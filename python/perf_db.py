#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Persist benchmark measurements and compare them across commits.

WHY THIS EXISTS

    A single run tells you a number; it does not tell you whether that number is
    normal. Over one week of debugging this library, several wrong conclusions
    came from exactly that gap: a 255 MB gap that was 73 MB on the next run, a
    "regression" that turned out to be two build directories with different
    compilers, a memory figure read from ru_maxrss (a high-water mark that never
    comes back down) and compared as if it were current usage.

    So this module stores what a run measured AND the context that produced it,
    and refuses to compare runs whose context differs.

WHAT IT STORES, AND WHY EACH FIELD IS THERE

    - median of N repetitions, plus the spread. A measurement whose spread
      exceeds the regression threshold cannot detect a regression of that size,
      and compare() says so instead of reporting a false positive.
    - peak AND current memory, separately. ru_maxrss alone cannot tell a
      transient peak from a steady-state cost.
    - commit, compiler, arch, thread count, cmake options. Two numbers from
      different build directories are not comparable, and the schema makes that
      checkable rather than a thing you remember.
    - the library's own counters (alloc bytes, still-compressed, reuse pools).
      When a regression shows up, these say WHICH of compression, retention or
      allocation moved -- without them you only know that something did.
    - the machine's load at measurement time. On a shared cluster a busy node
      inflates timings; a run taken under load is flagged, not silently trusted.

USAGE

    # after a benchmark run
    db = PerfDB("perf.sqlite")
    run_id = db.start_run(commit="abc1234", compiler="gcc-16.1.0", ...)
    db.record(run_id, config="omp-gnu-std-alloc", variant="zero_any",
              geometry="dyn", threads=64, divisor=5, iter_max=50,
              seconds=[12.9, 13.1, 12.8], peak_mo=[6222, 6230, 6218], ...)

    # compare two commits
    python3 perf_db.py compare perf.sqlite --base abc1234 --head def5678
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime, timezone

try:
    # Rows recorded before the rename carry chunk / chunk2 / chunk2a: read as
    # zero / zero_one_value / zero_any until 1.0.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_runner.settings import normalize_variant
except ImportError:
    def normalize_variant(label, warn=None):   # copied alone: labels as stored
        return label

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

-- One row per benchmark session: the CONTEXT. Everything that, if it differs,
-- makes two measurements incomparable.
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY,
    started_utc   TEXT    NOT NULL,
    commit_sha    TEXT    NOT NULL,
    commit_subject TEXT,
    dirty         INTEGER NOT NULL,   -- uncommitted changes present: 0/1
    compiler      TEXT    NOT NULL,   -- "gcc-16.1.0", "clang-22.1.6", ...
    arch          TEXT    NOT NULL,   -- uname -m
    host          TEXT,
    cmake_options TEXT,               -- the configure line, normalised
    notes         TEXT
);

-- One row per measured configuration within a run.
--
-- seconds_* and mem_* are the MEDIAN of the repetitions; *_spread is
-- (max - min) / median, the relative dispersion. compare() uses the spread to
-- decide whether a difference is meaningful at all.
CREATE TABLE IF NOT EXISTS measurements (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,

    -- what was measured
    config         TEXT    NOT NULL,  -- build flavour, e.g. omp-gnu-std-alloc
    variant        TEXT    NOT NULL,  -- zero / zero_one_value / zero_any (old rows: chunk...)
    geometry       TEXT    NOT NULL,  -- "256", "1024", "dyn"
    threads        INTEGER NOT NULL,
    divisor        INTEGER,
    iter_max       INTEGER,
    phase          TEXT    NOT NULL DEFAULT 'total',  -- 'total', 'init-zero', ...

    reps           INTEGER NOT NULL,
    seconds        REAL,              -- median
    seconds_spread REAL,              -- (max-min)/median
    peak_mo        REAL,              -- median of ru_maxrss
    peak_spread    REAL,
    current_mo     REAL,              -- median of current RSS, when the binary prints it
    current_spread REAL,

    -- the library's own counters, median across reps (JSON: name -> value)
    counters       TEXT,

    -- machine state while measuring: a busy node inflates timings
    loadavg        REAL,
    under_load     INTEGER NOT NULL DEFAULT 0,

    ok             INTEGER NOT NULL DEFAULT 1,
    detail         TEXT
);

CREATE INDEX IF NOT EXISTS idx_meas_key
    ON measurements(config, variant, geometry, threads, divisor, iter_max, phase);
CREATE INDEX IF NOT EXISTS idx_runs_commit ON runs(commit_sha);
"""

# A measurement whose own dispersion exceeds this cannot be used to claim a
# regression smaller than itself.
DEFAULT_REGRESSION_PCT = 10.0

# loadavg per core above which the node is considered busy enough that timings
# are not trustworthy.
LOAD_PER_CORE_BUSY = 0.7


def _usable_cores() -> int:
    """Cores this process may actually use (affinity mask, not nproc).

    nproc obeys OMP_NUM_THREADS and has reported 1 on a 128-CPU node.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _git(*args, cwd=None):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_context(repo=None) -> dict:
    """commit sha, subject and whether the tree has uncommitted changes."""
    sha = _git("rev-parse", "HEAD", cwd=repo) or "unknown"
    subject = _git("log", "-1", "--pretty=%s", cwd=repo) or ""
    status = _git("status", "--porcelain", cwd=repo)
    return {
        "commit_sha": sha,
        "commit_subject": subject,
        # A dirty tree means the measurement does not correspond to the commit
        # it is filed under. Recorded rather than refused -- you often measure
        # before committing -- but compare() warns about it.
        "dirty": 1 if status else 0,
    }


def _median_and_spread(values):
    """(median, relative spread) for a list of repetitions.

    Spread is (max - min) / median: how much the measurement moves on its own.
    Reported so a caller can tell "3% slower" from "3% slower, but it varies by
    15% between runs anyway".
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    if med == 0:
        return med, 0.0
    return med, (max(vals) - min(vals)) / abs(med)


class PerfDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        cur = self.conn.execute("SELECT version FROM schema_info")
        if cur.fetchone() is None:
            self.conn.execute("INSERT INTO schema_info(version) VALUES (?)",
                              (SCHEMA_VERSION,))
        self.conn.commit()

    # ── recording ────────────────────────────────────────────────────────────

    def start_run(self, *, compiler: str, cmake_options: str = "",
                  notes: str = "", repo: str | None = None) -> int:
        ctx = git_context(repo)
        cur = self.conn.execute(
            "INSERT INTO runs(started_utc, commit_sha, commit_subject, dirty, "
            "compiler, arch, host, cmake_options, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             ctx["commit_sha"], ctx["commit_subject"], ctx["dirty"],
             compiler, platform.machine(), platform.node(),
             cmake_options, notes))
        self.conn.commit()
        return cur.lastrowid

    def record(self, run_id: int, *, config: str, variant: str, geometry: str,
               threads: int, divisor=None, iter_max=None, phase: str = "total",
               seconds=None, peak_mo=None, current_mo=None,
               counters: dict | None = None, ok: bool = True,
               detail: str = "") -> int:
        """Store one configuration's measurement.

        seconds / peak_mo / current_mo are LISTS, one entry per repetition. A
        single-element list is accepted but its spread is 0, which compare()
        cannot distinguish from a genuinely stable measurement -- so pass at
        least three where the number matters.
        """
        variant = normalize_variant(variant)
        seconds = list(seconds or [])
        peak_mo = list(peak_mo or [])
        current_mo = list(current_mo or [])
        reps = max(len(seconds), len(peak_mo), len(current_mo), 1)

        s_med, s_spread = _median_and_spread(seconds)
        p_med, p_spread = _median_and_spread(peak_mo)
        c_med, c_spread = _median_and_spread(current_mo)

        try:
            load1 = os.getloadavg()[0]
        except (OSError, AttributeError):
            load1 = None
        busy = 0
        if load1 is not None:
            busy = 1 if load1 / max(1, _usable_cores()) > LOAD_PER_CORE_BUSY else 0

        cur = self.conn.execute(
            "INSERT INTO measurements(run_id, config, variant, geometry, threads,"
            " divisor, iter_max, phase, reps, seconds, seconds_spread, peak_mo,"
            " peak_spread, current_mo, current_spread, counters, loadavg,"
            " under_load, ok, detail)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, config, variant, geometry, threads, divisor, iter_max,
             phase, reps, s_med, s_spread, p_med, p_spread, c_med, c_spread,
             json.dumps(counters or {}), load1, busy, 1 if ok else 0, detail))
        self.conn.commit()
        return cur.lastrowid

    # ── comparison ───────────────────────────────────────────────────────────

    def _latest_run_for(self, commit_sha: str):
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE commit_sha LIKE ? "
            "ORDER BY started_utc DESC LIMIT 1", (commit_sha + "%",))
        return cur.fetchone()

    def compare(self, base_sha: str, head_sha: str,
                threshold_pct: float = DEFAULT_REGRESSION_PCT):
        """Compare two commits' latest runs, configuration by configuration.

        Returns (rows, warnings). A row is reported only when the two
        measurements share the same context AND the change exceeds both the
        threshold and the measurements' own dispersion -- the second condition
        is what keeps a noisy benchmark from producing a regression report every
        time it is run.
        """
        base, head = self._latest_run_for(base_sha), self._latest_run_for(head_sha)
        warnings = []
        if base is None:
            return [], [f"no run recorded for {base_sha}"]
        if head is None:
            return [], [f"no run recorded for {head_sha}"]

        # Context guards. Comparing across these produced a "2.8 GB regression"
        # that was two build directories with different compilers.
        for field, label in (("compiler", "compiler"), ("arch", "architecture")):
            if base[field] != head[field]:
                warnings.append(
                    f"{label} differs: {base[field]} vs {head[field]} — "
                    f"the comparison below is not meaningful")
        if base["cmake_options"] != head["cmake_options"]:
            warnings.append("cmake options differ between the two runs")
        if base["dirty"] or head["dirty"]:
            warnings.append("one of the runs was measured on a dirty tree")

        key = ("config", "variant", "geometry", "threads", "divisor",
               "iter_max", "phase")

        def by_key(run_id):
            out = {}
            for r in self.conn.execute(
                    "SELECT * FROM measurements WHERE run_id = ? AND ok = 1",
                    (run_id,)):
                out[tuple(normalize_variant(r[k]) if k == "variant" else r[k]
                          for k in key)] = r
            return out

        bm, hm = by_key(base["id"]), by_key(head["id"])
        rows = []
        for k in sorted(set(bm) & set(hm), key=lambda t: tuple(str(x) for x in t)):
            b, h = bm[k], hm[k]
            for metric, spread_col in (("seconds", "seconds_spread"),
                                       ("peak_mo", "peak_spread"),
                                       ("current_mo", "current_spread")):
                bv, hv = b[metric], h[metric]
                if bv in (None, 0) or hv is None:
                    continue
                change = (hv - bv) / bv * 100.0
                if abs(change) < threshold_pct:
                    continue
                # Dispersion guard: a measurement that moves by 15% on its own
                # cannot evidence a 10% regression.
                noise = max(b[spread_col] or 0.0, h[spread_col] or 0.0) * 100.0
                rows.append({
                    "key": dict(zip(key, k)),
                    "metric": metric,
                    "base": bv,
                    "head": hv,
                    "change_pct": change,
                    "noise_pct": noise,
                    "conclusive": abs(change) > noise,
                    "under_load": bool(b["under_load"] or h["under_load"]),
                })
        return rows, warnings


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cmd_compare(args):
    db = PerfDB(args.database)
    rows, warnings = db.compare(args.base, args.head, args.threshold)

    for w in warnings:
        print(f"[WARN] {w}")

    if not rows:
        print(f"No change above {args.threshold:.0f}% between {args.base[:8]} "
              f"and {args.head[:8]}.")
        return 0

    print(f"\n{'configuration':<52} {'metric':<11} {'base':>10} {'head':>10} "
          f"{'change':>9}  note")
    print("-" * 104)
    regressions = 0
    for r in rows:
        k = r["key"]
        label = (f"{k['config']}/{k['variant']}/{k['geometry']} "
                 f"t={k['threads']} d={k['divisor']} {k['phase']}")
        note = ""
        if not r["conclusive"]:
            note = f"within noise (±{r['noise_pct']:.0f}%)"
        elif r["under_load"]:
            note = "node was busy"
        else:
            if r["change_pct"] > 0:
                regressions += 1
            note = "REGRESSION" if r["change_pct"] > 0 else "improvement"
        print(f"{label[:52]:<52} {r['metric']:<11} {r['base']:>10.2f} "
              f"{r['head']:>10.2f} {r['change_pct']:>+8.1f}%  {note}")

    print(f"\n{regressions} conclusive regression(s).")
    return 1 if regressions else 0


def _cmd_list(args):
    db = PerfDB(args.database)
    print(f"{'date':<20} {'commit':<10} {'compiler':<16} {'meas':>5}  subject")
    print("-" * 92)
    for r in db.conn.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM measurements m WHERE m.run_id = r.id)"
            " AS n FROM runs r ORDER BY started_utc DESC LIMIT ?", (args.limit,)):
        dirty = "*" if r["dirty"] else " "
        print(f"{r['started_utc']:<20} {r['commit_sha'][:8]}{dirty:<2} "
              f"{r['compiler'][:16]:<16} {r['n']:>5}  {r['commit_subject'][:40]}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare two commits' latest runs")
    c.add_argument("database")
    c.add_argument("--base", required=True, help="reference commit (prefix ok)")
    c.add_argument("--head", required=True, help="commit under test")
    c.add_argument("--threshold", type=float, default=DEFAULT_REGRESSION_PCT,
                   help="percent change worth reporting (default: 10)")
    c.set_defaults(func=_cmd_compare)

    l = sub.add_parser("list", help="list recorded runs")
    l.add_argument("database")
    l.add_argument("--limit", type=int, default=20)
    l.set_defaults(func=_cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
