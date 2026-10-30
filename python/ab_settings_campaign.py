#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""ab_settings_campaign.py -- the A/B campaign that fixes the developer settings
of the re-chunk and of the fast allocator (docs/developer_settings.md, section 1).

Each setting still read from the environment is compared, one at a time, with
the defaults: the same shrink_bench run, the variable set or not. The runs give
the computation time ("Elapsed time computation") and the memory at the end of
the run, peak and current ("Memory consumption end"). A setting that changes
neither is fixed at its default and its variable can go; one that wins becomes
the default. The protocol is the one of the former Milan measurements
(KEEP_RECYCLED, SLICES, SCHEDULE): zero, zero_any and zero_one_value at dyn,
64 threads, three runs per point, the points interleaved so that a drift of the
node hits every one of them alike.

Run it INSIDE a job allocation on one compute node (ccc_msub / salloc):

    python3 python/ab_settings_campaign.py --build-root <dir with cmake-build-omp-gnu/> \\
        --out /ccc/work/.../ab_settings [--threads 64] [--reps 3]

or with --bindir <dir holding the shrink_bench executables>. --list prints the
plan and runs nothing. The results go to <out>/results.jsonl as the runs end:
the same command after an interruption (end of the job's time limit) runs only
what is missing, and a run whose executable changed since is refused (--force
mixes them anyway). --report-only rewrites the report from the results.

The points (--points selects some, by name or family):
    base         the defaults
    rthr0        CHUNK_RECLAIM_THREADS=0     madvise from every thread (1)
    pool16/256   CHUNK_DEFERRED_POOL_KB      deferred-write pool slabs of 16 / 256 KiB (64)

The first campaign (Milan, October 2026) measured eleven points; the settings it
fixed are gone with their points (CHANGELOG.md, Removed). CHUNK_RECLAIM_THREADS
stays at 1 until it is measured on Grace, where 288 threads would issue the
sweep's madvise calls, and CHUNK_DEFERRED_POOL_KB was never measured: run this
script there. Add a Point to POINTS to measure a new setting.

Every variable of the list is removed from the environment of the runs it does
not set, whatever the calling shell has: base is the defaults. CHUNK_LOG is
fixed at its default (warn,rechunk=info), so that the logs show the re-chunks.

The report (<out>/report.txt, and report.csv): per variant and point, the
median of the computation time, of the peak and of the end memory, with their
deviation from base. A deviation is marked "~" when it is within the noise:
the larger of the two points' spreads (max - min over the median) and 1%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# The executables carry the backend of their build (shrink_bench_<backend>_...):
# bench_runner's lookup tries the three, a build directory holding one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_runner.settings import find_bench_exe  # noqa: E402

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    name: str
    family: str
    env: dict = field(default_factory=dict)
    question: str = ""


POINTS = [
    Point("base", "base", {}, "the defaults"),
    Point("rthr0", "rthreads", {"CHUNK_RECLAIM_THREADS": "0"}, "madvise from every thread"),
    Point("pool16", "pool", {"CHUNK_DEFERRED_POOL_KB": "16"}, "deferred-write slabs of 16 KiB"),
    Point("pool256", "pool", {"CHUNK_DEFERRED_POOL_KB": "256"}, "deferred-write slabs of 256 KiB"),
]

# Every variable a point sets: removed from the environment of the others.
CAMPAIGN_VARS = sorted({k for p in POINTS for k in p.env})
# The other settings that would bias the runs if the calling shell had them.
CLEARED_VARS = CAMPAIGN_VARS + [
    "CHUNK_LOG", "CHUNK_LOG_FILE", "CHUNK_ARENA_REPORT", "CHUNK_LEDGER_REPORT", "CHUNK_OMPT_STATS",
    "CHUNK_AGGRESSIVE_COMPRESSION", "CHUNK_CST2_PROBE", "CHUNK_CST2_RECLAIM",
    "SHRINK_BENCH_EXACT_RSS", "SHRINK_BENCH_RSS_SAMPLE_MS", "SHRINK_BENCH_MALLOC_TRIM", "SHRINK_BENCH_KEEP_POOLS",
]

VARIANTS = ["zero", "zero_any", "zero_one_value"]
DEFAULT_ARGS = "--mode perf --no_io --multiply 2 --iter_max 500"  # bench_recycle.sh's workload


def select_points(spec: str | None) -> list[Point]:
    if not spec:
        return list(POINTS)
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    known = {p.name for p in POINTS} | {p.family for p in POINTS}
    bad = sorted(wanted - known)
    if bad:
        sys.exit(f"--points: unknown {', '.join(bad)} (known: {', '.join(sorted(known))})")
    chosen = [p for p in POINTS if p.name in wanted or p.family in wanted]
    if chosen[0].name != "base":
        chosen.insert(0, POINTS[0])  # every comparison needs base
    return chosen


# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------


def exe_name(variant: str, size: str, index: str) -> str:
    """The executable's name after its shrink_bench_<backend>_ (find_bench_exe adds it)."""
    return f"{variant}_int64_{size}" if index == "int64" else f"{variant}_{size}"


def find_exe(args, variant: str) -> Path | None:
    name = exe_name(variant, args.size, args.index)
    if args.bindir:
        dirs = [Path(args.bindir)]
    else:
        cfg = Path(args.build_root) / ("cmake-build-" + args.config.removeprefix("cmake-build-"))
        arch = platform.machine()
        dirs = sorted(d for d in cfg.glob("*/examples/shrink_bench") if arch in str(d))
        dirs += [cfg / "examples" / "shrink_bench"]
    for d in dirs:
        exe = find_bench_exe(str(d), name)
        if exe:
            return Path(exe).resolve()
    return None


def exe_stamp(exe: Path) -> str:
    st = exe.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

TIME_RE = re.compile(r"^Elapsed time computation : ([0-9.]+)\s*s", re.M)
MEM_END_RE = re.compile(r"^Memory consumption end : ([0-9.]+) Mo \(peak\), ([0-9.]+) Mo \(current\)", re.M)
RECHUNKS_RE = re.compile(r"^Re-chunks performed: (\d+)", re.M)
ENV_WARN_RE = re.compile(r"^\[shrink_array\]\[env\]\[W\].*$", re.M)


def numa_prefix(threads: int, mode: str) -> list[str]:
    if mode == "off":
        return []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import numa_launch  # bench_and_test.py's placement: --interleave over the nodes the run spans
    except ImportError:
        print("warning: numa_launch.py not found next to this script: no numactl", file=sys.stderr)
        return []
    return numa_launch.interleave_prefix({"OMP_NUM_THREADS": str(threads), "OMP_PROC_BIND": "close"})


def run_env(point: Point, threads: int) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in CLEARED_VARS}
    env.update(point.env)
    env.update(OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="close", OMP_PLACES="cores",
               CHUNK_LOG="warn,rechunk=info")
    return env


def run_one(args, exe: Path, variant: str, point: Point, rep: int, prefix: list[str], tag: str = "") -> dict:
    log = Path(args.out) / "logs" / f"{variant}_{point.name}_{tag or f'r{rep}'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = prefix + [str(exe)] + args.bench_args.split()
    t0 = time.monotonic()
    try:
        with open(log, "w") as out:
            out.write("# " + " ".join(f"{k}={v}" for k, v in point.env.items()) + " " + " ".join(cmd) + "\n")
            out.flush()
            proc = subprocess.run(cmd, env=run_env(point, args.threads), stdout=out, stderr=subprocess.STDOUT,
                                  cwd=log.parent, timeout=args.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = time.monotonic() - t0
    text = log.read_text(errors="replace")
    rec = {"variant": variant, "point": point.name, "rep": rep, "threads": args.threads,
           "exe": str(exe), "stamp": exe_stamp(exe), "rc": rc, "wall_s": round(wall, 3),
           "log": str(log.relative_to(args.out)), "env": point.env}
    m = TIME_RE.search(text)
    rec["comp_s"] = float(m.group(1)) if m else None
    m = MEM_END_RE.search(text)
    rec["peak_mb"], rec["end_mb"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = RECHUNKS_RE.search(text)
    rec["rechunks"] = int(m.group(1)) if m else None
    # A typo in a value is a warning on channel env, and the run would be base again.
    rec["env_warnings"] = sorted({w.split("] ", 1)[-1] for w in ENV_WARN_RE.findall(text)
                                  if any(k in w for k in CAMPAIGN_VARS)})
    rec["ok"] = rc == 0 and rec["comp_s"] is not None and rec["peak_mb"] is not None and not rec["env_warnings"]
    return rec


# ---------------------------------------------------------------------------
# Results and report
# ---------------------------------------------------------------------------


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a killed job
    return out


def spread(values: list[float]) -> float:
    med = statistics.median(values)
    return (max(values) - min(values)) / med if med else 0.0


def fmt_delta(val: float, base: float, noise: float) -> str:
    d = (val - base) / base if base else 0.0
    mark = "~" if abs(d) <= noise else " "
    return f"{d * 100:+6.1f}%{mark}"


def report(out: Path, results: list[dict], threads: int) -> str:
    ok = [r for r in results if r.get("ok") and r.get("threads") == threads]
    by: dict[tuple, list[dict]] = {}
    for r in ok:
        by.setdefault((r["variant"], r["point"]), []).append(r)
    lines = [f"A/B settings campaign -- {threads} threads, medians over the runs; "
             "~ = within the noise (max of the two spreads, 1% at least)", ""]
    rows = []
    for variant in VARIANTS:
        base = by.get((variant, "base"))
        if not base:
            continue
        b = {k: [r[k] for r in base] for k in ("comp_s", "peak_mb", "end_mb")}
        bm = {k: statistics.median(v) for k, v in b.items()}
        lines.append(f"== {variant} (base: {bm['comp_s']:.2f} s, peak {bm['peak_mb']:.0f} Mo, "
                     f"end {bm['end_mb']:.0f} Mo, n={len(base)}, time spread {spread(b['comp_s']) * 100:.1f}%)")
        lines.append(f"  {'point':<11} {'n':>2} {'time s':>8} {'d time':>9} {'peak Mo':>9} {'d peak':>9} "
                     f"{'end Mo':>9} {'d end':>9} {'rechunks':>8}  setting")
        for p in POINTS:
            runs = by.get((variant, p.name))
            if not runs:
                continue
            v = {k: [r[k] for r in runs] for k in ("comp_s", "peak_mb", "end_mb")}
            vm = {k: statistics.median(x) for k, x in v.items()}
            noise = {k: max(0.01, spread(b[k]), spread(v[k])) for k in v}
            rch = sorted({r["rechunks"] for r in runs if r["rechunks"] is not None})
            setting = " ".join(f"{k}={x}" for k, x in p.env.items()) or "defaults"
            lines.append(f"  {p.name:<11} {len(runs):>2} {vm['comp_s']:8.2f} "
                         f"{fmt_delta(vm['comp_s'], bm['comp_s'], noise['comp_s']):>9} {vm['peak_mb']:9.0f} "
                         f"{fmt_delta(vm['peak_mb'], bm['peak_mb'], noise['peak_mb']):>9} {vm['end_mb']:9.0f} "
                         f"{fmt_delta(vm['end_mb'], bm['end_mb'], noise['end_mb']):>9} "
                         f"{'/'.join(map(str, rch)) or '-':>8}  {setting}")
            rows.append({"variant": variant, "point": p.name, "family": p.family, "setting": setting,
                         "n": len(runs), "comp_s": vm["comp_s"], "peak_mb": vm["peak_mb"], "end_mb": vm["end_mb"],
                         "d_comp": (vm["comp_s"] - bm["comp_s"]) / bm["comp_s"],
                         "d_peak": (vm["peak_mb"] - bm["peak_mb"]) / bm["peak_mb"],
                         "d_end": (vm["end_mb"] - bm["end_mb"]) / bm["end_mb"],
                         "noise_comp": noise["comp_s"], "noise_peak": noise["peak_mb"],
                         "noise_end": noise["end_mb"]})
        lines.append("")
    if any(r["n"] < 2 for r in rows):
        lines.append("n=1: no spread is measured, the noise is the 1% floor only -- not a verdict.")
        lines.append("")
    bad = [r for r in results if not r.get("ok")]
    if bad:
        lines.append(f"{len(bad)} run(s) failed or unreadable (not counted):")
        for r in bad:
            why = r["env_warnings"][0] if r.get("env_warnings") else f"rc={r.get('rc')}"
            lines.append(f"  {r['variant']} {r['point']} r{r['rep']}: {why}  ({r['log']})")
    stamps = {}
    for r in ok:
        stamps.setdefault(r["variant"], set()).add(r["stamp"])
    mixed = [v for v, s in stamps.items() if len(s) > 1]
    if mixed:
        lines.append(f"WARNING: runs of different executables mixed for {', '.join(mixed)} (--force)")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    if rows:
        with open(out / "report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--build-root", help="directory holding cmake-build-<config>/ (compile_all.sh's root)")
    where.add_argument("--bindir", help="directory holding the shrink_bench executables")
    ap.add_argument("--config", default="omp-gnu", help="build configuration under --build-root (omp-gnu)")
    ap.add_argument("--out", required=True, help="output directory (in your work space); reused to resume")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3, help="runs per point (3)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="among " + ", ".join(VARIANTS))
    ap.add_argument("--size", default="dyn", help="chunk-size declination of the executables (dyn)")
    ap.add_argument("--index", choices=["int64", "int32"], default="int64",
                    help="int64: shrink_bench_<backend>_<v>_int64_<size> (SHRINK_BENCH_INT64=ON, the "
                         "default); int32: shrink_bench_<backend>_<v>_<size>")
    ap.add_argument("--points", help="comma-separated point names or families (default: all)")
    ap.add_argument("--args", dest="bench_args", default=DEFAULT_ARGS, help=f"shrink_bench arguments ({DEFAULT_ARGS})")
    ap.add_argument("--numa", choices=["auto", "off"], default="auto", help="numactl --interleave (auto)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per run (3600)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded base runs per variant before the campaign (1): page cache, frequency")
    ap.add_argument("--seed", type=int, default=1, help="seed of the per-repetition order of the points")
    ap.add_argument("--list", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--report-only", action="store_true", help="rewrite the report from results.jsonl")
    ap.add_argument("--force", action="store_true", help="resume even if an executable changed")
    args = ap.parse_args()

    out = Path(args.out)
    results_path = out / "results.jsonl"
    if args.report_only:
        print(report(out, load_results(results_path), args.threads), end="")
        return 0
    if not args.build_root and not args.bindir:
        ap.error("--build-root or --bindir is required")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        ap.error(f"--variants: unknown {', '.join(bad)}")
    points = select_points(args.points)
    exes = {}
    for v in variants:
        exe = find_exe(args, v)
        if exe is None:
            where_ = args.bindir or f"{args.build_root}/cmake-build-{args.config}"
            sys.exit(f"shrink_bench_{{omp,std,serial}}_{exe_name(v, args.size, args.index)} not found under {where_} "
                     f"(int64 executables need SHRINK_BENCH_INT64=ON; --index int32 takes the others)")
        exes[v] = exe

    # The plan: per repetition, every variant, its points in an order drawn from the seed.
    plan = []
    for rep in range(1, args.reps + 1):
        rng = random.Random(args.seed * 1000 + rep)
        for v in variants:
            order = list(points)
            rng.shuffle(order)
            plan += [(v, p, rep) for p in order]

    out.mkdir(parents=True, exist_ok=True)
    done = load_results(results_path)
    stamps = {v: exe_stamp(e) for v, e in exes.items()}
    stale = {(r["variant"], r["stamp"]) for r in done if r.get("ok") and r["variant"] in stamps
             and r["stamp"] != stamps[r["variant"]] and r.get("threads") == args.threads}
    if stale and not args.force:
        sys.exit(f"{out}: results of other executables for {', '.join(sorted({v for v, _ in stale}))} "
                 "(rebuilt since?): use another --out, or --force to mix them")
    have = {(r["variant"], r["point"], r["rep"]) for r in done if r.get("ok") and r.get("threads") == args.threads}
    todo = [t for t in plan if (t[0], t[1].name, t[2]) not in have]

    prefix = numa_prefix(args.threads, args.numa)
    print(f"{len(plan)} runs in the plan, {len(plan) - len(todo)} already done, {len(todo)} to run; "
          f"{args.threads} threads, {' '.join(prefix) or 'no numactl'}")
    for v, e in exes.items():
        print(f"  {v}: {e}")
    print(f"  shrink_bench {args.bench_args}")
    if args.list:
        for v, p, rep in todo:
            print(f"  r{rep} {v:<15} {p.name:<11} {' '.join(f'{k}={x}' for k, x in p.env.items())}")
        return 0
    (out / "meta.json").write_text(json.dumps({
        "host": platform.node(), "machine": platform.machine(), "threads": args.threads,
        "args": args.bench_args, "exes": {v: str(e) for v, e in exes.items()}, "stamps": stamps,
        "numa": prefix, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    if todo:
        for v in sorted({t[0] for t in todo}):
            for w in range(1, args.warmup + 1):
                rec = run_one(args, exes[v], v, POINTS[0], 0, prefix, tag=f"warmup{w}")
                print(f"warm-up {v}: {rec['wall_s']:.0f} s{'' if rec['ok'] else ' (FAILED: see ' + rec['log'] + ')'}",
                      flush=True)

    t_start = time.monotonic()
    for i, (v, p, rep) in enumerate(todo, 1):
        rec = run_one(args, exes[v], v, p, rep, prefix)
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        elapsed = time.monotonic() - t_start
        eta = elapsed / i * (len(todo) - i)
        state = "ok" if rec["ok"] else f"FAILED (rc={rec['rc']}{', env warning' if rec['env_warnings'] else ''})"
        comp = f"{rec['comp_s']:.2f} s" if rec["comp_s"] is not None else "-"
        peak = f"{rec['peak_mb']:.0f} Mo" if rec["peak_mb"] is not None else "-"
        print(f"[{i}/{len(todo)}] r{rep} {v:<15} {p.name:<11} {comp:>9} peak {peak:>9}  {state}"
              f"   (ETA {eta / 60:.0f} min)", flush=True)

    print()
    print(report(out, load_results(results_path), args.threads), end="")
    print(f"results: {results_path}\nreport:  {out / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
