#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""diag_zero_rechunk_slowdown.py -- why does dynamic zero run slower after its
re-chunk than static 1024 (docs/adaptive_rechunk.md, "Known limitations")?

Read-only diagnosis: nothing in the library changes, every factor is switched
by an environment variable the library already reads. Run it INSIDE a job
allocation on one compute node (ccc_msub / salloc), from anywhere:

    python3 diag_zero_rechunk_slowdown.py --bindir <build>/examples/shrink_bench \\
        --threads 64 --out /ccc/work/.../diag_zero [--perf] [--reps 2]

shrink_bench runs in --mode perf (large sizes, --iter_max iterations; the
default check mode stops at 33 iterations of a small problem and never
re-chunks) unless --args names a --mode.

Part 1 -- A/B on the rebuild's state (always). shrink_bench_<backend>_zero_1024
(static) and shrink_bench_<backend>_zero_dyn under:
    dyn            the default
                   The rebuild slices the array in 16 and spreads each slice
                   over ALL ranks (guided,16): a new chunk's buffer is
                   allocated, first-touched and owned (slab owner, NUMA node)
                   by an arbitrary rank, unrelated to the ranks that use it
                   afterwards. (The slice count, the rebuild's schedule and
                   its slab recycling are fixed now: CHUNK_RECHUNK_SLICES,
                   CHUNK_RECHUNK_SCHEDULE and CHUNK_RECHUNK_RECYCLE are gone,
                   with dyn_1slice, dyn_static1 and dyn_norecycle, and the
                   affinity mode with dyn_lazy: CHUNK_ALLOC_AFFINITY_EAGER is
                   gone too.)
    dyn_dryrun     --rechunk_dry_run         the advisor analyses, never moves
    dyn_noadvisor  --advisor_period 0        no advisor (stays at 64)
                   these two have no re-chunk: compare them with each other
                   (the analyses' cost) and with dyn over the same iterations
  The time per iteration comes from the timestamps of the "iter <k> : time"
  lines (printed every 100 iterations; stdout line-buffered by stdbuf), so
  each 100-iteration interval gives a mean. An interval in which a
  re-chunk phase line appeared is marked. The summary compares the intervals after
  the LAST re-chunk with the same intervals of the static run, for the time and
  for the resident memory (VmRSS, read at each iter line): a dyn run that holds
  much less than the static one pays its churn for that memory.
  It then gives the churn per array after the last re-chunk, from the decider's
  churn lines (rechunk=debug): materialisations per iteration and per phase, and
  the change of the resident materialised chunks. Each array has its row, so the
  churn shows up as one array's or as everybody's. Dynamic builds only: the
  static run has no per-array key, hence no such line. This is per array, not
  per chunk: the library has no trace of which chunk churns.

Part 2 -- probes during the slow phase (--perf), static and dyn only, at
iteration --probe-iter (default: 2 iterations after the last re-chunk of a
first dyn run, else 3/4 of the run):
    numastat -p        pages per NUMA node (placement)
    perf stat          task-clock / wall = busy cores, instructions / cycles
                       = IPC, context switches, page faults -- over --window s
    perf c2c           false sharing (HITM lines), same window
    perf record -g     where the idle cores are: libgomp barrier / lock
                       symbols versus library code
    perf report        the share of the samples in the compression passes and
                       around their window (start/finish and their barrier),
                       if those symbols are not inlined: what the passes cost,
                       not how many there were (no counter for that)
  plus one run of each with CHUNK_LOG=...,arena=info: freepath parked_foreign,
  the blocks freed by a rank that does not own their slab -- the allocator's
  view of the ownership mismatch the rebuild schedule would leave behind.
  (Separate runs: those counters are shared atomics and perturb the timing.)

Everything lands in --out: one directory per run (log, csv, probe outputs)
and summary.txt.
"""

import argparse
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

# The executables carry the backend of their build (shrink_bench_<backend>_zero_dyn):
# bench_runner's lookup tries the three, the --bindir holding one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_runner.settings import find_bench_exe  # noqa: E402

ITER_RE = re.compile(r"^iter (\d+) : time : ")
# The re-chunk's phase line, as the log writes it (rechunk=debug) and as it was printed before.
RECHUNK_RE = re.compile(r"^(?:\[RECHUNK\]|\[shrink_array\]\[rechunk\]\[D\]\[\d+\]) (\d+) -> (\d+) phases")
# The arena report's free-path line, as the log writes it ("[shrink_array][arena][I][<tid>] freepath
# <when> freed=..."), and as it was printed before the log ("[FREEPATH] <when> freed=...").
FREEPATH_RE = re.compile(r"(?:\[FREEPATH\]|\[arena\]\[I\]\[\d+\] freepath) (\S+)\s+freed=(\d+)\s+parked_own=(\d+)"
                         r"\s+parked_foreign=(\d+)")

# The decider's churn line (rechunk=debug), one per array at each advisor tick.
CHURN_RE = re.compile(r"churn (.+?) S=(\d+): (\d+) materialisations over (\d+) phases since the last tick .*?"
                      r"resident materialised (-?\d+) -> (-?\d+)")
# The perf report's groups of symbols: a label, the symbols (regex on the line).
PASS_SYMBOLS = [
    ("compression pass", r"make_block_compression|fill_list_to_compress|clear_int_st"),
    ("window start/finish", r"Compression\w*<.*>::(?:start|finish)\b"),
    ("materialisation", r"create_value|alloc_chunk"),
    ("OpenMP barrier", r"__kmp_\w*barrier|__kmp_hyper|gomp_\w*barrier|gomp_team_barrier"),
]

# (name, executable after its shrink_bench_<backend>_, environment, extra
# shrink_bench arguments)
CONFIGS = [
    ("static", "zero_1024", {}, []),
    ("dyn", "zero_dyn", {}, []),
    # the advisor's own share: analyses without any re-chunk, and none at all
    ("dyn_dryrun", "zero_dyn", {}, ["--rechunk_dry_run"]),
    ("dyn_noadvisor", "zero_dyn", {}, ["--advisor_period", "0"]),
]


def have(tool):
    return shutil.which(tool) is not None


def vm_rss_mb(pid):
    """The process's resident memory in MB, or None once it is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def churn_by_array(rs, after):
    """Per array, from the churn lines of runs @p rs past iteration @p after:
    (array, S, materialisations per iteration, per phase, resident first, last).
    Each line covers the iterations since the array's previous one."""
    rows = {}
    for r in rs:
        seen = {}
        for k, array, size, creates, phases, _, resident in r.churn:
            prev = seen.get(array)
            seen[array] = k
            if prev is None or prev <= after or k <= prev:
                continue
            row = rows.setdefault(array, {"size": size, "creates": 0, "phases": 0, "iters": 0,
                                          "first": resident, "last": resident})
            row["size"] = size
            row["creates"] += creates
            row["phases"] += phases
            row["iters"] += k - prev
            row["last"] = resident
    return [(array, v["size"], v["creates"] / v["iters"], v["creates"] / max(v["phases"], 1), v["first"], v["last"])
            for array, v in sorted(rows.items(), key=lambda kv: -kv[1]["creates"] / kv[1]["iters"])]


def symbol_shares(run):
    """The share of the samples of each group of PASS_SYMBOLS in the run's
    perf_report.txt ("" without one, or with none of them)."""
    f = run.dir / "perf_report.txt"
    if not f.exists():
        return ""
    shares = dict.fromkeys((label for label, _ in PASS_SYMBOLS), 0.0)
    for line in f.read_text(errors="replace").splitlines():
        m = re.match(r"\s*([\d.]+)%\s", line)
        if m:
            for label, pattern in PASS_SYMBOLS:
                if re.search(pattern, line):
                    shares[label] += float(m.group(1))
                    break
    return ", ".join(f"{label} {v:.1f}%" for label, v in shares.items() if v)


class Run:
    """One shrink_bench execution, its output timestamped line by line."""

    def __init__(self, name, exe, env_extra, args, outdir, threads, probe_iter=None, probes=None, window=10):
        self.name, self.exe, self.args = name, exe, args
        self.dir = Path(outdir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env.update({"OMP_NUM_THREADS": str(threads), "NO_IO": "1", "CHUNK_LOG": "warn,rechunk=debug"})
        self.env.setdefault("OMP_PROC_BIND", "close")
        self.env.setdefault("OMP_PLACES", "cores")
        self.env.update(env_extra)
        self.probe_iter, self.probes, self.window = probe_iter, probes or [], window
        self.iter_t = {}  # iteration -> timestamp of its line
        self.rechunk_iters = set()
        self.freepath = []  # (iteration, when, freed, parked_own, parked_foreign)
        self.rss_mb = {}    # k line -> VmRSS, MB
        self.churn = []     # (iteration, array, S, materialisations, phases, resident before, after)
        self.t0 = None
        self.rc = None

    def _probe(self, pid):
        w = str(self.window)
        if "numastat" in self.probes and have("numastat"):
            with open(self.dir / "numastat.txt", "w") as f:
                subprocess.run(["numastat", "-p", str(pid)], stdout=f, stderr=subprocess.STDOUT)
        procs = []
        if "stat" in self.probes and have("perf"):
            procs.append(subprocess.Popen(
                ["perf", "stat", "-e", "task-clock,cycles,instructions,context-switches,page-faults,cpu-migrations",
                 "-p", str(pid), "-o", str(self.dir / "perf_stat.txt"), "--", "sleep", w],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        if "c2c" in self.probes and have("perf"):
            procs.append(subprocess.Popen(
                ["perf", "c2c", "record", "-o", str(self.dir / "perf_c2c.data"), "-p", str(pid), "--", "sleep", w],
                stdout=subprocess.DEVNULL, stderr=open(self.dir / "perf_c2c.err", "w")))
        if "record" in self.probes and have("perf"):
            procs.append(subprocess.Popen(
                ["perf", "record", "-g", "-F", "499", "-o", str(self.dir / "perf.data"), "-p", str(pid), "--",
                 "sleep", w],
                stdout=subprocess.DEVNULL, stderr=open(self.dir / "perf_record.err", "w")))
        for p in procs:
            p.wait()
        (self.dir / "probe_done").write_text(f"probed at iteration {self.probe_iter}, window {w} s\n")

    def run(self):
        cmd = ["stdbuf", "-oL", "-eL", self.exe] + self.args if have("stdbuf") else [self.exe] + self.args
        log = open(self.dir / "run.log", "w")
        self.t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env, text=True,
                             bufsize=1, cwd=self.dir)
        cur_iter = -1
        probed = False
        prober = None
        for line in p.stdout:
            now = time.monotonic() - self.t0
            log.write(f"{now:10.3f} {line}")
            m = ITER_RE.search(line)
            if m:
                cur_iter = int(m.group(1))
                self.iter_t[cur_iter] = now
                rss = vm_rss_mb(p.pid)
                if rss is not None:
                    self.rss_mb[cur_iter] = rss
                if self.probes and not probed and self.probe_iter is not None and cur_iter >= self.probe_iter:
                    probed = True
                    prober = threading.Thread(target=self._probe, args=(p.pid,))
                    prober.start()
                continue
            if RECHUNK_RE.search(line):
                self.rechunk_iters.add(cur_iter)  # the interval that ends at the next k line
                continue
            m = CHURN_RE.search(line)
            if m:
                self.churn.append((cur_iter, m.group(1), *(int(m.group(j)) for j in range(2, 7))))
                continue
            m = FREEPATH_RE.search(line)
            if m:
                self.freepath.append((cur_iter, m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
        self.rc = p.wait()
        if prober:
            prober.join()
        log.close()
        self.write_csv()
        return self

    def iter_times(self):
        """Seconds per iteration over each interval (previous k line, this k]."""
        ks = sorted(self.iter_t)
        return {k: (self.iter_t[k] - self.iter_t[p]) / (k - p) for p, k in zip(ks, ks[1:])}

    def write_csv(self):
        with open(self.dir / "iterations.csv", "w") as f:
            f.write("interval_end_k,s_per_iteration,rechunk_in_interval,rss_mb\n")
            ks = sorted(self.iter_t)
            for p, k in zip(ks, ks[1:]):
                dt = (self.iter_t[k] - self.iter_t[p]) / (k - p)
                f.write(f"{k},{dt:.4f},{int(p in self.rechunk_iters)},{self.rss_mb.get(k, '')}\n")
        with open(self.dir / "churn.csv", "w") as f:
            f.write("iteration,array,S,materialisations,phases,resident_before,resident_after\n")
            for row in self.churn:
                f.write(",".join(map(str, row)) + "\n")

    def last_rechunk(self):
        """The k line after which the last re-chunk ran (its interval ends at the next k line)."""
        return max(self.rechunk_iters) if self.rechunk_iters else None


def perf_reports(run):
    """Text reports next to the perf data (best effort)."""
    d = run.dir
    if (d / "perf_c2c.data").exists():
        with open(d / "perf_c2c_report.txt", "w") as f:
            subprocess.run(["perf", "c2c", "report", "-i", str(d / "perf_c2c.data"), "--stdio", "-N",
                            "--coalesce", "iaddr"], stdout=f, stderr=subprocess.STDOUT)
    if (d / "perf.data").exists():
        with open(d / "perf_report.txt", "w") as f:
            subprocess.run(["perf", "report", "-i", str(d / "perf.data"), "--stdio", "--no-children",
                            "--sort", "dso,sym", "--percent-limit", "0.5"], stdout=f, stderr=subprocess.STDOUT)


def stat_summary(run):
    f = run.dir / "perf_stat.txt"
    if not f.exists():
        return ""
    txt = f.read_text()
    def num(name):
        m = re.search(r"([\d.,]+)\s+(?:msec\s+)?" + name, txt)
        return float(m.group(1).replace(",", "")) if m else None
    task, cyc, ins = num("task-clock"), num("cycles"), num("instructions")
    out = []
    if task:
        out.append(f"busy cores {task / 1000.0 / run.window:.1f}")
    if cyc and ins:
        out.append(f"IPC {ins / cyc:.2f}")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bindir", required=True, help="directory holding shrink_bench_<backend>_zero_dyn and _1024")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", required=True, help="output directory (put it in your work space)")
    ap.add_argument("--args", default="", help="extra shrink_bench arguments (default adds --mode perf --iter_max 2000)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions of part 1 (best per iteration kept)")
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS), help="part-1 configurations to run")
    ap.add_argument("--perf", action="store_true", help="part 2: numastat / perf stat / perf c2c / perf record")
    ap.add_argument("--probe-iter", type=int, default=None, help="iteration that starts the probes")
    ap.add_argument("--window", type=int, default=10, help="probe window, seconds")
    a = ap.parse_args()

    bindir = Path(a.bindir).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args = a.args.split()
    if "--mode" not in args:
        args = ["--mode", "perf"] + args
    if "--iter_max" not in args:
        args += ["--iter_max", "2000"]
    wanted = set(a.configs.split(","))
    path_of = {exe: find_bench_exe(str(bindir), exe) for name, exe, *_ in CONFIGS}
    missing = sorted({f"shrink_bench_{{omp,std,serial}}_{exe}" for name, exe, *_ in CONFIGS
                      if name in wanted and path_of[exe] is None})
    if missing:
        print(f"error: not found in {bindir}: {', '.join(missing)}", file=sys.stderr)
        return 1
    summary = [f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  threads={a.threads}  args='{' '.join(args)}'  "
               f"OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND', 'close')} "
               f"OMP_PLACES={os.environ.get('OMP_PLACES', 'cores')}  host={os.uname().nodename}"]

    # ---- part 1
    runs = {}
    for rep in range(a.reps):
        for name, exe, env, extra in CONFIGS:
            if name not in wanted:
                continue
            print(f"[part 1] {name} (rep {rep + 1}/{a.reps})", flush=True)
            r = Run(f"{name}_r{rep}", path_of[exe], env, args + extra, out, a.threads).run()
            if r.rc != 0:
                print(f"  {name}: exit code {r.rc}, see {r.dir / 'run.log'}", file=sys.stderr)
            runs.setdefault(name, []).append(r)

    def best_times(rs):
        """Per iteration, the fastest repetition."""
        out_t = {}
        for r in rs:
            for k, dt in r.iter_times().items():
                out_t[k] = min(dt, out_t.get(k, dt))
        return out_t

    def last_rechunk(rs):
        lrs = [r.last_rechunk() for r in rs if r.last_rechunk() is not None]
        return max(lrs) if lrs else None

    for name in runs:
        if name != "static" and last_rechunk(runs[name]) is None:
            summary.append(f"WARNING: {name}: no re-chunk seen (no 'a -> b phases' line) -- the comparison "
                           "below is then between two non-re-chunked runs")
    st = best_times(runs.get("static", []))
    summary.append("\n## Part 1: iteration wall time after the last re-chunk (vs static 1024, same iterations;"
                   " best of the repetitions)\n")
    summary.append(f"{'config':16s} {'last rechunk':>12s} {'intervals':>9s} {'s/iter':>9s} {'static':>9s} {'ratio':>7s}")
    for name, *_ in CONFIGS:
        if name not in runs:
            continue
        it = best_times(runs[name])
        lr = last_rechunk(runs[name])
        # intervals that end after the one holding the last re-chunk
        later = sorted(k for k in it if lr is None or k > lr)
        ks = [k for k in later[1:] if k in st] if lr is not None else [k for k in later if k in st]
        if not ks:
            summary.append(f"{name:16s} {str(lr):>12s} {'-':>9s}")
            continue
        m = statistics.mean(it[k] for k in ks)
        s = statistics.mean(st[k] for k in ks)
        summary.append(f"{name:16s} {'after k=' + str(lr):>12s} {len(ks):>9d} {m:9.4f} {s:9.4f} {m / s:7.2f}")

    # Memory, over the same iterations: what the dyn run's churn may be buying.
    def mean_rss(rs, ks):
        vals = [r.rss_mb[k] for r in rs for k in ks if k in r.rss_mb]
        return statistics.mean(vals) if vals else None

    st_rss = {k for r in runs.get("static", []) for k in r.rss_mb}
    summary.append("\n## Part 1: resident memory after the last re-chunk (VmRSS, MB, same iterations)\n")
    summary.append(f"{'config':16s} {'intervals':>9s} {'MB':>10s} {'static':>10s} {'ratio':>7s}")
    for name, *_ in CONFIGS:
        if name not in runs or name == "static":
            continue
        lr = last_rechunk(runs[name])
        ks = sorted(k for r in runs[name] for k in r.rss_mb if k in st_rss and (lr is None or k > lr))
        m, s = mean_rss(runs[name], ks), mean_rss(runs.get("static", []), ks)
        if m is None or s is None:
            summary.append(f"{name:16s} {'-':>9s}")
            continue
        summary.append(f"{name:16s} {len(set(ks)):>9d} {m:10.0f} {s:10.0f} {m / s:7.2f}")

    # Which arrays churn after the last re-chunk.
    dyn = runs.get("dyn", [])
    rows = churn_by_array(dyn, last_rechunk(dyn) or 0)
    summary.append("\n## Part 1: churn per array after the last re-chunk (dyn; static has no such line)\n")
    if rows:
        summary.append(f"{'array':24s} {'S':>6s} {'mat/iter':>10s} {'mat/phase':>10s} {'resident':>22s}")
        for array, size, per_iter, per_phase, first, last in rows:
            summary.append(f"{array[:24]:24s} {size:>6d} {per_iter:10.0f} {per_phase:10.0f} "
                           f"{str(first) + ' -> ' + str(last):>22s}")
    else:
        summary.append("(no churn line after the last re-chunk: too few advisor ticks, see max_backoff_ticks)")

    # ---- part 2
    if a.perf:
        dyn_runs = runs.get("dyn", [])
        probe_iter = a.probe_iter
        if probe_iter is None:
            lr = last_rechunk(dyn_runs)
            n = max(dyn_runs[0].iter_t) if dyn_runs and dyn_runs[0].iter_t else 32
            probe_iter = (lr + 1) if lr is not None and lr + 1 < n else (3 * n) // 4
        summary.append(f"\n## Part 2: probes from iteration {probe_iter}, {a.window} s window\n")
        for name in ("static", "dyn"):
            exe, env, extra = next((e, v, x) for n_, e, v, x in CONFIGS if n_ == name)
            print(f"[part 2] {name}: probes", flush=True)
            r = Run(f"{name}_probe", path_of[exe], env, args + extra, out, a.threads, probe_iter,
                    ["numastat", "stat", "c2c", "record"], a.window).run()
            perf_reports(r)
            summary.append(f"{name:8s} {stat_summary(r) or '(no perf stat)'}   -> {r.dir}")
            summary.append(f"{name:8s} samples: {symbol_shares(r) or '(no report, or no such symbol)'}")
            print(f"[part 2] {name}: allocator counters (CHUNK_LOG=...,arena=info)", flush=True)
            env2 = dict(env)
            env2["CHUNK_LOG"] = "warn,rechunk=debug,arena=info"
            r2 = Run(f"{name}_freepath", path_of[exe], env2, args + extra, out, a.threads).run()
            if r2.freepath:
                # cumulative counters: take the per-iteration increase of parked_foreign
                per_iter = {}
                for k, _, freed, own, foreign in r2.freepath:
                    per_iter[k] = (freed, own, foreign)
                ks = sorted(per_iter)
                lr = r2.last_rechunk() or 0
                deltas = [tuple((per_iter[k][j] - per_iter[p][j]) / (k - p) for j in range(3))
                          for p, k in zip(ks, ks[1:]) if p > lr]
                if deltas:
                    fr, po, pf = (statistics.mean(d[j] for d in deltas) for j in range(3))
                    summary.append(f"{name:8s} after iteration {lr}: per iteration freed {fr:.0f}, "
                                   f"parked_own {po:.0f}, parked_foreign {pf:.0f}")
            with open(r2.dir / "freepath.csv", "w") as f:
                f.write("iteration,when,freed,parked_own,parked_foreign\n")
                for row in r2.freepath:
                    f.write(",".join(map(str, row)) + "\n")

    text = "\n".join(summary) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
