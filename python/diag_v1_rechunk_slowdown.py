#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""diag_v1_rechunk_slowdown.py -- why does dynamic v1 run slower after its
re-chunk than static 1024 (docs/adaptive_rechunk.md, "Known limitations")?

Read-only diagnosis: nothing in the library changes, every factor is switched
by an environment variable the library already reads. Run it INSIDE a job
allocation on one compute node (ccc_msub / salloc), from anywhere:

    python3 diag_v1_rechunk_slowdown.py --bindir <build>/examples/shrink_bench \\
        --threads 64 --out /ccc/work/.../diag_v1 [--perf] [--reps 2]

shrink_bench runs in --mode perf (large sizes, --iter_max iterations; the
default check mode stops at 33 iterations of a small problem and never
re-chunks) unless --args names a --mode.

Part 1 -- A/B on the rebuild's state (always). shrink_bench_chunk_1024
(static) and shrink_bench_chunk_dyn under:
    dyn            the default
    dyn_1slice     CHUNK_RECHUNK_SLICES=1
    dyn_static1    CHUNK_RECHUNK_SLICES=1 CHUNK_RECHUNK_SCHEDULE=static
                   The default rebuild slices the array in 16 and spreads
                   each slice over ALL ranks (guided,16): a new chunk's
                   buffer is allocated, first-touched and owned (slab owner,
                   NUMA node) by an arbitrary rank, unrelated to the ranks
                   that use it afterwards. One pass in static gives each rank
                   one contiguous 1/T of the array instead. Neither is the
                   bench's own mapping (fill: one thread per array; compute:
                   schedule(static,1) blocks), so a gain here points at the
                   placement / ownership left by the rebuild, not a cure.
    dyn_norecycle  CHUNK_RECHUNK_RECYCLE=0       no slab recycling between
                   slices (recycled slabs keep the OLD geometry's pages, and
                   their NUMA placement)
    dyn_noreuse    CHUNK_RECHUNK_REUSE_LIST=0    no chunk_list buffer reuse
    dyn_lazy       CHUNK_ALLOC_AFFINITY_EAGER=0  per-array slab sets at the
                   first re-chunk, not at creation
  The time per iteration comes from the timestamps of the "iter <k> : time"
  lines (printed every 100 iterations; stdout line-buffered by stdbuf), so
  each 100-iteration interval gives a mean. An interval in which a
  [RECHUNK] line appeared is marked. The summary compares the intervals after
  the LAST re-chunk with the same intervals of the static run.

Part 2 -- probes during the slow phase (--perf), static and dyn only, at
iteration --probe-iter (default: 2 iterations after the last re-chunk of a
first dyn run, else 3/4 of the run):
    numastat -p        pages per NUMA node (placement)
    perf stat          task-clock / wall = busy cores, instructions / cycles
                       = IPC, context switches, page faults -- over --window s
    perf c2c           false sharing (HITM lines), same window
    perf record -g     where the idle cores are: libgomp barrier / lock
                       symbols versus library code
  plus one run of each with CHUNK_ARENA_REPORT=1: [FREEPATH] parked_foreign,
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

ITER_RE = re.compile(r"^iter (\d+) : time : ")
RECHUNK_RE = re.compile(r"^\[RECHUNK\] (\d+) -> (\d+) phases")
FREEPATH_RE = re.compile(r"\[FREEPATH\] (\S+)\s+freed=(\d+)\s+parked_own=(\d+)\s+parked_foreign=(\d+)")

CONFIGS = [
    ("static", "shrink_bench_chunk_1024", {}),
    ("dyn", "shrink_bench_chunk_dyn", {}),
    ("dyn_1slice", "shrink_bench_chunk_dyn", {"CHUNK_RECHUNK_SLICES": "1"}),
    ("dyn_static1", "shrink_bench_chunk_dyn", {"CHUNK_RECHUNK_SLICES": "1", "CHUNK_RECHUNK_SCHEDULE": "static"}),
    ("dyn_norecycle", "shrink_bench_chunk_dyn", {"CHUNK_RECHUNK_RECYCLE": "0"}),
    ("dyn_noreuse", "shrink_bench_chunk_dyn", {"CHUNK_RECHUNK_REUSE_LIST": "0"}),
    ("dyn_lazy", "shrink_bench_chunk_dyn", {"CHUNK_ALLOC_AFFINITY_EAGER": "0"}),
]


def have(tool):
    return shutil.which(tool) is not None


class Run:
    """One shrink_bench execution, its output timestamped line by line."""

    def __init__(self, name, exe, env_extra, args, outdir, threads, probe_iter=None, probes=None, window=10):
        self.name, self.exe, self.args = name, exe, args
        self.dir = Path(outdir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env.update({"OMP_NUM_THREADS": str(threads), "NO_IO": "1", "CHUNK_RECHUNK_TRACE": "1"})
        self.env.setdefault("OMP_PROC_BIND", "close")
        self.env.setdefault("OMP_PLACES", "cores")
        self.env.update(env_extra)
        self.probe_iter, self.probes, self.window = probe_iter, probes or [], window
        self.iter_t = {}  # iteration -> timestamp of its line
        self.rechunk_iters = set()
        self.freepath = []  # (iteration, when, freed, parked_own, parked_foreign)
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
                if self.probes and not probed and self.probe_iter is not None and cur_iter >= self.probe_iter:
                    probed = True
                    prober = threading.Thread(target=self._probe, args=(p.pid,))
                    prober.start()
                continue
            if RECHUNK_RE.search(line):
                self.rechunk_iters.add(cur_iter)  # the interval that ends at the next k line
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
            f.write("interval_end_k,s_per_iteration,rechunk_in_interval\n")
            ks = sorted(self.iter_t)
            for p, k in zip(ks, ks[1:]):
                dt = (self.iter_t[k] - self.iter_t[p]) / (k - p)
                f.write(f"{k},{dt:.4f},{int(p in self.rechunk_iters)}\n")

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
    ap.add_argument("--bindir", required=True, help="directory holding shrink_bench_chunk_dyn and _1024")
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
    summary = [f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  threads={a.threads}  args='{' '.join(args)}'  "
               f"OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND', 'close')} "
               f"OMP_PLACES={os.environ.get('OMP_PLACES', 'cores')}  host={os.uname().nodename}"]

    # ---- part 1
    runs = {}
    for rep in range(a.reps):
        for name, exe, env in CONFIGS:
            if name not in wanted:
                continue
            print(f"[part 1] {name} (rep {rep + 1}/{a.reps})", flush=True)
            r = Run(f"{name}_r{rep}", str(bindir / exe), env, args, out, a.threads).run()
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
            summary.append(f"WARNING: {name}: no re-chunk seen ([RECHUNK] a -> b phases) -- the comparison "
                           "below is then between two non-re-chunked runs")
    st = best_times(runs.get("static", []))
    summary.append("\n## Part 1: iteration wall time after the last re-chunk (vs static 1024, same iterations;"
                   " best of the repetitions)\n")
    summary.append(f"{'config':16s} {'last rechunk':>12s} {'intervals':>9s} {'s/iter':>9s} {'static':>9s} {'ratio':>7s}")
    for name, _, _ in CONFIGS:
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
            exe, env = next((e, v) for n_, e, v in CONFIGS if n_ == name)
            print(f"[part 2] {name}: probes", flush=True)
            r = Run(f"{name}_probe", str(bindir / exe), env, args, out, a.threads, probe_iter,
                    ["numastat", "stat", "c2c", "record"], a.window).run()
            perf_reports(r)
            summary.append(f"{name:8s} {stat_summary(r) or '(no perf stat)'}   -> {r.dir}")
            print(f"[part 2] {name}: allocator counters (CHUNK_ARENA_REPORT=1)", flush=True)
            env2 = dict(env)
            env2["CHUNK_ARENA_REPORT"] = "1"
            r2 = Run(f"{name}_freepath", str(bindir / exe), env2, args, out, a.threads).run()
            if r2.freepath:
                # cumulative counters: take the per-iteration increase of parked_foreign
                per_iter = {}
                for k, _, freed, own, foreign in r2.freepath:
                    per_iter[k] = (freed, own, foreign)
                ks = sorted(per_iter)
                lr = r2.last_rechunk() or 0
                deltas = [(k, (per_iter[k][2] - per_iter[p][2]) / (k - p), (per_iter[k][0] - per_iter[p][0]) / (k - p))
                          for p, k in zip(ks, ks[1:]) if p > lr]
                if deltas:
                    pf = statistics.mean(d[1] for d in deltas)
                    fr = statistics.mean(d[2] for d in deltas)
                    summary.append(f"{name:8s} after iteration {lr}: per iteration freed {fr:.0f}, "
                                   f"parked_foreign {pf:.0f}")
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
