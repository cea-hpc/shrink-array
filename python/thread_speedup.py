#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""thread_speedup.py -- thread speedup curves of shrink_bench executables on one
build target (Grace, Milan, or this machine), whatever runs the threads: OpenMP
(omp-* targets), the std::thread backend (std-*), or TBB (shrink_bench_tbb_*,
by full name). The thread count is OMP_NUM_THREADS, which the std backend and
the TBB bench read too (CHUNK_NUM_THREADS, which they would read first, is taken
out of the runs' environment). Formerly omp_speedup.py, which still runs this one.

For each executable and each thread count from --min-threads to --max-threads
(powers of two, plus the max), --reps runs of

    OMP_NUM_THREADS=<t> OMP_PROC_BIND=close OMP_PLACES=cores \\
        <exe> --mode perf --no_io --multiply <M> [extra args]

interleaved (repetition outer, then threads, then executables: a drift of the
machine hits every point alike). The time of a point is the BEST of its
repetitions (the least disturbed one); the speedup is against the same
executable at --min-threads, and the efficiency is speedup * min / threads.

Time measured: "Elapsed time computation" printed by shrink_bench (--metric
compute, the default: the parallel loops, without the array creation and the
fill), or the wall time of the process (--metric wall).

Output, in --out (default $CCCWORKDIR/thread_speedup/
speedup_<target>_<host>_<date>/ when that work area exists, else
~/thread_speedup/speedup_<target>_<host>_<date>/):
  results.csv   one row per run: target, exe, threads, rep, wall_s, compute_s,
                mem_mo, rc
  summary.csv   one row per (exe, threads): best and median times, speedup,
                efficiency -- the curve
  plot.gp       gnuplot: speedup and efficiency against threads, one line per
                executable, the ideal line;  gnuplot plot.gp  ->  speedup.png
  speedup.png   the same, drawn here when matplotlib is installed
  meta.json     the command, host, CPU model, arguments
  errors.log    one entry per failed run (what shrink_gui's Erreurs tab shows)
  logs/         the full output of every run

Examples (from python/, on a compute node):
  python3 thread_speedup.py --target omp-llvm --min-threads 1 --max-threads 72 \\
      --exec shrink_bench_vector_perf shrink_bench_zero_any_dyn --multiply 2
  python3 thread_speedup.py --target omp-gnu --min-threads 4 --max-threads 128 \\
      --exec zero@dyn zero_any@1024 --threads 4 8 16 32 64 96 128 --reps 5

--exec takes a full name (shrink_bench_zero_any_dyn), or base@size
(zero_any@dyn -> shrink_bench_zero_any_dyn), or "vector" (shrink_bench_vector_perf),
as bench_and_test.py names them. --all-exec takes every one the target has
for this machine: vector, then zero / zero_one_value / zero_any in each size
(--size narrows it); with --int64 their 64-bit declinations
(shrink_bench_*_int64_*, the only ones past INT_MAX elements: --multiply 2
and up).

--mode perf (the default) is the measurement: 1e9 elements x --multiply,
--no_io. --mode check runs the correctness mode (5e6 elements, read back
against the config's oracle): its speedup says whether the check path
scales, not how fast the library is. It runs in the oracle directory that
bench_and_test.py generated for the target (--oracle-dir, default
<bench_results>/<arch>/oracle/cmake-build-<target>), at the size the oracle
was made: --multiply 1, no --no_io. The build root is --build-root, else
$CHUNK_BUILD_ROOT, else $CHUNK_BUILD_REP (scripts/site_env.sh), else the
repository; the executable is looked up under
<root>/cmake-build-<target>/<dir ending with this machine's arch>/examples/shrink_bench/.

Python 3.6 or later, standard library only (matplotlib optional).
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["zero", "zero_one_value", "zero_any"]          # as multi_launch_stress.py --all_exec
try:
    # --exec chunk2a@dyn & co.: the old variant labels, read as their new names until 1.0.
    from bench_runner.settings import normalize_variant
except ImportError:
    def normalize_variant(label, warn=None):   # copied alone: the new labels only
        return label
SIZES = ["dyn", "1024", "256", "64"]
ORACLE_MARKER = "data_tab_ORACLE_COMPLETE.marker"  # bench_runner/builds.py
REPO = os.path.dirname(HERE)
EXE_SUBPATH = os.path.join("examples", "shrink_bench")
COMPUTE_RE = re.compile(r"Elapsed time computation\s*:\s*([0-9.eE+-]+)\s*s")
MEM_RE = re.compile(r"Memory consumption[^:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*Mo")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ------------------------------------------------------------------ helpers
def exe_name(spec, mode="perf"):
    """zero_any@dyn -> shrink_bench_zero_any_dyn, vector -> shrink_bench_vector_perf
    (vector_check in check mode: the one binary built twice), a full name or a
    path as it is."""
    if os.sep in spec or spec.startswith("shrink_bench_"):
        return spec
    if spec == "vector":
        return "shrink_bench_vector_" + mode
    if "@" in spec:
        base, size = spec.split("@", 1)
        return "shrink_bench_%s_%s" % (base, size)
    return "shrink_bench_" + spec


def glob_execs(root, target, pattern):
    """[(name, path)] of the executables of the target matching glob @p pattern
    (shrink_gui's "Choisir..." gives some), for this machine."""
    import fnmatch
    cfg = target if target.startswith("cmake-build-") else "cmake-build-" + target
    arch = platform.machine()
    out = {}
    for d in sorted(glob.glob(os.path.join(root, cfg, "*", EXE_SUBPATH))):
        if not os.path.basename(os.path.dirname(os.path.dirname(d))).endswith(arch):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if fnmatch.fnmatchcase(f, pattern) and os.path.isfile(p) and os.access(p, os.X_OK):
                out.setdefault(f, p)
    return sorted(out.items())


def all_execs(root, target, a):
    """--all-exec: the executables the target has for this machine, vector
    first, then each variant in each size (--size), 32-bit or --int64."""
    specs = ["vector_int64_perf" if a.int64 else "vector"]
    for v in VARIANTS:
        for sz in a.size or SIZES:
            specs.append("%s%s@%s" % (v, "_int64" if a.int64 else "", sz))
    out = []
    for spec in specs:
        name = exe_name(spec if not spec.startswith("vector_int64") else "shrink_bench_" + spec, a.mode)
        exe = find_exe(root, target, name)
        if exe:
            out.append((name, exe))
    return out


def default_out_base():
    """Where the output directories go without --out: the CCC work area
    ($CCCWORKDIR) when there is one, else ~/thread_speedup -- never the source tree
    (shrink_gui runs it from python/)."""
    work = os.environ.get("CCCWORKDIR", "")
    return os.path.join(work if work and os.path.isdir(work) else os.path.expanduser("~"), "thread_speedup")


def default_oracle_dir(target):
    """bench_and_test.py's oracle of @p target: <output dir>/<arch>/oracle/<config>,
    its default output dir being $CCCWORKDIR/bench_results (else the cwd)."""
    work = os.environ.get("CCCWORKDIR", "")
    base = os.path.join(work, "bench_results") if work and os.path.isdir(work) else os.getcwd()
    return os.path.join(base, platform.machine(), "oracle", "cmake-build-" + target)


def build_root(arg):
    for v in (arg, os.environ.get("CHUNK_BUILD_ROOT"), os.environ.get("CHUNK_BUILD_REP")):
        if v and v != ".":
            return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))
    return REPO


def find_exe(root, target, spec):
    """The executable @p spec (a name exe_name gave, or a path) of build
    target @p target, for this machine's architecture, or None."""
    name = spec
    if os.sep in name:
        return os.path.abspath(name) if os.access(name, os.X_OK) else None
    cfg = target if target.startswith("cmake-build-") else "cmake-build-" + target
    arch = platform.machine()
    cands = sorted(glob.glob(os.path.join(root, cfg, "*", EXE_SUBPATH, name)))
    cands = [c for c in cands
             if os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))).endswith(arch)]
    cands += [os.path.join(root, cfg, EXE_SUBPATH, name), os.path.join(root, cfg, name)]
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def usable_cores():
    """The CPUs this process may run on, one per physical core (OMP_PLACES=cores)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1
    seen = set()
    for c in cpus:
        base = "/sys/devices/system/cpu/cpu%d/topology/" % c
        try:
            key = (open(base + "physical_package_id").read().strip(), open(base + "core_id").read().strip())
        except OSError:
            key = ("cpu", c)
        seen.add(key)
    return len(seen)


def cpu_model():
    try:
        out = subprocess.check_output(["lscpu"], universal_newlines=True, stderr=subprocess.DEVNULL)
        for ln in out.splitlines():
            if ln.startswith("Model name:"):
                return ln.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def thread_counts(a):
    if a.threads:
        ts = sorted(set(a.threads))
    else:
        ts, t = [], 1
        while t <= a.max_threads:
            if t >= a.min_threads:
                ts.append(t)
            t *= 2
        ts = sorted(set(ts + [a.min_threads, a.max_threads]))
    return [t for t in ts if a.min_threads <= t <= a.max_threads]


def numa_prefix(mode, threads):
    """numactl --interleave over the nodes the run spans (python/numa_launch.py,
    as bench_and_test.py does), when --numa auto."""
    if mode != "auto":
        return []
    try:
        sys.path.insert(0, HERE)
        import numa_launch
        return list(numa_launch.interleave_prefix({"OMP_NUM_THREADS": str(threads),
                                                   "OMP_PROC_BIND": "close"}))
    except Exception:
        return []


def why(rc):
    if rc is None or rc >= 0:
        return ""
    try:
        name = signal.Signals(-rc).name
    except ValueError:
        name = "signal %d" % -rc
    return name + (" (out of memory? see the log)" if rc == -9 else "")


class Events:
    """bench_runner/events.py's records, in $CHUNK_BENCH_EVENTS when set (run by
    shrink_gui): its progress, its Erreurs column and first-error notice, the
    Phases of its Résumé tab. Best effort, never raises."""

    def __init__(self):
        self.path = os.environ.get("CHUNK_BENCH_EVENTS") or None

    def emit(self, kind, **fields):
        if not self.path:
            return
        rec = {"t": round(time.time(), 3), "kind": kind}
        rec.update(fields)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            pass


def fmt_hms(s):
    s = int(s)
    return "%dh%02dm" % (s // 3600, s % 3600 // 60) if s >= 3600 else "%dm%02ds" % (s // 60, s % 60)


# --------------------------------------------------------------------- runs
def run_one(exe, threads, bench_args, a, log_path):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               OMP_PROC_BIND=os.environ.get("OMP_PROC_BIND", "close"),
               OMP_PLACES=os.environ.get("OMP_PLACES", "cores"))
    # The std backend (and the TBB bench) read CHUNK_NUM_THREADS before
    # OMP_NUM_THREADS: left in the environment, it would pin every point.
    env.pop("CHUNK_NUM_THREADS", None)
    cmd = numa_prefix(a.numa, threads) + [exe] + bench_args
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=a.workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=a.timeout or None)
        out, rc = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        out += "\n[thread_speedup] TIMEOUT after %ss\n" % a.timeout
        rc = "timeout"
    wall = time.time() - t0
    with open(log_path, "w") as f:
        f.write("# cmd: OMP_NUM_THREADS=%d %s\n# rc: %s  wall: %.3f s\n" % (threads, " ".join(cmd), rc, wall))
        f.write(out)
    text = ANSI_RE.sub("", out)
    m = COMPUTE_RE.findall(text)
    mem = MEM_RE.findall(text)
    err = [ln.strip() for ln in text.splitlines() if ln.strip().lower().startswith("error")]
    return {"wall_s": wall, "compute_s": float(m[-1]) if m else None,
            "mem_mo": max(float(x) for x in mem) if mem else None, "rc": rc,
            "error": " / ".join(err[:2])}


def summarize(rows, metric, tmin):
    """[(exe, threads, n_ok, best, median, best_wall, best_compute, mem, speedup, efficiency)]"""
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["exe"], r["threads"]), []).append(r)
    base = {}
    for (exe, t) in sorted(by, key=lambda k: (k[0], k[1])):
        ok = [r for r in by[(exe, t)] if r["rc"] == 0]
        vals = [r[metric] for r in ok if r[metric] is not None]
        if not vals and metric == "compute_s":     # not printed: the wall time then
            vals = [r["wall_s"] for r in ok]
        best = min(vals) if vals else None
        med = statistics.median(vals) if vals else None
        walls = [r["wall_s"] for r in ok]
        comps = [r["compute_s"] for r in ok if r["compute_s"] is not None]
        mems = [r["mem_mo"] for r in ok if r["mem_mo"] is not None]
        if t == tmin and best:
            base[exe] = best
        sp = base[exe] / best if best and exe in base else None
        eff = sp * tmin / t if sp else None
        out.append([exe, t, len(ok), best, med, min(walls) if walls else None, min(comps) if comps else None,
                    max(mems) if mems else None, sp, eff])
    return out


def write_csvs(outdir, a, rows, summary):
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "exe", "threads", "rep", "wall_s", "compute_s", "mem_mo", "rc"])
        for r in rows:
            w.writerow([a.target, r["exe"], r["threads"], r["rep"], "%.4f" % r["wall_s"],
                        "" if r["compute_s"] is None else "%.4f" % r["compute_s"],
                        "" if r["mem_mo"] is None else "%.1f" % r["mem_mo"], r["rc"]])
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "exe", "threads", "ok_runs", "best_s", "median_s", "best_wall_s",
                    "best_compute_s", "peak_mem_mo", "speedup", "efficiency"])
        for row in summary:
            w.writerow([a.target] + [row[0], row[1], row[2]] +
                       ["" if v is None else ("%.4f" % v) for v in row[3:7]] +
                       ["" if row[7] is None else "%.1f" % row[7]] +
                       ["" if v is None else ("%.3f" % v) for v in row[8:]])


def write_gnuplot(outdir, a, execs, tmin, tmax):
    """plot.gp: speedup and efficiency against threads, one line per exe."""
    lines = ['# gnuplot plot.gp  ->  speedup.png (from summary.csv)',
             'set datafile separator ","', 'set terminal pngcairo size 1400,560 font ",11" noenhanced',
             'set output "speedup.png"', 'set multiplot layout 1,2 title "%s -- %s (%s, --multiply %g, %s time)"'
             % (a.target, socket.gethostname(), a.mode, a.multiply, a.metric.split("_")[0]),
             'set key top left', 'set grid', 'set logscale x 2', 'set xlabel "threads"',
             'set xrange [%g:%g]' % (tmin / 1.2, tmax * 1.2),
             'set ylabel "speedup vs %d thread(s)"' % tmin, 'set logscale y 2']
    plots = ['x/%d title "ideal" with lines dashtype 2 lc rgb "#898781"' % tmin]
    for e in execs:
        plots.append("'summary.csv' using (strcol(2) eq \"%s\" ? $3 : NaN):10 title \"%s\" with linespoints pt 7"
                     % (e, e.replace("shrink_bench_", "")))
    lines.append("plot " + ", \\\n     ".join(plots))
    lines += ['unset logscale y', 'set key bottom left', 'set ylabel "efficiency"', 'set yrange [0:1.1]']
    plots = ['1 title "ideal" with lines dashtype 2 lc rgb "#898781"']
    for e in execs:
        plots.append("'summary.csv' using (strcol(2) eq \"%s\" ? $3 : NaN):11 title \"%s\" with linespoints pt 7"
                     % (e, e.replace("shrink_bench_", "")))
    lines.append("plot " + ", \\\n     ".join(plots))
    lines.append("unset multiplot")
    with open(os.path.join(outdir, "plot.gp"), "w") as f:
        f.write("\n".join(lines) + "\n")


def draw_png(outdir, a, summary, tmin):
    """speedup.png with matplotlib, when it is there (False otherwise)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    execs = list(dict.fromkeys(r[0] for r in summary))
    ts = sorted({r[1] for r in summary})
    ax1.plot(ts, [t / tmin for t in ts], "--", color="#898781", label="ideal")
    ax2.axhline(1.0, ls="--", color="#898781", label="ideal")
    for e in execs:
        pts = [(r[1], r[8], r[9]) for r in summary if r[0] == e and r[8]]
        if not pts:
            continue
        lab = e.replace("shrink_bench_", "")
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=2, ms=6, label=lab)
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], "o-", lw=2, ms=6, label=lab)
    for ax in (ax1, ax2):
        try:
            ax.set_xscale("log", base=2)
        except TypeError:                 # matplotlib < 3.3
            ax.set_xscale("log", basex=2)
        ax.set_xticks(ts)
        ax.set_xticklabels([str(t) for t in ts])
        ax.set_xlabel("threads")
        ax.grid(True, alpha=0.3)
    try:                                  # the ideal line stays a straight line
        ax1.set_yscale("log", base=2)
    except TypeError:
        ax1.set_yscale("log", basey=2)
    from matplotlib.ticker import ScalarFormatter
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower left")
    ax1.set_ylabel("speedup vs %d thread(s)" % tmin)
    ax2.set_ylabel("efficiency")
    ax2.set_ylim(0, 1.1)
    fig.suptitle("%s -- %s (%s, --multiply %g, %s time)" % (a.target, socket.gethostname(), a.mode, a.multiply,
                                                         a.metric.split("_")[0]))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "speedup.png"), dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OpenMP speedup curves of shrink_bench executables on one build target "
                    "(CSV + gnuplot script + PNG when matplotlib is there).",
        epilog="Everything after -- goes to the executables (e.g. -- --iter_max 500).")
    p.add_argument("--target", required=True,
                   help="build target (compile_all.sh config): omp-llvm, omp-gnu, omp-oneapi... "
                        "or cmake-build-omp-llvm")
    p.add_argument("--exec", dest="execs", nargs="+", metavar="EXE",
                   help="executables, space- or comma-separated: full name (shrink_bench_zero_any_dyn), "
                        "base@size (zero_any@dyn), vector (shrink_bench_vector_perf), a glob pattern "
                        "(shrink_bench_zero_any_*), or a path")
    p.add_argument("--all-exec", action="store_true",
                   help="every executable the target has here: vector, then zero / zero_one_value / "
                        "zero_any in each size (with --exec: added to them)")
    p.add_argument("--size", type=lambda v: [x for x in v.split(",") if x], default=None,
                   metavar="dyn,1024,...", help="--all-exec: these sizes only (default: %s)" % ",".join(SIZES))
    p.add_argument("--int64", action="store_true",
                   help="--all-exec: the 64-bit declinations (shrink_bench_*_int64_*), needed past INT_MAX "
                        "elements (--multiply 2 and up); perf mode only")
    p.add_argument("--mode", choices=["perf", "check"], default="perf",
                   help="perf (default): the measurement. check: the correctness mode, run in the target's "
                        "oracle directory, --multiply 1 (see --oracle-dir)")
    p.add_argument("--oracle-dir", default=None,
                   help="--mode check: the directory holding the target's data_tab* reference (default "
                        "$CCCWORKDIR/bench_results/<arch>/oracle/cmake-build-<target>)")
    p.add_argument("--min-threads", type=int, required=True, help="first thread count (speedup reference)")
    p.add_argument("--max-threads", type=int, required=True, help="last thread count")
    p.add_argument("--threads", type=int, nargs="+",
                   help="explicit thread counts (kept between min and max); default: the powers of "
                        "two between min and max, plus min and max")
    p.add_argument("--multiply", type=float, default=1.0, help="--multiply of the workload (default 1)")
    p.add_argument("--reps", type=int, default=3, help="runs per point, the best kept (default 3)")
    p.add_argument("--metric", choices=["compute", "wall"], default="compute",
                   help="time of a point: shrink_bench's 'Elapsed time computation' (default) or the "
                        "process wall time")
    p.add_argument("--numa", choices=["off", "auto"], default="off",
                   help="auto: numactl --interleave over the nodes a run spans (numa_launch.py); "
                        "off (default): first touch, as the scalability guard")
    p.add_argument("--timeout", type=float, default=0, help="seconds per run, 0 = none (default)")
    p.add_argument("--build-root", default=None,
                   help="holds cmake-build-<target>/ (default $CHUNK_BUILD_ROOT, $CHUNK_BUILD_REP, "
                        "the repository)")
    p.add_argument("--out", default=None, help="output directory (default <CCC work or ~>/thread_speedup/speedup_<target>_<host>_<date>)")
    p.add_argument("--allow-oversubscribe", action="store_true",
                   help="keep the thread counts above the usable cores (dropped by default)")
    p.add_argument("--dry-run", action="store_true", help="print the runs and exit")
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    a.extra = extra
    a.metric = a.metric + "_s"
    a.execs = [x for e in a.execs or [] for x in e.split(",") if x]
    if not a.execs and not a.all_exec:
        p.error("give --exec EXE [EXE ...] and/or --all-exec")
    if a.mode == "check":
        if a.multiply != 1.0:
            p.error("--mode check: the oracle was generated at the default size, --multiply must stay 1")
        if a.int64:
            p.error("--int64: perf mode only (the int64 binaries have no check reference)")
    bad = [x for x in a.size or [] if x not in SIZES]
    if bad:
        p.error("--size: %s (valid: %s)" % (", ".join(bad), ", ".join(SIZES)))
    if a.min_threads < 1 or a.max_threads < a.min_threads:
        p.error("need 1 <= --min-threads <= --max-threads")
    if a.reps < 1:
        p.error("--reps must be >= 1")
    if a.multiply <= 0:
        p.error("--multiply must be > 0")
    return a


def main(argv=None):
    a = parse_args(argv)
    root = build_root(a.build_root)
    target = a.target[len("cmake-build-"):] if a.target.startswith("cmake-build-") else a.target
    a.target = target
    execs, missing = [], []
    for spec in a.execs or []:
        if any(c in spec for c in "*?[") and os.sep not in spec:
            hits = glob_execs(root, target, spec)
            execs += [h for h in hits if h[0] not in dict(execs)]
            if not hits:
                missing.append(spec)
            continue
        if os.sep not in spec and not spec.startswith("shrink_bench_"):
            spec = normalize_variant(spec, "--exec")      # chunk2a@dyn -> zero_any@dyn
        name = exe_name(spec, a.mode)
        exe = find_exe(root, target, name)
        (execs.append((name if os.sep not in name else os.path.basename(name), exe)) if exe
         else missing.append(name))
    if a.all_exec:
        found = all_execs(root, target, a)
        if not found:
            sys.exit("--all-exec: no shrink_bench executable for %s (%s) under %s/cmake-build-%s"
                     % (target, platform.machine(), root, target))
        execs += [e for e in found if e[0] not in dict(execs)]
    if missing:
        sys.exit("not found for %s (%s) in %s/cmake-build-%s/*%s/%s:\n  %s"
                 % (target, platform.machine(), root, target, platform.machine(), EXE_SUBPATH,
                    "\n  ".join(missing)))
    ts = thread_counts(a)
    cores = usable_cores()
    if not a.allow_oversubscribe:
        over = [t for t in ts if t > cores]
        ts = [t for t in ts if t <= cores]
        if over:
            print("note: %d usable cores here: %s thread(s) dropped (--allow-oversubscribe keeps them)"
                  % (cores, over))
    if not ts or ts[0] != a.min_threads:
        sys.exit("--min-threads %d cannot run here (%d usable cores)" % (a.min_threads, cores))
    if a.mode == "perf":
        bench_args = ["--mode", "perf", "--no_io", "--multiply", "%g" % a.multiply] + a.extra
        a.workdir = None
    else:
        # The check mode reads the oracle in its cwd, at its default size.
        bench_args = ["--mode", "check"] + a.extra
        a.workdir = os.path.abspath(os.path.expanduser(a.oracle_dir or default_oracle_dir(target)))
        if not os.path.isfile(os.path.join(a.workdir, ORACLE_MARKER)):
            sys.exit("--mode check: no complete oracle in %s (%s missing): generate it with "
                     "bench_and_test.py --configs cmake-build-%s (or --regen-ref), or give --oracle-dir"
                     % (a.workdir, ORACLE_MARKER, target))
    host = socket.gethostname().split(".")[0]
    outdir = os.path.abspath(a.out or os.path.join(
        default_out_base(), "speedup_%s_%s_%s" % (target, host, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    plan = [(r, t, name, exe) for r in range(1, a.reps + 1) for t in ts for name, exe in execs]
    print("%s on %s (%s, %d usable cores): %d run(s) -- threads %s, %s"
          % (target, host, cpu_model(), cores, len(plan), ts, " ".join(n for n, _ in execs)))
    print("args: " + " ".join(shlex.quote(x) for x in bench_args) + ("   numactl: auto" if a.numa == "auto" else ""))
    if a.dry_run:
        for r, t, name, exe in plan:
            print("rep %d  %4d threads  %s" % (r, t, exe))
        return 0
    logs = os.path.join(outdir, "logs")
    os.makedirs(logs, exist_ok=True)
    print("📁 Outputs in %s" % outdir, flush=True)       # what shrink_gui reads (out_re)
    ev = Events()
    ev.emit("start", total_runs=len(plan), target=target, execs=[n for n, _ in execs], threads=ts)
    ev.emit("phase_start", phase="speedup")
    errors_log = os.path.join(outdir, "errors.log")
    open(errors_log, "w").close()
    if a.workdir is None:
        a.workdir = outdir      # the binaries write memory_mo.txt and friends in their cwd
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"command": " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv),
                   "host": socket.gethostname(), "cpu": cpu_model(), "arch": platform.machine(),
                   "usable_cores": cores, "target": target, "build_root": root,
                   "execs": dict(execs), "threads": ts, "reps": a.reps, "multiply": a.multiply,
                   "bench_args": bench_args, "mode": a.mode, "cwd": a.workdir, "metric": a.metric,
                   "numa": a.numa,
                   "slurm_job": os.environ.get("SLURM_JOB_ID"),
                   "date": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    rows, t_start, n_fail = [], time.time(), 0
    interrupted = False
    try:
        for i, (rep, t, name, exe) in enumerate(plan, 1):
            log = os.path.join(logs, "%s__omp%d__r%d.log" % (name, t, rep))
            r = run_one(exe, t, bench_args, a, log)
            r.update(exe=name, threads=t, rep=rep)
            rows.append(r)
            done = time.time() - t_start
            eta = done / i * (len(plan) - i)
            tm = r["compute_s"] if a.metric == "compute_s" and r["compute_s"] is not None else r["wall_s"]
            status = ("%8.3f s" % tm) if r["rc"] == 0 else "FAIL rc=%s %s" % (r["rc"], why(r["rc"]) if isinstance(r["rc"], int) else "")
            if r["rc"] != 0:
                n_fail += 1
                if r["error"]:
                    status += "  -- " + r["error"][:200]
                head = "RUN_FAIL: %s @ %dth rep=%d rc=%s %s" % (name, t, rep, r["rc"],
                                                               why(r["rc"]) if isinstance(r["rc"], int) else "")
                with open(errors_log, "a") as f:
                    f.write(head.rstrip() + "\n" + ("  " + r["error"] + "\n" if r["error"] else "")
                            + "  full output: %s\n" % log)
                ev.emit("error", phase="speedup", error_kind="RUN_FAIL", summary=head.rstrip()[:300])
            ev.emit("progress", phase="speedup", done=i, total=len(plan))
            print("[%d/%d] rep %d %4d th  %-28s %s%s   ETA %s (%d s)"
                  % (i, len(plan), rep, t, name.replace("shrink_bench_", ""), status,
                     "  %.0f Mo" % r["mem_mo"] if r["mem_mo"] else "", fmt_hms(eta), eta), flush=True)
            write_csvs(outdir, a, rows, summarize(rows, a.metric, ts[0]))   # usable even if stopped
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: the CSVs hold the runs done so far")
    ev.emit("phase_end", phase="speedup", status="interrupted" if interrupted else ("errors" if n_fail else "ok"),
            errors=n_fail, seconds=round(time.time() - t_start, 1))
    ev.emit("end", errors=n_fail, interrupted=interrupted, minutes=round((time.time() - t_start) / 60, 2))
    summary = summarize(rows, a.metric, ts[0])
    write_csvs(outdir, a, rows, summary)
    write_gnuplot(outdir, a, [n for n, _ in execs], ts[0], ts[-1])
    png = draw_png(outdir, a, summary, ts[0])
    print("\n%-28s %7s %10s %10s %8s %6s" % ("exe", "threads", "best s", "median s", "speedup", "eff."))
    for row in summary:
        print("%-28s %7d %10s %10s %8s %6s" % (
            row[0].replace("shrink_bench_", ""), row[1],
            "%.3f" % row[3] if row[3] else "-", "%.3f" % row[4] if row[4] else "-",
            "%.2f" % row[8] if row[8] else "-", "%.2f" % row[9] if row[9] else "-"))
    print("\n%s/summary.csv (the curve), results.csv (every run), plot.gp%s, logs/"
          % (outdir, ", speedup.png" if png else " (gnuplot plot.gp -> speedup.png)"))
    if n_fail:
        print("%d run(s) failed: see logs/ (their points are left out)" % n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
