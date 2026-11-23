#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Run examples/access_cost over variants, thread counts and rounds, and print
the time per element of each access pattern on a std::vector and on the array.

Each access_cost_<variant>_<size> binary times the vector and the array itself,
alternating which goes first: the ratio compares the two in the same binary and
the same time window (docs/performance_measurement.md). This script runs the
binaries ROUND by round -- every variant once per round, at every thread count
-- so a drift of the machine over the session spreads over all of them, then
takes the median of every repetition of every round.

  python3 python/access_cost.py --build <build dir> [--threads 1,64]
          [--variants zero,zero_or_value,zero_any] [--rounds 3] [--n N] [--reps R]
          [--chunk-size S] [--csv out.csv]

The build dir is the configured tree (a perf configuration: Release, no CHECK);
the binaries are looked for under examples/access_cost, at the build's
CHUNK_DEFAULT_SIZE unless --chunk-size says otherwise. A run that fails or
prints no CSV is listed under the table, and the exit code is then 1. The runs
are pinned (OMP_PROC_BIND=close, OMP_PLACES=cores) unless the environment sets
them.
"""
import argparse
import csv
import io
import os
import pathlib
import statistics
import subprocess
import sys

try:
    # The old variant labels: refused on the command line, read as the new names
    # in the CSV rows of earlier runs.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_runner.settings import check_variant_label, normalize_variant
except ImportError:
    def check_variant_label(label, option):   # copied alone: the new labels only
        return label

    def normalize_variant(label):
        return label

CASES = ["read-varied", "read-constant", "read-random", "write-varied", "write-fresh", "rmw", "read-second", "read-bulk"]


def cached_default_size(build):
    """CHUNK_DEFAULT_SIZE of the build's CMakeCache.txt (the size access_cost is built at), or None."""
    try:
        with open(pathlib.Path(build) / "CMakeCache.txt") as f:
            for ln in f:
                if ln.startswith("CHUNK_DEFAULT_SIZE:"):
                    value = ln.split("=", 1)[1].strip()
                    return int(value) if value.isdigit() else None
    except OSError:
        pass
    return None


def find_binary(build, variant, size):
    """The access_cost_<variant>_<size> binary to run, or None.

    By chunk size as a NUMBER: sorted as strings, _1024 came before _256, and a
    tree holding both (a binary left by an earlier configure) ran 1024 unsaid.
    @p size: the one asked for (--chunk-size), else the build's
    CHUNK_DEFAULT_SIZE, else the only one there, else the smallest (with a note).
    """
    d = pathlib.Path(build) / "examples" / "access_cost"
    prefix = "access_cost_%s_" % variant
    # The size only after the prefix: access_cost_zero_* also lists zero_any's binaries.
    by_size = {int(p.name[len(prefix):]): p for p in d.glob(prefix + "*")
               if p.name[len(prefix):].isdigit() and p.is_file() and os.access(p, os.X_OK)}
    if not by_size:
        return None
    sizes = sorted(by_size)
    if size is not None:
        if size not in by_size:
            sys.stderr.write("access_cost_%s_%d not under %s (sizes there: %s)\n"
                             % (variant, size, d, ",".join(map(str, sizes))))
            return None
        return by_size[size]
    default = cached_default_size(build)
    if default in by_size:
        return by_size[default]
    if default is not None:
        sys.stderr.write("note: no access_cost_%s_%d (the build's CHUNK_DEFAULT_SIZE), running %d (sizes there: "
                         "%s)\n" % (variant, default, sizes[0], ",".join(map(str, sizes))))
    elif len(sizes) > 1:
        sys.stderr.write("note: access_cost_%s_* at sizes %s, running %d (choose with --chunk-size)\n"
                         % (variant, ",".join(map(str, sizes)), sizes[0]))
    return by_size[sizes[0]]


def run_one(binary, threads, n, reps):
    """(rows, None), or ([], why) for a run that gave no row: main() counts those."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    cmd = [str(binary), "--csv", "--reps", str(reps)] + (["--n", str(n)] if n else [])
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write("%s (threads=%d) failed, code %d:\n%s\n" % (binary.name, threads, res.returncode,
                                                                   res.stderr[-3000:]))
        return [], "exit code %d" % res.returncode
    # The CSV, not the stats lines: the header and the lines of as many fields
    # (an older binary has no passes column: its header says so).
    out = res.stdout.splitlines()
    header = next((ln for ln in out if ln.startswith("variant,chunk_size,")), None)
    if header is None:
        sys.stderr.write("%s (threads=%d): no CSV header in its output\n" % (binary.name, threads))
        return [], "no CSV in its output"
    lines = [ln for ln in out if ln.count(",") == header.count(",")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    return rows, (None if rows else "a CSV header and no row")


def report_failures(failures, missing):
    """List the runs that gave no row and the variants with no binary, under the table."""
    if missing:
        print("MISSING: no binary for variant(s) %s" % ", ".join(missing))
    if failures:
        print("FAILED: %d run(s) gave no row (their medians are over fewer rounds, or absent):" % len(failures))
        for r, t, name, why in failures:
            print("  round %d  %s  threads=%d: %s" % (r, name, t, why))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="the configured build tree")
    ap.add_argument("--threads", default="1,%d" % (os.cpu_count() or 1), help="thread counts, comma-separated")
    ap.add_argument("--variants", default="zero,zero_or_value,zero_any")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="elements (default: the binary's, max(2^27, threads x 2^20))")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per run")
    ap.add_argument("--chunk-size", type=int, help="binaries of this chunk size (default: the build's "
                    "CHUNK_DEFAULT_SIZE)")
    ap.add_argument("--csv", help="write every repetition here")
    a = ap.parse_args()

    threads = [int(t) for t in a.threads.split(",") if t.strip()]
    # The CPUs this process may run on (the job's cgroup, taskset), not the
    # node's: a job given one CPU ran 72 and 288 threads on it -- every case
    # at the 1-thread time per element, the rows meaningless. Said before and
    # in the table.
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    over = [t for t in threads if t > cpus]
    if over:
        sys.stderr.write("WARNING: %d CPU(s) available to this process (affinity), fewer than the thread "
                         "count(s) %s: those runs share the CPUs, their times per element are not the "
                         "threads' (ask the job for the cores: e.g. -c %d)\n"
                         % (cpus, ",".join(map(str, over)), max(threads)))
    variants = [check_variant_label(v.strip(), "--variants") for v in a.variants.split(",") if v.strip()]
    binaries = {}
    missing = []
    for v in variants:
        b = find_binary(a.build, v, a.chunk_size)
        if b is None:
            if a.chunk_size is None:
                sys.stderr.write("no access_cost_%s_* under %s/examples/access_cost (built?)\n" % (v, a.build))
            missing.append(v)
        else:
            binaries[v] = b
    if not binaries:
        return 1

    rows = []
    failures = []  # (round, threads, binary, why): a run that gave no row
    for r in range(a.rounds):
        for t in threads:
            for v, b in binaries.items():
                print("round %d/%d  %s  threads=%d" % (r + 1, a.rounds, b.name, t), file=sys.stderr, flush=True)
                got, why = run_one(b, t, a.n, a.reps)
                if why is not None:
                    failures.append((r + 1, t, b.name, why))
                for row in got:
                    row["round"] = str(r)
                    # An older binary prints its old label (chunk2a): read as the new one.
                    row["variant"] = normalize_variant(row.get("variant"))
                    rows.append(row)
    if not rows:
        report_failures(failures, missing)
        return 1

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("%-8s %7s %-14s %10s %10s %7s %7s" % ("variant", "threads", "case", "vector ns", "array ns",
                                                "ratio", "spread"))
    for v in binaries:
        for t in threads:
            for c in CASES:
                sel = [x for x in rows if x["variant"] == v and int(x["threads"]) == t and x["case"] == c]
                vec = [float(x["ns_per_elem"]) for x in sel if x["container"] == "vector"]
                arr = [float(x["ns_per_elem"]) for x in sel if x["container"] == "array"]
                if not vec or not arr:
                    continue
                vm, am = statistics.median(vec), statistics.median(arr)
                # Spread of the array's repetitions: (max - min) / median.
                spread = (max(arr) - min(arr)) / am if am > 0 else 0.0
                print("%-8s %7d %-14s %10.3f %10.3f %7.2f %6.0f%%%s"
                      % (v, t, c, vm, am, am / vm if vm else 0.0, 100 * spread,
                         "  OVERSUBSCRIBED (%d CPUs)" % cpus if t > cpus else ""))
    print("(medians over %d round(s) x %d repetition(s); ns per element = wall clock / n; "
          "ratio = array / vector; %d CPU(s) available)" % (a.rounds, a.reps, cpus))
    print("(binaries: %s)" % ", ".join(b.name for b in binaries.values()))
    # Under the table, not only in the stderr scroll above it: a row missing
    # from the table, or a median over fewer rounds, is otherwise unsaid.
    report_failures(failures, missing)
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
