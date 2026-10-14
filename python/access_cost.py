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
          [--variants chunk,chunk2,chunk2a] [--rounds 3] [--n N] [--reps R]
          [--csv out.csv]

The build dir is the configured tree (a perf configuration: Release, no CHECK);
the binaries are looked for under examples/access_cost. The runs are pinned
(OMP_PROC_BIND=close, OMP_PLACES=cores) unless the environment sets them.
"""
import argparse
import csv
import io
import os
import pathlib
import statistics
import subprocess
import sys

CASES = ["read-varied", "read-constant", "read-random", "write-varied", "write-fresh", "rmw"]


def find_binary(build, variant):
    d = pathlib.Path(build) / "examples" / "access_cost"
    found = sorted(d.glob("access_cost_%s_*" % variant))
    found = [p for p in found if p.is_file() and os.access(p, os.X_OK) and p.name.split("_")[-1].isdigit()]
    return found[0] if found else None


def run_one(binary, threads, n, reps):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    cmd = [str(binary), "--csv", "--reps", str(reps)] + (["--n", str(n)] if n else [])
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write("%s (threads=%d) failed, code %d:\n%s\n" % (binary.name, threads, res.returncode,
                                                                   res.stderr[-3000:]))
        return []
    lines = [ln for ln in res.stdout.splitlines() if ln.count(",") == 8]   # the CSV, not the stats lines
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="the configured build tree")
    ap.add_argument("--threads", default="1,%d" % (os.cpu_count() or 1), help="thread counts, comma-separated")
    ap.add_argument("--variants", default="chunk,chunk2,chunk2a")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="elements (default: the binary's, 2^27)")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per run")
    ap.add_argument("--csv", help="write every repetition here")
    a = ap.parse_args()

    threads = [int(t) for t in a.threads.split(",") if t.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    binaries = {}
    for v in variants:
        b = find_binary(a.build, v)
        if b is None:
            sys.stderr.write("no access_cost_%s_* under %s/examples/access_cost (built?)\n" % (v, a.build))
        else:
            binaries[v] = b
    if not binaries:
        return 1

    rows = []
    for r in range(a.rounds):
        for t in threads:
            for v, b in binaries.items():
                print("round %d/%d  %s  threads=%d" % (r + 1, a.rounds, b.name, t), file=sys.stderr, flush=True)
                for row in run_one(b, t, a.n, a.reps):
                    row["round"] = str(r)
                    rows.append(row)
    if not rows:
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
                print("%-8s %7d %-14s %10.3f %10.3f %7.2f %6.0f%%" % (v, t, c, vm, am, am / vm if vm else 0.0,
                                                                     100 * spread))
    print("(medians over %d round(s) x %d repetition(s); ns per element = wall clock / n; "
          "ratio = array / vector)" % (a.rounds, a.reps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
