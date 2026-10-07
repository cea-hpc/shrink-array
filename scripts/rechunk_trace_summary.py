#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Sum the CHUNK_RECHUNK_TRACE phase times of shrink_bench logs, one line per log.

For each log: the number of re-chunks, the bench's "Elapsed time" lines, then
each phase of the `[RECHUNK] a -> b phases (ms)` and `slices` lines summed over
every re-chunk of the run, in seconds. Comparing two thread counts shows which
phase grows.

Usage: scripts/rechunk_trace_summary.py <log>...
       e.g. scripts/rechunk_trace_summary.py recycle_1007_1133/omp-llvm_{64,128}th_rec1_r1.log
"""

import collections
import os
import re
import sys

LINE = re.compile(r"\[RECHUNK\] \d+ -> \d+ (phases|slices \d+) \(ms\): (.*)")


def summary(path):
    totals = collections.OrderedDict()
    n = 0
    elapsed = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("Elapsed time"):
                elapsed.append(re.sub(r"\s+", " ", line.strip()[len("Elapsed time "):]))
            m = LINE.match(line)
            if not m:
                continue
            n += m.group(1) == "phases"
            for part in m.group(2).split("|"):
                key, value = part.strip().rsplit(" ", 1)
                key = key.replace("of which ", "  ")
                totals[key] = totals.get(key, 0.0) + float(value)
    return n, elapsed, totals


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.split("\n\n")[-1], file=sys.stderr)
        return 1
    for path in sys.argv[1:]:
        n, elapsed, totals = summary(path)
        print(f"== {os.path.basename(path)}: {n} rechunk(s)  |  " + "  ;  ".join(elapsed))
        if totals:
            print("   " + "  ".join(f"{k}={v / 1000:.2f}s" for k, v in totals.items()))
        else:
            print("   (no [RECHUNK] phase line: not run with CHUNK_RECHUNK_TRACE=1?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
