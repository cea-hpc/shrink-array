# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""shrink_bench.sh's table: per (compiler, variant, threads), the medians over the
rounds of each mode, and each mode's change against mode 0.

Columns: compute = the compute phase (s), wall = the whole run (s), peak = shrink_bench's
peak RSS (MiB), kpeak = the kernel's (MiB, /usr/bin/time), retries = mode 7's retried
reads. Failed runs are counted and left out of the medians."""
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
data = defaultdict(lambda: defaultdict(list))
failed = defaultdict(int)
for row in rows:
    key = (row["comp"], row["variant"], int(row["threads"]), "m" + row["mode"])
    if row["status"] != "ok":
        failed[key] += 1
        continue
    for col in ("compute_s", "wall_s", "peak_mo", "kernel_peak_mo", "retries"):
        if row[col]:
            data[key][col].append(float(row[col]))


def med(key, col):
    vals = data.get(key, {}).get(col)
    return statistics.median(vals) if vals else None


def fmt(x, width, prec):
    return "%*s" % (width, "-") if x is None else "%*.*f" % (width, prec, x)


modes = sorted({k[3] for k in list(data) + list(failed)})
others = [m for m in modes if m != "m0"]
cols = (("compute_s", "compute", 2), ("wall_s", "wall", 1), ("peak_mo", "peak", 0), ("kernel_peak_mo", "kpeak", 0))
head = "%-5s %-14s %3s %-4s" % ("comp", "variant", "thr", "mode")
head += "".join(" %9s" % name for _, name, _ in cols) + " %9s %4s" % ("retries", "runs")
head += "".join(" %8s" % (name + "%") for _, name, _ in cols[:3])
print(head)
for comp in sorted({k[0] for k in data}):
    for v in ("zero", "zero_or_value", "zero_any"):
        for t in sorted({k[2] for k in data}):
            printed = False
            for m in modes:
                key = (comp, v, t, m)
                if key not in data and key not in failed:
                    continue
                printed = True
                line = "%-5s %-14s %3d %-4s" % (comp, v, t, m)
                line += "".join(" " + fmt(med(key, c), 9, p) for c, _, p in cols)
                line += " " + fmt(med(key, "retries"), 9, 0)
                n_ok = len(data.get(key, {}).get("wall_s", []))
                line += " %4s" % ("%d%s" % (n_ok, "!" if failed.get(key) else ""))
                if m in others:
                    for c, _, _ in cols[:3]:
                        b, x = med((comp, v, t, "m0"), c), med(key, c)
                        line += " %8s" % ("-" if not b or x is None else "%+.1f" % (100.0 * (x - b) / b))
                print(line)
            if printed:
                print()
if failed:
    print("failed runs (left out): " + ", ".join("%s %s %d thr %s: %d" % (k + (n,)) for k, n in sorted(failed.items())))
