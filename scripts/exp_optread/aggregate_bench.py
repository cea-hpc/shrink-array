# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""shrink_bench.sh's table: per (compiler, variant, threads), the medians over the
rounds of each mode, and each mode's change against mode 0 of the same run-time
setting (ENVS). A setting other than "default" shows as m<mode>@<setting>, with its
change against "default" of the same mode in the env columns.

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
    env = row.get("env") or "default"  # CSVs from before ENVS have no env column
    key = (row["comp"], row["variant"], int(row["threads"]), "m" + row["mode"], env)
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
envs = sorted({k[4] for k in list(data) + list(failed)}, key=lambda e: (e != "default", e))
with_env = envs != ["default"]


def pct(b, x):
    return "-" if not b or x is None else "%+.1f" % (100.0 * (x - b) / b)


cols = (("compute_s", "compute", 2), ("wall_s", "wall", 1), ("peak_mo", "peak", 0), ("kernel_peak_mo", "kpeak", 0))
mw = max([4] + [len(m) + (len(e) + 1 if e != "default" else 0) for m in modes for e in envs])
head = "%-5s %-14s %3s %-*s" % ("comp", "variant", "thr", mw, "mode")
head += "".join(" %9s" % name for _, name, _ in cols) + " %9s %4s" % ("retries", "runs")
head += "".join(" %8s" % (name + "%") for _, name, _ in cols[:3])
if with_env:
    head += " %9s %9s" % ("envcomp%", "envwall%")
print(head)
for comp in sorted({k[0] for k in data}):
    for v in ("zero", "zero_or_value", "zero_any"):
        for t in sorted({k[2] for k in data}):
            printed = False
            for e in envs:
              for m in modes:
                key = (comp, v, t, m, e)
                if key not in data and key not in failed:
                    continue
                printed = True
                label = m if e == "default" else m + "@" + e
                line = "%-5s %-14s %3d %-*s" % (comp, v, t, mw, label)
                line += "".join(" " + fmt(med(key, c), 9, p) for c, _, p in cols)
                line += " " + fmt(med(key, "retries"), 9, 0)
                n_ok = len(data.get(key, {}).get("wall_s", []))
                line += " %4s" % ("%d%s" % (n_ok, "!" if failed.get(key) else ""))
                for c, _, _ in cols[:3]:
                    line += " %8s" % ("" if m == "m0" else pct(med((comp, v, t, "m0", e), c), med(key, c)))
                if with_env and e != "default":
                    for c in ("compute_s", "wall_s"):
                        line += " %9s" % pct(med((comp, v, t, m, "default"), c), med(key, c))
                print(line.rstrip())
            if printed:
                print()
if failed:
    print("failed runs (left out): " + ", ".join("%s %s %d thr %s %s: %d" % (k + (n,)) for k, n in sorted(failed.items())))
