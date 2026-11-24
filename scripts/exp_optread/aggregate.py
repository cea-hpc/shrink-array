# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Median ns/elem of the array per (compiler, variant, threads, case, mode), over every round and rep."""
import csv, glob, os, statistics, sys
from collections import defaultdict
d = sys.argv[1]
data = defaultdict(list)
for f in glob.glob(os.path.join(d, "*_m*_r*.csv")):
    comp, m, _ = os.path.basename(f)[:-4].split("_")
    for row in csv.DictReader(open(f)):
        if row["container"] == "array":
            data[(comp, row["variant"], int(row["threads"]), row["case"], m)].append(float(row["ns_per_elem"]))
cases = ["read-varied", "read-constant", "read-random", "read-second", "read-bulk", "rmw", "write-varied"]
print("%-6s %-14s %3s %-14s %8s %8s %8s %8s %8s" % ("comp", "variant", "thr", "case", "announce", "none", "optim", "none%", "optim%"))
for comp in sorted({k[0] for k in data}):
    for v in ("zero", "zero_or_value", "zero_any"):
        for t in sorted({k[2] for k in data}):
            for c in cases:
                med = {m: statistics.median(data[(comp, v, t, c, m)]) for m in ("m0", "m1", "m2") if data.get((comp, v, t, c, m))}
                if len(med) < 3:
                    continue
                b = med["m0"]
                print("%-6s %-14s %3d %-14s %8.3f %8.3f %8.3f %+7.0f%% %+7.0f%%" % (comp, v, t, c, b, med["m1"], med["m2"],
                      100 * (med["m1"] / b - 1), 100 * (med["m2"] / b - 1)))
