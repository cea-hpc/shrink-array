# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Median ns/elem of the array per (compiler, variant, threads, case, mode), over every round and rep.

Columns: one per mode found (m0 announce, m1 none, m2 optimistic + fence, m3 optimistic +
address dependency, m4 no acquire at all, m7 optimistic + retry; suffix d = CHUNK_READ_DEP_ORDER=1,
i = CHUNK_RECHECK_INT_LOAD=1), then each mode's change against m0."""
import csv, glob, os, statistics, sys
from collections import defaultdict

d = sys.argv[1]
data = defaultdict(list)
for f in glob.glob(os.path.join(d, "*_m*_r*.csv")):
    comp, m, _ = os.path.basename(f)[:-4].split("_")
    for row in csv.DictReader(open(f)):
        if row["container"] == "array":
            data[(comp, row["variant"], int(row["threads"]), row["case"], m)].append(float(row["ns_per_elem"]))
modes = sorted({k[4] for k in data})
others = [m for m in modes if m != "m0"]
cases = ["read-varied", "read-constant", "read-random", "read-second", "read-bulk", "rmw", "write-varied"]
print("%-6s %-14s %3s %-14s " % ("comp", "variant", "thr", "case") + " ".join("%8s" % m for m in modes)
      + " " + " ".join("%7s" % (m + "%") for m in others))
for comp in sorted({k[0] for k in data}):
    for v in ("zero", "zero_or_value", "zero_any"):
        for t in sorted({k[2] for k in data}):
            for c in cases:
                med = {m: statistics.median(data[(comp, v, t, c, m)]) for m in modes if data.get((comp, v, t, c, m))}
                if "m0" not in med:
                    continue
                b = med["m0"]
                print("%-6s %-14s %3d %-14s " % (comp, v, t, c)
                      + " ".join("%8.3f" % med[m] if m in med else "%8s" % "-" for m in modes) + " "
                      + " ".join("%+6.0f%%" % (100 * (med[m] / b - 1)) if m in med else "%7s" % "-" for m in others))
