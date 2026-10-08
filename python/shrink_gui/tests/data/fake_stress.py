# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for multi_launch_stress.py in the tests: same log lines (progress,
output directory) and same timings_summary.csv columns, runs in milliseconds.
Deterministic: --speed scales every time, so two runs compare exactly."""
import argparse
import csv
import os
import random
import sys
import time

p = argparse.ArgumentParser(description="fake")
p.add_argument("--runs", type=int, default=3)
p.add_argument("--step", type=float, default=0.01)
p.add_argument("--rc", type=int, default=0, help="exit code of the last run")
p.add_argument("--fail_every", type=int, default=0)
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--quiet", action="store_true")
a = p.parse_args()
random.seed(7)
od = os.path.join(os.environ["HOME"], "mlr", "aarch64_%d" % os.getpid())
os.makedirs(od, exist_ok=True)
os.chdir(od)
print("📁 Outputs, CSV and log in %s" % od)
builds = ["omp-gnu", "omp-llvm", "omp-gnu-std-alloc"]
exes = ["shrink_bench_%s_%s" % (v, s) for v in ("chunk", "chunk2", "chunk2a") for s in ("256", "1024", "dyn")]
with open("timings_summary.csv", "w", newline="") as f:
    csv.writer(f).writerow(["run_index", "build", "executable", "num_threads", "wall_seconds",
                            "returncode", "stdout_file", "stderr_file"])
for i in range(a.runs):
    b = builds[i % 3]
    e = random.choice(exes)
    th = random.randint(40, 200)
    base = {"chunk": 6.0, "chunk2": 4.5, "chunk2a": 4.0}[e.split("_")[2]] * (1.15 if "std" in b else 1.0)
    wall = a.speed * base * (0.25 + 40.0 / th) * random.uniform(0.85, 1.25)
    rc = 1 if a.fail_every and (i + 1) % a.fail_every == 0 else (a.rc if i == a.runs - 1 else 0)
    print("⏳ Cycle %d: Elapsed Time: 1.00s | Estimated Time Left: %.2fs" % (i, (a.runs - i) * a.step))
    print("🏗️  Run %d/%d on build %s (1/3 of the cycle)" % (i + 1, a.runs, b))
    time.sleep(a.step)
    with open("timings_summary.csv", "a", newline="") as f:
        csv.writer(f).writerow([i + 1, b, e, th, "%.6f" % wall, rc, "", ""])
open("multi_launch_output.log", "w").write("log\n")
sys.exit(a.rc)
