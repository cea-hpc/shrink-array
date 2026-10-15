# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for bench_and_test.py in the tests: announces its output
directory as it does, writes bench_errors.log, failed_cases.jsonl (one
failure pointing to its full log, one not replayable) and run_logs/;
--rerun-failed replays the list, --rerun-repeat N times."""
import argparse
import json
import os

p = argparse.ArgumentParser(description="fake bench")
p.add_argument("--output-dir", default=None)
p.add_argument("--rerun-failed", nargs="?", const="", default=None)
p.add_argument("--rerun-repeat", type=int, default=1)
p.add_argument("--full-log-dir", default=None)
p.add_argument("--no-full-log", action="store_true")
p.add_argument("--speed", type=float, default=1.0, help="scales every time of bench_summary.csv")
a = p.parse_args()
od = os.path.join(a.output_dir or os.path.join(os.environ["HOME"], "bench_results"), "aarch64")
os.makedirs(od, exist_ok=True)
print("📁 Run outputs → %s  (arch: aarch64)" % od, flush=True)
logs = os.path.join(od, a.full_log_dir or "run_logs")
os.makedirs(logs, exist_ok=True)
cases = os.path.join(od, "failed_cases.jsonl")
if a.rerun_failed is not None:
    print("rerun %d" % a.rerun_repeat)
    log = os.path.join(logs, "shrink_bench_chunk2a__omp4__00001__rc-11.log")
    with open(log, "w") as f:
        f.write("# cmd: shrink_bench_chunk2a\n# ===== STDERR =====\nstill crashing\n")
    with open(cases, "w") as f:
        f.write(json.dumps({"kind": "CHECK_FAIL", "summary": "CHECK_FAIL: chunk2a t=4 code=-11",
                            "runs": [{"cmd": ["x"], "log": log}]}) + "\n")
    raise SystemExit(1)
log = os.path.join(logs, "shrink_bench_chunk2a__omp4__00003__rc-11.log")
with open(log, "w") as f:
    f.write("# cmd: shrink_bench_chunk2a\n# ===== STDERR =====\nSegmentation fault at chunk 17\n")
with open(os.path.join(logs, "shrink_bench_chunk__omp4__00001__ok.log"), "w") as f:
    f.write("fine\n")
with open(os.path.join(od, "bench_summary.csv"), "w") as f:
    f.write("Config,Variant,Threads,AvgTime_s,PeakMem_Mo,Speedup_vs_baseline,MemRatio_vs_baseline,"
            "MemSaved_Mo_vs_baseline\n")
    for cfg in ("omp-gnu", "omp-llvm"):
        for var, base in (("chunk", 4.0), ("chunk2", 3.0), ("chunk2a", 2.5)):
            for th in (1, 4, 16, 64):
                t = a.speed * base * (0.2 + 4.0 / th)
                sp = "" if var == "chunk" else "%.3f" % (4.0 / base)
                f.write("%s,%s,%d,%.6f,%.1f,%s,,\n" % (cfg, var, th, t, 100 * base, sp))
with open(os.path.join(od, "bench_errors.log"), "w") as f:
    f.write("CHECK_FAIL: chunk2a t=4 code=-11\n  stderr_tail:\n    Segmentation fault\n"
            "MEM_GUARD: chunk2a/chunk2 1.4 > 1.2\n")
with open(cases, "w") as f:
    f.write(json.dumps({"kind": "CHECK_FAIL", "summary": "CHECK_FAIL: chunk2a t=4 code=-11",
                        "runs": [{"cmd": ["x"], "log": log}]}) + "\n")
    f.write(json.dumps({"kind": "MEM_GUARD", "summary": "MEM_GUARD: chunk2a/chunk2 1.4 > 1.2",
                        "runs": []}) + "\n")
keep = os.environ.get("CHUNK_BENCH_KEEP_DIR")       # as bench_runner/report.py keep_run_outputs
if keep:
    import shutil
    for name in ("bench_summary.csv", "bench_errors.log", "failed_cases.jsonl"):
        shutil.copy2(os.path.join(od, name), os.path.join(keep, name))
raise SystemExit(1)
