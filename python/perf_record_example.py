#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Measure one configuration N times and file the result in the perf database.

Standalone on purpose: bench_and_test.py already knows how to build the matrix
and parse output, and wiring this in there is a few lines (see the bottom of
this file). This script is the smaller thing you can run today, and it is what
the comparison workflow needs -- repetitions, not single shots.

    # take a reference on the current commit
    python3 perf_record_example.py perf.sqlite \\
        --exe .../shrink_bench_chunk2a_dyn \\
        --threads 64 --divisor 5 --iter-max 50 --reps 3 \\
        --config omp-gnu-std-alloc --variant chunk2a --geometry dyn \\
        --compiler gcc-16.1.0

    # ... change something, rebuild, run it again ...

    python3 perf_db.py compare perf.sqlite --base <old-sha> --head <new-sha>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from perf_db import PerfDB

# "Memory consumption end : 6222.44 Mo (peak), 6222.44 Mo (current)"
# The "(current)" half is optional: older binaries print the peak alone.
_MEM_RE = re.compile(
    r"Memory consumption\s+end\s*:\s*([0-9.]+)\s*Mo\s*\(peak\)"
    r"(?:\s*,\s*([0-9.]+)\s*Mo\s*\(current\))?")
_MEM_PEAK_ONLY_RE = re.compile(r"Memory consumption\s+end\s*:\s*([0-9.]+)\s*Mo")

_TIME_RE = re.compile(r"Elapsed time computation\s*:\s*([0-9.]+)\s*s")
_INIT_RE = re.compile(r"Elapsed time Initialisation to zero\s*:\s*([0-9.]+)\s*s")

# Library counters worth keeping: when a regression appears, these say which of
# compression, retention or allocation moved.
_COUNTER_RES = {
    "alloc_bytes_mb":     re.compile(r"alloc\s+=\s+\d+\s*\n\s*bytes\s+=\s+(\d+)\s*MB"),
    "still_compressed":   re.compile(r"still compressed \(at exit\)\s*=\s*(\d+)"),
    "reuse_pools_mb":     re.compile(r"reuse pools\s*=\s*(\d+)\s*MB"),
    "compress_exec":      re.compile(r"compress \(during exec\)\s*=\s*(\d+)"),
}


def parse_output(text: str) -> dict:
    out = {}

    m = _MEM_RE.search(text)
    if m:
        out["peak_mo"] = float(m.group(1))
        if m.group(2):
            out["current_mo"] = float(m.group(2))
    else:
        m = _MEM_PEAK_ONLY_RE.search(text)
        if m:
            out["peak_mo"] = float(m.group(1))

    for name, rx, key in (("computation", _TIME_RE, "seconds"),
                          ("init-zero", _INIT_RE, "init_seconds")):
        mm = rx.search(text)
        if mm:
            out[key] = float(mm.group(1))

    counters = {}
    for name, rx in _COUNTER_RES.items():
        mm = rx.search(text)
        if mm:
            counters[name] = int(mm.group(1))
    out["counters"] = counters
    return out


def run_once(exe: str, threads: int, divisor, iter_max, extra_env: dict,
             timeout: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env.update(extra_env)

    cmd = [exe]
    if divisor is not None:
        cmd += ["--divisor", str(divisor)]
    if iter_max is not None:
        cmd += ["--iter_max", str(iter_max)]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=timeout, cwd=os.path.dirname(exe) or ".")
    if proc.returncode != 0:
        return {"ok": False, "detail": f"exit {proc.returncode}"}
    res = parse_output(proc.stdout + proc.stderr)
    res["ok"] = True
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("database")
    p.add_argument("--exe", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--divisor", type=int)
    p.add_argument("--iter-max", type=int)
    p.add_argument("--reps", type=int, default=3,
                   help="repetitions; 3 is the minimum for a usable spread")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--compiler", required=True)
    p.add_argument("--cmake-options", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--env", action="append", default=[],
                   metavar="K=V", help="extra environment, repeatable")
    args = p.parse_args(argv)

    extra_env = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        extra_env[k] = v

    if args.reps < 3:
        print("[WARN] fewer than 3 repetitions: the spread will understate the "
              "measurement's own noise, and compare() relies on it to tell a "
              "real change from a noisy one.", file=sys.stderr)

    seconds, init_seconds, peaks, currents = [], [], [], []
    counters_last = {}
    failed = None

    for i in range(1, args.reps + 1):
        print(f"  run {i}/{args.reps} ...", end="", flush=True)
        try:
            r = run_once(args.exe, args.threads, args.divisor, args.iter_max,
                         extra_env, args.timeout)
        except subprocess.TimeoutExpired:
            failed = "timeout"
            print(" timeout")
            break
        if not r.get("ok"):
            failed = r.get("detail", "failed")
            print(f" {failed}")
            break
        if "seconds" in r:      seconds.append(r["seconds"])
        if "init_seconds" in r: init_seconds.append(r["init_seconds"])
        if "peak_mo" in r:      peaks.append(r["peak_mo"])
        if "current_mo" in r:   currents.append(r["current_mo"])
        counters_last = r.get("counters", {})
        print(f" {r.get('seconds', float('nan')):.2f}s "
              f"{r.get('peak_mo', float('nan')):.0f} Mo")

    db = PerfDB(args.database)
    run_id = db.start_run(compiler=args.compiler,
                          cmake_options=args.cmake_options, notes=args.notes)

    common = dict(config=args.config, variant=args.variant,
                  geometry=args.geometry, threads=args.threads,
                  divisor=args.divisor, iter_max=args.iter_max)

    db.record(run_id, phase="total", seconds=seconds, peak_mo=peaks,
              current_mo=currents, counters=counters_last,
              ok=failed is None, detail=failed or "", **common)

    # The zero-init phase gets its own row: it is where a 109s regression hid
    # behind a total runtime that barely moved.
    if init_seconds:
        db.record(run_id, phase="init-zero", seconds=init_seconds,
                  ok=failed is None, detail=failed or "", **common)

    if failed:
        print(f"\nrecorded as FAILED ({failed})")
        return 1

    print(f"\nrecorded run {run_id} in {args.database}")
    if currents:
        print("  (peak and current memory both captured)")
    else:
        print("  [note] the binary printed only the peak; ru_maxrss is a "
              "high-water mark and cannot show a steady-state change")
    return 0


# ── Wiring this into bench_and_test.py ───────────────────────────────────────
#
# bench_and_test.py already parses memory (parse_peak_memory_mo) and the
# CHUNK_STATS counters (parse_chunk_stats_counters). To file its results:
#
#   from perf_db import PerfDB
#   db = PerfDB(args.perf_db)                       # new --perf-db option
#   run_id = db.start_run(compiler=<the module's compiler>,
#                         cmake_options=<configure line>)
#
# then, wherever a configuration finishes, with the repetitions it already runs:
#
#   db.record(run_id, config=cfg, variant=var, geometry=size,
#             threads=th, divisor=div, iter_max=it,
#             seconds=times, peak_mo=peaks, current_mo=currents,
#             counters=parse_chunk_stats_counters(stdout))
#
# The one thing to check on the way in: bench_and_test currently keeps the MAX
# of the memory lines (ru_maxrss is monotonic, so that is the peak). The current
# figure is a separate number on the same line and needs its own parse -- that
# is what _MEM_RE above does.

if __name__ == "__main__":
    sys.exit(main())
