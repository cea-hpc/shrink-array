#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""bimodal_probe.py -- what tells a slow run from a fast one, run after run.

The same binary, run N times, sometimes falls in a slow mode set at start-up
(docs/performance_measurement.md, 2.1: +18%, ~+40% on zero_one_value; same frequency,
same instructions, same LLC misses, lower IPC, more cycles stalled on memory).
Each candidate cause leaves its own trace, which this script records per run:

  NUMA placement   the process's pages per node (/proc/<pid>/numa_maps), and the
                   free memory of each node before the run: pages that landed on
                   another socket, or on a CPU-less GPU-memory node, cost more per
                   access with the same miss count.
  huge pages       AnonHugePages (/proc/<pid>/smaps_rollup) and the free blocks
                   of the highest order (/proc/buddyinfo) before the run: with THP
                   `always` the process gets huge pages or not depending on the
                   fragmentation it finds; without them, more page-table walks
                   (dtlb_walk), which stall on memory without being LLC misses.
  cache sets       L1D / L2 refills per instruction: buffers of different arrays
                   at addresses that share cache sets (the slab layout depends on
                   the allocation order) give conflict misses in L1/L2 that the
                   LLC still serves. What a slab colouring would address.

perf stat counts the events in user space (`:u`, allowed with
perf_event_paranoid <= 2); an event the CPU does not have reads as empty.

Usage (from a compute node, the binary's usual environment):
  bimodal_probe.py --runs 12 [--mode "CHUNK_ALLOC_AFFINITY_EAGER=0"] [--mode ""] \\
      [--csv out.csv] -- $BIN --mode perf --no_io
Several --mode: the runs alternate between them (A/B in the same time window).
A mode is a space-separated list of VAR=value, and may start with `numactl ...`
to prefix the command (e.g. --mode "numactl --membind=0").

Summary: the runs sorted by time, split at the largest gap; for each recorded
quantity, its median in each group and whether the two groups overlap. A
quantity whose groups do not overlap is a candidate cause; one that overlaps is
not.
"""

import argparse
import csv
import glob
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

# perf event lists, user space only. Neoverse V2 (Grace) names first, the
# generic perf names for x86.
EVENTS_ARM = ["cycles:u", "instructions:u", "stall_backend_mem:u", "l1d_cache_refill:u", "l2d_cache_refill:u",
              "ll_cache_miss_rd:u", "dtlb_walk:u", "l2d_tlb_refill:u", "remote_access:u"]
EVENTS_X86 = ["cycles:u", "instructions:u", "L1-dcache-load-misses:u", "l2_cache_req_stat.ic_dc_miss_in_l2:u",
              "LLC-load-misses:u", "dTLB-load-misses:u", "dTLB-loads:u"]

TIME_RE_DEFAULT = r"Elapsed time computation\s*:\s*([0-9.]+)"


def cpu_nodes():
    """The NUMA nodes with CPUs, sorted."""
    out = []
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            with open(os.path.join(d, "cpulist")) as f:
                if f.read().strip():
                    out.append(int(d.rsplit("node", 1)[1]))
        except (OSError, ValueError):
            pass
    return sorted(out)


def node_free_mib():
    """MemFree of every node, MiB."""
    free = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            node = int(d.rsplit("node", 1)[1])
            with open(os.path.join(d, "meminfo")) as f:
                for line in f:
                    if "MemFree:" in line:
                        free[node] = int(line.split()[-2]) // 1024
        except (OSError, ValueError, IndexError):
            pass
    return free


def buddy_top_order():
    """Free blocks of the highest order, per node (Normal zone), from /proc/buddyinfo."""
    top = {}
    try:
        with open("/proc/buddyinfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 4 and parts[2] == "zone" and parts[3] == "Normal":
                    node = int(parts[1].rstrip(","))
                    top[node] = top.get(node, 0) + int(parts[-1])
    except (OSError, ValueError):
        pass
    return top


def children_of(pid):
    """Direct children of @p pid (/proc/<pid>/task/*/children)."""
    kids = []
    for t in glob.glob(f"/proc/{pid}/task/*/children"):
        try:
            with open(t) as f:
                kids += [int(x) for x in f.read().split()]
        except (OSError, ValueError):
            pass
    return kids


def find_target(root, exe_name):
    """The process running @p exe_name under @p root (perf, numactl, env... in between)."""
    todo, seen = [root], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            if os.path.basename(os.readlink(f"/proc/{p}/exe")) == exe_name:
                return p
        except OSError:
            pass
        todo += children_of(p)
    return None


def sample(pid):
    """Pages per node (MiB), Rss and AnonHugePages (MiB) of @p pid, or None if gone."""
    per_node = {}
    try:
        with open(f"/proc/{pid}/numa_maps") as f:
            for line in f:
                m = re.search(r"kernelpagesize_kB=(\d+)", line)
                kb = int(m.group(1)) if m else 4
                for node, pages in re.findall(r"\bN(\d+)=(\d+)", line):
                    per_node[int(node)] = per_node.get(int(node), 0) + int(pages) * kb
        rss = ahp = 0
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
                elif line.startswith("AnonHugePages:"):
                    ahp = int(line.split()[1])
    except (OSError, ValueError):
        return None
    return {"node_mib": {n: v // 1024 for n, v in per_node.items()}, "rss_mib": rss // 1024, "ahp_mib": ahp // 1024}


def parse_perf(path):
    """perf stat -x, output -> {event: value} (None when not counted)."""
    vals = {}
    try:
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 3 or row[0].startswith("#"):
                    continue
                name = row[2]
                try:
                    vals[name] = float(row[0])
                except ValueError:
                    vals[name] = None  # <not supported> / <not counted>
    except OSError:
        pass
    return vals


def run_once(cmd, mode, args, events, time_re):
    env = dict(os.environ)
    prefix = []
    words = shlex.split(mode)
    while words and "=" in words[0] and not words[0].startswith("-"):
        k, v = words.pop(0).split("=", 1)
        env[k] = v
    prefix = words  # what remains: a command prefix, e.g. numactl --membind=0

    rec = {"mode": mode or "(default)"}
    free = node_free_mib()
    for n, v in buddy_top_order().items():
        rec[f"buddy_top_N{n}"] = v
    for n in cpu_nodes():
        rec[f"free_N{n}_mib"] = free.get(n)

    perf_out = None
    full = prefix + cmd
    if events:
        fd, perf_out = tempfile.mkstemp(prefix="bimodal_perf_", suffix=".csv")
        os.close(fd)
        full = ["perf", "stat", "-x,", "-o", perf_out, "-e", ",".join(events), "--"] + full

    t0 = time.monotonic()
    proc = subprocess.Popen(full, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    exe_name = os.path.basename(cmd[0])
    target, best, out_chunks = None, None, []
    # Sample in a side thread-free way: read the output non-blocking-ish by
    # polling the process, and sample every --every seconds.
    import threading

    def reader():
        for line in proc.stdout:
            out_chunks.append(line)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    next_sample = time.monotonic() + args.first
    while proc.poll() is None:
        time.sleep(0.05)
        if target is None:
            target = find_target(proc.pid, exe_name)
        if target is not None and time.monotonic() >= next_sample:
            s = sample(target)
            if s is not None and (best is None or s["rss_mib"] >= best["rss_mib"]):
                best = s  # keep the largest footprint seen: the steady state
            next_sample = time.monotonic() + args.every
    th.join(timeout=5)
    wall = time.monotonic() - t0
    out = "".join(out_chunks)

    m = re.search(time_re, out)
    rec["time_s"] = float(m.group(1)) if m else None
    rec["wall_s"] = round(wall, 2)
    rec["exit"] = proc.returncode
    if best is not None:
        rec["rss_mib"] = best["rss_mib"]
        rec["anon_huge_mib"] = best["ahp_mib"]
        total = sum(best["node_mib"].values()) or 1
        for n, v in sorted(best["node_mib"].items()):
            rec[f"mem_N{n}_mib"] = v
        local = cpu_nodes()
        # Share of the pages on the node holding the most of them, and outside
        # the CPU nodes (the GPU-memory nodes of a GH200).
        rec["mem_top_node_pct"] = round(100.0 * max(best["node_mib"].values(), default=0) / total, 1)
        rec["mem_off_cpu_nodes_pct"] = round(
            100.0 * sum(v for n, v in best["node_mib"].items() if local and n not in local) / total, 1)
    if perf_out:
        p = parse_perf(perf_out)
        os.unlink(perf_out)
        for k, v in p.items():
            rec[k.replace(":u", "")] = v
        ins = p.get("instructions:u") or p.get("instructions")
        cyc = p.get("cycles:u") or p.get("cycles")
        if ins and cyc:
            rec["ipc"] = round(ins / cyc, 3)
        if ins:
            for k, v in p.items():
                if v is not None and k not in ("instructions:u", "cycles:u", "instructions", "cycles"):
                    rec["pki_" + k.replace(":u", "")] = round(1000.0 * v / ins, 4)
    if rec["time_s"] is None:
        tail = out.strip().splitlines()[-5:]
        print("  (time not found in the output; last lines:)\n    " + "\n    ".join(tail), file=sys.stderr)
    return rec


def summarise(rows):
    timed = [r for r in rows if r.get("time_s") is not None]
    if len(timed) < 4:
        print("\nfewer than 4 timed runs: no summary")
        return
    timed.sort(key=lambda r: r["time_s"])
    times = [r["time_s"] for r in timed]
    gaps = [(times[i + 1] - times[i], i) for i in range(len(times) - 1)]
    gap, cut = max(gaps)
    fast, slow = timed[:cut + 1], timed[cut + 1:]
    print("\nruns by time:")
    for r in timed:
        print(f"  {r['time_s']:8.2f} s  {r['mode']}")
    spread = statistics.median(times)
    print(f"\nlargest gap: {gap:.2f} s between {times[cut]:.2f} and {times[cut + 1]:.2f} "
          f"({len(fast)} fast, {len(slow)} slow)")
    if gap < 0.03 * spread:
        print("  the gap is under 3% of the median: no clear bimodality in these runs")
    if min(len(fast), len(slow)) < 3:
        print("  a group has fewer than 3 runs: a YES below may be chance -- add runs")
    keys = sorted({k for r in timed for k in r} - {"mode", "time_s", "wall_s", "exit", "run"})
    print(f"\n{'quantity':34} {'fast (median)':>15} {'slow (median)':>15}  separates?")
    for k in keys:
        a = [r[k] for r in fast if isinstance(r.get(k), (int, float))]
        b = [r[k] for r in slow if isinstance(r.get(k), (int, float))]
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        sep = max(a) < min(b) or max(b) < min(a)
        mark = "YES" if sep else ""
        print(f"{k:34} {ma:15.4g} {mb:15.4g}  {mark}")
    modes = sorted({r["mode"] for r in timed})
    if len(modes) > 1:
        print("\nper mode (mean time, slow runs / runs):")
        slow_ids = {id(r) for r in slow}
        for md in modes:
            rr = [r for r in timed if r["mode"] == md]
            print(f"  {md:40} {statistics.mean(r['time_s'] for r in rr):8.2f} s   "
                  f"{sum(id(r) in slow_ids for r in rr)}/{len(rr)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10, help="runs in all (default 10)")
    ap.add_argument("--mode", action="append", default=None,
                    help='"VAR=v VAR2=w [prefix cmd ...]"; repeat to alternate modes (default: one, unchanged)')
    ap.add_argument("--events", default=None,
                    help="comma-separated perf events (default: a list for the CPU, :u)")
    ap.add_argument("--no-perf", action="store_true", help="do not run under perf stat")
    ap.add_argument("--time-regex", default=TIME_RE_DEFAULT, help="regex whose group 1 is the time, in seconds")
    ap.add_argument("--first", type=float, default=5.0, help="first sample after this many seconds (default 5)")
    ap.add_argument("--every", type=float, default=5.0, help="then one sample every this many seconds (default 5)")
    ap.add_argument("--csv", default="bimodal_probe.csv", help="per-run CSV (default bimodal_probe.csv)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- the command to run")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command (put it after --)")
    modes = args.mode or [""]
    events = [] if args.no_perf else (args.events.split(",") if args.events else
                                      (EVENTS_ARM if os.uname().machine == "aarch64" else EVENTS_X86))
    if events and subprocess.run(["sh", "-c", "command -v perf"], capture_output=True).returncode != 0:
        print("perf not found: running without counters (--no-perf)", file=sys.stderr)
        events = []
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            thp = f.read().strip()
        with open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size") as f:
            pmd = int(f.read()) >> 20
        print(f"THP: {thp}, PMD huge page {pmd} MiB, base page {os.sysconf('SC_PAGESIZE') >> 10} KiB")
    except (OSError, ValueError):
        pass
    print(f"CPU nodes: {cpu_nodes()}")

    rows = []
    for i in range(args.runs):
        mode = modes[i % len(modes)]
        r = run_once(cmd, mode, args, events, args.time_regex)
        r["run"] = i + 1
        rows.append(r)
        t = r.get("time_s")
        extra = "".join(f"  {k}={r[k]}" for k in ("ipc", "anon_huge_mib", "mem_top_node_pct") if k in r)
        print(f"run {i + 1:2}/{args.runs}  {mode or '(default)':30} "
              f"{(f'{t:.2f} s' if t is not None else 'no time'):>10}{extra}", flush=True)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-run values: {args.csv}")
    summarise(rows)


if __name__ == "__main__":
    main()
