#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
catch_crash.py -- runs a shrink_bench executable (--mode perf) in a loop under
gdb (batch) and captures the stack and the context at the FIRST crash
(SIGSEGV/SIGABRT) or probe report. Written to hunt a rare cross-thread
use-after-munmap.

Usage:
    python3 catch_crash.py --exe EXE [-n NRUNS] [-t THREADS] [--args "ARGS"] [--continue]

    --exe      the executable (default: $CATCH_CRASH_EXE)
    -n         maximum number of runs (default 1000)
    -t         fixed OMP_NUM_THREADS (default 64); "rand" = random in 2..128
    --args     arguments passed to the executable (default "--divisor 10")
    --continue do NOT stop at the first crash: archive it and go on
    --outdir   where the crashes are archived (default ./crashes)

Output: for each crash, a file crashes/crash_<i>_<threads>thr_<stamp>.log with
the program output, the backtrace of every thread and the allocator context.
"""

import argparse
import datetime
import os
import random
import subprocess
import sys
import time

# e.g. <build>/examples/shrink_bench/shrink_bench_chunk2a_dyn (pass --mode perf in --args)
DEFAULT_EXE = os.environ.get("CATCH_CRASH_EXE", "")

# The gdb commands run in batch. The program is started; if it stops on a
# signal, the context is dumped. If it ends normally, gdb goes on after `run`
# but $_exitcode is set and there is NO stack -- hence the test before `bt`.
# `set pagination off` is required in batch (gdb would block on --More--).
GDB_SCRIPT = r"""
set pagination off
set confirm off
set print pretty on
# Stop on the fatal signals only, let the others through.
handle SIGSEGV stop print
handle SIGABRT stop print
handle SIGFPE stop print
run
# --- Reached after a signal (program stopped) or after a normal end
# --- ($_exitcode set). Dump in the first case ONLY.
if $_isvoid($_exitcode)
  printf "\n=========== CRASH CONTEXT (signal) ===========\n"
  printf "\n--- faulting thread backtrace ---\n"
  bt
  printf "\n--- bench frame: k / values / tab_complex vs tab1 ---\n"
  frame 11
  print k
  print v_real
  print v_im
  print val_sum
  printf "\n--- faulting address (SIGSEGV) ---\n"
  print $_siginfo._sifields._sigfault.si_addr
  printf "\n--- all threads backtrace ---\n"
  thread apply all bt
  printf "\n--- registers ---\n"
  info registers x0 x1 x2 x19 x20
  printf "\n=========== END CONTEXT ===========\n"
else
  printf "\n___PROGRAM_EXITED_NORMALLY___ exitcode=%d\n", $_exitcode
end
quit
"""

# Interactive variant: gdb ITSELF loops on `run` until the program stops on a
# signal (SIGSEGV/SIGABRT). It then prints the context and gives the prompt back
# (no `quit`): you keep control to inspect (p, x, frame, info locals, thread
# apply all bt...). A normal end starts the `run` loop again. The mode for a rare
# crash to examine by hand: start it and let it run, it stops in gdb by itself
# at the crash. Leave with `quit`.
GDB_SCRIPT_INTERACTIVE = r"""
set pagination off
set confirm off
set print pretty on
handle SIGSEGV stop print
handle SIGABRT stop print
handle SIGFPE stop print
# Loop: run again while the program exits normally ($_exitcode set). Stop (and
# give the prompt back) as soon as a signal leaves $_exitcode void.
python
import gdb
while True:
    gdb.execute("run")
    try:
        ec = gdb.parse_and_eval("$_exitcode")
        # normal exit -> run again
        print("[catch_crash] normal exit, running again...")
        continue
    except gdb.error:
        # $_exitcode undefined => stopped on a signal
        print("\n=========== CRASH CONTEXT (signal) ===========")
        gdb.execute("bt")
        for _c in ("frame 11","print k","print v_real","print v_im","print val_sum"):
            try: gdb.execute(_c)
            except gdb.error: pass
        try:
            gdb.execute("print $_siginfo._sifields._sigfault.si_addr")
        except gdb.error:
            pass
        gdb.execute("thread apply all bt")
        print("=========== END CONTEXT -- the gdb prompt is yours (quit to leave) ===========")
        break
end
"""


def build_gdb_command(exe, args, gdb_script_path, interactive=False):
    # -batch: runs the script and quits (unattended, archived automatically).
    # -iex 'set auto-load safe-path /': no libstdc++ auto-load warning.
    #
    # interactive=True: instead of -batch (which quits at the crash), the same
    # script is loaded WITHOUT -batch and WITHOUT the final `quit`. gdb runs,
    # stops on the signal, prints the context, THEN gives the prompt back: you
    # keep control to inspect (p, x, frame, info locals...). For when a rare
    # crash finally happens and is to be examined by hand.
    if interactive:
        cmd = ["gdb", "-nx",
               "-iex", "set auto-load safe-path /",
               "-x", gdb_script_path, "--args", exe]
    else:
        cmd = ["gdb", "-batch", "-nx",
               "-iex", "set auto-load safe-path /",
               "-x", gdb_script_path, "--args", exe]
    cmd.extend(args)
    return cmd


def looks_like_crash(returncode, output):
    # gdb -batch returns 0 even after the inferior crashed, so the crash is
    # detected in the OUTPUT. A real signal or a probe report is required; a
    # program that exited normally is not a crash.
    if "___PROGRAM_EXITED_NORMALLY___" in output:
        return False
    markers = [
        "Program received signal SIGSEGV",
        "Program received signal SIGABRT",
        "Program received signal SIGFPE",
        "Arithmetic exception",
        "Program terminated with signal",
        "CRASH CONTEXT (signal)",     # printed in the signal branch only
        "ALLOC_PROBE",
        "FLUSH_PROBE",
        "POOL_LEDGER",
        "INVARIANT",
        "ThreadSanitizer",
    ]
    return any(m in output for m in markers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000, help="maximum number of runs")
    ap.add_argument("-t", default="64", help="OMP_NUM_THREADS (an integer or 'rand')")
    ap.add_argument("--args", default="--divisor 10", help="arguments of the executable")
    ap.add_argument("--exe", default=DEFAULT_EXE, help="the executable (default: $CATCH_CRASH_EXE)")
    ap.add_argument("--continue", dest="keep_going", action="store_true",
                    help="go on after a crash instead of stopping")
    ap.add_argument("--interactive", action="store_true",
                    help="at the crash, keep control in gdb (interactive prompt) "
                         "instead of archiving and quitting")
    ap.add_argument("--outdir", default="./crashes", help="where the crashes are archived")
    args = ap.parse_args()

    if not args.exe:
        print("No executable: give --exe or set CATCH_CRASH_EXE")
        sys.exit(1)
    if not (os.path.isfile(args.exe) and os.access(args.exe, os.X_OK)):
        print(f"Executable not found or not executable: {args.exe}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # the gdb script, written once and used by every run
    gdb_script_path = os.path.join(args.outdir, "_gdb_catch.txt")
    with open(gdb_script_path, "w") as f:
        f.write(GDB_SCRIPT_INTERACTIVE if args.interactive else GDB_SCRIPT)

    prog_args = args.args.split() if args.args else []
    start = time.time()
    crashes = 0

    for i in range(1, args.n + 1):
        threads = random.randint(2, 128) if args.t == "rand" else int(args.t)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))

        cmd = build_gdb_command(args.exe, prog_args, gdb_script_path,
                                interactive=args.interactive)

        if args.interactive:
            # Interactive mode: the gdb script ITSELF loops on `run` until the
            # crash, then gives the prompt back. gdb is started once in the
            # foreground (it inherits the terminal); the user takes over.
            print(f"Interactive mode: gdb runs the program again and again "
                  f"(threads={threads}) until it crashes, then gives the prompt back.\n"
                  f"Let it run; at the crash, inspect, then `quit`.\n",
                  flush=True)
            subprocess.run(cmd, env=env)
            print("\ngdb ended.")
            return

        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""

        if looks_like_crash(proc.returncode, out):
            crashes += 1
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.outdir, f"crash_{i:05d}_{threads}thr_{stamp}.log")
            with open(path, "w") as f:
                f.write(f"# run {i}/{args.n}  threads={threads}  "
                        f"rc={proc.returncode}  {stamp}\n")
                f.write(f"# cmd: {' '.join(cmd)}\n\n")
                f.write(out)
            print(f"[{i}/{args.n}] CRASH (threads={threads}) -> {path}")
            if not args.keep_going:
                print(f"\nStopped at the first crash. {crashes} crash(es). "
                      f"See {path}")
                break
        else:
            elapsed = time.time() - start
            rate = elapsed / i
            eta = (args.n - i) * rate
            # What the output says about how far the run got.
            n_recycle = out.count("[RECYCLE]")
            exited_ok = "___PROGRAM_EXITED_NORMALLY___" in out
            # the last "k :" reached (progress of the bench loop)
            last_k = ""
            for line in out.splitlines():
                if line.startswith("k :") or " val_sum " in line:
                    last_k = line.strip()[:60]
            status = "exit0" if exited_ok else "no-exit?"
            extra = f" last[{last_k}]" if last_k else ""
            print(f"[{i}/{args.n}] OK {status}  threads={threads}  "
                  f"recycle={n_recycle}  elapsed={elapsed:.0f}s  eta={eta:.0f}s{extra}",
                  flush=True)

    total = time.time() - start
    print(f"\nDone: {i} runs, {crashes} crash(es), {total:.0f}s.")
    if crashes == 0:
        print("No crash caught: raise -n, drop --divisor (larger runs), "
              "or -t rand to vary the load.")


if __name__ == "__main__":
    main()
