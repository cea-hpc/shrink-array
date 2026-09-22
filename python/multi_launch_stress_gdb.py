#!/usr/bin/env python3

import os
import random
import glob
import time
import csv
import re
import pexpect
import argparse
import difflib
import sys

# ============================================================
#  Custom ArgumentParser with typo suggestions
# ============================================================


class SuggestingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        parts = message.split()
        wrong_arg = next((p for p in parts if p.startswith("--")), "")
        valid_options = []
        for action in self._actions:
            valid_options.extend(action.option_strings)
        suggestion = difflib.get_close_matches(wrong_arg, valid_options, n=1)
        if suggestion:
            message += f"\nDid you mean {suggestion[0]} ?"
        super().error(message)


# ============================================================
#  Argument parsing
# ============================================================

parser = SuggestingArgumentParser(
    description="Run executable(s) under GDB with random thread counts."
)
parser.add_argument("--runs", type=int, default=10000, help="Total number of runs.")
parser.add_argument(
    "--min_threads", type=int, default=2, help="Minimum number of threads."
)
parser.add_argument(
    "--max_threads", type=int, default=512, help="Maximum number of threads."
)
parser.add_argument(
    "--no_io",
    action="store_true",
    help="Disable stdout/stderr file output and set NO_IO=1.",
)
parser.add_argument(
    "--all_exec", action="store_true", help="Run all executables sequentially."
)
parser.add_argument(
    "--shuffle_exec", action="store_true", help="Shuffle executables at each run."
)
parser.add_argument(
    "--exec_timeout", type=float, default=None, help="Max seconds allowed inside GDB."
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output."
)
args = parser.parse_args()

# Enabling shuffle implies running all executables
if args.shuffle_exec:
    args.all_exec = True

n = args.runs
min_threads = args.min_threads
max_threads = args.max_threads
no_io = args.no_io
verbose = args.verbose
exec_timeout = args.exec_timeout

# ============================================================
#  Executable list
# ============================================================

# Ces noms etaient FAUX bien avant la fusion check/perf : les cibles ont
# toujours porte une declinaison de taille, il n'a jamais existe de
# shrink_bench_chunk_check tout court. Corriges ici sur la declinaison dyn, la
# meme que le defaut de multi_launch_stress.py -- et sans le _check,
# qui est desormais un drapeau d'execution (MODE_ARGS plus bas).
base_exec_list = [
    "./shrink_bench_chunk_dyn",
    "./shrink_bench_chunk2_dyn",
    "./shrink_bench_chunk2a_dyn",
]

# check et perf sont un seul binaire depuis la fusion ; le mode se choisit ici.
MODE_ARGS = ["--mode", "check"]

executables = base_exec_list[:] if args.all_exec else ["./shrink_bench_chunk2_dyn"]

# ============================================================
#  Ensure .gdbinit contains the GroupBT command
# ============================================================

gdbinit_path = ".gdbinit"

gdbinit_content = r'''
python
import gdb

class GroupBT(gdb.Command):
    """Group threads by call stack."""

    def __init__(self):
        super(GroupBT, self).__init__("group-bt", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        threads = gdb.selected_inferior().threads()

        def signature(thread):
            frames = []
            try:
                f = thread.frame
                while f is not None:
                    name = f.name() or "??"
                    frames.append(name)
                    f = f.older()
            except gdb.error:
                frames.append("<??>")
            return tuple(frames)

        groups = {}
        for t in threads:
            sig = signature(t)
            groups.setdefault(sig, []).append(t.num)

        print("\n=== THREAD GROUPS BY CALL STACK ===")
        for sig, ths in sorted(groups.items(), key=lambda x: -len(x[1])):
            print(f"\nThreads: {ths}")
            for frame in sig:
                print(f"  {frame}")

GroupBT()
end
'''


def prepare_gdbinit():
    """Ensure .gdbinit exists and contains the GroupBT command."""
    if not os.path.exists(gdbinit_path):
        with open(gdbinit_path, "w") as f:
            f.write(gdbinit_content)
    else:
        with open(gdbinit_path, "r") as f:
            existing = f.read()
        if "GroupBT" not in existing:
            with open(gdbinit_path, "a") as f:
                f.write("\n" + gdbinit_content)


# ============================================================
#  Create GDB command file
# ============================================================


def prepare_cmd_file(verbose=False):
    """
    Generate the GDB command file used for each run.

    Important: do NOT add 'quit' here — pexpect must stay in control
    after a crash so it can detect the signal and hand off to interactive mode.
    """
    gdb_cmd_file = "cmd"
    with open(gdb_cmd_file, "w") as f:
        f.write("set logging file gdb.txt\n")  # Log all GDB output to gdb.txt
        f.write("set logging on\n")  # Enable logging
        f.write("handle SIGSEGV stop print\n")  # Stop on segmentation fault
        f.write(
            "handle SIGFPE stop print\n"
        )  # Stop on floating-point exception
        f.write("handle SIGABRT stop print\n")  # Stop on abort signal
        f.write("set prompt >>>GDBREADY<<<\n")
        f.write("run\n")  # Start the program
        # No 'quit' here: after a crash GDB stops at the signal and waits
        # at the (gdb) prompt, which pexpect will detect.
    if verbose:
        print(f"📄 GDB command file '{gdb_cmd_file}' prepared.")
    return gdb_cmd_file


# ============================================================
#  Signal detection helpers
# ============================================================

SIGNAL_PATTERNS = [
    re.compile(r"received signal (SIG[A-Z]+)"),
    re.compile(r"Thread \d+ .*?received signal (SIG[A-Z]+)"),
]


def extract_signal(text):
    """
    Return the first signal name found in text, or None.
    Tries multiple GDB message formats.
    """
    for pattern in SIGNAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def wait_for_signal_in_gdb_log(path, timeout=2.0, stable_duration=2.0, verbose=False):
    """
    Scan gdb.txt for crash signals.

    Returns the signal name if found, or None if the log stabilises
    without any signal within the timeout.
    """
    start = time.time()
    last_content = ""
    last_change_time = time.time()

    if verbose:
        print(
            f"⏳ Scanning {path} for signals (timeout={timeout}s, stable={stable_duration}s)"
        )

    while time.time() - start < timeout:
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read().replace("\r", "")

            signal_name = extract_signal(content)
            if signal_name:
                if verbose:
                    print(f"✅ Signal detected in gdb.txt: {signal_name}")
                return signal_name

            # Stop early if the file has not changed for stable_duration
            if content != last_content:
                last_content = content
                last_change_time = time.time()
            elif time.time() - last_change_time >= stable_duration:
                if verbose:
                    print(f"🛑 gdb.txt stable for {stable_duration}s — stopping scan")
                break
        else:
            if verbose:
                print("⚠️ gdb.txt not found yet")

        time.sleep(0.2)

    return None


# ============================================================
#  Run executable under GDB
# ============================================================


def run_with_gdb(
        run_index,
        exe,
        num_threads,
        env,
        stdout_file,
        stderr_file,
        gdb_cmd_file,
        verbose=False,
):
    """
    Spawn GDB for a single run and detect crashes.

    Signal detection strategy (two-stage):
      1. PRIMARY   — scan child.before (the pexpect PTY buffer captured between
                     the program start and the GDB prompt).  GDB writes the
                     signal message to the terminal immediately at crash time,
                     before gdb.txt is necessarily flushed, so this is the most
                     reliable and fastest source.
      2. FALLBACK  — scan gdb.txt in case the PTY buffer was truncated or the
                     signal message was flushed to the log file only.

    Flow:
      GDB starts → cmd file sets up handlers and calls run
      → program crashes → GDB stops at signal → prompt
      → pexpect catches prompt → signal detected in buffer or gdb.txt
      → save backtrace (one command at a time, always waiting for prompt)
      → hand off to interactive GDB

    Returns (returncode, signal_name):
        0   → clean run
        1   → crash detected, interactive GDB launched
       -3   → pexpect timeout
    """

    # Remove stale log from previous run
    if os.path.exists("gdb.txt"):
        os.remove("gdb.txt")

    exe_tag = os.path.basename(exe)

    gdb_args = [
        "-iex",
        "set pagination off",
        "-iex",
        "set confirm off",
        f"--command={gdb_cmd_file}",
        "--args",
        exe,
        *MODE_ARGS,
    ]

    if verbose:
        print(f"🐞 Spawning GDB: {exe_tag} (run {run_index}, threads={num_threads})")

    child = pexpect.spawn("gdb", gdb_args, env=env, encoding="utf-8", timeout=120)

    # Optionally capture GDB output to files
    if no_io:
        child.logfile = None
        child.logfile_read = None
    else:
        out = open(stdout_file, "w")
        err = open(stderr_file, "w")
        child.logfile = out
        child.logfile_read = err

    try:
        # --- Detect crash BEFORE waiting for the GDB prompt ---
        signal_name = wait_for_signal_in_gdb_log(
            "gdb.txt",
            timeout=2.0,
            stable_duration=1.0,
            verbose=verbose,
        )

        if signal_name:
            print(f"⚠️ Crash detected at run {run_index}: {signal_name}")

            # Force GDB to flush the prompt
            child.sendline("")

            # Extract backtrace safely
            child.sendline("thread apply all bt full")
            child.expect(r">>>GDBREADY<<<", timeout=20)
            bt_file = f"bt_run_{run_index}.txt"
            with open(bt_file, "w") as f:
                f.write(child.before)

            print(f"🧾 Backtrace saved to {bt_file}")

            # Quit GDB cleanly
            child.sendline("quit")
            return 1, signal_name
        # --- End crash detection block ---

        # Wait for GDB to return to the prompt after the program
        # finishes (normal exit or signal stop).
        child.expect(r">>>GDBREADY<<<", timeout=90)

        # Stage 1: check pexpect PTY buffer
        buffer_text = child.before or ""
        signal_name = extract_signal(buffer_text)

        if signal_name and verbose:
            print(f"✅ Signal detected in PTY buffer: {signal_name}")

        # Stage 2: fallback to gdb.txt
        if not signal_name:
            signal_name = wait_for_signal_in_gdb_log(
                "gdb.txt", timeout=2.0, stable_duration=2.0, verbose=verbose
            )

        if signal_name:
            print(f"🔍 Signal {signal_name} at run {run_index} for {exe_tag}")

            child.logfile = None
            child.logfile_read = None
            child.logfile_send = None

            child.sendline("set logging off")
            child.expect(r">>>GDBREADY<<<", timeout=10)

            bt_file = f"bt_run_{run_index}_{exe_tag}.txt"
            child.sendline(f"set logging file {bt_file}")
            child.expect(r">>>GDBREADY<<<", timeout=5)
            child.sendline("set logging on")
            child.expect(r">>>GDBREADY<<<", timeout=5)
            child.sendline("thread apply all bt full")
            child.expect(r">>>GDBREADY<<<", timeout=30)
            child.sendline("set logging off")
            child.expect(r">>>GDBREADY<<<", timeout=5)

            print(f"🧾 Backtrace saved to {bt_file}")
            print("💡 GDB is yours — type 'quit' when done.")

            child.interact()
            return 1, signal_name

        # Normal exit path
        child.sendline("quit")
        try:
            child.expect(r"Quit anyway\? \(y or n\)", timeout=5)
            child.sendline("y")
            child.expect(pexpect.EOF, timeout=5)
        except (pexpect.exceptions.EOF, pexpect.exceptions.TIMEOUT):
            pass

        if verbose:
            print(f"✅ Run {run_index} for {exe_tag} completed successfully")

        return 0, ""

    except pexpect.exceptions.TIMEOUT:
        print(f"⏱️ Timeout at run {run_index} — GDB did not respond for {exe_tag}")
        try:
            child.sendline("quit")
            child.expect(pexpect.EOF, timeout=5)
        except (pexpect.exceptions.EOF, pexpect.exceptions.TIMEOUT):
            pass
        return -3, ""

    finally:
        if not no_io:
            out.close()
            err.close()


# ============================================================
#  Cleanup old run files
# ============================================================

if not no_io:
    print("🧹 Cleaning up old output/error files...")
    for file in glob.glob("output_*.txt") + glob.glob("error_*.txt"):
        os.remove(file)

# ============================================================
#  Create CSV summary file
# ============================================================

summary_file = "timings_summary.csv"
print(f"📄 Creating summary file: {summary_file}")

with open(summary_file, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(
        [
            "run_index",
            "executable",
            "num_threads",
            "wall_seconds",
            "returncode",
            "signal",
            "stdout_file",
            "stderr_file",
            "retried",
        ]
    )

# ============================================================
#  Prepare GDB environment (gdbinit + cmd file)
# ============================================================

prepare_gdbinit()
gdb_cmd_file = prepare_cmd_file(verbose=verbose)

# ============================================================
#  Main loop
# ============================================================

start_time = time.time()
cycle_count = 0

for i in range(n):
    cycle_count += 1

    if cycle_count % 10 == 0:
        elapsed = time.time() - start_time
        eta = (n - i) * elapsed / cycle_count
        print(f"⏳ Cycle {cycle_count}: Elapsed={elapsed:.2f}s | ETA={eta:.2f}s")

    print("")

    if args.shuffle_exec:
        random.shuffle(executables)

    run_threads = random.randint(min_threads, max_threads)

    for exe in executables:
        exe_threads = (
            random.randint(min_threads, max_threads)
            if args.shuffle_exec
            else run_threads
        )
        exe_tag = os.path.basename(exe)

        print(f"🧵 Threads: {exe_threads}  →  {exe}")
        print(f"🚀 Run {i+1}/{n}: {exe}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(exe_threads)
        # Compression is off by default in shrink_bench (opt-in: --compress /
        # COMPRESS=1), so nothing to set here any more.
        if no_io:
            env["NO_IO"] = "1"

        stdout_file = f"output_{i+1}_{exe_tag}.txt"
        stderr_file = f"error_{i+1}_{exe_tag}.txt"

        t0 = time.perf_counter()
        retried = False

        returncode, signal_name = run_with_gdb(
            i + 1,
            exe,
            exe_threads,
            env,
            stdout_file,
            stderr_file,
            gdb_cmd_file,
            verbose=verbose,
            )

        if returncode == -3:
            retried = True
            print(f"🔁 Retrying run {i+1} for {exe} after timeout...")
            time.sleep(1)
            returncode, signal_name = run_with_gdb(
                i + 1,
                exe,
                exe_threads,
                env,
                stdout_file,
                stderr_file,
                gdb_cmd_file,
                verbose=verbose,
                )
            if returncode == -3:
                print(f"❌ Run {i+1} failed twice due to timeout — skipping")

        wall_seconds = time.perf_counter() - t0

        if no_io:
            stdout_rec = ""
            stderr_rec = ""
        else:
            if os.path.isfile(stderr_file) and os.stat(stderr_file).st_size == 0:
                os.remove(stderr_file)
                stderr_file = ""
            stdout_rec = stdout_file
            stderr_rec = stderr_file if returncode != 0 else ""

        with open(summary_file, "a", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(
                [
                    i + 1,
                    exe_tag,
                    exe_threads,
                    f"{wall_seconds:.6f}",
                    returncode,
                    signal_name,
                    stdout_rec,
                    stderr_rec,
                    "yes" if retried else "no",
                    ]
            )

        if returncode == 1:
            print(
                f"❌ Crash at run {i+1}/{n} for {exe}: signal {signal_name} — GDB session ended"
            )
            sys.exit(1)
        elif returncode == 0:
            print(f"✅ {exe_tag} run {i+1} OK in {wall_seconds:.3f}s")
        else:
            print(f"⚠️  Run {i+1}/{n} for {exe} ended with code {returncode}")
