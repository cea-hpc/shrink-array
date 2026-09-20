#!/usr/bin/env python3

import os
import random
import glob
import time
import csv
import subprocess
import argparse
import difflib
import sys

import atexit
import datetime


class _Tee:
    """Mirror everything printed to stdout into a log file as well, so a long
    stress run leaves a full transcript on disk in the directory it runs from.
    This script only uses plain print() (newline-terminated, no colours, no \r
    progress bars), so the file copy is a straight duplicate. The log is
    line-buffered and flushed on every write, so a Ctrl-C or a sys.exit() on the
    first failing run still leaves the complete output up to that point."""
    def __init__(self, real, log_file):
        self._real = real
        self._log = log_file

    def write(self, s):
        self._real.write(s)
        try:
            self._log.write(s)
            self._log.flush()
        except Exception:
            pass  # logging must never break the run

    def flush(self):
        self._real.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._real, "isatty", lambda: False)()


# 🧠 Custom ArgumentParser with suggestion on typo
class SuggestingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        parts = message.split()

        # Find the first argument-like token
        wrong_arg = next((p for p in parts if p.startswith("--")), "")

        valid_options = []
        for action in self._actions:
            valid_options.extend(action.option_strings)

        suggestion = difflib.get_close_matches(wrong_arg, valid_options, n=1)

        if suggestion:
            message += f"\nDid you mean {suggestion[0]} ?"

        super().error(message)

# 🛠️ Argument parser
parser = SuggestingArgumentParser(description="Run executable(s) with random thread counts.")
parser.add_argument("--runs", type=int, default=10000, help="Total number of runs (default: 10000)")
parser.add_argument("--min_threads", type=int, default=2, help="Minimum number of threads (default: 2)")
parser.add_argument("--max_threads", type=int, default=512, help="Maximum number of threads (default: 512)")
parser.add_argument("--no_io", action="store_true", help="Disable stdout/stderr file output and set NO_IO=1")
parser.add_argument("--all_exec", action="store_true",
                    help="Shortcut for --exec all --size all: run every variant in every "
                         "size declination (the full matrix of binaries present on disk)")
parser.add_argument("--shuffle_exec", action="store_true",
                    help="Shuffle the order of executables at each run")

# ⭐ Selection = VARIANTS x SIZES. Binaries are named
#   shrink_bench_<variant>[_<size>]  with size in 64/256/1024/dyn, lance
#   avec --mode check,
#   plus a bare 'default' declination with no suffix (compile-time default).
# Both options accept a comma-separated list or 'all', so subsets compose:
#   --size dyn                  -> the 3 dyn binaries (default behaviour)
#   --exec chunk2a              -> the 5 chunk2a binaries
#   --exec chunk,chunk2a --size 64,dyn -> 4 binaries
#   --all_exec                  -> the full matrix
_VARIANTS = ["chunk", "chunk2", "chunk2a"]
_SIZES         = ["64", "256", "1024", "dyn"]   # choix VALIDES (64 encore demandable via --size 64)
_DEFAULT_SIZES = ["256", "1024", "dyn"]              # tailles construites par defaut (64 retire)

def _csv_choices(name, valid):
    def parse(value):
        if value == "all":
            return list(valid)
        items = [v.strip() for v in value.split(",") if v.strip()]
        bad = [v for v in items if v not in valid]
        if bad:
            raise argparse.ArgumentTypeError(
                f"{name}: invalid value(s) {', '.join(bad)} "
                f"(valid: {', '.join(valid)}, or 'all', comma-separated)")
        # dedupe, keep user order
        return list(dict.fromkeys(items))
    return parse

parser.add_argument(
    "--exec",
    type=_csv_choices("--exec", _VARIANTS),
    default=None,
    metavar="{chunk,chunk2,chunk2a,all}[,...]",
    help="Variant(s) to run, comma-separated or 'all'. Default: chunk2 alone "
         "(historical behaviour) unless --all_exec is given."
)
parser.add_argument(
    "--size",
    type=_csv_choices("--size", _SIZES),
    default=None,
    metavar="{64,256,1024,dyn,default,all}[,...]",
    help="Size declination(s), comma-separated or 'all'. 'default' is the "
         "bare binary without suffix (compile-time default size). Default: dyn."
)

args = parser.parse_args()

# Resolve the selection. --all_exec expands both axes (unless the user
# narrowed one explicitly, which then wins for that axis).
if args.all_exec:
    variants = args.exec if args.exec else list(_VARIANTS)
    sizes    = args.size if args.size else list(_DEFAULT_SIZES)
else:
    variants = args.exec if args.exec else ["chunk2"]
    sizes    = args.size if args.size else ["dyn"]

# 📝 Tee all screen output into a log file in the current directory, next to
# timings_summary.csv and the per-run output_*/error_* files.
_console_log_path = "multi_launch_output.log"
try:
    _console_log_file = open(_console_log_path, "w")
    _console_log_file.write(
        f"# multi_launch_no_compress_inter.py run log — "
        f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
    _console_log_file.write("# command: " + " ".join(sys.argv) + "\n\n")
    _console_log_file.flush()
    sys.stdout = _Tee(sys.stdout, _console_log_file)
    atexit.register(lambda: (_console_log_file.flush(), _console_log_file.close()))
except OSError:
    pass  # if the log can't be opened, run without it

n = args.runs
min_threads = args.min_threads
max_threads = args.max_threads
no_io = args.no_io

# 🧹 Clean up previous output and error files (only if IO is enabled)
if not no_io:
    print("🧹 Cleaning up old files...")
    for file in glob.glob('output_*.txt') + glob.glob('error_*.txt'):
        os.remove(file)

# 🔧 Executables list: VARIANTS x SIZES, filtered by what actually exists on
# disk (a size declination may not be built in this tree — warn, don't die,
# unless the whole selection ends up empty).
# Le mode check n'est plus dans le NOM du binaire : check et perf sont un seul
# executable depuis la fusion, et le mode se choisit a l'execution par --mode
# (voir bench_config.h). D'ou shrink_bench_<variant>[_<size>] ici, et MODE_ARGS
# ajoute a chaque lancement plus bas.
MODE_ARGS = ["--mode", "check"]


def _binary_name(variant, size):
    suffix = "" if size == "default" else f"_{size}"
    return f"./shrink_bench_{variant}{suffix}"

executables = []
missing = []
for v in variants:
    for sz in sizes:
        exe = _binary_name(v, sz)
        (executables if os.path.isfile(exe) else missing).append(exe)

if missing:
    print(f"⚠️ {len(missing)} selected binary(ies) not found, skipped: {', '.join(missing)}")
if not executables:
    sys.exit("❌ No selected binary exists on disk — nothing to run "
             f"(variants={','.join(variants)} sizes={','.join(sizes)})")

print(f"🎯 {len(executables)} binary(ies) selected: {', '.join(e.lstrip('./') for e in executables)}")

# 📄 Summary CSV
summary_file = "timings_summary.csv"
print(f"📄 Creating summary file: {summary_file}")

with open(summary_file, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["run_index", "executable", "num_threads", "wall_seconds",
                     "returncode", "stdout_file", "stderr_file"])

start_time = time.time()
cycle_count = 0

# 🔁 Main loop
for i in range(n):
    cycle_count += 1

    if cycle_count % 10 == 0:
        elapsed_time = time.time() - start_time
        estimated_time_left = (n - i) * elapsed_time / cycle_count
        print(f"⏳ Cycle {cycle_count}: Elapsed Time: {elapsed_time:.2f}s | Estimated Time Left: {estimated_time_left:.2f}s")

    print("")

    # Shuffle executables if requested
    if args.shuffle_exec:
        random.shuffle(executables)

    # Draw a thread count for this run
    run_threads = random.randint(min_threads, max_threads)

    # Run each executable in sequence
    for exe in executables:

        # In shuffle mode → new draw per executable
        if args.shuffle_exec:
            exe_threads = random.randint(min_threads, max_threads)
        else:
            exe_threads = run_threads

        print(f"🧵 Threads used for {exe}: {exe_threads}")
        print(f"🚀 Launching {i+1}/{n} → {exe}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(exe_threads)
        # NUMA binding: pin threads to cores so data stays on the local socket.
        # On this 4-socket machine, unbound threads migrate across sockets and
        # hammer remote memory (huge slowdown). Overridable from the outer env.
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
        env["NO_COMPRESS"] = "1"
        if no_io:
            env["NO_IO"] = "1"

        exe_tag = os.path.basename(exe)
        stdout_file = f"output_{i + 1}_{exe_tag}.txt"
        stderr_file = f"error_{i + 1}_{exe_tag}.txt"

        t0 = time.perf_counter()
        try:
            if no_io:
                result = subprocess.run([exe] + MODE_ARGS, check=False, env=env)
            else:
                with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                    result = subprocess.run([exe] + MODE_ARGS, check=False, stdout=stdout, stderr=stderr, env=env)

            wall_seconds = time.perf_counter() - t0
            returncode = result.returncode

            if not no_io and os.path.isfile(stderr_file) and os.stat(stderr_file).st_size == 0:
                os.remove(stderr_file)
                stderr_file = ""

            with open(summary_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow([
                    i + 1,
                    exe_tag,
                    exe_threads,
                    f"{wall_seconds:.6f}",
                    returncode,
                    "" if no_io else stdout_file,
                    "" if no_io else (stderr_file if returncode != 0 else "")
                ])

            if returncode != 0:
                print(f"❌ Error during run {i+1}/{n} for {exe}: return code {returncode}")
                sys.exit(returncode)
                break
            else:
                print(f"✅ {exe_tag} succeeded in {wall_seconds:.3f}s")

        except Exception as e:
            wall_seconds = time.perf_counter() - t0
            print(f"⚠️ Exception during run {i+1}/{n} for {exe}: {e} (elapsed {wall_seconds:.3f}s)")
            with open(summary_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow([
                    i + 1,
                    exe_tag,
                    exe_threads,
                    f"{wall_seconds:.6f}",
                    -1,
                    "" if no_io else stdout_file,
                    "" if no_io else stderr_file
                ])
            break
