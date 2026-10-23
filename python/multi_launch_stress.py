#!/usr/bin/env python3
# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
multi_launch_stress.py -- stress runner: the check binaries, over and over,
each run at a random thread count.

Two ways to name the binaries:

  * historical: run it INSIDE a build directory, the binaries are
    ./shrink_bench_<variant>_<size> in the current directory;

  * --builds: run it from anywhere (the python/ directory, typically) and give
    it the builds to exercise, either inline or as a file, one name per line:

        python3 multi_launch_stress.py --builds build_arm.txt
        python3 multi_launch_stress.py --builds omp-llvm,omp-gnu

    A name is a directory under --build-root (default: the repository, i.e.
    the parent of this script, or $CHUNK_BUILD_ROOT); "omp-llvm" and
    "cmake-build-omp-llvm" both name cmake-build-omp-llvm. The binaries are
    found with bench_and_test.py's resolve_exe, so every layout it knows is
    accepted (the CEA one with its $arch level included). The runs then
    CYCLE through the builds: run 1 on the first, run 2 on the second, ...
    and back to the first, so a long session spreads its draws over all of
    them. A build's binaries execute in THEIR directory, where the reference
    files (data_tab_*, the oracle a check run reads) live, as in the
    historical mode; the outputs, the CSV and the log go to a directory of
    their own under the build root, multi_launch_runs/<arch>_<date_time>, one per
    session (--run-dir names another).
    <arch> is the machine (aarch64, x86_64): the build root is shared by
    both, so are the session directories.

  --regen-ref regenerates the reference before the runs, with the vector
  (std::vector) check binary: in the current directory in the historical
  mode, in every build's binary directory under --builds -- each build's
  reference comes from its own compiler. With --only-checksum, that
  reference holds the checksums only (no data): a mismatch aborts the run,
  a full reference is needed to locate it.
"""

import os
import random
import glob
import time
import csv
import subprocess
import argparse
import difflib
import sys

# --numa-interleave: numactl prefix for runs spanning several NUMA nodes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import numa_launch as _numa  # noqa: E402
except ImportError:
    # Copied alone next to the binaries (cp python/<script> build_dir/): run
    # without NUMA interleave rather than not at all. Copy numa_launch.py too,
    # or run the script from python/, to get it back.
    import types as _types  # noqa: E402

    _numa = _types.SimpleNamespace(MODES=("auto", "off"), interleave_prefix=lambda env, mode="auto": [])
# --exec chunk2a & co.: the old variant labels, read as their new names until 1.0.
try:
    from bench_runner.settings import normalize_variant  # noqa: E402
except ImportError:
    # Copied alone: the new labels only.
    def normalize_variant(label, warn=None):
        return label
    print(f"[{os.path.basename(__file__)}] numa_launch.py not found next to the script: "
          "--numa-interleave has no effect", file=sys.stderr)

import atexit
import datetime
import platform


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
parser.add_argument("--builds", default=None, metavar="LIST|FILE",
                    help="Builds to cycle through: a comma-separated list of build names, "
                         "or a file with one name per line ('#' comments). A name is a "
                         "directory under --build-root, with or without the cmake-build- "
                         "prefix. Without it, the binaries are taken from the current "
                         "directory (historical behaviour).")
parser.add_argument("--build-root", default=os.environ.get("CHUNK_BUILD_ROOT"),
                    metavar="DIR",
                    help="Where the build directories live (default: $CHUNK_BUILD_ROOT, "
                         "else the repository this script sits in).")
parser.add_argument("--run-dir", default=None, metavar="DIR",
                    help="Where the output_*/error_* files, the CSV and the log go. Default: "
                         "the current directory in the historical mode (the binaries run there "
                         "too and read the oracle data_tab* there); under --builds a directory "
                         "of its own, <build-root>/multi_launch_runs/<arch>_<date_time>, and each "
                         "build's binaries run in their own directory, with their own oracle.")
parser.add_argument("--regen-ref", action="store_true",
                    help="Regenerate the reference files (data_tab_*) with the vector check "
                         "binary before the runs: in the current directory (historical mode), "
                         "or in each build's binary directory (--builds), from that build's "
                         "own vector binary. The old data_tab_* files are removed first.")
# No "regen" in this help: the GUI's option filter matches flags AND help text.
parser.add_argument("--only-checksum", action="store_true",
                    help="With the reference rebuilt before the runs, the vector binary writes "
                         "the data_tab_*_check_sum files only (shrink_bench --checksum_only), no "
                         "data: ~20 GB less per oracle. The checks are then bit-exact (no "
                         "1e-10 / 1e-6 element tolerance) and a mismatch aborts without "
                         "locating the element: rebuild a full reference to diagnose it.")

# ⭐ Selection = VARIANTS x SIZES. Binaries are named
#   shrink_bench_<variant>[_<size>]  with size in 64/256/1024/dyn, run
#   with --mode check,
#   plus a bare 'default' declination with no suffix (compile-time default).
# Both options accept a comma-separated list or 'all', so subsets compose:
#   --size dyn                  -> the 3 dyn binaries (default behaviour)
#   --exec zero_any             -> the 5 zero_any binaries
#   --exec zero,zero_any --size 64,dyn -> 4 binaries
#   --all_exec                  -> the full matrix
_VARIANTS = ["zero", "zero_one_value", "zero_any"]
# The TBB bench (shrink_bench_tbb_<variant>_<size>, std-backend builds made
# with CHUNK_WITH_TBB): the same sources with the regions run as TBB tasks.
# Part of the selection like the others: 'all' and --all_exec include it, and
# so does the default (zero_one_value AND tbb_zero_one_value). A build without
# it (OpenMP, serial, a std build made without oneTBB) simply has no such
# binary: skipped silently unless the variant was named in --exec.
_TBB_VARIANTS = ["tbb_" + v for v in _VARIANTS]
_ALL_VARIANTS = _VARIANTS + _TBB_VARIANTS
_DEFAULT_VARIANTS = ["zero_one_value", "tbb_zero_one_value"]
_SIZES         = ["64", "256", "1024", "dyn"]   # VALID choices (64 can still be asked with --size 64)
_DEFAULT_SIZES = ["256", "1024", "dyn"]              # the sizes built by default (64 dropped)

class _Choice(list):
    """A parsed comma-separated selection; `named` is False when it came from 'all'."""
    named = True


def _csv_choices(name, valid):
    """Comma-separated values of valid, or 'all' (every one of them)."""
    def parse(value):
        if value == "all":
            out = _Choice(valid)
            out.named = False
            return out
        items = [v.strip() for v in value.split(",") if v.strip()]
        if name == "--exec":
            items = [normalize_variant(v, name) for v in items]
        bad = [v for v in items if v not in valid]
        if bad:
            raise argparse.ArgumentTypeError(
                f"{name}: invalid value(s) {', '.join(bad)} "
                f"(valid: {', '.join(valid)}, or 'all', comma-separated)")
        # dedupe, keep user order
        return _Choice(dict.fromkeys(items))
    return parse

parser.add_argument(
    "--exec",
    type=_csv_choices("--exec", _ALL_VARIANTS),
    default=None,
    metavar="{zero,zero_one_value,zero_any,tbb_zero,tbb_zero_one_value,tbb_zero_any,all}[,...]",
    help="Variant(s) to run, comma-separated or 'all' (the six). Default: zero_one_value "
         "and tbb_zero_one_value unless --all_exec is given. tbb_<variant>: the TBB bench "
         "(shrink_bench_tbb_<variant>_<size>, std-backend builds made with "
         "CHUNK_WITH_TBB); a build without it is skipped for those, silently "
         "unless the variant is named here."
)
parser.add_argument(
    "--size",
    type=_csv_choices("--size", _SIZES),
    default=None,
    metavar="{64,256,1024,dyn,default,all}[,...]",
    help="Size declination(s), comma-separated or 'all'. 'default' is the "
         "bare binary without suffix (compile-time default size). Default: dyn."
)

parser.add_argument("--numa-interleave", choices=list(_numa.MODES), default="auto",
                    help="auto: launch under `numactl --interleave=<nodes>` when the run spans "
                         "2+ NUMA nodes (the CPU nodes of the cores it uses). off: never.")

parser.add_argument("--keep-stderr", action="store_true",
                    help="Keep the stderr of SUCCESSFUL runs as stderr_<run>_<build>_<exe>.txt "
                         "(default: removed; error_*.txt is kept for failed runs only).")

args = parser.parse_args()

# Everything below works from --run-dir: outputs, CSV, log, and the oracle the
# check mode reads. The historical mode (no --builds) also finds its binaries
# there, as before. The paths given on the command line are resolved first,
# relative to where the command was typed.
if args.builds and os.path.isfile(args.builds):
    args.builds = os.path.abspath(args.builds)
if args.build_root:
    args.build_root = os.path.abspath(args.build_root)
if args.run_dir is None:
    if args.builds:
        # One directory per session under the build root (the work directory
        # on the cluster): nothing lands in python/, nothing overwrites the
        # previous campaign's files.
        _root = args.build_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # The architecture first: the build root is shared by the ARM and x86
        # machines, and their campaigns must not be told apart by date only.
        args.run_dir = os.path.join(_root, "multi_launch_runs",
                                    f"{platform.machine()}_"
                                    + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    else:
        args.run_dir = "."
os.makedirs(args.run_dir, exist_ok=True)
os.chdir(args.run_dir)
if args.builds:
    print(f"📁 Outputs, CSV and log in {os.getcwd()}")

# Resolve the selection. --all_exec expands both axes (unless the user
# narrowed one explicitly, which then wins for that axis).
if args.all_exec:
    variants = args.exec if args.exec else list(_ALL_VARIANTS)
    sizes    = args.size if args.size else list(_DEFAULT_SIZES)
else:
    variants = args.exec if args.exec else list(_DEFAULT_VARIANTS)
    sizes    = args.size if args.size else ["dyn"]
# A TBB binary missing from a build is only worth a warning when it was named.
_tbb_named = bool(args.exec) and args.exec.named and any(v in _TBB_VARIANTS for v in args.exec)

# 📝 Tee all screen output into a log file in the current directory, next to
# timings_summary.csv and the per-run output_*/error_* files.
_console_log_path = "multi_launch_output.log"
try:
    _console_log_file = open(_console_log_path, "w")
    _console_log_file.write(
        f"# multi_launch_stress.py run log — "
        f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
    _console_log_file.write("# command: " + " ".join(sys.argv) + "\n")
    _console_log_file.write("# run dir: " + os.getcwd() + "\n\n")
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
    for file in glob.glob('output_*.txt') + glob.glob('error_*.txt') + glob.glob('stderr_*.txt'):
        os.remove(file)

# 🔧 Executables list: VARIANTS x SIZES, filtered by what actually exists on
# disk (a size declination may not be built in this tree — warn, don't die,
# unless the whole selection ends up empty).
# The check mode is no longer in the binary's NAME: check and perf are one
# executable since they were merged, and the mode is chosen at run time with
# --mode (see bench_config.h). Hence shrink_bench_<variant>[_<size>] here, and
# MODE_ARGS added to every launch below.
MODE_ARGS = ["--mode", "check"]


def _binary_name(variant, size):
    suffix = "" if size == "default" else f"_{size}"
    return f"./shrink_bench_{variant}{suffix}"


def _read_builds(spec):
    """--builds: a file (one name per line, blank lines and '#' comments
    skipped) or an inline comma-separated list. Order kept, duplicates dropped."""
    if os.path.isfile(spec):
        with open(spec) as f:
            names = []
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.extend(n.strip() for n in line.split(",") if n.strip())
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        sys.exit(f"❌ --builds {spec}: no build name found")
    return list(dict.fromkeys(names))


def _build_dir_name(root, name):
    """'omp-llvm' -> 'cmake-build-omp-llvm' when that directory exists under
    root; a name that is already a directory is kept as it is."""
    for cand in (name, f"cmake-build-{name}"):
        if os.path.isdir(os.path.join(root, cand)):
            return cand
    # A run-time mode of another build (bench_runner/settings.py,
    # VIRTUAL_CONFIG_ENV): kept when its base build exists.
    from bench_runner.settings import virtual_base
    for cand in (name, f"cmake-build-{name}"):
        base = virtual_base(cand)
        if base and os.path.isdir(os.path.join(root, base)):
            return cand
    return None


# TBB binaries the selection asked for by default and that are not built.
_tbb_absent = []


def _select_in_cwd():
    """Historical mode: the binaries next to the oracle, in the current directory."""
    found, missing = [], []
    for v in variants:
        for sz in sizes:
            exe = _binary_name(v, sz)
            if os.path.isfile(exe):
                found.append(exe)
            elif v in _TBB_VARIANTS and not _tbb_named:
                _tbb_absent.append(exe)
            else:
                missing.append(exe)
    return found, missing


VECTOR_CHECK = "./shrink_bench_vector_check"


def _select_in_build(build_root, config):
    """--builds mode: bench_and_test.py's resolve_exe knows every layout a
    build directory may have (flat, examples/shrink_bench, the $arch level of
    the cluster), so the selection is the same wherever this runs."""
    found, missing = [], []
    for v in variants:
        for sz in sizes:
            exe = resolve_exe(build_root, config, f"{v}@{sz}", "check")
            if exe:
                found.append(exe)
            elif v in _TBB_VARIANTS and not _tbb_named:
                _tbb_absent.append(f"{config}/shrink_bench_{v}_{sz}")
            else:
                missing.append(f"{config}/shrink_bench_{v}_{sz}")
    return found, missing


# builds: the list to cycle through; executables_of[build] its binaries,
# ref_dir_of[build] where they run and read their oracle, vector_of[build]
# the vector check binary --regen-ref generates it with. The historical mode
# is one unnamed build, the current directory.
ref_dir_of = {}
vector_of = {}
if args.builds:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_runner.builds import resolve_exe  # the layout logic, shared with bench_and_test.py

    build_root = args.build_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builds = []
    executables_of = {}
    for name in _read_builds(args.builds):
        config = _build_dir_name(build_root, name)
        if config is None:
            print(f"⚠️ build '{name}': no directory {name} nor cmake-build-{name} under {build_root}, skipped")
            continue
        found, missing = _select_in_build(build_root, config)
        if missing:
            print(f"⚠️ {config}: {len(missing)} selected binary(ies) not found, skipped: {', '.join(missing)}")
        if not found:
            print(f"⚠️ {config}: no selected binary built, skipped")
            continue
        vector_exe = resolve_exe(build_root, config, "vector", "check")
        if args.regen_ref and vector_exe is None:
            print(f"⚠️ {config}: no vector check binary to regenerate the reference with, skipped")
            continue
        builds.append(config)
        executables_of[config] = found
        ref_dir_of[config] = os.path.dirname(found[0])
        vector_of[config] = vector_exe
        print(f"🎯 {config}: {len(found)} binary(ies): {', '.join(os.path.basename(e) for e in found)}")
    if not builds:
        sys.exit("❌ No build has a selected binary — nothing to run "
                 f"(variants={','.join(variants)} sizes={','.join(sizes)}, root={build_root})")
    print(f"🔁 {len(builds)} build(s), cycled run after run: {', '.join(builds)}")
    if _tbb_absent:
        with_tbb = [b for b in builds if any(os.path.basename(e).startswith("shrink_bench_tbb_")
                                             for e in executables_of[b])]
        print(f"ℹ️ TBB bench (tbb_*): {'in ' + ', '.join(with_tbb) if with_tbb else 'in no build'}; "
              "the other builds have none (OpenMP or serial, or a std build made without oneTBB)")
else:
    found, missing = _select_in_cwd()
    if missing:
        print(f"⚠️ {len(missing)} selected binary(ies) not found, skipped: {', '.join(missing)}")
    if not found:
        sys.exit("❌ No selected binary exists on disk — nothing to run "
                 f"(variants={','.join(variants)} sizes={','.join(sizes)})")
    builds = [""]
    executables_of = {"": found}
    ref_dir_of[""] = None  # the current directory
    vector_of[""] = VECTOR_CHECK if os.path.isfile(VECTOR_CHECK) else None
    if args.regen_ref and vector_of[""] is None:
        sys.exit(f"❌ --regen-ref: {VECTOR_CHECK} not found in {os.getcwd()}")
    print(f"🎯 {len(found)} binary(ies) selected: {', '.join(e.lstrip('./') for e in found)}")
    if _tbb_absent:
        print(f"ℹ️ no TBB bench here ({len(_tbb_absent)} tbb_* binary(ies) not built): skipped")

# 🧬 Reference regeneration: the vector (std::vector) check binary writes the
# data_tab_* oracle the chunk binaries compare against. Once per build, in the
# directory its binaries run in, after removing the previous files -- a check
# run against a partial oracle aborts on the first missing dump. Not under
# NO_IO, which would disable the very writes this is about.
if args.only_checksum and not args.regen_ref:
    print("⚠️ --only-checksum without --regen-ref: no effect, the existing reference is used as is")
if args.regen_ref:
    if no_io:
        print("⚠️ --regen-ref with --no_io: the runs will not read the reference just regenerated")
    # Half of --max_threads for the vector binary: it has no compression to
    # exercise, and at the top of the range its memory bandwidth, not its
    # thread count, is what bounds it.
    ref_threads = str(max(1, min(max_threads // 2, os.cpu_count() or 1)))
    for build in builds:
        ref_dir = ref_dir_of[build] or os.getcwd()
        old = glob.glob(os.path.join(ref_dir, "data_tab*"))
        for f in old:
            os.remove(f)
        label = build or os.path.basename(os.path.abspath(ref_dir))
        print(f"🧬 Regenerating the reference for {label} in {ref_dir} "
              f"({len(old)} old file(s) removed, {ref_threads} threads)...")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = ref_threads
        env.pop("NO_IO", None)
        log_name = f"regen_ref_{build}.txt" if build else "regen_ref.txt"
        t0 = time.perf_counter()
        with open(log_name, "w") as log:
            result = subprocess.run(_numa.interleave_prefix(env, args.numa_interleave) + [vector_of[build]] + MODE_ARGS
                                    + (["--checksum_only"] if args.only_checksum else []),
                                    check=False, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=ref_dir_of[build])
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            sys.exit(f"❌ Reference regeneration failed for {label} (return code {result.returncode}, "
                     f"see {log_name})")
        n_new = len(glob.glob(os.path.join(ref_dir, "data_tab*")))
        print(f"✅ Reference for {label}: {n_new} file(s) in {wall:.1f}s")

# 📄 Summary CSV
summary_file = "timings_summary.csv"
print(f"📄 Creating summary file: {summary_file}")

with open(summary_file, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["run_index", "build", "executable", "num_threads", "wall_seconds",
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

    # The build of this run: the next one in the cycle.
    build = builds[i % len(builds)]
    executables = list(executables_of[build])
    if build:
        print(f"🏗️  Run {i + 1}/{n} on build {build} ({(i % len(builds)) + 1}/{len(builds)} of the cycle)")

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

        # What the console shows for this binary: "<build>/<name>" in --builds
        # mode (the absolute path would bury the build), the path otherwise.
        exe_label = f"{build}/{os.path.basename(exe)}" if build else exe
        print(f"🧵 Threads used for {exe_label}: {exe_threads}")
        print(f"🚀 Launching {i+1}/{n} → {exe_label}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(exe_threads)
        if build:
            from bench_runner.settings import config_env
            env.update(config_env(build))  # a run-time mode: CHUNK_AGGRESSIVE_COMPRESSION=1
        # NUMA binding: pin threads to cores so data stays on the local socket.
        # On this 4-socket machine, unbound threads migrate across sockets and
        # hammer remote memory (huge slowdown). Overridable from the outer env.
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
        # The re-chunk decider's moves (the stats builds' trace on stderr):
        # log messages of level info, not written by default.
        env.setdefault("CHUNK_LOG", "warn,rechunk=info")
        # Compression is off by default in shrink_bench (opt-in: --compress /
        # COMPRESS=1), so nothing to set here any more.
        if no_io:
            env["NO_IO"] = "1"

        numa_prefix = _numa.interleave_prefix(env, args.numa_interleave)
        if numa_prefix:
            print(f"🧭 {' '.join(numa_prefix[1:])} ({exe_threads} threads span several NUMA nodes)")

        exe_tag = os.path.basename(exe)
        # The build in the file names too, so two builds' runs never collide.
        file_tag = f"{build}_{exe_tag}" if build else exe_tag
        stdout_file = f"output_{i + 1}_{file_tag}.txt"
        stderr_file = f"error_{i + 1}_{file_tag}.txt"

        t0 = time.perf_counter()
        try:
            # cwd: the build's own directory under --builds (its oracle lives
            # there), the current directory otherwise (None).
            if no_io:
                result = subprocess.run(numa_prefix + [exe] + MODE_ARGS, check=False, env=env, cwd=ref_dir_of[build])
            else:
                with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                    result = subprocess.run(numa_prefix + [exe] + MODE_ARGS, check=False, stdout=stdout,
                                            stderr=stderr, env=env, cwd=ref_dir_of[build])

            wall_seconds = time.perf_counter() - t0
            returncode = result.returncode

            # error_*.txt only for a FAILED run: a successful one may still
            # write to stderr (the re-chunk decider's moves, CHUNK_LOG=rechunk=info).
            if not no_io and os.path.isfile(stderr_file):
                if os.stat(stderr_file).st_size == 0:
                    os.remove(stderr_file)
                    stderr_file = ""
                elif returncode == 0:
                    if args.keep_stderr:
                        os.replace(stderr_file, "stderr_" + stderr_file[len("error_"):])
                    else:
                        os.remove(stderr_file)
                    stderr_file = ""

            with open(summary_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow([
                    i + 1,
                    build,
                    exe_tag,
                    exe_threads,
                    f"{wall_seconds:.6f}",
                    returncode,
                    "" if no_io else stdout_file,
                    "" if no_io else (stderr_file if returncode != 0 else "")
                ])

            if returncode != 0:
                print(f"❌ Error during run {i+1}/{n} for {exe_label}: return code {returncode}")
                sys.exit(returncode)
                break
            else:
                print(f"✅ {exe_label} succeeded in {wall_seconds:.3f}s")

        except Exception as e:
            wall_seconds = time.perf_counter() - t0
            print(f"⚠️ Exception during run {i+1}/{n} for {exe_label}: {e} (elapsed {wall_seconds:.3f}s)")
            with open(summary_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow([
                    i + 1,
                    build,
                    exe_tag,
                    exe_threads,
                    f"{wall_seconds:.6f}",
                    -1,
                    "" if no_io else stdout_file,
                    "" if no_io else stderr_file
                ])
            break
