#!/usr/bin/env python3
"""
bench_and_test.py — Benchmark + correctness stress runner for shrink-array.

Supported layout (the CEA cluster one):
    $build_root/{config}/$arch/examples/shrink_bench/shrink_bench_{var}_{stage}

Example:
    $CCCWORKDIR/shrink-array/
      ├── cmake-build-omp-gnu/RHEL_9__aarch64/examples/shrink_bench/shrink_bench_chunk2_dyn
      ├── cmake-build-omp-gnu-have-agressive-compression/.../shrink_bench_chunk2_dyn
      ├── cmake-build-omp-gnu-std-alloc/...
      ├── cmake-build-omp-llvm/...
      └── ...

Typical usage — perf comparison release vs aggressive:
    python3 bench_and_test.py --skip-check \\
        --build-root $CCCWORKDIR/shrink-array \\
        --configs cmake-build-omp-gnu \\
                  cmake-build-omp-gnu-have-agressive-compression \\
                  cmake-build-omp-gnu-std-alloc

Or more simply, if you are already INSIDE $build_root:
    python3 bench_and_test.py --skip-check

Notable features:
  * Auto-resolution via glob {config}/*/examples/shrink_bench/... ($arch is
    discovered). Also supports flat executables (the old layout).
  * Explicit warning per missing binary — never a silent skip.
  * Only the current machine's arch is run (aarch64 vs x86_64), so a build_root
    holding both architectures does not trigger "Exec format error".
  * One reference oracle PER CONFIG (output_dir/oracle/<config>/data_tab*),
    generated once by that config's own vector binary and kept across runs:
    the reference is not portable across compilers (see oracle_dir()).
  * Per-run timeout + capture of stderr from crashing runs.
  * Validation harnesses first (ctest -L validation in each build dir):
    rechunk_validation, dyn_geometry_validation, compress_roundtrip,
    pool_accounting, deferred_path_validation, open_types_validation.
    --skip-ctest to bypass.
  * Correctness before timings: check runs, then the seq-vs-OMP
    `still compressed (at exit)` compare, then the perf sweep. The two
    CHUNK_STATS checks (that compare, and the mark/create guard applied to
    every run) need a -stats config in the run -- the defaults include
    cmake-build-omp-gnu-stats / cmake-build-gnu-stats, checked and never
    timed; a run without any prints a warning up front.
  * Clean Ctrl+C: the CSV and the error log are written even on interruption.
"""
import subprocess
import os
import sys

# --numa-interleave: numactl prefix for runs spanning several NUMA nodes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numa_launch as _numa  # noqa: E402
_NUMA_MODE = "auto"  # set from --numa-interleave in main()
import time
import argparse
import json  # --rerun-failed: failed_cases.jsonl
import csv
from pathlib import Path

# --- Colour / display helpers --------------------------------------------
# Colour is emitted only to a real terminal and only when NO_COLOR is unset,
# so redirected logs / CSV pipelines stay clean (no escape sequences on disk).
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

# Optional per-run full-output capture (--full-log-dir). When set, run_cmd writes
# the COMPLETE stdout+stderr of every child run to its own file, so nothing is lost
# to the 30-line err_tail: you can see whether a re-chunk fired (it prints to stdout,
# which the summary otherwise drops) and the full crash context. Off by default.
import threading
_FULL_LOG_DIR = None
_FULL_LOG_LOCK = threading.Lock()
_FULL_LOG_COUNTER = 0

def _set_full_log_dir(path):
    global _FULL_LOG_DIR
    _FULL_LOG_DIR = path


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else str(s)


def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)
def blue(s):   return _c("34", s)
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)


import re as _re
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")

class _Tee:
    """Mirror everything written to the real stdout into a log file as well.

    The screen keeps the live experience (colours + \r-animated progress bars).
    The file gets a cleaned copy: ANSI colour codes stripped, and carriage-return
    progress redraws collapsed so the log holds only finalised lines (the same
    text you'd see if you scrolled back), not hundreds of half-drawn bars.
    """
    def __init__(self, real, log_file):
        self._real = real
        self._log = log_file
        self._pending = ""   # buffer for the current (possibly \r-rewritten) line

    def write(self, s):
        # 1) screen: unchanged, immediately
        self._real.write(s)
        # 2) file: strip colours, resolve \r rewrites into single final lines
        try:
            clean = _ANSI_RE.sub("", s)
            for ch in clean:
                if ch == "\r":
                    # progress redraw: drop whatever was buffered for this line
                    self._pending = ""
                elif ch == "\n":
                    self._log.write(self._pending + "\n")
                    self._pending = ""
                else:
                    self._pending += ch
        except Exception:
            pass  # logging must never break the run

    def flush(self):
        self._real.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        # keep colour/tty detection keyed to the REAL terminal
        return getattr(self._real, "isatty", lambda: False)()

    def close_log(self):
        try:
            if self._pending:
                self._log.write(self._pending + "\n")
                self._pending = ""
            self._log.flush()
            self._log.close()
        except Exception:
            pass


# Fixed field widths so every progress line aligns regardless of config/variant
# name length. CONFIG_W fits the longest short label
# ("llvm-have-agressive-compression" = 31) with one space of slack, so nothing
# is truncated (the old [:30] chopped the trailing 'n' off "...compression").
CONFIG_W = 32
VARIANT_W = 14  # longest label is "std_alloc@1024" (14); avoids truncating "chunk2a@256"

# Fixed-width column widths for the live progress line, shared by every phase
# (CHECK / SAN / PERF / STL) so their columns line up vertically. Any phase that
# doesn't use a column passes None and gets a blank of the right width.
TH_W  = 5   # "  8th"
RUN_W = 11  # "Run:  1/16 "


def _hms():
    """Wall-clock HH:MM:SS, stamped at the head of every per-test progress line
    in every phase.

    A full run is hours long and each line is rewritten in place over its own
    predecessor, so without a stamp the log says what was attempted but never
    WHEN: a run that took forty minutes on a loaded node and one that hung look
    identical after the fact, and a phase that turned out to have nothing to do
    (see the -stats pairs in run_seq_omp_compare_phase) is indistinguishable
    from one that did real work. The stamp is taken when the line is composed,
    i.e. just before the run it describes starts, so consecutive lines read as a
    timeline and the gap between two of them IS that run's duration.
    """
    return time.strftime("%H:%M:%S")


def _prog(phase, color_fn, cfg, var="", *, th=None, run=None, reps=None, extra=""):
    """Format one live progress line with fixed columns:

        HH:MM:SS PHASE  config......... | variant....... |   Nth | Run:  r/reps | extra

    The phase label is padded to width BEFORE colouring, so ANSI escape codes
    never enter the width computation (which would break alignment). Columns the
    phase doesn't use (th / run) are blanked to their fixed width.
    """
    th_s  = f"{th:>3}th" if th is not None else " " * TH_W
    run_s = (f"Run:{run:3}/{reps:<3}" if (run is not None and reps is not None)
             else " " * RUN_W)
    return (f"{dim(_hms())} "
            f"{color_fn(f'{phase:<5}')} "
            f"{short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
            f"{var[:VARIANT_W]:<{VARIANT_W}} | "
            f"{th_s} | {run_s} | {extra}")


# --- Default configuration -----------------------------------------------
# "Configs" = sub-directories of $build_root. The first entry is used as the
# baseline for the speedup computation.
# Config-name substrings that only change chunk* code paths (compression
# scheduling internals) and leave the vector binary byte-for-byte identical to
# the base config. On such configs we skip the compression-insensitive variants
# (see COMPRESSION_INSENSITIVE_VARIANTS) to avoid re-timing identical binaries.
COMPRESSION_ONLY_CONFIG_MARKERS = ["have-agressive-compression"]

# Variants whose binary does not depend on the aggressive-compression knob, so
# they never need to be run on a COMPRESSION_ONLY config. vector is a plain
# non-chunk path, identical to the base config there. (std_alloc is handled
# separately: it's only built in -std-alloc configs, see is_std_alloc_config.)
COMPRESSION_INSENSITIVE_VARIANTS = ["vector"]

DEFAULT_CONFIGS = [
    "cmake-build-omp-gnu",                             # release GNU (baseline)
    "cmake-build-omp-gnu-have-agressive-compression",  # aggressive scheduling
    "cmake-build-omp-gnu-std-alloc",                   # std allocator
    "cmake-build-omp-llvm",                            # release LLVM
    "cmake-build-omp-llvm-have-agressive-compression",
    "cmake-build-omp-llvm-std-alloc",
    # x86-64 only: the other OpenMP compiler compile_all.sh builds there (an
    # OMPT build). On ARM its directory does not exist and the absent-config
    # step drops it with one note. AOCC is not built by default any more
    # (AOCC 5 has no working C++23 with libstdc++ 15; compile_all.sh --with-aocc).
    "cmake-build-omp-oneapi",                          # release oneAPI (icpx)
    # Non-OpenMP builds — cover the `if constexpr(!use_openmp)` code paths so a
    # serial-path regression is caught. Build them with -DOMP:BOOL=OFF; if the
    # directory is absent it is skipped with a single note (see
    # drop_absent_configs), not one MISSING warning per variant.
    "cmake-build-gnu",                                 # non-OMP GNU
    "cmake-build-llvm",                                # non-OMP LLVM
    # CHUNK_STATS builds (-DCHUNK_STATS=ON). The two stats checks -- the
    # mark/create compression-health guard on every run, and the seq-vs-OMP
    # `still compressed (at exit)` compare -- only have something to read on
    # these: the counters are compiled out everywhere else, and with no stats
    # config in the run both checks silently pass. Checked, never timed
    # (PERF_EXCLUDED_CONFIGS). Absent directory = skipped with a note.
    "cmake-build-omp-gnu-stats",                       # OMP GNU, CHUNK_STATS
    "cmake-build-gnu-stats",                           # non-OMP GNU, CHUNK_STATS
]

DEFAULT_VARIANTS       = ["vector", "chunk", "chunk2", "chunk2a", "std_alloc"]

# ── Sanitizer CHECK configs ──────────────────────────────────────────────────
# Two llvm sanitizer build dirs (created by compile_all.sh) run in the CHECK
# phase only. They are diagnostic, not timed: run at a fixed LOW thread count
# (sanitizers are 5-20x slower, and TSan has a hard thread ceiling + exploding
# shadow memory past ~64, so 8 is the sweet spot — enough concurrency for a race
# to be *detectable* (TSan flags missing synchronisation even without a live
# overlap), cheap enough to run). Env options below silence the OpenMP-runtime
# false positives and make hits deterministic. A run is a FAIL if the exit code
# is non-zero OR the output contains a sanitizer report (ASan can continue past
# an error with exit 0 depending on options).
SANITIZER_CONFIGS = [
    "cmake-build-omp-llvm-debug-std-alloc-address-sanitizer",  # ASan + std-alloc (would have caught fw14)
    "cmake-build-omp-llvm-debug-thread-sanitizer",             # TSan (ARM race net)
]
SANITIZER_THREADS = 8
# One run per (config, variant): a sanitizer hit is deterministic enough on a
# single pass (TSan flags missing synchronisation without needing a live
# overlap, ASan reports on first touch), and each run is 5-20x a normal one --
# three passes tripled the cost of the CHECK phase for no extra signal.
SANITIZER_REPS    = 1

def sanitizer_env_for(config):
    """ASAN/TSAN_OPTIONS for a sanitizer config. TSan needs
    ignore_noninstrumented_modules=1 to avoid a flood of false positives from
    the (non-instrumented) OpenMP runtime; history_size widens the other-thread
    stack on real ARM races. ASan enables leak detection and deterministic abort.
    """
    if "thread-sanitizer" in config:
        return {"TSAN_OPTIONS": "ignore_noninstrumented_modules=1:halt_on_error=1:history_size=7"}
    # address sanitizer
    return {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1:halt_on_error=1"}

# Substrings marking a sanitizer report in stdout/stderr — a hit means FAIL even
# when the process exited 0. Kept to the unambiguous sanitizer banner names
# (each report prints its name before the SUMMARY line), so a normal bench run
# can never trip them.
SANITIZER_HIT_MARKERS = (
    "AddressSanitizer", "ThreadSanitizer", "LeakSanitizer",
)

def is_sanitizer_config(config):
    return "sanitizer" in config

# Configs to DROP from the PERF phase (kept in CHECK). llvm-std-alloc perf is
# redundant with gnu-std-alloc (near-identical numbers) and gnu is the perf/
# correctness judge; cutting it reclaims cluster time. Sanitizer configs are
# never timed.
PERF_EXCLUDED_CONFIGS = [
    "cmake-build-omp-llvm-std-alloc",
    # Instrumented builds: their timings measure the counters, not the library.
    "cmake-build-omp-gnu-stats",
    "cmake-build-gnu-stats",
]


def is_stats_config(config):
    """True for a CHUNK_STATS build (its dir name carries -stats). Only these
    print the CHUNK_STATS block the two stats checks read."""
    return config.endswith("-stats") or "-stats-" in config
DEFAULT_BASELINE       = "vector"     # used for the speedup computation
DEFAULT_THREADS_PERF   = [8, 16, 32, 64, 128, 288]
DEFAULT_THREADS_CHECK  = 64
DEFAULT_REPS_PERF      = 50
DEFAULT_REPS_CHECK     = 64
DEFAULT_TIMEOUT        = 300  # base seconds per run (see --timeout-per-thread)
DEFAULT_TIMEOUT_PER_THREAD = 0  # extra seconds added per thread (0 = disabled)

# Thread count(s) that get the full --reps_perf averaged measurement by default.
# All other thread counts in --threads get a single run (scalability curve).
DEFAULT_FULL_REPS_THREADS = [64]

# --- Seq-vs-OMP compression-classification compare (seed-pinned) ---------
# See run_seq_omp_compare_phase() for the full rationale. Defaults picked to
# match the exploratory session that motivated this phase: chunk-family
# variants only (vector/std_alloc have no compression state to classify),
# all three size declinations, a light thread sweep, a quick --divisor.
DEFAULT_SEQ_OMP_THREADS  = [8, 16, 64]
DEFAULT_SEQ_OMP_SIZES    = ["256", "1024", "dyn"]
DEFAULT_SEQ_OMP_VARIANTS = ["chunk", "chunk2", "chunk2a"]
DEFAULT_SEQ_OMP_SEED     = 42
DEFAULT_SEQ_OMP_ITER_MAX = 2000
DEFAULT_SEQ_OMP_DIVISOR  = 100

# --- Scalability guard (omp-llvm + omp-gnu) --------------------------------
# A short perf sweep whose POINT is the shape of the curve, not the absolute
# numbers: the wall time must keep going DOWN as threads are added, or at
# least not blow up. It catches the regressions the averaged perf phase hides
# -- a lock that only bites past 32 threads, a pass that became serial -- and
# it is cheap: one run per thread count and per perf variant (--variants x
# --sizes, vector always in as the reference curve), on two configs:
#   - release LLVM, the fastest of the matrix, whose curves are worth believing;
#   - release GNU, because libgomp has no OMPT: the phase-end path is not the
#     same (finalize_parallel_phase_after_flush calls destroy_after_flush after
#     EVERY phase), and that is where the AMD Milan stall was -- ~21 s of a
#     ~30 s chunk2a run on the master, 63 threads at the barrier -- which
#     neither the OMPT builds (LLVM, oneAPI, AOCC) nor this guard on LLVM alone
#     could see.
SCALING_CONFIGS  = ["cmake-build-omp-llvm", "cmake-build-omp-gnu"]
# The curve every container is read against: no chunk, no compression, so a
# bend it shares is the machine's, and one only the chunk variants show is ours.
SCALING_REFERENCE_VARIANT = "vector"
SCALING_THREADS  = [4, 8, 16, 32, 64, 128]
# The guard runs the int64 binaries (shrink_bench_<variant>_int64_<size>, see
# int64_variant) at --multiply 2: arrays past INT_MAX elements, so the curves
# also cover the 64-bit index path at the scale it exists for. --iter_max 500
# holds each point down; the sizes with no int64 declination (256) are left out.
SCALING_MULTIPLY = 2
SCALING_ITER_MAX = 500
# Floor of a point's timeout: twice the perf size at 4 threads is far from the
# few seconds of the old --divisor 50 points.
SCALING_TIMEOUT  = 3600
# Fallback of a config built WITHOUT the int64 binaries (-DSHRINK_BENCH_INT64
# off: a hand-configured build): its 32-bit binaries, on the shrunk workload
# the guard used before -- rather than no curve at all. Per config: a config
# that has int64 binaries stays int64, and its sizes without one (256) stay out.
SCALING_FALLBACK_DIVISOR  = 50
SCALING_FALLBACK_ITER_MAX = 200
# A point is a REGRESSION when it is slower than the BEST point so far by more
# than this factor. Against the best rather than the previous point, and 1.25
# rather than 1.10: measured on an i9-12900H (6 P-cores + 8 E-cores), the
# 16-thread point is 1.15x the 8-thread one because the threads past 12 land
# on E-cores -- the machine, not the library. A serialised pass is 1.5x and
# beyond, and against the best point a slow stretch cannot drift the reference
# along with it.
SCALING_MAX_SLOWDOWN = 1.25

# --- 64-bit index checks (int64 binaries) ----------------------------------
# A few --mode check runs of the int64 binaries against the 32-bit oracle of
# the same config: the data_tab* files hold element values, independent of the
# index type, so the int64 build must reproduce them exactly. One per variant,
# both geometries between them; chunk2a@dyn is the richest path (a second
# value per chunk, re-chunking). Not a sweep: the 32-bit checks cover the
# rest, and this only asks whether 64-bit indices change a result.
INT64_CHECK_VARIANTS = ["chunk@1024", "chunk2@1024", "chunk2a@dyn"]

# --- Mode presets --------------------------------------------------------
# Each preset overrides threads / reps_perf / full_reps_threads unless the user
# passes those flags explicitly. Vary both the thread distribution and the
# number of repetitions.
# The scalability guard (run_scaling_phase) is NOT part of these presets: it
# runs in every mode, on its own thread list, because a mode that trims the
# perf sweep is exactly the one that would otherwise miss a scaling
# regression. --skip-scaling is the way out.
#   fast   : compromise — 64/128 only, few reps on 64, no PERF on the serial
#            configs. About 1h30 on the full aarch64 matrix, not the ~10 min
#            this line used to claim: 1129 runs over 8 configs is not a
#            ten-minute job, and pretending otherwise made the mode look
#            broken rather than slow.
#   normal : the standard defaults above.
#   full   : the complete thread distribution with many reps on 64 and 128.
MODE_PRESETS = {
    "fast": {
        # 64 and 128 only. The 16-thread point was the most expensive run of the
        # sweep -- on aarch64, chunk2a@dyn took 100.3 s at 16 threads against a
        # 70.3 s average at 64 -- for the one end of the curve this mode is least
        # asked about. --mode normal/full still sweeps it.
        "threads":           [64, 128],
        # 3 reps instead of 10: on this workload the run-to-run spread at 64
        # threads is well under the differences the mode is meant to surface,
        # and the repetitions at 64 were the single biggest item left in the
        # perf phase (about 26 min of the 42 min gnu block at 5 reps). Use
        # --mode normal/full for numbers you intend to publish.
        "reps_perf":         3,
        "full_reps_threads": [64],
        "reps_check":        16,
        # Variants run as a single smoke test (one thread count, one run) just
        # to confirm they are not broken — they don't need the full sweep.
        "light_variants":    ["std_alloc"],
        "light_threads":     64,
    },
    "normal": {
        "threads":           DEFAULT_THREADS_PERF,
        "reps_perf":         DEFAULT_REPS_PERF,
        "full_reps_threads": DEFAULT_FULL_REPS_THREADS,
        "reps_check":        DEFAULT_REPS_CHECK,
        "light_variants":    [],
        "light_threads":     64,
    },
    "full": {
        "threads":           [1, 2, 4, 8, 16, 32, 64, 128, 192, 288],
        "reps_perf":         100,
        "full_reps_threads": [64, 128],
        "reps_check":        128,
        "light_variants":    [],
        "light_threads":     64,
    },
}

EXE_SUBPATH = "examples/shrink_bench"

# $arch sub-directory of the CURRENT machine. build_rep may hold builds for
# several architectures (RHEL_9.6__aarch64 AND RHEL_8.10__x86_64); we must only
# run the ones matching the machine the bench runs on, otherwise we get an
# "Exec format error" (x86 binary launched on ARM or vice-versa).
import platform as _platform
CURRENT_MACHINE = _platform.machine()  # 'aarch64' or 'x86_64'


def _arch_matches_current(arch_dirname):
    """True if the $arch directory name (e.g. 'RHEL_9.6__aarch64') matches the
    current machine. We match on the suffix after the last '__'."""
    # 'RHEL_9.6__aarch64' -> 'aarch64' ; robust if there is no '__'.
    tail = arch_dirname.rsplit("__", 1)[-1]
    return tail == CURRENT_MACHINE


def _variant_base(var):
    """Strip a "@<size>" declination suffix, returning the bare variant name
    (e.g. "chunk2a@256" -> "chunk2a", "vector" -> "vector"). Used wherever the
    logic keys on the variant identity rather than its size."""
    return var.split("@", 1)[0]


def int64_variant(var):
    """The int64 declination of a --variants entry, spelled for resolve_exe:
    "chunk2a@1024" -> "chunk2a_int64@1024" (shrink_bench_chunk2a_int64_1024),
    "vector" -> "vector_int64" (with stage perf: shrink_bench_vector_int64_perf).
    Built on the release OpenMP configs only, for 1024 and dyn (see
    examples/shrink_bench/CMakeLists.txt, SHRINK_BENCH_INT64)."""
    if "@" in var:
        base, size = var.split("@", 1)
        return f"{base}_int64@{size}"
    return f"{var}_int64"


def oracle_complete(output_dir, config):
    """Whether `config`'s oracle (oracle_dir) was written to the end."""
    return os.path.exists(os.path.join(oracle_dir(output_dir, config),
                                       "data_tab_ORACLE_COMPLETE.marker"))


def resolve_exe(build_root, config, var, stage):
    """
    Look up the executable for (config, var, stage) under several layouts:
      1. $build_root/$config/<arch>/examples/shrink_bench/shrink_bench_{var}_{stage}
         (CEA layout with $arch sub-directory — filtered on the current arch)
      2. $build_root/$config/examples/shrink_bench/shrink_bench_{var}_{stage}
         (build without the $arch level)
      3. $build_root/$config/shrink_bench_{var}_{stage}
         (config = flat directory)
      4. $build_root/shrink_bench_{var}_{stage}_{config}
         (old flat convention)
    Returns the absolute path of the first hit, or None.
    """
    # A variant may carry a size declination as "base@size" (e.g. "chunk2a@dyn",
    # "chunk2@256"). The executable name is then shrink_bench_{base}_{size}.
    #
    # No {stage} in that name any more: check and perf are ONE binary, the mode
    # being a run-time flag (--mode, added by run_with_rebuild_retry and by the
    # phases that build their own argument list). `stage` is still a parameter
    # because it still selects the MODE -- and because "vector" is the exception:
    # its two targets differ by compiler flags (fast-math on perf only), so they
    # stay separate executables and keep the staged name.
    if "@" in var:
        base, size = var.split("@", 1)
        exe_name = f"shrink_bench_{base}_{size}"
    else:
        exe_name = f"shrink_bench_{var}_{stage}"
    root = Path(build_root)

    # Layout 1: glob over the $arch level, but ONLY the current arch.
    for exe in sorted((root / config).glob(f"*/{EXE_SUBPATH}/{exe_name}")):
        # exe = .../{config}/{arch_dir}/examples/shrink_bench/{exe_name}
        # go up 3 levels to get the arch directory name.
        arch_dir = exe.parents[2].name
        if not _arch_matches_current(arch_dir):
            continue
        if exe.is_file() and os.access(exe, os.X_OK):
            return str(exe.resolve())

    # Layout 2: no $arch level
    candidate = root / config / EXE_SUBPATH / exe_name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    # Layout 3: flat executable inside config
    candidate = root / config / exe_name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    # Layout 4: old convention
    candidate = root / f"{exe_name}_{config}"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    return None


def config_dir_present(build_root, config):
    """True if `config` has at least one built executable for the CURRENT arch.
    Lets us skip an entirely-unbuilt config (e.g. a non-OMP dir the user did not
    build) with one note instead of a MISSING warning per variant."""
    # Probe both legacy names (vector, or a build without size declinations)
    # and the multi-size names (base@size) so a config built either way is
    # detected. We only need ONE hit to consider the config present.
    probe_vars = ["vector", "chunk", "chunk2", "chunk2a", "std_alloc"]
    probe_vars += [f"{v}@{s}" for v in ("chunk", "chunk2", "chunk2a", "std_alloc")
                   for s in ("256", "1024", "dyn")]
    for var in probe_vars:
        for stage in ("perf", "check"):
            if resolve_exe(build_root, config, var, stage):
                return True
    return False


def short_config_label(config):
    """Compact label for display / CSV."""
    if config.startswith("cmake-build-omp-"):
        return config.removeprefix("cmake-build-omp-")
    if config.startswith("cmake-build-"):
        # non-OMP build dirs (cmake-build-gnu, cmake-build-llvm) would otherwise
        # collapse onto the OMP label ("gnu"/"llvm") and look like a duplicate,
        # single-threaded run in the perf output. Tag them so they stay distinct
        # in the progress bar and the CSV.
        return config.removeprefix("cmake-build-") + "-noomp"
    return config


def is_compression_only_config(config):
    """True if this config only differs from its base in chunk* compression
    behaviour (e.g. the aggressive-compression variant)."""
    return any(m in config for m in COMPRESSION_ONLY_CONFIG_MARKERS)


def is_omp_config(config):
    """True if this build directory was configured WITH OpenMP.

    The only marker is the directory name: cmake-build-omp-* is an OpenMP build,
    everything else (cmake-build-gnu, cmake-build-llvm-std-alloc, ...) is serial
    and ignores OMP_NUM_THREADS entirely."""
    return config.startswith("cmake-build-omp-")


def is_std_alloc_config(config):
    """True if this is a std-allocator build (its dir name ends in -std-alloc).
    The std_alloc *variant* binary is only built there (CHUNK_STD_ALLOC=ON)."""
    return "std-alloc" in config


def skip_variant_on_config(var, config):
    """Whether to skip (var, config) entirely.

    - Compression-insensitive variants (vector) on a compression-only config
      (-have-agressive-compression): identical binary to the base config, so
      re-timing it is wasted work.
    - The std_alloc variant anywhere except a -std-alloc config: it is only
      built there now (gated on CHUNK_STD_ALLOC in CMake), so looking for it
      elsewhere would just produce spurious MISSING warnings.
    """
    base = _variant_base(var)
    if base in COMPRESSION_INSENSITIVE_VARIANTS and is_compression_only_config(config):
        return True
    if base == "std_alloc" and not is_std_alloc_config(config):
        return True
    return False


def get_progress_bar(current, total, width=20):
    if total <= 0:
        return f"|{'-' * width}| n/a"
    percent = float(current) / total
    done = int(width * percent)
    bar = green("█" * done) + dim("-" * (width - done))
    return f"|{bar}| {percent:5.1%}"


import re as _re

# Matches the binary's line: "Memory consumption <msg> : <value> Mo"
_MEM_RE = _re.compile(r"Memory consumption[^:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*Mo")


def parse_peak_memory_mo(stdout_text):
    """Return the peak 'Memory consumption ... : X Mo' value from stdout, or None.
    The binary prints this several times (after init, per iteration, at end);
    ru_maxrss is monotonic, so we keep the maximum."""
    if not stdout_text:
        return None
    vals = [float(m) for m in _MEM_RE.findall(stdout_text)]
    return max(vals) if vals else None


# Final-checksum line, e.g.:
#   "k : 1641 val_sum = nan long_val_sum = 115... cv_val_sum = (-5.7e+09,-1.3e+13)"
# \b before val_sum avoids matching inside long_val_sum / cv_val_sum (the char
# before is '_', a word char -> no boundary); cv_val_sum has its own regex.
_VALSUM_RE = _re.compile(r"\bval_sum\s*=\s*([^\s]+)")
_CVSUM_RE  = _re.compile(r"\bcv_val_sum\s*=\s*\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)")


def _is_nonfinite(tok):
    """True if the numeric token is nan or +/-inf."""
    t = tok.strip().lower()
    if "nan" in t or "inf" in t:
        return True
    try:
        v = float(t)
    except ValueError:
        return False  # unrecognised token -> do not flag
    return v != v or v == float("inf") or v == float("-inf")


def detect_nonfinite_checksum(stdout_text):
    """Return a short reason if the LAST final checksum in the output is non-finite
    (nan/inf) -> corrupted result; else None. Only the final value matters (the
    binary prints one per iteration). No val_sum in output -> None, so this never
    fires on runs that do not print a checksum."""
    if not stdout_text:
        return None
    bad = []
    vs = _VALSUM_RE.findall(stdout_text)
    if vs and _is_nonfinite(vs[-1]):
        bad.append(f"val_sum={vs[-1]}")
    cv = _CVSUM_RE.findall(stdout_text)
    if cv:
        re_, im_ = cv[-1]
        if _is_nonfinite(re_) or _is_nonfinite(im_):
            bad.append(f"cv_val_sum=({re_},{im_})")
    return ", ".join(bad) if bad else None


# --- Compression-health guard (CHUNK_STATS) --------------------------------
# `still compressed (at exit)` (checked by the seq-vs-OMP phase) cannot see a
# collapse of IN-FLIGHT compression: the final sweep compresses everything
# anyway, so that counter stays identical while memory blows up. The counters
# that DO move when in-flight marking breaks are `mark` (writes that newly
# flagged a chunk) against `create` (chunks materialised): a healthy chunk2 run
# sits around mark/create ~0.65; the Sept-2026 chunk2a regression (isv
# pre-check rejecting the chunk2a-only predicate clause) dropped it to ~0.07
# while `still compressed (at exit)` did not budge. We flag a run whose ratio
# falls under a per-variant floor. Only variants listed in the floor map are
# checked; a run without a CHUNK_STATS block (non-stats build) is never flagged.
#
# Floors are deliberately loose (roughly half the healthy value) so they catch
# a collapse, not normal noise. Tune with --stats-mark-ratio-min.
_STATS_MARK_RE   = _re.compile(r"^\s*mark\s*=\s*(\d+)", _re.MULTILINE)
_STATS_CREATE_RE = _re.compile(r"^\s*create\s*=\s*(\d+)", _re.MULTILINE)
_STATS_CEXEC_RE  = _re.compile(r"^\s*compress \(during exec\)\s*=\s*(\d+)", _re.MULTILINE)

DEFAULT_STATS_MARK_RATIO_MIN = {"chunk2": 0.3, "chunk2a": 0.3}
# Runtime copy, overridden from --stats-mark-ratio-min / --no-stats-guard.
_STATS_MARK_RATIO_MIN = dict(DEFAULT_STATS_MARK_RATIO_MIN)
# Below this many created chunks the ratio is too noisy to judge (tiny runs).
_STATS_MIN_CREATE_FOR_GUARD = 10_000


def parse_chunk_stats_counters(stdout_text):
    """Return {stat_tag: {"mark": int, "create": int, "compress_exec": int|None}}
    for every CHUNK_STATS block in the output (tag = 'chunk', 'chunk2',
    'chunk2a'...). Missing counters are omitted from the inner dict."""
    out = {}
    if not stdout_text:
        return out
    for m in _CHUNK_STATS_BLOCK_RE.finditer(stdout_text):
        tag, body = m.group(1), m.group(2)
        d = {}
        mk = _STATS_MARK_RE.search(body)
        cr = _STATS_CREATE_RE.search(body)
        ce = _STATS_CEXEC_RE.search(body)
        if mk:
            d["mark"] = int(mk.group(1))
        if cr:
            d["create"] = int(cr.group(1))
        if ce:
            d["compress_exec"] = int(ce.group(1))
        out[tag] = d
    return out


def detect_compression_collapse(stdout_text):
    """Return a short reason if any checked CHUNK_STATS block shows in-flight
    marking collapsed (mark/create under its floor); else None. Never fires
    when the stats block, the counters, or the floor for that tag are absent."""
    reasons = []
    for tag, d in parse_chunk_stats_counters(stdout_text).items():
        floor = _STATS_MARK_RATIO_MIN.get(tag)
        if floor is None or "mark" not in d or "create" not in d:
            continue
        create = d["create"]
        if create < _STATS_MIN_CREATE_FOR_GUARD:
            continue
        ratio = d["mark"] / create
        if ratio < floor:
            extra = ""
            if d.get("compress_exec") is not None:
                extra = f", compress(during exec)={d['compress_exec']}"
            reasons.append(
                f"[{tag}] mark/create={ratio:.3f} < {floor} "
                f"(mark={d['mark']}, create={create}{extra})"
            )
    return "; ".join(reasons) if reasons else None


def _write_full_log(cmd, child_env, returncode, stdout_text, stderr_text):
    """When --full-log-dir is set, dump one child run's COMPLETE stdout+stderr to
    its own file. The name encodes binary, OMP thread count, a global run counter
    and the status, so each execution is identifiable (chunk2 vs chunk2a, which run
    re-chunked / crashed). No-op when off. Never raises: logging must not break the
    bench."""
    global _FULL_LOG_COUNTER
    if not _FULL_LOG_DIR:
        return
    try:
        binary = os.path.basename(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else "run"
        omp = str(child_env.get("OMP_NUM_THREADS", "?"))
        with _FULL_LOG_LOCK:
            _FULL_LOG_COUNTER += 1
            idx = _FULL_LOG_COUNTER
        status = "ok" if returncode == 0 else f"rc{returncode}"
        path = os.path.join(_FULL_LOG_DIR, f"{binary}__omp{omp}__{idx:05d}__{status}.log")
        with open(path, "w") as fh:
            fh.write(f"# cmd: {' '.join(map(str, cmd))}\n")
            fh.write(f"# returncode: {returncode}\n")
            fh.write("# ===== STDOUT =====\n")
            fh.write(stdout_text or "")
            if not (stdout_text or "").endswith("\n"):
                fh.write("\n")
            fh.write("# ===== STDERR =====\n")
            fh.write(stderr_text or "")
    except Exception:
        pass  # logging must never break the run


def _run_cmd_impl(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, must_contain=None):
    """Run cmd. Returns (success, duration_s, returncode, stderr_tail, mem_mo).

    extra_ld_library_path: path(s) to PREPEND to LD_LIBRARY_PATH for this run.
    Needed for llvm binaries linked with -stdlib=libc++ that look for
    libc++.so.1 in the llvm module's lib/ at runtime (or the gcc lib64 for
    libstdc++). Not needed once the binaries carry a proper RPATH.

    cwd: working directory for the child. The binaries read/write data_tab_*
    and memory_mo.txt relative to their cwd, so pointing this at the shared
    output directory keeps every run's data together.

    mem_mo: peak memory in MB parsed from the binary's stdout, or None.
    """
    start = time.perf_counter()
    # NUMA binding: pin OpenMP threads to cores (local-socket memory).
    # Unbound threads on a multi-socket node migrate and hit remote RAM,
    # which dominates and hides real scaling. Caller/outer env can override.
    _bind = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    child_env = {**_bind, **os.environ, **env_vars}
    if extra_ld_library_path:
        prev = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            extra_ld_library_path + (":" + prev if prev else "")
        )
    cmd = _numa.interleave_prefix(child_env, _NUMA_MODE) + list(cmd)  # --numa-interleave
    try:
        res = subprocess.run(
            cmd, env=child_env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout,
        )
        err_tail = "\n".join(res.stderr.splitlines()[-30:]) if res.stderr else ""
        mem_mo = parse_peak_memory_mo(res.stdout)
        _write_full_log(cmd, child_env, res.returncode, res.stdout, res.stderr)
        # Sanitizer hit detection: ASan/TSan write their report to stderr (and
        # sometimes stdout). A report means FAIL even if the process exited 0
        # (ASan can continue past an error depending on options). We surface it
        # by making success False and prefixing err_tail, so existing callers
        # treat it exactly like any other failure with no signature change.
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        san_hit = any(m in combined for m in SANITIZER_HIT_MARKERS)
        success = (res.returncode == 0) and not san_hit
        if san_hit:
            # Keep the sanitizer report lines (the most relevant tail), not just
            # the last 30 stderr lines which may be past the SUMMARY.
            report_lines = [ln for ln in combined.splitlines()
                            if any(m in ln for m in SANITIZER_HIT_MARKERS)]
            err_tail = "SANITIZER HIT:\n" + "\n".join(report_lines[-30:]) + \
                       ("\n--- stderr tail ---\n" + err_tail if err_tail else "")
        # Optional required-output assertion: a run that exits cleanly but did NOT
        # print `must_contain` is a FAIL (used by the adaptive re-chunk feature
        # test to require an [APPLIED] line). Non-invasive: default None.
        if success and must_contain is not None and must_contain not in combined:
            success = False
            err_tail = (f"MISSING EXPECTED OUTPUT: {must_contain!r} not found in run output\n"
                        "--- output tail ---\n"
                        + "\n".join(combined.splitlines()[-30:]))
        # Final-checksum sanity: a run can exit 0 yet produce a corrupted result
        # (val_sum = nan / inf from a bad re-chunk, a race, an overflow). The perf
        # runs otherwise never check correctness, so this passed silently. Flag a
        # non-finite final checksum as a FAIL, like any other failure.
        if success:
            ck_bad = detect_nonfinite_checksum(res.stdout)
            if ck_bad is not None:
                success = False
                err_tail = ("CORRUPTED CHECKSUM: " + ck_bad + "\n"
                            "--- output tail ---\n"
                            + "\n".join(combined.splitlines()[-30:]))
        # Compression-health guard: a run can exit 0 with a perfect checksum and
        # still have stopped compressing in flight (memory then grows several-
        # fold). `still compressed (at exit)` cannot see it; mark/create can.
        # Same FAIL plumbing as the checks above, no signature change.
        if success:
            cc_bad = detect_compression_collapse(res.stdout)
            if cc_bad is not None:
                success = False
                err_tail = ("COMPRESSION COLLAPSE: " + cc_bad + "\n"
                            "--- output tail ---\n"
                            + "\n".join(combined.splitlines()[-30:]))
        return success, time.perf_counter() - start, res.returncode, err_tail, mem_mo
    except subprocess.TimeoutExpired:
        return False, float(timeout), -9, f"TIMEOUT after {timeout}s", None
    except FileNotFoundError as e:
        # The binary vanished between resolution and launch — almost always a
        # concurrent rebuild replacing the file. Signalled with -2 so the caller
        # can wait and re-resolve instead of spamming the same error.
        return False, 0.0, -2, f"FileNotFoundError: {e}", None
    except OSError as e:
        # ETXTBSY (Errno 26, "Text file busy"): the binary is open for writing
        # by another process — a concurrent link/copy or `cmake --build` that
        # just regenerated it (the non-OMP configs are built last, so the bench
        # can reach them mid-write). Transient, exactly like a vanished binary,
        # so signal -2 to trigger the wait-and-retry path instead of a hard
        # FAIL. Other OSErrors keep the generic -1.
        if getattr(e, "errno", None) == 26:  # errno.ETXTBSY
            return False, 0.0, -2, f"ETXTBSY (text file busy): {e}", None
        return False, 0.0, -1, f"{type(e).__name__}: {e}", None
    except Exception as e:
        return False, 0.0, -1, f"{type(e).__name__}: {e}", None


# ── --rerun-failed: recording the failed runs ────────────────────────────────
# Every FAILED run of the two executors is remembered here until the phase that
# made it logs its failure (_append_error_now), which moves it to
# failed_cases.jsonl. A later SUCCESSFUL run of the same command (the
# rebuild-and-retry path) forgets it, so a failure that healed is not replayed.
_PENDING_FAILED_RUNS = []
_CTEST_FAILED_NAME_RE = __import__("re").compile(r"^\s*\d+\s+-\s+(\S+)\s+\(", __import__("re").M)


def _run_key(r):
    return (tuple(r["cmd"]), r.get("cwd"), tuple(sorted((r.get("env") or {}).items())))


def _note_run(executor, ok, cmd, env_vars, timeout, extra_ld_library_path, cwd, must_contain, stdout_text=None):
    rec = {"executor": executor, "cmd": [str(c) for c in cmd], "env": dict(env_vars or {}),
           "timeout": timeout, "ld": extra_ld_library_path, "cwd": cwd, "must_contain": must_contain}
    if ok:
        k = _run_key(rec)
        _PENDING_FAILED_RUNS[:] = [r for r in _PENDING_FAILED_RUNS if _run_key(r) != k]
        return
    # ctest: replay only the tests that failed, not the whole label.
    if os.path.basename(rec["cmd"][0]) == "ctest" and stdout_text:
        tail = stdout_text.split("The following tests FAILED", 1)
        names = _CTEST_FAILED_NAME_RE.findall(tail[1]) if len(tail) == 2 else []
        if names:
            import re as _re2
            rec["cmd"] = [rec["cmd"][0], "-R", "^(" + "|".join(_re2.escape(n) for n in names) + ")$",
                          "--output-on-failure"]
            rec["ctest_tests"] = names
    _PENDING_FAILED_RUNS.append(rec)


def run_cmd(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, must_contain=None):
    """_run_cmd_impl(), recording a failed run for --rerun-failed."""
    res = _run_cmd_impl(cmd, env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
                        cwd=cwd, must_contain=must_contain)
    _note_run("run_cmd", res[0], cmd, env_vars, timeout, extra_ld_library_path, cwd, must_contain)
    return res


def run_capture_stdout(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None):
    """_run_capture_stdout_impl(), recording a failed run for --rerun-failed."""
    res = _run_capture_stdout_impl(cmd, env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
                                   cwd=cwd)
    _note_run("capture", res[0], cmd, env_vars, timeout, extra_ld_library_path, cwd, None,
              stdout_text=res[3])
    return res


def _failed_cases_path(errors_log):
    return os.path.join(os.path.dirname(os.path.abspath(errors_log)), "failed_cases.jsonl")


def _append_error_now(errors_log, entry):
    """Append a single error entry to the log immediately, so failures are on
    disk the instant they happen — not only at the final flush_summary(). This
    is what lets you `tail -f` the log during a run and see the real stderr of a
    FAIL right away instead of waiting for the run to finish (or guessing when it
    is interrupted). Best-effort: never let logging raise into the run loop.

    Also appends the failed runs behind this entry to failed_cases.jsonl, the
    list --rerun-failed replays (none: the entry is recorded as not
    replayable -- a comparison across runs, a guard)."""
    try:
        with open(errors_log, "a") as f:
            f.write(entry.rstrip("\n") + "\n")
    except OSError:
        pass
    try:
        first = entry.strip().splitlines()[0] if entry.strip() else ""
        case = {"kind": first.split(":", 1)[0], "summary": first[:300],
                "runs": list(_PENDING_FAILED_RUNS)}
        _PENDING_FAILED_RUNS.clear()
        with open(_failed_cases_path(errors_log), "a") as f:
            f.write(json.dumps(case) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def run_rerun_failed(args):
    """--rerun-failed: replay the recorded failed runs; returns the exit code."""
    src = args.rerun_failed or _failed_cases_path(args.errors_log)
    if not os.path.isfile(src):
        print(red(f"--rerun-failed: {src} not found (no failed run recorded there)"))
        return 2
    cases = []
    with open(src) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    cases.append(json.loads(ln))
                except ValueError:
                    pass
    reps = max(1, args.rerun_repeat)
    print(cyan(bold(f"\n🔁 Re-running the failed cases of {src}  ({len(cases)} case(s), x{reps})")))
    still, n_ok, n_fail, n_skip = [], 0, 0, 0
    for case in cases:
        runs = case.get("runs") or []
        head = case.get("summary", "")[:150]
        if not runs:
            n_skip += 1
            print(dim(f"  ⏭  not replayable (no single failed run behind it): {head}"))
            continue
        case_fail = None
        for rep in range(reps):
            for r in runs:
                if r.get("executor") == "capture":
                    ok, _dur, code, _out, tail = _run_capture_stdout_impl(
                        r["cmd"], r.get("env") or {}, r.get("timeout"),
                        extra_ld_library_path=r.get("ld"), cwd=r.get("cwd"))
                    if not tail:  # ctest reports on stdout
                        tail = "\n".join((_out or "").splitlines()[-12:])
                else:
                    ok, _dur, code, tail, _mem = _run_cmd_impl(
                        r["cmd"], r.get("env") or {}, r.get("timeout"),
                        extra_ld_library_path=r.get("ld"), cwd=r.get("cwd"),
                        must_contain=r.get("must_contain"))
                if not ok:
                    case_fail = (rep + 1, code, tail)
                    break
            if case_fail:
                break
        if case_fail:
            n_fail += 1
            still.append(case)
            rep, code, tail = case_fail
            print(red(f"  ✗ still fails (repeat {rep}/{reps}, code {code}): {head}"))
            for tl in (tail or "").splitlines()[-8:]:
                print(dim("      " + tl))
        else:
            n_ok += 1
            print(green(f"  ✓ passes x{reps}: {head}"))
    # Keep the previous list, then keep only what still fails (the not
    # replayable entries are kept too: they still need a full run to clear).
    dst = _failed_cases_path(args.errors_log)
    try:
        if os.path.abspath(src) == os.path.abspath(dst):
            os.replace(dst, dst.replace(".jsonl", ".prev.jsonl"))
        with open(dst, "w") as f:
            for case in still + [c for c in cases if not (c.get("runs") or [])]:
                f.write(json.dumps(case) + "\n")
    except OSError:
        pass
    print(bold(f"\n🔁 {n_ok} pass, {n_fail} still fail, {n_skip} not replayable  -> {dst}"))
    return 1 if n_fail else 0


def _memory_guard_errors(norm, baseline_variant,
                         chunk2a_vs_chunk2_max, vs_baseline_max):
    """Turn the memory figures flush_summary already computes into verdicts.

    norm maps (cfg, var, tc) -> (avg_time_s, peak_mem_mo_or_None).

    Two guards, both on the same (cfg, tc) and same size declination so the
    runs processed identical data:

    1. chunk2a vs chunk2: by design Chunk2A must never cost MORE memory than
       Chunk2 (it is Chunk2 plus a second constant). A ratio above
       `chunk2a_vs_chunk2_max` is the exact signature of the Sept-2026
       regression (x2.7 at 64 threads), reported as an error.
    2. any chunk variant vs the baseline (default `vector`): a chunk build
       that uses more than `vs_baseline_max` x the plain-vector memory has lost
       its reason to exist. Off when `vs_baseline_max` is None.

    Returns a list of error strings (possibly empty)."""
    errs = []
    if not norm:
        return errs

    def _size_of(var):
        return var.split("@", 1)[1] if "@" in var else ""

    # (cfg, tc, size) -> mem, per bare variant
    by_key = {}
    for (cfg, var, tc), (_avg, mem) in norm.items():
        if mem is None or mem <= 0:
            continue
        by_key[(cfg, tc, _size_of(var), _variant_base(var))] = mem

    if chunk2a_vs_chunk2_max is not None:
        for (cfg, tc, sz, base), mem in by_key.items():
            if base != "chunk2a":
                continue
            ref = by_key.get((cfg, tc, sz, "chunk2"))
            if ref is None:
                continue
            ratio = mem / ref
            if ratio > chunk2a_vs_chunk2_max:
                errs.append(
                    f"MEMORY REGRESSION [chunk2a vs chunk2] "
                    f"cfg={short_config_label(cfg)} threads={tc} size={sz or '-'}: "
                    f"chunk2a={mem:.0f} Mo, chunk2={ref:.0f} Mo, "
                    f"ratio={ratio:.2f} > {chunk2a_vs_chunk2_max}"
                )

    if vs_baseline_max is not None:
        for (cfg, tc, sz, base), mem in by_key.items():
            if base == baseline_variant or not base.startswith("chunk"):
                continue
            # The baseline (vector) has no size declination: match on (cfg, tc)
            # with any size.
            refs = [m for (c, t, _s, b), m in by_key.items()
                    if c == cfg and t == tc and b == baseline_variant]
            if not refs:
                continue
            ref = refs[0]
            ratio = mem / ref
            if ratio > vs_baseline_max:
                errs.append(
                    f"MEMORY REGRESSION [{base} vs {baseline_variant}] "
                    f"cfg={short_config_label(cfg)} threads={tc} size={sz or '-'}: "
                    f"{base}={mem:.0f} Mo, {baseline_variant}={ref:.0f} Mo, "
                    f"ratio={ratio:.2f} > {vs_baseline_max}"
                )
    return errs


def flush_summary(results, errors, summary_csv, errors_log, elapsed_min,
                  interrupted, baseline_variant=DEFAULT_BASELINE,
                  mem_chunk2a_vs_chunk2_max=None, mem_vs_baseline_max=None):
    """Write CSV + log — called even when interrupted.

    results maps (cfg, var, tc) -> (avg_time_s, peak_mem_mo_or_None).
    The CSV also carries, for non-baseline variants, the time speedup and the
    memory ratio vs the baseline variant (same cfg + thread count).

    Memory guards (see _memory_guard_errors) are evaluated here, once every
    run is in, and appended to `errors` so they count as real failures.
    """
    label = "interrupted" if interrupted else "finished"
    done_dot = blue("🔵") if not interrupted else yellow("🟡")
    print(bold(f"\n{done_dot} Test {label} in {elapsed_min:.2f} minutes."))

    # Normalise every entry to (avg, mem) and index baselines by (cfg, tc).
    norm = {}
    base_time = {}   # (cfg, tc) -> avg
    base_mem  = {}   # (cfg, tc) -> mem
    for (cfg, var, tc), val in results.items():
        avg, mem = val if isinstance(val, tuple) else (val, None)
        norm[(cfg, var, tc)] = (avg, mem)
        if var == baseline_variant:
            base_time[(cfg, tc)] = avg
            if mem is not None:
                base_mem[(cfg, tc)] = mem

    # Memory verdicts: computed from the same figures the CSV reports, but a
    # bad ratio is a FAIL, not a column in a spreadsheet.
    mem_errs = _memory_guard_errors(norm, baseline_variant,
                                    mem_chunk2a_vs_chunk2_max, mem_vs_baseline_max)
    if mem_errs:
        for e in mem_errs:
            print(red("⚠️  " + e))
        errors.extend(mem_errs)

    if errors:
        print(red(f"⚠️  {len(errors)} error(s) detected — see {errors_log}"))
        with open(errors_log, "w") as f:
            f.write("\n\n".join(errors))
    else:
        print(green("✅ No crash or error detected."))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Variant", "Threads", "AvgTime_s", "PeakMem_Mo",
                    "Speedup_vs_baseline", "MemRatio_vs_baseline",
                    "MemSaved_Mo_vs_baseline"])
        for (cfg, var, tc), (avg, mem) in sorted(norm.items()):
            mem_str = f"{mem:.1f}" if mem is not None else ""
            # Time speedup (>1 = faster than baseline).
            speedup_str = ""
            if var != baseline_variant and (cfg, tc) in base_time and avg > 0:
                speedup_str = f"{base_time[(cfg, tc)] / avg:.3f}"
            # Memory ratio (>1 = uses less memory than baseline) + absolute saving.
            mem_ratio_str = ""
            mem_saved_str = ""
            if var != baseline_variant and mem is not None \
                    and (cfg, tc) in base_mem and mem > 0:
                bm = base_mem[(cfg, tc)]
                mem_ratio_str = f"{bm / mem:.3f}"
                mem_saved_str = f"{bm - mem:.1f}"
            w.writerow([short_config_label(cfg), var, tc, f"{avg:.6f}", mem_str,
                        speedup_str, mem_ratio_str, mem_saved_str])
    print(f"📄 Summary written to {summary_csv}")


def effective_timeout(base_timeout, per_thread, tc):
    """Timeout for a run at `tc` threads: base + per_thread * tc."""
    return int(base_timeout + per_thread * tc)


def run_with_rebuild_retry(build_root, config, var, stage, exe, env_vars,
                           timeout, extra_ld_library_path,
                           rebuild_retries, rebuild_wait, cwd=None):
    """Run `exe`; if it vanished (concurrent rebuild → code -2), wait, re-resolve
    and retry up to rebuild_retries times. Returns
    (ok, dur, code, err_tail, mem_mo, exe_used)."""
    for attempt in range(rebuild_retries + 1):
        # --mode carries what the binary name used to: one executable, the stage
        # chosen at run time. Harmless on the vector targets, which still come in
        # two flavours and simply get told the mode they were built for.
        ok, dur, code, err_tail, mem_mo = run_cmd(
            [exe, "--mode", stage], env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
            cwd=cwd
        )
        if code != -2:
            return ok, dur, code, err_tail, mem_mo, exe
        # Binary disappeared — likely a rebuild in progress.
        if attempt < rebuild_retries:
            sys.stdout.write(
                f"\n   ⏳ binary busy/missing (rebuild in progress?), waiting "
                f"{rebuild_wait:.0f}s and re-resolving "
                f"[{attempt + 1}/{rebuild_retries}]...\n"
            )
            sys.stdout.flush()
            time.sleep(rebuild_wait)
            new_exe = resolve_exe(build_root, config, var, stage)
            if new_exe:
                exe = new_exe
    return ok, dur, code, err_tail, mem_mo, exe


# --- Seq-vs-OMP compression-classification compare (seed-pinned) ---------
# Compares `still compressed (at exit)` between a SEQUENTIAL (non-OMP) LLVM
# build and its OpenMP LLVM counterpart, at several thread counts, with the
# bench's RNG seed PINNED so every single run — sequential and every OMP
# thread count — operates on byte-identical generated data (the bench draws
# tab_size/tab_start/tab_nba1/tab_nba2 from an std::mt19937 whose seed is
# read from / saved to seed<iter_max>.txt, see initializeRNG in
# bench_utils.cpp). Without pinning the seed, an observed difference in
# `still compressed (at exit)` cannot be told apart from the runs simply
# having processed different random data.
#
# Also cross-checks long_val_sum — an exact INTEGER checksum the bench
# prints each iteration (associative, so immune to parallel-reduction
# reordering, unlike the floating-point val_sum) — between the sequential
# run and every OMP thread count. A disagreement there, seed pinned, means
# the array's own final CONTENT differs by thread count: a correctness
# question upstream of anything the chunk library's compression bookkeeping
# can explain, reported as its own failure rather than folded into a
# still@exit mismatch.

_CHUNK_STATS_BLOCK_RE = _re.compile(
    r"----- CHUNK_STATS \[(\w+)\] -----\n(.*?)\n-{5,}", _re.DOTALL
)
_STILL_AT_EXIT_RE = _re.compile(r"still compressed \(at exit\)\s*=\s*(\d+)")
_CHECKSUM_RE = _re.compile(
    r"k\s*:\s*(\d+)\s+val_sum\s*=\s*(\S+)\s+long_val_sum\s*=\s*(\S+)\s+cv_val_sum\s*=\s*(\([^)]*\))"
)
_SEED_FALLBACK_RE = _re.compile(r"Save seed in file")


def _parse_still_at_exit(stdout_text, variant_base):
    """`still compressed (at exit)` from `variant_base`'s CHUNK_STATS block in
    this run's stdout, or None if that block never appeared (e.g. the binary
    reports a different stat_variant tag than expected, or zero activity)."""
    for m in _CHUNK_STATS_BLOCK_RE.finditer(stdout_text):
        if m.group(1) == variant_base:
            sm = _STILL_AT_EXIT_RE.search(m.group(2))
            return int(sm.group(1)) if sm else None
    return None


def _parse_last_long_val_sum(stdout_text):
    """The LAST long_val_sum printed (the final iteration's exact integer
    checksum), or None if the bench never printed one (e.g. a build without
    the checksum line, or a run that crashed before finishing)."""
    matches = list(_CHECKSUM_RE.finditer(stdout_text))
    return matches[-1].group(3) if matches else None


def _seed_was_pinned(stdout_text):
    """False if the run fell back to a random seed (our pinned seed failed to
    load from seed<iter_max>.txt) -- see initializeRNG in bench_utils.cpp."""
    return _SEED_FALLBACK_RE.search(stdout_text) is None


def _write_pinned_seed(output_dir, iter_max, seed):
    """(Re)write the fixed seed into seed<iter_max>.txt in output_dir -- the
    SAME cwd every run in this phase uses, so all of them read the identical
    value regardless of which binary/config is launched. Rewritten before
    EVERY run: initializeRNG only regenerates when the file is absent or
    holds 0, so a pinned non-zero value is never overwritten by the bench
    itself, but writing it fresh each time removes any doubt (and self-heals
    if a previous run's crash left the file missing or truncated)."""
    seed_path = os.path.join(output_dir, f"seed{iter_max}.txt")
    with open(seed_path, "w") as f:
        f.write(str(seed))
    return seed_path


def oracle_dir(output_dir, config):
    """The directory holding `config`'s reference files (data_tab*), under
    output_dir/oracle/<config>: the check binaries of that config run there.

    One oracle PER CONFIG, generated by that config's own vector binary. The
    reference is not portable across compilers: the bench's branches that
    depend on a floating-point value (tab_int[i] = tab_bool[i] ? (tab_float[i]
    > 0 ? 1 : 2) : 0, ...) turn a last-bit difference into 0/1 differences on
    the integer and bool arrays, which the read-back's tolerance cannot
    absorb -- a clang chunk binary against a gcc oracle failed at iteration 1
    with `tab1[29581] 0 != 1` on x86 at -O0. The single oracle this script
    shared across every config (output_dir/data_tab*) is no longer read."""
    return os.path.join(output_dir, "oracle", config)


def _run_dir(output_dir, *parts):
    """A working directory of this run's own, under output_dir.

    The binaries read and write data_<tab>_<k>.bin relative to their cwd, and
    that name carries no variant, no geometry and no compiler -- so two
    binaries sharing a directory silently share reference files. A static
    chunk_256 writing data_tab_bool_1.bin and a chunk2a_dyn reading it back is
    a mismatch at element 450159 on iteration 1, which looks like a library bug
    and is not one. One directory per run, and the question cannot arise.

    Cheap: mkdir plus, where the caller asks for it, a copy of the pinned seed.
    """
    safe = [str(p).replace("/", "_").replace("@", "-").replace(" ", "_") for p in parts if p]
    d = os.path.join(output_dir, "runs", *safe) if safe else output_dir
    os.makedirs(d, exist_ok=True)
    return d


def _run_capture_stdout_impl(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None):
    """Sibling of run_cmd() that returns the FULL stdout instead of parsing
    peak memory / sanitizer hits / must_contain. This phase needs the raw
    CHUNK_STATS text and the checksum line, neither of which run_cmd()
    surfaces to its caller. Kept as a separate function rather than changing
    run_cmd()'s established return shape, which every other phase depends on.

    Returns (success, duration_s, returncode, stdout_text, stderr_tail).
    """
    start = time.perf_counter()
    _bind = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    child_env = {**_bind, **os.environ, **env_vars}
    if extra_ld_library_path:
        prev = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            extra_ld_library_path + (":" + prev if prev else "")
        )
    cmd = _numa.interleave_prefix(child_env, _NUMA_MODE) + list(cmd)  # --numa-interleave
    try:
        res = subprocess.run(
            cmd, env=child_env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout,
        )
        err_tail = "\n".join(res.stderr.splitlines()[-30:]) if res.stderr else ""
        _write_full_log(cmd, child_env, res.returncode, res.stdout, res.stderr)
        return (res.returncode == 0, time.perf_counter() - start, res.returncode,
                res.stdout or "", err_tail)
    except subprocess.TimeoutExpired:
        return False, float(timeout), -9, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return False, 0.0, -2, "", f"FileNotFoundError: {e}"
    except OSError as e:
        if getattr(e, "errno", None) == 26:  # ETXTBSY
            return False, 0.0, -2, "", f"ETXTBSY (text file busy): {e}"
        return False, 0.0, -1, "", f"{type(e).__name__}: {e}"
    except Exception as e:
        return False, 0.0, -1, "", f"{type(e).__name__}: {e}"


def find_llvm_seq_omp_pairs(build_root, configs):
    """Among `configs`, find every OpenMP config that has a matching non-OMP
    (sequential) counterpart actually present on disk, e.g.
      "cmake-build-omp-llvm"           <-> "cmake-build-llvm"
      "cmake-build-omp-gnu-stats"      <-> "cmake-build-gnu-stats"
      "cmake-build-omp-llvm-std-alloc" <-> "cmake-build-llvm-std-alloc"
    Returns a list of (omp_config, seq_config) pairs. An OMP config without a
    present sequential counterpart is skipped with a single note, not an
    error -- most build trees only have the plain pair, not a sequential
    build of every OMP variant (-std-alloc, -have-agressive-compression).
    (Used to be LLVM-only, from when only the LLVM pair existed; the compare
    reads `still compressed (at exit)`, which any CHUNK_STATS pair prints --
    the name is kept for its callers.)"""
    pairs, skipped = [], []
    for cfg in configs:
        if not cfg.startswith("cmake-build-omp-"):
            continue
        seq_cfg = cfg.replace("cmake-build-omp-", "cmake-build-", 1)
        if config_dir_present(build_root, seq_cfg):
            pairs.append((cfg, seq_cfg))
        else:
            skipped.append(cfg)
    if skipped:
        print(dim("   seq/omp compare: no sequential counterpart built for "
                  + ", ".join(short_config_label(c) for c in skipped)
                  + " -- skipped"))
    return pairs



# --- Randomised parameter sweep (fuzz) ------------------------------------
# The pools a case is drawn from. Constants, not flags: the point of the phase
# is to reach parameter combinations nobody thought to ask for, and a flag that
# narrows the pool works against that. Widen a pool by editing it here.
FUZZ_SIZES     = ["256", "1024", "dyn"]
FUZZ_DIVISORS  = [5, 10, 20, 50, 100, 200]
FUZZ_ITER_MAX  = [200, 500, 1000, 2000]
FUZZ_THREADS   = [2, 4, 8, 16, 64]


def run_fuzz_phase(args, extra_ldpath, errors):
    """Run each chunk-family variant on RANDOM workload parameters instead of
    the one trajectory every other phase repeats.

    Rationale: the whole suite exercises a single geometry/iteration profile,
    over and over. A defect that only shows up at some other divisor, iteration
    count or thread count is invisible to it no matter how many repetitions are
    run -- which is exactly how a re-chunk teardown crash can sit unnoticed
    while 690 check runs come back green.

    The oracle is the one the bench already trusts: the run must exit 0, must
    not print a sanitizer report, and its integer checksum (long_val_sum) must
    match the SEQUENTIAL run of the same parameters -- with the seed pinned, so
    a difference means behaviour, not scheduling.

    NOT the data_tab_* oracle, hence --no_io. In --mode check a chunk build
    reads the dumps the vector baseline wrote and compares element by element;
    that oracle was produced with the suite's default divisor / iter_max, and
    a case drawn here has other ones, so the arrays cannot match -- every case
    aborted in read_check_binary_file with the same stale checksum on its
    "read" side. Regenerating the dumps per case would cost a baseline run
    each and overwrite the oracle the other phases share; --no_io keeps the
    check-mode workload and skips only the file comparison.

    --fuzz-cases is the only knob. The pools are the FUZZ_* constants above,
    and the master seed comes from the clock and is echoed, so the phase has
    nothing to configure and nothing to get wrong.

    Replay does not need a flag: every case prints its variant, parameters and
    seed, and a failure is logged as the literal command line that produced it.
    """
    import random as _random
    import time as _time

    if args.fuzz_cases <= 0:
        return 0, 0

    # Drawn from the clock, deliberately not settable: a sweep you can pin is a
    # sweep that stops finding new combinations. Replay is per case, below.
    rng = _random.Random(int(_time.time()))
    print(cyan(bold(f"\n🎲 Randomised parameter sweep — {args.fuzz_cases} case(s)")))
    print(dim("   replay a case with the parameters and seed printed on its line"))

    bases = [b for b in ("chunk", "chunk2", "chunk2a")
             if any(_variant_base(v) == b for v in args.variants)]
    if not bases:
        print(dim("   (no chunk-family variant selected -- skipped)"))
        return 0, 0

    n_pass = n_total = 0
    for case in range(1, args.fuzz_cases + 1):
        base    = rng.choice(bases)
        size    = rng.choice(FUZZ_SIZES)
        divisor = rng.choice(FUZZ_DIVISORS)
        itmax   = rng.choice(FUZZ_ITER_MAX)
        threads = rng.choice(FUZZ_THREADS)
        seed    = rng.randrange(1, 2 ** 31 - 1)
        var     = f"{base}@{size}"

        cfg = next((c for c in args.configs
                    if not skip_variant_on_config(var, c)
                    and resolve_exe(args.build_root, c, var, "check")), None)
        if cfg is None:
            continue

        exe = resolve_exe(args.build_root, cfg, var, "check")
        to  = effective_timeout(args.timeout, args.timeout_per_thread, threads)
        bench_args = ["--mode", "check", "--no_io",
                      "--divisor", str(divisor), "--iter_max", str(itmax)]

        lead = (f"{dim(_hms())} [{case:3}/{args.fuzz_cases}] "
                f"{short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                f"{var[:VARIANT_W]:<{VARIANT_W}} | div={divisor:<4} it={itmax:<5} "
                f"{threads:3}th seed={seed:<11} | ")
        n_total += 1

        # Sequential reference first, then the threaded run, same pinned seed.
        _write_pinned_seed(args.output_dir, itmax, seed)
        sys.stdout.write("\r" + lead + "seq…      ")
        sys.stdout.flush()
        ok_seq, _, code_seq, tail_seq, _ = run_cmd(
            [exe] + bench_args, {"OMP_NUM_THREADS": "1"}, to,
            extra_ld_library_path=extra_ldpath, cwd=args.output_dir)

        ref = _parse_last_long_val_sum(tail_seq) if ok_seq else None

        _write_pinned_seed(args.output_dir, itmax, seed)
        sys.stdout.write("\r" + lead + f"{threads}th…     ")
        sys.stdout.flush()
        ok_omp, dur, code_omp, tail_omp, _ = run_cmd(
            [exe] + bench_args, {"OMP_NUM_THREADS": str(threads)}, to,
            extra_ld_library_path=extra_ldpath, cwd=args.output_dir)

        got = _parse_last_long_val_sum(tail_omp) if ok_omp else None

        if not ok_seq or not ok_omp:
            which = "sequential" if not ok_seq else f"{threads}-thread"
            code  = code_seq if not ok_seq else code_omp
            tail  = tail_seq if not ok_seq else tail_omp
            sys.stdout.write("\r" + lead + red(f"FAIL ({which}, code {code})") + "\n")
            _entry = (f"FUZZ_FAIL: {exe} {' '.join(bench_args)} "
                      f"OMP_NUM_THREADS={threads} seed={seed} ({which}, code={code})\n"
                      f"  stderr_tail:\n    " + tail.replace('\n', '\n    '))
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        elif ref is not None and got is not None and ref != got:
            sys.stdout.write("\r" + lead + red("CHECKSUM MISMATCH") + "\n")
            _entry = (f"FUZZ_CHECKSUM: {exe} {' '.join(bench_args)} seed={seed}\n"
                      f"  sequential long_val_sum = {ref}\n"
                      f"  {threads}-thread  long_val_sum = {got}")
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        else:
            n_pass += 1
            sys.stdout.write("\r" + lead + green("ok") + f" ({dur:.1f}s)\n")
        sys.stdout.flush()

    print(f"   {n_pass}/{n_total} case(s) passed")
    return n_pass, n_total


def run_seq_omp_compare_phase(args, extra_ldpath, errors):
    """For each LLVM seq/OMP config pair, each requested chunk-family variant
    and size, and each requested OMP thread count: run the SEQUENTIAL binary
    once and the OMP binary once per thread count, all with the SAME PINNED
    SEED, and compare `still compressed (at exit)`. With the seed pinned,
    this number is a function of the array's final CONTENT alone (verified
    via long_val_sum) -- it must be identical across every mode and thread
    count. A mismatch is a real, actionable signal, not scheduling noise.
    """
    print(cyan(bold("\n🌱 Seq vs OMP compression-classification compare "
                    f"(seed={args.seq_omp_seed}, threads={args.seq_omp_threads})")))

    if args.seq_omp_seed == 0:
        print(red("   --seq-omp-seed=0 is invalid: the bench treats a 0 seed "
                  "as \"generate a random one\", which defeats this phase's "
                  "whole point. Skipped."))
        return 0, 0

    pairs = find_llvm_seq_omp_pairs(args.build_root, args.configs)
    if not pairs:
        print(dim("   (no seq/omp config pair present -- skipped)"))
        return 0, 0

    # The compare reads `still compressed (at exit)` out of the CHUNK_STATS
    # block, which ONLY a -DCHUNK_STATS=ON build prints. Whether a pair is a
    # stats build is knowable from its name (is_stats_config), so drop the
    # others here -- the loop below used to find out the expensive way, by
    # running a full sequential bench per pair whose only possible outcome was
    # "no CHUNK_STATS block in the sequential output -- not a stats build".
    # That cost one pointless multi-minute run per non-stats pair, every time.
    non_stats = [(o, q) for o, q in pairs
                 if not (is_stats_config(o) and is_stats_config(q))]
    pairs = [(o, q) for o, q in pairs
             if is_stats_config(o) and is_stats_config(q)]
    if non_stats:
        print(dim("   seq/omp compare: not a CHUNK_STATS build, nothing to compare for "
                  + ", ".join(short_config_label(o) for o, _ in non_stats)
                  + " -- skipped"))
    if not pairs:
        print("   Seq vs OMP compare: "
              + yellow(f"nothing compared: {len(non_stats)} pair(s) without CHUNK_STATS "
                       "-- build the -stats pair (-DCHUNK_STATS=ON, OMP on and off)"))
        return 0, 0

    bench_args = ["--mode", "perf",
                  "--divisor", str(args.seq_omp_divisor),
                  "--iter_max", str(args.seq_omp_iter_max)]
    to = effective_timeout(args.timeout, args.timeout_per_thread,
                           max(args.seq_omp_threads))

    n_pass = n_total = 0
    n_pairs_skipped = 0
    for omp_cfg, seq_cfg in pairs:
        # The compare reads CHUNK_STATS. A pair whose binaries are not stats
        # builds has nothing to compare: found out on its FIRST sequential run
        # and skipped as a whole, rather than running every variant, size and
        # thread count to print "no still@exit data" each time.
        pair_skipped = False
        for base in args.seq_omp_variants:
            if pair_skipped:
                break
            for size in args.seq_omp_sizes:
                if pair_skipped:
                    break
                var = f"{base}@{size}"
                if skip_variant_on_config(var, omp_cfg) or \
                   skip_variant_on_config(var, seq_cfg):
                    continue
                seq_exe = resolve_exe(args.build_root, seq_cfg, var, "perf")
                omp_exe = resolve_exe(args.build_root, omp_cfg, var, "perf")
                if not seq_exe or not omp_exe:
                    continue

                lead = (f"{dim(_hms())} {short_config_label(omp_cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | ")

                # --- sequential reference run -----------------------------
                _write_pinned_seed(args.output_dir, args.seq_omp_iter_max,
                                   args.seq_omp_seed)
                sys.stdout.write("\r" + lead + "sequential…          ")
                sys.stdout.flush()
                ok, _dur, code, out, err_tail = run_capture_stdout(
                    [seq_exe] + bench_args, {}, to,
                    extra_ld_library_path=extra_ldpath, cwd=args.output_dir,
                )
                if not ok:
                    sys.stdout.write("\r" + lead + red("sequential FAIL") +
                                     f" (code {code})\n")
                    _entry = (f"SEQ_OMP_COMPARE_RUN_FAIL: {seq_exe} "
                             f"{' '.join(bench_args)} code={code}\n"
                             f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    continue
                if not _seed_was_pinned(out):
                    _entry = (f"SEQ_OMP_COMPARE_SEED_FALLBACK: {seq_exe} did not use "
                             f"the pinned seed (seed{args.seq_omp_iter_max}.txt failed "
                             f"to load) -- comparison skipped for {var} on "
                             f"{short_config_label(omp_cfg)}")
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    continue
                seq_still = _parse_still_at_exit(out, base)
                seq_checksum = _parse_last_long_val_sum(out)
                if seq_still is None:
                    sys.stdout.write("\r" + lead +
                                     yellow("no CHUNK_STATS block in the sequential output "
                                            "-- not a stats build, pair skipped") + "\n")
                    pair_skipped = True
                    n_pairs_skipped += 1
                    break

                # --- one OMP run per requested thread count ----------------
                for tc in args.seq_omp_threads:
                    _write_pinned_seed(args.output_dir, args.seq_omp_iter_max,
                                       args.seq_omp_seed)
                    sys.stdout.write("\r" + lead + f"omp {tc:>3}th…            ")
                    sys.stdout.flush()
                    ok, _dur, code, out, err_tail = run_capture_stdout(
                        [omp_exe] + bench_args, {"OMP_NUM_THREADS": str(tc)}, to,
                        extra_ld_library_path=extra_ldpath, cwd=args.output_dir,
                    )
                    n_total += 1
                    if not ok:
                        sys.stdout.write("\r" + lead + red(f"omp {tc}th FAIL") +
                                         f" (code {code})\n")
                        _entry = (f"SEQ_OMP_COMPARE_RUN_FAIL: {omp_exe} "
                                 f"{' '.join(bench_args)} threads={tc} code={code}\n"
                                 f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue
                    if not _seed_was_pinned(out):
                        _entry = (f"SEQ_OMP_COMPARE_SEED_FALLBACK: {omp_exe} @ {tc}th "
                                 f"did not use the pinned seed -- comparison skipped")
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue

                    omp_checksum = _parse_last_long_val_sum(out)
                    if seq_checksum is not None and omp_checksum is not None \
                            and seq_checksum != omp_checksum:
                        sys.stdout.write("\r" + lead +
                                         red(f"omp {tc}th CONTENT MISMATCH") + "\n")
                        _entry = (
                            f"SEQ_OMP_COMPARE_CONTENT_MISMATCH: {var} on "
                            f"{short_config_label(omp_cfg)} @ {tc}th -- long_val_sum "
                            f"differs from sequential DESPITE the pinned seed "
                            f"(seq={seq_checksum} omp={omp_checksum}). The array's own "
                            f"final content differs by thread count; this is a "
                            f"correctness question upstream of chunk-compression "
                            f"bookkeeping."
                        )
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue

                    omp_still = _parse_still_at_exit(out, base)
                    if seq_still is None or omp_still is None:
                        sys.stdout.write("\r" + lead +
                                         yellow(f"omp {tc}th no still@exit data") + "\n")
                        continue

                    if seq_still == omp_still:
                        n_pass += 1
                        sys.stdout.write(
                            "\r" + lead +
                            green(f"omp {tc:>3}th still@exit={omp_still} == seq") + "\n")
                    else:
                        sys.stdout.write(
                            "\r" + lead +
                            red(f"omp {tc:>3}th still@exit={omp_still} != "
                                f"seq({seq_still})") + "\n")
                        _entry = (
                            f"SEQ_OMP_COMPARE_MISMATCH: {var} on "
                            f"{short_config_label(omp_cfg)} @ {tc}th -- still compressed "
                            f"(at exit): seq={seq_still} omp={omp_still} "
                            f"(seed={args.seq_omp_seed} pinned, long_val_sum matched -- "
                            f"same data, different classification)"
                        )
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                    sys.stdout.flush()

    if n_total == 0 and n_pairs_skipped:
        tag = yellow(f"nothing compared: {n_pairs_skipped} pair(s) without CHUNK_STATS "
                     "-- build the -stats pair (-DCHUNK_STATS=ON, OMP on and off)")
    elif n_total == 0:
        tag = yellow("no runs (missing binaries?)")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Seq vs OMP compare: {tag}")
    return n_pass, n_total


def resolve_build_dir(build_root, config):
    """The CMake build directory of `config` for the CURRENT arch -- where
    ctest is run from. Same layouts as resolve_exe():
      1. $build_root/$config/<arch>/   (CEA layout, filtered on the arch)
      2. $build_root/$config/
    A build directory is one holding a CTestTestfile.cmake. Returns the
    absolute path, or None."""
    root = Path(build_root)
    for d in sorted((root / config).glob("*/")):
        if d.is_dir() and _arch_matches_current(d.name) and (d / "CTestTestfile.cmake").is_file():
            return str(d.resolve())
    d = root / config
    if (d / "CTestTestfile.cmake").is_file():
        return str(d.resolve())
    return None


_CTEST_SUMMARY_RE = _re.compile(
    r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")


def run_ctest_phase(args, extra_ldpath, errors):
    """Validation harnesses, through ctest.

    examples/validation registers exit-0/1 programs under the `validation`
    label (rechunk_validation, dyn_geometry_validation, compress_roundtrip,
    pool_accounting, deferred_path_validation, open_types_validation -- one
    executable per variant, the last four tested at 1 and 8 threads). They
    are quick and they run FIRST: a library that cannot round-trip a
    compression pass or tear itself down is not worth benchmarking, and the
    failure is far easier to read from a 150-line harness than from the bench.

    Runs `ctest -L <label> --output-on-failure` in each config's build
    directory (the one holding CTestTestfile.cmake, with or without the $arch
    level). ctest's own per-test timeout applies; the phase timeout is the
    bench one scaled by the test count. A config with no build directory is
    noted once and skipped.
    """
    print(cyan(bold(f"\n🧪 Validation harnesses — ctest -L {args.ctest_label}")))

    n_pass = n_total = 0
    for cfg in args.configs:
        bdir = resolve_build_dir(args.build_root, cfg)
        lead = f"{dim(_hms())} {short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
        if not bdir:
            print(lead + dim("no CTestTestfile.cmake (not a CMake build dir) — skipped"))
            continue

        cmd = ["ctest", "-L", args.ctest_label, "--output-on-failure",
               "-j", str(args.ctest_jobs)]
        if args.ctest_regex:
            cmd += ["-R", args.ctest_regex]
        # The harnesses set OMP_NUM_THREADS themselves (test property); only
        # the sanitizer options need to reach them.
        env = sanitizer_env_for(cfg) if is_sanitizer_config(cfg) else {}
        to  = effective_timeout(args.timeout, args.timeout_per_thread, 8) * 4

        sys.stdout.write("\r" + lead + "running…       ")
        sys.stdout.flush()
        ok, dur, code, out, err_tail = run_capture_stdout(
            cmd, env, to, extra_ld_library_path=extra_ldpath, cwd=bdir)

        m = _CTEST_SUMMARY_RE.search(out)
        if m:
            failed, total = int(m.group(2)), int(m.group(3))
            passed = total - failed
        else:
            failed, total, passed = (0 if ok else 1), (1 if not ok else 0), 0
        n_pass  += passed
        n_total += total

        if ok and total > 0:
            sys.stdout.write("\r" + lead + green(f"{passed}/{total} passed") +
                             f" ({dur:.1f}s)\n")
        elif total == 0 and ok:
            sys.stdout.write("\r" + lead + yellow("no test carries the label") + "\n")
        else:
            sys.stdout.write("\r" + lead + red(f"{passed}/{total} passed") +
                             f" (ctest code {code}, {dur:.1f}s)\n")
            # Keep what ctest prints for the failing tests (--output-on-failure):
            # the harness names the first mismatching element, chunk and offset.
            failing = [l for l in out.splitlines()
                       if "***" in l or "Failed" in l or "FAIL" in l]
            _entry = (
                f"CTEST_FAIL: {' '.join(cmd)} in {bdir} code={code}\n"
                f"  {failed} of {total} test(s) failed:\n    "
                + "\n    ".join(failing[:40])
                + ("\n  stderr_tail:\n    " + err_tail.replace("\n", "\n    ") if err_tail else "")
            )
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no validation test found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Validation harnesses: {tag}")
    return n_pass, n_total


def run_rechunk_phase(args, extra_ldpath, errors):
    """Feature test for the runtime adaptive re-chunk (dynamic builds).

    For every chunk-family variant it runs the _dyn binary of BOTH stages
    (perf and check) with the advisor enabled and asserts that a re-chunk is
    actually applied ([APPLIED]) with no crash / sanitizer hit — i.e. the
    advisor -> decider -> rechunk_to pipeline works end to end in the live loop.

      * no --compress            : quiescent between-phase path (no in-loop compr).
      * CHUNK_ADVISOR_PERIOD=4    : tick often so a verdict lands within the run.
      * CHUNK_COARSEN_MIN_META_MB : the floor that decides a coarsening; 0 here
                                    so the move fires on the shrunk arrays this
                                    test uses.
      * CHUNK_ADVISOR_MIN_GAIN_MB : the absolute deadband (8 MB by policy); 0
                                    here for the same reason -- at --divisor 200
                                    the 64 -> 128 metadata gain is ~0.7 MB, and
                                    the deadband alone refused every tick.
      * check stage              : --mode check reads the oracle (data_<tab>_<k>
                                   .bin, written once by the baseline run of
                                   the correctness phase in output_dir), so it
                                   runs THERE, read-only, and is skipped with a
                                   notice when the oracle is absent (--skip-check,
                                   or a failed baseline). It used to run in a
                                   directory of its own, empty: the binary
                                   aborted on the first missing file and the
                                   phase reported FAIL (code -6) on every
                                   chunk-family variant.
      * perf stage               : its own cwd (nothing read, nothing kept);
                                   --divisor/--iter_max shrink the huge perf data
                                   so the feature test stays quick (it validates
                                   the feature under -DPERF, not perf numbers).
    """
    print(cyan(bold("\n🔁 Adaptive re-chunk feature test — dyn binaries (perf + check)")))

    bases = []
    for v in args.variants:
        b = _variant_base(v)
        if b in ("chunk", "chunk2", "chunk2a") and b not in bases:
            bases.append(b)
    if not bases:
        print(dim("   (no chunk-family variants selected — skipped)"))
        return 0, 0

    stage_extra_args = {
        "check": ["--mode", "check"],
        "perf":  ["--mode", "perf", "--divisor", "200", "--iter_max", "40"],
    }
    env_common = {
        "OMP_NUM_THREADS":      str(args.check_threads),
        "CHUNK_ADVISOR_PERIOD": "4",
        # Both gates of a coarsening, off: the metadata floor and the absolute
        # deadband (the decider takes the larger of the two, so one alone
        # still refused the ~0.7 MB gain of the arrays --divisor leaves).
        "CHUNK_COARSEN_MIN_META_MB": "0",
        "CHUNK_ADVISOR_MIN_GAIN_MB": "0",
    }

    def _oracle_ok(cfg):
        return oracle_complete(args.output_dir, cfg)

    n_pass = n_total = 0
    for cfg in args.configs:
        san = sanitizer_env_for(cfg)
        to  = effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads)
        if not _oracle_ok(cfg):
            print(yellow(f"   {short_config_label(cfg)}: no complete oracle in "
                         f"{oracle_dir(args.output_dir, cfg)}: check stage skipped, perf only"))
        for base in bases:
            var = f"{base}@dyn"
            if skip_variant_on_config(var, cfg):
                continue
            for stage in ("check", "perf"):
                if stage == "check" and not _oracle_ok(cfg):
                    continue
                exe = resolve_exe(args.build_root, cfg, var, stage)
                if not exe:
                    continue
                n_total += 1
                lead = (f"{dim(_hms())} {short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | {stage:5} | ")
                sys.stdout.write("\r" + lead + "running…       ")
                sys.stdout.flush()
                # check reads the config's oracle where the correctness phase
                # wrote it (a shrink binary never writes reference files:
                # io_res is 2, read-only, in every chunk build); perf touches
                # no file and gets a directory of its own.
                rdir = (oracle_dir(args.output_dir, cfg) if stage == "check"
                        else _run_dir(args.output_dir, "rechunk", short_config_label(cfg), var, stage))
                ok, dur, code, err_tail, _ = run_cmd(
                    [exe] + stage_extra_args[stage],
                    {**env_common, **san}, to,
                    extra_ld_library_path=extra_ldpath, cwd=rdir,
                    # "[APPLIED]", not "[RECHUNK-APPLY]": the library prints
                    # "[RECHUNK-DECIDE] <array>: re-chunk S=.. -> S'=.. (...) [APPLIED]".
                    # The old marker existed nowhere in the tree, so this test
                    # reported FAIL on a process that had exited 0.
                    must_contain="[APPLIED]",
                )
                if ok:
                    n_pass += 1
                    sys.stdout.write("\r" + lead + green("re-chunk applied ✓") +
                                     f" ({dur:.1f}s)\n")
                else:
                    sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                    _entry = (
                        f"RECHUNK_FEATURE_FAIL: {exe} "
                        f"{' '.join(stage_extra_args[stage])} stage={stage} code={code}\n"
                        f"  (expected '[APPLIED]'; env CHUNK_ADVISOR_PERIOD=4 "
                        f"CHUNK_COARSEN_MIN_META_MB=0 CHUNK_ADVISOR_MIN_GAIN_MB=0)\n"
                        f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    ')
                    )
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no _dyn binaries found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Adaptive re-chunk feature: {tag}")
    return n_pass, n_total


def run_scaling_phase(args, extra_ldpath, errors):
    """Scalability guard: one perf run per thread count and per variant, on
    omp-llvm and omp-gnu (SCALING_CONFIGS).

    The averaged perf phase answers "how fast", this one answers "does it
    still scale". It runs EVERY perf binary of the sweep (--variants x --sizes,
    as the perf phase does, plus vector as the reference curve even when
    --variants leaves it out) over SCALING_THREADS (4..128) with a shrunk
    workload, prints each curve with the speedup against its first point, and
    FAILS when a point is slower than the BEST point of the same variant so far
    by more than SCALING_MAX_SLOWDOWN (1.25: a single run has spread, the top
    of the curve legitimately plateaus, and on a hybrid machine the points past
    the performance cores are genuinely slower -- a serialised pass shows up as
    1.5x and beyond). Each variant is its own curve: a regression confined to
    one container (a lock only chunk2 takes, a pass only v1 runs) is exactly
    what a single-variant guard would miss.

    Everything it needs is a constant at the top of this file, deliberately:
    seven knobs for a guard that either says "still scales" or names the point
    where it stopped would be seven ways to make its verdict unreadable.
    --skip-scaling turns it off, and it runs in EVERY mode -- most of all in
    --mode fast, which cuts the perf sweep down to 64/128 and would otherwise
    see no scalability regression at all.

    Why these two configs: LLVM is the fastest of the matrix, hence the one
    whose curves are worth believing; GNU is the one without OMPT, whose
    phase-end path differs (see SCALING_CONFIGS). The allocator, aggressive-
    compression and stats configs would add runs, not information.
    The binaries are the int64 ones at --multiply 2 (past INT_MAX elements) and
    --iter_max 500 (see SCALING_MULTIPLY): a REGRESSION signal on the 64-bit
    path at full scale, not a measurement campaign. A config built without
    the int64 binaries falls back to its 32-bit ones at --divisor 50
    --iter_max 200 (SCALING_FALLBACK_*). The thread counts
    the machine cannot honour (more than CHUNK_PMP_MAX_THREADS / the CPUs
    present) are dropped with a note rather than measured against an
    oversubscribed run.
    """
    print(cyan(bold("\n📈 Scalability guard — "
                    + ", ".join(short_config_label(c) for c in SCALING_CONFIGS))))

    # Oversubscription measures the scheduler, not the library: drop the
    # points the machine cannot run one thread per CPU.
    ncpu = os.cpu_count() or 1
    threads = [t for t in SCALING_THREADS if t <= ncpu]
    dropped = [t for t in SCALING_THREADS if t > ncpu]
    if dropped:
        print(dim(f"   {ncpu} CPUs: thread count(s) {dropped} dropped (would oversubscribe)"))
    if len(threads) < 2:
        print(dim("   (fewer than two usable thread counts -- skipped)"))
        return 0, 0

    args_int64 = ["--mode", "perf", "--no_io",
                  "--multiply", str(SCALING_MULTIPLY),
                  "--iter_max", str(SCALING_ITER_MAX)]
    args_32bit = ["--mode", "perf", "--no_io",
                  "--divisor", str(SCALING_FALLBACK_DIVISOR),
                  "--iter_max", str(SCALING_FALLBACK_ITER_MAX)]

    # vector first: the reference every other curve is read against.
    scaling_vars = [SCALING_REFERENCE_VARIANT] + [v for v in args.variants
                                                  if v != SCALING_REFERENCE_VARIANT]

    n_pass = n_total = 0
    n_curves = 0
    failed_vars = []
    for config in SCALING_CONFIGS:
        label = short_config_label(config)
        if config not in args.configs:
            print(dim(f"   ({label} not in --configs -- skipped)"))
            continue

        candidates = [v for v in scaling_vars if not skip_variant_on_config(v, config)]
        variants   = []
        missing    = []
        for var in candidates:
            exe = resolve_exe(args.build_root, config, int64_variant(var), "perf")
            if exe:
                variants.append((var, exe))
            else:
                missing.append(var)

        if variants:
            bench_args = args_int64
            if missing:
                print(dim(f"   (no int64 binary in {label} for {', '.join(missing)} -- skipped)"))
        else:
            # No int64 binary at all in this config: fall back to its 32-bit
            # ones, on the shrunk workload (SCALING_FALLBACK_*).
            bench_args = args_32bit
            for var in candidates:
                exe = resolve_exe(args.build_root, config, var, "perf")
                if exe:
                    variants.append((var, exe))
                else:
                    print(dim(f"   (no {var} perf binary in {label} -- skipped)"))
            if not variants:
                print(dim(f"   (no perf binary in {label} -- skipped)"))
                continue
            print(yellow(f"   {label}: no int64 binary (built without -DSHRINK_BENCH_INT64=ON) -- "
                         f"32-bit fallback, {' '.join(bench_args[3:])}"))
        is_int64 = bench_args is args_int64
        n_curves += len(variants)

        for var, exe in variants:
            rdir = _run_dir(args.output_dir, "scaling", label, var)
            v_pass = v_total = 0
            best_t = best_dur = None
            first_dur = None
            for tc in threads:
                v_total += 1
                lead = (f"{dim(_hms())} {label[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | {tc:3}th | ")
                sys.stdout.write("\r" + lead + "running…       ")
                sys.stdout.flush()

                ok, dur, code, err_tail, _ = run_cmd(
                    [exe] + bench_args, {"OMP_NUM_THREADS": str(tc)},
                    (max(SCALING_TIMEOUT, effective_timeout(args.timeout, args.timeout_per_thread, tc))
                     if is_int64 else effective_timeout(args.timeout, args.timeout_per_thread, tc)),
                    extra_ld_library_path=extra_ldpath, cwd=rdir)

                if not ok:
                    sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                    _entry = (f"SCALING_RUN_FAIL: {exe} {' '.join(bench_args)} threads={tc} code={code}\n"
                              f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    sys.stdout.flush()
                    continue

                if first_dur is None:
                    first_dur = dur
                speedup = first_dur / dur if dur > 0 else 0.0
                ratio   = (dur / best_dur) if best_dur else 1.0
                regress = best_dur is not None and ratio > SCALING_MAX_SLOWDOWN

                note = f" x{speedup:5.2f} vs {threads[0]}th"
                if best_dur is not None:
                    note += f", {ratio:4.2f}x the best ({best_t}th)"
                if regress:
                    sys.stdout.write("\r" + lead + red(f"{dur:7.2f}s") + red(note) + "\n")
                    _entry = (f"SCALING_REGRESSION: {label} "
                              f"{var} best {best_t}th {best_dur:.2f}s -> {tc}th {dur:.2f}s "
                              f"({ratio:.2f}x, limit {SCALING_MAX_SLOWDOWN})\n"
                              f"  ({' '.join(bench_args)})\n")
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                else:
                    v_pass += 1
                    sys.stdout.write("\r" + lead + bold(f"{dur:7.2f}s") + dim(note) + "\n")
                sys.stdout.flush()
                if best_dur is None or dur < best_dur:
                    best_t, best_dur = tc, dur
            n_pass  += v_pass
            n_total += v_total
            if v_pass != v_total:
                failed_vars.append(f"{label}/{var}")

    if n_total == 0:
        tag = yellow("nothing to run")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} points scale ({n_curves} curves)")
    else:
        tag = red(f"{n_pass}/{n_total} points scale ({n_curves} curves; "
                  f"failing: {', '.join(failed_vars)})")
    print(f"   Scalability: {tag}")
    return n_pass, n_total


def run_int64_check_phase(args, extra_ldpath, errors):
    """64-bit index checks: a few --mode check runs of the int64 binaries.

    INT64_CHECK_VARIANTS, once each, at --check-threads, in every config that
    has both int64 binaries (the release OpenMP ones) and a complete oracle.
    They run in that oracle, read-only, as the 32-bit checks do: the reference
    files are the 32-bit vector_check's, element values that do not depend on
    the index type, so an int64 binary must reproduce them. A failure is a
    32/64-bit divergence -- in the library's index path, or an index
    expression of the bench that overflows at 32 bits.
    """
    print(cyan(bold("\n🔢 64-bit index checks — int64 binaries against the 32-bit oracle")))

    env = {"OMP_NUM_THREADS": str(args.check_threads)}
    to  = effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads)

    n_pass = n_total = 0
    for cfg in args.configs:
        label = short_config_label(cfg)
        exes  = [(var, resolve_exe(args.build_root, cfg, int64_variant(var), "check"))
                 for var in INT64_CHECK_VARIANTS]
        exes  = [(var, exe) for var, exe in exes if exe]
        if not exes:
            continue  # not an int64 config: nothing to say
        if not oracle_complete(args.output_dir, cfg):
            print(yellow(f"   {label}: no complete oracle in {oracle_dir(args.output_dir, cfg)} -- skipped"))
            continue
        for var, exe in exes:
            n_total += 1
            shown = int64_variant(var)
            lead  = (f"{dim(_hms())} {label[:CONFIG_W]:<{CONFIG_W}} | "
                     f"{shown[:VARIANT_W + 6]:<{VARIANT_W + 6}} | ")
            sys.stdout.write("\r" + lead + "running…       ")
            sys.stdout.flush()
            ok, dur, code, err_tail, _ = run_cmd(
                [exe, "--mode", "check"], {**env, **sanitizer_env_for(cfg)}, to,
                extra_ld_library_path=extra_ldpath, cwd=oracle_dir(args.output_dir, cfg))
            if ok:
                n_pass += 1
                sys.stdout.write("\r" + lead + green("check OK ✓") + f" ({dur:.1f}s)\n")
            else:
                sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                _entry = (f"INT64_CHECK_FAIL: {exe} --mode check threads={args.check_threads} code={code}\n"
                          f"  (int64 build against the 32-bit oracle of {label})\n"
                          f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                errors.append(_entry)
                _append_error_now(args.errors_log, _entry)
            sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no int64 binaries found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   64-bit index checks: {tag}")
    return n_pass, n_total


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark + correctness runner for shrink-array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--build-root", default=os.environ.get("CHUNK_BUILD_ROOT", "."),
                        help="Root directory holding the cmake-build-omp-*/ build "
                             "sub-directories. Default: cwd (or $CHUNK_BUILD_ROOT).")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="Build sub-directories to test (in order).")
    parser.add_argument("--skip-sanitizers", action="store_true",
                        help="Skip the ASan/TSan CHECK phase (2d). Sanitizers run "
                             f"at {SANITIZER_THREADS} threads, {SANITIZER_REPS} run(s) per variant, "
                             "diagnostic only.")
    parser.add_argument("--skip-stl-test", action="store_true",
                        help="Skip the --test_stl STL-interface self-check (2c).")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="Variants to test.")
    parser.add_argument("--sizes", nargs="+", default=["256", "1024", "dyn"],
                        help="Chunk-size declinations to test for the shrink "
                             "variants: any of 64/256/1024 (static compile-time "
                             "sizes) and 'dyn' (runtime size). 'vector' is "
                             "size-agnostic and always tested once. Executables "
                             "are named shrink_bench_<variant>_<stage>_<size>.")
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE,
                        help="Variant used as the baseline for the speedup computation.")
    # --- Regression guards (compression health + memory) ---
    parser.add_argument("--stats-mark-ratio-min", nargs="*", default=None,
                        metavar="TAG=RATIO",
                        help="Per-variant floor on mark/create from the CHUNK_STATS "
                             "block; a run below its floor is a FAIL "
                             "(default: %s). E.g. --stats-mark-ratio-min chunk2a=0.4 "
                             "chunk=0.2. Needs a CHUNK_STATS build; silently skipped "
                             "otherwise." % " ".join(
                                 f"{k}={v}" for k, v in DEFAULT_STATS_MARK_RATIO_MIN.items()))
    parser.add_argument("--no-stats-guard", action="store_true",
                        help="Disable the mark/create compression-health guard.")
    parser.add_argument("--mem-chunk2a-vs-chunk2-max", type=float, default=1.10,
                        help="FAIL if chunk2a peak memory exceeds chunk2's by more than "
                             "this ratio on the same (config, threads, size) "
                             "(default 1.10). Set <= 0 to disable.")
    parser.add_argument("--mem-vs-baseline-max", type=float, default=None,
                        help="FAIL if a chunk variant's peak memory exceeds the baseline "
                             "variant's by more than this ratio (default: off).")
    parser.add_argument("--mode", choices=list(MODE_PRESETS.keys()), default="normal",
                        help="Preset controlling thread distribution AND repetitions: "
                             "'fast' (~1h30 on the full matrix), 'normal' (standard), "
                             "'full' (complete sweep, many reps). Any of --threads, "
                             "--reps_perf, --reps_check, --full-reps-threads passed "
                             "explicitly overrides the preset.")
    parser.add_argument("--threads", nargs="+", type=int, default=None,
                        help="OMP_NUM_THREADS values for the perf loop. "
                             "Overrides the --mode preset.")
    parser.add_argument("--light-variants", nargs="*", default=None,
                        help="Variants run as a single smoke test (one thread count, "
                             "one run) instead of the full sweep — just to confirm they "
                             "still work. Default from --mode (fast: std_alloc).")
    parser.add_argument("--light-threads", type=int, default=None,
                        help="Thread count used for --light-variants smoke runs "
                             "(default 64).")
    parser.add_argument("--full-reps-threads", nargs="+", type=int, default=None,
                        help="Thread count(s) on which to run --reps_perf repetitions "
                             "(for a reliable averaged measurement). On ALL other thread "
                             "counts from --threads, a single run is done (just for the "
                             "scalability curve). Pass a count absent from --threads to "
                             "disable. Overrides the --mode preset.")
    parser.add_argument("--check-threads", type=int, default=DEFAULT_THREADS_CHECK,
                        help="OMP_NUM_THREADS value for the correctness loop.")
    parser.add_argument("--reps_perf",  type=int, default=None,
                        help="Repetitions per (config, variant, thread count) in the "
                             "perf loop (subject to --full-reps-threads). "
                             "Overrides the --mode preset.")
    parser.add_argument("--reps_check", type=int, default=None,
                        help="Repetitions per (config, variant) in the correctness loop. "
                             "Overrides the --mode preset.")
    parser.add_argument("--timeout",    type=int, default=DEFAULT_TIMEOUT,
                        help="Base wall-time cap per run (seconds). The effective "
                             "timeout is --timeout + --timeout-per-thread * threads, "
                             "so high thread counts (e.g. 288) get more time.")
    parser.add_argument("--timeout-per-thread", type=float,
                        default=DEFAULT_TIMEOUT_PER_THREAD,
                        help="Extra seconds added to the timeout per OMP thread "
                             "(effective_timeout = --timeout + this * threads). "
                             "0 disables scaling. Try e.g. 2 so 288 threads get "
                             "--timeout + 576 s.")
    parser.add_argument("--rebuild-retries", type=int, default=5,
                        help="If a binary vanishes mid-run (concurrent rebuild), "
                             "re-resolve and retry this many times before giving up, "
                             "waiting --rebuild-wait s between tries.")
    parser.add_argument("--rebuild-wait", type=float, default=10.0,
                        help="Seconds to wait before re-resolving a binary that "
                             "disappeared (concurrent rebuild).")
    parser.add_argument("--ld-library-path", default=None,
                        help="Path(s) to prepend to LD_LIBRARY_PATH for EVERY run. "
                             "Needed for llvm -stdlib=libc++ binaries when the llvm "
                             "module is not loaded (libc++.so.1 not found), or for the "
                             "gcc lib64 (libstdc++ GLIBCXX_* not found). Not needed once "
                             "the binaries carry a proper RPATH. "
                             "e.g. /ccc/products/llvm-20.1.8/system__cuda--12.4/default/lib")
    parser.add_argument("--auto-libpath", action="store_true",
                        help="Derive the lib/ dir automatically from `command -v clang++` "
                             "(replaces <bin> with <lib>). Load the llvm module first.")
    parser.add_argument("--skip-perf",  action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-ctest", action="store_true",
                        help="Skip the validation harnesses (ctest -L validation "
                             "in each config's build directory: rechunk_validation, "
                             "dyn_geometry_validation, compress_roundtrip, "
                             "pool_accounting, deferred_path_validation, "
                             "open_types_validation).")
    parser.add_argument("--ctest-label", default="validation",
                        help="ctest label selecting the harnesses (default: "
                             "validation, set by examples/validation/CMakeLists.txt).")
    parser.add_argument("--ctest-regex", default=None,
                        help="Restrict the harnesses to the tests matching this "
                             "regex (ctest -R), e.g. compress_roundtrip.")
    parser.add_argument("--ctest-jobs", type=int, default=4,
                        help="Parallel ctest jobs per config (default 4; the "
                             "harnesses are small, the 8-thread ones share cores).")
    parser.add_argument("--skip-scaling", action="store_true",
                        help="Skip the scalability guard: one perf run per thread count "
                             f"({', '.join(map(str, SCALING_THREADS))}) and per perf variant (vector "
                             "included as the reference) on "
                             f"{', '.join(short_config_label(c) for c in SCALING_CONFIGS)}, which FAILS when a point "
                             f"is more than {SCALING_MAX_SLOWDOWN}x the best. On in every mode; the constants are at "
                             "the top of this file (SCALING_*), not options -- the guard is "
                             "a regression signal, not a measurement campaign.")
    parser.add_argument("--skip-int64-check", action="store_true",
                        help="Skip the 64-bit index checks: --mode check of the int64 binaries ("
                             + ", ".join(INT64_CHECK_VARIANTS) + ") against the 32-bit oracle.")
    parser.add_argument("--skip-rechunk", action="store_true",
                        help="Skip the adaptive re-chunk feature test on the "
                             "_dyn binaries (perf + check stages).")
    parser.add_argument("--skip-seq-omp-compare", action="store_true",
                        help="Skip the seed-pinned sequential-vs-OpenMP "
                             "compression-classification comparison (LLVM "
                             "seq/omp config pairs, chunk/chunk2/chunk2a, "
                             "256/1024/dyn, several thread counts).")
    parser.add_argument("--seq-omp-threads", nargs="+", type=int,
                        default=list(DEFAULT_SEQ_OMP_THREADS),
                        help="OMP thread counts for the seq-vs-omp compare "
                             "phase.")
    parser.add_argument("--seq-omp-sizes", nargs="+",
                        default=list(DEFAULT_SEQ_OMP_SIZES),
                        help="Chunk size declinations for the seq-vs-omp "
                             "compare phase.")
    parser.add_argument("--seq-omp-variants", nargs="+",
                        default=list(DEFAULT_SEQ_OMP_VARIANTS),
                        help="Chunk-family variant bases for the seq-vs-omp "
                             "compare phase.")
    # --- Randomised parameter sweep (fuzz) --------------------------------
    parser.add_argument("--fuzz-cases", type=int, default=0,
                        help="Run N randomised cases: variant, size, divisor, "
                             "iter_max and thread count drawn at random, each "
                             "checked against its own sequential run with the "
                             "seed pinned. 0 (default) disables the phase. Each "
                             "case prints its parameters and seed, and a failure "
                             "is logged as the command line that produced it, so "
                             "replay needs no flag. The pools drawn from are the "
                             "FUZZ_* constants in this file.")
    parser.add_argument("--seq-omp-seed", type=int, default=DEFAULT_SEQ_OMP_SEED,
                        help="Fixed RNG seed pinned before every run in the "
                             "seq-vs-omp compare phase, so sequential and "
                             "every OMP thread count see identical data. 0 is "
                             "invalid -- the bench treats it as \"generate a "
                             "random seed\", which would defeat the "
                             "comparison.")
    parser.add_argument("--seq-omp-iter-max", type=int,
                        default=DEFAULT_SEQ_OMP_ITER_MAX,
                        help="--iter_max passed to every seq-vs-omp compare "
                             "run; also determines the pinned seed's file "
                             "name (seed<N>.txt).")
    parser.add_argument("--seq-omp-divisor", type=int,
                        default=DEFAULT_SEQ_OMP_DIVISOR,
                        help="--divisor passed to every seq-vs-omp compare "
                             "run, to keep it quick.")
    parser.add_argument("--regen-ref", action="store_true",
                        help="Force regeneration of the data_tab* reference "
                             "oracles: delete the existing (possibly partial) "
                             "one of every config under --output-dir/oracle/, "
                             "then run each config's vector check to regenerate "
                             "a complete set. Use after rebuilding (another "
                             "compiler, other flags), or whenever a CHECK aborts "
                             "on a missing dump or reads back 0/1 differences.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory where ALL run outputs go: the summary "
                             "CSV, the errors log, the reference oracles "
                             "(oracle/<config>/data_tab*, one per config, "
                             "generated by that config's own vector binary: the "
                             "reference is not portable across compilers) and "
                             "the binaries' memory_mo.txt. Default: "
                             "$CCCWORKDIR/bench_results when that "
                             "work tree exists (never the script's own directory), "
                             "else the current directory. Created if missing.")
    parser.add_argument("--errors-log",  default="bench_errors.log",
                        help="Errors log filename (placed in --output-dir).")
    parser.add_argument("--summary-csv", default="bench_summary.csv",
                        help="Summary CSV filename (placed in --output-dir).")
    parser.add_argument("--full-log-dir", default=None,
                        help="Directory for the COMPLETE stdout+stderr of EVERY run "
                             "(one file per run). Relative -> under --output-dir. "
                             "Defaults to <output-dir>/run_logs (i.e. under "
                             "bench_results/<arch>). Captures re-chunk lines and the "
                             "full crash context the 30-line err_tail drops. Pass "
                             "--full-log-dir . to write directly into the arch dir, "
                             "or --no-full-log to disable.")
    parser.add_argument("--numa-interleave", choices=list(_numa.MODES), default="auto",
                        help="auto: run under `numactl --interleave=<nodes>` when the run spans "
                             "2+ NUMA nodes (only the CPU nodes of the cores it uses; Grace, "
                             "144 threads: -6.9%%). off: never (timings comparable with runs "
                             "made before this option).")
    parser.add_argument("--rerun-failed", nargs="?", const="", default=None, metavar="FILE",
                        help="Replay ONLY the runs that failed in a previous run, recorded in "
                             "<output-dir>/failed_cases.jsonl (or FILE), and skip every phase. "
                             "Rewrites the list with what still fails (previous one kept as "
                             "failed_cases.prev.jsonl). Exit code 1 if any case still fails.")
    parser.add_argument("--rerun-repeat", type=int, default=1,
                        help="With --rerun-failed: re-run each failed case up to N times, "
                             "stopping at its first failure (for intermittent failures).")
    parser.add_argument("--no-full-log", action="store_true",
                        help="Disable the default per-run full stdout+stderr "
                             "capture (see --full-log-dir).")
    args = parser.parse_args()

    # Regression guards: apply CLI overrides to the module-level floor map used
    # inside run_cmd (which has no access to args).
    global _STATS_MARK_RATIO_MIN, _NUMA_MODE
    _NUMA_MODE = args.numa_interleave
    if args.no_stats_guard:
        _STATS_MARK_RATIO_MIN = {}
    elif args.stats_mark_ratio_min is not None:
        floors = {}
        for item in args.stats_mark_ratio_min:
            if "=" not in item:
                parser.error(f"--stats-mark-ratio-min expects TAG=RATIO, got {item!r}")
            tag, val = item.split("=", 1)
            try:
                floors[tag.strip()] = float(val)
            except ValueError:
                parser.error(f"--stats-mark-ratio-min: bad ratio in {item!r}")
        _STATS_MARK_RATIO_MIN = floors
    mem_c2a_vs_c2 = (args.mem_chunk2a_vs_chunk2_max
                     if args.mem_chunk2a_vs_chunk2_max and args.mem_chunk2a_vs_chunk2_max > 0
                     else None)

    # Resolve the output directory and make the csv/log paths absolute inside it.
    # A caller can still pass an absolute --summary-csv/--errors-log to override.
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        # Default OUTSIDE the source tree: run outputs (oracle, CSV, logs,
        # memory_mo.txt) must not pile up in python/. Use the CCC work area
        # when present; fall back to the cwd elsewhere (local workstation).
        _ccc_work = f"/ccc/work/contXXX/project/{os.environ.get('USER', '')}"
        out_dir = (os.path.join(_ccc_work, "bench_results")
                   if os.path.isdir(_ccc_work) else os.getcwd())
    # Separate run outputs by architecture. aarch64 and x86_64 runs commonly
    # share a build_root and are launched with the SAME --output-dir; without an
    # arch level they write the same bench_summary.csv / bench_errors.log and,
    # worse, the same data_tab_* oracle in cwd=output_dir. Concurrent runs then
    # corrupt each other's files, causing spurious mid-run crashes. Appending the
    # current machine keeps each architecture's results (and oracle) isolated.
    out_dir = os.path.join(out_dir, CURRENT_MACHINE)
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir
    print(f"📁 Run outputs → {out_dir}  (arch: {CURRENT_MACHINE})")
    if not os.path.isabs(args.summary_csv):
        args.summary_csv = os.path.join(out_dir, args.summary_csv)
    if not os.path.isabs(args.errors_log):
        args.errors_log = os.path.join(out_dir, args.errors_log)
    # Per-run full-output capture is ON by default, landing in <out_dir>/run_logs
    # (arch-aware: out_dir already ends in bench_results/<arch>). Override the
    # location with --full-log-dir, or switch it off with --no-full-log.
    if not args.no_full_log:
        _fld = args.full_log_dir if args.full_log_dir is not None else "run_logs"
        if not os.path.isabs(_fld):
            _fld = os.path.join(out_dir, _fld)
        os.makedirs(_fld, exist_ok=True)
        _set_full_log_dir(_fld)
        print(f"\U0001f5d2  Full per-run logs \u2192 {_fld}")

    # Tee all screen output into bench_results/bench_output.log so every run
    # leaves a full, colour-free transcript on disk next to the CSV and error
    # log. Screen output is unchanged (colours + live progress bars).
    _console_log_path = os.path.join(out_dir, "bench_output.log")
    try:
        _console_log_file = open(_console_log_path, "w")
        import datetime as _dt0
        _console_log_file.write(
            f"# bench_and_test.py run log — {_dt0.datetime.now().isoformat(timespec='seconds')}\n")
        _console_log_file.write("# command: " + " ".join(sys.argv) + "\n\n")
        sys.stdout = _Tee(sys.stdout, _console_log_file)
        import atexit as _atexit
        _atexit.register(lambda: isinstance(sys.stdout, _Tee) and sys.stdout.close_log())
    except OSError:
        _console_log_file = None  # if we can't open it, just run without the log
    # --rerun-failed: replay the recorded failures only, and stop there (the
    # errors log of the failing run is left as it was).
    if args.rerun_failed is not None:
        sys.exit(run_rerun_failed(args))
    # Truncate the errors log at startup so incremental _append_error_now()
    # writes reflect only THIS run (append mode would otherwise accumulate
    # across runs and mix stale failures with current ones).
    try:
        open(args.errors_log, "w").close()
    except OSError:
        pass
    # Same for the replay list of --rerun-failed.
    try:
        open(_failed_cases_path(args.errors_log), "w").close()
    except OSError:
        pass
    print(f"📂 Output directory: {out_dir}")

    # --- Apply the --mode preset for any arg the user did NOT set explicitly ---
    preset = MODE_PRESETS[args.mode]
    if args.threads is None:
        args.threads = list(preset["threads"])
    if args.reps_perf is None:
        args.reps_perf = preset["reps_perf"]
    if args.reps_check is None:
        args.reps_check = preset["reps_check"]
    if args.full_reps_threads is None:
        args.full_reps_threads = list(preset["full_reps_threads"])
    if args.light_variants is None:
        args.light_variants = list(preset.get("light_variants", []))
    if args.light_threads is None:
        args.light_threads = preset.get("light_threads", 64)
    # --- Expand variants across the requested size declinations -----------
    # Shrink variants (chunk/chunk2/chunk2a/std_alloc) gain a "@<size>" suffix
    # so each (variant, size) pair is handled independently by the existing
    # machinery (resolve_exe decodes "base@size"). "vector" has no chunk
    # geometry and is left as-is. light_variants get the same treatment so a
    # light variant like std_alloc still resolves to real per-size binaries.
    SIZE_AGNOSTIC = {"vector"}

    def _expand_with_sizes(variant_list):
        out = []
        for v in variant_list:
            if v in SIZE_AGNOSTIC or "@" in v:
                out.append(v)
            else:
                out.extend(f"{v}@{s}" for s in args.sizes)
        return out

    args.variants        = _expand_with_sizes(args.variants)
    args.light_variants  = _expand_with_sizes(args.light_variants)
    # The baseline variant may also need a size to resolve its check binary.
    # vector is size-agnostic, so the common baseline (vector) is unaffected; if
    # a shrink variant is used as baseline, pin it to the first requested size.
    if args.baseline_variant not in SIZE_AGNOSTIC and "@" not in args.baseline_variant:
        args.baseline_variant = f"{args.baseline_variant}@{args.sizes[0]}"

    light_note = (f" light={args.light_variants}@{args.light_threads}t"
                  if args.light_variants else "")
    print(f"🎚️  Mode '{args.mode}': threads={args.threads} reps_perf={args.reps_perf} "
          f"full_reps_threads={args.full_reps_threads} reps_check={args.reps_check}{light_note}")

    # --- Fast-mode matrix reduction --------------------------------------
    # Only trims work the user did not pin explicitly on the command line.
    perf_excluded = set(PERF_EXCLUDED_CONFIGS)
    if args.mode == "fast":
        # 1. Drop -have-agressive-compression configs: benchmarks confirm they
        #    are within noise of the release configs on this workload, so running
        #    them doubles the matrix for no signal. (Skipped only if the user did
        #    not name configs explicitly.)
        if args.configs == DEFAULT_CONFIGS:
            kept = [c for c in args.configs if not is_compression_only_config(c)]
            dropped = [short_config_label(c) for c in args.configs if c not in kept]
            if dropped:
                print(dim(f"   fast: skipping aggressive-compression configs "
                          f"({', '.join(dropped)}) — ≈ release on this workload"))
            args.configs = kept
        args.light_variants = sorted(set(args.light_variants) | {"std_alloc"})
        # There used to be a rule here demoting chunk/chunk2/chunk2a on a
        # -std-alloc config to a light smoke run, "since only the allocator
        # differs". Removed, for two reasons. It never ran: it compared bare
        # names against `var`, which carries its @<size> declination since the
        # expansion above, so "chunk@256" never matched "chunk" (the same
        # bare-key bug that once reported 0 STL runs; _variant_base() is what
        # such a test needs). And it should not run: the allocator moves the
        # numbers well outside smoke-test territory -- on aarch64, chunk@1024 at
        # 64 threads is 14.53 s under gnu and 12.60 s under gnu-std-alloc (+15%),
        # and chunk@dyn 56.4 s vs 40.3 s (+40%). Those are results, not
        # duplicates.
        #
        # 2. Drop the PERF phase of every serial (non-OpenMP) config. They stay
        #    in CHECK, in the validation harnesses and in the seq/omp compare --
        #    only the timing sweep goes. It was the single largest item in the
        #    campaign: on aarch64, gnu-noomp 89 min and llvm-noomp about 2 h, so
        #    3h29 of a 5 h run, to time a single-threaded build whose numbers do
        #    not move between campaigns. `--mode normal` and `--mode full` still
        #    measure them.
        _serial = [c for c in args.configs if not is_omp_config(c)]
        if _serial:
            perf_excluded |= set(_serial)
            print(dim(f"   fast: skipping the PERF phase of the serial configs "
                      f"({', '.join(short_config_label(c) for c in _serial)}) "
                      f"— kept in CHECK and validation"))

    # --- Resolve the extra LD_LIBRARY_PATH -------------------------------
    extra_ldpath = args.ld_library_path
    if args.auto_libpath and not extra_ldpath:
        import shutil
        cxx = shutil.which("clang++")
        if cxx:
            # /.../llvm-20.1.8/.../bin/clang++ -> /.../llvm-20.1.8/.../lib
            lib = str(Path(cxx).resolve().parent.parent / "lib")
            if Path(lib).is_dir():
                extra_ldpath = lib
                print(f"🔧 --auto-libpath: LD_LIBRARY_PATH += {lib}")
            else:
                print(f"⚠️  --auto-libpath: {lib} does not exist, ignored.")
        else:
            print("⚠️  --auto-libpath: clang++ not found in PATH "
                  "(load the llvm module first).")
    if extra_ldpath:
        print(f"🔧 Extra LD_LIBRARY_PATH for every run: {extra_ldpath}")

    # --- Drop configs whose build directory is absent (one note each) ----
    present, absent = [], []
    for cfg in args.configs:
        (present if config_dir_present(args.build_root, cfg) else absent).append(cfg)
    if absent:
        print(dim("ℹ️  Skipping unbuilt configs (no binary for this arch): "
                  + ", ".join(short_config_label(c) for c in absent)))
    args.configs = present
    if not args.configs:
        print(red(f"\n❌ None of the requested configs are built under "
                  f"{args.build_root!r}."))
        sys.exit(2)
    # The two stats checks read the CHUNK_STATS block; without a stats build in
    # the run they pass without having looked at anything. Say so up front
    # rather than letting a green run suggest the compression health was
    # verified.
    if not any(is_stats_config(c) for c in args.configs):
        print(yellow("⚠️  No CHUNK_STATS config in this run: the mark/create "
                     "compression-health guard and the seq-vs-OMP still@exit "
                     "compare have nothing to read and will pass vacuously. "
                     "Build cmake-build-omp-gnu-stats / cmake-build-gnu-stats "
                     "(-DCHUNK_STATS=ON) to arm them."))

    # --- Phase 0: binary discovery ---------------------------------------
    print(f"🔎 Resolving executables under build_root={args.build_root!r}")
    print(f"   Current machine: {CURRENT_MACHINE} → only *__{CURRENT_MACHINE} "
          f"builds will be run")
    perf_exes  = {}   # (config, var) -> path
    check_exes = {}   # (config, var) -> path   (non-baseline variants)
    # config -> path of ITS vector check binary: every config generates its
    # own oracle (see oracle_dir), whatever skip_variant_on_config says about
    # running the vector variant on it.
    baseline_check_exe_of = {}
    missing_perf, missing_check = [], []

    for cfg in args.configs:
        if not args.skip_check:
            _bp = resolve_exe(args.build_root, cfg, args.baseline_variant, "check")
            if _bp:
                baseline_check_exe_of[cfg] = _bp
        for var in args.variants:
            if skip_variant_on_config(var, cfg):
                # e.g. vector on -have-agressive-compression: identical binary
                # to the base config, not re-run. Don't resolve or flag missing.
                continue
            if cfg in perf_excluded:
                pass  # kept in CHECK, dropped from PERF (redundant with gnu-std-alloc)
            else:
                p = resolve_exe(args.build_root, cfg, var, "perf")
                if p:
                    perf_exes[(cfg, var)] = p
                else:
                    missing_perf.append((cfg, var))

            if var == args.baseline_variant:
                # The baseline check (vector, uncompressed) only generates the
                # oracle: resolved per config above (baseline_check_exe_of),
                # never a CHECK run of its own.
                continue
            p = resolve_exe(args.build_root, cfg, var, "check")
            if p:
                check_exes[(cfg, var)] = p
            else:
                missing_check.append((cfg, var))

    def _fmt(cfg, var):
        return f"[{short_config_label(cfg):40} / {var:10}]"

    for (cfg, var), p in sorted(perf_exes.items()):
        print(f"   ✅ perf  {_fmt(cfg, var)} → {p}")
    for (cfg, var), p in sorted(check_exes.items()):
        print(f"   ✅ check {_fmt(cfg, var)} → {p}")
    for cfg, var in missing_perf:
        print(f"   ⚠️  MISSING perf  {_fmt(cfg, var)} — skipped")
    for cfg, var in missing_check:
        print(f"   ⚠️  MISSING check {_fmt(cfg, var)} — skipped")

    if not perf_exes and not check_exes:
        print(f"\n❌ No executable found under {args.build_root!r}.")
        print("   Check --build-root (default = cwd), or pass --configs "
              "explicitly, or --variants.")
        sys.exit(2)

    # --- Pre-flight: check dynamic linkage -------------------------------
    # Run ldd on one binary per config (with extra_ldpath applied) and warn if
    # any .so is missing, BEFORE launching thousands of runs that would all fail
    # with code 127 (e.g. libc++.so.1 / libomp.so for the llvm builds).
    import shutil as _shutil
    if _shutil.which("ldd"):
        seen_configs = set()
        libpb = False
        for (cfg, var), exe in sorted({**perf_exes, **check_exes}.items()):
            if cfg in seen_configs:
                continue
            seen_configs.add(cfg)
            env = {**os.environ}
            if extra_ldpath:
                prev = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = extra_ldpath + (":" + prev if prev else "")
            try:
                r = subprocess.run(["ldd", exe], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, timeout=30)
                miss = [ln.strip() for ln in r.stdout.splitlines() if "not found" in ln]
                if miss:
                    libpb = True
                    print(f"   ❌ LINKAGE [{short_config_label(cfg)}]: "
                          f"{len(miss)} missing lib(s):")
                    for m in miss:
                        print(f"        {m}")
            except Exception:
                pass
        if libpb:
            print("\n⚠️  Some shared libraries are missing (above).")
            print("   → load the matching module BEFORE running, or pass")
            print("     --ld-library-path <dir containing the .so>, or --auto-libpath")
            print("     (after loading the llvm module). The clean fix is to build")
            print("     with a proper RPATH so binaries are self-contained.")
            print("   The affected runs will fail with code 127 if you continue.\n")

    # --- Perf repetition policy ------------------------------------------
    # 1. The baseline (vector): a single run (no need to average).
    # 2. --full-reps-threads: if given, only those thread counts get --reps_perf
    #    repetitions (averaged measurement); all other thread counts get a single
    #    run (scalability curve). If not given, every thread count gets
    #    --reps_perf.
    full_reps_tc = set(args.full_reps_threads) if args.full_reps_threads else None

    light_set = set(args.light_variants or [])

    def _is_light(var, cfg):
        """A (var, cfg) pair runs as a single smoke test rather than the full
        sweep when the variant is globally light (--light-variants)."""
        return var in light_set

    def threads_for(var, cfg):
        """Thread counts to sweep for (var, cfg). Light pairs run only on the
        single light_threads count (a smoke test), never the full sweep."""
        if _is_light(var, cfg):
            return [args.light_threads]
        # A build without OpenMP ignores OMP_NUM_THREADS, so sweeping it just
        # runs the SAME measurement N times. It showed: in the aarch64 fast
        # campaign, gnu-noomp vector came out at 109.154 / 108.772 / 109.094 s
        # for 16 / 64 / 128 threads, and the two noomp configs alone took 3h29
        # of a 5h run -- two thirds of it for nothing. Keep one count, preferring
        # the reference one so the noomp rows line up with the OMP rows.
        if not is_omp_config(cfg):
            if full_reps_tc:
                for tc in args.threads:
                    if tc in full_reps_tc:
                        return [tc]
            return [args.threads[0]]
        return args.threads

    def reps_perf_for(var, tc, cfg):
        if _is_light(var, cfg):
            return 1                       # light: one smoke run only
        # Non-OMP builds are single-threaded and deterministic: repeating them
        # yields identical timings, so one run is enough (and saves a lot of
        # wall time — a noomp chunk run can be minutes). OMP configs keep the
        # full repetition count (scheduling / NUMA / contention add real noise).
        if not is_omp_config(cfg):
            return 1
        if var == args.baseline_variant:
            return 1
        if full_reps_tc is None:
            return args.reps_perf
        return args.reps_perf if tc in full_reps_tc else 1

    if full_reps_tc is not None:
        missing = full_reps_tc - set(args.threads)
        if missing:
            print(f"⚠️  --full-reps-threads {sorted(missing)} not in --threads "
                  f"{args.threads} → those values will never be run.")
        print(f"ℹ️  {args.reps_perf} runs on thread count(s) {sorted(full_reps_tc)}; "
              f"1 run on the others (scalability).")

    # The baseline check (vector, uncompressed) ONLY serves to generate the
    # reference files data_tab* (an expensive output). One oracle PER CONFIG
    # (see oracle_dir: the reference is not portable across compilers), each
    # generated once by the config's own vector binary and kept across runs:
    #   - only if the config has check binaries to compare (no oracle for a
    #     config that would not read it),
    #   - and only if its oracle is absent or --regen-ref says so.
    import glob as _glob
    _MARKER = "data_tab_ORACLE_COMPLETE.marker"

    def _oracle_files(d):
        return [f for f in _glob.glob(os.path.join(d, "data_tab*")) if not f.endswith(".marker")]

    def _purge_oracle(d):
        n = 0
        for f in _glob.glob(os.path.join(d, "data_tab*")):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
        return n

    # A single oracle from before Sept '26 sits in output_dir itself; nothing
    # reads it any more, say so once rather than silently keeping it around.
    if not args.skip_check and os.path.isfile(os.path.join(args.output_dir, _MARKER)):
        print(f"ℹ️  A legacy single oracle sits in {args.output_dir} (data_tab*): it is no "
              f"longer used -- each config now has its own under oracle/<config>/. "
              f"Delete it when convenient.")

    # config -> True when its baseline must run to (re)generate the oracle.
    oracle_needed = {}
    configs_checked = sorted({cfg for (cfg, _v) in check_exes})
    for cfg in configs_checked if not args.skip_check else []:
        d = oracle_dir(args.output_dir, cfg)
        marker = os.path.join(d, _MARKER)
        files = _oracle_files(d)
        # The oracle counts as present ONLY if the completeness marker exists
        # (written by the baseline AFTER a successful full generation): files
        # without it are an interrupted generation, purged and regenerated.
        present = os.path.isfile(marker) and bool(files)
        if files and not present:
            _purge_oracle(d)
            print(f"⚠️  Partial oracle for {short_config_label(cfg)} in {d} "
                  f"({len(files)} file(s), no completeness marker) → purged, regenerating.")
        if args.regen_ref and (present or files):
            n = _purge_oracle(d)
            print(f"ℹ️  --regen-ref: removed {n} file(s) in {d} → regenerating.")
            present = False
        if present:
            print(f"ℹ️  Oracle for {short_config_label(cfg)} present in {d} → "
                  f"'{args.baseline_variant}' run SKIPPED.")
            oracle_needed[cfg] = False
        elif cfg in baseline_check_exe_of:
            print(f"ℹ️  No oracle for {short_config_label(cfg)} → "
                  f"'{args.baseline_variant}' of that config run once to generate it in {d}.")
            oracle_needed[cfg] = True
        else:
            print(f"⚠️  No oracle for {short_config_label(cfg)} AND no "
                  f"'{args.baseline_variant}' check binary in that config → its CHECK "
                  f"runs will abort on missing references. Build the vector variant there.")
            oracle_needed[cfg] = False
    run_baseline_check = any(oracle_needed.values())

    # --- Totals (accounting for the policies above) ----------------------
    total_perf_runs = 0
    if not args.skip_perf:
        for (cfg, var) in perf_exes:
            if skip_variant_on_config(var, cfg):
                continue
            for tc in threads_for(var, cfg):
                total_perf_runs += reps_perf_for(var, tc, cfg)

    def reps_check_for(cfg):
        # Non-OMP builds are single-threaded and deterministic: one CHECK run is
        # enough (repeating gives identical results). OMP configs keep the full
        # reps_check count, where scheduling/contention make repetition useful.
        if not is_omp_config(cfg):
            return 1
        return args.reps_check

    total_check_runs = 0
    if not args.skip_check:
        total_check_runs = sum(
            reps_check_for(cfg) for (cfg, _var) in check_exes.keys()
        )
        total_check_runs += sum(1 for v in oracle_needed.values() if v)

    # STL self-check: one run per gnu/llvm front-end present (phase 2c).
    # One STL self-check per compiler front-end (gnu/llvm). The check is
    # functional and config-agnostic, so prefer the plain config name but fall
    # back to ANY config of that front-end that has a chunk2a/chunk check exe —
    # otherwise a build set with only suffixed configs (e.g. -std-alloc) reports
    # 0 STL runs and silently skips the check.
    def _stl_exe_for(config):
        """A chunk2a (else chunk) check binary of `config`, whatever its size.

        FIX: this used to be `check_exes.get((config, "chunk2a"))`, with a BARE
        variant name. check_exes is keyed by the EXPANDED variant -- "chunk2a@dyn",
        "chunk@256" -- ever since the size declinations arrived, so the lookup
        never matched, `reps` came out empty, the fallback below used the same
        bare keys and missed too, and the run reported "0 STL" while silently
        skipping the phase. The check is functional and size-agnostic, so any
        declination will do; chunk2a first because it exercises the widest surface.
        """
        for base in ("chunk2a", "chunk"):
            for (c, var), exe in check_exes.items():
                if c == config and (var == base or var.startswith(base + "@")):
                    return exe
        return None

    def _stl_representative_exes():
        reps = {}
        for c in args.configs:
            if "omp-gnu" in c:
                fe = "gnu"
            elif "omp-llvm" in c:
                fe = "llvm"
            else:
                continue
            exe = _stl_exe_for(c)
            if not exe:
                continue
            is_plain = c in ("cmake-build-omp-gnu", "cmake-build-omp-llvm")
            if fe not in reps or is_plain:
                reps[fe] = (c, exe)
        # Fallback: if no gnu/llvm-named config is built (e.g. fast mode on an arch
        # where only fast_alloc configs are present, whose dir names don't carry
        # the omp-gnu/omp-llvm marker), the STL self-check would silently be
        # skipped (0 STL). It is a functional, config-agnostic test, so run it once
        # on the first present config that has a chunk2a/chunk check exe.
        if not reps:
            for c in args.configs:
                exe = _stl_exe_for(c)
                if exe:
                    reps["any"] = (c, exe)
                    break
        return list(reps.values())

    total_stl_runs = 0
    if not args.skip_check and not args.skip_stl_test:
        total_stl_runs = len(_stl_representative_exes())

    # Sanitizer CHECK runs (phase 2d): SANITIZER_REPS per present (config,variant).
    total_san_runs = 0
    if not args.skip_check and not args.skip_sanitizers:
        for cfg in SANITIZER_CONFIGS:
            if not config_dir_present(args.build_root, cfg):
                continue
            for var in args.variants:
                if skip_variant_on_config(var, cfg):
                    continue
                if resolve_exe(args.build_root, cfg, var, "check"):
                    total_san_runs += SANITIZER_REPS

    overall_runs = (total_perf_runs + total_check_runs
                    + total_stl_runs + total_san_runs)
    current_run_idx = 0

    print(bold(f"\n🚀 Starting: {cyan(total_perf_runs)} perf runs + "
               f"{cyan(total_check_runs)} check runs + "
               f"{cyan(total_stl_runs)} STL + "
               f"{cyan(total_san_runs)} sanitizer = {cyan(overall_runs)} total."))

    results = {}
    errors  = []
    global_start_time = time.time()
    interrupted = False

    try:
        # --- Phase 0b: validation harnesses (ctest) ----------------------
        # First on purpose: quick, and a failure here reads better than the
        # same fault surfacing as a bench crash three phases later.
        if not args.skip_ctest:
            run_ctest_phase(args, extra_ldpath, errors)

        # --- Phase 1: Correctness ---------------------------------------
        if not args.skip_check:
            print(cyan(bold(f"\n🔍 Correctness check — {args.check_threads} threads")))

            # 2a. Generate the reference of every config that needs one, with
            #     the config's own vector baseline, uncompressed, in its
            #     oracle directory (one run each).
            for cfg_b in configs_checked:
                if not oracle_needed.get(cfg_b):
                    continue
                exe_b = baseline_check_exe_of[cfg_b]
                odir = oracle_dir(args.output_dir, cfg_b)
                os.makedirs(odir, exist_ok=True)
                current_run_idx += 1
                sys.stdout.write(
                    f"\r{get_progress_bar(current_run_idx, overall_runs)} "
                    f"CHECK {short_config_label(cfg_b)[:30]:30} | "
                    f"{args.baseline_variant:10} | uncompressed (ref data_tab*) "
                )
                sys.stdout.flush()
                try:
                    os.remove(os.path.join(odir, _MARKER))
                except OSError:
                    pass
                ok, _, code, err_tail, _ = run_cmd(
                    [exe_b],
                    {"OMP_NUM_THREADS": str(args.check_threads)},
                    effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads),
                    extra_ld_library_path=extra_ldpath, cwd=odir
                )
                if ok:
                    # Completeness marker: the gate above only trusts the oracle
                    # when this file exists, so an interrupted generation can
                    # never be mistaken for a complete one.
                    with open(os.path.join(odir, _MARKER), "w") as _mf:
                        _mf.write(f"generated by {exe_b} (uncompressed), "
                                  f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                    sys.stdout.write(" -> ✅ reference generated\n")
                else:
                    sys.stdout.write(f" -> ❌ failed (code {code})\n")
                    _entry = (
                        f"BASELINE_REF_FAIL: {exe_b} code={code}\n"
                        f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    ')
                    )
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                sys.stdout.flush()

            # 2b. Checks for the non-baseline variants, each in its config's
            #     oracle directory.
            for cfg in args.configs:
                for var in args.variants:
                    exe = check_exes.get((cfg, var))
                    if not exe:
                        continue

                    reps_check_this = reps_check_for(cfg)
                    success_in_row = 0
                    for r in range(reps_check_this):
                        current_run_idx += 1
                        sys.stdout.write(
                            f"\r{get_progress_bar(current_run_idx, overall_runs)} "
                            + _prog("CHECK", cyan, cfg, var,
                                    th=args.check_threads, run=r + 1,
                                    reps=reps_check_this,
                                    extra=f"OK:{success_in_row:3} ")
                        )
                        sys.stdout.flush()

                        ok, _, code, err_tail, _ = run_cmd(
                            [exe], {"OMP_NUM_THREADS": str(args.check_threads)},
                    effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads),
                            extra_ld_library_path=extra_ldpath, cwd=oracle_dir(args.output_dir, cfg)
                        )
                        if ok:
                            success_in_row += 1
                        else:
                            _entry = (
                                f"CHECK_FAIL: {exe} iter={r+1} code={code}\n"
                                f"  stderr_tail:\n    "
                                + err_tail.replace('\n', '\n    ')
                            )
                            errors.append(_entry)
                            _append_error_now(args.errors_log, _entry)
                    verdict = f"{success_in_row}/{reps_check_this}"
                    if success_in_row == reps_check_this:
                        tag = green(f"{verdict} Pass")
                    else:
                        tag = red(f"{verdict} FAIL")
                    sys.stdout.write(f" -> {tag}\n")
                    sys.stdout.flush()

            # 2c. STL interface self-checks: functional, thread-independent, so
            #     run ONCE per (compiler) front-end, not once per matrix cell.
            #     Uses --test_stl --no_io on the check binary; pass/fail read
            #     from the run's exit code (the section prints ALL PASSED / N
            #     FAILURE(S) and exits non-zero on failure).
            if not args.skip_stl_test:
                for cfg, exe in _stl_representative_exes():
                    current_run_idx += 1
                    sys.stdout.write(
                        f"\r{get_progress_bar(current_run_idx, overall_runs)} "
                        + _prog("STL", cyan, cfg,
                                th=SANITIZER_THREADS,
                                extra="--test_stl --no_io ")
                    )
                    sys.stdout.flush()
                    ok, _, code, err_tail, _ = run_cmd(
                        [exe, "--test_stl", "--no_io"],
                        {"OMP_NUM_THREADS": str(SANITIZER_THREADS)},
                        effective_timeout(args.timeout, args.timeout_per_thread, SANITIZER_THREADS),
                        extra_ld_library_path=extra_ldpath, cwd=args.output_dir
                    )
                    if ok:
                        sys.stdout.write(" -> " + green("PASS") + "\n")
                    else:
                        sys.stdout.write(" -> " + red(f"FAIL (code {code})") + "\n")
                        _entry = (f"STL_TEST_FAIL: {exe} --test_stl --no_io code={code}\n"
                                  f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                    sys.stdout.flush()

            # 2d. Sanitizer CHECK: ASan (std-alloc) + TSan, at a fixed low thread
            #     count, SANITIZER_REPS run(s). Diagnostic, not timed. A run FAILS on a
            #     non-zero exit OR any sanitizer report in the output (run_cmd
            #     detects the latter and flips success to False).
            if not args.skip_sanitizers:
                for cfg in SANITIZER_CONFIGS:
                    if not config_dir_present(args.build_root, cfg):
                        continue
                    san_env = sanitizer_env_for(cfg)
                    for var in args.variants:
                        if skip_variant_on_config(var, cfg):
                            continue
                        # The baseline (vector) holds no chunk code at all, so a
                        # sanitizer run on it can only ever exercise libstdc++.
                        # At 5-20x slowdown x SANITIZER_REPS that is pure cost
                        # for zero signal -- skip it.
                        if _variant_base(var) == args.baseline_variant:
                            continue
                        exe = resolve_exe(args.build_root, cfg, var, "check")
                        if not exe:
                            continue
                        ok_in_row = 0
                        for r in range(SANITIZER_REPS):
                            current_run_idx += 1
                            sys.stdout.write(
                                f"\r{get_progress_bar(current_run_idx, overall_runs)} "
                                + _prog("SAN", cyan, cfg, var,
                                        th=SANITIZER_THREADS, run=r + 1,
                                        reps=SANITIZER_REPS,
                                        extra=f"OK:{ok_in_row:3} ")
                            )
                            sys.stdout.flush()
                            ok, _, code, err_tail, _ = run_cmd(
                                [exe, "--no_io"],
                                {"OMP_NUM_THREADS": str(SANITIZER_THREADS), **san_env},
                                effective_timeout(args.timeout, args.timeout_per_thread,
                                                  SANITIZER_THREADS),
                                extra_ld_library_path=extra_ldpath, cwd=args.output_dir
                            )
                            if ok:
                                ok_in_row += 1
                            else:
                                _entry = (
                                    f"SANITIZER_FAIL: {exe} --no_io "
                                    f"({'TSan' if 'thread' in cfg else 'ASan'}) "
                                    f"iter={r+1} code={code} threads={SANITIZER_THREADS}\n"
                                    f"  report:\n    " + err_tail.replace('\n', '\n    ')
                                )
                                errors.append(_entry)
                                _append_error_now(args.errors_log, _entry)
                        verdict = f"{ok_in_row}/{SANITIZER_REPS}"
                        tag = green(f"{verdict} clean") if ok_in_row == SANITIZER_REPS \
                            else red(f"{verdict} HIT")
                        sys.stdout.write(f" -> {tag}\n")
                        sys.stdout.flush()

        # --- Phase 1b: seq-vs-omp compression-classification compare ----
        # A correctness check, so it sits with the others, BEFORE the perf
        # sweep: a classification that differs between the sequential and the
        # OpenMP build is a bug to look at now, not after an hour of timings.
        # Reads CHUNK_STATS, so it only bites on a -stats pair.
        if not args.skip_seq_omp_compare:
            run_seq_omp_compare_phase(args, extra_ldpath, errors)

        # --- Phase 2: Performance ---------------------------------------
        if not args.skip_perf:
            print(cyan(bold(f"\n🚀 Performance — threads {args.threads}")))
            baseline_times = {}
            baseline_mems  = {}
            for cfg in args.configs:
                # Baseline per (variant, tc): time of the baseline_variant for
                # this tc, on this config. A compression-only config (e.g.
                # -have-agressive-compression) does NOT re-run vector — its
                # binary is identical to the base config — so it inherits the
                # baseline captured on the preceding base config instead of
                # resetting it.
                if not is_compression_only_config(cfg):
                    baseline_times = {}
                    baseline_mems  = {}
                for var in args.variants:
                    if skip_variant_on_config(var, cfg):
                        continue
                    exe = perf_exes.get((cfg, var))
                    if not exe:
                        continue

                    for tc in threads_for(var, cfg):
                        run_times = []
                        run_mems  = []
                        _reps = reps_perf_for(var, tc, cfg)
                        _timeout = effective_timeout(
                            args.timeout, args.timeout_per_thread, tc)
                        for r in range(_reps):
                            current_run_idx += 1

                            elapsed = time.time() - global_start_time
                            eta_min = ((elapsed / current_run_idx) *
                                       (overall_runs - current_run_idx) / 60)
                            sys.stdout.write(
                                f"\r{get_progress_bar(current_run_idx, overall_runs)} "
                                + _prog("PERF", yellow, cfg, var,
                                        th=tc, run=r + 1, reps=_reps,
                                        extra=dim(f"ETA:{eta_min:5.1f}min") + "  ")
                            )
                            sys.stdout.flush()

                            ok, dur, code, err_tail, mem_mo, exe = run_with_rebuild_retry(
                                args.build_root, cfg, var, "perf", exe,
                                {"OMP_NUM_THREADS": str(tc)}, _timeout,
                                extra_ldpath,
                                args.rebuild_retries, args.rebuild_wait,
                                cwd=args.output_dir
                            )
                            if ok:
                                run_times.append(dur)
                                if mem_mo is not None:
                                    run_mems.append(mem_mo)
                            else:
                                _entry = (
                                    f"PERF_ERR: {exe} @ {tc}th run={r+1} code={code}\n"
                                    f"  stderr_tail:\n    "
                                    + err_tail.replace('\n', '\n    ')
                                )
                                errors.append(_entry)
                                _append_error_now(args.errors_log, _entry)

                        if run_times:
                            avg = sum(run_times) / len(run_times)
                            peak_mem = max(run_mems) if run_mems else None
                            results[(cfg, var, tc)] = (avg, peak_mem)
                            mem_str = (f" | {peak_mem:7.0f} Mo"
                                       if peak_mem is not None else " |          ")

                            if var == args.baseline_variant:
                                baseline_times[tc] = avg
                                if peak_mem is not None:
                                    baseline_mems[tc] = peak_mem
                                sys.stdout.write(
                                    f" -> Avg: {avg:8.3f}s{mem_str} (baseline)\n")
                            else:
                                # Time speedup vs vector (>1 = faster).
                                if tc in baseline_times:
                                    speedup = baseline_times[tc] / avg
                                    _sp = f"speedup: {speedup:.2f}x"
                                    speed_str = green(_sp) if speedup >= 1.0 else red(_sp)
                                else:
                                    speed_str = None
                                # Memory ratio vs vector (>1 = uses LESS memory =
                                # compression gain). Also show the absolute saving.
                                if peak_mem is not None and tc in baseline_mems \
                                        and peak_mem > 0:
                                    base_mem = baseline_mems[tc]
                                    mem_ratio = base_mem / peak_mem
                                    mem_saved = base_mem - peak_mem
                                    # Wording matters here: mem_ratio is
                                    # base/peak, so 2.89x means the variant uses
                                    # 2.89x LESS memory. The old label read
                                    # "mem: 2.89x (+56772 Mo vs vector)", whose
                                    # "x" and leading "+" both suggest the exact
                                    # opposite. Say less/more explicitly and sign
                                    # the delta by what actually happened.
                                    if mem_ratio >= 1.0:
                                        _mem = (
                                            f"mem: {mem_ratio:.2f}x less "
                                            f"(-{mem_saved:.0f} Mo vs {args.baseline_variant})"
                                        )
                                    else:
                                        _mem = (
                                            f"mem: {1.0 / mem_ratio:.2f}x more "
                                            f"(+{-mem_saved:.0f} Mo vs {args.baseline_variant})"
                                        )
                                    mem_gain_str = green(_mem) if mem_ratio >= 1.0 else yellow(_mem)
                                else:
                                    mem_gain_str = None

                                parts = [p for p in (speed_str, mem_gain_str) if p]
                                suffix = f" ({' | '.join(parts)})" if parts else ""
                                sys.stdout.write(
                                    f" -> Avg: {bold(f'{avg:8.3f}s')}{mem_str}{suffix}\n")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(f" -> {red(f'⚠️  no valid run out of {_reps}')}\n")
                            sys.stdout.flush()

        # --- Phase 3: adaptive re-chunk feature test (dyn) --------------
        if not args.skip_rechunk:
            run_rechunk_phase(args, extra_ldpath, errors)

        # --- Phase 3b: 64-bit index checks (int64 binaries) --------------
        # Needs the oracles of the correctness phase, like the rechunk check.
        if not args.skip_check and not args.skip_int64_check:
            run_int64_check_phase(args, extra_ldpath, errors)

        # --- Phase 4: scalability guard (omp-llvm, omp-gnu) -------------
        # After the perf sweep: it reads the same binaries, and a failure here
        # is about the SHAPE of the curve, which only means something once the
        # timings above have been produced without incident.
        if not args.skip_scaling and not args.skip_perf:
            run_scaling_phase(args, extra_ldpath, errors)

        # --- Phase 5: randomised parameter sweep (opt-in) ---------------
        # Last on purpose: it is the only phase that does NOT repeat the same
        # trajectory, so a failure here is new information rather than a
        # re-confirmation, and running it after the deterministic phases keeps
        # their timings clean.
        run_fuzz_phase(args, extra_ldpath, errors)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⛔ User interrupt — writing partial results...")

    total_min = (time.time() - global_start_time) / 60
    flush_summary(results, errors, args.summary_csv, args.errors_log,
                  total_min, interrupted, baseline_variant=args.baseline_variant,
                  mem_chunk2a_vs_chunk2_max=mem_c2a_vs_c2,
                  mem_vs_baseline_max=args.mem_vs_baseline_max)

    sys.exit(1 if errors else (130 if interrupted else 0))


if __name__ == "__main__":
    main()
