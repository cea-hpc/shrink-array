# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Defaults, presets and the machine fitting of the thread counts."""
import os


# --- Default configuration -----------------------------------------------
# "Configs" = sub-directories of $build_root. The first entry is used as the
# baseline for the speedup computation.
# Config-name substrings that only change chunk* code paths (compression
# scheduling internals) and leave the vector binary byte-for-byte identical to
# the base config. On such configs we skip the compression-insensitive variants
# (see COMPRESSION_INSENSITIVE_VARIANTS) to avoid re-timing identical binaries.
COMPRESSION_ONLY_CONFIG_MARKERS = ["have-aggressive-compression"]

# Configurations that are a RUN-TIME mode of another build, not a build of their
# own: the suffix of the name, and the environment of their runs. Their binaries
# are the base build's (builds.physical_config). -have-aggressive-compression:
# the base build with CHUNK_AGGRESSIVE_COMPRESSION=1 (chunk/config.h), a build of
# its own until the switch became a run-time one; a directory of that name still
# on disk (an old build) is used as it is.
VIRTUAL_CONFIG_ENV = {
    "-have-aggressive-compression": {"CHUNK_AGGRESSIVE_COMPRESSION": "1"},
}


def virtual_base(config):
    """The build a run-time-mode configuration runs, or None for a real one."""
    for suffix in VIRTUAL_CONFIG_ENV:
        if config.endswith(suffix):
            return config[: -len(suffix)]
    return None


def config_env(config):
    """The environment every run of `config` gets: its mode's (empty for a real build)."""
    for suffix, env in VIRTUAL_CONFIG_ENV.items():
        if config.endswith(suffix):
            return dict(env)
    return {}

# Variants whose binary does not depend on the aggressive-compression knob, so
# they never need to be run on a COMPRESSION_ONLY config. vector is a plain
# non-chunk path, identical to the base config there. (std_alloc is handled
# separately: it's only built in -std-alloc configs, see is_std_alloc_config.)
COMPRESSION_INSENSITIVE_VARIANTS = ["vector"]

DEFAULT_CONFIGS = [
    "cmake-build-omp-gnu",                             # release GNU (baseline)
    "cmake-build-omp-gnu-have-aggressive-compression",  # aggressive scheduling
    "cmake-build-omp-gnu-std-alloc",                   # std allocator
    "cmake-build-omp-llvm",                            # release LLVM
    "cmake-build-omp-llvm-have-aggressive-compression",
    "cmake-build-omp-llvm-std-alloc",
    # The std backend (CHUNK_THREADS=std: the library's teams on std::jthread
    # workers of its own, no OpenMP; OMP_NUM_THREADS still sets the team size).
    # The release builds carry the TBB bench too (tbb_* variants below) when
    # oneTBB was there at compile time. The -check builds (CHECK_OPTIM) run the
    # validation harnesses of that backend -- std_thread_validation,
    # tbb_validation -- and the check runs; never timed (PERF_EXCLUDED_CONFIGS).
    "cmake-build-std-gnu",                             # release GNU, std backend
    "cmake-build-std-llvm",                            # release LLVM, std backend
    "cmake-build-std-gnu-check",                       # CHECK_OPTIM, std backend
    "cmake-build-std-llvm-check",
    # x86-64 only: the other two OpenMP compilers compile_all.sh builds there
    # (both OMPT builds). On ARM their directories do not exist and the
    # absent-config step drops them with one note.
    "cmake-build-omp-oneapi",                          # release oneAPI (icpx)
    "cmake-build-omp-aocc",                            # RelWithDebInfo AOCC (AMD clang)
    "cmake-build-std-oneapi",                          # oneAPI, std backend (+ TBB bench)
    "cmake-build-std-aocc",                            # AOCC, std backend (+ TBB bench)
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

# The TBB bench: the same sources with the regions run as TBB tasks
# (shrink_bench_tbb_<variant>_<size>, examples/shrink_bench/CMakeLists.txt). It
# exists only in the std-backend builds made with CHUNK_WITH_TBB, so these
# variants are looked for there and nowhere else (builds.skip_variant_on_config).
TBB_VARIANTS = ["tbb_zero", "tbb_zero_one_value", "tbb_zero_any"]
SHRINK_VARIANTS = ["zero", "zero_one_value", "zero_any"]   # the three array variants
DEFAULT_VARIANTS       = ["vector"] + SHRINK_VARIANTS + ["std_alloc"] + TBB_VARIANTS


def is_tbb_variant(var):
    """True for a TBB bench variant ("tbb_zero_any", "tbb_zero_any@256")."""
    return var.split("@", 1)[0] in TBB_VARIANTS


# Every shrink_bench executable carries the threading backend of its build in
# its name (examples/shrink_bench/CMakeLists.txt): shrink_bench_<backend>_<rest>,
# <backend> one of these, <rest> zero_any_256, zero_int64_dyn, vector_check,
# std_alloc_1024, ... The build directory's name does not tell the backend
# (cmake-build-gnu-stats is serial, cmake-build-omp-gnu-stats OpenMP, and a
# hand-made one says nothing), but a build directory holds ONE backend: so an
# executable is found by trying the three names, the one built there being the
# only one present. The TBB bench is the exception, its "tbb" naming what runs
# its regions: shrink_bench_tbb_<variant>_<size>, one name.
#
# No fallback to the unprefixed names the executables had before: a build
# directory rebuilt since still holds those binaries, stale, and they must not
# be run in place of the new ones.
BENCH_BACKENDS = ("omp", "std", "serial")


def bench_exe_names(rest):
    """The names the executable `rest` (its name after "shrink_bench_", without
    the backend: "zero_any_256", "vector_check", "tbb_zero_any_dyn") may have
    in a build directory, in the order they are tried. Python 3.6: the
    standalone scripts that copy it (thread_speedup.py, multi_launch_stress.py)
    run there."""
    if rest.startswith("tbb_"):
        return ["shrink_bench_" + rest]
    return ["shrink_bench_%s_%s" % (b, rest) for b in BENCH_BACKENDS]


def find_bench_exe(directory, rest):
    """The path of the executable `rest` (bench_exe_names) in `directory`, the
    first of its names present there as an executable file, or None."""
    for name in bench_exe_names(rest):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# The variants were named chunk / chunk2 / chunk2a until they took the name of
# their namespace. Command lines, perf databases, bench_summary.csv files, JSON
# logs and CHUNK_STATS tags of older builds still carry the old labels: they
# are read as their new names until 1.0.
LEGACY_VARIANTS = {
    "chunk": "zero", "chunk2": "zero_one_value", "chunk2a": "zero_any",
    "tbb_chunk": "tbb_zero", "tbb_chunk2": "tbb_zero_one_value",
    "tbb_chunk2a": "tbb_zero_any",
}


def normalize_variant(label, warn=None):
    """The current name of a variant label: an old one (chunk2a, chunk2a@256,
    chunk2a_int64@dyn, tbb_chunk2) becomes its new name (zero_any, zero_any@256,
    zero_any_int64@dyn, tbb_zero_one_value); anything else comes back unchanged.
    `warn`: a context string -- when given and the label is an old one, a
    one-line deprecation note goes to stderr."""
    if not isinstance(label, str):
        return label
    base, at, size = label.partition("@")
    suffix = ""
    if base.endswith("_int64"):
        base, suffix = base[:-len("_int64")], "_int64"
    new = LEGACY_VARIANTS.get(base)
    if new is None:
        return label
    out = new + suffix + at + size
    if warn:
        import sys
        print(f"note: {warn}: variant '{label}' is the old name of '{out}' "
              "(accepted until 1.0)", file=sys.stderr)
    return out

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
    "cmake-build-omp-llvm-debug-std-alloc-address-sanitizer",  # ASan + std-alloc (sees what the arenas hide)
    "cmake-build-omp-llvm-debug-thread-sanitizer",             # TSan (ARM race net)
    # TSan on the std backend, optimised, CHECK_OPTIM: standard C++ all the way
    # down, so TSan sees every synchronisation of the library's teams (libomp's
    # annotations are not needed). Its TBB bench only exists when the build was
    # given an instrumented oneTBB (CHUNK_TBB_TSAN_DIR); the phase skips what is
    # not built.
    "cmake-build-std-llvm-relwithdebinfo-check-thread-sanitizer",
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


def _legacy_va_layout():
    """True when the node forces the legacy (bottom-up, 0x2aaa...) mmap layout:
    vm.legacy_va_layout = 1. Unreadable or absent counts as False."""
    try:
        with open("/proc/sys/vm/legacy_va_layout") as fh:
            return fh.read().strip() not in ("", "0")
    except OSError:
        return False


def sanitizer_skip_reason(config):
    """Why `config` cannot run on this node, or None when it can.

    TSan needs the standard top-down layout to place its shadow memory. With
    vm.legacy_va_layout = 1 it stops at startup ("memory layout is
    incompatible, even though ASLR is disabled") whatever the stack limit,
    and a user can neither change the sysctl nor the process personality
    (setarch is refused). Every run would be a FAIL that says nothing about
    the code."""
    if "thread-sanitizer" in config and _legacy_va_layout():
        return ("vm.legacy_va_layout = 1 on this node: ThreadSanitizer cannot "
                "start here (memory layout is incompatible) -- run TSan on "
                "another node")
    return None

# Configs to DROP from the PERF phase (kept in CHECK). llvm-std-alloc perf is
# redundant with gnu-std-alloc (near-identical numbers) and gnu is the perf/
# correctness judge; cutting it reclaims cluster time. Sanitizer configs are
# never timed.
PERF_EXCLUDED_CONFIGS = [
    "cmake-build-omp-llvm-std-alloc",
    # CHECK_OPTIM builds of the std backend: there for the checks and the
    # validation harnesses; the release std builds carry the timings.
    "cmake-build-std-gnu-check",
    "cmake-build-std-llvm-check",
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
DEFAULT_SEQ_OMP_VARIANTS = list(SHRINK_VARIANTS)
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
#     ~30 s zero_any run on the master, 63 threads at the barrier -- which
#     neither the OMPT builds (LLVM, oneAPI, AOCC) nor this guard on LLVM alone
#     could see.
SCALING_CONFIGS  = ["cmake-build-omp-llvm", "cmake-build-omp-gnu"]
# The curve every container is read against: no chunk, no compression, so a
# bend it shares is the machine's, and one only the chunk variants show is ours.
SCALING_REFERENCE_VARIANT = "vector"
SCALING_THREADS  = [4, 8, 16, 32, 64, 128]
# The guard runs the int64 binaries (shrink_bench_<backend>_<variant>_int64_<size>, see
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
# The guard's runs never go through numactl --interleave, whatever
# --numa-interleave says: interleaving puts half of each thread's slabs on the
# other socket, where first touch keeps them local. Grace, zero@dyn int64
# --multiply 2, 128 threads (2 sockets), best of 3 interleaved rounds: llvm
# 14.57 s interleaved / 13.02 s not, gnu 16.49 / 14.68 (-11%); 64 threads
# (one socket) unchanged. With it, the 128-thread point failed the guard
# (1.34x the 64-thread one on gnu) on a cost the library does not choose.
SCALING_NUMA_INTERLEAVE = "off"

# --- 64-bit index checks (int64 binaries) ----------------------------------
# A few --mode check runs of the int64 binaries against the 32-bit oracle of
# the same config: the data_tab* files hold element values, independent of the
# index type, so the int64 build must reproduce them exactly. One per variant,
# both geometries between them; zero_any@dyn is the richest path (a second
# value per chunk, re-chunking). Not a sweep: the 32-bit checks cover the
# rest, and this only asks whether 64-bit indices change a result.
INT64_CHECK_VARIANTS = ["zero@1024", "zero_one_value@1024", "zero_any@dyn"]

# --- Mode presets --------------------------------------------------------
# Each preset overrides threads / reps_perf / full_reps_threads unless the user
# passes those flags explicitly. Vary both the thread distribution and the
# number of repetitions.
# The scalability guard (run_scaling_phase) is NOT part of these presets: it
# runs in every mode, on its own thread list, because a mode that trims the
# perf sweep is exactly the one that would otherwise miss a scaling
# regression. --skip-scaling is the way out.
#   fast   : about 1 h on the full aarch64 matrix. Is anything broken? The
#            release builds (gnu, llvm, oneAPI/AOCC on x86) at 64 threads, 2
#            runs each; 4 check runs per variant; no sanitizers; a short
#            scalability guard (llvm, dyn, 16/64/128 threads).
#   normal : about 6 h. The matrix: 16/64/128/288 threads, 3 runs at 64,
#            16 check runs, the sanitizers, the guard on gnu and llvm from 16
#            threads.
#   full   : about 24 h. Every thread count, 10 runs at 64 and 128, 32 check
#            runs, the guard from 4 threads.
# The durations are targets, set from the run counts; the per-phase times of
# the summary (Summary tab of shrink_gui, events.jsonl) are what to adjust them
# with.
#
# Keys besides the perf/check ones (an option given on the command line wins):
#   skip           phases the mode leaves out ("sanitizers")
#   perf_drop      config name markers whose PERF phase the mode drops (kept
#                  in CHECK), on top of PERF_EXCLUDED_CONFIGS
#   scaling        the scalability guard: configs, threads, sizes (None = all)
#   seq_omp        the seq-vs-omp compare: threads, sizes
MODE_PRESETS = {
    "fast": {
        # One point, the reference: the curve is normal's and full's job, and
        # the 16-thread point was the most expensive run of the sweep (on
        # aarch64, zero_any@dyn 100.3 s at 16 threads against 70.3 s at 64).
        "threads":           [64],
        # 2 runs: enough to see a run that crashed or doubled, not to publish.
        "reps_perf":         2,
        "full_reps_threads": [64],
        "reps_check":        4,
        # Variants run as a single smoke test (one thread count, one run) just
        # to confirm they are not broken — they don't need the full sweep.
        "light_variants":    ["std_alloc"],
        "light_threads":     64,
        "skip":              ["sanitizers"],
        # The allocator moves the numbers (see reduce_matrix), but timing it
        # is a normal-mode question; fast still checks the std-alloc builds.
        "perf_drop":         ["-std-alloc"],
        "scaling": {"configs": ["cmake-build-omp-llvm"], "threads": [16, 64, 128], "sizes": ["dyn"]},
        "seq_omp": {"threads": [8, 64], "sizes": ["dyn"]},
    },
    "normal": {
        "threads":           [16, 64, 128, 288],
        "reps_perf":         3,
        "full_reps_threads": DEFAULT_FULL_REPS_THREADS,
        "reps_check":        16,
        "light_variants":    [],
        "light_threads":     64,
        "skip":              [],
        "perf_drop":         [],
        # From 16 threads: the low points of int64 --multiply 2 are the
        # longest runs of the guard (minutes each); full keeps them.
        "scaling": {"configs": SCALING_CONFIGS, "threads": [16, 32, 64, 128], "sizes": None},
        "seq_omp": {"threads": DEFAULT_SEQ_OMP_THREADS, "sizes": DEFAULT_SEQ_OMP_SIZES},
    },
    "full": {
        "threads":           [1, 2, 4, 8, 16, 32, 64, 128, 192, 288],
        "reps_perf":         10,
        "full_reps_threads": [64, 128],
        "reps_check":        32,
        "light_variants":    [],
        "light_threads":     64,
        "skip":              [],
        "perf_drop":         [],
        "scaling": {"configs": SCALING_CONFIGS, "threads": SCALING_THREADS, "sizes": None},
        "seq_omp": {"threads": DEFAULT_SEQ_OMP_THREADS, "sizes": DEFAULT_SEQ_OMP_SIZES},
    },
}

# --- Fitting the thread counts to the machine ----------------------------
# Every thread list above was written for the aarch64 node (4 x 72 cores, no
# SMT), where 64 threads is the reference point. A 12-core desktop (AMD
# 9900X, 24 hardware threads) cannot run 64, 128 or 288 threads, and an
# oversubscribed run measures the scheduler and spinning barriers, not the
# library. So, unless a flag pins it, every thread count is fitted to the CPUs
# this process may use (its affinity mask: a Slurm allocation or a taskset is
# honoured), capped by --max-threads:
#   - the reference count -- full repetitions, smoke runs -- is
#     min(REFERENCE_THREADS, physical cores);
#   - a list keeps the counts the machine can run; when it had to drop some,
#     it gains the machine's own top points instead: its physical cores and
#     its CPUs (the reference count too, so the full repetitions still have a
#     point to run on). On the 288-core node nothing is dropped: the lists
#     are unchanged.
REFERENCE_THREADS = 64


def machine_cpus():
    """(usable logical CPUs, physical cores among them, sockets among them).

    Usable = this process's affinity mask, not os.cpu_count(): a job confined
    to 32 CPUs of a 288-CPU node must plan for 32. Physical cores are the
    distinct SMT sibling sets (thread_siblings_list), which -- unlike core_id
    -- are unique on every architecture. Without sysfs, every CPU counts as a
    core of a single socket."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = list(range(os.cpu_count() or 1))
    cores, sockets = set(), set()
    for c in cpus:
        topo = f"/sys/devices/system/cpu/cpu{c}/topology/"
        try:
            with open(topo + "thread_siblings_list") as fh:
                cores.add(fh.read().strip())
            with open(topo + "physical_package_id") as fh:
                sockets.add(fh.read().strip())
        except OSError:
            return len(cpus), len(cpus), 1
    return len(cpus), len(cores), max(1, len(sockets))


def fit_threads(wanted, cap, machine_points):
    """`wanted` without the counts above `cap`; when some were dropped, plus
    `machine_points` (those <= cap). Sorted, without duplicates."""
    kept = {t for t in wanted if t <= cap}
    if len(kept) < len(set(wanted)):
        kept |= {t for t in machine_points if 1 <= t <= cap}
    return sorted(kept)


DEFAULT_STATS_MARK_RATIO_MIN = {"zero_one_value": 0.3, "zero_any": 0.3}
