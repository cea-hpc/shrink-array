# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Command line and the campaign itself (bench_and_test.py calls main())."""
import os
import sys
import time
import argparse

from . import events, share, state
from .builds import CURRENT_MACHINE, short_config_label
from .pool import Pool
from .phases.check import run_check_phase
from .phases.ctest import run_ctest_phase
from .phases.fuzz import run_fuzz_phase
from .phases.int64_check import run_int64_check_phase
from .phases.perf import run_perf_phase
from .phases.rechunk import run_rechunk_phase
from .phases.sanitizers import run_sanitizer_phase
from .phases.scaling import run_scaling_phase
from .phases.seq_omp import run_seq_omp_compare_phase
from .phases.stl import run_stl_phase
from .plan import Plan, drop_absent_configs, reduce_matrix, resolve_ldpath, resolve_run_parameters
from .progress import RunCounter
from .report import flush_summary
from .runs import _failed_cases_path, run_rerun_failed
from .settings import (
    DEFAULT_BASELINE, DEFAULT_CONFIGS, DEFAULT_SEQ_OMP_DIVISOR, DEFAULT_SEQ_OMP_ITER_MAX,
    DEFAULT_SEQ_OMP_SEED, DEFAULT_SEQ_OMP_SIZES, DEFAULT_SEQ_OMP_THREADS,
    DEFAULT_SEQ_OMP_VARIANTS, DEFAULT_STATS_MARK_RATIO_MIN, DEFAULT_THREADS_CHECK,
    DEFAULT_TIMEOUT, DEFAULT_TIMEOUT_PER_THREAD, DEFAULT_VARIANTS, INT64_CHECK_VARIANTS,
    MODE_PRESETS, REFERENCE_THREADS, SANITIZER_REPS, SANITIZER_THREADS, SCALING_CONFIGS,
    SCALING_MAX_SLOWDOWN, SCALING_THREADS, normalize_variant,
)
from .state import _numa, _set_full_log_dir
from .term import _Tee


class _OldOptionName(argparse.Action):
    """An option's old spelling, hidden from --help: stores into the new
    option's dest and says so on stderr (accepted until 1.0)."""
    def __init__(self, option_strings, dest, new_name=None, **kw):
        self.new_name = new_name
        super().__init__(option_strings, dest, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"note: {option_string} is the old name of {self.new_name} (accepted until 1.0)",
              file=sys.stderr)
        setattr(namespace, self.dest, values)


def build_parser():
    """The command line: every option, its default and its help."""
    parser = argparse.ArgumentParser(
        description="Benchmark + correctness runner for shrink-array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--build-root", default=os.environ.get("CHUNK_BUILD_ROOT", "."),
                        help="Root directory holding the cmake-build-*/ build "
                             "sub-directories. Default: cwd (or $CHUNK_BUILD_ROOT).")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="Build sub-directories to test (in order).")
    parser.add_argument("--skip-sanitizers", action="store_true",
                        help="Skip the ASan/TSan CHECK phase (2d). Sanitizers run "
                             f"at {SANITIZER_THREADS} threads (fewer on a smaller machine), "
                             f"{SANITIZER_REPS} run(s) per variant, "
                             "diagnostic only.")
    parser.add_argument("--skip-stl-test", action="store_true",
                        help="Skip the --test_stl STL-interface self-check (2c).")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="Variants to test. tbb_zero, tbb_zero_one_value, tbb_zero_any: the TBB "
                             "bench (shrink_bench_tbb_<variant>_<size>), looked for in the "
                             "std-backend builds (cmake-build-std-*) only.")
    parser.add_argument("--sizes", nargs="+", default=["256", "1024", "dyn"],
                        help="Chunk-size declinations to test for the shrink "
                             "variants: any of 64/256/1024 (static compile-time "
                             "sizes) and 'dyn' (runtime size). 'vector' is "
                             "size-agnostic and always tested once. Executables "
                             "are named shrink_bench_<backend>_<variant>_<size>.")
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE,
                        help="Variant used as the baseline for the speedup computation.")
    # --- Regression guards (compression health + memory) ---
    parser.add_argument("--stats-mark-ratio-min", nargs="*", default=None,
                        metavar="TAG=RATIO",
                        help="Per-variant floor on mark/create from the CHUNK_STATS "
                             "block; a run below its floor is a FAIL "
                             "(default: %s). E.g. --stats-mark-ratio-min zero_any=0.4 "
                             "zero=0.2. Needs a CHUNK_STATS build; silently skipped "
                             "otherwise." % " ".join(
                                 f"{k}={v}" for k, v in DEFAULT_STATS_MARK_RATIO_MIN.items()))
    parser.add_argument("--no-stats-guard", action="store_true",
                        help="Disable the mark/create compression-health guard.")
    parser.add_argument("--mem-zero-any-vs-zero-one-value-max", type=float, default=1.10,
                        dest="mem_zero_any_vs_zero_one_value_max",
                        help="FAIL if zero_any peak memory exceeds zero_one_value's by more than "
                             "this ratio on the same (config, threads, size) "
                             "(default 1.10). Set <= 0 to disable.")
    parser.add_argument("--mem-chunk2a-vs-chunk2-max", type=float, action=_OldOptionName,
                        dest="mem_zero_any_vs_zero_one_value_max", default=argparse.SUPPRESS,
                        new_name="--mem-zero-any-vs-zero-one-value-max", help=argparse.SUPPRESS)
    parser.add_argument("--mem-vs-baseline-max", type=float, default=None,
                        help="FAIL if a chunk variant's peak memory exceeds the baseline "
                             "variant's by more than this ratio (default: off).")
    parser.add_argument("--mode", choices=list(MODE_PRESETS.keys()), default="normal",
                        help="Preset for every phase: perf threads and repetitions, "
                             "check repetitions, sanitizers, scalability guard, seq-vs-omp "
                             "compare. Targets on the full aarch64 matrix: 'fast' ~1 h "
                             "(is anything broken?), 'normal' ~6 h, 'full' ~24 h. An option "
                             "given explicitly (--threads, --reps_perf, --reps_check, "
                             "--full-reps-threads, --seq-omp-*) overrides the preset.")
    parser.add_argument("--threads", nargs="+", type=int, default=None,
                        help="OMP_NUM_THREADS values for the perf loop. "
                             "Overrides the --mode preset.")
    parser.add_argument("--light-variants", nargs="*", default=None,
                        help="Variants run as a single smoke test (one thread count, "
                             "one run) instead of the full sweep — just to confirm they "
                             "still work. Default from --mode (fast: std_alloc).")
    parser.add_argument("--light-threads", type=int, default=None,
                        help="Thread count used for --light-variants smoke runs "
                             f"(default: the reference count, min({REFERENCE_THREADS}, "
                             "physical cores)).")
    parser.add_argument("--full-reps-threads", nargs="+", type=int, default=None,
                        help="Thread count(s) on which to run --reps_perf repetitions "
                             "(for a reliable averaged measurement). On ALL other thread "
                             "counts from --threads, a single run is done (just for the "
                             "scalability curve). Pass a count absent from --threads to "
                             "disable. Overrides the --mode preset.")
    parser.add_argument("--check-threads", type=int, default=None,
                        help="OMP_NUM_THREADS value for the correctness loop "
                             f"(default: {DEFAULT_THREADS_CHECK}, or the usable CPUs "
                             "when there are fewer).")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="Highest thread count any phase may use (default: the "
                             "CPUs this process may run on, its affinity mask). The "
                             "--mode thread lists, the reference count and the "
                             "check / seq-omp / fuzz / sanitizer counts are fitted "
                             "to it; counts passed explicitly are kept as given.")
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
                             "seq/omp config pairs, zero/zero_one_value/zero_any, "
                             "256/1024/dyn, several thread counts).")
    parser.add_argument("--seq-omp-threads", nargs="+", type=int, default=None,
                        help="OMP thread counts for the seq-vs-omp compare "
                             f"phase (default: {DEFAULT_SEQ_OMP_THREADS}, fitted "
                             "to the machine).")
    parser.add_argument("--seq-omp-sizes", nargs="+", default=None,
                        help="Chunk size declinations for the seq-vs-omp "
                             f"compare phase (default: the --mode preset's; "
                             f"normal and full: {DEFAULT_SEQ_OMP_SIZES}).")
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
    parser.add_argument("--events-file", default=os.environ.get("CHUNK_BENCH_EVENTS"),
                        help="Write the run's structured progress there, one JSON object per "
                             "line (phase start/end, runs done, end of run): what an outside "
                             "tool such as shrink_gui follows. Default: $CHUNK_BENCH_EVENTS, "
                             "else none.")
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
    parser.add_argument("--pool", default=None, metavar="ID",
                        help="Share the campaign with the other jobs given the same ID (shrink_gui's "
                             "\"Nœuds\" field adds it, one job per node): within each phase, each "
                             "(phase, configuration) unit is done by the first job that claims it in "
                             "<output-dir>/<arch>/.pools/ID (or ID itself when it holds a /), the others "
                             "skip it. Each configuration's oracle is generated once, by the job that "
                             "claims it, the others waiting for it ($CHUNK_POOL_REF_WAIT seconds at "
                             "most, default 6 h); oracle/ stays shared. Every other output (CSV, errors "
                             "log, failure list, console log, run_logs/, runs/) goes to the job's own "
                             "<output-dir>/<arch>/pool_ID/<job> (n<K> under shrink_gui, else "
                             "<host>.<pid>; an absolute --summary-csv / --errors-log / --full-log-dir "
                             "gets <job> as a last directory). A failed case is recorded and the job "
                             "goes on, as without it; an interrupted or crashed job stops them all. "
                             "With --rerun-failed: replays the failures of every job of the pool, in "
                             "this job alone.")
    return parser


def apply_regression_guards(args, parser):
    """Hand --numa-interleave and the stats-guard floors to the run helpers
    (state), and return the zero_any-vs-zero_one_value memory ceiling (None: off).
    The variant options given with an old label (chunk2a, tbb_chunk2...) are
    read as their new names here, with a note on stderr."""
    args.variants = [normalize_variant(v, "--variants") for v in args.variants]
    if args.light_variants is not None:
        args.light_variants = [normalize_variant(v, "--light-variants") for v in args.light_variants]
    args.seq_omp_variants = [normalize_variant(v, "--seq-omp-variants") for v in args.seq_omp_variants]
    args.baseline_variant = normalize_variant(args.baseline_variant, "--baseline-variant")
    # Regression guards: apply CLI overrides to the module-level floor map used
    # inside run_cmd (which has no access to args).
    state._NUMA_MODE = args.numa_interleave
    if args.no_stats_guard:
        state._STATS_MARK_RATIO_MIN = {}
    elif args.stats_mark_ratio_min is not None:
        floors = {}
        for item in args.stats_mark_ratio_min:
            if "=" not in item:
                parser.error(f"--stats-mark-ratio-min expects TAG=RATIO, got {item!r}")
            tag, val = item.split("=", 1)
            try:
                floors[normalize_variant(tag.strip(), "--stats-mark-ratio-min")] = float(val)
            except ValueError:
                parser.error(f"--stats-mark-ratio-min: bad ratio in {item!r}")
        state._STATS_MARK_RATIO_MIN = floors
    mem_za_vs_zov = (args.mem_zero_any_vs_zero_one_value_max
                     if args.mem_zero_any_vs_zero_one_value_max
                     and args.mem_zero_any_vs_zero_one_value_max > 0
                     else None)
    return mem_za_vs_zov


def _open_pool(args, shared_dir):
    """--pool: this job's Pool (share.pool), its directory <shared_dir>/.pools/<id>
    made and announced (not under --rerun-failed: a replay claims nothing).
    Returns the job's tag (n<K> under shrink_gui, else <host>.<pid>), or ""
    without --pool."""
    if not args.pool:
        return ""
    import re as _re
    if not _re.match(r"^[A-Za-z0-9._/-]+$", args.pool) or args.pool.strip("/.") == "":
        sys.exit(f"--pool {args.pool}: letters, digits, '.', '_', '-' (and '/' for a path) only")
    share.pool = Pool(args.pool, shared_dir)
    if args.rerun_failed is None:
        share.pool.open("(phase, configuration) units shared with the other jobs")
        share.register()
    return share.pool.tag


def setup_outputs(args):
    """Output directory (one per architecture), CSV and error-log paths, the
    per-run logs and the console transcript. --rerun-failed replays the
    recorded failures and exits here."""
    # Resolve the output directory and make the csv/log paths absolute inside it.
    # A caller can still pass an absolute --summary-csv/--errors-log to override.
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        # Default OUTSIDE the source tree: run outputs (oracle, CSV, logs,
        # memory_mo.txt) must not pile up in python/. Use the CCC work area
        # ($CCCWORKDIR, set by the TGCC environment) when present; fall back to
        # the cwd elsewhere (local workstation).
        _ccc_work = os.environ.get("CCCWORKDIR", "")
        out_dir = (os.path.join(_ccc_work, "bench_results")
                   if _ccc_work and os.path.isdir(_ccc_work) else os.getcwd())
    # Separate run outputs by architecture. aarch64 and x86_64 runs commonly
    # share a build_root and are launched with the SAME --output-dir; without an
    # arch level they write the same bench_summary.csv / bench_errors.log and,
    # worse, the same data_tab_* oracle in cwd=output_dir. Concurrent runs then
    # corrupt each other's files, causing spurious mid-run crashes. Appending the
    # current machine keeps each architecture's results (and oracle) isolated.
    out_dir = os.path.join(out_dir, CURRENT_MACHINE)
    os.makedirs(out_dir, exist_ok=True)
    # The architecture's directory: oracle/ and, under --pool, .pools/. The
    # same as args.output_dir without --pool.
    args.shared_dir = out_dir
    job_tag = _open_pool(args, out_dir)
    if job_tag:
        # --pool: the jobs run at the same time on the same architecture, and
        # each one rewrites its CSV, errors log, failure list and console log
        # from the start, numbers its run_logs/ files from 1 and runs its
        # binaries in runs/ and in output_dir itself (memory_mo.txt, the
        # pinned seed<N>.txt of seq/omp and fuzz): all of that goes to a
        # directory of the job's own. Only oracle/ (generated once per
        # configuration, read-only afterwards) and the pool's claims stay in
        # the shared one. The shared directory's perf seeds are copied in, so
        # the job's perf runs draw the same data a run without --pool would.
        out_dir = os.path.join(out_dir, "pool_" + os.path.basename(os.path.normpath(args.pool)), job_tag)
        os.makedirs(out_dir, exist_ok=True)
        import glob as _glob
        import shutil as _shutil
        for _seed in _glob.glob(os.path.join(args.shared_dir, "seed*.txt")):
            if not os.path.exists(os.path.join(out_dir, os.path.basename(_seed))):
                try:
                    _shutil.copy2(_seed, out_dir)
                except OSError:
                    pass
    args.output_dir = out_dir
    # shrink_gui reads the first path of this line as the run's output
    # directory (PROGRAMS out_re): under --pool, the job's own.
    print(f"📁 Run outputs → {out_dir}  (arch: {CURRENT_MACHINE})")
    if job_tag:
        print(f"📁 Shared by the pool: {args.shared_dir} (oracle/, .pools/)")

    def _own(path):
        """An absolute path given on the command line, under --pool: the
        same for every job of the pool, hence made the job's own by a last
        directory level named after it (and failed_cases.jsonl, beside the
        errors log, with it)."""
        if not job_tag:
            return path
        d = os.path.join(os.path.dirname(path), job_tag)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, os.path.basename(path))

    args.summary_csv = (_own(args.summary_csv) if os.path.isabs(args.summary_csv)
                        else os.path.join(out_dir, args.summary_csv))
    args.errors_log = (_own(args.errors_log) if os.path.isabs(args.errors_log)
                       else os.path.join(out_dir, args.errors_log))
    # Per-run full-output capture is ON by default, landing in <out_dir>/run_logs
    # (arch-aware: out_dir already ends in bench_results/<arch>). Override the
    # location with --full-log-dir, or switch it off with --no-full-log.
    if not args.no_full_log:
        _fld = args.full_log_dir if args.full_log_dir is not None else "run_logs"
        if not os.path.isabs(_fld):
            _fld = os.path.join(out_dir, _fld)
        elif job_tag:
            _fld = os.path.join(_fld, job_tag)
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
        if job_tag and not args.rerun_failed:
            # --pool: the failures of every job of the pool (each recorded its
            # own, in its directory beside this one's), replayed here alone --
            # a replay is short, and the list it rewrites is this job's.
            import glob as _glob
            mine = os.path.realpath(_failed_cases_path(args.errors_log))
            args.rerun_sources = [p for p in sorted(_glob.glob(os.path.join(
                os.path.dirname(out_dir), "*", "failed_cases.jsonl"))) if os.path.realpath(p) != mine]
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    mem_za_vs_zov = apply_regression_guards(args, parser)
    setup_outputs(args)
    resolve_run_parameters(args)
    perf_excluded = reduce_matrix(args)
    extra_ldpath = resolve_ldpath(args)
    drop_absent_configs(args)
    plan = Plan(args, perf_excluded, extra_ldpath)
    counter = RunCounter(plan.count_runs())
    events.open_file(args.events_file)
    events.emit("start", total_runs=counter.total, configs=list(args.configs),
                variants=list(args.variants), output_dir=args.output_dir, argv=sys.argv[1:],
                **({"pool": args.pool, "node": share.tag(), "shared_dir": args.shared_dir}
                   if share.active() else {}))

    results = {}
    errors  = []
    interrupted = False
    stopped_by = None       # --pool: another job stopped the pool (its "<job>: <why>")

    try:
        # --- Phase 0b: validation harnesses (ctest) ----------------------
        # First on purpose: quick, and a failure here reads better than the
        # same fault surfacing as a bench crash three phases later.
        if not args.skip_ctest:
            with events.phase("ctest", errors):
                run_ctest_phase(args, extra_ldpath, errors)

        # --- Phase 1: Correctness ---------------------------------------
        if not args.skip_check:
            with events.phase("check", errors):
                run_check_phase(args, plan, counter, errors)
            if not args.skip_stl_test:
                with events.phase("stl", errors):
                    run_stl_phase(args, plan, counter, errors)
            if not args.skip_sanitizers:
                with events.phase("sanitizer", errors):
                    run_sanitizer_phase(args, plan, counter, errors)

        # --- Phase 1b: seq-vs-omp compression-classification compare ----
        # A correctness check, so it sits with the others, BEFORE the perf
        # sweep: a classification that differs between the sequential and the
        # OpenMP build is a bug to look at now, not after an hour of timings.
        # Reads CHUNK_STATS, so it only bites on a -stats pair.
        if not args.skip_seq_omp_compare:
            with events.phase("seq_omp_compare", errors):
                run_seq_omp_compare_phase(args, extra_ldpath, errors)

        # --- Phase 2: Performance ---------------------------------------
        if not args.skip_perf:
            with events.phase("perf", errors):
                run_perf_phase(args, plan, counter, errors, results)

        # --- Phase 3: adaptive re-chunk feature test (dyn) --------------
        if not args.skip_rechunk:
            with events.phase("rechunk", errors):
                run_rechunk_phase(args, extra_ldpath, errors)

        # --- Phase 3b: 64-bit index checks (int64 binaries) --------------
        # Needs the oracles of the correctness phase, like the rechunk check.
        if not args.skip_check and not args.skip_int64_check:
            with events.phase("int64_check", errors):
                run_int64_check_phase(args, extra_ldpath, errors)

        # --- Phase 4: scalability guard (omp-llvm, omp-gnu) -------------
        # After the perf sweep: it reads the same binaries, and a failure here
        # is about the SHAPE of the curve, which only means something once the
        # timings above have been produced without incident.
        if not args.skip_scaling and not args.skip_perf:
            with events.phase("scaling", errors):
                run_scaling_phase(args, extra_ldpath, errors)

        # --- Phase 5: randomised parameter sweep (opt-in) ---------------
        # Last on purpose: it is the only phase that does NOT repeat the same
        # trajectory, so a failure here is new information rather than a
        # re-confirmation, and running it after the deterministic phases keeps
        # their timings clean.
        with events.phase("fuzz", errors):
            run_fuzz_phase(args, extra_ldpath, errors)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⛔ User interrupt — writing partial results...")
        # --pool: this job's units left half done, an oracle it was making
        # never finished: the others stop at their next claim rather than
        # wait for it (a failed CASE stops nothing: recorded, and the job
        # goes on, as without --pool).
        share.stop(f"{share.tag()} interrupted")
    except share.PoolStopped as e:
        stopped_by = str(e)
        print(f"\n\n🛑 [pool] {args.pool}: stopped by {stopped_by}; this job ({share.tag()}) claims "
              "nothing more — writing its partial results...")
    except BaseException as e:
        # A crash of the runner itself (not a failed run): the traceback goes
        # out as before, the pool is stopped first.
        share.stop(f"{share.tag()} died: {type(e).__name__}: {e}")
        raise

    total_min = (time.time() - counter.start) / 60
    if share.active():
        print(share.summary_line())
    flush_summary(results, errors, args.summary_csv, args.errors_log,
                  total_min, interrupted, baseline_variant=args.baseline_variant,
                  mem_zero_any_vs_zero_one_value_max=mem_za_vs_zov,
                  mem_vs_baseline_max=args.mem_vs_baseline_max)

    events.emit("end", errors=len(errors), interrupted=interrupted, minutes=round(total_min, 2),
                summary_csv=args.summary_csv, errors_log=args.errors_log,
                **({"stopped_by": stopped_by} if stopped_by else {}))
    # Stopped by another job of the pool: 1, its share of the campaign not done.
    sys.exit(1 if errors or stopped_by else (130 if interrupted else 0))


if __name__ == "__main__":
    main()
