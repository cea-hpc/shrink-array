"""Command line and the campaign itself (bench_and_test.py calls main())."""
import subprocess
import os
import sys
import time
import argparse
from pathlib import Path

from . import state
from .builds import (
    CURRENT_MACHINE, _variant_base, config_dir_present, is_compression_only_config,
    is_omp_config, oracle_dir, resolve_exe, short_config_label, skip_variant_on_config,
)
from .phases.ctest import run_ctest_phase
from .phases.fuzz import FUZZ_THREADS, run_fuzz_phase
from .phases.int64_check import run_int64_check_phase
from .phases.rechunk import run_rechunk_phase
from .phases.scaling import run_scaling_phase
from .phases.seq_omp import run_seq_omp_compare_phase
from .progress import _prog, get_progress_bar
from .report import flush_summary
from .runs import (
    _append_error_now, _failed_cases_path, effective_timeout, run_cmd, run_rerun_failed,
    run_with_rebuild_retry,
)
from .settings import (
    DEFAULT_BASELINE, DEFAULT_CONFIGS, DEFAULT_SEQ_OMP_DIVISOR, DEFAULT_SEQ_OMP_ITER_MAX,
    DEFAULT_SEQ_OMP_SEED, DEFAULT_SEQ_OMP_SIZES, DEFAULT_SEQ_OMP_THREADS,
    DEFAULT_SEQ_OMP_VARIANTS, DEFAULT_STATS_MARK_RATIO_MIN, DEFAULT_THREADS_CHECK,
    DEFAULT_TIMEOUT, DEFAULT_TIMEOUT_PER_THREAD, DEFAULT_VARIANTS, INT64_CHECK_VARIANTS,
    MODE_PRESETS, PERF_EXCLUDED_CONFIGS, REFERENCE_THREADS, SANITIZER_CONFIGS, SANITIZER_REPS,
    SANITIZER_THREADS, SCALING_CONFIGS, SCALING_MAX_SLOWDOWN, SCALING_THREADS, fit_threads,
    is_stats_config, machine_cpus, sanitizer_env_for,
)
from .state import _numa, _set_full_log_dir
from .term import _Tee, bold, cyan, dim, green, red, yellow


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
                             f"at {SANITIZER_THREADS} threads (fewer on a smaller machine), "
                             f"{SANITIZER_REPS} run(s) per variant, "
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
                             "seq/omp config pairs, chunk/chunk2/chunk2a, "
                             "256/1024/dyn, several thread counts).")
    parser.add_argument("--seq-omp-threads", nargs="+", type=int, default=None,
                        help="OMP thread counts for the seq-vs-omp compare "
                             f"phase (default: {DEFAULT_SEQ_OMP_THREADS}, fitted "
                             "to the machine).")
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
                floors[tag.strip()] = float(val)
            except ValueError:
                parser.error(f"--stats-mark-ratio-min: bad ratio in {item!r}")
        state._STATS_MARK_RATIO_MIN = floors
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

    # --- Fit the thread counts to this machine (see machine_cpus) -----------
    usable, physical, sockets = machine_cpus()
    if args.max_threads is None:
        args.max_threads = usable
    cap  = args.max_threads
    phys = min(physical, cap)
    ref  = min(REFERENCE_THREADS, phys)
    machine_points = (ref, phys, cap)
    print(f"🖥️  Machine: {usable} usable CPUs, {physical} physical cores, {sockets} socket(s)"
          f" -> up to {cap} threads, reference {ref}")

    # --- Apply the --mode preset for any arg the user did NOT set explicitly ---
    preset = MODE_PRESETS[args.mode]
    if args.threads is None:
        args.threads = fit_threads(preset["threads"], cap, machine_points)
    if args.reps_perf is None:
        args.reps_perf = preset["reps_perf"]
    if args.reps_check is None:
        args.reps_check = preset["reps_check"]
    if args.full_reps_threads is None:
        args.full_reps_threads = fit_threads(preset["full_reps_threads"], cap, (ref,))
        # --threads given without --full-reps-threads: the full repetitions go
        # to the count of that list nearest the reference, rather than to a
        # count the sweep never runs.
        if not set(args.full_reps_threads) & set(args.threads):
            args.full_reps_threads = [min(args.threads, key=lambda t: (abs(t - ref), t))]
    if args.light_variants is None:
        args.light_variants = list(preset.get("light_variants", []))
    if args.light_threads is None:
        light = preset.get("light_threads", REFERENCE_THREADS)
        args.light_threads = light if light <= cap else ref
    if args.check_threads is None:
        args.check_threads = min(DEFAULT_THREADS_CHECK, cap)
    if args.seq_omp_threads is None:
        args.seq_omp_threads = fit_threads(DEFAULT_SEQ_OMP_THREADS, cap, machine_points)
    args.fuzz_threads      = fit_threads(FUZZ_THREADS, cap, machine_points)
    args.sanitizer_threads = min(SANITIZER_THREADS, cap)
    over = sorted({t for t in [*args.threads, args.check_threads, *args.seq_omp_threads] if t > cap})
    if over:
        print(yellow(f"⚠️  Thread count(s) {over} passed explicitly exceed the {cap} usable CPUs: "
                     "those runs are oversubscribed"))
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
                                th=args.sanitizer_threads,
                                extra="--test_stl --no_io ")
                    )
                    sys.stdout.flush()
                    ok, _, code, err_tail, _ = run_cmd(
                        [exe, "--test_stl", "--no_io"],
                        {"OMP_NUM_THREADS": str(args.sanitizer_threads)},
                        effective_timeout(args.timeout, args.timeout_per_thread, args.sanitizer_threads),
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
                                        th=args.sanitizer_threads, run=r + 1,
                                        reps=SANITIZER_REPS,
                                        extra=f"OK:{ok_in_row:3} ")
                            )
                            sys.stdout.flush()
                            ok, _, code, err_tail, _ = run_cmd(
                                [exe, "--no_io"],
                                {"OMP_NUM_THREADS": str(args.sanitizer_threads), **san_env},
                                effective_timeout(args.timeout, args.timeout_per_thread,
                                                  args.sanitizer_threads),
                                extra_ld_library_path=extra_ldpath, cwd=args.output_dir
                            )
                            if ok:
                                ok_in_row += 1
                            else:
                                _entry = (
                                    f"SANITIZER_FAIL: {exe} --no_io "
                                    f"({'TSan' if 'thread' in cfg else 'ASan'}) "
                                    f"iter={r+1} code={code} threads={args.sanitizer_threads}\n"
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
