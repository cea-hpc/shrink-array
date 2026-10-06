#!/usr/bin/env python3
"""
bench_and_test.py — Benchmark + correctness stress runner for shrink-array.

Supported layout (the CEA cluster one):
    $build_root/{config}/$arch/examples/shrink_bench/shrink_bench_{var}_{stage}

Example:
    $CCCWORKDIR/shrink-array/
      ├── cmake-build-omp-gnu/RHEL_9__aarch64/examples/shrink_bench/shrink_bench_chunk2_dyn
      ├── cmake-build-omp-gnu-have-aggressive-compression/.../shrink_bench_chunk2_dyn
      ├── cmake-build-omp-gnu-std-alloc/...
      ├── cmake-build-omp-llvm/...
      └── ...

Typical usage — perf comparison release vs aggressive:
    python3 bench_and_test.py --skip-check \\
        --build-root $CCCWORKDIR/shrink-array \\
        --configs cmake-build-omp-gnu \\
                  cmake-build-omp-gnu-have-aggressive-compression \\
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

The code lives in the bench_runner package next to this script (one module
per concern, one per phase under bench_runner/phases/): copy the package
along with this file, and numa_launch.py for --numa-interleave.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from bench_runner.cli import main  # noqa: E402
except ImportError as exc:
    if getattr(exc, "name", None) not in ("bench_runner", "bench_runner.cli"):
        raise
    sys.exit(f"{os.path.basename(__file__)}: the bench_runner package is not next to this script "
             f"({os.path.dirname(os.path.abspath(__file__))}). Run it from python/, or copy "
             "python/bench_runner/ along with it.")

if __name__ == "__main__":
    main()
