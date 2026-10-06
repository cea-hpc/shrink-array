#!/bin/bash

# NOTE: CMAKE_BUILD_PARALLEL_LEVEL is no longer hard-coded here. It is derived
# in functions_for_compile.sh from the core count divided by CHUNK_MAX_JOBS,
# so the cores are SHARED between the concurrent builds instead of being
# multiplied by them. Set it in the environment to override.

# Site layer: paths, compilers, Environment Modules or not (site_env.sh).
source "$(dirname "${BASH_SOURCE[0]}")/site_env.sh"

# Targeted module unloads instead of a module purge (avoids datadir conflicts on ARM)
module unload gcc 2>/dev/null || true
module unload cmake 2>/dev/null || true
module unload ninja 2>/dev/null || true
module unload c/gcc 2>/dev/null || true
module unload c++/gcc 2>/dev/null || true
module unload fortran/gcc 2>/dev/null || true

# The newest CMake the modules offer (4.0.3), else 3.31.4, else the default.
chunk_module_load_first cmake/4.0.3 cmake/3.31.4 cmake
module load ninja

# Parse command-line arguments
DRYRUN=false
BUILD_ONLY=false
DEBUG_ONLY=false
FRESH_CACHE=false   # --fresh-cache: rm each build dir's CMakeCache.txt before configuring
NO_IPA_ICF=false
NO_HARDENING=false # --no-hardening: skip CHUNK_GLIBCXX_ASSERTIONS / CHUNK_UBSAN on Debug+sanitizer builds

while [[ $# -gt 0 ]]; do
    case $1 in
        --dryrun)
            DRYRUN=true
            shift
            ;;
         --build)
            BUILD_ONLY=true
            shift
            ;;
        --debug-only)
            DEBUG_ONLY=true
            shift
            ;;
        --fresh-cache)
            FRESH_CACHE=true
            shift
            ;;
        --no-ipa-icf)
            NO_IPA_ICF=true
            shift
            ;;
        --jobs=*)
            # How many build DIRS may compile at once. The core budget is split
            # across them, so this bounds total compiler processes rather than
            # multiplying them. 1 serialises entirely.
            export CHUNK_MAX_JOBS="${1#*=}"
            case "$CHUNK_MAX_JOBS" in
                ''|*[!0-9]*) echo "ERROR: --jobs=$CHUNK_MAX_JOBS (expected a positive integer)" >&2; exit 2 ;;
                0)           echo "ERROR: --jobs=0 (expected a positive integer)" >&2; exit 2 ;;
            esac
            shift
            ;;
        --no-hardening)
            # Debug and sanitizer flavours normally get -DCHUNK_GLIBCXX_ASSERTIONS=ON,
            # and the address-sanitizer ones -DCHUNK_UBSAN=ON as well (rule in
            # functions_for_compile.sh::hardening_opts_for). This turns that off.
            NO_HARDENING=true
            shift
            ;;
        --target=*)
            export CHUNK_BUILD_TARGET="${1#*=}"
            shift
            ;;
        --exec=*)
            # Build a single Ninja target (one executable) inside each selected
            # build dir, e.g. --exec=shrink_bench_chunk_64. Combine with
            # --target=<glob> to also restrict WHICH build dirs are configured.
            export CHUNK_BUILD_EXEC="${1#*=}"
            shift
            ;;
        *)
            # Previously swallowed silently — a --debug_only (underscore) went
            # unnoticed and the intended flag was never applied.
            echo "ERROR: unknown argument '$1' (expected --dryrun|--build|--debug-only|--fresh-cache|--no-ipa-icf|--no-hardening|--jobs=<N>|--target=<glob>|--exec=<ninja-target>)" >&2
            exit 2
            ;;
    esac
done

# --debug-only: build only the Debug flavours.
skip_release() { [ "$DEBUG_ONLY" = true ]; }

if [ "$NO_IPA_ICF" = true ]; then
    ICF_CMAKE_OPT="-DCHUNK_NO_IPA_ICF=ON"
    ICF_SUFFIX="-noicf"
    echo "compile_all_gnu.sh: --no-ipa-icf enabled → adding $ICF_CMAKE_OPT (build dirs suffixed with '$ICF_SUFFIX')"
else
    ICF_CMAKE_OPT=""
    ICF_SUFFIX=""
fi

export FRESH_CACHE   # seen by compile_with_cmake / run_cmd via maybe_wipe_cache

# Hardening switch, read by hardening_opts_for (functions_for_compile.sh).
# Default on; --no-hardening (or an inherited CHUNK_HARDENING=off from
# compile_all.sh) turns it off.
if [ "$NO_HARDENING" = true ]; then
    export CHUNK_HARDENING=off
    echo "compile_all_gnu.sh: --no-hardening → Debug/sanitizer builds without CHUNK_GLIBCXX_ASSERTIONS / CHUNK_UBSAN"
else
    export CHUNK_HARDENING="${CHUNK_HARDENING:-on}"
fi

source "$CHUNK_SCRIPT_DIR/functions_for_compile.sh"

# --target=<glob> that matches none of this script's configurations: stop here
# with the closest names (skipped when compile_all.sh already checked).
check_target_arg $(known_build_names "$0")

build_rep="$CHUNK_BUILD_REP"

create_build_rep

# --- Architecture detection & compiler/flags selection ---
if [ "$(uname -m)" = "aarch64" ]; then
    echo "[ARM/aarch64 detected] Loading gcc/16.1.0 with -mcpu=native"
    module load binutils/2.43
    module load gcc/16.1.0
    ARCH_FLAGS="-DCMAKE_CXX_FLAGS=-mcpu=native" # C++23: required by the CMakeLists itself
else
    echo "[x86_64 detected] Loading gcc/15.2.0 with -march=native"
    module load gcc/15.2.0
    ARCH_FLAGS="-DCMAKE_CXX_FLAGS=-march=native"
fi

# Fetch the g++ path from the loaded module (without modules: CHUNK_GXX,
# the newest g++-N on PATH, see site_env.sh)
if [ "$CHUNK_HAVE_MODULES" = true ]; then
    gxx="$(command -v g++ 2>/dev/null)"
else
    gxx="$(command -v "$CHUNK_GXX" 2>/dev/null)"
    echo "[site] no modules: gcc is $CHUNK_GXX"
fi
[ -z "$gxx" ] && gxx="g++"  # dry-run fallback: bare name, resolved at configure time
export CXX="$gxx"

arch="$(source /etc/os-release && echo ${ID^^}_${VERSION_ID}__$(uname -m))"

# Cleaned-up calls: no more ambiguous quoting in the forwarded string
if skip_release; then
    echo "[debug-only] skipping gnu openmp"
else
    compile_with_cmake "$build_rep/cmake-build-omp-gnu${ICF_SUFFIX}/$arch" \
      "-DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT $INT64_OPT" \
      "gnu openmp"
fi

if skip_release; then
    echo "[debug-only] skipping gnu openmp with check"
else
    compile_with_cmake "$build_rep/cmake-build-omp-gnu-check${ICF_SUFFIX}/$arch" \
      "-DCHECK_OPTIM=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu openmp with check"
fi

# Optimised build with the CHUNK_STATS instrumentation counters ON (per-variant,
# per-thread; reduced and printed at end of run). Kept as a SINGLE stats target
# per compiler on purpose -- crossing CHUNK_STATS with every other axis would
# blow up the build matrix, and the counters are orthogonal to alloc/sanitizer.
if skip_release; then
    echo "[debug-only] skipping gnu openmp with stats"
else
    compile_with_cmake "$build_rep/cmake-build-omp-gnu-stats${ICF_SUFFIX}/$arch" \
      "-DCHUNK_STATS:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu openmp with stats"
fi

if skip_release; then
    echo "[debug-only] skipping gnu openmp std allocator"
else
    compile_with_cmake "$build_rep/cmake-build-omp-gnu-std-alloc${ICF_SUFFIX}/$arch" \
      "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu openmp std allocator"
fi

if skip_release; then
    echo "[debug-only] skipping gnu openmp have_agressive_compression"
else
    compile_with_cmake "$build_rep/cmake-build-omp-gnu-have-agressive-compression${ICF_SUFFIX}/$arch" \
      "-DHAVE_AGRESSIVE_COMPRESSION:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu openmp have_agressive_compression"
fi

if skip_release; then
    echo "[debug-only] skipping gnu"
else
    compile_with_cmake "$build_rep/cmake-build-gnu${ICF_SUFFIX}/$arch" \
      "-DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu"
fi

if skip_release; then
    echo "[debug-only] skipping gnu with check (no openmp)"
else
    compile_with_cmake "$build_rep/cmake-build-gnu-check${ICF_SUFFIX}/$arch" \
      "-DCHECK_OPTIM=ON -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu with check (no openmp)"
fi

if skip_release; then
    echo "[debug-only] skipping gnu std allocator (no openmp)"
else
    compile_with_cmake "$build_rep/cmake-build-gnu-std-alloc${ICF_SUFFIX}/$arch" \
      "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu std allocator (no openmp)"
fi

# Sequential twin of cmake-build-omp-gnu-stats above: same release build, same
# CHUNK_STATS counters, OpenMP OFF. bench_and_test.py's seq-vs-OMP compare
# (find_llvm_seq_omp_pairs) derives "cmake-build-gnu-stats" from
# "cmake-build-omp-gnu-stats" by stripping the "omp-", and compares `still
# compressed (at exit)` between the two -- a number that must not depend on the
# thread count. Without this build the GNU stats pair is half-present and the
# whole phase reports "no sequential counterpart built for gnu-stats", which is
# exactly what it did until now (compile_all.sh has carried the llvm twin for a
# while; the GNU one was simply never added).
if skip_release; then
    echo "[debug-only] skipping gnu with stats (no openmp)"
else
    compile_with_cmake "$build_rep/cmake-build-gnu-stats${ICF_SUFFIX}/$arch" \
      "-DCHUNK_STATS:BOOL=ON -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
      "gnu with stats (no openmp)"
fi

compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug${ICF_SUFFIX}/$arch" \
  "-DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp"

# Debug OpenMP WITH the pool ledger forced on. The ledger is off by default
# under OpenMP (its global mutex serialises every alloc/free and distorts the
# concurrency a debug build is meant to exercise -- see pool_ledger.h), so
# without this target its double-free / illegal-park detection would never run
# in the one mode where those bugs actually appear. Kept as a SINGLE dedicated
# target, like the stats one: crossing CHUNK_FORCE_LEDGER with every other axis
# would blow up the matrix for no added coverage.
# NB: passed as a CMake option, NOT via -DCMAKE_CXX_FLAGS -- $ARCH_FLAGS already
# sets CMAKE_CXX_FLAGS (-mcpu=native on ARM) and the second one would win.
compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-ledger${ICF_SUFFIX}/$arch" \
  "-DCHUNK_FORCE_LEDGER:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT $SYNC_CMAKE_OPT" \
  "gnu debug openmp with ledger"

# Debug + std allocator + OpenMP, WITHOUT a sanitizer. The two std-alloc debug
# flavours below only exist in thread/address-sanitizer form, so a plain debug
# run of the std-alloc path had no build dir of its own -- and a sanitizer build
# is not a substitute: it is ~10x slower and its allocator interposition hides
# the very std-allocator behaviour this config is meant to exercise.
compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-std-alloc${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp std allocator"

compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-thread-sanitizer${ICF_SUFFIX}/$arch" \
  "-DOMP:BOOL=ON -DSANITIZER=thread -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp thread sanitizer"

# (No fast-allocator ASan build: with the fast allocator the chunk buffers
# live inside the library's slab arenas, which ASan sees as a few large blocks
# -- an overflow or use-after-free inside a slab goes unreported. ASan runs on
# the std-alloc builds below, where every buffer is a real malloc.)

compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-std-alloc-thread-sanitizer${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DSANITIZER=thread -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp std-alloc thread sanitizer"

compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-std-alloc-address-sanitizer${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DSANITIZER=address -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp std-alloc address sanitizer"

# Sanitizers on OPTIMISED code (RelWithDebInfo, i.e. -O2 + -g).
#
# Why these exist: every other sanitizer flavour above is CMAKE_BUILD_TYPE=Debug,
# so a bug that only shows up under optimisation is never seen by TSan or ASan --
# their silence proves nothing about the -O2 binaries. That was exactly the case
# for the std-alloc heap corruption ("corrupted size vs. prev_size" in
# malloc_consolidate, from ChunkBase::alloc_values under a live rechunk): it does
# not reproduce in Debug, and the Debug sanitizer builds had nothing to report.
#
# -O2 also changes what the tools see: inlining and reordering are part of the
# failing behaviour, and dropped frame pointers would make the reports unusable
# (chunk/CMakeLists.txt adds -fno-omit-frame-pointer for any SANITIZER build).
#
# Usage note: TSan's shadow memory scales with the thread count -- start these
# at OMP_NUM_THREADS=8..16, not 249; a race shows up long before that.
compile_with_cmake "$build_rep/cmake-build-omp-gnu-relwithdebinfo-std-alloc-thread-sanitizer${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DSANITIZER=thread -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu relwithdebinfo openmp std-alloc thread sanitizer"

compile_with_cmake "$build_rep/cmake-build-omp-gnu-relwithdebinfo-std-alloc-address-sanitizer${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=ON -DSANITIZER=address -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu relwithdebinfo openmp std-alloc address sanitizer"

# Optimised TSan build with the switches of the binary that diverged on
# 2026-09-14 (shrink_bench_chunk2a_256 --mode check, libgomp, 2283 cycles): fast
# allocator (default), HAVE_AGRESSIVE_COMPRESSION, CHECK_OPTIM as in every
# *_check_* target, RelWithDebInfo so inlining and scheduling are the failing
# ones. GNU-specific caveat: libgomp is not TSan-instrumented, so TSan does not
# see its barriers and reports a race at every parallel-region boundary (the
# "destroy_after_flush vs flush_deferred" report is one). Read this build's
# output next to the LLVM+libomp twin in compile_all.sh, whose runtime carries
# the barrier annotations; a report that only the GNU build shows is a
# barrier the tool did not see, a report both show is a bug.
compile_with_cmake "$build_rep/cmake-build-omp-gnu-relwithdebinfo-thread-sanitizer-have-agressive-compression${ICF_SUFFIX}/$arch" \
  "-DHAVE_AGRESSIVE_COMPRESSION:BOOL=ON -DCHECK_OPTIM=ON -DOMP:BOOL=ON -DSANITIZER=thread -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu relwithdebinfo openmp thread sanitizer have_agressive_compression"

compile_with_cmake "$build_rep/cmake-build-omp-gnu-debug-have-agressive-compression${ICF_SUFFIX}/$arch" \
  "-DHAVE_AGRESSIVE_COMPRESSION:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug openmp have_agressive_compression"

compile_with_cmake "$build_rep/cmake-build-gnu-debug${ICF_SUFFIX}/$arch" \
  "-DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug"

compile_with_cmake "$build_rep/cmake-build-gnu-std-alloc-debug${ICF_SUFFIX}/$arch" \
  "-DCHUNK_STD_ALLOC:BOOL=ON -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=Debug $ARCH_FLAGS $ICF_CMAKE_OPT" \
  "gnu debug std allocator (no openmp)"

# Wait for every gnu build and print the summary as they finish.
# Mandatory: when launched from compile_all.sh, this script would otherwise
# return with its ninjas orphaned, out of reach of the parent's wait.
echo -e '\n─── Waiting for gnu builds ───\n'
wait_all
rc=$?
# Standalone run: a --target / --exec that let nothing through is an error
# (under compile_all.sh the parent gives that verdict for both scripts).
if ! check_arg_epilogue; then
    echo -e "\ncompile_all_gnu: nothing built\n"
    exit 1
fi
if [ "$rc" -eq 0 ]; then
    echo -e "\ncompile_all_gnu: all builds OK\n"
else
    echo -e "\ncompile_all_gnu: $rc failed build(s) (logs: $LOG_DIR)\n"
fi
exit "$rc"
