#!/bin/bash

# BUILD_FLAGS used to force "-- -j 8" onto every cmake --build. That overrode
# CMAKE_BUILD_PARALLEL_LEVEL and produced "ninja-build -j 32 -j 8": ninja keeps
# the last one, so the whole run was pinned at 8 jobs per build whatever the
# node. Left empty; the per-build job count now comes from
# CMAKE_BUILD_PARALLEL_LEVEL, derived in functions_for_compile.sh.
BUILD_FLAGS=""

# NOTE: CMAKE_BUILD_PARALLEL_LEVEL is no longer hard-coded here. It is derived
# in functions_for_compile.sh from the core count divided by CHUNK_MAX_JOBS,
# so the cores are SHARED between the concurrent builds instead of being
# multiplied by them. Set it in the environment to override.

# Unload/reload the main build tools
module unload cmake 2>/dev/null || true
module unload ninja 2>/dev/null || true
# The same CMake as compile_all_gnu.sh (the default module is 3.20).
module load cmake/3.31.4 2>/dev/null || module load cmake
module load ninja

# Default value for dry-run mode
DRYRUN=false
BUILD_ONLY=false
FRESH_CACHE=false   # --fresh-cache: wipe CMakeCache.txt before each configure (functions_for_compile.sh::maybe_wipe_cache)
DEBUG_ONLY=false
arch="$(source /etc/os-release && echo ${ID^^}_${VERSION_ID}__$(uname -m))"

# ARM detection — oneapi is x86-only on CEA machines, so we skip that whole
# section on aarch64.
is_arm=false
if [ "$(uname -m)" = "aarch64" ]; then
    is_arm=true
    echo "[INFO] aarch64 detected — skipping oneapi builds"
fi

# ── mold ─────────────────────────────────────────────────────────────────────
# The module name differs per machine: aarch64 has a standalone `mold`, x86
# packages it behind gnu/11. Load it only if mold is not already reachable, and
# NEVER let its dependency stay in front of the compiler we actually build with:
# loading gnu/11 next to gcc 15.2 rewrote COMPILER_PATH and the link then died
# with "collect2: fatal error: cannot find 'ld'".
#
# So: load, capture mold's absolute path, unload the dependency straight away.
# The top-level CMakeLists finds mold by path and routes the driver to it
# without touching the toolchain search path (see USE_MOLD there). Set
# CHUNK_NO_MOLD=1 to skip all of this.
load_mold() {
    [ -n "${CHUNK_NO_MOLD:-}" ] && { echo "[INFO] mold disabled (CHUNK_NO_MOLD)"; return 0; }

    if command -v mold >/dev/null 2>&1; then
        export CHUNK_MOLD_PATH="$(command -v mold)"
        echo "[INFO] mold already in PATH: $CHUNK_MOLD_PATH"
        return 0
    fi

    # Resolve mold's path in a SUBSHELL, never in this environment.
    #
    # On x86 the mold module depends on gnu/11, which conflicts with the
    # gnu/15.2.0 already loaded for the build:
    #     Loading gnu/11.2.0
    #       ERROR: Module cannot be loaded due to a conflict.
    #     ERROR: Your current Module state is not compatible with 'mold/2.42.0'!
    # And when it DOES load, gnu/11 displaces the build compiler's own ld and the
    # link dies with "collect2: fatal error: cannot find 'ld'".
    #
    # Either way the module environment is the problem, and we only ever needed
    # one thing from it: where the mold binary lives. So swap the conflicting
    # module inside a subshell, read the path, and let the subshell take the
    # damaged environment with it. CMake is handed the absolute path
    # (CHUNK_MOLD_PATH) and routes the driver to it without touching the
    # toolchain search path.
    local found
    found=$(
        {
            if [ "$is_arm" = true ]; then
                module load mold
            else
                # -f: let the module system swap the conflicting gnu version
                # rather than refuse. Confined to this subshell.
                module unload gnu 2>/dev/null || true
                module load gnu/11 mold/2.42.0 2>/dev/null || module load mold 2>/dev/null || true
            fi
        } >/dev/null 2>&1
        command -v mold 2>/dev/null
    )

    if [ -n "$found" ] && [ -x "$found" ]; then
        export CHUNK_MOLD_PATH="$found"
        echo "[INFO] mold: $CHUNK_MOLD_PATH (resolved in a subshell; module state untouched)"
    else
        echo "[WARN] mold not found — builds will use the default linker"
    fi
}
load_mold

# oneAPI (icpx) builds, x86-only. They compile again (Sept '26). The icpx
# 2026.0 ICE in clang CodeGen (EmitCtorList -> getPointerBitCastOrAddrSpaceCast
# on a null Constant, emitting global_ctors for the explicit ManagerChunk
# instantiations; Intel priority-support ticket) has its fallback in
# chunk/CMakeLists.txt: -DCHUNK_ICPX_O0_WORKAROUND=ON compiles the three TUs it
# hit at -O0 under icpx. Set to false to skip oneAPI.
enable_oneapi=true

# AOCC (AMD Optimizing C/C++ Compiler, clang-based) — x86-only, like oneapi.
# Flip to false to skip. Auto-detects the newest aocc-compiler/<ver> module.
enable_aocc=true

# Per-compiler / sanitizer switches (default on). Toggle here, or via CLI:
#   --no-gnu --no-oneapi --no-nvhpc --no-llvm --no-aocc --no-sanitizers
#   --only=<gnu|oneapi|nvhpc|llvm|aocc>   (build just that compiler)
# nvhpc is OFF by default for now: about 1h30 per build on x86 and no
# performance gain over the others. --with-nvhpc or
# --only=nvhpc turns it back on.
enable_gnu=true
enable_nvhpc=false
enable_llvm=true
enable_sanitizers=true   # the llvm debug thread/address-sanitizer builds
llvm_cs_size=""          # on-demand llvm build with a custom CHUNK_SIZE (e.g. 1024); empty = off
NO_HARDENING=false       # --no-hardening: skip CHUNK_GLIBCXX_ASSERTIONS / CHUNK_UBSAN on Debug+sanitizer builds

# Parse command-line arguments
usage() {
    cat >&2 <<EOF
Usage: $0 [--dryrun] [--build] [--debug-only]

  --dryrun      Print the commands instead of running them.
  --build       Skip the cmake configure step (rebuild only).
  --fresh-cache Remove each selected build dir's CMakeCache.txt before
                configuring, so every option is re-read from CMakeLists.txt
                instead of a value cached by an earlier configure. Use it
                after any change to CMakeLists.txt or to a -D option, and
                whenever a measurement is about to be trusted: a stale
                CHUNK_RECLAIM_EMPTY_SLABS=OFF in one cache cost two days.
                Forwarded to compile_all_gnu.sh.
  --debug-only  Compile only the debug builds (skip release / RelWithDebInfo).
  --only=<c>       Build only compiler <c> (gnu|oneapi|nvhpc|llvm|aocc).
  --target=<glob>  Build only the config(s) whose name matches <glob>
                   (e.g. omp-llvm, 'omp-llvm*', '*sanitizer*', '*check*').
                   Combine with --only=<c> to also skip other compilers' configure.
                   A glob that matches no known configuration is an error
                   (exit 2) naming the closest ones, before anything runs.
  --exec=<glob>    Build only the Ninja target(s) matching <glob> inside each
                   selected build dir. A literal name builds one executable,
                   e.g. --exec=shrink_bench_chunk_64. A pattern builds a
                   family, e.g. --exec='*_dyn' (all dynamic exes) or
                   --exec='*_256' (all static-256 exes) or
                   --exec='shrink_bench_chunk2a_*'. Patterns are matched against
                   the real Ninja target list; quote them so the shell does not
                   expand the glob first. Ninja still builds each target's deps
                   (the chunk libs), just not the other exes. A name or pattern
                   that matches nothing in a build dir skips that dir (warned),
                   never falls back to a full build; matching nothing anywhere
                   is an error (exit 1) naming the closest targets. Combine with
                   --target=<glob> to also restrict which build dirs are configured.
  --no-<c>         Skip compiler <c> (gnu|oneapi|nvhpc|llvm|aocc).
  --with-nvhpc     Also build nvhpc, off by default (slow: ~1h30 on x86,
                   no performance gain).
  --no-sanitizers  Skip the llvm thread/address-sanitizer builds.
  --jobs=N         Build at most N configurations at once (default 4). The core
                   budget is split across them, so this bounds the total number
                   of compiler processes instead of multiplying it. Use 1 to
                   serialise when a build gets OOM-killed.
  --no-hardening   Do not add -DCHUNK_GLIBCXX_ASSERTIONS=ON to Debug/sanitizer builds,
                   nor -DCHUNK_UBSAN=ON to the address-sanitizer builds (default: added).
  --llvm-cs=<N>    Also build an llvm RelWithDebInfo variant with CHUNK_SIZE=N (power of two).
  -h, --help    Show this help.
EOF
}

for arg in "$@"; do
    case $arg in
        --dryrun)
            DRYRUN=true
            shift
            ;;
        --build)
            BUILD_ONLY=true
            shift
            ;;
        --fresh-cache)
            FRESH_CACHE=true
            shift
            ;;
        --debug-only)
            DEBUG_ONLY=true
            shift
            ;;
        --no-gnu)        enable_gnu=false;        shift ;;
        --no-oneapi)     enable_oneapi=false;     shift ;;
        --no-nvhpc)      enable_nvhpc=false;      shift ;;
        --with-nvhpc)    enable_nvhpc=true;       shift ;;
        --no-llvm)       enable_llvm=false;       shift ;;
        --no-aocc)       enable_aocc=false;       shift ;;
        --no-sanitizers) enable_sanitizers=false; shift ;;
        --no-hardening)  NO_HARDENING=true;       shift ;;
        --jobs=*)
            # How many build DIRS may compile at once (see functions_for_compile.sh).
            export CHUNK_MAX_JOBS="${arg#*=}"
            case "$CHUNK_MAX_JOBS" in
                ''|*[!0-9]*|0) echo "ERROR: --jobs=$CHUNK_MAX_JOBS (expected a positive integer)" >&2; exit 2 ;;
            esac
            shift ;;
        --llvm-cs=*)
            llvm_cs_size="${arg#*=}"
            case "$llvm_cs_size" in
                ''|*[!0-9]*) echo "ERROR: --llvm-cs=$llvm_cs_size (expected a positive integer)" >&2; exit 1 ;;
            esac
            # power-of-two check: (n & (n-1)) == 0
            if [ $(( llvm_cs_size & (llvm_cs_size - 1) )) -ne 0 ]; then
                echo "ERROR: --llvm-cs=$llvm_cs_size must be a power of two" >&2; exit 1
            fi
            shift ;;
        --target=*)
            export CHUNK_BUILD_TARGET="${arg#*=}"
            shift
            ;;
        --exec=*)
            export CHUNK_BUILD_EXEC="${arg#*=}"
            shift
            ;;
        --only=*)
            only="${arg#*=}"
            enable_gnu=false; enable_oneapi=false; enable_nvhpc=false; enable_llvm=false; enable_aocc=false
            case "$only" in
                gnu)    enable_gnu=true ;;
                oneapi) enable_oneapi=true ;;
                nvhpc)  enable_nvhpc=true ;;
                llvm)   enable_llvm=true ;;
                aocc)   enable_aocc=true ;;
                *) echo "ERROR: --only=$only (expected gnu|oneapi|nvhpc|llvm|aocc)" >&2; exit 1 ;;
            esac
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$arg'" >&2
            usage
            exit 2
            ;;
    esac
done

run_cmd() {
    if [ "$DRYRUN" = true ]; then
        echo "$@"
    else
        eval "$@"
    fi
}

skip_release() { [ "$DEBUG_ONLY" = true ]; }

# ── Module hygiene between compiler families ─────────────────────────────────
#
# The module environment is process-global, and the families are configured one
# after another in THIS shell, so whatever one leaves loaded the next inherits.
# Each family used to unload only what it knew about by name -- gcc, llvm, cuda,
# inteloneapi -- and nothing unloaded the TRANSITIVE dependencies. That is a real
# failure, not a tidiness issue: compile_all_gnu.sh loads binutils/2.43 for
# gcc/16.1.0 on aarch64, nvhpc then needs binutils/2.42, and the load fails in
# cascade:
#
#     Loading binutils/2.42
#       ERROR: Module cannot be loaded due to a conflict.
#     Loading c/gcc/15.2.0
#       ERROR: Load of requirement binutils/2.42 failed
#     Loading gcc/15.2.0
#       ERROR: Load of requirement c/gcc/15.2.0 failed
#
# ...and the build silently proceeds with whatever compiler was left in PATH.
#
# So: unload the whole known set before each family, by BARE name so the loaded
# version goes whatever it is (`module unload binutils` beats
# `module unload binutils/2.43`, which misses 2.42 and vice versa). Unloading
# something that is not loaded is not an error.
#
# Not done with a subshell per family, which would be the tidier answer: the
# builds are started with launch_build in the BACKGROUND, and jobs started inside
# a subshell are invisible to the parent's wait_all.
reset_compiler_modules() {
    local m
    for m in gcc llvm binutils cuda inteloneapi aocc-compiler nvhpc \
             c/gcc c++/gcc fortran/gcc; do
        run_cmd "module unload $m 2>/dev/null || true"
    done
}



run_cmd "module load ninja"
run_cmd "module load cmake/3.31.4 2>/dev/null || module load cmake"

source ./functions_for_compile.sh

# --target=<glob> that matches none of the configurations either script knows:
# stop here with the closest names, before a single configure runs.
check_target_arg $(known_build_names "$0" "$(dirname "$0")/compile_all_gnu.sh")

# Hardening switch, read by hardening_opts_for (functions_for_compile.sh) for
# every cmake configure issued here, and inherited by compile_all_gnu.sh.
if [ "$NO_HARDENING" = true ]; then
    export CHUNK_HARDENING=off
    echo "compile_all.sh: --no-hardening → Debug/sanitizer builds without CHUNK_GLIBCXX_ASSERTIONS / CHUNK_UBSAN"
else
    export CHUNK_HARDENING=on
fi

build_rep=$CCCWORKDIR/shrink-array
run_cmd "echo $build_rep"

run_cmd "create_build_rep"

if [ "$enable_oneapi" = true ] && [ "$is_arm" = false ]; then

set_build_header 'Compiling oneapi openmp:'

# Use GCC 15.2.0 so icpx finds <format> (libstdc++)

reset_compiler_modules
run_cmd "module load gcc/15.2.0"
run_cmd "module load inteloneapi/26.0.0"
run_cmd "module load ninja"

ONEAPI_GCC_TOOLCHAIN_DIR="/ccc/products/gcc-15.2.0/system/default"
ONEAPI_STDLIB_OPT="-DCMAKE_CXX_FLAGS=--gcc-toolchain=$ONEAPI_GCC_TOOLCHAIN_DIR"

if skip_release; then
    echo "[debug-only] skipping oneapi release builds"
else
    if [ "$BUILD_ONLY" = false ]; then
        run_cmd "CXX=icpx cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-oneapi/$arch $ONEAPI_STDLIB_OPT -DOMP:BOOL=ON $INT64_OPT -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-omp-oneapi/$arch" $BUILD_FLAGS
    # The three -manual-compression twins that used to follow each oneapi
    # build are gone: -DHAVE_MANUAL_COMPRESSION is read by no CMakeLists and
    # by no source any more (the manual mode left when compression became the
    # embedded, incremental one), so they compiled the very same code a second
    # time -- three oneapi builds' worth of wall time for nothing.

    set_build_header 'Compiling oneapi:'
    if [ "$BUILD_ONLY" = false ]; then
        run_cmd "CXX=icpx cmake -S ~/shrink-array -B $build_rep/cmake-build-oneapi/$arch $ONEAPI_STDLIB_OPT -DOMP:BOOL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-oneapi/$arch" $BUILD_FLAGS
fi

set_build_header 'Compiling oneapi debug openmp:'
if [ "$BUILD_ONLY" = false ]; then
    run_cmd "CXX=icpx cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-oneapi-debug/$arch $ONEAPI_STDLIB_OPT -DCMAKE_BUILD_TYPE=Debug -DOMP:BOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-omp-oneapi-debug/$arch" $BUILD_FLAGS

else
    echo "[SKIP] oneapi builds disabled (enable_oneapi=$enable_oneapi, is_arm=$is_arm)"
fi  # end of "if enable_oneapi && !is_arm"

# Delegate to the GNU builds
gnu_args=""
[ "$DRYRUN" = true ]     && gnu_args="$gnu_args --dryrun"
[ "$BUILD_ONLY" = true ] && gnu_args="$gnu_args --build"
[ "$DEBUG_ONLY" = true ] && gnu_args="$gnu_args --debug-only"
[ "$NO_HARDENING" = true ] && gnu_args="$gnu_args --no-hardening"
[ -n "${CHUNK_MAX_JOBS:-}" ] && gnu_args="$gnu_args --jobs=$CHUNK_MAX_JOBS"
# CHUNK_SLOT_DIR is exported by __slot_dir(): passing it down makes the gnu
# script share THIS run's slot pool instead of opening a second one, which is
# what let 9 ninjas run under two independent caps of 4.
__slot_dir >/dev/null   # create it now: with no build of our own it would not exist yet
export CHUNK_SLOT_DIR
[ -n "${CHUNK_BUILD_EXEC:-}" ] && gnu_args="$gnu_args --exec=$CHUNK_BUILD_EXEC"
[ "$FRESH_CACHE" = true ] && gnu_args="$gnu_args --fresh-cache"
if [ "$enable_gnu" = true ]; then
    # Drain OUR OWN builds before handing over.
    #
    # The oneapi/llvm builds started above run in the background, and
    # compile_all_gnu.sh is a separate process with its own job table and its own
    # [RUN ] block. Delegating while ours are still compiling meant two families
    # sharing the node with neither aware of the other, and only the gnu
    # progress on screen -- clang++ processes churning invisibly in top while
    # the display showed four gnu builds. Waiting here keeps one family, one
    # display, one load at a time.
    #
    # The slot semaphore already bounds the TOTAL, so this is about legibility
    # and node pressure rather than correctness; --no-gnu / --only= still let a
    # family be run on its own.
    if [ "${#JOB_NAME[@]}" -gt 0 ]; then
        target_filter_summary
echo -e '\n─── Waiting for the builds started here before delegating to gnu ───\n'
        wait_all
    fi

    # The gnu script waits for its own builds and returns its failure count;
    # aggregate it into the final summary.
    #
    # Run it directly, NOT through run_cmd. In --dryrun, run_cmd echoes its
    # argument instead of executing it, so the delegation itself was printed and
    # the child never started -- and since --only=gnu skips everything this
    # script builds on its own, `--only=gnu --dryrun` printed one line and
    # nothing else. The child gets --dryrun in $gnu_args, so it prints ITS cmake
    # commands, which is the point of asking for a dry run.
    # CHUNK_ARGS_CHECKED_BY_PARENT: --target was validated above and the
    # --exec verdict is given once, by our check_arg_epilogue, over the markers
    # both processes leave in the slot dir.
    CHUNK_ARGS_CHECKED_BY_PARENT=1 ./compile_all_gnu.sh$gnu_args
    gnu_rc=$?
    if [ "$gnu_rc" -ne 0 ]; then
        echo "[WARN] compile_all_gnu.sh: $gnu_rc failed gnu build(s)"
        N_FAILED=$(( N_FAILED + gnu_rc ))
    fi
else
    echo "[SKIP] gnu builds disabled (enable_gnu=$enable_gnu)"
fi

# NVHPC Section
if [ "$enable_nvhpc" = true ]; then
set_build_header 'Compiling nvhpc openmp:'
NVHPC_GCC_TOOLCHAIN_DIR="/ccc/products/gcc-15.2.0/system/default"
latest_nvhpc=$(module -t av nvhpc 2>&1 | tr -s '[:space:]' '\n' | grep -oE 'nvhpc/[0-9]+(\.[0-9]+)*' | sort -V -u | tail -1)

if [ -n "$latest_nvhpc" ]; then
    reset_compiler_modules
    if run_cmd "module load $latest_nvhpc"; then
        run_cmd "module load gcc/15.2.0"
        nvcxx="$(command -v nvc++ 2>/dev/null)"
    [ -z "$nvcxx" ] && nvcxx="nvc++"  # same dry-run fallback as clangxx
        if [ -n "$nvcxx" ]; then
            NVHPC_STDLIB_OPT="-DCMAKE_CXX_FLAGS=--gcc-toolchain=$NVHPC_GCC_TOOLCHAIN_DIR"

            if skip_release; then
                echo "[debug-only] skipping nvhpc release build"
            else
                if [ "$BUILD_ONLY" = false ]; then
                    run_cmd "CXX='$nvcxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-nvhpc/$arch -DCMAKE_CXX_COMPILER='$nvcxx' $NVHPC_STDLIB_OPT -DOMP:BOOL=ON $INT64_OPT -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja || echo '[WARN] nvhpc release configure failed'"
                fi
                launch_build "$build_rep/cmake-build-omp-nvhpc/$arch" $BUILD_FLAGS
            fi

            set_build_header 'Compiling nvhpc debug openmp:'
            if [ "$BUILD_ONLY" = false ]; then
                run_cmd "CXX='$nvcxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-nvhpc-debug/$arch -DCMAKE_CXX_COMPILER='$nvcxx' $NVHPC_STDLIB_OPT -DCMAKE_BUILD_TYPE=Debug -DOMP:BOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja || echo '[WARN] nvhpc debug configure failed'"
            fi
            launch_build "$build_rep/cmake-build-omp-nvhpc-debug/$arch" $BUILD_FLAGS
        fi
        run_cmd "module unload $latest_nvhpc"
    fi
fi
else
    echo "[SKIP] nvhpc builds disabled (enable_nvhpc=$enable_nvhpc)"
fi

if [ "$enable_oneapi" = true ] && [ "$is_arm" = false ]; then
    run_cmd "module unload inteloneapi/26.0.0"
fi

# LLVM Section
if [ "$enable_llvm" = true ]; then
set_build_header 'Compiling llvm openmp:'
reset_compiler_modules

latest_llvm=$(module av llvm 2>&1 | tr -s '[:space:]' '\n' | grep -oE 'llvm/[0-9]+(\.[0-9]+)*' | sort -V -u | tail -1)
if [ -z "$latest_llvm" ]; then
    echo "ERROR: no llvm module available on this system" >&2
    exit 1
fi

echo "Using $latest_llvm (latest available)"
run_cmd "module load $latest_llvm"
# Reload ninja in case loading LLVM unloaded it
run_cmd "module load ninja"

clangxx="$(command -v clang++ 2>/dev/null)"
if [ -z "$clangxx" ]; then
    # Dry-run (module load only echoed) or clang++ not yet in PATH: keep the
    # bare name. CMake resolves it in PATH at configure time, and the echoed
    # dry-run command stays copy-pastable into a shell with modules loaded —
    # the empty expansion used to bake CXX='' -DCMAKE_CXX_COMPILER= into the
    # output, which then failed with "No CMAKE_CXX_COMPILER could be found".
    clangxx="clang++"
fi
LLVM_CXX_OPT="-DCMAKE_CXX_COMPILER=$clangxx"

run_cmd "module load gcc/15.2.0"
GCC_TOOLCHAIN_DIR="/ccc/products/gcc-15.2.0/system/default"
LLVM_STDLIB_OPT="-DUSE_LIBCXX=OFF -DGCC_TOOLCHAIN_DIR=$GCC_TOOLCHAIN_DIR"
echo "llvm → libstdc++ from $GCC_TOOLCHAIN_DIR (USE_LIBCXX=OFF)"

if skip_release; then
   echo "[debug-only] skipping llvm release build"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON $INT64_OPT -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm/$arch"
fi

# Release + CHUNK_STATS: same flags as the plain release OMP build above, plus
# the instrumentation counters. Needed for anything that wants a still-fast
# binary with CHUNK_STATS on (e.g. bench_and_test.py's seq-vs-omp compare
# phase, which reads `still compressed (at exit)` from the run's stdout — the
# plain release build never prints it, since CHUNK_STATS is off there). Kept
# as its own target rather than always-on so ordinary release runs stay at
# the plain build's baseline perf. No -DCMAKE_BUILD_TYPE override, same as the
# plain target above: stays at CMake's own default (Release), it just adds
# the counters.
set_build_header 'Compiling llvm release openmp with stats (CHUNK_STATS):'
if skip_release; then
   echo "[debug-only] skipping llvm release-stats build"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-stats/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DCHUNK_STATS:BOOL=ON -DOMP:BOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm-stats/$arch"
fi

set_build_header 'Compiling llvm openmp with check (CHECK_OPTIM):'
if skip_release; then
   echo "[debug-only] skipping llvm openmp with check"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-check/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DCHECK_OPTIM=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm-check/$arch"
fi

set_build_header 'Compiling llvm debug openmp:'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug/$arch"

# Debug + CHUNK_STATS: the counters are cheap enough to keep in a debug build, and
# a debug binary is what you want when a counter disagrees with reality (gdb on the
# very path that increments it). Kept separate from the optimised -stats target.
set_build_header 'Compiling llvm debug openmp with stats (CHUNK_STATS):'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-stats/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DCHUNK_STATS:BOOL=ON -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-stats/$arch"

if [ "$enable_sanitizers" = true ]; then

# Token (alloc-token census) on the std_alloc axis only: it instruments
# malloc / operator new, and under fast_allocators the chunk buffers come from
# mmap'd slabs the census never sees.
set_build_header 'Compiling llvm debug openmp std_alloc token sanitizer:'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-std-alloc-token-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=token"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-std-alloc-token-sanitizer/$arch"

set_build_header 'Compiling llvm debug openmp thread sanitizer:'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-thread-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=thread"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-thread-sanitizer/$arch"

# (fast-allocator ASan build dropped, Sept '26: with the fast allocator the
# chunk buffers live inside the library's slab arenas, which ASan sees as a few
# large blocks -- an overflow or use-after-free inside a slab goes unreported.
# ASan runs on the std-alloc builds, where every buffer is a real malloc.)

set_build_header 'Compiling llvm debug openmp std_alloc thread sanitizer:'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-std-alloc-thread-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=thread"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-std-alloc-thread-sanitizer/$arch"

set_build_header 'Compiling llvm debug openmp std_alloc address sanitizer:'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-std-alloc-address-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=address"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-std-alloc-address-sanitizer/$arch"
else
    echo "[SKIP] sanitizer builds disabled (enable_sanitizers=$enable_sanitizers)"
fi

set_build_header 'Compiling llvm openmp std_alloc:'
if skip_release; then
   echo "[debug-only] skipping llvm std_alloc (release) build"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-std-alloc/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm-std-alloc/$arch"
fi

set_build_header 'Compiling llvm openmp have_agressive_compression:'
if skip_release; then
   echo "[debug-only] skipping llvm have_agressive_compression (release) build"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-have-agressive-compression/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DHAVE_AGRESSIVE_COMPRESSION:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm-have-agressive-compression/$arch"
fi

# Optimised thread-sanitizer build, same switches as the target that diverged
# (shrink_bench_chunk2a_256 --mode check after 2283 cycles, 2026-09-14): RelWithDebInfo
# so the scheduling and the inlining are the production ones, not the Debug
# ones -- the Debug TSan build above found nothing on that run --,
# HAVE_AGRESSIVE_COMPRESSION and CHECK_OPTIM as in the *_check_* binaries. LLVM
# libomp carries the TSan annotations for barriers / omp for / tasks (Archer),
# which is what makes the reports meaningful across region boundaries; the same
# build with libgomp reports a race at every region end. Hardening
# (CHUNK_GLIBCXX_ASSERTIONS) is added by default as for every -DSANITIZER build;
# --no-hardening removes it if it perturbs the timing too much.
if [ "$enable_sanitizers" = true ]; then
set_build_header 'Compiling llvm relwithdebinfo openmp have_agressive_compression thread sanitizer:'
if skip_release; then
   echo "[debug-only] skipping llvm relwithdebinfo have_agressive_compression thread-sanitizer build"
else
   if [ "$BUILD_ONLY" = false ]; then
      run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-relwithdebinfo-have-agressive-compression-thread-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DHAVE_AGRESSIVE_COMPRESSION:BOOL=ON -DCHECK_OPTIM=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=thread"
   fi
   launch_build "$build_rep/cmake-build-omp-llvm-relwithdebinfo-have-agressive-compression-thread-sanitizer/$arch"
fi
fi

# ─── llvm custom CHUNK_SIZE variants (large-chunk validation, on demand) ─────
# Two builds: RelWithDebInfo (perf/repro at scale) and Debug (-O0 -g, for gdb
# investigation of the large-chunk bugs). The Debug build is always produced
# when --llvm-cs is requested (that is the point of the flag); the release one
# is skipped under --debug-only like the other release builds.
if [ -n "$llvm_cs_size" ]; then
    echo -e "\nCompilation llvm openmp CHUNK_SIZE=$llvm_cs_size (RelWithDebInfo):\n"
    if skip_release; then
        echo "[SKIP] llvm cs$llvm_cs_size release (skipped by --debug-only)"
    else
        if [ "$BUILD_ONLY" = false ]; then
           run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-cs$llvm_cs_size/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_SIZE=$llvm_cs_size -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
        fi
        launch_build "$build_rep/cmake-build-omp-llvm-cs$llvm_cs_size/$arch"
    fi

    echo -e "\nCompilation llvm openmp CHUNK_SIZE=$llvm_cs_size (Debug):\n"
    if [ "$BUILD_ONLY" = false ]; then
       run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-cs$llvm_cs_size-debug/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_SIZE=$llvm_cs_size -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-omp-llvm-cs$llvm_cs_size-debug/$arch"
fi

set_build_header 'Compiling llvm debug:'
if [ "$BUILD_ONLY" = false ]; then
   # FIX: -DOMP:BOOL=OFF instead of -DOMP:OFF
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm-debug/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-llvm-debug/$arch"

# Serial debug + CHUNK_STATS: same idea without OpenMP, so a counter can be checked
# on the serial code path (no threads, no deferred ops) before blaming concurrency.
set_build_header 'Compiling llvm debug with stats (CHUNK_STATS):'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm-debug-stats/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DCHUNK_STATS:BOOL=ON -DOMP:BOOL=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-llvm-debug-stats/$arch"

# Non-OMP std-alloc Debug (no sanitizer): serial ledger repro of the rechunk
# double-free without the ASan slowdown. Runs deterministically at small sizes
# (e.g. --divisor 100), so it is the fast gdb/ledger path.
set_build_header 'Compiling llvm debug std_alloc (no openmp):'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm-debug-std-alloc/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=OFF -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-llvm-debug-std-alloc/$arch"

# OMP std-alloc Debug (no sanitizer): gdb + ledger repro of the rechunk
# double-free in the PARALLEL teardown/compression path (the crash fires from an
# omp worker, thr>0), without the ASan/TSan slowdown. Reproduces at small sizes
# with rechunk on (e.g. --divisor 4).
set_build_header 'Compiling llvm debug openmp std_alloc (no sanitizer):'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-llvm-debug-std-alloc/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=ON -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
fi
launch_build "$build_rep/cmake-build-omp-llvm-debug-std-alloc/$arch"

# Non-OMP std-alloc Debug + AddressSanitizer: the killer for the double-free.
# Serial + small + deterministic means ASan is fast here, and it prints the
# alloc / first-free / second-free sites directly (no guessing).
set_build_header 'Compiling llvm debug std_alloc address sanitizer (no openmp):'
if [ "$BUILD_ONLY" = false ]; then
   run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm-debug-std-alloc-address-sanitizer/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=OFF -DCHUNK_STD_ALLOC:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DSANITIZER=address"
fi
launch_build "$build_rep/cmake-build-llvm-debug-std-alloc-address-sanitizer/$arch"

# Non-OMP llvm RELEASE build. Builds the _check targets (declared
# unconditionally in CMakeLists; OMP only adds linkage) so bench_and_test can
# run CHECK on the non-OMP path — the one that showed the Chunk2A type-2
# divergence. Skipped by --debug-only like other release builds.
if skip_release; then
    echo "[debug-only] skipping llvm non-omp release build"
else
    if [ "$BUILD_ONLY" = false ]; then
       run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DOMP:BOOL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-llvm/$arch"
fi

# Sequential twin of cmake-build-omp-llvm-stats above: same release build,
# CHUNK_STATS on, OMP off. This is the other half of the seq-vs-omp compare
# pair bench_and_test.py's Phase 4 looks for (find_llvm_seq_omp_pairs derives
# "cmake-build-llvm-stats" from "cmake-build-omp-llvm-stats" by stripping
# "omp-"). Without both halves built, that phase has nothing to compare
# against and skips itself with a note.
set_build_header 'Compiling llvm release with stats (CHUNK_STATS):'
if skip_release; then
    echo "[debug-only] skipping llvm non-omp release-stats build"
else
    if [ "$BUILD_ONLY" = false ]; then
       run_cmd "CXX='$clangxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-llvm-stats/$arch $LLVM_CXX_OPT $LLVM_STDLIB_OPT -DCHUNK_STATS:BOOL=ON -DOMP:BOOL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-llvm-stats/$arch"
fi

else
    echo "[SKIP] llvm builds disabled (enable_llvm=$enable_llvm)"
fi

# ─── AOCC Section (AMD Optimizing C/C++ Compiler, clang-based — x86-only) ─────
if [ "$enable_aocc" = true ] && [ "$is_arm" = false ]; then
set_build_header 'Compiling aocc openmp:'
reset_compiler_modules

# Pick the newest aocc-compiler/<ver> module available (5.2.0 > 5.0.0 > 4.1.0).
latest_aocc=$(module av aocc 2>&1 | tr -s '[:space:]' '\n' | grep -oE 'aocc-compiler/[0-9]+(\.[0-9]+)*' | sort -V -u | tail -1)
if [ -z "$latest_aocc" ]; then
    echo "[SKIP] no aocc-compiler module available on this system"
else
    echo "Using $latest_aocc (latest available)"
    run_cmd "module load $latest_aocc"
    run_cmd "module load ninja"

    # After loading the aocc module, clang++ in PATH is AOCC's clang++.
    aoccxx="$(command -v clang++ 2>/dev/null)"
    [ -z "$aoccxx" ] && aoccxx="clang++"  # same dry-run fallback as clangxx
    AOCC_CXX_OPT="-DCMAKE_CXX_COMPILER=$aoccxx"

    # AOCC provides its own clang front-end but relies on the system libstdc++;
    # point it at the gcc toolchain, exactly like the llvm section does.
    run_cmd "module load gcc/15.2.0"
    GCC_TOOLCHAIN_DIR="/ccc/products/gcc-15.2.0/system/default"
    AOCC_STDLIB_OPT="-DUSE_LIBCXX=OFF -DGCC_TOOLCHAIN_DIR=$GCC_TOOLCHAIN_DIR"
    echo "aocc → libstdc++ from $GCC_TOOLCHAIN_DIR (USE_LIBCXX=OFF)"

    if skip_release; then
        echo "[debug-only] skipping aocc release build"
    else
        if [ "$BUILD_ONLY" = false ]; then
            run_cmd "CXX='$aoccxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-aocc/$arch $AOCC_CXX_OPT $AOCC_STDLIB_OPT -DOMP:BOOL=ON $INT64_OPT -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
        fi
        launch_build "$build_rep/cmake-build-omp-aocc/$arch" $BUILD_FLAGS
    fi

    set_build_header 'Compiling aocc debug openmp:'
    if [ "$BUILD_ONLY" = false ]; then
        run_cmd "CXX='$aoccxx' cmake -S ~/shrink-array -B $build_rep/cmake-build-omp-aocc-debug/$arch $AOCC_CXX_OPT $AOCC_STDLIB_OPT -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja"
    fi
    launch_build "$build_rep/cmake-build-omp-aocc-debug/$arch" $BUILD_FLAGS

    run_cmd "module unload $latest_aocc"
fi
else
    echo "[SKIP] aocc builds disabled (enable_aocc=$enable_aocc, is_arm=$is_arm)"
fi


# Wait for every build and print the summary as they finish.
echo -e '\n─── Waiting for remaining builds ───\n'
wait_all
rc=$?
# A --target / --exec that let nothing through is an error, not a quiet "OK".
if ! check_arg_epilogue; then
    echo -e "\nend — nothing built\n"
    exit 1
fi
if [ "$rc" -eq 0 ]; then
    echo -e "\nend — all builds OK\n"
else
    echo -e "\nend — $rc failed build(s) (logs: $LOG_DIR)\n"
fi
exit "$rc"
