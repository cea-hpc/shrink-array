#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
# device_check.sh -- build and run the device validations and device_bench on
# the GPU node it runs on: the commands of a GH200 or MI300A check in one file,
# instead of a script pasted into a terminal.
#
#   scripts/device_check.sh cuda|hip [--no-bench] [--repeat N] [--jobs N]
#                           [--kokkos <kokkos-5.3.0.tar.gz>]
#
# --kokkos also builds the Kokkos facade (-DCHUNK_DEVICE_KOKKOS=ON, Kokkos from the
# tarball given: the compute nodes do not reach GitHub) in build-kokkos-<lang>-<arch>
# and runs kokkos_validation. KOKKOS_ARCH overrides the Kokkos architecture
# (default: HOPPER90, AMPERE86 or TURING75 from CUDA_ARCH; AMD_GFX942_APU for hip).
# --jobs: compilations at once, 8 by default -- a step that does not hold the whole
# node has little memory, and 48 HIP compilations of Kokkos were killed (MI300A).
#
# Run it ON the GPU node (the GH200's host is aarch64: a build made on an x86
# login node does not run there): the build tree is build-device-<lang>-<arch>
# in the repository, configured once, then only rebuilt. The logs go to
# device-check-<lang>-<commit>/ in the repository.
#
# The toolchain comes from the environment (module load before, or the
# variables below); the script loads nothing.
#   cuda: nvcc on PATH (or CMAKE_CUDA_COMPILER), a g++ it accepts;
#         CUDA_ARCH, default 90 (GH200)
#   hip:  ROCM_PATH (default: from the clang++ on PATH, else /opt/rocm);
#         HIP_ARCH, default gfx942 (MI300A). The root is given to CMake
#         (CMAKE_HIP_COMPILER_ROCM_ROOT) and to clang (--rocm-path): CMake takes
#         it from `clang++ -v -print-targets`, without the flags, and ROCm
#         10.2's clang reports llvm/bin/../../.. -- the layout with
#         lib/llvm/bin, one level too high where llvm/ sits at the root (inti's
#         module); its driver also looked for libamdhip64.so there.
#
# Exit status: 0 when every validation passes (and the repeats of
# device_assign_validation), 1 otherwise.
# ─────────────────────────────────────────────────────────────────────────────
set -u

lang=${1:-}
shift || true
bench=1
repeat=10
jobs=8
kokkos=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-bench) bench=0 ;;
        --repeat) repeat=$2; shift ;;
        --jobs) jobs=$2; shift ;;
        --kokkos) kokkos=$(readlink -f "$2"); shift ;;
        *) echo "device_check.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
case "$lang" in
    cuda | hip) ;;
    *) echo "usage: scripts/device_check.sh cuda|hip [--no-bench] [--repeat N] [--jobs N] [--kokkos <tarball>]" >&2; exit 2 ;;
esac
if [ -n "$kokkos" ] && [ ! -f "$kokkos" ]; then
    echo "device_check.sh: --kokkos: no file $kokkos" >&2
    exit 2
fi

src=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$src" || exit 1
commit=$(git log -1 --format=%h 2>/dev/null || echo unknown)
build=build-device-$lang-$(uname -m)
logs=device-check-$lang-$commit
mkdir -p "$logs"
gxx=$(command -v g++)
gen=()
command -v ninja > /dev/null && gen=(-G Ninja)

echo "device_check: $lang, repo $src at $(git log --oneline -1 2>/dev/null), host $(uname -m), build $build"

targets="device_assign_validation device_write_validation device_arena_validation device_api_validation device_convert_validation device_memory_validation_zero_any"
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$gxx"
            -DCHUNK_BUILD_VALIDATION=ON -DCHUNK_DEVICE_CORE_CHECK=ON)
if [ "$lang" = cuda ]; then
    command -v nvidia-smi > /dev/null && nvidia-smi -L
    targets="$targets device_core_validation"
    cmake_args+=(-DCHUNK_DEVICE_LANGUAGE=CUDA -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-90}"
                 -DCMAKE_CUDA_HOST_COMPILER="$gxx")
else
    if [ -z "${ROCM_PATH:-}" ]; then
        clang=$(command -v clang++ || true)
        ROCM_PATH=$([ -n "$clang" ] && dirname "$(dirname "$(dirname "$clang")")" || echo /opt/rocm)
    fi
    command -v rocminfo > /dev/null && rocminfo | grep -m1 -o "gfx9[0-9a-f]*"
    gcc_dir=$(dirname "$("$gxx" -print-libgcc-file-name)")
    cmake_args+=(-DCHUNK_DEVICE_LANGUAGE=HIP -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH:-gfx942}"
                 -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_PREFIX_PATH="$ROCM_PATH"
                 -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH"
                 "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM_PATH --gcc-install-dir=$gcc_dir")
    echo "device_check: ROCM_PATH=$ROCM_PATH"
fi

if [ ! -f "$build/CMakeCache.txt" ]; then
    if ! cmake -S . -B "$build" "${gen[@]}" "${cmake_args[@]}" > "$logs/cmake.log" 2>&1; then
        tail -30 "$logs/cmake.log"
        rm -rf "$build" # the next run configures afresh
        echo "device_check: configure failed ($logs/cmake.log)"
        exit 1
    fi
fi
# shellcheck disable=SC2086 # the target list splits on purpose
if ! cmake --build "$build" -j "$jobs" --target $targets device_bench > "$logs/build.log" 2>&1; then
    tail -30 "$logs/build.log"
    echo "device_check: build failed ($logs/build.log)"
    exit 1
fi

bin=$build/examples
status=0
for t in $targets; do
    timeout 900 "$bin/validation/$t" > "$logs/$t.log" 2>&1
    rc=$?
    echo "$t rc=$rc $(tail -1 "$logs/$t.log")"
    [ $rc -eq 0 ] || status=1
done
fails=0
for k in $(seq 1 "$repeat"); do
    timeout 900 "$bin/validation/device_assign_validation" > "$logs/assign_repeat.log" 2>&1 || {
        echo "device_assign_validation repeat $k rc=$?"
        fails=$((fails + 1))
    }
done
echo "device_assign_validation repeats: $fails failure(s) out of $repeat"
[ $fails -eq 0 ] || status=1

if [ $bench = 1 ]; then
    "$bin/device_bench/device_bench" 30 10 > "$logs/device_bench.log" 2>&1
    echo "device_bench rc=$?"
    grep -E "^== S|assign\(\)|claim waits" "$logs/device_bench.log"
fi
if [ -n "$kokkos" ]; then
    kbuild=build-kokkos-$lang-$(uname -m)
    if [ "$lang" = cuda ]; then
        case "${CUDA_ARCH:-90}" in
            90) karch=HOPPER90 ;;
            86) karch=AMPERE86 ;;
            80) karch=AMPERE80 ;;
            75) karch=TURING75 ;;
            *) karch="" ;;
        esac
    else
        karch=AMD_GFX942_APU
    fi
    karch=${KOKKOS_ARCH:-$karch}
    if [ -z "$karch" ]; then
        echo "device_check: --kokkos: set KOKKOS_ARCH for CUDA_ARCH=${CUDA_ARCH}"
        exit 1
    fi
    if [ ! -f "$kbuild/CMakeCache.txt" ]; then
        if ! cmake -S . -B "$kbuild" "${gen[@]}" "${cmake_args[@]}" -DCHUNK_DEVICE_KOKKOS=ON \
            -DCHUNK_KOKKOS_ARCH="$karch" -DCHUNK_KOKKOS_URL="$kokkos" -DCHUNK_STATIC_SIZES=256 \
            -DCHUNK_BUILD_DYN=OFF > "$logs/cmake_kokkos.log" 2>&1; then
            tail -30 "$logs/cmake_kokkos.log"
            rm -rf "$kbuild"
            echo "device_check: Kokkos configure failed ($logs/cmake_kokkos.log)"
            exit 1
        fi
    fi
    if ! cmake --build "$kbuild" -j "$jobs" --target kokkos_validation > "$logs/build_kokkos.log" 2>&1; then
        tail -30 "$logs/build_kokkos.log"
        echo "device_check: Kokkos build failed ($logs/build_kokkos.log; 'Killed': fewer --jobs)"
        exit 1
    fi
    timeout 900 "$kbuild/examples/kokkos/kokkos_validation" > "$logs/kokkos_validation.log" 2>&1
    rc=$?
    echo "kokkos_validation ($karch) rc=$rc $(tail -1 "$logs/kokkos_validation.log")"
    [ $rc -eq 0 ] || status=1
fi
echo "device_check: logs in $src/$logs"
exit $status
