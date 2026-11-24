#!/bin/bash
# Runs on the Grace node: 2 compilers x CHUNK_READ_SECTION_MODE 0/1/2, then
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# access_cost in 3 rounds interleaving the 6 builds, then the table.
#   SRC  the exp/optimistic-read checkout (default: this script's repository)
#   WORK where the builds and the CSVs go (default: $CCCWORKDIR/optread)
#   THREADS (default 1,72)   N (default 33554432)   JOBS (default 32)

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
WORK=${WORK:-${CCCWORKDIR:-$HOME}/optread}
THREADS=${THREADS:-1,72}
N=${N:-33554432}
JOBS=${JOBS:-32}
mkdir -p "$WORK/csv"
echo "[optread] $(hostname) $(uname -m) SRC=$SRC WORK=$WORK commit $(git -C "$SRC" log --oneline -1)"
[ "$(uname -m)" = aarch64 ] || echo "[optread] WARNING: not an aarch64 node"
source "$SRC/scripts/site_env.sh"

# load_env gcc|llvm: the compiler's modules, in a clean module state; sets CXX / CC.
load_env() {
    module purge >/dev/null 2>&1
    chunk_module_load_first cmake/4.0.3 cmake/3.31.4 cmake
    local gcc_mod; gcc_mod=$(chunk_newest_module gcc)
    if [ "$1" = gcc ]; then
        module load "$gcc_mod"
        local bu; bu=$(chunk_newest_module binutils); [ -n "$bu" ] && module load "$bu"
        CXX=$(command -v g++); CC=$(command -v gcc)
    else
        if module -t avail gcc/15.2.0 2>&1 | grep -q '^gcc/15\.2\.0'; then gcc_mod=gcc/15.2.0; fi
        module load "$gcc_mod"
        GCC_TC=$(dirname "$(dirname "$(readlink -f "$(command -v g++)")")")
        module load "$(chunk_newest_module llvm)"
        CXX=$(command -v clang++); CC=$(command -v clang)
    fi
    module load ninja >/dev/null 2>&1; chunk_ensure_ninja
    export CXX CC
}

for comp in gcc llvm; do
  (
    load_env $comp
    echo "[optread] $comp: $($CXX --version | head -1)"
    for m in 0 1 2; do
        B=$WORK/$comp-mode$m
        extra=()
        [ $comp = llvm ] && extra=(-DCHUNK_USE_LIBCXX=OFF -DCHUNK_GCC_TOOLCHAIN_DIR=$GCC_TC)
        cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
            -DCHUNK_OPENMP=ON -DCHUNK_STATIC_SIZES=256 -DCHUNK_BUILD_DYN=OFF -DCHUNK_BUILD_VALIDATION=OFF \
            "${extra[@]}" -DCMAKE_CXX_FLAGS=-DCHUNK_READ_SECTION_MODE=$m > "$B.configure.log" 2>&1 \
            || { echo "[optread] configure FAILED $comp $m"; tail -20 "$B.configure.log"; exit 1; }
        grep 'Threading backend' "$B.configure.log"
        ninja -C "$B" -j "$JOBS" access_cost_zero_256 access_cost_zero_or_value_256 access_cost_zero_any_256 \
            > "$B.build.log" 2>&1 || { echo "[optread] build FAILED $comp $m"; tail -20 "$B.build.log"; exit 1; }
        echo "[optread] $comp mode $m built"
    done
  ) || exit 1
done

PY=$(command -v python3.11 || command -v python3)
for r in 1 2 3; do
  for comp in gcc llvm; do
    for m in 0 1 2; do
      (
        load_env $comp >/dev/null 2>&1
        echo "[optread] round $r $comp mode $m $(date +%T)"
        "$PY" "$SRC/python/access_cost.py" --build "$WORK/$comp-mode$m" --threads "$THREADS" --rounds 1 \
            --n "$N" --reps 5 --csv "$WORK/csv/${comp}_m${m}_r${r}.csv" > "$WORK/csv/${comp}_m${m}_r${r}.txt" 2>&1 \
            || { echo "[optread] run FAILED $comp $m $r"; tail -5 "$WORK/csv/${comp}_m${m}_r${r}.txt"; }
      )
    done
  done
done
"$PY" "$HERE/aggregate.py" "$WORK/csv" | tee "$WORK/result.txt"
echo "[optread] done: $WORK/result.txt"
