#!/bin/bash
# Runs on the Grace node: 2 compilers x CHUNK_READ_SECTION_MODE 0/1/2, then
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# access_cost in 3 rounds interleaving the 6 builds, then the table.
#   SRC  the exp/optimistic-read checkout (default: this script's repository)
#   WORK where the builds and the CSVs go (default: $CCCWORKDIR/optread)
#   THREADS (default 1,72)   N (default 33554432)   JOBS (default 32)
#   CONFIGS (default "gcc:0 gcc:1 gcc:3 gcc:4")   ROUNDS (default 2)

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
WORK=${WORK:-${CCCWORKDIR:-$HOME}/optread11}
THREADS=${THREADS:-1,72}
N=${N:-33554432}
JOBS=${JOBS:-32}
mkdir -p "$WORK/csv"
# A fresh CSV set: the table is built from every CSV there.
[ -n "$(ls "$WORK/csv" 2>/dev/null)" ] && mv "$WORK/csv" "$WORK/csv.$(date +%Y%m%d-%H%M%S)" && mkdir -p "$WORK/csv"
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

# CONFIGS: <compiler>:<mode>[d], d = also -DCHUNK_READ_DEP_ORDER=1 (relaxed `values` load, AArch64)
CONFIGS=${CONFIGS:-"gcc:0 gcc:7 llvm:0 llvm:7"}
ROUNDS=${ROUNDS:-3}
LOOP_SYM=_ZN12_GLOBAL__N_18arr_readERN12shrink_array13zero_or_value5ArrayIdiEEi._omp_fn.0
for cfg in $CONFIGS; do
  (
    comp=${cfg%%:*}; mode=${cfg#*:}; m=${mode%d}
    flags=-DCHUNK_READ_SECTION_MODE=$m
    [ "$mode" != "$m" ] && flags="$flags -DCHUNK_READ_DEP_ORDER=1"
    load_env $comp
    echo "[optread] $cfg: $($CXX --version | head -1), CXXFLAGS $flags"
    B=$WORK/$comp-mode$mode
    extra=()
    [ $comp = llvm ] && extra=(-DCHUNK_USE_LIBCXX=OFF -DCHUNK_GCC_TOOLCHAIN_DIR=$GCC_TC)
    cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
        -DCHUNK_OPENMP=ON -DCHUNK_STATIC_SIZES=256 -DCHUNK_BUILD_DYN=OFF -DCHUNK_BUILD_VALIDATION=OFF \
        "${extra[@]}" "-DCMAKE_CXX_FLAGS=$flags" > "$B.configure.log" 2>&1 \
        || { echo "[optread] configure FAILED $cfg"; tail -20 "$B.configure.log"; exit 1; }
    echo "[optread] $cfg $(grep 'Threading backend' "$B.configure.log")"
    ninja -C "$B" -j "$JOBS" access_cost_zero_256 access_cost_zero_or_value_256 access_cost_zero_any_256 \
        > "$B.build.log" 2>&1 || { echo "[optread] build FAILED $cfg"; tail -20 "$B.build.log"; exit 1; }
    # What the read loop is made of: calls (an inlining miss), and the ordering instructions.
    bin="$B/examples/access_cost/access_cost_zero_or_value_256"
    # The mangled name differs between compilers (gcc 16: ..._1L8arr_read...): found by nm.
    sym=$(nm "$bin" | grep -E '8arr_read.*(omp_fn\.0|omp_outlined)$' | awk '{print $3}' | head -1)
    objdump -d --no-show-raw-insn --disassemble="${sym:-$LOOP_SYM}" "$bin" > "$B.loop.s" 2>/dev/null
    echo "[optread] $cfg loop: $(grep -cE '\sbl\s' "$B.loop.s") bl, $(grep -c dmb "$B.loop.s") dmb," \
         "$(grep -c stlr "$B.loop.s") stlr, $(grep -cE 'ldar\s' "$B.loop.s") ldar, $(grep -c ldapr "$B.loop.s") ldapr" \
         "($(grep -c . "$B.loop.s") lines, $B.loop.s)"
  ) || exit 1
done

PY=$(command -v python3.11 || command -v python3)
for r in $(seq 1 $ROUNDS); do
  for cfg in $CONFIGS; do
    (
      comp=${cfg%%:*}; mode=${cfg#*:}
      load_env $comp >/dev/null 2>&1
      echo "[optread] round $r $cfg $(date +%T)"
      "$PY" "$SRC/python/access_cost.py" --build "$WORK/$comp-mode$mode" --threads "$THREADS" --rounds 1 \
          --n "$N" --reps 5 --csv "$WORK/csv/${comp}_m${mode}_r${r}.csv" > "$WORK/csv/${comp}_m${mode}_r${r}.txt" 2>&1 \
          || { echo "[optread] run FAILED $cfg $r"; tail -5 "$WORK/csv/${comp}_m${mode}_r${r}.txt"; }
    )
  done
done
"$PY" "$HERE/aggregate.py" "$WORK/csv" | tee "$WORK/result.txt"
echo "[optread] done: $WORK/result.txt"
