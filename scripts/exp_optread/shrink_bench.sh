#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# EXPERIMENT exp/optimistic-read: shrink_bench, mode 0 against mode 7, on the Grace node.
# One build per CONFIGS entry (the same configure as run.sh), then ROUNDS rounds of
# `shrink_bench --mode perf` interleaving the builds. Per run: the compute phase's time,
# the wall time, the peak RSS (shrink_bench's own figure, and the kernel's when
# /usr/bin/time exists) and, in mode 7, the reads retried. Then aggregate_bench.py's table.
#   SRC      the exp/optimistic-read checkout (default: this script's repository)
#   WORK     builds, logs and CSV (default: $CCCWORKDIR/optread_sb1)
#   CONFIGS  (default "gcc:0 gcc:7 llvm:0 llvm:7"), as run.sh: <compiler>:<mode>[d][i]
#   VARIANTS (default "zero zero_or_value zero_any")
#   THREADS  (default "8 72"; 72 = one Grace socket)
#   ROUNDS   (default 3)   JOBS (default 32)   ARGS (default "--mode perf")
#   SEED     (default 20261126): every run draws the same intervals. shrink_bench
#            takes its seed from seed<iter_max>.txt in its working directory and,
#            without one, draws a new seed from std::random_device.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
WORK=${WORK:-${CCCWORKDIR:-$HOME}/optread_sb1}
CONFIGS=${CONFIGS:-"gcc:0 gcc:7 llvm:0 llvm:7"}
VARIANTS=${VARIANTS:-"zero zero_or_value zero_any"}
THREADS=${THREADS:-"8 72"}
ROUNDS=${ROUNDS:-3}
JOBS=${JOBS:-32}
ARGS=${ARGS:-"--mode perf"}
SEED=${SEED:-20261126}
# The seed file's name follows --iter_max (shrink_bench's default 2000).
ITER=2000
read -r -a _args <<< "$ARGS"
for ((i = 0; i + 1 < ${#_args[@]}; ++i)); do [ "${_args[i]}" = --iter_max ] && ITER=${_args[i + 1]}; done
mkdir -p "$WORK/logs"
CSV=$WORK/shrink_bench.csv
# A fresh CSV: the table is built from this one file.
[ -f "$CSV" ] && mv "$CSV" "$CSV.$(date +%Y%m%d-%H%M%S)"
echo "comp,mode,variant,threads,round,status,compute_s,wall_s,peak_mo,kernel_peak_mo,retries" > "$CSV"
echo "[sbench] $(hostname) $(uname -m) SRC=$SRC WORK=$WORK commit $(git -C "$SRC" log --oneline -1)"
echo "[sbench] CONFIGS=$CONFIGS VARIANTS=$VARIANTS THREADS=$THREADS ROUNDS=$ROUNDS ARGS=$ARGS SEED=$SEED (seed$ITER.txt)"
[ "$(uname -m)" = aarch64 ] || echo "[sbench] WARNING: not an aarch64 node"
source "$SRC/scripts/site_env.sh"

# load_env gcc|llvm: as run.sh (the compiler's modules, in a clean module state).
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

# build_dir <cfg>: where run.sh would put it, under this WORK.
build_dir() { local comp=${1%%:*} mode=${1#*:}; echo "$WORK/$comp-mode$mode"; }

targets=()
for v in $VARIANTS; do targets+=("shrink_bench_omp_${v}_256"); done

for cfg in $CONFIGS; do
  (
    comp=${cfg%%:*}; mode=${cfg#*:}; m=${mode%%[!0-9]*}; suffix=${mode#"$m"}
    flags=-DCHUNK_READ_SECTION_MODE=$m
    case $suffix in *d*) flags="$flags -DCHUNK_READ_DEP_ORDER=1" ;; esac
    case $suffix in *i*) flags="$flags -DCHUNK_RECHECK_INT_LOAD=1" ;; esac
    case $suffix in *[!di]*) echo "[sbench] unknown suffix in $cfg"; exit 1 ;; esac
    load_env "$comp"
    echo "[sbench] $cfg: $($CXX --version | head -1), CXXFLAGS $flags"
    B=$(build_dir "$cfg")
    extra=()
    [ "$comp" = llvm ] && extra=(-DCHUNK_USE_LIBCXX=OFF -DCHUNK_GCC_TOOLCHAIN_DIR=$GCC_TC)
    cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
        -DCHUNK_OPENMP=ON -DCHUNK_STATIC_SIZES=256 -DCHUNK_BUILD_DYN=OFF -DCHUNK_BUILD_VALIDATION=OFF \
        "${extra[@]}" "-DCMAKE_CXX_FLAGS=$flags" > "$B.configure.log" 2>&1 \
        || { echo "[sbench] configure FAILED $cfg"; tail -20 "$B.configure.log"; exit 1; }
    echo "[sbench] $cfg $(grep 'Threading backend' "$B.configure.log")"
    ninja -C "$B" -j "$JOBS" "${targets[@]}" > "$B.build.log" 2>&1 \
        || { echo "[sbench] build FAILED $cfg"; tail -20 "$B.build.log"; exit 1; }
  ) || exit 1
done

TIME_BIN=
[ -x /usr/bin/time ] && TIME_BIN=/usr/bin/time
export OMP_PROC_BIND=close OMP_PLACES=cores
for r in $(seq 1 "$ROUNDS"); do
  for v in $VARIANTS; do
    for t in $THREADS; do
      for cfg in $CONFIGS; do
        (
          comp=${cfg%%:*}; mode=${cfg#*:}
          load_env "$comp" >/dev/null 2>&1
          bin="$(build_dir "$cfg")/examples/shrink_bench/shrink_bench_omp_${v}_256"
          tag=${comp}_m${mode}_${v}_t${t}_r${r}
          log=$WORK/logs/$tag.log
          # Its own working directory: shrink_bench writes memory_mo.txt (and the
          # reference files in check mode) where it runs.
          run=$WORK/run.$tag; rm -rf "$run"; mkdir -p "$run"; cd "$run" || exit 1
          printf '%s' "$SEED" > "seed$ITER.txt"
          echo "[sbench] round $r $v $t thr $cfg $(date +%T)"
          t0=$(date +%s.%N)
          # shellcheck disable=SC2086  # ARGS: several arguments
          if [ -n "$TIME_BIN" ]; then
              OMP_NUM_THREADS=$t "$TIME_BIN" -f '[time] maxrss_kib %M' "$bin" $ARGS > "$log" 2>&1
          else
              OMP_NUM_THREADS=$t "$bin" $ARGS > "$log" 2>&1
          fi
          rc=$?
          t1=$(date +%s.%N)
          cd "$WORK" && rm -rf "$run"
          status=ok; [ $rc -eq 0 ] || { status=rc$rc; echo "[sbench] FAILED $tag (rc $rc)"; tail -5 "$log"; }
          compute=$(grep -oE 'Elapsed time computation : [0-9.e+-]+' "$log" | tail -1 | awk '{print $NF}')
          peak=$(grep -oE 'Memory consumption end : [0-9.]+ Mo \(peak\)' "$log" | tail -1 | awk '{print $5}')
          kpeak=$(grep -oE '\[time\] maxrss_kib [0-9]+' "$log" | awk '{printf "%.1f", $3 / 1024}')
          retries=$(grep -oE '\[optimistic\] reads retried [0-9]+' "$log" | awk '{print $NF}')
          wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')
          echo "$comp,$mode,$v,$t,$r,$status,${compute:-},$wall,${peak:-},${kpeak:-},${retries:-}" >> "$CSV"
        )
      done
    done
  done
done
PY=$(command -v python3.11 || command -v python3)
"$PY" "$HERE/aggregate_bench.py" "$CSV" | tee "$WORK/result_shrink_bench.txt"
echo "[sbench] done: $WORK/result_shrink_bench.txt, logs in $WORK/logs"
