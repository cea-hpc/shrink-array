#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# EXPERIMENT exp/optimistic-read: mode 7's correctness on the Grace node (AArch64).
# The validations so far ran on x86 only, whose TSO keeps the loads in order by
# itself; mode 7 rests on ARMv8's address dependencies. Two parts:
#   A  a check build (CHUNK_CHECK_OPTIM) in mode 7 per compiler: every ctest validation.
#   B  a Release build with the read-delay hook (CHUNK_TEST_READ_DELAY_EVERY) per
#      compiler x STRESS_MODES (default 7, and 3 as the control: no retry, so it must
#      FAIL -- otherwise the test does not catch a stale read on this machine):
#      read_reclaim_validation plain / bulk / compound, VARIANTS x THREADS.
# Then summary.txt: per run, the exit status against the expected one.
#   SRC   the exp/optimistic-read checkout (default: this script's repository)
#   WORK  (default: $CCCWORKDIR/optread_val1)
#   COMPS (default "gcc llvm")   PARTS (default "A B")   JOBS (default 64)
#   CTEST_JOBS (default 24: the tests run at 8 threads)
#   STRESS_MODES (default "7 3"; suffix s: -DCHUNK_GRACE_SYMMETRIC=1, the grace period
#                without membarrier, e.g. "7s 3s")   VARIANTS (default "zero zero_or_value zero_any")
#   THREADS (default "8 72")   CASES (default "plain bulk compound")
#   SECONDS_PER_RUN (default 20: READ_RECLAIM_SECONDS; on x86 the mode 3 plain control caught
#                    7 stale reads in 10 s, 1 in 2 s -- too few to trust less)
#   TIMING (default 0; 1: part B built with -DCHUNK_READ_RECLAIM_TIMING=1 -- each run's
#           "[grace]" lines, how long a pass holds its released buffers, follow its row)
#   READ_RECLAIM_DELAY_US (unset: the build's 50 us), passed through to the runs
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
WORK=${WORK:-${CCCWORKDIR:-$HOME}/optread_val1}
COMPS=${COMPS:-"gcc llvm"}
PARTS=${PARTS:-"A B"}
JOBS=${JOBS:-64}
CTEST_JOBS=${CTEST_JOBS:-24}
STRESS_MODES=${STRESS_MODES:-"7 3"}
VARIANTS=${VARIANTS:-"zero zero_or_value zero_any"}
THREADS=${THREADS:-"8 72"}
CASES=${CASES:-"plain bulk compound"}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-20}
TIMING=${TIMING:-0}
mkdir -p "$WORK/logs"
SUMMARY=$WORK/summary.txt
[ -f "$SUMMARY" ] && mv "$SUMMARY" "$SUMMARY.$(date +%Y%m%d-%H%M%S)"
echo "[val] $(hostname) $(uname -m) SRC=$SRC WORK=$WORK commit $(git -C "$SRC" log --oneline -1)" | tee "$SUMMARY"
echo "[val] COMPS=$COMPS PARTS=$PARTS STRESS_MODES=$STRESS_MODES VARIANTS=$VARIANTS THREADS=$THREADS" \
     "CASES=$CASES SECONDS_PER_RUN=$SECONDS_PER_RUN" | tee -a "$SUMMARY"
[ "$(uname -m)" = aarch64 ] || echo "[val] WARNING: not an aarch64 node" | tee -a "$SUMMARY"
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

# configure <comp> <build dir> <CXXFLAGS> <cmake args...>
configure() {
    local comp=$1 B=$2 flags=$3; shift 3
    local extra=()
    [ "$comp" = llvm ] && extra=(-DCHUNK_USE_LIBCXX=OFF -DCHUNK_GCC_TOOLCHAIN_DIR=$GCC_TC)
    cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DCHUNK_OPENMP=ON \
        "${extra[@]}" "-DCMAKE_CXX_FLAGS=$flags" "$@" > "$B.configure.log" 2>&1 \
        || { echo "[val] configure FAILED $B"; tail -20 "$B.configure.log"; return 1; }
    echo "[val] $(basename "$B"): $($CXX --version | head -1), CXXFLAGS $flags, $(grep 'Threading backend' "$B.configure.log")"
}

export OMP_PROC_BIND=close OMP_PLACES=cores

# ---- A: every validation, check build, mode 7 -------------------------------------------
case " $PARTS " in *" A "*)
for comp in $COMPS; do
  (
    load_env "$comp"
    B=$WORK/check-$comp-mode7
    configure "$comp" "$B" "-DCHUNK_READ_SECTION_MODE=7" -DCMAKE_BUILD_TYPE=Release -DCHUNK_CHECK_OPTIM=ON \
        -DCHUNK_BUILD_VALIDATION=ON || exit 1
    # The whole tree: the validation executables are most of it, and -k 0 lets one
    # failed target leave the others built (their tests still run).
    if ! ninja -C "$B" -j "$JOBS" -k 0 > "$B.build.log" 2>&1; then
        echo "[val] A $comp: build FAILED (tests of the built targets still run)" | tee -a "$SUMMARY"
        grep -E 'error:|FAILED:' "$B.build.log" | head -10 | tee -a "$SUMMARY"
    fi
    echo "[val] A $comp: ctest -L validation $(date +%T)"
    (cd "$B" && ctest -L validation -j "$CTEST_JOBS" --timeout 1800 --output-on-failure > "$WORK/logs/A_$comp.ctest.log" 2>&1)
    {
        echo "== A $comp mode 7 (check build): $(grep -E 'tests passed|tests failed' "$WORK/logs/A_$comp.ctest.log" | tail -1)"
        sed -n '/The following tests FAILED/,$p' "$WORK/logs/A_$comp.ctest.log" | head -40
    } | tee -a "$SUMMARY"
  )
done
;; esac

# ---- B: read_reclaim_validation under the read delay, mode 7 and the control ------------
case " $PARTS " in *" B "*)
for comp in $COMPS; do
  for m in $STRESS_MODES; do
    (
      load_env "$comp"
      B=$WORK/stress-$comp-mode$m
      # Release (bulk needs bulk_fast: no stats, no CHUNK_CHECK), the hooks compiled in.
      timing_flag=; [ "$TIMING" = 1 ] && timing_flag=" -DCHUNK_READ_RECLAIM_TIMING=1"
      sym_flag=; case $m in *s) sym_flag=" -DCHUNK_GRACE_SYMMETRIC=1" ;; esac
      configure "$comp" "$B" "-DCHUNK_READ_SECTION_MODE=${m%s} -DCHUNK_TEST_READ_DELAY_EVERY=16$timing_flag$sym_flag" \
          -DCMAKE_BUILD_TYPE=Release -DCHUNK_BUILD_VALIDATION=ON -DCHUNK_STATIC_SIZES=256 -DCHUNK_BUILD_DYN=OFF || exit 1
      tg=(); for v in $VARIANTS; do tg+=("validation_${v}_256"); done
      ninja -C "$B" -j "$JOBS" "${tg[@]}" > "$B.build.log" 2>&1 \
          || { echo "[val] B $comp mode $m: build FAILED" | tee -a "$SUMMARY"; tail -20 "$B.build.log"; exit 1; }
    ) || continue
  done
done

{
    echo
    echo "== B read_reclaim_validation, CHUNK_TEST_READ_DELAY_EVERY=16, ${SECONDS_PER_RUN} s per run," \
         "read delay ${READ_RECLAIM_DELAY_US:-build default} us, TIMING=$TIMING"
    echo "   expected: mode 7 -> 0 (pass); mode 3 plain / bulk -> 1 WITH stale reads (the control must"
    echo "   catch them: a FAIL premise -- no pass inside the region -- does not count); 77 = skipped"
    printf '%-5s %-4s %-14s %4s %-9s %4s %-14s %10s  %s\n' comp mode variant thr case rc verdict retries detail
} | tee -a "$SUMMARY"
n_ok=0; n_fail=0; n_ctl_ok=0; n_ctl_bad=0; n_skip=0
for comp in $COMPS; do
  for m in $STRESS_MODES; do
    bin=$WORK/stress-$comp-mode$m/examples/validation
    for v in $VARIANTS; do
      exe=$bin/validation_${v}_256
      [ -x "$exe" ] || { echo "[val] B $comp mode $m $v: no executable" | tee -a "$SUMMARY"; continue; }
      for t in $THREADS; do
        for c in $CASES; do
          arg=(); [ "$c" = plain ] || arg=("$c")
          log=$WORK/logs/B_${comp}_m${m}_${v}_t${t}_${c}.log
          (
            load_env "$comp" >/dev/null 2>&1
            OMP_NUM_THREADS=$t READ_RECLAIM_SECONDS=$SECONDS_PER_RUN timeout $((SECONDS_PER_RUN * 6 + 120)) \
                "$exe" read_reclaim_validation "${arg[@]}" > "$log" 2>&1
          )
          rc=$?
          # The harness's verdict line: OK / FAIL <n> stale / FAIL premise / SKIP.
          detail=$(grep -E 'read-reclaim[^]]*\] (OK:|FAIL [0-9]+ |FAIL premise|SKIP)' "$log" | tail -1 \
                   | sed 's/^.*\] //' | cut -c1-100)
          [ -n "$detail" ] || detail="(no verdict line: $(tail -1 "$log" | cut -c1-80))"
          if [ $rc -eq 77 ]; then verdict=skipped; n_skip=$((n_skip + 1))
          elif [ "${m%s}" = 3 ] && [ "$c" != compound ]; then
              if [ $rc -eq 1 ] && grep -qE 'FAIL stale' "$log"; then verdict="ok(control)"; n_ctl_ok=$((n_ctl_ok + 1))
              else verdict="CONTROL-BAD"; n_ctl_bad=$((n_ctl_bad + 1)); fi
          elif [ "${m%s}" = 3 ]; then verdict="info"
          elif [ $rc -eq 0 ]; then verdict=ok; n_ok=$((n_ok + 1))
          else verdict=FAIL; n_fail=$((n_fail + 1)); fi
          retries=$(grep -oE '\[optimistic\] reads retried [0-9]+' "$log" | awk '{print $NF}')
          printf '%-5s %-4s %-14s %4s %-9s %4s %-14s %10s  %s\n' "$comp" "$m" "$v" "$t" "$c" "$rc" "$verdict" \
              "${retries:--}" "$detail" | tee -a "$SUMMARY"
          [ "$TIMING" = 1 ] && grep -E '^\[grace\]' "$log" | sed 's/^/      /' | tee -a "$SUMMARY"
        done
      done
    done
  done
done
{
    echo
    echo "[val] B: $n_ok ok, $n_fail FAIL; controls (mode 3): $n_ctl_ok caught stale reads," \
         "$n_ctl_bad did not (CONTROL-BAD); $n_skip skipped"
} | tee -a "$SUMMARY"
;; esac

echo "[val] done: $SUMMARY, logs in $WORK/logs"
