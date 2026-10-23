#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# zero_any allocator diagnosis: build one configuration, then run it in a loop
# with a progress line.
# Usage: ./diag_alloc.sh <variant> [nb_runs]
#   variants: fast | std | noreuse | tsan
#   e.g.: ./diag_alloc.sh fast 100
#         ./diag_alloc.sh tsan 5
# Environment: CHUNK_BUILD_REP (where the build trees go; default: the
# repository), GCC_MODULE / GCC_TSAN_MODULE (the compiler modules; CEA defaults).
set -u

# ── Paths ────────────────────────────────────────────────────────────────────
SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${CHUNK_BUILD_REP:-$SRC}"
GCC_MODULE="${GCC_MODULE:-c++/gcc/15.2.0}"
GCC_TSAN_MODULE="${GCC_TSAN_MODULE:-c++/gcc/14.1.0}"
arch="$(source /etc/os-release && echo ${ID^^}_${VERSION_ID}__$(uname -m))"

ts(){ date +%H:%M:%S; }
step(){ echo "[$(ts)] >>> $*"; }

variant="${1:-}"; runs="${2:-100}"
case "$variant" in
  fast)    name=fastpur;  opt="";                         gccmod="$GCC_MODULE";      san="";     label="FAST only (slab, the crashing reference)" ;;
  std)     name=std;      opt="-DCHUNK_STD_ALLOC=ON";     gccmod="$GCC_MODULE";      san="";     label="STD (malloc, the clean reference)"        ;;
  noreuse) name=noreuse;  opt="-DCHUNK_POOL_NO_REUSE=ON"; gccmod="$GCC_MODULE";      san="";     label="NO_REUSE (new/delete)"                    ;;
  tsan)    name=tsan;     opt="-DCHUNK_STD_ALLOC=ON -DSANITIZER=thread"; gccmod="$GCC_TSAN_MODULE"; san="tsan"; label="TSan (oracle, STD)" ;;
  *) echo "usage: $0 {fast|std|noreuse|tsan} [nb_runs]"; exit 2 ;;
esac
[ "$variant" = "tsan" ] && runs="${2:-5}"

echo "══════════════════════════════════════════════════════════════════"
echo " [$(ts)] variant: $label"
echo "         module : $gccmod     runs=$runs threads=64"
echo "══════════════════════════════════════════════════════════════════"

# ── 0) are the sources up to date? ───────────────────────────────────────────
step "git: are the sources up to date?"
cd "$SRC" || { echo "!! SRC not found"; exit 1; }
git fetch origin >/dev/null 2>&1
LOCAL=$(git rev-parse @ 2>/dev/null); REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "?")
echo "         local HEAD : $(git log --oneline -1)"
if [ "$REMOTE" != "?" ] && [ "$LOCAL" != "$REMOTE" ]; then
  echo "  !! WARNING: local != origin -- 'git pull'? (local=$LOCAL remote=$REMOTE)"
fi

# ── 1) compiler module ───────────────────────────────────────────────────────
step "module load $gccmod"
module load $gccmod 2>/dev/null
echo "         g++ = $(which g++)  ($(g++ --version | head -1))"
if [ "$san" = "tsan" ]; then
  LT=$(find "$(dirname $(dirname $(which g++)))" -name 'libtsan.so*' 2>/dev/null | head -1)
  [ -z "$LT" ] && { echo "  !! no libtsan with this compiler: TSan impossible here"; exit 1; }
  echo "         libtsan = $LT"
  export LD_LIBRARY_PATH="$(dirname "$LT"):${LD_LIBRARY_PATH:-}"
fi

B="$WORK/cmake-build-$name/$arch"

# ── 2) configure ─────────────────────────────────────────────────────────────
step "cmake configure  (build dir: $B)"
rm -rf "$B"
extra=""
[ "$san" = "tsan" ] && extra='-DCMAKE_CXX_COMPILER='"$(which g++)"' -DCMAKE_C_COMPILER='"$(which gcc)"' -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread'
if ! cmake -S "$SRC" -B "$B" -DOMP:BOOL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $opt $extra >/tmp/cfg_$name.log 2>&1; then
  echo "  !! CONFIGURE FAILED:"; tail -20 /tmp/cfg_$name.log; exit 1
fi
grep -q "Manually-specified variables were not used" /tmp/cfg_$name.log \
  && echo "  !! option ignored ($opt): sources not up to date?"
echo "         configure OK"

# ── 3) build (timed) ─────────────────────────────────────────────────────────
step "cmake build  (compiling, be patient...)"
t0=$(date +%s)
if ! cmake --build "$B" --target shrink_bench_zero_any_dyn -j >/tmp/build_$name.log 2>&1; then
  echo "  !! BUILD FAILED:"; tail -25 /tmp/build_$name.log; exit 1
fi
echo "         build OK in $(( $(date +%s)-t0 ))s"
# Every target carries its size (_256, _1024, _dyn); the perf mode is a
# run-time flag (--mode), not part of the name.
exe="$B/examples/shrink_bench/shrink_bench_zero_any_dyn"
[ -x "$exe" ] || { echo "  !! no executable"; exit 1; }
if [ "$san" = "tsan" ]; then
  echo "         TSan instrumentation: $(nm "$exe" 2>/dev/null | grep -c __tsan_func_entry) __tsan symbols (must be > 0)"
fi

# ── 4) runs ──────────────────────────────────────────────────────────────────
step "starting $runs runs (64 threads) -- about 1-2 min a run"
cd "$B/examples/shrink_bench"; ulimit -c 0
c=0 races=0 T0=$(date +%s)
for i in $(seq 1 "$runs"); do
  r0=$(date +%s)
  if [ "$san" = "tsan" ]; then
    TSAN_OPTIONS="history_size=7:halt_on_error=0" OMP_NUM_THREADS=64 ./shrink_bench_zero_any_dyn --mode perf >/tmp/${name}_$i.log 2>&1
    nr=$(grep -c "data race" /tmp/${name}_$i.log); races=$((races+nr)); st="races=$nr"
  else
    OMP_NUM_THREADS=64 ./shrink_bench_zero_any_dyn --mode perf >/dev/null 2>&1 && st="ok" || { st="CRASH"; c=$((c+1)); }
  fi
  el=$(( $(date +%s)-T0 )); rt=$(( $(date +%s)-r0 ))
  # rough ETA
  avg=$(( el / i )); eta=$(( avg * (runs - i) ))
  printf "\r  [%s] run %3d/%s | %-9s | crash %3d | this run %3ds | elapsed %4ds | ETA ~%4ds  " \
         "$(ts)" "$i" "$runs" "$st" "$c" "$rt" "$el" "$eta"
done
echo ""
echo "══════════════════════════════════════════════════════════════════"
if [ "$san" = "tsan" ]; then
  echo " [$(ts)] TSan RESULT: $races 'data race' reports over $runs runs"
  echo " Race sites (grouped):"
  grep -h -A3 "SUMMARY: ThreadSanitizer" /tmp/${name}_*.log 2>/dev/null \
    | grep -oE "chunk[a-z_0-9]*\.h:[0-9]+|unordered_dense\.h:[0-9]+|flat_hash_map\.hpp:[0-9]+|compression_manager[a-z_]*\.h:[0-9]+" \
    | sort | uniq -c | sort -rn | sed 's/^/    /'
else
  echo " [$(ts)] RESULT $label: $c crash(es) / $runs   (in $(( $(date +%s)-T0 ))s)"
fi
echo "══════════════════════════════════════════════════════════════════"
