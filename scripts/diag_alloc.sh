#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# Diagnostic Chunk2A aarch64 — build + runs avec progression détaillée.
# Usage : ./diag_alloc.sh <variante> [nb_runs]
#   variantes : fast | std | noreuse | op | ctx | tsan
#   ex : ./diag_alloc.sh fast 100
#        ./diag_alloc.sh tsan 5
set -u

# ── Chemins (édite si besoin) ────────────────────────────────────────────────
SRC=$HOME/shrink-array
WORK=$CCCWORKDIR/shrink-array
arch="$(source /etc/os-release && echo ${ID^^}_${VERSION_ID}__$(uname -m))"

ts(){ date +%H:%M:%S; }
step(){ echo "[$(ts)] >>> $*"; }

variant="${1:-}"; runs="${2:-100}"
case "$variant" in
  fast)    name=fastpur;  opt="";                         gccmod="c++/gcc/15.2.0"; san="";       label="FAST pur (slab, réf crash)"      ;;
  std)     name=std;      opt="-DCHUNK_STD_ALLOC=ON";     gccmod="c++/gcc/15.2.0"; san="";       label="STD (malloc, réf clean)"         ;;
  noreuse) name=noreuse;  opt="-DCHUNK_POOL_NO_REUSE=ON"; gccmod="c++/gcc/15.2.0"; san="";       label="NO_REUSE (new/delete)"           ;;
  op)      name=siteop;   opt="-DCHUNK_STD_ALLOC_OP=ON";  gccmod="c++/gcc/15.2.0"; san="";       label="OP->malloc, ctx slab"            ;;
  ctx)     name=sitectx;  opt="-DCHUNK_STD_ALLOC_CTX=ON"; gccmod="c++/gcc/15.2.0"; san="";       label="CTX->malloc, op slab"            ;;
  tsan)    name=tsan;     opt="-DCHUNK_STD_ALLOC=ON -DSANITIZER=thread"; gccmod="c++/gcc/14.1.0"; san="tsan"; label="TSan (oracle, STD)"      ;;
  *) echo "usage: $0 {fast|std|noreuse|op|ctx|tsan} [nb_runs]"; exit 2 ;;
esac
[ "$variant" = "tsan" ] && runs="${2:-5}"

echo "══════════════════════════════════════════════════════════════════"
echo " [$(ts)] variante : $label"
echo "         module   : $gccmod     runs=$runs threads=64"
echo "══════════════════════════════════════════════════════════════════"

# ── 0) source à jour ? ───────────────────────────────────────────────────────
step "git : vérification de la fraîcheur des sources"
cd "$SRC" || { echo "!! SRC introuvable"; exit 1; }
git fetch origin >/dev/null 2>&1
LOCAL=$(git rev-parse @ 2>/dev/null); REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "?")
echo "         HEAD local  : $(git log --oneline -1)"
if [ "$REMOTE" != "?" ] && [ "$LOCAL" != "$REMOTE" ]; then
  echo "  !! ATTENTION : local != origin — pense à 'git pull' (local=$LOCAL remote=$REMOTE)"
fi

# ── 1) module compilateur ────────────────────────────────────────────────────
step "module load $gccmod"
module load $gccmod 2>/dev/null
echo "         g++ = $(which g++)  ($(g++ --version | head -1))"
if [ "$san" = "tsan" ]; then
  LT=$(find "$(dirname $(dirname $(which g++)))" -name 'libtsan.so*' 2>/dev/null | head -1)
  [ -z "$LT" ] && { echo "  !! libtsan absent de ce compilateur — TSan impossible ici"; exit 1; }
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
  echo "  !! CONFIGURE FAILED :"; tail -20 /tmp/cfg_$name.log; exit 1
fi
grep -q "Manually-specified variables were not used" /tmp/cfg_$name.log \
  && echo "  !! option ignorée ($opt) — sources pas à jour ?"
echo "         configure OK"

# ── 3) build (avec chrono) ───────────────────────────────────────────────────
step "cmake build  (compilation en cours, patiente...)"
t0=$(date +%s)
if ! cmake --build "$B" --target shrink_bench_chunk2a_dyn -j >/tmp/build_$name.log 2>&1; then
  echo "  !! BUILD FAILED :"; tail -25 /tmp/build_$name.log; exit 1
fi
echo "         build OK en $(( $(date +%s)-t0 ))s"
# Le nom sans declinaison de taille n'a jamais existe (les cibles portent
# toujours _64/_256/_1024/_dyn), et le mode perf n'est plus dans le nom :
# c'est un drapeau depuis la fusion check/perf.
exe="$B/examples/shrink_bench/shrink_bench_chunk2a_dyn"
[ -x "$exe" ] || { echo "  !! pas d'exe"; exit 1; }
if [ "$san" = "tsan" ]; then
  echo "         instrumentation TSan : $(nm "$exe" 2>/dev/null | grep -c __tsan_func_entry) symboles __tsan (doit être > 0)"
fi

# ── 4) runs ──────────────────────────────────────────────────────────────────
step "lancement de $runs runs (64 threads) — un run ≈ 1-2 min"
cd "$B/examples/shrink_bench"; ulimit -c 0
c=0 races=0 T0=$(date +%s)
for i in $(seq 1 "$runs"); do
  r0=$(date +%s)
  if [ "$san" = "tsan" ]; then
    TSAN_OPTIONS="history_size=7:halt_on_error=0" OMP_NUM_THREADS=64 ./shrink_bench_chunk2a_dyn --mode perf >/tmp/${name}_$i.log 2>&1
    nr=$(grep -c "data race" /tmp/${name}_$i.log); races=$((races+nr)); st="races=$nr"
  else
    OMP_NUM_THREADS=64 ./shrink_bench_chunk2a_dyn --mode perf >/dev/null 2>&1 && st="ok" || { st="CRASH"; c=$((c+1)); }
  fi
  el=$(( $(date +%s)-T0 )); rt=$(( $(date +%s)-r0 ))
  # ETA grossier
  avg=$(( el / i )); eta=$(( avg * (runs - i) ))
  printf "\r  [%s] run %3d/%s | %-9s | crash %3d | ce run %3ds | écoulé %4ds | ETA ~%4ds  " \
         "$(ts)" "$i" "$runs" "$st" "$c" "$rt" "$el" "$eta"
done
echo ""
echo "══════════════════════════════════════════════════════════════════"
if [ "$san" = "tsan" ]; then
  echo " [$(ts)] RÉSULTAT TSan : $races 'data race' cumulées sur $runs runs"
  echo " Sites de course (regroupés) :"
  grep -h -A3 "SUMMARY: ThreadSanitizer" /tmp/${name}_*.log 2>/dev/null \
    | grep -oE "chunk[a-z_0-9]*\.h:[0-9]+|unordered_dense\.h:[0-9]+|flat_hash_map\.hpp:[0-9]+|compression_manager[a-z_]*\.h:[0-9]+" \
    | sort | uniq -c | sort -rn | sed 's/^/    /'
else
  echo " [$(ts)] RÉSULTAT $label : $c crash / $runs   (en $(( $(date +%s)-T0 ))s)"
fi
echo "══════════════════════════════════════════════════════════════════"
