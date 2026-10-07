#!/usr/bin/env bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# shrink_bench int64 binaries (chunk@dyn by default), the scaling guard's workload,
# with/without slab recycling and numactl --interleave.
# Usage: bench_recycle.sh <build_root> [reps] [dir containing numa_launch.py]
#    or: bench_recycle.sh <build_root> <python_dir> [reps]
#   build_root: the directory holding cmake-build-omp-llvm/ and cmake-build-omp-gnu/
#   python dir: found next to this script, in ../python or in build_root/python if omitted
# Environment (defaults in brackets):
#   THREADS ["32 64 128"]  RECYCLE ["1 0"]  CONFIGS ["omp-llvm omp-gnu"]
#   VARIANTS ["chunk@dyn"]: base@size, run as shrink_bench_<base>_int64_<size>
#            (chunk2@1024, chunk2a@dyn, ...); CHUNK_RECHUNK_RECYCLE only acts on chunk
#   MULTIPLY [2]: --multiply of the workload
#   NUMA ["auto"]: auto = numactl --interleave over the nodes the run spans, as
#                  bench_and_test.py does; off = no numactl; "auto off" = both
#   ENVS ["-"]: environment sets to compare, each tag:VAR=VAL[,VAR=VAL...] or "-"
#               for none; the tag (no "_", "@" or ":") is appended to the label.
#               e.g. ENVS="- coarsen8:CHUNK_RECHUNK_MAX_COARSEN=8 nolist:CHUNK_RECHUNK_REUSE_LIST=0"
set -u
export LC_ALL=C   # printf %.2f and awk read and write "1.23" whatever the user locale
usage() { echo "usage: $0 <build_root> [reps=3] [python_dir]" >&2; exit 1; }
[ $# -ge 1 ] || usage
ROOT=$1; REPS=${2:-3}; PYDIR=${3:-}
# Old calling convention (build_root python_dir): a directory as 2nd argument is the python dir.
if [ -d "$REPS" ]; then PYDIR=$REPS; REPS=${3:-3}; fi
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "reps must be a positive integer: '$REPS'" >&2; usage; }
[ -d "$ROOT/cmake-build-omp-llvm" ] || [ -d "$ROOT/cmake-build-omp-gnu" ] ||
  { echo "$ROOT: no cmake-build-omp-llvm/ or cmake-build-omp-gnu/ here -- wrong build_root?" >&2; usage; }
if [ -z "$PYDIR" ]; then
  here=$(cd "$(dirname "$0")" && pwd)
  for d in "$here" "$here/../python" "$ROOT/python"; do
    [ -f "$d/numa_launch.py" ] && { PYDIR=$d; break; }
  done
fi
[ -n "$PYDIR" ] && [ -f "$PYDIR/numa_launch.py" ] ||
  { echo "numa_launch.py not found: pass the python dir as 3rd argument" >&2; usage; }
OUT=${OUT:-$PWD/recycle_$(date +%m%d_%H%M)}; mkdir -p "$OUT"
ARGS=(--mode perf --no_io --multiply "${MULTIPLY:-2}" --iter_max 500)
export OMP_PROC_BIND=close OMP_PLACES=cores
numa() { python3 -c "import sys; sys.path.insert(0,'$PYDIR'); import numa_launch as n
print(' '.join(n.interleave_prefix({'OMP_NUM_THREADS':'$1','OMP_PROC_BIND':'close'})))"; }
THREADS=${THREADS:-32 64 128}; RECYCLE=${RECYCLE:-1 0}; NUMA=${NUMA:-auto}
CONFIGS=${CONFIGS:-omp-llvm omp-gnu}; VARIANTS=${VARIANTS:-chunk@dyn}
for m in $NUMA; do [ "$m" = auto ] || [ "$m" = off ] || { echo "NUMA: auto or off, not '$m'" >&2; exit 1; }; done
ENVS=${ENVS:--}
for e in $ENVS; do
  [ "$e" = - ] && continue
  tag=${e%%:*}
  [[ "$e" == *:* && -n "$tag" && "$tag" != *[_@]* ]] ||
    { echo "ENVS: tag:VAR=VAL[,VAR=VAL...] (tag without _ or @), not '$e'" >&2; exit 1; }
  for kv in $(tr , ' ' <<<"${e#*:}"); do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "ENVS: '$kv' is not VAR=VAL (in '$e')" >&2; exit 1; }
  done
done
echo "config threads recycle numa rep seconds" | tee "$OUT/results.txt"
for c in $CONFIGS; do
 cfg=cmake-build-${c#cmake-build-}
 for v in $VARIANTS; do
  [[ "$v" == *@* ]] || { echo "VARIANTS: base@size, not '$v'" >&2; exit 1; }
  name=shrink_bench_${v%@*}_int64_${v#*@}
  exe=$(ls "$ROOT/$cfg"/*/examples/shrink_bench/$name 2>/dev/null | grep "$(uname -m)" | head -1)
  [ -x "$exe" ] || { echo "missing $name in $cfg" >&2; continue; }
  # one label per config and variant; chunk@dyn alone keeps the bare config name
  lab0=${cfg#cmake-build-}; [ "$VARIANTS" = chunk@dyn ] || lab0=$lab0:$v
  for es in $ENVS; do
  if [ "$es" = - ]; then lab=$lab0; envset=(); else lab=$lab0+${es%%:*}; read -ra envset <<<"$(tr , ' ' <<<"${es#*:}")"; fi
  for t in $THREADS; do
    for r in $(seq "$REPS"); do
      for m in $NUMA; do    # interleaved, so drift hits every side equally
        if [ "$m" = auto ]; then pre=$(numa $t); else pre=; fi
        for rec in $RECYCLE; do
          log="$OUT/${lab/@/-}_${t}th_rec${rec}_numa-${m}_r$r.log"
          s=$(date +%s.%N)
          ( cd "$OUT" && env ${envset[@]+"${envset[@]}"} OMP_NUM_THREADS=$t CHUNK_RECHUNK_RECYCLE=$rec CHUNK_RECHUNK_TRACE=1 \
              $pre "$exe" "${ARGS[@]}" ) >"$log" 2>&1 || echo "  FAIL: $log" >&2
          e=$(date +%s.%N)
          printf "%s %s %s %s %s %.2f\n" "$lab" $t $rec $m $r \
            "$(awk -v a=$s -v b=$e 'BEGIN{print b-a}')" | tee -a "$OUT/results.txt"
        done
      done
    done
  done
  done
 done
done
echo; echo "== best of $REPS (s) ==";
awk 'NR>1{k=$1" "$2"th rec="$3" numa="$4; if(!(k in b)||$6<b[k])b[k]=$6} END{for(k in b)print k, b[k]}' "$OUT/results.txt" | sort -k1,1 -k2,2n -k3,3 -k4,4
echo "logs (with CHUNK_RECHUNK_TRACE phase times): $OUT"
# Rechunk phase totals of the first repetition, recycling on when it was run.
summ="$(cd "$(dirname "$0")" && pwd)/rechunk_trace_summary.py"
rec1=$(set -- $RECYCLE; case " $RECYCLE " in *" 1 "*) echo 1;; *) echo "$1";; esac)
if [ -f "$summ" ] && ls "$OUT"/*_rec${rec1}_*_r1.log >/dev/null 2>&1; then
  echo; echo "== rechunk phases (rep 1, recycle=$rec1) =="
  python3 "$summ" $(cd "$OUT" && ls *_rec${rec1}_*_r1.log | sort -t_ -k1,1 -k2,2n -k4,4 | sed "s|^|$OUT/|")
fi
