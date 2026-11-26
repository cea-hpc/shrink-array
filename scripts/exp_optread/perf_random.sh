#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# EXPERIMENT exp/optimistic-read: where the random read spends its cycles, per mode.
# Runs on the Grace node AFTER run.sh (reuses its builds under $WORK). For each
# <compiler>-mode<m> build: perf stat of the whole access_cost run (cycles,
# instructions, backend stalls), then perf record + annotate of the random-read
# loop, the hottest instructions first.
#   WORK    run.sh's work directory (default: $CCCWORKDIR/optread10)
#   BUILDS  (default "gcc-mode0 gcc-mode7 llvm-mode0 llvm-mode7")
#   N       elements (default 33554432)
set -u
WORK=${WORK:-${CCCWORKDIR:-$HOME}/optread11}
BUILDS=${BUILDS:-"gcc-mode0 gcc-mode7 llvm-mode0 llvm-mode7"}
N=${N:-33554432}
OUT=$WORK/perf_random
mkdir -p "$OUT"
command -v perf > /dev/null || { echo "[perf_random] no perf on $(hostname)"; exit 1; }
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
for b in $BUILDS; do
    bin=$WORK/$b/examples/access_cost/access_cost_zero_or_value_256
    [ -x "$bin" ] || { echo "[perf_random] $b: no binary ($bin)"; continue; }
    perf stat -e cycles,instructions,stalled-cycles-backend,stalled-cycles-frontend -o "$OUT/$b.stat.txt" \
        "$bin" --n "$N" --reps 3 > /dev/null 2>&1
    perf record -q -e cycles:u -o "$OUT/$b.data" "$bin" --n "$N" --reps 3 > /dev/null 2>&1
    # Annotate everything, then keep the random-read loop's block (perf -s wants the
    # demangled name, which differs between compilers).
    perf annotate -i "$OUT/$b.data" --stdio > "$OUT/$b.all.annotate.txt" 2> /dev/null
    # One block per function, from its " Percent" header; keep the block whose
    # function label (the first "<...>:" line) is the random-read loop's.
    awk 'function flush() { if (is_random) printf "%s", blk; blk = ""; is_random = 0; label = 0 }
         /^ Percent/ { flush() }
         { blk = blk $0 "\n" }
         !label && /<.*>:$/ { label = 1; if ($0 ~ /arr_read_random/) is_random = 1 }
         END { flush() }' "$OUT/$b.all.annotate.txt" > "$OUT/$b.random.annotate.txt"
    echo "== $b"
    grep -E 'cycles|instructions|stalled' "$OUT/$b.stat.txt" | sed 's/^/   /'
    # The hottest lines of the random loop.
    grep -E '^\s+[0-9]+\.[0-9]+ :' "$OUT/$b.random.annotate.txt" 2> /dev/null | sort -rn | head -12 | sed 's/^/   /'
done
echo "[perf_random] files in $OUT"
