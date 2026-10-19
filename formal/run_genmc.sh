#!/usr/bin/env bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# formal/run_genmc.sh -- run the reduced models under GenMC and compare with the expected verdicts.
#
#   ./run_genmc.sh [genmc options]        e.g. ./run_genmc.sh --imm
#
# Needs `genmc` on the PATH (https://github.com/MPI-SWS/genmc, built against LLVM 19+) and
# the clang GenMC was built with on the PATH too. Exit status: 0 if every verdict is as
# expected, 1 otherwise. One line per case: EXPECTED / ACTUAL / status.

set -u
cd "$(dirname "$0")"
GENMC=${GENMC:-genmc}
EXTRA=("$@")
fail=0

# run <expected: ok|error> <label> <genmc flags...> -- <cflags...> <file>
run() {
    local expected=$1 label=$2; shift 2
    local gflags=() cflags=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do gflags+=("$1"); shift; done
    shift
    cflags=("$@")
    local out
    out=$("$GENMC" --disable-estimation "${EXTRA[@]}" "${gflags[@]}" -- "${cflags[@]}" 2>&1)
    local actual
    if grep -q "No errors were detected" <<<"$out"; then
        actual=ok
    elif grep -qE "Error detected|violation|Assertion|assertion|use-after-free|freed memory|Liveness|liveness|Non-terminating|non-terminating" <<<"$out"; then
        actual=error
    else
        actual=unknown
    fi
    local execs
    execs=$(grep -oE "complete executions explored: [0-9]+" <<<"$out" | grep -oE "[0-9]+$")
    local status=OK
    if [ "$actual" != "$expected" ]; then status=MISMATCH; fail=1; fi
    printf "%-58s expected %-5s actual %-7s (%s executions)  %s\n" "$label" "$expected" "$actual" "${execs:-?}" "$status"
    if [ "$status" = MISMATCH ] || [ "$actual" = unknown ]; then
        grep -E "Error|error|violation|Assertion|freed|WARNING|warning" <<<"$out" | head -5 | sed 's/^/        /'
    fi
}

M=writer_compressor.c

echo "== handshake 1: writer <-> compressor ($M) =="
# The protocol as shipped, every configuration: must pass.
run ok    "current, marks, buffer start"                     -- -DSHAPE_SYMMETRIC $M
run ok    "current, marks, constant start"                   -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DBOUNDED_COMPRESSOR_WAIT $M
run ok    "current, no marks, constant start"                -- -DSHAPE_SYMMETRIC -DNO_MARKS -DINIT_CONSTANT -DBOUNDED_COMPRESSOR_WAIT $M
echo "-- finding F1: the no-marks gate (SHRINK_SMALL_CHUNK) as shipped, and with the acquire --"
run error "no marks AS SHIPPED: relaxed load of hard_inflight"  -- -DSHAPE_SYMMETRIC -DNO_MARKS $M
run ok    "no marks + FIX_NO_MARKS_ACQUIRE"                  -- -DSHAPE_SYMMETRIC -DNO_MARKS -DFIX_NO_MARKS_ACQUIRE $M
run ok    "no marks + fix, 3 writers"                        -- -DSHAPE_SYMMETRIC -DNO_MARKS -DFIX_NO_MARKS_ACQUIRE -DNWRITERS=3 $M
run ok    "current, 3 writers"                               -- -DSHAPE_SYMMETRIC -DNWRITERS=3 $M
# Liveness of the shipped protocol: the compressor's wait is bounded in the library.
run ok    "current, liveness, bounded compressor wait"       --check-liveness -- -DSHAPE_SYMMETRIC -DBOUNDED_COMPRESSOR_WAIT $M
run ok    "current, liveness, constant start, bounded wait"  --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DBOUNDED_COMPRESSOR_WAIT $M

echo "-- what the bound buys: the same protocol with an UNBOUNDED compressor wait --"
run ok    "unbounded wait, buffer start (no materialisation)" --check-liveness -- -DSHAPE_SYMMETRIC $M
run error "unbounded wait, constant start: materialise vs parked" --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT $M
echo "-- finding F2: the candidate fix, unbounded compressor wait, safety and liveness --"
run ok    "FIX_MATERIALISE_NO_WAIT_UNDER_FLAG, constant start" --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DFIX_MATERIALISE_NO_WAIT_UNDER_FLAG $M
run ok    "FIX_MATERIALISE_NO_WAIT_UNDER_FLAG, 3 writers"      --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DNWRITERS=3 -DFIX_MATERIALISE_NO_WAIT_UNDER_FLAG $M

echo "-- mutations: each must be refused --"
run error "MUT_NO_CHECK2 (no look after the announce)"       -- -DSHAPE_SYMMETRIC -DMUT_NO_CHECK2 $M
run error "SHAPE_ASYMMETRIC_NAKED (asymmetric, no membarrier)" -- -DSHAPE_ASYMMETRIC_NAKED $M
run error "MUT_NO_FENCE_COMPRESSOR"                          -- -DSHAPE_SYMMETRIC -DMUT_NO_FENCE_COMPRESSOR $M
run error "MUT_WAIT_ON_HARD, no marks (the triangle)"        --check-liveness -- -DSHAPE_SYMMETRIC -DNO_MARKS -DFIX_NO_MARKS_ACQUIRE -DINIT_CONSTANT -DMUT_WAIT_ON_HARD $M
run error "MUT_RELAXED_MARK (section 5.3: acquire is load-bearing)" -- -DSHAPE_SYMMETRIC -DMUT_RELAXED_MARK $M

exit $fail
