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
    elif grep -qE "Error detected|Error:|violation|race|Assertion|assertion|use-after-free|freed memory|Liveness|liveness|Non-terminating|non-terminating" <<<"$out"; then
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
# The protocol as in the source, every configuration: must pass.
run ok    "source, marks, buffer start"                      -- -DSHAPE_SYMMETRIC $M
run ok    "source, marks, constant start"                    -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT $M
run ok    "source, marks, 3 writers"                         -- -DSHAPE_SYMMETRIC -DNWRITERS=3 $M
run ok    "source, marks, constant start, 3 writers"         -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DNWRITERS=3 $M
run ok    "source, no marks (SHRINK_SMALL_CHUNK)"            -- -DSHAPE_SYMMETRIC -DNO_MARKS $M
run ok    "source, no marks, constant start"                 -- -DSHAPE_SYMMETRIC -DNO_MARKS -DINIT_CONSTANT -DBOUNDED_COMPRESSOR_WAIT $M
run ok    "source, no marks, 3 writers"                      -- -DSHAPE_SYMMETRIC -DNO_MARKS -DNWRITERS=3 $M
# Liveness: with the marks, even an UNBOUNDED compressor wait terminates (F2).
run ok    "source, liveness, unbounded compressor wait"      --check-liveness -- -DSHAPE_SYMMETRIC $M
run ok    "source, liveness, constant start, unbounded wait" --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT $M
run ok    "source, liveness, constant start, 3 writers"      --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DNWRITERS=3 $M
run ok    "source, liveness, bounded wait (the library)"     --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DBOUNDED_COMPRESSOR_WAIT $M
run ok    "source, liveness, 5 writers, buffer start"        --check-liveness -- -DSHAPE_SYMMETRIC -DNWRITERS=5 $M
run ok    "source, liveness, 5 writers, constant start"      --check-liveness -- -DSHAPE_SYMMETRIC -DNWRITERS=5 -DINIT_CONSTANT $M

echo "-- mutations: each must be refused --"
run error "MUT_NO_CHECK2 (no look after the announce)"       -- -DSHAPE_SYMMETRIC -DMUT_NO_CHECK2 $M
run error "SHAPE_ASYMMETRIC_NAKED (asymmetric, no membarrier)" -- -DSHAPE_ASYMMETRIC_NAKED $M
run error "MUT_NO_FENCE_COMPRESSOR"                          -- -DSHAPE_SYMMETRIC -DMUT_NO_FENCE_COMPRESSOR $M
run error "MUT_WAIT_ON_HARD, no marks (the triangle)"        --check-liveness -- -DSHAPE_SYMMETRIC -DNO_MARKS -DINIT_CONSTANT -DMUT_WAIT_ON_HARD $M
run error "MUT_RELAXED_MARK (section 5.3: acquire is load-bearing)" -- -DSHAPE_SYMMETRIC -DMUT_RELAXED_MARK $M
echo "-- the two behaviours the model changed in the source (docs/formal_models.md, F1 and F2) --"
run error "MUT_NO_MARKS_RELAXED_GATE (F1, before)"           -- -DSHAPE_SYMMETRIC -DNO_MARKS -DMUT_NO_MARKS_RELAXED_GATE $M
run error "MUT_MATERIALISE_WAIT_UNDER_FLAG (F2, before), unbounded wait" --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DMUT_MATERIALISE_WAIT_UNDER_FLAG $M
run ok    "MUT_MATERIALISE_WAIT_UNDER_FLAG, bounded wait: safe, stalls only" --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DMUT_MATERIALISE_WAIT_UNDER_FLAG -DBOUNDED_COMPRESSOR_WAIT $M

M2=compressor_flush.c
echo "== handshake 2: compressor <-> deferred-write flush ($M2) =="
run ok    "source, 1 writer (deferred then flushed)"         --check-liveness -- $M2
run ok    "source, 2 writers"                                --check-liveness -- -DNWRITERS=2 $M2
run ok    "source, 3 writers"                                --check-liveness -- -DNWRITERS=3 $M2
echo "-- mutations: each must be refused --"
run error "MUT_NO_SECOND_LOOK (section 5.4: the lost replay)" -- -DMUT_NO_SECOND_LOOK $M2
run error "MUT_NO_SECOND_LOOK, 2 writers"                    -- -DMUT_NO_SECOND_LOOK -DNWRITERS=2 $M2
run error "MUT_NO_SECOND_LOOK, race detection off: the lost write itself" --disable-race-detection -- -DMUT_NO_SECOND_LOOK $M2
run error "MUT_NO_SECOND_LOOK + MUT_NO_POSTCHECK"            -- -DMUT_NO_SECOND_LOOK -DMUT_NO_POSTCHECK $M2
run error "MUT_FLUSH_NO_DRAIN (replay without draining the stealers)" -- -DMUT_FLUSH_NO_DRAIN $M2
echo "-- what the document says is NOT load-bearing: must still pass --"
run ok    "MUT_NO_FIRST_LOOK (the early-out 'pairs with nothing')" -- -DMUT_NO_FIRST_LOOK $M2
run ok    "MUT_NO_FIRST_LOOK, 2 writers"                     -- -DMUT_NO_FIRST_LOOK -DNWRITERS=2 $M2
run ok    "MUT_NO_POSTCHECK (post-store confirmation: a second line)" -- -DMUT_NO_POSTCHECK $M2
run ok    "MUT_NO_POSTCHECK, 2 writers"                      -- -DMUT_NO_POSTCHECK -DNWRITERS=2 $M2

exit $fail
