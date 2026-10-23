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
run error "MUT_WAIT_ON_HARD (the triangle of 5.3)"            --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT -DMUT_WAIT_ON_HARD $M
run error "MUT_RELAXED_MARK (section 5.3: acquire is load-bearing)" -- -DSHAPE_SYMMETRIC -DMUT_RELAXED_MARK $M
echo "-- the behaviour the model changed in the source (docs/formal_models.md, F2) --"
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

M3=reader_reclaim.c
echo "== handshake 3: readers <-> buffer reclamation ($M3) =="
run ok    "source, 2 readers, one read each"                 --check-liveness -- $M3
run ok    "source, 3 readers"                                --check-liveness -- -DNREADERS=3 $M3
run ok    "source, 2 reads per reader"                       --check-liveness -- -DNREADS=2 $M3
run ok    "source, 3 reads per reader"                       --check-liveness -- -DNREADS=3 $M3
run ok    "source, 3 readers x 2 reads"                      --check-liveness -- -DNREADERS=3 -DNREADS=2 $M3
run ok    "source, overflow reader"                          --check-liveness -- -DOVERFLOW_READER $M3
run ok    "source, overflow reader, 2 reads"                 --check-liveness -- -DOVERFLOW_READER -DNREADS=2 $M3
echo "-- mutations: each must be refused --"
run error "MUT_NO_GRACE (the hazard itself)"                 -- -DMUT_NO_GRACE $M3
run error "MUT_NO_HEAVY_BARRIER"                             -- -DMUT_NO_HEAVY_BARRIER $M3
run error "MUT_LEAVE_RELAXED (the even store must be release)" -- -DMUT_LEAVE_RELAXED $M3
run error "SHAPE_ASYMMETRIC_NAKED (reader without its fence)" -- -DSHAPE_ASYMMETRIC_NAKED $M3
echo "-- the two behaviours the model changed in the source (F3, F4) --"
run error "MUT_ENTER_RELAXED, 2 reads (F3, before)"          -- -DNREADS=2 -DMUT_ENTER_RELAXED $M3
run ok    "MUT_ENTER_RELAXED, 1 read: the hole needs back-to-back reads" -- -DMUT_ENTER_RELAXED $M3
run ok    "MUT_WAIT_FOR_EVEN, 2 reads (the pass-side way out, not taken)" --check-liveness -- -DNREADS=2 -DMUT_WAIT_FOR_EVEN $M3
run error "MUT_OVERFLOW_NO_FENCE (F4, before)"               -- -DOVERFLOW_READER -DMUT_OVERFLOW_NO_FENCE $M3

M4=switch_mt.c
echo "== the in-place constant switch: writer <-> compression pass ($M4) =="
# The switch of zero_any::Chunk::fill_constant_st against the pass, properties P1-P3 (docs/formal_models.md 6.1).
run ok    "source, switch to a second value"                 -- $M4
run ok    "source, switch to the first value"                -- -DTO_FIRST $M4
run ok    "source, chunk materialised: left alone"           -- -DINIT_BUF $M4
run ok    "source, materialised, first value"                -- -DTO_FIRST -DINIT_BUF $M4
run ok    "isolation (STRICT_SCAN), second value"            -- -DSTRICT_SCAN $M4
run ok    "isolation (STRICT_SCAN), first value"             -- -DSTRICT_SCAN -DTO_FIRST $M4
run ok    "isolation (STRICT_SCAN), materialised"            -- -DSTRICT_SCAN -DINIT_BUF $M4
run ok    "isolation (STRICT_SCAN), materialised, first value" -- -DSTRICT_SCAN -DTO_FIRST -DINIT_BUF $M4
echo "-- mutations of the handshake: refused by the isolation property --"
run error "MUT_NO_RECHECK, isolation"                        -- -DSTRICT_SCAN -DTO_FIRST -DMUT_NO_RECHECK $M4
run error "MUT_CHECK_FIRST, isolation"                       -- -DSTRICT_SCAN -DTO_FIRST -DMUT_CHECK_FIRST $M4
run error "MUT_NO_WINDOW, isolation"                         -- -DSTRICT_SCAN -DTO_FIRST -DMUT_NO_WINDOW $M4
run error "MUT_NO_FENCE, isolation"                          -- -DSTRICT_SCAN -DTO_FIRST -DMUT_NO_FENCE $M4
echo "-- the same mutations under P1-P3 alone, and the orders readers would need: must still pass (7.6) --"
run ok    "MUT_NO_RECHECK, P1-P3"                            -- -DTO_FIRST -DMUT_NO_RECHECK $M4
run ok    "MUT_CHECK_FIRST, P1-P3"                           -- -DTO_FIRST -DMUT_CHECK_FIRST $M4
run ok    "MUT_NO_WINDOW, P1-P3"                             -- -DTO_FIRST -DMUT_NO_WINDOW $M4
run ok    "MUT_NO_FENCE, P1-P3"                              -- -DTO_FIRST -DMUT_NO_FENCE $M4
run ok    "MUT_RELAXED_VALUES (no reader in the model)"      -- -DTO_FIRST -DMUT_RELAXED_VALUES $M4
run ok    "MUT_META_AFTER (no reader in the model)"          -- -DTO_FIRST -DMUT_META_AFTER $M4
echo "-- what the switch does rest on --"
run error "MUT_NO_BUF_CHECK on a materialised chunk (buffer lost)" -- -DINIT_BUF -DMUT_NO_BUF_CHECK $M4
run error "SECOND_MATERIALISER: the span's precondition broken"    -- -DSECOND_MATERIALISER $M4
run ok    "SECOND_MATERIALISER, switch by CAS (not taken)"   -- -DSECOND_MATERIALISER -DSWITCH_CAS $M4
run ok    "switch by CAS, first value, materialised"         -- -DSWITCH_CAS -DTO_FIRST -DINIT_BUF $M4

exit $fail
