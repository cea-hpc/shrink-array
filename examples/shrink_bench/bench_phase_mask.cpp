/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// @file bench_phase_mask.cpp
/// @brief tab_mask's phases: its fill and its per-iteration update. See
///        bench_array.h (BenchMaskArray, bench_mask_*) and bench_phases.h.
///
/// tab_bool is 96 % constant chunks by the end of a run: there, packing bools
/// saves almost nothing. tab_mask is the other case -- a large bool array that
/// is mostly NOT constant: segments of random masks (half the bits set) and of
/// scattered flags (~3 % set), between whole constant segments (60 % of them,
/// all false or all true -- bench_array.h, BenchMaskKind, says why 60). The constant half is what a chunked container
/// compresses and std::vector<bool> cannot; the mixed half is what a byte per
/// bool pays 8x for.
///
/// Every loop runs over whole 64-element blocks, one thread per block: in the
/// std::vector<bool> baseline two threads writing the same word would race.
/// The chunked containers do not need it (ArrayChunkBits writes each word
/// atomically), but the workload has to be the same in every build.

#include <cstdint> // std::uint64_t

#include "bench_phases.h" // bench_phase_mask_fill / _compute (defined here), BenchArrays

namespace
{

/// @brief Is tab_mask[i] flipped at iteration @p k? ~1 element in 61 of the
///        mixed segments, a different set each iteration.
constexpr bool mask_flips(Int i, Int k) noexcept
{
    return bench_mask_hash((static_cast<std::uint64_t>(i) * 131u) + (static_cast<std::uint64_t>(k) * 0x9E37u)) % 61 ==
           0;
}

} // namespace

// ─── Fill ────────────────────────────────────────────────────────────────────
// The starting pattern, bench_mask_initial(): whole segments of each kind.
void bench_phase_mask_fill(BenchArrays& bench_arr)
{
    if (!bench_arr.with_mask) // --no_mask
    {
        return;
    }

    auto&     tab_mask = bench_arr.tab_mask;
    const Int n        = static_cast<Int>(tab_mask.size());
    const Int n_blocks = (n + bench_mask_block - 1) / bench_mask_block;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& t)
        {
            begin_parallel_phase_g();

            BENCH_FOR_STATIC_NOWAIT_SCALAR(t, b, 0, n_blocks)
            {
                const Int lo = b * bench_mask_block;
                const Int hi = lo + bench_mask_block < n ? lo + bench_mask_block : n;
                for (Int i = lo; i < hi; ++i)
                {
                    // A fresh array is all false: only the true elements are written,
                    // so a false segment stays a constant chunk nothing materialised.
                    if (bench_mask_initial(i))
                    {
                        tab_mask[i] = true;
                    }
                }
            }

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();
}

// ─── Compute ─────────────────────────────────────────────────────────────────
// Iteration k:
//  - in the mixed segments, one block in 16 (a different sixteenth each
//    time): ~1 element in 61 flips (read, negate, write back) -- the masks
//    and flags evolve, and stay mixed;
//  - a constant segment in 32 turns to the other constant, WHOLE (all its
//    blocks this iteration, so it stays uniform): its chunks materialise,
//    then the pass compresses them again, to the other constant.
// Then a read sweep of the blocks written, into @p mask_count, as the other
// compute phases read what they wrote.
namespace
{
/// @brief Does iteration @p k write block @p b of tab_mask?
bool mask_block_written(Int b, Int k) noexcept
{
    const Int seg = b * bench_mask_block / bench_mask_segment;
    switch (bench_mask_kind(seg))
    {
        case BenchMaskKind::all_false:
        case BenchMaskKind::all_true:
            return (seg + k) % 32 == 0;
        case BenchMaskKind::random:
        case BenchMaskKind::scattered:
            return b % 16 == k % 16;
    }
    return false;
}
} // namespace

void bench_phase_mask_compute(BenchArrays& bench_arr, Int k, long& mask_count)
{
    if (!bench_arr.with_mask) // --no_mask
    {
        return;
    }

    auto&     tab_mask = bench_arr.tab_mask;
    const Int n        = static_cast<Int>(tab_mask.size());
    const Int n_blocks = (n + bench_mask_block - 1) / bench_mask_block;

    long count = 0;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& t)
        {
            begin_parallel_phase_g();

            BENCH_FOR_STATIC_NOWAIT_SCALAR(t, b, 0, n_blocks)
            {
                if (!mask_block_written(b, k))
                {
                    continue;
                }
                const Int  lo       = b * bench_mask_block;
                const Int  hi       = lo + bench_mask_block < n ? lo + bench_mask_block : n;
                const auto kind     = bench_mask_kind(lo / bench_mask_segment);
                const bool constant = kind == BenchMaskKind::all_false || kind == BenchMaskKind::all_true;
                if (constant)
                {
                    const bool now = tab_mask[lo]; // the whole segment's value
                    for (Int i = lo; i < hi; ++i)
                    {
                        tab_mask[i] = !now;
                    }
                }
                else
                {
                    for (Int i = lo; i < hi; ++i)
                    {
                        if (mask_flips(i, k))
                        {
                            const bool now = tab_mask[i];
                            tab_mask[i]    = !now;
                        }
                    }
                }
            }

            // Same schedule and bounds: a thread reads back its own blocks.
            BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, b, 0, n_blocks, count)
            {
                if (!mask_block_written(b, k))
                {
                    continue;
                }
                const Int lo = b * bench_mask_block;
                const Int hi = lo + bench_mask_block < n ? lo + bench_mask_block : n;
                for (Int i = lo; i < hi; ++i)
                {
                    BENCH_RED(count) += tab_mask[i] ? 1 : 0;
                }
            }

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();

    mask_count += count;
}
