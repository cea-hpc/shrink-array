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

/// @file bench_phase_struct.cpp
/// @brief The struct arrays' phases: their fill, their compute window and their
///        reset. See bench_phases.h and bench_struct.h.
///
/// Point3 and Cell are element types like the others -- what this file shows
/// is that the library needs nothing from them beyond the element contract.
/// Three things differ from the arithmetic phases, all on the bench's side:
///
///  - no compound operators: a struct has no +=, so an update reads the
///    element, builds the new one and assigns it;
///  - no `simd` on the std::vector baseline's loops (the _SCALAR macros): a
///    store of a 24-byte struct is not what LLVM vectorises, and asking for
///    it only drew a -Wpass-failed per loop (see bench_prelude.h);
///  - the reduction is on a double built from the members, since a struct
///    is not a reduction type.

#include <algorithm> // std::min
#include <cstdint>   // int32_t
#include <vector>    // std::vector

#include "bench_phases.h" // bench_phase_struct_* (defined here), BenchArrays

// ─── Fill ────────────────────────────────────────────────────────────────────
// The same partition as bench_phase_fill: [0, sz/2) stays at the null element
// (a fresh array is T{} everywhere -- nothing to write, and the std::vector
// baseline is value-initialised to the same), [sz/2, sz - sz/3) holds the
// second value, [sz - sz/3, sz - sz/4) another constant, then one interval of
// varied elements. Whole chunks of nulls and of one repeated point are what
// the pass collapses, exactly like runs of 0.0 and of a second value.
void bench_phase_struct_fill(BenchArrays& bench_arr)
{
    if (!bench_arr.with_struct) // --no_struct
    {
        return;
    }

    auto& tab_point = bench_arr.tab_point;
    auto& tab_cell  = bench_arr.tab_cell;

    const Int sz       = bench_arr.sz;
    const Int interval = bench_arr.interval;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& t)
        {
            begin_parallel_phase_g();

            Int begin_i = sz / 2;
            Int end_i   = sz - (sz / 3);

            t.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab_point[i] = second_value_p;
                    tab_cell[i]  = second_value_cell;
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = sz - (sz / 4);

            t.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab_point[i] = Point3{2., 0., 0.};
                    tab_cell[i]  = Cell{2, 1.f};
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = std::min(begin_i + interval, sz);

            t.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab_point[i] = Point3{static_cast<double>(i), 0.5, static_cast<double>(i % 7)};
                    tab_cell[i]  = Cell{static_cast<int32_t>(i % 5), static_cast<float>(i % 3) * 0.25f};
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();
}

// ─── Compute ─────────────────────────────────────────────────────────────────
// On the iteration's window [start0, end0): a run of null elements up to
// tab_nba1[k], a run of the second value up to tab_nba2[k], and the rest
// updated from its current value -- so that every iteration re-creates runs
// the pass can take AND overwrites some it took. Then a read-only sweep of
// the window into point_sum, twice the reads of the writes, as the other
// compute loops do.
void bench_phase_struct_compute(BenchArrays&            bench_arr,
                                Int                     k,
                                Int                     start0,
                                Int                     end0,
                                const std::vector<Int>& tab_nba1,
                                const std::vector<Int>& tab_nba2,
                                double&                 point_sum)
{
    if (!bench_arr.with_struct) // --no_struct
    {
        return;
    }

    auto& tab_point = bench_arr.tab_point;
    auto& tab_cell  = bench_arr.tab_cell;

    const Int  nba1 = tab_nba1[k];
    const Int  nba2 = tab_nba2[k];
    const auto dk   = static_cast<double>(k % 3);

    // Shared accumulator of the region, reduced by the loop (bench_prelude.h
    // on why not a `parallel reduction`).
    double sum = 0;

    // Round-robin, chunk-aligned blocks (bench_sched_for): the writer's cost
    // depends on the index, and both loops must share the schedule.
    const BenchSched sched = bench_sched_for(start0, end0);
    // Named by the pragma macros only: unused in a serial build.
    [[maybe_unused]] const Int sched_blk  = sched.blk;
    const Int                  sched_base = sched.base;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& t)
        {
            begin_parallel_phase_g();

            t.for_static_chunks<Int>(
                sched_base, end0, bench_par::block(sched_blk),
                [&](const Int i) __attribute__((always_inline))
                {
                    if (i < start0) // alignment head: same trip count in both loops
                    {
                        return;
                    }
                    if (i < nba1)
                    {
                        tab_point[i] = Point3{};
                        tab_cell[i]  = Cell{};
                    }
                    else if (i < nba2)
                    {
                        tab_point[i] = second_value_p;
                        tab_cell[i]  = second_value_cell;
                    }
                    else
                    {
                        const Point3& p = tab_point[i];
                        const Cell&   c = tab_cell[i];
                        tab_point[i]    = Point3{p.x + (0.5 * dk), (0.5 * p.y) + 1., p.z + p.x};
                        tab_cell[i]     = Cell{(c.material + 1) & 7, (0.5f * c.fraction) + 0.25f};
                    }
                },
                bench_par::nowait);

            // No barrier between the two loops: same bounds, same schedule and block
            // size, so a thread reads back the elements it wrote itself (and a read
            // of its own deferred write goes through the read gate anyway).
            bench_par::for_static_chunks_sum(
                t, sched_base, end0, bench_par::block(sched_blk), sum,
                [&](const Int i, auto& sum_part) __attribute__((always_inline))
                {
                    if (i < start0)
                    {
                        return;
                    }
                    const Point3& p = tab_point[i];
                    const Cell&   c = tab_cell[i];
                    sum_part += p.x + (p.y * static_cast<double>(c.fraction)) + p.z + static_cast<double>(c.material);
                });

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();

    point_sum += sum;
}

// ─── Reset ───────────────────────────────────────────────────────────────────
// The struct arrays' share of the raz sweep: the window back to the null
// element, a whole run for the next pass.
void bench_phase_struct_raz(BenchArrays& bench_arr, Int start0, Int end0)
{
    if (!bench_arr.with_struct) // --no_struct
    {
        return;
    }

    auto& tab_point = bench_arr.tab_point;
    auto& tab_cell  = bench_arr.tab_cell;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& t)
        {
            begin_parallel_phase_g();

            t.for_static<Int>(
                start0, end0,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab_point[i] = Point3{};
                    tab_cell[i]  = Cell{};
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();
}
