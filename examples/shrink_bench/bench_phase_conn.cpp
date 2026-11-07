/*
 * Copyright 2026 CEA (www.cea.fr)
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

/// @file bench_phase_conn.cpp
/// @brief tab_conn's phases: its fill and its per-iteration update. See
///        bench_array.h (BenchConnArray, bench_conn_*) and bench_phases.h.
///
/// tab_conn is a mesh connectivity: the 8 node numbers of each hexahedral cell,
/// cell after cell, rows of bench_conn_nx cells. No chunk of it is uniform, so
/// the constant-compressing variants keep it whole; but consecutive cells are the
/// same 8 numbers plus one, so a chunk of whole cells is one motif plus an offset,
/// which pattern::Array compresses (docs/algo_pattern.md).
///
/// Each iteration renumbers one band of rows (row % bench_conn_bands): the node
/// order of its cells rotated by one corner, or put back. Its chunks then hold
/// another motif -- shared by the whole band, so compressed again once the pass
/// has seen it twice -- and the pass has work at every iteration. A read sweep
/// of the band before its rewrite plays the solver's gather.
///
/// Every loop runs over whole rows, one thread per row.

#include "bench_phases.h" // bench_phase_conn_fill / _compute (defined here), BenchArrays

namespace
{

/// @brief The rows of tab_conn and the node offset of its top layer.
struct ConnGeometry
{
    Int rows  = 0;
    Int layer = 0;
};

ConnGeometry conn_geometry(const BenchArrays& bench_arr)
{
    const Int rows = static_cast<Int>(bench_arr.tab_conn.size()) / bench_conn_row;
    return {rows, (bench_conn_nx + 1) * (rows + 1)};
}

} // namespace

// ─── Fill ────────────────────────────────────────────────────────────────────
// The mesh in its first numbering, row by row.
void bench_phase_conn_fill(BenchArrays& bench_arr)
{
    if (!bench_arr.with_conn) // --no_conn
    {
        return;
    }

    auto&              tab_conn = bench_arr.tab_conn;
    const ConnGeometry geo      = conn_geometry(bench_arr);

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();

            team.for_static<Int>(
                0, geo.rows,
                [&](const Int row) __attribute__((always_inline))
                {
                    const Int lo = row * bench_conn_row;
                    for (Int i = lo; i < lo + bench_conn_row; ++i)
                    {
                        tab_conn[i] = bench_conn_value(i, geo.layer, 0);
                    }
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();
}

// ─── Compute ─────────────────────────────────────────────────────────────────
// Iteration k: band k % bench_conn_bands read into @p conn_sum (the gather),
// then rewritten in its next numbering (bench_conn_rotation).
void bench_phase_conn_compute(BenchArrays& bench_arr, Int iter, long& conn_sum)
{
    if (!bench_arr.with_conn) // --no_conn
    {
        return;
    }

    auto&              tab_conn = bench_arr.tab_conn;
    const ConnGeometry geo      = conn_geometry(bench_arr);
    const Int          band     = iter % bench_conn_bands;

    long sum = 0;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();

            // The gather: the band's node numbers, as they stand.
            bench_par::for_static_sum(team, 0, geo.rows, sum,
                                      [&](const Int row, auto& sum_part) __attribute__((always_inline))
                                      {
                                          if (row % bench_conn_bands != band)
                                          {
                                              return;
                                          }
                                          const Int lo = row * bench_conn_row;
                                          for (Int i = lo; i < lo + bench_conn_row; ++i)
                                          {
                                              sum_part += tab_conn[i];
                                          }
                                      });

            // The renumbering: same schedule and bounds, a thread rewrites the rows it read.
            team.for_static<Int>(
                0, geo.rows,
                [&](const Int row) __attribute__((always_inline))
                {
                    if (row % bench_conn_bands != band)
                    {
                        return;
                    }
                    const Int rotation = bench_conn_rotation(row, iter);
                    const Int lo       = row * bench_conn_row;
                    for (Int i = lo; i < lo + bench_conn_row; ++i)
                    {
                        tab_conn[i] = bench_conn_value(i, geo.layer, rotation);
                    }
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();

    conn_sum += sum;
}
