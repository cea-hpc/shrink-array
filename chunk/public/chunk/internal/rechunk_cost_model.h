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

#pragma once

// ============================================================================
// Re-chunk cost model (adaptive chunk sizing)
// ============================================================================
//
// Estimates the time cost of re-chunking one array from geometry S_old to a
// coarser or finer one, so the re-chunk decision can weigh it against the
// memory gain the ChunkSizeAdvisor predicts.
//
// Cost formula:
//   cost = n_elems      * cost_elem_ns    [the refill walks every element]
//        + n_chunks_old * cost_chunk_ns   [and every chunk: state, tables,
//                                          temporary, teardown, pool drain]
//        + data_bytes   * cost_byte_ns    [materialised data re-written]
//
//   n_chunks_old : chunks at the current size; n_elems = n_chunks_old * S_old.
//   data_bytes   : mat_old * S_old * elem_bytes, the materialised chunks'
//                  bytes (elem_bytes is 1/8 for packed bits).
//
// Compressed chunks carry no buffer, but they are not free: counting the
// materialised data alone underestimates sparse arrays by an order of magnitude.
//
// The constants were fitted (least squares) on rebuilds measured with
// CHUNK_RECHUNK_TRACE on one ARM machine at 64 threads (median error under
// 10%). They are HARDWARE-specific; elsewhere the formula's shape holds and
// the scalars move. The [RECHUNK-DECIDE] lines log the estimate, the chunk and
// materialised counts, and CHUNK_RECHUNK_TRACE the measured total, which is
// what a refit needs.
//
// Thread count: the constants hold at 64 threads, and the rebuild does not
// all run in parallel (the refill scales almost linearly, the teardown by
// half, the prologue, temporary, swap and pool drain not at all). So
//   cost(T) = cost_64 * (s + (1 - s) * (64 / T)^alpha),  s = 0.5, alpha = 0.63
// fitted between 8 and 64 threads. Below 8 nothing was measured and the
// extrapolation is unreliable: T is taken as 8 there.
//
// This module has no side effects and is not gated: it is pure arithmetic.

#include <algorithm> // std::max (the thread factor)
#include <cmath>     // std::pow (same)
#include <cstddef>   // std::size_t
#include <cstdint>   // std::int64_t

namespace shrink_array
{

// Hardware-specific scalars (see above). Mutable so a caller can set measured
// values for its machine.
inline double g_rechunk_cost_elem_ns  = 0.032; // per element of the array
inline double g_rechunk_cost_chunk_ns = 4.2;   // per chunk at the current size
inline double g_rechunk_cost_byte_ns  = 0.042; // per byte of materialised data

// The thread factor's scalars (see above).
inline double g_rechunk_cost_fit_threads  = 64.;  // thread count the constants above hold at
inline double g_rechunk_cost_serial_share = 0.5;  // s: the share that does not scale
inline double g_rechunk_cost_thread_exp   = 0.63; // alpha: how the rest scales
inline double g_rechunk_cost_min_threads  = 8.;   // below the measured range: taken as this

/// @brief How much longer a rebuild takes on @p threads than at
///        g_rechunk_cost_fit_threads (1 there, below 1 above it).
[[nodiscard]] inline double rechunk_thread_factor(int threads) noexcept
{
    const double t = std::max(static_cast<double>(threads), g_rechunk_cost_min_threads);
    return g_rechunk_cost_serial_share +
           ((1. - g_rechunk_cost_serial_share) * std::pow(g_rechunk_cost_fit_threads / t, g_rechunk_cost_thread_exp));
}

// Estimated re-chunk time in milliseconds for one array leaving S_old, given
// its chunk count and materialised-chunk count there, the bytes one element
// takes in a materialised chunk, and the threads the rebuild will run on
// (CHUNK_PMP_MAX_THREADS at the decision: the refill's region takes the
// default team).
[[nodiscard]] inline double rechunk_cost_ms(std::int64_t n_chunks_old,
                                            unsigned     S_old,
                                            std::int64_t mat_old,
                                            double       elem_bytes,
                                            int          threads) noexcept
{
    const double n_elems    = static_cast<double>(n_chunks_old) * static_cast<double>(S_old);
    const double data_bytes = static_cast<double>(mat_old) * static_cast<double>(S_old) * elem_bytes;
    const double ns         = (n_elems * g_rechunk_cost_elem_ns) +
                      (static_cast<double>(n_chunks_old) * g_rechunk_cost_chunk_ns) +
                      (data_bytes * g_rechunk_cost_byte_ns);
    return ns / 1.0e6 * rechunk_thread_factor(threads);
}

// Convenience: is a re-chunk worth it? gain_bytes is the advisor's predicted
// memory saving; the caller passes a policy (min gain, max cost). Kept separate
// from the cost so the decision can layer hysteresis/cooldown on top. Returns true
// when the move clears both thresholds.
[[nodiscard]] inline bool
rechunk_is_worth_it(std::size_t gain_bytes, double cost_ms, std::size_t min_gain_bytes, double max_cost_ms) noexcept
{
    return gain_bytes >= min_gain_bytes && cost_ms <= max_cost_ms;
}

} // namespace shrink_array
