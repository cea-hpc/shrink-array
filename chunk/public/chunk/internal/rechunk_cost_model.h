#pragma once

// ============================================================================
// Re-chunk cost model (phase 2 of adaptive chunk sizing)
// ============================================================================
//
// Estimates the time cost of re-chunking one array from geometry S_old to a
// coarser or finer one, so the phase-3 decision can weigh it against the memory
// gain the ChunkSizeAdvisor predicts. Objective recap: minimum memory; this
// module supplies the "what does it cost to get there" half.
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
// Compressed chunks carry no buffer, but they are not free: the first model
// counted the materialised data alone and said 5 ms for rebuilds measured at
// 85 ms on sparse arrays (tab_long, 7% of the chunks materialised), 10 to 20
// times short, and 0-19 ms for 370-1070 ms in a Debug build.
//
// The constants were fitted (least squares) on 61 rebuilds measured with
// CHUNK_RECHUNK_TRACE: shrink_bench_chunk2_dyn, Release, ARM, 64 threads,
// --divisor 1 and 10, 64 -> 256, 256 -> 1024 and 1024 -> 4096, fifteen
// element types of 1 to 24 bytes, 11 to 179 ms. Median error 7%, worst 33%.
// They are HARDWARE- and THREAD-COUNT-specific (the refill is parallel);
// elsewhere the formula's shape holds and the scalars move. The
// [RECHUNK-DECIDE] lines log the estimate, the chunk and materialised counts,
// and CHUNK_RECHUNK_TRACE the measured total, which is what a refit needs.
//
// This module has no side effects and is not gated: it is pure arithmetic.

#include <cstddef>
#include <cstdint>

namespace shrink_array
{

// Hardware-specific scalars (see above). Mutable so a caller can set measured
// values for its machine.
inline double g_rechunk_cost_elem_ns  = 0.032; // per element of the array
inline double g_rechunk_cost_chunk_ns = 4.2;   // per chunk at the current size
inline double g_rechunk_cost_byte_ns  = 0.042; // per byte of materialised data

// Estimated re-chunk time in milliseconds for one array leaving S_old, given
// its chunk count and materialised-chunk count there, and the bytes one
// element takes in a materialised chunk.
[[nodiscard]] inline double rechunk_cost_ms(std::int64_t n_chunks_old, unsigned S_old, std::int64_t mat_old,
                                            double elem_bytes) noexcept
{
    const double n_elems    = static_cast<double>(n_chunks_old) * static_cast<double>(S_old);
    const double data_bytes = static_cast<double>(mat_old) * static_cast<double>(S_old) * elem_bytes;
    const double ns         = n_elems * g_rechunk_cost_elem_ns +
                      static_cast<double>(n_chunks_old) * g_rechunk_cost_chunk_ns + data_bytes * g_rechunk_cost_byte_ns;
    return ns / 1.0e6;
}

// Convenience: is a re-chunk worth it? gain_bytes is the advisor's predicted
// memory saving; the caller passes a policy (min gain, max cost). Kept separate
// from the cost so phase 3 can layer hysteresis/cooldown on top. Returns true
// when the move clears both thresholds.
[[nodiscard]] inline bool
rechunk_is_worth_it(std::size_t gain_bytes, double cost_ms, std::size_t min_gain_bytes, double max_cost_ms) noexcept
{
    return gain_bytes >= min_gain_bytes && cost_ms <= max_cost_ms;
}

} // namespace shrink_array
