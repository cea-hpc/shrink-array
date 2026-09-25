#pragma once

// ============================================================================
// Re-chunk cost model (phase 2 of adaptive chunk sizing)
// ============================================================================
//
// Estimates the time cost of re-chunking one array from geometry S_old to
// S_new, so the phase-3 decision can weigh it against the memory gain the
// ChunkSizeAdvisor predicts. Objective recap: minimum memory; this module
// supplies the "what does it cost to get there" half.
//
// Key insight (measured): a re-chunk only pays for MATERIALISED chunks.
// Compressed (constant) chunks carry no private buffer — their value lives in
// the shared cst buffer — so re-chunking them is a pointer/metadata operation,
// negligible next to copying real data. With ~0.1% of chunks materialised in
// practice, re-chunking a compressible array is cheap (sub-millisecond on the
// data the advisor was run against).
//
// Cost formula:
//   cost = (mat_old * S_old) * cost_copy_ns          [re-write materialised data]
//        + (mat_old + mat_new) * cost_allocfree_ns    [free old + alloc new bufs]
//
//   mat_old / mat_new : number of MATERIALISED chunks before / after (the
//                       advisor already counts these per candidate size).
//   S_old             : old chunk width (elements).
//   cost_copy_ns      : per-element re-write cost (memory-bandwidth bound).
//   cost_allocfree_ns : per-chunk alloc+free cost (roughly size-independent).
//
// The two constants are HARDWARE-SPECIFIC. The defaults below were measured on
// a local x86 dev box with ::operator new/delete (a pessimistic proxy — the
// real fast-alloc pool is faster, so the default cost_allocfree is an upper
// bound, which is the safe side for a "should we re-chunk?" decision). On ARM
// Grace with the real pool they will differ: call calibrate_rechunk_cost()
// once at startup to overwrite them with measured values, or set them from a
// config. The formula's SHAPE is hardware-independent; only the scalars move.
//
// This module has no side effects and is not gated: it is pure arithmetic plus
// an opt-in calibration routine. Callers decide when (if ever) to invoke it.

#include <chrono>
#include <cstddef>
#include <cstring>
#include <vector>

namespace shrink_array
{

// Hardware-specific scalars. Defaults are local-x86 measurements; recalibrate
// on the target (see calibrate_rechunk_cost). Mutable so calibration can update
// them in place.
inline double g_rechunk_cost_copy_ns      = 0.65; // per re-written element
inline double g_rechunk_cost_allocfree_ns = 25.0; // per chunk buffer alloc+free

// Estimated re-chunk time in milliseconds for one array going S_old -> S_new,
// given how many chunks are materialised before (mat_old) and after (mat_new).
// Compressed chunks are intentionally omitted (near-zero cost).
[[nodiscard]] inline double rechunk_cost_ms(long mat_old, unsigned S_old, long mat_new, unsigned /*S_new*/) noexcept
{
    const double elems_to_copy = static_cast<double>(mat_old) * static_cast<double>(S_old);
    const double alloc_ops     = static_cast<double>(mat_old) + static_cast<double>(mat_new);
    const double ns            = elems_to_copy * g_rechunk_cost_copy_ns + alloc_ops * g_rechunk_cost_allocfree_ns;
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

// Measure cost_copy_ns and cost_allocfree_ns on the CURRENT hardware and store
// them into the globals above. Call once at startup on the target machine. T is
// the element type whose buffers you actually allocate (sizeof matters for the
// alloc cost). sample_size tunes the copy measurement; alloc_reps the alloc one.
// NOTE: this uses ::operator new/delete as a portable proxy. If you want the
// exact fast-alloc pool figure, feed a lambda variant (left as an extension) —
// the pool is faster, so this proxy stays a safe upper bound.
template <typename T>
void calibrate_rechunk_cost(std::size_t sample_size = (16u << 20),
                            unsigned    chunk_size  = 2048,
                            int         alloc_reps  = 200000) noexcept
{
    using clk  = std::chrono::steady_clock;
    auto to_ns = [](clk::duration dur) { return std::chrono::duration<double, std::nano>(dur).count(); };

    // --- copy component ---
    {
        std::vector<T> src(sample_size), dst(sample_size);
        for (std::size_t i = 0; i < sample_size; ++i)
            src[i] = static_cast<T>(i & 1023);
        std::memcpy(dst.data(), src.data(), (sample_size < 4096 ? sample_size : 4096) * sizeof(T));
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < sample_size; ++i)
            dst[i] = src[i];
        const auto t1 = clk::now();
        // prevent the loop being optimised away
        volatile T sink = dst[sample_size / 2];
        (void)sink;
        g_rechunk_cost_copy_ns = to_ns(t1 - t0) / static_cast<double>(sample_size);
    }

    // --- alloc+free component ---
    {
        const std::size_t bytes = static_cast<std::size_t>(chunk_size) * sizeof(T);
        for (int i = 0; i < 1000; ++i)
        {
            void* ptr = ::operator new(bytes);
            ::operator delete(ptr);
        }
        volatile std::size_t sink  = 0;
        const auto           start = clk::now();
        for (int i = 0; i < alloc_reps; ++i)
        {
            void* ptr = ::operator new(bytes);
            std::memset(ptr, 0, 64);
            sink += reinterpret_cast<std::size_t>(ptr);
            ::operator delete(ptr);
        }
        const auto stop = clk::now();
        (void)sink;
        g_rechunk_cost_allocfree_ns = to_ns(stop - start) / static_cast<double>(alloc_reps);
    }
}

} // namespace shrink_array
