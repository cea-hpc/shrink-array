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

/**
 * @file bench_array.h
 * @brief The array type this build benches, as one alias.
 *
 * Lived in shrink_bench.cpp until its self-check sections moved to their own
 * translation unit: both need the same alias, so it had to become shared.
 * Nothing else changed -- ActiveArray and Array are verbatim what they were.
 */

#pragma once

#include <cstdint> // std::uint64_t (bench_mask_hash)
#include <type_traits>
#include <vector> // std::vector<bool>: tab_mask in the baseline

#include "bench_prelude.h"

#include "bench_struct.h" // Point3, Cell: the aggregate element types

// The bulk range API: the chunked arrays' overloads (constrained on the
// ChunkedArray concept, so they select nothing in the std::vector build) on
// top of the std::vector fallback they include. Also what drags in the
// ArrayChunk* definitions the alias below selects between.
#include "chunk/array_chunk_range_ops.h"
#include "chunk/array_chunk_bits.h" // BENCH_BOOL_BITS: tab_bool packed 8 per byte

namespace shrink_array
{

// `Int` is the bench's index type (bench_int.h): `int` by default, `long`
// under -DBENCH_INT64. It is handed to the library as `ind_type`, so the array
// indexes exactly as wide as the bench computes -- a 64-bit bench on a 32-bit
// array would overflow inside the library instead of inside main().
template <typename T>
using ActiveArray = std::conditional_t<
    chunk_config::use_chunk2a,
    ArrayChunk2A<T, Int>,
    std::conditional_t<chunk_config::use_chunk2,
                       ArrayChunk2<T, Int>,
                       std::conditional_t<chunk_config::use_chunk, ArrayChunk<T, Int>, std::vector<T>>>>;


// Runtime chunk size for CHUNK_DYNAMIC builds. Seeded into g_active_chunk_size
// before init_indexes (below), so every array constructed afterwards adopts this
// geometry by default. Ignored in static builds (geometry fixed by __chunk_size).
// Must be a power of two in [ParamChunk::size_min, ParamChunk::size_max]; 512
// differs from the usual compile-time __chunk_size (64) so a dynamic run visibly
// exercises a new geometry.
#ifndef DYN_CHUNK_SIZE
    // Deliberately sub-optimal starting geometry for dynamic builds: 64 is small,
    // so a run visibly exercises the re-chunk path (the advisor grows it). Override
    // at configure time with -DDYN_CHUNK_SIZE=<pow2 in [size_min, size_max]>.
    #define DYN_CHUNK_SIZE 64
#endif
inline constexpr unsigned int chunk_dyn_size = DYN_CHUNK_SIZE;
} // namespace shrink_array

/// @brief Short name for the array type this build benches.
template <typename T>
using Array = shrink_array::ActiveArray<T>;

// ---------------------------------------------------------------------------
// The workload's arrays and the values it fills them with.
//
// Both used to be locals of main(). They are shared state now because the bench
// phases live in their own translation units: a phase that writes tab1..tab_ulong
// has to be handed them, and handing them over as a struct of REFERENCES keeps
// the extracted phases ORDINARY FUNCTIONS -- which is the whole point. A
// template over the array types would be re-instantiated in every caller and buy
// nothing; that is what limited the first extraction attempt.
// ---------------------------------------------------------------------------

/// @name The second values the workload writes
/// Moved verbatim out of main(); `inline constexpr` so every phase sees the same
/// ones. Names unchanged so the moved code reads exactly as it did.
/// @{
inline constexpr double second_value_d  = 1.;
inline constexpr double second_value_d0 = 3.5;
inline constexpr double second_value_d1 = 2;
inline constexpr double second_value_d2 = 100;

inline constexpr float second_value_f0 = 1;
inline constexpr float second_value_f1 = 12;

inline constexpr short second_value_s0 = 5;
inline constexpr short second_value_s1 = 7;
inline constexpr short second_value_s2 = 4;

inline constexpr unsigned char second_value_c0 = 3;
inline constexpr unsigned char second_value_c1 = 4;

inline constexpr int            second_value_i       = 1;
inline constexpr int            other_second_value_i = 10;
inline constexpr long           second_value_l       = 1L;
inline constexpr unsigned long  second_value_ul      = 1UL;
inline constexpr bool           second_value_b       = true;
inline constexpr complex_double second_value_c       = std::complex<double>(1., 0.);
inline constexpr complex_double other_second_value_c = std::complex<double>(10., 20.);

inline constexpr complex_double zero_complex = std::complex<double>(0., 0.);
/// @}

/// @brief What main() declares tab_bool as. A shrink build uses a real
///        Array<bool>; the std::vector baseline uses short, because
///        std::vector<bool>'s proxy is useless here. The asymmetry predates the
///        split -- this alias just names it so BenchArrays can hold a reference.
///
///        BENCH_BOOL_BITS (the shrink_bench_chunk2bits_* targets): tab_bool is
///        an ArrayChunkBits -- bools packed 8 per byte -- instead, everything
///        else unchanged, so the two builds compare the containers on the same
///        workload and against the same data_tab_bool_*.bin references.
#ifdef BENCH_BOOL_BITS
static_assert(chunk_config::use_chunk2, "BENCH_BOOL_BITS: ArrayChunkBits is ArrayChunk2<Bit>, a chunk2 build");
using BenchBoolArray = shrink_array::ArrayChunkBits<Int>;
#else
using BenchBoolArray = std::conditional_t<chunk_config::is_shrink_enabled, Array<bool>, std::vector<short>>;
#endif

/// @brief Distinct levels tab_quant may take, hence the distinct second values
///        it registers.
///
/// @details Exercises the registry at a scale the other arrays never reach:
///          they each declare one second value, while a quantised field gives
///          every level its own. 200 000 is past the 65 536 a dyn build capped
///          at while each value carried a filled array, and far under the 4 M
///          the value table allows now. A constant rather than a build option
///          on purpose -- the point is to exercise the capacity, not to make it
///          adjustable.
inline constexpr Int bench_double_bands = 200000;

/**
 * @brief What tab_mask is: a large bool array of random masks and scattered
 *        flags (bench_phase_mask.cpp), where bools are NOT mostly constant.
 *
 * @details The array a bit-packed container is for, and the one std::vector's
 *          own bit packing is the reference of:
 *            - chunk2 builds: ArrayChunkBits (bit.h), 8 bools per byte in a
 *              materialised chunk, and constant chunks at no data cost;
 *            - chunk (v1) / chunk2a builds: the variant's Array<bool>, one byte
 *              per bool -- ArrayChunkBits is chunk2's machinery;
 *            - the std::vector baseline: std::vector<bool>, one bit per bool
 *              and nothing compressed. Its phases write whole 64-element
 *              blocks, one thread per block, so no word of it has two writers.
 */
using BenchMaskArray = std::conditional_t<
    chunk_config::use_chunk2,
    shrink_array::ArrayChunkBits<Int>,
    std::conditional_t<chunk_config::is_shrink_enabled, Array<bool>, std::vector<bool>>>;

/// @brief tab_mask's default size, as a fraction of sz (--mask_scale sets it).
/// @details 1/32. Its per-iteration update writes ~1/16 of the random and
///          scattered blocks and 1/32 of the constant segments, element by
///          element through the proxies in a chunk (v1) / chunk2A build: at the
///          8 x sz it was created with (capped at 2^31 in the 32-bit builds) it
///          took ~75% of a 43 s chunk2a run (64 threads) that is ~7 s without it.
///          1/32 keeps thousands of 4096-element segments of each kind.
inline constexpr double bench_mask_default_scale = 1.0 / 32.0;

/// @brief Segment of tab_mask with one kind of content: whole chunks at every
///        geometry up to 4096, so a constant segment is constant chunks.
inline constexpr Int bench_mask_segment = 4096;

/// @brief Elements per block of the tab_mask phases: one 64-bit word of a
///        packed layout, written by one thread.
inline constexpr Int bench_mask_block = 64;

/// @brief A 64-bit mix (splitmix64's finaliser): the deterministic "random"
///        of tab_mask -- integers only, so every build computes the same bits.
[[nodiscard]] constexpr std::uint64_t bench_mask_hash(std::uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/// @brief What a segment of tab_mask holds.
enum class BenchMaskKind
{
    all_false, ///< 30 %: a constant chunk -- compressible
    all_true,  ///< 30 %: the other constant -- compressible
    random,    ///< 25 %: a random mask, half the bits set
    scattered  ///< 15 %: scattered flags, ~3 % set
};

// Why 60 % constant. ArrayChunkBits costs ~16 bytes of header per chunk plus
// S/8 bytes per chunk that is not constant; std::vector<bool> costs S/8 per
// chunk's worth of elements. With a fraction c of constant chunks the ratio is
// 16 / (S/8) + (1 - c): at S = 256, 0.5 + 0.4 = 0.9 of std::vector<bool>
// (break-even at c = 0.5), at S = 1024, 0.125 + 0.4 = 0.53. Past a slab
// allocator's floor (about 512 KiB per thread rank): below a few million
// elements that floor, not the layout, is what a measurement sees.

/// @brief The kind of segment @p seg (fixed for the run).
[[nodiscard]] constexpr BenchMaskKind bench_mask_kind(Int seg) noexcept
{
    const auto h = bench_mask_hash(static_cast<std::uint64_t>(seg) * 0x51ED27ull) % 100;
    if (h < 30)
        return BenchMaskKind::all_false;
    if (h < 60)
        return BenchMaskKind::all_true;
    if (h < 85)
        return BenchMaskKind::random;
    return BenchMaskKind::scattered;
}

/// @brief The starting value of tab_mask[i].
[[nodiscard]] constexpr bool bench_mask_initial(Int i) noexcept
{
    const auto h = bench_mask_hash(static_cast<std::uint64_t>(i));
    switch (bench_mask_kind(i / bench_mask_segment))
    {
    case BenchMaskKind::all_false:
        return false;
    case BenchMaskKind::all_true:
        return true;
    case BenchMaskKind::random:
        return (h & 1u) != 0;
    case BenchMaskKind::scattered:
        return h % 32 == 0;
    }
    return false;
}

/**
 * @brief The fifteen arrays the workload drives, by reference, plus their
 *        geometry.
 *
 * @note References, not values: the arrays stay owned by main(), which builds
 *       and destroys them. This is a handle onto them, cheap to pass and
 *       impossible to copy by accident.
 * @warning tab_bool's type varies by build -- see BenchBoolArray above.
 */
struct BenchArrays
{
    Array<double>& tab1;
    Array<double>& tab2;
    Array<double>& tab3;
    Array<double>& tab4;
    Array<double>& tab5;

    Array<int>&            tab_int;
    BenchBoolArray&        tab_bool;
    Array<complex_double>& tab_complex;
    Array<float>&          tab_float;
    Array<short>&          tab_short;
    Array<unsigned char>&  tab_char;
    Array<long>&           tab_long;
    Array<unsigned long>&  tab_ulong;
    Array<double>&         tab_quant; ///< quantised: up to bench_double_bands distinct second values
    Array<Point3>&         tab_point; ///< an aggregate element: three doubles (bench_struct.h)
    Array<Cell>&           tab_cell;  ///< an aggregate element: an Int and a float
    BenchMaskArray&        tab_mask;  ///< mask_scale * sz bools (--mask_scale, default 1/32): masks and flags (bench_phase_mask.cpp)

    Int sz       = 0; ///< element count of the double arrays (others are sz, 2*sz or sz/2)
    Int interval = 0; ///< the workload's stride, get_size_interval()
    /// false under --no_struct: tab_point and tab_cell are of size 1 and every
    /// phase, check, stat and checksum of theirs is skipped.
    bool with_struct = true;
    /// false under --no_mask: tab_mask is of size 1 and its phases, check and
    /// compression are skipped.
    bool with_mask = true;
};
