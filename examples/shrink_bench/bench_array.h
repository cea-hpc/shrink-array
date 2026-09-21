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

#include <type_traits>
#include <vector>

#include "bench_prelude.h"
#include "bench_struct.h" // Point3, Cell: the aggregate element types

// The bulk range API: the chunked arrays' overloads (constrained on the
// ChunkedArray concept, so they select nothing in the std::vector build) on
// top of the std::vector fallback they include. Also what drags in the
// ArrayChunk* definitions the alias below selects between.
#include "chunk/array_chunk_range_ops.h"

namespace shrink_array
{

template <typename T>
using ActiveArray =
    std::conditional_t<chunk_config::use_chunk2a,
                       ArrayChunk2A<T>,
                       std::conditional_t<chunk_config::use_chunk2,
                                          ArrayChunk2<T>,
                                          std::conditional_t<chunk_config::use_chunk, ArrayChunk<T>, std::vector<T>>>>;


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
using BenchBoolArray = std::conditional_t<chunk_config::is_shrink_enabled, Array<bool>, std::vector<short>>;

/**
 * @brief The fifteen arrays the workload drives, by reference, plus their
 *        geometry.
 *
 * @note References, not values: the arrays stay owned by main(), which builds
 *       and destroys them. This is a handle onto them, cheap to pass and
 *       impossible to copy by accident.
 * @warning tab_bool's type varies by build -- see BenchBoolArray above.
 */
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
inline constexpr int bench_double_bands = 200000;

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
    Array<Cell>&           tab_cell;  ///< an aggregate element: an int and a float

    int sz;       ///< element count of the double arrays (others are sz, 2*sz or sz/2)
    int interval; ///< the workload's stride, get_size_interval()
    /// false under --no_struct: tab_point and tab_cell are of size 1 and every
    /// phase, check, stat and checksum of theirs is skipped.
    bool with_struct;
};
