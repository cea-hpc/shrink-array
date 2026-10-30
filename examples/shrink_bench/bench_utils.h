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

#pragma once

#include "bench_prelude.h" // Int, chunk_config::has_ompt
#include "bench_struct.h"  // Point3, Cell and their std::formatter (bench_text)

#include <atomic>      // std::atomic
#include <bit>         // std::bit_cast (checksum_word)
#include <cstddef>     // size_t
#include <cstdint>     // uint64_t, int64_t
#include <format>      // std::format (bench_text)
#include <random>      // std::mt19937
#include <string>      // std::string
#include <type_traits> // std::true_type, std::false_type
#include <vector>      // std::vector

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS, CHUNK_PMP_THREAD_RANK
#include "chunk/bit.h"       // shrink_array::is_bool_like_v (bench_text), is_packed_bit_v (checksum_word)

// -----------------------------------------------------------------------------
// Cache-line-padded uint64 used as a per-thread accumulator to avoid false
// sharing when threads write their local checksum contribution.
// -----------------------------------------------------------------------------
struct Uint64Padding
{
    /// @brief Zero-initialise the value.
    Uint64Padding() : value(0) {}
    /// @brief Initialise the value to @p val.
    Uint64Padding(uint64_t val) : value(val) {}

    /// @brief The stored value.
    operator uint64_t() const
    {
        return value;
    }
    /// @brief Bitwise complement of the stored value.
    uint64_t operator~() const
    {
        return ~value;
    }

    /// @brief Add @p other's value to this one.
    /// @return Reference to the stored value.
    uint64_t& operator+=(const Uint64Padding& other)
    {
        value += other.value;
        return value;
    }

    union
    {
        uint64_t value;
        char     padding[cache_line_size];
    };
};

/// @brief A bench value as text, the way operator<< printed it: the narrow
///        character types, bool and Bit as numbers (tab_char would otherwise show
///        a control byte), floating point as %g, complex as "(re,im)".
/// @return The text of @p v.
template <class V>
std::string bench_text(const V& v)
{
    if constexpr (shrink_array::is_bool_like_v<V> || std::is_same_v<V, char> || std::is_same_v<V, signed char> ||
                  std::is_same_v<V, unsigned char>)
    {
        return std::format("{}", static_cast<int>(v));
    }
    else if constexpr (std::is_floating_point_v<V>)
    {
        return std::format("{:g}", v);
    }
    else if constexpr (std::is_same_v<V, ComplexDouble>)
    {
        return std::format("({:g},{:g})", v.real(), v.imag());
    }
    else
    {
        return std::format("{}", v);
    }
}

// -----------------------------------------------------------------------------
// Trait: is T a std::vector<...> ?
// -----------------------------------------------------------------------------
template <typename T>
struct is_vector : public std::false_type
{
};

/// @brief std::vector<T, Alloc> is a vector.
template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : public std::true_type
{
};

/// @brief A class derived from std::vector (the baseline's BaselineVector) is a vector.
template <typename T>
    requires(std::is_base_of_v<std::vector<typename T::value_type, typename T::allocator_type>, T> &&
             !std::is_same_v<T, std::vector<typename T::value_type, typename T::allocator_type>>)
struct is_vector<T> : public std::true_type
{
};

// -----------------------------------------------------------------------------
// Trait: is T a std::atomic<pointer> ?
// -----------------------------------------------------------------------------
template <typename T>
struct is_atomic_pointer : public std::false_type
{
};

/// @brief std::atomic<U*> is an atomic pointer.
template <typename U>
struct is_atomic_pointer<std::atomic<U*>> : public std::true_type
{
};

// -----------------------------------------------------------------------------
// The checksum of an array: the sum, over its elements different from T{}, of
// a hash of the element's VALUE MIXED WITH ITS INDEX. A sum of the bytes alone,
// which no index enters, would miss two chunks swapped, a chunk's content
// copied to the wrong index while another took its place, compensating byte
// errors -- the very class of geometry / re-chunk bugs the check runs hunt.
// The sum stays order-free (a parallel reduction, modulo 2^64).
//   - integral elements hash their VALUE, not their bytes: tab_bool is a
//     std::vector<short> in the baseline and an Array<bool> in a shrink build
//     (bench_array.h), and their checksums must agree; so does a packed Bit
//     (tab_bool_bits and tab_mask in a zero_one_value build), hashed as its bool --
//     its byte folded would never match the baseline's bool / short;
//   - floating-point elements their bit pattern (-0.0 == T{} is skipped, as it
//     always was: the library compresses it to +0.0 by design);
//   - aggregates (complex, Point3, Cell -- no padding, bench_struct.h) their
//     bytes, folded into one word.
// Changing it changes every reference checksum: regenerate the reference
// (multi_launch_stress.py --regen-ref) once.
// -----------------------------------------------------------------------------
namespace bench_checksum
{
/// @brief splitmix64's finaliser: a bijective 64-bit mix.
constexpr std::uint64_t mix64(std::uint64_t value) noexcept
{
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

/// @brief One element as a 64-bit word: its value for an integral type, its
///        bits for a floating one, its bytes folded (FNV-1a) otherwise.
template <typename T>
inline std::uint64_t checksum_word(const T& elem) noexcept
{
    if constexpr (std::is_integral_v<T>)
    {
        return static_cast<std::uint64_t>(elem);
    }
    else if constexpr (shrink_array::is_packed_bit_v<T>)
    {
        return static_cast<std::uint64_t>(static_cast<bool>(elem));
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return std::bit_cast<std::uint64_t>(elem);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return std::bit_cast<std::uint32_t>(elem);
    }
    else
    {
        const auto*   bytes = reinterpret_cast<const unsigned char*>(&elem);
        std::uint64_t hash  = 0xcbf29ce484222325ULL;
        for (std::size_t i = 0; i < sizeof(T); ++i)
        {
            hash = (hash ^ bytes[i]) * 0x100000001b3ULL;
        }
        return hash;
    }
}

/// @brief The contribution of element @p elem at index @p index.
template <typename T>
inline std::uint64_t contribution(std::uint64_t index, const T& elem) noexcept
{
    return mix64(checksum_word(elem) ^ mix64(index + 0x9e3779b97f4a7c15ULL));
}
} // namespace bench_checksum

// -----------------------------------------------------------------------------
// Checksum templates. calculate_checksum() runs its own parallel region;
// calculate_checksum_0() assumes it is already inside one (uses `omp for`
// nowait) and writes into a per-thread accumulator slot.
// -----------------------------------------------------------------------------
/// @brief Checksum of @p vec, in its own parallel region: the sum of bench_checksum::contribution(k, element)
///        over the elements different from V::value_type{}.
/// @param vec Container to sum.
/// @return The bitwise complement of the sum.
template <typename V>
#if defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__clang__)
// Same pin as calculate_checksum_0 below: at -O3, gcc 16 dies on this loop
// instantiated on a struct element (24-byte Point3), an ICE in the RTL
// jump-label pass of the serial build.
__attribute__((optimize("O2")))
#endif
uint64_t calculate_checksum(const V& vec)
{
    uint64_t checksum = 0;

    using T      = V::value_type;
    const Int nn = vec.size();

    constexpr T zero{}; // T{}, not 0: the struct element types have no conversion from Int

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

    // `omp parallel for schedule(static) reduction(+ : ...)` with OpenMP.
    checksum = shrink_array::pmp::parallel_sum<uint64_t, 1>(Int{0}, nn, shrink_array::pmp::Options{},
                                                            [&](Int k, auto& acc)
                                                            {
                                                                const T element = vec[k];

                                                                if (element != zero)
                                                                {
                                                                    acc[0] += bench_checksum::contribution(
                                                                        static_cast<std::uint64_t>(k), element);
                                                                }
                                                            })[0];

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);

    // Return the bitwise complement.
    return ~checksum;
}

/// @brief Inside a parallel region (`omp for nowait`): add bench_checksum::contribution(k, element) of every
///        element of @p vec different from V::value_type{} to the calling thread's slot of @p checksum_thr.
/// @param team The region's team (bench_par::region).
/// @param vec Container to sum.
/// @param checksum_thr Per-thread accumulators, indexed by thread rank.
template <typename V>
#if defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__clang__)
__attribute__((optimize("O2")))
#endif
void calculate_checksum_0(bench_par::Team& team, const V& vec, std::vector<Uint64Padding>& checksum_thr)
{
    using T      = V::value_type;
    const Int nn = vec.size();

    constexpr T zero{};

    // The rank in the region's team (a TBB task's virtual rank): one slot per share.
    auto& checksum = checksum_thr[static_cast<std::size_t>(team.rank())];

    team.for_static<Int>(
        0, nn,
        [&](const Int k) __attribute__((always_inline))
        {
            const T element = vec[k];

            if (element != zero)
            {
                checksum +=
                    static_cast<Uint64Padding>(bench_checksum::contribution(static_cast<std::uint64_t>(k), element));
            }
        },
        bench_par::nowait);
}

/// @brief A container reference with one padded checksum accumulator per thread.
template <typename V>
struct VectorChecksum
{
    /// @brief Keep a reference to @p container and size checksum_thr to CHUNK_PMP_MAX_THREADS zeroed slots.
    /// @param container Container to checksum.
    VectorChecksum(const V& container) : vec(container)
    {
        checksum_thr.resize(CHUNK_PMP_MAX_THREADS, 0);
    }
    const V&                   vec;
    std::vector<Uint64Padding> checksum_thr;
};

// -----------------------------------------------------------------------------
// Detect at compile time whether the array type has a set_second_value(v)
// member (only zero_one_value::Array / zero_any::Array do; plain std::vector doesn't).
// -----------------------------------------------------------------------------
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

// -----------------------------------------------------------------------------
// The same for set_debug_name(name): the chunked arrays print it in their
// end-of-life stats line, next to a table number that every re-chunk changes.
// -----------------------------------------------------------------------------
template <typename Arr>
void maybe_set_debug_name(Arr& arr, const char* name)
{
    if constexpr (requires(Arr& probe) { probe.set_debug_name(name); })
    {
        arr.set_debug_name(name);
    }
}

// -----------------------------------------------------------------------------
// Non-template utilities defined in bench_utils.cpp.
// -----------------------------------------------------------------------------
extern bool debug_flag;

extern const std::vector<std::string> valid_args;

/// @brief Write @p seed to @p filename; reports an error on std::cerr if the file cannot be opened.
/// @param seed Seed to save.
/// @param filename Destination file.
void save_seed_to_file(unsigned seed, const std::string& filename);
/// @brief Read a seed from @p filename.
/// @param filename Source file.
/// @return The seed read, or 0 when the file cannot be opened.
unsigned load_seed_to_file(const std::string& filename);
/// @brief Seed @p rng from @p seed_file_name; when that yields 0, draw a seed from std::random_device and save it.
/// @param rng Generator to seed.
/// @param seed_file_name Seed file.
void initialize_rng(std::mt19937& rng, const std::string& seed_file_name);

/// @brief Bytes to MiB (the bench's "Mo"). A pure conversion: no I/O.
/// @param memory_value Size in bytes.
/// @return The same size in MiB.
double convert_memory(double memory_value);
/// @brief Print the memory line labelled @p msg: peak and current resident size, plus the parked (and, with
///        CHUNK_STATS, materialised) bytes in shrink builds.
/// @param msg Label of the line.
void memory_consumption(const std::string& msg);

/// @brief Linear map of @p index from [0, sz] to [0, sz1].
/// @return (sz1 / sz) * index, truncated to Int.
Int linear_correlation(Int index, Int sz, Int sz1);
/// @brief Power-law map of @p index from [0, sz] to [0, sz1].
/// @return sz1 * (index / sz)^rate, truncated to Int.
Int exponential_correlation(Int index, Int sz, Int sz1, double rate);
/// @brief Square-root map of @p index from [0, sz] to [0, sz1].
/// @return sz1 * sqrt(index / sz), truncated to Int.
Int interpolation(Int index, Int sz, Int sz1);

/// @brief Edit distance between @p lhs and @p rhs (insertions, deletions, substitutions).
/// @return The Levenshtein distance.
Int levenshtein(const std::string& lhs, const std::string& rhs);
/// @brief Print the valid command-line options (valid_args) on std::cout.
void print_help();
/// @brief The entry of valid_args closest to @p input by levenshtein() distance.
/// @param input Unrecognised argument.
/// @return The closest valid argument (empty when valid_args is empty).
std::string suggest_closest(const std::string& input);
