/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include "bench_prelude.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "chunk/PMP_CHUNK.h"

// -----------------------------------------------------------------------------
// Cache-line-padded uint64 used as a per-thread accumulator to avoid false
// sharing when threads write their local checksum contribution.
// -----------------------------------------------------------------------------
struct uint64_t_padding
{
    /// @brief Zero-initialise the value.
    uint64_t_padding() : value(0) {}
    /// @brief Initialise the value to @p val.
    uint64_t_padding(uint64_t val) : value(val) {}

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
    uint64_t& operator+=(const uint64_t_padding& other)
    {
        value += other.value;
        return value;
    }

    union
    {
        uint64_t value;
        char     padding[CACHE_LINE_SIZE];
    };
};

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
// Checksum templates. calculate_checksum() runs its own parallel region;
// calculate_checksum_0() assumes it is already inside one (uses `omp for`
// nowait) and writes into a per-thread accumulator slot.
// -----------------------------------------------------------------------------
/// @brief Checksum of @p vec, in its own parallel region: the sum of the bytes of every element different from
///        V::value_type{}.
/// @param vec Container to sum.
/// @return The bitwise complement of the sum.
template <typename V>
#if defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__clang__)
// Same pin as calculate_checksum_0 below: at -O3, gcc 16 dies on this loop
// instantiated on a struct element (24-byte Point3), an ICE in the RTL
// jump-label pass of the serial build.
__attribute__((optimize("O2")))
#endif
uint64_t
calculate_checksum(const V& vec)
{
    uint64_t checksum = 0;

    using T      = V::value_type;
    const Int nn = vec.size();

    constexpr T zero{}; // T{}, not 0: the struct element types have no conversion from Int

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

#pragma omp parallel for reduction(+ : checksum)
    for (Int k = 0; k < nn; ++k)
    {
        const T element = vec[k];

        if (element != zero)
        {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&element);
            for (size_t i = 0; i < sizeof(T); ++i)
            {
                checksum += bytes[i];
            }
        }
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);

    // Return the bitwise complement (matches historical behaviour).
    return ~checksum;
}

/// @brief Inside a parallel region (`omp for nowait`): add the bytes of every element of @p vec different from
///        V::value_type{} to the calling thread's slot of @p checksum_thr.
/// @param vec Container to sum.
/// @param checksum_thr Per-thread accumulators, indexed by thread rank.
template <typename V>
#if defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__clang__)
__attribute__((optimize("O2")))
#endif
void calculate_checksum_0(const V& vec, std::vector<uint64_t_padding>& checksum_thr)
{
    using T      = V::value_type;
    const Int nn = vec.size();

    constexpr T zero{};

    auto& checksum = checksum_thr[CHUNK_PMP_THREAD_RANK];

#pragma omp for schedule(static) nowait
    for (Int k = 0; k < nn; ++k)
    {
        const T element = vec[k];

        if (element != zero)
        {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&element);
            for (size_t i = 0; i < sizeof(T); ++i)
            {
                checksum += static_cast<uint64_t_padding>(bytes[i]);
            }
        }
    }
}

/// @brief A container reference with one padded checksum accumulator per thread.
template <typename V>
struct VectorChecksum
{
    /// @brief Keep a reference to @p vec_ and size checksum_thr to CHUNK_PMP_MAX_THREADS zeroed slots.
    /// @param vec_ Container to checksum.
    VectorChecksum(const V& vec_) : vec(vec_)
    {
        checksum_thr.resize(CHUNK_PMP_MAX_THREADS, 0);
    }
    const V&                      vec;
    std::vector<uint64_t_padding> checksum_thr;
};

// -----------------------------------------------------------------------------
// Detect at compile time whether the array type has a set_second_value(v)
// member (only ArrayChunk2 / ArrayChunk2A do; plain std::vector doesn't).
// -----------------------------------------------------------------------------
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
        a.set_second_value(v);
}

// -----------------------------------------------------------------------------
// Non-template utilities defined in bench_utils.cpp.
// -----------------------------------------------------------------------------
// __mem_page_size is defined in bench_utils.cpp. fast_allocators declares it
// too, but that header is not in a CHUNK_STD_ALLOC build: the declaration has
// to be here (removed once as a duplicate -- the std-alloc builds broke).
extern int  __mem_page_size; // NOT Int: shared with fast_allocators, which declares it `int`
extern bool __debug_flag;

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
void initializeRNG(std::mt19937& rng, const std::string& seed_file_name);

/// @brief Bytes to MiB (the bench's "Mo"). A pure conversion: no I/O.
/// @param memoryValue Size in bytes.
/// @return The same size in MiB.
double convert_memory(double memoryValue);
/// @brief Print the memory line labelled @p msg: peak and current resident size, plus the parked (and, with
///        CHUNK_STATS, materialised) bytes in shrink builds.
/// @param msg Label of the line.
void memory_consumption(const std::string& msg);

/// @brief Linear map of @p i from [0, sz] to [0, sz1].
/// @return (sz1 / sz) * i, truncated to Int.
Int linear_correlation(Int i, Int sz, Int sz1);
/// @brief Power-law map of @p i from [0, sz] to [0, sz1].
/// @return sz1 * (i / sz)^p, truncated to Int.
Int exponential_correlation(Int i, Int sz, Int sz1, double p);
/// @brief Square-root map of @p i from [0, sz] to [0, sz1].
/// @return sz1 * sqrt(i / sz), truncated to Int.
Int interpolation(Int i, Int sz, Int sz1);

/// @brief Edit distance between @p a and @p b (insertions, deletions, substitutions).
/// @return The Levenshtein distance.
Int levenshtein(const std::string& a, const std::string& b);
/// @brief Print the valid command-line options (valid_args) on std::cout.
void print_help();
/// @brief The entry of valid_args closest to @p input by levenshtein() distance.
/// @param input Unrecognised argument.
/// @return The closest valid argument (empty when valid_args is empty).
std::string suggest_closest(const std::string& input);
