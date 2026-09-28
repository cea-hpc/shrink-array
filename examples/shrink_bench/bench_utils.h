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
    union
    {
        uint64_t value;
        char     padding[CACHE_LINE_SIZE];
    };

    operator uint64_t() const
    {
        return value;
    }
    uint64_t operator~() const
    {
        return ~value;
    }

    uint64_t& operator+=(const uint64_t_padding& other)
    {
        value += other.value;
        return value;
    }

    uint64_t_padding() : value(0) {}
    uint64_t_padding(uint64_t val) : value(val) {}
};

// -----------------------------------------------------------------------------
// Trait: is T a std::vector<...> ?
// -----------------------------------------------------------------------------
template <typename T>
struct is_vector : std::false_type
{
};

template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type
{
};

// -----------------------------------------------------------------------------
// Trait: is T a std::atomic<pointer> ?
// -----------------------------------------------------------------------------
template <typename T>
struct is_atomic_pointer : std::false_type
{
};

template <typename U>
struct is_atomic_pointer<std::atomic<U*>> : std::true_type
{
};

// -----------------------------------------------------------------------------
// Checksum templates. calculate_checksum() runs its own parallel region;
// calculate_checksum_0() assumes it is already inside one (uses `omp for`
// nowait) and writes into a per-thread accumulator slot.
// -----------------------------------------------------------------------------
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

template <typename V>
struct VectorChecksum
{
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

void     save_seed_to_file(unsigned seed, const std::string& filename);
unsigned load_seed_to_file(const std::string& filename);
void     initializeRNG(std::mt19937& rng, const std::string& seed_file_name);

double convert_memory(double memoryValue);
void   memory_consumption(const std::string& msg);

Int linear_correlation(Int i, Int sz, Int sz1);
Int exponential_correlation(Int i, Int sz, Int sz1, double p);
Int interpolation(Int i, Int sz, Int sz1);

Int         levenshtein(const std::string& a, const std::string& b);
void        print_help();
std::string suggest_closest(const std::string& input);
