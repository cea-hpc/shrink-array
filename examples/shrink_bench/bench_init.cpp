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

/// @file bench_init.cpp
/// @brief Geometry validation and library start-up. See bench_phases.h.

#include <cstdlib> // std::abort
#include <cstdio>  // stderr
#include <print>   // std::println, std::print

#include "bench_phases.h" // bench_init_library (defined here)

// As in shrink_bench.cpp, where this body comes from: it names ParamChunk,
// ChunkDef and __chunk_size unqualified. In a .cpp the directive stays
// contained -- it would have no place in a header.
using namespace shrink_array;

void bench_init_library()
{
    if constexpr (chunk_config::is_shrink_enabled)
    {
        constexpr Int chunk_size = static_cast<Int>(__chunk_size);

        static_assert(chunk_size <= ChunkDef::size_max);

        if constexpr (chunk_size == 0)
        {
            std::println(stderr, "The chunk size can't be null");
            std::abort();
        }

        if (const bool is_pow2 = ParamChunk::check_pow2(chunk_size); !is_pow2)
        {
            std::print(stderr, "The chunk size must be power of two");
            std::abort();
        }

        if constexpr (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
        {
            std::println(stderr, "The chunk size must must be superior or equal to {} and inferior or equal to {}",
                         ParamChunk::size_min, ParamChunk::size_max);
            std::abort();
        }

        // The checks above only validate __chunk_size — the COMPILE-TIME default.
        // In dynamic builds the effective geometry is chunk_dyn_size (seeded just
        // below), now overridable via -DDYN_CHUNK_SIZE: validate it with the same
        // rules. chunk_dyn_size is constexpr, so this is free and fires at compile
        // time (also checked in static builds, where the seed is defined but
        // unused — keeping it valid costs nothing).
        static_assert(std::has_single_bit(shrink_array::chunk_dyn_size),
                      "chunk_dyn_size (DYN_CHUNK_SIZE) must be a power of two");
        static_assert(shrink_array::chunk_dyn_size >= ParamChunk::size_min &&
                          shrink_array::chunk_dyn_size <= ParamChunk::size_max,
                      "chunk_dyn_size (DYN_CHUNK_SIZE) out of [ParamChunk::size_min, size_max]");

        if constexpr (build_config::chunk_dynamic)
        {
            // Seed the process-wide active chunk size before the allocators are
            // primed by init_indexes. In static builds this branch is discarded.
            shrink_array::set_active_chunk_size(shrink_array::chunk_dyn_size);
        }

        ParamChunk::init_indexes(false);
        ParamChunk::print_info_chunk();
    }
}
