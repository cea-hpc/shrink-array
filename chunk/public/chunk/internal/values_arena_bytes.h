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

/**
 * @file values_arena_bytes.h
 * @brief What the chunk values' arena holds, in every build: the one place
 *        that knows whether there is such an arena.
 *
 * @details Fast-alloc builds carve the chunk buffers out of
 *          fast_alloc::value_arena(); std-alloc builds malloc them, and have
 *          no fast_allocators include path at all. The preprocessor test lives
 *          here, once, so that a memory report can ask with an ordinary
 *          `if constexpr (build_config::use_std_alloc)` or with the optional
 *          below. An `if constexpr` in the caller could not replace this:
 *          fast_alloc::value_arena() is a non-dependent name, looked up even in
 *          a discarded branch, and it is not declared in a std-alloc build.
 */

#include <cstddef>  // std::size_t
#include <optional> // std::optional, std::nullopt

#if !defined CHUNK_STD_ALLOC                // std-alloc builds have no fast_allocators include path
    #include "fast_allocators/slab_arena.h" // fast_alloc::value_arena
#endif

namespace shrink_array::internal
{

/// @brief The values arena's byte counts (see fast_alloc::SlabArena).
struct ArenaBytes
{
    std::size_t resident = 0; ///< pages actually backed
    std::size_t mapped   = 0; ///< address space reserved
    std::size_t live     = 0; ///< slabs handed out to the allocators
};

/// @brief The values arena's byte counts, or nullopt in a std-alloc build
///        (the chunk buffers are plain mallocs: there is no arena to report).
[[nodiscard]] inline std::optional<ArenaBytes> values_arena_bytes() noexcept
{
#if defined CHUNK_STD_ALLOC
    return std::nullopt;
#else
    const auto& va = fast_alloc::value_arena();
    return ArenaBytes{va.resident_bytes(), va.mapped_bytes(), va.live_bytes()};
#endif
}

} // namespace shrink_array::internal
