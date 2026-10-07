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

// The count of arrays alive in the process, for the teardown verbs
// (release_chunk_pools, destroy_block_chunk, chunk_std.cpp): with arrays still
// alive -- a global or static array, destroyed after main returns -- they keep
// the structures those arrays will park their buffers in, rather than pull
// them from under them.

#include <atomic>  // std::atomic
#include <cstdint> // std::int64_t

namespace shrink_array::internal
{

/// @brief Arrays alive in the process: every array object, empty or moved-from
///        ones included, from its construction to the end of its destructor.
inline std::atomic<std::int64_t> g_live_arrays{0};

/// @brief The arrays alive now (live_array_count() == 0: no array left).
[[nodiscard]] inline std::int64_t live_array_count() noexcept
{
    return g_live_arrays.load(std::memory_order_acquire);
}

/**
 * @brief Member of every array: counts it in g_live_arrays.
 * @details A member, not code in the arrays' constructors, so that every
 *          constructor -- default, copy, move, the sized ones -- counts. It is
 *          destroyed after the array's destructor body has released the
 *          storage, so a count of zero means every buffer is back in the pools.
 */
class LiveArrayToken
{
  public:
    LiveArrayToken() noexcept
    {
        g_live_arrays.fetch_add(1, std::memory_order_relaxed);
    }
    LiveArrayToken(const LiveArrayToken& /*other*/) noexcept : LiveArrayToken() {}
    LiveArrayToken(LiveArrayToken&& /*other*/) noexcept : LiveArrayToken() {}
    LiveArrayToken& operator=(const LiveArrayToken& /*other*/) noexcept = default;
    LiveArrayToken& operator=(LiveArrayToken&& /*other*/) noexcept      = default;
    ~LiveArrayToken()
    {
        g_live_arrays.fetch_sub(1, std::memory_order_release);
    }
};

} // namespace shrink_array::internal
