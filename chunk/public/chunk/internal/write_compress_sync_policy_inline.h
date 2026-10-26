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
#include "chunk/internal/write_compress_sync_policy.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::WriteCompressSyncPolicy<ProxyT, Enabled>
// ---------------------------------------------------------------------------

template <typename ProxyT, bool Enabled>
void WriteCompressSyncPolicy<ProxyT, Enabled>::ensure_initialized()
{
    static std::once_flag flag;
    std::call_once(flag,
                   []
                   {
                       per_thread_writing =
                           std::vector<WritingFlag>(static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY));
                       num_thread_flushing.store(0, std::memory_order_relaxed);
                       // Process-wide and idempotent; done here so it precedes
                       // the first parallel write of this variant.
                       dekker::register_membarrier();
                   });
}

template <typename ProxyT, bool Enabled>
[[gnu::always_inline]] inline void WriteCompressSyncPolicy<ProxyT, Enabled>::announce_write(int thr,
                                                                                            const void* chunk) noexcept
{
    if (dekker::asymmetric_active()) [[likely]]
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EventCode::policy_begin_write, static_cast<uint32_t>(thr));
        }
        per_thread_writing[thr].v.store(chunk, std::memory_order_relaxed);
        std::atomic_signal_fence(std::memory_order_seq_cst); // compiler: keep the store before the loads
        return;
    }
    begin_write(thr, chunk);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

template <typename ProxyT, bool Enabled>
template <typename ChunkT>
bool WriteCompressSyncPolicy<ProxyT, Enabled>::any_thread_writing_on_marked(int max_thr) noexcept
{
    for (int i = 0; i < max_thr; ++i)
    {
        const auto* chunk = static_cast<const ChunkT*>(per_thread_writing[i].v.load(std::memory_order_acquire));
        if (chunk != nullptr && chunk->steal_marked())
        {
            return true;
        }
    }
    return false;
}

template <typename ProxyT, bool Enabled>
bool WriteCompressSyncPolicy<ProxyT, Enabled>::any_thread_writing(int max_thr) noexcept
{
    for (int i = 0; i < max_thr; ++i)
    {
        if (per_thread_writing[i].v.load(std::memory_order_acquire) != nullptr)
        {
            return true;
        }
    }
    return false;
}

} // namespace shrink_array
