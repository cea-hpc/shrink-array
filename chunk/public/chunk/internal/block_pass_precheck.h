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

/**
 * @file block_pass_precheck.h
 * @brief make_block_compression's early-out BEFORE it claims its candidates.
 *
 * A block pass marks every candidate of the rank's list (ChunkBase::steal_mark),
 * then opens the window with compression_mgr.start(), whose first check is
 * "is any thread flushing its deferred writes?". When it is, start() declines
 * and the pass unmarks every candidate -- and the rank's next write, its list
 * still above the threshold, does it all again. On Grace at 128 threads (two
 * sockets) that was 10.4 million declined starts in one shrink_bench run, 99.99%
 * of them on that check, each costing two stores per candidate into Chunk
 * objects the other socket's writers read: 96.8% of 6 billion remote-socket
 * accesses, and a 3x slower run than at 64 threads.
 *
 * block_pass_yields_to_flush() asks start()'s first question first: one seq_cst
 * load of the flush counter. start() still asks it too (and again after its
 * writer wait), so a flush beginning after this check is caught as before.
 */

#include "chunk/config.h"                              // build_config::use_openmp
#include "chunk/internal/chunk_stats.h"                // increm_chunk_gate_start*
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED
#include "chunk/internal/write_compress_sync_policy.h" // any_thread_flushing

namespace shrink_array
{

/**
 * @brief Would compression_mgr.start() decline right now because a flush is in
 *        progress? If so, count it as start() would, and tell the caller to
 *        return before marking anything.
 * @tparam ProxyType the proxy whose WriteCompressSyncPolicy start() consults
 * @tparam ChunkType the chunk type the gate counters are kept for
 * @param max_thr rank bound passed on to any_thread_flushing()
 */
template <typename ProxyType, typename ChunkType>
[[nodiscard]] inline bool block_pass_yields_to_flush([[maybe_unused]] int max_thr) noexcept
{
    if constexpr (!build_config::use_openmp)
    {
        return false;
    }
    else
    {
        if (!WriteCompressSyncPolicy<ProxyType>::any_thread_flushing(max_thr))
        {
            return false;
        }
        // What a declined start() counts: an attempt, yielded to a flush.
        increm_chunk_gate_start<ChunkType>();
        increm_chunk_gate_start_flush_yield<ChunkType>();
        return true;
    }
}

/**
 * @brief Check builds: trip if a flush is in progress while a compressor runs
 *        OUTSIDE CompressionManager::start().
 * @details The v1 replay bodies (proxy_inline.h, `*_async`) store through a
 *          values pointer loaded once, with no retry: they rely on no
 *          compressor changing `values` during a flush drain. start() refuses
 *          while a flush is announced; the compressors that bypass it
 *          (Chunk::compress_sweep_rank, RechunkHooks::rechunk_drop_candidates)
 *          are safe only because they run where no drain can: outside the
 *          writers' parallel region, after the flush. This turns a caller that
 *          breaks that into an CHUNK_INVARIANT_FAILED instead of a write lost in
 *          silence. A tripwire, not a lock: a flush announced after the check
 *          is not seen.
 * @tparam ProxyType the proxy whose WriteCompressSyncPolicy the drains announce on
 * @param where caller, for the message
 */
template <typename ProxyType>
inline void check_no_flush_outside_start([[maybe_unused]] const char* where) noexcept
{
    if constexpr (build_config::use_openmp && build_config::any_check)
    {
        if (WriteCompressSyncPolicy<ProxyType>::any_thread_flushing(0)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: compresses outside CompressionManager::start() while a deferred-op flush is in "
                                   "progress; a v1 replay storing into this chunk would be lost (proxy_inline.h)",
                                   where);
        }
    }
}

} // namespace shrink_array
