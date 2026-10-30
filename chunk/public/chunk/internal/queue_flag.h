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
 * @file queue_flag.h
 * @brief The "queued for compression" flag of a chunk
 *        (tab_is_inside_chunk_list_to_compress): its protocol between the
 *        writers that queue the chunk and the compression pass that hands it
 *        back.
 *
 * @details Four values:
 *            - free (0): in no candidate list;
 *            - queued (1): in one list, once -- a writer that finds it gives up;
 *            - handed_back (2): the pass that holds the chunk's entry has
 *              scanned it and is handing it back (its mark goes down before the
 *              flag does);
 *            - written_back (3): a writer queued it during the hand-back.
 *
 *          The hand-back used to be "steal_unmark(), then the flag to 0". In
 *          between, a writer finds the mark down and writes; if its write makes
 *          the chunk constant, its CAS 0 -> 1 meets the flag still up and gives
 *          up, and the reset then drops the chunk from every list: a constant
 *          chunk left materialised (F6, docs/formal_models.md section 8.7,
 *          formal/queue_handback.c). The order itself is load-bearing -- once
 *          the flag is down another pass may queue and mark the chunk, and a
 *          late unmark would erase its claim -- so it is kept, and the window
 *          is closed by the flag instead: the pass sets it to handed_back while
 *          its marks still keep the writers out, a writer that finds
 *          handed_back turns it into written_back rather than giving up, and
 *          the pass's reset is a CAS handed_back -> free that, on
 *          written_back, queues the chunk again itself.
 *
 *          handed_back and written_back exist only between a pass's scans and
 *          the end of its hand-back loop; everything outside a pass sees free
 *          or queued, and "queued <=> in a list" holds there.
 *
 *          Serial builds have no concurrent pass: they keep plain 0 / 1 stores.
 */

#include <atomic> // std::atomic, std::memory_order

#include "chunk/config.h"                   // build_config::any_check
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array::queue_flag
{

inline constexpr char free         = 0; ///< in no candidate list
inline constexpr char queued       = 1; ///< in one list
inline constexpr char handed_back  = 2; ///< its pass is handing it back
inline constexpr char written_back = 3; ///< queued by a writer during the hand-back

/**
 * @brief Writer side: claim the flag for a push into the caller's list.
 * @return true if the caller must push the chunk (free -> queued); false if it
 *         is listed already, or if its pass is handing it back -- then the
 *         write is noted (handed_back -> written_back) and the pass queues it.
 * @note A relaxed probe first: the flag is almost always queued, and a CAS
 *       takes the line exclusively even when it fails.
 */
[[nodiscard]] inline bool try_queue(std::atomic<char>& flag) noexcept
{
    char v = flag.load(std::memory_order_relaxed);
    for (;;)
    {
        if (v == queued || v == written_back)
        {
            return false;
        }
        const char next = (v == free) ? queued : written_back;
        if (flag.compare_exchange_weak(v, next, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return next == queued;
        }
    }
}

/**
 * @brief Pass side: once the chunk is scanned, before its mark goes down.
 * @pre The chunk is in this pass's list (queued), and still marked: no writer
 *      touches the flag.
 */
inline void hand_back(std::atomic<char>& flag) noexcept
{
    const char was = flag.exchange(handed_back, std::memory_order_acq_rel);
    if constexpr (build_config::any_check)
    {
        if (was != queued) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("queue_flag::hand_back: flag {} on a listed candidate, expected {} (queued)",
                                   static_cast<int>(was), static_cast<int>(queued));
        }
    }
}

/**
 * @brief Pass side: the end of the hand-back, after the chunk's mark is down.
 * @return true if a writer queued the chunk meanwhile: the flag is queued again
 *         and the caller must push the chunk into its own list; false if the
 *         flag is free.
 */
[[nodiscard]] inline bool release(std::atomic<char>& flag) noexcept
{
    char v = handed_back;
    if (flag.compare_exchange_strong(v, free, std::memory_order_acq_rel, std::memory_order_acquire))
    {
        return false;
    }
    if constexpr (build_config::any_check)
    {
        if (v != written_back) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("queue_flag::release: flag {} at the end of a hand-back, expected {} or {}",
                                   static_cast<int>(v), static_cast<int>(handed_back), static_cast<int>(written_back));
        }
    }
    flag.store(queued, std::memory_order_release);
    return true;
}

} // namespace shrink_array::queue_flag
