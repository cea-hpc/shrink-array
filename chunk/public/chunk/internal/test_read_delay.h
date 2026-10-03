#pragma once
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

/**
 * @file test_read_delay.h
 * @brief TEST KNOB: widen the window between a pure read's pointer load and
 *        its dereference.
 *
 * @details A pure read (ChunkBase::element_at) loads the chunk's `values`
 *          pointer once, then reads `vals[i]`. Nothing ties that pair to the
 *          compressor: a pass running on another thread inside the same region
 *          may CAS `values` to the constant and free the old buffer in between
 *          (clear_int_st -> free_block_to_pool). Under CHUNK_STD_ALLOC the
 *          buffer goes back to malloc at once (use-after-free); with the fast
 *          allocator it is parked at the top of the rank's LIFO reuse list and
 *          the next materialisation takes it and overwrites it (a stale value).
 *          The window is two instructions wide; this knob makes it observable.
 *
 *          CHUNK_TEST_READ_DELAY_EVERY = N > 0: on a thread that has opted in
 *          (read_delay_enabled = true), one pure read in N waits
 *          CHUNK_TEST_READ_DELAY_US microseconds between the load and the
 *          dereference. 0 (the default) compiles everything out, and a thread
 *          that has not opted in -- every library-internal reader, the
 *          compressor's scans included -- is never delayed.
 *
 *          Scoped to a harness target, like CHUNK_FORCE_DEFER_EVERY: the read
 *          path is header-only and instantiated in the executable.
 *          @see examples/validation/read_reclaim_validation.cpp
 */

#include <chrono>

#include "chunk/internal/cpu_pause.h"

#ifndef CHUNK_TEST_READ_DELAY_EVERY
    #define CHUNK_TEST_READ_DELAY_EVERY 0
#endif
#ifndef CHUNK_TEST_READ_DELAY_US
    #define CHUNK_TEST_READ_DELAY_US 50
#endif

namespace shrink_array::test_hooks
{

inline constexpr unsigned read_delay_every = CHUNK_TEST_READ_DELAY_EVERY;

/// @brief Opt-in, per thread: only the harness's reader threads set it.
inline thread_local bool read_delay_enabled = false;

/// @brief The wait, in microseconds (CHUNK_TEST_READ_DELAY_US by default; a
///        harness may change it before its readers start).
inline unsigned read_delay_us = CHUNK_TEST_READ_DELAY_US;

/// @brief One call in read_delay_every on an opted-in thread spins
///        CHUNK_TEST_READ_DELAY_US microseconds. Compiled out at 0.
inline void maybe_read_delay() noexcept
{
    if constexpr (read_delay_every > 0)
    {
        if (!read_delay_enabled)
        {
            return;
        }
        thread_local unsigned count = 0;
        if (++count < read_delay_every)
        {
            return;
        }
        count           = 0;
        const auto t0   = std::chrono::steady_clock::now();
        const auto wait = std::chrono::microseconds(read_delay_us);
        while (std::chrono::steady_clock::now() - t0 < wait)
        {
            CPU_PAUSE();
        }
    }
}

// ── Pass side ────────────────────────────────────────────────────────────────
// CHUNK_TEST_PASS_DELAY_EVERY = N > 0: on a thread that has opted in
// (pass_delay_enabled = true), one candidate in N of a compression pass's
// closing loop waits CHUNK_TEST_PASS_DELAY_US microseconds at the point where
// the pass hands the candidate back -- between releasing its mark and
// releasing the dedup flag that lets another thread queue it again. That is
// the window in which another pass could claim the chunk; a pass that touched
// the mark after the flag would erase that claim. 0 compiles it out.
#ifndef CHUNK_TEST_PASS_DELAY_EVERY
    #define CHUNK_TEST_PASS_DELAY_EVERY 0
#endif
#ifndef CHUNK_TEST_PASS_DELAY_US
    #define CHUNK_TEST_PASS_DELAY_US 200
#endif

inline constexpr unsigned pass_delay_every = CHUNK_TEST_PASS_DELAY_EVERY;

/// @brief Opt-in, per thread: only the harness's slow-pass thread sets it.
inline thread_local bool pass_delay_enabled = false;

/// @brief The wait, in microseconds (CHUNK_TEST_PASS_DELAY_US by default).
inline unsigned pass_delay_us = CHUNK_TEST_PASS_DELAY_US;

/// @brief One call in pass_delay_every on an opted-in thread spins
///        pass_delay_us microseconds. Compiled out at 0.
inline void maybe_pass_delay() noexcept
{
    if constexpr (pass_delay_every > 0)
    {
        if (!pass_delay_enabled)
        {
            return;
        }
        thread_local unsigned count = 0;
        if (++count < pass_delay_every)
        {
            return;
        }
        count           = 0;
        const auto t0   = std::chrono::steady_clock::now();
        const auto wait = std::chrono::microseconds(pass_delay_us);
        while (std::chrono::steady_clock::now() - t0 < wait)
        {
            CPU_PAUSE();
        }
    }
}

} // namespace shrink_array::test_hooks
