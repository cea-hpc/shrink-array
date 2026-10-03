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
 * @file test_hooks.h
 * @brief TEST HOOKS: widen the windows the validation harnesses are about.
 *
 * @details Three hooks, each a run-time value that is 0 -- off -- until a
 *          harness sets it:
 *          - force_defer_every: one write in N of every thread takes the
 *            deferred path as if a compressor held the chunk
 *            (deferred_path_validation);
 *          - read_delay_every: one pure read in N of an opted-in thread waits
 *            between its `values` load and its element load
 *            (read_reclaim_validation, pure-read and `compound` modes);
 *          - pass_delay_every: one candidate in N of an opted-in thread's
 *            compression pass waits at the point where the pass hands it back
 *            (read_reclaim_validation, mode `remark`).
 *
 *          They are COMPILED in only where the harnesses are built
 *          (CHUNK_TEST_HOOKS, defined for the whole tree by the top-level
 *          CMakeLists in Debug, CHECK_OPTIM and sanitizer builds), so the
 *          harnesses share one executable -- one definition of the
 *          header-only engine -- with the others. Elsewhere they compile to
 *          nothing. The former per-target macros still work for a whole-tree
 *          build: CHUNK_FORCE_DEFER_EVERY (config.h, build_config::
 *          force_defer_every), CHUNK_TEST_READ_DELAY_EVERY / _US and
 *          CHUNK_TEST_PASS_DELAY_EVERY / _US give the initial values, and a
 *          non-zero one compiles its hook in.
 */

#include <atomic> // std::atomic, std::memory_order_relaxed
#include <chrono> // std::chrono::steady_clock, std::chrono::microseconds

#include "chunk/config.h"             // build_config::force_defer_every, build_config::use_openmp
#include "chunk/internal/cpu_pause.h" // CPU_PAUSE

#ifndef CHUNK_TEST_READ_DELAY_EVERY
    #define CHUNK_TEST_READ_DELAY_EVERY 0
#endif
#ifndef CHUNK_TEST_READ_DELAY_US
    #define CHUNK_TEST_READ_DELAY_US 50
#endif
#ifndef CHUNK_TEST_PASS_DELAY_EVERY
    #define CHUNK_TEST_PASS_DELAY_EVERY 0
#endif
#ifndef CHUNK_TEST_PASS_DELAY_US
    #define CHUNK_TEST_PASS_DELAY_US 200
#endif

namespace shrink_array::test_hooks
{

/// @brief The hooks are compiled in: CHUNK_TEST_HOOKS (the harnesses' builds).
#ifdef CHUNK_TEST_HOOKS
inline constexpr bool compiled = true;
#else
inline constexpr bool compiled = false;
#endif

// ── Write side: forced deferrals ─────────────────────────────────────────────

/// @brief The forced-deferral hook is compiled in: OpenMP builds only, the
///        deferred path does not exist without them.
inline constexpr bool force_defer_compiled =
    build_config::use_openmp && (compiled || build_config::force_defer_every > 0);

/// @brief One write in N of every thread takes the deferred path (0 = off).
///        Set by the harness before its regions; read on the write path.
inline unsigned force_defer_every = build_config::force_defer_every;

/// @brief Writes the hook has sent down the deferred path so far: lets a
///        harness check that the hook really fired.
inline std::atomic<unsigned long> forced_deferrals{0};

// ── Read side: pure reads ────────────────────────────────────────────────────

/// @brief The read-delay hook is compiled in.
inline constexpr bool read_delay_compiled = compiled || CHUNK_TEST_READ_DELAY_EVERY > 0;

/// @brief One pure read in N of an opted-in thread waits (0 = off).
inline unsigned read_delay_every = CHUNK_TEST_READ_DELAY_EVERY;

/// @brief Opt-in, per thread: only the harness's reader threads set it.
inline thread_local bool read_delay_enabled = false;

/// @brief The wait, in microseconds.
inline unsigned read_delay_us = CHUNK_TEST_READ_DELAY_US;

// ── Pass side: hand-back of the candidates ───────────────────────────────────

/// @brief The pass-delay hook is compiled in.
inline constexpr bool pass_delay_compiled = compiled || CHUNK_TEST_PASS_DELAY_EVERY > 0;

/// @brief One candidate in N of an opted-in thread's pass waits (0 = off).
inline unsigned pass_delay_every = CHUNK_TEST_PASS_DELAY_EVERY;

/// @brief Opt-in, per thread: only the harness's slow-pass thread sets it.
inline thread_local bool pass_delay_enabled = false;

/// @brief The wait, in microseconds.
inline unsigned pass_delay_us = CHUNK_TEST_PASS_DELAY_US;

/// @brief Spin @p us microseconds.
inline void spin_for_us(unsigned us) noexcept
{
    const auto t0   = std::chrono::steady_clock::now();
    const auto wait = std::chrono::microseconds(us);
    while (std::chrono::steady_clock::now() - t0 < wait)
    {
        CPU_PAUSE();
    }
}

/// @brief One call in read_delay_every on an opted-in thread spins
///        read_delay_us. Nothing when the hook is not compiled in.
inline void maybe_read_delay() noexcept
{
    if constexpr (read_delay_compiled)
    {
        if (!read_delay_enabled || read_delay_every == 0)
        {
            return;
        }
        thread_local unsigned count = 0;
        if (++count < read_delay_every)
        {
            return;
        }
        count = 0;
        spin_for_us(read_delay_us);
    }
}

/// @brief One call in pass_delay_every on an opted-in thread spins
///        pass_delay_us, at the point where a pass hands a candidate back
///        (between releasing its mark and its dedup flag). Nothing when the
///        hook is not compiled in.
inline void maybe_pass_delay() noexcept
{
    if constexpr (pass_delay_compiled)
    {
        if (!pass_delay_enabled || pass_delay_every == 0)
        {
            return;
        }
        thread_local unsigned count = 0;
        if (++count < pass_delay_every)
        {
            return;
        }
        count = 0;
        spin_for_us(pass_delay_us);
    }
}

/// @brief The forced-deferral decision for one write: 0 = take the normal
///        path, 1 = defer at check #1, 2 = defer at check #2 (alternating).
///        Always 0 when the hook is not compiled in or force_defer_every is 0.
inline int forced_deferral() noexcept
{
    if constexpr (force_defer_compiled)
    {
        const unsigned every = force_defer_every;
        if (every == 0)
        {
            return 0;
        }
        thread_local unsigned count = 0;
        thread_local bool     odd   = false;
        if (++count >= every) [[unlikely]]
        {
            count = 0;
            odd   = !odd;
            forced_deferrals.fetch_add(1, std::memory_order_relaxed);
            return odd ? 2 : 1;
        }
    }
    return 0;
}

} // namespace shrink_array::test_hooks
