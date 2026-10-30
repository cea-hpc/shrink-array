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
 * @file phase_guard.h
 * @brief Guard of the parallel-phase protocol (manage_ompt.h).
 *
 * Without OMPT the library does not see region boundaries. A write can be
 * DEFERRED to the writing thread's queue (once the chunk's candidate list is
 * flagged dangerous), and it is the application that drains those queues
 * between regions: begin_parallel_phase() by every thread inside the region,
 * finalize_parallel_phase() (or the only_flush / after_flush pair) outside
 * it. Skip the brackets and a deferred write of one region lands AFTER a
 * direct write of the next one: values silently wrong, only on the deferred
 * chunks, only at thread counts that push a list past the threshold.
 *
 * Measured (8 threads, forced deferrals, team size changing from region to
 * region): finalize forgotten between regions replays a freed deferred op
 * (DEFERRED_LIFECYCLE: op magic poisoned); begin and finalize both forgotten
 * leave wrong values with no message in a release build. With a fixed team
 * both went unnoticed -- the trap only springs when the team changes.
 *
 * The guard is a generation counter: every phase end bumps it, every
 * begin_parallel_phase() stamps the calling thread with the current value.
 * Two checks:
 *  - finalize forgotten: a begin_parallel_phase() by a thread whose stamp is
 *    ALREADY current -- it began in this phase and no phase end came since
 *    (check_begin()). Deterministic: it does not wait for a write to be
 *    deferred, which a normal run almost never does. Only at the outermost
 *    level: a region nested in it runs on the same thread, in the same
 *    phase, and may call begin again.
 *  - begin forgotten: a write about to be DEFERRED inside a parallel region
 *    by a thread whose stamp is not current (check_write(), at the enqueue).
 *    Not on every write: a write needs no begin as such (the rank cache
 *    fills itself), and an array created and used inside an `omp single`
 *    without the brackets is legitimate (region_storage_validation). What
 *    needs the brackets is a write left queued in a rank's queue.
 *
 * Only where the protocol is manual (build_config::has_not_ompt, OpenMP):
 * under OMPT the runtime callbacks make the calls themselves (`tracked`).
 * The finalize check runs in EVERY such build, release included: it costs
 * one level query and one comparison per thread and per region, never per
 * write, and what it stops is a freed op replayed or values silently wrong.
 * The begin check is a check-build one (`active`, build_config::any_check):
 * alone, a forgotten begin leaves the values right. shrink_array::
 * parallel_phase() (manage_ompt.h) makes both calls for the program.
 *
 * One top-level team at a time: the generation is global, so with two
 * teams started by different host threads one team's phase end makes the
 * other's threads stale (check_write fails) and can hide its forgotten
 * finalize. Such a program is outside the protocol anyway (manage_ompt.h).
 */

#include <atomic>  // std::atomic, std::memory_order_relaxed
#include <cstddef> // std::size_t (teams::rank_holders)

#include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_*, chunk_pmp_team_rank
#include "chunk/config.h"                   // build_config::any_check, build_config::has_not_ompt
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array::phase_guard
{
/// @brief Phase generation: bumped by every phase end, read by begin and by the check.
inline std::atomic<unsigned> generation{1};

/// @brief The generation this thread's last begin_parallel_phase() saw.
inline thread_local unsigned armed_for = 0;

/// @brief begin_parallel_phase(): stamp the calling thread with the current generation.
inline void arm() noexcept
{
    armed_for = generation.load(std::memory_order_relaxed);
}

/// @brief Phase end: open a new generation, so a thread that does not re-arm shows as stale.
inline void next_phase() noexcept
{
    generation.fetch_add(1, std::memory_order_relaxed);
}

/// @brief Is the calling thread armed for the current phase?
[[nodiscard]] inline bool armed() noexcept
{
    return armed_for == generation.load(std::memory_order_relaxed);
}

/// @brief The protocol is manual in this build (OpenMP without OMPT): the
///        generations are kept and the finalize check runs, in every build type.
inline constexpr bool tracked = build_config::threaded && build_config::has_not_ompt;

/// @brief ... and it is a check build: the begin check (check_write) runs too.
inline constexpr bool active = tracked && build_config::any_check;

/// @brief begin_parallel_phase(): fail if the calling thread already began in
///        this phase (finalize forgotten), then stamp it.
/// @details Outside any region (level 0) it does nothing: there is no region
///          to bracket -- except during an application phase of the std
///          backend (pmp::in_application_phase()), where the application's
///          threads, in no team, are the region's threads: it stamps them,
///          and a second begin of the same thread in the phase is allowed
///          (same rank, same queue, its writes stay in program order; the
///          phase has no region whose end could have been forgotten). In a
///          nested region it only stamps (see the file); so does a team of
///          the std backend opened during an application phase, nested in it.
inline void check_begin() noexcept
{
    if constexpr (tracked)
    {
        const int  level  = CHUNK_PMP_LEVEL;
        const bool in_app = pmp::in_application_phase();
        if (level == 0 && !in_app)
        {
            return;
        }
        if (level == 1 && !in_app && armed()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "[PHASE PROTOCOL] begin_parallel_phase() twice in the same phase on thread {}: the region before "
                "this one did not end with finalize_parallel_phase() (or the only_flush / after_flush pair), so its "
                "deferred writes are still queued and replay after this region's -- wrong values, or a freed op "
                "replayed when the team size changes. Call finalize_parallel_phase() after every region that "
                "called begin_parallel_phase(), or let shrink_array::parallel_phase() do both.",
                chunk_pmp_team_rank());
        }
        arm();
    }
}

/// @brief A write about to be deferred: fail if it runs in a parallel region
///        on a thread that did not call begin_parallel_phase() in this phase.
/// @param chunk the chunk written, @param num the element (for the message).
inline void check_write(const void* chunk, long num) noexcept
{
    if constexpr (active)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION && !armed()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "[PHASE PROTOCOL] deferred write in a parallel region by thread {}, which did not call "
                "begin_parallel_phase() since the last finalize_parallel_phase(): without OMPT every thread of a "
                "region that writes calls begin_parallel_phase() before its first write, and "
                "finalize_parallel_phase() runs after the region, or this write replays after a later region's "
                "writes -- or let shrink_array::parallel_phase() do both. chunk={} num={}",
                chunk_pmp_team_rank(), chunk, num);
        }
    }
}
// ── One top-level team at a time (every OpenMP build) ────────────────────────
// The per-rank structures are indexed by the rank in the team, and the end of a
// phase drains every rank's queue: two top-level teams writing arrays at once --
// opened by two host threads -- share ranks 0..n-1, one team's phase end replays
// the other's writes in the middle of its region, and their writers clear each
// other's flags (wrong values in a release build). The second team is refused,
// in every build, once an array exists (before, nothing is shared):
//  - OMPT: the parallel_begin / parallel_end callbacks count the top-level
//    regions (outer_team_begin / outer_team_end, manage_ompt.cpp);
//  - otherwise: every begin_parallel_phase() at level 1 takes its rank for
//    the phase (manual_begin); a rank another thread holds in the open phase
//    is a second team. The phase end frees them all (manual_end).
// A nested region is not a top-level one: it is allowed (nested_region_validation).
namespace teams
{
/// @brief An array exists: set by the first manager's registration (all_type_chunk.cpp).
inline std::atomic<bool> arrays_exist{false};

/// @brief Top-level regions running now (OMPT builds).
inline std::atomic<int> outer_active{0};

/// @brief Per rank: the thread that holds it in the open phase (manual protocol),
///        or null. Sized to the capacity on first use, immortal.
[[nodiscard]] inline std::atomic<const void*>* rank_holders() noexcept
{
    static auto* const holders = new std::atomic<const void*>[static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY)]();
    return holders;
}

/// @brief An address unique to the calling thread, for as long as it lives.
[[nodiscard]] inline const void* this_thread_id() noexcept
{
    thread_local const char marker = 0;
    return &marker;
}

/// @brief A top-level region begins (OMPT parallel_begin): refuse a second one.
inline void outer_team_begin() noexcept
{
    const int before = outer_active.fetch_add(1, std::memory_order_acq_rel);
    if (before > 0 && arrays_exist.load(std::memory_order_acquire)) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED(
            "[PHASE PROTOCOL] {} top-level OpenMP teams at once: a parallel region began while another one, opened "
            "by another host thread, is still running. The arrays' per-thread structures are indexed by the rank "
            "in the team, so the two teams share them, and one team's phase end replays the other's writes. Run "
            "the parts of the program that use the arrays one after the other, or in one team "
            "(docs/getting_started.md, section 5).",
            before + 1);
    }
}

/// @brief A top-level region ended (OMPT parallel_end).
inline void outer_team_end() noexcept
{
    outer_active.fetch_sub(1, std::memory_order_acq_rel);
}

/// @brief begin_parallel_phase() at level 1 (manual protocol): the calling
///        thread takes its rank for the phase; a rank another live thread
///        already holds in it means two teams at once. Every thread checks,
///        not only the master: the workers of a second team write as soon as
///        they begin.
inline void manual_begin(int rank) noexcept
{
    auto&       holder = rank_holders()[rank];
    const void* me     = this_thread_id();
    const void* cur    = holder.load(std::memory_order_acquire);
    if (cur == me || (cur == nullptr && holder.compare_exchange_strong(cur, me, std::memory_order_acq_rel)))
    {
        return;
    }
    if (cur != me && arrays_exist.load(std::memory_order_acquire)) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED(
            "[PHASE PROTOCOL] two top-level OpenMP teams at once: rank {} began a phase on a thread while another "
            "thread holds that rank in the phase still open -- two teams opened by two host threads run together, "
            "or finalize_parallel_phase() was forgotten after the other's region. The arrays' per-thread "
            "structures are indexed by the rank in the team, so two teams share them. Run the parts of the "
            "program that use the arrays one after the other, or in one team (docs/getting_started.md, "
            "section 5).",
            rank);
    }
    holder.store(me, std::memory_order_release); // no array yet: nothing shared, take it over
}

/// @brief The phase ended (manual protocol): every rank is free for the next one.
inline void manual_end() noexcept
{
    // Every rank handed out so far (serial builds: 1), a seq_cst load of the registry's high-water mark.
    const int bound   = CHUNK_PMP_RANK_BOUND;
    auto*     holders = rank_holders();
    for (int r = 0; r < bound; ++r)
    {
        holders[r].store(nullptr, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
}
} // namespace teams
} // namespace shrink_array::phase_guard
