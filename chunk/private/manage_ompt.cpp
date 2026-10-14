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
 * @file manage_ompt.cpp
 * @brief OMPT tool wiring the chunk library into the OpenMP runtime.
 *
 * Compiled as a real tool only under `CHUNK_OPT_OMPT`.
 * The runtime discovers it through ompt_start_tool(), which registers three
 * callbacks; each of them maps a point of the OpenMP lifecycle onto one of the
 * phase transitions declared in chunk/manage_ompt.h, so an application built
 * this way needs no explicit call at all.
 *
 * In any other build this file only defines `g_ompt_suspended{1}` (callbacks
 * permanently suspended: none exists); the application drives the phase
 * transitions itself (see manage_ompt.h).
 *
 * @see chunk/manage_ompt.h for the semantics of the phase transitions.
 */

#include <array>     // std::array
#include <algorithm> // std::min (CHUNK_OMPT_STATS report)
#include <atomic>    // std::atomic, std::memory_order
#include <chrono>    // CHUNK_OMPT_STATS: time spent flushing
#include <cstdint>   // std::uint64_t
#include <cstdio>    // CHUNK_OMPT_STATS report
#include <cstdlib>   // std::getenv, std::abort (CHUNK_REQUIRE_OMPT)
#include <cstring>   // std::strlen (the refused callbacks' list)
#include <format>    // std::format_to_n (the refused callbacks' list)

#include "chunk/config.h"      // g_ompt_suspended (defined here), chunk_config::is_shrink_enabled
#include "chunk/singleton.h"   // SingletonChk
#include "chunk/manage_ompt.h" // begin_parallel_phase, finalize_parallel_phase, finalize_barrier_phase
#include "chunk/PMP_CHUNK.h"   // g_fast_omp_rank

#include "chunk/all_type_chunk.h"          // AllTypeChunk
#include "chunk/internal/churn_counters.h" // churn::note_phase_end

#if CHUNK_OPT_OMPT
    #include <omp-tools.h> // ompt_set_callback, ompt_start_tool_result_t
#endif

// Stays a directive: it gates the WHOLE translation unit -- the global switch
// below and every callback -- so there is no statement for an `if constexpr` to
// sit in. A build without OMPT compiles only the stand-ins of the #else branch.
#if CHUNK_OPT_OMPT

    #include <omp.h> // omp_in_parallel: the allocator flush runs outside any active region
    // Directive, like all_type_chunk.cpp's: a std-alloc build does not put
    // fast_allocators/public on the include path at all.
    #if !CHUNK_OPT_STD_ALLOC
        #include "fast_allocators/internal/static_const_array.h" // StaticConstArray::clearInfoFreeElt
    #endif

/**
 * @brief Suspension counter of the OMPT callbacks, 0 (active) at start in
 *        this build.
 *
 * Every callback returns immediately while it is non-zero, which lets a
 * program suspend the embedded compression hooks at runtime (e.g. around a
 * region that must not trigger a flush) without unregistering the tool. A
 * counter, not a flag: see its declaration in chunk/config.h.
 */
shrink_array::OmptSuspendCounter shrink_array::g_ompt_suspended{0}; // 0 = callbacks active

namespace
{
/// @brief What ompt_initialize() managed to install, for check_ompt_tool_loaded().
/// @details 0: ompt_initialize never ran (the runtime did not load the tool);
///          1: it ran and every callback was accepted; 2: it ran but
///          ompt_set_callback was missing or refused a callback.
std::atomic<int> g_ompt_tool_state{0};
/// @brief The callbacks the runtime refused (state 2), for the warning.
/// @details A constant-initialised buffer, not a std::string: ompt_initialize()
///          runs when the OpenMP runtime starts, which may be before this
///          file's dynamic initialisation.
char g_ompt_refused[128] = "";
} // namespace

// ── CHUNK_OMPT_STATS: what the sync-region callback does, counted ───────────
//
// Measurement only. Off unless CHUNK_OMPT_STATS is set (read once, in
// ompt_initialize, before any team exists): one load of a plain bool per
// callback. On: each thread owns one 128-byte-aligned slot, claimed once, and
// is its only writer -- no shared cache line, no read-modify-write, and no
// OpenMP API call from inside a callback. Read at ompt_finalize, after the
// runtime has shut its threads down.
namespace
{
constexpr int kOmptKinds    = 11; // ompt_sync_region_t values 1..10
constexpr int kOmptMaxSlots = 1024;

bool g_ompt_stats = false;

/// Per [kind][endpoint: 0 begin, 1 end].
using OmptCounters = std::array<std::array<std::atomic<std::uint64_t>, 2>, kOmptKinds>;

struct alignas(128) OmptSyncSlot
{
    OmptCounters calls;
    OmptCounters flush_calls;
    OmptCounters skipped_flag;
    OmptCounters flushed;
    OmptCounters flush_ns;
};

std::array<OmptSyncSlot, kOmptMaxSlots> g_ompt_slots;
std::atomic<int>                        g_ompt_next_slot{0};
thread_local OmptSyncSlot*              t_ompt_slot = nullptr;

OmptSyncSlot* ompt_my_slot() noexcept
{
    if (t_ompt_slot == nullptr)
    {
        const int slot_idx = g_ompt_next_slot.fetch_add(1, std::memory_order_relaxed); // once per thread
        if (slot_idx >= kOmptMaxSlots)
        {
            return nullptr;
        }
        t_ompt_slot = &g_ompt_slots[slot_idx];
    }
    return t_ompt_slot;
}

/// Single-writer increment: the slot belongs to the calling thread.
void ompt_bump(std::atomic<std::uint64_t>& counter, std::uint64_t v = 1) noexcept
{
    counter.store(counter.load(std::memory_order_relaxed) + v, std::memory_order_relaxed);
}

const char* ompt_kind_name(int k) noexcept
{
    switch (k)
    {
        case 1:
            return "barrier";
        case 2:
            return "barrier_implicit";
        case 3:
            return "barrier_explicit";
        case 4:
            return "barrier_implementation";
        case 5:
            return "taskwait";
        case 6:
            return "taskgroup";
        case 7:
            return "reduction";
        case 8:
            return "barrier_implicit_workshare";
        case 9:
            return "barrier_implicit_parallel";
        case 10:
            return "barrier_teams";
        default:
            return "?";
    }
}

void ompt_finalize(ompt_data_t*)
{
    if (!g_ompt_stats)
    {
        return;
    }
    const int nslots = std::min(g_ompt_next_slot.load(std::memory_order_acquire), kOmptMaxSlots);
    std::fprintf(stderr, "\n[OMPT-SYNC] %d threads seen. Per sync-region kind and endpoint, summed over threads:\n",
                 nslots);
    std::fprintf(stderr, "[OMPT-SYNC] %-27s %-5s %12s %12s %12s %12s %10s %10s\n", "kind", "end", "calls",
                 "flush_tried", "skip_flag", "flushed", "flush_ms", "max_thr_ms");
    for (int k = 0; k < kOmptKinds; ++k)
    {
        for (int e = 0; e < 2; ++e)
        {
            std::uint64_t calls = 0, tried = 0, skip = 0, fl = 0, ns = 0, ns_max = 0;
            for (int s = 0; s < nslots; ++s)
            {
                const auto& sl = g_ompt_slots[s];
                calls += sl.calls[k][e].load(std::memory_order_relaxed);
                tried += sl.flush_calls[k][e].load(std::memory_order_relaxed);
                skip += sl.skipped_flag[k][e].load(std::memory_order_relaxed);
                fl += sl.flushed[k][e].load(std::memory_order_relaxed);
                const std::uint64_t flush_ns = sl.flush_ns[k][e].load(std::memory_order_relaxed);
                ns += flush_ns;
                ns_max = flush_ns > ns_max ? flush_ns : ns_max;
            }
            if (calls == 0)
            {
                continue;
            }
            std::fprintf(stderr, "[OMPT-SYNC] %-27s %-5s %12llu %12llu %12llu %12llu %10.1f %10.1f\n",
                         ompt_kind_name(k), e == 0 ? "begin" : "end", (unsigned long long)calls,
                         (unsigned long long)tried, (unsigned long long)skip, (unsigned long long)fl,
                         static_cast<double>(ns) * 1e-6, static_cast<double>(ns_max) * 1e-6);
        }
    }
    std::fflush(stderr);
}

/**
 * @brief `ompt_callback_implicit_task`: start/end of a thread's implicit task.
 *
 * Runs on EVERY thread of the team, and is the only callback that does. The
 * `ompt_scope_begin` endpoint seeds the thread-local rank cache via
 * shrink_array::begin_parallel_phase(). The `ompt_scope_end` endpoint, and a
 * begin while the callbacks are suspended, clear it (-1): the next
 * get_thr_rank_fast() recomputes it. Otherwise a thread coming back from a
 * nested (serialised) region would keep the inner rank 0, and a region run
 * while suspended the previous region's rank -- sharing another rank's
 * candidate list and deferred-op queue.
 *
 * @param endpoint Which side of the implicit task is being reported.
 */
void on_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int flags)
{
    // The INITIAL task is the program's own thread, not a parallel region: no
    // phase to begin, and the rank cache is recomputed on demand. Do not seed
    // it here: libomp reports it from inside the program's first OpenMP call,
    // which can be the omp_get_max_threads() of chunk_thread_capacity()'s
    // static initialiser; begin_parallel_phase() -> CHUNK_PMP_THREAD_RANK ->
    // chunk_thread_capacity() would re-enter that initialisation
    // (recursive_init_error at start-up).
    if ((flags & ompt_task_initial) != 0)
    {
        return;
    }
    if (endpoint != ompt_scope_begin || shrink_array::g_ompt_suspended.load(std::memory_order_relaxed) != 0)
    {
        shrink_array::g_fast_omp_rank = -1; // recomputed on the next get_thr_rank_fast()
        return;
    }
    // ompt_scope_begin is the start of the thread's work in the parallel region:
    // seed the thread-local rank cache (and arm the phase guard).
    shrink_array::begin_parallel_phase();
}

// No `ompt_callback_parallel_begin`: pre-reserving the candidate lists is
// useless, the passes empty them with clear(), which keeps their capacity.

/**
 * @brief The chunk library's part of `ompt_callback_parallel_end`.
 *
 * The library's main phase boundary: flushes the deferred writes through
 * shrink_array::finalize_parallel_phase(), then compresses the candidates
 * accumulated during the region. Both steps are suppressed during a
 * chunk-level rechunk rebuild (`g_in_rechunk_rebuild`); see the inline comments.
 */
void parallel_end_chunk_work()
{
    shrink_array::churn::note_phase_end(); // one compression phase (churn_counters.h)
    // Vector-mode build: chunk lib is inactive, nothing to finalize.
    if constexpr (!shrink_array::chunk_config::is_shrink_enabled)
    {
        return;
    }

    // During the chunk-level rechunk rebuild, skip the DEFERRED FLUSH: the
    // rebuild drives the deferred write path inside its own parallel region, and
    // flushing those half-applied ops here corrupts grow regions spanning several
    // old chunks (rechunk_to flushes them afterwards, at a quiescent point).
    const bool in_rebuild = shrink_array::g_in_rechunk_rebuild.load(std::memory_order_acquire);
    if (shrink_array::g_need_flush_deferred && !in_rebuild)
    {
        shrink_array::finalize_parallel_phase();
    }

    // During the rebuild, ALSO skip block compression. The rebuild writes tmp's
    // new chunks inside its own parallel region; compressing them here (before
    // rechunk_to's quiescent flush/drop point) re-materialises an
    // all-second_value chunk spanning the old-chunk boundary at the OLD
    // geometry, and every element past old_size reads back first_value.
    // rechunk_to compresses the new chunks itself, at the new geometry. (Buffer
    // reclaim during the rebuild: rechunk_drain_reuse_list, a quiescent path.)
    if (in_rebuild)
    {
        return;
    }

    const auto& ens_type_chunk = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef();

    if (!ens_type_chunk.have_chunk_to_compress_potentially())
    {
        return;
    }

    // No `omp barrier` here: at an outermost parallel_end the team has
    // already joined (the region's implicit barrier made its writes visible),
    // so one would bind to a team of one and do nothing; at a nested region's
    // end it would bind to the ENCLOSING team, which only this thread reaches.
    ens_type_chunk.compress_chunk_to_compress_potentially(true);
}

/**
 * @brief Hand the fast_allocators per-thread free batches back to the ladder.
 *
 * @details Inside a region StaticConstArray::freeBlock / freeBlockAlign park
 *          freed blocks in a per-thread batch that is flushed only when it
 *          overflows (4 MiB or 10000 blocks, per ladder): without this, up to
 *          that much per thread stays out of every free list after the region.
 *
 *          Outside any ACTIVE region only (the outermost parallel_end): the
 *          flush takes no lock, so no rank may be freeing meanwhile. A nested
 *          region's end leaves its batches to the enclosing one's.
 */
void flush_allocator_free_batches()
{
    #if !CHUNK_OPT_STD_ALLOC
    if (omp_in_parallel() == 0)
    {
        fast_alloc::StaticConstArray::clearInfoFreeElt();
    }
    #endif
}

/**
 * @brief `ompt_callback_parallel_end`: the team of a parallel region has joined.
 *
 * Runs the chunk library's phase boundary (parallel_end_chunk_work), then
 * flushes the allocator's free batches -- AFTER it: the compression pass opens
 * its own region with these callbacks suspended, and its frees are batched
 * too.
 */
void on_parallel_end(ompt_data_t*, const ompt_frame_t*, ompt_data_t*, uint32_t, const void*)
{
    if (shrink_array::g_ompt_suspended.load(std::memory_order_relaxed) != 0)
    {
        return;
    }

    parallel_end_chunk_work();

    flush_allocator_free_batches();
}

/**
 * @brief `ompt_callback_sync_region`: a synchronisation region (barrier) event.
 *
 * Makes the values written before the barrier visible after it, by flushing the
 * deferred write queues through shrink_array::finalize_barrier_phase().
 * Suppressed during a rechunk rebuild for the same reason as on_parallel_end().
 */
//
// The signature is the runtime's ompt_callback_sync_region_t. The runtime
// calls this at BOTH the begin and the end of EVERY sync region, whatever its
// kind (barriers, taskwait, taskgroup, reductions). CHUNK_OMPT_STATS counts them.
void on_barrier_end(ompt_sync_region_t    kind,
                    ompt_scope_endpoint_t endpoint,
                    ompt_data_t*          parallel_data,
                    ompt_data_t*,
                    const void*)
{
    if (shrink_array::g_ompt_suspended.load(std::memory_order_relaxed) != 0)
    {
        return;
    }

    OmptSyncSlot* slot = nullptr;
    int           k    = 0;
    int           ep   = 0;
    if (g_ompt_stats) [[unlikely]]
    {
        slot = ompt_my_slot();
        k    = (static_cast<int>(kind) > 0 && static_cast<int>(kind) < kOmptKinds) ? static_cast<int>(kind) : 0;
        ep   = endpoint == ompt_scope_begin ? 0 : 1;
        if (slot)
        {
            ompt_bump(slot->calls[k][ep]);
        }
    }

    // The end of the implicit barrier that closes a parallel region: a worker
    // has LEFT the team by then (libomp passes parallel_data null; older
    // runtimes spell the kind ompt_sync_region_barrier_implicit), and the rank
    // lookup of the flush below (omp_get_level()) would read a team that is
    // gone. Skipping loses nothing: the master's on_parallel_end() runs
    // finalize_parallel_phase() right after, for EVERY rank.
    if (endpoint == ompt_scope_end && (kind == ompt_sync_region_barrier_implicit_parallel || parallel_data == nullptr))
    {
        return;
    }

    // Same suppression as on_parallel_end (see parallel_end_chunk_work): the
    // rebuild's own parallel region ends with an implicit barrier too.
    const bool in_rebuild = shrink_array::g_in_rechunk_rebuild.load(std::memory_order_acquire);
    if (shrink_array::g_need_flush_deferred && !in_rebuild)
    {
        if (slot) [[unlikely]]
        {
            const auto t0      = std::chrono::steady_clock::now();
            const bool flushed = shrink_array::finalize_barrier_phase();
            const auto dt      = std::chrono::steady_clock::now() - t0;
            ompt_bump(slot->flush_calls[k][ep]);
            if (flushed)
            {
                ompt_bump(slot->flushed[k][ep]);
            }
            ompt_bump(slot->flush_ns[k][ep],
                      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()));
        }
        else
        {
            shrink_array::finalize_barrier_phase();
        }
    }
    else if (slot && !in_rebuild) [[unlikely]]
    {
        // Skipped because ANOTHER thread's flush_all_deferred() had cleared
        // the flag for the duration of its own pass.
        ompt_bump(slot->skipped_flag[k][ep]);
    }
}


/**
 * @brief OMPT tool initialiser, called once by the runtime at startup.
 *
 * Looks up `ompt_set_callback` and registers the three callbacks above. If the
 * lookup fails the tool stays inert (an error is printed): no phase transition
 * happens at all, which shows up as retained memory rather than as a crash.
 *
 * @param lookup Runtime-provided entry-point lookup function.
 * @return 1 to keep the tool active for the whole program run.
 */
int ompt_initialize(ompt_function_lookup_t lookup, int, ompt_data_t*)
{
    {
        const char* env = std::getenv("CHUNK_OMPT_STATS");
        g_ompt_stats    = env != nullptr && env[0] != '\0' && env[0] != '0'; // before any team: no race
    }

    auto ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
    if (ompt_set_callback)
    {
        // ompt_set_error / _never / _impossible: the callback will not be
        // dispatched (check_ompt_tool_loaded() reports it).
        bool       all_set = true;
        const auto set     = [&](ompt_callbacks_t which, ompt_callback_t cb, const char* name)
        {
            if (ompt_set_callback(which, cb) < ompt_set_sometimes)
            {
                all_set                = false;
                const std::size_t used = std::strlen(g_ompt_refused);
                *std::format_to_n(g_ompt_refused + used, sizeof g_ompt_refused - used - 1, " {}", name).out = '\0';
            }
        };
        set(ompt_callback_parallel_end, (ompt_callback_t)on_parallel_end, "parallel_end");
        set(ompt_callback_sync_region, (ompt_callback_t)on_barrier_end, "sync_region");

        set(ompt_callback_implicit_task, (ompt_callback_t)on_implicit_task, "implicit_task");
        g_ompt_tool_state.store(all_set ? 1 : 2, std::memory_order_release);
    }
    else
    {
        *std::format_to_n(g_ompt_refused, sizeof g_ompt_refused - 1, " (ompt_set_callback is null)").out = '\0';
        g_ompt_tool_state.store(2, std::memory_order_release);
        std::fprintf(stderr, "[OMPT] ompt_set_callback is null: the chunk library's callbacks are not installed\n");
    }
    return 1;
}

} // namespace

/**
 * @brief OMPT entry point looked up by the OpenMP runtime at program start.
 *
 * Advertises ompt_initialize() as the tool initialiser and ompt_finalize() as
 * its finaliser (the CHUNK_OMPT_STATS report; the tool allocates nothing).
 *
 * @return Pointer to a function-local static result, valid for the whole run.
 */
// extern "C": the runtime finds the tool with dlsym("ompt_start_tool"). Some
// omp-tools.h (LLVM 18) do not declare it, and a C++-mangled definition is
// never found: the tool stays silently inert.
extern "C" ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*)
{
    static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize, &ompt_finalize, {0}};
    return &ompt_start_tool_result;
}

/**
 * @brief See manage_ompt.h. Once per process (a magic static).
 */
void shrink_array::check_ompt_tool_loaded()
{
    [[maybe_unused]] static const bool checked = []
    {
        // The runtime loads its tool while it initialises: make sure it has.
        (void)omp_get_max_threads();
        const int state = g_ompt_tool_state.load(std::memory_order_acquire);
        if (state == 1)
        {
            return true;
        }

        const char* req   = std::getenv("CHUNK_REQUIRE_OMPT");
        const bool  fatal = req && req[0] && req[0] != '0';
        if (state == 0)
        {
            std::fprintf(stderr,
                         "[OMPT] %s: this build (OMPT tool) runs its phase boundaries -- deferred-write flush, "
                         "compression between parallel regions -- from an OMPT tool, and the OpenMP runtime did "
                         "not load it. Usual causes: OMP_TOOL=disabled; another tool found first through "
                         "OMP_TOOL_LIBRARIES; ompt_start_tool not found in the executable. "
                         "OMP_TOOL_VERBOSE_INIT=stderr shows the runtime's search.%s\n",
                         fatal ? "FATAL" : "WARNING", fatal ? "" : " CHUNK_REQUIRE_OMPT=1 makes this fatal.");
        }
        else
        {
            std::fprintf(stderr,
                         "[OMPT] %s: the chunk library's OMPT tool is loaded but the runtime refused:%s -- the "
                         "phase boundaries they run (deferred-write flush, compression between parallel "
                         "regions) will not all happen.%s\n",
                         fatal ? "FATAL" : "WARNING", g_ompt_refused,
                         fatal ? "" : " CHUNK_REQUIRE_OMPT=1 makes this fatal.");
        }
        std::fflush(stderr);
        if (fatal)
        {
            std::abort();
        }
        return false;
    }();
}

#else

/**
 * @brief Suspension counter of the OMPT callbacks, held at 1 in a build
 *        without OMPT (they run only while it is zero, and none exists here).
 *
 * The symbol is still defined so that code testing it links in every variant;
 * here no callback exists and the application drives the phase transitions.
 */
shrink_array::OmptSuspendCounter shrink_array::g_ompt_suspended{1}; // no callback exists in this build

/// @brief No OMPT tool in this build: nothing to check (see manage_ompt.h).
void shrink_array::check_ompt_tool_loaded() {}

#endif
