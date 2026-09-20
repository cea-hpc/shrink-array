//
// Created by ballereaup on 21/04/2026.
//
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
 * Compiled as a real tool only under `HAVE_EMBEDDED_COMPRESSION_WITH_OMPT`.
 * The runtime discovers it through ompt_start_tool(), which registers four
 * callbacks; each of them maps a point of the OpenMP lifecycle onto one of the
 * phase transitions declared in chunk/manage_ompt.h, so an application built
 * this way needs no explicit call at all.
 *
 * In any other build this file only defines `__global_ompt = false`; the
 * application drives the phase transitions itself (see manage_ompt.h).
 *
 * @see chunk/manage_ompt.h for the semantics of the phase transitions.
 */

#include <atomic>

#include "chunk/config.h"
#include "chunk/singleton.h"
#include "chunk/internal/manager_chunk.h"
#include "chunk/manage_ompt.h"

#include "chunk/all_type_chunk.h"

#ifdef HAVE_OMPT
    #include <omp-tools.h>
#endif

// Stays a directive: it gates the WHOLE translation unit -- the global switch
// below and every callback -- so there is no statement for an `if constexpr` to
// sit in. A build without OMPT compiles this file to nothing.
#if defined HAVE_EMBEDDED_COMPRESSION_WITH_OMPT

/**
 * @brief Master switch of the OMPT callbacks, true in this build.
 *
 * Every callback returns immediately when it is false, which lets a program
 * disable the embedded compression hooks at runtime (e.g. around a region that
 * must not trigger a flush) without unregistering the tool.
 */
bool __global_ompt = true;

/**
 * @brief `ompt_callback_implicit_task`: start/end of a thread's implicit task.
 *
 * Runs on EVERY thread of the team, and is the only callback that does. Used
 * on the `ompt_scope_begin` endpoint to seed the thread-local rank cache via
 * shrink_array::begin_parallel_phase(); the `ompt_scope_end` endpoint is
 * ignored.
 *
 * @param endpoint Which side of the implicit task is being reported.
 */
void on_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int)
{
    if (!__global_ompt)
        return;

    // ompt_scope_begin is the start of the thread's work in the parallel region
    if (endpoint == ompt_scope_begin)
    {
        // Initialize the thread-local rank cache using the ID provided by OMPT
        // This is much faster than calling omp_get_thread_num() later.
        shrink_array::begin_parallel_phase();
    }
}

/**
 * @brief `ompt_callback_parallel_begin`: a parallel region is about to start.
 *
 * Clears the compression-candidate list left over from the previous phase, so
 * the region starts with an empty backlog. Runs on the encountering thread
 * only, before the team is formed.
 */
void on_parallel_begin(ompt_data_t*, const ompt_frame_t*, ompt_data_t*, uint32_t, const void*)
{
    if (!__global_ompt)
        return;

    // Vector-mode build: chunk lib has nothing to do. Avoid touching the
    // AllTypeChunk singleton — its lazy init calls omp_get_max_threads()
    // from an OMPT callback context, which segfaults in libomp (LWP crash
    // in omp_get_max_threads with vector_check target confirmed).
    if constexpr (!chunk_config::is_shrink_enabled)
        return;

    auto& ens_type_chunk = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef();

    ens_type_chunk.clean_chunk_to_compress_potentially();
}

/**
 * @brief `ompt_callback_parallel_end`: the team of a parallel region has joined.
 *
 * The library's main phase boundary: flushes the deferred writes through
 * shrink_array::finalize_parallel_phase(), then compresses the candidates
 * accumulated during the region. Both steps are individually suppressed during
 * a chunk-level rechunk rebuild (`g_in_rechunk_rebuild`) — see the inline
 * comments, which record why each suppression is needed and what regressed
 * when it was missing.
 */
void on_parallel_end(ompt_data_t*, const ompt_frame_t*, ompt_data_t*, uint32_t, const void*)
{
    if (!__global_ompt)
        return;

    // Vector-mode build: chunk lib is inactive, nothing to finalize.
    if constexpr (!chunk_config::is_shrink_enabled)
        return;

    // During the chunk-level rechunk rebuild, skip the DEFERRED FLUSH only: the
    // rebuild drives the deferred write path inside its own parallel region, and
    // flushing those half-applied ops here corrupts grow regions spanning several
    // old chunks (rechunk_to flushes them properly afterwards, at a quiescent
    // point). The COMPRESSION below is NOT skipped — it drains the freshly
    // written chunks' compression candidates (e.g. Chunk2A raw-fill), which is
    // what reclaims their materialised buffers; suppressing it caused a large
    // memory retention on global re-chunks.
    if (shrink_array::__global_need_flush_deferred && !shrink_array::g_in_rechunk_rebuild)
    {
        shrink_array::finalize_parallel_phase();
    }

    // During the chunk-level rechunk rebuild, ALSO skip block compression here.
    // The rebuild writes tmp's new chunks inside its own parallel region; if we
    // compress them at this parallel-end (before rechunk_to reaches its quiescent
    // flush/drop point), an all-second_value new chunk that spans the old-chunk
    // boundary gets re-materialised at the OLD geometry, dropping every element
    // past old_size (they read back first_value). rechunk_to compresses/drops the
    // new chunks itself, afterwards, at the correct new geometry. (Buffer reclaim
    // that this used to do during the rebuild is handled by rechunk_drain_reuse_
    // list on the swapped-out storage — a separate, quiescent path.)
    if (shrink_array::g_in_rechunk_rebuild)
        return;

    auto& ens_type_chunk = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef();

    if (!ens_type_chunk.have_chunk_to_compress_potentially())
        return;

    #pragma omp barrier

    ens_type_chunk.compress_chunk_to_compress_potentially(true);
}

/**
 * @brief `ompt_callback_sync_region`: a synchronisation region (barrier) event.
 *
 * Makes the values written before the barrier visible after it, by flushing the
 * deferred write queues through shrink_array::finalize_barrier_phase().
 * Suppressed during a rechunk rebuild for the same reason as on_parallel_end().
 */
void on_barrier_end( // ompt_sync_region_kind_t kind,
    ompt_data_t*,
    const ompt_frame_t*,
    ompt_data_t*,
    uint32_t,
    const void*)
{
    if (!__global_ompt)
        return;

    // Same suppression as on_parallel_end: the rechunk rebuild's own parallel
    // region ends with an implicit barrier; flushing the half-applied deferred
    // writes here corrupts grow regions spanning several old chunks (rechunk_to
    // flushes them properly afterwards). Without this guard the barrier path
    // reintroduced the cst2 grow loss that on_parallel_end's guard alone fixed.
    if (shrink_array::__global_need_flush_deferred && !shrink_array::g_in_rechunk_rebuild)
    {
        shrink_array::finalize_barrier_phase();
    }
}


/**
 * @brief OMPT tool initialiser, called once by the runtime at startup.
 *
 * Looks up `ompt_set_callback` and registers the four callbacks above. If the
 * lookup fails the tool stays inert (an error is printed): no phase transition
 * happens at all, which shows up as retained memory rather than as a crash.
 *
 * @param lookup Runtime-provided entry-point lookup function.
 * @return 1 to keep the tool active for the whole program run.
 */
int ompt_initialize(ompt_function_lookup_t lookup, int, ompt_data_t*)
{
    auto ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
    if (ompt_set_callback)
    {
        printf("Callback 'on_parallel_begin' defined\n");
        ompt_set_callback(ompt_callback_parallel_begin, (ompt_callback_t)on_parallel_begin);
        printf("Callback 'on_parallel_end' defined\n");
        ompt_set_callback(ompt_callback_parallel_end, (ompt_callback_t)on_parallel_end);
        printf("Callback 'on_barrier' defined\n");
        ompt_set_callback(ompt_callback_sync_region, (ompt_callback_t)on_barrier_end);

        printf("Callback 'on_implicit_task' defined (for rank cache)\n");
        ompt_set_callback(ompt_callback_implicit_task, (ompt_callback_t)on_implicit_task);
    }
    else
    {
        printf("Erreur: 'ompt_set_callback' est nullptr\n");
    }
    return 1;
}

/**
 * @brief OMPT entry point looked up by the OpenMP runtime at program start.
 *
 * Advertises ompt_initialize() as the tool initialiser (no finaliser needed:
 * nothing is allocated by the tool itself).
 *
 * @return Pointer to a function-local static result, valid for the whole run.
 */
ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*)
{
    static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize, nullptr, {0}};
    return &ompt_start_tool_result;
}

#else

/**
 * @brief Master switch of the OMPT callbacks, false in a build without OMPT.
 *
 * The symbol is still defined so that code testing it links in every variant;
 * here no callback exists and the application drives the phase transitions.
 */
bool __global_ompt = false;

#endif
