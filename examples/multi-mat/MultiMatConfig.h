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
 * @file MultiMatConfig.h
 * @brief The compression mode of this example: who drives the chunk library.
 *
 * The library compresses from its write path in every OpenMP build: a write
 * that closes a constant run queues its chunk, and a thread whose list fills
 * runs a pass over the candidates. What a pass needs from the application is
 * the PHASES (chunk/manage_ompt.h): a write that lands on a chunk the pass
 * holds is deferred, and somebody has to replay the deferred writes before
 * they are read -- at the barriers -- and to drain the candidates left when
 * the region ends. Two ways to provide them, two executables:
 *
 *  - manual (default, every compiler): the application makes the phase
 *    calls itself -- the four one-line wrappers below, and nothing else --
 *    and compresses its field arrays by explicit compress() calls every
 *    freq_compress iterations (ComputeLoops.h). This binary carries no OMPT
 *    tool, on any compiler;
 *
 *  - embedded (MULTIMAT_EMBEDDED_COMPRESSION): no call at all, the parallel
 *    loops are plain OpenMP. The OMPT tool of manage_ompt.cpp, next to this
 *    file, makes the phase calls from the runtime's callbacks --
 *    implicit-task begin, barrier end, parallel end. That tool is what the
 *    mode stands on, so a build without OMPT is refused here rather than
 *    compiled to a version that loses the deferred writes: HAVE_OMPT is set
 *    by the top-level CMakeLists for Clang and IntelLLVM (OMP=ON, not under
 *    TSan); gcc's libgomp has no OMPT.
 *
 * The presence fraction (a Chunk2 array) is compressed by that engine in
 * both modes: its compress() is an audit pass, not the compressor.
 */

#include "chunk/config.h"
#include "chunk/manage_ompt.h"

#if defined(MULTIMAT_EMBEDDED_COMPRESSION) && !defined(HAVE_OMPT)
    #error "multi-mat-embedded rests on the OMPT tool of the chunk library, which this build has not (HAVE_OMPT): build with Clang or IntelLLVM, OMP=ON, no TSan -- gcc's libgomp has no OMPT"
#endif

namespace multi_mat
{

#ifdef MULTIMAT_EMBEDDED_COMPRESSION
inline constexpr bool embedded_compression = true;
#else
inline constexpr bool embedded_compression = false;
#endif

// The manual phase calls, elided in the embedded executable.

/// @brief Inside the region, by every thread, before the first chunk write.
inline void begin_parallel_phase()
{
    if constexpr (!embedded_compression)
        shrink_array::begin_parallel_phase();
}

/// @brief After a barrier, before the reads that must see the writes of
///        before it: every thread replays its deferred writes, then a
///        barrier again so no thread reads while another still replays.
inline void barrier_phase()
{
    if constexpr (!embedded_compression)
    {
        shrink_array::finalize_barrier_phase();
#pragma omp barrier
    }
}

/// @brief Last thing inside the region: the deferred writes, replayed by
///        the team.
inline void end_parallel_phase_in_region()
{
    if constexpr (!embedded_compression)
        shrink_array::finalize_parallel_phase_only_flush();
}

/// @brief First thing after the region: the pools reset and the candidates
///        left over compressed.
inline void end_parallel_phase_after_region()
{
    if constexpr (!embedded_compression)
        shrink_array::finalize_parallel_phase_after_flush();
}

} // namespace multi_mat
