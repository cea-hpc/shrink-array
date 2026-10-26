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
#include "fast_allocators/internal/pmp_loops.h" // the class these bodies belong to

// pmp_loops.h declares this Team only off the std backend (which has its own,
// pmp_std.h): its bodies follow the same switch.
#if !FAST_ALLOC_PMP_BACKEND_STD

namespace fast_alloc
{

// ---------------------------------------------------------------------------
// fast_alloc::Team
// ---------------------------------------------------------------------------

template <typename Index, typename Body>
void pmp::Team::for_static(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for schedule(static) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename Index, typename Body>
void pmp::Team::for_static_simd(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for simd schedule(static) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for simd schedule(static)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename Index, typename Body>
void pmp::Team::for_static_chunks(Index first, Index last, int chunk, Body&& body, [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    [[maybe_unused]] const int step = chunk > 0 ? chunk : 1;
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for schedule(static, step) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for schedule(static, step)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename Index, typename Body>
void pmp::Team::for_static_chunks_simd(Index first,
                                       Index last,
                                       int chunk,
                                       Body&& body,
                                       [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    [[maybe_unused]] const int step = chunk > 0 ? chunk : 1;
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for simd schedule(static, step) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for simd schedule(static, step)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename Index, typename Body>
void pmp::Team::for_dynamic(Index first, Index last, int chunk, Body&& body, [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    [[maybe_unused]] const int step = chunk > 0 ? chunk : 1;
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for schedule(dynamic, step) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for schedule(dynamic, step)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename Index, typename Body>
void pmp::Team::for_runtime(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp for schedule(runtime) nowait
        for (Index i = first; i < last; ++i)
        {
            body(i);
        }
        return;
    }
    #pragma omp for schedule(runtime)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
#endif
}

template <typename F>
void pmp::Team::single(F&& func, [[maybe_unused]] WaitMode wait_mode)
{
#ifdef _OPENMP
    if (wait_mode == WaitMode::nowait)
    {
    #pragma omp single nowait
        func();
        return;
    }
    #pragma omp single
    func();
#else
    func();
#endif
}

template <typename F>
void pmp::Team::masked(F&& func)
{
#ifdef _OPENMP
    #pragma omp masked
    func();
#else
    func();
#endif
}

} // namespace fast_alloc

#endif // !FAST_ALLOC_PMP_BACKEND_STD
