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
#include "fast_allocators/internal/pmp_std.h" // the class these bodies belong to

namespace fast_alloc
{

// ---------------------------------------------------------------------------
// fast_alloc::TeamCtx
// ---------------------------------------------------------------------------

inline pmp::std_backend::Slot& pmp::std_backend::TeamCtx::enter(int rank, std::uint64_t& seq)
{
    seq = met[static_cast<std::size_t>(rank)]++;
    const std::scoped_lock lock(mutex);
    auto&                  s = slots[seq];
    if (!s)
    {
        s       = std::make_unique<Slot>();
        s->left = size;
    }
    return *s;
}

inline void pmp::std_backend::TeamCtx::leave(std::uint64_t seq)
{
    const std::scoped_lock lock(mutex);
    const auto             it = slots.find(seq);
    if (--it->second->left == 0)
    {
        slots.erase(it);
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::Pool
// ---------------------------------------------------------------------------

template <typename Job>
void pmp::std_backend::Pool::run(int n, Job& job)
{
    const std::scoped_lock lock(team_mutex_);
    while (static_cast<int>(workers_.size()) < n - 1)
    {
        const int           r     = static_cast<int>(workers_.size()) + 1;
        const std::uint64_t start = word_.load(std::memory_order_acquire);
        workers_.emplace_back([this, r, start] { worker_loop(r, start); });
    }
    fn_  = [](void* j, int rank) { (*static_cast<Job*>(j))(rank); };
    arg_ = &job;
    pending_.store(n - 1, std::memory_order_relaxed);
    const std::uint64_t gen = (word_.load(std::memory_order_relaxed) >> 16) + 1;
    word_.store((gen << 16) | static_cast<std::uint64_t>(n), std::memory_order_release);
    word_.notify_all();
    call(0);
    for (int p = pending_.load(std::memory_order_acquire); p != 0; p = pending_.load(std::memory_order_acquire))
    {
        pending_.wait(p, std::memory_order_acquire);
    }
}

inline void pmp::std_backend::Pool::call(int rank) noexcept
{
    try
    {
        fn_(arg_, rank);
    }
    catch (...)
    {
        std::terminate(); // as OpenMP: an exception must not leave a region
    }
}

inline void pmp::std_backend::Pool::worker_loop(int rank, std::uint64_t seen) noexcept
{
    for (;;)
    {
        word_.wait(seen, std::memory_order_acquire);
        seen = word_.load(std::memory_order_acquire);
        if (rank < static_cast<int>(seen & 0xFFFFU))
        {
            call(rank);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                pending_.notify_one();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::Team
// ---------------------------------------------------------------------------

template <typename Index, typename Body>
void pmp::Team::for_static(Index first, Index last, Body&& body, WaitMode w)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    long long b = 0;
    long long e = static_cast<long long>(last) - static_cast<long long>(first);
    if (ctx_ != nullptr && e > 0)
    {
        std_backend::static_block(e, ctx_->size, rank_, b, e);
    }
    for (long long k = b; k < e; ++k)
    {
        body(static_cast<Index>(first + static_cast<Index>(k)));
    }
    end_construct(w);
}

template <typename Index, typename Body>
void pmp::Team::for_static_chunks(Index first, Index last, int chunk, Body&& body, WaitMode w)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    const long long n  = static_cast<long long>(last) - static_cast<long long>(first);
    const long long c  = chunk > 0 ? chunk : 1;
    const long long nt = size();
    for (long long b = rank_ * c; b < n; b += nt * c)
    {
        const long long e = b + c < n ? b + c : n;
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
    }
    end_construct(w);
}

template <typename Index, typename Body>
void pmp::Team::for_dynamic(Index first, Index last, int chunk, Body&& body, WaitMode w)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    const long long n = static_cast<long long>(last) - static_cast<long long>(first);
    const long long c = chunk > 0 ? chunk : 1;
    if (ctx_ == nullptr)
    {
        for (long long k = 0; k < n; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
        return;
    }
    std::uint64_t      seq  = 0;
    std_backend::Slot& slot = ctx_->enter(rank_, seq);
    for (long long b = slot.next.fetch_add(c, std::memory_order_relaxed); b < n;
         b           = slot.next.fetch_add(c, std::memory_order_relaxed))
    {
        const long long e = b + c < n ? b + c : n;
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
    }
    ctx_->leave(seq);
    end_construct(w);
}

template <typename Index, typename Body>
void pmp::Team::for_runtime(Index first, Index last, Body&& body, WaitMode w)
{
    const auto& s     = std_backend::settings();
    const int   kind  = s.sched_kind.load(std::memory_order_relaxed) & 0x7FFFFFFF; // monotonic bit off
    const int   chunk = s.sched_chunk.load(std::memory_order_relaxed);
    if (kind == 2 || kind == 3)
    {
        for_dynamic(first, last, chunk, std::forward<Body>(body), w);
    }
    else if (kind == 1 && chunk > 0)
    {
        for_static_chunks(first, last, chunk, std::forward<Body>(body), w);
    }
    else
    {
        for_static(first, last, std::forward<Body>(body), w);
    }
}

template <typename F>
void pmp::Team::single(F&& f, WaitMode w)
{
    if (ctx_ == nullptr)
    {
        f();
        return;
    }
    std::uint64_t      seq  = 0;
    std_backend::Slot& slot = ctx_->enter(rank_, seq);
    if (!slot.claimed.exchange(true, std::memory_order_acq_rel))
    {
        f();
    }
    ctx_->leave(seq);
    end_construct(w);
}

} // namespace fast_alloc
