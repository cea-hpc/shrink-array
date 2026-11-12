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
#include <exception> // std::terminate (a body that throws)
#include <atomic> // std::memory_order_acq_rel, std::memory_order_acquire, std::memory_order_relaxed, std::memory_order_release
#include <cstddef> // std::size_t
#include <cstdint> // std::uint32_t, std::uint64_t
#include <mutex>   // std::try_to_lock, std::unique_lock
#include <thread>  // std::this_thread::yield (a dynamic slot still in use)

#include "fast_allocators/internal/pmp_std.h" // the class these bodies belong to

namespace fast_alloc
{

// ---------------------------------------------------------------------------
// fast_alloc::SpinBarrier
// ---------------------------------------------------------------------------

inline void pmp::std_backend::SpinBarrier::arrive_and_wait() noexcept
{
    const std::uint32_t gen = gen_.load(std::memory_order_acquire);
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
    {
        count_.store(n_, std::memory_order_relaxed);
        gen_.store(gen + 1U, std::memory_order_release);
        gen_.notify_all();
        return;
    }
    if (spin_until(n_, [&]() noexcept { return gen_.load(std::memory_order_acquire) != gen; }))
    {
        return;
    }
    while (gen_.load(std::memory_order_acquire) == gen)
    {
        gen_.wait(gen, std::memory_order_acquire);
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::TeamCtx
// ---------------------------------------------------------------------------

inline pmp::std_backend::DynSlot& pmp::std_backend::TeamCtx::enter_dynamic(int rank, std::uint64_t& seq) noexcept
{
    seq              = met[static_cast<std::size_t>(rank)]++;
    DynSlot&   slot  = dyn[seq % dyn_ring];
    const auto ready = 2 * (seq + 1);
    for (unsigned spins = 1;; ++spins)
    {
        std::uint64_t tag = slot.tag.load(std::memory_order_acquire);
        if (tag == ready)
        {
            return slot; // claimed and reset by the first thread of this loop
        }
        // Free: even (not being reset), an older loop's, and nobody left in it.
        // A tag above `ready` cannot be: a later loop on this slot needs this
        // thread out of this one first.
        if ((tag & 1U) == 0U && tag < ready && slot.left.load(std::memory_order_acquire) == 0 &&
            slot.tag.compare_exchange_strong(tag, ready - 1, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            slot.next.store(0, std::memory_order_relaxed);
            slot.left.store(size, std::memory_order_relaxed);
            slot.tag.store(ready, std::memory_order_release);
            return slot;
        }
        // Being reset by another thread, or still in use by a thread
        // dyn_ring constructs behind: it leaves without anyone's help.
        cpu_relax();
        if ((spins & 255U) == 0U)
        {
            std::this_thread::yield();
        }
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::Pool
// ---------------------------------------------------------------------------

template <typename Job>
bool pmp::std_backend::Pool::try_run(int n, Job& job)
{
    const std::unique_lock lock(team_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
        return false;
    }
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
    // Spin first (spin_budget_ns): the team's last worker is usually done
    // within microseconds of rank 0, and a futex wake-up costs more.
    if (spin_until(n, [this]() noexcept { return pending_.load(std::memory_order_acquire) == 0; }))
    {
        return true;
    }
    for (int p = pending_.load(std::memory_order_acquire); p != 0; p = pending_.load(std::memory_order_acquire))
    {
        pending_.wait(p, std::memory_order_acquire);
    }
    return true;
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
        // Spin first (spin_budget_ns): teams opened back to back (a re-chunk's
        // slices) find their workers awake; past the budget, sleep on the word.
        // The last team's size stands for the next one's (oversubscription).
        const auto last_team = static_cast<int>(seen & 0xFFFFU);
        if (!spin_until(last_team, [this, seen]() noexcept { return word_.load(std::memory_order_acquire) != seen; }))
        {
            word_.wait(seen, std::memory_order_acquire);
        }
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
    std::uint64_t         seq  = 0;
    std_backend::DynSlot& slot = ctx_->enter_dynamic(rank_, seq);
    for (long long b = slot.next.fetch_add(c, std::memory_order_relaxed); b < n;
         b           = slot.next.fetch_add(c, std::memory_order_relaxed))
    {
        const long long e = b + c < n ? b + c : n;
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
    }
    // acq_rel: this thread's uses of the slot happen before its reuse
    // (the next claimer reads left == 0, acquire).
    slot.left.fetch_sub(1, std::memory_order_acq_rel);
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
    // Lock-free: every thread meets the constructs in the same order, so this
    // one's number seq is the same for all (the count enter_dynamic() keeps too). The
    // mark is "every single numbered below it has been claimed"; the thread
    // that moves it past seq claims this one. A thread that finds it already
    // past seq loses: whoever moved it there went through this single first
    // (same order), and claimed it or found it claimed.
    const std::uint64_t seq  = ctx_->met[static_cast<std::size_t>(rank_)]++;
    std::uint64_t       mark = ctx_->single_mark.load(std::memory_order_acquire);
    bool                mine = false;
    while (mark <= seq)
    {
        if (ctx_->single_mark.compare_exchange_weak(mark, seq + 1, std::memory_order_acq_rel,
                                                    std::memory_order_acquire))
        {
            mine = true;
            break;
        }
    }
    if (mine)
    {
        f();
    }
    end_construct(w);
}

} // namespace fast_alloc
