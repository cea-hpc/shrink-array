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

#include <algorithm>       // std::fill_n (init_block)
#include <atomic>          // std::atomic_thread_fence
#include <cstddef>         // std::size_t
#include <cstdio>          // stderr
#include <print>           // std::println
#include <source_location> // std::source_location
#include <tuple>           // the candidate-list entries
#include <type_traits>     // std::remove_cvref_t (the re-queued candidates)

#include "chunk/zero_chunk.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/config.h"     // build_config::chunk_dynamic, chunk_config::stats
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED
#include "chunk/internal/chunk_size_policy.h"   // alloc_chunk_count
#include "chunk/internal/chunk_stats.h"         // increm_chunk_* / add_chunk_* (CHUNK_STATS)
#include "chunk/internal/manager_chunk.h"       // manager_chunk:: candidate lists (dependent calls)
#include "chunk/internal/zero_proxy.h"          // Proxy<T>: the compression pass's proxy
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_*, pmp::team, pmp::in_application_phase
#include "chunk/internal/pool_ledger.h"         // CHUNK_LEDGER_*
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush
#include "chunk/internal/test_hooks.h"          // TEST KNOB: test_hooks::maybe_pass_delay
#include "chunk/internal/queue_flag.h"          // queue_flag::hand_back / release: the pass's hand-back

namespace shrink_array
{
// No build guard: the ChunkThreadPolicy definitions below belong to the PRIMARY template,
// instantiated only with Enabled=true (OpenMP); the serial build uses the <..., false>
// specialisation. Chunk's own members compile in both modes.

// Out of line on purpose, as operator delete[] below: see there.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void* zero::ChunkThreadPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* chunks = static_cast<zero::Chunk<T>*>(::operator new(sz));
    init_chunks_cst_val(chunks, sz / sizeof(zero::Chunk<T>));
    return static_cast<void*>(chunks);
}

// Out of line on purpose, as operator new[] above: avoids a false -Wmismatched-new-delete
// (see zero_any::ChunkThreadPolicy::operator delete[] in zero_any_chunk_inline.h).
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void zero::ChunkThreadPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par(int                 thr,
                                                                                        VecTupleChunkpInt2& ll_compt)
{
    const auto& vec_tup = Derived::Manager::chunk_to_compress_potentially_thr[thr];

    const int sz = vec_tup.size();

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tup[k];

        const auto& chunk_t   = std::get<0>(tup);
        const int&  num_chunk = std::get<1>(tup);
        const int   num_tab   = std::get<2>(tup);

        // The chunk's OWN geometry, captured at enqueue -- not the ambient size.
        const auto count = ll_compt.size();
        chunk_t->fill_list_to_compress(ll_compt, num_chunk, num_tab, std::get<3>(tup).count());
        // Scanned, still marked: hand its queue flag back before the mark can go down,
        // here for the chunks the scan drops (queue_flag.h, F6).
        queue_flag::hand_back(zero::Proxy<T>::get_tab_is_inside_list_to_compress(num_tab)[num_chunk]);
        // Not selected by the scan: the writers may have it back now.
        if (ll_compt.size() == count)
        {
            chunk_t->steal_unmark();
        }
    }
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_to_compress(this Derived&       self,
                                                                           VecTupleChunkpInt2& ll_compt,
                                                                           int                 num_chunk,
                                                                           int                 num_tab,
                                                                           unsigned int        n_elems)
{
    if (self.written())
    {
        // CHUNK_STATS: one candidate scanned (scan reads, not application reads).
        increm_chunk_compress_attempt<Derived>();
        if (self.is_constant_to(zero, n_elems))
        {
            ll_compt.push_back(std::make_tuple(std::addressof(self), num_chunk, num_tab, ChunkElems{n_elems}));
        }
    }
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::compress(this Derived& self, int thr_rank, unsigned int n_elems)
{
    // Already compressed: nothing to do. Required: a chunk can be reached twice (its queued
    // candidate outlives a compression by another path, e.g. the re-chunk fold). On a cst1
    // chunk `values` IS the shared zero sentinel: is_constant_to(0) would answer true and
    // clear_st() would park the SENTINEL into a reuse list, to be handed out as a buffer.
    if (!self.written())
    {
        return;
    }

    self.Base::debug_check_written();

    // CHUNK_STATS: counted here so the scan yield reflects what actually runs.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress_attempt<ChunkType>();
    }

    if (self.Base::is_constant_to(T{}, n_elems))
    {
        self.clear_st(thr_rank, n_elems);
    }
}

template <typename T>
void zero::Chunk<T>::compress_sweep_rank(int thr)
{
    // Bypasses CompressionManager::start(): see check_no_flush_outside_start.
    check_no_flush_outside_start<zero::Proxy<T>>("Chunk::compress_sweep_rank");
    auto& chunk_t = Manager::chunk_to_compress_potentially_thr[thr];
    // Dynamic geometry: pin the ambient chunk size to EACH candidate's own (its ChunkElems),
    // restored after the loop. free_block_to_pool() parks in the list of the AMBIENT size,
    // which after a re-chunk may be another array's: a 64-buffer parked in the 128 list
    // is later handed out as a 128-buffer. zero_one_value and zero_any pin the same way.
    [[maybe_unused]] const unsigned int saved_sz = thread_chunk_size();
    for (auto& tup : chunk_t)
    {
        auto* chunk_p = std::get<0>(tup);
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
        }
        chunk_p->compress(thr, std::get<3>(tup).count());

        // Release the dedup gate: add_to_compression_list queues a chunk only when its
        // is_inside flag CASes 0->1, so a flag left at 1 means the chunk is never queued
        // again (compression stalls). Reset it TOGETHER with the list clear below: one
        // without the other lets a write re-append the chunk and make_block_compression
        // double-clears it (see zero_any::Chunk<T>::compress_chunk_to_compress_in_par).
        zero::Proxy<T>::reset_tab_inside_list_to_compress(std::get<2>(tup), std::get<1>(tup));
    }

    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(saved_sz);
    }
    // clear() keeps the capacity for the next pass.
    chunk_t.clear();
}

template <typename T>
void zero::Chunk<T>::compress_chunk_to_compress(bool /*apply*/)
{
    // Each rank its own list, the ranks beyond the team too: a flag left at 1 is a
    // chunk never queued again (see ChunkBase::sweep_every_rank).
    const auto sweep = [](int thr) { compress_sweep_rank(thr); };
    Base::sweep_every_rank(sweep, sweep);
}


template <typename T, typename Derived, bool Enabled>
bool zero::ChunkFastCompPolicy<T, Derived, Enabled>::have_chunk_to_compress(int max_thr)
{
    if (Manager::chunk_to_compress_potentially_thr.empty())
    {
        return false;
    }

    for (int thr = 0; thr < max_thr; ++thr)
    {
        const auto& chunk_t = Manager::chunk_to_compress_potentially_thr[thr];

        const bool is_empty = chunk_t.empty();
        if (!is_empty)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief Zero-fill a freshly allocated zero::Chunk buffer before it is published (create_value's fresh-block
 *        initialiser).
 * @param vals buffer of shrink_array::alloc_chunk_count() elements
 * @details Only the pooled allocator needs the fill (svcalloc_align returns zeroed memory).
 *          The std path must stay on a zeroing allocator: plain malloc leaves garbage in
 *          cells read before written.
 */
template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::zero_fill_fresh(T* vals) noexcept
{
    if constexpr (build_config::use_fast_alloc)
    {
        std::fill_n(vals, alloc_chunk_count(), Base::zero);
    }
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::create_value(this Derived& self, int thr_rank) noexcept
{
    // Reuse pool first, then a fresh allocation (ChunkAllocPolicy::allocate_chunk_internal,
    // shared with zero_one_value/zero_any). Initialisers differ from zero_one_value/zero_any: a zero::Chunk is
    // compressed only when all-zero, so a recycled buffer is already zero and needs no fill.
    T* values_loc = self.Base::allocate_chunk_internal(
        thr_rank, [](T*) noexcept { /* recycled: already zero, see above */ },
        [](T* vals) noexcept { zero_fill_fresh(vals); });

    if (self.compare_and_swap(values_loc))
    {
        self.Base::recup_value(values_loc, thr_rank);
    }

    // Release fence: the caller's following stores cannot be reordered before the CAS (AArch64).
    std::atomic_thread_fence(std::memory_order_release);
}

template <typename T, typename Derived>
void zero::ChunkThreadPolicy<T, Derived, false>::create_value(this Derived& self) noexcept
{
    // Serially there is no reuse list: the first initialiser is ignored (serial
    // ChunkAllocPolicy).
    self.values_ = self.Base::allocate_chunk_internal(
        0, [](T*) noexcept {},
        [](T* vals) noexcept
        {
            if constexpr (build_config::use_fast_alloc)
            {
                std::fill_n(vals, alloc_chunk_count(), Base::zero);
            }
        });
}

// No guard needed: the primary template is only instantiated with Enabled=true.
template <typename T, typename Derived, bool Enabled>
void zero::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr, const zero::Proxy<T>* proxy)
{
    // Before the first parallel region the per-rank lists do not exist (sized at region
    // begin), yet zero::Array::resize() calls this from create(): do not index them.
    if (thr < 0 || static_cast<size_t>(thr) >= Manager::chunk_to_compress_potentially_thr.size())
    {
        return;
    }

    auto& vec_tuple_to_compress = Manager::chunk_to_compress_potentially_thr[thr];

    const int sz = vec_tuple_to_compress.size();

    if (sz == 0)
    {
        return;
    }

    const int max_thr = CHUNK_PMP_MAX_THREADS;
    // Other threads may write the candidates: a team of more than one, or --
    // std backend -- an application phase, whose threads no team counts (a
    // team of one there still has the application's writers around it). Taken
    // once: the start() below and its finish() at the end must agree.
    const bool guarded = proxy != nullptr && (max_thr > 1 || pmp::in_application_phase());

    if (guarded)
    {
        // A flush in progress makes start() decline: ask first, before marking, or the pass
        // spins in mark/unmark rounds (block_pass_precheck.h).
        if (block_pass_yields_to_flush<zero::Proxy<T>, zero::Chunk<T>>(max_thr))
        {
            return;
        }
        // Per-chunk exclusion (ChunkBase::steal_mark_): claim every candidate
        // BEFORE start(), whose heavy barrier publishes the marks; released
        // per chunk as the pass goes, and all again at the end.
        for (int k = 0; k < sz; ++k)
        {
            if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
            {
                chunk_t->steal_mark();
            }
        }
        const bool ret = proxy->compression_mgr.template start<zero::Proxy<T>>();

        if (!ret)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                {
                    chunk_t->steal_unmark();
                }
            }
            return;
        }
        // Check builds: window shut, commit the claims (tested by store_value()'s guard).
        if constexpr (build_config::any_check)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                {
                    chunk_t->steal_commit();
                }
            }
        }
    }

    VecTupleChunkpInt2 ll_compt;

    ll_compt.reserve((sz / 10) + 1);

    // The scan hands every candidate's queue flag back as it goes, before any mark goes
    // down (fill_list_chunk_to_compress_in_par: queue_flag.h, F6); the loop at the end
    // releases them.
    fill_list_chunk_to_compress_in_par(thr, ll_compt);

    // Geometry: a rank's list mixes chunks of EVERY array, at their own sizes. clear_int_st()
    // frees into the reuse list of the AMBIENT size, so pin each chunk to its own n_elems
    // right before its clear (as zero_one_value / zero_any) -- never once to the triggering array's.
    [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();
    for (const auto& tup : ll_compt)
    {
        auto* chunk_ptr = std::get<0>(tup);

        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
        }
        chunk_ptr->clear_int_st(thr);
        chunk_ptr->steal_unmark();
    }
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }

    // The candidates a writer queued while this pass handed them back (queue_flag::release()).
    std::remove_cvref_t<decltype(vec_tuple_to_compress)> requeued;

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tuple_to_compress[k];

        const int num_chunk = std::get<1>(tup);

        const int num_tab = std::get<2>(tup);
        // Hand the candidate back: its mark FIRST, then the dedup flag (the reverse order can
        // erase another pass's claim: see zero_any::Chunk<T>::make_block_compression).
        if (auto* const chunk_t = std::get<0>(tup))
        {
            chunk_t->steal_unmark();
        }
        test_hooks::maybe_pass_delay(); // TEST KNOB (test_hooks.h)
        if constexpr (build_config::any_check)
        {
            if (auto* const marked = std::get<0>(tup); marked != nullptr && marked->steal_marked()) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("compression pass: candidate {} was marked while this pass still held its "
                                       "dedup flag -- two passes on one chunk",
                                       marked);
            }
        }
        if (build_config::threaded && std::get<0>(tup) != nullptr)
        {
            if (queue_flag::release(zero::Proxy<T>::get_tab_is_inside_list_to_compress(num_tab)[num_chunk]))
            {
                requeued.push_back(tup);
            }
        }
        else
        {
            zero::Proxy<T>::reset_tab_inside_list_to_compress(num_tab, num_chunk);
        }
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);
    for (const auto& tup : requeued)
    {
        vec_tuple_to_compress.push_back(tup);
    }

    if (guarded)
    {
        if (proxy->is_not_cst_val())
        {
            proxy->add_to_compression_list(vec_tuple_to_compress);
        }

        proxy->compression_mgr.template finish<zero::Proxy<T>>();
    }
}

// CHUNK_STATS: release counter is "pool" under OpenMP, "free" serially. The OMP zero::Chunk
// path does not come here (clear_int_st releases the block itself and counts there): do not
// count in both if this entry point is ever wired up.
template <typename T>
void zero::Chunk<T>::block_clear(this auto& self, [[maybe_unused]] int thr_rank)
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<zero::Chunk<T>>();
        add_chunk_bytes_freed<zero::Chunk<T>>(static_cast<long>(sizeof(T)) *
                                              static_cast<long>(shrink_array::alloc_chunk_count()));
        if constexpr (build_config::threaded)
        {
            increm_chunk_release_pool<zero::Chunk<T>>();
        }
        else
        {
            increm_chunk_release_free<zero::Chunk<T>>();
        }
    }

    if constexpr (build_config::threaded)
    {
        self.Base::free_block_to_pool(thr_rank, self.values_);
    }
    else
    {
        self.Base::free_values();
        self.init_values_cst_val();
    }
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::clear_int_st(this Derived& self, int thr_rank)
{
    T* old = self.try_mark_cst_and_get_old();
    if (old)
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_compress<zero::Chunk<T>>();
            add_chunk_bytes_freed<zero::Chunk<T>>(static_cast<long>(sizeof(T)) *
                                                  static_cast<long>(shrink_array::alloc_chunk_count()));
            increm_chunk_release_pool<zero::Chunk<T>>();
        }
        self.Base::free_block_to_pool(thr_rank, old);
    }
}


template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::clear_st(this Derived& self, int thr_rank, unsigned int n_elems)
{
    // CHUNK_STATS: a compression completing. Count here, not in init_values_cst_val (also
    // reached from teardown/neutralisation, which are not compressions). The re-chunk
    // rebuild releases through release_st() directly so it is not counted as compression.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<zero::Chunk<T>>();
        add_chunk_bytes_freed<zero::Chunk<T>>(static_cast<long>(sizeof(T)) * static_cast<long>(n_elems));
        increm_chunk_release_park<zero::Chunk<T>>();
    }
    self.release_st(thr_rank, n_elems, /*count_release=*/false);
}

template <typename T, typename Derived, bool Enabled>
void zero::ChunkThreadPolicy<T, Derived, Enabled>::release_st(this Derived& self,
                                                              int           thr_rank,
                                                              unsigned int  n_elems,
                                                              bool          count_release)
{
    // CHUNK_STATS: a release that is not a compression (re-chunk rebuild): bytes still
    // leave the live figure.
    if constexpr (chunk_config::stats)
    {
        if (count_release)
        {
            add_chunk_released<zero::Chunk<T>>(1, static_cast<long>(sizeof(T)) * static_cast<long>(n_elems));
        }
    }

    if constexpr (build_config::any_check)
    {
        if (!self.written())
        {
            invariant_failed(std::source_location::current(), "trying to clear a chunk that is not written (chunk={})",
                             &self);
        }
    }

    // Park into the list of the buffer's OWN geometry (n_elems), not the ambient one:
    // get_list_block_free_thr(thr) routes by thread_chunk_size(), i.e. by whichever array
    // wrote last. With two geometries alive, a 64-element buffer would land in the 256 list
    // and be popped for a 256-element chunk (overflow). The memset and the push must agree.
    auto& ll_block_free_thr = Manager::get_list_block_free_thr_for_size(thr_rank, n_elems);

    Base::init_block(self.values_.load(), n_elems);
    CHUNK_LEDGER_PARK(self.values_.load(), "clear_st.park"); // LIVE -> parked before it re-enters the pool
    ll_block_free_thr.push_back(self.values_);

    self.init_values_cst_val();
}

template <typename T, typename Derived>
void zero::ChunkThreadPolicy<T, Derived, false>::clear_st(this Derived& self)
{
    // CHUNK_STATS: serial twin of the OpenMP clear_st above.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<zero::Chunk<T>>();
        add_chunk_bytes_freed<zero::Chunk<T>>(static_cast<long>(sizeof(T)) *
                                              static_cast<long>(shrink_array::alloc_chunk_count()));
        increm_chunk_release_free<zero::Chunk<T>>();
    }
    self.release_st(/*count_release=*/false);
}

template <typename T, typename Derived>
void zero::ChunkThreadPolicy<T, Derived, false>::release_st(this Derived& self, bool count_release)
{
    if constexpr (chunk_config::stats)
    {
        if (count_release)
        {
            add_chunk_released<zero::Chunk<T>>(1, static_cast<long>(sizeof(T)) *
                                                      static_cast<long>(shrink_array::alloc_chunk_count()));
        }
    }

    if constexpr (build_config::is_debug)
    {
        if (!self.written())
        {
            // As ChunkBase::debug_check_written: clearing an unwritten chunk frees the shared
            // sentinel as a buffer.
            CHUNK_INVARIANT_FAILED("trying to clear a chunk that is not written");
        }
    }

    self.Base::free_values();

    self.init_values_cst_val();
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkThreadPolicy<T, Derived, Enabled>
// ---------------------------------------------------------------------------

template <typename T, typename Derived, bool Enabled>
T* zero::ChunkThreadPolicy<T, Derived, Enabled>::try_mark_cst_and_get_old(this Derived& self)
{
    T* old = self.Base::values_load();
    if (old == Base::get_cst_val1()) [[unlikely]]
    {
        return nullptr;
    }

    T* expected = old;
    T* new_ptr  = Base::get_cst_val1();

    if (!self.values_.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel, std::memory_order_acquire))
        [[unlikely]]
    {
        return nullptr;
    }
    return old;
}

// ---------------------------------------------------------------------------
// shrink_array::Chunk<T>
// ---------------------------------------------------------------------------

template <typename T>
zero::Chunk<T>::Chunk(const Chunk& other) : ChunkCounterPolicy<chunk_config::use_compression_with_counter>(other)
{
    if constexpr (build_config::threaded)
    {
        values_.store(other.values_.load());
    }
    else
    {
        values_ = other.values_;
    }
}

template <typename T>
zero::Chunk<T>& zero::Chunk<T>::operator=(const Chunk<T>& other)
{
    if (this == &other)
    {
        return *this;
    }

    // Same reason as the copy ctor: propagate the zero-counter state.
    ChunkCounterPolicy<chunk_config::use_compression_with_counter>::operator=(other);

    if constexpr (build_config::threaded)
    {
        T* loaded_value = other.values.load();
        values_.store(loaded_value);
    }
    else
    {
        values_ = other.values;
    }

    return *this;
}

template <typename T>
void zero::Chunk<T>::alloc_chunk(this auto& self, [[maybe_unused]] int thr_rank)
{
    if (self.is_cst_val1())
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_write<ChunkType>();
        }

        if constexpr (build_config::threaded)
        {
            self.create_value(thr_rank);
        }
        else
        {
            self.create_value();
        }
    }
}

template <typename T>
auto& zero::Chunk<T>::get_manager() noexcept
{
    // Do not make this a function-local static: its lazy guard's first init
    // can run from compress() inside an OpenMP region and deadlock threads on
    // __cxa_guard_acquire. A LazyImmortal (chunk/internal/immortal.h) is built
    // by the manager's registration (ManagerChunk::init_static_state), before
    // any array or thread reaches it, and never destroyed (a global array
    // still uses it after main).
    return *s_manager_;
}

} // namespace shrink_array