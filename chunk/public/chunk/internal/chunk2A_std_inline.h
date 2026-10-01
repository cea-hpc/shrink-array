
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

#include "chunk/internal/in_t.h"            // in_t<T>: the element snapshots
#include "chunk/internal/chunk_array_new.h" // chunk_array_new::init_chunks (operator new[])
#include "chunk/internal/invariant_check.h" // INVARIANT_FAILED (steal-mark check)
#include "chunk/internal/dependent_false.h" // dependent_false_v (wrong-type branches)
#include <algorithm>                        // std::fill_n (the create_value fills)
#include <cstdlib>                          // std::abort
#include <optional>                         // std::nullopt (the clears)

#include "chunk/internal/chunk_size_policy.h" // ChunkElems, set_thread_chunk_size
#include "chunk/PMP_CHUNK.h"
#include "chunk/config.h"
#include "chunk/internal/cpu_pause.h" // CPU_PAUSE (spin in create_value*)
#include "chunk/chunk2A_std.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/internal/proxy2A.h"
#include "chunk/internal/manager_chunk.h"
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush

/**
 * @file chunk2A_std_inline.h
 * @brief Out-of-line definitions of Chunk2A: materialisation and compression.
 *
 * A Chunk2A block is in one of three states, and every function here either
 * moves it between them or decides that it should move:
 *  - **cst_val1**: `values` points at the shared all-`first_value` block;
 *  - **cst_val2**: `values` points at the shared block of THIS chunk's own
 *    second value (Chunk2A's distinguishing feature — Chunk2 has one second
 *    value per array, Chunk2A one per chunk);
 *  - **materialised**: `values` points at a private heap buffer.
 *
 * The two directions:
 *  - **materialise** (create_value(), create_value_bool()): allocate a
 *    buffer, pre-fill it, then CAS it into `values`. Publication is the CAS,
 *    which is what makes a concurrent reader
 *    see either the old constant or the finished buffer, never a half-built one;
 *  - **compress** (the clear / block_clear / clear_int_st family): release the
 *    buffer and point `values` back at a shared constant block.
 *
 * Around those sits the candidate pipeline: a write flags a chunk as a
 * candidate, fill_list_to_compress() scans it and sorts it into the
 * "all-first_value" or "all-one-other-value" list, and make_block_compression()
 * applies the clears and clears the flags.
 *
 * Two things to keep in mind when reading:
 *
 *  - **Almost every function exists twice**, under `_OPENMP` and without. The
 *    serial twin is not just the parallel one minus the locking: it often takes
 *    a different, shorter path (no CAS, no rank, no compression handshake).
 *  - **Geometry is pinned per chunk, never inherited.** The candidate list can
 *    hold chunks from several arrays at once, so a single pin at the top of a
 *    pass is wrong; each entry carries its owning array's ChunkElems and the
 *    ambient size is set from it right before the clear. Getting this wrong
 *    frees a buffer through the wrong size pool.
 *
 * @see chunk2A_std.h for the declarations, chunk_base.h for the shared
 *      allocate/free machinery, proxy2A_inline.h for the write paths that feed
 *      the candidate lists.
 */

namespace shrink_array
{

// ── Array-new (placement-new arrays; see chunk_array_new.h) ────────────────
template <typename T, typename Derived, bool Enabled>
void* Chunk2AOmpPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* v = static_cast<Chunk2A<T>*>(::operator new(sz));
    init_chunks_cst_val(v, sz / sizeof(Chunk2A<T>));
    return static_cast<void*>(v);
}

template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::init_chunks_cst_val(Chunk2A<T>* v, std::size_t count) noexcept
{
    chunk_array_new::init_chunks(v, count, [](Chunk2A<T>& chunk) { chunk.init_after_constructor(); });
}

/**
 * @brief Records which of the two constants this chunk is being compressed to.
 *
 * @param num_compress_value 0 for the first constant, anything else for the second.
 * @param ptr_type_cst Slot of the per-chunk type table to write.
 */
template <typename T>
void Chunk2A<T>::set_cst_val_to_cst_val(int num_compress_value, char* ptr_type_cst) noexcept
{
    if (num_compress_value == 0)
    {
        set_cst_val_to_cst_val1(ptr_type_cst);
    }
    else
    {
        set_cst_val_to_cst_val2(ptr_type_cst);
    }
}

/**
 * @brief Scans this chunk and sorts it into one of the two compression lists.
 *
 * Reads the whole chunk once, tracking two hypotheses at the same time — "every
 * element is first_value" and "every element is this one other value" — and
 * stops as soon as both are disproved. A chunk that satisfies both (a constant
 * run of first_value) is filed as compact1 only.
 *
 * Two cases, on whether the chunk is currently compressed to its second value:
 *  - **materialised**: the candidate second value is taken from element 0, and
 *    a compact2 hit is queued WITH that value, for the caller to register;
 *  - **already cst_val2**: the entry is queued with a `T{}` marker instead.
 *
 * @param ll_compt1  List of chunks compressible to first_value.
 * @param ll_compt2  List of chunks compressible to one other value.
 * @param num_chunk  Index of this chunk in its array.
 * @param num_tab    Table number of the owning array.
 * @param size       Geometry of the owning array, carried through to the clear.
 * @param claimed    Whether the caller claimed (steal-marked) the chunk first;
 *                   check builds assert it.
 *
 * @note Chunks already compressed to first_value are skipped outright.
 * @note The `T{}` marker is only unambiguous because `first_value` is 0: a
 *       genuinely all-zero chunk is classified compact1, so it never reaches
 *       ll_compt2 with a zero value. Nothing in the library reassigns
 *       first_value today, but the coupling is worth knowing about.
 */
template <typename T>
void Chunk2A<T>::fill_list_to_compress(vec_tuple_chunkA2p_int2&  ll_compt1,
                                       vec_tuple_chunkA2p_Tint2& ll_compt2,
                                       int                       num_chunk,
                                       int                       num_tab,
                                       ChunkElems                size,
                                       [[maybe_unused]] bool     claimed)
{
    if (Base::is_not_cst_val1())
    {
        // The chunk about to be scanned must be claimed. make_block_compression()
        // marks every entry of vec_tuple_to_compress before start(), whose
        // barrier publishes the marks to the writers -- but that list holds the
        // candidates already known, and scanning is how new ones are found. A
        // chunk reached that way was never marked, and no writer has a reason to
        // stay off it: TSan reports scan_constants() reading a chunk that
        // another thread's store_value() is writing, at 2 threads as at 8.
        //
        // Asserting rather than assuming: if this fires, the hole is where it
        // looks. If it never fires, the race comes from start()'s window
        // instead -- a writer already past the filter when the marks went out.
        if constexpr (build_config::any_check && chunk_config::use_chunk_steal_mark)
        {
            // Only where a claim was made. The resize entry point
            // (ArrayChunk2A::make_block_compression -> Chunk2A::
            // make_block_compression(thr, nullptr)) runs outside any parallel
            // region and skips the claim loop by design: no writer can be in
            // flight, so no mark is needed and its absence proves nothing.
            //
            // On the claimed path it proves a lot: the claim covers
            // vec_tuple_to_compress, the candidates already known, while
            // scanning is how new ones are DISCOVERED -- and a chunk reached
            // that way was never marked, which is what TSan reports as
            // scan_constants() reading a chunk another thread is writing.
            if (claimed && !Base::steal_marked()) [[unlikely]]
                INVARIANT_FAILED("fill_list_to_compress on an unclaimed chunk (num_chunk=%d num_tab=%d)", num_chunk,
                                 num_tab);
        }

        // CHUNK_STATS: one compression candidate scanned. compress/compress_attempt
        // is the yield of the scan (how much of this work pays off).
        increm_chunk_compress_attempt<Chunk2A<T>>();

        // Already compressed to its second value: nothing to scan, and
        // nothing that may be scanned. A candidate can be in that state when
        // the single-chunk compress() (compress_int, sequential build) ran
        // over it while it was still queued: its flag is set, its entry is
        // still in vec_tuple_to_compress, and it comes back through here at
        // the next pass. The branch that used to scan it read through
        // `values` -- a second-value slot, a region that was never
        // written, so every element read as 0 == first_value, the chunk
        // went on ll_compt1 and block_clear() freed the region pointer as a
        // buffer (SIGSEGV in free_elt_pow2 on aarch64, the invariant check
        // "Attempted to free 'cst_val2'" in a debug build). With the arrays
        // filled (before Oct '26) it was merely useless: the re-scan sent the chunk to
        // ll_compt2, and the OpenMP CAS refused the clear.
        if (this->is_cst_val2())
            return;

        // No second value yet: whatever the first element holds is the
        // candidate second constant.
        const in_t<T> second_v = Base::read_value_scan(0);
        const auto    scan     = Base::scan_constants(first_value, second_v, size.count());

        if (scan.first)
        {
            ll_compt1.emplace_back(std::make_tuple(this, num_chunk, num_tab, size));
        }
        if (scan.second)
        {
            ll_compt2.emplace_back(std::make_tuple(this, second_v, num_chunk, num_tab, size));
        }
    }
}

/**
 * @brief Runs a full compression pass over one thread's candidate list.
 *
 * The compressor proper. Scans every queued candidate into the two lists
 * (fill_list_chunk_to_compress_in_par), applies the clears, then resets the
 * per-chunk "already queued" flags and empties the list.
 *
 * The second list needs one extra step: the chunk's new second value must be
 * REGISTERED (find_or_set_ind_cst_val2) before the clear, so that
 * clear_int_st() can commit the resulting pointer and a later read returns the
 * right constant. Entries carrying the `T{}` marker are skipped — those chunks
 * are already compressed to their second value.
 *
 * @param thr     Rank whose candidate list is processed.
 * @param proxy2A Proxy of the write that triggered the pass; may be null. Used
 *                ONLY for the compression handshake and for re-queueing that
 *                chunk afterwards — never for sizing, since the list can hold
 *                chunks from other arrays.
 *
 * @note Under OpenMP the whole body is gated by
 *       `compression_mgr.start()`, which can decline: another thread is already
 *       compressing, and this pass returns having done nothing.
 */
template <typename T>
void Chunk2A<T>::make_block_compression(int thr, Proxy2A<T>* proxy2A)
{
    // NOTE: an earlier version of this fix pinned the ambient size ONCE here,
    // to proxy2A->chunk_size(). That was insufficient: this function
    // processes vec_tuple_to_compress, a list that can hold candidates from
    // MORE THAN ONE array (proxy2A is only "whichever write triggered this
    // pass", not necessarily the owner of every chunk below) -- confirmed via
    // a live repro that kept crashing after that fix, from this exact
    // function, with the ambient still wrong for some later chunk in the
    // list. Each ll_compt1/ll_compt2 entry instead carries its OWN owning
    // array's geometry (a ChunkElems, captured by push_to_compress_list() at
    // the original write that queued it) -- pin PER-CHUNK from that, inside
    // each loop below, right before the clear_int_st() call that eventually
    // reaches ChunkBase::free_values() (which routes by the ambient
    // thread_chunk_size(), not by the buffer being freed -- see
    // Proxy2::manage_full_first_value()'s comment in proxy2.h for the full
    // free-routing rationale). proxy2A itself is only used for the
    // compression_mgr start/finish handshake below, not for sizing.
    auto& vec_tuple_to_compress = manager_chunk2A::get_chunk2A_to_compress_potentially(thr);

    const auto sz = static_cast<int>(vec_tuple_to_compress.size());

    if (sz == 0)
        return;

    // The compression body runs in BOTH modes. Only the inter-thread
    // coordination (compression_mgr start/finish) is OpenMP-specific; the type-2
    // detection/registration (fill_list -> find_or_set_ind_cst_val2 ->
    // clear_int_st) is identical, just single-threaded when OpenMP is off.
    if constexpr (build_config::use_openmp)
    {
        if (proxy2A)
        {
            // Per-chunk exclusion (ChunkBase::steal_mark_): claim every
            // candidate BEFORE start(), whose heavy barrier publishes the
            // marks to the writers (its window then waits only for the
            // writers whose flag points to a marked chunk). Each chunk
            // is released as the pass is done with it (scanned and left, or
            // stolen), and all of them again at the end, whatever happened.
            // A flush in progress makes start() decline: ask first, before
            // marking -- see block_pass_precheck.h.
            if (block_pass_yields_to_flush<Proxy2A<T>, Chunk2A<T>>(CHUNK_PMP_MAX_THREADS))
                return;
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                    chunk_t->steal_mark();
            }
            const bool ret = proxy2A->compression_mgr.template start<Proxy2A<T>>();
            if (!ret)
            {
                for (int k = 0; k < sz; ++k)
                {
                    if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                        chunk_t->steal_unmark();
                }
                return;
            }
            // Check builds: the window is shut, commit the claims -- what the
            // store_value() guard tests (ChunkBase::steal_commit).
            if constexpr (build_config::any_check)
            {
                for (int k = 0; k < sz; ++k)
                {
                    if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                        chunk_t->steal_commit();
                }
            }
        }
    }

    // Now we're inside the compression phase. Log scope from here to the end.
    //  ScopedRingLog<EV_BLOCK_COMPRESS_BEGIN, EV_BLOCK_COMPRESS_END> block_log{static_cast<uint32_t>(sz) &
    //                                                                         0x00FFFFFFu};
    vec_tuple_chunkA2p_int2  ll_compt1;
    vec_tuple_chunkA2p_Tint2 ll_compt2;

    ll_compt1.reserve((sz / 50) + 1);
    ll_compt2.reserve((sz / 50) + 1);

    // `claimed`: the claim loop above runs only with a proxy. Without one
    // this is the resize path, outside any parallel region.
    fill_list_chunk_to_compress_in_par(thr, ll_compt1, ll_compt2, proxy2A != nullptr);

    for (auto& tup : ll_compt1)
    {
        const int num_chunk = std::get<1>(tup);
        const int num_tab   = std::get<2>(tup);

        auto* chunk_ptr = std::get<0>(tup);

        // Per-chunk geometry pin -- see this function's header comment.
        if constexpr (build_config::chunk_dynamic)
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());

        char* ptr_type_cst = &Proxy2A<T>::tab_type_cst_val[num_tab][num_chunk];

        // Long form in both modes: thr is 0 with OpenMP off. clear_int_st(5-arg)
        // is now available in non-OMP (step 1).
        const SecondValueRef<T> ptr_second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
        chunk_ptr->clear_int_st(0, ptr_type_cst, ptr_second_value, thr);
        chunk_ptr->steal_unmark();
    }

    for (const auto& tup : ll_compt2)
    {
        auto [chunk_ptr, second_v, num_chunk, num_tab, csize] = tup;

        // Per-chunk geometry pin -- see this function's header comment.
        if constexpr (build_config::chunk_dynamic)
            shrink_array::set_thread_chunk_size(csize.count());

        if constexpr (Base::is_not_bool)
        {
            if (second_v != T{})
            {
                char* ptr_type_cst = &Proxy2A<T>::tab_type_cst_val[num_tab][num_chunk];

                // Type-2 registration, identical in both modes. find_or_set records
                // the second value in the chunk's slot and returns the shared
                // array; clear_int_st installs it as the chunk's values pointer
                // (the CAS). thr is 0 with OpenMP off.
                const SecondValueRef<T> ptr_second_value =
                    Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);

                T* new_cst_val2_ptr =
                    chunk_ptr->tab_cst_val2_holder::find_or_set_ind_cst_val2(second_v, ptr_second_value);

                chunk_ptr->clear_int_st(1, ptr_type_cst, ptr_second_value, thr, new_cst_val2_ptr);
            }
        }
        else
        {
            // bool: no second-value pointer, ordering issue does not apply
            if (second_v)
            {
                chunk_ptr->tab_cst_val2_holder::find_or_set_ind_cst_val2();
            }

            char* ptr_type_cst = &Proxy2A<T>::tab_type_cst_val[num_tab][num_chunk];

            const SecondValueRef<T> ptr_second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            chunk_ptr->clear_int_st(1, ptr_type_cst, ptr_second_value, thr);
        }
        chunk_ptr->steal_unmark();
    }

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup       = vec_tuple_to_compress[k];
        const int   num_chunk = std::get<1>(tup);
        // Formerly `num_chunk != -1`: nothing sets -1 any more (purged entries
        // are erased from the lists), so a negative num_chunk is a corrupt entry
        // -- reported in check builds, skipped otherwise.
        if constexpr (build_config::any_check)
        {
            if (num_chunk < 0) [[unlikely]]
                INVARIANT_FAILED("Chunk2A block pass: candidate {} of {} has num_chunk={}", k, sz, num_chunk);
        }
        if (num_chunk >= 0)
        {
            const int num_tab = std::get<2>(tup);

            Proxy2A<T>::Base::reset_tab_inside_list_to_compress(num_tab, num_chunk);
        }
        // The pass is over for every candidate: no mark may outlive it
        // (idempotent for the ones released above).
        if (auto* const chunk_t = std::get<0>(tup))
            chunk_t->steal_unmark();
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);

    if (proxy2A)
    {
        if (proxy2A->is_not_cst_val())
        {
            proxy2A->set_info_on_chunk();
            // We are in a critical section: this guarantees that only one thread executes this part of the code at
            // a time, but another thread may be running add_to_compression_list at the same time
            proxy2A->add_to_compression_list(vec_tuple_to_compress);
        }

        if constexpr (build_config::use_openmp)
        {
            proxy2A->compression_mgr.template finish<Proxy2A<T>>();
        }
    }
}

/**
 * @brief Scans one chunk and compresses it in place if it is constant.
 *
 * The single-chunk counterpart of make_block_compression(): same two-hypothesis
 * scan, but the clear happens right here instead of being collected into a
 * list.
 *
 * @param thr_rank       Calling thread's rank (OMPT path only).
 * @param num_chunk      Index of this chunk in its array.
 * @param num_tab        Table number of the owning array.
 * @param apply          true: scan, and compress the chunk if constant;
 *                       false: nothing at all in an OpenMP build without OMPT
 *                       (the candidate lists' purge walks call it so: the
 *                       scan with no clear to show for it measured ~40 % of an
 *                       aarch64 profile), the scan alone elsewhere.
 * @param size           Geometry of the owning array.
 *
 * @note In an OpenMP build without OMPT the between-region pass is
 *       make_block_compression() (compress_chunk_to_compress); this is what
 *       the rechunk fold calls on a chunk of its own, with apply = true.
 *       Until Sept '26 it returned at once there whatever the flag, and a
 *       GNU build's fold counted every merged constant span as refused.
 */
template <typename T>
void Chunk2A<T>::compress([[maybe_unused]] int  thr_rank,
                          int                   num_chunk,
                          int                   num_tab,
                          [[maybe_unused]] bool apply,
                          ChunkElems            size)
{
    // In the GNU/GOMP build (use_openmp && no OMPT) NEITHER clear_int_st branch
    // at the bottom of this function is compiled in:
    //   - the `if constexpr (!use_openmp)` block is dead (use_openmp == true),
    //   - the `if constexpr (use_compression_with_ompt)` block is dead (no OMPT).
    // So the whole body would only SCAN every value (O(chunk_size) per chunk)
    // and call find_or_set_ind_cst_val2 to register a cst_val2 — a registration
    // that make_block_compression redoes itself (it is fully self-contained:
    // scan via fill_list_to_compress, its own find_or_set, then clear_int_st).
    // find_or_set is idempotent, so that registration is redundant and the
    // discarded new_cst_val2_ptr is only used by the OMPT clear. This scan was
    // ~40% of the aarch64 profile with zero compression effect: the lists'
    // purge walks (apply = false) skip it entirely. With apply = true -- the
    // rechunk fold, on a chunk no writer can reach -- the scan and the clear
    // run here as in every other build.
    if constexpr (build_config::use_openmp && chunk_config::not_use_compression_with_ompt)
    {
        if (!apply)
            return;
    }

    // Only a MATERIALISED chunk can be compressed: a queued entry can outlive
    // the chunk's materialisation (see Chunk2FastCompPolicy::compress). Testing
    // cst_val1 alone let a chunk already at cst_val2 through, whose scan then
    // read the shared second-value block and cleared it a second time.
    if (this->written())
    {
        // CHUNK_STATS: one compression candidate scanned. compress/compress_attempt
        // is the yield of the scan (how much of this work pays off).
        increm_chunk_compress_attempt<Chunk2A>();
        bool compact1 = false;
        bool compact2 = false;

        // new_cst_val2_ptr: result of find_or_set, the array clear_int_st
        // installs as the chunk's values pointer.
        T* new_cst_val2_ptr = nullptr;

        if constexpr (Base::is_not_bool)
        {
            // Against values[0], as fill_list_to_compress does, whatever
            // second value the chunk has registered: a chunk uniform at a
            // value other than its current one is compressed to THAT value,
            // which find_or_set registers and the clear commits. Scanning
            // against the registered value alone (until Sept '26) left such a
            // chunk materialised for good, until it passed through first_value.
            const SecondValueRef<T> ptr_second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            const in_t<T>           second_v         = Base::read_value_scan(0);
            const auto              scan             = Base::scan_constants(first_value, second_v, size.count());
            compact1                                 = scan.first;
            compact2                                 = scan.second;
            if (compact2)
                new_cst_val2_ptr = tab_cst_val2_holder::find_or_set_ind_cst_val2(second_v, ptr_second_value);
        }
        else
        {
            const auto scan = Base::scan_constants(static_cast<T>(false), static_cast<T>(true), size.count());
            compact1        = scan.first;
            compact2        = scan.second;
            if (compact2)
                tab_cst_val2_holder::find_or_set_ind_cst_val2();
        }

        // Serial: clears here, outside any callback. OMPT: clears only from the
        // callback. Both go through the long clear_int_st(): the type-2 clear
        // MUST commit new_cst_val2_ptr first, or init_values_cst_val() points
        // the chunk at a second-value slot that was never installed (nullptr)
        // -- the serial path used the short two-argument form and the first
        // read of such a chunk after a rechunk_to() dereferenced null
        // (rechunk_validation_chunk2a_dyn, scale-64-to-tgt, serial build).
        // The serial build used to clear on !apply (the flag was named
        // is_inside_ompt and the serial callers passed false), and the
        // OpenMP build without OMPT never: every caller now says what it
        // wants, so every build reads it the same way.
        if (apply)
        {
            const SecondValueRef<T> ptr_second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            char*                   ptr_type_cst     = &Proxy2A<T>::tab_type_cst_val[num_tab][num_chunk];
            if (compact1)
                clear_int_st(0, ptr_type_cst, ptr_second_value, thr_rank);
            if (compact2)
                clear_int_st(1, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
        }
    }
}

/**
 * @brief Compresses every thread's candidate chunks, opening a parallel region if needed.
 *
 * Entry point used between parallel phases. Reuses the current team when
 * already inside a region, and opens its own otherwise. The OMPT callbacks are
 * disarmed for the duration, so the region this opens does not re-enter the
 * compression machinery through its own parallel-begin/end. Without OMPT the
 * per-chunk compress() is a no-op, and the pass is make_block_compression()
 * over every rank instead (see below).
 *
 * @param apply          Forwarded to compress().
 */
template <typename T>
void Chunk2A<T>::compress_sweep_rank(int thr, bool apply)
{
    if constexpr (build_config::use_openmp && chunk_config::not_use_compression_with_ompt)
    {
        // An OpenMP build without OMPT had no between-region compressor for
        // chunk2a until Sept '26: compress() returns at once there, so a
        // candidate waited until its rank's list reached the in-loop
        // threshold, and what the last phases of a run left constant was
        // never compressed (chunk2a_1024 --mode perf, 6 threads: 1 GB of
        // buffers more returned to the pool with this pass, -3 % RSS, time
        // within noise). Between two regions no writer is live, so the
        // writers' pass runs without its handshake (nullptr proxy), each
        // rank on its own list -- the lists are per rank, the second-value
        // registry takes its mutex, the in-loop passes already run
        // concurrently -- and the ranks beyond the team's size serially
        // (compress_sweep_mops_up), as Chunk2's pass does.
        if (apply)
        {
            make_block_compression(thr, nullptr);
            return;
        }
    }
    compress_chunk_to_compress_in_par(thr, apply);
}

template <typename T>
void Chunk2A<T>::compress_chunk_to_compress(bool apply)
{
    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        // Inside a region: this rank's own list, and nothing else. Opening
        // the region below from inside one (as this did, with no `else`) ran
        // a serialised copy of the pass on every thread of the team, and the
        // mop-up swept every rank's list from every thread at once.
        compress_chunk_to_compress_in_par(CHUNK_PMP_THREAD_RANK, apply);
    }
    else
    {
        int team = 0;
#pragma omp parallel
        {
            const int thr = CHUNK_PMP_THREAD_RANK;
            if (thr == 0)
                team = CHUNK_PMP_NUM_THREADS;
            compress_sweep_rank(thr, apply);
        }
        if (compress_sweep_mops_up(apply))
            for (int thr = team; thr < CHUNK_PMP_RANK_BOUND; ++thr)
                compress_sweep_rank(thr, apply);
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
}

/**
 * @brief Is there any chunk waiting to be compressed?
 *
 * Cheap gate for the callers that would otherwise open a parallel region for
 * nothing.
 *
 * @param max_thr Number of per-thread lists to examine (OpenMP builds only).
 * @return True as soon as one list is non-empty.
 */
template <typename T>
bool Chunk2A<T>::have_chunk_to_compress([[maybe_unused]] int max_thr)
{
    const bool is_empty = is_vec_tuple_to_compress_potentially_empty_thr();
    if (is_empty)
    {
        return false;
    }

    if constexpr (build_config::use_openmp)
    {
        for (int thr = 0; thr < max_thr; ++thr)
        {
            const auto& chunk_t      = get_vec_tuple_to_compress_potentially_thr(thr);
            const bool  is_empty_thr = chunk_t.empty();
            if (!is_empty_thr)
                return true;
        }
    }
    else
    {
        if (!is_empty)
            return true;
    }

    return false;
}

/**
 * @brief Compresses one thread's candidate chunks, from inside a parallel region.
 *
 * Walks the rank's list and calls compress() on each entry, using the geometry
 * each entry carries rather than any ambient size.
 *
 * @param thr            Rank whose list is processed.
 * @param apply          Forwarded to compress().
 *
 * @warning The "already queued" flag reset and the list clear MUST stay gated on
 *          the same condition. Clearing the flag while leaving the entry in the
 *          list lets a later write queue the same chunk twice, and the second
 *          clear then hits an already-constant chunk. That is why the flag reset
 *          sits under the same `use_compression_with_ompt` condition as the
 *          clear at the end.
 */
template <typename T>
void Chunk2A<T>::compress_chunk_to_compress_in_par([[maybe_unused]] int thr, bool apply)
{
    auto& vec_tup = manager_chunk2A::get_chunk2A_to_compress_potentially(thr);

    if constexpr (build_config::use_openmp)
    {
        // Geometry: the scan inside compress() reads csize explicitly, but the
        // clear it triggers reaches free_block_to_pool(), which routes by the
        // ambient thread_chunk_size() (reuse list AND allocator). This list
        // mixes chunks of every array this rank wrote -- an array and the tmp
        // rechunk_to() builds from it, at two sizes -- so pin each chunk to its
        // own csize and restore afterwards, exactly as make_block_compression()
        // does. Same defect as Chunk2FastCompPolicy::
        // compress_chunk_to_compress_in_par_impl (rechunk_validation PRE-FAIL
        // on clang/OMPT: wrong-width blocks popped from the reuse list).
        [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();

        for (const auto& tup : vec_tup)
        {
            auto* const      chunk_t   = std::get<0>(tup);
            const int        num_chunk = std::get<1>(tup);
            const int        num_tab   = std::get<2>(tup);
            const ChunkElems csize     = std::get<3>(tup);

            if (chunk_t)
            {
                if constexpr (build_config::chunk_dynamic)
                    set_thread_chunk_size(csize.count());

                chunk_t->compress(thr, num_chunk, num_tab, apply, csize);
                // DEDUP FLAG / LIST must be reset TOGETHER. The is_inside flag is
                // the CAS gate in add_to_compression_list (0->1 admits a chunk).
                // Resetting it to 0 here while leaving the entry physically in
                // vec_tup (which happens when vec_tup.clear() below is compiled
                // out — i.e. in every non-OMPT build) desyncs the two: a later
                // write CAS-flips 0->1 and appends the SAME chunk a second time,
                // and make_block_compression then double-clears it (2nd
                // clear_int_st hits an already-cst_val1 chunk ->
                // check_not_constant_before_clear fires: Values == Cst1).
                // So gate the flag reset on the SAME condition as the list clear.
                if constexpr (chunk_config::use_compression_with_ompt)
                    Proxy2A<T>::set_tab_is_inside_chunk_list_to_compress(num_tab, num_chunk, 0);
            }
        }

        if constexpr (build_config::chunk_dynamic)
            set_thread_chunk_size(saved_sz);
    }
    else
    {
        for (auto& tup : vec_tup)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);

            // Per-entry geometry, exactly like the OpenMP branch above — the
            // former call relied on the (now removed) `= __chunk_size` default,
            // which under-scanned dynamic chunks (see chunk2A_std.h).
            std::get<0>(tup)->compress(0, num_chunk, num_tab, apply, std::get<3>(tup));
            Proxy2A<T>::set_tab_is_inside_chunk_list_to_compress(num_tab, num_chunk, 0);
        }
    }

    if constexpr (chunk_config::use_compression_with_ompt)
    {
        // clear() keeps the capacity: the list is already reserved for the
        // next pass, at the size this one reached.
        vec_tup.clear();
    }
}

/**
 * @brief Scans one thread's candidate list into the two compression lists.
 *
 * Forwards each live entry to fill_list_to_compress(), passing the geometry the
 * entry carries. Null chunk pointers are the batch sentinels and are skipped.
 *
 * @param thr       Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to one other value.
 * @param claimed   Whether the caller claimed (steal-marked) the candidates
 *                  first; forwarded to fill_list_to_compress().
 */
template <typename T>
void Chunk2A<T>::fill_list_chunk_to_compress_in_par([[maybe_unused]] int      thr,
                                                    vec_tuple_chunkA2p_int2&  ll_compt1,
                                                    vec_tuple_chunkA2p_Tint2& ll_compt2,
                                                    [[maybe_unused]] bool     claimed)
{
    auto& vec_tup = manager_chunk2A::get_chunk2A_to_compress_potentially(thr);

    const int sz = vec_tup.size();

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tup[k];

        auto* const chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);
            // Per-entry geometry (see chunk2A_std.h): in dynamic builds each
            // queued chunk carries the chunk size of its owning array.
            const ChunkElems csize = std::get<3>(tup);

            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress(ll_compt1, ll_compt2, num_chunk, num_tab, csize, claimed);
            // Not selected: nothing will be stolen, the writers may have it
            // back now rather than at the end of the pass.
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
                chunk_t->steal_unmark();
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk2AOmpPolicy<T, Derived, true>: materialise and clear, OpenMP mode.
// ---------------------------------------------------------------------------
/**
 * @brief Materialise-side exclusion against the compressor's steal.
 *
 * make_block_compression() CAS-steals a chunk's `values` and
 * free_block_to_pool()s it; allocating and filling a block for that chunk
 * while that is in flight raced the free on the same recycled block (TSan,
 * July '26, fatal on the dense slab). The wait used to be on the narrow
 * GLOBAL condition, is_hard_compression_inflight() -- not
 * is_compression_active(), which also counts slow-path threads merely
 * draining flush_deferred and deadlocks from a flusher's write body -- and
 * that was invisible while in-loop compressions yielded 98% of the time.
 * Once they ran, every materialisation that fell in a pass spun for the
 * whole scan: 72% of perform_core_assignment<double> on the load of
 * stealing_inflight_, 10% of the perf bench at 6 threads on x86.
 *
 * With the per-chunk marks the compressor only ever steals a chunk it has
 * MARKED, and it holds the mark from before its start() to after the CAS: so
 * this chunk's own mark is the exact condition. A writer on the fast path
 * never reaches here with the chunk marked (the gate stopped it), a flush
 * replay never overlaps a stealer (flush_deferred waits stealing_inflight_
 * out and blocks start() while it replays), so the wait is a no-op in
 * practice and the load is on the header line, not on the compressors'
 * shared counter. Builds without the marks keep the global condition.
 *
 * @param chunk The chunk about to be materialised.
 */
template <typename T, typename ChunkT>
void wait_before_materialise(const ChunkT& chunk) noexcept
{
    if constexpr (chunk_config::use_chunk_steal_mark)
    {
        while (chunk.steal_marked())
            CPU_PAUSE();
    }
    else
    {
        while (CompressionManager<T, ChunkT>::is_hard_compression_inflight())
            CPU_PAUSE();
    }
}

/**
 * @brief Materialises this chunk: allocate a buffer, fill it, publish it by CAS.
 *
 * The buffer is filled with either first_value or the chunk's second value
 * BEFORE it is published, and publication is a single CAS against the constant
 * pointer the chunk currently holds. A reader therefore sees the old constant
 * or the finished buffer, never a partially built one. Losing the CAS means
 * another thread materialised first; apply_cas_or_recup() recycles our buffer.
 *
 * @param need_set_second_value True to pre-fill with the second value, false for first_value.
 * @param ptr_second_value      The chunk's second value; read only when the flag is set.
 * @param thr_rank              Rank whose allocator provides the buffer.
 *
 * @warning Spins until no HARD compression is in flight. The wait is on that
 *          narrow condition on purpose: waiting on the broad
 *          "compression active" deadlocks, because this can run from a
 *          flusher's write body while a slow-path thread waits on the lock that
 *          flusher holds.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::create_value(bool              need_set_second_value,
                                                         SecondValueRef<T> ptr_second_value,
                                                         int               thr_rank)
{
    if constexpr (is_not_bool)
    {
        // Piste 1 — exclude create_value from a concurrent compression.
        // make_block_compression() (another thread) CAS-steals this chunk's
        // `values` and free_block_to_pool()s it; if we allocate+memset a block
        // now we race that free on the same recycled block (TSan:
        // chunk_base_inline.h:78 <-> chunk_base.h:369, fatal on the dense slab).
        // Mirror the write path: wait until no compression may steal THIS
        // chunk -- see wait_before_materialise() for the condition, and why
        // the broad "any driver or hard compression" predicate would deadlock
        // here.
        wait_before_materialise<T>(derived());

        // Lambda to select the right initialization function
        const auto init_func = [need_set_second_value, ptr_second_value](T* ptr)
        {
            if (!need_set_second_value)
            {
                Base::init_block(ptr);
            }
            else
            {
                // Object-level fill: safe for double, complex, float, etc.
                // Compilers will optimize this into SIMD or optimized assembly.
                std::fill_n(ptr, shrink_array::alloc_chunk_count(), ptr_second_value.get());
            }
        };

        T* values_loc = Base::allocate_chunk_internal(thr_rank, init_func);

        // Reuse the unified CAS/Recycle logic from Chunk2Base.
        Base::apply_cas_or_recup(values_loc, thr_rank,
                                 [this, need_set_second_value](T* cst_ptr)
                                 {
                                     // Single CAS operation targeting the specific constant pointer.
                                     return this->compare_and_swap(cst_ptr, need_set_second_value);
                                 });
    }

    else
    {
        static_assert(dependent_false_v<T>,
                      "create_value(..., second value) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(), simplified by bool having no
 * per-chunk second value: the flag doubles as the fill value.
 *
 * @param need_set_second_value Fill value — true for the second constant, false for the first.
 * @param thr_rank              Rank whose allocator provides the buffer.
 * @note No compression-inflight wait here, unlike create_value().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::create_value_bool(bool need_set_second_value, int thr_rank)
{
    if constexpr (is_bool)
    {
        // Same exclusion as create_value(), which this twin was missing: make_block_compression() is a single function
        // for every T, so a bool chunk's `values` is CAS-stolen and free_block_to_pool()d exactly like any other.
        // Allocating and filling a block here while that is in flight races the free on the same recycled block -- see
        // wait_before_materialise() for the condition.
        wait_before_materialise<T>(derived());

        const auto init_func = [need_set_second_value](bool* ptr)
        {
            // need_set_second_value is false (0) or true (1),
            // matching exactly the required initialization state.
            std::fill_n(ptr, shrink_array::alloc_chunk_count(), need_set_second_value);
        };

        T* values_loc = Base::allocate_chunk_internal(thr_rank, init_func);

        Base::apply_cas_or_recup(values_loc, thr_rank,
                                 [this, need_set_second_value](bool* cst_ptr)
                                 {
                                     // Single CAS operation targeting the specific constant pointer.
                                     return this->compare_and_swap(cst_ptr, need_set_second_value);
                                 });
    }
    else
    {
        static_assert(dependent_false_v<T>,
                      "create_value_bool() is for bool elements: use create_value(..., second value)");
    }
}

/// @brief create_value() on the calling thread's rank.
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::create_value(bool need_set_second_value, SecondValueRef<T> ptr_second_value)
{
    create_value(need_set_second_value, ptr_second_value, get_thr_rank_fast());
}

/// @brief create_value_bool() on the calling thread's rank.
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::create_value_bool(bool need_set_second_value)
{
    create_value_bool(need_set_second_value, get_thr_rank_fast());
}

// ---------------------------------------------------------------------------
// Chunk2AOmpPolicy<T, Derived, false>: materialise and clear, serial mode.
// ---------------------------------------------------------------------------

/**
 * @brief Materialises this chunk (serial build): allocate, fill, assign.
 *
 * No CAS and no inflight wait — with nothing else running, the buffer is simply
 * assigned to `values`.
 *
 * @param need_set_second_value True to pre-fill with the second value, false for first_value.
 * @param ptr_second_value      The chunk's second value; read only when the flag is set.
 */
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::create_value(bool need_set_second_value, SecondValueRef<T> ptr_second_value)
{
    // Lambda to select the right initialization function
    const auto init_func = [need_set_second_value, ptr_second_value](T* ptr)
    {
        if (!need_set_second_value)
        {
            Base::init_block(ptr);
        }
        else
        {
            for (unsigned int i = 0; i < alloc_chunk_count(); ++i)
            {
                ptr[i] = ptr_second_value.get();
            }
        }
    };

    values = Base::allocate_chunk_internal(0, init_func);
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 * @param need_set_second_value Fill value — true for the second constant, false for the first.
 */
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::create_value_bool(bool need_set_second_value)
{
    // bool's second value is always `true`, so this IS create_value() at that
    // value. The body used to be written out again with the fill hand-rolled:
    // its "init to false" loop is what Base::init_block() does (a memset for any
    // arithmetic T, bool included), and its "init to true" loop is the
    // second-value fill with ptr_second_value.get() == true.
    static T k_second_value = static_cast<T>(true); // a slot the handle can point at; never written
    create_value(need_set_second_value, SecondValueRef<T>{&k_second_value});
}

/**
 * @brief Compresses this chunk back to a constant, if it is materialised.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot.
 * @param thr_rank         Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::clear(char* ptr_type_cst, SecondValueRef<T> ptr_second_value, int thr_rank)
{
    Base::clear_impl(std::nullopt, thr_rank, ptr_type_cst, ptr_second_value);
}

/**
 * @brief Compresses this chunk to the FIRST constant and forgets its second value.
 *
 * Serial-build variant used at teardown: releases the buffer, resets the
 * chunk's second value to zero, and points it back at the first constant.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot, reset to `T{}`.
 */
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::clear(char* ptr_type_cst, SecondValueRef<T> ptr_second_value)
{
    if (this->written())
    {
        Base::free_values(values);

        ptr_second_value.reset();
        Derived::set_cst_val_to_cst_val1(ptr_type_cst);
        derived().init_values_cst_val(ptr_type_cst, ptr_second_value);
    }
}

/**
 * @brief Unconditional twin of clear(): compresses without testing `written()`.
 *
 * For callers that have already established the chunk is materialised; the test
 * is kept as a debug-only assertion.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot, reset to `T{}`.
 */
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::clear_st(char* ptr_type_cst, SecondValueRef<T> ptr_second_value)
{
    // Check condition only if debug mode is active at compile-time
    Base::debug_check_written();

    // Core logic: executed always in Release, and only if written() is true in Debug
    Base::free_values(values);

    ptr_second_value.reset();
    Derived::set_cst_val_to_cst_val1(ptr_type_cst);
    derived().init_values_cst_val(ptr_type_cst, ptr_second_value);
}

/**
 * @brief Releases this chunk's buffer and points it back at a constant.
 *
 * The single point every compression path funnels through, in both modes and
 * every chunk variant — which is why the "block compressed" statistic is
 * counted here and nowhere else.
 *
 * @param num_compress_value 0 for the first constant, otherwise the second.
 * @param ptr_type_cst       Slot of the per-chunk type table to update.
 * @param ptr_second_value   The chunk's second-value slot.
 * @param thr_rank           Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::block_clear(int                  num_compress_value,
                                                        char*                ptr_type_cst,
                                                        SecondValueRef<T>    ptr_second_value,
                                                        [[maybe_unused]] int thr_rank,
                                                        T*                   cst2_ptr)
{
    // CHUNK_STATS: one materialised block released by a successful compression.
    // Counted here because block_clear is the single point every variant and
    // both OMP/non-OMP modes funnel through (the old counter only fired on the
    // plain-chunk sweep, so chunk2/chunk2a always reported 0).
    increm_chunk_compress<Derived>();
    // Bytes released, as Chunk2 counts them: with bytes_alloc and bytes_reused
    // this gives the buffers still materialised at any point of a run.
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));
    Base::block_clear_common(num_compress_value, 0, thr_rank, ptr_type_cst, ptr_second_value, cst2_ptr);
}

/**
 * @brief Compresses this chunk, committing its second-value pointer first.
 *
 * The full compression step. When compressing to the second constant it
 * publishes the registered pointer BEFORE releasing the buffer: block_clear()
 * reads that pointer while doing so, and would otherwise pick up a stale one.
 *
 * @param num_compress_value 0 for the first constant, 1 for the second.
 * @param ptr_type_cst       Slot of the per-chunk type table to update.
 * @param ptr_second_value   The chunk's second-value slot.
 * @param thr_rank           Rank whose pool receives the buffer.
 * @param new_cst_val2_ptr   Pointer returned by find_or_set_ind_cst_val2();
 *                           required when @p num_compress_value is 1.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2AOmpPolicy<T, Derived, Enabled>::clear_int_st(int               num_compress_value,
                                                         char*             ptr_type_cst,
                                                         SecondValueRef<T> ptr_second_value,
                                                         int               thr_rank,
                                                         T*                new_cst_val2_ptr)
{
    Base::check_not_constant_before_clear(num_compress_value, std::nullopt);

    // The array to install for a type-2 compression: the caller's, or the one
    // registered for the chunk's second value (the bool path, which passes
    // none; there it is the single shared array).
    if (num_compress_value == 1 && new_cst_val2_ptr == nullptr)
        new_cst_val2_ptr = Derived::tab_cst_val2_holder::find_cst_val2_ptr(ptr_second_value.get());

    block_clear(num_compress_value, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
}

/// @brief Serial block_clear(): frees the buffer outright instead of parking it
///        in a rank pool, and points the chunk at a constant.
/// @param num_compress_value 0 for the first constant, otherwise the second.
/// @param ptr_type_cst       Slot of the per-chunk type table to update.
/// @param ptr_second_value   The chunk's second-value slot.
/// @param cst2_ptr           Shared array of the second constant (type 2).
/// @note The rank argument is unused: there is no rank pool serially.
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::block_clear(int               num_compress_value,
                                                      char*             ptr_type_cst,
                                                      SecondValueRef<T> ptr_second_value,
                                                      int,
                                                      T* cst2_ptr)
{
    // Same refusal as try_mark_cst_and_get_old() on the OpenMP side: a chunk
    // whose values already point at a constant has no buffer to release, and
    // handing that pointer to the allocator corrupts it (or faults: a
    // second-value identifier is not backed by a page). The type table is left as it is -- it
    // describes the constant in place.
    if (values == Base::get_cst_val1() || Derived::tab_cst_val2_holder::is_cst_val2_ptr(values)) [[unlikely]]
        return;
    increm_chunk_compress<Derived>();
    Base::free_values(values);
    Derived::set_cst_val_to_cst_val(num_compress_value, ptr_type_cst);
    // Serial: no CAS, the values pointer is set outright to the constant --
    // the caller's array for a type 2, from the tables otherwise.
    if (num_compress_value == 1 && cst2_ptr != nullptr)
        values = cst2_ptr;
    else
        derived().init_values_cst_val(ptr_type_cst, ptr_second_value);
}

/**
 * @brief Serial twin of the five-argument clear_int_st(): same commit-then-
 *        block_clear order, so the sequential compression can register a
 *        cst_val2 exactly like the OpenMP path. thr_rank is 0 here.
 * @copydetails Chunk2AOmpPolicy::clear_int_st
 */
template <typename T, typename Derived>
void Chunk2AOmpPolicy<T, Derived, false>::clear_int_st(int               num_compress_value,
                                                       char*             ptr_type_cst,
                                                       SecondValueRef<T> ptr_second_value,
                                                       int               thr_rank,
                                                       T*                new_cst_val2_ptr)
{
    Base::check_not_constant_before_clear(num_compress_value, std::nullopt);
    if (num_compress_value == 1 && new_cst_val2_ptr == nullptr)
        new_cst_val2_ptr = Derived::tab_cst_val2_holder::find_cst_val2_ptr(ptr_second_value.get());
    block_clear(num_compress_value, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
}

// clear_st(int num_compress_value, char* ptr_type_cst, SecondValueRef<T> ptr_second_value) --
// the entry point that supplies the calling thread's rank -- is defined
// in-class on the OpenMP Chunk2AOmpPolicy specialisation.

} // namespace shrink_array