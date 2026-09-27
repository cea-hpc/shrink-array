#pragma once

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

/**
 * @file array_chunk_dirty_bitset.h
 * @brief The per-chunk "dirty" bitset — an OpenMP-only concern, isolated in its
 *        own mixin so it disappears entirely from a serial build.
 *
 * @details The bitset is the coarse pre-filter of the deferred-op read gate: the
 *          enqueue path sets a chunk's bit (mark_chunk_dirty) and
 *          ArrayChunkBase::sync_chunk tests it before paying for the precise
 *          per-chunk gate. Both the marking and the testing exist only under
 *          OpenMP — in a serial build there is no deferred path, sync_chunk
 *          compiles to nothing and the flush loops are behind
 *          `if constexpr (build_config::use_openmp)`.
 *
 *          The bitset used to live in ArrayChunkBase, allocated unconditionally:
 *          a serial array still paid one heap allocation plus 128 bytes per 64
 *          chunks that nothing ever read. Splitting it into
 *          DirtyBitsetSupport<Derived, use_openmp> keeps the OpenMP version
 *          byte-for-byte identical and makes the serial version an empty class
 *          (no member, no allocation, zero bytes via the empty-base
 *          optimisation) whose verbs are no-ops the optimiser deletes.
 *
 * @see ArrayChunkBase::sync_chunk — the sole hot-path reader of the bits
 */

#include <atomic>
#include <cstddef> // std::size_t
#include <cstdint>
#include <memory>
#include <algorithm> // std::min
#include <utility>   // std::swap

#include "chunk/config.h"

namespace shrink_array
{

/**
 * @brief One 64-bit word of the per-chunk dirty bitset, padded to its own cache line.
 *
 * @details The bitset is written concurrently by the deferred-enqueue path
 *          (mark_chunk_dirty) from every thread. Packing the words together
 *          would make neighbouring chunks share a cache line and turn each mark
 *          into false sharing; alignas(128) gives each word a line of its own
 *          (128 bytes covers the 128-byte sectored lines of recent server CPUs).
 */
struct alignas(128) PaddedDirtyWord
{
    std::atomic<uint64_t> word{0};
};

/**
 * @brief Per-chunk dirty bitset, present only when the build has OpenMP.
 *
 * @tparam Derived the array core that owns it, ArrayChunkCore<chunk_type>
 *         (needed by clear_all_dirty_bitsets to walk the per-type registry,
 *         whose keys are addresses of that core).
 * @tparam Enabled defaulted to build_config::use_openmp; never pass it
 *         explicitly outside of tests — the point of the parameter is that the
 *         two specialisations below are selected by the build mode.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
class DirtyBitsetSupport;

// ── OpenMP build: the real bitset ────────────────────────────────────────────
/**
 * @brief OpenMP specialisation: owns the bit array and the dirty-bit verbs.
 * @copydetails DirtyBitsetSupport
 */
template <typename Derived>
class DirtyBitsetSupport<Derived, true>
{
    // One bit per CHUNK, whatever the chunk size: the 64 below is the width of
    // a word (64 chunks per word), not a chunk geometry. chunk_idx >> 6 and
    // chunk_idx & 63 read like the decalage / modulo of a 64-element chunk,
    // which they are not.
    static constexpr int kChunksPerWord = 64;                 ///< Bits in a uint64_t word.
    static constexpr int kWordShift     = 6;                  ///< log2(kChunksPerWord).
    static constexpr int kBitInWordMask = kChunksPerWord - 1; ///< chunk_idx -> bit within its word.
    static_assert(kChunksPerWord == 1 << kWordShift, "kWordShift must be log2(kChunksPerWord)");

  public:
    /**
     * @brief Allocate a fresh, all-clear bitset for @p nb_chunk chunks.
     * @param nb_chunk number of chunks the owning array now has
     * @note Drops whatever bits were set: the creation / full-resize paths call
     *       this exactly where the chunk list itself is rebuilt.
     */
    void alloc_dirty_bitset(int nb_chunk)
    {
        nb_words_ = words_for(nb_chunk);
        // + 1: the "some bit was set" flag (any_marked_word), after the bits.
        bitset_ = std::make_unique<PaddedDirtyWord[]>(static_cast<std::size_t>(nb_words_) + 1);
    }

    /**
     * @brief Reallocates and copies the dirty bitset during resize.
     * Centralized logic to prevent duplication and buffer overflows.
     * @param old_nb_chunk chunk count the current bitset was sized for
     * @param new_nb_chunk chunk count to size the new bitset for
     */
    void reallocate_dirty_bitset(int old_nb_chunk, int new_nb_chunk)
    {
        if (new_nb_chunk == old_nb_chunk)
            return;

        const int old_nb_words = words_for(old_nb_chunk);
        const int new_nb_words = words_for(new_nb_chunk);

        auto new_bitset = std::make_unique<PaddedDirtyWord[]>(static_cast<std::size_t>(new_nb_words) + 1);

        if (bitset_)
        {
            const int words_to_copy = std::min(old_nb_words, new_nb_words);
            for (int i = 0; i < words_to_copy; ++i)
            {
                const uint64_t val = bitset_[i].word.load(std::memory_order_relaxed);
                new_bitset[i].word.store(val, std::memory_order_relaxed);
            }
            // Carry the "some bit was set" flag (conservatively: it may now
            // cover only bits that were cut off, which costs one useless clear).
            new_bitset[new_nb_words].word.store(bitset_[old_nb_words].word.load(std::memory_order_relaxed),
                                                std::memory_order_relaxed);
        }
        bitset_   = std::move(new_bitset);
        nb_words_ = new_nb_words;
    }

    /**
     * @brief Zero the whole dirty bitset.
     * @note One relaxed store per word — safe to call after an OMP barrier
     *       (serial or single region), not while writers may still be marking.
     *       Not a memset: the words are std::atomic, which has no trivial
     *       copy-assignment (-Wclass-memaccess), and writing an atomic object
     *       through a non-atomic access is undefined behaviour even at a
     *       quiescent point. A relaxed store is the defined way to reset it and
     *       costs the same here: each word sits on its own 128-byte line, so a
     *       memset would touch exactly the same lines.
     */
    void clear_dirty_bitset() noexcept
    {
        if (!bitset_)
            return;
        // Nothing marked since the last clear: the words are already zero.
        // This is the common case -- only the deferred-enqueue path marks --
        // and it matters: clear_all_dirty_bitsets() runs at every phase end of
        // a non-OMPT build (finalize_parallel_phase_after_flush), serially, and
        // rewriting one 128-byte line per 64 chunks of every array took ~21 s
        // of a ~30 s shrink_bench_chunk2a_256 run on AMD Milan (GNU, 64 threads).
        if (any_marked_word().exchange(0, std::memory_order_acq_rel) == 0)
            return;
        PaddedDirtyWord* const words = bitset_.get();
        for (int i = 0; i < nb_words_; ++i)
            words[i].word.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Mark a chunk as having received a deferred operation.
     * @param chunk_idx The global index of the chunk (not of an element: one
     *        bit per chunk, whatever the chunk size).
     * @note Concurrent: an acq_rel fetch_or, called from the deferred-enqueue
     *       path on any thread.
     */
    void mark_chunk_dirty(int chunk_idx) noexcept
    {
        const int      word_idx = chunk_idx >> kWordShift;
        const uint64_t bit_mask = 1ULL << (chunk_idx & kBitInWordMask);
        bitset_[word_idx].word.fetch_or(bit_mask, std::memory_order_acq_rel);
        // Raise the "some bit was set" flag (read first: raised once per phase,
        // then every later mark is a plain load).
        auto& any = any_marked_word();
        if (any.load(std::memory_order_relaxed) == 0)
            any.store(1, std::memory_order_release);
    }

    /**
     * @brief Checks if a chunk is marked as dirty.
     * @param chunk_idx The global index of the chunk (not of an element).
     * @return true if a deferred operation was queued for this chunk during the
     *         current phase (the bit is sticky until clear_dirty_bitset()).
     */
    [[nodiscard]] bool is_chunk_dirty(int chunk_idx) const noexcept
    {
        const int      word_idx = chunk_idx >> kWordShift;
        const uint64_t bit_mask = 1ULL << (chunk_idx & kBitInWordMask);

        // Use acquire to synchronize with previous mark_chunk_safe/dirty calls
        return (bitset_[word_idx].word.load(std::memory_order_acquire) & bit_mask) != 0;
    }

    /**
     * @brief Exchange bitsets with another array, as part of an O(1) container swap.
     * @param other the array to exchange the bitset with
     */
    void swap_dirty_bitset(DirtyBitsetSupport& other) noexcept
    {
        using std::swap;
        swap(bitset_, other.bitset_);
        swap(nb_words_, other.nb_words_);
    }

    /**
     * @brief Clear the dirty bitset of EVERY live array of this variant.
     * @details Walks the per-type registry of arrays and calls
     *          clear_dirty_bitset() on each. Quiescent point only (the reset is
     *          not synchronised against concurrent marks).
     */
    static void clear_all_dirty_bitsets()
    {
        using proxy_type = Derived::proxy_type;
        proxy_type::share_info_array.for_each_key(
            [](const void* key)
            {
                if (key)
                {
                    auto* arr = static_cast<Derived*>(const_cast<void*>(key));
                    arr->clear_dirty_bitset();
                }
            });
    }

  private:
    /// @brief The "some bit was set since the last clear" flag: the padded word
    ///        right after the bits (so it moves and swaps with them).
    /// @pre bitset_ is allocated.
    [[nodiscard]] std::atomic<uint64_t>& any_marked_word() const noexcept
    {
        return bitset_[static_cast<std::size_t>(nb_words_)].word;
    }

    /** @brief Number of 64-bit words needed to hold @p nb_chunk bits (one per chunk). */
    [[nodiscard]] static constexpr int words_for(int nb_chunk) noexcept
    {
        return (nb_chunk + kChunksPerWord - 1) / kChunksPerWord;
    }

    std::unique_ptr<PaddedDirtyWord[]> bitset_;      ///< One bit per chunk, one cache line per 64 chunks.
    int                                nb_words_{0}; ///< Length of @ref bitset_, so clearing needs no chunk count.
};

// ── Serial build: nothing at all ─────────────────────────────────────────────
/**
 * @brief Serial specialisation: an empty class with the same verbs, all inert.
 *
 * @details Nothing in a serial build ever queries a dirty bit — sync_chunk is
 *          behind `if constexpr (build_config::use_openmp)` and every
 *          flush_pending_deferred_ops() body is too — so is_chunk_dirty() can
 *          answer false unconditionally and the array carries no bitset at all.
 *          The verbs are kept (rather than compiled out at every call site) so
 *          the creation, resize, swap and re-chunk paths stay mode-agnostic.
 */
template <typename Derived>
class DirtyBitsetSupport<Derived, false>
{
  public:
    /// @name No-op verbs (no bitset in this mode)
    /// @{
    void alloc_dirty_bitset(int /*nb_chunk*/) noexcept {}
    void reallocate_dirty_bitset(int /*old_nb_chunk*/, int /*new_nb_chunk*/) noexcept {}
    void clear_dirty_bitset() noexcept {}
    void mark_chunk_dirty(int /*chunk_idx*/) noexcept {}
    /// @brief Never dirty: no deferred write can exist in this mode.
    [[nodiscard]] bool is_chunk_dirty(int /*chunk_idx*/) const noexcept
    {
        return false;
    }
    void        swap_dirty_bitset(DirtyBitsetSupport& /*other*/) noexcept {}
    static void clear_all_dirty_bitsets() noexcept {}
    /// @}
};

} // namespace shrink_array
