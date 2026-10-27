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
#include "chunk/internal/array_chunk_dirty_bitset.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::DirtyBitsetSupport<Derived, true>
// ---------------------------------------------------------------------------

template <typename Derived>
void DirtyBitsetSupport<Derived, true>::mark_chunk_dirty(int chunk_idx) noexcept
{
    const int      word_idx = chunk_idx >> word_shift;
    const uint64_t bit_mask = 1ULL << (chunk_idx & bit_in_word_mask);
    auto&          word     = bitset_[word_idx].word;
    if ((word.load(std::memory_order_relaxed) & bit_mask) == 0)
    {
        word.fetch_or(bit_mask, std::memory_order_acq_rel);
    }
    // Raise the "some bit was set" flag (read first: raised once per phase,
    // then every later mark is a plain load).
    auto& any = any_marked_word();
    if (any.load(std::memory_order_relaxed) == 0)
    {
        any.store(1, std::memory_order_release);
    }
}

template <typename Derived>
void DirtyBitsetSupport<Derived, true>::clear_all_dirty_bitsets()
{
    using ProxyType = Derived::ProxyType;
    ProxyType::share_info_array->for_each_key(
        [](ArrayKey<typename Derived::ChunkType> key)
        {
            if (key != nullptr)
            {
                // The registry hands out const keys; clearing the bitset is
                // the quiescent point's own write.
                const_cast<Derived*>(key)->clear_dirty_bitset();
            }
        });
}

template <typename Derived>
void DirtyBitsetSupport<Derived, true>::reallocate_dirty_bitset(int old_nb_chunk, int new_nb_chunk)
{
    if (new_nb_chunk == old_nb_chunk)
    {
        return;
    }

    const int old_nb_words = words_for(old_nb_chunk);
    const int new_nb_words = words_for(new_nb_chunk);

    auto new_bitset = parallel_storage_init::make_filled<PaddedDirtyWord>(static_cast<std::size_t>(new_nb_words) + 1);

    if (!bitset_.empty())
    {
        const auto words_to_copy = static_cast<std::size_t>(std::min(old_nb_words, new_nb_words));
        for (std::size_t i = 0; i < words_to_copy; ++i)
        {
            const uint64_t val = bitset_[i].word.load(std::memory_order_relaxed);
            new_bitset[i].word.store(val, std::memory_order_relaxed);
        }
        // Carry the "some bit was set" flag (conservatively: it may now
        // cover only bits that were cut off, which costs one useless clear).
        new_bitset[static_cast<std::size_t>(new_nb_words)].word.store(
            bitset_[static_cast<std::size_t>(old_nb_words)].word.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
    bitset_   = std::move(new_bitset);
    nb_words_ = new_nb_words;
}

template <typename Derived>
void DirtyBitsetSupport<Derived, true>::clear_dirty_bitset() noexcept
{
    if (bitset_.empty())
    {
        return;
    }
    // Nothing marked since the last clear: the words are already zero.
    // The common case (only the deferred-enqueue path marks), and do not
    // drop it: clear_all_dirty_bitsets() runs serially at every phase end
    // of a non-OMPT build, and rewriting every line of every array then
    // dominates the run time of a benchmark.
    if (any_marked_word().exchange(0, std::memory_order_acq_rel) == 0)
    {
        return;
    }
    PaddedDirtyWord* const words = bitset_.data();
    for (int i = 0; i < nb_words_; ++i)
    {
        words[i].word.store(0, std::memory_order_relaxed);
    }
}

} // namespace shrink_array
