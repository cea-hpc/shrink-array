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
#include <array>       // std::array (report_key_not_found)
#include <format>      // std::format_to_n (report_key_not_found)
#include <string_view> // std::string_view (report_key_not_found)
#include <cstddef>     // std::size_t
#include <vector>      // std::vector

#include "chunk/internal/zero_or_value_pointer_map.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::ZeroOrValueArrayInfoPolicy<T, Enabled>
// ---------------------------------------------------------------------------

template <typename T, bool Enabled>
[[nodiscard]] ZeroOrValueArrayInfoPolicy<T, Enabled>::ArrayInfo*
ZeroOrValueArrayInfoPolicy<T, Enabled>::find_array_info(InT<T> key) const
{
    if (const auto it = array_info_map_.find(key); it != array_info_map_.end())
    {
        return it->second;
    }
    return nullptr;
}

template <typename T, bool Enabled>
void ZeroOrValueArrayInfoPolicy<T, Enabled>::erase_array_info_by_ptr(const ArrayInfo* self_ptr)
{
    for (auto it = array_info_map_.begin(); it != array_info_map_.end();)
    {
        if (it->second == self_ptr)
        {
            it = array_info_map_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::PointerMap<T>
// ---------------------------------------------------------------------------

template <typename T>
void zero_or_value::PointerMap<T>::report_key_not_found(Key key, const char* site) const noexcept
{
    if constexpr (build_config::any_check)
    {
        // -1 means "outside any parallel region"; CHUNK_PMP_* already
        // folds to that in a serial build.
        const int    rank  = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_THREAD_RANK : -1;
        const size_t count = vec_.size();

        // The valid keys, bounded so a corrupt size can't run away (and
        // cut at the log's line length anyway).
        std::array<char, logging::line_max> keys{};
        std::size_t                         n   = 0;
        const size_t                        lim = (count > 64u) ? 64u : count;
        try
        {
            for (size_t i = 0; i < lim && n < keys.size(); ++i)
            {
                n += static_cast<std::size_t>(std::format_to_n(keys.data() + n, keys.size() - n, " [{} array_id={}]",
                                                               static_cast<const void*>(vec_[i].first),
                                                               vec_[i].second.array_id)
                                                  .size);
            }
        }
        catch (...)
        {
            // Unreachable (a format string checked at compile time, a pointer
            // and an int): the keys listed so far are kept.
        }
        CHUNK_LOG(error, general,
                  "share_info {}: KEY NOT FOUND: this={} rank={} searched_key={} vec.size={}; valid keys:{}{}", site,
                  this, rank, key, count, std::string_view{keys.data(), n < keys.size() ? n : keys.size()},
                  count > lim ? " ...(truncated)" : "");
    }
}

template <typename T>
void zero_or_value::PointerMap<T>::replace_all_other_elm(Key key, InT<T> val, int idx) noexcept
{
    const internal::RegistryWriteLock registry_lock(internal::registry_mutex());
    for (auto& item : vec_)
    {
        if (item.first == key)
        {
            auto& info        = item.second;
            info.second_value = val;
            info.idx          = idx;
            return;
        }
    }
    report_key_not_found(key, "replace_all_other_elm");
    CHUNK_INVARIANT_FAILED("[share_info replace_all_other_elm] key NOT FOUND: this={} size={} key={}", this,
                           vec_.size(), key);
}

template <typename T>
[[nodiscard]] int zero_or_value::PointerMap<T>::find_second_value_index(InT<T> key) const
{
    if constexpr (is_not_bool)
    {
        if (const auto it = second_value_idx_map_.find(key); it != second_value_idx_map_.end())
        {
            return it->second;
        }
        return -1;
    }
    else
    {
        return 0;
    }
}

template <typename T>
void zero_or_value::PointerMap<T>::replace(Key key, InT<T> val, int idx) noexcept
{
    const internal::RegistryWriteLock registry_lock(internal::registry_mutex());
    for (auto& item : vec_)
    {
        if (item.first == key)
        {
            item.second.second_value = val;
            item.second.idx          = idx;
            return;
        }
    }

    report_key_not_found(key, "replace");
    CHUNK_INVARIANT_FAILED("[share_info replace] key NOT FOUND: this={} size={} key={}", this, vec_.size(), key);
}

template <typename T>
[[nodiscard]] T zero_or_value::PointerMap<T>::get_second_value(Key key) const noexcept
{
    const internal::RegistryReadLock registry_lock(internal::registry_mutex());
    if constexpr (is_not_bool)
    {
        for (const auto& item : vec_)
        {
            if (item.first == key)
            {
                return item.second.second_value;
            }
        }
    }
    else
    {
        return true;
    }
    report_key_not_found(key, "get_second_value");
    CHUNK_INVARIANT_FAILED("[share_info get_second_value] key NOT FOUND: this={} size={} key={}", this, vec_.size(),
                           key);
}

template <typename T>
[[nodiscard]] int zero_or_value::PointerMap<T>::get_array_id(Key key) const noexcept
{
    const internal::RegistryReadLock registry_lock(internal::registry_mutex());
    for (const auto& item : vec_)
    {
        if (item.first == key)
        {
            return item.second.array_id;
        }
    }
    report_key_not_found(key, "get_array_id");
    CHUNK_INVARIANT_FAILED("[share_info get_array_id] key NOT FOUND: this={} size={} key={}", this, vec_.size(), key);
}

template <typename T>
[[nodiscard]] zero_or_value::PointerMap<T>::ArrayInfo
zero_or_value::PointerMap<T>::get_info_second_value(Key key) const noexcept
{
    const internal::RegistryReadLock registry_lock(internal::registry_mutex());
    if constexpr (is_not_bool)
    {
        if constexpr (build_config::any_check)
        {
            // Snapshot size() ONCE and index stable deque slots (see the storage
            // note above): a concurrent append cannot move the loop bound.
            const size_t count = vec_.size();

            if (count > (1U << 30U))
            {
                CHUNK_INVARIANT_FAILED("[share_info get] corrupt vec: this={} size={} key={}", this, count, key);
            }

            for (size_t i = 0; i < count; ++i)
            {
                if (vec_[i].first == key)
                {
                    return vec_[i].second;
                }
            }
            CHUNK_INVARIANT_FAILED("[share_info get] key NOT FOUND: this={} size={} key={} "
                                   "(dangling ctx->ptr_on_array, or array destroyed while a deferred op "
                                   "still referenced it?)",
                                   this, count, key);
        }
        else
        {
            // deque: snapshot size() and index stable slots (see above).
            const size_t count = vec_.size();
            for (size_t i = 0; i < count; ++i)
            {
                if (vec_[i].first == key)
                {
                    return vec_[i].second;
                }
            }
            report_key_not_found(key, "get_info_second_value");
            CHUNK_INVARIANT_FAILED("[share_info get_info_second_value] key NOT FOUND: this={} size={} key={}", this,
                                   vec_.size(), key);
        }
    }
    else
    {
        // bool: second_value is always `true`, but array_id is PER-ARRAY, so look
        // the key up. Do not return vec[0]: every bool array (rechunk_to's tmp
        // included) would get the first one's array_id and its compression-flag
        // write would index the wrong array's array_is_inside_chunk_list_to_compress
        // (heap overflow). Same hazard as PointerMapLinear::erase().
        const size_t count = vec_.size();
        for (size_t i = 0; i < count; ++i)
        {
            if (vec_[i].first == key)
            {
                return vec_[i].second;
            }
        }
        return vec_[0].second; // fallback: not registered (should not happen)
    }
}

// ---------------------------------------------------------------------------
// shrink_array::PointerMap<T>
// ---------------------------------------------------------------------------

template <typename T>
void zero_or_value::PointerMap<T>::insert_only_array_id(Key key, int array_id)
{
    // No second value yet: the sentinel index, not 0 (which is the slot
    // of whichever second value was registered first, by any array).
    const T                           val{};
    const int                         idx = chunk_config::no_second_value_index;
    const internal::RegistryWriteLock registry_lock(internal::registry_mutex());
    vec_.emplace_back(key, ArrayInfo{val, idx, array_id});
}

} // namespace shrink_array
