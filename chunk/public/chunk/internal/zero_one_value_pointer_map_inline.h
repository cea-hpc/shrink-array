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
#include "chunk/internal/zero_one_value_pointer_map.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::PointerMapChunk2InfArrayPolicy<T, Enabled>
// ---------------------------------------------------------------------------

template <typename T, bool Enabled>
[[nodiscard]] std::vector<typename PointerMapChunk2InfArrayPolicy<T, Enabled>::InfArrayT*>
PointerMapChunk2InfArrayPolicy<T, Enabled>::get_all_inf_array() const
{
    std::vector<InfArrayT*> values;

    for (const auto& [key, v] : map_inf_array_)
    {
        values.push_back(v);
    }

    return values;
}

template <typename T, bool Enabled>
[[nodiscard]] PointerMapChunk2InfArrayPolicy<T, Enabled>::InfArrayT*
PointerMapChunk2InfArrayPolicy<T, Enabled>::find_inf_array(InT<T> key) const
{
    const auto it = map_inf_array_.find(key);
    if (it != map_inf_array_.end())
    {
        return it->second;
    }
    return nullptr;
}

template <typename T, bool Enabled>
void PointerMapChunk2InfArrayPolicy<T, Enabled>::erase_inf_array_by_ptr(const InfArrayT* self_ptr)
{
    for (auto it = map_inf_array_.begin(); it != map_inf_array_.end();)
    {
        if (it->second == self_ptr)
        {
            it = map_inf_array_.erase(it);
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
void zero_one_value::PointerMap<T>::report_key_not_found(Key key, const char* site) const noexcept
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
        for (size_t i = 0; i < lim && n < keys.size(); ++i)
        {
            n += static_cast<std::size_t>(std::format_to_n(keys.data() + n, keys.size() - n, " [{} num_tab={}]",
                                                           static_cast<const void*>(vec_[i].first),
                                                           vec_[i].second.num_tab)
                                              .size);
        }
        CHUNK_LOG(error, general,
                  "share_info {}: KEY NOT FOUND: this={} rank={} searched_key={} vec.size={}; valid keys:{}{}", site,
                  this, rank, key, count, std::string_view{keys.data(), n < keys.size() ? n : keys.size()},
                  count > lim ? " ...(truncated)" : "");
    }
}

template <typename T>
void zero_one_value::PointerMap<T>::replace_all_other_elm(Key key, InT<T> val, int ind) noexcept
{
    for (auto& item : vec_)
    {
        if (item.first == key)
        {
            auto& info        = item.second;
            info.second_value = val;
            info.ind          = ind;
            return;
        }
    }
    report_key_not_found(key, "replace_all_other_elm");
    CHUNK_INVARIANT_FAILED("[share_info replace_all_other_elm] key NOT FOUND: this={} size={} key={}", this,
                           vec_.size(), key);
}

template <typename T>
[[nodiscard]] std::vector<T> zero_one_value::PointerMap<T>::get_all_second_value() const
{
    std::vector<T> keys;

    for (const auto& [key, v] : map_ind_cst_val2_)
    {
        keys.push_back(key);
    }

    return keys;
}

template <typename T>
[[nodiscard]] int zero_one_value::PointerMap<T>::find_ind_tab_cst2(InT<T> key) const
{
    if constexpr (is_not_bool)
    {
        const auto it = map_ind_cst_val2_.find(key);
        if (it != map_ind_cst_val2_.end())
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
void zero_one_value::PointerMap<T>::replace(Key key, InT<T> val, int ind) noexcept
{
    for (auto& item : vec_)
    {
        if (item.first == key)
        {
            item.second.second_value = val;
            item.second.ind          = ind;
            return;
        }
    }

    report_key_not_found(key, "replace");
    CHUNK_INVARIANT_FAILED("[share_info replace] key NOT FOUND: this={} size={} key={}", this, vec_.size(), key);
}

template <typename T>
[[nodiscard]] T zero_one_value::PointerMap<T>::get_second_value(Key key) const noexcept
{
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
[[nodiscard]] int zero_one_value::PointerMap<T>::get_num_tab(Key key) const noexcept
{
    for (const auto& item : vec_)
    {
        if (item.first == key)
        {
            return item.second.num_tab;
        }
    }
    report_key_not_found(key, "get_num_tab");
    CHUNK_INVARIANT_FAILED("[share_info get_num_tab] key NOT FOUND: this={} size={} key={}", this, vec_.size(), key);
}

template <typename T>
[[nodiscard]] zero_one_value::PointerMap<T>::InfArrayT
zero_one_value::PointerMap<T>::get_info_second_value(Key key) const noexcept
{
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
        // bool: second_value is always `true`, but num_tab is PER-ARRAY, so look
        // the key up. Do not return vec[0]: every bool array (rechunk_to's tmp
        // included) would get the first one's num_tab and its compression-flag
        // write would index the wrong array's tab_is_inside_chunk_list_to_compress
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

} // namespace shrink_array
