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

// ─────────────────────────────────────────────────────────────────────────────
// PointerMapChunk2 — per-array registry of Chunk2's SECOND constant
// ─────────────────────────────────────────────────────────────────────────────
//
// Chunk2 compresses a chunk to one of TWO constants: first_value is fixed for the
// type, the second is chosen per array at run time and a chunk folded to it stores
// only an index. This header holds that lookup, keyed by the array's identity
// (a void* handle) and by the value:
//
//   PointerMapChunk2<T>                  the map itself (key -> SecondValueInfo).
//   PointerMapChunk2InfArrayPolicy<T,B>  the value -> inf_array side, as a policy:
//                                        the <T,false> specialisation is all no-ops.
//   ValueToInfTupleMapHelper<T>          its std::map type, ordered by ValueLess
//                                        (value_less.h: complex has no operator<).
//
// The key -> value store is a LINEAR scan over a std::deque (PointerMapLinear; see
// the storage note in PointerMapChunk2). Only the value -> inf_array side is a std::map.

#include <cstdio>      // fprintf / fflush in report_key_not_found
#include <deque>       // std::deque: share_info storage, no realloc invalidation
#include <map>         // std::map
#include <type_traits> // std::is_same_v
#include <utility>     // std::pair
#include <vector>      // std::vector

#include "chunk/bit.h"                         // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/pointer_map_linear.h" // linear scan shared with PointerMapChunk
#include "chunk/PMP_CHUNK.h"                   // CHUNK_PMP_* — thread rank without an #ifdef _OPENMP
#include "chunk/internal/value_less.h"         // ValueLess
#include "chunk/internal/map_helper.h"         // ValueToIntMapHelper
#include "chunk/internal/in_t.h"               // in_t<T>
#include "chunk/config.h"                      // build_config::any_check
#include "chunk/internal/invariant_check.h"    // CHUNK_INVARIANT_FAILED
#include "chunk/internal/second_value_info.h"  // SecondValueInfo (inf_array_t)

namespace shrink_array
{

/// @brief Names the map from a second value to its inf_array record (SecondValueInfo<T>), ordered by ValueLess<T>.
template <typename T>
struct ValueToInfTupleMapHelper
{
    using inf_array_t = SecondValueInfo<T>;

    // Keyed by the second value, ordered by ValueLess (value_less.h).
    using type = std::map<T, inf_array_t*, ValueLess<T>>;
};

/**
 * @brief Policy holding the fast-block-compression-only value -> inf_array
 *        lookup for PointerMapChunk2.
 *
 * @details An inf_array is Chunk2's `(second_value, index, num_tab)` tuple;
 *          the map, keyed by second value, is ValueToInfTupleMapHelper's.
 *          v1 (PointerMapChunk, proxy.h) has no such lookup.
 *
 * @tparam T       Element type PointerMapChunk2 is keyed on.
 * @tparam Enabled Whether the map actually exists (fast-block-compression
 *                 builds) or is a zero-overhead no-op (counter-mode builds):
 *                 chunk_config::use_fast_block_compression.
 */
template <typename T, bool Enabled>
class PointerMapChunk2InfArrayPolicy
{
  public:
    using inf_array_t = SecondValueInfo<T>;

    /**
     * @brief All registered inf_array pointers, in map order.
     * @return Vector of every second value's currently-registered pointer.
     */
    [[nodiscard]] std::vector<inf_array_t*> get_all_inf_array() const
    {
        std::vector<inf_array_t*> values;

        for (const auto& [key, v] : map_inf_array)
        {
            values.push_back(v);
        }

        return values;
    }

    /**
     * @brief Look up the inf_array record pointer registered for a second value.
     * @param key Second value to look up.
     * @return The registered pointer, or nullptr if @p key is not registered.
     */
    [[nodiscard]] inf_array_t* find_inf_array(in_t<T> key) const
    {
        const auto it = map_inf_array.find(key);
        if (it != map_inf_array.end())
        {
            return it->second;
        }
        return nullptr;
    }

    /**
     * @brief Drop every entry still pointing at @p self_ptr.
     *
     * @details Called from ArrayChunk2::clear_core so a destroyed array's
     *          inf_array MEMBER cannot dangle here, where
     *          compress_chunk_to_compress_potentially() would dereference it via
     *          get_all_inf_array(). Only entries this array currently OWNS are
     *          removed: one re-registered by a later array points elsewhere and
     *          stays. O(n), teardown-only.
     *
     * @param self_ptr Address of the departing array's inf_array member.
     */
    void erase_inf_array_by_ptr(const inf_array_t* self_ptr)
    {
        for (auto it = map_inf_array.begin(); it != map_inf_array.end();)
        {
            if (it->second == self_ptr)
            {
                it = map_inf_array.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

  protected:
    using MapType0 = ValueToInfTupleMapHelper<T>::type;

    /**
     * @brief Register the inf_array record pointer for a second value.
     * @param key Second value to register.
     * @param ind inf_array record pointer to associate with @p key.
     */
    void insert(in_t<T> key, inf_array_t* ind)
    {
        map_inf_array[key] = ind;
    }

    /// Map from a second value to its registered inf_array record pointer.
    MapType0 map_inf_array;
};

/**
 * @brief Specialization for counter-mode builds: no map, zero overhead.
 *
 * @details The entry points stay declared so callers need no `if constexpr`
 *          of their own; they simply do nothing and report an empty registry.
 */
template <typename T>
class PointerMapChunk2InfArrayPolicy<T, false>
{
  public:
    using inf_array_t = SecondValueInfo<T>;

    /// @brief No registry: always empty.
    [[nodiscard]] std::vector<inf_array_t*> get_all_inf_array() const
    {
        return {};
    }
    /// @brief No registry: never found.
    [[nodiscard]] inf_array_t* find_inf_array(in_t<T> /*key*/) const
    {
        return nullptr;
    }
    /// @brief No registry: nothing to erase.
    static void erase_inf_array_by_ptr(const inf_array_t* /*self_ptr*/) {}

  protected:
    /// @brief No registry: nothing recorded.
    static void insert(in_t<T> /*key*/, inf_array_t* /*ind*/) {}
};

/**
 * @brief Per-array registry of Chunk2's second value: array address ->
 *        (second value, its index in the shared table, num_tab).
 *
 * @tparam T Element type.
 */
template <typename T>
class PointerMapChunk2
    : public PointerMapLinear<SecondValueInfo<T>, std::deque<std::pair<const void*, SecondValueInfo<T>>>>,
      public PointerMapChunk2InfArrayPolicy<T, chunk_config::use_fast_block_compression>
{
  private:
    /// Internal alias for tuple representing a value with index and table number.
    using inf_array_t = SecondValueInfo<T>;

    using Linear = PointerMapLinear<inf_array_t, std::deque<std::pair<const void*, inf_array_t>>>;
    /// @brief Bring the base's storage into this class's scope.
    /// @details Avoids `this->vec` everywhere (dependent base).
    using Linear::vec;

    /// @brief Policy carrying the fast-block-compression-only inf_array map.
    using InfArrayPolicy = PointerMapChunk2InfArrayPolicy<T, chunk_config::use_fast_block_compression>;

    // The storage must be a std::deque, NOT a std::vector: a vector's emplace_back
    // reallocates and frees the old buffer under a reader walking it in parallel
    // (get_info_second_value, from a deferred compression's write_body on another
    // thread). A deque's emplace_back never relocates elements nor invalidates
    // references; only iterators are invalidated, so the traversals below capture
    // size() first, then index stable slots.
    using MapType = ValueToIntMapHelper<T>::type;

    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    // Keys are inserted once at setup, before any parallel region, so a miss is
    // either a GARBAGE key or a VALID key that vanished (container raced). Printing
    // the searched key beside every stored one tells the two apart.
    /// @brief Check-only: dump the searched @p key and every stored
    ///        (key, num_tab) on stderr before a lookup from @p site fails.
    void report_key_not_found(const void* key, const char* site) const noexcept
    {
        if constexpr (build_config::any_check)
        {
            // -1 means "outside any parallel region"; CHUNK_PMP_* already
            // folds to that in a serial build.
            const int    rank  = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_THREAD_RANK : -1;
            const size_t count = vec.size();

            std::fprintf(stderr,
                         "[share_info %s] KEY NOT FOUND: this=%p rank=%d searched_key=%p "
                         "vec.size=%zu\n  valid keys:",
                         site, static_cast<const void*>(this), rank, key, count);
            // Bound the dump so a corrupt size can't run away.
            const size_t lim = (count > 64u) ? 64u : count;
            for (size_t i = 0; i < lim; ++i)
            {
                std::fprintf(stderr, " [%p num_tab=%d]", static_cast<const void*>(vec[i].first), vec[i].second.num_tab);
            }
            std::fprintf(stderr, "%s\n", (count > lim) ? " …(truncated)" : "");
            std::fflush(stderr);
        }
    }

    /// Map from values to constant index values.
    MapType map_ind_cst_val2;

  public:
    /// @brief The policy's insert(second value, inf_array pointer), protected
    ///        there, is part of this registry's public surface (ArrayChunk2
    ///        registers its inf_array through it).
    using InfArrayPolicy::insert;

    /**
     * @brief Inserts a default value and table number for a given key.
     *
     * @param key Pointer used as a key in the vector.
     * @param num_tab Table number associated with the key.
     */
    void insert_only_num_tab(void* key, int num_tab)
    {
        // No second value yet: the sentinel index, not 0 (which is the slot
        // of whichever second value was registered first, by any array).
        const T   val{};
        const int ind = chunk_config::no_second_value_index;
        vec.emplace_back(key, inf_array_t{val, ind, num_tab});
    }

    /**
     * @brief Entries get_info_second_value() scans, and the high-water mark.
     *
     * @details One entry per live array of this element type: a handful is
     *          expected, more means entries outlive their arrays. Tells a hot
     *          caller of get_info_second_value() (fix: caching) from a grown list
     *          (fix: erasing), since the scan can dominate a profile at high thread counts.
     */
    [[nodiscard]] std::size_t share_info_entries() const noexcept
    {
        return vec.size();
    }

    /**
     * @brief Replaces the value and index of an existing entry with the same key.
     *
     * @param key Pointer key to match.
     * @param val New value to assign.
     * @param ind New index to assign.
     * @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
     */
    void replace_all_other_elm(void* key, in_t<T> val, int ind) noexcept
    {
        for (auto& item : vec)
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
        CHUNK_INVARIANT_FAILED("[share_info replace_all_other_elm] key NOT FOUND: this={} size={} key={}",
                               static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
    }

    /// @brief Record that second value @p key sits at index @p ind of the
    ///        shared value table.
    void insert_map_ind_cst_val2(in_t<T> key, int ind)
    {
        map_ind_cst_val2[key] = ind;
    }

    /// @brief Every registered second value, in map order.
    [[nodiscard]] std::vector<T> get_all_second_value() const
    {
        std::vector<T> keys;

        for (const auto& [key, v] : map_ind_cst_val2)
        {
            keys.push_back(key);
        }

        return keys;
    }

    /// @brief Index of second value @p key in the shared value table, or -1
    ///        when it is not registered (0 for bool, which has no table).
    [[nodiscard]] int find_ind_tab_cst2(in_t<T> key) const
    {
        if constexpr (is_not_bool)
        {
            const auto it = map_ind_cst_val2.find(key);
            if (it != map_ind_cst_val2.end())
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

    /// @brief Change the second value and its table index recorded for
    ///        array @p key.
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    void replace(void* key, in_t<T> val, int ind) noexcept
    {
        for (auto& item : vec)
        {
            if (item.first == key)
            {
                item.second.second_value = val;
                item.second.ind          = ind;
                return;
            }
        }

        report_key_not_found(key, "replace");
        CHUNK_INVARIANT_FAILED("[share_info replace] key NOT FOUND: this={} size={} key={}",
                               static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
    }

    /// @brief The second value recorded for array @p key (true for bool).
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    [[nodiscard]] T get_second_value(void* key) const noexcept
    {
        if constexpr (is_not_bool)
        {
            for (const auto& item : vec)
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
        CHUNK_INVARIANT_FAILED("[share_info get_second_value] key NOT FOUND: this={} size={} key={}",
                               static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
    }
    /// @brief The num_tab recorded for array @p key.
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    [[nodiscard]] int get_num_tab(void* key) const noexcept
    {
        for (const auto& item : vec)
        {
            if (item.first == key)
            {
                return item.second.num_tab;
            }
        }
        report_key_not_found(key, "get_num_tab");
        CHUNK_INVARIANT_FAILED("[share_info get_num_tab] key NOT FOUND: this={} size={} key={}",
                               static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
    }

    /// @brief The whole (second value, index, num_tab) tuple recorded for
    ///        array @p key.
    [[nodiscard]] inf_array_t get_info_second_value(const void* key) const noexcept
    {
        if constexpr (is_not_bool)
        {
            if constexpr (build_config::any_check)
            {
                // Snapshot size() ONCE and index stable deque slots (see the storage
                // note above): a concurrent append cannot move the loop bound.
                const size_t count = vec.size();

                if (count > (1U << 30U))
                {
                    CHUNK_INVARIANT_FAILED("[share_info get] corrupt vec: this={} size={} key={}",
                                           static_cast<const void*>(this), count, key);
                }

                for (size_t i = 0; i < count; ++i)
                {
                    if (vec[i].first == key)
                    {
                        return vec[i].second;
                    }
                }
                CHUNK_INVARIANT_FAILED("[share_info get] key NOT FOUND: this={} size={} key={} "
                                       "(dangling ctx->ptr_on_array, or array destroyed while a deferred op "
                                       "still referenced it?)",
                                       static_cast<const void*>(this), count, key);
            }
            else
            {
                // deque: snapshot size() and index stable slots (see above).
                const size_t count = vec.size();
                for (size_t i = 0; i < count; ++i)
                {
                    if (vec[i].first == key)
                    {
                        return vec[i].second;
                    }
                }
                report_key_not_found(key, "get_info_second_value");
                CHUNK_INVARIANT_FAILED("[share_info get_info_second_value] key NOT FOUND: this={} size={} key={}",
                                       static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
            }
        }
        else
        {
            // bool: second_value is always `true`, but num_tab is PER-ARRAY, so look
            // the key up. Do not return vec[0]: every bool array (rechunk_to's tmp
            // included) would get the first one's num_tab and its compression-flag
            // write would index the wrong array's tab_is_inside_chunk_list_to_compress
            // (heap overflow). Same hazard as PointerMapLinear::erase().
            const size_t count = vec.size();
            for (size_t i = 0; i < count; ++i)
            {
                if (vec[i].first == key)
                {
                    return vec[i].second;
                }
            }
            return vec[0].second; // fallback: not registered (should not happen)
        }
    }

    // erase(), swap_entries(), exists(), size() and for_each_key() come from
    // PointerMapLinear (shared with PointerMapChunk).
};

} // namespace shrink_array