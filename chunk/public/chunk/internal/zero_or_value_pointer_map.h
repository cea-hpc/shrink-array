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
// zero_or_value::PointerMap — per-array registry of zero_or_value's SECOND constant
// ─────────────────────────────────────────────────────────────────────────────
//
// zero_or_value compresses a chunk to one of TWO constants: first_value is fixed for the
// type, the second is chosen per array at run time and a chunk folded to it stores
// only an index. This header holds that lookup, keyed by the array's identity
// (a void* handle) and by the value:
//
//   zero_or_value::PointerMap<T>                  the map itself (key -> SecondValueInfo).
//   ZeroOrValueArrayInfoPolicy<T,B>  the value -> array_info side, as a policy:
//                                        the <T,false> specialisation is all no-ops.
//   ValueToInfTupleMapHelper<T>          its std::map type, ordered by ValueLess
//                                        (value_less.h: complex has no operator<).
//
// The key -> value store is a LINEAR scan over a std::deque (PointerMapLinear; see
// the storage note in zero_or_value::PointerMap). Only the value -> array_info side is a std::map.

#include <array>       // std::array (report_key_not_found)
#include <cstddef>     // std::size_t
#include <format>      // std::format_to_n (report_key_not_found)
#include <string_view> // std::string_view (report_key_not_found)
#include <deque>       // std::deque: share_info storage, no realloc invalidation
#include <map>         // std::map
#include <ranges>      // std::views::keys / values, std::ranges::to (get_all_*)
#include <utility>     // std::pair
#include <vector>      // std::vector

#include "chunk/bit.h"                         // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/array_chunk_for.h"    // ArrayKey, zero_or_value::Chunk (the keys)
#include "chunk/internal/pointer_map_linear.h" // linear scan shared with zero::PointerMap
#include "chunk/internal/pmp_chunk.h"          // CHUNK_PMP_* — thread rank without an #ifdef _OPENMP
#include "chunk/internal/value_less.h"         // ValueLess
#include "chunk/internal/map_helper.h"         // ValueToIntMapHelper
#include "chunk/internal/in_t.h"               // InT<T>
#include "chunk/config.h"                      // build_config::any_check
#include "chunk/internal/invariant_check.h"    // CHUNK_INVARIANT_FAILED
#include "chunk/internal/second_value_info.h"  // SecondValueInfo (ArrayInfo)
#include "chunk/log.h"                         // CHUNK_LOG (report_key_not_found)

namespace shrink_array
{

/// @brief Names the map from a second value to its array_info record (SecondValueInfo<T>), ordered by ValueLess<T>.
template <typename T>
struct ValueToInfTupleMapHelper
{
    using ArrayInfo = SecondValueInfo<T>;

    // Keyed by the second value, ordered by ValueLess (value_less.h). The map
    // OWNS the record of each value: a std::map node does not move, so the
    // pointers handed out (find_array_info, all_array_infos) stay valid, and
    // no array's lifetime decides whether a value is drained.
    using type = std::map<T, ArrayInfo, ValueLess<T>>;
};

/**
 * @brief Policy holding the fast-block-compression-only value -> array_info
 *        lookup for zero_or_value::PointerMap.
 *
 * @details An array_info is zero_or_value's `(second_value, index, array_id)` tuple;
 *          the map, keyed by second value, is ValueToInfTupleMapHelper's.
 *          zero (zero::PointerMap, zero_proxy.h) has no such lookup.
 *
 * @tparam T       Element type zero_or_value::PointerMap is keyed on.
 * @tparam Enabled Whether the map actually exists (fast-block-compression
 *                 builds) or is a zero-overhead no-op (counter-mode builds):
 *                 chunk_config::use_fast_block_compression.
 */
template <typename T, bool Enabled>
class ZeroOrValueArrayInfoPolicy
{
  public:
    using ArrayInfo = SecondValueInfo<T>;

    /**
     * @brief The record of every registered second value, in map order.
     * @return One pointer per value, into the map: valid while the registry lives.
     */
    [[nodiscard]] std::vector<const ArrayInfo*> all_array_infos() const
    {
        // ranges::to in its call form, not `| std::ranges::to<std::vector>()`:
        // clang 23.1 (the apt.llvm.org snapshot of the nightly) rejects the
        // pipe's closure object with libstdc++ 14.
        return std::ranges::to<std::vector>(array_info_map_ | std::views::values |
                                            std::views::transform([](const ArrayInfo& v) { return &v; }));
    }

    /**
     * @brief Look up the record registered for a second value.
     * @param key Second value to look up.
     * @return A pointer into the map, or nullptr if @p key is not registered.
     */
    [[nodiscard]] const ArrayInfo* find_array_info(InT<T> key) const;

  protected:
    using MapType0 = ValueToInfTupleMapHelper<T>::type;

    /**
     * @brief Register second value @p key with the (value, index) of @p info.
     * @details A copy, not the array's member: the record belongs to the value,
     *          and every array declaring it shares it. It was a pointer to the
     *          member of the LAST array that declared the value, erased with
     *          that array or left on its new value when it took another (copy,
     *          move-assign, swap): the other arrays declaring the value were no
     *          longer drained. A value keeps its index for the registry's life
     *          (second_value_idx_map_ never erases), so re-registering it only
     *          rewrites the same record.
     * @param key  Second value to register.
     * @param info (second value, index, array_id) of the declaring array.
     */
    void insert(InT<T> key, const ArrayInfo* info)
    {
        array_info_map_.insert_or_assign(key, *info);
    }

    /// Map from a second value to its record.
    MapType0 array_info_map_;
};

/**
 * @brief Specialization for counter-mode builds: no map, zero overhead.
 *
 * @details The entry points stay declared so callers need no `if constexpr`
 *          of their own; they simply do nothing and report an empty registry.
 */
template <typename T>
class ZeroOrValueArrayInfoPolicy<T, false>
{
  public:
    using ArrayInfo = SecondValueInfo<T>;

    /// @brief No registry: always empty.
    [[nodiscard]] static std::vector<const ArrayInfo*> all_array_infos()
    {
        return {};
    }
    /// @brief No registry: never found.
    [[nodiscard]] static const ArrayInfo* find_array_info(InT<T> /*key*/) noexcept
    {
        return nullptr;
    }

  protected:
    /// @brief No registry: nothing recorded.
    static void insert(InT<T> /*key*/, const ArrayInfo* /*info*/) {}
};

namespace zero_or_value
{

/**
 * @brief Per-array registry of zero_or_value's second value: array address ->
 *        (second value, its index in the shared table, array_id).
 *
 * @tparam T Element type.
 */
template <typename T>
class PointerMap
    : public PointerMapLinear<SecondValueInfo<T>, std::deque<std::pair<ArrayKey<Chunk<T>>, SecondValueInfo<T>>>>,
      public ZeroOrValueArrayInfoPolicy<T, chunk_config::use_fast_block_compression>
{
  private:
    /// Internal alias for tuple representing a value with index and table number.
    using ArrayInfo = SecondValueInfo<T>;

    using Linear = PointerMapLinear<ArrayInfo, std::deque<std::pair<ArrayKey<Chunk<T>>, ArrayInfo>>>;
    /// @brief Bring the base's storage into this class's scope.
    /// @details Avoids `this->vec` everywhere (dependent base).
    using Linear::vec_;

  public:
    /// @brief The key: the array's identity.
    using Key = Linear::Key;

  private:
    /// @brief Policy carrying the fast-block-compression-only array_info map.
    using ArrayInfoPolicy = ZeroOrValueArrayInfoPolicy<T, chunk_config::use_fast_block_compression>;

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
    /// @brief Check-only: write the searched @p key and every stored
    ///        (key, array_id) on the log (an error) before a lookup from @p
    ///        site fails.
    void report_key_not_found(Key key, const char* site) const noexcept;

    /// Map from values to constant index values.
    MapType second_value_idx_map_;

  public:
    /// @brief The policy's insert(second value, array_info pointer), protected
    ///        there, is part of this registry's public surface (zero_or_value::Array
    ///        registers its array_info through it).
    using ArrayInfoPolicy::insert;

    /**
     * @brief Inserts a default value and table number for a given key.
     *
     * @param key Pointer used as a key in the vector.
     * @param array_id Table number associated with the key.
     */
    void insert_only_array_id(Key key, int array_id);

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
        return vec_.size();
    }

    /**
     * @brief Replaces the value and index of an existing entry with the same key.
     *
     * @param key Pointer key to match.
     * @param val New value to assign.
     * @param idx New index to assign.
     * @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
     */
    void replace_all_other_elm(Key key, InT<T> val, int idx) noexcept;

    /// @brief Record that second value @p key sits at index @p idx of the
    ///        shared value table.
    void insert_second_value_idx(InT<T> key, int idx)
    {
        second_value_idx_map_[key] = idx;
    }

    /// @brief Every registered second value, in map order.
    [[nodiscard]] std::vector<T> get_all_second_value() const
    {
        return std::ranges::to<std::vector>(second_value_idx_map_ | std::views::keys); // call form, as above
    }

    /// @brief Index of second value @p key in the shared value table, or -1
    ///        when it is not registered (0 for bool, which has no table).
    [[nodiscard]] int find_second_value_index(InT<T> key) const;

    /// @brief Change the second value and its table index recorded for
    ///        array @p key.
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    void replace(Key key, InT<T> val, int idx) noexcept;

    /// @brief The second value recorded for array @p key (true for bool).
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    [[nodiscard]] T get_second_value(Key key) const noexcept;
    /// @brief The array_id recorded for array @p key.
    /// @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
    [[nodiscard]] int get_array_id(Key key) const noexcept;

    /// @brief The whole (second value, index, array_id) tuple recorded for
    ///        array @p key.
    [[nodiscard]] ArrayInfo get_info_second_value(Key key) const noexcept;

    // erase(), swap_entries(), exists(), size() and for_each_key() come from
    // PointerMapLinear (shared with zero::PointerMap).
};

} // namespace zero_or_value

} // namespace shrink_array

#include "chunk/internal/zero_or_value_pointer_map_inline.h" // the template bodies
