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

/**
 * @file pointer_map_linear.h
 * @brief The linear-scan core shared by the array→metadata registries.
 *
 * @details zero::PointerMap (zero_proxy.h) and zero_one_value::PointerMap (zero_one_value_pointer_map.h)
 *          map an array's address to its metadata by walking flat (key, payload)
 *          pairs; the plain scans live here once. Payload: a table number (zero),
 *          a (second_value, ind, num_tab) record (zero_one_value).
 *
 * @note The CONTAINER is a template parameter on purpose: zero_one_value::PointerMap
 *       needs a std::deque, since a std::vector reallocates as it grows and a
 *       concurrent reader (get_info_second_value, from a deferred compression
 *       body) would read freed memory. See zero_one_value::PointerMap's storage note.
 *       zero::PointerMap uses a std::vector.
 *
 * @note NOT used by zero_any::PointerMap (zero_any_proxy.h), whose small vector turns
 *       into a hash map past a threshold.
 */

#include <utility> // std::swap

namespace shrink_array
{

/**
 * @brief Flat (key → payload) registry addressed by pointer identity.
 *
 * @tparam Payload the metadata stored per key.
 * @tparam Storage the sequence holding the pairs — see the container note in
 *         the file header; each registry supplies its own.
 */
template <typename Payload, typename Storage>
class PointerMapLinear
{
  public:
    /**
     * @brief Remove the entry for @p key, if present.
     *
     * @details Called from the array destructors: otherwise a later array at
     *          this address (systematic with rechunk_to's stack tmp) is SHADOWED
     *          by the dead entry (first-match lookups). Idempotent, O(n).
     *
     * @param key the array address to drop.
     */
    void erase(const void* key)
    {
        for (auto it = vec_.begin(); it != vec_.end(); ++it)
        {
            if (it->first == key)
            {
                vec_.erase(it);
                return;
            }
        }
    }

    /**
     * @brief Exchange the payloads of two registered keys.
     *
     * @details The identity swap of rechunk_to and swap(): each address then
     *          resolves the other's metadata. Only payloads move, so no
     *          reference into the container is invalidated.
     *
     * @param key_a the first array address.
     * @param key_b the second array address.
     * @return true when both keys were present and their payloads swapped;
     *         false, with nothing changed, when either is missing.
     */
    bool swap_entries(const void* key_a, const void* key_b) noexcept
    {
        Payload* pa = nullptr;
        Payload* pb = nullptr;
        for (auto& item : vec_)
        {
            if (item.first == key_a)
            {
                pa = &item.second;
            }
            else if (item.first == key_b)
            {
                pb = &item.second;
            }
        }
        if (pa == nullptr || pb == nullptr)
        {
            return false;
        }
        std::swap(*pa, *pb);
        return true;
    }

    /**
     * @brief Whether @p key is currently registered.
     * @param key the array address to test.
     */
    [[nodiscard]] bool exists(const void* key) const noexcept
    {
        return find_payload(key) != nullptr;
    }

    /// @brief Number of registered keys.
    [[nodiscard]] int size() const noexcept
    {
        return static_cast<int>(vec_.size());
    }

    /**
     * @brief Apply @p func to every registered key, in insertion order.
     * @tparam F callable taking the key.
     */
    template <typename F>
    void for_each_key(F&& func) const
    {
        for (const auto& item : vec_)
        {
            func(item.first);
        }
    }

  protected:
    /**
     * @brief Address of the payload registered for @p key, or nullptr.
     *
     * @details FIRST match only, like every lookup here (see erase()).
     */
    [[nodiscard]] const Payload* find_payload(const void* key) const noexcept
    {
        for (const auto& item : vec_)
        {
            if (item.first == key)
            {
                return &item.second;
            }
        }
        return nullptr;
    }

    /// The (key, payload) pairs, in insertion order. Derived registries read it
    /// directly for the lookups that are specific to them.
    Storage vec_;
};

} // namespace shrink_array
