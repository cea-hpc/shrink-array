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
 * @details PointerMapChunk (proxy.h) and PointerMapChunk2 (pointer_map_chunk2.h)
 *          both map an array's address to that array's metadata, and both do it
 *          by walking a flat sequence of (key, payload) pairs. Their erase(),
 *          exists(), size(), for_each_key() and swap_entries() are the same
 *          code. Only the payload differs: a table number for the first, a
 *          (second_value, ind, num_tab) record for the second.
 *
 *          Everything that is a plain scan lives here once. Each registry keeps
 *          what is specific to it: v1 its table-number accessors, v2 its whole
 *          second-value surface.
 *
 * @note The CONTAINER is a template parameter, and that is deliberate — do not
 *       "simplify" it away. PointerMapChunk2 stores in a std::deque precisely
 *       because a std::vector reallocates as it grows, and a reader walking the
 *       registry concurrently (get_info_second_value, reached from a deferred
 *       compression body on another thread) would then read freed memory. That
 *       showed up as intermittent "Key not found" failures and vtable
 *       corruption on aarch64 in release builds. See the note on
 *       PointerMapChunk2's storage for the full account. PointerMapChunk uses
 *       a std::vector.
 *
 * @note NOT used by PointerMapChunk2A (proxy2A.h): it stores in a small vector
 *       that switches to a hash map past a threshold, so its lookups are not
 *       linear scans at all. Folding it in here would trade that away.
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
     * @details Called from the array destructors. Without it a later array
     *          reusing this address (rechunk_to's tmp lives on the stack, so the
     *          reuse is systematic) is SHADOWED by the dead entry, because every
     *          lookup returns the first match — the new array would read the
     *          dead one's metadata. Idempotent, first match only, O(n),
     *          teardown-only.
     *
     * @param key the array address to drop.
     */
    void erase(const void* key)
    {
        for (auto it = vec.begin(); it != vec.end(); ++it)
        {
            if (it->first == key)
            {
                vec.erase(it);
                return;
            }
        }
    }

    /**
     * @brief Exchange the payloads of two registered keys.
     *
     * @details rechunk_to's identity swap: once *this and tmp have exchanged
     *          storage, the registry exchanges which metadata each address maps
     *          to, so get_num_tab(this) resolves tmp's fresh row (and
     *          get_num_tab(&tmp) the old one, for tmp's coherent destruction).
     *          Only rebinds payloads — no element is erased or moved, so no
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
        for (auto& item : vec)
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
        if (!pa || !pb)
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
        return static_cast<int>(vec.size());
    }

    /**
     * @brief Apply @p func to every registered key, in insertion order.
     * @tparam F callable taking the key.
     */
    template <typename F>
    void for_each_key(F&& func) const
    {
        for (const auto& item : vec)
        {
            func(item.first);
        }
    }

  protected:
    /**
     * @brief Address of the payload registered for @p key, or nullptr.
     *
     * @details FIRST match only, like every lookup here: a stale entry left by a
     *          destroyed array would otherwise shadow a live one that reused its
     *          address, which is exactly what erase() exists to prevent.
     */
    [[nodiscard]] const Payload* find_payload(const void* key) const noexcept
    {
        for (const auto& item : vec)
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
    Storage vec;
};

} // namespace shrink_array
