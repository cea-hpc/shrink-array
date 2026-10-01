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

#include <algorithm> // std::min, std::copy_n, std::fill (the resize copy)
#include <atomic>
#include <cstdlib>   // std::abort (debug stubs)
#include <exception> // std::exception (PointerMapChunk2A::erase)
#include <iostream>  // std::cerr (debug stubs)
#include <memory>    // std::unique_ptr
#include <type_traits>
#include <utility> // std::pair, std::swap
#include <vector>
#ifndef __NVCOMPILER
    #include <concepts>
#endif

#include <ankerl/unordered_dense.h> // PointerMapChunk2A::big

#include "chunk/config.h"
#include "chunk/internal/chunk_size_policy.h"
#include "chunk/internal/concept_is_clean_assignement.h"
#include "chunk/internal/crp_helper.h"
#include "chunk/internal/in_t.h"
#include "chunk/internal/invariant_check.h"     // INVARIANT_FAILED
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef
#include "chunk/internal/TChunk.h"
#include "chunk/internal/proxy_base.h"
#include "chunk/internal/proxy2_base.h"

namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkType>
class CompressionManager;

template <template <typename> class CHUNK_T, typename T>
class ManagerChunk;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Forward declaration of Chunk2A used by Proxy2A/PointerMapChunk2A.
 * @tparam T element type
 */

template <typename T>
class Chunk2A;

/**
 * @brief Registry mapping array-element keys (pointers) to their table
 *        number, for Chunk2A. Two-phase storage: a linear-scan vector while
 *        the entry count stays small (cheap, cache-friendly for the common
 *        case), switching once and for all to a hash map past SmallN
 *        entries (avoids O(n) lookups for programs with many arrays).
 * @tparam T Element type this registry is keyed on (unused by the registry
 *           itself; kept for interface symmetry with PointerMapChunk).
 * @tparam SmallN Entry-count threshold for switching from the linear-scan
 *                vector to the hash map. Default 16.
 */
template <typename T, size_t SmallN = 16>
class PointerMapChunk2A
{
  public:
    PointerMapChunk2A()                                        = default;
    PointerMapChunk2A(const PointerMapChunk2A&)                = default;
    PointerMapChunk2A(PointerMapChunk2A&&) noexcept            = default;
    PointerMapChunk2A& operator=(const PointerMapChunk2A&)     = default;
    PointerMapChunk2A& operator=(PointerMapChunk2A&&) noexcept = default;

    /**
     * @brief Look up the table number registered for a key.
     * @param key Key to look up.
     * @return The registered table number.
     * @note A key that is not registered is an invariant failure (INVARIANT_FAILED: message, then abort).
     */
    [[nodiscard]] int get_num_tab(const void* key) const noexcept
    {
        if (!using_big)
        {
            for (const auto& [k, tab] : small)
            {
                if (k == key)
                    return tab;
            }
            INVARIANT_FAILED("[PointerMapChunk2A get_num_tab] key NOT FOUND: this={} size={} key={}",
                             static_cast<const void*>(this), small.size(), key);
        }

        if (const auto it = big.find(key); it != big.end())
            return it->second;
        INVARIANT_FAILED("[PointerMapChunk2A get_num_tab] key NOT FOUND: this={} size={} key={}",
                         static_cast<const void*>(this), big.size(), key);
    }

    /**
     * @brief Remove the entry for `key`, if present. Must be called when an
     *        array is destroyed, otherwise its (address -> num_tab) entry
     *        lingers. Stack arrays (e.g. rechunk_to's tmp) get their
     *        addresses reused by later arrays; a stale entry would then
     *        shadow the new array's real num_tab in get_num_tab() (which
     *        returns the FIRST match in phase "small") -> the new array
     *        reads another tab's metadata tables -> out-of-bounds /
     *        corruption. Idempotent.
     * @param key Key to remove.
     */
    void erase(const void* key) noexcept
    {
        if (!using_big)
        {
            for (auto it = small.begin(); it != small.end(); ++it)
            {
                if (it->first == key)
                {
                    small.erase(it);
                    return;
                }
            }
        }
        else
        {
            // ankerl's erase throws on a key it finds changed; ~ArrayChunk2A
            // calls this and is noexcept: fail with the key instead.
            try
            {
                big.erase(key);
            }
            catch (const std::exception& e)
            {
                INVARIANT_FAILED("[PointerMapChunk2A erase] failed: {} (this={} key={})", e.what(),
                                 static_cast<const void*>(this), key);
            }
        }
    }

    /**
     * @brief Register a new key -> num_tab mapping. Once the entry count
     *        crosses SmallN, transparently migrates all entries from the
     *        linear-scan vector to the hash map (one-way; never migrates back).
     * @param key Key to register.
     * @param num_tab Table number to associate with `key`.
     */
    void insert(const void* key, int num_tab)
    {
        if (!using_big)
        {
            // Phase "small"
            small.emplace_back(key, num_tab);

            if (small.size() > SmallN)
            {
                using_big = true;
                big.reserve(small.size());
                for (const auto& [k, tab] : small)
                {
                    big[k] = tab;
                }
                small.clear();
            }
        }
        else
        {
            // Phase "big"
            big[key] = num_tab;
        }
    }

    /**
     * @brief Re-map an EXISTING key to a new num_tab. Unlike insert(), this
     *        updates the existing entry in place (in phase "small",
     *        insert() would append a duplicate and get_num_tab() would then
     *        return the stale first match).
     * @param key Key to update.
     * @param new_num_tab New table number.
     * @return false if the key was not present (caller error: nothing to
     *         re-map); true otherwise.
     */
    bool set_num_tab(const void* key, int new_num_tab) noexcept
    {
        if (!using_big)
        {
            for (auto& [k, tab] : small)
            {
                if (k == key)
                {
                    tab = new_num_tab;
                    return true;
                }
            }
            return false;
        }

        const auto it = big.find(key);
        if (it == big.end())
            return false;
        it->second = new_num_tab;
        return true;
    }

    /**
     * @brief Swap the num_tab values associated with two existing keys. Used
     *        by rechunk_to's array-swap: after two arrays exchange their
     *        storage and identities, the registry must exchange which
     *        num_tab each address maps to.
     * @param key_a First key.
     * @param key_b Second key.
     * @return true if both keys were found (and swapped); false otherwise.
     */
    bool swap_keys(const void* key_a, const void* key_b) noexcept
    {
        int  na = 0;
        int  nb = 0;
        bool fa = false;
        bool fb = false;
        if (!using_big)
        {
            for (const auto& [k, tab] : small)
            {
                if (k == key_a)
                {
                    na = tab;
                    fa = true;
                }
                if (k == key_b)
                {
                    nb = tab;
                    fb = true;
                }
            }
            if (!fa || !fb)
                return false;
            for (auto& [k, tab] : small)
            {
                if (k == key_a)
                    tab = nb;
                else if (k == key_b)
                    tab = na;
            }
            return true;
        }

        const auto ia = big.find(key_a);
        const auto ib = big.find(key_b);
        if (ia == big.end() || ib == big.end())
            return false;
        std::swap(ia->second, ib->second);
        return true;
    }

    /**
     * @brief Call `func` with each registered key, regardless of which
     *        storage phase is currently active.
     * @tparam F Callable invoked with a `const void*` key.
     * @param func Callback to invoke for every key.
     */
    template <typename F>
    void for_each_key(F&& func) const
    {
        if (!using_big)
        {
            for (const auto& [k, tab] : small)
                func(k);
        }
        else
        {
            for (const auto& [k, tab] : big)
                func(k);
        }
    }

  private:
    std::vector<std::pair<const void*, int>>       small;
    ankerl::unordered_dense::map<const void*, int> big;
    bool                                           using_big = false;
};

/*!
 * \brief Specification class for managing auxiliary second-value arrays in chunk compression.
 *
 * ProxySpec2A provides static storage and access to per-tab, per-chunk second values
 * used by compression-aware proxies such as Proxy2A. It abstracts the layout and access
 * to these auxiliary values, allowing Proxy2A to retrieve and update second-value data
 * without managing the underlying container directly.
 *
 * This class is designed to be extended by Proxy2A<T>, which uses it to coordinate
 * compression metadata and value tracking across chunked arrays.
 *
 * \tparam T Type of the second value stored in the chunk.
 */

/// @brief Reallocate @p ptr from @p old_size to @p new_size elements,
///        keeping the first min(old_size, new_size).
/// @warning Elements [old_size, new_size) are left INDETERMINATE: the caller
///          fills them (ProxySpec2A<bool>::resize does). The block is
///          default-initialised (`new T[n]`), not value-initialised as
///          make_unique<T[]> did: its zero-fill was overwritten by the copy
///          and by the caller's fill anyway -- three write passes for one.
template <typename T>
void realloc_unique(std::unique_ptr<T[]>& ptr, std::size_t old_size, std::size_t new_size)
{
    std::unique_ptr<T[]> new_ptr(new T[new_size]);

    const std::size_t kept = std::min(old_size, new_size);
    if (kept != 0)
        std::copy_n(ptr.get(), kept, new_ptr.get()); // a memmove for a trivial T

    ptr = std::move(new_ptr);
}

/// @brief Per-table, per-chunk second-value storage used by Proxy2A (see the description above realloc_unique).
template <typename T, typename Derived>
class ProxySpec2A : public Proxy2Base<Derived, T, Chunk2A<T>>
{
  public:
    using Base2 = Proxy2Base<Derived, T, Chunk2A<T>>;

    /// @brief What a slot holds: the value, or for a wide T the identifier
    ///        of its shared array (SecondValueTraits) -- 4 bytes per chunk
    ///        instead of 8 or 16.
    using second_value_slot_t = SecondValueTraits<T>::slot_t;

    using Base2::Base2;

    /*!
     * \brief Retrieves a pointer to a specific second value in the chunk.
     * \param num_tab Index of the tab.
     * \param num_chunk Index of the chunk within the tab.
     * \return Pointer to the second value.
     */
    [[nodiscard]] static SecondValueRef<T> get_tab_second_value(int num_tab, int num_chunk) noexcept
    {
        return SecondValueRef<T>{&tab_second_value[num_tab][num_chunk]};
    }

    /*!
     * \brief Retrieves the vector of second values for a specific tab.
     * \param num_tab Index of the tab.
     * \return Reference to the vector of second values.
     */
    [[nodiscard]] static std::vector<second_value_slot_t>& get_tab_second_value(int num_tab) noexcept
    {
        return tab_second_value[static_cast<std::size_t>(num_tab)];
    }

    /*!
     * \brief Sets a specific second value in a tab and chunk.
     * \param num_tab Index of the tab.
     * \param i Index of the chunk within the tab.
     * \param v Value to assign.
     */
    static void set(int num_tab, int i, in_t<T> v)
    {
        get_tab_second_value(num_tab, i).set(v);
    }

  protected:
    /*!
     * \brief Resizes the number of tabs in the second value container.
     * \param nb_tab New number of tabs.
     */
    static void resize(int nb_tab)
    {
        tab_second_value.resize(nb_tab);
    }

    /*!
     * \brief Resizes the number of chunks in a specific tab and initializes values to zero.
     * \param nb_tab Index of the tab to resize.
     * \param nb_chunk_old Number of chunks before the resize.
     * \param nb_chunk New number of chunks in the tab.
     */
    static void resize(int nb_tab, int nb_chunk_old, int nb_chunk)
    {
        auto& row = tab_second_value[static_cast<std::size_t>(nb_tab)];
        row.resize(static_cast<std::size_t>(nb_chunk)); // keeps [0, min(old, new))
        // The tail from the OLD count is cleared, as the reallocation did: the
        // row may hold more than nb_chunk_old entries.
        for (int k = nb_chunk_old; k < nb_chunk; ++k)
        {
            row[static_cast<std::size_t>(k)] = second_value_slot_t{};
        }
    }

    /*!
     * \brief Clears all second values in a specific tab.
     * \param num_tab Index of the tab to clear.
     */
    static void clear([[maybe_unused]] int num_tab)
    {
        // The array's per-chunk second-value slots, released, not just emptied
        // (a re-chunk destroys the old storage under its own num_tab: this row
        // leaked 4 bytes per old chunk at every re-chunk until Sept '26).
        // Swapped with an empty row: `= {}` would keep the capacity.
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_second_value.size())
            std::vector<second_value_slot_t>().swap(tab_second_value[static_cast<std::size_t>(num_tab)]);
    }

    SecondValueRef<T> ptr_second_value; ///< this chunk's slot in tab_second_value[num_tab]

  private:
    /*!
     * \brief Static container holding second values for each tab and chunk.
     * Indexed as tab_second_value[num_tab][num_chunk].
     */
    static inline std::vector<std::vector<second_value_slot_t>> tab_second_value;
};

/*!
 * \brief Specialization of ProxySpec2A for boolean values.
 *
 * This specialization bypasses dynamic allocation and uses a static pointer
 * from Chunk2A<bool> for all second value access. It disables resizing and
 * mutation operations since the boolean compression model uses a shared constant.
 */

template <typename Derived>
class ProxySpec2A<bool, Derived> : public Proxy2Base<Derived, bool, Chunk2A<bool>>
{
  public:
    using ChunkType = Chunk2A<bool>;

    using Base2 = Proxy2Base<Derived, bool, ChunkType>;

    using values_type_ref = Proxy2Base<Derived, bool, ChunkType>::values_type_ref;

    /// @brief Forward the element reference, index and chunk to Proxy2Base.
    /// @param v Reference to the chunk's values.
    /// @param num_ Index of the element in the chunk.
    /// @param chunk_ Owning chunk.
    explicit ProxySpec2A(values_type_ref v, int num_, ChunkType& chunk_) noexcept : Base2(v, num_, chunk_) {}

    /*!
     * \brief Retrieves a pointer to a specific second value in the chunk.
     * \param num_tab Index of the tab.
     * \param num_chunk Index of the chunk within the tab.
     * \return Pointer to the second value.
     */
    [[nodiscard]] static SecondValueRef<bool> get_tab_second_value(int num_tab, int num_chunk) noexcept
    {
        return SecondValueRef<bool>{&tab_second_value[num_tab][num_chunk]};
    }

    /*!
     * \brief Retrieves the vector of second values for a specific tab.
     * \param num_tab Index of the tab.
     * \return Reference to the vector of second values.
     */
    [[nodiscard]] static std::unique_ptr<bool[]>& get_tab_second_value(int num_tab) noexcept
    {
        return tab_second_value[num_tab];
    }

    /*!
     * \brief Sets a specific second value in a tab and chunk.
     * \param num_tab Index of the tab.
     * \param i Index of the chunk within the tab.
     * \param v Value to assign.
     */
    static void set(int num_tab, int i, bool v) noexcept
    {
        tab_second_value[num_tab][i] = v;
    }

  protected:
    /*!
     * \brief Resizes the number of tabs in the second value container.
     * \param nb_tab New number of tabs.
     */
    static void resize(int nb_tab)
    {
        tab_second_value.resize(nb_tab);
    }

    /*!
     * \brief Resizes the number of chunks in a specific tab and initializes values to zero.
     * \param nb_tab Index of the tab to resize.
     * \param nb_chunk_old Old number of chunks in the tab.
     * \param nb_chunk New number of chunks in the tab.
     */
    static void resize(int nb_tab, int nb_chunk_old, int nb_chunk)
    {
        realloc_unique(tab_second_value[nb_tab], nb_chunk_old, nb_chunk);

        // realloc_unique leaves the new tail indeterminate: fill it here.
        if (nb_chunk > nb_chunk_old)
            std::fill(tab_second_value[nb_tab].get() + nb_chunk_old, tab_second_value[nb_tab].get() + nb_chunk, false);
    }

    /*!
     * \brief Clears all second values in a specific tab.
     * \param num_tab Index of the tab to clear.
     */
    static void clear([[maybe_unused]] int num_tab)
    {
        // Released, not just emptied -- see the non-bool specialisation.
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_second_value.size())
            tab_second_value[static_cast<std::size_t>(num_tab)].reset();
    }

  private:
    /*!
     * \brief Static container holding second values for each tab and chunk.
     * Indexed as tab_second_value[num_tab][num_chunk].
     */
    static inline std::vector<std::unique_ptr<bool[]>> tab_second_value;
};

/**
 * @brief Policy to handle Fast Block Compression specific logic for Proxy2A.
 * Encapsulates per-thread list management and trigger logic.
 */

template <typename Derived, typename T, typename ChunkType, bool Enabled>
class Proxy2AFastCompPolicy : public ProxySpec2A<T, Derived>
{
  public:
    using BaseSpec = ProxySpec2A<T, Derived>;

    using CRTPSupport<Derived>::derived;

    using values_type_ref = BaseSpec::values_type_ref;

    using Base  = ProxyBase<Derived, T, ChunkType>;
    using Base2 = Proxy2Base<Derived, T, ChunkType>;
    using Base::chunk;

    using BaseSpec::BaseSpec;

    // --- per-thread candidate list plumbing (OpenMP only) -----------------
    template <typename LAMBDA>
    void execute_simple_operation_async(this Derived& self, auto* ctx, LAMBDA operation);

    /**
     * @brief Executes an in-place modification (bitwise, inc/dec) asynchronously.
     */
    template <typename LAMBDA>
    void execute_inplace_operation_async(this const Derived& self, auto* ctx, LAMBDA operation);

    template <typename OP, typename LBD, typename U>
    void execute_arithmetic_compound_operation_async(auto* ctx, OP op, const U& v, LBD lbd) const;

    /// @brief Rank @p thr's Chunk2A candidate list.
    TChunk2AComp<Chunk2A, T>& get_vec_tuple_to_compress_potentially_thr(int thr) const noexcept
    {
        return chunk.get_vec_tuple_to_compress_potentially_thr(thr);
    }

    /// @brief Common initialisation of the fast path: table number, then the
    ///        "already queued" flag pointer.
    void set_num_tab_ptr_is_inside_chunk(this Derived& self) noexcept
    {
        self.set_num_tab(); // Proxy2A's no-arg overload (looks the table up)
        self.Base::set_ptr_is_inside_chunk_list_to_compress();
    }

  private:
    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    void ensure_writable_storage_async(const Derived& proxy) const;
};
/**
 * @brief Specialization for Disabled Fast Compression (Zero Overhead).
 */
template <typename Derived, typename T, typename ChunkType>
class Proxy2AFastCompPolicy<Derived, T, ChunkType, false> : public ProxySpec2A<T, Derived>
{
  public:
    using BaseSpec = ProxySpec2A<T, Derived>;

    using values_type_ref = BaseSpec::values_type_ref;

    using Base = ProxyBase<Derived, T, ChunkType>;
    using Base::chunk;

    using BaseSpec::BaseSpec;

    /// @brief The single candidate list of a serial build (Chunk2AOmpPolicy<false>).
    TChunk2AComp<Chunk2A, T>& get_vec_tuple_to_compress_potentially() const noexcept
    {
        return chunk.get_vec_tuple_to_compress_potentially();
    }
};
/**
 * @brief Proxy2A class providing element access to a chunked sparse array representation.
 *
 * Proxy2A performs read/write operations targeting a specific slot inside a chunked
 * container. It implements atomic-aware fast-path/slow-path locking for compression
 * and manages auxiliary per-chunk metadata (second values, type-of-constant flags,
 * and compression bookkeeping). Many operations are forwarded to the underlying
 * Chunk2A<T> implementation via create_value() and other helpers.
 *
 * The class mirrors much of the original project behavior; only stylistic/modern
 * changes and documentation were applied.
 *
 * @tparam T stored element type
 */

/// @brief Proxy to one element of a Chunk2A<T> chunk (see the class description above).
/// @tparam T Element type.
template <typename T>
class Proxy2A : public Proxy2AFastCompPolicy<Proxy2A<T>, T, Chunk2A<T>, chunk_config::use_fast_block_compression>
{
  public:
    /// @brief Stream output of the element's value (defined below).
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Proxy2A<U>& proxy);

    using ChunkType          = Chunk2A<T>;
    using Base2              = Proxy2Base<Proxy2A, T, ChunkType>;
    using Base               = ProxyBase<Proxy2A, T, ChunkType>;
    using BaseFastCompPolicy = Proxy2AFastCompPolicy<Proxy2A, T, ChunkType, chunk_config::use_fast_block_compression>;
    using BaseSpec           = ProxySpec2A<T, Proxy2A>;

    friend Base;
    friend Base2;
    friend BaseFastCompPolicy;

    template <typename, typename, typename, bool>
    friend class ProxyBaseFastCompPolicy;

    /**
     * @brief Construct a Proxy2A for OpenMP build (atomic reference to values).
     * @param values_ reference to array pointer (atomic)
     * @param num_ index within array
     * @param chunk_ owning chunk
     */
    explicit Proxy2A(BaseFastCompPolicy::values_type_ref values_, int num_, Chunk2A<T>& chunk_) noexcept
        : Proxy2AFastCompPolicy<Proxy2A, T, Chunk2A<T>, chunk_config::use_fast_block_compression>(values_, num_, chunk_)
    {
    }

    /** @brief Clean path for safe assignments (small operand, by value). */
    template <typename U>
        requires IsCleanAssignment<T, U>
    T operator=(const U& v) noexcept
    {
        return assign_impl<U>(v);
    }

    /** @brief Trap path for risky assignments (e.g., float = 1.01). */
    template <typename U>
        requires(!IsCleanAssignment<T, U>) && std::assignable_from<T&, U>
    [[deprecated("HPC WARNING: Type mismatch in assignment. Use 'f' suffix for floats.")]]
    T operator=(const U& v) noexcept
    {
        return assign_impl<T>(static_cast<T>(v));
    }

    const Proxy2A& operator=(const Proxy2A& from) = delete;
    Proxy2A(const Proxy2A& other)                 = default;

    /**
     * @brief Address-of operator is disabled to prevent unsafe access.
     */
    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
     * @brief Compute and store num_tab using share_info_array and ptr_on_array.
     */
    void set_num_tab() noexcept
    {
        // Armed by the array's operator[] (set_this_and_num_chunk with its
        // num_tab): nothing to look up. A proxy armed otherwise -- a replay
        // reconstruction, a path that only set the array pointer -- still
        // resolves it through the registry.
        if (num_tab >= 0) [[likely]]
        {
            if constexpr (build_config::any_check)
            {
                if (num_tab != share_info_array.get_num_tab(ptr_on_array))
                    INVARIANT_FAILED("Proxy2A::set_num_tab: armed num_tab {} differs from the registry's for array {}",
                                     num_tab, ptr_on_array);
            }
            return;
        }
        num_tab = share_info_array.get_num_tab(ptr_on_array);
    }

    /**
     * @brief Initialize per-chunk pointers for this proxy (must call set_num_tab first).
     */
    void set_info_on_chunk() noexcept;

    void set_ptr_second_value() noexcept;

    /// @brief This proxy's array index (ProxyBase::get_num_tab(), re-exported
    ///        past the policy layers).
    [[nodiscard]] int get_num_tab() const noexcept
    {
        return Base::get_num_tab();
    }

    /**
     * @brief Resize per-tab containers for nb_tab tabs.
     * @param nb_tab number of tabs
     */
    static void resize_tab_is_int(int nb_tab);

    /**
     * @brief Resize per-tab per-chunk containers for a given tab.
     * @param num_tab tab index
     * @param nb_chunk_old previous chunk count for the tab
     * @param nb_chunk new chunk count for the tab
     * @warning The "already queued for compression" flags of this tab are
     *          REALLOCATED, not grown: every chunk's flag is reset to 0, not
     *          only the new ones. Safe only while no chunk of the tab sits in a
     *          candidate list, which could otherwise be queued a second time.
     */
    static void resize_tab_is_int(int num_tab, int nb_chunk_old, int nb_chunk);

    /**
     * @brief Clear per-chunk arrays for a tab.
     * @param num_tab tab index
     */
    static void clear_tab_is_int(int num_tab)
    {
        BaseSpec::clear(num_tab);
        // released, not just emptied: clear() kept the capacity, so every
        // re-chunk (new storage under a new num_tab, the old one destroyed)
        // left one byte per old chunk behind.
        std::vector<char>().swap(tab_type_cst_val[num_tab]);
        // And the candidate flags (one byte per chunk), which leaked the same way.
        Base::release_tab_is_int(num_tab);
    }

    /**
     * @brief Per-tab array of flags indicating the type of constant value used.
     */
    static inline std::vector<std::vector<char>> tab_type_cst_val; ///< type-of-constant flags

    /**
     * @brief Mapping from pointer to tab information for chunk lookup.
     */
    static inline PointerMapChunk2A<T> share_info_array; ///< pointer->tab mapping

  private:
    using manager_chunk2A = ManagerChunk<Chunk2A, T>;

    using Base::read_values;
    using Base::values;

    using Base::num;
    /// Reference to the associated chunk.
    ///
    using Base::chunk; ///< owning chunk

    using Base::ptr_on_array; ///< pointer to array (for tab lookup)

    using Base::num_tab; ///< tab index for this array

    using Base::num_chunk; ///< chunk index

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using Base::set_num_tab;

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;
    static bool constexpr verbose = false;

    // Small helpers about chunk const-value modes

    /// @brief Is this proxy's chunk compressed to its second value?
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        return chunk.is_cst_val2();
    }

    // get_vec_tuple_to_compress_potentially_thr() / set_num_tab_ptr_is_inside_chunk()
    // (OpenMP) and get_vec_tuple_to_compress_potentially() (serial) live on the
    // matching Proxy2AFastCompPolicy specialisation.

    /**
     * @brief Marks the current chunk for compression if the new value requires it.
     * @param new_value The value being assigned.
     * @note Re-reads the current value from the slot; prefer
     *       mark_for_compression_if_needed_with_value() when the caller already
     *       holds it, so no concurrent writer can be read in between.
     */
    void mark_for_compression_if_needed(in_t<T> new_value) const;

    /**
     * @brief Marks chunk for potential compression with explicit old value
     * @param old_value Previous value
     * @param new_value New value being written
     */
    void mark_for_compression_if_needed_with_value(in_t<T> old_value, in_t<T> new_value) const;

    /// @brief Materialise this proxy's chunk: allocate its buffer and fill it
    ///        with the constant it was compressed to (first value, or
    ///        @p ptr_second_value). Not for bool (aborts in Debug).
    /// @note Dynamic geometry: the allocation is routed to THIS array's
    ///       chunk-size pool first, whatever the calling thread's ambient size.
    void create_value(SecondValueRef<T> ptr_second_value) const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            // Route this (possibly worker-thread) allocation to the allocator
            // for THIS array's chunk size: set the per-thread active size from
            // the array right before materializing, so parallel-region writes
            // on any worker allocate from the correct size pool.
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        if constexpr (is_not_bool)
        {
            chunk.create_value(is_cst_val2(), ptr_second_value);
        }
        else
        {
            static_assert(false, "create_value(..., second value) is for non-bool elements: use create_value_bool()");
        }
    }

    /// @brief create_value() for bool, whose second value is the chunk flag
    ///        itself. Not for other types (aborts in Debug).
    void create_value_bool() const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        if constexpr (is_bool)
        {
            chunk.create_value_bool(is_cst_val2());
        }
        else
        {
            static_assert(false, "create_value_bool() is for bool elements: use create_value(..., second value)");
        }
    }

    /**
     * @brief Initialize per-chunk pointers except second value.
     */
    void set_other_info_on_chunk() noexcept;

    void set_ptr_second_value_and_num_tab() noexcept;

    inline void trigger_compression_if_needed(int thr_rank);

    inline void perform_compression() noexcept;

    inline void perform_core_assignment(in_t<T> v) const;

    /**
     * @brief Checks if a value change should trigger compression consideration
     * @param old_value Previous value
     * @param new_value New value
     * @param second_value Current second value for comparison
     * @param first_value First constant value for comparison
     * @return true if the chunk should be marked for compression
     *
     * @details Beyond the shared predicate (a write of one of the two
     *          constants over something else), chunk2a queues two writes of
     *          its own: any non-first value on a chunk that has no second
     *          value yet (how it discovers one), and a write that LEAVES the
     *          chunk's second value (old == second, new is neither constant):
     *          the chunk may be turning uniform at a third value W, which
     *          the pass compresses (fill_list_to_compress scans against
     *          values[0] and registers what it finds). Without that clause a
     *          chunk compressed to V once and uniformly W later was never
     *          queued again, hence never compressed until it passed through
     *          first_value: 100 out of 100 in the Sept '26 probe.
     */
    ///
    /// Both Chunk2A clauses also require may_become_uniform(new_value): they
    /// used to fire on ANY such write, so a materialised chunk of varied data
    /// was queued, scanned, found not uniform, reset -- and re-queued by the
    /// next write; the writer then drove make_block_compression on the slow
    /// path again and again (~3 us per write on chunk2a_256, 64 threads).
    /// Non-static: the filter reads the chunk.
    [[nodiscard]] bool should_mark_for_compression(in_t<T> old_value,
                                                   in_t<T> new_value,
                                                   in_t<T> second_value,
                                                   in_t<T> first_value) const noexcept
    {
        if (CompressionManager<T, ChunkType>::should_mark_for_compression(old_value, new_value, second_value,
                                                                          first_value))
            return true;
        const bool discover = (second_value == first_value && new_value != first_value);
        const bool leave    = (second_value != first_value && old_value == second_value && new_value != second_value);
        if (!discover && !leave)
            return false;
        if constexpr (chunk_config::use_fast_block_compression)
        {
            // Already queued: add_to_compression_list() is a no-op, exactly as
            // before -- one relaxed load, and the reads of the filter are only
            // paid once the pass has reset the flag.
            if (this->ptr_is_inside_chunk_list_to_compress != nullptr &&
                this->ptr_is_inside_chunk_list_to_compress->load(std::memory_order_relaxed) != 0)
                return true;
        }
        return may_become_uniform(new_value);
    }

    /// @brief Could the chunk be uniform at @p v once this write lands?
    /// @details NECESSARY condition: two other elements of the chunk (the first
    ///          and the last, or their neighbours when one of them is the
    ///          element being written) already hold @p v. The write that
    ///          completes a uniform chunk always passes -- every other element
    ///          holds the value by then. Read with the scan accessor (internal
    ///          read, counted as such in CHUNK_STATS). Under concurrent writes
    ///          on the same chunk, two final writes landing exactly on the two
    ///          reference elements may both read the other's old value and both
    ///          skip: a missed hint (the chunk stays materialised, values
    ///          correct), never a wrong value.
    ///
    ///          Tolerating a stale value does not make the read race-free: the
    ///          two elements belong to other writers, so both sides are relaxed
    ///          atomics (relaxed_element.h; the stores in store_value()). A type
    ///          that cannot take them (the aggregates) skips the filter and
    ///          queues as it did before it: correct, only without the saving.
    [[nodiscard]] bool may_become_uniform(in_t<T> v) const noexcept
    {
        if constexpr (!internal::relaxed_element_access_v<T>)
        {
            return true;
        }
        else
        {
            const int S = static_cast<int>(this->chunk_size());
            if (S < 3) [[unlikely]]
                return true;
            const int i  = this->num;
            const int r0 = (i == 0) ? 1 : 0;
            const int r1 = (i == S - 1) ? S - 2 : S - 1;
            return this->chunk.read_value_scan_relaxed(r0) == v && this->chunk.read_value_scan_relaxed(r1) == v;
        }
    }

    /**
     * @brief Chunk2A-specific fast pre-check guarding the marking path.
     *
     * @details Shadows Proxy2Base::pre_check_mark_for_compression(). The strict
     *          constant-to-constant test inherited from the base only admits a
     *          chunk that is ALREADY at one constant and is being written the
     *          other one. Chunk2A's should_mark_for_compression() above accepts
     *          one further case -- a chunk whose second constant is not yet
     *          distinct from the first (`second_value == first_value`) receiving
     *          any value other than the first -- which is precisely how Chunk2A
     *          discovers its second constant and earns its advantage over
     *          Chunk2. That case is NOT a constant-to-constant transition, so
     *          the strict pre-check rejected it and the clause was unreachable:
     *          chunks were never queued, their buffers never returned to the
     *          reuse list, and memory grew several-fold.
     *
     *          Mirrors the extra clause of should_mark_for_compression() exactly,
     *          keeping the invariant stated in
     *          Proxy2Base::is_cst_val_new_value_is_other_cst_val(): the
     *          pre-check must never reject what the predicate would accept.
     *
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth examining further.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) const noexcept
    {
        // The third clause of should_mark_for_compression() (leaving the second
        // value) depends on the old value, which this filter does not see: a
        // write of a non-constant on a chunk that has a second value goes
        // through, and the predicate decides on the old value it reads anyway.
        return this->pre_check_mark_for_compression_strict(new_value, first_value, second_val) ||
               (second_val == first_value && new_value != first_value) ||
               (second_val != first_value && new_value != second_val);
    }

    /**
     * @brief Ensures the chunk has allocated storage for modification
     *
     * @details
     * If the chunk is in constant mode (cst_val1 or cst_val2), this allocates
     * a full array so individual elements can be modified.
     *
     * @pre metadata_.second_value must be initialized
     * @post Chunk has allocated storage for modifications
     */
    void ensure_writable_storage();

    template <typename ModifyFn>
    void execute_simple_operation(ModifyFn modify_fn);

    template <typename OP, typename ModifyFn, typename U>
    void execute_arithmetic_compound_operation(OP op, const U& v, ModifyFn modify_fn);

    template <typename ModifyFn>
    void execute_inplace_operation(ModifyFn modify_fn);

    inline void handle_bool_assignment(in_t<T> v);

    inline void handle_no_bool_assignment(in_t<T> v);

    /**
     * @brief Assigns a value to the proxy element.
     * @param v Value to assign.
     * @return Assigned value.
     */
    template <typename U>
    // in_t<U>, not const U&: this is the write engine, the one frame the
    // operators do NOT inline into the caller, so its parameter is what
    // crosses a call -- a char by value stays in a register, a struct goes by
    // reference. Non-deducible, hence the explicit assign_impl<U>(v) below.
    T assign_impl(in_t<U> v) noexcept;

    // ========================================================================
    // COMPOUND ARITHMETIC OPERATORS
    // ========================================================================
    template <typename U, typename Op>
    void apply_compound_operation(Op      op,
                                  in_t<U> v,
                                  bool /*check_first_value_result*/ = false) noexcept; // in_t: see assign_impl

    /**
     * @brief UNSAFE: a raw `T&` into this element's chunk buffer, for callers
     *        that must bind a real reference (e.g. passing `a[i]` to a
     *        function taking `T&`). Materialises the chunk first if it is
     *        compressed.
     * @warning A write through the returned reference bypasses the proxy:
     *          the chunk is not marked as a compression candidate, deferred
     *          ops are not replayed first, and nothing orders it against
     *          other threads. The reference dangles as soon as the chunk is
     *          recompressed or freed. Prefer the proxy's own operators.
     * @note FIX (as getVal()): it called `create_value(Base::is_cst_val())`,
     *       two compile errors at the first instantiation, latent because it
     *       has no caller: a bool passed to create_value(SecondValueRef<T>),
     *       and is_cst_val() looked up in ProxyBase instead of Proxy2Base
     *       (Base2). It materialises as the write path (set_value) does,
     *       gated on Base2::is_cst_val().
     * @return Reference to this element's live value.
     */
    [[nodiscard]] T& unsafe_ref() noexcept
    {
        if (Base2::is_cst_val())
        {
            if constexpr (is_not_bool)
                create_value(BaseSpec::ptr_second_value);
            else
                create_value_bool();
        }
        return values[num];
    }

    // static inline NanoSleeper nano_sleeper;
    char* ptr_type_cst = nullptr; ///< pointer into tab_type_cst_val[num_tab]
};

/// @brief Write the element's value, read through proxy.read_values(), to @p os.
/// @return @p os.
template <typename T>
std::ostream& operator<<(std::ostream& os, const Proxy2A<T>& proxy)
{
    // Through the proxy's read entry: indexing values[] directly prints a zero
    // for a chunk compressed to a second constant (an unbacked slot).
    os << proxy.read_values();
    return os;
}

} // namespace shrink_array

#include "chunk/internal/proxy2A_inline.h"
