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

#include <algorithm>   // std::min, std::copy_n, std::ranges::fill (the resize copy)
#include <atomic>      // std::memory_order_relaxed
#include <cstdlib>     // std::abort (debug stubs)
#include <exception>   // std::exception (zero_any::PointerMap::erase)
#include <iostream>    // std::cerr (debug stubs)
#include <memory>      // std::unique_ptr, std::addressof (operator=: operator& is deleted)
#include <type_traits> // std::is_same_v
#include <utility>     // std::pair, std::swap
#include <vector>      // std::vector
#ifndef __NVCOMPILER
    #include <concepts> // std::assignable_from
#endif

#include "chunk/vendor/ankerl/unordered_dense.h" // zero_any::PointerMap::big

#include "chunk/config.h"                     // chunk_config::use_fast_block_compression, build_config::chunk_dynamic
#include "chunk/chunk_traits.h"               // ChunkTraits<ChunkT>::ManagerType (ProxyImpl)
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size, set_thread_alloc_key
#include "chunk/internal/cst_slot.h"          // CstSlot (tab_type_cst_val)
#include "chunk/internal/concept_is_clean_assignement.h" // IsCleanAssignment
#include "chunk/internal/crp_helper.h"                   // CRTPSupport
#include "chunk/internal/in_t.h"                         // InT<T>
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/tab_cst_val2_holder.h"          // SecondValueRef
#include "chunk/internal/TChunk.h"                       // TChunk2ACompOf
#include "chunk/internal/proxy_base.h"                   // ProxyBase
#include "chunk/internal/second_value_proxy_base.h"      // SecondValueProxyBase
#include "chunk/internal/dependent_false.h"              // dependent_false_v (wrong-type branches)
#include "chunk/internal/immortal.h"                     // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/parallel_storage_init.h"        // NoInitVector, resize_filled, for_each_chunk

namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkT>
struct CompressionManager;

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace zero_any
{

// Used by zero_any::Proxy and zero_any::PointerMap.
template <typename T>
class Chunk;

/**
 * @brief Registry mapping array keys (pointers) to their table number, for
 *        zero_any: a linear-scan vector while small, switching once and for all
 *        to a hash map past SmallN entries.
 * @tparam T Element type: the keys are arrays of zero_any::Chunk<T>.
 * @tparam SmallN Entry count past which the hash map is used.
 */
template <typename T, size_t SmallN = 16>
class PointerMap
{
  public:
    /// @brief The key: the array's identity.
    using Key = ArrayKey<Chunk<T>>;

    PointerMap()                                 = default;
    ~PointerMap()                                = default;
    PointerMap(const PointerMap&)                = default;
    PointerMap(PointerMap&&) noexcept            = default;
    PointerMap& operator=(const PointerMap&)     = default;
    PointerMap& operator=(PointerMap&&) noexcept = default;

    /**
     * @brief Look up the table number registered for a key.
     * @param key Key to look up.
     * @return The registered table number.
     * @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
     */
    [[nodiscard]] int get_num_tab(Key key) const noexcept
    {
        if (!using_big_)
        {
            for (const auto& [entry_key, tab] : small_)
            {
                if (entry_key == key)
                {
                    return tab;
                }
            }
            CHUNK_INVARIANT_FAILED("[zero_any::PointerMap get_num_tab] key NOT FOUND: this={} size={} key={}", this,
                                   small_.size(), key);
        }

        if (const auto it = big_.find(key); it != big_.end())
        {
            return it->second;
        }
        CHUNK_INVARIANT_FAILED("[zero_any::PointerMap get_num_tab] key NOT FOUND: this={} size={} key={}", this,
                               big_.size(), key);
    }

    /**
     * @brief Remove the entry for `key`, if present. Idempotent. Must be called
     *        when an array is destroyed: a later array at the same address (stack
     *        arrays, e.g. rechunk_to's tmp) would otherwise get the stale num_tab
     *        from get_num_tab() (FIRST match in phase "small") and read another
     *        tab's metadata.
     * @param key Key to remove.
     */
    void erase(Key key) noexcept
    {
        if (!using_big_)
        {
            for (auto it = small_.begin(); it != small_.end(); ++it)
            {
                if (it->first == key)
                {
                    small_.erase(it);
                    return;
                }
            }
        }
        else
        {
            // ankerl's erase throws on a key it finds changed; ~zero_any::Array
            // calls this and is noexcept: fail with the key instead.
            try
            {
                big_.erase(key);
            }
            catch (const std::exception& e)
            {
                CHUNK_INVARIANT_FAILED("[zero_any::PointerMap erase] failed: {} (this={} key={})", e.what(), this, key);
            }
        }
    }

    /**
     * @brief Register a new key -> num_tab mapping; past SmallN entries, migrates
     *        everything to the hash map (one-way).
     * @param key Key to register.
     * @param num_tab Table number to associate with `key`.
     */
    void insert(Key key, int num_tab)
    {
        if (!using_big_)
        {
            small_.emplace_back(key, num_tab);

            if (small_.size() > SmallN)
            {
                using_big_ = true;
                big_.reserve(small_.size());
                for (const auto& [key, tab] : small_)
                {
                    big_[key] = tab;
                }
                small_.clear();
            }
        }
        else
        {
            big_[key] = num_tab;
        }
    }

    /**
     * @brief Re-map an EXISTING key to a new num_tab, in place. Do not use insert()
     *        for this: in phase "small" it appends a duplicate, and get_num_tab()
     *        returns the stale first match.
     * @param key Key to update.
     * @param new_num_tab New table number.
     * @return false if the key was not present (caller error); true otherwise.
     */
    bool set_num_tab(Key key, int new_num_tab) noexcept
    {
        if (!using_big_)
        {
            for (auto& [k, tab] : small_)
            {
                if (k == key)
                {
                    tab = new_num_tab;
                    return true;
                }
            }
            return false;
        }

        const auto it = big_.find(key);
        if (it == big_.end())
        {
            return false;
        }
        it->second = new_num_tab;
        return true;
    }

    /**
     * @brief Swap the num_tab values of two existing keys (rechunk_to's array swap,
     *        where two arrays exchange storage and identities).
     * @param key_a First key.
     * @param key_b Second key.
     * @return true if both keys were found (and swapped); false otherwise.
     */
    bool swap_keys(Key key_a, Key key_b) noexcept
    {
        int  na = 0;
        int  nb = 0;
        bool fa = false;
        bool fb = false;
        if (!using_big_)
        {
            for (const auto& [key, tab] : small_)
            {
                if (key == key_a)
                {
                    na = tab;
                    fa = true;
                }
                if (key == key_b)
                {
                    nb = tab;
                    fb = true;
                }
            }
            if (!fa || !fb)
            {
                return false;
            }
            for (auto& [key, tab] : small_)
            {
                if (key == key_a)
                {
                    tab = nb;
                }
                else if (key == key_b)
                {
                    tab = na;
                }
            }
            return true;
        }

        const auto ia = big_.find(key_a);
        const auto ib = big_.find(key_b);
        if (ia == big_.end() || ib == big_.end())
        {
            return false;
        }
        std::swap(ia->second, ib->second);
        return true;
    }

    /**
     * @brief Call `func` with each registered key.
     * @tparam F Callable invoked with a Key.
     * @param func Callback to invoke for every key.
     */
    template <typename F>
    void for_each_key(F&& func) const
    {
        if (!using_big_)
        {
            for (const auto& [key, tab] : small_)
            {
                func(key);
            }
        }
        else
        {
            for (const auto& [key, tab] : big_)
            {
                func(key);
            }
        }
    }

  private:
    std::vector<std::pair<Key, int>>       small_;
    ankerl::unordered_dense::map<Key, int> big_;
    bool                                   using_big_ = false;
};

} // namespace zero_any

/*!
 * \brief zero_any::ProxySpec: static per-tab, per-chunk second-value storage for zero_any::Proxy
 *        (indexed tab_second_value[num_tab][num_chunk]), so the proxy never manages
 *        the container itself.
 *
 * \tparam T Type of the second value stored in the chunk.
 */

/// @brief Reallocate @p ptr from @p old_size to @p new_size elements,
///        keeping the first min(old_size, new_size).
/// @warning Elements [old_size, new_size) are left INDETERMINATE: the caller
///          fills them (zero_any::ProxySpec<bool>::resize does). Default-initialised on
///          purpose: a zero-fill would be overwritten by the copy and the fill.
template <typename T>
void realloc_unique(std::unique_ptr<T[]>& ptr, std::size_t old_size, std::size_t new_size)
{
    std::unique_ptr<T[]> new_ptr(new T[new_size]);

    const std::size_t kept = std::min(old_size, new_size);
    if (kept != 0)
    {
        std::copy_n(ptr.get(), kept, new_ptr.get()); // a memmove for a trivial T
    }

    ptr = std::move(new_ptr);
}

namespace zero_any
{

/// @brief Per-table, per-chunk second-value storage used by zero_any::Proxy (see the description above realloc_unique).
template <typename T, typename Derived, typename ChunkT>
class ProxySpec : public SecondValueProxyBase<Derived, T, ChunkT>
{
  public:
    using Base2 = SecondValueProxyBase<Derived, T, ChunkT>;

    /// @brief What a slot holds: the value, or for a wide T the identifier
    ///        of its shared array (SecondValueTraits) -- 4 bytes per chunk
    ///        instead of 8 or 16.
    using SecondValueSlotT = SecondValueTraits<T>::SlotT;

    /// @brief One array's slots, one per chunk: same element type, filled by
    ///        a team for a large array (resize_filled).
    using SecondValueRowT = parallel_storage_init::NoInitVector<SecondValueSlotT>;

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
    [[nodiscard]] static SecondValueRowT& get_tab_second_value(int num_tab) noexcept
    {
        return tab_second_value[static_cast<std::size_t>(num_tab)];
    }

    /// @brief How many rows there are: one per num_tab ever handed out. A
    ///        released row is empty (clear()).
    [[nodiscard]] static std::size_t second_value_row_count() noexcept
    {
        return tab_second_value.size();
    }

    /*!
     * \brief Sets a specific second value in a tab and chunk.
     * \param num_tab Index of the tab.
     * \param i Index of the chunk within the tab.
     * \param v Value to assign.
     */
    static void set(int num_tab, int num_chunk, InT<T> value)
    {
        get_tab_second_value(num_tab, num_chunk).set(value);
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
        auto&     row  = tab_second_value[static_cast<std::size_t>(nb_tab)];
        const int held = static_cast<int>(row.size());
        // Keeps [0, min(old, new)); new slots value-initialised (by a team for a large array).
        parallel_storage_init::resize_filled(row, static_cast<std::size_t>(nb_chunk), SecondValueSlotT{});
        // Clear from the OLD count too: the row may hold more than nb_chunk_old entries.
        // Only up to what it held: resize_filled() initialised the rest.
        for (int k = nb_chunk_old; k < std::min(held, nb_chunk); ++k)
        {
            row[static_cast<std::size_t>(k)] = SecondValueSlotT{};
        }
    }

    /*!
     * \brief Clears all second values in a specific tab.
     * \param num_tab Index of the tab to clear.
     */
    static void clear([[maybe_unused]] int num_tab)
    {
        // Released, not just emptied: a re-chunk destroys the old storage under its own
        // num_tab, so a row left allocated leaks at every re-chunk. Swap with an empty
        // row: `= {}` would keep the capacity.
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_second_value.size())
        {
            SecondValueRowT().swap(tab_second_value[static_cast<std::size_t>(num_tab)]);
        }
    }

    SecondValueRef<T> ptr_second_value_; ///< this chunk's slot in tab_second_value[num_tab]

  private:
    /*!
     * \brief Static container holding second values for each tab and chunk.
     * Indexed as tab_second_value[num_tab][num_chunk].
     */
    static inline constinit internal::Immortal<std::vector<SecondValueRowT>> tab_second_value_storage_{};
    static inline std::vector<SecondValueRowT>& tab_second_value = tab_second_value_storage_.value;
};

} // namespace zero_any

/*!
 * \brief Specialization of zero_any::ProxySpec for bool: one flag per chunk in a plain
 *        unique_ptr<bool[]> row per tab.
 */

namespace zero_any
{

template <typename Derived, typename ChunkT>
class ProxySpec<bool, Derived, ChunkT> : public SecondValueProxyBase<Derived, bool, ChunkT>
{
  public:
    using ChunkType = ChunkT;

    using Base2 = SecondValueProxyBase<Derived, bool, ChunkType>;

    using ValuesTypeRef = SecondValueProxyBase<Derived, bool, ChunkType>::ValuesTypeRef;

    /// @brief Forward the element reference, index and chunk to SecondValueProxyBase.
    /// @param values Reference to the chunk's values.
    /// @param num Index of the element in the chunk.
    /// @param chunk Owning chunk.
    explicit ProxySpec(ValuesTypeRef values, int num, ChunkType& chunk) noexcept : Base2(values, num, chunk) {}

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
    static void set(int num_tab, int num_chunk, bool value) noexcept
    {
        tab_second_value[num_tab][num_chunk] = value;
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
        {
            std::ranges::fill(tab_second_value[nb_tab].get() + nb_chunk_old, tab_second_value[nb_tab].get() + nb_chunk,
                              false);
        }
    }

    /*!
     * \brief Clears all second values in a specific tab.
     * \param num_tab Index of the tab to clear.
     */
    static void clear([[maybe_unused]] int num_tab)
    {
        // Released, not just emptied -- see the non-bool specialisation.
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_second_value.size())
        {
            tab_second_value[static_cast<std::size_t>(num_tab)].reset();
        }
    }

  private:
    /*!
     * \brief Static container holding second values for each tab and chunk.
     * Indexed as tab_second_value[num_tab][num_chunk].
     */
    static inline constinit internal::Immortal<std::vector<std::unique_ptr<bool[]>>> tab_second_value_storage_{};
    static inline std::vector<std::unique_ptr<bool[]>>& tab_second_value = tab_second_value_storage_.value;
};

} // namespace zero_any

/**
 * @brief Policy to handle Fast Block Compression specific logic for zero_any::Proxy.
 * Encapsulates per-thread list management and trigger logic.
 */

namespace zero_any
{

template <typename Derived, typename T, typename ChunkT, bool Enabled>
class ProxyFastCompPolicy : public ProxySpec<T, Derived, ChunkT>
{
  public:
    using BaseSpec = ProxySpec<T, Derived, ChunkT>;

    using CRTPSupport<Derived>::derived;

    using ValuesTypeRef = BaseSpec::ValuesTypeRef;

    using Base  = ProxyBase<Derived, T, ChunkT>;
    using Base2 = SecondValueProxyBase<Derived, T, ChunkT>;
    using Base::chunk_;

    using BaseSpec::BaseSpec;

    // --- per-thread candidate list plumbing (OpenMP only) -----------------
    template <typename Lambda>
    void execute_simple_operation_async(auto* ctx, Lambda operation);

    /**
     * @brief Executes an in-place modification (bitwise, inc/dec) asynchronously.
     */
    template <typename Lambda>
    void execute_inplace_operation_async(auto* ctx, Lambda operation) const;

    template <typename Op, typename Body, typename U>
    void execute_arithmetic_compound_operation_async(auto* ctx, Op op, const U& operand, Body lbd) const;

    /// @brief Rank @p thr's zero_any candidate list.
    TChunk2ACompOf<ChunkT>& get_vec_tuple_to_compress_potentially_thr(int thr) const noexcept
    {
        return chunk_.get_vec_tuple_to_compress_potentially_thr(thr);
    }

    /// @brief Common initialisation of the fast path: table number, then the
    ///        "already queued" flag pointer.
    void set_num_tab_ptr_is_inside_chunk() noexcept
    {
        derived().set_num_tab(); // zero_any::ProxyImpl's no-arg overload (looks the table up)
        Base::set_ptr_is_inside_chunk_list_to_compress();
    }

  private:
    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    void ensure_writable_storage_async(const Derived& proxy) const;
};
/**
 * @brief Specialization for Disabled Fast Compression (Zero Overhead).
 */
template <typename Derived, typename T, typename ChunkT>
class ProxyFastCompPolicy<Derived, T, ChunkT, false> : public ProxySpec<T, Derived, ChunkT>
{
  public:
    using BaseSpec = ProxySpec<T, Derived, ChunkT>;

    using ValuesTypeRef = BaseSpec::ValuesTypeRef;

    using Base = ProxyBase<Derived, T, ChunkT>;
    using Base::chunk_;

    using BaseSpec::BaseSpec;

    /// @brief The single candidate list of a serial build (zero_any::ChunkThreadPolicy<false>).
    TChunk2ACompOf<ChunkT>& get_vec_tuple_to_compress_potentially() const noexcept
    {
        return chunk_.get_vec_tuple_to_compress_potentially();
    }
};

} // namespace zero_any

namespace zero_any
{

/**
 * @brief Body of the proxy to one element of a zero_any-family chunk
 *        (zero_any::Proxy, and later pattern::Proxy).
 *
 * Reads/writes one slot of a ChunkT, with fast-path/slow-path locking around
 * compression, and manages the per-chunk metadata (second values, type-of-constant
 * flags, compression bookkeeping). The tables are static members: one set per
 * instantiation, so per member of the family.
 *
 * @tparam T Element type.
 * @tparam Derived The final proxy type (CRTP), e.g. zero_any::Proxy<T>.
 * @tparam ChunkT The chunk type, e.g. zero_any::Chunk<T>.
 */
template <typename T, typename Derived, typename ChunkT>
// no move assignment on purpose -- assigning a proxy
// WRITES the element (see its operator=); a defaulted move would rebind it instead.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class ProxyImpl : public ProxyFastCompPolicy<Derived, T, ChunkT, chunk_config::use_fast_block_compression>
{
  public:
    /// @brief Stream output of the element's value (defined below).
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Proxy<U>& proxy);

    using ChunkType          = ChunkT;
    using Base2              = SecondValueProxyBase<Derived, T, ChunkType>;
    using Base               = ProxyBase<Derived, T, ChunkType>;
    using BaseFastCompPolicy = ProxyFastCompPolicy<Derived, T, ChunkType, chunk_config::use_fast_block_compression>;
    using BaseSpec           = ProxySpec<T, Derived, ChunkType>;

    friend Base;
    friend Base2;
    friend BaseFastCompPolicy;

    // Qualified: an unqualified friend template would declare a new
    // zero_any::ProxyBaseFastCompPolicy, and make_proxy_with_info's requires
    // checks on the private setters would then fail silently.
    template <typename, typename, typename, bool>
    friend class shrink_array::ProxyBaseFastCompPolicy;

    /**
     * @brief Construct the proxy of element @p num of @p chunk.
     * @param values reference to array pointer (atomic in OpenMP builds)
     * @param num index within array
     * @param chunk owning chunk
     */
    explicit ProxyImpl(BaseFastCompPolicy::ValuesTypeRef values, int num, ChunkT& chunk) noexcept
        : BaseFastCompPolicy(values, num, chunk)
    {
    }

    /** @brief Clean path for safe assignments (small operand, by value). */
    template <typename U>
        requires IsCleanAssignment<T, U>
    T operator=(const U& operand) noexcept
    {
        return assign_impl<U>(operand);
    }

    /** @brief Trap path for risky assignments (e.g., float = 1.01). */
    template <typename U>
        requires(!IsCleanAssignment<T, U>) && std::assignable_from<T&, U>
    [[deprecated("HPC WARNING: Type mismatch in assignment. Use 'f' suffix for floats.")]]
    T operator=(const U& operand) noexcept
    {
        return assign_impl<T>(static_cast<T>(operand));
    }

    /**
     * @brief Proxy-to-proxy assignment, `a[i] = b[j]`: read @p from's value
     *        through operator T() (the gated read), then write it through
     *        assign_impl() -- Proxy::operator='s meaning. The final class
     *        forwards its own copy assignment here.
     * @param from Proxy to read the value from.
     * @return *this.
     */
    const ProxyImpl& operator=(const ProxyImpl& from) noexcept
    {
        if (std::addressof(from) != this) // operator& is deleted
        {
            const InT<T> value = static_cast<T>(from); // complete before the write starts
            assign_impl<T>(value);
        }
        return *this;
    }
    ProxyImpl(const ProxyImpl& other) = default;

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
        // Armed by the array's operator[]: nothing to look up. A proxy armed otherwise
        // (replay reconstruction, only the array pointer set) resolves it through the registry.
        if (num_tab_ >= 0) [[likely]]
        {
            if constexpr (build_config::any_check)
            {
                if (num_tab_ != share_info_array->get_num_tab(ptr_on_array_))
                {
                    CHUNK_INVARIANT_FAILED(
                        "zero_any::ProxyImpl::set_num_tab: armed num_tab {} differs from the registry's for array {}",
                        num_tab_, ptr_on_array_);
                }
            }
            return;
        }
        num_tab_ = share_info_array->get_num_tab(ptr_on_array_);
    }

    /**
     * @brief Initialize per-chunk pointers for this proxy (must call set_num_tab first).
     */
    void set_info_on_chunk() noexcept;

    /// @brief Arm ptr_second_value from num_tab / num_chunk (no-op for bool).
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
     * @warning The "already queued" flags of this tab are REALLOCATED: every chunk's
     *          flag is reset to 0. Safe only while no chunk of the tab sits in a
     *          candidate list (it could be queued a second time).
     */
    static void resize_tab_is_int(int num_tab, int nb_chunk_old, int nb_chunk);

    /**
     * @brief Clear per-chunk arrays for a tab.
     * @param num_tab tab index
     */
    static void clear_tab_is_int(int num_tab)
    {
        BaseSpec::clear(num_tab);
        // Released, not just emptied (clear() keeps the capacity): every re-chunk would
        // leave one byte per old chunk behind.
        parallel_storage_init::NoInitVector<CstSlot>().swap(tab_type_cst_val[num_tab]);
        // Same for the candidate flags.
        Base::release_tab_is_int(num_tab);
    }

    /**
     * @brief Per-tab array of flags indicating the type of constant value used.
     */
    static inline constinit internal::Immortal<std::vector<parallel_storage_init::NoInitVector<CstSlot>>>
        tab_type_cst_val_storage_{}; ///< type-of-constant flags
    static inline std::vector<parallel_storage_init::NoInitVector<CstSlot>>& tab_type_cst_val =
        tab_type_cst_val_storage_.value;

    /**
     * @brief Mapping from pointer to tab information for chunk lookup.
     */
    static inline constinit internal::LazyImmortal<PointerMap<T>> share_info_array; ///< pointer->tab mapping

  private:
    using ManagerChunk2A = typename ChunkTraits<ChunkT>::ManagerType;

    using Base::read_values;
    using Base::values_;

    using Base::chunk_; ///< owning chunk
    using Base::num_;

    using Base::ptr_on_array_; ///< pointer to array (for tab lookup)

    using Base::num_tab_; ///< tab index for this array

    using Base::num_chunk_; ///< chunk index

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using Base::set_num_tab;

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;
    static bool constexpr verbose     = false;

    // Small helpers about chunk const-value modes

    /// @brief Is this proxy's chunk compressed to its second value?
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        return chunk_.is_cst_val2();
    }

    // The candidate-list accessors live on the zero_any::ProxyFastCompPolicy specialisations.

    /**
     * @brief Marks the current chunk for compression if the new value requires it.
     * @param new_value The value being assigned.
     * @param vals The chunk's values pointer, loaded by the caller inside the
     *        write session; the non-bool path reads the slot through it.
     * @note Re-reads the current value from the slot; prefer
     *       mark_for_compression_if_needed_with_value() when the caller already
     *       holds it, so no concurrent writer can be read in between.
     */
    void mark_for_compression_if_needed(InT<T> new_value, const T* vals) const;

    /**
     * @brief Marks chunk for potential compression with explicit old value
     * @param old_value Previous value
     * @param new_value New value being written
     */
    void mark_for_compression_if_needed_with_value(InT<T> old_value, InT<T> new_value) const;

    /// @brief Materialise this proxy's chunk with the constant it was compressed to
    ///        (first value, or @p ptr_second_value). Not for bool.
    /// @note Dynamic geometry: allocates from THIS array's chunk-size pool, whatever
    ///       the calling thread's ambient size.
    void create_value(SecondValueRef<T> ptr_second_value) const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            // Set the per-thread active size from THIS array right before materialising, so a
            // write on any worker allocates from the right size pool.
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        if constexpr (is_not_bool)
        {
            chunk_.create_value(is_cst_val2(), ptr_second_value);
        }
        else
        {
            static_assert(dependent_false_v<T>,
                          "create_value(..., second value) is for non-bool elements: use create_value_bool()");
        }
    }

    /// @brief create_value() for bool, whose second value is the chunk flag itself.
    void create_value_bool() const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        if constexpr (is_bool)
        {
            chunk_.create_value_bool(is_cst_val2());
        }
        else
        {
            static_assert(dependent_false_v<T>,
                          "create_value_bool() is for bool elements: use create_value(..., second value)");
        }
    }

    inline void trigger_compression_if_needed(int thr_rank);

    inline void perform_compression() noexcept;

    inline void perform_core_assignment(T* vals, InT<T> value) const;

    /**
     * @brief Checks if a value change should trigger compression consideration
     * @param old_value Previous value
     * @param new_value New value
     * @param second_value Current second value for comparison
     * @param first_value First constant value for comparison
     * @return true if the chunk should be marked for compression
     *
     * @details Beyond the shared predicate (a write of one of the two constants
     *          over something else), zero_any queues two writes of its own:
     *          - discover: any non-first value on a chunk with no second value yet;
     *          - leave: old == second, new is neither constant. The chunk may be
     *            turning uniform at a third value W, which the pass compresses
     *            (fill_list_to_compress scans against values[0]). Without this
     *            clause a chunk once compressed to V, later uniformly W, is never
     *            queued again until it passes through first_value.
     *
     *          Both clauses also require may_become_uniform(new_value): otherwise
     *          a chunk of varied data is re-queued by every write and the writer
     *          drives make_block_compression on the slow path again and again
     *          (microseconds per write). Non-static: the filter reads the chunk.
     */
    [[nodiscard]] bool should_mark_for_compression(InT<T> old_value,
                                                   InT<T> new_value,
                                                   InT<T> second_value,
                                                   InT<T> first_value) const noexcept
    {
        if (CompressionManager<T, ChunkType>::should_mark_for_compression(old_value, new_value, second_value,
                                                                          first_value))
        {
            return true;
        }
        const bool discover = (second_value == first_value && new_value != first_value);
        const bool leave    = (second_value != first_value && old_value == second_value && new_value != second_value);
        if (!discover && !leave)
        {
            return false;
        }
        if constexpr (chunk_config::use_fast_block_compression)
        {
            // Already queued: add_to_compression_list() is a no-op. One relaxed load; the
            // filter's reads are only paid once the pass has reset the flag.
            if (this->ptr_is_inside_chunk_list_to_compress_ != nullptr &&
                this->ptr_is_inside_chunk_list_to_compress_->load(std::memory_order_relaxed) != 0)
            {
                return true;
            }
        }
        return may_become_uniform(new_value);
    }

    /// @brief Could the chunk be uniform at @p value once this write lands?
    /// @details NECESSARY condition: two other elements of the chunk (the first
    ///          and the last, or their neighbours when one is the element being
    ///          written) already hold @p value, so the write completing a uniform
    ///          chunk always passes. Read with the scan accessor (counted as
    ///          internal in CHUNK_STATS). Two concurrent final writes landing on the
    ///          two reference elements may both read the other's old value and
    ///          skip: a missed hint (chunk stays materialised), never a wrong value.
    ///
    ///          Tolerating a stale value does not make the read race-free: the two
    ///          elements belong to other writers, so both sides are relaxed atomics
    ///          (relaxed_element.h; the stores in store_value()). A type that cannot
    ///          take them (the aggregates) skips the filter and always queues.
    [[nodiscard]] bool may_become_uniform(InT<T> value) const noexcept
    {
        if constexpr (!internal::relaxed_element_access_v<T>)
        {
            return true;
        }
        else
        {
            const int chunk_sz = static_cast<int>(this->chunk_size());
            if (chunk_sz < 3) [[unlikely]]
            {
                return true;
            }
            const int offset = this->num_;
            const int r0     = (offset == 0) ? 1 : 0;
            const int r1     = (offset == chunk_sz - 1) ? chunk_sz - 2 : chunk_sz - 1;
            return this->chunk_.read_value_scan_relaxed(r0) == value &&
                   this->chunk_.read_value_scan_relaxed(r1) == value;
        }
    }

    /**
     * @brief zero_any-specific fast pre-check guarding the marking path.
     *
     * @details Shadows SecondValueProxyBase::pre_check_mark_for_compression(), whose strict
     *          constant-to-constant test would reject the extra clauses of
     *          should_mark_for_compression() above (discovering the second
     *          constant is not a constant-to-constant transition): those chunks
     *          would never be queued and their buffers never reused.
     *
     *          Invariant (SecondValueProxyBase::is_cst_val_new_value_is_other_cst_val()): the
     *          pre-check must never reject what the predicate would accept.
     *
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth examining further.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression(InT<T> new_value, InT<T> first_value, InT<T> second_val) const noexcept
    {
        // The "leave" clause depends on the old value, which this filter does not see: let
        // any write of a non-second value through; the predicate decides on the old value.
        return this->pre_check_mark_for_compression_strict(new_value, first_value, second_val) ||
               (second_val == first_value && new_value != first_value) ||
               (second_val != first_value && new_value != second_val);
    }

    /**
     * @brief Ensures the chunk has allocated storage for modification
     *
     * @details Materialises a chunk in constant mode (cst_val1 or cst_val2). Arms
     * the cached metadata itself (set_info_on_chunk()), after sampling is_cst_val2().
     *
     * @post Chunk has allocated storage for modifications
     */
    void ensure_writable_storage();

    template <typename ModifyFn>
    void execute_simple_operation(ModifyFn modify_fn);

    template <typename Op, typename ModifyFn, typename U>
    void execute_arithmetic_compound_operation(Op op, const U& operand, ModifyFn modify_fn);

    template <typename ModifyFn>
    void execute_inplace_operation(ModifyFn modify_fn);

    inline void handle_bool_assignment(InT<T> value);

    inline void handle_no_bool_assignment(InT<T> value);

    /**
     * @brief Assigns a value to the proxy element.
     * @param value Value to assign.
     * @return Assigned value.
     */
    template <typename U>
    // InT<U>, not const U&: the write engine is the one frame not inlined into the caller,
    // so a small type crosses the call in a register. Non-deducible: call assign_impl<U>(v).
    T assign_impl(InT<U> value) noexcept;

    // ========================================================================
    // COMPOUND ARITHMETIC OPERATORS
    // ========================================================================
    template <typename U, typename Op>
    void apply_compound_operation(Op op, InT<U> operand) noexcept; // InT: see assign_impl

    CstSlot* ptr_type_cst_ = nullptr; ///< pointer into tab_type_cst_val[num_tab]
};

/**
 * @brief Proxy to one element of a zero_any::Chunk<T> chunk: ProxyImpl with
 *        itself as the proxy type and zero_any::Chunk<T> as the chunk type.
 *
 * @tparam T Element type.
 */
template <typename T>
// no move assignment on purpose -- assigning a proxy
// WRITES the element (see its operator=); a defaulted move would rebind it instead.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Proxy : public ProxyImpl<T, Proxy<T>, Chunk<T>>
{
    using Impl = ProxyImpl<T, Proxy<T>, Chunk<T>>;

  public:
    using Impl::Impl;
    using Impl::operator=; // the value assignments (clean and trap paths)

    /// @brief Proxy-to-proxy assignment, `a[i] = b[j]`: a WRITE of @p from's
    ///        value (ProxyImpl::operator=), not a rebinding.
    /// @return *this.
    const Proxy& operator=(const Proxy& from) noexcept
    {
        Impl::operator=(from);
        return *this;
    }
    Proxy(const Proxy& other) = default;
};

} // namespace zero_any

/// @brief Write the element's value, read through proxy.read_values(), to @p os.
/// @return @p os.
template <typename T>
std::ostream& operator<<(std::ostream& os, const zero_any::Proxy<T>& proxy)
{
    // Through the proxy's read entry: values[] directly is wrong for a chunk
    // compressed to a second constant (an unbacked slot).
    os << proxy.read_values();
    return os;
}

} // namespace shrink_array

#include "chunk/internal/zero_any_proxy_inline.h" // the template bodies
