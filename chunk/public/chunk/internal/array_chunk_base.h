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
 * @file array_chunk_base.h
 * @brief CRTP base shared by every chunked array variant (zero::Array,
 *        zero_one_value::Array, zero_any::Array).
 *
 * @details It gathers everything the three variants have in common: the
 *          index-independent core (ArrayChunkCore: geometry policy, chunk list,
 *          the dirty bitset that pre-filters the OpenMP read gate, identity),
 *          the STL-like surface (size/empty, at/front/back, assign/swap,
 *          const_iterator) and, through RechunkSupport, the generic re-chunking
 *          algorithm. Anything variant-specific is reached through derived()
 *          (CRTP) or through the `rechunk_*` hooks, whose no-op defaults live in
 *          RechunkSupport (array_chunk_rechunk.h).
 *
 *          Declarations only: the out-of-line template bodies are in
 *          array_chunk_base_inline.h, included at the end of this file.
 *
 * @see array_chunk_base_inline.h for the method bodies
 * @see chunk_size_policy.h for the geometry policy (static vs dynamic)
 */

#include <compare>  // std::strong_ordering (const_iterator's operator<=>)
#include <array>    // std::array (the bulk-read buffer)
#include <concepts> // std::invocable (for_each_span's run overload)
#include <cstddef>  // std::size_t (size_type)
#include <span>     // std::span (for_each_span)
#include <utility>  // std::swap, std::as_const
#include <iterator> // std::reverse_iterator, iterator tags — const_iterator, input_iterator (assign)

#include "chunk/config.h"                       // build_config::any_check, chunk_config::do_check
#include "chunk/internal/in_t.h"                // InT<T>
#include "chunk/chunk_traits.h"                 // ArrayChunkTraits, zero_one_value::Proxy
#include "chunk/internal/crp_helper.h"          // CRTPSupport
#include "chunk/internal/array_chunk_core.h"    // ArrayChunkCore — geometry, chunks, bitset, identity
#include "chunk/internal/array_chunk_rechunk.h" // RechunkSupport — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED — used HERE (swap check)
#include "chunk/internal/def_force_inline.h"    // CHUNK_FORCE_INLINE
#include "chunk/internal/element_ref.h"         // ElementRef: what operator[] returns
#include "chunk/internal/live_arrays.h"         // LiveArrayToken — the teardown verbs' live-array count

namespace shrink_array
{
/**
 * @brief Trait: does @p D provide a CHUNK-indexed `chunk_written(int)`?
 *
 * @details zero_one_value::Array does (zero_one_value::Chunk::written() needs the array's second-value
 *          index); for zero::Array and zero_any::Array is_written_at() asks the chunk.
 *          Do not test `written(int)` instead: every variant has it, but it takes
 *          an ELEMENT index.
 *
 *          Do not inline this as `if constexpr (requires { derived().chunk_written(j); })`
 *          in is_written_at(): nvc++ (EDG) takes the false branch for a
 *          requires-expression over a CRTP cast plus a local parameter. A free
 *          trait is evaluated alike by gcc, clang and nvc++.
 *
 * @tparam D the candidate derived array type
 * @see ArrayChunkBase::is_written_at
 */
template <typename D>
inline constexpr bool has_chunk_written_v = requires(const D& obj, int j) { obj.chunk_written(j); };

/**
 * @brief Common implementation of the chunked arrays, injected by CRTP.
 *
 * @details Storage is a vector of chunks of `chunk_size()` elements each; a
 *          global index @c i splits into `i >> chunk_decalage()` (chunk number)
 *          and `i & chunk_modulo()` (offset inside the chunk). A chunk may be
 *          constant-folded or compressed, so elements have no stable address —
 *          which is why the positional accessors below return a value on the
 *          const side and a write proxy on the non-const side rather than
 *          @c T& / @c const T&.
 *
 *          Variant-specific state is reached through derived() and the
 *          @c rechunk_* hooks. The index-independent part (geometry, chunk list,
 *          dirty bitset, registry identity) is ArrayChunkCore, the only part the
 *          proxies and registries see; this class adds m_size and the element API.
 *
 * @tparam Derived the concrete array type (zero::Array, zero_one_value::Array,
 *         zero_any::Array), which must specialise ArrayChunkTraits.
 * @see ArrayChunkTraits
 */
template <typename Derived>
class ArrayChunkBase : public CRTPSupport<Derived>,
                       public RegistryKeyCanaryBase,
                       public ArrayChunkCore<typename ArrayChunkTraits<Derived>::ChunkType>,
                       public RechunkSupport<Derived>
{
  public:
    // RechunkSupport swaps chunk_list / m_size / nb_chunk / size_policy_ with the
    // fresh array; as a base it cannot reach these protected members through a
    // Derived object without friendship (both specialisations: <typename, bool>).
    template <typename, bool>
    friend class RechunkSupport;

    // Same for RechunkHooks::rechunk_swap_identity, which swaps num_tab.
    template <typename, bool>
    friend class RechunkHooks;

    using Traits      = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using T           = Traits::value_type;        ///< Stored element type.
    using IndType     = Traits::IndexType;         ///< Signed global index type.
    using ChunkType   = Traits::ChunkType;         ///< Chunk type held in @ref chunk_list.
    using ProxyType   = Traits::ProxyType;         ///< Write proxy returned by the non-const accessors.
    using ManagerType = Traits::ManagerType;       ///< Per-type manager owning the compression machinery.
    using Core        = ArrayChunkCore<ChunkType>; ///< The index-independent part (array_chunk_core.h).

    // ── STL-vector-like member typedefs ──────────────────────────────────────
    /**
     * @name STL-vector-like member typedefs
     *
     * Deliberate divergences from std::vector, forced by the compressed storage:
     *  - reference is the write proxy, NOT T&: a raw T& would bypass the write gate;
     *  - const_reference is T, NOT const T&: an element has no stable address
     *    (its chunk may be constant, compressed or recycled);
     *  - no `pointer` / `data()`: elements are not contiguous.
     * @{
     */
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    /// non-const positional access: the write proxy, wrapped in an ElementRef
    /// (build_config::strict_element_ref; not with -DCHUNK_LOOSE_ELEMENT_REF)
    using reference       = std::conditional_t<build_config::strict_element_ref, ElementRef<ProxyType, T>, ProxyType>;
    using const_reference = T; // const positional access (value)

    /** @} */
    using CRTPSupport<Derived>::derived;

    // Re-export the mixin verbs so the container's own surface is unchanged for
    // callers (and for the derived arrays, which use them unqualified).
    using Core::alloc_dirty_bitset;
    using Core::clear_all_dirty_bitsets;
    using Core::clear_dirty_bitset;
    using Core::is_chunk_dirty;
    using Core::mark_chunk_dirty;
    using Core::reallocate_dirty_bitset;
    using Core::swap_dirty_bitset;

    using Core::alloc_key;
    using Core::chunk_count;
    using Core::chunk_ref;
    using Core::chunk_size;
    using Core::registry_key;

    using RechunkSupport<Derived>::rechunk_to;
    using RechunkSupport<Derived>::rechunk_all_to;
    using RechunkSupport<Derived>::rechunk_refill;
    using RechunkSupport<Derived>::rechunk_drop_candidates;
    using RechunkSupport<Derived>::rechunk_drain_reuse_list;
    using RechunkSupport<Derived>::rechunk_capture_state;
    using RechunkSupport<Derived>::rechunk_seed_fresh;
    using RechunkSupport<Derived>::rechunk_reown_after_swap;
    using RechunkSupport<Derived>::rechunk_swap_identity;

    // ── STL const_iterator (read-only, random access, intra-chunk cursor) ─────
    /**
     * @brief Read-only random-access iterator over the elements.
     *
     * @details Dereference yields a VALUE (T), like the const operator[]: the
     *          compressed store has no stable element address. Random access, so
     *          std::distance is O(1) and std::lower_bound & co. move in O(log n).
     *          Permuting algorithms (std::sort) are excluded by construction, as
     *          they swap through T&: sort a std::vector copy, then assign() it back.
     *
     *          Both tags are random_access, although a C++17
     *          LegacyRandomAccessIterator wants a real reference type: the classic
     *          algorithms pick their O(1) paths from iterator_category alone
     *          (std::vector<bool>::const_iterator does the same). The C++20
     *          std::random_access_iterator concept accepts a prvalue reference
     *          (iterator_validation asserts it, and that std::sortable is false).
     *
     *          Cursor, not index: the iterator keeps the chunk pointer and offset
     *          and syncs only on chunk entry, so ++ is ++offset in the common case.
     *          The read gate on entry applies EVERY op pending on the chunk (the
     *          counter is per chunk), so without a concurrent writer the whole
     *          chunk reads as the const operator[] would.
     *
     *          Jumps (+=, -=, +, -, []) go through seek(): one const operator[]
     *          read's worth. Walk with ++, not it[i] in a loop (a sync per element).
     *
     * @warning Invalidation contract = std::vector's: any resize/push_back/
     *          pop_back (reallocation) invalidates all iterators.
     * @warning Concurrency contract = std::vector's: no other thread may write
     *          the array during an iteration. A write deferred onto the current
     *          chunk after its entry is NOT seen; check builds (precise read gate)
     *          detect it at the next dereference.
     * @warning Iterators of two different arrays must not be compared or
     *          subtracted; check builds stop on it.
     */
    class const_iterator
    {
      public:
        using iterator_category = std::random_access_iterator_tag; ///< what std::lower_bound/distance dispatch on
        using iterator_concept  = std::random_access_iterator_tag; ///< C++20 concepts (std::ranges)
        using value_type        = T;                               ///< the element type
        using difference_type   = std::ptrdiff_t;                  ///< distance between two iterators
        using reference         = T;                               ///< value, not T& (see class comment)
        using pointer           = void;                            ///< no addressable element; arrow unsupported

        /** @brief Two iterators are equal when they sit on the same absolute index. */
        [[nodiscard]] friend bool operator==(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ == rhs.lin_;
        }

        // ── Random access: every jump is a seek() on lin_ ───────────────────

        /** @brief Jump @p n elements (negative: backwards). O(1), one chunk sync. */
        const_iterator& operator+=(difference_type n) noexcept
        {
            seek(lin_ + static_cast<IndType>(n));
            return *this;
        }

        /** @brief Jump back @p n elements. */
        const_iterator& operator-=(difference_type n) noexcept
        {
            seek(lin_ - static_cast<IndType>(n));
            return *this;
        }

        /** @brief The iterator @p n elements after @p it. */
        [[nodiscard]] friend const_iterator operator+(const_iterator it, difference_type n) noexcept
        {
            it += n;
            return it;
        }

        /** @brief The iterator @p n elements after @p it (n + it form). */
        [[nodiscard]] friend const_iterator operator+(difference_type n, const_iterator it) noexcept
        {
            it += n;
            return it;
        }

        /** @brief The iterator @p n elements before @p it. */
        [[nodiscard]] friend const_iterator operator-(const_iterator it, difference_type n) noexcept
        {
            it -= n;
            return it;
        }

        /** @brief Signed distance from @p rhs to @p lhs, in elements. */
        [[nodiscard]] friend difference_type operator-(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return static_cast<difference_type>(lhs.lin_) - static_cast<difference_type>(rhs.lin_);
        }

        /**
         * @brief The element @p n positions away, by VALUE (see the class comment).
         * @details A seek per call: a sync per element in a loop, where ++ syncs
         *          once per chunk.
         */
        [[nodiscard]] reference operator[](difference_type n) const noexcept
        {
            return *(*this + n);
        }

        /**
         * @brief Ordering by absolute index, like operator==. The compiler
         *        rewrites <, >, <= and >= from it (and != from operator==).
         */
        [[nodiscard]] friend std::strong_ordering operator<=>(const const_iterator& lhs,
                                                              const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ <=> rhs.lin_;
        }

        friend class ArrayChunkBase;

        /** @brief Singular iterator; only assignable and comparable. */
        const_iterator() noexcept = default;

        /**
         * @brief Read the element under the cursor.
         * @return the element VALUE (no reference exists in a compressed store)
         */
        [[nodiscard]] reference operator*() const noexcept
        {
            // chunk_ already sync'd by enter_chunk(): raw read, as chunk_t[num].
            check_no_write_since_entry();
            return (*chunk_)[num_];
        }

        /** @brief Advance one element, entering (and syncing) the next chunk on overflow. */
        const_iterator& operator++() noexcept
        {
            // lin_ is the ONLY thing operator==/!= compare: it must track the
            // cursor, or range-for runs past the logical end.
            ++lin_;
            ++num_;
            if (num_ > modulo_) [[unlikely]] // crossed a chunk boundary
            {
                num_ = 0;
                ++num_chunk_;
                enter_chunk(); // sync + repoint (skipped past the end)
            }
            return *this;
        }

        /** @brief Post-increment; prefer the prefix form on the hot path. */
        const_iterator operator++(int) noexcept
        {
            auto prev = *this;
            ++*this;
            return prev;
        }

        /** @brief Step back one element, entering (and syncing) the previous chunk on underflow. */
        const_iterator& operator--() noexcept
        {
            --lin_;                     // see operator++: lin_ is what == and != read
            if (num_ == 0) [[unlikely]] // crossed a chunk boundary backwards
            {
                --num_chunk_;
                num_ = modulo_;
                enter_chunk();
            }
            else
            {
                --num_;
            }
            return *this;
        }

        /** @brief Post-decrement; prefer the prefix form on the hot path. */
        const_iterator operator--(int) noexcept
        {
            auto prev = *this;
            --*this;
            return prev;
        }

      private:
        /**
         * @brief Build a cursor on @p owner positioned at absolute index @p lin.
         * @param owner the array being traversed
         * @param lin absolute linear index (may be m_size, i.e. end())
         */
        const_iterator(const ArrayChunkBase* owner, IndType lin) noexcept
            : owner_(owner)
            , lin_(lin)
            , num_chunk_(lin < 0 ? -1 : owner->chunk_index_of(lin))
            , num_(lin < 0 ? static_cast<int>(owner->size_policy_.modulo()) : owner->offset_in_chunk(lin))
            , modulo_(static_cast<int>(owner->size_policy_.modulo()))
        {
            enter_chunk();
        }

        /// @brief Check builds, precise read gate: nothing was deferred onto the
        ///        current chunk since enter_chunk() drained it -- i.e. no other
        ///        thread writes the array during the iteration (class contract).
        void check_no_write_since_entry() const noexcept
        {
            if constexpr (build_config::any_check && ChunkType::precise)
            {
                if (chunk_->read_gate_should_wait(num_)) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED(
                        "ArrayChunkBase::const_iterator: a write was deferred onto chunk {} of array {} "
                        "during the iteration (offset {}); the iterator would not see it -- no other "
                        "thread may write an array that is being iterated",
                        num_chunk_, owner_, num_);
                }
            }
        }

        /**
         * @brief Point chunk_ at chunk_list[num_chunk_] and sync it once.
         * @details Guarded so the end() position (one past the last element) never
         *          dereferences an out-of-range chunk: at end, num_chunk_ may index
         *          chunk_list.size().
         */
        void enter_chunk() noexcept
        {
            if (num_chunk_ >= 0 && static_cast<std::size_t>(num_chunk_) < owner_->chunk_list_.size()) [[likely]]
            {
                chunk_ = &owner_->chunk_list_[num_chunk_];
                owner_->sync_chunk(*chunk_, num_chunk_, num_); // once per chunk
            }
            else
            {
                chunk_ = nullptr; // end / rend sentinel: never dereferenced
            }
        }

        /**
         * @brief Move the cursor to absolute index @p lin: every jump comes here.
         * @details Always calls enter_chunk(), even within the same chunk: the
         *          read gate takes the offset, so a jump reads as the const
         *          operator[] does, never weaker. @p lin may be m_size (end) or -1
         *          (one before begin): the last slot of chunk -1, left null by
         *          enter_chunk().
         * @param lin absolute linear index
         */
        void seek(IndType lin) noexcept
        {
            // -1 handled apart: the chunk/offset split wants i >= 0.
            lin_       = lin;
            num_chunk_ = lin < 0 ? -1 : owner_->chunk_index_of(lin);
            num_       = lin < 0 ? modulo_ : owner_->offset_in_chunk(lin);
            enter_chunk();
        }

        /// @brief Check builds: @p lhs and @p rhs walk the same array (or are
        ///        both singular, which C++14 allows to compare).
        static void check_same_owner([[maybe_unused]] const const_iterator& lhs,
                                     [[maybe_unused]] const const_iterator& rhs) noexcept
        {
            if constexpr (build_config::any_check)
            {
                if (lhs.owner_ != rhs.owner_) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED("ArrayChunkBase::const_iterator: iterators of two different arrays compared "
                                           "or subtracted ({} vs {})",
                                           lhs.owner_, rhs.owner_);
                }
            }
        }

        const ArrayChunkBase* owner_     = nullptr; ///< Array being traversed.
        const ChunkType*      chunk_     = nullptr; ///< Current chunk, already sync'd; null at end/rend.
        IndType               lin_       = 0;       // absolute linear index (identity/compare)
        int                   num_chunk_ = 0;       ///< Index of @ref chunk_ in the chunk list.
        int                   num_       = 0;       // intra-chunk offset
        int                   modulo_    = 0;       ///< Cached geometry mask.
    };

    using iterator               = const_iterator;                        ///< No mutable iterator exists (proxy store).
    using const_reverse_iterator = std::reverse_iterator<const_iterator>; ///< Reverse read-only iterator.
    using reverse_iterator       = const_reverse_iterator;                ///< Same as const_reverse_iterator.

    T* data() = delete;
    /**
     * @brief Set the runtime chunk geometry.
     *
     * @details No-op with StaticChunkSize. With DynamicChunkSize it also
     *          registers the value allocators for @p new_chunk_size and records it
     *          as the calling thread's active size.
     *
     * @param new_chunk_size the new chunk size: a power of two or a multiple of
     *        64, at most cst_buffer_size (chunk_size_error); refused otherwise,
     *        with a warning, geometry kept
     * @warning Must be called on an empty container, before any
     *          resize/allocation, since it changes how indices map to chunks.
     */
    void set_chunk_size([[maybe_unused]] unsigned int new_chunk_size) noexcept;

    /** @brief log2(chunk_size()): the index shift of a power-of-two size, meaningless otherwise. */
    [[deprecated("chunk sizes may be multiples of 64: use i / chunk_size()")]] [[nodiscard]] unsigned int
    chunk_decalage() const noexcept
    {
        return size_policy_.decalage();
    }

    /** @brief chunk_size() - 1, the last offset in a chunk (a mask only for a power of two: use i % chunk_size()). */
    [[nodiscard]] unsigned int chunk_modulo() const noexcept
    {
        return size_policy_.modulo();
    }

    /** @brief Number of elements currently held. */
    [[nodiscard]] IndType size() const noexcept
    {
        return m_size_;
    }

    /** @brief True when the container holds no element. */
    [[nodiscard]] bool empty() const noexcept
    {
        return m_size_ == 0;
    }

    /**
     * @brief Count the number of chunks that have been modified (written).
     * Uses multi-threading if not already in a parallel region.
     * @return Total number of non-constant chunks.
     */
    [[nodiscard]] int get_nb_chunk_written() const;

    /**
     * @brief Calculate the filling rate of written chunks.
     * @return Percentage of modified chunks.
     */
    [[nodiscard]] double stats_seq() const;

    /**
     * @brief Parallel counterpart of stats_seq().
     * @return Percentage of modified chunks, 0 when the container has no chunk.
     */
    [[nodiscard]] double stats() const
    {
        return nb_chunk_ ? (100.0 * get_nb_chunk_written() / nb_chunk_) : 0.0;
    }

    // ── Capacity, and the growth by one ──────────────────────────────────────
    //
    // resize() to n allocates n / chunk_size + 1 chunks (the last is never full):
    // that spare room is the capacity. A push_back within it is ++m_size and one
    // proxied write; beyond it, resize(), a sequential point (compression pass,
    // drain of the deferred ops, tables regrown). reserve() pays that up front.
    // Sequential context, as resize().

    /// @return The element count the current chunks can hold without a resize().
    [[nodiscard]] size_type capacity() const noexcept
    {
        return static_cast<size_type>(nb_chunk_) * static_cast<size_type>(size_policy_.size());
    }

    /**
     * @brief Have room for @p new_capacity elements without a further resize().
     * @details A no-op when the capacity already covers @p new_capacity; otherwise the
     *          storage of a resize() to @p new_capacity, with the size left where it is.
     *          The new chunks are at the first constant, like any fresh chunk.
     */
    void reserve(IndType new_capacity);

    /**
     * @brief resize(), then @p val written to the NEW elements only
     *        ([old size, newsz) -- std::vector's semantics; assign() fills all).
     * @details The derived arrays' resize(newsz) does the resizing (they name
     *          this overload back with a using-declaration).
     */
    void resize(IndType newsz, InT<T> val)
    {
        const IndType old_size = m_size_;
        derived().resize(newsz);

        // A team of its own outside a region (the library's threshold, the
        // phase bracketed), the calling thread inside one.
        write_range_by_chunks(old_size, newsz, [&val](IndType) { return static_cast<T>(val); });
    }

    /**
     * @brief Append one element, growing the container by one.
     * @param val value to store at the new last position
     * @note Within the capacity: one increment and the write proxy, so the
     *       compression gate sees the write like any other. Beyond it:
     *       resize(), a sequential point (see capacity()). Invalidates all
     *       iterators.
     */
    void push_back(InT<T> val)
    {
        if (static_cast<size_type>(m_size_) < capacity()) [[likely]]
        {
            ++m_size_;
        }
        else
        {
            derived().resize(m_size_ + 1);
        }
        (*this)[m_size_ - 1] = val;
    }

    /**
     * @brief Append one element built from @p args.
     * @details push_back() of `T{args...}` (T is trivially copyable). Braces, so
     *          that an aggregate builds from its members.
     */
    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        push_back(T{std::forward<Args>(args)...});
    }

    /**
     * @name Iteration
     * Read-only ranges over the elements; all invalidated by any resize.
     * @{
     */
    /// @brief First element.
    [[nodiscard]] const_iterator begin() const noexcept
    {
        return const_iterator(this, 0);
    }

    /// @brief One past the last element.
    [[nodiscard]] const_iterator end() const noexcept
    {
        return const_iterator(this, m_size_);
    }

    /// @brief Same as begin().
    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return begin();
    }

    /// @brief Same as end().
    [[nodiscard]] const_iterator cend() const noexcept
    {
        return end();
    }

    /// @brief Last element (reverse order).
    [[nodiscard]] const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }

    /// @brief One before the first element (reverse order).
    [[nodiscard]] const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    /// @brief Same as rbegin().
    [[nodiscard]] const_reverse_iterator crbegin() const noexcept
    {
        return rbegin();
    }

    /// @brief Same as rend().
    [[nodiscard]] const_reverse_iterator crend() const noexcept
    {
        return rend();
    }

    /** @} */

    /**
     * @name Bulk reads
     * The read half of fill_range / copy_range. Call them through the free
     * functions shrink_array::for_each_value / for_each_span / read_range
     * (array_chunk_range_ops.h), which take a std::vector as well.
     * @{
     */
    /**
     * @brief Call @p f(value) on every element of [@p first, @p last), in order.
     *
     * @details One sync_chunk per chunk, then blocks of bulk_block elements
     *          copied under one read section each (ChunkBase::read_block), @p f
     *          called outside it. Stats and NaN-check builds read through the
     *          iterator, so every element still goes through the const operator[].
     * @param first first index, inclusive (clamped to [0, size()])
     * @param last  last index, exclusive (clamped to size())
     * @param f     called as f(value); its return value is ignored
     * @warning Same contract as the iterator: no other thread may write the
     *          array during the call. Check builds stop on a write deferred
     *          onto the chunk being read. Callable from a parallel region,
     *          every thread on its own slice.
     */
    template <typename F>
    void for_each_value(IndType first, IndType last, F&& f) const
    {
        if constexpr (bulk_fast)
        {
            std::array<T, bulk_block> buf;
            for_each_chunk_span(first, last,
                                [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += bulk_block)
                                    {
                                        const int n = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                        ch.read_block(j, n, buf.data());
                                        check_no_write_during_bulk_read(ch, num_chunk, j);
                                        for (int k = 0; k < n; ++k)
                                        {
                                            f(buf[k]);
                                        }
                                    }
                                });
        }
        else
        {
            if (first < 0)
            {
                first = 0;
            }
            if (last > m_size_)
            {
                last = m_size_;
            }
            if (first >= last)
            {
                return;
            }
            const const_iterator e = begin() + last;
            for (const_iterator it = begin() + first; it != e; ++it)
            {
                f(*it);
            }
        }
    }

    /// @brief for_each_value() over every element.
    template <typename F>
    void for_each_value(F&& f) const
    {
        for_each_value(IndType{0}, m_size_, std::forward<F>(f));
    }

    /**
     * @brief Call @p f(std::span<const T>) on [@p first, @p last), block by
     *        block, in order: the loop over the span can be vectorised.
     *
     * @details for_each_value() by blocks of up to bulk_block elements. When @p f
     *          is also callable as f(const T& v, std::size_t count), a compressed
     *          chunk is passed that way in O(1); other callables get it as spans.
     *          Stats and NaN-check builds fill blocks through the iterator and
     *          never call f(v, count).
     * @param first first index, inclusive (clamped to [0, size()])
     * @param last  last index, exclusive (clamped to size())
     * @param f     called as f(std::span<const T>), and f(v, count) when it
     *              accepts it; its return value is ignored. A span is only
     *              valid during its call.
     * @warning for_each_value()'s contract: no other thread may write the
     *          array during the call; callable from a parallel region, every
     *          thread on its own slice.
     */
    template <typename F>
    void for_each_span(IndType first, IndType last, F&& f) const
    {
        std::array<T, bulk_block> buf;
        if constexpr (bulk_fast)
        {
            for_each_chunk_span(first, last,
                                [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += bulk_block)
                                    {
                                        const int n = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                        if constexpr (std::invocable<F&, const T&, std::size_t>)
                                        {
                                            // A compressed chunk stays so for the rest of the
                                            // call (no writer): the constant covers [j, hi).
                                            T cst{};
                                            if (ch.read_block_or_constant(j, n, buf.data(), cst))
                                            {
                                                check_no_write_during_bulk_read(ch, num_chunk, j);
                                                f(std::as_const(cst), static_cast<std::size_t>(hi - j));
                                                return;
                                            }
                                        }
                                        else
                                        {
                                            ch.read_block(j, n, buf.data());
                                        }
                                        check_no_write_during_bulk_read(ch, num_chunk, j);
                                        f(std::span<const T>(buf.data(), static_cast<std::size_t>(n)));
                                    }
                                });
        }
        else
        {
            int n = 0;
            for_each_value(first, last,
                           [&](const T& x)
                           {
                               buf[n++] = x;
                               if (n == bulk_block)
                               {
                                   f(std::span<const T>(buf.data(), static_cast<std::size_t>(n)));
                                   n = 0;
                               }
                           });
            if (n > 0)
            {
                f(std::span<const T>(buf.data(), static_cast<std::size_t>(n)));
            }
        }
    }

    /// @brief for_each_span() over every element.
    template <typename F>
    void for_each_span(F&& f) const
    {
        for_each_span(IndType{0}, m_size_, std::forward<F>(f));
    }

    /**
     * @brief Copy the elements of [@p first, @p last) to @p out.
     * @details for_each_value() writing to @p out. Into a T*, each block is
     *          read straight into the destination: no intermediate copy.
     * @return @p out advanced past the last element written.
     */
    template <typename OutIt>
    OutIt read_range(IndType first, IndType last, OutIt out) const
    {
        if constexpr (bulk_fast && std::is_same_v<OutIt, T*>)
        {
            for_each_chunk_span(first, last,
                                [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += bulk_block)
                                    {
                                        const int n = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                        ch.read_block(j, n, out);
                                        check_no_write_during_bulk_read(ch, num_chunk, j);
                                        out += n;
                                    }
                                });
            return out;
        }
        else
        {
            for_each_value(first, last,
                           [&out](const T& x)
                           {
                               *out = x;
                               ++out;
                           });
            return out;
        }
    }
    /** @} */

    /**
     * @brief Drop the last element, if any.
     * @note Unlike std::vector, calling it on an empty container is harmless
     *       (no-op) rather than undefined. The capacity is kept, as
     *       std::vector keeps it: one decrement, no resize(). Invalidates all
     *       iterators.
     */
    void pop_back() noexcept
    {
        if (!empty())
        {
            --m_size_;
        }
    }

    /**
     * @brief What shrink_to_fit means for a container that compresses: every
     *        chunk that has become constant is folded now, and the buffers the
     *        pools keep parked for reuse are given back.
     * @details Both are process-wide (candidates and pools are per element type
     *          and rank, not per array), so this serves every array. Size and
     *          capacity do not change. Sequential context.
     */
    void shrink_to_fit();

    /**
     * @brief Checked read access.
     * @param i global element index
     * @return the element value
     * @throws std::out_of_range if @p i is negative or >= size()
     */
    [[nodiscard]] const_reference at(IndType i) const;

    /**
     * @brief Checked write access.
     * @param i global element index
     * @return the write proxy for that element
     * @throws std::out_of_range if @p i is negative or >= size()
     */
    [[nodiscard]] reference at(IndType i);

    /**
     * @name First / last element
     *
     * Unlike std::vector, these throw on an empty container (as at() does): an
     * access into an unallocated chunk would corrupt silently. Not hot-path.
     *
     * @throws std::out_of_range if the container is empty
     * @{
     */
    [[nodiscard]] const_reference front() const;
    [[nodiscard]] reference       front();
    [[nodiscard]] const_reference back() const;
    [[nodiscard]] reference       back();

    /** @} */

    /**
     * @brief assign(count, val): std::vector semantics.
     *
     * @details resize, then operator[] chunk by chunk: a team of its own for a
     *          large array outside a region (write_range_by_chunks), the
     *          calling thread otherwise.
     *
     * @param count new element count
     * @param val value written to every position
     */
    void assign(IndType count, T val); // T, not InT<T>: val may be a(i) -- resize() can free it

    /**
     * @brief assign(): the elements of [first, last), in order.
     * @details Sized once (std::distance, so a forward iterator at least),
     *          then one proxied write per element: chunk by chunk on a team of
     *          its own for a random-access source (write_range_by_chunks), in
     *          order on the calling thread for a forward one.
     */
    template <std::forward_iterator It>
    void assign(It first, It last);

    /**
     * @brief swap(): O(1) exchange of two containers.
     *
     * @details The chunk lists are exchanged with everything describing them:
     *          the identity (num_tab indexes the proxies' per-array tables and the
     *          registries map the address to it, as in rechunk_to()), the
     *          slab-affinity key and the dirty bitset. Exchanging the chunks alone
     *          would leave each array reading the other's metadata rows.
     *          Sequential context: both arrays' deferred writes are drained first
     *          (they carry the array they were queued on).
     *
     * @param other the container to exchange state with
     * @note noexcept: the drain and registry update only fail on memory
     *       exhaustion or a corrupt registry, which terminates.
     * @pre Both operands must share chunk geometry (enforced under check; with
     *      StaticChunkSize this is always true).
     */
    void swap(Derived& other) noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (size_policy_.size() != other.size_policy_.size())
            {
                CHUNK_INVARIANT_FAILED("zero::Array::swap: mismatched chunk geometry ({} vs {})", size_policy_.size(),
                                       other.size_policy_.size());
            }
        }
        swap_storage_and_identity(other);
    }

    /**
     * @brief Const read access (used by read_value()).
     * @param i global element index (unchecked)
     * @return the element value, after the read gate has been honoured
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T operator()(IndType i) const noexcept
    {
        const int   num_chunk = chunk_index_of(i);
        const int   num       = offset_in_chunk(i);
        const auto& chunk_t   = chunk_list_[num_chunk];

        sync_chunk(chunk_t, num_chunk, num);

        return chunk_t.read_value(num);
    }

    /**
     * @brief Raw reference access: arr(i) = value.
     * @param i global element index (unchecked)
     * @return a reference to the physical storage of the element
     * @warning Bypasses the write proxy: valid only where the chunk is known to
     *          be materialised. Prefer operator[] otherwise.
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T& operator()(IndType i) noexcept
    {
        const int num_chunk = chunk_index_of(i);
        const int num       = offset_in_chunk(i);
        auto&     chunk_t   = chunk_list_[num_chunk];

        sync_chunk(chunk_t, num_chunk, num);

        return chunk_t(num);
    }

    /**
     * @brief Unchecked read access (hot path).
     *
     * @details Force-inlined: GCC otherwise leaves it out-of-line, where it costs
     *          a large share of a many-thread profile. CHUNK_FORCE_INLINE is plain
     *          `inline` in debug and for nvc++; CHUNK_FLATTEN (opt-in,
     *          -DSHRINK_AGGRESSIVE_INLINE, never nvc++) pulls the whole read path in.
     *
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the element VALUE (see the member-typedef note on const_reference)
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T operator[](IndType i) const noexcept
    {
        const int num_chunk = chunk_index_of(i);
        const int num       = offset_in_chunk(i);
        // Lets the compiler drop bounds/sign checks (0 <= num <= modulo by construction).
        // modulo() is read into a local first: clang ignores an assumption that
        // calls a function it cannot prove side-effect free (-Wassume).
        [[maybe_unused]] const int chunk_mod = static_cast<int>(size_policy_.modulo());
        CHUNK_ASSUME(num_chunk >= 0);
        CHUNK_ASSUME(num >= 0 && num <= chunk_mod);
        const auto& chunk_t = chunk_list_[num_chunk];

        sync_chunk(chunk_t, num_chunk, num);

        return chunk_t[num];
    }

    /**
     * @brief Unchecked write access: THE write entry of the container.
     *
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the write proxy for that element, armed with the owning array and
     *         chunk number
     * @warning The proxy is a HANDLE into this array's chunk storage, not a
     *          value. It must not outlive an operation that rebuilds or releases
     *          that storage: resize() (push_back/emplace_back beyond the capacity
     *          included), reserve(), shrink_to_fit() (it releases buffers a
     *          proxy's `values` snapshot may point into), clear(), assign(),
     *          swap(), copy/move assignment, destruction, and any re-chunk
     *          (rechunk_to or a RechunkDecider). Keep a value: `T v = a[i];`, not
     *          `auto v = a[i];`. The compound operators and prefix ++/-- return
     *          the same kind of proxy. The ElementRef wrapper makes a use of
     *          `auto v = a[i];` fail to compile (internal/element_ref.h).
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE reference operator[](IndType i) noexcept
    {
        if constexpr (build_config::strict_element_ref)
        {
            return reference(element_proxy(i));
        }
        else
        {
            return element_proxy(i);
        }
    }

    /**
     * @brief The write proxy of element @p i, in every build: operator[]'s
     *        work, for the library code that names the proxy (operator[]
     *        wraps it in an ElementRef).
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the write proxy for that element, armed as operator[]'s
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE ProxyType element_proxy(IndType i) noexcept
    {
        const int        num_chunk = chunk_index_of(i);
        auto&            chunk_t   = chunk_list_[num_chunk];
        const int        num       = offset_in_chunk(i);
        const ProxyType& val       = chunk_t[num];

        // Arm ptr_on_array / num_chunk in ALL modes, not only under fast block
        // compression: in non-OpenMP builds too, set_num_tab() and
        // get_info_second_value() look ptr_on_array up in share_info_array.
        val.set_this_and_num_chunk(registry_key(), num_chunk, num_tab_);
        val.arm_storage_generation(&storage_generation_); // check builds only

        // zero_one_value only: arm the proxy with this array's registry mirror, so the
        // write path copies (second_value, ind, num_tab) instead of scanning
        // share_info_array. Tested on the DECLARED proxy type, not a `requires`:
        // a renamed member must fail to compile, not fall back silently.
        if constexpr (std::is_same_v<ProxyType, zero_one_value::Proxy<T>>)
        {
            val.arm_inf_array(derived().registry_mirror());
        }

        // No sync on the write entry: the read entries pull visibility
        // (ChunkBase::reader_sync), and a same-offset WAW is handled on the
        // direct write path. The const accessors and T& operator() keep sync_chunk.

        chunk_config::CheckNaN<chunk_config::do_check>::maybe(val, static_cast<long>(i));

        return val;
    }

  protected:
    // The core's members, named here so this class and the derived arrays use
    // them unqualified (the core is a dependent base).
    using Core::alloc_key_;
    using Core::assign_alloc_key;
    using Core::chunk_list_;
    using Core::nb_chunk_;
    using Core::nb_tab;
    using Core::next_alloc_key_;
    using Core::num_tab_;
    using Core::size_policy_;

    /// @brief Installs the core's way back to rechunk_to() (rechunk_erased).
    /// @note Protected, not private with `friend Derived`: zero_one_value and zero_any
    ///       reach this base through SecondValueArrayBase<Derived>, which has to
    ///       construct it too.
    ArrayChunkBase() noexcept
    {
        this->rechunk_fn_ = &rechunk_thunk;
    }

    /**
     * @name Index splitting
     * IndType (int or long) -> int chunk number and offset: the one place of
     * the narrowing, valid while size() < INT_MAX * chunk_size().
     * @{
     */
    /// @brief The chunk holding element @p i (>= 0): i / chunk_size() (SizePolicy::chunk_of).
    [[nodiscard]] int chunk_index_of(IndType i) const noexcept
    {
        return static_cast<int>(size_policy_.chunk_of(i));
    }

    /// @brief Element @p i's (>= 0) offset inside its chunk: i % chunk_size() (SizePolicy::offset_of).
    [[nodiscard]] int offset_in_chunk(IndType i) const noexcept
    {
        return static_cast<int>(size_policy_.offset_of(i));
    }

    /// @brief Chunks allocated for @p n elements: n / chunk_size() + 1 (the last
    ///        chunk is never full -- see capacity()).
    [[nodiscard]] int chunks_for(IndType n) const noexcept
    {
        return chunk_index_of(n) + 1;
    }

    /** @} */

    /**
     * @brief Creation steps common to every variant, run from the derived constructors.
     * @details Resets the chunk counter, re-arms the NaN checker, calls the
     *          variant's pre_create_hook() and sizes the proxy's per-array tables
     *          for this array's identity.
     */
    void create_common();

    /**
     * @brief Claim this array's identity and build its storage for @p sz
     *        elements: the constructors' body of zero::Array and zero_any::Array
     *        (zero_one_value::Array has its own). Registers the element type's manager on
     *        the way (AllTypeChunk::ensure_manager), runs create_common(), then
     *        the variant's rebuild_storage() -- the geometry-dependent
     *        allocation alone, which rechunk_to also calls on an already
     *        registered, just torn-down array WITHOUT re-running create_common()
     *        (that would re-register it under a fresh num_tab via pre_create_hook).
     */
    void create(IndType sz);

    /**
     * @brief The deep copy the derived arrays' copy constructor and
     *        copy-assignment are built on: sized to @p from, then every
     *        element read from @p from (a const read, through its gate) and
     *        written here through the proxy.
     *
     * @details Element by element on purpose, like rechunk_refill: correct for
     *          every variant, any geometry pair, and a non-empty destination.
     *          Repeated values are written materialised and folded back by the
     *          next compression pass. Called at a sequential point (resize() is
     *          one); the writes go chunk by chunk on a team of its own for a
     *          large array (write_range_by_chunks). zero_one_value's second value is
     *          re-declared first by the derived copy hooks.
     */
    void copy_from(const Derived& from);

    /**
     * @brief Elements [@p first, @p last) set to @p value_at(i), through the
     *        proxy, chunk by chunk.
     *
     * @details A team of its own when parallel_storage_init::use_own_team()
     *          accepts the chunk count (outside a region, from
     *          parallel_init_min_chunks chunks), each chunk written by one
     *          thread; the calling thread otherwise -- inside a region an
     *          orphaned `omp for` would write 1/N of the range. The team goes
     *          through parallel_phase(): without OMPT its threads begin the
     *          phase before writing and the deferred writes are flushed after
     *          it, as for any region that writes.
     * @param first    first index, inclusive
     * @param last     last index, exclusive (<= size())
     * @param value_at called as value_at(i); independent iterations, may run
     *                 on any thread of the team
     */
    template <typename ValueAt>
    void write_range_by_chunks(IndType first, IndType last, ValueAt&& value_at);

    /**
     * @brief The exchange behind swap() and the derived move-assignments:
     *        storage, geometry, affinity key, dirty bitset, then the identity
     *        (Derived::swap_identity()). No geometry precondition -- the
     *        geometry is exchanged with the chunks, as rechunk_to() does.
     */
    void swap_storage_and_identity(Derived& other)
    {
        if (&derived() == &other)
        {
            return;
        }
        derived().flush_pending_deferred_ops(/*force=*/true);
        other.flush_pending_deferred_ops(/*force=*/true);
        invalidate_proxies();
        other.invalidate_proxies();

        using std::swap;
        swap(chunk_list_, other.chunk_list_);
        swap(alloc_key_, other.alloc_key_);
        swap_dirty_bitset(other);
        swap(m_size_, other.m_size_);
        swap(nb_chunk_, other.nb_chunk_);
        swap(size_policy_, other.size_policy_);
        derived().swap_identity(other);
    }

    IndType m_size_; ///< Number of elements exposed by size().

    /// @brief Bumped by every operation that invalidates a proxy (see operator[]).
    ///        Check builds' proxies abort on use when it changed
    ///        (ProxyBase::check_not_stale). Not swapped: a swap invalidates both.
    unsigned int storage_generation_ = 0;

    /// @brief Counts this array among the live ones (live_arrays.h): the
    ///        teardown verbs keep the pools while any array is alive.
    [[no_unique_address]] internal::LiveArrayToken live_token_;

    /// @brief Every proxy of this array built so far is now stale.
    void invalidate_proxies() noexcept
    {
        ++storage_generation_;
    }

    /// @brief clear() takes the end-of-life census (print_compression_stats()).
    ///        False on a re-chunk's tmp: its state depends on the build (OpenMP
    ///        releases the source chunks during the copy, serial does not), so
    ///        counting it would make still-at-exit differ between the two.
    bool exit_census_ = true;

    /**
     * @brief Read the element at @p i.
     * @param i global element index (unchecked)
     * @return the element value
     * @note Protected, not private: zero_any::Array::operator== reads both of its
     *       operands through it.
     */
    T read_value(IndType i) const noexcept
    {
        return (*this)(i);
    }

  private:
    /// @brief Elements per read section in the bulk reads (8 to 64 measure
    ///        alike; the larger, the fewer sections on long ranges).
    static constexpr int bulk_block = 64;
    /// @brief Bulk reads by block only where per-element stats and NaN checks
    ///        do not rely on every element going through operator[].
    static constexpr bool bulk_fast = !chunk_config::stats && !chunk_config::do_check;

    /// @brief Call @p fn(chunk, num_chunk, lo, hi) for each chunk the range
    ///        [first, last) touches -- offsets [lo, hi) in it -- after one
    ///        sync_chunk, as the iterator does on chunk entry.
    template <typename Fn>
    void for_each_chunk_span(IndType first, IndType last, Fn&& fn) const
    {
        if (first < 0)
        {
            first = 0;
        }
        if (last > m_size_)
        {
            last = m_size_;
        }
        if (first >= last)
        {
            return;
        }
        const int c_first = chunk_index_of(first);
        const int c_last  = chunk_index_of(last - 1);
        const int full    = static_cast<int>(size_policy_.size());
        for (int c = c_first; c <= c_last; ++c)
        {
            const int        lo = c == c_first ? offset_in_chunk(first) : 0;
            const int        hi = c == c_last ? offset_in_chunk(last - 1) + 1 : full;
            const ChunkType& ch = chunk_list_[static_cast<std::size_t>(c)];
            sync_chunk(ch, c, lo);
            fn(ch, c, lo, hi);
        }
    }

    /// @brief Check builds, precise read gate: nothing was deferred onto
    ///        the chunk since its sync -- the iterator's check, per block.
    void check_no_write_during_bulk_read([[maybe_unused]] const ChunkType& ch,
                                         [[maybe_unused]] int              num_chunk,
                                         [[maybe_unused]] int              offset) const noexcept
    {
        if constexpr (build_config::any_check && ChunkType::precise)
        {
            if (ch.read_gate_should_wait(offset)) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("ArrayChunkBase bulk read: a write was deferred onto chunk {} of array {} "
                                       "during the read (offset {}); it would not be seen -- no other thread may "
                                       "write an array that is being read in bulk",
                                       num_chunk, this, offset);
            }
        }
    }

    /**
     * @brief Is chunk @p j materialised?
     * @details Routes to Derived::chunk_written(j) when it exists (zero_one_value::Array,
     *          whose chunks need the second-value index), else asks
     *          chunk_list[j].written() directly.
     * @param j CHUNK index
     * @return true if chunk @p j has been written (i.e. is not constant)
     * @see has_chunk_written_v
     */
    [[nodiscard]] bool is_written_at(int j) const noexcept;

    /**
     * @brief Sequential counterpart of get_nb_chunk_written().
     * @return Total number of non-constant chunks, counted on one thread.
     */
    [[nodiscard]] int get_nb_chunk_written_seq() const;

    /**
     * @brief Centralized synchronization logic.
     * @param chunk_t Reference to the actual chunk for OMP checks.
     * @param num_chunk Index of the chunk for bitset lookup.
     * @param offset Offset within the chunk.
     * @note Compiles to nothing in a serial build (`if constexpr (threaded)`).
     */
    CHUNK_FORCE_INLINE void sync_chunk(const ChunkType& chunk_t, int num_chunk, int offset) const noexcept
    {
        if constexpr (build_config::threaded)
        {
            // Coarse pre-filter: the deferred-enqueue path marks this bit with
            // the CHUNK index (proxy_base_inline.h), so a clear bit means no
            // deferred op was ever queued for this chunk: skip the gate. This
            // keeps the hot read path cheap. The bit is sticky until
            // clear_dirty_bitset(); the authoritative gate is the chunk's own
            // pending counter, below.
            if (derived().is_chunk_dirty(num_chunk)) [[unlikely]]
            {
                // precise per-chunk gate (DeferredReadGate, chunk_base.h).
                if (chunk_t.read_gate_should_wait(offset)) [[unlikely]]
                {
                    chunk_t.wait_for_availability(offset);
                }
            }
        }
    }

    /// @brief The core's rechunk_fn_: casts @p core back to Derived and calls rechunk_to(@p new_size).
    static void rechunk_thunk(Core& core, unsigned int new_size)
    {
        // Safe: rechunk_fn_ is only set by this class's ctor, inside a Derived.
        static_cast<Derived&>(core).rechunk_to(new_size);
    }
};
} // namespace shrink_array

#include "chunk/internal/array_chunk_base_inline.h"    // the template bodies
#include "chunk/internal/array_chunk_rechunk_inline.h" // the RechunkSupport bodies (CHUNK_DYNAMIC)
