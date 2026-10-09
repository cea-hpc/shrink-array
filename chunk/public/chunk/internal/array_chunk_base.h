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
 * @brief CRTP base shared by every chunked array variant (ArrayChunk,
 *        ArrayChunk2, ArrayChunk2A).
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

#include <concepts> // std::invocable (for_each_span's run overload)
#include <cstddef>  // std::size_t (size_type)
#include <span>     // std::span (for_each_span)
#include <utility>  // std::swap, std::as_const
#include <iterator> // std::reverse_iterator, iterator tags — const_iterator, input_iterator (assign)

#include "chunk/config.h"                       // build_config::any_check, chunk_config::do_check
#include "chunk/internal/in_t.h"                // in_t<T>
#include "chunk/chunk_traits.h"                 // ArrayChunkTraits, Proxy2
#include "chunk/internal/crp_helper.h"          // CRTPSupport
#include "chunk/internal/array_chunk_core.h"    // ArrayChunkCore — geometry, chunks, bitset, identity
#include "chunk/internal/array_chunk_rechunk.h" // RechunkSupport — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED — used HERE (swap check)
#include "chunk/internal/def_force_inline.h"    // CHUNK_FORCE_INLINE
#include "chunk/internal/element_ref.h"         // ElementRef: operator[] in check builds
#include "chunk/internal/live_arrays.h"         // LiveArrayToken — the teardown verbs' live-array count

namespace shrink_array
{
/**
 * @brief Trait: does @p D provide a CHUNK-indexed `chunk_written(int)`?
 *
 * @details ArrayChunk2 does (Chunk2::written() needs the array's second-value
 *          index); ArrayChunk and ArrayChunk2A do not, and is_written_at()
 *          asks the chunk directly for them.
 *
 *          Not `written(int)`: every variant has that overload, but it takes an
 *          ELEMENT index -- fed a chunk index, it tests elements of chunk 0 and
 *          reports chunks written that are not.
 *
 *          NOTE: deliberately NOT written inline as
 *          @code if constexpr (requires { derived().chunk_written(j); }) @endcode
 *          inside is_written_at(): nvc++ (EDG) mis-evaluates a requires-expression
 *          that references a CRTP-cast (`derived()`) plus a captured local
 *          parameter, taking the FALSE branch even when the overload exists.
 *          A free trait with explicit declval'd parameters is standard SFINAE
 *          that all three frontends (gcc/llvm/nvc++) evaluate identically.
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
 *          Everything variant-specific (deferred compression state, second
 *          value, registry identity) is reached through derived(); the
 *          @c rechunk_* hooks give each variant a place to plug into the shared
 *          re-chunking algorithm.
 *
 *          What does not depend on the index type -- geometry, chunk list,
 *          dirty bitset, registry identity -- is ArrayChunkCore, the one part of
 *          the array the proxies and the per-T registries see (see
 *          array_chunk_core.h). This class adds the index-typed part: m_size and
 *          the element API.
 *
 * @tparam Derived the concrete array type (ArrayChunk, ArrayChunk2,
 *         ArrayChunk2A), which must specialise ArrayChunkTraits.
 * @see ArrayChunkTraits
 */
template <typename Derived>
class ArrayChunkBase : public CRTPSupport<Derived>,
                       public RegistryKeyCanaryBase,
                       public ArrayChunkCore<typename ArrayChunkTraits<Derived>::chunk_type>,
                       public RechunkSupport<Derived>
{
  public:
    // RechunkSupport rebuilds the container from the outside — it swaps
    // chunk_list / m_size / nb_chunk / size_policy_ between *this and the fresh
    // array at the new geometry. It is a BASE of this class, so inheritance
    // alone gives it no access to these protected members through a Derived
    // object; the friendship does. Both specialisations are named, hence the
    // <typename, bool> form.
    template <typename, bool>
    friend class RechunkSupport;

    // Same reason for v1's re-chunk hooks: RechunkHooks::rechunk_swap_identity
    // swaps num_tab, which lives here and is protected.
    template <typename, bool>
    friend class RechunkHooks;

    using Traits       = ArrayChunkTraits<Derived>;  ///< Traits specialisation of the concrete array.
    using T            = Traits::value_type;         ///< Stored element type.
    using ind_type     = Traits::index_type;         ///< Signed global index type.
    using chunk_type   = Traits::chunk_type;         ///< Chunk type held in @ref chunk_list.
    using proxy_type   = Traits::proxy_type;         ///< Write proxy returned by the non-const accessors.
    using manager_type = Traits::manager_type;       ///< Per-type manager owning the compression machinery.
    using Core         = ArrayChunkCore<chunk_type>; ///< The index-independent part (array_chunk_core.h).

    // ── STL-vector-like member typedefs ──────────────────────────────────────
    /**
     * @name STL-vector-like member typedefs
     *
     * Provided for generic code that inspects Container::value_type etc. NOTE
     * the deliberate divergences from std::vector, forced by the compressed,
     * gate-driven storage (see operator[] and front/back below):
     *  - reference is proxy_type, NOT T&: a non-const positional access returns
     *    the write proxy that drives deferred compression and the write gate. A
     *    raw T& would bypass the gate.
     *  - const_reference is T (a value), NOT const T&: a const access returns the
     *    materialised value after sync_chunk; there is no stable address to
     *    reference (a chunk may be cst / compressed / recycled).
     *  - there is intentionally no `pointer` / `data()`: elements are not
     *    contiguous and may be constant-folded, so a raw pointer cannot exist.
     * @{
     */
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    /// non-const positional access: the write proxy, wrapped in an ElementRef in
    /// check builds (build_config::strict_element_ref)
    using reference = std::conditional_t<build_config::strict_element_ref, ElementRef<proxy_type, T>, proxy_type>;
    using const_reference = T;          // const positional access (value)

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
     * @details Read-only by design: dereference yields a VALUE (T), like the
     *          const operator[], never a T&/proxy — the compressed store has no
     *          stable element address, and a writing iterator would need a proxy
     *          reference that breaks permuting algorithms anyway. This iterator
     *          makes the read half of the STL work: range-for,
     *          std::copy/accumulate/find/equal/for_each, reverse traversal
     *          (rbegin/rend) and, being random access, it[n], it2 - it1,
     *          std::distance in O(1), and std::lower_bound / upper_bound /
     *          equal_range / binary_search in O(log n) moves, not only O(log n)
     *          comparisons.
     *
     *          std::sort and the other permuting algorithms stay out BY
     *          CONSTRUCTION: they swap through T&, and an element that may live
     *          in a chunk folded to a constant has no address. Sort a copy:
     *          std::vector<T> v(a.begin(), a.end()); std::sort(v.begin(), v.end());
     *          a.assign(v.begin(), v.end()).
     *
     *          Both tags are random_access. Formally a C++17
     *          LegacyRandomAccessIterator wants a real reference type, but the
     *          classic algorithms (std::lower_bound, std::distance, std::advance)
     *          pick their O(1) paths from iterator_category alone -- with
     *          iterator_concept only, just the std::ranges algorithms would --
     *          and std::vector<bool>::const_iterator does the same with
     *          reference = bool. The C++20 std::random_access_iterator concept
     *          accepts a prvalue reference as is (iterator_validation asserts it,
     *          and that std::sortable is false).
     *
     *          Cursor, not index: a naive iterator would recompute
     *          (i>>decalage, i&modulo) and run sync_chunk on EVERY dereference.
     *          Instead we keep the current chunk pointer + intra-chunk offset and
     *          only cross to the next chunk when the offset overflows — so ++ is
     *          a trivial ++offset in the common case, and sync_chunk is amortised
     *          once per chunk (called on chunk entry). On entry the read gate
     *          applies EVERY op pending on the chunk (the counter is per chunk,
     *          not per offset), so with no concurrent writer the whole chunk then
     *          reads exactly as the const operator[] would, 1/chunk_size as
     *          often.
     *
     *          Jumps (+=, -=, +, -, []) go through seek(): a shift, a mask and a
     *          sync of the target chunk -- one const operator[] read's worth.
     *          O(1), not free: walk with ++, not with it[i] in a loop, which
     *          pays a sync per element where ++ pays one per chunk.
     *
     * @warning Invalidation contract = std::vector's: any resize/push_back/
     *          pop_back (reallocation) invalidates all iterators.
     * @warning Concurrency contract = std::vector's too: no other thread may
     *          write the array while an iteration over it is under way. A write
     *          deferred onto the current chunk after its entry is NOT seen by the
     *          rest of the chunk (the const operator[] has the same window, on one
     *          element). Check builds (build_config::any_check, precise read gate)
     *          detect it at the next dereference.
     * @warning Iterators of two different arrays must not be compared or
     *          subtracted (undefined, as for std containers); check builds stop
     *          on it.
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

        /** @brief Negation of operator==. */
        [[nodiscard]] friend bool operator!=(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ != rhs.lin_;
        }

        // ── Random access: every jump is a seek() on lin_ ───────────────────

        /** @brief Jump @p n elements (negative: backwards). O(1), one chunk sync. */
        const_iterator& operator+=(difference_type n) noexcept
        {
            seek(lin_ + static_cast<ind_type>(n));
            return *this;
        }

        /** @brief Jump back @p n elements. */
        const_iterator& operator-=(difference_type n) noexcept
        {
            seek(lin_ - static_cast<ind_type>(n));
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

        /** @brief Orderings: by absolute index, like operator==. */
        [[nodiscard]] friend bool operator<(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ < rhs.lin_;
        }

        /** @brief Ordering by absolute index. */
        [[nodiscard]] friend bool operator>(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ > rhs.lin_;
        }

        /** @brief Ordering by absolute index. */
        [[nodiscard]] friend bool operator<=(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ <= rhs.lin_;
        }

        /** @brief Ordering by absolute index. */
        [[nodiscard]] friend bool operator>=(const const_iterator& lhs, const const_iterator& rhs) noexcept
        {
            check_same_owner(lhs, rhs);
            return lhs.lin_ >= rhs.lin_;
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
            // chunk_ already sync'd on entry (see enter_chunk(), reached by seek() and ++);
            // this is the raw per-element read, identical to chunk_t[num].
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
        const_iterator(const ArrayChunkBase* owner, ind_type lin) noexcept
            : owner_(owner)
            , lin_(lin)
            , num_chunk_(static_cast<int>(lin >> owner->size_policy_.decalage()))
            , num_(static_cast<int>(lin & owner->size_policy_.modulo()))
            , modulo_(static_cast<int>(owner->size_policy_.modulo()))
            , decalage_(owner->size_policy_.decalage())
        {
            enter_chunk();
        }

        /// @brief Check builds, precise read gate: nothing was deferred onto the
        ///        current chunk since enter_chunk() drained it -- i.e. no other
        ///        thread writes the array during the iteration (class contract).
        void check_no_write_since_entry() const noexcept
        {
            if constexpr (build_config::any_check && chunk_type::precise)
            {
                if (chunk_->read_gate_should_wait(num_)) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED(
                        "ArrayChunkBase::const_iterator: a write was deferred onto chunk {} of array {} "
                        "during the iteration (offset {}); the iterator would not see it -- no other "
                        "thread may write an array that is being iterated",
                        num_chunk_, static_cast<const void*>(owner_), num_);
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
            if (num_chunk_ >= 0 && static_cast<std::size_t>(num_chunk_) < owner_->chunk_list.size()) [[likely]]
            {
                chunk_ = &owner_->chunk_list[num_chunk_];
                owner_->sync_chunk(*chunk_, num_chunk_, num_); // once per chunk
            }
            else
            {
                chunk_ = nullptr; // end / rend sentinel: never dereferenced
            }
        }

        /**
         * @brief Move the cursor to absolute index @p lin: every jump comes here.
         * @details Chunk and offset recomputed from @p lin (shift, mask), then
         *          enter_chunk() -- ALWAYS, even when the jump stays in the same
         *          chunk: the read gate of sync_chunk takes the offset, so a jump
         *          reads exactly as the const operator[] does, never weaker.
         *          @p lin may be m_size (end) or -1 (one before begin, which --
         *          reaches too): the arithmetic shift gives chunk -1 there, and
         *          enter_chunk() leaves chunk_ null outside the list.
         * @param lin absolute linear index
         */
        void seek(ind_type lin) noexcept
        {
            lin_       = lin;
            num_chunk_ = static_cast<int>(lin >> decalage_);
            num_       = static_cast<int>(lin & static_cast<ind_type>(modulo_));
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
                                           static_cast<const void*>(lhs.owner_), static_cast<const void*>(rhs.owner_));
                }
            }
        }

        const ArrayChunkBase* owner_     = nullptr; ///< Array being traversed.
        const chunk_type*     chunk_     = nullptr; ///< Current chunk, already sync'd; null at end/rend.
        ind_type              lin_       = 0;       // absolute linear index (identity/compare)
        int                   num_chunk_ = 0;       ///< Index of @ref chunk_ in the chunk list.
        int                   num_       = 0;       // intra-chunk offset
        int                   modulo_    = 0;       ///< Cached geometry mask.
        unsigned int          decalage_  = 0;       ///< Cached geometry shift.
    };

    using iterator               = const_iterator;                        ///< No mutable iterator exists (proxy store).
    using const_reverse_iterator = std::reverse_iterator<const_iterator>; ///< Reverse read-only iterator.
    using reverse_iterator       = const_reverse_iterator;                ///< Same as const_reverse_iterator.

    T* data() = delete;
    /**
     * @brief Set the runtime chunk geometry.
     *
     * @details Only meaningful with DynamicChunkSize; with StaticChunkSize the
     *          geometry is fixed at compile time and this is a no-op (guarded by
     *          if constexpr so it compiles in both modes). In a dynamic build it
     *          also registers the value allocators for @p new_chunk_size and records @p new_chunk_size as
     *          the calling thread's active size.
     *
     * @param new_chunk_size the new chunk size (power of two)
     * @warning Must be called on an empty container, before any
     *          resize/allocation, since it changes how indices map to chunks.
     */
    void set_chunk_size([[maybe_unused]] unsigned int new_chunk_size) noexcept;

    /** @brief Right shift turning a global index into a chunk index: `i >> chunk_decalage()`. */
    [[nodiscard]] unsigned int chunk_decalage() const noexcept
    {
        return size_policy_.decalage();
    }

    /** @brief Mask extracting the offset inside a chunk: `i & chunk_modulo()`. */
    [[nodiscard]] unsigned int chunk_modulo() const noexcept
    {
        return size_policy_.modulo();
    }

    /** @brief Number of elements currently held. */
    [[nodiscard]] ind_type size() const noexcept
    {
        return m_size;
    }

    /** @brief True when the container holds no element. */
    [[nodiscard]] bool empty() const noexcept
    {
        return m_size == 0;
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
        return nb_chunk ? (100.0 * get_nb_chunk_written() / nb_chunk) : 0.0;
    }

    // ── Capacity, and the growth by one ──────────────────────────────────────
    //
    // A container is nb_chunk chunks of chunk_size() elements, and the last
    // chunk is never full: resize() to n allocates n / chunk_size + 1 of them.
    // That spare room is the capacity, exactly as std::vector's. A push_back
    // within it is ++m_size and one proxied write; only the push_back that
    // needs a new chunk goes through resize(), which is a sequential point
    // (a compression pass of the candidates, a drain of the deferred ops, the
    // chunk list and the per-chunk tables regrown). reserve() moves that cost
    // up front, once. Sequential context, as resize().

    /// @return The element count the current chunks can hold without a resize().
    [[nodiscard]] size_type capacity() const noexcept
    {
        return static_cast<size_type>(nb_chunk) * static_cast<size_type>(size_policy_.size());
    }

    /**
     * @brief Have room for @p new_capacity elements without a further resize().
     * @details A no-op when the capacity already covers @p new_capacity; otherwise the
     *          storage of a resize() to @p new_capacity, with the size left where it is.
     *          The new chunks are at the first constant, like any fresh chunk.
     */
    void reserve(ind_type new_capacity);

    /**
     * @brief Append one element, growing the container by one.
     * @param val value to store at the new last position
     * @note Within the capacity: one increment and the write proxy, so the
     *       compression gate sees the write like any other. Beyond it:
     *       resize(), a sequential point (see capacity()). Invalidates all
     *       iterators.
     */
    void push_back(in_t<T> val)
    {
        if (static_cast<size_type>(m_size) < capacity()) [[likely]]
        {
            ++m_size;
        }
        else
        {
            derived().resize(m_size + 1);
        }
        (*this)[m_size - 1] = val;
    }

    /**
     * @brief Append one element built from @p args.
     * @details An element type is trivially copyable, so this is push_back()
     *          of a `T{args...}` -- here for the interface, not for a move it
     *          could save. Braces, so that an aggregate (a struct of doubles)
     *          builds from its members.
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
        return const_iterator(this, m_size);
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
     * @details Chunk by chunk: one sync_chunk per chunk (as the iterator on
     *          chunk entry), then blocks of kBulkBlock elements copied under
     *          one read section each (ChunkBase::read_block), @p f called
     *          outside it. A range-for pays a section and the iterator's
     *          bookkeeping per element.
     *
     *          Stats and NaN-check builds read through the iterator instead,
     *          so every element still goes through the const operator[].
     * @param first first index, inclusive (clamped to [0, size()])
     * @param last  last index, exclusive (clamped to size())
     * @param f     called as f(value); its return value is ignored
     * @warning Same contract as the iterator: no other thread may write the
     *          array during the call. Check builds stop on a write deferred
     *          onto the chunk being read. Callable from a parallel region,
     *          every thread on its own slice.
     */
    template <typename F>
    void for_each_value(ind_type first, ind_type last, F&& f) const
    {
        if constexpr (kBulkFast)
        {
            T buf[kBulkBlock];
            for_each_chunk_span(first, last,
                                [&](const chunk_type& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += kBulkBlock)
                                    {
                                        const int n = (hi - j) < kBulkBlock ? (hi - j) : kBulkBlock;
                                        ch.read_block(j, n, buf);
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
            if (last > m_size)
            {
                last = m_size;
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
        for_each_value(ind_type{0}, m_size, std::forward<F>(f));
    }

    /**
     * @brief Call @p f(std::span<const T>) on [@p first, @p last), block by
     *        block, in order: the loop over the span can be vectorised.
     *
     * @details for_each_value() handing over each block of up to kBulkBlock
     *          elements instead of its elements one by one.
     *
     *          When @p f is also callable as f(const T& v, std::size_t count),
     *          a compressed chunk is passed that way: its constant and the
     *          number of elements of the range it covers, in O(1), nothing
     *          copied -- a sum adds v * count. Other callables get the
     *          constant as spans, like any block.
     *
     *          Stats and NaN-check builds fill each block through the
     *          iterator, so every element still goes through the const
     *          operator[], and never call f(v, count).
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
    void for_each_span(ind_type first, ind_type last, F&& f) const
    {
        T buf[kBulkBlock];
        if constexpr (kBulkFast)
        {
            for_each_chunk_span(first, last,
                                [&](const chunk_type& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += kBulkBlock)
                                    {
                                        const int n = (hi - j) < kBulkBlock ? (hi - j) : kBulkBlock;
                                        if constexpr (std::invocable<F&, const T&, std::size_t>)
                                        {
                                            // A compressed chunk stays so for the rest of the
                                            // call (no writer): the constant covers [j, hi).
                                            T cst{};
                                            if (ch.read_block_or_constant(j, n, buf, cst))
                                            {
                                                check_no_write_during_bulk_read(ch, num_chunk, j);
                                                f(std::as_const(cst), static_cast<std::size_t>(hi - j));
                                                return;
                                            }
                                        }
                                        else
                                        {
                                            ch.read_block(j, n, buf);
                                        }
                                        check_no_write_during_bulk_read(ch, num_chunk, j);
                                        f(std::span<const T>(buf, static_cast<std::size_t>(n)));
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
                               if (n == kBulkBlock)
                               {
                                   f(std::span<const T>(buf, static_cast<std::size_t>(n)));
                                   n = 0;
                               }
                           });
            if (n > 0)
            {
                f(std::span<const T>(buf, static_cast<std::size_t>(n)));
            }
        }
    }

    /// @brief for_each_span() over every element.
    template <typename F>
    void for_each_span(F&& f) const
    {
        for_each_span(ind_type{0}, m_size, std::forward<F>(f));
    }

    /**
     * @brief Copy the elements of [@p first, @p last) to @p out.
     * @details for_each_value() writing to @p out. Into a T*, each block is
     *          read straight into the destination: no intermediate copy.
     * @return @p out advanced past the last element written.
     */
    template <typename OutIt>
    OutIt read_range(ind_type first, ind_type last, OutIt out) const
    {
        if constexpr (kBulkFast && std::is_same_v<OutIt, T*>)
        {
            for_each_chunk_span(first, last,
                                [&](const chunk_type& ch, int num_chunk, int lo, int hi)
                                {
                                    for (int j = lo; j < hi; j += kBulkBlock)
                                    {
                                        const int n = (hi - j) < kBulkBlock ? (hi - j) : kBulkBlock;
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
            --m_size;
        }
    }

    /**
     * @brief What shrink_to_fit means for a container that compresses: every
     *        chunk that has become constant is folded now, and the buffers the
     *        pools keep parked for reuse are given back.
     * @details Both are process-wide -- the compression candidates and the
     *          reuse pools are per element type and per rank, not per array --
     *          so this serves every array of the process, and the memory it
     *          returns is theirs as much as this one's. The size and the
     *          capacity do not change: a chunk is the unit of storage, and the
     *          spare room in the last one is one chunk at most.
     *          Sequential context.
     */
    void shrink_to_fit();

    /**
     * @brief Checked read access.
     * @param i global element index
     * @return the element value
     * @throws std::out_of_range if @p i is negative or >= size()
     */
    [[nodiscard]] const_reference at(ind_type i) const;

    /**
     * @brief Checked write access.
     * @param i global element index
     * @return the write proxy for that element
     * @throws std::out_of_range if @p i is negative or >= size()
     */
    [[nodiscard]] reference at(ind_type i);

    /**
     * @name First / last element
     *
     * front()/back() on an empty container is UB in std::vector; here the store
     * is compressed/gate-driven, so an access into an unallocated chunk geometry
     * would corrupt silently rather than segfault cleanly. We throw instead — the
     * same safety contract as at(). These are not hot-path (the hot path is
     * operator[] in loops), so the emptiness test is free in practice.
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
     * @details Drop current contents, size to @p count, fill with @p val. Built on
     *          resize + operator[]. Serial (positional writes go through the
     *          proxy gate); callers that want a parallel fill use the bulk init.
     *
     * @param count new element count
     * @param val value written to every position
     */
    void assign(ind_type count, T val); // T, not in_t<T>: val may be a(i) -- resize() can free it

    /**
     * @brief assign(): the elements of [first, last), in order.
     * @details Sized once (std::distance, so a forward iterator at least),
     *          then one proxied write per element. Sequential.
     */
    template <std::forward_iterator It>
    void assign(It first, It last);

    /**
     * @brief swap(): O(1) exchange of two containers.
     *
     * @details No element moves: the chunk lists are exchanged, and with them
     *          everything that describes those chunks. The per-array tables of
     *          the proxies (constant types, Chunk2A's per-chunk second values,
     *          the compression candidates) are indexed by num_tab, and the
     *          registries map the array's address to it -- so the identity is
     *          exchanged too, as rechunk_to() does, and the metadata travel
     *          with their chunks. So do the slab-affinity key (the buffers were
     *          drawn under it) and the dirty bitset. Exchanging the chunks
     *          alone would leave each array reading the other's metadata rows.
     *
     *          Sequential context: the deferred writes of both arrays are
     *          drained first, since they carry the array they were queued on.
     *
     * @param other the container to exchange state with
     * @note noexcept, as std::vector::swap. The drain and the registry
     *       update around the O(1) exchange only fail on memory exhaustion or
     *       a corrupt registry, which terminates.
     * @pre Both operands must share chunk geometry (enforced under check; with
     *      StaticChunkSize this is always true).
     */
    void swap(Derived& other) noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (size_policy_.size() != other.size_policy_.size())
            {
                CHUNK_INVARIANT_FAILED("ArrayChunk::swap: mismatched chunk geometry ({} vs {})", size_policy_.size(),
                                       other.size_policy_.size());
            }
        }
        swap_storage_and_identity(other);
    }

    /**
     * @brief CONST operator() for fast READ access.
     * Marks the function as const so it can be called by read_value()
     * and other const contexts.
     * @param i global element index (unchecked)
     * @return the element value, after the read gate has been honoured
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T operator()(ind_type i) const noexcept
    {
        const int   num_chunk = chunk_index_of(i);
        const int   num       = offset_in_chunk(i);
        const auto& chunk_t   = chunk_list[num_chunk];

        sync_chunk(chunk_t, num_chunk, num);

        // Direct read from the chunk
        return chunk_t.read_value(num);
    }

    /**
     * @brief NON-CONST operator() for direct REFERENCE access (Write).
     * This version allows modification: arr(i) = value;
     * @param i global element index (unchecked)
     * @return a reference to the physical storage of the element
     * @warning Bypasses the write proxy: valid only where the chunk is known to
     *          be materialised. Prefer operator[] otherwise.
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T& operator()(ind_type i) noexcept
    {
        const int num_chunk = chunk_index_of(i);
        const int num       = offset_in_chunk(i);
        auto&     chunk_t   = chunk_list[num_chunk];

        // Read gate
        sync_chunk(chunk_t, num_chunk, num);

        // Reference to the physical storage
        return chunk_t(num);
    }

    /**
     * @brief Unchecked read access (hot path).
     *
     * @details Force-inlined: GNU otherwise leaves this out-of-line, and it showed
     *          up as ~30% of the 128-thread GNU profile (operator[] across all
     *          types). LLVM and nvc++ already inline it. CHUNK_FORCE_INLINE degrades to
     *          plain `inline` in debug and for nvc++; CHUNK_FLATTEN (opt-in,
     *          -DSHRINK_AGGRESSIVE_INLINE, never nvc++) additionally pulls the
     *          whole read path into this leaf on GNU/clang.
     *
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the element VALUE (see the member-typedef note on const_reference)
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE T operator[](ind_type i) const noexcept
    {
        const int num_chunk = chunk_index_of(i);
        const int num       = offset_in_chunk(i);
        // Loop index is always in range; lets the compiler drop bounds/negative
        // checks. Side-effect free (C++23 [[assume]] when available, else no-op).
        // num = i & size_policy_.modulo()  =>  0 <= num <= static_cast<int>(size_policy_.modulo()) by construction.
        // modulo() read into a local first: clang ignores an assumption that
        // calls a function it cannot prove side-effect free (-Wassume).
        [[maybe_unused]] const int chunk_mod = static_cast<int>(size_policy_.modulo());
        CHUNK_ASSUME(num_chunk >= 0);
        CHUNK_ASSUME(num >= 0 && num <= chunk_mod);
        const auto& chunk_t = chunk_list[num_chunk];

        sync_chunk(chunk_t, num_chunk, num);

        return chunk_t[num];
    }

    /**
     * @brief Unchecked write access: THE write entry of the container.
     *
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the write proxy for that element, armed with the owning array and
     *         chunk number
     * @warning The proxy is a HANDLE on the element, not a copy of its value:
     *          it holds references into this array's chunk storage. It must not
     *          outlive any operation that rebuilds or releases that storage -- resize()
     *          (a push_back/emplace_back beyond the capacity included),
     *          reserve(), shrink_to_fit(), clear(), assign(), swap(), a copy or
     *          move assignment, the array's destruction, and above all a
     *          re-chunk, whether called explicitly (rechunk_to) or decided by a
     *          RechunkDecider. (shrink_to_fit() keeps the chunks but compresses
     *          them and releases their buffers, which a proxy's ctor-time
     *          `values` snapshot may still point into.) Reading a stale proxy
     *          reads swapped-out or released storage.
     *          Take a value when it has to survive: `T v = a[i];`, not
     *          `auto v = a[i];` (which keeps the proxy). The compound operators
     *          (+= ... >>=) and the prefix ++/-- return the same kind of proxy.
     *          In check builds the proxy comes wrapped in an ElementRef, whose
     *          operations only a temporary may use: a use of
     *          `auto v = a[i];` does not compile (internal/element_ref.h).
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE reference operator[](ind_type i) noexcept
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
     *        wraps it in an ElementRef in check builds).
     * @param i global element index; the caller guarantees 0 <= i < size()
     * @return the write proxy for that element, armed as operator[]'s
     */
    CHUNK_FLATTEN CHUNK_FORCE_INLINE proxy_type element_proxy(ind_type i) noexcept
    {
        const int         num_chunk = chunk_index_of(i);
        auto&             chunk_t   = chunk_list[num_chunk];
        const int         num       = offset_in_chunk(i);
        const proxy_type& val       = chunk_t[num];

        // Arm ptr_on_array / num_chunk in ALL modes, not only under fast block
        // compression: in non-OpenMP builds too, set_num_tab() and
        // get_info_second_value() look ptr_on_array up in share_info_array.
        val.set_this_and_num_chunk(registry_key(), num_chunk, num_tab);
        val.arm_storage_generation(&storage_generation_); // check builds only

        // Chunk2 only: arm the proxy with the address of this array's registry
        // mirror, so the write path copies the (second_value, ind, num_tab)
        // tuple instead of scanning share_info_array for it. Discriminated on
        // the DECLARED proxy type (chunk_traits.h), not on a `requires`: a
        // renamed member must fail to compile, not silently fall back to the
        // registry.
        if constexpr (std::is_same_v<proxy_type, Proxy2<T>>)
        {
            val.arm_inf_array(derived().registry_mirror());
        }

        // The write entry does not sync: visibility is pulled by the read
        // entries (proxy operator T()/read_values -> ChunkBase::reader_sync),
        // and a same-offset WAW is handled on the direct write path. The const
        // operator[]/operator() and the raw-reference T& operator() keep
        // sync_chunk.

        chunk_config::CheckNaN<chunk_config::do_check>::maybe(val, static_cast<long>(i));

        return val;
    }

  protected:
    // The core's members, named here so this class and the derived arrays use
    // them unqualified (the core is a dependent base).
    using Core::alloc_key_;
    using Core::assign_alloc_key;
    using Core::chunk_list;
    using Core::nb_chunk;
    using Core::nb_tab;
    using Core::next_alloc_key_;
    using Core::num_tab;
    using Core::size_policy_;

    /// @brief Installs the core's way back to rechunk_to() (rechunk_erased).
    /// @note Protected, not private with `friend Derived`: Chunk2 and Chunk2A
    ///       reach this base through ArrayChunk2Base<Derived>, which has to
    ///       construct it too.
    ArrayChunkBase() noexcept
    {
        this->rechunk_fn_ = &rechunk_thunk;
    }

    /**
     * @name Index splitting
     * A global element index is an ind_type (int or long); the chunk it falls
     * in and its offset there are ints -- nb_chunk, the per-chunk tables and the
     * chunks' own offsets are int, which holds while size() < INT_MAX *
     * chunk_size(). The narrowing is written here, once, rather than implied at
     * every `const int c = i >> ...`.
     * @{
     */
    /// @brief The chunk holding element @p i: `i >> chunk_decalage()`.
    [[nodiscard]] int chunk_index_of(ind_type i) const noexcept
    {
        return static_cast<int>(i >> size_policy_.decalage());
    }

    /// @brief Element @p i's offset inside its chunk: `i & chunk_modulo()`.
    [[nodiscard]] int offset_in_chunk(ind_type i) const noexcept
    {
        return static_cast<int>(i & size_policy_.modulo());
    }

    /// @brief Chunks allocated for @p n elements: n / chunk_size() + 1 (the last
    ///        chunk is never full -- see capacity()).
    [[nodiscard]] int chunks_for(ind_type n) const noexcept
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
     * @brief The deep copy the derived arrays' copy constructor and
     *        copy-assignment are built on: sized to @p from, then every
     *        element read from @p from (a const read, through its gate) and
     *        written here through the proxy.
     *
     * @details Element by element on purpose, like rechunk_refill: the copy
     *          is then correct for every variant, for arrays of different
     *          geometries, and for a destination that already holds data. A
     *          chunk of @p from at the first constant costs one early-out per
     *          element here (the destination's fresh chunks are at the first
     *          constant too); a run of a second value or of any repeated
     *          value is written materialised and folded back by the next
     *          compression pass -- the copy is right at once, compact soon
     *          after. Sequential context: the writes are this thread's, and
     *          resize() is a sequential point.
     *
     *          chunk2's array-level second value is re-declared first when
     *          @p from has one (the derived copy hooks take care of it: this
     *          base knows no second value).
     */
    void copy_from(const Derived& from);

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
        swap(chunk_list, other.chunk_list);
        swap(alloc_key_, other.alloc_key_);
        swap_dirty_bitset(other);
        swap(m_size, other.m_size);
        swap(nb_chunk, other.nb_chunk);
        swap(size_policy_, other.size_policy_);
        derived().swap_identity(other);
    }

    ind_type m_size; ///< Number of elements exposed by size().

    /// @brief Bumped by every operation that rebuilds or releases the storage
    ///        -- the list that invalidates a proxy (see operator[]). A check
    ///        build's proxy keeps the value it saw and aborts on use when it
    ///        changed (ProxyBase::check_not_stale); other builds never read
    ///        it. Per array, not swapped: a swap invalidates both sides.
    unsigned int storage_generation_ = 0;

    /// @brief Counts this array among the live ones (live_arrays.h): the
    ///        teardown verbs keep the pools while any array is alive.
    [[no_unique_address]] internal::LiveArrayToken live_token_;

    /// @brief Every proxy of this array built so far is now stale.
    void invalidate_proxies() noexcept
    {
        ++storage_generation_;
    }

    /// @brief clear() takes the end-of-life census (print_compression_stats():
    ///        the per-array line and the "still compressed (at exit)" counter).
    ///        False on a re-chunk's tmp, which dies mid-run holding the OLD
    ///        geometry: its census is not an exit, and its state depends on the
    ///        build (an OpenMP re-chunk releases the source chunks during the
    ///        copy, a serial one leaves them materialised), so counting it
    ///        would make still-at-exit differ between sequential and OpenMP runs.
    bool exit_census_ = true;

    /**
     * @brief Read the element at @p i.
     * @param i global element index (unchecked)
     * @return the element value
     * @note Protected, not private: ArrayChunk2A::operator== reads both of its
     *       operands through it.
     */
    T read_value(ind_type i) const noexcept
    {
        return (*this)(i);
    }

  private:
    /// @brief Elements per read section in the bulk reads (8 to 64 measure
    ///        alike; the larger, the fewer sections on long ranges).
    static constexpr int kBulkBlock = 64;
    /// @brief Bulk reads by block only where per-element stats and NaN checks
    ///        do not rely on every element going through operator[].
    static constexpr bool kBulkFast = !chunk_config::stats && !chunk_config::do_check;

    /// @brief Call @p fn(chunk, num_chunk, lo, hi) for each chunk the range
    ///        [first, last) touches -- offsets [lo, hi) in it -- after one
    ///        sync_chunk, as the iterator does on chunk entry.
    template <typename Fn>
    void for_each_chunk_span(ind_type first, ind_type last, Fn&& fn) const
    {
        if (first < 0)
        {
            first = 0;
        }
        if (last > m_size)
        {
            last = m_size;
        }
        if (first >= last)
        {
            return;
        }
        const unsigned int dec     = size_policy_.decalage();
        const auto         mod     = static_cast<ind_type>(size_policy_.modulo());
        const int          c_first = static_cast<int>(first >> dec);
        const int          c_last  = static_cast<int>((last - 1) >> dec);
        for (int c = c_first; c <= c_last; ++c)
        {
            const int         lo = c == c_first ? static_cast<int>(first & mod) : 0;
            const int         hi = c == c_last ? static_cast<int>((last - 1) & mod) + 1 : static_cast<int>(mod) + 1;
            const chunk_type& ch = chunk_list[static_cast<std::size_t>(c)];
            sync_chunk(ch, c, lo);
            fn(ch, c, lo, hi);
        }
    }

    /// @brief Check builds, precise read gate: nothing was deferred onto
    ///        the chunk since its sync -- the iterator's check, per block.
    void check_no_write_during_bulk_read([[maybe_unused]] const chunk_type& ch,
                                         [[maybe_unused]] int               num_chunk,
                                         [[maybe_unused]] int               offset) const noexcept
    {
        if constexpr (build_config::any_check && chunk_type::precise)
        {
            if (ch.read_gate_should_wait(offset)) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("ArrayChunkBase bulk read: a write was deferred onto chunk {} of array {} "
                                       "during the read (offset {}); it would not be seen -- no other thread may "
                                       "write an array that is being read in bulk",
                                       num_chunk, static_cast<const void*>(this), offset);
            }
        }
    }

    /**
     * @brief Is chunk @p j materialised?
     * @details Routes to Derived::chunk_written(j) when it exists (ArrayChunk2,
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
     * @note Compiles to nothing in a serial build (`if constexpr (use_openmp)`).
     */
    CHUNK_FORCE_INLINE void sync_chunk(const chunk_type& chunk_t, int num_chunk, int offset) const noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            // Per-chunk pre-filter: the deferred-enqueue path marks this bit
            // with the CHUNK index (proxy_base_inline.h: mark_chunk_dirty(
            // ctx->num_chunk)), so a clear bit reliably means "no deferred op
            // was ever queued for this chunk" — skip the global check and the
            // flush entirely. Only chunks that actually received a deferred
            // write reach is_not_available(). This is what keeps the hot read
            // path cheap: without it, every read of every chunk would run
            // any_pending_ops() and (whenever anything is pending anywhere)
            // trigger flush_chunk_all_ranks.
            //
            // Note: the bit is sticky (only clear_dirty_bitset() resets it), so a
            // chunk stays "dirty" for the rest of the phase after its first
            // deferred op. That is acceptable: it is only a coarse pre-filter.
            // The precise, authoritative gate below is the chunk's OWN pending
            // counter — see next comment.
            if (derived().is_chunk_dirty(num_chunk)) [[unlikely]]
            {
                // Precise per-chunk gate (per-chunk pending_ops_ load) or coarse
                // is_not_available() fallback under SHRINK_SMALL_CHUNK — selected
                // by DeferredReadGate<Derived> (chunk_base.h). The
                // outer `if constexpr (use_openmp)` already excludes serial builds,
                // where the gate would be a no-op anyway.
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
        // The core IS the base of a Derived: its fn pointer was set by this
        // very class, in the constructor of that Derived.
        static_cast<Derived&>(core).rechunk_to(new_size);
    }
};
} // namespace shrink_array

#include "chunk/internal/array_chunk_base_inline.h"    // the template bodies
#include "chunk/internal/array_chunk_rechunk_inline.h" // the RechunkSupport bodies (CHUNK_DYNAMIC)
