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

//
// Created by ballereaup on 03/04/24.
//
#pragma once

#include <atomic>          // std::memory_order_* on the values pointer
#include <cstddef>         // std::size_t
#include <cstdint>         // uint32_t / uintptr_t (the ring-log chunk id)
#include <source_location> // std::source_location::current() in the invariants
#include <tuple>           // the candidate-list entries
#include <type_traits>     // std::is_same_v (concurrent-ops base)
#include <vector>

#include "chunk/config.h"
#include "chunk/internal/invariant_check.h"
#include "chunk/internal/tab_cst_val2_holder.h"
#include "chunk/chunk_define.h" // FAST_ALLOC
#include "chunk/chunk_def.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/ring_log_debug.h"             // ScopedRingLog, ring_log_* (RING_LOG builds)
#include "chunk/internal/chunk_size_policy.h" // ChunkElems
#include "chunk/internal/chunk2_base.h"
#include "chunk/internal/crp_helper.h" // crtp_cast_t
#include "chunk/internal/chunk_stats.h"

#ifdef FAST_ALLOC
    #include "fast_allocators/tc_allocator.h"
#endif

#include "chunk/internal/TChunk.h"


namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkType>
class CompressionManager;

template <typename T>
class Proxy2A;

template <template <typename> typename CHUNK_T, typename T>
class ManagerChunk;

/*!
 * \brief Block compression class (power of 2) 2 values automatic detection
 * This class manages a chunk of values that can either be constant (val1 or val2)
 * or dynamically allocated. It supports compression by identifying and optimizing
 * for repeated values.
 *
 * \tparam T Type of the values stored in the chunk
 */

// ─── Concurrent (OMP) chunk operations extracted from Chunk2A ───────────────
// CRTP policy base holding the lock-free CAS operations that exist only in
// OpenMP builds. Chunk2A inherits it through the OpenMP specialisation of
// Chunk2AOmpPolicy below, so the preprocessor guard around these two methods is
// replaced by a typed base: in a serial build the methods do not exist,
// exactly as before. Reaches Chunk2A's bases via self(): Base = Chunk2Base,
// Holder = TabCstVal2Holder. All accessed members are public (no friend needed).
template <typename T, typename Derived>
class Chunk2AConcurrentOps
{
  public:
    using Base   = Chunk2Base<T, Derived>;
    using Holder = TabCstVal2Holder<T>;

    /// @brief Atomically mark this chunk constant and hand back its old buffer.
    /// @param num_compress_value 0 for the first constant, 1 for the second.
    /// @param cst2_ptr The shared array to install when @p num_compress_value
    ///        is 1 (find_or_set_ind_cst_val2's result); ignored for 0.
    [[nodiscard]] T* try_mark_cst_and_get_old(int num_compress_value, T* cst2_ptr) noexcept
    {
        T* old = base()->values_load();
        if (old == Base::get_cst_val1() || Holder::is_cst_val2_ptr(old))
            return nullptr;

        T* new_ptr  = (num_compress_value == 0) ? Base::get_cst_val1() : cst2_ptr;
        T* expected = old;

        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = ring_chunk_id();
            const uint32_t info     = (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(num_compress_value + 1) << 24U);
            ring_log(get_tid(), EV_COMPRESS_ATTEMPT, info);
        }

        if (!self()->values.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel,
                                                    std::memory_order_acquire))
            return nullptr;

        if constexpr (build_config::any_check)
        {
            T* observed = base()->values_load();
            if (observed != new_ptr && observed != Base::get_cst_val1() && !Holder::is_cst_val2_ptr(observed))
                invariant_failed(std::source_location::current(),
                                 "try_mark_cst_and_get_old: post-CAS values={} not in expected set "
                                 "(new_ptr={}, cst1={}, a cst2 array)",
                                 static_cast<const void*>(observed), static_cast<const void*>(new_ptr),
                                 static_cast<const void*>(Base::get_cst_val1()));
        }

        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = ring_chunk_id();
            const uint32_t info     = (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(num_compress_value + 1) << 24U);
            ring_log(get_tid(), EV_CAS_VALUES_TO_CST, info);
        }
        return old;
    }

  protected:
    /// @brief Publish @p values_loc as this chunk's buffer, expecting the
    ///        chunk to still be compressed to the constant the caller saw
    ///        (@p expected_is_cst_val2: the second value, else the first).
    /// @return true when the swap FAILED (someone else materialised the chunk
    ///         first; the caller recycles @p values_loc), the convention of
    ///         ChunkBase::compare_and_swap_gen().
    [[nodiscard]] bool compare_and_swap(T* values_loc, bool expected_is_cst_val2) noexcept
    {
        // The array a chunk compressed to a second constant reads from IS its
        // values pointer (there is no per-chunk copy of it any more). The CAS
        // must still expect a CONSTANT, never "whatever values holds now": if
        // another thread materialised the chunk between the caller's look and
        // this point, values is its buffer, and a CAS that expected it would
        // succeed and replace that buffer -- with the value it just wrote --
        // by ours. Seen as a lost write in check mode. So: load, and if the
        // pointer is not in the second-value region, report the CAS as failed
        // without attempting it; the caller recycles our buffer, as it does
        // when the CAS itself fails.
        T* expected = nullptr;
        if constexpr (is_not_bool)
        {
            if (expected_is_cst_val2)
            {
                expected = base()->values_load();
                if (!Holder::is_cst_val2_ptr(expected))
                    return true; // failed: someone else materialised it first
            }
            else
            {
                expected = Base::get_cst_val1();
            }
        }
        else
        {
            expected = expected_is_cst_val2 ? Holder::m_ptr.data() : Base::get_cst_val1();
        }

        const bool failed = base()->compare_and_swap_gen(values_loc, expected);

        if constexpr (build_config::any_check)
        {
            if (!failed)
            {
                T* current = base()->values_load();
                if (current != values_loc && current != Base::get_cst_val1() && !Holder::is_cst_val2_ptr(current))
                    invariant_failed(std::source_location::current(),
                                     "post-create_value CAS: values={}, expected one of "
                                     "{{values_loc={}, cst1={}, a cst2 array}}",
                                     static_cast<const void*>(current), static_cast<const void*>(values_loc),
                                     static_cast<const void*>(Base::get_cst_val1()));
            }
        }
        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = ring_chunk_id();
            const uint32_t info =
                (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(expected_is_cst_val2 ? 2 : 1) << 24U);
            ring_log(get_tid(), failed ? EV_CAS_VALUES_FAIL : EV_CAS_VALUES_FROM_CST, info);
        }
        return failed;
    }

  private:
    static constexpr bool is_not_bool = !std::is_same_v<T, bool>;

    /// @brief The chunk, const on a const object (deducing this: one member).
    template <typename Self>
    [[nodiscard]] auto* self(this Self&& obj) noexcept
    {
        return &static_cast<crtp_cast_t<Self, Derived>&>(obj);
    }
    /// @brief The chunk's Chunk2Base / TabCstVal2Holder, const on a const
    ///        object (deducing this, as self()).
    template <typename Self>
    [[nodiscard]] auto* base(this Self&& obj) noexcept
    {
        return static_cast<crtp_cast_t<Self, Base>*>(obj.self());
    }
    template <typename Self>
    [[nodiscard]] auto* holder(this Self&& obj) noexcept
    {
        return static_cast<crtp_cast_t<Self, Holder>*>(obj.self());
    }

    /// @brief Compact identity of this chunk for the ring log: its address past
    ///        the alignment bits. Callers mask it to 24 bits and pack a tag into
    ///        the top byte. Was written out identically at three sites.
    [[nodiscard]] uint32_t ring_chunk_id() const noexcept
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(self()) >> 4U);
    }
};

// Read-side cst-state queries: lock-free classification of the atomic `values`
// pointer (cst_val1 / cst_val2 / written). The read counterpart to
// Chunk2AConcurrentOps' CAS writes, but BUILD-AGNOSTIC: these are needed in both
// serial and OMP builds, so this base is mixed into Chunk2A UNCONDITIONALLY (not
// gated on use_openmp like Chunk2AConcurrentOps). Same CRTP accessors, which
// resolve to const here since every query is a pure read.
template <typename T, typename Derived>
class Chunk2AStateQueries
{
  public:
    using Base   = Chunk2Base<T, Derived>;
    using Holder = TabCstVal2Holder<T>;

    /// @brief Is the chunk compressed to a second value (its values pointer
    ///        lies in the second-value region; for bool, IS the flag block)?
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        const T* current_value = base()->values_load();
        if constexpr (is_not_bool)
            return Holder::is_cst_val2_ptr(current_value); // a range test on the region
        else
            return current_value == Holder::m_ptr.data();
    }

    /// @brief is_cst_val2() asked of a pointer the caller already loaded.
    /// @details element_at() must test and read the SAME load: asking the chunk
    ///          loads `values` a second time, and a compressor claiming it in
    ///          between swaps that pointer for a region identifier, which the
    ///          read then indexes as a buffer.
    [[nodiscard]] bool is_cst_val2_ptr_of(const T* ptr) const noexcept
    {
        if constexpr (is_not_bool)
            return Holder::is_cst_val2_ptr(ptr);
        else
            return ptr == Holder::m_ptr.data();
    }

    /**
     * @brief The constant a chunk whose values pointer is @p ptr stands for,
     *        read from the registry: second_value_of_chunk() for a pointer
     *        the caller already loaded.
     *
     * @pre @p ptr satisfies is_cst_val2_ptr_of().
     * @details Used by ChunkBase::element_at(): a second constant has no
     *          buffer, its values pointer names an identifier. Deliberately does NOT dereference that pointer:
     *          id_of_ptr() is arithmetic on the region base and stride, and
     *          value_of_id() is one load of the value table -- the same table
     *          a write of a wide type already reads.
     */
    [[nodiscard]] T second_value_from_ptr(const T* ptr) const noexcept
    {
        if constexpr (is_not_bool)
            return Holder::value_of_id(Holder::id_of_ptr(ptr));
        else
            return true;
    }

    /**
     * @brief Give this chunk a real buffer, because a caller needs an address.
     *
     * @details ChunkBase::print_adr() hands out `&values[i]`, which no registry
     *          lookup can stand in for: there is no buffer behind a chunk
     *          compressed to a second constant. Its one caller is
     *          ArrayChunkBase's re-export, itself never called inside the
     *          library, so this is a diagnostic path and the chunk losing its
     *          compressed state costs nothing that matters.
     */
    void materialise_for_address() noexcept
    {
        if (is_cst_val2())
            base()->create_value(second_value_of_chunk());
    }

    /// @brief Is @p ptr one of the constant arrays (first or a second one)?
    /// @note For a values pointer the caller already loaded -- the pre-store
    ///       guards of the proxy -- where is_cst_val() would load it again.
    [[nodiscard]] static bool is_constant_ptr(const T* ptr) noexcept
    {
        if (ptr == Base::get_cst_val1())
            return true;
        if constexpr (is_not_bool)
            return Holder::is_cst_val2_ptr(ptr);
        else
            return ptr == Holder::m_ptr.data();
    }

    /// @brief Materialised (owns a buffer, i.e. compressed to neither constant)?
    [[nodiscard]] bool written() const noexcept
    {
        return base()->values_load() != Base::get_cst_val1() && is_not_cst_val2();
    }

    /// @brief written() reading the values member directly, without the
    ///        atomic load: for a sequential point (the array's stats), where no
    ///        writer or compressor can be publishing.
    [[nodiscard]] bool written_test() const noexcept
    {
        return self()->values != Base::get_cst_val1() && is_not_cst_val2();
    }

  protected:
    /// @brief Compressed to either constant?
    [[nodiscard]] bool is_cst_val() const noexcept
    {
        return base()->is_cst_val1() || is_cst_val2();
    }

  private:
    static constexpr bool is_not_bool = !std::is_same_v<T, bool>;

    /// @brief The chunk, const on a const object (deducing this, as in
    ///        Chunk2AConcurrentOps; every query below is const).
    template <typename Self>
    [[nodiscard]] auto* self(this Self&& obj) noexcept
    {
        return &static_cast<crtp_cast_t<Self, Derived>&>(obj);
    }
    /// @brief The chunk's Chunk2Base / TabCstVal2Holder, const on a const
    ///        object (deducing this, as self()).
    template <typename Self>
    [[nodiscard]] auto* base(this Self&& obj) noexcept
    {
        return static_cast<crtp_cast_t<Self, Base>*>(obj.self());
    }
    template <typename Self>
    [[nodiscard]] auto* holder(this Self&& obj) noexcept
    {
        return static_cast<crtp_cast_t<Self, Holder>*>(obj.self());
    }

    /// @brief Complement of is_cst_val2().
    [[nodiscard]] bool is_not_cst_val2() const noexcept
    {
        return !is_cst_val2();
    }

    /// @brief second_value_from_ptr() on this chunk's own values pointer.
    /// @pre is_cst_val2().
    [[nodiscard]] T second_value_of_chunk() const noexcept
    {
        if constexpr (is_not_bool)
            return Holder::value_of_id(Holder::id_of_ptr(base()->values_load()));
        else
            return true; // bool: the second value is the only other one
    }
};

/**
 * @brief Chunk2A's OpenMP-only surface: per-thread materialisation and clearing.
 *
 * @details A serial build has no thread ranks, no CAS window and no in-flight
 *          compression to wait out, so it materialises and clears a chunk
 *          differently; each specialisation below declares exactly the verbs
 *          its mode has -- the technique of ChunkOmpPolicy / Chunk2OmpPolicy.
 *          The OpenMP specialisation also carries the CAS operations
 *          (Chunk2AConcurrentOps).
 *
 *          Sits between Chunk2A and Chunk2Base in the inheritance chain.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk2A<T>)
 * @tparam Enabled build_config::use_openmp
 */
template <typename T, typename Derived, bool Enabled = build_config::use_openmp>
class Chunk2AOmpPolicy : public Chunk2Base<T, Derived>, public Chunk2AConcurrentOps<T, Derived>
{
  public:
    using Base                   = Chunk2Base<T, Derived>;
    using manager_chunk2A        = ManagerChunk<Chunk2A, T>;
    using CompressionManagerType = CompressionManager<T, Derived>;

    using Base::derived;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new Chunk2A<T>[n]` runs the DEFAULT CONSTRUCTORS after this,
    ///          so that initialisation is immediately overwritten; the operator
    ///          pays off for a placement-new array, where the caller does not
    ///          construct. Nothing in the library array-news a chunk --
    ///          ArrayChunk* hold a std::vector<chunk_type>, initialised by
    ///          their own loops.
    void* operator new[](std::size_t sz);
    /// @brief Matching array-delete.
    void operator delete[](void* ptr);

    /*!
     * \brief Materialise this chunk's buffer (allocate + fill + CAS-publish),
     *        for a specific thread. Non-bool only.
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param ptr_second_value      The chunk's second value; read only when the flag is set.
     * \param thr_rank              Rank whose allocator provides the buffer.
     */
    void create_value(this Derived& self, bool need_set_second_value, SecondValueRef<T> ptr_second_value, int thr_rank);

    /*!
     * \brief Bool specialization of create_value().
     * \param need_set_second_value Fill value -- true for the second constant, false for the first.
     * \param thr_rank              Rank whose allocator provides the buffer.
     */
    void create_value_bool(this Derived& self, bool need_set_second_value, int thr_rank);

    /// @brief create_value() on the calling thread's rank.
    void create_value(this Derived& self, bool need_set_second_value, SecondValueRef<T> ptr_second_value);

    /// @brief create_value_bool() on the calling thread's rank.
    void create_value_bool(this Derived& self, bool need_set_second_value);

    /*!
     * \brief Compress this chunk back to a constant, if it is materialised.
     * \param ptr_type_cst     Slot of the per-chunk type table to update.
     * \param ptr_second_value The chunk's second-value slot.
     * \param thr_rank         Rank whose pool receives the released buffer.
     */
    void clear(char* ptr_type_cst, SecondValueRef<T> ptr_second_value, int thr_rank);

    /*!
     * \brief Release this chunk's buffer and point it back at a constant --
     *        the single point every compression path funnels through.
     * \param num_compress_value 0 for the first constant, otherwise the second.
     * \param ptr_type_cst       Slot of the per-chunk type table to update.
     * \param ptr_second_value   The chunk's second-value slot.
     * \param thr_rank           Rank whose pool receives the released buffer.
     * \param cst2_ptr           Shared array of the second constant (type 2).
     */
    void block_clear(int               num_compress_value,
                     char*             ptr_type_cst,
                     SecondValueRef<T> ptr_second_value,
                     int               thr_rank,
                     T*                cst2_ptr);

    /*!
     * \brief Compress this chunk, committing its second-value pointer first.
     * \param num_compress_value 0 for the first constant, 1 for the second.
     * \param ptr_type_cst       Slot of the per-chunk type table to update.
     * \param ptr_second_value   The chunk's second-value slot.
     * \param thr_rank           Rank whose pool receives the buffer.
     * \param new_cst_val2_ptr   Pointer returned by find_or_set_ind_cst_val2();
     *                           required when @p num_compress_value is 1.
     */
    void clear_int_st(int               num_compress_value,
                      char*             ptr_type_cst,
                      SecondValueRef<T> ptr_second_value,
                      int               thr_rank,
                      T*                new_cst_val2_ptr = nullptr);

    /*!
     * \brief clear_int_st() on the calling thread's rank.
     * \param num_compress_value 0 for the first constant, otherwise the second.
     * \param ptr_type_cst       Slot of the per-chunk type table to update.
     * \param ptr_second_value   The chunk's second-value slot.
     */
    void clear_st(int num_compress_value, char* ptr_type_cst, SecondValueRef<T> ptr_second_value)
    {
        clear_int_st(num_compress_value, ptr_type_cst, ptr_second_value, CHUNK_PMP_THREAD_RANK);
    }

  private:
    /// @brief Point @p count freshly allocated chunks at the first constant.
    /// @details Parallel past 500 chunks, and only when called from OUTSIDE a
    ///          parallel region (inside one it stays on the calling thread: an
    ///          allocation is made by one thread, and an orphaned `omp for`
    ///          would hang waiting for the others). Shared by the three
    ///          variants -- chunk_array_new::init_chunks.
    static void init_chunks_cst_val(Chunk2A<T>* v, std::size_t count) noexcept;
};

/**
 * @brief Serial specialisation: no rank, no CAS, no in-flight wait; and the
 *        eager clear verbs the sequential compression uses.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk2A<T>)
 */
template <typename T, typename Derived>
class Chunk2AOmpPolicy<T, Derived, false> : public Chunk2Base<T, Derived>
{
  public:
    using Base            = Chunk2Base<T, Derived>;
    using manager_chunk2A = ManagerChunk<Chunk2A, T>;

    using Base::derived;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values;

    /// @brief The single candidate list of a serial build.
    static TChunk2AComp<Chunk2A, T>& get_vec_tuple_to_compress_potentially() noexcept
    {
        return manager_chunk2A::chunk2A_to_compress_potentially;
    }

    /*!
     * \brief Materialise this chunk's buffer (allocate + fill + assign).
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param ptr_second_value      The chunk's second value; read only when the flag is set.
     */
    void create_value(bool need_set_second_value, SecondValueRef<T> ptr_second_value);

    /// @brief Bool specialization of create_value().
    void create_value_bool(bool need_set_second_value);

    /*!
     * \brief The full type-2 clear (find_or_set path), so the sequential
     *        compression can register a cst_val2 exactly like the OpenMP path.
     *        thr_rank is 0 with OpenMP off.
     */
    void clear_int_st(this Derived&     self,
                      int               num_compress_value,
                      char*             ptr_type_cst,
                      SecondValueRef<T> ptr_second_value,
                      int               thr_rank,
                      T*                new_cst_val2_ptr = nullptr);


    /// @brief Compress to the FIRST constant and forget the second value, if materialised.
    void clear(this Derived& self, char* ptr_type_cst, SecondValueRef<T> ptr_second_value);

    /// @brief Unconditional twin of clear(char*, T*): no written() test.
    void clear_st(this Derived& self, char* ptr_type_cst, SecondValueRef<T> ptr_second_value);

    /// @brief Release the buffer outright (no rank pool) and point at a constant.
    void block_clear(this Derived&     self,
                     int               num_compress_value,
                     char*             ptr_type_cst,
                     SecondValueRef<T> ptr_second_value,
                     int,
                     T* cst2_ptr);
};

/// @brief Chunk of values compressible to the first constant or to a second constant shared through the
///        TabCstVal2Holder registry (see the class description above Chunk2AConcurrentOps).
/// @tparam T Element type.
template <typename T>
class Chunk2A : public Chunk2AOmpPolicy<T, Chunk2A<T>>,
                public TabCstVal2Holder<T>,
                public Chunk2AStateQueries<T, Chunk2A<T>>
{
  private:
    friend class Chunk2Base<T, Chunk2A>;
    friend class Proxy2A<T>;

    using Base      = Chunk2Base<T, Chunk2A>;
    using OmpPolicy = Chunk2AOmpPolicy<T, Chunk2A, build_config::use_openmp>;

  public:
    using manager_chunk2A = ManagerChunk<Chunk2A, T>;
    // Same tuple as TChunk.h's VectorTuple4ChunkInt2 (the manager storage).
    using vec_tuple_chunkA2p_int2  = std::vector<std::tuple<Chunk2A*, int, int, ChunkElems>>;
    using vec_tuple_chunkA2p_Tint2 = std::vector<std::tuple<Chunk2A*, T, int, int, ChunkElems>>;

    using Base::allocator_data;
    using Base::first_value;
    using Base::values;

    using tab_cst_val2_holder = TabCstVal2Holder<T>;

    // block_clear / clear_st / clear / clear_int_st / create_value* live on
    // Chunk2AOmpPolicy, one specialisation per mode (see above). Both declare
    // these names; the in-class bodies below call them unqualified.
    using OmpPolicy::block_clear;
    using OmpPolicy::clear;
    using OmpPolicy::clear_int_st;
    using OmpPolicy::clear_st;
    using OmpPolicy::create_value;
    using OmpPolicy::create_value_bool;

    // Statistics variant tag (CHUNK_STATS): 2 = chunk2a.
    static constexpr int stat_variant = 2;

    Chunk2A()  = default;
    ~Chunk2A() = default;

    /// @brief Copy the per-chunk state of @p other (Base::values_store(other.values_load())).
    Chunk2A(const Chunk2A& other)
    {
        Base::values_store(other.values_load()); // the whole per-chunk state
    }

    /// @brief Copy the per-chunk state of @p other unless it is this chunk.
    /// @return *this.
    Chunk2A& operator=(const Chunk2A& other)
    {
        if (this != &other)
            Base::values_store(other.values_load());
        return *this;
    }

    /**
     * @brief Initialize values after construction
     */
    void init_after_constructor() noexcept
    {
        Base::values = Base::get_cst_val1();
    }

    /**
     * @brief Set constant type to val1
     * @param ptr_type_cst Pointer to type indicator
     */
    static void set_cst_val_to_cst_val1(char* ptr_type_cst) noexcept
    {
        *ptr_type_cst = is_val1;
    }
    /**
     * @brief Set constant type to val2
     * @param ptr_type_cst Pointer to type indicator
     */
    static void set_cst_val_to_cst_val2(char* ptr_type_cst) noexcept
    {
        *ptr_type_cst = is_val2;
    }

    static void set_cst_val_to_cst_val(int num_compress_value, char* ptr_type_cst) noexcept;

    /**
     * @brief Initialize values depending on constant type
     * @param ptr_type_cst Pointer to constant type indicator
     */
    /// @param ptr_second_value The chunk's second-value slot: the array to
    ///        point at when the type says cst_val2 is found by value in the
    ///        registry (there is no per-chunk copy of it). May be null when
    ///        the type is cst_val1.
    void init_values_cst_val(const char* ptr_type_cst, SecondValueRef<T> ptr_second_value) noexcept
    {
        if (*ptr_type_cst == is_val1)
        {
            Base::values = Base::get_cst_val1();
            return;
        }
        if constexpr (is_not_bool)
        {
            T* cst_val2_ptr = tab_cst_val2_holder::find_cst_val2_ptr(ptr_second_value.get());
            if constexpr (build_config::any_check)
            {
                if (cst_val2_ptr == nullptr)
                    invariant_failed(std::source_location::current(),
                                     "init_values_cst_val: type says cst_val2 but no array is registered for the "
                                     "chunk's second value");
            }
            Base::values = cst_val2_ptr;
        }
        else
        {
            Base::values = tab_cst_val2_holder::m_ptr.data();
        }
    }

    /// @brief Get reference to the chunk manager (static access)
    /// @return Reference to the manager_chunk2A static instance
    ///
    static auto& get_manager() noexcept
    {
        // was a function-local `static manager_chunk2A s_manager;` (lazy Meyers
        // singleton). Chunk2A<T> is a class template; the lazy guard's first
        // init happened from compress() inside an OpenMP region, deadlocking
        // threads on __cxa_guard_acquire. An `inline static` member has one
        // definition across TUs (C++17), built during static init before main()
        // and before any thread — no guard, no race. Manager ctor is trivial.
        return s_manager_;
    }

    /// @brief Rank @p thr's candidate list.
    static TChunk2AComp<Chunk2A, T>& get_vec_tuple_to_compress_potentially_thr(int thr) noexcept
    {
        return manager_chunk2A::get_chunk2A_to_compress_potentially(thr);
    }

    /// @brief Materialise this chunk if it is currently compressed to either
    ///        constant; a no-op otherwise (already materialised).
    /// @param ptr_second_value The second value to fill from, when that is
    ///        the constant it holds.
    void alloc_chunk(SecondValueRef<T> ptr_second_value)
    {
        if (this->is_cst_val())
        {
            // bool has its own materialiser (the second value is the flag).
            if constexpr (is_not_bool)
                create_value(false, ptr_second_value);
            else
                create_value_bool(false);
        }
    }

    /// @brief alloc_chunk(), per-thread variant.
    /// @param ptr_second_value The second value to fill from.
    /// @param thr_rank Calling thread's rank (OpenMP builds; ignored in serial
    ///        builds).
    void alloc_chunk(SecondValueRef<T> ptr_second_value, [[maybe_unused]] int thr_rank)
    {
        if (this->is_cst_val())
        {
            if constexpr (chunk_config::stats)
            {
                increm_chunk_write<ChunkType>();
            }

            if constexpr (build_config::use_openmp)
            {
                if constexpr (is_not_bool)
                    create_value(false, ptr_second_value, thr_rank);
                else
                    create_value_bool(false, thr_rank);
            }
            else
            {
                if constexpr (is_not_bool)
                    create_value(false, ptr_second_value);
                else
                    create_value_bool(false);
            }
        }
    }

    void compress(int thr_rank, int num_chunk, int num_tab, bool apply, ChunkElems size);

    static void compress_chunk_to_compress(bool apply);

    /// @brief One rank's share of compress_chunk_to_compress() outside a
    ///        region: the body its region runs on each thread. Public for
    ///        AllTypeChunk's single sweep region (ManagerChunk::sweep_rank).
    static void compress_sweep_rank(int thr, bool apply);

    /// @brief Does that pass also sweep, serially, the ranks past the team?
    ///        Only the OpenMP non-OMPT apply pass, which drains with
    ///        make_block_compression().
    [[nodiscard]] static constexpr bool compress_sweep_mops_up(bool apply) noexcept
    {
        return build_config::use_openmp && chunk_config::not_use_compression_with_ompt && apply;
    }
    static bool have_chunk_to_compress(int max_thr);

    static void make_block_compression(int thr, Proxy2A<T>* proxy2A);

  private:
    using ChunkType              = Chunk2A;
    using CompressionManagerType = CompressionManager<T, ChunkType>;

    static constexpr char is_val1 = 0;
    static constexpr char is_val2 = 1;

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    /// @brief Is every rank's candidate list empty (the calling thread's, in a
    ///        serial build)?
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            return manager_chunk2A::is_chunk2A_to_compress_potentially_empty();
        }
        else
        {
            return manager_chunk2A::get_chunk2A_to_compress_potentially().empty();
        }
    }

    static void compress_chunk_to_compress_in_par(int thr, bool apply);

    // Chunk geometry is PER ENTRY in dynamic builds: every tuple queued into
    // chunk2A_to_compress_potentially carries the owning array's chunk size
    // (push_to_compress_list stores this->chunk_size() as field <3>). The scan
    // bounds below therefore come from that field — NEVER from a
    // `= __chunk_size` default. Such a default silently scanned only the first
    // 64 values of a 512-wide chunk in CHUNK_DYNAMIC runs: a chunk whose first
    // 64 elements happened to equal second_value was declared fully-constant
    // (compact2), block-cleared to the cst_val2 sentinel, and the remaining
    // 448 live values were destroyed. The defaults are gone on purpose: a new
    // call site must state its size explicitly or fail to compile.
    static void fill_list_chunk_to_compress_in_par(int                       thr,
                                                   vec_tuple_chunkA2p_int2&  ll_compt1,
                                                   vec_tuple_chunkA2p_Tint2& ll_compt2,
                                                   bool                      claimed = false);

    void fill_list_to_compress(vec_tuple_chunkA2p_int2&  ll_compt1,
                               vec_tuple_chunkA2p_Tint2& ll_compt2,
                               int                       num_chunk,
                               int                       num_tab,
                               ChunkElems                size,
                               bool                      claimed = false);

    inline static manager_chunk2A s_manager_{};
};

} // namespace shrink_array

#include "chunk/internal/chunk2A_std_inline.h"
