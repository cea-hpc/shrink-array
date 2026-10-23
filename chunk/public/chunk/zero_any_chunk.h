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

#include <atomic>          // std::memory_order_* on the values pointer
#include <cstddef>         // std::size_t
#include <cstdint>         // uint32_t / uintptr_t (the ring-log chunk id)
#include <source_location> // std::source_location::current() in the invariants
#include <tuple>           // the candidate-list entries
#include <type_traits>     // std::is_same_v (concurrent-ops base)
#include <vector>          // std::vector

#include "chunk/config.h"                       // build_config::threaded, chunk_config
#include "chunk/internal/invariant_check.h"     // invariant_failed
#include "chunk/internal/tab_cst_val2_holder.h" // TabCstVal2Holder, SecondValueRef
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_THREAD_RANK
#include "chunk/ring_log_debug.h"               // ScopedRingLog, ring_log_* (RING_LOG builds)
#include "chunk/internal/chunk_size_policy.h"   // ChunkElems
#include "chunk/internal/second_value_chunk_base.h"         // Chunk2Base
#include "chunk/internal/chunk_stats.h"         // increm_chunk_write

#include "chunk/internal/TChunk.h"   // TChunk2AComp, VectorTuple4ChunkInt2
#include "chunk/internal/immortal.h" // Immortal / LazyImmortal: statics safe for global arrays


namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkT>
struct CompressionManager;

namespace zero_any
{

template <typename T>
class Proxy;

} // namespace zero_any

template <template <typename> typename ChunkT, typename T>
class ManagerChunk;

/*!
 * \brief Block compression class (power of 2) 2 values automatic detection
 * This class manages a chunk of values that can either be constant (val1 or val2)
 * or dynamically allocated. It supports compression by identifying and optimizing
 * for repeated values.
 *
 * \tparam T Type of the values stored in the chunk
 */

namespace zero_any
{

// ─── Concurrent (OMP) chunk operations extracted from Chunk2A ───────────────
// CRTP policy base with the lock-free CAS operations of OpenMP builds only: Chunk2A inherits it
// through the OpenMP specialisation of Chunk2AOmpPolicy, so in a serial build the methods do not
// exist. Reaches Chunk2A's bases via self() (Base = Chunk2Base, Holder = TabCstVal2Holder).
/// @brief Chunk2A's lock-free CAS operations on `values` (OpenMP builds only).
template <typename T, typename Derived>
class ChunkConcurrentOps
{
  public:
    using Base   = SecondValueChunkBase<T, Derived>;
    using Holder = TabCstVal2Holder<T>;

    /// @brief Atomically mark this chunk constant and hand back its old buffer.
    /// @param num_compress_value 0 for the first constant, 1 for the second.
    /// @param cst2_ptr The shared array to install when @p num_compress_value
    ///        is 1 (find_or_set_ind_cst_val2's result); ignored for 0.
    [[nodiscard]] T* try_mark_cst_and_get_old(this Derived& self, int num_compress_value, T* cst2_ptr) noexcept
    {
        T* old = self.values_load();
        if (old == Base::get_cst_val1() || Holder::is_cst_val2_ptr(old))
        {
            return nullptr;
        }

        T* new_ptr  = (num_compress_value == 0) ? Base::get_cst_val1() : cst2_ptr;
        T* expected = old;

        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = self.ring_chunk_id();
            const uint32_t info     = (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(num_compress_value + 1) << 24U);
            ring_log(get_tid(), ev_compress_attempt, info);
        }

        if (!self.values_.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
        {
            return nullptr;
        }

        if constexpr (build_config::any_check)
        {
            T* observed = self.values_load();
            if (observed != new_ptr && observed != Base::get_cst_val1() && !Holder::is_cst_val2_ptr(observed))
            {
                invariant_failed(std::source_location::current(),
                                 "try_mark_cst_and_get_old: post-CAS values={} not in expected set "
                                 "(new_ptr={}, cst1={}, a cst2 array)",
                                 static_cast<const void*>(observed), static_cast<const void*>(new_ptr),
                                 static_cast<const void*>(Base::get_cst_val1()));
            }
        }

        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = self.ring_chunk_id();
            const uint32_t info     = (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(num_compress_value + 1) << 24U);
            ring_log(get_tid(), ev_cas_values_to_cst, info);
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
    [[nodiscard]] bool compare_and_swap(this Derived& self, T* values_loc, bool expected_is_cst_val2) noexcept
    {
        // A chunk's second constant IS its values pointer (no per-chunk copy). The CAS must
        // still expect a CONSTANT, never "whatever values holds now": if another thread
        // materialised the chunk since the caller looked, values is its buffer, and a CAS
        // expecting it would replace that buffer -- and its write -- by ours (a lost write).
        // So if the loaded pointer is not in the second-value region, report failure without
        // attempting the CAS; the caller recycles our buffer.
        T* expected = nullptr;
        if constexpr (is_not_bool)
        {
            if (expected_is_cst_val2)
            {
                expected = self.values_load();
                if (!Holder::is_cst_val2_ptr(expected))
                {
                    return true; // failed: someone else materialised it first
                }
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

        // Spelled out, not self.Base::: name lookup in `self` finds Chunk2A's own
        // private Base alias first (clang rejects it).
        const bool failed = self.SecondValueChunkBase<T, Derived>::compare_and_swap_gen(values_loc, expected);

        if constexpr (build_config::any_check)
        {
            if (!failed)
            {
                T* current = self.values_load();
                if (current != values_loc && current != Base::get_cst_val1() && !Holder::is_cst_val2_ptr(current))
                {
                    invariant_failed(std::source_location::current(),
                                     "post-create_value CAS: values={}, expected one of "
                                     "{{values_loc={}, cst1={}, a cst2 array}}",
                                     static_cast<const void*>(current), static_cast<const void*>(values_loc),
                                     static_cast<const void*>(Base::get_cst_val1()));
                }
            }
        }
        if constexpr (chunk_config::ring_log_debug)
        {
            const uint32_t chunk_id = self.ring_chunk_id();
            const uint32_t info =
                (chunk_id & 0x00FFFFFFu) | (static_cast<uint32_t>(expected_is_cst_val2 ? 2 : 1) << 24U);
            ring_log(get_tid(), failed ? ev_cas_values_fail : ev_cas_values_from_cst, info);
        }
        return failed;
    }

  private:
    static constexpr bool is_not_bool = !std::is_same_v<T, bool>;

    /// @brief Compact identity of this chunk for the ring log: its address past
    ///        the alignment bits. Callers mask it to 24 bits and pack a tag into
    ///        the top byte.
    [[nodiscard]] uint32_t ring_chunk_id(this const auto& obj) noexcept
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&obj) >> 4U);
    }
};

// Read-side counterpart of Chunk2AConcurrentOps: lock-free classification of `values`. Needed in
// every build, so mixed into Chunk2A UNCONDITIONALLY (not gated on threaded). Same
// deducing-this pattern, const-qualified: every query is a pure read.
/// @brief Chunk2A's lock-free reads of its constant state (cst_val1 / cst_val2 / written), every build.
template <typename T, typename Derived>
class ChunkStateQueries
{
  public:
    using Base   = SecondValueChunkBase<T, Derived>;
    using Holder = TabCstVal2Holder<T>;

    /// @brief Is the chunk compressed to a second value (its values pointer
    ///        lies in the second-value region; for bool, IS the flag block)?
    [[nodiscard]] bool is_cst_val2(this const Derived& self) noexcept
    {
        const T* current_value = self.values_load();
        if constexpr (is_not_bool)
        {
            return Holder::is_cst_val2_ptr(current_value); // a range test on the region
        }
        else
        {
            return current_value == Holder::m_ptr.data();
        }
    }

    /// @brief is_cst_val2() asked of a pointer the caller already loaded.
    /// @details element_at() must test and read the SAME load: asking the chunk
    ///          loads `values` a second time, and a compressor claiming it in
    ///          between swaps that pointer for a region identifier, which the
    ///          read then indexes as a buffer.
    [[nodiscard]] static bool is_cst_val2_ptr_of(const T* ptr) noexcept
    {
        if constexpr (is_not_bool)
        {
            return Holder::is_cst_val2_ptr(ptr);
        }
        else
        {
            return ptr == Holder::m_ptr.data();
        }
    }

    /**
     * @brief The constant a chunk whose values pointer is @p ptr stands for,
     *        read from the registry, for a pointer the caller already loaded.
     *
     * @pre @p ptr satisfies is_cst_val2_ptr_of().
     * @details Used by ChunkBase::element_at(). Must NOT dereference @p ptr (it names
     *          an identifier, not a buffer): id_of_ptr() is arithmetic, value_of_id()
     *          one load of the value table.
     */
    [[nodiscard]] static T second_value_from_ptr(const T* ptr) noexcept
    {
        if constexpr (is_not_bool)
        {
            return Holder::value_of_id(Holder::id_of_ptr(ptr));
        }
        else
        {
            return true;
        }
    }

    /// @brief Is @p ptr one of the constant arrays (first or a second one)?
    /// @note For a values pointer the caller already loaded -- the pre-store
    ///       guards of the proxy -- where is_cst_val() would load it again.
    [[nodiscard]] static bool is_constant_ptr(const T* ptr) noexcept
    {
        if (ptr == Base::get_cst_val1())
        {
            return true;
        }
        if constexpr (is_not_bool)
        {
            return Holder::is_cst_val2_ptr(ptr);
        }
        else
        {
            return ptr == Holder::m_ptr.data();
        }
    }

    /// @brief Materialised (owns a buffer, i.e. compressed to neither constant)?
    [[nodiscard]] bool written(this const Derived& self) noexcept
    {
        return self.values_load() != Base::get_cst_val1() && self.is_not_cst_val2();
    }

    /// @brief written() reading the values member directly, without the
    ///        atomic load: for a sequential point (the array's stats), where no
    ///        writer or compressor can be publishing.
    [[nodiscard]] bool written_test(this const Derived& self) noexcept
    {
        return self.values_ != Base::get_cst_val1() && self.is_not_cst_val2();
    }

  protected:
    /// @brief Compressed to either constant?
    [[nodiscard]] bool is_cst_val(this const Derived& self) noexcept
    {
        return self.Base::is_cst_val1() || self.is_cst_val2();
    }

  private:
    static constexpr bool is_not_bool = !std::is_same_v<T, bool>;

    /// @brief Complement of is_cst_val2().
    [[nodiscard]] bool is_not_cst_val2(this const Derived& self) noexcept
    {
        return !self.is_cst_val2();
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
 * @tparam Enabled build_config::threaded
 */
template <typename T, typename Derived, bool Enabled = build_config::threaded>
class ChunkOmpPolicy : public SecondValueChunkBase<T, Derived>, public ChunkConcurrentOps<T, Derived>
{
  public:
    using Base                   = SecondValueChunkBase<T, Derived>;
    using ManagerChunk2A         = ManagerChunk<Chunk, T>;
    using CompressionManagerType = CompressionManager<T, Derived>;

    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values_;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new Chunk2A<T>[n]` runs the DEFAULT CONSTRUCTORS after this, overwriting
    ///          that initialisation; it pays off only for a placement-new array. Nothing in
    ///          the library array-news a chunk (zero::Array* hold a std::vector).
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
    void clear(this Derived& self, char* ptr_type_cst, SecondValueRef<T> ptr_second_value, int thr_rank);

    /*!
     * \brief Release this chunk's buffer and point it back at a constant --
     *        the single point every compression path funnels through.
     * \param num_compress_value 0 for the first constant, otherwise the second.
     * \param ptr_type_cst       Slot of the per-chunk type table to update.
     * \param ptr_second_value   The chunk's second-value slot.
     * \param thr_rank           Rank whose pool receives the released buffer.
     * \param cst2_ptr           Shared array of the second constant (type 2).
     */
    void block_clear(this Derived&     self,
                     int               num_compress_value,
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
    void clear_int_st(this Derived&     self,
                      int               num_compress_value,
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
    void clear_st(this Derived& self, int num_compress_value, char* ptr_type_cst, SecondValueRef<T> ptr_second_value)
    {
        self.clear_int_st(num_compress_value, ptr_type_cst, ptr_second_value, CHUNK_PMP_THREAD_RANK);
    }

  private:
    /// @brief Point @p count freshly allocated chunks at the first constant.
    /// @details Parallel past 500 chunks, and only when called from OUTSIDE a
    ///          parallel region (inside one it stays on the calling thread: an
    ///          allocation is made by one thread, and an orphaned `omp for`
    ///          would hang waiting for the others). Shared by the three
    ///          variants -- chunk_array_new::init_chunks.
    static void init_chunks_cst_val(Chunk<T>* v, std::size_t count) noexcept;
};

/**
 * @brief Serial specialisation: no rank, no CAS, no in-flight wait; and the
 *        eager clear verbs the sequential compression uses.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk2A<T>)
 */
template <typename T, typename Derived>
class ChunkOmpPolicy<T, Derived, false> : public SecondValueChunkBase<T, Derived>
{
  public:
    using Base           = SecondValueChunkBase<T, Derived>;
    using ManagerChunk2A = ManagerChunk<Chunk, T>;

    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values_;

    /// @brief The single candidate list of a serial build.
    static TChunk2AComp<Chunk, T>& get_vec_tuple_to_compress_potentially() noexcept
    {
        return ManagerChunk2A::chunk2a_to_compress_potentially;
    }

    /*!
     * \brief Materialise this chunk's buffer (allocate + fill + assign).
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param ptr_second_value      The chunk's second value; read only when the flag is set.
     */
    void create_value(this Derived& self, bool need_set_second_value, SecondValueRef<T> ptr_second_value);

    /// @brief Bool specialization of create_value().
    void create_value_bool(this Derived& self, bool need_set_second_value);

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
                     int /*thr_rank*/,
                     T* cst2_ptr);
};

/// @brief Chunk of values compressible to the first constant or to a second constant shared through the
///        TabCstVal2Holder registry (see the class description above Chunk2AConcurrentOps).
/// @tparam T Element type.
template <typename T>
// no move operations on purpose -- a move falls back to
// the copy, which carries the per-chunk state the way the copy constructor documents.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Chunk : public ChunkOmpPolicy<T, Chunk<T>>,
                public TabCstVal2Holder<T>,
                public ChunkStateQueries<T, Chunk<T>>
{
  public:
    friend class SecondValueChunkBase<T, Chunk>;
    friend class Proxy<T>;

    using Base      = SecondValueChunkBase<T, Chunk>;                               ///< The CRTP chunk base.
    using OmpPolicy = ChunkOmpPolicy<T, Chunk, build_config::threaded>; ///< This build's thread-model policy.

    using ManagerChunk2A = ManagerChunk<Chunk, T>;
    // Same tuple as TChunk.h's VectorTuple4ChunkInt2 (the manager storage).
    using VecTupleChunkA2pInt2  = std::vector<std::tuple<Chunk*, int, int, ChunkElems>>;
    using VecTupleChunkA2pTint2 = std::vector<std::tuple<Chunk*, T, int, int, ChunkElems>>;

    using Base::allocator_data;
    using Base::first_value;
    using Base::values_;

    using Holder = TabCstVal2Holder<T>;

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

    Chunk()  = default;
    ~Chunk() = default;

    /// @brief Copy the per-chunk state of @p other (Base::values_store(other.values_load())).
    Chunk(const Chunk& other)
    {
        Base::values_store(other.values_load()); // the whole per-chunk state
    }

    /// @brief Copy the per-chunk state of @p other unless it is this chunk.
    /// @return *this.
    Chunk& operator=(const Chunk& other)
    {
        if (this != &other)
        {
            Base::values_store(other.values_load());
        }
        return *this;
    }

    /**
     * @brief Initialize values after construction
     */
    void init_after_constructor() noexcept
    {
        Base::values_ = Base::get_cst_val1();
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
            Base::values_ = Base::get_cst_val1();
            return;
        }
        if constexpr (is_not_bool)
        {
            T* cst_val2_ptr = Holder::find_cst_val2_ptr(ptr_second_value.get());
            if constexpr (build_config::any_check)
            {
                if (cst_val2_ptr == nullptr)
                {
                    invariant_failed(std::source_location::current(),
                                     "init_values_cst_val: type says cst_val2 but no array is registered for the "
                                     "chunk's second value");
                }
            }
            Base::values_ = cst_val2_ptr;
        }
        else
        {
            Base::values_ = Holder::m_ptr.data();
        }
    }

    /// @brief Get reference to the chunk manager (static access)
    /// @return Reference to the ManagerChunk2A static instance
    ///
    static auto& get_manager() noexcept
    {
        // Do not use a function-local static (lazy Meyers singleton): its first init can
        // happen from compress() inside an OpenMP region and deadlock threads on
        // __cxa_guard_acquire. The inline static member is built before any thread.
        return *s_manager_;
    }

    /// @brief Rank @p thr's candidate list.
    static TChunk2AComp<Chunk, T>& get_vec_tuple_to_compress_potentially_thr(int thr) noexcept
    {
        return ManagerChunk2A::get_chunk2a_to_compress_potentially(thr);
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
            {
                create_value(false, ptr_second_value);
            }
            else
            {
                create_value_bool(false);
            }
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

            if constexpr (build_config::threaded)
            {
                if constexpr (is_not_bool)
                {
                    create_value(false, ptr_second_value, thr_rank);
                }
                else
                {
                    create_value_bool(false, thr_rank);
                }
            }
            else
            {
                if constexpr (is_not_bool)
                {
                    create_value(false, ptr_second_value);
                }
                else
                {
                    create_value_bool(false);
                }
            }
        }
    }

    void compress(int thr_rank, int num_chunk, int num_tab, bool apply, ChunkElems size);

    /// @brief Sets every element of this chunk to @p v by its state alone,
    ///        with no buffer touched: a whole-chunk fill (fill_range, a
    ///        uniform copy_range span). Non-bool T only.
    /// @pre   Serial contract (no concurrent writer, reader or pass on the
    ///        array); the array's deferred writes drained.
    void fill_constant_st(int thr_rank, int num_chunk, int num_tab, InT<T> v);

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
        return build_config::threaded && chunk_config::not_use_compression_with_ompt && apply;
    }
    static bool have_chunk_to_compress(int max_thr);

    static void make_block_compression(int thr, Proxy<T>* proxy2a);

  private:
    using ChunkType              = Chunk;
    using CompressionManagerType = CompressionManager<T, ChunkType>;

    static constexpr char is_val1 = 0;
    static constexpr char is_val2 = 1;

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    /// @brief Is every rank's candidate list empty (the calling thread's, in a
    ///        serial build)?
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        if constexpr (build_config::threaded)
        {
            return ManagerChunk2A::is_chunk2a_to_compress_potentially_empty();
        }
        else
        {
            return ManagerChunk2A::get_chunk2a_to_compress_potentially().empty();
        }
    }

    static void compress_chunk_to_compress_in_par(int thr, bool apply);

    // Chunk geometry is PER ENTRY in dynamic builds: every queued tuple carries the owning
    // array's chunk size (field <3>, from push_to_compress_list). The scan bounds come from
    // that field. Do not add a `= static_chunk_size` default: it would scan only the first
    // static_chunk_size values of a wider chunk, declare it constant and destroy the rest.
    static void fill_list_chunk_to_compress_in_par(int                    thr,
                                                   VecTupleChunkA2pInt2&  ll_compt1,
                                                   VecTupleChunkA2pTint2& ll_compt2,
                                                   bool                   claimed = false);

    void fill_list_to_compress(VecTupleChunkA2pInt2&  ll_compt1,
                               VecTupleChunkA2pTint2& ll_compt2,
                               int                    num_chunk,
                               int                    num_tab,
                               ChunkElems             size,
                               bool                   claimed = false);

    static inline constinit internal::LazyImmortal<ManagerChunk2A> s_manager_;
};

} // namespace zero_any

} // namespace shrink_array

#include "chunk/internal/zero_any_chunk_inline.h" // the template bodies
