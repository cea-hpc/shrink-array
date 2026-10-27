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

#include <algorithm>   // std::fill_n
#include <array>       // std::array
#include <deque>       // tab_m_ptr_cst_val2: grows without moving its arrays
#include <atomic>      // std::memory_order_acquire, std::memory_order_release
#include <cstddef>     // std::size_t
#include <tuple>       // the candidate-list entries
#include <type_traits> // std::is_same_v, std::conditional_t
#include <vector>      // std::vector
#include <cstring>     // std::memset (the Bit second constant)

#include "chunk/internal/chunk_size_config.h"                   // static_chunk_size
#include "chunk/internal/chunk_size_policy.h"                   // ChunkElems, DynamicChunkSize
#include "chunk/internal/in_t.h"                                // InT
#include "chunk/internal/element_access.h"                      // Bit buffers: words of bits
#include "chunk/internal/bit_words.h"                           // bits::fill_bits, bits::WordT
#include "chunk/config.h"                                       // build_config::threaded, chunk_config
#include "chunk/internal/invariant_check.h"                     // CHUNK_INVARIANT_FAILED
#include "chunk/internal/bad_ptr_check.h"                       // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/PMP_CHUNK.h"                                    // get_thr_rank_fast, CHUNK_PMP_THREAD_RANK
#include "chunk/internal/zero_one_value_candidate_list_check.h" // chunk2_missing_candidate_list
#include "chunk/internal/zero_one_value_proxy.h"                // zero_one_value::Proxy
#include "chunk/internal/second_value_chunk_base.h"             // SecondValueChunkBase
#include "chunk/internal/conditional_counter.h"                 // ConditionalCounter, CounterValueT
#include "chunk/internal/TChunk.h"                              // VectorTuple4ChunkInt2 (the candidate lists)
#include "chunk/internal/immortal.h"          // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/second_value_info.h" // SecondValueInfo (InfArrayT)
#include "chunk/internal/chunk_array_new.h"   // chunk_array_new::init_chunks (init_chunks_cst_val)

namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkT>
struct CompressionManager;

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

namespace zero_one_value
{

template <typename T>
class Proxy;

/// @brief Static table of the registered second-constant arrays of element type T (tab_m_ptr_cst_val2),
///        with the lookups and checks on it.
template <typename T>
class ChunkSpec
{
  public:
    // The registered second-constant arrays. A chunk compressed to one stores its ADDRESS as
    // `values`, so the container must never move them: use a std::deque grown at the end, not a
    // std::vector (a reallocation would leave every such chunk pointing at freed memory).
    // zero_one_value_registry_growth_validation guards it. MM_PROTECT holds pointers to its own
    // mappings, which never move.
    using VecArrayChunkSizeMax =
        std::conditional_t<build_config::use_mm_protect, std::vector<T*>, std::deque<std::array<T, cst_buffer_size>>>;

    /*!
     * \brief Gets the pointer to the second constant value.
     * \param ind Index of the second value.
     * \return Pointer to the constant value array.
     */
    [[nodiscard]] static T* get_cst_val2(int ind) noexcept;
    /*!
     * \brief Checks whether `values` IS the second constant's pointer at
     *        index `ind` (i.e. the chunk is currently compressed to it).
     * \param values Pointer to compare.
     * \param ind Index of the second value.
     * \return True if values equals the second constant at ind.
     */
    [[nodiscard]] bool is_cst_val2(const T* values, int ind) const noexcept
    {
        return values == get_cst_val2(ind);
    }


    /*!
     * \brief Fill a freshly materialised buffer with the second value
     *        carried in `inf_array`.
     * \param values_loc Buffer to fill.
     * \param inf_array (second_value, ...) tuple; only the value is used here.
     */
    static void init_block_2nd_value(T* values_loc, const SecondValueInfo<T>& inf_array)
    {
        // InT: fill_n takes a const T& anyway; no copy of a struct on the way.
        const InT<T> value = inf_array.second_value;
        std::fill_n(values_loc, alloc_chunk_count(), value);
    }

    static inline constinit internal::LazyImmortal<VecArrayChunkSizeMax> tab_m_ptr_cst_val2;

    // mm_protect mode only: element count allocated at (*tab_m_ptr_cst_val2)[ind] (a raw
    // mmap'd T*; the plain mode uses fixed cst_buffer_size arrays). Lets set_second_value's
    // reuse path (a known second_value registered at a LARGER geometry) grow an undersized
    // allocation instead of overrunning it. Declared but unused in the other builds.
    static inline constinit internal::Immortal<std::vector<std::size_t>> tab_m_ptr_cst_val2_mm_elems_storage_{};
    static inline std::vector<std::size_t>& tab_m_ptr_cst_val2_mm_elems = tab_m_ptr_cst_val2_mm_elems_storage_.value;

  protected:
    /*!
     * \brief Whether `v` is ANY of the registered second-constant pointers,
     *        regardless of index. For teardown, where the chunk's own index is not
     *        to hand: is_cst_val2(v, ind) with a wrong index would read the marker
     *        of a chunk compressed at another index as a live buffer.
     * \param v Pointer to test.
     * \return True if v matches the second constant at some index.
     */
    [[nodiscard]] static bool is_cst_val2_any(const T* vals) noexcept;

  private:
    static constexpr bool use_mm_protect = build_config::use_mm_protect;

    /*!
     * \brief Checks that this chunk is NOT currently compressed to the
     *        second constant at index `ind`.
     * \param ind Index of the second value.
     * \return True if the chunk's own values pointer differs from the
     *         second constant at ind.
     */
    [[nodiscard]] bool is_not_cst_val2(int ind) const noexcept
    {
        return !is_cst_val2(ind);
    }
};

/// @brief zero_one_value::ChunkSpec for bool and Bit (bit.h): the second constant is `true`, one for the type.
template <typename T>
    requires is_bool_like_v<T>
class ChunkSpec<T>
{
  public:
    /*!
     * \brief Gets the pointer to the (single, shared) second constant value.
     * \return Pointer to the constant value array.
     */
    [[nodiscard]] static T* get_cst_val2() noexcept
    {
        return m_ptr_cst_val2.data();
    }

    /*!
     * \brief Checks whether `values` IS the second constant's pointer (i.e.
     *        the chunk is currently compressed to it).
     * \param values Pointer to compare.
     * \return True if values equals the second constant.
     */
    [[nodiscard]] static bool is_cst_val2(const T* values) noexcept
    {
        return values == get_cst_val2();
    }

    /*!
     * \brief Fill a freshly materialised buffer with `true` (bool's only
     *        possible second value).
     * \param values_loc Buffer to fill.
     */
    static void init_block_2nd_value(T* values_loc);

  protected:
    // Bit: every BIT set -- the buffer is read as 64-bit words, hence the
    // bytes 0xFF (a Bit{true} object is the byte 0x01) and the alignment.
    alignas(bits::WordT) static inline std::array<T, cst_buffer_size> m_ptr_cst_val2 = []
    {
        std::array<T, cst_buffer_size> arr{};
        if constexpr (is_packed_bit_v<T>)
        {
            std::memset(static_cast<void*>(arr.data()), 0xFF, sizeof(arr)); // raw words, not Bit objects
        }
        else
        {
            arr.fill(true);
        }
        return arr;
    }();

  private:
    /*!
     * \brief Negation of is_cst_val2().
     * \param values Pointer to compare.
     * \return True if values does NOT equal the second constant.
     */
    [[nodiscard]] static bool is_not_cst_val2(const T* values) noexcept
    {
        return !is_cst_val2(values);
    }
};

/*!
 * \brief Per-chunk element counters (how many elements currently equal the
 *        first/second constant), zero-cost when EnableCounters is false
 *        (ConditionalCounter collapses to an empty type).
 * \tparam EnableCounters Whether the counters actually exist and count.
 */
template <bool EnableCounters>
class ChunkCounterPolicy
{
  public:
    /// @brief Reset the second-value counter to 0.
    void reset_nb_val_eq_second_value() noexcept
    {
        nb_val_eq_second_value_ = 0;
    }

    /// @brief Set the second-value counter to an explicit value.
    /// @param count New counter value.
    void set_nb_val_eq_second_value(CounterValueT count) noexcept
    {
        nb_val_eq_second_value_ = count;
    }

    /// @brief Whether every element in the chunk currently equals the second
    ///        constant (the chunk is a candidate for type-2 compression).
    /// @param size Chunk geometry; defaults to the compile-time size (callers that know the
    ///             runtime geometry, e.g. under DynamicChunkSize, pass it).
    /// @return true if nb_val_eq_second_value == size.
    [[nodiscard]] bool is_full_second_value(unsigned int size = static_chunk_size) const noexcept
    {
        return nb_val_eq_second_value_ == size;
    }

    /// @brief Increment the second-value counter (one more element now
    ///        equals the second constant).
    void inc_nb_val_eq_second_value() noexcept
    {
        ++nb_val_eq_second_value_;
    }

    /// @brief Decrement the second-value counter (one fewer element now
    ///        equals the second constant).
    void dec_nb_val_eq_second_value() noexcept
    {
        --nb_val_eq_second_value_;
    }

    /// @brief Increment the first-value counter (one more element now
    ///        equals the first constant).
    void inc_nb_val_eq_first_value() noexcept
    {
        ++nb_val_eq_first_value_;
    }

    /// @brief Decrement the first-value counter (one fewer element now
    ///        equals the first constant).
    void dec_nb_val_eq_first_value() noexcept
    {
        --nb_val_eq_first_value_;
    }

    /// @brief Reset the first-value counter to 0.
    void reset_nb_val_eq_first_value() noexcept
    {
        nb_val_eq_first_value_ = 0;
    }

    /// @brief Set the first-value counter to an explicit value.
    /// @param count New counter value.
    void set_nb_val_eq_first_value(CounterValueT count) noexcept
    {
        nb_val_eq_first_value_ = count;
    }

    /// @brief Whether every element in the chunk currently equals the first
    ///        constant (the chunk is a candidate for type-1 compression).
    /// @param size Chunk geometry; see is_full_second_value().
    /// @return true if nb_val_eq_first_value == size.
    [[nodiscard]] bool is_full_first_value(unsigned int size = static_chunk_size) const noexcept
    {
        return nb_val_eq_first_value_ == size;
    }

    /// @brief Current first-value counter.
    /// @return Number of elements currently equal to the first constant.
    [[nodiscard]] CounterValueT get_nb_first_value() const noexcept
    {
        return nb_val_eq_first_value_;
    }

    /// @brief Current second-value counter.
    /// @return Number of elements currently equal to the second constant.
    [[nodiscard]] CounterValueT get_nb_second_value() const noexcept
    {
        return nb_val_eq_second_value_;
    }

  private:
    /// @brief Reset both counters to 0.
    void reset_nb_val_eq_value() noexcept
    {
        reset_nb_val_eq_first_value();
        reset_nb_val_eq_second_value();
    }

    // [[no_unique_address]] and two distinct tags: when disabled both counters are empty and
    // share offset 0, so the policy is empty and `values` stays at offset 0 (see
    // ChunkCounterPolicy for the padding this saves).
    [[no_unique_address]] ConditionalCounter<EnableCounters, 0> nb_val_eq_first_value_;
    [[no_unique_address]] ConditionalCounter<EnableCounters, 1> nb_val_eq_second_value_;
};
/**
 * @brief zero_one_value's OpenMP-only surface: per-thread materialisation and clearing.
 *
 * @details A serial build has no thread ranks, no CAS window and no per-thread
 *          free-lists, and maintains per-chunk element counters that the parallel
 *          path does not; each specialisation declares exactly the verbs its mode
 *          has (as ChunkThreadPolicy does for zero). Sits between
 *          zero_one_value::ChunkFastCompPolicy and SecondValueChunkBase, so zero_one_value inherits both through one
 * path.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (zero_one_value::Chunk<T>)
 * @tparam Enabled build_config::threaded
 */
template <typename T, typename Derived, bool Enabled = build_config::threaded>
class ChunkThreadPolicy : public SecondValueChunkBase<T, Derived>
{
  public:
    using Base          = SecondValueChunkBase<T, Derived>;
    using BaseSpec      = ChunkSpec<T>;
    using ManagerChunk2 = ManagerChunk<Chunk, T>;
    using InfArrayT     = SecondValueInfo<T>;

    using Base::first_value;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values_;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new zero_one_value::Chunk<T>[n]` runs the default constructors after this, which
    ///          overwrite that initialisation; it only matters for placement-new
    ///          arrays. The library never array-news a chunk (zero::Array* use a
    ///          std::vector<ChunkType>).
    void* operator new[](std::size_t sz);
    /// @brief Matching array-delete.
    void operator delete[](void* ptr);

    /*!
     * \brief Materialise this chunk's buffer (allocate + init), for a
     *        specific thread. Non-bool only.
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param inf_array (second_value, index, ...) tuple of the owning array.
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void
    create_value(this Derived& self, bool need_set_second_value, const InfArrayT& inf_array, int thr_rank) noexcept;

    /*!
     * \brief Bool specialization of create_value().
     * \param need_set_second_value True to pre-fill with `true`, false for first_value.
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void create_value_bool(this Derived& self, bool need_set_second_value, int thr_rank) noexcept;

    /*!
     * \brief create_value() with the calling thread's rank (get_thr_rank_fast()).
     *        Non-bool only (a compile error otherwise).
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param inf_array (second_value, index, ...) tuple of the owning array.
     */
    void create_value(this Derived& self, bool need_set_second_value, const InfArrayT& inf_array) noexcept;

    /*!
     * \brief create_value_bool() with the calling thread's rank. Bool only (a
     *        compile error otherwise).
     * \param need_set_second_value True to pre-fill with `true`, false for first_value.
     */
    void create_value_bool(this Derived& self, bool need_set_second_value) noexcept;

    /*!
     * \brief Install the constant and release the old buffer (every compression path).
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void block_clear(this Derived& self, int num_compress_value, int ind, int thr_rank);

    /*!
     * \brief Compress this chunk to a constant, for a specific thread. Non-bool.
     *        Called by make_block_compression_impl()'s clear_func.
     * \param num_compress_value 0 for the first constant, non-zero for the second at `ind`.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     * \pre The chunk is materialised -- compressing an already-constant chunk would
     *      release a shared constant block. Checked by check_not_constant_before_clear().
     */
    void clear_int_st(this Derived& self, int num_compress_value, int ind, int thr_rank)
    {
        self.Base::check_not_constant_before_clear(num_compress_value, ind);
        self.block_clear(num_compress_value, ind, thr_rank);
    }

    /*!
     * \brief Bool counterpart of the 3-arg clear_int_st().
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param thr_rank Calling thread's rank.
     */
    void clear_int_st_bool(this Derived& self, int num_compress_value, int thr_rank);

    /*!
     * \brief Compress this chunk to a constant, with the calling thread's rank.
     * \param num_compress_value 0 for the first constant, non-zero for the second at `ind`.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void clear_st(this Derived& self, int num_compress_value, int ind)
    {
        self.clear_int_st(num_compress_value, ind, CHUNK_PMP_THREAD_RANK);
    }

  private:
    /// @brief Point @p count freshly allocated chunks at the first constant.
    /// @details Parallel past 500 chunks, and only from OUTSIDE a parallel region. Inside
    ///          one it stays on the calling thread: only that thread allocates, and an
    ///          orphaned `omp for` would hang waiting for the others. Shared by the
    ///          variants (chunk_array_new::init_chunks).
    static void init_chunks_cst_val(Chunk<T>* chunks, std::size_t count) noexcept
    {
        chunk_array_new::init_chunks(chunks, count, [](Chunk<T>& chunk) { chunk.init_after_constructor(); });
    }

    /**
     * \brief Shared body of create_value() and create_value_bool().
     * \tparam IsBoolCas true to publish with compare_and_swap_bool, false with
     *         the indexed compare_and_swap -- selected here, NOT passed as a
     *         lambda; see the definition for why that is load-bearing.
     * \tparam InitFn callable (T*) pre-filling the allocated block.
     */
    template <bool IsBoolCas, typename InitFn>
    void create_value_impl(this Derived& self,
                           int           thr_rank,
                           InitFn&&      init_func,
                           int           ind,
                           bool          need_set_second_value) noexcept;

    /*!
     * \brief Compress this chunk to its second constant at index `ind`
     *        (OpenMP/deferred-queue algorithm). Non-bool.
     * \param ind Index of the second constant to compress to.
     * \param thr_rank Calling thread's rank.
     */
    void clear(this Derived& self, int ind, int thr_rank)
    {
        self.Base::clear_impl(ind, thr_rank);
    }
};

/**
 * @brief Serial specialisation: no rank, no CAS, and the element counters to reset.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (zero_one_value::Chunk<T>)
 */
template <typename T, typename Derived>
class ChunkThreadPolicy<T, Derived, false> : public SecondValueChunkBase<T, Derived>
{
  public:
    using Base      = SecondValueChunkBase<T, Derived>;
    using BaseSpec  = ChunkSpec<T>;
    using InfArrayT = SecondValueInfo<T>;

    using Base::first_value;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values_;

    /*!
     * \brief Materialise this chunk's buffer (allocate + init). Serial: no thr_rank.
     *        Non-bool only.
     * \param need_set_second_value True to pre-fill with the second value, false for first_value.
     * \param inf_array (second_value, index, ...) tuple of the owning array.
     */
    void create_value(this Derived& self, bool need_set_second_value, const InfArrayT& inf_array) noexcept;

    /*!
     * \brief Bool specialization of create_value() (serial builds).
     * \param need_set_second_value True to pre-fill with `true`, false for first_value.
     */
    void create_value_bool(this Derived& self, bool need_set_second_value) noexcept;

    /*!
     * \brief Compress this chunk to a constant (serial/counter algorithm: eager, no queue).
     * \param num_compress_value 0 for the first constant, non-zero for the second at `ind`.
     * \param ind Index of the second constant (unused when num_compress_value == 0, and for bool).
     */
    void clear_int_st(this Derived& self, int num_compress_value, int ind);

    /*!
     * \brief Compress this chunk to its first constant (serial/counter algorithm).
     * \param ind Only for the debug_check_written(ind) sanity check; the target is
     *            always the first constant.
     */
    void clear_st(this Derived& self, int ind);

    /*!
     * \brief Compress this chunk to a constant (serial twin of the OpenMP
     *        clear_st(int, int): no rank to supply).
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void clear_st(this Derived& self, int num_compress_value, int ind)
    {
        self.clear_int_st(num_compress_value, ind);
    }

    /*!
     * \brief Install the constant and free the old buffer outright (no rank pool);
     *        resets the per-chunk element counters.
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void block_clear(this Derived& self, int num_compress_value, int ind, int /*thr_rank*/);

  private:
    /**
     * \brief Shared body of the two serial create_value entry points: they
     *        differ only in how the second-value fill is spelled.
     * \tparam Init2ndFn callable (T*) filling the block with the second value.
     */
    template <typename Init2ndFn>
    void create_value_serial_impl(this Derived& self, bool need_set_second_value, Init2ndFn&& init_2nd) noexcept;
};

/// @brief zero_one_value's batched candidate queue and block compression (fast-block builds).
template <typename T, typename Derived, bool Enabled>
class ChunkFastCompPolicy : public ChunkSpec<T>, public ChunkThreadPolicy<T, Derived>
{
  public:
    using ManagerChunk2 = ManagerChunk<Chunk, T>;
    using Base          = SecondValueChunkBase<T, Derived>;
    using BaseSpec      = ChunkSpec<T>;

    using Base::first_value;
    using Base::values_;

    // Local aliases, as ChunkFastCompPolicy carries its own rather than reaching into Derived.
    // (chunk, num_chunk, num_tab, n_elems) -- n_elems is the chunk's OWN geometry, carried
    // from the candidate queue so the clear loop can pin the thread to it before freeing.
    using VecTupleChunk2pInt2 = std::vector<std::tuple<Chunk<T>*, int, int, unsigned int>>;
    using InfArrayT           = SecondValueInfo<T>;

    /**
     * @brief Scans all threads to see if any chunk is marked for compression with a specific value.
     */
    static bool have_chunk_to_compress(T second_value, int max_thr);

    /**
     * @brief Boolean specific version of the compression check.
     */
    static bool have_chunk_to_compress_bool(int max_thr)
    {
        // bool's second value is always `true`.
        return have_chunk_to_compress(static_cast<T>(true), max_thr);
    }


    /**
     * @brief CAS a chunk's second-value slot from its expected constant state
     *        to a newly materialised buffer.
     * @param values_loc New buffer to install on success.
     * @param ind Index used to resolve the expected second-value constant.
     * @param expected_is_cst_val2 Whether the expected current state is the
     *                             second constant (vs the first).
     * @return true on failure (caller owns values_loc), false on success.
     */
    [[nodiscard]] bool compare_and_swap(this Derived& self, T* values_loc, int ind, bool expected_is_cst_val2) noexcept;

    /**
     * @brief Bool path of compare_and_swap() -- no ind (bool has a single shared array).
     * @param values_loc New buffer to install on success.
     * @param expected_is_cst_val2 Whether the expected current state is the second constant.
     * @return true on failure (caller owns values_loc), false on success.
     */
    [[nodiscard]] bool compare_and_swap_bool(this Derived& self, T* values_loc, bool expected_is_cst_val2) noexcept;

    /**
     * @brief Compress every non-bool candidate chunk of (thr_rank, inf_array's second value).
     * @param thr_rank Thread whose candidate list to drain.
     * @param inf_array (second_value, ...) identifying the candidate list.
     * @param proxy2 Owning proxy, or nullptr for the thread-only sweep.
     */
    static void make_block_compression(int thr_rank, const InfArrayT& inf_array, Proxy<T>* proxy2);

    /**
     * @brief Bool specialization of make_block_compression().
     * @param thr_rank Thread whose candidate list to drain.
     * @param proxy2 Owning proxy, or nullptr for the thread-only sweep.
     */
    static void make_block_compression_bool(int thr_rank, Proxy<T>* proxy2);

    /**
     * @brief Thread-only sweep: drain every second_value candidate list of thr_rank.
     * @param thr_rank Thread whose candidate lists to drain.
     */
    static void make_block_compression(int thr_rank);

    // --- queue-driven compression ------------------------------------------
    // The false specialisation below has no queue and declares none of it.

    /*!
     * \brief Drain the compression-candidate queue for the given second
     *        value, across all threads.
     * \param inf_array (second_value, index, ...) identifying the candidate list.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress(const InfArrayT& inf_array, bool apply);

    /*!
     * \brief Bool specialization of compress_chunk_to_compress().
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_bool(bool apply);

    /// @brief One rank's share of compress_chunk_to_compress() for one second
    ///        value: the body its region runs on each thread. Public for
    ///        AllTypeChunk's single sweep region (ManagerChunk::sweep_rank).
    static void compress_sweep_rank(int thr_rank, const InfArrayT& inf_array, bool apply)
    {
        compress_chunk_to_compress_in_par(thr_rank, inf_array, apply);
    }
    /// @brief Bool twin of compress_sweep_rank().
    static void compress_sweep_rank_bool(int thr_rank, bool apply)
    {
        compress_chunk_to_compress_bool_in_par(thr_rank, apply);
    }

    /*!
     * \brief Per-candidate step of compress_chunk_to_compress_in_par(): re-scans THIS
     *        chunk's live content (the queue entry alone is not trusted) and, when
     *        @p apply, compresses it via clear_int_st() if still uniform.
     * \param thr_rank Calling thread's rank (for clear_int_st()).
     * \param inf_array (second_value, ...) identifying the constant and index to test.
     * \param apply The scan always runs; compression only when true.
     * \param n_elems Element count of this chunk (the owning array's geometry).
     */
    void compress(this Derived& self, int thr_rank, const InfArrayT& inf_array, bool apply, unsigned int n_elems);

  private:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /**
     * @brief Checks if the per-thread compression candidate map is empty.
     */
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        return ManagerChunk2::map_chunk2_to_compress_potentially_thr.empty() ||
               ManagerChunk2::map_chunk2_to_compress_potentially_thr[0].empty();
    }

    /**
     * @brief Per-(thread, second_value) list of chunks marked as compression
     *        candidates.
     * @param thr Thread rank whose candidate list to fetch.
     * @param second_value The type-2 constant these chunks are candidates for.
     * @return Reference to the (thr, second_value) candidate list.
     */
    static VectorTuple4ChunkInt2<Chunk, T>& get_vec_tuple_to_compress_potentially_thr(int    thr,
                                                                                      InT<T> second_value) noexcept;

    /**
     * @brief Core compression sweep: drain the candidate list of one (thread,
     *        second_value), RELEASE the buffer of every candidate found constant,
     *        and clear the list.
     * @tparam FillFunc Fills the two candidate buckets (type-1/type-2).
     * @tparam ClearFunc Clears one candidate chunk's constant slot.
     * @param thr_rank Thread whose candidate list to drain.
     * @param second_value The constant value candidates are being compressed to.
     * @param proxy2 Owning proxy (null for the thread-only sweep); pins the active chunk
     *               size and runs the compression-manager start/finish protocol.
     * @param fill_func Fills the candidate buckets (bool vs non-bool).
     * @param clear_func Clears one candidate's slot (bool vs non-bool).
     */
    template <typename FillFunc, typename ClearFunc>
    static void make_block_compression_impl(int         thr_rank,
                                            T           second_value,
                                            Proxy<T>*   proxy2,
                                            FillFunc&&  fill_func,
                                            ClearFunc&& clear_func);

    /*!
     * \brief Per-thread half of compress_chunk_to_compress(): drain only
     *        thread `thr_rank`'s candidate queue for the given second value.
     * \param thr_rank Thread whose candidate queue to drain.
     * \param inf_array (second_value, ...) identifying the candidate list.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_in_par(int thr_rank, const InfArrayT& inf_array, bool apply);

    /*!
     * \brief Bool specialization of compress_chunk_to_compress_in_par().
     * \param thr_rank Thread whose candidate queue to drain.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_bool_in_par(int thr_rank, bool apply);

    /**
     * \brief Shared body of the two compress_chunk_to_compress_in_par entry
     *        points: they differ only in the map key and the per-chunk compress.
     *
     * The candidate is compressed BEFORE its "already queued" flag is cleared;
     * see the definition for why that order is the safe one.
     *
     * \tparam CompressFn callable (ChunkType*, ChunkElems count).
     */
    template <typename CompressFn>
    static void compress_chunk_to_compress_in_par_impl(int thr_rank, InT<T> key, CompressFn&& compress_one);

    /*!
     * \brief Bool specialization of compress().
     * \param thr_rank Calling thread's rank (for clear_int_st_bool()).
     * \param apply The scan always runs; compression only when true.
     * \param n_elems Element count of this chunk (the owning array's geometry).
     */
    void compress_bool(this Derived& self, int thr_rank, bool apply, unsigned int n_elems);

    /**
     * \brief Shared body of compress() and compress_bool(): they differ only in
     *        how the second value is obtained and which clear_int_st overload
     *        applies it.
     * \tparam SecondValueFunc Nullary callable returning the second value.
     * \tparam ClearFunc callable (int num_compress_value).
     */
    template <typename SecondValueFunc, typename ClearFunc>
    void compress_impl(this Derived&   self,
                       bool            apply,
                       unsigned int    n_elems,
                       SecondValueFunc get_second_value,
                       ClearFunc&&     clear_one);

    /**
     * \brief Shared body of the two fill_list_chunk_to_compress_in_par entry
     *        points: they differ only in the map key and the per-chunk scan.
     * \tparam ScanFn callable (ChunkType*, num_chunk, num_tab, ChunkElems count).
     */
    template <typename ScanFn>
    static void fill_list_chunk_to_compress_in_par_impl(int thr_rank, InT<T> key, ScanFn&& scan_one);

    /*!
     * \brief Sort thread `thr_rank`'s queued candidates into the two
     *        make_block_compression_impl() buckets, each chunk re-assessing itself
     *        (fill_list_to_compress()).
     * \param thr_rank Thread whose candidate list to distribute.
     * \param ll_compt1 Output: candidates confirmed uniform on the first constant.
     * \param ll_compt2 Output: candidates confirmed uniform on the second constant.
     * \param inf_array (second_value, ...) identifying the candidate list.
     */
    static void fill_list_chunk_to_compress_in_par(int                  thr_rank,
                                                   VecTupleChunk2pInt2& ll_compt1,
                                                   VecTupleChunk2pInt2& ll_compt2,
                                                   const InfArrayT&     inf_array);

    /*!
     * \brief Bool specialization of fill_list_chunk_to_compress_in_par().
     * \param thr_rank Thread whose candidate list to distribute.
     * \param ll_compt1 Output: candidates confirmed uniform on the first constant.
     * \param ll_compt2 Output: candidates confirmed uniform on the second constant.
     */
    static void fill_list_chunk_to_compress_bool_in_par(int                  thr_rank,
                                                        VecTupleChunk2pInt2& ll_compt1,
                                                        VecTupleChunk2pInt2& ll_compt2);

    /*!
     * \brief Re-scan THIS chunk's live content and, if now uniform on the first and/or
     *        second constant, append it to the matching bucket. Called once per
     *        candidate by fill_list_chunk_to_compress_in_par().
     * \tparam SecondValueFunc Callable returning the second constant.
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     * \param get_second_value Supplies the second constant to compare against.
     * \param check_cst_val2 Whether to test the second constant (skipped when the
     *                        chunk is already known to hold it).
     * \param n_elems Element count of this chunk (the owning array's geometry).
     */
    template <typename SecondValueFunc>
    void fill_list_to_compress_impl(this Derived&        self,
                                    VecTupleChunk2pInt2& ll_compt1,
                                    VecTupleChunk2pInt2& ll_compt2,
                                    int                  num_chunk,
                                    int                  num_tab,
                                    SecondValueFunc&&    get_second_value,
                                    bool                 check_cst_val2,
                                    unsigned int         n_elems);

    /*!
     * \brief Non-bool wrapper around fill_list_to_compress_impl() that
     *        supplies the second constant from `inf_array`.
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param inf_array (second_value, ...) tuple supplying the second constant.
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     * \param n_elems Element count of this chunk (the owning array's geometry).
     */
    void fill_list_to_compress(this Derived&        self,
                               VecTupleChunk2pInt2& ll_compt1,
                               VecTupleChunk2pInt2& ll_compt2,
                               const InfArrayT&     inf_array,
                               int                  num_chunk,
                               int                  num_tab,
                               unsigned int         n_elems);

    /*!
     * \brief Bool wrapper around fill_list_to_compress_impl() (the second
     *        constant is always `true` for bool).
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     * \param n_elems Element count of this chunk (the owning array's geometry).
     */
    void fill_list_to_compress_bool(this Derived&        self,
                                    VecTupleChunk2pInt2& ll_compt1,
                                    VecTupleChunk2pInt2& ll_compt2,
                                    int                  num_chunk,
                                    int                  num_tab,
                                    unsigned int         n_elems)
    {
        self.fill_list_to_compress_impl(
            ll_compt1, ll_compt2, num_chunk, num_tab, [] { return true; }, self.is_not_cst_val2(), n_elems);
    }
};

/// @brief zero_one_value::ChunkFastCompPolicy with fast block compression compiled out:
///        no candidate list, so every query answers "empty" and every pass
///        is a no-op.
template <typename T, typename Derived>
class ChunkFastCompPolicy<T, Derived, false> : public ChunkThreadPolicy<T, Derived>, public ChunkSpec<T>
{
  public:
    /// (chunk, num_chunk, num_tab, n_elems); see the true specialisation.
    using VecTupleChunk2pInt2 = std::vector<std::tuple<Chunk<T>*, int, int, unsigned int>>;
    /// (second value, its index in the shared value table, num_tab).
    using InfArrayT = SecondValueInfo<T>;

    /// @name No-op counterparts of the true specialization's queries and passes
    ///       (same pattern as ChunkFastCompPolicy<T, Derived, false>)
    /// @{
    static bool have_chunk_to_compress(T /*second_value*/, int /*max_thr*/)
    {
        return false;
    }
    static bool have_chunk_to_compress_bool(int /*max_thr*/)
    {
        return false;
    }

    static void make_block_compression(int /*thr_rank*/, const InfArrayT& /*inf_array*/, Proxy<T>* /*proxy2*/) {}

    static void make_block_compression_bool(int /*thr_rank*/, Proxy<T>* /*proxy2*/) {}

    static void make_block_compression(int /*thr_rank*/) {}
    /// @}

  private:
    /// @name Empty candidate lists
    /// @{
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        return true;
    }
    /// @}

    /// @name No-op compression passes
    /// @{
    template <typename FillFunc, typename ClearFunc>
    static void make_block_compression_impl(int /*thr_rank*/,
                                            T /*second_value*/,
                                            Proxy<T>* /*proxy2*/,
                                            FillFunc&& /*fill_func*/,
                                            ClearFunc&& /*clear_func*/)
    {
    }

    /// @}
};

/*!
 * \brief Block compression class (power of 2) 2 values automatic detection
 */
template <typename T>
// no move operations on purpose -- a move falls back to
// the copy, which carries the per-chunk state the way the copy constructor documents.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Chunk : public ChunkCounterPolicy<chunk_config::use_compression_with_counter>,
              public ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>
{
  public:
    friend class Proxy<T>;
    friend class SecondValueChunkBase<T, Chunk>;

    using Base     = SecondValueChunkBase<T, Chunk>;
    using BaseSpec = ChunkSpec<T>;

    using ChunkType              = Chunk;
    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using Base::allocator_data;
    using Base::first_value;
    using Base::values_;

    using CounterPolicy =
        ChunkCounterPolicy; ///< The counter base, zero_one_value::ChunkCounterPolicy<use_compression_with_counter>.

    using FastCompPolicy = ChunkFastCompPolicy<T, Chunk, chunk_config::use_fast_block_compression>;
    using ThreadPolicy   = ChunkThreadPolicy<T, Chunk, build_config::threaded>;

    // Both ThreadPolicy specialisations declare these (with mode-specific
    // signatures); the in-class bodies below call them unqualified.
    using ThreadPolicy::block_clear;
    using ThreadPolicy::clear_int_st;
    using ThreadPolicy::clear_st;
    using ThreadPolicy::create_value;
    using ThreadPolicy::create_value_bool;

    using VecTupleChunk2pInt2 = std::vector<std::tuple<Chunk*, int, int>>;

    using ManagerChunk2 = ManagerChunk<Chunk, T>;
    // (second value, index, num_tab). Redeclared here on purpose: the policy
    // bases declare the same name, and this one hides theirs so zero_one_value's own
    // members see one type. It must stay THEIR type -- checked below.
    using InfArrayT = SecondValueInfo<T>;

    // Statistics variant tag (CHUNK_STATS): 1 = zero_one_value.
    static constexpr int stat_variant = 1;

    /*!
     * \brief Default constructor.
     */
    Chunk() = default;
    /*!
     * \brief Default destructor.
     */
    ~Chunk() = default;

    /// @brief Get reference to the chunk manager (static access)
    /// @return Reference to the ManagerChunk2 static instance
    /// @details On the class, not on the threaded policy: ChunkBase::release_to_owner()
    ///          names self.get_manager() in every build (unreachable serially but
    ///          must compile).
    ///
    ///          An `inline static` member, NOT a function-local static: the lazy guard's
    ///          first init can happen from compress() inside an OpenMP region and
    ///          deadlock threads on __cxa_guard_acquire. An inline static is built during
    ///          static init, before any thread. The manager ctor is trivial.
    static auto& get_manager() noexcept
    {
        return *s_manager_;
    }

    /**
     * @brief Copy constructor: copies the values pointer (shallow) AND the
     *        zero_one_value::ChunkCounterPolicy counters.
     * @param other Chunk to copy from.
     * @warning The counter base MUST be copied: default-constructed, the counters restart
     *          at 0 and can no longer reach `size`, so the chunk's compression is missed
     *          for good. This ctor also suppresses the implicit move, so every
     *          std::vector<zero_one_value::Chunk> reallocation goes through it. Same rule as zero
     *          (zero_chunk.h). `values` is atomic and is copied by hand.
     */
    Chunk(const Chunk& other);

    /**
     * @brief Copy assignment: copies the values pointer (shallow) AND the counters.
     * @param other Chunk to copy from.
     * @return *this.
     * @see Chunk(const Chunk&) for why the counters must follow `values`.
     */
    Chunk<T>& operator=(const Chunk<T>& other);

    /*!
     * \brief Point this chunk's values at the shared first-constant buffer
     *        (compress to cst_val1).
     */
    void init_values_cst_val1() noexcept
    {
        values_ = Base::get_cst_val1();
    }

    /*!
     * \brief Initializes internal pointers after construction.
     */
    void init_after_constructor();

    /*!
     * \brief Non-bool: point this chunk's values at the requested constant.
     * \param num_compress_value 0 for the first constant, non-zero for the second at `ind`.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void init_values_cst_val(int num_compress_value, int ind);

    /*!
     * \brief Bool-only: point this chunk's values at the requested constant.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            (single, shared) second constant.
     */
    void init_values_cst_val_bool(int num_compress_value);

    /*!
     * \brief Bool-only: whether this chunk is currently materialised (holds
     *        real data, neither constant).
     * \return True if values differs from both the first and second constant.
     */
    [[nodiscard]] bool written() const;

    /*!
     * \brief Whether this chunk owns a real heap buffer that must be
     *        parked/freed at teardown, as opposed to pointing at cst_val1 or
     *        ANY cst_val2 marker.
     * \note clear_core must key on THIS, not written(ind): a single index misclassifies
     *       chunks compressed at other indices, and their static markers would be
     *       parked/freed as if allocated.
     * \return True if the chunk's values pointer is a live, owned buffer.
     */
    [[nodiscard]] bool owns_live_buffer() const noexcept;

    /*!
     * \brief Non-bool: whether this chunk is currently materialised (holds
     *        real data, neither constant) at second-value index `ind`.
     * \param ind Index of the second constant value.
     * \return True if values differs from both constants.
     */
    [[nodiscard]] bool written(int ind) const;

    /*!
     * \brief Checks whether the chunk IS currently compressed to the second
     *        constant at index `ind`.
     * \param ind Index of the second value.
     * \return True if values equals the second constant at ind.
     */
    [[nodiscard]] bool is_cst_val2(int ind) const noexcept;

    /*!
     * \brief Bool-only: checks whether the chunk IS currently compressed to
     *        the (single, shared) second constant.
     * \return True if values equals the second constant.
     */
    [[nodiscard]] bool is_cst_val2() const noexcept;

    /*!
     * \brief Checks that the chunk does NOT use the second constant at
     *        index `ind`.
     * \param ind Index of the second value.
     * \return True if values differs from the second constant at ind.
     */
    [[nodiscard]] bool is_not_cst_val2(int ind) const noexcept;

    /*!
     * \brief Bool-only: negation of is_cst_val2().
     * \return True if values does NOT equal the (single, shared) second constant.
     */
    [[nodiscard]] bool is_not_cst_val2() const noexcept;

    /*!
     * \brief Materialise this chunk if it is currently compressed to either
     *        constant; a no-op otherwise (already materialised).
     * \param inf_array (second_value, ...) tuple describing the target state.
     */
    void alloc_chunk(const InfArrayT& inf_array);

    /*!
     * \brief Per-thread variant of alloc_chunk().
     * \param inf_array (second_value, ...) tuple describing the target state.
     * \param thr_rank Calling thread's rank (ignored in serial builds).
     */
    void alloc_chunk(const InfArrayT& inf_array, [[maybe_unused]] int thr_rank);

    static_assert(
        std::is_same_v<InfArrayT,
                       typename ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>::InfArrayT>,
        "zero_one_value::Chunk::inf_array_t must be the policies' inf_array_t");

    // --- where the rest of zero_one_value lives ----------------------------------
    // Materialisation and clearing (create_value*, clear*, block_clear) are on
    // zero_one_value::ChunkThreadPolicy, one specialisation per mode; the queue-driven compression
    // (compress*, fill_list*, compress_chunk_to_compress*) is on
    // zero_one_value::ChunkFastCompPolicy<T, zero_one_value::Chunk, true>, next to make_block_compression.

  private:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /*!
     * \brief Stores a pointer to the chunk's values.
     * \param ptr Pointer to store.
     */
    void values_store(T* ptr) noexcept;

    /*!
     * \brief Attempt to atomically compress the chunk to either cst_val1 or
     *        cst_val2(ind).
     * \param num_compress_value 0 to target cst_val1, non-zero for the second at `ind`.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     * \return nullptr if the chunk was already compressed or another thread
     *         won the race (caller must not free anything); otherwise the
     *         old pointer, which the caller now owns and must free/recycle.
     */
    [[nodiscard]] T* try_mark_cst_and_get_old(this auto& self, int num_compress_value, int ind);

    /*!
     * \brief Bool-only: whether the chunk is currently compressed to EITHER
     *        constant (cst1 or cst2).
     * \return true if the chunk is compressed to either constant.
     */
    [[nodiscard]] bool is_cst_val() const noexcept;

    /*!
     * \brief Whether the chunk is currently compressed to EITHER constant
     *        (cst1 or cst2). Valid for bool too: `ind` is then ignored.
     * \param ind Index of the second constant (used only for non-bool chunks).
     * \return true if the chunk is compressed to either constant.
     */
    [[nodiscard]] bool is_cst_val(int ind) const noexcept;

    static inline constinit internal::LazyImmortal<ManagerChunk2> s_manager_;
};

} // namespace zero_one_value

} // end namespace shrink_array

#include "chunk/internal/zero_one_value_chunk_inline.h" // the template bodies
