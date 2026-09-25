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

#include <vector>
#include <array>

#include "chunk/internal/chunk_size_config.h"
#include "chunk/internal/chunk_size_policy.h"

#include "chunk/internal/in_t.h"
#include "chunk/config.h"
#include "chunk/chunk_define.h"
//#include "chunk/chunk_def.h"
//#include "chunk/chunk2_def.h"
#include "chunk/internal/invariant_check.h"
#include "chunk/internal/bad_ptr_check.h" // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/PMP_CHUNK.h"
#include "chunk/internal/chunk2_candidate_list_check.h" // chunk2_missing_candidate_list
#include "chunk/internal/proxy2.h"
#include "chunk/internal/chunk2_base.h"
#include "chunk/internal/conditional_counter.h"

namespace shrink_array
{

template <typename T>
class Proxy2;

template <typename T>
struct ChunkSpec2
{
    using vec_array_chunk_size_max =
        std::conditional_t<build_config::use_mm_protect, std::vector<T*>, std::vector<std::array<T, cst_buffer_size>>>;

    static constexpr bool use_mm_protect = build_config::use_mm_protect;

    static inline vec_array_chunk_size_max tab_m_ptr_cst_val2;

    // mm_protect mode only: element count currently allocated at
    // tab_m_ptr_cst_val2[ind] (a raw mmap'd T*, unlike the plain-array mode
    // where each slot is a fixed cst_buffer_size std::array and always big
    // enough). Lets set_second_value's reuse path (a later array registering
    // an already-known second_value at a LARGER geometry) detect an
    // undersized allocation and grow it instead of overrunning it. Empty /
    // short in the non-mm_protect build (still declared so it links, but
    // logically unused).
    static inline std::vector<std::size_t> tab_m_ptr_cst_val2_mm_elems;

    /*!
     * \brief Gets the pointer to the second constant value.
     * \param ind Index of the second value.
     * \return Pointer to the constant value array.
     */

    [[nodiscard]] static T* get_cst_val2(int ind) noexcept
    {
        // An array with no second value (chunk_config::no_second_value_index)
        // has no block: nullptr, which no chunk's `values` ever equals.
        if (ind < 0)
            return nullptr;
        if constexpr (build_config::any_check)
        {
            if (static_cast<size_t>(ind) >= tab_m_ptr_cst_val2.size()) [[unlikely]]
            {
                INVARIANT_FAILED("[chunk] suspect get_cst_val2<{}>: ind={} size={}", typeid(T).name(), ind,
                                 tab_m_ptr_cst_val2.size());
                return nullptr;
            }
        }

        T* ret;
        if constexpr (use_mm_protect)
        {
            ret = tab_m_ptr_cst_val2[ind];
        }
        else
        {
            ret = tab_m_ptr_cst_val2[ind].data();
        }
        CHUNK_CHECK_VALUES_PTR(ret, "get_cst_val2");
        return ret;
    }
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

    // ANY-index marker test: is `v` one of the cst_val2 constant pointers,
    // whichever index? Needed at teardown where the chunk's own index is not to
    // hand: is_cst_val2(v, ind) with a single/wrong index misclassifies a chunk
    // compressed at ANOTHER index (its marker then reads as a live buffer).
    /*!
     * \brief Whether `v` is ANY of the registered second-constant pointers,
     *        regardless of index.
     * \param v Pointer to test.
     * \return True if v matches the second constant at some index.
     */
    [[nodiscard]] static bool is_cst_val2_any(const T* v) noexcept
    {
        const int nb_cst_val2 = static_cast<int>(tab_m_ptr_cst_val2.size());
        for (int i = 0; i < nb_cst_val2; ++i)
            if (v == get_cst_val2(i))
                return true;
        return false;
    }

    // is_cst_val2_async / is_not_cst_val2_async removed: dead code, zero
    // callers anywhere in the tree (the bool one had a latent bug -- it called
    // is_cst_val2(ctx) instead of is_cst_val2_async(ctx), a type mismatch never
    // caught because the template body was never instantiated).

    /*!
     * \brief Fill a freshly materialised buffer with the second value
     *        carried in `inf_array`.
     * \param values_loc Buffer to fill.
     * \param inf_array (second_value, ...) tuple; only the value is used here.
     */
    static void init_block_2nd_value(T* values_loc, const std::tuple<T, int, int>& inf_array)
    {
        // in_t: fill_n takes a const T& anyway; no copy of a struct on the way.
        const in_t<T> value = std::get<0>(inf_array);
        std::fill_n(values_loc, alloc_chunk_count(), value);
    }
};

template <>
struct ChunkSpec2<bool>
{
    static inline std::array<bool, cst_buffer_size> m_ptr_cst_val2 = []()
    {
        std::array<bool, cst_buffer_size> arr{};
        arr.fill(true); // Initialize everything to 'true'
        return arr;
    }();

    /*!
     * \brief Gets the pointer to the (single, shared) second constant value.
     * \return Pointer to the constant value array.
     */
    [[nodiscard]] static bool* get_cst_val2() noexcept
    {
        return m_ptr_cst_val2.data();
    }

    /*!
     * \brief Checks whether `values` IS the second constant's pointer (i.e.
     *        the chunk is currently compressed to it).
     * \param values Pointer to compare.
     * \return True if values equals the second constant.
     */
    [[nodiscard]] static bool is_cst_val2(const bool* values) noexcept
    {
        return values == get_cst_val2();
    }

    /*!
     * \brief Negation of is_cst_val2().
     * \param values Pointer to compare.
     * \return True if values does NOT equal the second constant.
     */
    [[nodiscard]] static bool is_not_cst_val2(const bool* values) noexcept
    {
        return !is_cst_val2(values);
    }

    /*!
     * \brief Fill a freshly materialised buffer with `true` (bool's only
     *        possible second value).
     * \param values_loc Buffer to fill.
     */
    static void init_block_2nd_value(bool* values_loc)
    {
        std::fill_n(values_loc, alloc_chunk_count(), true);
    }
};

// Base policy for counters
/*!
 * \brief Per-chunk element counters (how many elements currently equal the
 *        first/second constant), zero-cost when EnableCounters is false
 *        (ConditionalCounter collapses to an empty type).
 * \tparam EnableCounters Whether the counters actually exist and count.
 */
template <bool EnableCounters>
struct Chunk2CounterPolicy
{
    // [[no_unique_address]] and two distinct tags: disabled, both counters
    // are empty types that may share offset 0, so the policy is empty and
    // the empty-base optimisation keeps `values` at offset 0 -- see
    // ChunkCounterPolicy for the 8 bytes of padding this saves per chunk.
    [[no_unique_address]] ConditionalCounter<EnableCounters, 0> nb_val_eq_first_value;
    [[no_unique_address]] ConditionalCounter<EnableCounters, 1> nb_val_eq_second_value;

    /// @brief Reset the second-value counter to 0.
    void reset_nb_val_eq_second_value() noexcept
    {
        nb_val_eq_second_value = 0;
    }

    /// @brief Set the second-value counter to an explicit value.
    /// @param v New counter value.
    void set_nb_val_eq_second_value(counter_value_t v) noexcept
    {
        nb_val_eq_second_value = v;
    }

    // size defaults to the compile-time geometry so existing callers keep
    // working; callers that know the runtime geometry (the array/proxy, via
    // size_policy_) pass it explicitly. Foundation for DynamicChunkSize.
    /// @brief Whether every element in the chunk currently equals the second
    ///        constant (the chunk is a candidate for type-2 compression).
    /// @param size Chunk geometry to compare against; defaults to the
    ///             compile-time size (pass the runtime size under
    ///             DynamicChunkSize).
    /// @return true if nb_val_eq_second_value == size.
    [[nodiscard]] bool is_full_second_value(unsigned int size = __chunk_size) const noexcept
    {
        return nb_val_eq_second_value == size;
    }

    /// @brief Increment the second-value counter (one more element now
    ///        equals the second constant).
    void inc_nb_val_eq_second_value() noexcept
    {
        ++nb_val_eq_second_value;
    }

    /// @brief Decrement the second-value counter (one fewer element now
    ///        equals the second constant).
    void dec_nb_val_eq_second_value() noexcept
    {
        --nb_val_eq_second_value;
    }

    /// @brief Increment the first-value counter (one more element now
    ///        equals the first constant).
    void inc_nb_val_eq_first_value() noexcept
    {
        ++nb_val_eq_first_value;
    }

    /// @brief Decrement the first-value counter (one fewer element now
    ///        equals the first constant).
    void dec_nb_val_eq_first_value() noexcept
    {
        --nb_val_eq_first_value;
    }

    /// @brief Reset the first-value counter to 0.
    void reset_nb_val_eq_first_value() noexcept
    {
        nb_val_eq_first_value = 0;
    }

    /// @brief Set the first-value counter to an explicit value.
    /// @param v New counter value.
    void set_nb_val_eq_first_value(counter_value_t v) noexcept
    {
        nb_val_eq_first_value = v;
    }

    /// @brief Reset both counters to 0.
    void reset_nb_val_eq_value() noexcept
    {
        reset_nb_val_eq_first_value();
        reset_nb_val_eq_second_value();
    }

    /// @brief Whether every element in the chunk currently equals the first
    ///        constant (the chunk is a candidate for type-1 compression).
    /// @param size Chunk geometry to compare against; defaults to the
    ///             compile-time size (pass the runtime size under
    ///             DynamicChunkSize).
    /// @return true if nb_val_eq_first_value == size.
    [[nodiscard]] bool is_full_first_value(unsigned int size = __chunk_size) const noexcept
    {
        return nb_val_eq_first_value == size;
    }

    /// @brief Current first-value counter.
    /// @return Number of elements currently equal to the first constant.
    [[nodiscard]] unsigned char get_nb_first_value() const noexcept
    {
        return nb_val_eq_first_value;
    }

    /// @brief Current second-value counter.
    /// @return Number of elements currently equal to the second constant.
    [[nodiscard]] unsigned char get_nb_second_value() const noexcept
    {
        return nb_val_eq_second_value;
    }
};
/**
 * @brief Chunk2's OpenMP-only surface: per-thread materialisation and clearing.
 *
 * @details A serial build has no thread ranks, no CAS window and no per-thread
 *          free-lists, and it maintains per-chunk element counters
 *          (use_compression_with_counter) that the parallel path does not. The
 *          two modes therefore materialise and clear a chunk differently, and
 *          each specialisation below declares exactly the verbs its mode has --
 *          the technique of ChunkOmpPolicy for Chunk v1.
 *
 *          Sits between Chunk2FastCompPolicy and Chunk2Base in the inheritance
 *          chain, so Chunk2 inherits both policies through one path.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk2<T>)
 * @tparam Enabled build_config::use_openmp
 */
template <typename T, typename Derived, bool Enabled = build_config::use_openmp>
struct Chunk2OmpPolicy : public Chunk2Base<T, Derived>
{
    using Base           = Chunk2Base<T, Derived>;
    using BaseSpec       = ChunkSpec2<T>;
    using manager_chunk2 = ManagerChunk<Chunk2, T>;
    using inf_array_t    = std::tuple<T, int, int>;

    using Base::derived;
    using Base::first_value;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new Chunk2<T>[n]` runs the DEFAULT CONSTRUCTORS after this, so
    ///          that initialisation is immediately overwritten; the operator
    ///          pays off for a placement-new array, where the caller does not
    ///          construct. Nothing in the library array-news a chunk --
    ///          ArrayChunk* hold a std::vector<chunk_type>, initialised by
    ///          their own loops.
    void* operator new[](std::size_t sz);
    /// @brief Matching array-delete.
    void operator delete[](void* ptr);
    /// @brief Point @p n freshly allocated chunks at the first constant.
    /// @details Parallel past 500 chunks, and only when called from OUTSIDE a
    ///          parallel region (inside one it stays on the calling thread: an
    ///          allocation is made by one thread, and an orphaned `omp for`
    ///          would hang waiting for the others). Shared by the three
    ///          variants -- chunk_array_new::init_chunks.
    static void init_chunks_cst_val(Chunk2<T>* v, std::size_t count) noexcept;

    /**
     * \brief Shared body of create_value() and create_value_bool().
     * \tparam IsBoolCas true to publish with compare_and_swap_bool, false with
     *         the indexed compare_and_swap -- selected here, NOT passed as a
     *         lambda; see the definition for why that is load-bearing.
     * \tparam InitFn callable (T*) pre-filling the allocated block.
     */
    template <bool IsBoolCas, typename InitFn>
    void create_value_impl(int thr_rank, InitFn&& init_func, int ind, bool need_set_second_value) noexcept;

    /*!
     * \brief Materialise this chunk's buffer (allocate + init), for a
     *        specific thread. Non-bool only.
     * \param need_set_second_value Whether to also register/set the
     *                               chunk's second-value slot.
     * \param inf_array (second_value, ...) tuple describing the target state.
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void create_value(bool need_set_second_value, const inf_array_t& inf_array, int thr_rank) noexcept;

    /*!
     * \brief Bool specialization of create_value().
     * \param need_set_second_value Whether to also register the second-value slot.
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void create_value_bool(bool need_set_second_value, int thr_rank) noexcept;

    /*!
     * \brief Convenience overload of create_value() that resolves the
     *        calling thread's rank automatically (get_thr_rank_fast()).
     *        Non-bool only; aborts in debug builds otherwise.
     * \param need_set_second_value Whether to also register the second-value slot.
     * \param inf_array (second_value, ...) tuple describing the target state.
     */
    void create_value(bool need_set_second_value, const inf_array_t& inf_array) noexcept
    {
        if constexpr (is_not_bool)
        {
            create_value(need_set_second_value, inf_array, get_thr_rank_fast());
        }
        else if constexpr (build_config::is_debug)
        {
            std::cerr << "Chunk2_std::create_value() called" << '\n';
            std::abort();
        }
    }

    /*!
     * \brief Convenience overload of create_value_bool() that resolves the
     *        calling thread's rank automatically (get_thr_rank_fast()).
     *        Bool-only; aborts in debug builds otherwise.
     * \param need_set_second_value Whether to also register the second-value slot.
     */
    void create_value_bool(bool need_set_second_value) noexcept
    {
        if constexpr (is_bool)
        {
            create_value_bool(need_set_second_value, get_thr_rank_fast());
        }
        else if constexpr (build_config::is_debug)
        {
            std::cerr << "Chunk2_std::create_value_bool() called" << '\n';
            std::abort();
        }
    }

    /*!
     * \brief Compress this chunk to its second constant at index `ind`
     *        (OpenMP/deferred-queue algorithm). Non-bool.
     * \param ind Index of the second constant to compress to.
     * \param thr_rank Calling thread's rank.
     */
    void clear(int ind, int thr_rank);

    /*!
     * \brief Shared "actually install the constant and release the old
     *        buffer" step used by every compression path.
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void block_clear(int num_compress_value, int ind, int thr_rank);

    /*!
     * \brief Compress this chunk to a constant, for a specific thread.
     *        Non-bool. Called by the make_block_compression_impl()
     *        clear_func lambda (see Chunk2FastCompPolicy).
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            second constant at index `ind`.
     * \param ind Index of the second constant (unused when
     *            num_compress_value == 0).
     * \param thr_rank Calling thread's rank (selects the allocator pool).
     */
    void clear_int_st(int num_compress_value, int ind, int thr_rank);

    /*!
     * \brief Bool specialization of the 3-arg clear_int_st(), for a
     *        specific thread.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            (single, shared) second constant.
     * \param thr_rank Calling thread's rank.
     */
    void clear_int_st_bool(int num_compress_value, int thr_rank);

    /*!
     * \brief Compress this chunk to a constant (no thread dimension): the
     *        calling thread's rank is supplied here.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            second constant at index `ind`.
     * \param ind Index of the second constant (unused when
     *            num_compress_value == 0).
     */
    void clear_st(int num_compress_value, int ind)
    {
        clear_int_st(num_compress_value, ind, CHUNK_PMP_THREAD_RANK);
    }
};

/**
 * @brief Serial specialisation: no rank, no CAS, and the element counters to reset.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk2<T>)
 */
template <typename T, typename Derived>
struct Chunk2OmpPolicy<T, Derived, false> : public Chunk2Base<T, Derived>
{
    using Base        = Chunk2Base<T, Derived>;
    using BaseSpec    = ChunkSpec2<T>;
    using inf_array_t = std::tuple<T, int, int>;

    using Base::derived;
    using Base::first_value;
    using Base::is_bool;
    using Base::is_not_bool;
    using Base::values;

    /*!
     * \brief Materialise this chunk's buffer (allocate + init). Serial
     *        builds have a single rank, so no thr_rank parameter. Non-bool
     *        only.
     * \param need_set_second_value Whether to also register/set the
     *                               chunk's second-value slot.
     * \param inf_array (second_value, ...) tuple describing the target state.
     */
    void create_value(bool need_set_second_value, const inf_array_t& inf_array) noexcept;

    /*!
     * \brief Bool specialization of create_value() (serial builds).
     * \param need_set_second_value Whether to also register the second-value slot.
     */
    void create_value_bool(bool need_set_second_value) noexcept;

    /**
     * \brief Shared body of the two serial create_value entry points: they
     *        differ only in how the second-value fill is spelled.
     * \tparam Init2ndFn callable (T*) filling the block with the second value.
     */
    template <typename Init2ndFn>
    void create_value_serial_impl(bool need_set_second_value, Init2ndFn&& init_2nd) noexcept;

    /*!
     * \brief Compress this chunk to the second constant at index `ind`
     *        (serial/counter algorithm: eager, no queue). Non-bool.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            second constant at index `ind`.
     * \param ind Index of the second constant (unused when
     *            num_compress_value == 0).
     */
    void clear_int_st(int num_compress_value, int ind);

    /*!
     * \brief Compress this chunk to its first constant (serial/counter
     *        algorithm).
     * \param ind Passed through to debug_check_written(ind) as a sanity
     *            check; the compression itself always targets the first
     *            constant regardless of ind.
     */
    void clear_st(int ind);

    /*!
     * \brief Compress this chunk to a constant (serial twin of the OpenMP
     *        clear_st(int, int): no rank to supply).
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void clear_st(int num_compress_value, int ind)
    {
        clear_int_st(num_compress_value, ind);
    }

    /*!
     * \brief Shared "install the constant and release the old buffer" step.
     *        Frees the buffer outright (no rank pool) and resets the
     *        per-chunk element counters.
     * \param num_compress_value 0 for the first constant, non-zero for the second.
     * \param ind Index of the second constant (unused when num_compress_value == 0).
     */
    void block_clear(int num_compress_value, int ind, int);
};

// --- In chunk2_std.h ---

template <typename T, typename Derived, bool Enabled>
struct Chunk2FastCompPolicy : public ChunkSpec2<T>, public Chunk2OmpPolicy<T, Derived>
{
    using manager_chunk2 = ManagerChunk<Chunk2, T>;
    using Base           = Chunk2Base<T, Derived>;
    using BaseSpec       = ChunkSpec2<T>;

    using Base::derived;
    using Base::first_value;
    using Base::values;

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;
    /**
     * @brief Checks if the per-thread compression candidate map is empty.
     */
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        return manager_chunk2::map_chunk2_to_compress_potentially_thr.empty() ||
               manager_chunk2::map_chunk2_to_compress_potentially_thr[0].empty();
    }

    /**
     * @brief Scans all threads to see if any chunk is marked for compression with a specific value.
     */
    static bool have_chunk_to_compress(T second_value, int max_thr)
    {
        if (is_vec_tuple_to_compress_potentially_empty_thr())
            return false;

        for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
        {
            auto& map_thr = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank];
            auto  it      = map_thr.find(second_value);
            if (it != map_thr.end() && !it->second->empty())
                return true;
        }
        return false;
    }

    /**
     * @brief Boolean specific version of the compression check.
     */
    static bool have_chunk_to_compress_bool(int max_thr)
    {
        // bool's second value is always `true`, so this IS the general lookup at
        // that key -- the body used to be duplicated verbatim with `true`
        // hard-coded in place of the parameter.
        return have_chunk_to_compress(static_cast<T>(true), max_thr);
    }

    /**
     * @brief Per-(thread, second_value) list of chunks marked as compression
     *        candidates.
     * @param thr Thread rank whose candidate list to fetch.
     * @param second_value The type-2 constant these chunks are candidates for.
     * @return Reference to the (thr, second_value) candidate list.
     */
    static VectorTuple4ChunkInt2<Chunk2, T>& get_vec_tuple_to_compress_potentially_thr(int     thr,
                                                                                       in_t<T> second_value) noexcept
    {
        // find(), not operator[]: see Proxy2::get_vec_tuple_to_compress_potentially_thr.
        auto&      lists_by_value = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr];
        const auto it             = lists_by_value.find(second_value);
        if (it == lists_by_value.end() || it->second == nullptr) [[unlikely]]
            chunk2_missing_candidate_list(thr);
        return *it->second;
    }

    /**
     * @brief Bool-specific version of get_vec_tuple_to_compress_potentially_thr()
     *        (bool has a single shared second_value, always true).
     * @param thr Thread rank whose candidate list to fetch.
     * @return Reference to thread thr's candidate list.
     */
    // Removed: get_vec_tuple_to_compress_potentially_bool_thr(thr). It was
    // get_vec_tuple_to_compress_potentially_thr(thr, true) written out again, and
    // it had no caller anywhere in the tree.


    /**
     * @brief CAS a chunk's second-value slot from its expected constant state
     *        to a newly materialised buffer.
     * @param values_loc New buffer to install on success.
     * @param ind Index used to resolve the expected second-value constant.
     * @param expected_is_cst_val2 Whether the expected current state is the
     *                             second constant (vs the first).
     * @return true on failure (caller owns values_loc), false on success.
     */
    [[nodiscard]] bool compare_and_swap(T* values_loc, int ind, bool expected_is_cst_val2) noexcept
    {
        if constexpr (is_not_bool)
        {
            // Single CAS against the precise expected pointer. See rationale in
            // Chunk2A::compare_and_swap — combining two CAS with && could silently
            // succeed on the wrong side after a cst_val1 <-> cst_val2 race,
            // installing a buffer whose init does not match the observed state.
            T* expected = expected_is_cst_val2 ? BaseSpec::get_cst_val2(ind) : Base::get_cst_val1();
            return Base::compare_and_swap_gen(values_loc, expected);
        }
        else
        {
            std::abort();
        }
    }

    // Bool path — same rationale, no ind parameter (bool has a single shared array).
    /**
     * @brief Bool path of compare_and_swap() -- no ind parameter (bool has a
     *        single shared array).
     * @param values_loc New buffer to install on success.
     * @param expected_is_cst_val2 Whether the expected current state is the
     *                             second constant (vs the first).
     * @return true on failure (caller owns values_loc), false on success.
     */
    [[nodiscard]] bool compare_and_swap_bool(bool* values_loc, bool expected_is_cst_val2) noexcept
    {
        if constexpr (is_bool)
        {
            bool* expected = expected_is_cst_val2 ? BaseSpec::get_cst_val2() : Base::get_cst_val1();
            return Base::compare_and_swap_gen(values_loc, expected);
        }
        else
        {
            std::abort();
        }
    }

    // --- Same axis as this policy's own Enabled parameter
    // (chunk_config::use_fast_block_compression), following the precedent of
    // ChunkFastCompPolicy::make_block_compression for plain Chunk<T>. Local
    // aliases mirrored here the same way ChunkFastCompPolicy carries its own
    // vec_tuple_chunkp_int2 rather than reaching into Derived for them.
    // (chunk, num_chunk, num_tab, n_elems) -- n_elems is the chunk's OWN
    // geometry, carried from the candidate queue so the clear loop can pin the
    // thread to it before freeing (same shape as vec_tuple_chunkA2p_int2).
    using vec_tuple_chunk2p_int2 = std::vector<std::tuple<Chunk2<T>*, int, int, unsigned int>>;
    using inf_array_t            = std::tuple<T, int, int>;

    /**
     * @brief Core compression sweep: drain the candidate list for one
     *        (thread, second_value) pair, RELEASE the heap buffer of every
     *        candidate that turns out to be constant, and clear the list.
     * @tparam FillFunc Callable that fills the two candidate-list buckets
     *                  (type-1/type-2) for this (thread, second_value) pair.
     * @tparam ClearFunc Callable that clears one candidate chunk's constant slot.
     * @param thr_rank Thread whose candidate list to drain.
     * @param second_value The constant value candidates are being compressed to.
     * @param proxy2 Owning proxy (may be null for the thread-only sweep
     *               overload below); used to pin the active chunk size and
     *               to run the compression-manager start/finish protocol.
     * @param fill_func Fills the candidate buckets (bool vs non-bool differ here).
     * @param clear_func Clears one candidate's slot (bool vs non-bool differ here).
     */
    template <typename FillFunc, typename ClearFunc>
    static void make_block_compression_impl(int         thr_rank,
                                            T           second_value,
                                            Proxy2<T>*  proxy2,
                                            FillFunc&&  fill_func,
                                            ClearFunc&& clear_func);

    /**
     * @brief Compress every non-bool candidate chunk for (thr_rank, the
     *        second value carried by inf_array) to its constant second-value
     *        slot.
     * @param thr_rank Thread whose candidate list to drain.
     * @param inf_array (second_value, ...) tuple identifying which candidate
     *                  list to drain.
     * @param proxy2 Owning proxy, or nullptr for the thread-only sweep.
     */
    static void make_block_compression(int thr_rank, const inf_array_t& inf_array, Proxy2<T>* proxy2);

    /**
     * @brief Bool specialization of make_block_compression().
     * @param thr_rank Thread whose candidate list to drain.
     * @param proxy2 Owning proxy, or nullptr for the thread-only sweep.
     */
    static void make_block_compression_bool(int thr_rank, Proxy2<T>* proxy2);

    /**
     * @brief Thread-only sweep: drain every second_value candidate list
     *        registered for thr_rank (dispatches to the per-second_value
     *        overload above once per distinct second_value).
     * @param thr_rank Thread whose candidate lists to drain.
     */
    static void make_block_compression(int thr_rank);

    // --- queue-driven compression, moved here from Chunk2<T> --------------
    // Same axis as this policy (chunk_config::use_fast_block_compression ==
    // the former HAVE_EMBEDDED_FAST_BLOCK_COMPRESSION fence); the false
    // specialisation below has no queue and so declares none of it.

    /*!
     * \brief Drain the compression-candidate queue for the given second
     *        value, across all threads.
     * \param inf_array (second_value, ...) tuple identifying which
     *                  candidate list to drain.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress(const inf_array_t& inf_array, bool apply);

    /*!
     * \brief Bool specialization of compress_chunk_to_compress().
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_bool(bool apply);

    /*!
     * \brief Per-thread half of compress_chunk_to_compress(): drain only
     *        thread `thr`'s candidate queue for the given second value.
     * \param thr Thread whose candidate queue to drain.
     * \param inf_array (second_value, ...) tuple identifying which
     *                  candidate list to drain.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_in_par(int thr, const inf_array_t& inf_array, bool apply);

    /*!
     * \brief Bool specialization of compress_chunk_to_compress_in_par().
     * \param thr Thread whose candidate queue to drain.
     * \param apply true: compress the constant candidates; false: walk only.
     */
    static void compress_chunk_to_compress_bool_in_par(int thr, bool apply);

    /**
     * \brief Shared body of the two compress_chunk_to_compress_in_par entry
     *        points: they differ only in the map key and the per-chunk compress.
     *
     * The candidate is compressed BEFORE its "already queued" flag is cleared;
     * see the definition for why that order is the safe one.
     *
     * \tparam CompressFn callable (chunk_type*, ChunkElems count).
     */
    template <typename CompressFn>
    static void compress_chunk_to_compress_in_par_impl(int thr_rank, in_t<T> key, CompressFn&& compress_one);

    /*!
     * \brief Per-candidate verify-and-apply step, called once per chunk by
     *        compress_chunk_to_compress_in_par() while draining the
     *        candidate queue -- NOT itself a queue-draining entry point.
     *        Independently re-scans THIS chunk's live content (does not
     *        trust the queue entry alone) and, when @p apply is true,
     *        compresses it via clear_int_st() if still uniform. A no-op
     *        (scan only, no compress) when @p apply is false.
     * \param thr_rank Calling thread's rank (used only for the
     *                 clear_int_st() call).
     * \param inf_array (second_value, ...) tuple identifying the second
     *                  constant and index to test against.
     * \param apply The scan always runs; the compression only happens when
     *              this is true.
     */
    void compress(int thr_rank, const inf_array_t& inf_array, bool apply, unsigned int n_elems);

    /*!
     * \brief Bool specialization of compress().
     * \param thr_rank Calling thread's rank (used only for the
     *                 clear_int_st_bool() call).
     * \param apply The scan always runs; the compression only happens when
     *              this is true.
     */
    void compress_bool(int thr_rank, bool apply, unsigned int n_elems);

    /**
     * \brief Shared body of compress() and compress_bool(): they differ only in
     *        how the second value is obtained and which clear_int_st overload
     *        applies it.
     * \tparam SecondValueFunc Nullary callable returning the second value.
     * \tparam ClearFunc callable (int num_compress_value).
     */
    template <typename SecondValueFunc, typename ClearFunc>
    void compress_impl(bool apply, unsigned int n_elems, SecondValueFunc get_second_value, ClearFunc&& clear_one);

    /**
     * \brief Shared body of the two fill_list_chunk_to_compress_in_par entry
     *        points: they differ only in the map key and the per-chunk scan.
     * \tparam ScanFn callable (chunk_type*, num_chunk, num_tab, ChunkElems count).
     */
    template <typename ScanFn>
    static void fill_list_chunk_to_compress_in_par_impl(int thr_rank, in_t<T> key, ScanFn&& scan_one);

    /*!
     * \brief Distribute thread `thr`'s already-queued compression candidates
     *        into the two make_block_compression_impl() buckets, by asking
     *        each candidate chunk to (re-)assess itself
     *        (fill_list_to_compress()).
     * \param thr Thread whose candidate list to distribute.
     * \param ll_compt1 Output: candidates confirmed uniform on the first constant.
     * \param ll_compt2 Output: candidates confirmed uniform on the second constant.
     * \param inf_array (second_value, ...) tuple identifying which
     *                  candidate list to read.
     */
    static void fill_list_chunk_to_compress_in_par(int                     thr,
                                                   vec_tuple_chunk2p_int2& ll_compt1,
                                                   vec_tuple_chunk2p_int2& ll_compt2,
                                                   const inf_array_t&      inf_array);

    /*!
     * \brief Bool specialization of fill_list_chunk_to_compress_in_par().
     * \param thr Thread whose candidate list to distribute.
     * \param ll_compt1 Output: candidates confirmed uniform on the first constant.
     * \param ll_compt2 Output: candidates confirmed uniform on the second constant.
     */
    static void fill_list_chunk_to_compress_bool_in_par(int                     thr,
                                                        vec_tuple_chunk2p_int2& ll_compt1,
                                                        vec_tuple_chunk2p_int2& ll_compt2);

    /*!
     * \brief Re-scan THIS chunk's live content and, if it is now uniform on
     *        the first and/or second constant, append it to the matching
     *        output bucket. Called once per candidate by
     *        fill_list_chunk_to_compress_in_par().
     * \tparam SecondValueFunc Callable returning the second constant to test
     *                          against.
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     * \param get_second_value Supplies the second constant to compare against.
     * \param check_cst_val2 Whether the second-constant check is worth doing
     *                        at all (skipped if the chunk is already known
     *                        to hold the second constant).
     */
    template <typename SecondValueFunc>
    void fill_list_to_compress_impl(vec_tuple_chunk2p_int2& ll_compt1,
                                    vec_tuple_chunk2p_int2& ll_compt2,
                                    int                     num_chunk,
                                    int                     num_tab,
                                    SecondValueFunc&&       get_second_value,
                                    bool                    check_cst_val2,
                                    unsigned int            n_elems);

    /*!
     * \brief Non-bool wrapper around fill_list_to_compress_impl() that
     *        supplies the second constant from `inf_array`.
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param inf_array (second_value, ...) tuple supplying the second constant.
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     */
    void fill_list_to_compress(vec_tuple_chunk2p_int2& ll_compt1,
                               vec_tuple_chunk2p_int2& ll_compt2,
                               const inf_array_t&      inf_array,
                               int                     num_chunk,
                               int                     num_tab,
                               unsigned int            n_elems);

    /*!
     * \brief Bool wrapper around fill_list_to_compress_impl() (the second
     *        constant is always `true` for bool).
     * \param ll_compt1 Output bucket for "uniform on the first constant".
     * \param ll_compt2 Output bucket for "uniform on the second constant".
     * \param num_chunk This chunk's index within its array.
     * \param num_tab Which array this chunk belongs to.
     */
    void fill_list_to_compress_bool(vec_tuple_chunk2p_int2& ll_compt1,
                                    vec_tuple_chunk2p_int2& ll_compt2,
                                    int                     num_chunk,
                                    int                     num_tab,
                                    unsigned int            n_elems);
};

/// @brief Chunk2FastCompPolicy with fast block compression compiled out:
///        no candidate list, so every query answers "empty" and every pass
///        is a no-op.
template <typename T, typename Derived>
struct Chunk2FastCompPolicy<T, Derived, false> : public Chunk2OmpPolicy<T, Derived>, public ChunkSpec2<T>
{
    /// @name Empty candidate lists
    /// @{
    [[nodiscard]] static bool is_vec_tuple_to_compress_potentially_empty_thr() noexcept
    {
        return true;
    }
    static bool have_chunk_to_compress(T, int)
    {
        return false;
    }
    static bool have_chunk_to_compress_bool(int)
    {
        return false;
    }
    /// @}

    // No-op counterparts of the 4 methods added to the true specialization
    // above -- same pattern as ChunkFastCompPolicy<T,Derived,false>::
    // make_block_compression(int, const Proxy<T>*) {}.
    // (chunk, num_chunk, num_tab, n_elems) -- n_elems is the chunk's OWN
    // geometry, carried from the candidate queue so the clear loop can pin the
    // thread to it before freeing (same shape as vec_tuple_chunkA2p_int2).
    using vec_tuple_chunk2p_int2 = std::vector<std::tuple<Chunk2<T>*, int, int, unsigned int>>;
    using inf_array_t            = std::tuple<T, int, int>;

    /// @name No-op compression passes
    /// @{
    template <typename FillFunc, typename ClearFunc>
    static void make_block_compression_impl(int, T, Proxy2<T>*, FillFunc&&, ClearFunc&&)
    {
    }

    static void make_block_compression(int, const inf_array_t&, Proxy2<T>*) {}

    static void make_block_compression_bool(int, Proxy2<T>*) {}

    static void make_block_compression(int) {}
    /// @}
};

/*!
 * \brief Block compression class (power of 2) 2 values automatic detection
 */

template <typename T>
class Chunk2 : public Chunk2CounterPolicy<chunk_config::use_compression_with_counter>,
               public Chunk2FastCompPolicy<T, Chunk2<T>, chunk_config::use_fast_block_compression>
{
  public:
    // Statistics variant tag (CHUNK_STATS): 1 = chunk2.
    static constexpr int stat_variant = 1;

    using Base     = Chunk2Base<T, Chunk2>;
    using BaseSpec = ChunkSpec2<T>;

    using ChunkType              = Chunk2;
    using CompressionManagerType = CompressionManager<T, ChunkType>;

    friend class Proxy2<T>;
    friend class Chunk2Base<T, Chunk2>;

    using Base::allocator_data;
    using Base::first_value;
    using Base::values;

    using CounterPolicy = Chunk2CounterPolicy;

    using FastCompPolicy = Chunk2FastCompPolicy<T, Chunk2, chunk_config::use_fast_block_compression>;
    using OmpPolicy      = Chunk2OmpPolicy<T, Chunk2, build_config::use_openmp>;

    // Both OmpPolicy specialisations declare these (with mode-specific
    // signatures); the in-class bodies below call them unqualified.
    using OmpPolicy::block_clear;
    using OmpPolicy::clear_int_st;
    using OmpPolicy::clear_st;
    using OmpPolicy::create_value;
    using OmpPolicy::create_value_bool;

  private:
    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

  public:
    /*!
     * \brief Default constructor.
     */
    Chunk2() = default;
    /*!
     * \brief Default destructor.
     */
    ~Chunk2() = default;

    using vec_tuple_chunk2p_int2 = std::vector<std::tuple<Chunk2*, int, int>>;

    using manager_chunk2 = ManagerChunk<Chunk2, T>;
    using inf_array_t    = std::tuple<T, int, int>;

    /// @brief Get reference to the chunk manager (static access)
    /// @return Reference to the manager_chunk2 static instance
    /// @details On the class, not on the OpenMP policy (where it used to sit,
    ///          unlike Chunk / Chunk2A): ChunkBase::release_to_owner() names
    ///          derived().get_manager() in every build -- its foreign-block
    ///          park is unreachable serially but must still compile
    ///          (gnu-debug chunk2 dyn: "Chunk2<double> has no member named
    ///          get_manager").
    ///
    ///          NOTE: was a function-local `static manager_chunk2 s_manager;`
    ///          (lazy Meyers singleton). Chunk2<T> is a class template compiled
    ///          across several TUs / library targets; the lazy guard's first
    ///          init happened from compress() inside an OpenMP region,
    ///          deadlocking threads on __cxa_guard_acquire. An `inline static`
    ///          member has one definition across TUs (C++17) and is built
    ///          during static init, before main() / any thread -- no guard, no
    ///          race. The manager ctor is trivial (sets max_thr only).
    static auto& get_manager() noexcept
    {
        return s_manager_;
    }

  private:
    inline static manager_chunk2 s_manager_{};

  public:
    /**
     * @brief Copy constructor: copies the values pointer (shallow copy; the
     *        chunk does not own/duplicate the pointed-to buffer) AND the
     *        Chunk2CounterPolicy counters.
     * @param other Chunk to copy from.
     * @warning The counter base MUST be copied. Left out of the initialiser
     *          list it is default-constructed and both counters restart at 0,
     *          and since this user-declared copy ctor suppresses the implicit
     *          move ctor, every std::vector<Chunk2> reallocation goes through
     *          here. The counter then under-counts by the elements already
     *          equal to the value and can no longer reach `size`: compression
     *          of the chunk is delayed or missed for good. Same defect as the
     *          one fixed in Chunk v1 (chunk_std.h). `values` is atomic, so it
     *          cannot come through a base copy and is copied by hand below.
     */
    Chunk2(const Chunk2& other) : Chunk2CounterPolicy<chunk_config::use_compression_with_counter>(other)
    {
        if constexpr (build_config::use_openmp)
        {
            values.store(other.values.load());
        }
        else
        {
            values = other.values;
        }
    }

    /**
     * @brief Copy assignment: copies the values pointer (shallow copy; the
     *        chunk does not own/duplicate the pointed-to buffer) AND the
     *        Chunk2CounterPolicy counters.
     * @param other Chunk to copy from.
     * @return *this.
     * @see Chunk2(const Chunk2&) for why the counters must follow `values`.
     */
    Chunk2<T>& operator=(const Chunk2<T>& other)
    {
        if (this == &other)
            return *this;

        // Same reason as the copy ctor: propagate the counter state.
        Chunk2CounterPolicy<chunk_config::use_compression_with_counter>::operator=(other);

        if constexpr (build_config::use_openmp)
        {
            T* loaded_value = other.values.load();
            values.store(loaded_value);
        }
        else
        {
            values = other.values;
        }

        return *this;
    }

    /*!
     * \brief Stores a pointer to the chunk's values.
     * \param ptr Pointer to store.
     */

    void values_store(T* ptr) noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            values.store(ptr, std::memory_order_release);
        }
        else
        {
            values = ptr;
        }
    }

    /*!
     * \brief Point this chunk's values at the shared first-constant buffer
     *        (compress to cst_val1).
     */
    void init_values_cst_val1() noexcept
    {
        values = Base::get_cst_val1();
    }
    /*!
     * \brief Initializes internal pointers after construction.
     */

    void init_after_constructor()
    {
        if constexpr (is_not_bool)
        {
            init_values_cst_val1();
        }
        else
        {
            init_values_cst_val_bool(0);
        }
    }

    /*!
     * \brief Non-bool: point this chunk's values at the requested constant.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            second constant at index `ind`.
     * \param ind Index of the second constant (unused when
     *            num_compress_value == 0).
     */
    void init_values_cst_val(int num_compress_value, int ind)
    {
        if (num_compress_value == 0)
        {
            values = Base::get_cst_val1();
        }
        else
        {
            if constexpr (is_not_bool)
            {
                values = BaseSpec::get_cst_val2(ind);
            }
            else
            {
                if constexpr (build_config::any_check)
                {
                    std::cerr << "Logic Error: init_values_cst_val called for boolean type with index "
                              << num_compress_value << ". Use init_values_cst_val_bool instead." << '\n';
                    std::abort();
                }
            }
        }
    }

    /*!
     * \brief Bool-only: point this chunk's values at the requested constant.
     * \param num_compress_value 0 for the first constant, non-zero for the
     *                            (single, shared) second constant.
     */
    void init_values_cst_val_bool(int num_compress_value)
    {
        if constexpr (is_bool)
        {
            values = num_compress_value == 0 ? Base::get_cst_val1() : BaseSpec::m_ptr_cst_val2.data();
        }
        else
        {
            // This branch is reached if the method is called on a non-boolean chunk
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: init_values_cst_val_bool called on a non-boolean chunk type." << '\n';
                std::abort();
            }
        }
    }

    /*!
     * \brief Attempt to atomically compress the chunk to either cst_val1 or
     *        cst_val2(ind).
     * \param num_compress_value 0 to target cst_val1, non-zero to target
     *                            the second constant at index `ind`.
     * \param ind Index of the second constant (unused when
     *            num_compress_value == 0).
     * \return nullptr if the chunk was already compressed or another thread
     *         won the race (caller must not free anything); otherwise the
     *         old pointer, which the caller now owns and must free/recycle.
     */
    T* try_mark_cst_and_get_old(int num_compress_value, int ind)
    {
        // Load the current pointer (may point to allocated data or to a constant block)
        T* old = Base::values_load();
        if constexpr (is_bool)
        {
            // If already compressed, nothing to do
            if (old == Base::get_cst_val1() || BaseSpec::is_cst_val2(old)) [[unlikely]]
                return nullptr;
        }
        else
        {
            if (old == Base::get_cst_val1() || BaseSpec::is_cst_val2(old, ind)) [[unlikely]]
                return nullptr;
            // No second value declared: there is no block to compress to. The
            // scans never file such a chunk under the second constant
            // (scan_constants reports a run of first_value as `first` only),
            // so this is a guard, not a path.
            if (num_compress_value != 0 && ind < 0) [[unlikely]]
            {
                if constexpr (build_config::any_check)
                    INVARIANT_FAILED("[chunk] compress to cst_val2 on an array with no second value");
                return nullptr;
            }
        }

        // We will attempt to replace 'old' with the appropriate constant pointer
        T* expected = old;
        T* new_ptr  = nullptr;

        // Select the target constant pointer depending on compression mode
        if (num_compress_value == 0)
        {
            // Compress to cst_val1
            new_ptr = Base::get_cst_val1();
        }
        else
        {
            if constexpr (is_bool)
            {
                new_ptr = BaseSpec::get_cst_val2();
            }
            else
            {
                // Compress to cst_val2(ind)
                new_ptr = BaseSpec::get_cst_val2(ind);
            }
        }

        // Atomic compare-and-swap:
        // Only the thread that successfully replaces 'old' by 'new_ptr'
        // is allowed to free/recycle the old block.
        CHUNK_CHECK_VALUES_PTR_C(new_ptr, "try_mark.new_ptr", this);
        if (!values.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel, std::memory_order_acquire))
            [[unlikely]]
        {
            // Another thread changed 'values' in the meantime.
            // This thread must not free anything.
            return nullptr;
        }

        // Success: this thread performed the unique transition to a constant value.
        // It must free/recycle the old pointer.
        return old;
    }

    /*!
     * \brief Bool-only: whether this chunk is currently materialised (holds
     *        real data, neither constant).
     * \return True if values differs from both the first and second constant.
     */
    [[nodiscard]] bool written() const
    {
        if constexpr (is_bool)
        {
            return values != Base::get_cst_val1() && is_not_cst_val2();
        }
        else
        {
            // This version is only for boolean chunks
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: written() called on a non-boolean chunk. "
                          << "For non-bool types, use written(int ind) instead." << '\n';
                std::abort();
            }
            return false;
        }
    }

    // Teardown ownership: does this chunk own a real heap buffer (to be parked/
    // freed) rather than cst_val1 or ANY cst_val2 marker? clear_core must key on
    // THIS, not written(ind): a single index misclassifies chunks compressed at
    // other indices, so clear_core parked their never-allocated markers ->
    // b_park of UNTRACKED (std-alloc: double-free of a static marker).
    /*!
     * \brief Whether this chunk owns a real heap buffer that must be
     *        parked/freed at teardown, as opposed to pointing at cst_val1 or
     *        ANY cst_val2 marker.
     * \return True if the chunk's values pointer is a live, owned buffer.
     */
    [[nodiscard]] bool owns_live_buffer() const noexcept
    {
        const T* v = Base::values_load();
        if (v == Base::get_cst_val1())
            return false;
        if constexpr (is_bool)
            return !BaseSpec::is_cst_val2(v);
        else
            return !BaseSpec::is_cst_val2_any(v);
    }

    /*!
     * \brief Non-bool: whether this chunk is currently materialised (holds
     *        real data, neither constant) at second-value index `ind`.
     * \param ind Index of the second constant value.
     * \return True if values differs from both constants.
     */
    [[nodiscard]] bool written(int ind) const
    {
        if constexpr (is_not_bool)
        {
            return values != Base::get_cst_val1() && is_not_cst_val2(ind);
        }
        else
        {
            // This version expects an index, which boolean chunks do not use
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: written(int ind) called on a boolean chunk. "
                          << "Boolean chunks use a single shared second value, use written() instead." << '\n';
                std::abort();
            }
            return false;
        }
    }

    /*!
     * \brief Checks whether the chunk IS currently compressed to the second
     *        constant at index `ind`.
     * \param ind Index of the second value.
     * \return True if values equals the second constant at ind.
     */
    [[nodiscard]] bool is_cst_val2(int ind) const noexcept
    {
        if constexpr (is_not_bool)
        {
            return values == BaseSpec::get_cst_val2(ind);
        }
        else
        {
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: is_cst_val2(int ind) called on a boolean chunk. "
                          << "Boolean chunks do not use indexed constants." << '\n';
                std::abort();
            }
            return false;
        }
    }

    /*!
     * \brief Bool-only: checks whether the chunk IS currently compressed to
     *        the (single, shared) second constant.
     * \return True if values equals the second constant.
     */
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        if constexpr (is_bool)
        {
            return values == BaseSpec::get_cst_val2();
        }
        else
        {
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: is_cst_val2() called on a non-boolean chunk. "
                          << "Non-bool types require an index (int ind)." << '\n';
                std::abort();
            }
            return false;
        }
    }

    /*!
     * \brief Checks that the chunk does NOT use the second constant at
     *        index `ind`.
     * \param ind Index of the second value.
     * \return True if values differs from the second constant at ind.
     */
    [[nodiscard]] bool is_not_cst_val2(int ind) const noexcept
    {
        if constexpr (is_not_bool)
        {
            return !is_cst_val2(ind);
        }
        else
        {
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: is_not_cst_val2(int ind) called on a boolean chunk." << '\n';
                std::abort();
            }
            return true;
        }
    }

    /*!
     * \brief Bool-only: negation of is_cst_val2().
     * \return True if values does NOT equal the (single, shared) second constant.
     */
    [[nodiscard]] bool is_not_cst_val2() const noexcept
    {
        if constexpr (is_bool)
        {
            return !is_cst_val2();
        }
        else
        {
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: is_not_cst_val2() called on a non-boolean chunk." << '\n';
                std::abort();
            }
            return true;
        }
    }

    /*!
     * \brief Bool-only: whether the chunk is currently compressed to EITHER
     *        constant (cst1 or cst2).
     * \return true if the chunk is compressed to either constant.
     */
    [[nodiscard]] bool is_cst_val() const noexcept
    {
        if constexpr (is_bool)
        {
            // For booleans, check if it's either cst1 (false) or cst2 (true)
            return Base::is_cst_val1() || is_cst_val2();
        }
        else
        {
            if constexpr (build_config::is_debug)
            {
                std::cerr << "Logic Error: is_cst_val() without index called on a non-boolean chunk." << '\n';
                std::abort();
            }
            return false;
        }
    }

    /*!
     * \brief Whether the chunk is currently compressed to EITHER constant
     *        (cst1 or cst2). Works for both bool and non-bool chunks --
     *        unlike most other predicates in this class, this one does not
     *        abort on the "wrong" type: `ind` is simply ignored for bool
     *        (which has a single shared second constant).
     * \param ind Index of the second constant (used only for non-bool chunks).
     * \return true if the chunk is compressed to either constant.
     */
    [[nodiscard]] bool is_cst_val(int ind) const noexcept
    {
        if constexpr (is_not_bool)
        {
            return Base::is_cst_val1() || BaseSpec::is_cst_val2(values, ind);
        }
        else
        {
            return Base::is_cst_val1() || BaseSpec::is_cst_val2(values);
        }
    }

    /*!
     * \brief Materialise this chunk if it is currently compressed to either
     *        constant; a no-op otherwise (already materialised).
     * \param inf_array (second_value, ...) tuple describing the target state.
     */
    void alloc_chunk(const inf_array_t& inf_array)
    {
        if constexpr (is_not_bool)
        {
            if (is_cst_val(std::get<1>(inf_array)))
            {
                create_value(false, inf_array);
            }
        }
        else
        {
            if (is_cst_val())
            {
                create_value(false, inf_array);
            }
        }
    }

    /*!
     * \brief Materialise this chunk if it is currently compressed to either
     *        constant; a no-op otherwise. Per-thread variant.
     * \param inf_array (second_value, ...) tuple describing the target state.
     * \param thr_rank Calling thread's rank (OMP builds only; ignored in
     *                 serial builds).
     */
    void alloc_chunk(const inf_array_t& inf_array, [[maybe_unused]] int thr_rank)
    {
        if (is_cst_val(std::get<1>(inf_array)))
        {
            if constexpr (chunk_config::stats)
            {
                increm_chunk_write<ChunkType>();
            }

            if constexpr (build_config::use_openmp)
            {
                create_value(false, inf_array, thr_rank);
            }
            else
            {
                create_value(false, inf_array);
            }
        }
    }

    // --- what used to follow here, and where it went ---------------------
    // The OpenMP surface (the create_value family, clear(),
    // block_clear(), the clear_int_st family, clear_st()) sits on
    // Chunk2OmpPolicy<T, Chunk2, build_config::use_openmp>, one specialisation
    // per mode; the serial one carries the counter resets the parallel path
    // does not maintain. The queue-driven compression (compress*,
    // fill_list*, compress_chunk_to_compress*)
    // sits on Chunk2FastCompPolicy<T, Chunk2, true>, next to
    // make_block_compression -- same technique as ChunkOmpPolicy for Chunk v1.
};

} // end namespace shrink_array

#include "chunk/internal/chunk2_std_inline.h"
