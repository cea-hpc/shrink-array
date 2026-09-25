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

// #define CHUNK_STATS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <complex>
#include <tuple> // the candidate-list entries
#include <vector>

/// @brief The complex element type of the library (chunk_std.cpp instantiates it).
using complex_double = std::complex<double>;

#include "chunk/internal/process_verbs.h" // compress_now_and_trim_pools, defined in chunk_std.cpp
#include "chunk/internal/chunk_base.h"
#include "chunk/chunk_define.h"               // build macros
#include "chunk/internal/chunk_size_config.h" // ChunkElems
#include "chunk/internal/chunk_stats.h"       // increm_chunk_write (CHUNK_STATS)
#include "chunk/internal/conditional_counter.h"
#include "chunk/internal/proxy.h"

namespace shrink_array
{


/// @brief The per-type facade these verbs drive. Forward-declared, not included:
///        chunk_std.h sits inside the chunk_func.h -> chunk_std.h -> proxy.h ->
///        manager_chunk.h include cycle and cannot pull all_type_chunk.h in.
///        Only chunk_std.cpp, which defines the bodies, needs the full type.
class AllTypeChunk;

/**
 * @brief The chunk reuse pools' load-balancing and teardown surface, split on
 *        the thread model.
 *
 * @tparam Enabled build_config::use_openmp.
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING -- same role, same
 *                 reason, as on ParamChunkParallelInterface (param_chunk.h):
 *                 it keeps the bodies TEMPLATES so a build only ever
 *                 instantiates its own, and it keeps the calls DEPENDENT so
 *                 phase-1 lookup does not resolve them against the concrete
 *                 facade at definition time. Without it these bodies cannot
 *                 coexist, because the two halves of the teardown are verbs of
 *                 DIFFERENT policies: AllTypeChunkParallelInterface<Derived,
 *                 true> declares clear_list_block_free_thr() and
 *                 delete_list_block_chunk_free_itself(), <Derived, false>
 *                 declares delete_chunk_allocator(), and neither declares the
 *                 other's. That asymmetry is what the #ifdef _OPENMP in
 *                 chunk_std.cpp used to paper over.
 *
 * @note EVERY verb exists in BOTH specialisations, so a caller needs no build
 *       guard of its own. The ones with nothing to do in a given model are
 *       honest no-ops -- a serial build has no per-thread free lists to clear
 *       or balance, an OpenMP build frees its pools through the per-rank pair
 *       rather than through the single allocator.
 * @see release_chunk_pools(), which is the verb to call when what you want is
 *      simply "let go of the reuse pools", whichever build this is.
 */
template <bool Enabled = build_config::use_openmp, typename AllT = AllTypeChunk>
struct ChunkPoolParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: the pools are per-rank.
template <typename AllT>
struct ChunkPoolParallelInterface<true, AllT>
{
    /// @return True as soon as the per-rank free lists are unbalanced enough to
    ///         be worth a pass.
    [[nodiscard]] static bool need_load_balance_list_block_free();
    /// @brief Redistribute the free blocks of every type across the ranks.
    static void load_balance_list_block_free();
    /// @brief Empty every type's per-rank free-block lists.
    static void clear_all_list_block_free_thr();
    /// @brief Release the per-rank list containers themselves.
    static void delete_list_block_chunk_free_itself();
    /// @brief No-op here: an OpenMP build has no single allocator to delete, it
    ///        releases through the per-rank pair above. Present so callers need
    ///        no `#ifdef _OPENMP`.
    static void delete_chunk_allocator() {}
    /// @copydoc shrink_array::release_chunk_pools
    static void release_chunk_pools();
};

/// @brief Serial builds: one allocator, no per-rank lists to balance or clear.
template <typename AllT>
struct ChunkPoolParallelInterface<false, AllT>
{
    /// @return Always false: there is a single free list, so nothing to balance.
    [[nodiscard]] static bool need_load_balance_list_block_free()
    {
        return false;
    }
    /// @brief No-op: see need_load_balance_list_block_free().
    static void load_balance_list_block_free() {}
    /// @brief No-op: a serial build keeps no per-rank free lists.
    static void clear_all_list_block_free_thr() {}
    /// @brief No-op: see clear_all_list_block_free_thr().
    static void delete_list_block_chunk_free_itself() {}
    /// @brief Delete this build's single chunk allocator.
    static void delete_chunk_allocator();
    /// @copydoc shrink_array::release_chunk_pools
    static void release_chunk_pools();
};

extern void destroy_block_chunk();

// Build-agnostic free functions over the policy above. Declared -- and defined,
// in chunk_std.cpp -- in EVERY build, which is the point: delete_chunk_allocator()
// and need_load_balance_list_block_free() used to be declared here
// unconditionally but defined in one model only, so calling either from the
// other build was a link error waiting to happen.
extern void delete_chunk_allocator();
extern void clear_all_list_block_free_thr();
extern void load_balance_list_block_free();
extern void delete_list_block_chunk_free_itself();
extern bool need_load_balance_list_block_free();

/**
 * @brief Let go of the chunk reuse pools, whichever thread model this build is.
 *
 * @details The teardown halves are not the same verb in the two models, so a
 *          caller used to have to know which it was in: OpenMP releases the
 *          per-rank lists (clear_all_list_block_free_thr() then
 *          delete_list_block_chunk_free_itself()), a serial build deletes its
 *          single allocator (delete_chunk_allocator()). This does the right one.
 * @note Pair it with destroy_block_chunk(), which is build-agnostic already.
 */
extern void release_chunk_pools();


/**
 * @brief Per-chunk zero-counter, the serial path's compression trigger.
 *
 * @details Chunk v1 compresses a chunk once ALL its elements are zero. Rather
 *          than rescanning the buffer on every write, the serial build keeps a
 *          running count of zero elements and compresses when it reaches the
 *          chunk size. The OpenMP build does not use this: it queues candidates
 *          and scans them in batches instead (see ChunkFastCompPolicy), which is
 *          why the counter is behind a policy rather than always present.
 *
 * @tparam EnableCounter chunk_config::use_compression_with_counter; false makes
 *                       ConditionalCounter an empty type, so the counter costs
 *                       nothing in the modes that do not consult it
 * @warning The count must track the buffer exactly. Chunk's copy constructor and
 *          copy assignment forward to this base explicitly for that reason --
 *          see their comments; a desynchronised counter compresses a chunk that
 *          is not actually all-zero, and later reads return the constant.
 */

template <bool EnableCounter>
struct ChunkCounterPolicy
{
    // [[no_unique_address]]: a DISABLED counter is an empty type, but an
    // empty MEMBER still takes a byte unless told otherwise -- and that byte,
    // as a base of Chunk, pushed `values` to offset 8 and the chunk header
    // from 16 to 24 bytes in every OpenMP build (where the counter mode is
    // off). Enabled, the attribute changes nothing.
    [[no_unique_address]] ConditionalCounter<EnableCounter> nb_val_eq_zero;

    /// @brief Record one more zero element.
    void inc_nb_val_eq_zero() noexcept
    {
        ++nb_val_eq_zero; // Uses operator++
    }

    /// @brief Record one fewer zero element.
    void dec_nb_val_eq_zero() noexcept
    {
        --nb_val_eq_zero; // Uses operator--
    }

    /// @brief Set the zero count outright.
    /// @param v new count
    void set_nb_val_eq_zero(counter_value_t v) noexcept
    {
        nb_val_eq_zero = v; // Uses operator=
    }

    /// @brief Is every element of the chunk zero?
    /// @param size the chunk's element count
    /// @return true when the count has reached @p size -- the compression trigger
    [[nodiscard]] bool is_full_zero(unsigned int size = __chunk_size) const noexcept
    {
        // Implicit conversion to unsigned char happens here
        return nb_val_eq_zero == size;
    }
};


/**
 * @brief Chunk's OpenMP-only surface: per-thread allocation, CAS and clearing.
 *
 * @details A serial build has no thread ranks, no CAS-stealing window and no
 *          per-thread free-lists, so the members below have no meaning there.
 *          Splitting them onto the build_config::use_openmp axis -- the same
 *          technique used by ChunkDeferredOpsPolicy and
 *          CompressionManagerOmpPolicy -- replaces the preprocessor blocks that
 *          used to fence them off, and lets each mode declare exactly the
 *          clearing verb it actually has: clear_st(int) here, clear_st() in the
 *          serial specialisation.
 *
 *          Sits between ChunkFastCompPolicy and ChunkBase in the inheritance
 *          chain, so Chunk inherits both policies through one path.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 * @tparam Enabled build_config::use_openmp
 */
template <typename T, typename Derived, bool Enabled = build_config::use_openmp>
struct ChunkOmpPolicy : public ChunkBase<T, Derived>
{
    using Base          = ChunkBase<T, Derived>;
    using ChunkType     = Derived;
    using manager_chunk = ManagerChunk<Chunk, T>;

    using Base::derived;
    using Base::init_block;
    using Base::is_cst_val1;
    using Base::self;
    using Base::values;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new Chunk<T>[n]` runs the DEFAULT CONSTRUCTORS after this, so
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
    ///          would hang waiting for the others). Public: an application
    ///          that placement-news an array of chunks initialises it here.
    static void init_chunks_cst_val(Chunk<T>* v, std::size_t count) noexcept;

    /// @brief CAS this chunk's values pointer into place.
    /// @param values_loc buffer to publish
    /// @return true if this thread won the exchange
    bool compare_and_swap(T* values_loc) noexcept
    {
        return Base::compare_and_swap1(values_loc);
    }

    /**
     * @brief Zero-fill a buffer that allocate_chunk_internal has just created.
     * @param vals freshly allocated buffer of shrink_array::alloc_chunk_count() elements
     * @note Only the fresh-allocation branch needs it -- see create_value().
     */
    static void zero_fill_fresh(T* vals) noexcept;

    /**
     * @brief Materialise this chunk's storage from @p thr_rank's free-list.
     * @param thr_rank calling thread's rank, selecting its allocator pool
     */
    void create_value(int thr_rank) noexcept;

    /// @brief Materialise storage for the calling thread.
    /// @details Delegates to the rank-aware overload using this thread's rank.
    void create_value() noexcept;

    /**
     * @brief Atomically mark this chunk constant and hand back its old buffer.
     * @return the previous values pointer, or nullptr if the chunk was already
     *         constant or another thread won the exchange
     * @note This is the CAS-steal that makes a compressor's window dangerous to
     *       concurrent writers -- see CompressionManager's hard_inflight_.
     */
    T* try_mark_cst_and_get_old()
    {
        T* old = Base::values_load();
        if (old == Base::get_cst_val1()) [[unlikely]]
            return nullptr;

        T* expected = old;
        T* new_ptr  = Base::get_cst_val1();

        if (!values.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel, std::memory_order_acquire))
            [[unlikely]]
        {
            return nullptr;
        }
        return old;
    }

    /// @brief Compress: release this chunk's buffer to @p thr_rank's reuse
    ///        list and count it as a compression (CHUNK_STATS).
    void clear_st(int thr_rank, unsigned int n_elems);
    /// @brief Release this chunk's buffer to @p thr_rank's reuse list WITHOUT
    ///        counting a compression -- the re-chunk rebuild releasing an old
    ///        chunk once copied (clear_st = count + release_st).
    void release_st(int thr_rank, unsigned int n_elems, bool count_release = true);
    /// @brief Release the buffer outright, without parking it for reuse.
    void clear_int_st(int thr_rank);

    /**
     * @brief Compress this chunk if all @p n_elems elements are zero.
     * @param thr_rank calling thread's rank
     * @param n_elems  THIS chunk's element count, carried by the candidate
     *                 tuple; using the ambient size instead overflows a buffer
     *                 allocated at another geometry
     * @note The queue consumer's per-chunk step (Chunk::compress_chunk_to_compress).
     */
    void compress(int thr_rank, unsigned int n_elems);
};

/**
 * @brief Serial specialisation: one clearing verb, no thread rank.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 */
template <typename T, typename Derived>
struct ChunkOmpPolicy<T, Derived, false> : public ChunkBase<T, Derived>
{
    using Base = ChunkBase<T, Derived>;
    using Base::derived;
    using Base::values;

    /// @brief Materialise this chunk's storage.
    /// @details Allocates directly: there is no per-thread free-list to consult.
    void create_value() noexcept;

    /// @brief Compress: release this chunk's buffer and count it as a
    ///        compression (CHUNK_STATS).
    void clear_st();
    /// @brief Release this chunk's buffer WITHOUT counting a compression --
    ///        the teardown, counted as teardown_clear (clear_st = count +
    ///        release_st).
    void release_st(bool count_release = true);
};

/**
 * @brief Batched, queue-driven compression -- the OpenMP path's trigger.
 *
 * @details Where the serial build compresses eagerly on a zero-counter (see
 *          ChunkCounterPolicy), this one defers: a write that leaves a chunk
 *          eligible queues it on the writer's per-thread candidate list, and a
 *          later pass scans and compresses the batch. That keeps the write path
 *          free of a full buffer scan, at the cost of the candidate bookkeeping
 *          held here.
 * *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 * @tparam Enabled chunk_config::use_fast_block_compression, i.e. embedded
 *                 compression AND OpenMP; the false specialisation below is all
 *                 no-ops
 */
template <typename T, typename Derived, bool Enabled>
struct ChunkFastCompPolicy : public ChunkOmpPolicy<T, Derived>
{
    // 4th element: the chunk's OWN geometry, captured at enqueue by
    // push_to_compress_list. compress() must scan exactly that many elements --
    // NOT alloc_chunk_count(), which is the CONSUMING thread's ambient size and
    // may differ (heap-buffer-overflow: a 64-element buffer scanned at 256).
    using vec_tuple_chunkp_int2 = std::vector<std::tuple<Chunk<T>*, int, int, ChunkElems>>;

    using manager_chunk = ManagerChunk<Chunk, T>;
    using Base          = ChunkBase<T, Derived>;

    using Base::derived;
    using Base::self;

    constexpr static T zero{};

    /**
     * @brief Scan @p thr's queued candidates and collect those now compressible.
     * @param thr      rank whose candidate list is scanned
     * @param ll_compt out: the chunks found compressible, with their geometry
     */
    static void fill_list_chunk_to_compress_in_par(int thr, vec_tuple_chunkp_int2& ll_compt);
    /**
     * @brief Append this chunk to @p ll_compt if all @p n_elems elements are zero.
     * @param ll_compt  out: list of compressible chunks
     * @param num_chunk this chunk's index in its array
     * @param num_tab   its array's tab index
     * @param n_elems   THIS chunk's element count -- never the ambient size, see
     *                  TChunk.h's note on the field it is carried in
     */
    void fill_list_to_compress(vec_tuple_chunkp_int2& ll_compt, int num_chunk, int num_tab, unsigned int n_elems);

    /**
     * @brief Compress every chunk @p thr has queued, in one batch.
     * @param thr   rank whose candidates are compressed
     * @param proxy proxy driving the pass; may be null for a generic sweep
     */
    static void make_block_compression(int thr, const Proxy<T>* proxy);

    /**
     * @brief Is any thread holding chunks queued for compression?
     * @param max_thr number of threads whose candidate lists are scanned
     * @return true as soon as one non-empty list is found
     */
    static bool have_chunk_to_compress(int max_thr);
};

/**
 * @brief No-op specialisation: no candidate queue outside fast block compression.
 * @details Every member is a stub, because the call sites are shared with the
 *          enabled path and guarded only by `if constexpr`.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 */
template <typename T, typename Derived>
struct ChunkFastCompPolicy<T, Derived, false> : public ChunkOmpPolicy<T, Derived>
{
    // 4th element: the chunk's OWN geometry, captured at enqueue by
    // push_to_compress_list. compress() must scan exactly that many elements --
    // NOT alloc_chunk_count(), which is the CONSUMING thread's ambient size and
    // may differ (heap-buffer-overflow: a 64-element buffer scanned at 256).
    using vec_tuple_chunkp_int2 = std::vector<std::tuple<Chunk<T>*, int, int, ChunkElems>>;

    /// @brief No candidate list to fill in this mode.
    static void fill_list_chunk_to_compress_in_par(int /*thr*/, vec_tuple_chunkp_int2& /*ll_compt*/) {}
    /// @copydoc fill_list_chunk_to_compress_in_par
    static void fill_list_to_compress(vec_tuple_chunkp_int2& /*ll_compt*/,
                                      int /*num_chunk*/,
                                      int /*num_tab*/,
                                      unsigned int /*n_elems*/)
    {
    }
    /// @brief No block compression in this mode.
    static void make_block_compression(int /*thr*/, const Proxy<T>* /*proxy*/) {}

    /// No candidate lists exist in this mode, so there is never anything queued.
    static bool have_chunk_to_compress(int /*max_thr*/)
    {
        return false;
    }
};
/**
 * @brief One chunk of a ChunkArray: a power-of-two block that collapses to zero.
 *
 * @details The simplest of the three variants. A chunk holds either a real
 *          buffer or, once every element is zero, the shared cst_val1 constant
 *          -- so `values` doubles as the state flag: written() is just
 *          "values != cst_val1". Chunk2 and Chunk2A extend this with a second
 *          constant, shared or per-chunk respectively.
 *
 *          Three policies supply what depends on the build, so this class holds
 *          only what every mode needs:
 *          - ChunkCounterPolicy -- the eager zero-counter trigger (serial)
 *          - ChunkFastCompPolicy -- the batched candidate queue (OpenMP)
 *          - ChunkOmpPolicy -- per-thread allocation, CAS and clearing
 *
 * @tparam T stored element type
 * @note The chunk does NOT know its own element count: the geometry lives in
 *       the owning array's size_policy_, and is passed in by callers. Reading a
 *       buffer with the wrong count is the failure mode the candidate lists
 *       guard against by carrying it -- see TChunk.h.
 */

template <typename T>
class Chunk : private ChunkCounterPolicy<chunk_config::use_compression_with_counter>,
              public ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>
{
  public:
    // Statistics variant tag (CHUNK_STATS): 0 = chunk.
    static constexpr int stat_variant = 0;

    using Base      = ChunkBase<T, Chunk>;
    using ChunkType = Chunk;

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using Base::allocate_chunk_internal;
    using Base::first_value;

    using Base::init_block;
    using Base::is_cst_val1;
    using Base::m_ptr_cst_val1;
    using Base::values;
    using Base::values_load;
    using Base::values_store;

    using Base::free_values;

    friend class Proxy<T>;

    using manager_chunk = ManagerChunk<Chunk, T>;

    using FastCompPolicy = ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>;

    using FastCompPolicy::derived;
    using FastCompPolicy::self;

    Chunk()  = default;
    ~Chunk() = default;

    Chunk(const Chunk& other)
        // Copy the ChunkCounterPolicy base: it holds nb_val_eq_zero, the count
        // of zero elements. A user-defined copy ctor otherwise DEFAULT-constructs
        // the bases, resetting the counter to 0. That desync makes
        // inc_nb_val_eq_zero later reach __chunk_size on a chunk that is not
        // actually all-zero, so manage_full_zero() wrongly re-compresses it to
        // cst_val1 (observed: a later read returns the constant instead of the
        // stored value). We can't copy the ChunkFastCompPolicy base the same way
        // because it contains the atomic `values` (non-copyable) — that's why
        // `values` is copied by hand below.
        : ChunkCounterPolicy<chunk_config::use_compression_with_counter>(other)
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

    Chunk& operator=(const Chunk<T>& other)
    {
        if (this == &other)
            return *this;

        // Same reason as the copy ctor: propagate the zero-counter state.
        ChunkCounterPolicy<chunk_config::use_compression_with_counter>::operator=(other);

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

    /// @brief Point this chunk at the shared zero constant (compressed state).
    void init_values_cst_val()
    {
        values = Base::get_cst_val1();
    }

    /// @brief Does this chunk hold a real buffer?
    /// @return false while it is compressed to cst_val1
    [[nodiscard]] bool written() const noexcept
    {
        return values != Base::get_cst_val1();
    }

    /// @brief Materialise storage if the chunk is still compressed.
    void alloc_chunk()
    {
        if (is_cst_val1())
        {
            this->create_value();
        }
    }

    /// @brief Materialise storage if still compressed, from @p thr_rank's pool.
    /// @param thr_rank calling thread's rank; ignored in serial builds
    void alloc_chunk([[maybe_unused]] int thr_rank)
    {
        if (is_cst_val1())
        {
            if constexpr (chunk_config::stats)
            {
                increm_chunk_write<ChunkType>();
            }

            if constexpr (build_config::use_openmp)
            {
                this->create_value(thr_rank);
            }
            else
            {
                this->create_value();
            }
        }
    }


    /**
     * @brief Release this chunk's buffer and return to the compressed state.
     * @param thr_rank owning thread's rank; unused serially
     * @note The release counter differs by mode (pool vs free) -- see the
     *       definition's comment.
     */
    void block_clear(int thr_rank);

    /// @brief Get reference to the chunk manager (static access)
    static auto& get_manager() noexcept
    {
        // NOTE: was a function-local `static manager_chunk s_manager;` (lazy Meyers
        // singleton). Chunk<T> is a class template compiled across several TUs /
        // library targets; the lazy guard's first init happened from compress()
        // inside an OpenMP region, deadlocking threads on __cxa_guard_acquire.
        // An `inline static` member has one definition across TUs (C++17) and is
        // built during static init, before main() / any thread — no guard, no
        // race. The manager ctor is trivial (sets max_thr only).
        return s_manager_;
    }

  private:
    inline static manager_chunk s_manager_{};

  public:
    /**
     * @brief Drain and compress every thread's candidate list.
     * @param apply true: compress the candidates found constant; false: walk
     *        the lists only (v1 ignores it: its pass always compresses)
     */
    static void compress_chunk_to_compress(bool apply);

    // compress(int thr_rank, unsigned n_elems) lives in ChunkOmpPolicy: it is
    // the queue consumer's per-chunk step (compress_chunk_to_compress above)
    // and needs a thread rank, so it belongs on the OpenMP surface with
    // clear_st(int, unsigned) rather than behind a preprocessor guard here.
    // An OMPT-flavoured overload compress(int, bool, unsigned) used to sit
    // next to it; nothing called it and its fast-comp branch ended in
    // std::abort(), so it is gone.

  private:
    constexpr static T zero{};
};

} // namespace shrink_array

#include "chunk/internal/chunk_std_inline.h"
