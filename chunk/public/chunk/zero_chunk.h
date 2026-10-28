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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <atomic>  // std::memory_order_acquire, std::memory_order_acq_rel
#include <cstddef> // std::size_t
#include <tuple>   // the candidate-list entries
#include <vector>  // std::vector

#include "chunk/internal/chunk_base.h"          // ChunkBase
#include "chunk/config.h"                       // build_config::threaded, chunk_config
#include "chunk/internal/chunk_size_config.h"   // ChunkElems
#include "chunk/internal/chunk_size_policy.h"   // alloc_chunk_count
#include "chunk/internal/chunk_stats.h"         // increm_chunk_write (CHUNK_STATS)
#include "chunk/internal/conditional_counter.h" // ConditionalCounter
#include "chunk/internal/zero_proxy.h"          // Proxy<T>
#include "chunk/internal/immortal.h"            // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/chunk_array_new.h"     // chunk_array_new::init_chunks (init_chunks_cst_val)

namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkT>
struct CompressionManager;

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

/// @brief The per-type facade these verbs drive. Forward-declared, not included:
///        zero_chunk.h sits inside an include cycle (chunk_func.h -> zero_chunk.h ->
///        zero_proxy.h -> manager_chunk.h). Only chunk_std.cpp needs the full type.
class AllTypeChunk;

/**
 * @brief The chunk reuse pools' load-balancing and teardown surface, split on
 *        the thread model.
 *
 * @tparam Enabled build_config::threaded.
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING (as on
 *                 ParamChunkParallelInterface, param_chunk.h): it keeps the
 *                 bodies TEMPLATES, so a build instantiates only its own, and
 *                 the calls DEPENDENT, so phase-1 lookup does not resolve them
 *                 against the concrete facade. The two teardown halves are verbs
 *                 of DIFFERENT policies (AllTypeChunkParallelInterface<Derived,
 *                 true> vs <Derived, false>), neither declaring the other's.
 *
 * @note EVERY verb exists in BOTH specialisations, so callers need no build
 *       guard; those with nothing to do in a model are no-ops.
 * @see release_chunk_pools(), the verb for "let go of the reuse pools" in any build.
 */
template <bool Enabled = build_config::threaded, typename AllT = AllTypeChunk>
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
    ///        no <tt>\#ifdef _OPENMP</tt>.
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

/// @brief When fast allocation is enabled, call destroy_block_chunk() on the AllTypeChunk singleton; otherwise
///        do nothing. Nothing either while an array is alive (see release_chunk_pools()).
extern void destroy_block_chunk();

// Build-agnostic free functions over the policy above, declared and defined
// (chunk_std.cpp) in EVERY build, so calling any of them links in both models.
// The teardown ones (delete_chunk_allocator, clear_all_list_block_free_thr,
// delete_list_block_chunk_free_itself) are sequential points, as
// release_chunk_pools() says.
/// @brief ChunkPoolParallelInterface<>::delete_chunk_allocator() of this build.
extern void delete_chunk_allocator();
/// @brief ChunkPoolParallelInterface<>::clear_all_list_block_free_thr() of this build.
extern void clear_all_list_block_free_thr();
/// @brief ChunkPoolParallelInterface<>::load_balance_list_block_free() of this build.
extern void load_balance_list_block_free();
/// @brief ChunkPoolParallelInterface<>::delete_list_block_chunk_free_itself() of this build.
extern void delete_list_block_chunk_free_itself();
/// @brief ChunkPoolParallelInterface<>::need_load_balance_list_block_free() of this build.
/// @return Its result.
extern bool need_load_balance_list_block_free();

/**
 * @brief Let go of the chunk reuse pools, whichever thread model this build is.
 *
 * @details OpenMP releases the per-rank lists (clear_all_list_block_free_thr()
 *          then delete_list_block_chunk_free_itself()); a serial build deletes
 *          its single allocator (delete_chunk_allocator()).
 * @note Pair it with destroy_block_chunk(). Both are optional: the process can
 *       simply exit.
 * @note While an array is alive (typically a global one, destroyed after main)
 *       this only empties the free lists, and destroy_block_chunk(),
 *       delete_list_block_chunk_free_itself() and delete_chunk_allocator() do
 *       nothing: the array parks its buffers there when it dies.
 * @warning Sequential point: call it once, after the last parallel region. From
 *          inside a region (OpenMP's, the std backend's team or application
 *          phase) a check build aborts; any other build says so on stderr and
 *          does nothing. Every thread of the team would otherwise empty the
 *          same pools at once. The same holds for destroy_block_chunk(),
 *          delete_chunk_allocator(), clear_all_list_block_free_thr(),
 *          delete_list_block_chunk_free_itself() and an array's shrink_to_fit().
 */
extern void release_chunk_pools();

namespace zero
{

/**
 * @brief Per-chunk zero-counter, the serial path's compression trigger.
 *
 * @details zero compresses a chunk once ALL its elements are zero. The
 *          serial build keeps a running count of zero elements instead of
 *          rescanning; the OpenMP build queues candidates and scans them in
 *          batches instead (see ChunkFastCompPolicy).
 *
 * @tparam EnableCounter chunk_config::use_compression_with_counter; false makes
 *                       ConditionalCounter an empty type, so the counter costs
 *                       nothing in the modes that do not consult it
 * @warning The count must track the buffer exactly (hence Chunk's copy
 *          constructor and assignment forward to this base explicitly): a
 *          desynchronised counter compresses a chunk that is not all-zero.
 */
template <bool EnableCounter>
class ChunkCounterPolicy
{
  public:
    /// @brief Record one more zero element.
    void inc_nb_val_eq_zero() noexcept
    {
        ++nb_val_eq_zero_;
    }

    /// @brief Record one fewer zero element.
    void dec_nb_val_eq_zero() noexcept
    {
        --nb_val_eq_zero_;
    }

    /// @brief Set the zero count outright.
    /// @param count new count
    void set_nb_val_eq_zero(CounterValueT count) noexcept
    {
        nb_val_eq_zero_ = count;
    }

    /// @brief Is every element of the chunk zero?
    /// @param size the chunk's element count
    /// @return true when the count has reached @p size -- the compression trigger
    [[nodiscard]] bool is_full_zero(unsigned int size = static_chunk_size) const noexcept
    {
        return nb_val_eq_zero_ == size;
    }

  private:
    // [[no_unique_address]]: a DISABLED counter is an empty type, but an empty
    // MEMBER still takes a byte, which grows the chunk header from 16 to 24
    // bytes in OpenMP builds (counter off).
    [[no_unique_address]] ConditionalCounter<EnableCounter> nb_val_eq_zero_;
};

/**
 * @brief Chunk's OpenMP-only surface: per-thread allocation, CAS and clearing.
 *
 * @details A serial build has no thread ranks, CAS-stealing window or
 *          per-thread free-lists. Split on build_config::threaded (as
 *          ChunkDeferredOpsPolicy, CompressionManagerThreadPolicy): each mode
 *          declares the clearing verb it has, clear_st(int) here, clear_st()
 *          serially.
 *
 *          Sits between ChunkFastCompPolicy and ChunkBase in the inheritance
 *          chain, so Chunk inherits both policies through one path.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 * @tparam Enabled build_config::threaded
 */
template <typename T, typename Derived, bool Enabled = build_config::threaded>
class ChunkThreadPolicy : public ChunkBase<T, Derived>
{
  public:
    using Base      = ChunkBase<T, Derived>;
    using ChunkType = Derived;
    using Manager   = ManagerChunk<Chunk, T>;

    using Base::init_block;
    using Base::is_cst_val1;
    using Base::values_;

    /// @brief Array-new for chunks: allocates and points every chunk at the
    ///        first constant (init_chunks_cst_val).
    /// @warning `new Chunk<T>[n]` runs the DEFAULT CONSTRUCTORS after this,
    ///          overwriting that initialisation; it pays off only for a
    ///          placement-new array. The library never array-news a chunk.
    void* operator new[](std::size_t sz);
    /// @brief Matching array-delete.
    void operator delete[](void* ptr);

    /**
     * @brief Materialise this chunk's storage from @p thr_rank's free-list.
     * @param thr_rank calling thread's rank, selecting its allocator pool
     */
    void create_value(this Derived& self, int thr_rank) noexcept;

    /// @brief Materialise storage for the calling thread.
    /// @details Delegates to the rank-aware overload using this thread's rank.
    void create_value(this Derived& self) noexcept
    {
        self.create_value(get_thr_rank_fast());
    }

    /// @brief Compress: release this chunk's buffer to @p thr_rank's reuse
    ///        list and count it as a compression (CHUNK_STATS).
    void clear_st(this Derived& self, int thr_rank, unsigned int n_elems);
    /// @brief Release this chunk's buffer to @p thr_rank's reuse list WITHOUT
    ///        counting a compression -- the re-chunk rebuild releasing an old
    ///        chunk once copied (clear_st = count + release_st).
    /// @param thr_rank      rank whose reuse list receives the buffer
    /// @param n_elems       element count of this chunk (its array's geometry)
    /// @param count_release count it as a release (CHUNK_STATS `released`);
    ///        clear_st passes false, having counted a compression instead
    void release_st(this Derived& self, int thr_rank, unsigned int n_elems, bool count_release = true);
    /// @brief Release the buffer outright, without parking it for reuse.
    void clear_int_st(this Derived& self, int thr_rank);

    /**
     * @brief Compress this chunk if all @p n_elems elements are zero.
     * @param thr_rank calling thread's rank
     * @param n_elems  THIS chunk's element count, carried by the candidate
     *                 tuple; using the ambient size instead overflows a buffer
     *                 allocated at another geometry
     * @note The queue consumer's per-chunk step (Chunk::compress_chunk_to_compress).
     */
    void compress(this Derived& self, int thr_rank, unsigned int n_elems);

  private:
    /// @brief CAS this chunk's values pointer into place.
    /// @param values_loc buffer to publish
    /// @return true if this thread won the exchange
    bool compare_and_swap(this Derived& self, T* values_loc) noexcept
    {
        return self.Base::compare_and_swap1(values_loc);
    }

    /**
     * @brief Atomically mark this chunk constant and hand back its old buffer.
     * @return the previous values pointer, or nullptr if the chunk was already
     *         constant or another thread won the exchange
     * @note This is the CAS-steal that makes a compressor's window dangerous to
     *       concurrent writers -- see CompressionManager's hard_inflight_.
     */
    T* try_mark_cst_and_get_old(this Derived& self);

    /// @brief Point @p count freshly allocated chunks at the first constant.
    /// @details Parallel past 500 chunks, and only from OUTSIDE a parallel
    ///          region (inside one, an orphaned `omp for` would hang waiting for
    ///          the other threads). An application that placement-news an array
    ///          of chunks initialises it here.
    static void init_chunks_cst_val(Chunk<T>* chunks, std::size_t count) noexcept
    {
        chunk_array_new::init_chunks(chunks, count, [](Chunk<T>& chunk) { chunk.init_values_cst_val(); });
    }

    /**
     * @brief Zero-fill a buffer that allocate_chunk_internal has just created.
     * @param vals freshly allocated buffer of shrink_array::alloc_chunk_count() elements
     * @note Only the fresh-allocation branch needs it -- see create_value().
     */
    static void zero_fill_fresh(T* vals) noexcept;
};

/**
 * @brief Serial specialisation: one clearing verb, no thread rank.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 */
template <typename T, typename Derived>
class ChunkThreadPolicy<T, Derived, false> : public ChunkBase<T, Derived>
{
  public:
    using Base = ChunkBase<T, Derived>;
    using Base::values_;

    /// @brief Materialise this chunk's storage.
    /// @details Allocates directly: there is no per-thread free-list to consult.
    void create_value(this Derived& self) noexcept;

    /// @brief Compress: release this chunk's buffer and count it as a
    ///        compression (CHUNK_STATS).
    void clear_st(this Derived& self);
    /// @brief Release this chunk's buffer WITHOUT counting a compression --
    ///        the teardown, counted as teardown_clear (clear_st = count +
    ///        release_st).
    /// @param count_release as in the threaded policy's release_st
    void release_st(this Derived& self, bool count_release = true);
};

/**
 * @brief Batched, queue-driven compression -- the OpenMP path's trigger.
 *
 * @details Where the serial build compresses eagerly on a zero-counter (see
 *          ChunkCounterPolicy), this one defers: a write that leaves a chunk
 *          eligible queues it on the writer's per-thread candidate list, and a
 *          later pass scans and compresses the batch, keeping full buffer
 *          scans off the write path.
 *
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 * @tparam Enabled chunk_config::use_fast_block_compression, i.e. embedded
 *                 compression AND OpenMP; the false specialisation below is all
 *                 no-ops
 */
template <typename T, typename Derived, bool Enabled>
class ChunkFastCompPolicy : public ChunkThreadPolicy<T, Derived>
{
  public:
    // 4th element: the chunk's OWN geometry, captured at enqueue by
    // push_to_compress_list. compress() must scan exactly that many elements,
    // NOT alloc_chunk_count() (the CONSUMING thread's ambient size, may differ).
    using VecTupleChunkpInt2 = std::vector<std::tuple<Chunk<T>*, int, int, ChunkElems>>;

    using Manager = ManagerChunk<Chunk, T>;
    using Base    = ChunkBase<T, Derived>;

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

  private:
    constexpr static T zero{};

    /**
     * @brief Append this chunk to @p ll_compt if all @p n_elems elements are zero.
     * @param ll_compt  out: list of compressible chunks
     * @param num_chunk this chunk's index in its array
     * @param num_tab   its array's tab index
     * @param n_elems   THIS chunk's element count -- never the ambient size, see
     *                  TChunk.h's note on the field it is carried in
     */
    void fill_list_to_compress(this Derived&       self,
                               VecTupleChunkpInt2& ll_compt,
                               int                 num_chunk,
                               int                 num_tab,
                               unsigned int        n_elems);

    /**
     * @brief Scan @p thr's queued candidates and collect those now compressible.
     * @param thr      rank whose candidate list is scanned
     * @param ll_compt out: the chunks found compressible, with their geometry
     */
    static void fill_list_chunk_to_compress_in_par(int thr, VecTupleChunkpInt2& ll_compt);
};

/**
 * @brief No-op specialisation: no candidate queue outside fast block compression.
 * @details Every member is a stub, because the call sites are shared with the
 *          enabled path and guarded only by `if constexpr`.
 * @tparam T       stored element type
 * @tparam Derived CRTP derived chunk type (Chunk<T>)
 */
template <typename T, typename Derived>
class ChunkFastCompPolicy<T, Derived, false> : public ChunkThreadPolicy<T, Derived>
{
  public:
    // 4th element: the chunk's own geometry (see the enabled policy).
    using VecTupleChunkpInt2 = std::vector<std::tuple<Chunk<T>*, int, int, ChunkElems>>;

    /// @brief No block compression in this mode.
    static void make_block_compression(int /*thr*/, const Proxy<T>* /*proxy*/) {}

    /// No candidate lists exist in this mode, so there is never anything queued.
    static bool have_chunk_to_compress(int /*max_thr*/)
    {
        return false;
    }

  private:
    /// @brief No candidate list to fill in this mode.
    static void fill_list_chunk_to_compress_in_par(int /*thr*/, VecTupleChunkpInt2& /*ll_compt*/) {}
    /// @copydoc fill_list_chunk_to_compress_in_par
    static void fill_list_to_compress(VecTupleChunkpInt2& /*ll_compt*/,
                                      int /*num_chunk*/,
                                      int /*num_tab*/,
                                      unsigned int /*n_elems*/)
    {
    }
};
/**
 * @brief One chunk of a ChunkArray: a block of chunk-size elements that collapses to zero.
 *
 * @details The simplest of the three variants. A chunk holds either a real
 *          buffer or, once every element is zero, the shared cst_val1 constant
 *          -- so `values` doubles as the state flag: written() is just
 *          "values != cst_val1". zero_one_value and zero_any extend this with a second
 *          constant, shared or per-chunk respectively.
 *
 *          Three policies supply what depends on the build, so this class holds
 *          only what every mode needs:
 *          - ChunkCounterPolicy -- the eager zero-counter trigger (serial)
 *          - ChunkFastCompPolicy -- the batched candidate queue (OpenMP)
 *          - ChunkThreadPolicy -- per-thread allocation, CAS and clearing
 *
 * @tparam T stored element type
 * @note The chunk does NOT know its own element count: the geometry lives in
 *       the owning array's size_policy_, and is passed in by callers. Reading a
 *       buffer with the wrong count is the failure mode the candidate lists
 *       guard against by carrying it -- see TChunk.h.
 */
template <typename T>
// no move operations on purpose -- a move falls back to
// the copy, which carries the per-chunk state the way the copy constructor documents.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Chunk : private ChunkCounterPolicy<chunk_config::use_compression_with_counter>,
              public ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>
{
  public:
    friend class Proxy<T>;

    using Base      = ChunkBase<T, Chunk>;
    using ChunkType = Chunk;

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using Manager = ManagerChunk<Chunk, T>;

    using FastCompPolicy = ChunkFastCompPolicy<T, Chunk<T>, chunk_config::use_fast_block_compression>;

    using Base::allocate_chunk_internal;
    using Base::first_value;

    using Base::init_block;
    using Base::is_cst_val1;
    using Base::m_ptr_cst_val1;
    using Base::values_;
    using Base::values_load;
    using Base::values_store;

    using Base::free_values;

    // Statistics variant tag (CHUNK_STATS): 0 = chunk.
    static constexpr int stat_variant = 0;

    Chunk()  = default;
    ~Chunk() = default;

    /// @brief Copy the zero-counter base and the `values` pointer of @p other.
    ///
    /// The ChunkCounterPolicy base must be copied explicitly: a user-defined
    /// copy ctor otherwise DEFAULT-constructs it, resetting nb_val_eq_zero_, and
    /// manage_full_zero() later re-compresses a chunk that is not all-zero.
    /// ChunkFastCompPolicy holds the atomic `values` (non-copyable), copied by
    /// hand below.
    Chunk(const Chunk& other);

    /// @brief Copy the zero-counter base and the `values` pointer of @p other unless it is this chunk.
    /// @return *this.
    Chunk& operator=(const Chunk<T>& other);

    /// @brief Point this chunk at the shared zero constant (compressed state).
    void init_values_cst_val(this auto& self)
    {
        self.values_ = Base::get_cst_val1();
    }

    /// @brief Does this chunk hold a real buffer?
    /// @return false while it is compressed to cst_val1
    [[nodiscard]] bool written(this const auto& self) noexcept
    {
        return self.values_ != Base::get_cst_val1();
    }

    /// @brief Materialise storage if the chunk is still compressed.
    void alloc_chunk(this auto& self)
    {
        if (self.is_cst_val1())
        {
            self.create_value();
        }
    }

    /// @brief Materialise storage if still compressed, from @p thr_rank's pool.
    /// @param thr_rank calling thread's rank; ignored in serial builds
    void alloc_chunk(this auto& self, [[maybe_unused]] int thr_rank);

    /// @brief Get reference to the chunk manager (static access)
    static auto& get_manager() noexcept;

    /**
     * @brief Drain and compress every thread's candidate list.
     * @param apply true: compress the candidates found constant; false: walk
     *        the lists only (zero ignores it: its pass always compresses)
     */
    static void compress_chunk_to_compress(bool apply);

    /// @brief One rank's share of compress_chunk_to_compress(): the body its
    ///        region runs on each thread. Public for AllTypeChunk's single
    ///        sweep region (ManagerChunk::sweep_rank).
    /// @param thr rank whose candidate list to drain
    static void compress_sweep_rank(int thr);

  private:
    // compress(int thr_rank, unsigned n_elems) lives in ChunkThreadPolicy (needs a rank).
    constexpr static T zero{};

    /**
     * @brief Release this chunk's buffer and return to the compressed state.
     * @param thr_rank owning thread's rank; unused serially
     * @note The release counter differs by mode (pool vs free) -- see the
     *       definition's comment.
     */
    void block_clear(this auto& self, int thr_rank);

    static inline constinit internal::LazyImmortal<Manager> s_manager_;
};

} // namespace zero

} // namespace shrink_array

#include "chunk/internal/zero_chunk_inline.h" // the template bodies
