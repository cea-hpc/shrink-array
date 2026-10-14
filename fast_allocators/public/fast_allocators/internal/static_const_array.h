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
 * @file static_const_array.h
 * @brief A ladder of fixed-size buckets, for allocations too small to deserve a
 *        slab allocator of their own.
 *
 * ::StaticConstArray holds fifteen ::TCTypeAllocatorEBC buckets, 4 bytes to
 * 65536, and routes a request to the smallest one that fits. A larger request
 * is not an error: the lookup returns null and the caller falls back to malloc,
 * hence the free path's bool -- "the ladder took it" or "it was never mine".
 *
 * @par Who uses it
 * ::TypeAllocator, hence TCAllocator's `operator new` (slab HEADER objects, not
 * payloads; serial builds only, tc_allocator.cpp), and ::BlkAllocator, the entry
 * point for code outside the library.
 *
 * @par Threading
 * A bucket is not thread-safe: every ladder access inside a parallel region
 * holds ladder_mutex(). To keep frees from serialising, a freed block is
 * batched in a per-thread ::InfoFreeElt, released at once when the batch fills.
 *
 * Allocation and free hold the same mutex, so they exclude each other. It is
 * the ladder's own (it replaced `omp critical(fast_alloc_ladder)`, threads
 * branch step 2): never shared with a lock of the caller's, and not reentrant
 * -- nothing that runs under it may allocate from or free to the ladder.
 *
 * @par Configuration
 * The macros are read in alloc_config.h; the bodies branch with `if constexpr`.
 * What still has to be a directive, and why, is written at each site.
 */

#include <array>   // std::array
#include <bit>     // std::bit_width (bucket_index)
#include <cassert> // TypeAllocator::alloc: the size its free will route by
#include <cstddef> // std::size_t
#include <mutex>   // std::mutex (ladder_mutex)
#include <tuple>   // std::tuple, std::get, std::apply (the ladder)

#include "fast_allocators/internal/svalloc.h" // fast_alloc::alignment
#include "fast_allocators/internal/pmp_alloc.h" // FAST_ALLOC_PMP_MAX_THREADS / FAST_ALLOC_PMP_THREAD_RANK / FAST_ALLOC_PMP_IN_PARALLEL_REGION

#include "fast_allocators/internal/int_alloc.h"    // int_alloc, int_free, int_alloc_align, int_free_align
#include "fast_allocators/internal/alloc_config.h" // alloc_config::use_std_alloc
#include "fast_allocators/tc_allocator.h"          // TCTypeAllocatorEBC

namespace fast_alloc
{

/**
 * @brief A per-thread batch of blocks freed but not yet returned to the ladder.
 *
 * @details A free inside a parallel region only records the block here; the
 *          batch is handed back when it fills -- one critical section instead
 *          of one per free.
 *
 * @note One instance exists, StaticConstArray::tab_info_free_elt().
 *
 * @warning A batch that is still filling holds blocks that are free but not
 *          reusable: they belong to no bucket's free list until the flush.
 *
 * @warning The per-rank arrays are sized ONCE (#nb_thr = FAST_ALLOC_PMP_MAX_THREADS,
 *          before main) and indexed unchecked: callers must bound the rank
 *          (freeBlock() frees a rank >= #nb_thr's blocks at once).
 */
class InfoFreeElt
{
  private:
    /// @brief Batch capacity, in blocks. The other bound is 4 MiB of payload.
    static constexpr int max_free_elt = 10000;

    int* somme_sz{nullptr}; ///< Per rank: bytes batched, bounded at 4 MiB.

  public:
    using IntMaxFreeElt  = std::array<int, max_free_elt>;   ///< One rank's batched sizes.
    using VoidMaxFreeElt = std::array<void*, max_free_elt>; ///< One rank's batched blocks.

    /// @brief Allocates the four per-thread arrays, sized by FAST_ALLOC_PMP_MAX_THREADS.
    ///        A serial build allocates nothing (pointers stay null, never reached).
    InfoFreeElt();

    /// @brief Frees them. Does NOT release blocks still held in a batch.
    ~InfoFreeElt();

    // Owns the four arrays: a copy would free them twice.
    InfoFreeElt(const InfoFreeElt&)            = delete;
    InfoFreeElt& operator=(const InfoFreeElt&) = delete;
    InfoFreeElt(InfoFreeElt&&)                 = delete;
    InfoFreeElt& operator=(InfoFreeElt&&)      = delete;

    /**
     * @brief Record one freed block, if the batch still has room.
     * @param ptr The block.
     * @param sz Its size, needed to route it to the right bucket at flush time.
     * @param thr_rank The calling thread's rank.
     * @return TRUE when the batch is FULL and @p ptr was NOT recorded -- the
     *         caller must then free @p ptr itself and flush the batch. False
     *         means it was taken.
     * @note The return reads "did this overflow", not "did this succeed".
     *
     * @note NOT const, like reset(): both mutate the batch, through member
     *       pointers, which is the only reason const would compile.
     */
    bool push_back(void* ptr, int sz, int thr_rank)
    {
        int& ref_compt = compt[thr_rank];

        if ((ref_compt < max_free_elt && somme_sz[thr_rank] < 4 * 1024 * 1024))
        {
            tab_ptr[thr_rank][ref_compt] = ptr;
            tab_sz[thr_rank][ref_compt]  = sz;

            somme_sz[thr_rank] += sz;

            ref_compt++;
            return false;
        }

        return true;
    }

    /// @brief Forget the batch. Call only after its blocks have been released.
    void reset(int thr_rank) noexcept
    {
        somme_sz[thr_rank] = compt[thr_rank] = 0;
    }

    VoidMaxFreeElt* tab_ptr{nullptr}; ///< Per rank: the batched blocks.
    IntMaxFreeElt*  tab_sz{nullptr};  ///< Per rank: their sizes.
    int*            compt{nullptr};   ///< Per rank: how many are batched.

    /// @brief Ranks the four arrays were sized for (FAST_ALLOC_PMP_MAX_THREADS at
    ///        construction); 0 in a serial build, which allocates nothing.
    int nb_thr = 0;
};

/**
 * @brief The bucket ladder itself, and the entry points that walk it.
 *
 * @details All static: one ladder per process. The class scopes the buckets and
 *          is the base of the front ends (::BlkAllocator by size,
 *          ::FixedSizeAllocator by type).
 *
 * @note The `*Int` methods are the bare lookup; the others add the size bound
 *       and, in a parallel region, the critical section. Call the outer ones.
 */
class StaticConstArray
{
  public:
    // Per-thread batching of freed blocks: OpenMP free path only, declared in
    // every build so that body type-checks serially.

    /// @brief Return thread @p thr_rank's batched blocks to the ladder.
    /// @warning Does NOT reset the batch; the caller must call
    ///          InfoFreeElt::reset() or the same blocks are freed twice.
    static void clearInfoFreeElt(int thr_rank);

    /// @brief Flush every thread's batch and reset it.
    /// @warning Quiescent point only: no rank may be freeing meanwhile (no
    ///          ladder_mutex() here). Called by the chunk library's OMPT
    ///          parallel_end callback, so a region's parked blocks do not
    ///          outlive it.
    static void clearInfoFreeElt();

    /// @brief The ladder's buckets, smallest first: element sizes 4 << i bytes.
    using Ladder = std::tuple<TCTypeAllocatorEBC<4, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<8, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<16, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<32, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<64, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<128, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<256, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<512, SIZE_BLOCK>,
                              TCTypeAllocatorEBC<1024, 2 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<2048, 4 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<4096, 8 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<8192, 16 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<16384, 16 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<32768, 16 * SIZE_BLOCK>,
                              TCTypeAllocatorEBC<65536, 32 * SIZE_BLOCK>>;

    /// @brief How many buckets the ladder has.
    static constexpr std::size_t n_buckets = std::tuple_size_v<Ladder>;

    /// @brief Largest size the ladder serves, in bytes: its top bucket's.
    ///        Past it, getBlockAlloc() returns null and the caller uses malloc.
    /// @note `int`: freeBlock() compares an `int` size with it, the front ends a
    ///       `size_t` one, and neither mixes signedness against a variable.
    static constexpr int max_const_block_size = 65536;

    /**
     * @brief The ladder: one bucket per size class, 4 bytes up to 65536.
     *
     * @details Block size grows with element size so a bucket always holds a
     *          useful number of elements -- 1 MiB up to the 512-byte class,
     *          then doubling, to 32 MiB for the 65536-byte one.
     *
     * @warning IMMORTAL, and the leak is deliberate: see the comment below.
     */
    // IMMORTAL: a never-deleted heap object, like the arenas in slab_arena.h
    // (rationale there). These buckets allocate TCAllocator HEADER objects
    // (TypeAllocator<TCAllocator>). Do not make it a plain static: its exit-time
    // destruction is unordered against other units' statics, e.g. a never-used
    // variant's ChunkBase::allocator_pool_serial_, whose destructor would then
    // walk already-unmapped memory.
    //
    // A function-local static, not a `static inline Ladder& = *new Ladder()`
    // member: that reference is DYNAMICALLY initialised only in the units that
    // include this header, so a global array built from another unit could see
    // a null ladder. The local static is built at first use, whatever the order.
    [[nodiscard]] static Ladder& buckets()
    {
        static auto* const ladder = new Ladder();
        return *ladder;
    }

    /// @brief The bucket getBlockAllocInt() / freeBlockWithInt() use for
    ///        @p sz bytes: the smallest that holds it, the top one past 32768.
    [[nodiscard]] static constexpr std::size_t bucket_index(unsigned int sz) noexcept
    {
        if (sz <= 4)
        {
            return 0;
        }
        // (2^(k-1), 2^k] -> bucket k - 2, the one of 2^k bytes.
        const auto k = static_cast<std::size_t>(std::bit_width(sz - 1));
        return k - 2 < n_buckets ? k - 2 : n_buckets - 1;
    }

  protected:
    /**
     * @brief Take a block of at least @p sz bytes from the ladder.
     * @param sz Requested size in bytes.
     * @return The block, or NULL when @p sz exceeds max_const_block_size --
     *         which is not a failure, it means "too big for the ladder" and the
     *         caller should fall back to malloc.
     * @note Takes the critical section when called inside a parallel region.
     */
    static void* getBlockAlloc(unsigned int sz);

    /**
     * @brief Give a block back to the ladder.
     * @param sz The size it was requested with.
     * @param ptr The block.
     * @return TRUE if the ladder took it; FALSE if @p sz is past the top bucket,
     *         meaning the block came from malloc and the caller must free it.
     * @note Inside a parallel region the block is batched rather than returned
     *       immediately. @see InfoFreeElt
     */
    static bool freeBlock(int sz, void* ptr);

  private:
    /// @brief Bare ladder lookup: smallest bucket that fits @p sz.
    /// @return Null if @p sz is past the top bucket. Not locked, not bounded.
    static void* getBlockAllocInt(unsigned int sz);

    /// @brief Bare ladder release, the counterpart of getBlockAllocInt().
    /// @pre @p ptr came from the bucket that @p sz selects.
    static void freeBlockWithInt(int sz, void* ptr);

    /// @brief Run @p f on bucket @p i of the ladder; its result, if any.
    /// @pre @p i < n_buckets.
    template <class F>
    static decltype(auto) with_bucket(std::size_t i, F&& f);

    /// @brief The per-thread batches. IMMORTAL and built at first use, like the
    ///        ladder: a global array frees through them before main and after
    ///        it returns.
    [[nodiscard]] static InfoFreeElt& tab_info_free_elt()
    {
        static auto* const batches = new InfoFreeElt();
        return *batches;
    }

    /// @brief The lock of the ladder inside a parallel region (see Threading
    ///        at the top of this file). IMMORTAL and built at first use, like
    ///        the ladder it guards, for the same reason.
    [[nodiscard]] static std::mutex& ladder_mutex()
    {
        static auto* const mutex = new std::mutex();
        return *mutex;
    }
};

static_assert(StaticConstArray::bucket_index(StaticConstArray::max_const_block_size) ==
                      StaticConstArray::n_buckets - 1 &&
                  StaticConstArray::bucket_index(StaticConstArray::max_const_block_size / 2) ==
                      StaticConstArray::n_buckets - 2,
              "max_const_block_size must be the size of the ladder's top bucket");

/**
 * @brief Ladder front end that falls back to malloc, by size. PUBLIC API.
 * @details Try a bucket; past the top of the ladder, int_alloc()/int_free().
 *          free_block() works out which from the size again.
 *
 *          Two pairs, never mixed: alloc_block()/free_block(size) when the
 *          caller knows the size at free time, alloc_tab()/free_tab() when it
 *          does not (a `fast_alloc::alignment`-byte header keeps it as a
 *          size_t). Every size is compared with the ceiling in size_t, before
 *          any narrowing.
 *          Thread-safety: within ONE OpenMP team only, no nesting, no other
 *          threads -- a thread is its omp_get_thread_num() (two teams share
 *          ranks, hence batches), and outside a region no lock is taken.
 *
 *          Not called by shrink-array: the entry point for outside code,
 *          exercised by blk_allocator_validation.
 */
struct BlkAllocator : public StaticConstArray
{
    /// @brief A block of @p size bytes, from the ladder or from malloc.
    static void* alloc_block(size_t size)
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            return int_alloc(size);
        }
        else
        {
            // Compared in size_t BEFORE narrowing to unsigned int (4 GiB + 64
            // would become 64).
            void* ptr = size <= max_const_block_size ? getBlockAlloc(static_cast<unsigned int>(size)) : nullptr;

            if (ptr == nullptr)
            {
                ptr = int_alloc(size);
            }

            return ptr;
        }
    }

    /// @brief Give back a block from alloc_block().
    /// @param ptr  The block; null is a no-op.
    /// @param size Must be the size it was allocated with -- that is what decides
    ///        whether it goes to the ladder or to int_free().
    static void free_block(void* ptr, size_t size)
    {
        if (ptr == nullptr)
        {
            return;
        }

        if constexpr (alloc_config::use_std_alloc)
        {
            int_free(ptr);
        }
        else
        {
            // false: not a ladder size, so it came from int_alloc(). Test in
            // size_t first: static_cast<int> alone turns [2 GiB, 4 GiB) negative.
            if (size <= max_const_block_size && freeBlock(static_cast<int>(size), ptr))
            {
                return;
            }

            int_free(ptr);
        }
    }

    /**
     * @brief A block of @p size bytes that remembers its own size.
     * @return A pointer past a header of `fast_alloc::alignment` bytes holding @p size,
     *         so free_tab() needs no size argument. The alignment padding is
     *         what makes the returned pointer suitably aligned.
     */
    static void* alloc_tab(size_t size)
    {
        // size_t header: an int one would wrap past INT_MAX.
        static_assert(fast_alloc::alignment >= sizeof(std::size_t), "alloc_tab: the header must hold a size_t");
        // size + header must not wrap: near SIZE_MAX it would route to a small
        // ladder block that the caller then writes past.
        if (size > static_cast<std::size_t>(-1) - fast_alloc::alignment) [[unlikely]]
        {
            fast_alloc::out_of_memory("alloc_tab", size, 1, 0);
        }
        const std::size_t total = size + fast_alloc::alignment;

        auto* ptr = static_cast<std::size_t*>(
            total <= max_const_block_size ? getBlockAlloc(static_cast<unsigned int>(total)) : nullptr);

        if (ptr == nullptr)
        {
            ptr = static_cast<std::size_t*>(int_alloc_align(total));
        }

        ptr[0] = size;

        return reinterpret_cast<char*>(ptr) + fast_alloc::alignment;
    }

    /// @brief Give back an alloc_tab() block, reading its size from its header.
    static void free_tab(void* user_ptr)
    {
        if (user_ptr == nullptr)
        {
            return;
        }

        auto* m_ptr = reinterpret_cast<std::size_t*>(static_cast<char*>(user_ptr) - fast_alloc::alignment);

        const std::size_t total = m_ptr[0] + fast_alloc::alignment;

        // Same routing as alloc_tab(), decided in size_t.
        if (total > max_const_block_size || !freeBlock(static_cast<int>(total), m_ptr))
        {
            int_free_align(m_ptr);
        }
    }
};

/**
 * @brief Ladder front end for a fixed element TYPE.
 *
 * @tparam ElemT The type whose `sizeof` is the block size.
 *
 * @details Same fallback as ::BlkAllocator. The size is a constant, so
 *          bucket_index() folds.
 *
 *          The library's one instantiation is TypeAllocator<TCAllocator> (slab
 *          headers, serial builds only); chunks and their buffers never come here.
 */
template <typename ElemT>
class FixedSizeAllocator : private StaticConstArray
{
  public:
    /// @brief One block of `sizeof(ElemT)` bytes.
    static void* alloc_block()
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            return int_alloc(sizeof(ElemT));
        }
        else
        {
            void* ptr = getBlockAlloc(static_cast<unsigned int>(sizeof(ElemT)));

            if (ptr == nullptr)
            {
                return int_alloc(sizeof(ElemT));
            }

            return ptr;
        }
    }

    /// @brief Give back a block from alloc_block().
    static void free_block(void* ptr)
    {
        if (ptr == nullptr)
        {
            return;
        }

        if constexpr (alloc_config::use_std_alloc)
        {
            int_free(ptr);
        }
        else
        {
            // freeBlock() returns false for a size the ladder does not serve:
            // the block came from int_alloc().
            if (freeBlock(static_cast<int>(sizeof(ElemT)), ptr))
            {
                return;
            }

            int_free(ptr);
        }
    }
};

/**
 * @brief The name the rest of the library allocates small objects through.
 *
 * @tparam T The object type. It selects the ::FixedSizeAllocator instantiation;
 *         the size actually comes from the argument.
 *
 * @details Lets TCAllocator::operator new (tc_allocator.cpp, serial builds) use
 *          the ladder without tc_allocator.h knowing it.
 */
template <typename T>
struct TypeAllocator
{
    /// @brief A block of @p size bytes for a T.
    /// @pre `size == sizeof(T)`: free() routes by sizeof(T), so any other size
    ///      (an operator new reached by a larger derived class) would go back
    ///      to the wrong bucket.
    static void* alloc([[maybe_unused]] size_t size)
    {
        assert(size == sizeof(T) && "TypeAllocator: free() routes by sizeof(T)");
        return FixedSizeAllocator<T>::alloc_block();
    }
    /// @brief Give it back.
    /// @warning Routes by `sizeof(T)`. @see FixedSizeAllocator::free_block
    static void free(void* ptr)
    {
        FixedSizeAllocator<T>::free_block(ptr);
    }
};

/**
 * @brief Total what every allocator in the process is holding.
 * @param sum_UsedSize Bytes ever carved out of a block.
 * @param sum_nn_ref Bytes currently handed out.
 * @param alloc_sz Bytes held in blocks, whether handed out or not.
 * @details Walks the ladder, then the registry of ::TCTypeAllocator instances.
 * @note Only a serial build fills the registry, so an OpenMP build reports
 *       the ladder alone.
 */
void compute_sum_allocated_size_TCAlloc(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz);

} // namespace fast_alloc

#include "fast_allocators/internal/static_const_array_inline.h" // the template bodies
