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
 * 65536, and routes a request to the smallest one that fits. A request larger
 * than the top bucket is not an error: the lookup returns null and the caller
 * falls back to plain malloc. That two-step is why the free path returns a
 * bool -- "the ladder took it" or "it was never mine".
 *
 * @par Who uses it
 * ::TypeAllocator, and through it TCAllocator's own `operator new` -- so what
 * the library allocates here is the slab HEADER objects of the other
 * allocators, not the payloads they hand out; that path exists in serial
 * builds only (tc_allocator.cpp). And ::BlkAllocator, the entry point for code
 * outside the library.
 *
 * @par Threading
 * A bucket is not thread-safe, so every ladder access from inside a parallel
 * region is wrapped in `omp critical`. To keep that from serialising the free
 * path, a freed block is instead batched in a per-thread ::InfoFreeElt and the
 * whole batch is released at once when it fills.
 *
 * The critical sections are NAMED (fast_alloc_ladder), all the same name, so
 * that an allocation and a free still exclude each other. Unnamed, they were
 * the one process-wide lock every unnamed `omp critical` shares: a caller
 * reaching the allocator from inside its own unnamed critical deadlocked (a
 * critical is not reentrant), and every unrelated critical section in the
 * program waited on the ladder and the ladder on it.
 *
 * @par Configuration
 * The macros are read in alloc_config.h; the bodies branch with `if constexpr`.
 * What still has to be a directive, and why, is written at each site.
 *
 * SIZE_BLOCK is tc_allocator.h's `inline constexpr unsigned int`;
 * static_const_array_define.h only keeps MAX_CONST_BLOCK_SIZE.
 */

#include <array>       // std::array
#include <bit>         // std::bit_width (bucket_index)
#include <cassert>     // TypeAllocator::alloc: the size its free will route by
#include <cstddef>     // std::size_t
#include <tuple>       // std::tuple, std::get, std::apply (the ladder)
#include <type_traits> // std::is_same_v (Allocator8 alias check)

#include "fast_allocators/internal/svalloc.h"   // fast_alloc::alignment
#include "fast_allocators/internal/pmp_alloc.h" // PMP_MAX_THREADS / PMP_THREAD_RANK / PMP_IN_PARALLEL_REGION

#include "fast_allocators/internal/int_alloc.h"                 // int_alloc, int_free, int_alloc_align, int_free_align
#include "fast_allocators/internal/alloc_config.h"              // constexpr mirror of STD_ALLOC
#include "fast_allocators/internal/static_const_array_define.h" // MAX_CONST_BLOCK_SIZE (BlkAllocator routing)
#include "fast_allocators/tc_allocator.h"                       // TCTypeAllocatorEBC

/**
 * @brief A per-thread batch of blocks freed but not yet returned to the ladder.
 *
 * @details Returning a block to its bucket needs the `omp critical` section,
 *          which every thread contends for. So a free inside a parallel region
 *          only records the block here, and the whole batch is handed back at
 *          once when it fills -- one critical section instead of one per free.
 *
 * @note One instance exists, StaticConstArray::tab_info_free_elt.
 *
 * @warning A batch that is still filling holds blocks that are free but not
 *          reusable: they belong to no bucket's free list until the flush.
 *
 * @warning The per-rank arrays are sized ONCE, by PMP_MAX_THREADS when the
 *          static instances are built (before main): #nb_thr ranks. Every
 *          method indexes by rank unchecked, so callers must bound it:
 *          freeBlock() does not batch a rank >= #nb_thr (a team grown later)
 *          and frees its blocks at once instead.
 */
class InfoFreeElt
{
  private:
    /// @brief Batch capacity, in blocks. The other bound is 4 MiB of payload.
    static constexpr int max_free_elt = 10000;

    int* somme_sz{nullptr}; ///< Per rank: bytes batched, bounded at 4 MiB.

  public:
    using IntMaxFreeElt  = std::array<int, max_free_elt>; ///< One rank's batched sizes.
    using VoidMaxFreeElt = void* [max_free_elt];          ///< One rank's batched blocks.

    /// @brief Allocates the four per-thread arrays, sized by PMP_MAX_THREADS.
    ///        A serial build has no batching to do and allocates nothing (the
    ///        pointers stay null; the batching path is never reached there).
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
     * @note The return is inverted from what the name suggests. It reads as
     *       "did this overflow", not "did this succeed".
     *
     * @note NOT const, and its sibling reset() is the reason: both mutate the
     *       batch, both do it through member POINTERS, and const would only be
     *       accepted here because of that indirection. A const method that
     *       changes what the object means is worse than an honest non-const one.
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

    /// @brief Ranks the four arrays were sized for (PMP_MAX_THREADS at
    ///        construction); 0 in a serial build, which allocates nothing.
    int nb_thr = 0;
};

/**
 * @brief The bucket ladder itself, and the entry points that walk it.
 *
 * @details Everything is static -- there is one ladder per process, not one per
 *          instance. The class exists to scope the buckets and to be a private
 *          base for the front ends (::BlkAllocator, public API by size, and
 *          ::FixedSizeAllocator, by type).
 *
 * @note The `*Int` methods are the bare lookup: pick a bucket by size and call
 *       it. The methods without the suffix wrap that in the size bound and, in
 *       a parallel region, the critical section. Call the outer ones.
 */
class StaticConstArray
{
  public:
    // Per-thread batching of freed blocks. Only the OpenMP free path uses it
    // (freeBlock, under alloc_config::use_openmp); the declarations exist in
    // every build so that body type-checks serially.

    /// @brief Return thread @p thr_rank's batched blocks to the ladder.
    /// @warning Does NOT reset the batch; the caller must call
    ///          InfoFreeElt::reset() or the same blocks are freed twice.
    static void clearInfoFreeElt(int thr_rank);

    /// @brief Flush every thread's batch and reset it.
    /// @warning Quiescent point only: no rank may be freeing meanwhile, since
    ///          this runs without the overflow path's `omp critical`. Called
    ///          by the chunk library's OMPT parallel_end callback, outside any
    ///          active region, so that a region's parked blocks do not outlive
    ///          it.
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

    /**
     * @brief The ladder: one bucket per size class, 4 bytes up to 65536.
     *
     * @details Block size grows with element size so a bucket always holds a
     *          useful number of elements -- 1 MiB up to the 512-byte class,
     *          then doubling, to 32 MiB for the 65536-byte one.
     *
     * @warning IMMORTAL, and the leak is deliberate: see the comment below.
     */
    // IMMORTAL: bound to a heap-allocated, never-deleted object, exactly like
    // op_arena()/ctx_arena()/value_arena() in slab_arena.h (see that file's
    // comment for the full rationale). These buckets back TCAllocator's own
    // custom operator new/delete (TypeAllocator<TCAllocator>::alloc/free) --
    // i.e. they allocate the TCAllocator HEADER objects themselves, not just
    // chunk payloads. A directly-embedded static (the previous form) is
    // destroyed at program exit in an order that is UNSPECIFIED relative to
    // other translation units' static inline members -- including
    // ChunkBase<T,Derived>::allocator_pool_serial_ for a variant (e.g.
    // Chunk2A<double> in a CHUNK2-only build) that is never actively used and
    // so never explicitly cleared during main(): if the relevant bucket here
    // is destroyed first, that map's own destructor later reads
    // already-unmapped memory walking its block/block_full lists --
    // observed as SIGSEGV / "Cannot access memory" in
    // TCTypeAllocator<T>::destroy_int() at static teardown, well after
    // main() returns cleanly with every in-run integrity check passing.
    // A reference has no destructor of its own to run, so leaking the
    // pointee here is intentional and harmless, matching the arena.
    static inline Ladder& buckets = *(new Ladder());

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
     * @return The block, or NULL when @p sz exceeds MAX_CONST_BLOCK_SIZE --
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

    static InfoFreeElt tab_info_free_elt; ///< The per-thread batches.
};

static_assert(std::is_same_v<std::tuple_element_t<1, StaticConstArray::Ladder>, TCTypeAllocatorEBC<8, SIZE_BLOCK>>,
              "Allocator8 aliases the ladder's bucket 1: it must stay the 8-byte one");

/// @brief The 8-byte bucket, the most common size, by name.
/// @details IS the ladder's bucket 1 (sizes 5..8), not a pool of its own:
///          ::FixedSizeAllocator reaches it directly, skipping the size
///          comparisons of the ladder walk. One pool, not a second 8-byte
///          global (with its own 1 MiB slab and an unspecified destruction
///          order): immortal, with the ladder. Same lock either way
///          (critical(fast_alloc_ladder)).
/// @note Inline, defined after StaticConstArray in this header, which every
///       user includes: partially-ordered initialisation sets up `buckets`
///       before this reference binds to it.
inline TCTypeAllocatorEBC<8, SIZE_BLOCK>& Allocator8 = std::get<1>(StaticConstArray::buckets);

/// @brief Standalone bucket for 24-byte blocks (the ladder goes 16 -> 32).
/// @details IMMORTAL, like the ladder and for the same reason: it backs
///          TypeAllocator's 24-byte allocations, which may still be released
///          during static teardown. Leaked on purpose. @see Allocator8
inline TCTypeAllocatorEBC<24, SIZE_BLOCK>& Allocator24 = *(new TCTypeAllocatorEBC<24, SIZE_BLOCK>());

/**
 * @brief Ladder front end that falls back to malloc, by size. PUBLIC API.
 * @details The two-step the ladder's null/false conventions are designed for:
 *          try a bucket, and if the size is past the top of the ladder, use
 *          int_alloc()/int_free() instead. The caller never has to know which
 *          happened -- free_block() works it out from the size again.
 *
 *          Two pairs, never mixed: alloc_block()/free_block(size) when the
 *          caller knows the size at free time, alloc_tab()/free_tab() when it
 *          does not (a header of `fast_alloc::alignment` bytes keeps it as a
 *          size_t). Every size is compared with the ladder's ceiling in size_t,
 *          before any narrowing to the ladder's int / unsigned int.
 *          Thread-safe: inside an OpenMP region the ladder takes its critical
 *          section and frees are batched per thread until
 *          StaticConstArray::clearInfoFreeElt() runs at a quiescent point.
 *          Within ONE OpenMP team, no nesting, no other threads: a thread is
 *          its omp_get_thread_num() (two teams, or a nested one, share ranks
 *          and so batches), and outside a region no lock is taken (a
 *          std::thread allocating then races a team under its critical).
 *
 *          Nothing in shrink-array calls it: it is fast_allocators' entry point
 *          for code outside the library, exercised by blk_allocator_validation.
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
            // Compared in size_t BEFORE narrowing: getBlockAlloc() takes an
            // unsigned int, and 4 GiB + 64 would reach it as 64 -- a 64-byte
            // block for a 4 GiB request.
            void* ptr = size <= MAX_CONST_BLOCK_SIZE ? getBlockAlloc(static_cast<unsigned int>(size)) : nullptr;

            // getBlockAlloc() returns null for a size past the largest bucket;
            // that is not a failure, it is "too big for the ladder".
            if (!ptr)
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
            // freeBlock() returns false for a size the ladder does not serve --
            // it came from int_alloc(), so it goes back to int_free().
            // Same size_t test as alloc_block(): static_cast<int> alone made
            // [2 GiB, 4 GiB) negative, hence "<= MAX_CONST_BLOCK_SIZE", and a
            // malloc'd block went to the ladder at an out-of-range bucket.
            if (size <= MAX_CONST_BLOCK_SIZE && freeBlock(static_cast<int>(size), ptr))
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
        // size_t header: an int one silently wrapped past INT_MAX, and
        // free_tab() then routed the block by a garbage size.
        static_assert(fast_alloc::alignment >= sizeof(std::size_t), "alloc_tab: the header must hold a size_t");
        // size + header must not wrap: near SIZE_MAX it would route to a small
        // ladder block that the caller then writes past.
        if (size > static_cast<std::size_t>(-1) - fast_alloc::alignment) [[unlikely]]
        {
            fast_alloc::out_of_memory("alloc_tab", size, 1, 0);
        }
        const std::size_t total = size + fast_alloc::alignment;

        auto* ptr = static_cast<std::size_t*>(
            total <= MAX_CONST_BLOCK_SIZE ? getBlockAlloc(static_cast<unsigned int>(total)) : nullptr);

        if (!ptr)
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
        if (total > MAX_CONST_BLOCK_SIZE || !freeBlock(static_cast<int>(total), m_ptr))
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
 * @details Same fallback as ::BlkAllocator (a bucket, malloc past the top of
 *          the ladder), with a shortcut: 8 and 24 bytes are
 *          by far the most common sizes here, so they go straight to
 *          ::Allocator8 / ::Allocator24 instead of walking the ladder.
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
            if (sizeof(ElemT) == 8)
            {
                return shortcut_alloc(Allocator8);
            }
            if (sizeof(ElemT) == 24)
            {
                return shortcut_alloc(Allocator24);
            }

            void* ptr = getBlockAlloc(sizeof(ElemT));

            if (!ptr)
            {
                return int_alloc(sizeof(ElemT));
            }

            return ptr;
        }
    }

    /// @brief One block of @p size bytes, ignoring `sizeof(ElemT)`.
    /// @note Used by ::TypeAllocator, where the caller knows the size and the
    ///       type parameter only selects the pool.
    static void* alloc_block(size_t size)
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            return int_alloc(size);
        }
        else
        {
            if (size == 8)
            {
                return shortcut_alloc(Allocator8);
            }
            if (size == 24)
            {
                return shortcut_alloc(Allocator24);
            }

            // size_t test before narrowing, as BlkAllocator::alloc_block().
            void* ptr = size <= MAX_CONST_BLOCK_SIZE ? getBlockAlloc(static_cast<unsigned int>(size)) : nullptr;

            if (!ptr)
            {
                return int_alloc(size);
            }

            return ptr;
        }
    }

    /// @brief Give back a block from either alloc_block() overload.
    /// @warning Routes by `sizeof(ElemT)`, NOT by the size the block was
    ///          actually taken with. A block from `alloc_block(size)` with
    ///          `size != sizeof(ElemT)` goes back to the wrong bucket;
    ///          TypeAllocator::alloc() asserts it never is.
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
            if (sizeof(ElemT) == 8)
            {
                shortcut_free(Allocator8, ptr);
            }
            else if (sizeof(ElemT) == 24)
            {
                shortcut_free(Allocator24, ptr);
            }
            else
            {
                if (freeBlock(sizeof(ElemT), ptr))
                {
                    return;
                }

                int_free(ptr);
            }
        }
    }

  private:
    /// @brief Allocator8 / Allocator24 are plain buckets, no more thread-safe
    ///        than the ladder's: same guard as getBlockAlloc(). Serially
    ///        PMP_IN_PARALLEL_REGION is the constant false and the branch folds
    ///        away.
    template <typename Bucket>
    static void* shortcut_alloc(Bucket& bucket)
    {
        if (PMP_IN_PARALLEL_REGION)
        {
            void* ptr = nullptr;
#pragma omp critical(fast_alloc_ladder)
            {
                ptr = bucket.alloc();
            }
            return ptr;
        }
        return bucket.alloc();
    }

    /// @brief Counterpart of shortcut_alloc(), same guard.
    template <typename Bucket>
    static void shortcut_free(Bucket& bucket, void* ptr)
    {
        if (PMP_IN_PARALLEL_REGION)
        {
#pragma omp critical(fast_alloc_ladder)
            {
                bucket.free_elt(ptr);
            }
            return;
        }
        bucket.free_elt(ptr);
    }
};

/**
 * @brief The name the rest of the library allocates small objects through.
 *
 * @tparam T The object type. It selects the ::FixedSizeAllocator instantiation;
 *         the size actually comes from the argument.
 *
 * @details What lets TCAllocator::operator new (tc_allocator.cpp, serial
 *          builds) allocate slab headers out of the ladder without
 *          tc_allocator.h knowing the ladder exists.
 */
template <typename T>
struct TypeAllocator
{
    /// @brief A block of @p size bytes for a T.
    /// @pre `size == sizeof(T)`: free() routes by sizeof(T), so any other size
    ///      (an operator new reached by a larger derived class) would go back
    ///      to the wrong bucket.
    static void* alloc(size_t size)
    {
        assert(size == sizeof(T) && "TypeAllocator: free() routes by sizeof(T)");
        return FixedSizeAllocator<T>::alloc_block(size);
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

#include "fast_allocators/internal/static_const_array_inline.h" // the template bodies
