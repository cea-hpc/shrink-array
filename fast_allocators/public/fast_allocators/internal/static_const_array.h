/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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
 * There are two parallel ladders. The plain one serves any size; the aligned
 * one skips the buckets narrower than `__ALIGNEMENT__`, since a 4-byte bucket
 * cannot satisfy a 32-byte alignment.
 *
 * @par Who uses it
 * Only ::TypeAllocator, and through it TCAllocator's own `operator new` -- so
 * what this file allocates is the slab HEADER objects of the other allocators,
 * not the payloads they hand out. That path exists in serial builds only; see
 * the note on TCAllocator::operator new.
 *
 * @par Threading
 * A bucket is not thread-safe, so every ladder access from inside a parallel
 * region is wrapped in `omp critical`. To keep that from serialising the free
 * path, a freed block is instead batched in a per-thread ::InfoFreeElt and the
 * whole batch is released at once when it fills.
 *
 * @par Configuration
 * The macros are read in alloc_config.h; the bodies branch with `if constexpr`.
 * What still has to be a directive, and why, is written at each site.
 *
 * @warning static_const_array_define.h, included below, `#define`s SIZE_BLOCK --
 *          AFTER tc_allocator.h has declared `inline constexpr unsigned int
 *          SIZE_BLOCK` for the same value. From this point on the macro wins,
 *          so the constexpr that file went out of its way to introduce is
 *          shadowed everywhere static_const_array.h has been included. The two
 *          values agree, so nothing misbehaves today.
 */

#include "fast_allocators/internal/svalloc.h"
#include "fast_allocators/internal/pmp_alloc.h" // PMP_MAX_THREADS / PMP_THREAD_RANK / PMP_IN_PARALLEL_REGION

#include "fast_allocators/internal/int_alloc.h"
#include "fast_allocators/internal/alloc_config.h" // constexpr mirror of STD_ALLOC
#include "fast_allocators/tc_allocator.h"
#include "fast_allocators/internal/static_const_array_define.h"

/**
 * @brief A per-thread batch of blocks freed but not yet returned to the ladder.
 *
 * @details Returning a block to its bucket needs the `omp critical` section,
 *          which every thread contends for. So a free inside a parallel region
 *          only records the block here, and the whole batch is handed back at
 *          once when it fills -- one critical section instead of one per free.
 *
 * @note Two instances exist, StaticConstArray::tab_info_free_elt for the plain
 *       ladder and StaticConstArray::tab_info_free_elt_align for the aligned
 *       one, because the two have different free routines.
 *
 * @warning A batch that is still filling holds blocks that are free but not
 *          reusable: they belong to no bucket's free list until the flush.
 */
struct InfoFreeElt
{
    /// @brief Batch capacity, in blocks. The other bound is 4 MiB of payload.
    static constexpr int max_free_elt = 10000;

    using IntMaxFreeElt  = int[max_free_elt];
    using VoidMaxFreeElt = void* [max_free_elt];

    VoidMaxFreeElt* tab_ptr;
    IntMaxFreeElt*  tab_sz;

    int* somme_sz;
    int* compt;

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
    void reset(int thr_rank)
    {
        somme_sz[thr_rank] = compt[thr_rank] = 0;
    }

    /// @brief Allocates the four per-thread arrays, sized by PMP_MAX_THREADS.
    ///        A serial build has no batching to do and allocates nothing (the
    ///        pointers stay null; the batching path is never reached there).
    InfoFreeElt();

    /// @brief Frees them. Does NOT release blocks still held in a batch.
    ~InfoFreeElt();
};

/**
 * @brief The bucket ladder itself, and the entry points that walk it.
 *
 * @details Everything is static -- there is one ladder per process, not one per
 *          instance. The class exists to scope the buckets and to be a private
 *          base for the front ends (::BlkAllocator, ::FixedSizeAllocator).
 *
 * @note The `*Int` methods are the bare lookup: pick a bucket by size and call
 *       it. The methods without the suffix wrap that in the size bound and, in
 *       a parallel region, the critical section. Call the outer ones.
 */
class StaticConstArray
{
  public:
    /**
     * @brief The ladder: one bucket per size class, 4 bytes up to 65536.
     *
     * @details Block size grows with element size so a bucket always holds a
     *          useful number of elements -- 1 MiB up to the 512-byte class,
     *          then doubling, to 32 MiB for the 65536-byte one.
     *
     * @warning IMMORTAL, and the leak is deliberate. See the note below.
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
    static inline TCTypeAllocatorEBC<4, SIZE_BLOCK>&   _Allocator4   = *(new TCTypeAllocatorEBC<4, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<8, SIZE_BLOCK>&   _Allocator8   = *(new TCTypeAllocatorEBC<8, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<16, SIZE_BLOCK>&  _Allocator16  = *(new TCTypeAllocatorEBC<16, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<32, SIZE_BLOCK>&  _Allocator32  = *(new TCTypeAllocatorEBC<32, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<64, SIZE_BLOCK>&  _Allocator64  = *(new TCTypeAllocatorEBC<64, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<128, SIZE_BLOCK>& _Allocator128 = *(new TCTypeAllocatorEBC<128, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<256, SIZE_BLOCK>& _Allocator256 = *(new TCTypeAllocatorEBC<256, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<512, SIZE_BLOCK>& _Allocator512 = *(new TCTypeAllocatorEBC<512, SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<1024, 2 * SIZE_BLOCK>& _Allocator1024 =
        *(new TCTypeAllocatorEBC<1024, 2 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<2048, 4 * SIZE_BLOCK>& _Allocator2048 =
        *(new TCTypeAllocatorEBC<2048, 4 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<4096, 8 * SIZE_BLOCK>& _Allocator4096 =
        *(new TCTypeAllocatorEBC<4096, 8 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<8192, 16 * SIZE_BLOCK>& _Allocator8192 =
        *(new TCTypeAllocatorEBC<8192, 16 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<16384, 16 * SIZE_BLOCK>& _Allocator16384 =
        *(new TCTypeAllocatorEBC<16384, 16 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<32768, 16 * SIZE_BLOCK>& _Allocator32768 =
        *(new TCTypeAllocatorEBC<32768, 16 * SIZE_BLOCK>());
    static inline TCTypeAllocatorEBC<65536, 32 * SIZE_BLOCK>& _Allocator65536 =
        *(new TCTypeAllocatorEBC<65536, 32 * SIZE_BLOCK>());

    // Per-thread batching of freed blocks. Only the OpenMP free path uses it
    // (freeBlock / freeBlockAlign, under alloc_config::use_openmp); the
    // declarations exist in every build so those bodies type-check serially.

    /// @brief Return thread @p thr_rank's batched blocks to the plain ladder.
    /// @warning Does NOT reset the batch; the caller must call
    ///          InfoFreeElt::reset() or the same blocks are freed twice.
    static void clearInfoFreeElt(int thr_rank);

    /// @brief clearInfoFreeElt() for the aligned ladder. Same warning.
    static void clearInfoFreeEltAlign(int thr_rank);

    /// @brief Flush every thread's batch, both ladders.
    /// @warning Nothing calls this. It also clears only `compt` and not
    ///          `somme_sz`, so a batch it emptied still counts as 4 MiB full
    ///          and would never batch again. Fix the reset before using it.
    static void clearInfoFreeElt();

    static InfoFreeElt tab_info_free_elt;       ///< Batches for the plain ladder.
    static InfoFreeElt tab_info_free_elt_align; ///< Batches for the aligned one.

  private:
    /// @brief Bare ladder lookup: smallest bucket that fits @p sz.
    /// @return Null if @p sz is past the top bucket. Not locked, not bounded.
    static inline void* getBlockAllocInt(unsigned int sz);

    /// @brief Bare ladder release, the counterpart of getBlockAllocInt().
    /// @pre @p ptr came from the bucket that @p sz selects.
    static inline void freeBlockWithInt(int sz, void* ptr);

    /// @brief getBlockAllocInt() over the aligned ladder.
    /// @details Skips the buckets narrower than `__ALIGNEMENT__`.
    static inline void* getBlockAllocAlignInt(int sz);

    /// @brief freeBlockWithInt() over the aligned ladder.
    static inline void freeBlockWithAlignInt(int sz, void* ptr);

  public:
    StaticConstArray()  = default;
    ~StaticConstArray() = default;

    /**
     * @brief Take a block of at least @p sz bytes from the ladder.
     * @param sz Requested size in bytes.
     * @return The block, or NULL when @p sz exceeds MAX_CONST_BLOCK_SIZE --
     *         which is not a failure, it means "too big for the ladder" and the
     *         caller should fall back to malloc.
     * @note Takes the critical section when called inside a parallel region.
     */
    static inline void* getBlockAlloc(unsigned int sz);

    /// @brief getBlockAlloc() over the aligned ladder. Same null convention.
    static inline void* getBlockAllocAlign(int sz);

    /**
     * @brief Give a block back to the ladder.
     * @param sz The size it was requested with.
     * @param ptr The block.
     * @return TRUE if the ladder took it; FALSE if @p sz is past the top bucket,
     *         meaning the block came from malloc and the caller must free it.
     * @note Inside a parallel region the block is batched rather than returned
     *       immediately. @see InfoFreeElt
     */
    static inline bool freeBlock(int sz, void* ptr);

    /// @brief freeBlock() over the aligned ladder. Same bool convention.
    static inline bool freeBlockAlign(int sz, void* ptr);

    /// @brief The largest size the ladder serves.
    /// @warning Asserts in a STD_ALLOC build, where there is no ladder to size.
    static inline int computeBlockSize();
};

/// @brief Standalone bucket for 8-byte blocks, the most common size.
/// @details Not part of the ladder: ::FixedSizeAllocator reaches it directly,
///          skipping the size comparisons the ladder walk would cost.
extern TCTypeAllocatorEBC<8, SIZE_BLOCK> Allocator8;

/// @brief Standalone bucket for 24-byte blocks. @see Allocator8
extern TCTypeAllocatorEBC<24, SIZE_BLOCK> Allocator24;

/**
 * @brief Ladder front end that falls back to malloc, by size.
 * @details The two-step the ladder's null/false conventions are designed for:
 *          try a bucket, and if the size is past the top of the ladder, use
 *          int_alloc()/int_free() instead. The caller never has to know which
 *          happened -- free_block() works it out from the size again.
 */
struct BlkAllocator : public StaticConstArray
{
    /// @brief A block of @p s bytes, from the ladder or from malloc.
    static void* alloc_block(size_t s)
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            return int_alloc(s);
        }
        else
        {
            void* ptr = getBlockAlloc(s);

            // getBlockAlloc() returns null for a size past the largest bucket;
            // that is not a failure, it is "too big for the ladder".
            if (!ptr)
            {
                int_alloc((char*&)ptr, s);
                return ptr;
            }

            return ptr;
        }
    }

    /// @brief Give back a block from alloc_block().
    /// @param s Must be the size it was allocated with -- that is what decides
    ///        whether it goes to the ladder or to int_free().
    static void free_block(void* ptr, size_t s)
    {
        if (ptr == nullptr)
            return;

        if constexpr (alloc_config::use_std_alloc)
        {
            int_free(ptr);
        }
        else
        {
            // freeBlock() returns false for a size the ladder does not serve --
            // it came from int_alloc(), so it goes back to int_free().
            if (freeBlock(s, ptr))
            {
                return;
            }

            int_free(ptr);
        }
    }

    /**
     * @brief A block of @p s bytes that remembers its own size.
     * @return A pointer past a header of `__ALIGNEMENT__` bytes holding @p s,
     *         so free_tab() needs no size argument. The alignment padding is
     *         what makes the returned pointer suitably aligned.
     */
    static void* alloc_tab(size_t s)
    {
        int* ptr = (int*)getBlockAlloc(s + __ALIGNEMENT__);

        if (!ptr)
        {
            ptr = (int*)int_alloc_align(s + __ALIGNEMENT__);
        }

        ptr[0] = s;

        char* p = (char*)ptr;

        p += __ALIGNEMENT__;

        return p;
    }

    /// @brief Give back an alloc_tab() block, reading its size from its header.
    static void free_tab(void* p)
    {
        if (p == nullptr)
            return;

        int* m_ptr = (int*)((char*)p - __ALIGNEMENT__);

        const int s = m_ptr[0];

        if (!freeBlock(s + __ALIGNEMENT__, m_ptr))
        {
            int_free_align(m_ptr);
        }
    }
};

/**
 * @brief Ladder front end for a fixed element TYPE.
 *
 * @tparam _ElementType The type whose `sizeof` is the block size.
 *
 * @details Same fallback as ::BlkAllocator, with a shortcut: 8 and 24 bytes are
 *          by far the most common sizes here, so they go straight to
 *          ::Allocator8 / ::Allocator24 instead of walking the ladder.
 */
template <typename _ElementType>
struct FixedSizeAllocator : private StaticConstArray
{
    // 12 and 24 are very common sizes in the code --> specialized
    // for the AutoPtr case, otherwise it's 8 and 24

    /// @brief One block of `sizeof(_ElementType)` bytes.
    static void* alloc_block()
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            size_t sz = sizeof(_ElementType);
            return int_alloc(sz);
        }
        else
        {
            if (sizeof(_ElementType) == 8)
            {
                return Allocator8.alloc();
            }
            if (sizeof(_ElementType) == 24)
            {
                return Allocator24.alloc();
            }

            void* ptr = getBlockAlloc(sizeof(_ElementType));

            if (!ptr)
            {
                return int_alloc(sizeof(_ElementType));
            }

            return ptr;
        }
    }

    /// @brief One block of @p s bytes, ignoring `sizeof(_ElementType)`.
    /// @note Used by ::TypeAllocator, where the caller knows the size and the
    ///       type parameter only selects the pool.
    static void* alloc_block(size_t s)
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            return int_alloc(s);
        }
        else
        {
            if (s == 8)
            {
                return Allocator8.alloc();
            }
            if (s == 24)
            {
                return Allocator24.alloc();
            }

            void* ptr = getBlockAlloc(s);

            if (!ptr)
            {
                return int_alloc(s);
            }

            return ptr;
        }
    }

    /// @brief Give back a block from either alloc_block() overload.
    /// @warning Routes by `sizeof(_ElementType)`, NOT by the size the block was
    ///          actually taken with. A block from `alloc_block(s)` with
    ///          `s != sizeof(_ElementType)` goes back to the wrong bucket.
    static void free_block(void* ptr)
    {
        if (ptr == nullptr)
            return;

        if constexpr (alloc_config::use_std_alloc)
        {
            int_free(ptr);
        }
        else
        {
            if (sizeof(_ElementType) == 8)
            {
                Allocator8.free_elt(ptr);
            }
            else if (sizeof(_ElementType) == 24)
            {
                Allocator24.free_elt(ptr);
            }
            else
            {
                if (freeBlock(sizeof(_ElementType), ptr))
                {
                    return;
                }

                int_free(ptr);
            }
        }
    }
    //
    //    static inline void* alloc_tab(size_t s)
    //    {
    //        int* ptr = (int*)getBlockAlloc(s + __ALIGNEMENT__);
    //
    //        if (!ptr)
    //        {
    //            ptr = (int*)int_alloc_align(s + __ALIGNEMENT__);
    //        }
    //
    //        ptr[0] = s;
    //
    //        char* p = (char*)ptr;
    //
    //        p += __ALIGNEMENT__;
    //
    //        return p;
    //    }
    //
    //    static inline void free_tab(void* p)
    //    {
    //        if (p == nullptr)
    //            return;
    //
    //        int* m_ptr = (int*)((char*)p - __ALIGNEMENT__);
    //
    //        int s = m_ptr[0];
    //
    //        if (!freeBlock(s + __ALIGNEMENT__, m_ptr))
    //        {
    //            int_free_align(m_ptr);
    //        }
    //    }
};

/**
 * @brief The name the rest of the library allocates small objects through.
 *
 * @tparam T The object type. It selects the ::FixedSizeAllocator instantiation;
 *         the size actually comes from the argument.
 *
 * @details Declared in tc_allocator.h and defined here, which is what lets
 *          TCAllocator::operator new allocate slab headers out of the ladder
 *          without tc_allocator.h knowing the ladder exists.
 */
template <typename T>
struct TypeAllocator
{
    /// @brief A block of @p s bytes for a T.
    static void* alloc(size_t s)
    {
        return FixedSizeAllocator<T>::alloc_block(s);
    }
    /// @brief Give it back.
    /// @warning Routes by `sizeof(T)`. @see FixedSizeAllocator::free_block
    static void free(void* p)
    {
        FixedSizeAllocator<T>::free_block(p);
    }
};

/**
 * @brief Total what every allocator in the process is holding.
 * @param sum_UsedSize Bytes ever carved out of a block.
 * @param sum_nn_ref Bytes currently handed out.
 * @param alloc_sz Bytes held in blocks, whether handed out or not.
 * @details Walks the ladder, then the registry of ::TCTypeAllocator instances.
 * @note Serial builds only report the registry part -- see the note there.
 */
void compute_sum_allocated_size_TCAlloc(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz);

#include "fast_allocators/internal/static_const_array_inline.h"
