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

#include <algorithm>   // std::max (slab sizing)
#include <bit>         // std::has_single_bit (the FOREIGN-block report schedule)
#include <cassert>     // assert
#include <cstddef>     // std::byte, std::size_t
#include <cstdint>     // std::uintptr_t (the block probe)
#include <memory>      // std::unique_ptr
#include <mutex>       // std::mutex, std::scoped_lock (the pools of exited threads)
#include <type_traits> // std::conditional_t (DeferredByteAllocSlot)
#include <utility>     // std::move
#include <vector>      // std::vector
#include <sys/mman.h>  // msync / MS_ASYNC (the block probe)
#include <unistd.h>    // sysconf (same)

#include "chunk/config.h"                   // build_config::use_fast_alloc, build_config::deferred_pool_granularity
#include "chunk/internal/env_parse.h"       // env_number (CHUNK_DEFERRED_POOL_KB)
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (the block probe)
#include "chunk/internal/pool_ledger.h"     // pool_ledger::on_op_alloc / on_ctx_alloc and the frees
#include "chunk/ring_log_debug.h"           // ring_log_op, dump_ring_for_addr, ev_op_alloc / ev_op_free
#include "fast_allocators/internal/pmp_runtime.h" // fast_alloc::pmp::backend (the pools of exited threads)
#include "chunk/log.h"                            // CHUNK_LOG (recycle)

#if CHUNK_OPT_FAST_ALLOC
    #include "fast_allocators/slab_arena.h"   // fast_alloc::ArenaKind
    #include "fast_allocators/tc_allocator.h" // TCTypeAllocator<std::byte>
#else
/**
 * @brief Fallback definitions when FAST_ALLOC is disabled.
 * Ensures the template signatures in this file remain valid.
 */
namespace fast_alloc
{
/// @brief Fallback arena kinds (op, ctx, value), matching the names of the fast allocator's enum.
enum class ArenaKind
{
    op,
    ctx,
    value
};

/// @brief std-alloc stand-in for the fast allocator: operator new/delete
///        behind the same call surface (init/set_arena no-ops, every
///        element "free-able").
template <typename T>
class TCTypeAllocator
{
  public:
    /// @brief No-op.
    void init(size_t, size_t, size_t) {}
    /// @brief No-op.
    void set_arena(fast_alloc::ArenaKind) {}
    /// @brief Always true.
    bool is_free_able_elt(std::byte*)
    {
        return true;
    }

    /// @brief Allocate sizeof(T) bytes with the global operator new.
    std::byte* alloc()
    {
        return static_cast<std::byte*>(::operator new(sizeof(T)));
    }
};
} // namespace fast_alloc
#endif

namespace shrink_array
{

// ============================================================================
// Base class for deferred operations
// ============================================================================

/// @brief A queued write, type-erased: replayed by write_body(), returned to
///        its pool by destroy().
struct DeferredOperationBase
{
    DeferredOperationBase()                   = default;
    virtual ~DeferredOperationBase() noexcept = default;

    // Polymorphic, and pool-owned through destroy(): never copied or moved.
    DeferredOperationBase(const DeferredOperationBase&)            = delete;
    DeferredOperationBase& operator=(const DeferredOperationBase&) = delete;
    DeferredOperationBase(DeferredOperationBase&&)                 = delete;
    DeferredOperationBase& operator=(DeferredOperationBase&&)      = delete;

    /// @brief Replay the write.
    virtual void write_body() = 0;
    /// @brief Run the destructor and hand the block back to its pool.
    virtual void destroy() noexcept = 0;
};

/// @brief Deleter of DeferredOpPtr: does nothing with the pointer.
struct DeferredOpDeleter
{
    /// @brief No-op: the operation is not destroyed or freed here.
    static void operator()(const DeferredOperationBase* op) noexcept
    {
        (void)op;
    }
};

/// @brief Non-owning handle on a queued op (destroy() releases it, not the pointer).
using DeferredOpPtr = std::unique_ptr<DeferredOperationBase, DeferredOpDeleter>;

// The per-thread deferred-op / ctx pools. Only the OpenMP write path allocates
// from them, but they compile in every build: serially the thread_local
// registries stay empty and the reset entry points are no-ops.

// ============================================================================
// Polymorphic allocator base
// ============================================================================

/// @brief Type-erased pool of one block size (a FastAllocatorConcrete per
///        object type, found by AllocatorRegistry).
struct FastAllocatorBase
{
    /// @param block_bytes bytes per block
    explicit FastAllocatorBase(size_t block_bytes) noexcept : block_size(block_bytes) {}
    virtual ~FastAllocatorBase() = default;

    // Polymorphic, held by unique_ptr in the registry: never copied or moved.
    FastAllocatorBase(const FastAllocatorBase&)            = delete;
    FastAllocatorBase& operator=(const FastAllocatorBase&) = delete;
    FastAllocatorBase(FastAllocatorBase&&)                 = delete;
    FastAllocatorBase& operator=(FastAllocatorBase&&)      = delete;

    /// @brief Take a block (uninitialised).
    virtual void* allocate_raw() = 0;
    /// @brief Return a block to the pool.
    virtual void recycle(void* block) = 0;
    /// @brief The slab the blocks come from (the ledger's identity).
    [[nodiscard]] virtual const void* slab_addr() const noexcept = 0;

    size_t block_size = 0; ///< bytes per block
};

// ============================================================================
// Concrete allocator for a given type T
// ============================================================================

/**
 * @brief Per-element-type storage slot for the byte allocator of allocator().
 *
 * Do not hold a plain `thread_local TCTypeAllocator<std::byte>` in allocator():
 * that type does not depend on T, so for two same-sized types (a 56-byte ctx and
 * a 56-byte op) the allocator() bodies are byte-identical and GCC's -fipa-icf (on
 * at -O2, target-dependent) folds them into ONE function sharing ONE pool: the
 * ctx and op pools alias and hand out a live ctx's block to a new op.
 *
 * Wrapping the thread_local in a T-parameterised type gives each element type
 * disjoint storage and makes the bodies non-identical, so ICF cannot fold them,
 * without relying on -fno-ipa-icf in the build.
 */
/**
 * @brief Std backend: the op / ctx pools of a thread outlive it, until the
 *        next phase end.
 * @details An application thread under the std backend (a std::thread, docs/
 *          threading_backends.md step 7c) may exit with writes still queued on
 *          its rank: the serial contract joins the writers, THEN calls
 *          finalize_parallel_phase(), which replays them. Its thread_local
 *          allocator would hand its slabs back to the Op / Ctx arena at the
 *          exit, where another thread's next slab takes them and overwrites the
 *          queued ops. So, under the std backend, the allocator lives on the
 *          heap, an exiting thread hands it to exited_thread_pools(), and
 *          finalize_parallel_phase() destroys those after its flush
 *          (retire_exited_thread_pools()). OpenMP's threads live as long as the
 *          runtime: their pools stay in the thread_local, as before.
 */
inline constexpr bool deferred_pools_outlive_thread =
    build_config::use_fast_alloc && ::fast_alloc::pmp::backend == ::fast_alloc::pmp::BackendKind::std_threads;

namespace detail
{
/// @brief The op / ctx byte allocators of the threads that exited since the
///        last phase end (std backend; see deferred_pools_outlive_thread).
struct ExitedThreadPools
{
    std::mutex                                           mutex; ///< guards list
    std::vector<fast_alloc::TCTypeAllocator<std::byte>*> list;  ///< owned, destroyed by retire_exited_thread_pools()
};

/// @brief The process-wide list (immortal: threads add to it from their thread_local destructors).
[[nodiscard]] inline ExitedThreadPools& exited_thread_pools()
{
    static auto* const p = new ExitedThreadPools; // NOLINT(cppcoreguidelines-owning-memory): immortal
    return *p;
}

/// @brief In place of the inline allocator when it lives on the heap.
struct NoInlineAlloc
{
};
} // namespace detail

template <typename T>
struct DeferredByteAllocSlot
{
    using Alloc = fast_alloc::TCTypeAllocator<std::byte>;
    /// deferred_pools_outlive_thread, made dependent on T: the branch not
    /// taken below is then not checked (it names the other member's type).
    static constexpr bool on_heap = deferred_pools_outlive_thread && sizeof(T) > 0;

    DeferredByteAllocSlot() = default;
    // Owns its allocator (inline, or on the heap until the thread exits).
    DeferredByteAllocSlot(const DeferredByteAllocSlot&)            = delete;
    DeferredByteAllocSlot& operator=(const DeferredByteAllocSlot&) = delete;
    DeferredByteAllocSlot(DeferredByteAllocSlot&&)                 = delete;
    DeferredByteAllocSlot& operator=(DeferredByteAllocSlot&&)      = delete;

    /// Std backend: the thread exits; its allocator waits for the phase end.
    ~DeferredByteAllocSlot()
    {
        if constexpr (on_heap)
        {
            if (heap_ != nullptr)
            {
                auto&                  ex = detail::exited_thread_pools();
                const std::scoped_lock lock(ex.mutex);
                ex.list.push_back(heap_);
            }
        }
    }

    /// @brief This thread's byte allocator for T.
    [[nodiscard]] Alloc& alloc()
    {
        if constexpr (on_heap)
        {
            if (heap_ == nullptr) [[unlikely]]
            {
                heap_ = new Alloc; // NOLINT(cppcoreguidelines-owning-memory): handed over at thread exit
            }
            return *heap_;
        }
        else
        {
            return inline_;
        }
    }

    bool initialized = false; ///< alloc set up (allocator())

  private:
    /// The allocator, in the thread_local itself (OpenMP, serial).
    [[no_unique_address]] std::conditional_t<on_heap, detail::NoInlineAlloc, Alloc> inline_{};
    Alloc* heap_ = nullptr; ///< std backend: the allocator, on the heap
};

/**
 * @brief Std backend: destroy the op / ctx allocators of the threads that
 *        exited (deferred_pools_outlive_thread). At a phase end, after the
 *        flush: no op of theirs is queued any more. A no-op otherwise.
 */
inline void retire_exited_thread_pools()
{
    if constexpr (deferred_pools_outlive_thread)
    {
        std::vector<fast_alloc::TCTypeAllocator<std::byte>*> retired;
        {
            auto&                  ex = detail::exited_thread_pools();
            const std::scoped_lock lock(ex.mutex);
            retired.swap(ex.list);
        }
        for (auto* a : retired)
        {
            delete a; // NOLINT(cppcoreguidelines-owning-memory): owned by the list
        }
    }
}

/**
 * @brief Per-element-type free-list slot.
 *
 * Same -fipa-icf hazard as DeferredByteAllocSlot: std::vector<T*> code is
 * identical for every T, and a folded free_list() would recycle a ctx block into
 * the op pool (and vice versa). The T-parameterised wrapper keeps them disjoint.
 */
template <typename T>
struct DeferredFreeListSlot
{
    DeferredFreeListSlot() = default;
    // Owns the parked blocks in std-alloc builds (see the destructor): a copy
    // would delete them twice.
    DeferredFreeListSlot(const DeferredFreeListSlot&)            = delete;
    DeferredFreeListSlot& operator=(const DeferredFreeListSlot&) = delete;
    DeferredFreeListSlot(DeferredFreeListSlot&&)                 = delete;
    DeferredFreeListSlot& operator=(DeferredFreeListSlot&&)      = delete;

    // std-alloc only: the blocks parked in `fl` come from ::operator new and
    // nothing else owns them, so they are deleted here (else a bounded leak at
    // thread exit). Under fast-alloc `fl` owns nothing: ~TCTypeAllocator frees
    // the slabs. Under CHUNK_POOL_NO_REUSE recycle() deletes at once, `fl` stays
    // empty and this loop is a no-op (no double free).
    ~DeferredFreeListSlot()
    {
        if constexpr (!build_config::use_fast_alloc)
        {
            for (T* p : fl)
            {
                ::operator delete(static_cast<void*>(p));
            }
        }
    }

    std::vector<T*> fl; ///< recycled blocks, reused by allocate_raw()
};

/**
 * @brief Per-thread pool allocator for objects of type T.
 * Uses a calculated number of blocks based on the global granularity config.
 */
template <typename T, fast_alloc::ArenaKind Arena = fast_alloc::ArenaKind::op>
class FastAllocatorConcrete : public FastAllocatorBase
{
  public:
    using ByteAlloc = fast_alloc::TCTypeAllocator<std::byte>;

    /// @brief Set block_size to sizeof(T) and, under fast-alloc, create this thread's allocator.
    FastAllocatorConcrete() : FastAllocatorBase(sizeof(T))
    {
        if constexpr (build_config::use_fast_alloc)
        {
            (void)allocator();
        }
    }

  private:
    // Aliasing check (POOL_LEDGER): identity of this (type,thread) pool -- the
    // byte-slab address under fast-alloc, else the free-list address. Two element
    // types share it iff -fipa-icf folded their allocator()/free_list().
    [[nodiscard]] const void* slab_addr() const noexcept override
    {
        if constexpr (build_config::use_fast_alloc)
        {
            return static_cast<const void*>(&allocator());
        }
        else
        {
            return static_cast<const void*>(&free_list());
        }
    }

    /**
     * @brief Allocates raw memory for one object of type T.
     */
    void* allocate_raw() override
    {
        if constexpr (build_config::pool_no_reuse)
        {
            // Diagnostic mode (CHUNK_POOL_NO_REUSE): bypass the free-list AND the fast
            // pool, so recycle() is a real ::operator delete and sanitizers see a
            // use-after-recycle that the free-list would hide (stale but mapped data).
            return ::operator new(sizeof(T));
        }
        auto& fl = free_list();
        if (!fl.empty())
        {
            T* block = fl.back();
            fl.pop_back();
            // Check-build guard: a firing here blames the free-list (a corrupt
            // recycled pointer), not the slab.
            if constexpr (build_config::any_check)
            {
                probe_returned_block(static_cast<void*>(block), "freelist");
            }
            return block;
        }

        if constexpr (build_config::use_fast_alloc)
        {
            std::byte* raw = allocator().alloc();
            // Check-build guard: a firing here blames TCTypeAllocator itself
            // (corrupt block/bump state).
            if constexpr (build_config::any_check)
            {
                probe_returned_block(static_cast<void*>(raw), "slab");
            }
            return static_cast<void*>(raw);
        }
        else
        {
            return ::operator new(sizeof(T));
        }
    }

    /**
     * @brief Recycles a raw memory block back into the per-thread free list.
     *
     * CROSS-THREAD SAFETY: a block may reach recycle() on a thread other than
     * the one that allocated it (destroy_after_flush()'s defensive drain runs
     * another thread's ops here; see ManagerChunk::destroy_after_flush and
     * reset_all). A foreign block pushed on THIS thread's free_list would outlive
     * its owner's thread_local TCTypeAllocator, which munmap()s its slabs: a later
     * allocate_raw() would hand out an unmapped pointer.
     *
     * So only blocks of THIS thread's slab are recycled
     * (TCTypeAllocator::is_free_able_elt); a foreign one is dropped -- a bounded
     * leak, its owner still frees the whole slab at teardown.
     */
    void recycle(void* block) override
    {
        if constexpr (build_config::pool_no_reuse)
        {
            ::operator delete(block); // CHUNK_POOL_NO_REUSE: real free, visible to sanitizers
            return;
        }
        if constexpr (build_config::use_fast_alloc)
        {
            // Foreign block (cross-thread drain): drop it (see above).
            if (!allocator().is_free_able_elt(static_cast<std::byte*>(block)))
            {
                if constexpr (build_config::any_check)
                {
                    // Expected, not an anomaly: recycle=debug reports the
                    // first, then on a doubling schedule with the running total
                    // (~log2(N) lines per thread); recycle=trace (formerly
                    // CHUNK_RECYCLE_TRACE) one line per block.
                    thread_local std::size_t n_foreign = 0;
                    ++n_foreign;
                    if (std::has_single_bit(n_foreign))
                    {
                        CHUNK_LOG(debug, recycle,
                                  "dropped FOREIGN block {} (not owned by this thread's slab) -- cross-thread drain, "
                                  "would be use-after-munmap if recycled; {} on this thread so far",
                                  block, n_foreign);
                    }
                    else
                    {
                        CHUNK_LOG(trace, recycle,
                                  "dropped FOREIGN block {} (not owned by this thread's slab) -- cross-thread drain, "
                                  "would be use-after-munmap if recycled; {} on this thread so far",
                                  block, n_foreign);
                    }
                }
                return;
            }
        }
        free_list().push_back(static_cast<T*>(block));
    }

    /**
     * @brief Target slab size of the op / ctx pools, in bytes: the build's
     *        deferred_pool_granularity (64 KiB), or CHUNK_DEFERRED_POOL_KB KiB.
     * @details Read once. The knob is for measuring: the pools are per thread,
     *          per element type and per op/ctx variant, and op slabs are never
     *          returned, so the slab size trades allocator traffic against
     *          resident memory -- the more so under transparent huge pages
     *          (`always`), where a fresh 2 MiB-aligned slab can fault a whole
     *          huge page in at its first object.
     */
    [[nodiscard]] static uintptr_t deferred_pool_slab_granularity() noexcept
    {
        static const uintptr_t bytes = []
        {
            constexpr std::size_t def_kb = build_config::deferred_pool_granularity / 1024;
            const auto            kb     = shrink_array::env_number<std::size_t>("CHUNK_DEFERRED_POOL_KB", def_kb);
            return static_cast<uintptr_t>(kb == 0 ? def_kb : kb) * 1024;
        }();
        return bytes;
    }

    /**
     * @brief Returns the per-thread byte allocator for type T.
     * @details Slabs of deferred_pool_slab_granularity() bytes, at least 64
     *          objects (TCSlabPool rounds the size up to a power of two).
     */
    static ByteAlloc& allocator()
    {
        // T-tagged storage against -fipa-icf folding: see DeferredByteAllocSlot.
        thread_local DeferredByteAllocSlot<T> slot;

        if (!slot.initialized)
        {
            // Op and ctx pools MUST draw from disjoint arenas: a freed ctx slab
            // reobtained by the op pool would overwrite a live op's vtable.
            slot.alloc().set_arena(Arena);

            constexpr size_t element_size = sizeof(T);
            // TCAllocator threads its free list through the freed elements (an
            // int in each): TCAllocator::init()'s run-time check, at compile time.
            static_assert(element_size >= sizeof(int) && element_size % alignof(int) == 0,
                          "deferred-op pool: an object must hold TCAllocator's free-list index");

            // Slab size in BYTES (TCSlabPool::init's second parameter; not an
            // object count). Never fewer than 64 objects per slab.
            const uintptr_t slab_request =
                std::max<uintptr_t>(deferred_pool_slab_granularity(), uintptr_t{64} * element_size);

            // nb_max_block_empty = -1: NEVER munmap an empty slab. A freed op
            // slab's address could be remapped by fast_alloc::map_anonymous() for
            // the VALUE pool on another thread and overwrite a still-live op. Op
            // slabs are small and bounded.
            slot.alloc().init(element_size, slab_request, -1);
            slot.initialized = true;
        }
        return slot.alloc();
    }

    /**
     * @brief Per-thread free list of recycled memory blocks.
     */
    static std::vector<T*>& free_list()
    {
        // T-tagged storage against -fipa-icf folding: see DeferredFreeListSlot.
        thread_local DeferredFreeListSlot<T> slot;
        return slot.fl;
    }

    // Do NOT discriminate on the address range: map_anonymous() uses
    // mmap(nullptr, ...), so a valid slab may sit high in the 0xffff... band (with
    // thread stacks). msync() asks the kernel instead: ENOMEM if unmapped, never faults.
    /// @brief Check-only: fail the invariant if @p ptr, returned by @p path,
    ///        is null, misaligned or not a mapped page.
    static void probe_returned_block(const void* ptr, const char* path) noexcept
    {
        if constexpr (build_config::any_check)
        {
            const auto addr = reinterpret_cast<std::uintptr_t>(ptr);

            if (ptr == nullptr)
            {
                CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] allocate_raw returned NULL for T size={}", path, sizeof(T));
            }

            if ((addr & (alignof(std::max_align_t) - 1)) != 0 && (addr & 0x7U) != 0)
            {
                CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] misaligned block p={} (T size={})", path,
                                       static_cast<const void*>(ptr), sizeof(T));
            }

            // msync(MS_ASYNC) only reads the VMA table: 0 = mapped, -1/ENOMEM = not.
            // The object may straddle two pages: check both ends. Page bases by
            // pointer arithmetic from p, not int -> pointer casts (performance-no-int-to-ptr).
            const long  pg   = ::sysconf(_SC_PAGESIZE);
            const auto  mask = static_cast<std::uintptr_t>(pg - 1);
            const auto* base = static_cast<const char*>(ptr);
            const void* pg0  = base - (addr & mask);
            const void* pg1  = base + (sizeof(T) - 1) - ((addr + sizeof(T) - 1) & mask);

            if (::msync(const_cast<void*>(pg0), 1, MS_ASYNC) != 0 || ::msync(const_cast<void*>(pg1), 1, MS_ASYNC) != 0)
            {
                CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] block NOT MAPPED (wild ptr): p={} T size={} slab={}", path,
                                       static_cast<const void*>(ptr), sizeof(T),
                                       build_config::use_fast_alloc ? static_cast<const void*>(&allocator()) : nullptr);
            }
        }
    }
};

// ============================================================================
// Allocator Registry and Tags
// ============================================================================

/// @brief Registry tag of the deferred-op allocators (OpRegistry).
struct OpTag
{
};
/// @brief Registry tag of the context allocators (CtxRegistry).
struct CtxTag
{
};

/// @brief The arena of a registry tag, so op and ctx pools allocate slabs
///        from disjoint address spaces.
template <typename Tag>
struct arena_of_tag;
/// @brief Op allocators: the Op arena.
template <>
struct arena_of_tag<OpTag>
{
    static constexpr fast_alloc::ArenaKind value = fast_alloc::ArenaKind::op;
};
/// @brief Context allocators: the Ctx arena.
template <>
struct arena_of_tag<CtxTag>
{
    static constexpr fast_alloc::ArenaKind value = fast_alloc::ArenaKind::ctx;
};

/**
 * @brief One thread's cached answer of AllocatorRegistry<Tag>::get_or_create<T>().
 * @details T-tagged storage, as DeferredByteAllocSlot: the get_or_create<T>()
 *          bodies of two same-sized types stay distinct, so -fipa-icf cannot
 *          fold them onto one slot.
 */
template <typename Tag, typename T>
struct CachedAllocatorSlot
{
    FastAllocatorBase* alloc      = nullptr; ///< the allocator found, or nullptr
    unsigned int       generation = 0;       ///< the registry generation it was found in
};

/**
 * @brief Registry that manages a collection of FastAllocators, one per object size.
 */
template <typename Tag>
class AllocatorRegistry
{
  public:
    /// @brief This thread's allocator for objects of sizeof(T) bytes, created on first request.
    /// @tparam T Object type; allocators are shared between types of the same size.
    /// @return The allocator whose block_size is sizeof(T).
    template <typename T>
    static FastAllocatorBase* get_or_create()
    {
        // Hot (every alloc/release): cached per thread until reset() moves the generation.
        thread_local CachedAllocatorSlot<Tag, T> cached;
        if (cached.alloc != nullptr && cached.generation == generation()) [[likely]]
        {
            return cached.alloc;
        }
        cached.alloc      = find_or_create<T>();
        cached.generation = generation();
        return cached.alloc;
    }

    /// @brief Destroy this thread's allocators (teardown).
    static void reset()
    {
        allocs().clear();
        ++generation(); // the cached answers point at the destroyed descriptors
    }

  private:
    /// @brief The registry scan behind get_or_create(): this thread's allocator
    ///        for objects of sizeof(T) bytes, created on first request.
    template <typename T>
    static FastAllocatorBase* find_or_create()
    {
        auto&        v  = allocs();
        const size_t sz = sizeof(T);

        for (auto& a : v)
        {
            if (a->block_size == sz)
            {
                return a.get();
            }
        }

        auto               ptr = std::make_unique<FastAllocatorConcrete<T, arena_of_tag<Tag>::value>>();
        FastAllocatorBase* raw = ptr.get();
        v.push_back(std::move(ptr));
        return raw;
    }

    /// @brief This thread's registry generation: moved on by reset(), it
    ///        invalidates every cached answer of get_or_create().
    /// @note Two tags may share it (identical bodies, -fipa-icf): a reset of
    ///       one then also invalidates the other's cache, which only costs a
    ///       rescan.
    static unsigned int& generation()
    {
        thread_local unsigned int g = 0;
        return g;
    }

    /// @brief This thread's allocators (one per object size seen so far).
    static std::vector<std::unique_ptr<FastAllocatorBase>>& allocs()
    {
        thread_local std::vector<std::unique_ptr<FastAllocatorBase>> v;
        return v;
    }
};

using OpRegistry  = AllocatorRegistry<OpTag>;  ///< the deferred-op allocators
using CtxRegistry = AllocatorRegistry<CtxTag>; ///< the context allocators

// ============================================================================
// Allocation API
// ============================================================================

/// @brief CHUNK_ALIGN_CHECK builds (CMake option): print and abort if pooled
///        block @p mem is misaligned for the T about to be constructed on it.
/// @details On aarch64 a misaligned atomic access (LDAR/STLR/LDXR) faults.
template <typename T>
void chunk_align_probe([[maybe_unused]] const char* site, [[maybe_unused]] void* mem) noexcept
{
    if constexpr (build_config::align_check)
    {
        if ((reinterpret_cast<std::uintptr_t>(mem) & (alignof(T) - 1)) != 0)
        {
            CHUNK_INVARIANT_FAILED("[SLAB MISALIGN] {}: ptr={} alignof={} sizeof={} (block from slab is under-aligned "
                                   "for T)",
                                   site, mem, alignof(T), sizeof(T));
        }
    }
}

/**
 * @brief Allocates and constructs a deferred operation of type T.
 */
template <typename T, typename... Args>
[[nodiscard]] T* pool_new_op(Args&&... args)
{
    FastAllocatorBase* alloc = OpRegistry::get_or_create<T>();
    void*              mem   = alloc->allocate_raw();
    assert(mem && "OpRegistry allocator returned null");
    chunk_align_probe<T>("op", mem);
    // Check builds: the block must not be live as a ctx or an op already.
    if constexpr (pool_ledger::pool_ledger_enabled)
    {
        // slab_addr() is virtual: not called at all when the ledger is off.
        pool_ledger::on_op_alloc(mem, sizeof(T), alloc->slab_addr());
    }
    T* constructed = std::construct_at(static_cast<T*>(mem), std::forward<Args>(args)...);
    // Debug ring log AFTER construct_at, so OP_ALLOC marks a fully-built op
    // (vptr written).
    if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
    {
        ring_log_op(get_tid(), ev_op_alloc, 0, mem);
    }
    return constructed;
}

/**
 * @brief Allocates and constructs a context of type T.
 */
template <typename T, typename... Args>
[[nodiscard]] T* pool_new_ctx(Args&&... args)
{
    FastAllocatorBase* alloc = CtxRegistry::get_or_create<T>();
    void*              mem   = alloc->allocate_raw();
    assert(mem && "CtxRegistry allocator returned null");
    chunk_align_probe<T>("ctx", mem);
    if constexpr (pool_ledger::pool_ledger_enabled)
    {
        pool_ledger::on_ctx_alloc(mem, sizeof(T), alloc->slab_addr()); // virtual: only when the ledger is on
    }
    return std::construct_at(static_cast<T*>(mem), std::forward<Args>(args)...);
}

// ============================================================================
// Deallocation API
// ============================================================================

/// @brief Record the release in the pool ledger and recycle @p op into its
///        OpRegistry allocator.
/// @param op Operation block to recycle; not destroyed here.
template <typename T>
void free_op(T* op)
{
    // Debug ring log: which thread recycles this op block, so dump_ring_for_addr
    // can show a free while the op was still referenced by op_map.
    if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
    {
        ring_log_op(get_tid(), ev_op_free, 0, static_cast<const void*>(op));
    }
    pool_ledger::on_op_free(static_cast<void*>(op));
    FastAllocatorBase* alloc = OpRegistry::get_or_create<T>();
    alloc->recycle(static_cast<void*>(op));
}

/// @brief Record the release in the pool ledger and recycle @p context into its CtxRegistry allocator.
/// @param context Context block to recycle; not destroyed here.
template <typename T>
void free_ctx(T* context)
{
    pool_ledger::on_ctx_free(static_cast<void*>(context));
    FastAllocatorBase* alloc = CtxRegistry::get_or_create<T>();
    alloc->recycle(static_cast<void*>(context));
}

/// @brief Destroy the calling thread's op allocators (OpRegistry::reset()).
inline void reset_op_allocators()
{
    OpRegistry::reset();
}
/// @brief Destroy the calling thread's context allocators (CtxRegistry::reset()).
inline void reset_ctx_allocators()
{
    CtxRegistry::reset();
}

/**
 * @brief Clears the allocator REGISTRY of the CALLING thread only.
 *
 * @warning This does NOT free or recycle any pool memory:
 *  - allocs() is thread_local: only the calling thread's descriptor
 *    vector is cleared. Worker threads' registries are untouched.
 *  - FastAllocatorConcrete descriptors do not own the memory:
 *    allocator() (TCAllocator pool) and free_list() are STATIC
 *    thread_local, independent of descriptor lifetime. Destroying
 *    descriptors releases nothing; blocks live for program lifetime.
 *
 * DO NOT make descriptors own their pools without first fixing cross-thread
 * free_list references (destroy_after_flush's defensive drain recycles other
 * threads' blocks into the draining thread's free_list): this call would become
 * a deferred use-after-free.
 */
inline void reset_all()
{
    reset_op_allocators();
    reset_ctx_allocators();
}

} // namespace shrink_array