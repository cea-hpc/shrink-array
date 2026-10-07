#pragma once

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

#include <algorithm>  // std::max (slab sizing)
#include <cassert>    // assert
#include <cstddef>    // std::byte, std::size_t
#include <cstdint>    // std::uintptr_t (the block probe)
#include <cstdio>     // std::fprintf (diagnostics)
#include <cstdlib>    // std::abort
#include <memory>     // std::unique_ptr
#include <utility>    // std::move
#include <vector>     // std::vector
#include <sys/mman.h> // msync / MS_ASYNC (the block probe)
#include <unistd.h>   // sysconf (same)

#include "chunk/config.h"                   // build_config::use_fast_alloc, build_config::deferred_pool_granularity
#include "chunk/internal/env_parse.h"       // env_number (CHUNK_DEFERRED_POOL_KB), env_flag (CHUNK_RECYCLE_TRACE)
#include "chunk/internal/invariant_check.h" // INVARIANT_FAILED (the block probe)
#include "chunk/internal/pool_ledger.h"     // pool_ledger::on_op_alloc / on_ctx_alloc and the frees
#include "chunk/ring_log_debug.h"           // ring_log_op, dump_ring_for_addr, EV_OP_ALLOC / EV_OP_FREE

#ifdef FAST_ALLOC
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
    Op,
    Ctx,
    Value
};
} // namespace fast_alloc

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

    /// @brief Allocate sizeof(T) bytes with ::operator new.
    std::byte* alloc()
    {
        return static_cast<std::byte*>(::operator new(sizeof(T)));
    }
};
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
    void operator()(DeferredOperationBase* op) const noexcept
    {
        (void)op;
    }
};

/// @brief Non-owning handle on a queued op (destroy() releases it, not the pointer).
using DeferredOpPtr = std::unique_ptr<DeferredOperationBase, DeferredOpDeleter>;

// Everything below is the per-thread deferred-op / ctx pool machinery. Only the
// OpenMP write path ever allocates from it, but it compiles in every build:
// the registries are thread_local vectors that stay empty serially, so the
// reset entry points are no-ops there without a separate definition.

// ============================================================================
// Polymorphic allocator base
// ============================================================================

/// @brief Type-erased pool of one block size (a FastAllocatorConcrete per
///        object type, found by AllocatorRegistry).
struct FastAllocatorBase
{
    FastAllocatorBase()          = default;
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
 * The byte allocator itself is TCTypeAllocator<std::byte> — a type that does
 * NOT depend on T. Its only T-dependent input is element_size = sizeof(T).
 * When two element types have the SAME size (e.g. a 56-byte ctx and a 56-byte
 * op), FastAllocatorConcrete<Ctx>::allocator() and FastAllocatorConcrete<Op>::
 * allocator() compile to byte-identical bodies, and GCC's -fipa-icf (on at -O2)
 * folds them into ONE function sharing ONE thread_local pool — so the ctx pool
 * and the op pool alias and hand out the same address for a live ctx and a new
 * op (caught by POOL_LEDGER: "op_alloc ... but block is LIVE as ctx"; ARM-only
 * because folding decisions are target-dependent).
 *
 * Wrapping the thread_local in a T-parameterised type makes the two allocator()
 * bodies reference DIFFERENT types, which (a) gives each element type genuinely
 * disjoint storage and (b) makes the bodies non-identical so ICF cannot fold
 * them. This is portable — it does not rely on -fno-ipa-icf at the build level.
 */
template <typename T>
struct DeferredByteAllocSlot
{
    TCTypeAllocator<std::byte> alloc;               ///< this thread's byte allocator for T
    bool                       initialized = false; ///< alloc set up (allocator())
};

/**
 * @brief Per-element-type free-list slot.
 *
 * std::vector<T*> operations are pointer-generic and compile to identical code
 * regardless of T, so free_list() is foldable by -fipa-icf just like allocator()
 * — a shared free list would recycle a ctx block into the op pool (and vice
 * versa). Wrapping the vector in a T-parameterised type keeps each element
 * type's free list disjoint and the free_list() bodies non-identical.
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

    // In std-alloc, allocate_raw() hands out blocks via
    // ::operator new and recycle() parks them here in `fl` for reuse. Nothing
    // else owns them, so when this thread_local slot dies the vector is
    // destroyed but the pointed-to blocks are not — a real (if bounded, one-shot)
    // leak that LSan flags. fast-alloc has no such gap: there the blocks live in
    // the slab allocator, whose ~TCTypeAllocator frees every slab at thread
    // teardown, and `fl` does not own anything. So we release here ONLY in
    // std-alloc, making std-alloc symmetric with fast-alloc without needing the
    // CHUNK_POOL_NO_REUSE diagnostic define. Under CHUNK_POOL_NO_REUSE the
    // free-list is bypassed entirely (recycle deletes immediately), so `fl` is
    // empty and this loop is a no-op — the two mechanisms don't double-free.
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
template <typename T, fast_alloc::ArenaKind Arena = fast_alloc::ArenaKind::Op>
class FastAllocatorConcrete : public FastAllocatorBase
{
  public:
    using ByteAlloc = TCTypeAllocator<std::byte>;

    /// @brief Set block_size to sizeof(T) and, under fast-alloc, create this thread's allocator.
    FastAllocatorConcrete()
    {
        block_size = sizeof(T);
        if constexpr (build_config::use_fast_alloc)
        {
            (void)allocator();
        }
    }

  private:
    // Aliasing check (POOL_LEDGER): identity of this (type,thread) pool. Under
    // fast-alloc it's the byte-slab address; otherwise the free-list address.
    // Two element types share this address iff -fipa-icf folded their
    // allocator()/free_list() (or the T-tag failed to prevent it).
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
            // Diagnostic mode (CHUNK_POOL_NO_REUSE): bypass the free-list AND the fast pool.
            // Every block is a fresh global allocation, so a later recycle() is a
            // real ::operator delete. This gives ASan/TSan full visibility into
            // use-after-free / double-free of deferred ctx/op blocks — which the
            // recycling free-list normally hides (freed blocks stay mapped and
            // readable, so a use-after-recycle silently returns stale data).
            return ::operator new(sizeof(T));
        }
        auto& fl = free_list();
        if (!fl.empty())
        {
            T* block = fl.back();
            fl.pop_back();
            // Check-build guard: validate the pointer taken from the
            // thread-local free-list BEFORE returning it. The crash is a wild
            // address returned by allocate_raw and written by construct_at; this
            // catches it at the source and blames the free-list path (a corrupt
            // recycled pointer, e.g. a bad recycle() push) rather than the slab.
            if constexpr (build_config::any_check)
            {
                probe_returned_block(static_cast<void*>(block), "freelist");
            }
            return block;
        }

        if constexpr (build_config::use_fast_alloc)
        {
            std::byte* raw = allocator().alloc();
            // Check-build guard: validate the pointer fresh from the slab
            // bump-allocator. If THIS fires, the TCTypeAllocator itself produced
            // a wild address (corrupt block/bump state); if only the free-list
            // probe fires, recycling is the culprit.
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
     * the one that allocated it — notably destroy_after_flush()'s defensive
     * drain, which runs another thread's ops on the calling thread (see the
     * warning in ManagerChunk::destroy_after_flush and reset_all). If we blindly
     * push a foreign block into THIS thread's free_list, it survives here after
     * the owning thread's thread_local TCTypeAllocator is destroyed and munmap()s
     * its slabs — a later allocate_raw() then hands out a pointer into an
     * unmapped page.
     *
     * Guard: only recycle blocks that belong to THIS thread's slab (checked via
     * TCTypeAllocator::is_free_able_elt, which resolves the block header from the
     * pointer). A foreign block is dropped rather than recycled — a bounded,
     * one-phase leak that is vastly preferable to a deferred use-after-free. The
     * block's real owner still frees its whole slab at teardown.
     */
    void recycle(void* block) override
    {
        if constexpr (build_config::pool_no_reuse)
        {
            ::operator delete(block); // CHUNK_POOL_NO_REUSE: real free for ASan/TSan
            return;
        }
        if constexpr (build_config::use_fast_alloc)
        {
            // Only pool blocks owned by this thread's slab may go on this
            // thread's free_list. Foreign blocks (cross-thread drain) are
            // dropped to avoid a use-after-munmap when their owner's slab dies.
            if (!allocator().is_free_able_elt(static_cast<std::byte*>(block)))
            {
                if constexpr (build_config::any_check)
                {
                    // Dropping a foreign block is the EXPECTED outcome of a
                    // cross-thread drain -- the guard doing its job, not an
                    // anomaly. One line per block buried every other message in
                    // check/debug runs, so report the first, then on a doubling
                    // schedule with the running total: the fact stays visible,
                    // the output stays bounded (~log2(N) lines per thread) and a
                    // change of ORDER of magnitude is still obvious.
                    // CHUNK_RECYCLE_TRACE=1 restores one line per block.
                    static const bool               trace     = env_flag("CHUNK_RECYCLE_TRACE");
                    static thread_local std::size_t n_foreign = 0;
                    ++n_foreign;
                    if (trace || (n_foreign & (n_foreign - 1)) == 0)
                    {
                        std::fprintf(stderr,
                                     "[RECYCLE] dropped FOREIGN block %p (not owned by this thread's slab) "
                                     "— cross-thread drain, would be use-after-munmap if recycled; "
                                     "%zu on this thread so far\n",
                                     block, n_foreign);
                        std::fflush(stderr);
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
            const std::size_t     kb     = shrink_array::env_number<std::size_t>("CHUNK_DEFERRED_POOL_KB", def_kb);
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
        // T-tagged storage: distinct type per element type so -fipa-icf cannot
        // fold this body across same-sized element types and make ctx/op pools
        // share one slab. See DeferredByteAllocSlot above.
        thread_local DeferredByteAllocSlot<T> slot;

        if (!slot.initialized)
        {
            // Route this pool's slabs to its own arena. Op and ctx pools use the
            // same TCTypeAllocator<byte> but MUST draw from disjoint arenas -- a
            // freed ctx slab reobtained by the op pool overwrites a live op's
            // vtable with ctx layout. Arena is a template param so op and ctx
            // get genuinely separate never-unmapped arenas.
            slot.alloc.set_arena(Arena);

            constexpr size_t element_size = sizeof(T);
            // TCAllocator threads its free list through the freed elements (an
            // int in each): the same condition TCAllocator::init() checks at run
            // time, caught here at compile time where the size is known.
            static_assert(element_size >= sizeof(int) && element_size % alignof(int) == 0,
                          "deferred-op pool: an object must hold TCAllocator's free-list index");

            // Slab size in BYTES -- what TCSlabPool::init's second parameter is
            // (an OBJECT count there would give an object above a few KiB a
            // slab smaller than itself). Never fewer than 64 objects per slab.
            const uintptr_t slab_request =
                std::max<uintptr_t>(deferred_pool_slab_granularity(), uintptr_t{64} * element_size);

            // Last arg = nb_max_block_empty = -1 -> the op allocator NEVER
            // munmaps an empty slab: a freed op slab's address could be
            // reobtained by fast_alloc::map_anonymous() for the chunk VALUE pool
            // on another thread and overwrite a still-live op. Op slabs are small
            // and bounded, so the retained memory is modest.
            slot.alloc.init(element_size, slab_request, -1);
            slot.initialized = true;
        }
        return slot.alloc;
    }

    /**
     * @brief Per-thread free list of recycled memory blocks.
     */
    static std::vector<T*>& free_list()
    {
        // T-tagged storage (same anti-ICF rationale as allocator()): keeps each
        // element type's recycled-block list disjoint even when vector<T*> code
        // is identical across types.
        thread_local DeferredFreeListSlot<T> slot;
        return slot.fl;
    }

    // Check-build guard -- validate a pointer just before allocate_raw hands
    // it to construct_at (wild addresses, mapped in NO region, have been seen).
    //
    // Do NOT discriminate on the address range: fast_alloc::map_anonymous()
    // uses mmap(nullptr, ...), so the kernel may place a valid slab high in the
    // 0xffff... band (same band as thread stacks). Ask the kernel whether the
    // page is mapped, via msync(): it returns ENOMEM for an unmapped address
    // and never faults. Only genuinely unmapped / misaligned / null pointers
    // trip the probe.
    /// @brief Check-only: fail the invariant if @p ptr, returned by @p path,
    ///        is null, misaligned or not a mapped page.
    static void probe_returned_block(const void* ptr, const char* path) noexcept
    {
        if constexpr (build_config::any_check)
        {
            const auto addr = reinterpret_cast<std::uintptr_t>(ptr);

            if (ptr == nullptr)
            {
                INVARIANT_FAILED("[ALLOC_PROBE {}] allocate_raw returned NULL for T size={}", path, sizeof(T));
            }

            if ((addr & (alignof(std::max_align_t) - 1)) != 0 && (addr & 0x7U) != 0)
            {
                INVARIANT_FAILED("[ALLOC_PROBE {}] misaligned block p={} (T size={})", path,
                                 static_cast<const void*>(ptr), sizeof(T));
            }

            // Page-containing-p mapped? msync(MS_ASYNC) touches nothing but the
            // kernel's VMA table: 0 = mapped, -1/ENOMEM = not mapped. The whole
            // object [p, p+sizeof(T)) may straddle two pages, so check both ends.
            // Page bases as pointer arithmetic from p, not integer -> pointer
            // casts (performance-no-int-to-ptr): p minus its offset in its page,
            // and the same for the last byte of the object.
            const long  pg   = ::sysconf(_SC_PAGESIZE);
            const auto  mask = static_cast<std::uintptr_t>(pg - 1);
            const auto* base = static_cast<const char*>(ptr);
            const void* pg0  = base - (addr & mask);
            const void* pg1  = base + (sizeof(T) - 1) - ((addr + sizeof(T) - 1) & mask);

            if (::msync(const_cast<void*>(pg0), 1, MS_ASYNC) != 0 || ::msync(const_cast<void*>(pg1), 1, MS_ASYNC) != 0)
            {
                INVARIANT_FAILED("[ALLOC_PROBE {}] block NOT MAPPED (wild ptr): p={} T size={} slab={}", path,
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
    static constexpr fast_alloc::ArenaKind value = fast_alloc::ArenaKind::Op;
};
/// @brief Context allocators: the Ctx arena.
template <>
struct arena_of_tag<CtxTag>
{
    static constexpr fast_alloc::ArenaKind value = fast_alloc::ArenaKind::Ctx;
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

    /// @brief Destroy this thread's allocators (teardown).
    static void reset()
    {
        allocs().clear();
    }

  private:
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
            std::fprintf(stderr,
                         "[SLAB MISALIGN] %s: ptr=%p alignof=%zu sizeof=%zu (block from slab is under-aligned for T)\n",
                         site, mem, static_cast<size_t>(alignof(T)), sizeof(T));
            std::fflush(stderr);
            std::abort();
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
    pool_ledger::on_op_alloc(mem, sizeof(T), alloc->slab_addr());
    T* constructed = std::construct_at(static_cast<T*>(mem), std::forward<Args>(args)...);
    // Debug ring log AFTER construct_at, so OP_ALLOC marks a fully-built op
    // (vptr written).
    if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
    {
        ring_log_op(get_tid(), EV_OP_ALLOC, 0, mem);
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
    pool_ledger::on_ctx_alloc(mem, sizeof(T), alloc->slab_addr());
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
        ring_log_op(get_tid(), EV_OP_FREE, 0, static_cast<const void*>(op));
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
 * Kept for now as a phase-boundary hook. DO NOT refactor descriptors
 * into owning their pools without first fixing cross-thread free_list
 * references (destroy_after_flush's defensive drain recycles other
 * threads' blocks into the draining thread's free_list) — otherwise
 * this call becomes a deferred use-after-free.
 */
inline void reset_all()
{
    reset_op_allocators();
    reset_ctx_allocators();
}

} // namespace shrink_array