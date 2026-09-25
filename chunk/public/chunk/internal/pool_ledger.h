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

// ============================================================================
// CHECK-BUILD INSTRUMENTATION — block-lifecycle ledger (grep tag: POOL_LEDGER).
// Born of the ctx/op pool aliasing hunt; kept as a permanent guard because it
// costs nothing in release, and it has since paid for itself on the teardown
// double-free (see manager_chunk_inline.h). -DCHUNK_NO_LEDGER drops it.
//
// The DEFERRED_LIFECYCLE sentinels proved that, on a single thread, a block
// referenced as a deferred `ctx` was reused as a deferred `op` (its slot held
// op_dead_magic). ctx and op are supposed to come from two disjoint pools
// (CtxRegistry vs OpRegistry); this ledger verifies that invariant directly.
//
// It maintains two GLOBAL address sets (ctx-live, op-live), protected by a
// mutex — global, not thread_local, because cross-thread flush legitimately
// frees a block on a thread other than the one that allocated it (see the
// warning in reset_all()). A thread_local ledger would false-positive on that
// path; a global one does not.
//
// The check that is expected to fire first for this bug:
//   op_alloc(p): p is still live as a ctx  →  a pending op still references p
//                as its ctx, yet the pool just handed p out as an op.
//
// All hooks compile to nothing when build_config::any_check is false.
// ============================================================================

#include <cstddef> // std::size_t
#include <cstdint> // std::uintptr_t (the aarch64 alignment probe)
#include <cstdio>  // std::fprintf
#include <cstdlib> // std::abort
#include <mutex>

#include "chunk/config.h"
#include "chunk/internal/invariant_check.h"

#include "ankerl/unordered_dense.h" // the live maps

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_IN_PARALLEL_REGION / _THREAD_RANK, and <omp.h> with it

namespace shrink_array::pool_ledger
{

// One record per live block: its size and the rank that allocated it, for
// richer diagnostics when an invariant trips.
struct Rec
{
    std::size_t size;
    int         alloc_rank;
    const void* slab = nullptr; // identity of the pool that served this block (aliasing check)
};

inline std::mutex& mtx()
{
    static std::mutex ledger_mutex;
    return ledger_mutex;
}
inline ankerl::unordered_dense::map<void*, Rec>& ctx_live()
{
    static ankerl::unordered_dense::map<void*, Rec> live;
    return live;
}
inline ankerl::unordered_dense::map<void*, Rec>& op_live()
{
    static ankerl::unordered_dense::map<void*, Rec> live;
    return live;
}

// Standalone rank probe — no PMP_CHUNK dependency to keep include order clean.
inline int current_rank() noexcept
{
    // No directive. The note that used to sit here was right about the problem
    // -- this is not a template, so an `if constexpr` branch is still compiled
    // in a serial build where omp_in_parallel() does not exist -- but the
    // answer is not a directive, it is PMP_CHUNK.h: the shims ARE the serial
    // build's declarations (false and 0), so the expression compiles either way
    // and still yields -1 outside a parallel region.
    return CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_THREAD_RANK : -1;
}

// Master switch for BOTH ledgers (ctx/op above + block below).
//
// Every hook takes a GLOBAL std::mutex plus a global map lookup. The cost is
// not symmetric between build modes, so neither is the default:
//
//   - serial build: the lock is never contended. The overhead is marginal and
//     the detection (double-free, illegal park, leaked LIVE block) is free ->
//     ON by default under any_check.
//   - OpenMP build: the single lock serialises EVERY allocation across all
//     threads. At 16+ threads it dominates run time, but the real problem is
//     that it changes the concurrency the debug build exists to exercise: a
//     race on the allocation path can become unreproducible precisely because
//     the ledger is holding a global lock around it -> OFF by default.
//
// Overrides:
//   -DCHUNK_FORCE_LEDGER : force ON wherever any_check is on, OpenMP included.
//                          Use it when hunting a multi-threaded double-free --
//                          and see the cmake-build-omp-gnu-debug-ledger target,
//                          which exists so this coverage does not silently
//                          vanish from the build matrix.
//   -DCHUNK_NO_LEDGER    : force OFF everywhere, whatever the build mode.
//
// Either way, every other any_check invariant/assert stays active.
#ifdef CHUNK_NO_LEDGER
inline constexpr bool pool_ledger_enabled = false;
#elifdef CHUNK_FORCE_LEDGER
inline constexpr bool pool_ledger_enabled = build_config::any_check;
#else
inline constexpr bool pool_ledger_enabled = build_config::any_check && !build_config::use_openmp;
#endif

inline void on_ctx_alloc(void* ptr, std::size_t sz, const void* slab = nullptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (auto it = op_live().find(ptr); it != op_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] ctx_alloc at {} but block is LIVE as op "
                             "(op_size={}, op_alloc_rank={}, ctx_size={}, op_slab={}, ctx_slab={})",
                             ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab);
        if (auto it = ctx_live().find(ptr); it != ctx_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] ctx_alloc DOUBLE-alloc at {} — block already live as ctx "
                             "(prev_size={}, prev_alloc_rank={}, new_size={}, prev_slab={}, new_slab={}, "
                             "SLABS_{}). Means the ctx pool returned a still-live block → double-recycle "
                             "of this ctx, or the slab handed out a live block.",
                             ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                             (it->second.slab == slab) ? "SAME" : "DIFF");
        ctx_live().emplace(ptr, Rec{.size = sz, .alloc_rank = current_rank(), .slab = slab});
    }
}

inline void on_ctx_free(void* ptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (auto it = op_live().find(ptr); it != op_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] ctx_free at {} but block is LIVE as op "
                             "(op_size={}, op_alloc_rank={})",
                             ptr, it->second.size, it->second.alloc_rank);
        if (ctx_live().erase(ptr) == 0)
            INVARIANT_FAILED("[POOL_LEDGER] ctx_free at {} of an unknown/already-freed ctx block", ptr);
    }
}

inline void on_op_alloc(void* ptr, std::size_t sz, const void* slab = nullptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        // Prime suspect for this bug: a pending op still holds p as its ctx.
        // ctx_slab vs op_slab tells us WHY they alias:
        //   ctx_slab == op_slab  -> ctx & op share one pool (ICF fold / tag fail)
        //   ctx_slab != op_slab  -> two distinct pools returned the same block
        //                           (TCTypeAllocator slab hands out overlapping mem)
        if (auto it = ctx_live().find(ptr); it != ctx_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] op_alloc at {} but block is LIVE as ctx "
                             "(ctx_size={}, ctx_alloc_rank={}, op_size={}, ctx_slab={}, op_slab={}, "
                             "SLABS_{})",
                             ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                             (it->second.slab == slab) ? "SHARED" : "DISTINCT");
        if (auto it = op_live().find(ptr); it != op_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] op_alloc DOUBLE-alloc at {} — block already live as op "
                             "(prev_size={}, prev_alloc_rank={}, new_size={}, prev_slab={}, new_slab={}, "
                             "SLABS_{})",
                             ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                             (it->second.slab == slab) ? "SAME" : "DIFF");
        op_live().emplace(ptr, Rec{.size = sz, .alloc_rank = current_rank(), .slab = slab});
    }
}

inline void on_op_free(void* ptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (auto it = ctx_live().find(ptr); it != ctx_live().end())
            INVARIANT_FAILED("[POOL_LEDGER] op_free at {} but block is LIVE as ctx "
                             "(ctx_size={}, ctx_alloc_rank={})",
                             ptr, it->second.size, it->second.alloc_rank);
        if (op_live().erase(ptr) == 0)
            INVARIANT_FAILED("[POOL_LEDGER] op_free at {} of an unknown/already-freed op block", ptr);
    }
}

// ============================================================================
// BLOCK LEDGER — value-block lifecycle (SEGV hunt on Chunk2 / Chunk2A)
// ----------------------------------------------------------------------------
// Distinct from the ctx/op ledger above: this one tracks the heap chunk-blocks
// that flow through allocate_chunk_internal / recup_value / free_block_to_pool.
// It is the deterministic replacement for the TSan "init_block (memset) <->
// free_block_to_pool (svfree) on the same address" race: instead of sampling a
// race, we trip at the FIRST illegal transition with the address and BOTH ranks.
//
// State per block, with legal transitions (op : from -> to):
//   b_alloc   : FREE   -> LIVE     fresh block from allocator/svcalloc
//   b_pop     : PARKED -> LIVE     popped from a per-thread free-list
//   b_park    : LIVE   -> PARKED   pushed into a per-thread free-list (recup / free_block_to_pool)
//   b_release : LIVE   -> FREE     returned to allocator (free_elt_pow2)
//
// Illegal (each trips the assert, naming previous state + previous rank):
//   double-park  (park a PARKED block)     → two chunks think they own it
//   double-free  (release a FREE block)    → the classic UAF/double-free
//   pop LIVE     (hand a LIVE block twice) → the "init_block <-> free" TSan pair
//   park/pop/release of an UNTRACKED block → foreign or already-recycled ptr
// ============================================================================

enum class BState : std::uint8_t
{
    FREE   = 0,
    LIVE   = 1,
    PARKED = 2
};

inline const char* bstate_name(BState state) noexcept
{
    switch (state)
    {
        case BState::FREE:
            return "FREE";
        case BState::LIVE:
            return "LIVE";
        case BState::PARKED:
            return "PARKED";
    }
    return "??";
}

struct BRec
{
    BState      state     = BState::FREE;
    int         last_rank = -1;
    const char* last_site = "";
};

inline ankerl::unordered_dense::map<const void*, BRec>& block_live()
{
    static ankerl::unordered_dense::map<const void*, BRec> live;
    return live;
}

// ── End-of-run report (diagnostic) ───────────────────────────────────────────
// Lists every block still recorded in block_live(), grouped by (state, last
// site). Enabled only when the ledger is on AND CHUNK_LEDGER_REPORT is set in
// the environment, so it costs nothing otherwise. It separates the causes a
// LeakSanitizer report cannot: a block left LIVE was allocated and never parked
// nor released (its reference was dropped); a block left PARKED was parked but
// not freed by the exit drain; a leaked block absent from the table was never
// seen by the ledger's hooks at all.
inline void report_live_blocks() noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        std::size_t            n_live = 0, n_parked = 0;
        // (state, site) -> count ; a plain array of pairs keeps this free of
        // any extra container dependency.
        struct Row
        {
            int         state;
            const char* site;
            std::size_t count;
        };
        Row rows[64];
        int n_rows = 0;
        for (const auto& kv : block_live())
        {
            const BRec& rec = kv.second;
            if (rec.state == BState::FREE)
                continue;
            if (rec.state == BState::LIVE)
                ++n_live;
            else
                ++n_parked;
            int i = 0;
            for (; i < n_rows; ++i)
                if (rows[i].state == static_cast<int>(rec.state) && rows[i].site == rec.last_site)
                    break;
            if (i == n_rows)
            {
                if (n_rows == 64)
                    continue; // table full: still counted in the totals above
                rows[n_rows++] = Row{static_cast<int>(rec.state), rec.last_site, 0};
            }
            ++rows[i].count;
        }
        std::fprintf(stderr,
                     "\n[POOL_LEDGER/REPORT] blocks still recorded at exit: "
                     "%zu LIVE, %zu PARKED\n",
                     n_live, n_parked);
        for (int i = 0; i < n_rows; ++i)
            std::fprintf(stderr, "[POOL_LEDGER/REPORT]   %-7s %8zu  last site: %s\n",
                         rows[i].state == static_cast<int>(BState::LIVE) ? "LIVE" : "PARKED", rows[i].count,
                         rows[i].site);
        if (n_live == 0 && n_parked == 0)
            std::fprintf(stderr, "[POOL_LEDGER/REPORT]   (none) -- any LSan leak comes "
                                 "from a path the ledger does not trace\n");
        std::fflush(stderr);
    }
}

// Registered once, on the first block the ledger sees.
//
// ORDER MATTERS: block_live() is a function-local static, destroyed at exit in
// the REVERSE order of construction relative to std::atexit registrations. The
// map must therefore be CONSTRUCTED BEFORE the handler is registered, so that
// the handler runs first and reads a live map. Registering first (the naive
// order) destroys the map before the report runs, which then reads freed
// memory and prints arbitrary states -- observed: a LIVE block reported as
// PARKED. The explicit block_live() call below forces construction first.
inline void maybe_register_live_report() noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        static const bool registered = []
        {
            if (std::getenv("CHUNK_LEDGER_REPORT") != nullptr)
            {
                (void)block_live();                        // construct the map FIRST
                std::atexit([] { report_live_blocks(); }); // ...then register
            }
            return true;
        }();
        (void)registered;
    }
}


// Core transition. `allow_unknown` is true only for b_alloc: a never-seen
// pointer is implicitly FREE and becomes LIVE. For pop/park/release, an unknown
// pointer is itself a bug (we lost track of its birth, or it is foreign).
inline void block_transition(const void* block,
                             BState      expected,
                             BState      next,
                             const char* site,
                             const char* op,
                             bool        allow_unknown) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const int rank = current_rank();

        // Guard: a stray non-pointer value leaking in as a "block" key (e.g. a
        // double like 42.0 = 0x4045000000000008) corrupts the ledger map and
        // later SIGSEGVs in hashing. Catch it at the source with the site.
        // Architecture-aware range check:
        //  - x86-64: canonical user pointers have the top 17 bits clear (<2^47).
        //  - aarch64: user virtual addresses legitimately use up to 48 bits (52
        //    with LVA), so the x86 "<2^47" rule false-flags valid high heap keys
        //    (e.g. 0x0000fdcf........). Reject only keys >= 2^52; this still
        //    catches a leaked normal-magnitude double, whose exponent bits (52-62)
        //    are set. (Standard ASan on aarch64 does not tag pointers; if hwasan/
        //    MTE tagging is ever enabled, strip the top byte before this check.)
        {
            const auto up = reinterpret_cast<std::uintptr_t>(block);
#ifdef __aarch64__
            const bool bad_high = (up >> 52) != 0;
#else
            const bool bad_high = (up >> 47) != 0;
#endif
            if (bad_high || (up & 0x7) != 0)
            {
                std::fprintf(stderr, "[LEDGER_BADPTR] non-canonical block key %p op=%s site=%s\n", block, op, site);
                std::fflush(stderr);
                print_backtrace();
                std::abort();
            }
        }

        const std::scoped_lock lk(mtx());

        auto it = block_live().find(block);
#ifdef RECHUNK_LEDGERTRACE
        {
            const char* cur = (it == block_live().end()) ? "ABSENT" : bstate_name(it->second.state);
            std::fprintf(stderr, "[LEDGER] %p op=%s cur=%s -> %s site=%s sz=%zu\n", block, op, cur, bstate_name(next),
                         site, block_live().size());
        }
#endif
        if (it == block_live().end())
        {
            if (!allow_unknown)
                INVARIANT_FAILED("[POOL_LEDGER/BLOCK] {} of UNTRACKED block {} — site='{}' rank={} "
                                 "(expected state {}) → foreign or already-recycled pointer",
                                 op, block, site, rank, bstate_name(expected));
            // Don't record a brand-new entry as FREE: a released unknown block
            // should simply not be tracked (see erase rationale below).
            if (next != BState::FREE)
                block_live().emplace(block, BRec{.state = next, .last_rank = rank, .last_site = site});
            return;
        }

        BRec& info = it->second;
        if (info.state != expected)
            INVARIANT_FAILED("[POOL_LEDGER/BLOCK] ILLEGAL {} of block {} — site='{}' rank={} : "
                             "state={} (expected {}), last op by rank={} at site='{}'",
                             op, block, site, rank, bstate_name(info.state), bstate_name(expected), info.last_rank,
                             info.last_site);

        // Released back to the system (svfree): remove the entry. A released
        // block does not belong in a live-block ledger; malloc may re-hand the
        // same address later, so the next b_alloc on it starts clean.
        if (next == BState::FREE)
        {
            block_live().erase(it);
            return;
        }

        info.state     = next;
        info.last_rank = rank;
        info.last_site = site;
    }
}

inline void on_block_alloc(const void* block, const char* site) noexcept
{
    maybe_register_live_report();
    block_transition(block, BState::FREE, BState::LIVE, site, "b_alloc", /*allow_unknown=*/true);
}
inline void on_block_pop(const void* block, const char* site) noexcept
{
    block_transition(block, BState::PARKED, BState::LIVE, site, "b_pop", /*allow_unknown=*/false);
}
inline void on_block_park(const void* block, const char* site) noexcept
{
    block_transition(block, BState::LIVE, BState::PARKED, site, "b_park", /*allow_unknown=*/false);
}
inline void on_block_release(const void* block, const char* site) noexcept
{
    // allow_unknown=true: a release of a never-tracked address is non-fatal here
    // (foreign/reused block), letting the real svfree proceed so an ASan build
    // still reports the double-free with both stacks. A release of a *tracked*
    // block in the wrong state (e.g. already FREE) still trips the ILLEGAL branch
    // with the first release site, thanks to the FREE tombstone above.
    block_transition(block, BState::LIVE, BState::FREE, site, "b_release", /*allow_unknown=*/true);
}
// Discard a PARKED block (in a reuse free-list) straight back to the system.
// Used when draining a reuse-list whose buffers are being svfree'd: they are
// PARKED (not LIVE), so a plain release would be an illegal transition.
inline void on_block_discard(const void* block, const char* site) noexcept
{
    block_transition(block, BState::PARKED, BState::FREE, site, "b_discard", /*allow_unknown=*/false);
}

} // namespace shrink_array::pool_ledger

// ── Call-site macros for the BLOCK ledger ────────────────────────────────────
// Defined unconditionally; the underlying hooks compile to nothing when
// build_config::any_check is false (same policy as the ctx/op ledger above).
// Grep tag: CHUNK_LEDGER_ / POOL_LEDGER to remove all scaffolding when settled.
// CHUNK_NO_LEDGER: hard off-switch for the block ledger, independent of
// build_config::any_check. Each hook takes a global std::mutex + a global
// map lookup, so under any_check at high thread counts the single lock
// serialises every thread and dominates debug run time. Define CHUNK_NO_LEDGER
// (e.g. -DCHUNK_NO_LEDGER) to compile the hooks to nothing for a fast debug
// build while keeping every other any_check invariant/assert active. The block
// ledger is not the detector for the current double-free (glibc catches that),
// so turning it off loses no coverage for it.
#ifdef CHUNK_NO_LEDGER
    #define CHUNK_LEDGER_ALLOC(p, site) ((void)0)
    #define CHUNK_LEDGER_POP(p, site) ((void)0)
    #define CHUNK_LEDGER_PARK(p, site) ((void)0)
    #define CHUNK_LEDGER_RELEASE(p, site) ((void)0)
    #define CHUNK_LEDGER_DISCARD(p, site) ((void)0)
#else
    #define CHUNK_LEDGER_ALLOC(p, site) ::shrink_array::pool_ledger::on_block_alloc((p), (site))
    #define CHUNK_LEDGER_POP(p, site) ::shrink_array::pool_ledger::on_block_pop((p), (site))
    #define CHUNK_LEDGER_PARK(p, site) ::shrink_array::pool_ledger::on_block_park((p), (site))
    #define CHUNK_LEDGER_RELEASE(p, site) ::shrink_array::pool_ledger::on_block_release((p), (site))
    #define CHUNK_LEDGER_DISCARD(p, site) ::shrink_array::pool_ledger::on_block_discard((p), (site))
#endif
