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

// ============================================================================
// CHECK-BUILD INSTRUMENTATION — block-lifecycle ledger (grep tag: POOL_LEDGER).
// Costs nothing in release; -DCHUNK_NO_LEDGER drops it.
//
// Invariant checked: deferred `ctx` and `op` blocks come from two disjoint pools
// (CtxRegistry vs OpRegistry), so no address is ever live as both. The typical
// trip is op_alloc(p) while p is still live as a ctx: a pending op still
// references p as its ctx, yet the pool just handed p out as an op.
//
// Two GLOBAL address sets (ctx-live, op-live) under one mutex. Not thread_local:
// a cross-thread flush legitimately frees a block on another thread than the one
// that allocated it (see deferred_op_pool.h, reset_all()); a per-thread ledger
// would false-positive there.
//
// All hooks compile to nothing when build_config::any_check is false.
// ============================================================================

#include <array>   // std::array
#include <cstddef> // std::size_t
#include <cstdint> // std::uintptr_t (the aarch64 alignment probe)
#include <cstdio>  // std::fprintf
#include <cstdlib> // std::abort
#include <mutex>   // std::mutex, std::scoped_lock
#include <utility> // std::cmp_equal

#include "chunk/config.h"                   // build_config::any_check
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

#include "chunk/vendor/ankerl/unordered_dense.h" // the live maps

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_IN_PARALLEL_REGION / _THREAD_RANK

namespace shrink_array::pool_ledger
{

/// @brief One record per live ctx / op block: its size, the rank that
///        allocated it and the pool that served it, for the diagnostics when
///        an invariant trips.
struct Rec
{
    std::size_t size;
    int         alloc_rank;
    const void* slab = nullptr; // identity of the pool that served this block (aliasing check)
};

/// @brief The global mutex guarding every ledger map.
inline std::mutex& mtx()
{
    static std::mutex ledger_mutex;
    return ledger_mutex;
}
/// @brief Live ctx blocks, keyed by address.
inline ankerl::unordered_dense::map<void*, Rec>& ctx_live()
{
    // Never destroyed: a global array is destroyed after main, and its buffers
    // go through the ledger then (see chunk/internal/immortal.h).
    static auto* live = new ankerl::unordered_dense::map<void*, Rec>();
    return *live;
}
/// @brief Live op blocks, keyed by address.
inline ankerl::unordered_dense::map<void*, Rec>& op_live()
{
    // Never destroyed: a global array is destroyed after main, and its buffers
    // go through the ledger then (see chunk/internal/immortal.h).
    static auto* live = new ankerl::unordered_dense::map<void*, Rec>();
    return *live;
}

/// @brief The calling thread's rank, or -1 outside a parallel region.
inline int current_rank() noexcept
{
    // No directive needed: PMP_CHUNK.h's serial shims (false, 0) make this
    // compile in both builds, and it yields -1 outside a parallel region.
    return CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_THREAD_RANK : -1;
}

// Master switch for BOTH ledgers (ctx/op above + block below).
//
// Every hook takes a GLOBAL std::mutex plus a global map lookup, so the default
// differs by build mode:
//
//   - serial build: the lock is never contended, the overhead is marginal ->
//     ON by default under any_check.
//   - OpenMP build: the single lock serialises EVERY allocation across threads.
//     Beyond the cost, it changes the concurrency the debug build exists to
//     exercise: a race on the allocation path can hide behind the global lock
//     -> OFF by default.
//
// Overrides:
//   -DCHUNK_FORCE_LEDGER : force ON wherever any_check is on, OpenMP included
//                          (scripts/compile_all.sh --with-ledger builds the
//                          cmake-build-omp-gnu-debug-ledger config with it).
//   -DCHUNK_NO_LEDGER    : force OFF everywhere, whatever the build mode.
//
// Either way, every other any_check invariant/assert stays active.
#ifdef CHUNK_NO_LEDGER
inline constexpr bool pool_ledger_enabled = false;
#elif CHUNK_OPT_FORCE_LEDGER
inline constexpr bool pool_ledger_enabled = build_config::any_check;
#else
inline constexpr bool pool_ledger_enabled = build_config::any_check && !build_config::use_openmp;
#endif

/// @brief Record a ctx block allocation; fails an invariant if @p ptr is already live as ctx or as op.
/// @param ptr Block address.
/// @param sz Block size.
/// @param slab Identity of the pool that served the block.
inline void on_ctx_alloc(void* ptr, std::size_t sz, const void* slab = nullptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (const auto it = op_live().find(ptr); it != op_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] ctx_alloc at {} but block is LIVE as op "
                                   "(op_size={}, op_alloc_rank={}, ctx_size={}, op_slab={}, ctx_slab={})",
                                   ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab);
        }
        if (const auto it = ctx_live().find(ptr); it != ctx_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] ctx_alloc DOUBLE-alloc at {} — block already live as ctx "
                                   "(prev_size={}, prev_alloc_rank={}, new_size={}, prev_slab={}, new_slab={}, "
                                   "SLABS_{}). Means the ctx pool returned a still-live block → double-recycle "
                                   "of this ctx, or the slab handed out a live block.",
                                   ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                                   (it->second.slab == slab) ? "SAME" : "DIFF");
        }
        ctx_live().emplace(ptr, Rec{.size = sz, .alloc_rank = current_rank(), .slab = slab});
    }
}

/// @brief Record a ctx block release; fails an invariant if @p ptr is live as op or not live as ctx.
/// @param ptr Block address.
inline void on_ctx_free(void* ptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (const auto it = op_live().find(ptr); it != op_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] ctx_free at {} but block is LIVE as op "
                                   "(op_size={}, op_alloc_rank={})",
                                   ptr, it->second.size, it->second.alloc_rank);
        }
        if (ctx_live().erase(ptr) == 0)
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] ctx_free at {} of an unknown/already-freed ctx block", ptr);
        }
    }
}

/// @brief Record an op block allocation; fails an invariant if @p ptr is already live as ctx or as op.
/// @param ptr Block address.
/// @param sz Block size.
/// @param slab Identity of the pool that served the block.
inline void on_op_alloc(void* ptr, std::size_t sz, const void* slab = nullptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        // A pending op still holds p as its ctx. ctx_slab vs op_slab tells WHY
        // they alias:
        //   ctx_slab == op_slab  -> ctx & op share one pool (ICF fold / tag fail)
        //   ctx_slab != op_slab  -> two distinct pools returned the same block
        //                           (TCTypeAllocator slab hands out overlapping mem)
        if (const auto it = ctx_live().find(ptr); it != ctx_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] op_alloc at {} but block is LIVE as ctx "
                                   "(ctx_size={}, ctx_alloc_rank={}, op_size={}, ctx_slab={}, op_slab={}, "
                                   "SLABS_{})",
                                   ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                                   (it->second.slab == slab) ? "SHARED" : "DISTINCT");
        }
        if (const auto it = op_live().find(ptr); it != op_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] op_alloc DOUBLE-alloc at {} — block already live as op "
                                   "(prev_size={}, prev_alloc_rank={}, new_size={}, prev_slab={}, new_slab={}, "
                                   "SLABS_{})",
                                   ptr, it->second.size, it->second.alloc_rank, sz, it->second.slab, slab,
                                   (it->second.slab == slab) ? "SAME" : "DIFF");
        }
        op_live().emplace(ptr, Rec{.size = sz, .alloc_rank = current_rank(), .slab = slab});
    }
}

/// @brief Record an op block release; fails an invariant if @p ptr is live as ctx or not live as op.
/// @param ptr Block address.
inline void on_op_free(void* ptr) noexcept
{
    if constexpr (pool_ledger_enabled)
    {
        const std::scoped_lock lk(mtx());
        if (const auto it = ctx_live().find(ptr); it != ctx_live().end())
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] op_free at {} but block is LIVE as ctx "
                                   "(ctx_size={}, ctx_alloc_rank={})",
                                   ptr, it->second.size, it->second.alloc_rank);
        }
        if (op_live().erase(ptr) == 0)
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER] op_free at {} of an unknown/already-freed op block", ptr);
        }
    }
}

// ============================================================================
// BLOCK LEDGER — value-block lifecycle (check builds, Chunk2 / Chunk2A)
// ----------------------------------------------------------------------------
// Distinct from the ctx/op ledger above: this one tracks the heap chunk-blocks
// that flow through allocate_chunk_internal / recup_value / free_block_to_pool.
// It trips deterministically at the FIRST illegal transition, with the address
// and BOTH ranks, where a race detector only samples the resulting race.
//
// State per block, with legal transitions (op : from -> to):
//   b_alloc   : FREE   -> LIVE     fresh block from allocator/svcalloc
//   b_pop     : PARKED -> LIVE     popped from a per-thread free-list
//   b_park    : LIVE   -> PARKED   pushed into a per-thread free-list (recup / free_block_to_pool)
//   b_release : LIVE   -> FREE     returned to allocator (free_elt_pow2)
//
// Illegal (each trips the assert, naming previous state + previous rank):
//   double-park  (park a PARKED block)     → two chunks think they own it
//   pop LIVE     (hand a LIVE block twice) → one chunk inits a block another frees
//   park/pop/discard of an UNTRACKED block → foreign or already-recycled ptr
// A released block leaves the table (malloc may hand its address out again),
// so a second release of it is not seen here: glibc / ASan report that one.
// ============================================================================

/// @brief A value block's state in the block ledger.
enum class BState : std::uint8_t
{
    FREE   = 0,
    LIVE   = 1,
    PARKED = 2
};

/// @brief Name of a block state.
/// @return "FREE", "LIVE", "PARKED", or "??" for any other value.
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

/// @brief Block ledger record: current state and the rank and site of the last transition.
struct BRec
{
    BState      state     = BState::FREE;
    int         last_rank = -1;
    const char* last_site = "";
};

/// @brief Tracked value blocks, keyed by address.
inline ankerl::unordered_dense::map<const void*, BRec>& block_live()
{
    static auto* live = new ankerl::unordered_dense::map<const void*, BRec>(); // never destroyed, as above
    return *live;
}

/// @brief End-of-run report: every block still in the ledger, by (state, last site).
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
        std::size_t            n_live   = 0;
        std::size_t            n_parked = 0;
        // (state, site) -> count ; a plain array of pairs keeps this free of
        // any extra container dependency.
        struct Row
        {
            int         state;
            const char* site;
            std::size_t count;
        };
        std::array<Row, 64> rows{};
        int                 n_rows = 0;
        for (const auto& kv : block_live())
        {
            const BRec& rec = kv.second;
            if (rec.state == BState::FREE)
            {
                continue;
            }
            if (rec.state == BState::LIVE)
            {
                ++n_live;
            }
            else
            {
                ++n_parked;
            }
            int i = 0;
            for (; i < n_rows; ++i)
            {
                if (rows[i].state == static_cast<int>(rec.state) && rows[i].site == rec.last_site)
                {
                    break;
                }
            }
            if (i == n_rows)
            {
                if (std::cmp_equal(n_rows, rows.size()))
                {
                    continue; // table full: still counted in the totals above
                }
                rows[n_rows++] = Row{.state = static_cast<int>(rec.state), .site = rec.last_site, .count = 0};
            }
            ++rows[i].count;
        }
        std::fprintf(stderr,
                     "\n[POOL_LEDGER/REPORT] blocks still recorded at exit: "
                     "%zu LIVE, %zu PARKED\n",
                     n_live, n_parked);
        for (int i = 0; i < n_rows; ++i)
        {
            std::fprintf(stderr, "[POOL_LEDGER/REPORT]   %-7s %8zu  last site: %s\n",
                         rows[i].state == static_cast<int>(BState::LIVE) ? "LIVE" : "PARKED", rows[i].count,
                         rows[i].site);
        }
        if (n_live == 0 && n_parked == 0)
        {
            std::fprintf(stderr, "[POOL_LEDGER/REPORT]   (none) -- any LSan leak comes "
                                 "from a path the ledger does not trace\n");
        }
        std::fflush(stderr);
    }
}

/// @brief Register report_live_blocks() at exit (CHUNK_LEDGER_REPORT), once,
///        on the first block the ledger sees.
//
// ORDER MATTERS: function-local statics and std::atexit handlers run in the
// REVERSE order of construction/registration. Construct block_live() BEFORE
// registering the handler, so the handler runs first on a live map; the other
// order makes the report read freed memory and print arbitrary states.
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


/// @brief Core transition of the block ledger: @p block must be in state
///        @p expected and moves to @p next (FREE: leaves the table).
// `allow_unknown` is true for b_alloc -- a never-seen pointer is implicitly
// FREE and becomes LIVE -- and for b_release (see on_block_release). For
// pop/park/discard, an unknown pointer is itself a bug (we lost track of its
// birth, or it is foreign).
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

        // Guard: a stray non-pointer value used as a "block" key (e.g. a double
        // like 42.0 = 0x4045000000000008) would corrupt the map; catch it here,
        // with the site. Architecture-aware range check:
        //  - x86-64: canonical user pointers are < 2^47.
        //  - aarch64: user addresses use up to 48 bits (52 with LVA), so the x86
        //    rule false-flags valid heap keys. Reject only keys >= 2^52, which
        //    still catches a normal-magnitude double (exponent bits 52-62 set).
        //    If hwasan/MTE tagging is ever enabled, strip the top byte first.
        {
            const auto up = reinterpret_cast<std::uintptr_t>(block);
#ifdef __aarch64__
            const bool bad_high = (up >> 52U) != 0;
#else
            const bool bad_high = (up >> 47U) != 0;
#endif
            if (bad_high || (up & 0x7U) != 0)
            {
                CHUNK_INVARIANT_FAILED("[LEDGER_BADPTR] non-canonical block key {} op={} site={}",
                                       static_cast<const void*>(block), op, site);
            }
        }

        const std::scoped_lock lk(mtx());

        const auto it = block_live().find(block);
        if constexpr (build_config::rechunk_ledger_trace)
        {
            const char* cur = (it == block_live().end()) ? "ABSENT" : bstate_name(it->second.state);
            std::fprintf(stderr, "[LEDGER] %p op=%s cur=%s -> %s site=%s sz=%zu\n", block, op, cur, bstate_name(next),
                         site, block_live().size());
        }
        if (it == block_live().end())
        {
            if (!allow_unknown)
            {
                CHUNK_INVARIANT_FAILED("[POOL_LEDGER/BLOCK] {} of UNTRACKED block {} — site='{}' rank={} "
                                       "(expected state {}) → foreign or already-recycled pointer",
                                       op, block, site, rank, bstate_name(expected));
            }
            // Don't record a brand-new entry as FREE: a released unknown block
            // should simply not be tracked (see erase rationale below).
            if (next != BState::FREE)
            {
                block_live().emplace(block, BRec{.state = next, .last_rank = rank, .last_site = site});
            }
            return;
        }

        BRec& info = it->second;
        if (info.state != expected)
        {
            CHUNK_INVARIANT_FAILED("[POOL_LEDGER/BLOCK] ILLEGAL {} of block {} — site='{}' rank={} : "
                                   "state={} (expected {}), last op by rank={} at site='{}'",
                                   op, block, site, rank, bstate_name(info.state), bstate_name(expected),
                                   info.last_rank, info.last_site);
        }

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

/// @brief Record a fresh block (FREE -> LIVE); an untracked block is accepted.
/// @param block Block address.
/// @param site Label of the calling site.
inline void on_block_alloc(const void* block, const char* site) noexcept
{
    maybe_register_live_report();
    block_transition(block, BState::FREE, BState::LIVE, site, "b_alloc", /*allow_unknown=*/true);
}
/// @brief Record a block popped from a free-list (PARKED -> LIVE).
/// @param block Block address.
/// @param site Label of the calling site.
inline void on_block_pop(const void* block, const char* site) noexcept
{
    block_transition(block, BState::PARKED, BState::LIVE, site, "b_pop", /*allow_unknown=*/false);
}
/// @brief Record a block pushed into a free-list (LIVE -> PARKED).
/// @param block Block address.
/// @param site Label of the calling site.
inline void on_block_park(const void* block, const char* site) noexcept
{
    block_transition(block, BState::LIVE, BState::PARKED, site, "b_park", /*allow_unknown=*/false);
}
/// @brief Record a block returned to the allocator (LIVE -> FREE); an untracked block is accepted.
/// @param block Block address.
/// @param site Label of the calling site.
inline void on_block_release(const void* block, const char* site) noexcept
{
    // allow_unknown=true: a release of a never-tracked address is non-fatal here
    // (foreign/reused block, or a second release: the first one removed the
    // entry), letting the real svfree proceed so an ASan build still reports
    // the double-free with both stacks. A release of a tracked PARKED block
    // still trips the ILLEGAL branch, with the site of its last transition.
    block_transition(block, BState::LIVE, BState::FREE, site, "b_release", /*allow_unknown=*/true);
}
/// @brief Record a parked block freed straight from its reuse list (PARKED -> FREE).
/// @param block Block address.
/// @param site Label of the calling site.
// Used when draining a reuse-list whose buffers are being svfree'd: they are
// PARKED (not LIVE), so a plain release would be an illegal transition.
inline void on_block_discard(const void* block, const char* site) noexcept
{
    block_transition(block, BState::PARKED, BState::FREE, site, "b_discard", /*allow_unknown=*/false);
}

} // namespace shrink_array::pool_ledger

// ── Call-site macros for the BLOCK ledger ────────────────────────────────────
// The hooks compile to nothing when build_config::any_check is false.
// CHUNK_NO_LEDGER compiles them to nothing regardless (a fast debug build that
// keeps every other any_check invariant); a double free is still reported by
// glibc / ASan (see the table above).
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
