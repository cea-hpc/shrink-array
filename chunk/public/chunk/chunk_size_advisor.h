// SPDX-License-Identifier: Apache-2.0
#pragma once

// ============================================================================
// ChunkSizeAdvisor — STEP 0: pure observation, zero mutation, zero hot-path.
// ----------------------------------------------------------------------------
// Estimates, for one array, the model memory footprint it WOULD have at other
// chunk sizes, without re-chunking anything:
//
//     mem(S) ≈ n_chunks(S) * meta_bytes_per_chunk  +  U_mat(S) * S * sizeof(T)
//     (S / 8 bytes per materialised chunk for Bit, ArrayChunkBits: storage_units)
//
//   meta term  : fixed per-chunk cost (sizeof(chunk) + tracking list + dirty
//                bitset + slack), pushes toward LARGER chunks.
//   data term  : U_mat(S) = number of chunks that stay MATERIALIZED at size S
//                (one exceptional value pollutes a whole chunk), pushes toward
//                SMALLER chunks.
//
// How U_mat(S) is obtained without re-chunking:
//   * COARSER sizes (S*2, S*4, ...): exact fold over current chunk STATES.
//     A merged chunk is compressible iff all its children are the SAME
//     constant. For Chunk2A the identity of a cst_val2 constant IS the
//     values_load() pointer (a TabCstVal2Holder slot), so the rule is pure
//     pointer comparison:
//         cst1 + cst1           -> cst1
//         cst2(p) + cst2(p)     -> cst2
//         anything else         -> materialized
//     One pass over chunk_list, all levels folded simultaneously. NOTE: this
//     rule is deliberately CONSERVATIVE for Chunk2A's mixed dual-value
//     patterns; the demo harness cross-checks it against ground truth.
//   * FINER sizes (S/2, S/4): only materialized chunks can improve; their
//     DATA is scanned (sampled, capped) to count constant sub-blocks, then
//     extrapolated. Constant-at-finer sub-blocks are counted as compressed
//     regardless of which constant (cst1 vs cst2 registry growth is a
//     second-order cost, ignored here and documented).
//
// CONCURRENCY CONTRACT: call at a QUIESCENT phase boundary only (same contract
// as flush_pending_deferred_ops: no concurrent writers/compressors, deferred
// ops drained). The walk reads chunk states and, for finer sizes, chunk data.
// ============================================================================

#include <cmath>   // std::lround (the extrapolated materialised count)
#include <cstddef> // std::size_t
#include <cstdint>
#include <cstdio> // std::printf (print)
#include <vector>

#include "chunk/internal/in_t.h" // in_t<T>: the element snapshots
#include "chunk/bit.h"                    // is_bool_like_v, is_packed_bit_v
#include "chunk/internal/element_access.h" // detail::as_words: Bit sub-blocks scanned by word
#include "chunk/config.h"        // chunk_config::use_chunk2 / use_chunk2a
#include "chunk/internal/rechunk_cost_model.h"
#include "chunk/internal/chunk_size_policy.h" // shrink_array::cst_buffer_size

namespace shrink_array
{

struct ChunkSizeEstimate
{
    uint32_t size      = 0;     // candidate chunk size (elements)
    int64_t  n_chunks  = 0;     // total chunks at that size
    int64_t  mat       = 0;     // materialized chunks (estimated or exact)
    int64_t  cst       = 0;     // compressed chunks (cst_val1 + cst_val2)
    double   mem_bytes = 0.;    // model memory
    bool     exact     = false; // true: exact fold; false: sampled estimate
};

struct ChunkSizeReport
{
    uint32_t                       current_size = 0;
    double                         meta_bytes   = 0.;
    std::vector<ChunkSizeEstimate> table;     // ascending sizes
    int                            best = -1; // index into table
    int                            cur  = -1; // index of current size

    /// @brief Bytes the best size would save over the current one (0 when
    ///        either is unknown).
    [[nodiscard]] double best_gain_bytes() const noexcept
    {
        if (best < 0 || cur < 0)
            return 0.;
        return table[cur].mem_bytes - table[best].mem_bytes;
    }
};

template <typename T>
class ChunkSizeAdvisor
{
  public:
    struct Options
    {
        int     coarser_levels = 4;     // evaluate S*2 .. S*2^levels
        int     finer_levels   = 2;     // evaluate S/2 .. S/2^levels (data sampled)
        int64_t max_mat_scans  = 65536; // cap on materialized chunks whose DATA is read
        // Per-chunk metadata beyond sizeof(chunk): tracking-list entry (~0.8 B,
        // see the GLOBAL COMPRESSION LISTS monitoring table: List Memory /
        // Allocated Chunks) + dirty bitset (128 B / 64 chunks = 2 B) + slack.
        double   meta_extra_bytes = 3.0;
        uint32_t min_size         = 64; // don't evaluate below this. Raised from
                                        // 16: the cost model under-counts real
                                        // per-chunk overhead (slab block +
                                        // fragmentation, not just meta_bytes),
                                        // so it over-shrinks toward tiny chunks
                                        // that fragment. 64 caps the descent at
                                        // a sane overhead ratio.
    };

    /// @brief Estimate, from @p tab's constant runs, the memory each
    ///        candidate chunk size would give it. Read-only; call at a
    ///        quiescent point. Variant-generic via @p Arr (ArrayChunk /
    ///        ArrayChunk2 / ArrayChunk2A): the only per-variant part is the
    ///        cst2 classification, selected with if constexpr on chunk_config.
    template <typename Arr>
    static ChunkSizeReport analyze(const Arr& tab, const Options& opt = {})
    {
        ChunkSizeReport rep;

        const uint32_t S  = tab.chunk_size();
        const int64_t  nc = tab.nb_chunk;
        rep.current_size  = S;
        rep.meta_bytes    = static_cast<double>(sizeof(typename Arr::chunk_type)) + opt.meta_extra_bytes;

        if (nc <= 0)
            return rep;

        // Chunk2 keys cst2 by the array-level second-value index; fetch it once.
        // v1 has no cst2; 2A keys cst2 per chunk by the values_load() pointer.
        [[maybe_unused]] const int chunk2_ind = [&]
        {
            // bool / Bit: one second value, `true`, no index --
            // get_ind_second_value() aborts for them; classify() below asks
            // the unindexed is_cst_val2().
            if constexpr (chunk_config::use_chunk2 && !is_bool_like_v<T>)
                return tab.get_ind_second_value();
            else if constexpr (chunk_config::use_chunk2)
                return 0;
            else
                return -1;
        }();

        // Classify one chunk: kind 0 = cst1, 1 = cst2, 2 = mat; `key`
        // distinguishes cst2 constants for the coarser fold (two cst2 children
        // merge iff same key). 2A: per-chunk value -> key = buffer ptr. Chunk2:
        // one array-level value -> all cst2 share a sentinel key. v1: no cst2.
        // Which constant values this variant can actually store WITHOUT keeping
        // the buffer. Mirrors classify() below, at value level rather than at
        // chunk-state level: classify() asks "is this chunk already constant?",
        // this asks "would a uniform run of THIS value be storable as one?".
        //
        // v1 has a single designated constant (first_value). chunk2/2A add a
        // second one, per array; when it is not yet designated the second slot
        // is simply unavailable and only first_value counts -- which is the
        // conservative answer, and the right one for a predictor whose errors
        // have been one-sided so far.
        const in_t<T> cst1_value = Arr::chunk_type::first_value;
        bool          has_cst2   = false;
        T             cst2_value = cst1_value;
        if constexpr ((chunk_config::use_chunk2 || chunk_config::use_chunk2a) && requires { tab.get_second_value(); })
        {
            cst2_value = tab.get_second_value();
            has_cst2   = !(cst2_value == cst1_value);
        }
        const auto compressible_value = [&]([[maybe_unused]] const T& elem) -> bool
        {
            // Chunk2A: any uniform value is compressible -- each chunk
            // registers its own second value, and compress() compresses a
            // chunk uniform at ANY value to that value (find_or_set). Its
            // arrays have no array-level second value, so the gate below kept
            // first_value only: the finer rows missed every other uniform
            // sub-block, and a coarsening that turned out costly (writes
            // materialising whole large chunks) could never look worse than
            // refining back (tab1 stuck at "best=1024" while its data grew
            // 406 -> 565 MB over six ticks at S=512).
            if constexpr (chunk_config::use_chunk2a)
                return true;
            else
            {
                if (elem == cst1_value)
                    return true;
                return has_cst2 && (elem == cst2_value);
            }
        };

        const auto classify = [&](const auto& chunk, const void*& key) -> int
        {
            key = nullptr;
            if (chunk.is_cst_val1())
                return 0;
            if constexpr (chunk_config::use_chunk2a)
            {
                if (chunk.is_cst_val2())
                {
                    key = static_cast<const void*>(chunk.values_load());
                    return 1;
                }
            }
            else if constexpr (chunk_config::use_chunk2)
            {
                bool is_cst2 = false;
                if constexpr (is_bool_like_v<T>)
                    is_cst2 = chunk.is_cst_val2();
                else
                    is_cst2 = chunk2_ind >= 0 && chunk.is_cst_val2(chunk2_ind);
                if (is_cst2)
                {
                    static constexpr int cst2_sentinel = 0; // one address: array-level, all cst2 equal
                    key                                = &cst2_sentinel;
                    return 1;
                }
            }
            // v1: cst1 or mat only.
            return 2;
        };

        // ── Single pass: classify current states and fold all coarser levels ──
        // Level l aggregates blocks of 2^l consecutive chunks. A block aggregate
        // is {kind, key}: kind CST1 / CST2(key) / MAT; folding a child into a
        // block degrades it per the merge rule above.
        // Classify every chunk ONCE, in parallel. This is the expensive part
        // (is_cst_val1 / is_cst_val2 / values_load per chunk over up to millions
        // of chunks) and it is strictly PURE-READ: those accessors are const and
        // mutate nothing (is_cst_val2 only compares a pointer to a stable shared
        // sentinel), so concurrent reads are race-free. We store each chunk's
        // (kind, key) so the coarser fold below can stay SERIAL and byte-identical
        // (it reads the arrays instead of re-classifying). Only this classification
        // is parallel; the sequential fold is never parallelised. analyze() is
        // called at a quiescent serial point, so spawning a region here is safe.
        std::vector<std::uint8_t> kindv(static_cast<std::size_t>(nc));
        std::vector<const void*>  keyv(static_cast<std::size_t>(nc), nullptr);

        int64_t mat0 = 0;

        // ── Coarser fold and finer sampling: parallel too, same results.
        // Both used to run SERIALLY after the classification, and the sampling
        // re-classified every chunk (classify() again, kindv already held the
        // answer): the master thread spent 3.7 s of a 17 s run here (four arrays,
        // 64 threads, the other 63 idle). Both are now shared out, and every
        // count they produce is an integer computed exactly as before, so the
        // report is byte-identical:
        //  - the fold is split into super-blocks of 2^L chunks: every level-l
        //    block (l <= L) lies inside one super-block, so each super-block
        //    folds independently with its own run[] and the per-level counts are
        //    summed;
        //  - the sampling SELECTION stays sequential (it depends on the running
        //    count of materialised chunks) but reads kindv, and only the chosen
        //    chunks' DATA are scanned in parallel.
        const int coarser = opt.coarser_levels;
        struct Agg
        {
            int         kind;
            const void* key;
        }; // kind: 0 cst1, 1 cst2, 2 mat
        std::vector<int64_t> mat_at(static_cast<std::size_t>(coarser) + 1, 0); // materialized blocks per level

        const int            finer = opt.finer_levels;
        std::vector<int64_t> fine_cst(static_cast<std::size_t>(finer) + 1, 0); // constant sub-blocks found
        int64_t              scanned = 0;
        std::vector<int64_t> picks; // chunks whose data the finer sampling reads

        const int64_t super   = int64_t{1} << (coarser > 0 ? coarser : 0);
        const int64_t n_super = (nc + super - 1) / super;

#pragma omp parallel if (nc > 16384)
        {
#pragma omp for schedule(static) reduction(+ : mat0)
            for (int64_t j = 0; j < nc; ++j)
            {
                const void* key                    = nullptr;
                const int   kind                   = classify(tab.chunk_list[j], key);
                kindv[static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(kind);
                keyv[static_cast<std::size_t>(j)]  = key;
                if (kind != 0 && kind != 1)
                {
                    ++mat0;
                }
            }
            // (implicit barrier: kindv, keyv and mat0 are complete)

            // Coarser fold, one super-block of 2^L chunks per iteration.
            {
                std::vector<Agg>     run(static_cast<std::size_t>(coarser) + 1, Agg{0, nullptr});
                std::vector<int64_t> mat_loc(static_cast<std::size_t>(coarser) + 1, 0);
#pragma omp for schedule(static) nowait
                for (int64_t s = 0; s < n_super; ++s)
                {
                    const int64_t j0 = s * super;
                    const int64_t j1 = (j0 + super) < nc ? (j0 + super) : nc;
                    for (int64_t j = j0; j < j1; ++j)
                    {
                        const int   kind = kindv[static_cast<std::size_t>(j)];
                        const void* key  = keyv[static_cast<std::size_t>(j)];

                        for (int l = 1; l <= coarser; ++l)
                        {
                            Agg& agg = run[static_cast<std::size_t>(l)];
                            if ((j & ((int64_t{1} << l) - 1)) == 0)
                                agg = {kind, key}; // block opens with this child
                            else if (agg.kind != 2 && (agg.kind != kind || agg.key != key))
                                agg.kind = 2; // mixed constants -> materialized
                            if (((j + 1) & ((int64_t{1} << l) - 1)) == 0 || j + 1 == nc)
                                if (agg.kind == 2)
                                    ++mat_loc[static_cast<std::size_t>(l)]; // commit block at boundary
                        }
                    }
                }
                for (int l = 1; l <= coarser; ++l)
                    if (mat_loc[static_cast<std::size_t>(l)] != 0)
                    {
#pragma omp atomic
                        mat_at[static_cast<std::size_t>(l)] += mat_loc[static_cast<std::size_t>(l)];
                    }
            }

            // ── Finer sizes: sample the DATA of materialized chunks ───────────
            // Selection: the same chunks as the former sequential loop -- every
            // stride-th materialised chunk, at most max_mat_scans -- read from
            // kindv instead of classify()ing every chunk again.
#pragma omp single
            {
                if (finer > 0 && mat0 > 0 && S >= opt.min_size * 2)
                {
                    const int64_t stride =
                        mat0 > opt.max_mat_scans ? (mat0 + opt.max_mat_scans - 1) / opt.max_mat_scans : 1;
                    int64_t seen = 0;
                    picks.reserve(static_cast<std::size_t>(mat0 < opt.max_mat_scans ? mat0 : opt.max_mat_scans));
                    for (int64_t j = 0; j < nc && static_cast<int64_t>(picks.size()) < opt.max_mat_scans; ++j)
                    {
                        if (kindv[static_cast<std::size_t>(j)] != 2) // skip cst1 / cst2 (only mat has data)
                            continue;
                        if ((seen++ % stride) != 0)
                            continue;
                        picks.push_back(j);
                    }
                    scanned = static_cast<int64_t>(picks.size());
                }
            }
            // (implicit barrier: picks is complete)

            {
                std::vector<int64_t> fine_loc(static_cast<std::size_t>(finer) + 1, 0);
                const auto           n_picks = static_cast<int64_t>(picks.size());
#pragma omp for schedule(static) nowait
                for (int64_t p = 0; p < n_picks; ++p)
                {
                    const auto& chunk = tab.chunk_list[picks[static_cast<std::size_t>(p)]];
                    const T*    v     = chunk.values_load();
                    for (int k = 1; k <= finer; ++k)
                    {
                        const uint32_t sub = S >> k;
                        if (sub < opt.min_size)
                            break;
                        for (uint32_t b = 0; b < S; b += sub)
                        {
                            bool cst = true;
                            T    run{}; // the sub-block's value, when uniform
                            if constexpr (is_packed_bit_v<T>)
                            {
                                // Bit: v is words of bits (bit.h), a sub-block
                                // is a bit range -- classified word by word.
                                const bits::Uniform u = bits::scan_range(detail::as_words(v), b, sub);
                                cst                   = u != bits::Uniform::mixed;
                                run                   = T{u == bits::Uniform::all_true};
                            }
                            else
                            {
                                for (uint32_t i = 1; i < sub; ++i)
                                    if (!(v[b + i] == v[b]))
                                    {
                                        cst = false;
                                        break;
                                    }
                                if (cst)
                                    run = v[b];
                            }
                            // A sub-block that is merely uniform is NOT compressible:
                            // the runtime only ever stores a chunk as a constant when
                            // that constant is one it can REPRESENT -- cst_val1 (the
                            // designated first value), plus cst_val2 for chunk2/2A.
                            // A run of 7.0 stays materialised in v1 whatever its
                            // length. Counting every uniform run made the model
                            // predict that almost everything would compress after a
                            // split: on tab1 it forecast a 5743 MB gain where the
                            // measured one was 251 MB, and the decider acted on it.
                            // Gate on the value, not just on the uniformity.
                            if (cst)
                                cst = compressible_value(run);
                            fine_loc[static_cast<std::size_t>(k)] += cst ? 1 : 0;
                        }
                    }
                }
                for (int k = 1; k <= finer; ++k)
                    if (fine_loc[static_cast<std::size_t>(k)] != 0)
                    {
#pragma omp atomic
                        fine_cst[static_cast<std::size_t>(k)] += fine_loc[static_cast<std::size_t>(k)];
                    }
            }
        }

        // ── Assemble the table (ascending sizes) ──────────────────────────────
        auto push = [&](uint32_t size, int64_t nchunks, int64_t mat, bool exact)
        {
            ChunkSizeEstimate est;
            est.size      = size;
            est.n_chunks  = nchunks;
            est.mat       = mat;
            est.cst       = nchunks - mat;
            est.exact     = exact;
            est.mem_bytes = (static_cast<double>(nchunks) * rep.meta_bytes) +
                            (static_cast<double>(mat) * static_cast<double>(Arr::chunk_type::storage_units(size)) *
                             sizeof(T)); // Bit: size / 8 bytes, not size
            rep.table.push_back(est);
        };

        for (int k = finer; k >= 1; --k)
        {
            const uint32_t sub = S >> k;
            if (sub < opt.min_size)
                continue;
            // constant fraction among SCANNED mat sub-blocks, extrapolated to all
            const int64_t sub_per_chunk = int64_t{1} << k;
            const int64_t scanned_subs  = scanned * sub_per_chunk;
            const double  frac_cst = scanned_subs > 0 ? static_cast<double>(fine_cst[static_cast<std::size_t>(k)]) /
                                                            static_cast<double>(scanned_subs)
                                                      : 0.;
            const int64_t nchunks  = nc * sub_per_chunk; // cst chunks split into cst children
            const auto    mat      = static_cast<int64_t>(
                std::lround(static_cast<double>(mat0) * static_cast<double>(sub_per_chunk) * (1. - frac_cst)));
            push(sub, nchunks, mat, /*exact=*/false);
        }

        rep.cur = static_cast<int>(rep.table.size());
        push(S, nc, mat0, /*exact=*/true);

        for (int l = 1; l <= coarser; ++l)
        {
            // Never propose a size the shared cst buffers can't cover: rechunk_to
            // would refuse it (cst_buffer_size ceiling). Stop the coarser walk here.
            if ((S << l) > shrink_array::cst_buffer_size)
                break;
            const int64_t nchunks = (nc + (int64_t{1} << l) - 1) >> l;
            push(S << l, nchunks, mat_at[static_cast<std::size_t>(l)], /*exact=*/true);
        }

        rep.best = 0;

        for (std::size_t i = 1; i < rep.table.size(); ++i)
            if (rep.table[i].mem_bytes < rep.table[static_cast<std::size_t>(rep.best)].mem_bytes)
                rep.best = static_cast<int>(i);
        return rep;
    }

    /// @brief Print @p rep as a table on stdout, headed by @p name.
    static void print(const ChunkSizeReport& rep, const char* name)
    {
        std::printf("── ChunkSizeAdvisor [%s] current S=%u, meta=%.1f B/chunk ──\n", name, rep.current_size,
                    rep.meta_bytes);
        for (std::size_t i = 0; i < rep.table.size(); ++i)
        {
            const auto& row = rep.table[i];
            std::printf("   S=%5u : chunks=%9lld  mat=%9lld (%5.1f%%)  mem=%9.2f Mo %s%s%s\n", row.size,
                        static_cast<long long>(row.n_chunks), static_cast<long long>(row.mat),
                        row.n_chunks ? 100. * static_cast<double>(row.mat) / static_cast<double>(row.n_chunks) : 0.,
                        row.mem_bytes / (1024. * 1024.), row.exact ? "" : "~",
                        static_cast<int>(i) == rep.cur ? " <- current" : "",
                        static_cast<int>(i) == rep.best ? "  ** BEST **" : "");
        }
        std::printf("   estimated gain best vs current : %.2f Mo\n", rep.best_gain_bytes() / (1024. * 1024.));

        // Re-chunk cost (phase 2 model): only meaningful when best != current.
        // Uses the materialized-chunk counts the table already holds. The gain
        // is memory (bytes), the cost is time (ms) — different units, shown side
        // by side so the reader can judge the trade-off. Constants are hardware
        // specific; calibrate_rechunk_cost<T>() should be called once on the
        // target, else these defaults are a (pessimistic) x86 proxy.
        if (rep.best >= 0 && rep.cur >= 0 && rep.best != rep.cur)
        {
            const auto&  cur_e      = rep.table[static_cast<std::size_t>(rep.cur)];
            const auto&  best_e     = rep.table[static_cast<std::size_t>(rep.best)];
            const double cost_ms    = rechunk_cost_ms(cur_e.mat, cur_e.size, best_e.mat, best_e.size);
            const auto   gain_bytes = static_cast<std::size_t>(rep.best_gain_bytes() > 0 ? rep.best_gain_bytes() : 0.);
            // Observation-phase thresholds (not a decision, just a hint label):
            // >= 1 Mo gained and <= 50 ms spent reads as clearly worth it.
            const bool worth = rechunk_is_worth_it(gain_bytes, cost_ms, std::size_t{1} << 20, 50.0);
            std::printf("   estimated re-chunk cost       : %.3f ms  (%s)\n", cost_ms, worth ? "worth it" : "marginal");
        }
    }
};

} // namespace shrink_array
