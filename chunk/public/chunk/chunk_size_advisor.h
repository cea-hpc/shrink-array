/*
 * Copyright 2026 CEA (www.cea.fr)
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

#include <cmath>     // std::lround (the extrapolated materialised count)
#include <cstddef>   // std::size_t
#include <cstdint>   // int64_t, uint32_t
#include <vector>    // std::vector
#include <algorithm> // std::ranges::min_element (best)
#include <print>     // std::println (print)
#include <ranges>    // std::views::enumerate (print)

#include "chunk/internal/in_t.h"               // in_t<T>: the element snapshots
#include "chunk/bit.h"                         // is_bool_like_v, is_packed_bit_v
#include "chunk/internal/element_access.h"     // internal::as_words: Bit sub-blocks scanned by word
#include "chunk/internal/bit_words.h"          // bits::scan_range, bits::Uniform
#include "chunk/config.h"                      // chunk_config::use_chunk2 / use_chunk2a
#include "chunk/PMP_CHUNK.h"                   // CHUNK_PMP_NUM_THREADS (the parts per thread)
#include "chunk/internal/rechunk_cost_model.h" // rechunk_cost_ms, rechunk_is_worth_it
#include "chunk/internal/chunk_size_policy.h"  // shrink_array::cst_buffer_size

namespace shrink_array
{

/// @brief Model figures for one candidate chunk size: chunk counts and estimated memory.
struct ChunkSizeEstimate
{
    uint32_t size      = 0;     ///< candidate chunk size (elements)
    int64_t  n_chunks  = 0;     ///< total chunks at that size
    int64_t  mat       = 0;     ///< materialized chunks (estimated or exact)
    int64_t  cst       = 0;     ///< compressed chunks (cst_val1 + cst_val2)
    double   mem_bytes = 0.;    ///< model memory (bytes)
    bool     exact     = false; ///< true: exact fold; false: sampled estimate
};

/// @brief ChunkSizeAdvisor's result: one ChunkSizeEstimate per candidate size, with the best and current indices.
struct ChunkSizeReport
{
    /// @brief Bytes the best size would save over the current one (0 when
    ///        either is unknown).
    [[nodiscard]] double best_gain_bytes() const noexcept
    {
        if (best < 0 || cur < 0)
        {
            return 0.;
        }
        return table[cur].mem_bytes - table[best].mem_bytes;
    }

    uint32_t                       current_size = 0;  ///< the array's chunk size now (elements)
    double                         meta_bytes   = 0.; ///< per-chunk metadata: sizeof(chunk) + Options::meta_extra_bytes
    double                         elem_bytes   = 0.; ///< per element in a materialised chunk (1/8 for Bit)
    std::vector<ChunkSizeEstimate> table;             ///< ascending sizes
    int                            best = -1;         ///< index into table (-1: none)
    int                            cur  = -1;         ///< index of current size (-1: none)
};

/// @brief Estimates an array's model memory at other chunk sizes without re-chunking (see the header comment).
/// @tparam T Element type.
template <typename T>
struct ChunkSizeAdvisor
{
    /// @brief Which sizes analyze() evaluates, its scan cap and the per-chunk metadata cost it adds.
    struct Options
    {
        int     coarser_levels = 4;     ///< evaluate S*2 .. S*2^levels
        int     finer_levels   = 2;     ///< evaluate S/2 .. S/2^levels (data sampled)
        int64_t max_mat_scans  = 65536; ///< cap on materialized chunks whose DATA is read
        /// Per-chunk metadata beyond sizeof(chunk): tracking-list entry (~0.8 B,
        /// see the GLOBAL COMPRESSION LISTS monitoring table: List Memory /
        /// Allocated Chunks) + dirty bitset (128 B / 64 chunks = 2 B) + slack.
        double   meta_extra_bytes = 3.0;
        uint32_t min_size         = 64; ///< don't evaluate below this. Raised from
                                        ///< 16: the cost model under-counts real
                                        ///< per-chunk overhead (slab block +
                                        ///< fragmentation, not just meta_bytes),
                                        ///< so it over-shrinks toward tiny chunks
                                        ///< that fragment. 64 caps the descent at
                                        ///< a sane overhead ratio.
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
        rep.elem_bytes    = S != 0 ? static_cast<double>(Arr::chunk_type::storage_units(S)) * sizeof(T) / S : 0.;

        if (nc <= 0)
        {
            return rep;
        }

        // Chunk2 keys cst2 by the array-level second-value index; fetch it once.
        // v1 has no cst2; 2A keys cst2 per chunk by the values_load() pointer.
        [[maybe_unused]] const int chunk2_ind = [&]
        {
            // bool / Bit: one second value, `true`, no index --
            // get_ind_second_value() aborts for them; classify() below asks
            // the unindexed is_cst_val2().
            if constexpr (chunk_config::use_chunk2 && !is_bool_like_v<T>)
            {
                return tab.get_ind_second_value();
            }
            else if constexpr (chunk_config::use_chunk2)
            {
                return 0;
            }
            else
            {
                return -1;
            }
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
            {
                return true;
            }
            else
            {
                if (elem == cst1_value)
                {
                    return true;
                }
                return has_cst2 && (elem == cst2_value);
            }
        };

        const auto classify = [&](const auto& chunk, const void*& key) -> int
        {
            key = nullptr;
            if (chunk.is_cst_val1())
            {
                return 0;
            }
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
                {
                    is_cst2 = chunk.is_cst_val2();
                }
                else
                {
                    is_cst2 = chunk2_ind >= 0 && chunk.is_cst_val2(chunk2_ind);
                }
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
        // ── Classification, coarser fold and sampling selection ─────────────
        // No per-chunk scratch array: each pass classifies the chunks it reads
        // (classify() is pure-read -- const accessors, a pointer compare against
        // stable sentinels -- so concurrent reads are race-free; analyze() is
        // called at a quiescent serial point, so spawning a region here is safe).
        // Storing the classification instead (9 bytes per chunk: 140 MB for a
        // 15.6-million-chunk array, zeroed and faulted in at EVERY call) cost
        // 1-2 s of a 6.5 s run on the 64-thread chunk2a_dyn bench; classifying
        // twice costs a fraction of that.
        //
        // Work unit: a PART, a contiguous run of whole super-blocks of 2^L chunks.
        //  - the coarser fold: every level-l block (l <= L) lies inside one
        //    super-block, so each part folds independently with its own run[],
        //    and the per-level counts are summed -- the same integers as the
        //    sequential fold;
        //  - each part counts its materialised chunks; the selection gives part p
        //    the running count of the parts before it, so it picks exactly the
        //    chunks the sequential walk picked (every stride-th materialised
        //    chunk), and the parts' picks are concatenated in part order: the same
        //    list, in the same order. The report is byte-identical.
        // Parts are indexed by the loop, never by a thread rank, so a serial build
        // runs the same code in order.
        int64_t mat0 = 0;

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

        int64_t                           n_parts  = 1;
        int64_t                           stride   = 1;
        bool                              sampling = false;
        std::vector<int64_t>              part_mat;   // materialised chunks per part, then their offsets
        std::vector<std::vector<int64_t>> part_picks; // the chunks each part selected

        // Chunks [j0, j1) of part p (whole super-blocks).
        const auto part_range = [&](int64_t p, int64_t& j0, int64_t& j1)
        {
            const int64_t s0 = n_super * p / n_parts;
            const int64_t s1 = n_super * (p + 1) / n_parts;
            j0               = s0 * super;
            j1               = (s1 * super) < nc ? (s1 * super) : nc;
        };

#pragma omp parallel if (nc > 16384)
        {
#pragma omp single
            {
                n_parts = static_cast<int64_t>(CHUNK_PMP_NUM_THREADS) * 4;
                if (n_parts > n_super)
                {
                    n_parts = n_super > 0 ? n_super : 1;
                }
                part_mat.assign(static_cast<std::size_t>(n_parts), 0);
                part_picks.assign(static_cast<std::size_t>(n_parts), {});
            }
            // (implicit barrier: n_parts and the per-part vectors are set)

            // Coarser fold + materialised count, one part per iteration.
            {
                std::vector<Agg>     run(static_cast<std::size_t>(coarser) + 1, Agg{0, nullptr});
                std::vector<int64_t> mat_loc(static_cast<std::size_t>(coarser) + 1, 0);
#pragma omp for schedule(dynamic, 1) nowait
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    int64_t j0 = 0;
                    int64_t j1 = 0;
                    part_range(p, j0, j1);
                    int64_t m = 0;
                    for (int64_t j = j0; j < j1; ++j)
                    {
                        const void* key  = nullptr;
                        const int   kind = classify(tab.chunk_list[j], key);
                        if (kind == 2)
                        {
                            ++m;
                        }

                        for (int l = 1; l <= coarser; ++l)
                        {
                            Agg& agg = run[static_cast<std::size_t>(l)];
                            if ((j & ((int64_t{1} << l) - 1)) == 0)
                            {
                                agg = {kind, key}; // block opens with this child
                            }
                            else if (agg.kind != 2 && (agg.kind != kind || agg.key != key))
                            {
                                agg.kind = 2; // mixed constants -> materialized
                            }
                            if (((j + 1) & ((int64_t{1} << l) - 1)) == 0 || j + 1 == nc)
                            {
                                if (agg.kind == 2)
                                {
                                    ++mat_loc[static_cast<std::size_t>(l)]; // commit block at boundary
                                }
                            }
                        }
                    }
                    part_mat[static_cast<std::size_t>(p)] = m;
                }
                for (int l = 1; l <= coarser; ++l)
                {
                    if (mat_loc[static_cast<std::size_t>(l)] != 0)
                    {
#pragma omp atomic
                        mat_at[static_cast<std::size_t>(l)] += mat_loc[static_cast<std::size_t>(l)];
                    }
                }
            }
#pragma omp barrier

            // mat0, the sampling stride, and each part's starting rank among the
            // materialised chunks (exclusive prefix sum, in place).
#pragma omp single
            {
                int64_t run_count = 0;
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    const int64_t m                       = part_mat[static_cast<std::size_t>(p)];
                    part_mat[static_cast<std::size_t>(p)] = run_count;
                    run_count += m;
                }
                mat0     = run_count;
                sampling = finer > 0 && mat0 > 0 && S >= opt.min_size * 2;
                if (sampling)
                {
                    stride = mat0 > opt.max_mat_scans ? (mat0 + opt.max_mat_scans - 1) / opt.max_mat_scans : 1;
                }
            }
            // (implicit barrier: mat0, stride and the offsets are set)

            // ── Finer sizes: sample the DATA of materialized chunks ───────────
            // Selection: every stride-th materialised chunk -- at most
            // max_mat_scans, since stride = ceil(mat0 / max_mat_scans) -- the same
            // chunks as the sequential walk, found part by part.
            if (sampling)
            {
#pragma omp for schedule(dynamic, 1)
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    int64_t j0 = 0;
                    int64_t j1 = 0;
                    part_range(p, j0, j1);
                    int64_t seen = part_mat[static_cast<std::size_t>(p)];
                    auto&   mine = part_picks[static_cast<std::size_t>(p)];
                    for (int64_t j = j0; j < j1; ++j)
                    {
                        const void* key = nullptr;
                        if (classify(tab.chunk_list[j], key) != 2) // skip cst1 / cst2 (only mat has data)
                        {
                            continue;
                        }
                        if ((seen++ % stride) != 0)
                        {
                            continue;
                        }
                        mine.push_back(j);
                    }
                }
            }
#pragma omp single
            {
                for (const auto& mine : part_picks)
                {
                    picks.insert(picks.end(), mine.begin(), mine.end());
                }
                scanned = static_cast<int64_t>(picks.size());
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
                        {
                            break;
                        }
                        for (uint32_t b = 0; b < S; b += sub)
                        {
                            bool cst = true;
                            T    run{}; // the sub-block's value, when uniform
                            if constexpr (is_packed_bit_v<T>)
                            {
                                // Bit: v is words of bits (bit.h), a sub-block
                                // is a bit range -- classified word by word.
                                const bits::Uniform u = bits::scan_range(internal::as_words(v), b, sub);
                                cst                   = u != bits::Uniform::mixed;
                                run                   = T{u == bits::Uniform::all_true};
                            }
                            else
                            {
                                for (uint32_t i = 1; i < sub; ++i)
                                {
                                    if (!(v[b + i] == v[b]))
                                    {
                                        cst = false;
                                        break;
                                    }
                                }
                                if (cst)
                                {
                                    run = v[b];
                                }
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
                            {
                                cst = compressible_value(run);
                            }
                            fine_loc[static_cast<std::size_t>(k)] += cst ? 1 : 0;
                        }
                    }
                }
                for (int k = 1; k <= finer; ++k)
                {
                    if (fine_loc[static_cast<std::size_t>(k)] != 0)
                    {
#pragma omp atomic
                        fine_cst[static_cast<std::size_t>(k)] += fine_loc[static_cast<std::size_t>(k)];
                    }
                }
            }
        }

        // ── Assemble the table (ascending sizes) ──────────────────────────────
        const auto push = [&](uint32_t size, int64_t nchunks, int64_t mat, bool exact)
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
            {
                continue;
            }
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
            {
                break;
            }
            const int64_t nchunks = (nc + (int64_t{1} << l) - 1) >> l;
            push(S << l, nchunks, mat_at[static_cast<std::size_t>(l)], /*exact=*/true);
        }

        // The first of the smallest footprints: on a tie the smaller size wins.
        rep.best = static_cast<int>(std::ranges::min_element(rep.table, {}, &ChunkSizeEstimate::mem_bytes) -
                                    rep.table.begin());
        return rep;
    }

    /// @brief Print @p rep as a table on stdout, headed by @p name.
    static void print(const ChunkSizeReport& rep, const char* name)
    {
        std::println("── ChunkSizeAdvisor [{}] current S={}, meta={:.1f} B/chunk ──", name, rep.current_size,
                     rep.meta_bytes);
        for (const auto& [i, row] : std::views::enumerate(rep.table))
        {
            std::println(
                "   S={:5} : chunks={:9}  mat={:9} ({:5.1f}%)  mem={:9.2f} Mo {}{}{}", row.size, row.n_chunks, row.mat,
                row.n_chunks != 0 ? 100. * static_cast<double>(row.mat) / static_cast<double>(row.n_chunks) : 0.,
                row.mem_bytes / (1024. * 1024.), row.exact ? "" : "~", i == rep.cur ? " <- current" : "",
                i == rep.best ? "  ** BEST **" : "");
        }
        std::println("   estimated gain best vs current : {:.2f} Mo", rep.best_gain_bytes() / (1024. * 1024.));

        // Re-chunk cost (phase 2 model): only meaningful when best != current.
        // Uses the chunk and materialised-chunk counts the table already holds.
        // The gain is memory (bytes), the cost is time (ms) — different units,
        // shown side by side so the reader can judge the trade-off. Constants
        // are hardware specific (rechunk_cost_model.h: fitted on ARM, 64
        // threads).
        if (rep.best >= 0 && rep.cur >= 0 && rep.best != rep.cur)
        {
            const auto&  cur_e = rep.table[static_cast<std::size_t>(rep.cur)];
            const double cost_ms =
                rechunk_cost_ms(cur_e.n_chunks, cur_e.size, cur_e.mat, rep.elem_bytes, CHUNK_PMP_MAX_THREADS);
            const auto gain_bytes = static_cast<std::size_t>(rep.best_gain_bytes() > 0 ? rep.best_gain_bytes() : 0.);
            // Observation-phase thresholds (not a decision, just a hint label):
            // >= 1 Mo gained and <= 50 ms spent reads as clearly worth it.
            const bool worth = rechunk_is_worth_it(gain_bytes, cost_ms, 1uz << 20U, 50.0);
            std::println("   estimated re-chunk cost       : {:.3f} ms  ({})", cost_ms,
                         worth ? "worth it" : "marginal");
        }
    }
};

} // namespace shrink_array
