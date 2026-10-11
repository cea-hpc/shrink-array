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
// The candidate sizes: S / 2, S / 4 (finer_levels), S * 2 .. S * 16
// (coarser_levels), and between those bounds every multiple of 64 whose C / 64
// has at most grid_bits significant bits (2: 192, 384, 768, ...; 3: 320, 448,
// 640 too): a chunk size is a power of two or a multiple of 64
// (chunk_size_policy.h).
//
// How U_mat(C) is obtained without re-chunking: the array is cut into
// windows of C elements, each classified from the CURRENT chunks it overlaps.
//   * Inside a run of constant chunks of one constant: compressed. Exact, from
//     the chunk STATES alone. For Chunk2A the identity of a cst_val2 constant
//     IS the values_load() pointer (a TabCstVal2Holder slot), so a run is a
//     pointer comparison:
//         cst1 + cst1           -> cst1
//         cst2(p) + cst2(p)     -> cst2
//         anything else         -> materialized
//   * Covering a whole materialised chunk: materialised.
//   * Overlapping a materialised chunk in part: its data is read (sampled,
//     capped) -- uniform at a compressible value, and matching the rest of the
//     window, it compresses -- then extrapolated.
//   When S divides C (S * 2, S * 3, ...) the last case never occurs: the row is
//   EXACT, the fold of the current states. NOTE: this rule is deliberately
//   CONSERVATIVE for Chunk2A's mixed dual-value patterns; the demo harness
//   cross-checks it against ground truth. Constant-at-C windows read from the
//   data are counted as compressed regardless of which constant (cst1 vs cst2
//   registry growth is a second-order cost, ignored here and documented).
//
// CONCURRENCY CONTRACT: call at a QUIESCENT phase boundary only (same contract
// as flush_pending_deferred_ops: no concurrent writers/compressors, deferred
// ops drained). The walk reads chunk states and, for finer sizes, chunk data.
// ============================================================================

#include <algorithm> // std::sort, std::unique, std::upper_bound
#include <bit>       // std::bit_width, std::countr_zero
#include <cmath>     // std::lround (the extrapolated materialised count)
#include <cstddef>   // std::size_t
#include <cstdint>   // int64_t, uint32_t
#include <cstdio>    // std::printf (print)
#include <vector>    // std::vector

#include "chunk/internal/in_t.h"               // in_t<T>: the element snapshots
#include "chunk/bit.h"                         // is_bool_like_v, is_packed_bit_v
#include "chunk/internal/element_access.h"     // internal::as_words: Bit sub-blocks scanned by word
#include "chunk/internal/bit_words.h"          // bits::scan_range, bits::Uniform, bits::word_t
#include "chunk/config.h"                      // chunk_config::use_chunk2 / use_chunk2a
#include "chunk/PMP_CHUNK.h"                   // CHUNK_PMP_NUM_THREADS (the parts per thread)
#include "chunk/internal/rechunk_cost_model.h" // rechunk_cost_ms, rechunk_is_worth_it
#include "chunk/internal/chunk_size_policy.h"  // shrink_array::cst_buffer_size
#include "chunk/internal/env_parse.h"          // env_number (CHUNK_ADVISOR_GRID_BITS)

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
    /// @brief Options::grid_bits' default: CHUNK_ADVISOR_GRID_BITS, read once, else 2.
    /// @details Measured (shrink_bench chunk2a_dyn, --divisor 4, 16 threads,
    ///          end of run): 1 (powers of two) 2320 MB, 2 2317 MB, 3 2396 MB.
    ///          At 3 the cheapest size is often a small step (256 -> 384) whose
    ///          metadata gain the decider's rate gate refuses -- and it does not
    ///          fall back to the next best: two arrays stayed at 256.
    [[nodiscard]] static int default_grid_bits() noexcept
    {
        static const int bits = env_number<int>("CHUNK_ADVISOR_GRID_BITS", 2);
        return bits;
    }

    /// @brief Which sizes analyze() evaluates, its scan cap and the per-chunk metadata cost it adds.
    struct Options
    {
        int     coarser_levels = 4;     ///< evaluate S*2 .. S*2^levels
        int     finer_levels   = 2;     ///< evaluate S/2 .. S/2^levels (data sampled)
        int     grid_bits      = default_grid_bits(); ///< also every multiple of 64 between
                                                      ///< S/2^finer_levels and S*2^coarser_levels whose
                                                      ///< C/64 has at most this many significant bits
                                                      ///< (2: 192, 384, 768...: steps of at most 50%;
                                                      ///< 3: 320, 448, 640...: 25%); 0: none
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

        // ── Candidate sizes ───────────────────────────────────────────────────
        // The powers-of-two ratios S >> k (k <= finer_levels) and S << l
        // (l <= coarser_levels), plus, between the same bounds, every multiple
        // of 64 whose C / 64 has at most grid_bits significant bits (2: 64 x
        // 1, 2, 3, 4, 6, 8, 12, 16, ... -- steps of at most 50%; 3 adds 5, 7,
        // 10, 14, ...). Never above cst_buffer_size: rechunk_to would refuse it.
        std::vector<uint32_t> cand;
        {
            const auto     cap = static_cast<uint64_t>(shrink_array::cst_buffer_size);
            const uint32_t lo  = opt.finer_levels > 0 && opt.finer_levels < 32 ? S >> opt.finer_levels : S;
            const uint64_t hi_wanted =
                opt.coarser_levels > 0 && opt.coarser_levels < 32 ? uint64_t{S} << opt.coarser_levels : uint64_t{S};
            const uint64_t hi = hi_wanted < cap ? hi_wanted : (S < cap ? cap : uint64_t{S});
            cand.push_back(S);
            for (int k = 1; k <= opt.finer_levels && k < 32; ++k)
            {
                const uint32_t sub = S >> k;
                if (sub >= opt.min_size && is_valid_chunk_size(sub) && S % sub == 0)
                {
                    cand.push_back(sub);
                }
            }
            for (int l = 1; l <= opt.coarser_levels && l < 32 && (uint64_t{S} << l) <= cap; ++l)
            {
                cand.push_back(S << l);
            }
            if (opt.grid_bits > 0)
            {
                const uint64_t lo_grid = lo > opt.min_size ? lo : opt.min_size;
                for (uint64_t m = (lo_grid + 63) / 64; m * 64 <= hi; ++m)
                {
                    if (std::bit_width(m) - static_cast<unsigned>(std::countr_zero(m)) <=
                        static_cast<unsigned>(opt.grid_bits))
                    {
                        cand.push_back(static_cast<uint32_t>(m * 64));
                    }
                }
            }
            std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        }
        const auto n_cand = static_cast<int>(cand.size());

        // ── The window model ──────────────────────────────────────────────────
        // At a candidate size C the array is cut into windows [w*C, (w+1)*C)
        // of the current extent L = nc * S (the last one shorter). A window
        // stays compressed iff it holds one compressible value -- read from
        // the CURRENT chunks it overlaps:
        //   * inside a run of constant chunks of one (kind, key): compressed.
        //     Counted exactly, per run, without reading any data;
        //   * covering a whole materialised chunk: materialised (the chunk
        //     was not compressible at S, it is not at C either: the coarser
        //     fold's rule);
        //   * overlapping a materialised chunk only in part: the ambiguous
        //     case -- the overlapped data may be uniform at the window's value
        //     (a finer or a misaligned size). Settled by reading the data, on
        //     a sample of the materialised chunks.
        // When S divides C every window covers whole chunks: no ambiguous
        // window, the row is exact -- the coarser fold, at any multiple. When
        // C divides S the ambiguous windows are the sub-blocks of the
        // materialised chunks: the former finer sampling.
        //
        // The ambiguous window is attributed to ONE owner, its first
        // materialised chunk overlapped in part (only its first and last
        // chunks can be), so that sampling the owners counts each once; the
        // count is scaled by mat0 / scanned (picked owners).
        const int64_t L    = nc * static_cast<int64_t>(S);
        const auto    nwin = [&](uint32_t c) { return (L + c - 1) / c; };

        // Windows of size C inside the constant run of chunks [r0, r1).
        const auto windows_inside = [&](int64_t r0, int64_t r1, uint32_t c) -> int64_t
        {
            const int64_t a    = r0 * S;
            const int64_t wmin = (a + c - 1) / c;
            const int64_t wmax = r1 == nc ? nwin(c) - 1 : (r1 * S) / c - 1;
            return wmax >= wmin ? wmax - wmin + 1 : 0;
        };

        struct Seg
        {
            int64_t     start;
            int64_t     end;
            int         kind; // -1: none, 0 cst1, 1 cst2, 2 mat
            const void* key;
        };
        const auto count_run = [&](const Seg& seg, std::vector<int64_t>& inside)
        {
            if (seg.kind < 0 || seg.kind == 2)
            {
                return;
            }
            for (int ci = 0; ci < n_cand; ++ci)
            {
                inside[static_cast<std::size_t>(ci)] += windows_inside(seg.start, seg.end, cand[static_cast<std::size_t>(ci)]);
            }
        };

        // ── Classification, constant runs and sampling selection ──────────────
        // No per-chunk scratch array: each pass classifies the chunks it reads
        // (classify() is pure-read -- const accessors, a pointer compare against
        // stable sentinels -- so concurrent reads are race-free; analyze() is
        // called at a quiescent serial point, so spawning a region here is safe).
        // Storing the classification instead (9 bytes per chunk: 140 MB for a
        // 15.6-million-chunk array, zeroed and faulted in at EVERY call) cost
        // 1-2 s of a 6.5 s run on the 64-thread chunk2a_dyn bench; classifying
        // twice costs a fraction of that.
        //
        // Work unit: a PART, a contiguous run of chunks.
        //  - the runs: a part counts the runs that lie inside it, and leaves its
        //    first and last ones (they may continue in the parts around it) to
        //    a serial merge -- the same runs, so the same integers, as a
        //    sequential walk;
        //  - each part counts its materialised chunks; the selection gives part p
        //    the running count of the parts before it, so it picks exactly the
        //    chunks the sequential walk picked (every stride-th materialised
        //    chunk), and the parts' picks are concatenated in part order: the same
        //    list, in the same order. The report is byte-identical.
        // Parts are indexed by the loop, never by a thread rank, so a serial build
        // runs the same code in order.
        int64_t mat0 = 0;

        std::vector<int64_t> inside(static_cast<std::size_t>(n_cand), 0); // windows inside constant runs
        std::vector<int64_t> found(static_cast<std::size_t>(n_cand), 0);  // compressible ambiguous windows
        bool                 any_sampled = false;                         // a candidate that S does not divide
        for (const uint32_t c : cand)
        {
            any_sampled = any_sampled || c % S != 0;
        }
        int64_t              scanned = 0;
        std::vector<int64_t> picks; // chunks whose data the sampling reads

        int64_t                           n_parts  = 1;
        int64_t                           stride   = 1;
        bool                              sampling = false;
        std::vector<int64_t>              part_mat;    // materialised chunks per part, then their offsets
        std::vector<Seg>                  part_first;  // each part's first run
        std::vector<Seg>                  part_last;   // ... and its last one
        std::vector<char>                 part_single; // the part is one run
        std::vector<std::vector<int64_t>> part_picks;  // the chunks each part selected

        // Chunks [j0, j1) of part p.
        const auto part_range = [&](int64_t p, int64_t& j0, int64_t& j1)
        {
            j0 = nc * p / n_parts;
            j1 = nc * (p + 1) / n_parts;
        };

#pragma omp parallel if (nc > 16384)
        {
#pragma omp single
            {
                n_parts = static_cast<int64_t>(CHUNK_PMP_NUM_THREADS) * 4;
                if (n_parts > nc)
                {
                    n_parts = nc;
                }
                part_mat.assign(static_cast<std::size_t>(n_parts), 0);
                part_first.assign(static_cast<std::size_t>(n_parts), Seg{0, 0, -1, nullptr});
                part_last.assign(static_cast<std::size_t>(n_parts), Seg{0, 0, -1, nullptr});
                part_single.assign(static_cast<std::size_t>(n_parts), 0);
                part_picks.assign(static_cast<std::size_t>(n_parts), {});
            }
            // (implicit barrier: n_parts and the per-part vectors are set)

            // Runs + materialised count, one part per iteration.
            {
                std::vector<int64_t> inside_loc(static_cast<std::size_t>(n_cand), 0);
#pragma omp for schedule(dynamic, 1) nowait
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    int64_t j0 = 0;
                    int64_t j1 = 0;
                    part_range(p, j0, j1);
                    int64_t m          = 0;
                    Seg     cur        = {j0, j0, -1, nullptr};
                    bool    first_done = false;
                    for (int64_t j = j0; j < j1; ++j)
                    {
                        const void* key  = nullptr;
                        const int   kind = classify(tab.chunk_list[j], key);
                        if (kind == 2)
                        {
                            ++m;
                        }
                        if (kind == cur.kind && key == cur.key)
                        {
                            cur.end = j + 1;
                            continue;
                        }
                        if (cur.kind >= 0)
                        {
                            if (first_done)
                            {
                                count_run(cur, inside_loc);
                            }
                            else
                            {
                                part_first[static_cast<std::size_t>(p)] = cur;
                                first_done                              = true;
                            }
                        }
                        cur = {j, j + 1, kind, key};
                    }
                    if (first_done)
                    {
                        part_last[static_cast<std::size_t>(p)] = cur;
                    }
                    else
                    {
                        part_first[static_cast<std::size_t>(p)]  = cur;
                        part_single[static_cast<std::size_t>(p)] = 1;
                    }
                    part_mat[static_cast<std::size_t>(p)] = m;
                }
                for (int ci = 0; ci < n_cand; ++ci)
                {
                    if (inside_loc[static_cast<std::size_t>(ci)] != 0)
                    {
#pragma omp atomic
                        inside[static_cast<std::size_t>(ci)] += inside_loc[static_cast<std::size_t>(ci)];
                    }
                }
            }
#pragma omp barrier

            // The runs across part boundaries; mat0, the sampling stride, and
            // each part's starting rank among the materialised chunks
            // (exclusive prefix sum, in place).
#pragma omp single
            {
                Seg open = {0, 0, -1, nullptr};
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    const Seg& first = part_first[static_cast<std::size_t>(p)];
                    if (open.kind == first.kind && open.key == first.key && open.end == first.start)
                    {
                        open.end = first.end;
                    }
                    else
                    {
                        count_run(open, inside);
                        open = first;
                    }
                    if (part_single[static_cast<std::size_t>(p)] == 0)
                    {
                        count_run(open, inside);
                        open = part_last[static_cast<std::size_t>(p)];
                    }
                }
                count_run(open, inside);

                int64_t run_count = 0;
                for (int64_t p = 0; p < n_parts; ++p)
                {
                    const int64_t m                       = part_mat[static_cast<std::size_t>(p)];
                    part_mat[static_cast<std::size_t>(p)] = run_count;
                    run_count += m;
                }
                mat0     = run_count;
                sampling = any_sampled && mat0 > 0;
                if (sampling)
                {
                    stride = mat0 > opt.max_mat_scans ? (mat0 + opt.max_mat_scans - 1) / opt.max_mat_scans : 1;
                }
            }
            // (implicit barrier: inside, mat0, stride and the offsets are set)

            // ── Sampling: pick the materialised chunks whose DATA is read ─────
            // Every stride-th materialised chunk -- at most max_mat_scans, since
            // stride = ceil(mat0 / max_mat_scans) -- the same chunks as the
            // sequential walk, found part by part.
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

            // ── The ambiguous windows of each picked chunk ────────────────────
            {
                std::vector<int64_t>  found_loc(static_cast<std::size_t>(n_cand), 0);
                std::vector<uint32_t> seg_start; // the picked chunk's runs of equal values ...
                std::vector<T>        seg_val;   // ... and their values

                // Element i of a materialised buffer, and whether [b, e) of it
                // is uniform (its value in `val`).
                const auto value_of = [](const T* v, std::size_t i) -> T
                {
                    if constexpr (is_packed_bit_v<T>)
                    {
                        const auto* w = internal::as_words(v);
                        return T{((w[i / bits::word_bits] >> (i % bits::word_bits)) & 1u) != 0};
                    }
                    else
                    {
                        return v[i];
                    }
                };
                const auto uniform_range = [&](const T* v, std::size_t b, std::size_t e, T& val) -> bool
                {
                    if constexpr (is_packed_bit_v<T>)
                    {
                        const bits::Uniform u = bits::scan_range(internal::as_words(v), b, e - b);
                        val                   = T{u == bits::Uniform::all_true};
                        return u != bits::Uniform::mixed;
                    }
                    else
                    {
                        for (std::size_t i = b + 1; i < e; ++i)
                        {
                            if (!(v[i] == v[b]))
                            {
                                return false;
                            }
                        }
                        val = v[b];
                        return true;
                    }
                };

                const auto n_picks = static_cast<int64_t>(picks.size());
#pragma omp for schedule(dynamic, 64) nowait
                for (int64_t p = 0; p < n_picks; ++p)
                {
                    const int64_t j  = picks[static_cast<std::size_t>(p)];
                    const T*      vj = tab.chunk_list[j].values_load();

                    // Chunk j's runs of equal values: [b, e) of it is uniform iff
                    // b and e - 1 fall in the same run.
                    seg_start.clear();
                    seg_val.clear();
                    seg_start.push_back(0);
                    seg_val.push_back(value_of(vj, 0));
                    if constexpr (is_packed_bit_v<T>)
                    {
                        // A word at a time: the bits that differ from the one below.
                        const auto*       w     = internal::as_words(vj);
                        const std::size_t nw    = (S + bits::word_bits - 1) / bits::word_bits;
                        auto              below = static_cast<bits::word_t>(w[0] & 1u);
                        for (std::size_t k = 0; k < nw; ++k)
                        {
                            const bits::word_t x = w[k];
                            bits::word_t       t = x ^ ((x << 1) | below);
                            below                = x >> (bits::word_bits - 1);
                            while (t != 0)
                            {
                                const std::size_t i = (k * bits::word_bits) + static_cast<std::size_t>(std::countr_zero(t));
                                t &= t - 1;
                                if (i == 0 || i >= S)
                                {
                                    continue;
                                }
                                seg_start.push_back(static_cast<uint32_t>(i));
                                seg_val.push_back(value_of(vj, i));
                            }
                        }
                    }
                    else
                    {
                        for (uint32_t i = 1; i < S; ++i)
                        {
                            if (!(vj[i] == vj[i - 1]))
                            {
                                seg_start.push_back(i);
                                seg_val.push_back(vj[i]);
                            }
                        }
                    }
                    const auto seg_of = [&](int64_t i)
                    {
                        return std::upper_bound(seg_start.begin(), seg_start.end(), static_cast<uint32_t>(i)) -
                               seg_start.begin() - 1;
                    };

                    const int64_t x0 = j * S;
                    const int64_t x1 = x0 + S;
                    for (int ci = 0; ci < n_cand; ++ci)
                    {
                        const uint32_t c = cand[static_cast<std::size_t>(ci)];
                        if (c % S == 0)
                        {
                            continue; // whole chunks only: no ambiguous window
                        }
                        for (int64_t w = x0 / c; w * c < x1; ++w)
                        {
                            const int64_t a0 = w * c;
                            const int64_t a1 = a0 + c < L ? a0 + c : L;
                            if (a0 <= x0 && x1 <= a1)
                            {
                                continue; // covers chunk j whole: materialised
                            }
                            const int64_t ca = a0 / S;
                            const int64_t cb = (a1 - 1) / S;
                            if (j != ca)
                            {
                                // j is the window's last chunk: the first one owns
                                // the window when it is materialised in part too.
                                const void* key = nullptr;
                                if (a0 > ca * S && classify(tab.chunk_list[ca], key) == 2)
                                {
                                    continue;
                                }
                            }
                            bool ok    = true;
                            bool has_v = false;
                            T    val{};
                            for (int64_t cc = ca; cc <= cb && ok; ++cc)
                            {
                                const int64_t p0 = a0 > cc * S ? a0 : cc * S;
                                const int64_t p1 = a1 < (cc + 1) * S ? a1 : (cc + 1) * S;
                                const auto&   ch = tab.chunk_list[cc];
                                const void*   key = nullptr;
                                T             part{};
                                if (classify(ch, key) != 2)
                                {
                                    part = ch.read_value_scan(0); // the constant (a scan read: no app stats)
                                }
                                else if (p0 == cc * S && p1 == (cc + 1) * S)
                                {
                                    ok = false; // a whole materialised chunk
                                }
                                else if (cc == j)
                                {
                                    const auto s = seg_of(p0 - x0);
                                    ok           = s == seg_of(p1 - 1 - x0);
                                    part         = seg_val[static_cast<std::size_t>(s)];
                                }
                                else
                                {
                                    ok = uniform_range(ch.values_load(), static_cast<std::size_t>(p0 - (cc * S)),
                                                       static_cast<std::size_t>(p1 - (cc * S)), part);
                                }
                                if (ok && has_v)
                                {
                                    ok = part == val;
                                }
                                val   = part;
                                has_v = true;
                            }
                            // Uniform is not enough: the value must be one the
                            // variant can store as a constant (compressible_value).
                            found_loc[static_cast<std::size_t>(ci)] += ok && compressible_value(val) ? 1 : 0;
                        }
                    }
                }
                for (int ci = 0; ci < n_cand; ++ci)
                {
                    if (found_loc[static_cast<std::size_t>(ci)] != 0)
                    {
#pragma omp atomic
                        found[static_cast<std::size_t>(ci)] += found_loc[static_cast<std::size_t>(ci)];
                    }
                }
            }
        }

        // ── Assemble the table (ascending sizes) ──────────────────────────────
        for (int ci = 0; ci < n_cand; ++ci)
        {
            const uint32_t c     = cand[static_cast<std::size_t>(ci)];
            const bool     exact = c % S == 0;
            const int64_t  nch   = nwin(c);
            int64_t        mat   = 0;
            if (c == S)
            {
                rep.cur = static_cast<int>(rep.table.size());
                mat     = mat0;
            }
            else
            {
                const double amb = scanned > 0 ? static_cast<double>(found[static_cast<std::size_t>(ci)]) *
                                                     static_cast<double>(mat0) / static_cast<double>(scanned)
                                               : 0.;
                mat = nch - inside[static_cast<std::size_t>(ci)] - std::lround(amb);
                mat = mat < 0 ? 0 : (mat > nch ? nch : mat);
            }
            ChunkSizeEstimate est;
            est.size      = c;
            est.n_chunks  = nch;
            est.mat       = mat;
            est.cst       = nch - mat;
            est.exact     = exact;
            est.mem_bytes = (static_cast<double>(nch) * rep.meta_bytes) +
                            (static_cast<double>(mat) * static_cast<double>(Arr::chunk_type::storage_units(c)) *
                             sizeof(T)); // Bit: size / 8 bytes, not size
            rep.table.push_back(est);
        }

        rep.best = 0;

        for (std::size_t i = 1; i < rep.table.size(); ++i)
        {
            if (rep.table[i].mem_bytes < rep.table[static_cast<std::size_t>(rep.best)].mem_bytes)
            {
                rep.best = static_cast<int>(i);
            }
        }
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
                        row.n_chunks != 0 ? 100. * static_cast<double>(row.mat) / static_cast<double>(row.n_chunks)
                                          : 0.,
                        row.mem_bytes / (1024. * 1024.), row.exact ? "" : "~",
                        static_cast<int>(i) == rep.cur ? " <- current" : "",
                        static_cast<int>(i) == rep.best ? "  ** BEST **" : "");
        }
        std::printf("   estimated gain best vs current : %.2f Mo\n", rep.best_gain_bytes() / (1024. * 1024.));

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
            const bool worth = rechunk_is_worth_it(gain_bytes, cost_ms, std::size_t{1} << 20U, 50.0);
            std::printf("   estimated re-chunk cost       : %.3f ms  (%s)\n", cost_ms, worth ? "worth it" : "marginal");
        }
    }
};

} // namespace shrink_array
