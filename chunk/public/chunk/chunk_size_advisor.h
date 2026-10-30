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
// windows of C elements, each classified from the CURRENT chunks it overlaps
// (details at "The window model" in analyze()). For zero_any the identity of a
// cst_val2 constant IS the values_load() pointer (a TabCstVal2Holder slot), so
// a constant run is a pointer comparison:
//         cst1 + cst1           -> cst1
//         cst2(p) + cst2(p)     -> cst2
//         anything else         -> materialized
// When S divides C the row is EXACT; otherwise partly-overlapped materialised
// chunks are sampled (capped) and extrapolated. Deliberately CONSERVATIVE for
// zero_any's mixed dual-value patterns. Constant-at-C windows count as
// compressed whichever constant (cst2 registry growth is ignored).
//
// CONCURRENCY CONTRACT: call at a QUIESCENT phase boundary only (same contract
// as flush_pending_deferred_ops: no concurrent writers/compressors, deferred
// ops drained). The walk reads chunk states and, for finer sizes, chunk data.
// ============================================================================

#include <algorithm>   // std::sort, std::unique, std::upper_bound
#include <array>       // std::array (motif_chunk_data's buffers)
#include <atomic>      // std::atomic_ref (the per-thread counts merged)
#include <bit>         // std::bit_width, std::countr_zero
#include <cmath>       // std::lround (the extrapolated materialised count)
#include <cstddef>     // std::size_t
#include <cstdint>     // int64_t, uint32_t
#include <format>      // std::format (write_table's lines)
#include <string_view> // std::string_view (write_table's sink)
#include <vector>      // std::vector

#include "chunk/internal/in_t.h"               // InT<T>: the element snapshots
#include "chunk/bit.h"                         // is_bool_like_v, is_packed_bit_v
#include "chunk/internal/element_access.h"     // internal::as_words: Bit sub-blocks scanned by word
#include "chunk/internal/bit_words.h"          // bits::scan_range, bits::Uniform, bits::WordT
#include "chunk/config.h"                      // chunk_config::use_zero_one_value / use_zero_any
#include "chunk/PMP_CHUNK.h"                   // CHUNK_PMP_NUM_THREADS (the parts per thread), pmp::team
#include "chunk/internal/rechunk_cost_model.h" // rechunk_cost_ms, rechunk_is_worth_it
#include "chunk/internal/chunk_size_policy.h"  // shrink_array::cst_buffer_size
#include "chunk/internal/print.h"              // shrink_array::println
#include "chunk/log.h"                         // CHUNK_LOG (log(): channel advisor)
#include "chunk/chunk_traits.h"                // is_zero_any_family_v, has_motif_state_v (pattern::Array)
#include "chunk/internal/relaxed_element.h"    // load_element_relaxed (a motif chunk's offset)

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

namespace internal
{
/**
 * @brief The data of chunk @p idx of @p tab, a member with the motif state
 *        (pattern::Array): its buffer; for a motif chunk, its motif plus its
 *        offset, written into this thread's scratch buffer @p which (0 or 1:
 *        the analysis holds two chunks' data at once).
 * @details For ChunkSizeAdvisor::analyze(), at a quiescent point: the slot is
 *          stable. A function template of its own, with its own buffers, called
 *          from discarded branches for the other variants: their code -- the
 *          analysis lambdas and what they capture -- is unchanged. GCC orders a
 *          lambda's captures by first appearance, discarded branches included:
 *          the callers name nothing the lambdas did not already name first.
 */
template <typename Arr>
[[nodiscard]] const Arr::value_type* motif_chunk_data(const Arr& tab, int64_t idx, int which)
{
    using ChunkT = Arr::ChunkType;
    using T      = Arr::value_type;
    static thread_local std::array<std::vector<T>, 2> buffers;
    std::vector<T>&                                   scratch = buffers[static_cast<std::size_t>(which)];
    const T* const                                    vals    = tab.chunk_list_[idx].values_load();
    if (!ChunkT::is_motif_ptr(vals))
    {
        return vals;
    }
    const T        off      = load_element_relaxed(tab.motif_offset_slot(static_cast<int>(idx)));
    const uint32_t chunk_sz = tab.chunk_size();
    scratch.resize(chunk_sz);
    for (uint32_t i = 0; i < chunk_sz; ++i)
    {
        scratch[i] = ChunkT::motif_value(vals, static_cast<int>(i), off);
    }
    return scratch.data();
}
} // namespace internal

/// @brief Estimates an array's model memory at other chunk sizes without re-chunking (see the header comment).
/// @tparam T Element type.
template <typename T>
struct ChunkSizeAdvisor
{
    /// @brief Which sizes analyze() evaluates, its scan cap and the per-chunk metadata cost it adds.
    struct Options
    {
        int coarser_levels = 4;        ///< evaluate S*2 .. S*2^levels
        int finer_levels   = 2;        ///< evaluate S/2 .. S/2^levels (data sampled)
        int grid_bits      = 2;        ///< also every multiple of 64 between
                                       ///< S/2^finer_levels and S*2^coarser_levels whose
                                       ///< C/64 has at most this many significant bits
                                       ///< (2: 192, 384, 768...: steps of at most 50%;
                                       ///< 3: 320, 448, 640...: 25%); 0: none. 1 and 2
                                       ///< give about the same footprint, 3 a few % more:
                                       ///< at 3 the cheapest size is often a small step
                                       ///< (256 -> 384) that the decider's rate gate
                                       ///< refuses, with no fallback to the next best.
                                       ///< RechunkPolicy::advisor_grid_bits sets it for
                                       ///< RechunkDecider::consider().
        int64_t max_mat_scans = 65536; ///< cap on materialized chunks whose DATA is read
        /// Per-chunk metadata beyond sizeof(chunk): tracking-list entry (~0.8 B,
        /// see the GLOBAL COMPRESSION LISTS monitoring table: List Memory /
        /// Allocated Chunks) + dirty bitset (128 B / 64 chunks = 2 B) + slack.
        double   meta_extra_bytes = 3.0;
        uint32_t min_size         = 64; ///< don't evaluate below this: the model
                                        ///< under-counts real per-chunk overhead
                                        ///< (slab block, fragmentation), so a
                                        ///< lower floor over-shrinks to tiny chunks.
    };

    /// @brief Estimate, from @p tab's constant runs, the memory each
    ///        candidate chunk size would give it. Read-only; call at a
    ///        quiescent point. Variant-generic via @p Arr (zero::Array /
    ///        zero_one_value::Array / zero_any::Array): the only per-variant part is the
    ///        cst2 classification, selected with if constexpr on chunk_config.
    template <typename Arr>
    static ChunkSizeReport analyze(const Arr& tab, const Options& opt = {})
    {
        ChunkSizeReport rep;

        const uint32_t chunk_sz = tab.chunk_size();
        const int64_t  nc       = tab.nb_chunk_;
        rep.current_size        = chunk_sz;
        rep.meta_bytes          = static_cast<double>(sizeof(typename Arr::ChunkType)) + opt.meta_extra_bytes;
        rep.elem_bytes =
            chunk_sz != 0 ? static_cast<double>(Arr::ChunkType::storage_units(chunk_sz)) * sizeof(T) / chunk_sz : 0.;

        // No chunk, or no geometry (never on a live array: every setter refuses
        // a zero size): nothing to analyse. Every division below is by
        // chunk_sz or by a candidate derived from it, hence this test first.
        if (nc <= 0 || chunk_sz == 0)
        {
            return rep;
        }

        // zero_one_value keys cst2 by the array-level second-value index; fetch it once.
        // zero has no cst2; zero_any keys cst2 per chunk by the values_load() pointer.
        [[maybe_unused]] const int chunk2_ind = [&]
        {
            // bool / Bit: one second value, `true`, no index --
            // get_ind_second_value() aborts for them; classify() below asks
            // the unindexed is_cst_val2().
            // A zero_any-family array in a zero_one_value build (pattern::Array):
            // no index, the zero_any branches below.
            if constexpr (chunk_config::use_zero_one_value && !is_zero_any_family_v<typename Arr::ChunkType> &&
                          !is_bool_like_v<T>)
            {
                return tab.get_ind_second_value();
            }
            else if constexpr (chunk_config::use_zero_one_value && !is_zero_any_family_v<typename Arr::ChunkType>)
            {
                return 0;
            }
            else
            {
                return -1;
            }
        }();

        // compressible_value: would a uniform run of THIS value be storable as
        // a constant? (classify() below asks whether a chunk already is one.)
        // zero: first_value only. zero_one_value: also the array's second value, when
        // designated (otherwise only first_value: the conservative answer).
        const InT<T> cst1_value = Arr::ChunkType::first_value;
        bool         has_cst2   = false;
        T            cst2_value = cst1_value;
        if constexpr ((chunk_config::use_zero_one_value || is_zero_any_family_v<typename Arr::ChunkType>) &&
                      requires { tab.get_second_value(); })
        {
            cst2_value = tab.get_second_value();
            has_cst2   = !(cst2_value == cst1_value);
        }
        const auto compressible_value = [&]([[maybe_unused]] const T& elem) -> bool
        {
            // zero_any: any uniform value is compressible (compress() registers
            // it per chunk, find_or_set). Do not apply the first_value gate
            // below to it: the finer rows would miss every other uniform
            // sub-block, and a costly coarsening could never look worse than
            // refining back.
            // The family, not the build: pattern::Array in a zero_one_value build too.
            if constexpr (is_zero_any_family_v<typename Arr::ChunkType>)
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

        // classify: kind 0 = cst1, 1 = cst2, 2 = mat; two cst2 chunks merge iff
        // same `key` (zero_any: buffer ptr; zero_one_value: one shared sentinel).
        const auto classify = [&](const auto& chunk, const void*& key) -> int
        {
            key = nullptr;
            if (chunk.is_cst_val1())
            {
                return 0;
            }
            if constexpr (has_motif_state_v<typename Arr::ChunkType>)
            {
                // A motif chunk (pattern::Array): is_cst_val2() for the write engine,
                // but no constant. Its data -- motif + offset, internal::motif_chunk_data() -- is
                // read as a materialised chunk's (no motif forecast: never optimistic).
                if (Arr::ChunkType::is_motif_ptr(chunk.values_load()))
                {
                    return 2;
                }
            }
            // The family, not the build (as compressible_value).
            if constexpr (is_zero_any_family_v<typename Arr::ChunkType>)
            {
                if (chunk.is_cst_val2())
                {
                    key = static_cast<const void*>(chunk.values_load());
                    return 1;
                }
            }
            else if constexpr (chunk_config::use_zero_one_value)
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
            // zero: cst1 or mat only.
            return 2;
        };

        // ── Candidate sizes ───────────────────────────────────────────────────
        // As in the header comment. Never above cst_buffer_size: rechunk_to
        // would refuse it.
        std::vector<uint32_t> cand;
        {
            const auto     cap = static_cast<uint64_t>(shrink_array::cst_buffer_size);
            const uint32_t lo = opt.finer_levels > 0 && opt.finer_levels < 32 ? chunk_sz >> opt.finer_levels : chunk_sz;
            const uint64_t hi_wanted = opt.coarser_levels > 0 && opt.coarser_levels < 32
                                           ? uint64_t{chunk_sz} << opt.coarser_levels
                                           : uint64_t{chunk_sz};
            const uint64_t hi        = hi_wanted < cap ? hi_wanted : (chunk_sz < cap ? cap : uint64_t{chunk_sz});
            cand.push_back(chunk_sz);
            for (int k = 1; k <= opt.finer_levels && k < 32; ++k)
            {
                const uint32_t sub = chunk_sz >> k;
                if (sub >= opt.min_size && is_valid_chunk_size(sub) && chunk_sz % sub == 0)
                {
                    cand.push_back(sub);
                }
            }
            for (int l = 1; l <= opt.coarser_levels && l < 32 && (uint64_t{chunk_sz} << l) <= cap; ++l)
            {
                cand.push_back(chunk_sz << l);
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
        //   * inside a run of constant chunks of one (kind, key): compressed,
        //     counted exactly per run without reading data;
        //   * covering a whole materialised chunk: materialised;
        //   * overlapping a materialised chunk only in part: ambiguous, settled
        //     by reading the data of a sample of the materialised chunks.
        // When S divides C there is no ambiguous window: the row is exact.
        //
        // An ambiguous window is attributed to ONE owner, its first chunk
        // materialised and overlapped in part, so sampling counts it once; the
        // count is scaled by mat0 / scanned.
        const int64_t len  = nc * static_cast<int64_t>(chunk_sz);
        const auto    nwin = [&](uint32_t cand_sz) { return (len + cand_sz - 1) / cand_sz; };

        // Windows of size C inside the constant run of chunks [r0, r1).
        const auto windows_inside = [&](int64_t r0, int64_t r1, uint32_t cand_sz) -> int64_t
        {
            const int64_t begin = r0 * chunk_sz;
            const int64_t wmin  = (begin + cand_sz - 1) / cand_sz;
            const int64_t wmax  = r1 == nc ? nwin(cand_sz) - 1 : (r1 * chunk_sz) / cand_sz - 1;
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
                inside[static_cast<std::size_t>(ci)] +=
                    windows_inside(seg.start, seg.end, cand[static_cast<std::size_t>(ci)]);
            }
        };

        // ── Classification, constant runs and sampling selection ──────────────
        // No per-chunk scratch array: each pass re-classifies the chunks it reads
        // (classify() is pure-read, so concurrent reads are race-free; analyze()
        // runs at a quiescent serial point, so a region here is safe). Storing
        // the classification (9 B per chunk, faulted in at every call) costs far
        // more than classifying twice.
        //
        // Work unit: a PART, a contiguous run of chunks.
        //  - a part counts the runs inside it and leaves its first and last ones
        //    to a serial merge -- the same runs as a sequential walk;
        //  - part p starts from the materialised count of the parts before it,
        //    so it picks exactly the sequential walk's chunks (every stride-th),
        //    concatenated in part order. The report is byte-identical.
        // Parts are indexed by the loop, never by a thread rank, so a serial build
        // runs the same code in order.
        int64_t mat0 = 0;

        std::vector<int64_t> inside(static_cast<std::size_t>(n_cand), 0); // windows inside constant runs
        std::vector<int64_t> found(static_cast<std::size_t>(n_cand), 0);  // compressible ambiguous windows
        bool                 any_sampled = false;                         // a candidate that S does not divide
        for (const uint32_t cand_sz : cand)
        {
            any_sampled = any_sampled || cand_sz % chunk_sz != 0;
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
        const auto part_range = [&](int64_t part, int64_t& j0, int64_t& j1)
        {
            j0 = nc * part / n_parts;
            j1 = nc * (part + 1) / n_parts;
        };

        pmp::team(
            pmp::Options{.use_team = nc > 16384},
            [&](pmp::Team& team)
            {
                team.single(
                    [&]
                    {
                        n_parts = std::min<int64_t>(static_cast<int64_t>(CHUNK_PMP_NUM_THREADS) * 4, nc);
                        part_mat.assign(static_cast<std::size_t>(n_parts), 0);
                        part_first.assign(static_cast<std::size_t>(n_parts), Seg{0, 0, -1, nullptr});
                        part_last.assign(static_cast<std::size_t>(n_parts), Seg{0, 0, -1, nullptr});
                        part_single.assign(static_cast<std::size_t>(n_parts), 0);
                        part_picks.assign(static_cast<std::size_t>(n_parts), {});
                    });
                // (implicit barrier: n_parts and the per-part vectors are set)

                // Runs + materialised count, one part per iteration.
                {
                    std::vector<int64_t> inside_loc(static_cast<std::size_t>(n_cand), 0);
                    team.for_dynamic<int64_t>(
                        0, n_parts, 1,
                        [&](int64_t part)
                        {
                            int64_t j0 = 0;
                            int64_t j1 = 0;
                            part_range(part, j0, j1);
                            int64_t n_mat      = 0;
                            Seg     cur        = {j0, j0, -1, nullptr};
                            bool    first_done = false;
                            for (int64_t j = j0; j < j1; ++j)
                            {
                                const void* key  = nullptr;
                                const int   kind = classify(tab.chunk_list_[j], key);
                                if (kind == 2)
                                {
                                    ++n_mat;
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
                                        part_first[static_cast<std::size_t>(part)] = cur;
                                        first_done                                 = true;
                                    }
                                }
                                cur = {j, j + 1, kind, key};
                            }
                            if (first_done)
                            {
                                part_last[static_cast<std::size_t>(part)] = cur;
                            }
                            else
                            {
                                part_first[static_cast<std::size_t>(part)]  = cur;
                                part_single[static_cast<std::size_t>(part)] = 1;
                            }
                            part_mat[static_cast<std::size_t>(part)] = n_mat;
                        },
                        pmp::nowait);
                    for (int ci = 0; ci < n_cand; ++ci)
                    {
                        if (inside_loc[static_cast<std::size_t>(ci)] != 0)
                        {
                            // Relaxed, as `omp atomic` was: the barrier below
                            // publishes the sums.
                            std::atomic_ref<int64_t>(inside[static_cast<std::size_t>(ci)])
                                .fetch_add(inside_loc[static_cast<std::size_t>(ci)], std::memory_order_relaxed);
                        }
                    }
                }
                team.barrier();

                // The runs across part boundaries; mat0, the sampling stride, and
                // each part's starting rank among the materialised chunks
                // (exclusive prefix sum, in place).
                team.single(
                    [&]
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
                            const int64_t n_mat                   = part_mat[static_cast<std::size_t>(p)];
                            part_mat[static_cast<std::size_t>(p)] = run_count;
                            run_count += n_mat;
                        }
                        mat0     = run_count;
                        sampling = any_sampled && mat0 > 0;
                        if (sampling)
                        {
                            stride = mat0 > opt.max_mat_scans ? (mat0 + opt.max_mat_scans - 1) / opt.max_mat_scans : 1;
                        }
                    });
                // (implicit barrier: inside, mat0, stride and the offsets are set)

                // ── Sampling: pick the materialised chunks whose DATA is read ─────
                // Every stride-th materialised chunk -- at most max_mat_scans, since
                // stride = ceil(mat0 / max_mat_scans) -- the same chunks as the
                // sequential walk, found part by part.
                if (sampling)
                {
                    team.for_dynamic<int64_t>(0, n_parts, 1,
                                              [&](int64_t part)
                                              {
                                                  int64_t j0 = 0;
                                                  int64_t j1 = 0;
                                                  part_range(part, j0, j1);
                                                  int64_t seen = part_mat[static_cast<std::size_t>(part)];
                                                  auto&   mine = part_picks[static_cast<std::size_t>(part)];
                                                  for (int64_t j = j0; j < j1; ++j)
                                                  {
                                                      const void* key = nullptr;
                                                      if (classify(tab.chunk_list_[j], key) !=
                                                          2) // skip cst1 / cst2 (only mat has data)
                                                      {
                                                          continue;
                                                      }
                                                      if ((seen++ % stride) != 0)
                                                      {
                                                          continue;
                                                      }
                                                      mine.push_back(j);
                                                  }
                                              });
                }
                team.single(
                    [&]
                    {
                        for (const auto& mine : part_picks)
                        {
                            picks.insert(picks.end(), mine.begin(), mine.end());
                        }
                        scanned = static_cast<int64_t>(picks.size());
                    });
                // (implicit barrier: picks is complete)

                // ── The ambiguous windows of each picked chunk ────────────────────
                {
                    std::vector<int64_t>  found_loc(static_cast<std::size_t>(n_cand), 0);
                    std::vector<uint32_t> seg_start; // the picked chunk's runs of equal values ...
                    std::vector<T>        seg_val;   // ... and their values


                    // Element index of a materialised buffer, and whether [first, last)
                    // of it is uniform (its value in `val`).
                    const auto value_of = [](const T* vals, std::size_t index) -> T
                    {
                        if constexpr (is_packed_bit_v<T>)
                        {
                            const auto* words = internal::as_words(vals);
                            return T{((words[index / bits::word_bits] >> (index % bits::word_bits)) & 1u) != 0};
                        }
                        else
                        {
                            return vals[index]; // raw-read: the advisor runs at a quiescent point (see analyze)
                        }
                    };
                    const auto uniform_range = [&](const T* vals, std::size_t first, std::size_t last, T& val) -> bool
                    {
                        if constexpr (is_packed_bit_v<T>)
                        {
                            const bits::Uniform scan = bits::scan_range(internal::as_words(vals), first, last - first);
                            val                      = T{scan == bits::Uniform::all_true};
                            return scan != bits::Uniform::mixed;
                        }
                        else
                        {
                            // raw-read: the advisor runs at a quiescent point (see analyze)
                            for (std::size_t i = first + 1; i < last; ++i)
                            {
                                // raw-read: as above
                                if (!(vals[i] == vals[first]))
                                {
                                    return false;
                                }
                            }
                            val = vals[first]; // raw-read: as above
                            return true;
                        }
                    };

                    const auto n_picks = static_cast<int64_t>(picks.size());
                    team.for_dynamic<int64_t>(
                        0, n_picks, 64,
                        [&](int64_t pick)
                        {
                            const int64_t chunk_idx = picks[static_cast<std::size_t>(pick)];
                            const T*      vj        = tab.chunk_list_[chunk_idx].values_load();
                            if constexpr (has_motif_state_v<typename Arr::ChunkType>)
                            {
                                vj = internal::motif_chunk_data(tab, chunk_idx, 0); // pattern::Array
                            }

                            // Chunk chunk_idx's runs of equal values: [b, e) of it is uniform iff
                            // b and e - 1 fall in the same run.
                            seg_start.clear();
                            seg_val.clear();
                            seg_start.push_back(0);
                            seg_val.push_back(value_of(vj, 0));
                            if constexpr (is_packed_bit_v<T>)
                            {
                                // A word at a time: the bits that differ from the one below.
                                const auto*       words = internal::as_words(vj);
                                const std::size_t nw    = (chunk_sz + bits::word_bits - 1) / bits::word_bits;
                                auto              below = static_cast<bits::WordT>(words[0] & 1u);
                                for (std::size_t k = 0; k < nw; ++k)
                                {
                                    const bits::WordT word        = words[k];
                                    bits::WordT       transitions = word ^ ((word << 1) | below);
                                    below                         = word >> (bits::word_bits - 1);
                                    while (transitions != 0)
                                    {
                                        const std::size_t bit_pos =
                                            (k * bits::word_bits) +
                                            static_cast<std::size_t>(std::countr_zero(transitions));
                                        transitions &= transitions - 1;
                                        if (bit_pos == 0 || bit_pos >= chunk_sz)
                                        {
                                            continue;
                                        }
                                        seg_start.push_back(static_cast<uint32_t>(bit_pos));
                                        seg_val.push_back(value_of(vj, bit_pos));
                                    }
                                }
                            }
                            else
                            {
                                for (uint32_t i = 1; i < chunk_sz; ++i)
                                {
                                    if (!(vj[i] == vj[i - 1]))
                                    {
                                        seg_start.push_back(i);
                                        seg_val.push_back(vj[i]);
                                    }
                                }
                            }
                            const auto seg_of = [&](int64_t index)
                            {
                                return std::upper_bound(seg_start.begin(), seg_start.end(),
                                                        static_cast<uint32_t>(index)) -
                                       seg_start.begin() - 1;
                            };

                            const int64_t x0 = chunk_idx * chunk_sz;
                            const int64_t x1 = x0 + chunk_sz;
                            for (int ci = 0; ci < n_cand; ++ci)
                            {
                                const uint32_t cand_sz = cand[static_cast<std::size_t>(ci)];
                                if (cand_sz % chunk_sz == 0)
                                {
                                    continue; // whole chunks only: no ambiguous window
                                }
                                for (int64_t w = x0 / cand_sz; w * cand_sz < x1; ++w)
                                {
                                    const int64_t a0 = w * cand_sz;
                                    const int64_t a1 = a0 + cand_sz < len ? a0 + cand_sz : len;
                                    if (a0 <= x0 && x1 <= a1)
                                    {
                                        continue; // covers chunk chunk_idx whole: materialised
                                    }
                                    const int64_t ca = a0 / chunk_sz;
                                    const int64_t cb = (a1 - 1) / chunk_sz;
                                    if (chunk_idx != ca)
                                    {
                                        // chunk_idx is the window's last chunk: the first one owns
                                        // the window when it is materialised in part too.
                                        const void* key = nullptr;
                                        if (a0 > ca * chunk_sz && classify(tab.chunk_list_[ca], key) == 2)
                                        {
                                            continue;
                                        }
                                    }
                                    bool ok    = true;
                                    bool has_v = false;
                                    T    val{};
                                    for (int64_t cc = ca; cc <= cb && ok; ++cc)
                                    {
                                        const int64_t p0  = a0 > cc * chunk_sz ? a0 : cc * chunk_sz;
                                        const int64_t p1  = a1 < (cc + 1) * chunk_sz ? a1 : (cc + 1) * chunk_sz;
                                        const auto&   ch  = tab.chunk_list_[cc];
                                        const void*   key = nullptr;
                                        T             part{};
                                        if (classify(ch, key) != 2)
                                        {
                                            part = ch.read_value_scan(0); // the constant (a scan read: no app stats)
                                        }
                                        else if (p0 == cc * chunk_sz && p1 == (cc + 1) * chunk_sz)
                                        {
                                            ok = false; // a whole materialised chunk
                                        }
                                        else if (cc == chunk_idx)
                                        {
                                            const auto seg = seg_of(p0 - x0);
                                            ok             = seg == seg_of(p1 - 1 - x0);
                                            part           = seg_val[static_cast<std::size_t>(seg)];
                                        }
                                        else
                                        {
                                            if constexpr (has_motif_state_v<typename Arr::ChunkType>)
                                            {
                                                ok =
                                                    uniform_range(internal::motif_chunk_data(tab, cc, 1),
                                                                  static_cast<std::size_t>(p0 - (cc * chunk_sz)),
                                                                  static_cast<std::size_t>(p1 - (cc * chunk_sz)), part);
                                            }
                                            else
                                            {
                                                ok = uniform_range(
                                                    ch.values_load(), static_cast<std::size_t>(p0 - (cc * chunk_sz)),
                                                    static_cast<std::size_t>(p1 - (cc * chunk_sz)), part);
                                            }
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
                        },
                        pmp::nowait);
                    for (int ci = 0; ci < n_cand; ++ci)
                    {
                        if (found_loc[static_cast<std::size_t>(ci)] != 0)
                        {
                            // Relaxed, as `omp atomic` was: the end of the
                            // region publishes the sums.
                            std::atomic_ref<int64_t>(found[static_cast<std::size_t>(ci)])
                                .fetch_add(found_loc[static_cast<std::size_t>(ci)], std::memory_order_relaxed);
                        }
                    }
                }
            });

        // ── Assemble the table (ascending sizes) ──────────────────────────────
        for (int ci = 0; ci < n_cand; ++ci)
        {
            const uint32_t cand_sz = cand[static_cast<std::size_t>(ci)];
            const bool     exact   = cand_sz % chunk_sz == 0;
            const int64_t  nch     = nwin(cand_sz);
            int64_t        mat     = 0;
            if (cand_sz == chunk_sz)
            {
                rep.cur = static_cast<int>(rep.table.size());
                mat     = mat0;
            }
            else
            {
                const double amb = scanned > 0 ? static_cast<double>(found[static_cast<std::size_t>(ci)]) *
                                                     static_cast<double>(mat0) / static_cast<double>(scanned)
                                               : 0.;
                mat              = nch - inside[static_cast<std::size_t>(ci)] - std::lround(amb);
                mat              = mat < 0 ? 0 : (mat > nch ? nch : mat);
            }
            ChunkSizeEstimate est;
            est.size      = cand_sz;
            est.n_chunks  = nch;
            est.mat       = mat;
            est.cst       = nch - mat;
            est.exact     = exact;
            est.mem_bytes = (static_cast<double>(nch) * rep.meta_bytes) +
                            (static_cast<double>(mat) * static_cast<double>(Arr::ChunkType::storage_units(cand_sz)) *
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
        write_table(rep, name, [](std::string_view line) { println("{}", line); });
    }

    /// @brief Write @p rep as a table to the log, channel advisor at debug
    ///        (CHUNK_LOG=advisor=debug), headed by @p name.
    static void log(const ChunkSizeReport& rep, const char* name)
    {
        write_table(rep, name, [](std::string_view line) { CHUNK_LOG(debug, advisor, "{}", line); });
    }

  private:
    /// @brief Hand the lines of @p rep's table, headed by @p name, to @p out
    ///        (a callable taking a std::string_view): print() and log().
    template <typename Out>
    static void write_table(const ChunkSizeReport& rep, const char* name, Out&& out)
    {
        out(std::format("── ChunkSizeAdvisor [{}] current S={}, meta={:.1f} B/chunk ──", name, rep.current_size,
                        rep.meta_bytes));
        for (std::size_t i = 0; i < rep.table.size(); ++i)
        {
            const auto& row = rep.table[i];
            out(std::format("   S={:5} : chunks={:9}  mat={:9} ({:5.1f}%)  mem={:9.2f} Mo {}{}{}", row.size,
                            static_cast<long long>(row.n_chunks), static_cast<long long>(row.mat),
                            row.n_chunks != 0 ? 100. * static_cast<double>(row.mat) / static_cast<double>(row.n_chunks)
                                              : 0.,
                            row.mem_bytes / (1024. * 1024.), row.exact ? "" : "~",
                            static_cast<int>(i) == rep.cur ? " <- current" : "",
                            static_cast<int>(i) == rep.best ? "  ** BEST **" : ""));
        }
        out(std::format("   estimated gain best vs current : {:.2f} Mo", rep.best_gain_bytes() / (1024. * 1024.)));

        // Re-chunk cost, when best != current: gain (bytes) and cost (ms) are
        // shown side by side. The constants are hardware specific
        // (rechunk_cost_model.h).
        if (rep.best >= 0 && rep.cur >= 0 && rep.best != rep.cur)
        {
            const auto&  cur_e = rep.table[static_cast<std::size_t>(rep.cur)];
            const double cost_ms =
                rechunk_cost_ms(cur_e.n_chunks, cur_e.size, cur_e.mat, rep.elem_bytes, CHUNK_PMP_MAX_THREADS);
            const auto gain_bytes = static_cast<std::size_t>(rep.best_gain_bytes() > 0 ? rep.best_gain_bytes() : 0.);
            // Observation-phase thresholds (not a decision, just a hint label):
            // >= 1 Mo gained and <= 50 ms spent reads as clearly worth it.
            const bool worth = rechunk_is_worth_it(gain_bytes, cost_ms, std::size_t{1} << 20U, 50.0);
            out(std::format("   estimated re-chunk cost       : {:.3f} ms  ({})", cost_ms,
                            worth ? "worth it" : "marginal"));
        }
    }
};

} // namespace shrink_array
