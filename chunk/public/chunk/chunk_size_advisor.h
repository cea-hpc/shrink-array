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
//     (S / 8 bytes per materialised chunk for Bit, BitArray: storage_units)
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
// (details at "The window model" in internal/chunk_size_scan.h, whose
// ChunkSizeScan does the counting). For zero_any the identity of a
// const_val2 constant IS the values_load() pointer (a SecondValueRegistry slot), so
// a constant run is a pointer comparison:
//         const1 + const1           -> const1
//         const2(p) + const2(p)     -> const2
//         anything else         -> materialized
// When S divides C the row is EXACT; otherwise partly-overlapped materialised
// chunks are sampled (capped) and extrapolated. Deliberately CONSERVATIVE for
// zero_any's mixed dual-value patterns. Constant-at-C windows count as
// compressed whichever constant (const2 registry growth is ignored).
//
// CONCURRENCY CONTRACT: call at a QUIESCENT phase boundary only (same contract
// as flush_pending_deferred_ops: no concurrent writers/compressors, deferred
// ops drained). The walk reads chunk states and, for finer sizes, chunk data.
// ============================================================================

#include <algorithm>   // std::ranges::min_element (best), std::min (the motif rows)
#include <cmath>       // std::lround (the extrapolated materialised count)
#include <cstddef>     // std::size_t
#include <cstdint>     // int64_t, uint32_t
#include <format>      // std::format (write_table's lines)
#include <string_view> // std::string_view (write_table's sink)
#include <vector>      // std::vector

#include "chunk/internal/chunk_size_scan.h"    // internal::ChunkSizeScan: analyze()'s counting passes
#include "chunk/internal/pmp_chunk.h"          // CHUNK_PMP_MAX_THREADS (write_table's re-chunk cost)
#include "chunk/internal/rechunk_cost_model.h" // rechunk_cost_ms, rechunk_is_worth_it
#include "chunk/print.h"                       // shrink_array::println
#include "chunk/log.h"                         // CHUNK_LOG (log(): channel advisor)
#include "chunk/internal/chunk_traits.h"       // has_motif_state_v (pattern::Array: the motif rows)

#include "chunk/internal/module_export.h" // SHRINK_ARRAY_EXPORT

namespace shrink_array
{

/// @brief Model figures for one candidate chunk size: chunk counts and estimated memory.
SHRINK_ARRAY_EXPORT struct ChunkSizeEstimate
{
    uint32_t size      = 0;  ///< candidate chunk size (elements)
    int64_t  n_chunks  = 0;  ///< total chunks at that size
    int64_t  mat       = 0;  ///< materialized chunks (estimated or exact)
    int64_t  constant  = 0;  ///< compressed chunks (const_val1 + const_val2)
    double   mem_bytes = 0.; ///< model memory (bytes): metadata, materialised data and,
                             ///< for pattern::Array, one entry per distinct motif
    bool    exact = false;   ///< true: exact fold; false: sampled estimate
    int64_t motif = 0;       ///< chunks compressed to a motif (pattern::Array; 0 otherwise):
                             ///< exact at the current size, forecast at the others
};

/// @brief ChunkSizeAdvisor's result: one ChunkSizeEstimate per candidate size, with the best and current indices.
SHRINK_ARRAY_EXPORT struct ChunkSizeReport
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
 * @brief The row of candidate @p ci in ChunkSizeAdvisor::analyze()'s table,
 *        from the counts of @p scan (run): the windows that stay materialised
 *        -- exact when the current size divides the candidate, the sampled
 *        ambiguous ones extrapolated otherwise -- and the model memory, with
 *        @p meta_bytes per chunk.
 */
template <typename Arr>
[[nodiscard]] ChunkSizeEstimate advisor_row(const ChunkSizeScan<Arr>& scan, int ci, double meta_bytes)
{
    using T                 = Arr::value_type;
    const uint32_t chunk_sz = scan.chunk_size();
    const uint32_t cand_sz  = scan.candidates()[static_cast<std::size_t>(ci)];
    const int64_t  nch      = scan.windows(cand_sz);
    int64_t        mat      = scan.materialised();
    if (cand_sz != chunk_sz)
    {
        const double amb = scan.scanned() > 0
                               ? static_cast<double>(scan.found(ci)) * static_cast<double>(scan.materialised()) /
                                     static_cast<double>(scan.scanned())
                               : 0.;
        mat              = nch - scan.inside(ci) - std::lround(amb);
        mat              = mat < 0 ? 0 : (mat > nch ? nch : mat);
    }
    ChunkSizeEstimate est;
    est.size      = cand_sz;
    est.n_chunks  = nch;
    est.mat       = mat;
    est.constant  = nch - mat;
    est.exact     = cand_sz % chunk_sz == 0;
    est.mem_bytes = (static_cast<double>(nch) * meta_bytes) +
                    (static_cast<double>(mat) * static_cast<double>(Arr::ChunkType::storage_units(cand_sz)) *
                     sizeof(T)); // Bit: size / 8 bytes, not size
    if constexpr (has_motif_state_v<typename Arr::ChunkType>)
    {
        // The motif windows were counted materialised above (no constant
        // run, no sampled uniform window holds a motif chunk): moved to
        // `motif`, plus one entry per distinct motif. At the current size
        // the materialised count never included the motif chunks: the
        // constants are what is left.
        const int64_t motif = std::min(scan.motif_windows(ci), cand_sz == chunk_sz ? nch - mat : mat);
        if (cand_sz != chunk_sz)
        {
            est.mat = mat - motif;
        }
        est.motif     = motif;
        est.constant  = nch - est.mat - motif;
        est.mem_bytes = (static_cast<double>(nch) * meta_bytes) +
                        (static_cast<double>(est.mat + scan.motif_entries(ci)) *
                         static_cast<double>(Arr::ChunkType::storage_units(cand_sz)) * sizeof(T));
    }
    return est;
}
} // namespace internal

/// @brief Estimates an array's model memory at other chunk sizes without re-chunking (see the header comment).
/// @tparam T Element type.
SHRINK_ARRAY_EXPORT template <typename T>
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

    /// @brief Estimate, from @p arr's constant runs, the memory each
    ///        candidate chunk size would give it. Read-only; call at a
    ///        quiescent point. Variant-generic via @p Arr (zero::Array /
    ///        zero_or_value::Array / zero_any::Array): the only per-variant part is the
    ///        const2 classification, selected with if constexpr on chunk_config.
    template <typename Arr>
    static ChunkSizeReport analyze(const Arr& arr, const Options& opt = {})
    {
        ChunkSizeReport rep;

        const uint32_t chunk_sz = arr.chunk_size();
        const int64_t  nc       = arr.chunk_count_;
        rep.current_size        = chunk_sz;
        rep.meta_bytes          = static_cast<double>(sizeof(typename Arr::ChunkType)) + opt.meta_extra_bytes;
        rep.elem_bytes =
            chunk_sz != 0 ? static_cast<double>(Arr::ChunkType::storage_units(chunk_sz)) * sizeof(T) / chunk_sz : 0.;

        // No chunk, or no geometry (never on a live array: every setter refuses
        // a zero size): nothing to analyse. Every division of the scan is by
        // chunk_sz or by a candidate derived from it, hence this test first.
        if (nc <= 0 || chunk_sz == 0)
        {
            return rep;
        }

        // The counts (internal/chunk_size_scan.h), then the table, ascending sizes.
        internal::ChunkSizeScan<Arr> scan(arr, opt);
        scan.run();
        const auto n_cand = static_cast<int>(scan.candidates().size());
        for (int ci = 0; ci < n_cand; ++ci)
        {
            rep.table.push_back(internal::advisor_row(scan, ci, rep.meta_bytes));
            if (rep.table.back().size == chunk_sz)
            {
                rep.cur = ci;
            }
        }

        // The first of the smallest footprints: on a tie the smaller size wins.
        rep.best = static_cast<int>(std::ranges::min_element(rep.table, {}, &ChunkSizeEstimate::mem_bytes) -
                                    rep.table.begin());
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
        // An index loop, not std::views::enumerate: libc++ has it from 20 only.
        for (std::size_t i = 0; i < rep.table.size(); ++i)
        {
            const auto& row = rep.table[i];
            out(std::format("   S={:5} : chunks={:9}  mat={:9} ({:5.1f}%)  mem={:9.2f} Mo {}{}{}{}", row.size,
                            static_cast<long long>(row.n_chunks), static_cast<long long>(row.mat),
                            row.n_chunks != 0 ? 100. * static_cast<double>(row.mat) / static_cast<double>(row.n_chunks)
                                              : 0.,
                            row.mem_bytes / (1024. * 1024.), row.exact ? "" : "~",
                            row.motif != 0 ? std::format("  motif={}", static_cast<long long>(row.motif)) : "",
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
            const bool worth = rechunk_is_worth_it(gain_bytes, cost_ms, 1uz << 20U, 50.0);
            out(std::format("   estimated re-chunk cost       : {:.3f} ms  ({})", cost_ms,
                            worth ? "worth it" : "marginal"));
        }
    }
};

} // namespace shrink_array
