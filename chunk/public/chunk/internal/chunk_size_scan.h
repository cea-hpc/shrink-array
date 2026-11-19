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
// ChunkSizeScan — the passes of ChunkSizeAdvisor::analyze() over one array.
// ----------------------------------------------------------------------------
// analyze() (chunk_size_advisor.h, whose header comment gives the model) asks
// one question per candidate chunk size C: how many windows of C elements
// would stay compressed. This class answers it with counts, and knows nothing
// of bytes: analyze() turns them into the report's rows.
//
// The window model. At a candidate size C the array is cut into windows
// [w*C, (w+1)*C) of the current extent L = nc * S (the last one shorter). A
// window stays compressed iff it holds one compressible value -- read from
// the CURRENT chunks it overlaps:
//   * inside a run of constant chunks of one (kind, key): compressed,
//     counted exactly per run without reading data (inside());
//   * covering a whole materialised chunk: materialised;
//   * overlapping a materialised chunk only in part: ambiguous, settled
//     by reading the data of a sample of the materialised chunks (found(),
//     out of scanned() chunks read among materialised()).
// When S divides C there is no ambiguous window: the row is exact.
//
// An ambiguous window is attributed to ONE owner, its first chunk
// materialised and overlapped in part, so sampling counts it once; analyze()
// scales the count by materialised() / scanned().
//
// The passes, in run()'s order, one member function each:
//   size_parts      the work units (PARTS, contiguous runs of chunks)
//   count_runs      per part: the constant runs and the materialised count
//   merge_parts     serial: the runs across part boundaries, the sampling stride
//   pick_part       per part: the materialised chunks whose data is read
//   count_ambiguous per picked chunk: its ambiguous windows that are uniform
//   forecast_motifs pattern::Array only: the motif windows per candidate
//
// No per-chunk scratch array: each pass re-classifies the chunks it reads
// (classify() is pure-read, so concurrent reads are race-free; analyze() runs
// at a quiescent serial point, so a region here is safe). Storing the
// classification (9 B per chunk, faulted in at every call) costs far more
// than classifying twice.
//
// A part counts the runs inside it and leaves its first and last ones to the
// serial merge -- the same runs as a sequential walk; part p starts from the
// materialised count of the parts before it, so it picks exactly the
// sequential walk's chunks (every stride-th), concatenated in part order. The
// counts are identical whatever the team: the report is byte-identical. Parts
// are indexed by the loop, never by a thread rank, so a serial build runs the
// same code in order.
// ============================================================================

#include <array>   // std::array (motif_chunk_data's buffers)
#include <cstddef> // std::size_t
#include <cstdint> // int64_t, uint32_t, std::uint64_t
#include <vector>  // std::vector

#include "chunk/internal/in_t.h"            // InT<T>: the first value's snapshot
#include "chunk/internal/pmp_chunk.h"       // pmp::Team (the passes run in analyze()'s team)
#include "chunk/internal/chunk_traits.h"    // has_motif_state_v (pattern::Array)
#include "chunk/internal/relaxed_element.h" // load_element_relaxed (a motif chunk's offset)
#include "chunk/internal/module_export.h"   // SHRINK_ARRAY_EXPORT (ChunkSizeAdvisor)

namespace shrink_array
{

// Marked as its definition is: the module exports a name from its first
// declaration only (chunk/internal/module_export.h).
SHRINK_ARRAY_EXPORT template <typename T>
struct ChunkSizeAdvisor; // chunk_size_advisor.h: its Options are this scan's

namespace internal
{
/**
 * @brief The data of chunk @p idx of @p arr, a member with the motif state
 *        (pattern::Array): its buffer; for a motif chunk, its motif plus its
 *        offset, written into this thread's scratch buffer @p which (0 or 1:
 *        the analysis holds two chunks' data at once).
 * @details For ChunkSizeScan, at a quiescent point: the slot is stable. Called
 *          from discarded branches for the other variants.
 */
template <typename Arr>
[[nodiscard]] const Arr::value_type* motif_chunk_data(const Arr& arr, int64_t idx, int which)
{
    using ChunkT = Arr::ChunkType;
    using T      = Arr::value_type;
    static thread_local std::array<std::vector<T>, 2> buffers;
    std::vector<T>&                                   scratch = buffers[static_cast<std::size_t>(which)];
    const T* const                                    vals    = arr.chunk_list_[idx].values_load();
    if (!ChunkT::is_motif_ptr(vals))
    {
        return vals;
    }
    const T        off      = load_element_relaxed(arr.motif_offset_slot(static_cast<int>(idx)));
    const uint32_t chunk_sz = arr.chunk_size();
    scratch.resize(chunk_sz);
    for (uint32_t i = 0; i < chunk_sz; ++i)
    {
        scratch[i] = ChunkT::motif_value(vals, static_cast<int>(i), off);
    }
    return scratch.data();
}

/// @brief The counting passes of ChunkSizeAdvisor::analyze() over @p Arr (see
///        the file comment): per candidate size, the windows that would stay
///        compressed. Built on a non-empty array at a quiescent point; run()
///        once, then read the counts.
/// @tparam Arr zero::Array, zero_or_value::Array, zero_any::Array or
///         pattern::Array: the only per-variant parts are classify() and
///         compressible_value(), selected with if constexpr.
template <typename Arr>
class ChunkSizeScan
{
  public:
    using T       = Arr::value_type;              ///< the element type
    using ChunkT  = Arr::ChunkType;               ///< the chunk type
    using Options = ChunkSizeAdvisor<T>::Options; ///< analyze()'s options

    /// @brief classify()'s kinds. A motif chunk (pattern::Array) has a kind of
    ///        its own; for the other variants, which have none, kind_motif is
    ///        kind_mat and every test on it folds away.
    static constexpr int kind_none   = -1;
    static constexpr int kind_first_value = 0;                                        ///< the first constant
    static constexpr int kind_second_value = 1;                                        ///< a second constant
    static constexpr int kind_mat    = 2;                                        ///< materialised
    static constexpr int kind_motif  = has_motif_state_v<ChunkT> ? 3 : kind_mat; ///< a motif chunk

    /// @brief Chunks from which the passes run in a team.
    static constexpr int64_t team_min_chunks = 16384;

    /// @brief Bind to @p arr (at least one chunk, a non-zero chunk size) and
    ///        list the candidate sizes @p opt asks for.
    ChunkSizeScan(const Arr& arr, const Options& opt);

    /// @brief Run the passes (see the file comment). Once.
    void run();

    /// @brief The candidate sizes, ascending, the current one among them.
    [[nodiscard]] const std::vector<uint32_t>& candidates() const noexcept
    {
        return cand_;
    }
    /// @brief The array's chunk size now.
    [[nodiscard]] uint32_t chunk_size() const noexcept
    {
        return chunk_sz_;
    }
    /// @brief Windows (chunks) the array would have at size @p cand_sz.
    [[nodiscard]] int64_t windows(uint32_t cand_sz) const noexcept
    {
        return (len_ + cand_sz - 1) / cand_sz;
    }
    /// @brief Materialised chunks at the current size (motif chunks apart).
    [[nodiscard]] int64_t materialised() const noexcept
    {
        return mat0_;
    }
    /// @brief Materialised chunks whose data the sampling read.
    [[nodiscard]] int64_t scanned() const noexcept
    {
        return scanned_;
    }
    /// @brief Windows of candidate @p ci inside a run of constant chunks.
    [[nodiscard]] int64_t inside(int ci) const noexcept
    {
        return inside_[static_cast<std::size_t>(ci)];
    }
    /// @brief Ambiguous windows of candidate @p ci found uniform and
    ///        compressible, among those the scanned chunks own.
    [[nodiscard]] int64_t found(int ci) const noexcept
    {
        return found_[static_cast<std::size_t>(ci)];
    }
    /// @brief pattern::Array: windows of candidate @p ci forecast as motifs.
    [[nodiscard]] int64_t motif_windows(int ci) const noexcept
    {
        return motif_win_[static_cast<std::size_t>(ci)];
    }
    /// @brief pattern::Array: distinct motifs forecast for candidate @p ci.
    [[nodiscard]] int64_t motif_entries(int ci) const noexcept
    {
        return motif_entries_[static_cast<std::size_t>(ci)];
    }

  private:
    /// @brief A run of chunks [start, end) of one (kind, key).
    struct Run
    {
        int64_t     start;
        int64_t     end;
        int         kind; // kind_none, kind_first_value, kind_second_value, kind_mat, kind_motif
        const void* key;
    };

    /// @brief A materialised chunk's runs of equal values: [b, e) of it is
    ///        uniform iff b and e - 1 fall in the same run.
    struct ValueRuns
    {
        std::vector<uint32_t> start; ///< each run's first element
        std::vector<T>        val;   ///< ... and its value
    };

    // ── Candidates and classification ─────────────────────────────────────

    /// @brief The sizes analyze() evaluates from @p chunk_sz, as the header
    ///        comment of chunk_size_advisor.h lists them. Never above
    ///        const_buffer_size: rechunk_to would refuse it.
    [[nodiscard]] static std::vector<uint32_t> candidate_sizes(uint32_t chunk_sz, const Options& opt);

    /// @brief The key zero_or_value gives its const2 chunks: the array-level
    ///        second-value index, fetched once (0 for bool / Bit, which have
    ///        one second value and no index; -1 for the other variants).
    [[nodiscard]] static int second_value_index(const Arr& arr);

    /// @brief Would a uniform run of @p elem be storable as a constant?
    ///        (classify() asks whether a chunk already is one.) zero:
    ///        first_value only. zero_or_value: also the array's second
    ///        value, when designated. zero_any: any value.
    [[nodiscard]] bool compressible_value(const T& elem) const;

    /// @brief @p chunk's kind; two kind_second_value chunks merge iff same @p key
    ///        (zero_any: the buffer pointer; zero_or_value: one sentinel).
    [[nodiscard]] int classify(const ChunkT& chunk, const void*& key) const;

    // ── Runs of constant chunks ───────────────────────────────────────────

    /// @brief Windows of size @p cand_sz inside the constant run of chunks [@p r0, @p r1).
    [[nodiscard]] int64_t windows_inside(int64_t r0, int64_t r1, uint32_t cand_sz) const;

    /// @brief Add to @p inside, per candidate, the windows inside @p run (a
    ///        constant one; nothing for the other kinds).
    void count_run(const Run& run, std::vector<int64_t>& inside) const;

    /// @brief Chunks [@p j0, @p j1) of part @p part.
    void part_range(int64_t part, int64_t& j0, int64_t& j1) const;

    /// @brief How many parts, and their per-part vectors. In a `single`.
    void size_parts();

    /// @brief Part @p part: its runs counted into @p inside_loc but the first
    ///        and the last, left to merge_parts(); its materialised count.
    void scan_part(int64_t part, std::vector<int64_t>& inside_loc);

    /// @brief Every thread of @p team: scan_part() over the parts, then its
    ///        counts added to the shared ones. No barrier at the end.
    void count_runs(pmp::Team& team);

    /// @brief The runs across part boundaries; the materialised total, the
    ///        sampling stride, and each part's starting rank among the
    ///        materialised chunks (exclusive prefix sum, in place). In a `single`.
    void merge_parts();

    // ── Sampling ──────────────────────────────────────────────────────────

    /// @brief Part @p part: every stride-th materialised chunk -- at most
    ///        max_mat_scans in all, the same chunks as a sequential walk.
    void pick_part(int64_t part);

    /// @brief The parts' picks, concatenated in part order. In a `single`.
    void gather_picks();

    /// @brief Every thread of @p team: scan_pick() over the picked chunks,
    ///        then its counts added to the shared ones.
    void count_ambiguous(pmp::Team& team);

    /// @brief Element @p index of a materialised buffer.
    [[nodiscard]] static T value_of(const T* vals, std::size_t index);

    /// @brief Whether [@p first, @p last) of a materialised buffer is uniform
    ///        (its value in @p val).
    [[nodiscard]] static bool uniform_range(const T* vals, std::size_t first, std::size_t last, T& val);

    /// @brief @p runs := the runs of equal values of the chunk whose data is @p vals.
    void value_runs(const T* vals, ValueRuns& runs) const;

    /// @brief value_runs() for Bit: a word at a time, the bits that differ
    ///        from the one below.
    void bit_runs(const T* vals, ValueRuns& runs) const;

    /// @brief The ambiguous windows chunk @p chunk_idx owns, per candidate:
    ///        those found uniform and compressible added to @p found_loc.
    ///        @p runs is the calling thread's scratch.
    void scan_pick(int64_t chunk_idx, ValueRuns& runs, std::vector<int64_t>& found_loc) const;

    /// @brief Is chunk @p chunk_idx, materialised, the owner of the window
    ///        starting at element @p a0? (Its first chunk materialised and
    ///        overlapped in part.)
    [[nodiscard]] bool owns_window(int64_t chunk_idx, int64_t a0) const;

    /// @brief Does the window [@p a0, @p a1) hold one value (in @p val)?
    ///        Read from the chunks it overlaps; @p runs are those of chunk
    ///        @p chunk_idx, whose first element is @p x0.
    [[nodiscard]] bool window_value(int64_t a0, int64_t a1, int64_t chunk_idx, const ValueRuns& runs, T& val) const;

    /// @brief The part [@p p0, @p p1) of chunk @p cc: one value (in @p part)?
    ///        False for a whole materialised chunk.
    [[nodiscard]] bool
    part_value(int64_t cc, int64_t p0, int64_t p1, int64_t chunk_idx, const ValueRuns& runs, T& part) const;

    // ── pattern::Array: the motif forecast ────────────────────────────────

    /// @brief The motif windows and distinct motifs per candidate (see the
    ///        comment of the definition). Nothing for the other variants, nor
    ///        on an array without a motif chunk.
    void forecast_motifs();

    /// @brief Candidate @p ci's motif windows and entries; @p motif0 and
    ///        @p distinct0 are the counts at the current size.
    void forecast_candidate(int ci, int64_t motif0, int64_t distinct0);

    /// @brief The key of window @p win at size @p cand_sz, in @p key: false
    ///        when it holds a materialised chunk or no motif chunk.
    [[nodiscard]] bool motif_window_key(int64_t win, uint32_t cand_sz, std::uint64_t& key) const;

    /// @brief One step of the window keys' hash.
    [[nodiscard]] static std::uint64_t mix(std::uint64_t hash, std::uint64_t value) noexcept;

    /// @brief @p to[i] += @p from[i], atomically (relaxed): a thread's counts
    ///        into the shared ones; a barrier or the region's end publishes them.
    static void add_counts(std::vector<int64_t>& to, const std::vector<int64_t>& from);

    const Arr&     array_;
    const Options& opt_;
    uint32_t       chunk_sz_; ///< the array's chunk size now (S)
    int64_t        nc_;       ///< its chunks
    int64_t        len_;      ///< nc_ * chunk_sz_: the extent the windows cut

    int    second_value_idx_;   ///< second_value_index()
    InT<T> first_value_;       ///< the first constant
    bool   has_second_value_ = false; ///< a second value is designated, distinct from the first
    T      second_value_;       ///< ... that value

    std::vector<uint32_t> cand_;                ///< candidate_sizes()
    int                   n_cand_      = 0;     ///< their count
    bool                  any_sampled_ = false; ///< a candidate that S does not divide

    int64_t                           n_parts_ = 1;
    std::vector<int64_t>              part_mat_;    ///< materialised chunks per part, then their offsets
    std::vector<Run>                  part_first_;  ///< each part's first run
    std::vector<Run>                  part_last_;   ///< ... and its last one
    std::vector<char>                 part_single_; ///< the part is one run
    std::vector<std::vector<int64_t>> part_picks_;  ///< the chunks each part selected

    int64_t              mat0_     = 0;     ///< materialised chunks at the current size
    int64_t              stride_   = 1;     ///< one materialised chunk in stride_ is read
    bool                 sampling_ = false; ///< there are ambiguous windows to settle
    std::vector<int64_t> picks_;            ///< chunks whose data the sampling reads
    int64_t              scanned_ = 0;      ///< their count

    std::vector<int64_t> inside_;        ///< per candidate: windows inside constant runs
    std::vector<int64_t> found_;         ///< per candidate: compressible ambiguous windows
    std::vector<int64_t> motif_win_;     ///< per candidate: motif windows (pattern::Array)
    std::vector<int64_t> motif_entries_; ///< per candidate: distinct motifs (pattern::Array)
};

} // namespace internal
} // namespace shrink_array

#include "chunk/internal/chunk_size_scan_inline.h" // the method bodies
