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

// Out-of-line bodies of internal::ChunkSizeScan (declared in
// chunk_size_scan.h, which includes this file after the class and whose file
// comment gives the window model and the passes).

#include <algorithm> // std::sort, std::unique, std::upper_bound, std::min
#include <atomic>    // std::atomic_ref (the per-thread counts merged)
#include <bit>       // std::bit_width, std::countr_zero
#include <cstddef>   // std::size_t
#include <cstdint>   // int64_t, uint32_t, std::uint64_t, std::uintptr_t
#include <vector>    // std::vector

#include "chunk/internal/chunk_size_scan.h"   // the declarations these bodies belong to
#include "chunk/bit.h"                        // is_bool_like_v, is_packed_bit_v
#include "chunk/config.h"                     // chunk_config::use_zero_or_value
#include "chunk/internal/pmp_chunk.h"         // CHUNK_PMP_NUM_THREADS (the parts per thread), pmp::team
#include "chunk/internal/bit_words.h"         // bits::scan_range, bits::Uniform, bits::WordT
#include "chunk/internal/chunk_size_policy.h" // shrink_array::const_buffer_size, is_valid_chunk_size
#include "chunk/internal/chunk_traits.h"      // is_zero_any_family_v, has_motif_state_v (pattern::Array)
#include "chunk/internal/element_access.h"    // internal::as_words: Bit sub-blocks scanned by word
#include "chunk/internal/relaxed_element.h"   // load_element_relaxed (a motif chunk's offset)
#include "chunk/internal/same_value.h"        // same_value

namespace shrink_array::internal
{

template <typename Arr>
ChunkSizeScan<Arr>::ChunkSizeScan(const Arr& arr, const Options& opt)
    : array_(arr)
    , opt_(opt)
    , chunk_sz_(arr.chunk_size())
    , nc_(arr.chunk_count_)
    , len_(arr.chunk_count_ * static_cast<int64_t>(arr.chunk_size()))
    , second_value_idx_(second_value_index(arr))
    , first_value_(ChunkT::first_value)
    , second_value_(first_value_)
    , cand_(candidate_sizes(arr.chunk_size(), opt))
    , n_cand_(static_cast<int>(cand_.size()))
{
    // zero_or_value and the zero_any family: the array's second value, when
    // designated (otherwise only first_value: the conservative answer).
    if constexpr ((chunk_config::use_zero_or_value || is_zero_any_family_v<ChunkT>) && requires { arr.second_value(); })
    {
        second_value_     = arr.second_value();
        has_second_value_ = !same_value(second_value_, first_value_);
    }
    for (const uint32_t cand_sz : cand_)
    {
        any_sampled_ = any_sampled_ || cand_sz % chunk_sz_ != 0;
    }
    inside_.assign(static_cast<std::size_t>(n_cand_), 0);
    found_.assign(static_cast<std::size_t>(n_cand_), 0);
}

template <typename Arr>
std::vector<uint32_t> ChunkSizeScan<Arr>::candidate_sizes(uint32_t chunk_sz, const Options& opt)
{
    std::vector<uint32_t> cand;
    const auto            cap = static_cast<uint64_t>(shrink_array::const_buffer_size);
    const uint32_t        lo  = opt.finer_levels > 0 && opt.finer_levels < 32 ? chunk_sz >> opt.finer_levels : chunk_sz;
    const uint64_t        hi_wanted = opt.coarser_levels > 0 && opt.coarser_levels < 32
                                          ? uint64_t{chunk_sz} << opt.coarser_levels
                                          : uint64_t{chunk_sz};
    const uint64_t        hi        = hi_wanted < cap ? hi_wanted : (chunk_sz < cap ? cap : uint64_t{chunk_sz});
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
            if (std::bit_width(m) - static_cast<unsigned>(std::countr_zero(m)) <= static_cast<unsigned>(opt.grid_bits))
            {
                cand.push_back(static_cast<uint32_t>(m * 64));
            }
        }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    return cand;
}

template <typename Arr>
int ChunkSizeScan<Arr>::second_value_index([[maybe_unused]] const Arr& arr)
{
    // bool / Bit: one second value, `true`, no index -- get_idx_second_value()
    // aborts for them; classify() asks the unindexed is_second_value().
    // A zero_any-family array in a zero_or_value build (pattern::Array): no
    // index, the zero_any branches of classify().
    if constexpr (chunk_config::use_zero_or_value && !is_zero_any_family_v<ChunkT> && !is_bool_like_v<T>)
    {
        return arr.get_idx_second_value();
    }
    else if constexpr (chunk_config::use_zero_or_value && !is_zero_any_family_v<ChunkT>)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

template <typename Arr>
bool ChunkSizeScan<Arr>::compressible_value([[maybe_unused]] const T& elem) const
{
    // zero_any: any uniform value is compressible (compress() registers it per
    // chunk, find_or_set). Do not apply the first_value gate below to it: the
    // finer rows would miss every other uniform sub-block, and a costly
    // coarsening could never look worse than refining back.
    // The family, not the build: pattern::Array in a zero_or_value build too.
    if constexpr (is_zero_any_family_v<ChunkT>)
    {
        return true;
    }
    else
    {
        if (same_value(elem, first_value_))
        {
            return true;
        }
        return has_second_value_ && same_value(elem, second_value_);
    }
}

template <typename Arr>
int ChunkSizeScan<Arr>::classify(const ChunkT& chunk, const void*& key) const
{
    key = nullptr;
    if (chunk.is_first_value())
    {
        return kind_first_value;
    }
    if constexpr (has_motif_state_v<ChunkT>)
    {
        // A motif chunk (pattern::Array): is_second_value() for the write engine,
        // but no constant, and no buffer of its own: a kind of its own,
        // counted and forecast apart (forecast_motifs()), never sampled as data.
        if (ChunkT::is_motif_ptr(chunk.values_load()))
        {
            return kind_motif;
        }
    }
    // The family, not the build (as compressible_value).
    if constexpr (is_zero_any_family_v<ChunkT>)
    {
        if (chunk.is_second_value())
        {
            key = static_cast<const void*>(chunk.values_load());
            return kind_second_value;
        }
    }
    else if constexpr (chunk_config::use_zero_or_value)
    {
        bool is_second = false;
        if constexpr (is_bool_like_v<T>)
        {
            is_second = chunk.is_second_value();
        }
        else
        {
            is_second = second_value_idx_ >= 0 && chunk.is_second_value(second_value_idx_);
        }
        if (is_second)
        {
            static constexpr int second_value_sentinel = 0; // one address: array-level, all second values equal
            key                                        = &second_value_sentinel;
            return kind_second_value;
        }
    }
    // zero: first value or mat only.
    return kind_mat;
}

template <typename Arr>
int64_t ChunkSizeScan<Arr>::windows_inside(int64_t r0, int64_t r1, uint32_t cand_sz) const
{
    const int64_t begin = r0 * chunk_sz_;
    const int64_t wmin  = (begin + cand_sz - 1) / cand_sz;
    const int64_t wmax  = r1 == nc_ ? windows(cand_sz) - 1 : (r1 * chunk_sz_) / cand_sz - 1;
    return wmax >= wmin ? wmax - wmin + 1 : 0;
}

template <typename Arr>
void ChunkSizeScan<Arr>::count_run(const Run& run, std::vector<int64_t>& inside) const
{
    if (run.kind < 0 || run.kind == kind_mat || run.kind == kind_motif)
    {
        return;
    }
    for (int ci = 0; ci < n_cand_; ++ci)
    {
        inside[static_cast<std::size_t>(ci)] += windows_inside(run.start, run.end, cand_[static_cast<std::size_t>(ci)]);
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::part_range(int64_t part, int64_t& j0, int64_t& j1) const
{
    j0 = nc_ * part / n_parts_;
    j1 = nc_ * (part + 1) / n_parts_;
}

template <typename Arr>
void ChunkSizeScan<Arr>::add_counts(std::vector<int64_t>& to, const std::vector<int64_t>& from)
{
    for (std::size_t i = 0; i < from.size(); ++i)
    {
        if (from[i] != 0)
        {
            // Relaxed, as `omp atomic` was.
            std::atomic_ref<int64_t>(to[i]).fetch_add(from[i], std::memory_order_relaxed);
        }
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::run()
{
    pmp::team(pmp::Options{.use_team = nc_ > team_min_chunks},
              [&](pmp::Team& team)
              {
                  team.single([&] { size_parts(); });
                  // (implicit barrier: n_parts_ and the per-part vectors are set)

                  count_runs(team);
                  team.barrier();

                  team.single([&] { merge_parts(); });
                  // (implicit barrier: inside_, mat0_, stride_ and the offsets are set)

                  if (sampling_)
                  {
                      team.for_dynamic<int64_t>(0, n_parts_, 1, [&](int64_t part) { pick_part(part); });
                  }
                  team.single([&] { gather_picks(); });
                  // (implicit barrier: picks_ is complete)

                  count_ambiguous(team);
              });
    forecast_motifs();
}

template <typename Arr>
void ChunkSizeScan<Arr>::size_parts()
{
    n_parts_ = std::min<int64_t>(static_cast<int64_t>(CHUNK_PMP_NUM_THREADS) * 4, nc_);
    part_mat_.assign(static_cast<std::size_t>(n_parts_), 0);
    part_first_.assign(static_cast<std::size_t>(n_parts_), Run{0, 0, kind_none, nullptr});
    part_last_.assign(static_cast<std::size_t>(n_parts_), Run{0, 0, kind_none, nullptr});
    part_single_.assign(static_cast<std::size_t>(n_parts_), 0);
    part_picks_.assign(static_cast<std::size_t>(n_parts_), {});
}

template <typename Arr>
void ChunkSizeScan<Arr>::scan_part(int64_t part, std::vector<int64_t>& inside_loc)
{
    int64_t j0 = 0;
    int64_t j1 = 0;
    part_range(part, j0, j1);
    int64_t n_mat      = 0;
    Run     cur        = {j0, j0, kind_none, nullptr};
    bool    first_done = false;
    for (int64_t j = j0; j < j1; ++j)
    {
        const void* key  = nullptr;
        const int   kind = classify(array_.chunk_list_[j], key);
        if (kind == kind_mat)
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
                part_first_[static_cast<std::size_t>(part)] = cur;
                first_done                                  = true;
            }
        }
        cur = {j, j + 1, kind, key};
    }
    if (first_done)
    {
        part_last_[static_cast<std::size_t>(part)] = cur;
    }
    else
    {
        part_first_[static_cast<std::size_t>(part)]  = cur;
        part_single_[static_cast<std::size_t>(part)] = 1;
    }
    part_mat_[static_cast<std::size_t>(part)] = n_mat;
}

template <typename Arr>
void ChunkSizeScan<Arr>::count_runs(pmp::Team& team)
{
    std::vector<int64_t> inside_loc(static_cast<std::size_t>(n_cand_), 0);
    team.for_dynamic<int64_t>(0, n_parts_, 1, [&](int64_t part) { scan_part(part, inside_loc); }, pmp::nowait);
    add_counts(inside_, inside_loc); // the caller's barrier publishes the sums
}

template <typename Arr>
void ChunkSizeScan<Arr>::merge_parts()
{
    Run open = {0, 0, kind_none, nullptr};
    for (int64_t p = 0; p < n_parts_; ++p)
    {
        if (const Run& first = part_first_[static_cast<std::size_t>(p)];
            open.kind == first.kind && open.key == first.key && open.end == first.start)
        {
            open.end = first.end;
        }
        else
        {
            count_run(open, inside_);
            open = first;
        }
        if (part_single_[static_cast<std::size_t>(p)] == 0)
        {
            count_run(open, inside_);
            open = part_last_[static_cast<std::size_t>(p)];
        }
    }
    count_run(open, inside_);

    int64_t run_count = 0;
    for (int64_t p = 0; p < n_parts_; ++p)
    {
        const int64_t n_mat                    = part_mat_[static_cast<std::size_t>(p)];
        part_mat_[static_cast<std::size_t>(p)] = run_count;
        run_count += n_mat;
    }
    mat0_     = run_count;
    sampling_ = any_sampled_ && mat0_ > 0;
    if (sampling_)
    {
        // At least one chunk read: max_mat_scans = 0 (or less) divided by zero here.
        const int64_t max_scans = std::max<int64_t>(opt_.max_mat_scans, 1);
        stride_                 = mat0_ > max_scans ? (mat0_ + max_scans - 1) / max_scans : 1;
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::pick_part(int64_t part)
{
    int64_t j0 = 0;
    int64_t j1 = 0;
    part_range(part, j0, j1);
    int64_t seen = part_mat_[static_cast<std::size_t>(part)];
    auto&   mine = part_picks_[static_cast<std::size_t>(part)];
    for (int64_t j = j0; j < j1; ++j)
    {
        // skip first value / second value (only mat has data)
        if (const void* key = nullptr; classify(array_.chunk_list_[j], key) != kind_mat)
        {
            continue;
        }
        if ((seen++ % stride_) != 0)
        {
            continue;
        }
        mine.push_back(j);
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::gather_picks()
{
    for (const auto& mine : part_picks_)
    {
        picks_.insert(picks_.end(), mine.begin(), mine.end());
    }
    scanned_ = static_cast<int64_t>(picks_.size());
}

template <typename Arr>
void ChunkSizeScan<Arr>::count_ambiguous(pmp::Team& team)
{
    std::vector<int64_t> found_loc(static_cast<std::size_t>(n_cand_), 0);
    ValueRuns            runs; // the picked chunk's runs of equal values
    const auto           n_picks = static_cast<int64_t>(picks_.size());
    team.for_dynamic<int64_t>(
        0, n_picks, 64, [&](int64_t pick) { scan_pick(picks_[static_cast<std::size_t>(pick)], runs, found_loc); },
        pmp::nowait);
    add_counts(found_, found_loc); // the end of the region publishes the sums
}

template <typename Arr>
auto ChunkSizeScan<Arr>::value_of(const T* vals, std::size_t index) -> T
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
}

template <typename Arr>
bool ChunkSizeScan<Arr>::uniform_range(const T* vals, std::size_t first, std::size_t last, T& val)
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
            if (!same_value(vals[i], vals[first]))
            {
                return false;
            }
        }
        val = vals[first]; // raw-read: as above
        return true;
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::bit_runs(const T* vals, ValueRuns& runs) const
{
    const auto*       words = internal::as_words(vals);
    const std::size_t nw    = (chunk_sz_ + bits::word_bits - 1) / bits::word_bits;
    auto              below = static_cast<bits::WordT>(words[0] & 1u);
    for (std::size_t k = 0; k < nw; ++k)
    {
        const bits::WordT word        = words[k];
        bits::WordT       transitions = word ^ ((word << 1) | below);
        below                         = word >> (bits::word_bits - 1);
        while (transitions != 0)
        {
            const std::size_t bit_pos = (k * bits::word_bits) + static_cast<std::size_t>(std::countr_zero(transitions));
            transitions &= transitions - 1;
            if (bit_pos == 0 || bit_pos >= chunk_sz_)
            {
                continue;
            }
            runs.start.push_back(static_cast<uint32_t>(bit_pos));
            runs.val.push_back(value_of(vals, bit_pos));
        }
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::value_runs(const T* vals, ValueRuns& runs) const
{
    runs.start.clear();
    runs.val.clear();
    runs.start.push_back(0);
    runs.val.push_back(value_of(vals, 0));
    if constexpr (is_packed_bit_v<T>)
    {
        bit_runs(vals, runs);
    }
    else
    {
        for (uint32_t i = 1; i < chunk_sz_; ++i)
        {
            // raw-read: the advisor runs at a quiescent point (see analyze)
            if (!same_value(vals[i], vals[i - 1]))
            {
                runs.start.push_back(i);
                runs.val.push_back(vals[i]); // raw-read: as above
            }
        }
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::scan_pick(int64_t chunk_idx, ValueRuns& runs, std::vector<int64_t>& found_loc) const
{
    const T* vj = array_.chunk_list_[chunk_idx].values_load();
    if constexpr (has_motif_state_v<ChunkT>)
    {
        vj = internal::motif_chunk_data(array_, chunk_idx, 0); // pattern::Array
    }
    value_runs(vj, runs);

    const int64_t x0 = chunk_idx * chunk_sz_;
    const int64_t x1 = x0 + chunk_sz_;
    for (int ci = 0; ci < n_cand_; ++ci)
    {
        const uint32_t cand_sz = cand_[static_cast<std::size_t>(ci)];
        if (cand_sz % chunk_sz_ == 0)
        {
            continue; // whole chunks only: no ambiguous window
        }
        for (int64_t w = x0 / cand_sz; w * cand_sz < x1; ++w)
        {
            const int64_t a0 = w * cand_sz;
            const int64_t a1 = a0 + cand_sz < len_ ? a0 + cand_sz : len_;
            if (a0 <= x0 && x1 <= a1)
            {
                continue; // covers chunk chunk_idx whole: materialised
            }
            if (!owns_window(chunk_idx, a0))
            {
                continue;
            }
            T          val{};
            const bool ok = window_value(a0, a1, chunk_idx, runs, val);
            // Uniform is not enough: the value must be one the variant can
            // store as a constant (compressible_value).
            found_loc[static_cast<std::size_t>(ci)] += ok && compressible_value(val) ? 1 : 0;
        }
    }
}

template <typename Arr>
bool ChunkSizeScan<Arr>::owns_window(int64_t chunk_idx, int64_t a0) const
{
    const int64_t ca = a0 / chunk_sz_;
    if (chunk_idx == ca)
    {
        return true;
    }
    // chunk_idx is the window's last chunk: the first one owns the window
    // when it is materialised in part too.
    const void* key = nullptr;
    return !(a0 > ca * chunk_sz_ && classify(array_.chunk_list_[ca], key) == kind_mat);
}

template <typename Arr>
bool ChunkSizeScan<Arr>::window_value(int64_t a0, int64_t a1, int64_t chunk_idx, const ValueRuns& runs, T& val) const
{
    const int64_t ca    = a0 / chunk_sz_;
    const int64_t cb    = (a1 - 1) / chunk_sz_;
    bool          ok    = true;
    bool          has_v = false;
    for (int64_t cc = ca; cc <= cb && ok; ++cc)
    {
        const int64_t p0 = a0 > cc * chunk_sz_ ? a0 : cc * chunk_sz_;
        const int64_t p1 = a1 < (cc + 1) * chunk_sz_ ? a1 : (cc + 1) * chunk_sz_;
        T             part{};
        ok = part_value(cc, p0, p1, chunk_idx, runs, part);
        if (ok && has_v)
        {
            ok = part == val;
        }
        val   = part;
        has_v = true;
    }
    return ok;
}

template <typename Arr>
bool ChunkSizeScan<Arr>::part_value(int64_t          cc,
                                    int64_t          p0,
                                    int64_t          p1,
                                    int64_t          chunk_idx,
                                    const ValueRuns& runs,
                                    T&               part) const
{
    const auto& ch  = array_.chunk_list_[cc];
    const void* key = nullptr;
    if (const int kc = classify(ch, key); kc != kind_mat && kc != kind_motif)
    {
        part = ch.read_value_scan(0); // the constant (a scan read: no app stats)
        return true;
    }
    if (p0 == cc * chunk_sz_ && p1 == (cc + 1) * chunk_sz_)
    {
        return false; // a whole materialised chunk
    }
    if (cc == chunk_idx)
    {
        const int64_t x0     = chunk_idx * chunk_sz_;
        const auto    run_of = [&](int64_t index)
        {
            return std::upper_bound(runs.start.begin(), runs.start.end(), static_cast<uint32_t>(index)) -
                   runs.start.begin() - 1;
        };
        const auto run = run_of(p0 - x0);
        part           = runs.val[static_cast<std::size_t>(run)];
        return run == run_of(p1 - 1 - x0);
    }
    const auto first = static_cast<std::size_t>(p0 - (cc * chunk_sz_));
    const auto last  = static_cast<std::size_t>(p1 - (cc * chunk_sz_));
    if constexpr (has_motif_state_v<ChunkT>)
    {
        return uniform_range(internal::motif_chunk_data(array_, cc, 1), first, last, part);
    }
    else
    {
        return uniform_range(ch.values_load(), first, last, part);
    }
}

// ── pattern::Array: the motif forecast ────────────────────────────────────
// At the current size, the motif chunks and the distinct motifs, counted.
// At another size C, a window [a0, a1) that holds no materialised chunk
// and at least one motif chunk holds the same normalised data as another
// window iff their parts match: for each chunk it overlaps, in order, a
// motif chunk's (motif pointer, first element in the motif, length) or a
// constant chunk's length, each with its first value minus the window's
// first value. Equal keys, equal data (the converse can fail: two motifs
// with an equal sub-range; the forecast then misses a share, never invents
// one). A key seen c >= 2 times: c - 1 motif windows, one materialised
// (the first sighting the pass leaves on probation, docs/algo_pattern.md
// section 3), one motif entry. Uniform windows among them are rare (a motif
// chunk is never uniform) and counted as motifs. Read from the pointers
// and offset slots alone, one candidate size per task; nothing to do on
// an array without a motif chunk.
template <typename Arr>
void ChunkSizeScan<Arr>::forecast_motifs()
{
    if constexpr (has_motif_state_v<ChunkT>)
    {
        motif_win_.assign(static_cast<std::size_t>(n_cand_), 0);
        motif_entries_.assign(static_cast<std::size_t>(n_cand_), 0);
        std::vector<const T*> motifs_now;
        for (int64_t j = 0; j < nc_; ++j)
        {
            const void* key = nullptr;
            if (classify(array_.chunk_list_[j], key) == kind_motif)
            {
                motifs_now.push_back(array_.chunk_list_[j].values_load());
            }
        }
        const auto motif0 = static_cast<int64_t>(motifs_now.size());
        std::sort(motifs_now.begin(), motifs_now.end());
        const auto distinct0 =
            static_cast<int64_t>(std::unique(motifs_now.begin(), motifs_now.end()) - motifs_now.begin());
        if (motif0 > 0)
        {
            pmp::team(
                pmp::Options{.use_team = nc_ > team_min_chunks}, [&](pmp::Team& team)
                { team.for_dynamic<int>(0, n_cand_, 1, [&](int ci) { forecast_candidate(ci, motif0, distinct0); }); });
        }
    }
}

template <typename Arr>
void ChunkSizeScan<Arr>::forecast_candidate(int ci, int64_t motif0, int64_t distinct0)
{
    const uint32_t cand_sz = cand_[static_cast<std::size_t>(ci)];
    if (cand_sz == chunk_sz_)
    {
        motif_win_[static_cast<std::size_t>(ci)]     = motif0;
        motif_entries_[static_cast<std::size_t>(ci)] = distinct0;
        return;
    }
    std::vector<std::uint64_t> keys;
    const int64_t              n_w = windows(cand_sz);
    for (int64_t w = 0; w < n_w; ++w)
    {
        std::uint64_t key = 0;
        if (motif_window_key(w, cand_sz, key))
        {
            keys.push_back(key);
        }
    }
    std::sort(keys.begin(), keys.end());
    int64_t n_motif = 0;
    int64_t n_entry = 0;
    for (std::size_t i = 0; i < keys.size();)
    {
        std::size_t k = i + 1;
        while (k < keys.size() && keys[k] == keys[i])
        {
            ++k;
        }
        if (k - i >= 2)
        {
            n_motif += static_cast<int64_t>(k - i) - 1;
            ++n_entry;
        }
        i = k;
    }
    motif_win_[static_cast<std::size_t>(ci)]     = n_motif;
    motif_entries_[static_cast<std::size_t>(ci)] = n_entry;
}

template <typename Arr>
bool ChunkSizeScan<Arr>::motif_window_key(int64_t win, uint32_t cand_sz, std::uint64_t& key) const
{
    const int64_t a0        = win * cand_sz;
    const int64_t a1        = a0 + cand_sz < len_ ? a0 + cand_sz : len_;
    const int64_t ca        = a0 / chunk_sz_;
    const int64_t cb        = (a1 - 1) / chunk_sz_;
    bool          has_motif = false;
    T             first{};
    std::uint64_t h = mix(0x51ED27C3A9F1B6D5ULL, static_cast<std::uint64_t>(a1 - a0));
    for (int64_t cc = ca; cc <= cb; ++cc)
    {
        const auto&   ch      = array_.chunk_list_[cc];
        const void*   key_ptr = nullptr;
        const int     kc      = classify(ch, key_ptr);
        const int64_t p0      = a0 > cc * chunk_sz_ ? a0 : cc * chunk_sz_;
        const int64_t p1      = a1 < (cc + 1) * chunk_sz_ ? a1 : (cc + 1) * chunk_sz_;
        const auto    from    = static_cast<std::uint64_t>(p0 - (cc * chunk_sz_));
        const auto    cnt     = static_cast<std::uint64_t>(p1 - p0);
        T             v0{};
        if (kc == kind_mat)
        {
            return false; // a materialised chunk: the sampled windows' business
        }
        if (kc == kind_motif)
        {
            const T* motif = ch.values_load();
            const T  off   = internal::load_element_relaxed(array_.motif_offset_slot(static_cast<int>(cc)));
            v0             = ChunkT::motif_value(motif, static_cast<int>(from), off);
            h              = mix(h, reinterpret_cast<std::uintptr_t>(motif));
            h              = mix(h, from);
            has_motif      = true;
        }
        else
        {
            v0 = ch.read_value_scan(0); // the constant (a scan read: no app stats)
            h  = mix(h, 0x0C0175A27ULL);
        }
        if (cc == ca)
        {
            first = v0;
        }
        h = mix(h, cnt);
        h = mix(h, static_cast<std::uint64_t>(static_cast<ChunkT::UnsignedT>(ChunkT::wrapping_sub(v0, first))));
    }
    key = h;
    return has_motif;
}

template <typename Arr>
std::uint64_t ChunkSizeScan<Arr>::mix(std::uint64_t hash, std::uint64_t value) noexcept
{
    hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 6U) + (hash >> 2U);
    hash *= 0xBF58476D1CE4E5B9ULL;
    return hash ^ (hash >> 31U);
}

} // namespace shrink_array::internal
