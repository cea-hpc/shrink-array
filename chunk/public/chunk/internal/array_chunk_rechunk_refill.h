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

/**
 * @file array_chunk_rechunk_refill.h
 * @brief The re-chunk's refill: RechunkSupport<Derived, true>::Refill rebuilds
 *        the fresh array, chunk by chunk, from the array being re-chunked.
 *
 * @details One Refill per re-chunk, built and run by rechunk_refill(). It holds
 *          what the rebuild of every new chunk shares (the two arrays, the two
 *          geometries, the second-value lookups, the per-old-chunk release
 *          counters, the slice state), so each step of rebuild_one() is a
 *          method of its own:
 *
 *            scan_sources     what the old chunks under the new one hold
 *            install_cst2     zero_any: one shared constant, installed, no buffer
 *            read_span        bulk read of the old chunks into the buffer
 *            release_sources  free the old chunks once fully transcribed
 *            copy_range       fill the new chunk (the public bulk API)
 *            fold_back        zero_any / zero_one_value: compress it at once
 *
 *          run() drives the slices (16 with the fast allocator and threads, 1
 *          otherwise or for ArrayChunkBits), each a pmp::team region whose closing slice_boundary()
 *          sends the slice's released blocks home and recycles the slabs they
 *          emptied.
 *
 *          Nested in RechunkSupport<Derived, true>, so it has the access the
 *          array's friendship gives the mixin (chunk_list_, m_size_, num_tab_...).
 *
 * @see array_chunk_rechunk.h, rechunk_refill() for the contract
 * @see array_chunk_rechunk_refill_inline.h for the bodies
 */

#include <array>   // std::array (the rebuild schedule's text)
#include <atomic>  // the release counters, the census totals
#include <chrono>  // steady_clock (the slice boundaries' timers)
#include <cstddef> // std::size_t
#include <utility> // std::declval (SecondInfo)
#include <vector>  // the release counters, the recycling scratch

#include "chunk/internal/PMP_CHUNK.h"                 // pmp::Team, ChunkPmpSchedule
#include "chunk/bit.h"                                // is_packed_bit_v (releases_sources)
#include "chunk/internal/chunk_traits.h"              // is_zero_any_family_v
#include "chunk/config.h"                             // chunk_config::use_zero_one_value
#include "chunk/internal/array_chunk_rechunk.h"       // RechunkSupport, the class Refill is nested in
#include "chunk/internal/array_chunk_rechunk_trace.h" // SliceProbe

namespace shrink_array::internal
{
/// @brief The rebuild loop's schedule: guided,16. Slightly faster than
///        dynamic,64 on Grace, and large dynamic chunks are much slower on
///        small re-chunks; on Milan (64 threads) guided,16, static and
///        dynamic,64 are within 1% for zero and zero_any, static is 7.6%
///        slower for zero_one_value. Fixed: the former CHUNK_RECHUNK_SCHEDULE
///        is gone. A serial build ignores it (chunk_pmp_set_schedule() is then
///        a no-op).
struct RebuildSchedule
{
    ChunkPmpSchedule     schedule{.kind = ChunkPmpSched::sched_guided, .chunk = 16}; ///< the loop's schedule
    std::array<char, 32> text{"guided,16"};                                          ///< the same, for the trace
};

/// @brief The rebuild loop's schedule (see RebuildSchedule).
inline const RebuildSchedule& rebuild_schedule() noexcept
{
    static constexpr RebuildSchedule schedule{};
    return schedule;
}

/// @brief The refill folds through zero_one_value's second-value index: a
///        zero_one_value build, a chunk outside the zero_any family (not
///        pattern::Array) and a non-bool element (bool registers no second
///        value: get_ind_second_value() aborts).
/// @details The test on Derived::is_not_bool is NESTED in a discarded branch:
///          `&&` short-circuits evaluation, never name lookup, and zero::Array
///          has no is_not_bool. Instantiated in the bodies only, where Derived
///          is complete.
template <typename Derived, typename ChunkType>
inline constexpr bool rechunk_second_index_v = []
{
    if constexpr (chunk_config::use_zero_one_value && !is_zero_any_family_v<ChunkType>)
    {
        return Derived::is_not_bool;
    }
    else
    {
        return false;
    }
}();

/// @brief The SOURCE array's second-value index (the scan reads it), or 0 when
///        the refill does not fold through one (rechunk_second_index_v).
template <typename Derived, typename ChunkType>
[[nodiscard]] int rechunk_second_ind(const Derived& array) noexcept
{
    if constexpr (rechunk_second_index_v<Derived, ChunkType>)
    {
        return array.get_ind_second_value();
    }
    else
    {
        return 0;
    }
}

/// @brief The FRESH array's second-value record (the fold writes it;
///        rechunk_seed_fresh() has declared it), or 0 as above. Deduced, not
///        named: only zero_one_value::Array has InfArrayT, and a discarded
///        branch is never instantiated.
template <typename Derived, typename ChunkType>
[[nodiscard]] auto rechunk_second_info(const Derived& array) noexcept
{
    if constexpr (rechunk_second_index_v<Derived, ChunkType>)
    {
        return array.get_info_second_value();
    }
    else
    {
        return 0;
    }
}
} // namespace shrink_array::internal

namespace shrink_array
{

/**
 * @brief The refill of one re-chunk: the fresh array @c tmp (new geometry)
 *        rebuilt from @c src (old geometry), chunk by chunk.
 *
 * @details Pure read of @c src (quiescent: rechunk_to() has flushed its
 *          deferred ops and dropped its candidates) and armed writes of @c tmp.
 *          rebuild_one() is independent across new chunks: each is written by
 *          exactly one thread, the reads are const, and the write path
 *          partitions allocator / compression state by thread rank.
 *
 *          Materialisation is tracked with is_cst_val1() (values == the cst1
 *          sentinel), universal and index-free -- unlike written()/is_cst_val()
 *          on zero_one_value, which need the second_value index. While filled,
 *          a chunk only goes cst1 -> real buffer (never cst2, a compression
 *          result), so "still cst1" == "not yet materialised" for every variant.
 */
template <typename Derived>
class RechunkSupport<Derived, true>::Refill
{
  public:
    /**
     * @brief Prepare the refill of @p tmp from @p src.
     * @param src the array being re-chunked (read side, OLD geometry)
     * @param tmp the fresh array (write side, NEW geometry)
     * @param old_size chunk size of @p src
     * @param new_size chunk size of @p tmp
     * @param old_key the allocator key @p src's storage was built under (the
     *        set the slice boundaries empty)
     */
    Refill(Derived& src, Derived& tmp, int old_size, int new_size, unsigned int old_key);

    Refill(const Refill&)            = delete;
    Refill& operator=(const Refill&) = delete;
    Refill(Refill&&)                 = delete;
    Refill& operator=(Refill&&)      = delete;
    ~Refill()                        = default;

    /// @brief Rebuild every new chunk, slice by slice; then the memory census.
    void run();

  private:
    using ManagerT   = Traits::ManagerType;       ///< the manager (its team drain)
    using SliceClock = std::chrono::steady_clock; ///< the slice boundaries' clock

    /// zero_one_value folds through its second-value index; the zero_any
    /// family carries its constants implicitly.
    static constexpr bool second_index = internal::rechunk_second_index_v<Derived, ChunkType>;
    /// The refill frees each old chunk once transcribed (release_sources()):
    /// threaded builds, every variant and element type but Bit. An
    /// ArrayChunkBits (zero_one_value::Array<Bit>) packs 8 elements per byte
    /// and has a geometry of its own (packed_bit_birth_factor): its rebuild
    /// stays one pass that frees nothing early. Array<bool> is one byte per
    /// bool, released like any other type.
    static constexpr bool releases_sources = build_config::threaded && !is_packed_bit_v<T>;
    /// The fresh array's second-value record (zero_one_value), int otherwise.
    using SecondInfo = decltype(internal::rechunk_second_info<Derived, ChunkType>(std::declval<const Derived&>()));

    /// Per-thread fold counters, merged once per thread and slice after the
    /// loop (merge()): ++ on the shared atomics, per chunk, would bounce a
    /// cache line every thread also reads at each call.
    struct Tally
    {
        long folded    = 0; ///< spans the rebuild tried to fold back
        long refused   = 0; ///< ... of which compress() declined
        long installed = 0; ///< ... of which zero_any installed directly (no buffer)
    };

    /// One new chunk's elements, and the old chunks they come from.
    struct Span
    {
        IndType base;      ///< first element (global index)
        IndType end;       ///< one past the last element
        int     old_first; ///< first old chunk under the span
        int     old_last;  ///< last old chunk under the span
    };

    /// What the old chunks under a span hold (scan_sources()).
    struct Sources
    {
        /// Every source is still the first constant: nothing to rebuild.
        bool all_cst1 = true;
        /// Every source is a compressed constant (cst1 or cst2): the new chunk,
        /// once filled, holds only constants and can be folded back at once.
        /// Tracked for zero_any and zero_one_value (non-bool); zero has cst1
        /// only (covered by all_cst1).
        bool all_folded = true;
        /// zero_any: some source is cst1.
        bool any_cst1 = false;
        /// zero_any: every cst2 source points at the SAME shared array (same value).
        bool same_cst2 = true;
        /// zero_any: the first cst2 source's values pointer.
        const T* cst2 = nullptr;
    };

    /// @brief Slices of the refill: 16 with releases_sources + fast allocator (fewer
    ///        on a small array), 1 otherwise.
    [[nodiscard]] int slice_count() const noexcept;

    /// @brief One slice: a team region over the new chunks of slice @p slice.
    void run_slice(int slice);

    /// @brief The slice boundary, inside the region: team drain of the old
    ///        size's reuse lists, then the emptied slabs recycled.
    void slice_boundary(pmp::Team& team);

    /// @brief Rebuild new chunk @p new_chunk through @p buf (one per thread).
    void rebuild_one(int new_chunk, std::vector<T>& buf, Tally& tally);

    /// @brief What the old chunks under @p span hold.
    [[nodiscard]] Sources scan_sources(const Span& span) const;

    /// @brief zero_any: all sources are the same second constant -> install it
    ///        on the new chunk without a buffer. @return true if installed.
    bool install_cst2(int new_chunk, const Span& span, const Sources& sources, Tally& tally);

    /// @brief Read the elements of @p span from the old chunks into @p buf.
    void read_span(int new_chunk, const Span& span, std::vector<T>& buf) const;

    /// @brief Check builds: the bulk read of @p span agrees with the
    ///        per-element path (@p copied: elements the bulk read wrote).
    void check_read(int new_chunk, const Span& span, std::size_t copied, const std::vector<T>& buf) const;

    /// @brief Release the old chunks @p span has just finished reading (releases_sources).
    void release_sources(const Span& span);

    /// @brief Fold new chunk @p new_chunk back at once when its sources were
    ///        all compressed constants (zero_any, zero_one_value).
    void fold_back(int new_chunk, const Sources& sources, Tally& tally);

    /// @brief Add one thread's @p tally to the census totals.
    void merge(const Tally& tally) noexcept;

    /// @brief CHUNK_LOG=memory=debug: what the new geometry actually holds.
    void log_census() const;

    Derived&           src_;      ///< the array being re-chunked (OLD geometry)
    Derived&           tmp_;      ///< the fresh array (NEW geometry)
    const int          old_size_; ///< chunk size of src_
    const int          new_size_; ///< chunk size of tmp_
    const unsigned int old_key_;  ///< the allocator key of src_'s storage
    const IndType      total_;    ///< elements of the array
    const int          n_new_;    ///< chunks at the new geometry

    // The traces, read once for the whole refill (rechunk_to() reads them at
    // its start too): rechunk=debug times, rechunk=trace the slices,
    // memory=debug the census.
    const bool trace_on_;     ///< rechunk=debug
    const bool trace_mem_on_; ///< memory=debug
    const bool slice_trace_;  ///< rechunk=trace

    // zero_one_value's cst2 predicate and its compress() both need the array's
    // second-value registry entry; the lookups are hoisted here, once per
    // re-chunk.
    [[maybe_unused]] const int        ind_old2_; ///< src_'s second-value index (the scan)
    [[maybe_unused]] const SecondInfo inf_new2_; ///< tmp_'s second-value record (the fold)

    /// releases_sources: every old chunk lies inside one new chunk (the new size
    /// is a multiple of the old one), so old_done_ is not needed.
    [[maybe_unused]] const bool aligned_coarsening_;
    /// releases_sources, not aligned: how many new chunks have finished with each
    /// old chunk (release_sources()).
    std::vector<std::atomic<int>> old_done_;

    std::atomic<long> n_folded_{0};    ///< census: spans the rebuild tried to fold back
    std::atomic<long> n_installed_{0}; ///< census: ... of which zero_any installed directly
    std::atomic<long> n_refused_{0};   ///< census: ... of which compress() declined: the span
                                       ///< merged constants that DIFFER

    const int n_slices_; ///< slice_count()

    // Slice boundaries run INSIDE the slice's region (see run_slice()): the
    // team drain's scratch lives across them, and so do the trace timers.
    typename ManagerT::TeamDrainScratch team_drain_; ///< the team drain's scratch
    /// The ranks the team holds at a boundary (each recycles its own slabs,
    /// one thread the others').
    std::vector<unsigned char> recycle_held_;
    /// An address buffer per rank, reused. Sized at the first boundary.
    std::vector<std::vector<void*>> recycle_scratch_;
    SliceClock::time_point          slice_t0_{};     ///< written by the master, read after barriers
    SliceClock::time_point          slice_t1_{};     ///< written in a single, read after its barrier
    internal::SliceProbe            probe_;          ///< rechunk=debug: the slice regions' split
    long                            rss_start_ = -1; ///< rechunk=trace: the RSS before the first slice
};

} // namespace shrink_array

#include "chunk/internal/array_chunk_rechunk_refill_inline.h" // Refill's bodies
