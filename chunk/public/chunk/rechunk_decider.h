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

#pragma once

// ============================================================================
// Re-chunk decider (phase 3 of adaptive chunk sizing)
// ============================================================================
//
// Consumes a ChunkSizeReport (advisor: predicted best size + memory gain) and a
// rechunk cost estimate, and decides whether a re-chunk is worthwhile;
// consider() then performs it (rechunk_to) unless Policy::dry_run or
// CHUNK_RECHUNK_DRY_RUN asks it only to log its verdict.
//
// State is per-array, persists across advisor ticks, and is keyed by a stable
// identifier from the caller (the array's address). Hysteresis, three layers:
//   * Damping (best stable for N ticks): a lone tick whose "best" jumps must not
//     trigger. Acting resets the stability counter.
//   * Deadband (min gain): no ping-pong between near-equivalent sizes.
//   * Cooldown (min ticks between actions): a cheap last-resort frequency floor;
//     0 disables.
//
// Thresholds are fields on Policy. The rebuild's cost is weighed in the gain gates
// (rechunk_cost_model.h); hysteresis here is about avoiding OSCILLATION.

#include <cstddef>       // std::size_t
#include <cstdio>        // std::FILE, stdout
#include <string>        // std::string
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include "chunk/chunk_size_advisor.h" // ChunkSizeReport

namespace shrink_array
{

/// @brief Thresholds of RechunkDecider: stability ticks, gain deadbands, cooldown, caps and dry-run switch.
struct RechunkPolicy
{
    int         stability_ticks = 3;        ///< best must hold this many ticks
    std::size_t min_gain_bytes  = 8u << 20; ///< deadband: ignore gains below 8 MB
    double      min_gain_frac   = 0.15;     ///< ALSO require gain >= this fraction of
                                            ///< CURRENT memory (relative deadband):
                                            ///< refuses marginal moves on a large
                                            ///< array, the main thrashing driver.
                                            ///< 0 disables (absolute deadband only).
    int cooldown_ticks      = 5;            ///< min ticks between actions (0 = off)
    int anti_reversal_ticks = 30;           ///< do NOT re-chunk back to the size we
                                            ///< just left within this many ticks
                                            ///< (kills 512<->1024 oscillation). 0=off.
    unsigned max_coarsen_factor = 4;        ///< cap a coarsening step to this factor x
                                            ///< cur_size per rechunk; 1 disables (fining
                                            ///< is never capped). CHUNK_RECHUNK_MAX_COARSEN
                                            ///< overrides it at run time. 4 gave the
                                            ///< lowest end memory. Not higher: the old
                                            ///< size must stay within the advisor's two
                                            ///< finer levels, so the next tick measures
                                            ///< the move at unchanged data and a bad step
                                            ///< can be seen and reverted.
    std::size_t coarsen_min_meta_gain_bytes = 1u << 20;
    ///< A coarsening must gain at least this much METADATA to be counted at all
    ///< (below it the gain is not measurable); whether it is worth a rebuild is
    ///< coarsen_min_meta_rate's question. Small on purpose: a larger floor makes
    ///< the decision depend on the problem scale. 0 disables.
    double coarsen_min_meta_rate = 250.;
    ///< ... and at least this many MB of metadata per second of rebuild, as
    ///< rechunk_cost_ms() estimates it. Gain and rebuild time both grow with the
    ///< array, so the ratio does not depend on its size. The estimate counts the
    ///< rebuild's threads: fewer make it longer, and the last steps are refused.
    ///< The gains fall geometrically with each coarsening, and the step where they
    ///< stop being worth a rebuild (well below this rate) is also where the data
    ///< forecast stops being trustworthy. 0 disables.
    unsigned coarsen_gain_multiple = 3; ///< coarsening must clear this many times
                                        ///< the normal gain gate: the model is exact
                                        ///< when fining (constant chunks split into
                                        ///< constant children) and optimistic when
                                        ///< coarsening (one differing element keeps
                                        ///< the larger chunk materialised).
                                        ///< 1 = same bar as fining.
    double coarsen_trust_ratio = 0.5;   ///< after a coarsening lands, compare the
                                        ///< measured gain with the predicted one;
                                        ///< below this ratio coarsening is barred
                                        ///< for THIS array from then on (fining
                                        ///< stays). 0 disables the feedback.
    bool dry_run = true;                ///< never mutate, only log the verdict
    int  max_backoff_ticks = 8;         ///< an array whose verdict is a settled "no move"
                                        ///< ("already optimal", "gain below deadband") is
                                        ///< looked at again after 2, 4, ... ticks, up to
                                        ///< this many; any other verdict, or a move, brings
                                        ///< it back to every tick. The analysis scans the
                                        ///< array's data, so once the geometry has settled
                                        ///< it was a fixed cost for nothing (shrink_bench
                                        ///< v1 dyn, 64 threads: ~9% of an iteration).
                                        ///< 1 or 0 disables. CHUNK_ADVISOR_BACKOFF_MAX
                                        ///< overrides it at run time.
};

/// @brief Decides, per array, whether a re-chunk is worthwhile from ChunkSizeReport ticks and a RechunkPolicy.
class RechunkDecider
{
  public:
    /// One committed re-chunk, for the end-of-run report.
    struct RechunkEvent
    {
        int         iter;       ///< iteration at which it happened
        std::string name;       ///< array name
        unsigned    old_size;   ///< chunk size before the re-chunk (elements)
        unsigned    new_size;   ///< chunk size after the re-chunk (elements)
        double      mem_before; ///< bytes (model) before the re-chunk
        double      mem_pred;   ///< bytes (model) predicted at the new size
        double      mem_after;  ///< bytes (model) after the re-chunk
        /// Model memory at the OLD size, from the after-tick's analysis (the
        /// advisor evaluates the finer sizes too): old_mem_now - mem_after is
        /// what the move freed at UNCHANGED data. -1 when not available.
        double mem_after_old = -1.;
    };

    /// @brief Store the policy.
    /// @param pol Decision thresholds.
    explicit RechunkDecider(RechunkPolicy pol = {}) : pol_(pol) {}

    /// @brief Table index to act on: rep.best, or a capped coarsening step
    ///        (at most pol_.max_coarsen_factor x cur_size; rep.cur when
    ///        @p banned). Fining is never capped.
    [[nodiscard]] int capped_best_index(const ChunkSizeReport& rep, bool banned = false) const;

    /// @brief Feed one advisor observation for array @p key at logical time
    ///        @p tick. Logs the verdict; true iff it WOULD re-chunk this tick.
    bool observe(const void* key, int tick, const ChunkSizeReport& rep, const char* name);

    /// @brief One-call convenience: analyse @p arr, apply the memory +
    ///        hysteresis policy, and re-chunk it when warranted (no-op in
    ///        static builds). True iff a re-chunk was decided this tick.
    template <typename T, typename Arr>
    bool consider(Arr& arr, int tick, int iter, const char* name);

    /// @brief Print the per-event table: when, old->new, model memory
    ///        before -> after (measured) and the advisor's prediction.
    void print_rechunk_log(std::FILE* out = stdout) const;

  private:
    struct Hist
    {
        static constexpr int kVisited = 8;

        unsigned last_best_size   = 0;
        int      stable_count     = 0;
        int      last_action_tick = -1000000000;
        unsigned last_from_size   = 0; // size the last action re-chunked FROM
        // Set once a coarsening on this array under-delivered against its forecast
        // (see coarsen_trust_ratio): further coarsening is refused, fining stays.
        bool coarsen_banned = false;

        // Multi-step anti-reversal: ring of the last sizes this array OCCUPIED, with
        // the tick it left them. Blocks re-chunking BACK to any of them within
        // anti_reversal_ticks (last_from_size alone misses multi-step oscillations).
        // kVisited covers the geometry ladder (16..1024 = 7 rungs) with headroom.
        unsigned visited_size[kVisited] = {0, 0, 0, 0, 0, 0, 0, 0};
        int      visited_tick[kVisited] = {-1000000000, -1000000000, -1000000000, -1000000000,
                                           -1000000000, -1000000000, -1000000000, -1000000000};
        int      visited_head           = 0; // next write slot (ring)
        // CHUNK_RECHUNK_CHURN: the previous tick's counters (the baseline is
        // reset when the key changes, i.e. after a re-chunk of this array).
        unsigned churn_key     = 0;
        long     churn_creates = -1;
        long     churn_phases  = 0;
        long     churn_mat     = 0;
        // Back-off (max_backoff_ticks): the current gap, in ticks, between two
        // analyses of this array, and the first tick it is analysed again.
        int backoff         = 1;
        int next_look_tick  = -1000000000;
    };

    // `tick` increments once per advisor call (NOT the raw iteration index), so
    // cooldown_ticks compares directly against tick deltas.
    RechunkPolicy                                pol_;
    std::unordered_map<const void*, Hist>        hist_;
    std::vector<RechunkEvent>                    log_;
    std::unordered_map<const void*, std::size_t> pending_after_; // array -> log_ idx awaiting "after"
};

} // namespace shrink_array

#include "chunk/internal/rechunk_decider_inline.h" // the method bodies
