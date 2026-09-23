#pragma once

// ============================================================================
// Re-chunk decider (phase 3, morceau A — DRY-RUN, no mutation)
// ============================================================================
//
// Consumes a ChunkSizeReport (advisor: predicted best size + memory gain) and a
// rechunk cost estimate, and decides whether a re-chunk would be worthwhile —
// WITHOUT performing it. It only logs its verdict. This lets us watch the
// decisions (cadence, stability, whether hysteresis behaves) before the real
// re-chunk operation exists.
//
// State is per-array and persists across advisor ticks, keyed by a stable
// identifier the caller provides (the array's address is a good key: stable for
// the array's lifetime). Hysteresis, three layers, each guarding a different
// failure:
//   * Damping (best stable for N ticks): filters sampling noise — a lone tick
//     whose "best" jumps must not trigger. Acting resets the stability counter,
//     so a further move must re-stabilise.
//   * Deadband (min gain): ignores marginal gains between near-equivalent sizes
//     (the advisor's valleys can be shallow; don't ping-pong for 2 MB).
//   * Cooldown (min ticks between actions): a hard frequency floor, redundant
//     with the damping reset in most cases but a cheap last-resort guard; set to
//     0 to disable.
//
// All thresholds are fields on Policy — tune from observation. Because measured
// re-chunk cost is sub-millisecond, hysteresis here is about avoiding
// OSCILLATION, not about protecting from an expensive operation.

#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "chunk/chunk_size_advisor.h"

namespace shrink_array
{

struct RechunkPolicy
{
    int         stability_ticks = 3;        // best must hold this many ticks
    std::size_t min_gain_bytes  = 8u << 20; // deadband: ignore gains below 8 MB
    double      min_gain_frac   = 0.15;     // ALSO require gain >= this fraction of
                                            // CURRENT memory (relative deadband).
                                            // Kills marginal moves on a large array
                                            // (48 MB on 3 GB = 1.5% -> refused),
                                            // which was the main thrashing driver.
                                            // 0 disables (absolute deadband only).
    int cooldown_ticks      = 5;            // min ticks between actions (0 = off)
    int anti_reversal_ticks = 30;           // do NOT re-chunk back to the size we
                                            // just left within this many ticks
                                            // (kills 512<->1024 oscillation). 0=off.
    unsigned max_coarsen_factor = 2;        // cap a coarsening step to this factor x
                                            // cur_size per rechunk. A big jump (x16)
                                            // leans on the least reliable model
                                            // extrapolation and can mispredict
                                            // (64->1024 cost +40 MB); stepping via the
                                            // intermediate size re-checks with real
                                            // chunk data. 1 disables (fining never
                                            // capped).
                                            // Set to 2 (one doubling per decision):
                                            // x8 still lands three rungs away in a
                                            // single bet -- 512->4096 was taken on a
                                            // 35.8 MB forecast and the run gained
                                            // nothing. One rung at a time is
                                            // self-correcting: a bad step is seen at
                                            // the next tick and can be reverted, and
                                            // a good one is repeated, reaching the
                                            // same size in three checked moves.
    std::size_t coarsen_min_meta_gain_bytes = 24u << 20;
                                            // A coarsening must gain at least this
                                            // much METADATA to be worth doing at
                                            // all. The metadata term is exact, so
                                            // it needs no margin for trust -- this
                                            // is about cost: a coarsening is a full
                                            // rebuild, ~0.6 s per array on the
                                            // 64-thread bench, and the gains fall
                                            // geometrically (tab1, 6.1 GB: 211 MB,
                                            // 105, 52, 25, then 11). The last step
                                            // cleared the 8 MB absolute deadband on
                                            // an exact 10.9 MB forecast and returned
                                            // 3.7 -- the forecast stops being
                                            // trustworthy at the same step the gain
                                            // stops being worth having. 0 disables
                                            // (absolute deadband only).
    unsigned coarsen_gain_multiple = 3;     // coarsening must clear this many times
                                            // the normal gain gate. The model is
                                            // reliable when fining (constant chunks
                                            // split into constant children) and
                                            // optimistic when coarsening (one
                                            // differing element keeps the whole,
                                            // now much larger, chunk materialised),
                                            // so the two directions should not face
                                            // the same bar. 1 = same bar as fining.
    double coarsen_trust_ratio = 0.5;       // after a coarsening lands, compare the
                                            // measured gain with the predicted one.
                                            // Below this ratio the model was wrong
                                            // about THIS array, and coarsening is
                                            // barred for it from then on. Fining
                                            // stays available. 0 disables the
                                            // feedback.
    bool dry_run = true;                    // phase-3A: never mutate, only log
};

class RechunkDecider
{
  public:
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

    // One committed re-chunk, for the end-of-run report.
    struct RechunkEvent
    {
        int         iter; // iteration at which it happened
        std::string name; // array name
        unsigned    old_size, new_size;
        double      mem_before, mem_pred, mem_after; // bytes (model)
    };

    /// @brief Print the per-event table: when, old->new, model memory
    ///        before -> after (measured) and the advisor's prediction.
    void print_rechunk_log(std::FILE* out = stdout) const;

  private:
    struct Hist
    {
        unsigned last_best_size   = 0;
        int      stable_count     = 0;
        int      last_action_tick = -1000000000;
        unsigned last_from_size   = 0; // size the last action re-chunked FROM
        // Multi-step anti-reversal: ring of the last few sizes this array
        // OCCUPIED (cur_size at each committed action) with the tick it left
        // them. Blocks re-chunking BACK to any size occupied within
        // anti_reversal_ticks -- single last_from_size misses the multi-step
        // 512->..->16->256 oscillation. kVisited covers the geometry ladder
        // (16..1024 = 7 rungs) with headroom.
        // Set once a coarsening move on this array under-delivered against its
        // own forecast (see coarsen_trust_ratio). The model has then been shown
        // wrong on this data, so further coarsening is refused; fining, whose
        // model is exact, stays available.
        bool                 coarsen_banned         = false;
        static constexpr int kVisited               = 8;
        unsigned             visited_size[kVisited] = {0, 0, 0, 0, 0, 0, 0, 0};
        int                  visited_tick[kVisited] = {-1000000000, -1000000000, -1000000000, -1000000000,
                                                       -1000000000, -1000000000, -1000000000, -1000000000};
        int                  visited_head           = 0; // next write slot (ring)
    };

    // `tick` is a monotone counter that increments once per advisor call (the
    // caller passes such a small integer, NOT the raw iteration index), so
    // cooldown_ticks compares directly against tick deltas.
    RechunkPolicy                                pol_;
    std::unordered_map<const void*, Hist>        hist_;
    std::vector<RechunkEvent>                    log_;
    std::unordered_map<const void*, std::size_t> pending_after_; // array -> log_ idx awaiting "after"
};

} // namespace shrink_array

// Method bodies. Included here (after the class) so every translation unit
// using RechunkDecider sees the definitions and instantiates the T/Arr it needs.
#include "chunk/internal/rechunk_decider_inline.h"
#include "chunk/internal/chunk_size_policy.h" // shrink_array::cst_buffer_size
