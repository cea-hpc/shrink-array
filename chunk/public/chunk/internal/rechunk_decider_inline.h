#pragma once

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

// Out-of-line bodies of RechunkDecider (declared in rechunk_decider.h, which
// includes this file after the class). Only the one-liners stay in the class.

#include <algorithm> // std::max
#include <cstddef>   // std::size_t
#include <cstdio>    // std::fprintf, std::fflush, std::snprintf
#include <cstdlib>   // std::getenv -- CHUNK_FORCE_RECHUNK debug hooks
#include <cstring>   // std::strcmp (CHUNK_ADVISOR_TABLE)
#include <string>    // std::string
#include <optional>  // std::optional

#include "chunk/rechunk_decider.h"             // the declarations these bodies belong to
#include "chunk/chunk_size_advisor.h"          // ChunkSizeAdvisor, ChunkSizeReport
#include "chunk/config.h"                      // build_config::chunk_dynamic
#include "chunk/internal/env_parse.h"          // env_number / env_flag (checked, not atoi)
#include "chunk/PMP_CHUNK.h"                   // CHUNK_PMP_MAX_THREADS (the rebuild's team, for its cost)
#include "chunk/internal/churn_counters.h"     // CHUNK_RECHUNK_CHURN
#include "chunk/internal/exact_rss.h"          // exact_rss_kib ([RECHUNK-MEM])
#include "chunk/internal/values_arena_bytes.h" // the values arena, if the build has one ([RECHUNK-MEM])

namespace shrink_array
{

// A size in MB from an environment variable, in bytes; 0 or less is 0.
inline std::size_t mb_to_bytes(double mb) noexcept
{
    return mb > 0. ? static_cast<std::size_t>(mb * 1024. * 1024.) : 0u;
}

// Cap a coarsening move to at most pol_.max_coarsen_factor x cur_size per
// rechunk. A large coarsening (e.g. x16) leans on the least reliable part of
// the model -- mat_at extrapolated over many merge levels -- so one wrong
// jump can COST memory (observed 64->1024: predicted -91 MB, actual +40 MB).
// Restricting the step lets the advisor re-evaluate at the intermediate size
// with real chunks to scan (not an extrapolation) and step further only if
// the gain is real. Fining (best < cur) is reliable and never capped. Returns
// the table index to act on -- rep.best when no cap applies.
/// @brief CHUNK_RECHUNK_DRY_RUN=1 (read once): analyse and log, never apply a
///        move, whatever the policy says -- the way to measure the analyses'
///        cost on the same binary (a huge CHUNK_ADVISOR_MIN_GAIN_MB no longer
///        blocks moves since the gain-per-cost gate).
inline bool rechunk_env_dry_run() noexcept
{
    static const bool v = []
    {
        const char* e = std::getenv("CHUNK_RECHUNK_DRY_RUN");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return v;
}

inline int RechunkDecider::capped_best_index(const ChunkSizeReport& rep, bool banned) const
{
    // CHUNK_RECHUNK_MAX_COARSEN=<n> overrides the policy's factor (1 = no
    // cap), read once. The cap of 2 came from two bad big jumps measured
    // before Sept '26's fixes (lagging VmRSS, Chunk2A's leaked metadata rows,
    // folds through a buffer); this measures whether larger steps are safe
    // now, without a rebuild.
    // Unset, not a number (warned) or 0: keep the policy.
    static const unsigned env_factor = shrink_array::env_optional<unsigned>("CHUNK_RECHUNK_MAX_COARSEN").value_or(0u);
    const unsigned        max_factor = env_factor != 0 ? env_factor : pol_.max_coarsen_factor;

    if (rep.best < 0 || rep.cur < 0 || max_factor <= 1)
    {
        return rep.best;
    }
    const unsigned cur_size = rep.table[static_cast<std::size_t>(rep.cur)].size;
    const unsigned best_sz  = rep.table[static_cast<std::size_t>(rep.best)].size;
    if (best_sz <= cur_size) // fining or same size: reliable, no cap
    {
        return rep.best;
    }
    if (banned) // this array already paid for one bad coarsening
    {
        return rep.cur;
    }
    const unsigned cap_sz = cur_size * max_factor;
    if (best_sz <= cap_sz)
    {
        return rep.best; // within the cap
    }
    // Jump exceeds the cap: pick the min-model-memory size in (cur_size,
    // cap_sz]; if none beats cur, fall back to cur (no move this tick).
    int    pick     = rep.cur;
    double best_mem = rep.table[static_cast<std::size_t>(rep.cur)].mem_bytes;
    for (std::size_t i = 0; i < rep.table.size(); ++i)
    {
        const auto& entry = rep.table[i];
        if (entry.size > cur_size && entry.size <= cap_sz && entry.mem_bytes < best_mem)
        {
            best_mem = entry.mem_bytes;
            pick     = static_cast<int>(i);
        }
    }
    return pick;
}

// Feed one advisor observation for array `key` at logical time `tick`.
// Logs the verdict; returns true iff it WOULD re-chunk (dry-run) this tick.

inline bool RechunkDecider::observe(const void* key, int tick, const ChunkSizeReport& rep, const char* name)
{
    if (rep.best < 0 || rep.cur < 0)
    {
        return false;
    }

    const int      eff_best   = capped_best_index(rep, hist_[key].coarsen_banned);
    const auto&    cur_e      = rep.table[static_cast<std::size_t>(rep.cur)];
    const auto&    best_e     = rep.table[static_cast<std::size_t>(eff_best)];
    const unsigned cur_size   = cur_e.size;
    const unsigned best_sz    = best_e.size;
    const double   gain_b     = cur_e.mem_bytes - best_e.mem_bytes;
    const auto     gain_bytes = static_cast<std::size_t>(gain_b > 0. ? gain_b : 0.);
    Hist&          hist       = hist_[key];

    // Damping bookkeeping: is `best` the same as last tick?
    if (best_sz == hist.last_best_size)
    {
        ++hist.stable_count;
    }
    else
    {
        hist.last_best_size = best_sz;
        hist.stable_count   = 1;
    }

    const char* reason = nullptr;
    bool        act    = false;

    // The model has two terms (chunk_size_advisor.h):
    //   mem(S) = n_chunks(S) * meta_bytes  +  U_mat(S) * S * sizeof(T)
    // and they do not deserve the same gate. The metadata term is EXACT --
    // the chunk count at a size is arithmetic, not a forecast -- and on a
    // coarsening it is always a gain. The data term is the forecast, and
    // on a coarsening it is the optimistic one (one differing element keeps
    // a chunk that is now several times larger materialised).
    //
    // Applying the coarsening margin and the relative deadband to the SUM
    // made a certain gain pay for an uncertain one: measured on
    // shrink_bench (chunk2 dyn, 64 threads, 10^9-element arrays), the
    // 64 -> 128 move carried a ~200 MB metadata gain identical across
    // double, short and complex arrays (so: not data), and was refused
    // "gain below deadband" at every one of 16 ticks, because 45% of a
    // multi-GB array's current memory is out of reach for any structural
    // gain. Zero re-chunks in 2000 iterations; the dyn build stayed at its
    // deliberately sub-optimal start size and paid ~4-5 GB over static 256.
    const bool coarsening = best_sz > cur_size;
    // What the cost model expects the rebuild to take: gates a coarsening
    // (coarsen_min_meta_rate), and is logged with the chunk and
    // materialised-chunk counts, to be set against CHUNK_RECHUNK_TRACE's
    // measured total when the constants are refitted.
    const double est_cost_ms =
        rechunk_cost_ms(cur_e.n_chunks, cur_size, cur_e.mat, rep.elem_bytes, CHUNK_PMP_MAX_THREADS);
    const double gain_meta = static_cast<double>(cur_e.n_chunks - best_e.n_chunks) * rep.meta_bytes;
    const double gain_data = gain_b - gain_meta;
    // MB of metadata per second of estimated rebuild (0 when the estimate is).
    const double meta_rate = est_cost_ms > 0. ? (gain_meta / (1024. * 1024.)) / (est_cost_ms / 1000.) : 0.;

    // Gate for the forecast (data) term: the LARGER of the absolute
    // deadband and a fraction of the current memory. On a large array a
    // fixed 8 MB is trivially cleared by noise; the relative gate is what
    // stops chasing sub-percent "gains" (the observed thrashing: ~48 MB
    // moves on multi-GB arrays). Coarsening faces a higher bar than fining,
    // for the model-optimism reason above.
    // CHUNK_ADVISOR_MIN_GAIN_MB overrides the absolute deadband, in MB; 0
    // disables it. Read once, like the floor below. Without it the feature
    // test of bench_and_test.py could not fire a move: it shrinks its arrays
    // with --divisor, the metadata gain of 64 -> 128 is then under a MB, and
    // the 8 MB policy value still gated the coarsening branch after
    // CHUNK_COARSEN_MIN_META_MB=0 had cleared the floor -- "gain below
    // deadband" at every tick, on every chunk-family variant.
    // Unset, or not a number: nothing, and the policy value is kept.
    static const std::optional<std::size_t> env_min_gain =
        shrink_array::env_optional<double>("CHUNK_ADVISOR_MIN_GAIN_MB").transform(mb_to_bytes);
    const std::size_t min_gain = env_min_gain.value_or(pol_.min_gain_bytes);

    const auto  rel_gate  = static_cast<std::size_t>(pol_.min_gain_frac * static_cast<double>(cur_e.mem_bytes));
    std::size_t data_gate = std::max(min_gain, rel_gate);
    if (coarsening && pol_.coarsen_gain_multiple > 1)
    {
        data_gate *= pol_.coarsen_gain_multiple;
    }

    // The move clears the deadband when EITHER term earns it on its own:
    //   - the exact metadata term alone exceeds the absolute deadband, with
    //     the forecast data term not eating it back (gain_b >= deadband);
    //   - or the forecast data term clears its own, larger, gate.
    // Fining keeps the old rule unchanged: there gain_meta <= 0 and the data
    // forecast is exact, so the sum is the right thing to gate.
    bool clears_gate;
    if (coarsening)
    {
        // The metadata branch carries its own gates: a floor (is the gain
        // measurable?) and a rate (is it worth a rebuild?). Both have to hold,
        // because an exact gain too small to pay for the work it costs is
        // still a bad move -- and it is the step where the data forecast
        // starts missing badly, so refusing it also keeps the decision inside
        // the range where the model is sound. Both are relative to nothing
        // but the move itself: an absolute floor (24 MB until Sept '26) let
        // the same arrays re-chunk at full size and never at a tenth of it.
        // CHUNK_COARSEN_MIN_META_MB overrides the floor, in MB, and
        // CHUNK_COARSEN_MIN_META_RATE the rate, in MB per second of estimated
        // rebuild; 0 disables either. Read once, here, which runs once per
        // array per advisor tick -- the same shape as the library's other
        // run-time knobs. The feature test needs them for the opposite
        // reason: it shrinks its arrays on purpose and still wants the move
        // to fire.
        static const std::optional<std::size_t> env_meta_floor =
            shrink_array::env_optional<double>("CHUNK_COARSEN_MIN_META_MB").transform(mb_to_bytes);
        static const std::optional<double> env_meta_rate =
            shrink_array::env_optional<double>("CHUNK_COARSEN_MIN_META_RATE");
        const std::size_t meta_floor = env_meta_floor.value_or(pol_.coarsen_min_meta_gain_bytes);
        const double      min_rate   = env_meta_rate.value_or(pol_.coarsen_min_meta_rate);
        // A zero-cost estimate (empty array) passes.
        const bool rate_ok = min_rate <= 0. || est_cost_ms <= 0. || meta_rate >= min_rate;
        clears_gate        = (gain_meta >= static_cast<double>(meta_floor) && rate_ok && gain_bytes >= meta_floor) ||
                             gain_bytes >= data_gate;
    }
    else
    {
        clears_gate = gain_bytes >= data_gate;
    }
    // Kept for the log line: what the old single gate would have demanded.
    const std::size_t gain_gate = data_gate;

    // Multi-step anti-reversal: would this move return us to a size we
    // occupied within the anti_reversal window? Scans the visited ring
    // (single last_from_size misses 512->..->16->256).
    bool reverting = false;
    if (pol_.anti_reversal_ticks > 0)
    {
        for (int _i = 0; _i < Hist::kVisited; ++_i)
        {
            if (hist.visited_size[_i] == best_sz && tick - hist.visited_tick[_i] < pol_.anti_reversal_ticks)
            {
                reverting = true;
                break;
            }
        }
    }

    if (best_sz == cur_size)
    {
        reason = "already optimal";
    }
    else if (pol_.cooldown_ticks > 0 && tick - hist.last_action_tick < pol_.cooldown_ticks)
    {
        reason = "cooldown";
    }
    else if (reverting)
    {
        reason = "anti-reversal"; // would return to a recently-occupied size -> oscillation
    }
    else if (!clears_gate)
    {
        reason = "gain below deadband";
    }
    else if (hist.stable_count < pol_.stability_ticks)
    {
        reason = "best not yet stable";
    }
    else
    {
        act    = true;
        reason = "RE-CHUNK";
    }

    if (act)
    {
        std::fprintf(stderr,
                     "[RECHUNK-DECIDE] %s: %s S=%u -> S'=%u "
                     "(gain %.1f MB = meta %.1f + data %.1f, stable %d ticks, est. cost %.0f ms, %.0f MB/s, chunks "
                     "%lld mat %lld)%s\n",
                     name, (pol_.dry_run || rechunk_env_dry_run()) ? "[dry-run] would re-chunk" : "re-chunk", cur_size,
                     best_sz, gain_b / (1024. * 1024.), gain_meta / (1024. * 1024.), gain_data / (1024. * 1024.),
                     hist.stable_count, est_cost_ms, meta_rate, static_cast<long long>(cur_e.n_chunks),
                     static_cast<long long>(cur_e.mat), (pol_.dry_run || rechunk_env_dry_run()) ? "" : " [APPLIED]");
        std::fflush(stderr);

        hist.last_action_tick = tick;
        hist.stable_count     = 0;        // must re-stabilise before next move
        hist.last_from_size   = cur_size; // remember what we left (single-step)
        // Record the size we just left in the visited ring (multi-step
        // anti-reversal): a later best == this size within the window is
        // refused, killing the 512->..->16->256 oscillation.
        hist.visited_size[hist.visited_head] = cur_size;
        hist.visited_tick[hist.visited_head] = tick;
        hist.visited_head                    = (hist.visited_head + 1) % Hist::kVisited;
        hist.last_best_size                  = best_sz;
        // In dry-run we do NOT change the array; a real phase would re-chunk
        // here and update the array's current geometry.
    }
    else
    {
        // Explain non-actions (why no re-chunk): "cost too high", "gain below
        // deadband", "cooldown", "best not yet stable", "already optimal".
        std::fprintf(stderr,
                     "[RECHUNK-DECIDE] %s: wait (%s) "
                     "S=%u best=%u gain %.1f MB "
                     "(meta %.1f + data %.1f; cur %.1f MB, data gate %.1f MB, est. cost %.0f ms, %.0f MB/s)\n",
                     name, reason, cur_size, best_sz, gain_b / (1024. * 1024.), gain_meta / (1024. * 1024.),
                     gain_data / (1024. * 1024.), cur_e.mem_bytes / (1024. * 1024.),
                     static_cast<double>(gain_gate) / (1024. * 1024.), est_cost_ms, meta_rate);
        std::fflush(stderr);
    }
    return act;
}

// One-call convenience: analyse `arr`, apply the memory + hysteresis policy,
// and actually re-chunk it when warranted. The advisor run, the decision and
// the rechunk_to all happen here, so a caller needs nothing else. No-op in
// static builds. Returns true iff a re-chunk was decided this tick.
/// @brief CHUNK_RECHUNK_MEMTRACE: the last tick printed -- one variable for
///        every consider<T, Arr> instantiation (a function-local static in the
///        template would print once per element type).
inline int& rechunk_memtrace_last_tick() noexcept
{
    static int t = -1000000000;
    return t;
}

template <typename T, typename Arr>
bool RechunkDecider::consider(Arr& arr, int tick, int iter, const char* name)
{
    if constexpr (!build_config::chunk_dynamic)
    {
        (void)arr;
        (void)tick;
        (void)iter;
        (void)name;
        return false;
    }
    else
    {
        // DEBUG: CHUNK_FORCE_RECHUNK=<size> forces every array to rechunk_to
        // <size> once, bypassing the advisor + hysteresis policy, so the rechunk
        // path can be exercised (e.g. under a sanitizer) even when --divisor makes
        // it un-beneficial and the advisor never fires. Self-limiting: once
        // arr.chunk_size()==<size> the guard is false, so it never repeats.
        // 0/unset disables. Read once.
        static const unsigned force_sz = shrink_array::env_number<unsigned>("CHUNK_FORCE_RECHUNK", 0u);
        // CHUNK_FORCE_RECHUNK_AT=<iter> delays the forced rechunk until iter >=
        // this (default 0 = at the first tick). Forcing at iter 0 rechunks fresh,
        // near-empty arrays; set e.g. 1000 to rechunk POPULATED arrays with
        // pending deferred ops -- the state under which the real bug appears.
        static const int force_at = shrink_array::env_number<int>("CHUNK_FORCE_RECHUNK_AT", 0);
        // CHUNK_FORCE_RECHUNK_TOGGLE=1 rechunks at EVERY tick (iter>=force_at),
        // alternating force_sz <-> CHUNK_FORCE_RECHUNK_ALT (default 64). Combined
        // with a short CHUNK_ADVISOR_PERIOD and a large --divisor, this hammers
        // the rechunk path many times in a short, ASan-cheap run to reproduce a
        // rechunk-triggered corruption. Default (0) keeps the one-shot behaviour.
        static const unsigned force_alt    = shrink_array::env_number<unsigned>("CHUNK_FORCE_RECHUNK_ALT", 64u);
        static const bool     force_toggle = shrink_array::env_number<int>("CHUNK_FORCE_RECHUNK_TOGGLE", 0) != 0;
        if (force_sz != 0u && !(pol_.dry_run || rechunk_env_dry_run()) && iter >= force_at)
        {
            const unsigned cur_sz = arr.chunk_size();
            const unsigned tgt    = force_toggle ? (cur_sz == force_sz ? force_alt : force_sz) : force_sz;
            if (cur_sz != tgt)
            {
                std::fprintf(stderr, "[FORCE_RECHUNK] %s: %u -> %u\n", name, cur_sz, tgt);
                arr.rechunk_to(tgt);
                // Log it so print_rechunk_log() ("Re-chunks performed") shows the
                // forced move. Mem figures are 0 (advisor bypassed, not measured).
                log_.emplace_back(RechunkEvent{iter, std::string(name), cur_sz, tgt, 0.0, 0.0, 0.0});
                return true;
            }
            return false;
        }

        // CHUNK_RECHUNK_MEMTRACE=1: once per tick, the process memory split
        // into the chunk value arena (resident pages, mincore) and the rest.
        // MiB, like [RECHUNK-RSS] (rss_mib_now), so the two traces add up.
        {
            static const bool memtrace = []
            {
                const char* e = std::getenv("CHUNK_RECHUNK_MEMTRACE");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            // Shared by every consider<T, Arr> instantiation: once per tick,
            // not once per element type.
            int& last_mem_tick = rechunk_memtrace_last_tick();
            if (memtrace && tick != last_mem_tick)
            {
                last_mem_tick = tick;
                // Exact (smaps_rollup): statm lags by gigabytes at 64 threads.
                const long rss_kib = shrink_array::exact_rss_kib();
                const long rss_mb  = rss_kib >= 0 ? rss_kib / 1024 : -1;
                if constexpr (build_config::use_std_alloc)
                {
                    // std-alloc: the values are plain mallocs, no arena to split off.
                    std::fprintf(stderr, "[RECHUNK-MEM] tick %d iter %d: rss %ld MB (std-alloc: no values arena)\n",
                                 tick, iter, rss_mb);
                }
                else
                {
                    // values_arena_bytes.h: no fast_alloc name here, so this
                    // branch compiles in a std-alloc build too.
                    const internal::ArenaBytes va     = internal::values_arena_bytes().value_or(internal::ArenaBytes{});
                    const long                 res_mb = static_cast<long>(static_cast<double>(va.resident) / 1048576.);
                    std::fprintf(stderr,
                                 "[RECHUNK-MEM] tick %d iter %d: rss %ld MB = values arena %ld MB + other %ld MB"
                                 "   (arena mapped %ld MB, slabs handed out %ld MB)\n",
                                 tick, iter, rss_mb, res_mb, rss_mb - res_mb,
                                 static_cast<long>(static_cast<double>(va.mapped) / 1048576.),
                                 static_cast<long>(static_cast<double>(va.live) / 1048576.));
                }
                std::fflush(stderr);
            }
        }

        const ChunkSizeReport& rep = ChunkSizeAdvisor<T>::analyze(arr);

        // CHUNK_RECHUNK_CHURN=1 (every array) or =<name>: the per-phase
        // transient the snapshot above cannot see (churn_counters.h).
        {
            static const char* const churn_env = std::getenv("CHUNK_RECHUNK_CHURN");
            if (churn_env != nullptr && churn_env[0] != '\0' && churn_env[0] != '0' &&
                (std::strcmp(churn_env, "1") == 0 || std::strcmp(churn_env, name) == 0) && rep.cur >= 0)
            {
                Hist&          h       = hist_[&arr];
                const unsigned key     = arr.alloc_key();
                const long     creates = churn::Materialisations<T>::total(key);
                const long     phases  = churn::phase_ends();
                const long     mat_now = static_cast<long>(rep.table[static_cast<std::size_t>(rep.cur)].mat);
                const unsigned S       = rep.current_size;
                if (key != 0 && h.churn_creates >= 0 && key == h.churn_key)
                {
                    const long   d_creates = creates - h.churn_creates;
                    const long   d_phases  = (phases - h.churn_phases) > 0 ? (phases - h.churn_phases) : 1;
                    const long   growth    = mat_now - h.churn_mat;
                    const double transient =
                        static_cast<double>(d_creates - (growth > 0 ? growth : 0)) / static_cast<double>(d_phases);
                    const double mb_S   = transient * S * sizeof(T) / 1e6;
                    const double mb_64s = transient * 64. * sizeof(T) / 1e6; // same chunk count at 64
                    std::fprintf(stderr,
                                 "[RECHUNK-CHURN] %s S=%u: %ld materialisations over %ld phases since the last "
                                 "tick (%.0f/phase), resident materialised %ld -> %ld: transient ~%.0f "
                                 "chunks/phase = %.1f MB at S=%u; at S=64: %.1f MB if scattered, %.1f MB if "
                                 "grouped\n",
                                 name, S, d_creates, d_phases, static_cast<double>(d_creates) / d_phases, h.churn_mat,
                                 mat_now, transient, mb_S, S, mb_64s, mb_S);
                }
                else
                {
                    std::fprintf(stderr, "[RECHUNK-CHURN] %s S=%u: baseline (key %u)\n", name, S, key);
                }
                std::fflush(stderr);
                h.churn_key     = key;
                h.churn_creates = creates;
                h.churn_phases  = phases;
                h.churn_mat     = mat_now;
            }
        }

        // CHUNK_ADVISOR_TABLE=1 (every array) or =<name> (that one): the whole
        // table this tick's decision is taken from -- the finer rows too.
        {
            static const char* const table_env = std::getenv("CHUNK_ADVISOR_TABLE");
            if (table_env != nullptr && table_env[0] != '\0' && table_env[0] != '0' &&
                (std::strcmp(table_env, "1") == 0 || std::strcmp(table_env, name) == 0))
            {
                char title[96];
                std::snprintf(title, sizeof title, "%s tick %d iter %d", name, tick, iter);
                ChunkSizeAdvisor<T>::print(rep, title);
                std::fflush(stdout);
            }
        }

        // If a PRIOR re-chunk of this array is awaiting its "after" figure,
        // fill it now: this tick's analysis sees the new geometry once it has
        // settled (re-compressed over the ticks since). Measuring right after
        // rechunk_to would instead catch the freshly-materialised, not-yet-
        // compressed chunks and report a memory INCREASE.
        if (auto it = pending_after_.find(&arr); it != pending_after_.end())
        {
            if (rep.cur >= 0)
            {
                RechunkEvent& ev = log_[it->second];
                ev.mem_after     = rep.table[static_cast<std::size_t>(rep.cur)].mem_bytes;
                // The OLD size evaluated on the SAME data (a finer row once
                // coarsened, a coarser one once refined): the drift-free
                // reference. mem_before is one advisor period older, and the
                // data moved in between (tab_complex: +290 MB between two
                // moves), which alone turned correct forecasts into "freed
                // -6.5 MB" and banned coarsening on it.
                for (const auto& row : rep.table)
                {
                    if (row.size == ev.old_size)
                    {
                        ev.mem_after_old = row.mem_bytes;
                    }
                }

                // Outcome feedback. Everything needed was already recorded
                // and never looked at: compare what the move actually
                // freed with what it promised. A coarsening that delivers
                // materially less than forecast means the model is wrong
                // about THIS array's data, so stop coarsening it -- no
                // amount of re-deciding on the same model will do better.
                // Fining is untouched: its model is exact.
                if (pol_.coarsen_trust_ratio > 0. && ev.new_size > ev.old_size)
                {
                    const double predicted = ev.mem_before - ev.mem_pred;
                    // Drift-free when the table had the old size, else the
                    // before/after difference, as before.
                    const double measured =
                        ev.mem_after_old >= 0. ? ev.mem_after_old - ev.mem_after : ev.mem_before - ev.mem_after;
                    if (predicted > 0. && measured < pol_.coarsen_trust_ratio * predicted)
                    {
                        hist_[&arr].coarsen_banned = true;
                        std::fprintf(stdout,
                                     "[RECHUNK-LEARN] %s: coarsening %u -> %u freed %.1f MB "
                                     "of the %.1f MB predicted -- coarsening disabled for "
                                     "this array\n",
                                     ev.name.c_str(), ev.old_size, ev.new_size, measured / 1e6, predicted / 1e6);
                    }
                }
            }
            pending_after_.erase(it);
        }

        const bool act = observe(&arr, tick, rep, name);
        if (act && !(pol_.dry_run || rechunk_env_dry_run()) && rep.best >= 0)
        {
            const int      eff_best = capped_best_index(rep, hist_[&arr].coarsen_banned);
            const auto&    cur_e    = rep.table[static_cast<std::size_t>(rep.cur)];
            const auto&    best_e   = rep.table[static_cast<std::size_t>(eff_best)];
            const double   mem_bef  = cur_e.mem_bytes;  // model mem at OLD geometry
            const double   mem_prd  = best_e.mem_bytes; // advisor's PREDICTION
            const unsigned old_sz   = cur_e.size;
            const unsigned new_sz   = best_e.size;

            arr.rechunk_to(new_sz);

            // "after" is measured at the NEXT tick on this array (settled +
            // re-compressed), not now. Seed it with the prediction so a run
            // that ends before the next tick still shows a sane figure.
            log_.emplace_back(RechunkEvent{iter, std::string(name), old_sz, new_sz, mem_bef, mem_prd, mem_prd});
            pending_after_[&arr] = log_.size() - 1;
        }
        return act;
    }
}

// Print the per-event table: when, old->new, and the model memory
// before -> after (measured), with the advisor's prediction for comparison.
inline void RechunkDecider::print_rechunk_log(std::FILE* out) const
{
    std::fprintf(out, "Re-chunks performed: %zu\n", log_.size());
    for (const RechunkEvent& e : log_)
    {
        const double d_meas = e.mem_before - e.mem_after;
        if (e.mem_after_old >= 0.)
        {
            std::fprintf(out,
                         "  iter %-6d %-12s S %5u -> %-5u   mem %7.1f -> %7.1f MB "
                         "(freed %6.1f MB, %6.1f at unchanged data; advisor predicted %6.1f)\n",
                         e.iter, e.name.c_str(), e.old_size, e.new_size, e.mem_before / 1e6, e.mem_after / 1e6,
                         d_meas / 1e6, (e.mem_after_old - e.mem_after) / 1e6, (e.mem_before - e.mem_pred) / 1e6);
        }
        else
        {
            std::fprintf(out,
                         "  iter %-6d %-12s S %5u -> %-5u   mem %7.1f -> %7.1f MB "
                         "(freed %6.1f MB; advisor predicted %6.1f)\n",
                         e.iter, e.name.c_str(), e.old_size, e.new_size, e.mem_before / 1e6, e.mem_after / 1e6,
                         d_meas / 1e6, (e.mem_before - e.mem_pred) / 1e6);
        }
    }
}

} // namespace shrink_array
