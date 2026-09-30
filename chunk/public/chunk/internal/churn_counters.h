// Copyright 2026 CEA (www.cea.fr)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

/**
 * @file churn_counters.h
 * @brief Measurement for the re-chunk model's "churn" term: how many chunks
 *        each phase materialises and releases again, per array.
 *
 * The advisor looks at an array between phases, after the compression pass:
 * it sees the data at rest. A write into a constant chunk materialises the
 * whole chunk until the phase-end compression releases it, and the pages it
 * took stay with the process: at S=512 that transient costs 8x what it costs
 * at S=64. These counters give the decider the per-phase rate to weigh it
 * (CHUNK_RECHUNK_CHURN trace; nothing decides on them yet).
 *
 * phase_ends(): process-wide count of compression phases (end of each
 * parallel phase). Materialisations<T>: per element type, per array
 * allocation key, per rank -- rank-local increments, summed by the reader at
 * a quiescent point (an advisor tick). Only dynamic builds give arrays their
 * own key; key 0 (the shared pool) is not counted.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"

namespace shrink_array::churn
{

inline std::atomic<long> g_phase_ends{0};

/// @brief One compression phase ended (called once per parallel phase).
inline void note_phase_end() noexcept
{
    g_phase_ends.fetch_add(1, std::memory_order_relaxed);
}

/// @brief Compression phases ended so far.
[[nodiscard]] inline long phase_ends() noexcept
{
    return g_phase_ends.load(std::memory_order_relaxed);
}

/// @brief Materialisations per (array allocation key, rank), for element type T.
template <typename T>
struct Materialisations
{
  private:
    static constexpr unsigned kKeys = 128; ///< key slots per rank (key % kKeys)

  public:
    /// @brief One rank's counters, one per key slot.
    struct alignas(128) RankSlot
    {
        std::array<long, kKeys> n;
    };

  private:
    static inline std::vector<RankSlot> slots_; ///< one per rank, sized once (ensure)
    static inline std::once_flag        once_;

    /// @brief Size slots_ once to CHUNK_PMP_MAX_THREADS entries (at least 1), zero-initialised.
    static void ensure() noexcept
    {
        std::call_once(once_,
                       []
                       {
                           const int n = CHUNK_PMP_MAX_THREADS;
                           slots_.resize(static_cast<std::size_t>(n > 0 ? n : 1)); // value-initialised: 0
                       });
    }

  public:
    /// @brief One materialisation of array @p key on rank @p thr (rank-local).
    static void note(int thr, unsigned key) noexcept
    {
        ensure();
        if (thr < 0 || static_cast<std::size_t>(thr) >= slots_.size())
            return;
        ++slots_[static_cast<std::size_t>(thr)].n[key % kKeys];
    }

    /// @brief All ranks' materialisations of array @p key (quiescent point).
    [[nodiscard]] static long total(unsigned key) noexcept
    {
        if (key == 0)
            return 0;
        long s = 0;
        for (const RankSlot& slot : slots_)
            s += slot.n[key % kKeys];
        return s;
    }
};

/// @brief Count one materialisation of the array whose allocation key is
///        @p key (dynamic builds; key 0, the shared pool, is not counted).
template <typename T>
inline void note_materialisation([[maybe_unused]] int thr, [[maybe_unused]] unsigned key) noexcept
{
    if constexpr (build_config::chunk_dynamic)
    {
        if (key != 0)
            Materialisations<T>::note(thr, key);
    }
}

} // namespace shrink_array::churn
