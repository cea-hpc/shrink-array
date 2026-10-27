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
#include "chunk/internal/churn_counters.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::Materialisations<T>
// ---------------------------------------------------------------------------

template <typename T>
void churn::Materialisations<T>::note(int thr, unsigned key) noexcept
{
    ensure();
    if (thr < 0 || static_cast<std::size_t>(thr) >= slots_.size())
    {
        return;
    }
    ++slots_[static_cast<std::size_t>(thr)].n[key % key_slots];
}

template <typename T>
[[nodiscard]] long churn::Materialisations<T>::total(unsigned key) noexcept
{
    if (key == 0)
    {
        return 0;
    }
    long sum = 0;
    for (const RankSlot& slot : slots_)
    {
        sum += slot.n[key % key_slots];
    }
    return sum;
}

template <typename T>
void churn::Materialisations<T>::ensure() noexcept
{
    std::call_once(once_,
                   []
                   {
                       const int nthreads = CHUNK_PMP_MAX_THREADS;
                       slots_.resize(static_cast<std::size_t>(nthreads > 0 ? nthreads : 1)); // value-initialised: 0
                   });
}

} // namespace shrink_array
