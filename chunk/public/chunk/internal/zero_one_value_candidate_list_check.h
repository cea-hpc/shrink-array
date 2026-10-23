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

/**
 * @file zero_one_value_candidate_list_check.h
 * @brief The abort taken when a zero_one_value write reaches a per-second-value
 *        compression candidate list that does not exist.
 * @details Its own header because both zero_one_value_proxy.h and zero_one_value_chunk.h need it and
 *          zero_one_value_chunk.h can be parsed before zero_one_value_proxy.h's body.
 */

#pragma once

#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array
{

/// @brief A zero_one_value write reached a candidate list that does not exist (see
///        zero_one_value::Proxy::pre_check_mark_for_compression).
[[noreturn]] inline void chunk2_missing_candidate_list(int thr) noexcept
{
    CHUNK_INVARIANT_FAILED("[zero_one_value] rank {}: no compression candidate list for this array's second value -- "
                           "a write was queued for an array that never called set_second_value(); "
                           "zero_one_value::Proxy::pre_check_mark_for_compression() must reject it",
                           thr);
}

} // namespace shrink_array
