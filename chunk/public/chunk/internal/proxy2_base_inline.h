
#pragma once

#include <utility>                      // std::forward
#include "chunk/internal/proxy2_base.h" // the declarations these bodies belong to
#include "chunk/internal/in_t.h"

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
namespace shrink_array
{

/**
 * @brief Core unified logic for 2-value compression marking.
 * Handles both standard and 'pre' checks using variadic args for second_value/first_value.
 */

template <typename Derived, typename T, typename ChunkType>
template <typename... Args>
void Proxy2Base<Derived, T, ChunkType>::mark_for_compression_core(in_t<T> old_value,
                                                                  in_t<T> new_value,
                                                                  in_t<T> expected_in_memory,
                                                                  bool    isv,
                                                                  Args&&... compression_args) const
{
    // Check if transition is valid for compression OR if memory changed unexpectedly (Race)
    // isv (Is Valid) guards the entire check for performance.
    if (isv && (derived().should_mark_for_compression(old_value, new_value, std::forward<Args>(compression_args)...) ||
                (Base::load_value() != expected_in_memory))) [[unlikely]]
    {
        add_to_compression_list();
    }

    // NOTE: a dead check was removed here. It consulted
    // WriteCompressSyncPolicy<Derived>::is_slow_path_active() — but
    // active_slow_path is only ever set by the Chunk v1 proxy
    // (proxy_inline.h, activate_slow_path before make_block_compression), and
    // these Policy statics are per template instantiation: the v2/v2A
    // instantiations reaching this code NEVER activate it, so the predicate
    // was constant-false and cost one atomic load + branch on every write.
}

} // namespace shrink_array