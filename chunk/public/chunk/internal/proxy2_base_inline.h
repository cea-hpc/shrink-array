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

#include <utility> // std::forward

#include "chunk/internal/proxy2_base.h" // the declarations these bodies belong to
#include "chunk/internal/in_t.h"        // InT<T>

namespace shrink_array
{

/**
 * @brief Core unified logic for 2-value compression marking.
 * Handles both standard and 'pre' checks using variadic args for second_value/first_value.
 */

template <typename Derived, typename T, typename ChunkT>
template <typename... Args>
void SecondValueProxyBase<Derived, T, ChunkT>::mark_for_compression_core(InT<T> old_value,
                                                                         InT<T> new_value,
                                                                         InT<T> expected_in_memory,
                                                                         bool   isv,
                                                                         Args&&... compression_args) const
{
    // Check if transition is valid for compression OR if memory changed unexpectedly (Race)
    // isv (Is Valid) guards the entire check for performance.
    if (isv && (derived().should_mark_for_compression(old_value, new_value, std::forward<Args>(compression_args)...) ||
                (Base::load_value() != expected_in_memory))) [[unlikely]]
    {
        add_to_compression_list();
    }
}

} // namespace shrink_array