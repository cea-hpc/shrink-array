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

namespace shrink_array
{
/**
 * @brief Always false, but dependent on its arguments: the condition of a
 *        static_assert in the discarded branch of an `if constexpr`.
 * @details `static_assert(false)` in such a branch is ill-formed before C++23
 *          (P2593); `static_assert(dependent_false_v<T>, "...")` fires only
 *          when the branch is instantiated -- a member meant for bool called on
 *          a Chunk2<double>, say -- turning what would be a silent no-op in
 *          release (an abort in debug only) into a compile error.
 */
template <typename...>
inline constexpr bool dependent_false_v = false;
} // namespace shrink_array
