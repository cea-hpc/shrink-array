//
// Created by ballereaup on 27/04/2026.
//
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

// --- C++23 "deducing this" ----------------------------------------------------
// The library requires C++23: CRTP method bodies take an explicit object
// parameter (this Derived& self, or this auto& self for a final class). The
// derived()/self() helper mixin this header once provided is gone -- no call
// site survived the migration; only crtp_cast_t remains, for the local cast
// helpers of the array-level hook classes still converting through it.

#include <type_traits> // std::conditional_t, std::is_const_v

namespace shrink_array
{

/**
 * @brief `Derived`, const when the object `Self` names is: the target of a CRTP
 *        down-cast made from a `this Self&& self` member, so that ONE member
 *        serves a const and a non-const object (C++23 deducing this) where a
 *        const / non-const pair used to repeat the same cast.
 */
template <typename Self, typename Derived>
using crtp_cast_t = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const Derived, Derived>;

} // namespace shrink_array