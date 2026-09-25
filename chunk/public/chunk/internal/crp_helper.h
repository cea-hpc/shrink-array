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

// --- C++23 "deducing this" in CRTP methods ------------------------------------
// The library requires C++23: a CRTP method body may take an explicit object
// parameter (this auto& self) instead of calling derived(). The class keeps its
// Derived parameter and the CRTPSupport base, so a conversion stays local to the
// method bodies converted.

#include <memory>      // std::addressof
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

/**
 * @brief Simple mixin to provide derived() and self() cast helpers for CRTP.
 */
template <typename Derived>
struct CRTPSupport
{
    /// @brief `*this` as the derived class, const on a const object.
    template <typename Self>
    auto& derived(this Self&& self) noexcept
    {
        return static_cast<crtp_cast_t<Self, Derived>&>(self);
    }

    /// @brief derived()'s address. std::addressof: the proxies delete operator&.
    template <typename Self>
    auto* self(this Self&& obj) noexcept
    {
        return std::addressof(static_cast<crtp_cast_t<Self, Derived>&>(obj));
    }
};

} // namespace shrink_array