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

#include <version> // __cpp_explicit_this_parameter

// --- Optional C++23 "deducing this" path for CRTP methods --------------------
// When available (GCC 14+, Clang 18+, MSVC 19.32+), individual CRTP method
// bodies may take an explicit object parameter (this auto& self) instead of
// calling derived(). Gate each converted method with #if CRTP_USE_DEDUCING_THIS;
// the class keeps its Derived parameter and CRTPSupport base either way, so the
// change is strictly local to the method bodies you convert.
#if defined(CRTP_FORCE_DEDUCING_THIS)
    #define CRTP_USE_DEDUCING_THIS CRTP_FORCE_DEDUCING_THIS
#elif defined(__cpp_explicit_this_parameter) && __cpp_explicit_this_parameter >= 202110L
    #define CRTP_USE_DEDUCING_THIS 1
#else
    #define CRTP_USE_DEDUCING_THIS 0
#endif

namespace shrink_array
{

/**
 * @brief Simple mixin to provide derived() and self() cast helpers for CRTP.
 */
template <typename Derived>
struct CRTPSupport
{
    /// @brief `*this` as the derived class.
    Derived& derived() noexcept
    {
        return static_cast<Derived&>(*this);
    }
    /// @copydoc derived()
    [[nodiscard]] const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    /// @brief `this` as a pointer to the derived class.
    Derived* self() noexcept
    {
        return static_cast<Derived*>(this);
    }

    [[nodiscard]] const Derived* self() const noexcept
    {
        return static_cast<const Derived*>(this);
    }
};

} // namespace shrink_array