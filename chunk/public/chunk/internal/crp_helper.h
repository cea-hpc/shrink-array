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
// When available, a CRTP method body may take an explicit object parameter
// (this auto& self) instead of calling derived(); gate it with
// #if CHUNK_CRTP_USE_DEDUCING_THIS. The class keeps its Derived parameter and CRTPSupport base.
#ifdef CHUNK_CRTP_FORCE_DEDUCING_THIS
    #define CHUNK_CRTP_USE_DEDUCING_THIS CHUNK_CRTP_FORCE_DEDUCING_THIS
#elif defined(__cpp_explicit_this_parameter) && __cpp_explicit_this_parameter >= 202110L
    #define CHUNK_CRTP_USE_DEDUCING_THIS 1
#else
    #define CHUNK_CRTP_USE_DEDUCING_THIS 0
#endif

namespace shrink_array
{

/**
 * @brief Simple mixin to provide derived() and self() cast helpers for CRTP.
 */
template <typename Derived>
class CRTPSupport
{
  protected:
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
    /// @copydoc self()
    [[nodiscard]] const Derived* self() const noexcept
    {
        return static_cast<const Derived*>(this);
    }
};

} // namespace shrink_array