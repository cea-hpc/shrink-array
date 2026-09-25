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

namespace shrink_array
{

/**
 * @brief Simple mixin to provide derived() and self() cast helpers for CRTP.
 */
template <typename Derived>
struct CRTPSupport
{
    /// @brief `*this` as the derived class.
    Derived& derived()
    {
        return static_cast<Derived&>(*this);
    }
    /// @copydoc derived()
    const Derived& derived() const
    {
        return static_cast<const Derived&>(*this);
    }

    Derived* self()
    {
        return static_cast<Derived*>(this);
    }

    const Derived* self() const
    {
        return static_cast<const Derived*>(this);
    }
};

} // namespace shrink_array