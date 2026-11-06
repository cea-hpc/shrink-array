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

/**
 * @file pattern_proxy.h
 * @brief pattern::Proxy, the write proxy of pattern::Array: zero_any::ProxyImpl
 *        with pattern::Chunk as the chunk type (docs/algo_pattern.md).
 */

#include <ostream> // std::ostream (operator<<)

#include "chunk/chunk_traits.h"            // pattern::Chunk, pattern::Proxy (forward declarations)
#include "chunk/internal/zero_any_proxy.h" // zero_any::ProxyImpl

namespace shrink_array
{

namespace pattern
{

/**
 * @brief Proxy to one element of a pattern::Chunk<T>: zero_any::ProxyImpl with
 *        itself as the proxy type and pattern::Chunk<T> as the chunk type. Its
 *        tables (second values, type-of-constant flags, the registry of arrays)
 *        are ProxyImpl's statics, so pattern::Array has its own.
 * @tparam T Element type: an integer type other than bool.
 */
template <typename T>
// no move assignment on purpose -- assigning a proxy
// WRITES the element (see its operator=); a defaulted move would rebind it instead.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Proxy : public zero_any::ProxyImpl<T, Proxy<T>, Chunk<T>>
{
    using Impl = zero_any::ProxyImpl<T, Proxy<T>, Chunk<T>>;

  public:
    using Impl::Impl;
    using Impl::operator=; // the value assignments (clean and trap paths)

    /// @brief Write the element's value, read through the proxy's gated read
    ///        (operator T()), to @p os.
    /// @return @p os.
    friend std::ostream& operator<<(std::ostream& os, const Proxy& proxy)
    {
        os << static_cast<T>(proxy);
        return os;
    }

    /// @brief Proxy-to-proxy assignment, `a[i] = b[j]`: a WRITE of @p from's
    ///        value (ProxyImpl::operator=), not a rebinding.
    /// @return *this.
    const Proxy& operator=(const Proxy& from) noexcept
    {
        Impl::operator=(from);
        return *this;
    }
    Proxy(const Proxy& other) = default;

    /// @brief Does this write complete its chunk: its first or its last element
    ///        (docs/algo_pattern.md, section 4)?
    [[nodiscard]] bool is_completion_write() const noexcept
    {
        return this->num_ == 0 || this->num_ == static_cast<int>(this->chunk_size()) - 1;
    }

    /// @brief zero_any's fast pre-check (ProxyImpl::pre_check_mark_for_compression()),
    ///        widened to the completion writes, which should_mark_for_compression()
    ///        accepts for a member with the motif state: the pre-check must never
    ///        reject what the predicate accepts.
    /// @return true if the write is worth examining further.
    [[nodiscard]] bool
    pre_check_mark_for_compression(InT<T> new_value, InT<T> first_value, InT<T> second_val) const noexcept
    {
        return is_completion_write() || Impl::pre_check_mark_for_compression(new_value, first_value, second_val);
    }
};

} // namespace pattern

} // namespace shrink_array
