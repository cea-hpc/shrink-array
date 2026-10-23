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
 * @file value_hash.h
 * @brief value_hash<T>(v): a hash of a whole element value, consistent with
 *        operator== where the type lets us -- the second-value index of
 *        Chunk2A (tab_cst_val2_holder.h).
 *
 * @details - A scalar of at most 8 bytes: a 64-bit finaliser over its bits
 *            (the former index_hash formula), a floating zero normalised so
 *            -0.0 and +0.0 hash alike.
 *          - A class whose operator== is the defaulted member one, with
 *            C++26 reflection (SHRINK_ARRAY_HAS_VALUE_HASH_MEMBERWISE): member
 *            by member, each by value_hash -- the padding is not read, and a
 *            signed zero in a member hashes like the other.
 *          - Anything else: every byte of the object, 8 at a time through the
 *            finaliser. Equal values with different padding or zero signs may
 *            then take two hashes: a second identifier (memory), not a wrong
 *            value, since the registry compares with operator==.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t
#include <cstring>     // std::memcpy
#include <memory>      // std::addressof
#include <type_traits> // std::is_floating_point_v, std::is_class_v

#if defined(__cpp_impl_reflection)
    #include <meta> // std::meta (reflection)
#endif

#if defined(__cpp_impl_reflection) && defined(__cpp_lib_reflection)
    #define SHRINK_ARRAY_HAS_VALUE_HASH_MEMBERWISE 1
#else
    #define SHRINK_ARRAY_HAS_VALUE_HASH_MEMBERWISE 0
#endif

namespace shrink_array
{

namespace value_hash_detail
{

/// 64-bit finalize (MurmurHash3 fmix64).
constexpr std::uint64_t mix64(std::uint64_t bits) noexcept
{
    bits ^= bits >> 33U;
    bits *= 0xff51afd7ed558ccdULL;
    bits ^= bits >> 33U;
    bits *= 0xc4ceb9fe1a85ec53ULL;
    bits ^= bits >> 33U;
    return bits;
}

/// Every byte of the object at @p p, 8 at a time (the last word zero-padded).
/// One word: mix64(word), the former formula.
inline std::uint64_t bytes_hash(const void* p, std::size_t size) noexcept
{
    const auto*   bytes = static_cast<const unsigned char*>(p);
    std::uint64_t h     = 0;
    for (std::size_t off = 0; off < size; off += sizeof(std::uint64_t))
    {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes + off, size - off < sizeof(word) ? size - off : sizeof(word));
        h = mix64(h ^ word);
    }
    return h;
}

#if SHRINK_ARRAY_HAS_VALUE_HASH_MEMBERWISE
/// True when @p type is a class without base whose operator== is the
/// defaulted member one: then equality is member by member.
consteval bool has_memberwise_equality(std::meta::info type)
{
    if (!std::meta::bases_of(type, std::meta::access_context::unchecked()).empty())
    {
        return false;
    }
    for (std::meta::info m : std::meta::members_of(type, std::meta::access_context::unchecked()))
    {
        if (std::meta::is_operator_function(m) && std::meta::operator_of(m) == std::meta::operators::op_equals_equals &&
            std::meta::is_defaulted(m))
        {
            return true;
        }
    }
    return false;
}
#endif
} // namespace value_hash_detail

/// @brief Hash of the value @p v, over the whole value (see the file comment).
template <typename T>
[[nodiscard]] std::size_t value_hash(const T& v) noexcept
{
    using value_hash_detail::bytes_hash;
    using value_hash_detail::mix64;
    if constexpr (std::is_floating_point_v<T>)
    {
        const T normalised = v == T{0} ? T{0} : v; // -0.0 hashes like +0.0
        return bytes_hash(std::addressof(normalised), sizeof(T));
    }
#if SHRINK_ARRAY_HAS_VALUE_HASH_MEMBERWISE
    else if constexpr (std::is_class_v<T> && value_hash_detail::has_memberwise_equality(^^T))
    {
        std::uint64_t h = 0;
        template for (constexpr std::meta::info f :
                      std::define_static_array(std::meta::nonstatic_data_members_of(
                          ^^T, std::meta::access_context::unchecked())))
        {
            using F = [:std::meta::type_of(f):];
            if constexpr (std::is_array_v<F>)
            {
                for (const auto& e : v.[:f:])
                {
                    h = mix64(h ^ value_hash(e));
                }
            }
            else
            {
                h = mix64(h ^ value_hash<F>(v.[:f:]));
            }
        }
        return h;
    }
#endif
    else
    {
        return bytes_hash(std::addressof(v), sizeof(T));
    }
}

} // namespace shrink_array
