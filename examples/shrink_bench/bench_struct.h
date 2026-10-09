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

/**
 * @file bench_struct.h
 * @brief The two aggregate element types the bench drives next to its
 *        arithmetic ones, and their second values.
 *
 * An array of a struct is an array like any other to the library (the
 * element contract: trivially copyable, ==, T{} -- see the README), and the
 * bench shows what that buys: runs of null triplets and of a repeated point
 * collapse like runs of 0.0 and of a second value. Two shapes:
 *
 *  - Point3, three doubles: the mesh-point / velocity case, 24 bytes, no
 *    padding;
 *  - Cell, a material index and a volume fraction: the multi-material cell
 *    state, 8 bytes, no padding, two member types.
 *
 * NO PADDING on purpose: the check mode dumps the std::vector baseline's
 * bytes to a file and sums the bytes of every non-zero element into a
 * checksum, and padding bytes are not part of a value (a defaulted operator==
 * ignores them) but would be part of those bytes.
 *
 * What a struct element does NOT get: the compound operators of the proxies
 * (+=, *=, ++ ...) exist for the types that have the operation; Point3 has
 * no operator+, so `tab_point[i] += p` does not compile. The bench's struct
 * loops read the element, build the new one, assign it -- the shape the
 * library offers every element type.
 */
#pragma once

#include <compare> // the defaulted operator<=>
#include <cstdint> // std::int32_t
#include <format>  // std::formatter
#include <ostream> // std::ostream

/// Three doubles. The defaulted <=> gives the == every element type needs and
/// the < chunk2 asks of a floating-point aggregate (value_less.h).
struct Point3
{
    auto operator<=>(const Point3&) const = default;

    double x, y, z;
};

/// A material index and a volume fraction: the state of a multi-material cell.
struct Cell
{
    auto operator<=>(const Cell&) const = default;

    std::int32_t material;
    float        fraction;
};

/// @brief std::format of a Point3: "(x, y, z)", each coordinate as %g. No format spec.
template <>
struct std::formatter<Point3>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }
    auto format(const Point3& p, std::format_context& ctx) const
    {
        return std::format_to(ctx.out(), "({:g}, {:g}, {:g})", p.x, p.y, p.z);
    }
};

/// @brief std::format of a Cell: "{material, fraction}", the fraction as %g. No format spec.
template <>
struct std::formatter<Cell>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }
    auto format(const Cell& c, std::format_context& ctx) const
    {
        return std::format_to(ctx.out(), "{{{}, {:g}}}", c.material, c.fraction);
    }
};

/// @brief Print @p p as "(x, y, z)" (its std::formatter).
/// @return @p os.
inline std::ostream& operator<<(std::ostream& os, const Point3& p)
{
    return os << std::format("{}", p);
}

/// @brief Print @p c as "{material, fraction}" (its std::formatter).
/// @return @p os.
inline std::ostream& operator<<(std::ostream& os, const Cell& c)
{
    return os << std::format("{}", c);
}

/// @name The second values of the struct arrays
/// @{
inline constexpr Point3 second_value_p{1., 2., 3.};
inline constexpr Cell   second_value_cell{1, 0.5f};
/// @}
