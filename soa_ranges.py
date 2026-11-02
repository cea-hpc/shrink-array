#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
soa_ranges.py -- SoaArray<T>: the range operations member by member
(fill_range, copy_range from structs, their _mt forms, read_range back into
structs), resize(), push_back(), member_count / member(k) / column_at<k>().

Run from the root of the shrink-array tree (branch cpp26):  python3 soa_ranges.py
After soa_annotations.py.
Idempotent; writes nothing unless every hunk applies.

  chunk/public/chunk/soa_array.h                    the operations, the file comment
  examples/soa_array/soa_array_validation.cpp       ranges(): against a std::vector, _mt in a phase
  CHANGELOG.md                                      Unreleased Added (cpp26)
"""
import os
import re
import sys

ROOT = os.getcwd()

HUNKS = [
    ('CHANGELOG.md',
     r'''  `soa_array_validation` checks the column types, the values after parallel
  writes, compound ops and `set()`, and that the annotated compressed columns
  compress.

### Fixed (cpp26)

''',
     r'''  `soa_array_validation` checks the column types, the values after parallel
  writes, compound ops and `set()`, and that the annotated compressed columns
  compress.
- `SoaArray<T>`: the range operations, member by member, each column
  keeping its own semantics (a chunk filled with one value takes it as its
  constant, a `soa::plain` column is a `std::fill` / `std::copy`):
  `fill_range(a, lo, hi, v)`, `copy_range(a, lo, hi, src)` from structs (the
  AoS to SoA transposition, through a projection of each member),
  `fill_range_mt`, `copy_range_mt` (every thread of a phase on its own
  range), `read_range(a, lo, hi, out)` back into structs (a compressed chunk
  a fill of its member); `resize(n)`, `resize(n, v)`, `push_back(v)`; and
  `member_count`, `member(k)`, `column_at<k>()` to walk the columns by
  index. The file comment no longer announces a `compress()` it did not
  have. `soa_array_validation` checks them against a `std::vector`, the
  `_mt` forms from a phase, a range past the end clamped.

### Fixed (cpp26)

'''),
    ('chunk/public/chunk/soa_array.h',
     r''' *          auto& mat = a.column<^^Cell::mat>(); // the zero_any::Array<int> itself:
 *                                               // range operations, stats, mdspan views
 *          @endcode
 *
 *          A member can pick its own storage with an annotation (C++26,
 *          P3394), the other members keeping the array's Column:
''',
     r''' *          auto& mat = a.column<^^Cell::mat>(); // the zero_any::Array<int> itself:
 *                                               // range operations, stats, mdspan views
 *          @endcode
 *
 *          Range operations, member by member (each column keeps its own
 *          semantics: a chunk filled with one value takes it as its
 *          constant): fill_range(a, lo, hi, v), copy_range(a, lo, hi, src)
 *          from structs (the AoS to SoA transposition), their _mt forms
 *          (every thread of a phase on its own range), read_range(a, lo, hi,
 *          out) back into structs; resize(), push_back() forward to every
 *          column.
 *
 *          A member can pick its own storage with an annotation (C++26,
 *          P3394), the other members keeping the array's Column:
'''),
    ('chunk/public/chunk/soa_array.h',
     r'''
#if SHRINK_ARRAY_HAS_SOA_ARRAY

    #include <cstddef>     // std::size_t
    #include <ranges>      // std::views::iota
    #include <string_view> // std::string_view (the member names)
    #include <tuple>       // std::tuple, std::get, std::apply
    #include <type_traits> // std::is_class_v
    #include <utility>     // std::index_sequence, std::declval, std::as_const
    #include <vector>      // std::vector (the member lists, consteval; a soa::plain column)

    #include "chunk/array_chunk2A.h"                        // zero_any::Array, the default column
    #include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement

namespace shrink_array
''',
     r'''
#if SHRINK_ARRAY_HAS_SOA_ARRAY

    #include <cstddef>     // std::size_t, std::ptrdiff_t
    #include <iterator>    // std::random_access_iterator (the range operations)
    #include <ranges>      // std::views::iota, std::views::transform, std::ranges::subrange
    #include <span>        // std::span (read_range's visitor)
    #include <string_view> // std::string_view (the member names)
    #include <tuple>       // std::tuple, std::get, std::apply
    #include <type_traits> // std::is_class_v
    #include <utility>     // std::index_sequence, std::declval, std::as_const
    #include <vector>      // std::vector (the member lists, consteval; a soa::plain column)

    #include "chunk/array_chunk2A.h"         // zero_any::Array, the default column
    #include "chunk/array_chunk_range_ops.h" // fill_range(_mt), copy_range(_mt), for_each_span (chunked columns)
    #include "chunk/vector_range_ops.h"      // the same on a soa::plain column (std::vector)
    #include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement

namespace shrink_array
'''),
    ('chunk/public/chunk/soa_array.h',
     r'''    {
        std::apply([](auto&... c) { (c.shrink_to_fit(), ...); }, columns_);
    }
};

} // namespace shrink_array

''',
     r'''    {
        std::apply([](auto&... c) { (c.shrink_to_fit(), ...); }, columns_);
    }

    /// Sequential point: @p n elements, every column; the new ones T{}'s.
    void resize(size_type n)
    {
        std::apply([n](auto&... c) { (c.resize(static_cast<decltype(c.size())>(n)), ...); }, columns_);
    }

    /// Sequential point: @p n elements, the new ones @p v (std::vector's semantics).
    void resize(size_type n, const T& v)
    {
        template for (constexpr std::size_t k : std::views::iota(0uz, n_members_))
        {
            auto& c = std::get<k>(columns_);
            c.resize(static_cast<decltype(c.size())>(n), v.[:members_[k]:]);
        }
    }

    /// Append @p v, every member to its column (resize() beyond the capacity:
    /// a sequential point).
    void push_back(const T& v)
    {
        template for (constexpr std::size_t k : std::views::iota(0uz, n_members_))
        {
            std::get<k>(columns_).push_back(v.[:members_[k]:]);
        }
    }

    /// The number of members of T, the number of columns.
    static constexpr std::size_t member_count = n_members_;

    /// The reflection of member @p k of T (declaration order).
    static consteval std::meta::info member(std::size_t k)
    {
        return members_[k];
    }

    /// The column of member @p K (declaration order): the range operations' way in.
    template <std::size_t K>
    [[nodiscard]] auto& column_at() noexcept
    {
        return std::get<K>(columns_);
    }
    template <std::size_t K>
    [[nodiscard]] const auto& column_at() const noexcept
    {
        return std::get<K>(columns_);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Range operations: the arrays' own, member by member. Each column keeps its
// semantics -- a chunk filled with one value takes it as its constant, a
// soa::plain column is a std::fill / std::copy -- and the rules are the
// columns' (the _mt forms called by every thread of a phase on its own range).
// ─────────────────────────────────────────────────────────────────────────────

/// Every element of [@p first, @p last) set to @p v, member by member.
template <typename T, template <typename> class Column>
void fill_range(SoaArray<T, Column>& a, std::size_t first, std::size_t last, const T& v)
{
    template for (constexpr std::size_t k : std::views::iota(0uz, SoaArray<T, Column>::member_count))
    {
        fill_range(a.template column_at<k>(), first, last, v.[:SoaArray<T, Column>::member(k):]);
    }
}

/// fill_range() from every thread of a phase, each on its own range.
template <typename T, template <typename> class Column>
void fill_range_mt(SoaArray<T, Column>& a, std::size_t first, std::size_t last, const T& v)
{
    template for (constexpr std::size_t k : std::views::iota(0uz, SoaArray<T, Column>::member_count))
    {
        fill_range_mt(a.template column_at<k>(), first, last, v.[:SoaArray<T, Column>::member(k):]);
    }
}

namespace soa_detail
{
/// [src, src + n) seen as the values of member @p M: a random-access iterator.
template <std::meta::info M, std::random_access_iterator It>
auto member_values(It src, std::size_t n)
{
    return (std::ranges::subrange(src, src + static_cast<std::ptrdiff_t>(n)) |
            std::views::transform([](const auto& x) { return x.[:M:]; }))
        .begin();
}

/// for_each_span()'s visitor of read_range(): the values of a column into
/// member @p M of out[j], out[j + 1], ...
template <std::meta::info M, typename OutIt>
struct MemberWriter
{
    OutIt        out;
    std::size_t& j;

    template <typename V>
    void operator()(std::span<const V> s) const
    {
        for (const V& x : s)
        {
            out[static_cast<std::ptrdiff_t>(j++)].[:M:] = x;
        }
    }

    /// A compressed chunk: @p count times @p x.
    template <typename V>
    void operator()(const V& x, std::size_t count) const
    {
        for (std::size_t c = 0; c < count; ++c)
        {
            out[static_cast<std::ptrdiff_t>(j++)].[:M:] = x;
        }
    }
};

/// The elements of [first, last) clamped to @p size: their count.
inline std::size_t clamped_count(std::size_t first, std::size_t& last, std::size_t size)
{
    last = last < size ? last : size;
    return first < last ? last - first : 0;
}
} // namespace soa_detail

/// [@p first, @p last) from the structs [src, src + (last - first)) -- the
/// array-of-structs to structure-of-arrays transposition, member by member.
template <typename T, template <typename> class Column, std::random_access_iterator It>
void copy_range(SoaArray<T, Column>& a, std::size_t first, std::size_t last, It src)
{
    const std::size_t n = soa_detail::clamped_count(first, last, a.size());
    if (n == 0)
    {
        return;
    }
    template for (constexpr std::size_t k : std::views::iota(0uz, SoaArray<T, Column>::member_count))
    {
        copy_range(a.template column_at<k>(), first, last,
                   soa_detail::member_values<SoaArray<T, Column>::member(k)>(src, n));
    }
}

/// copy_range() from every thread of a phase, each on its own range.
template <typename T, template <typename> class Column, std::random_access_iterator It>
void copy_range_mt(SoaArray<T, Column>& a, std::size_t first, std::size_t last, It src)
{
    const std::size_t n = soa_detail::clamped_count(first, last, a.size());
    if (n == 0)
    {
        return;
    }
    template for (constexpr std::size_t k : std::views::iota(0uz, SoaArray<T, Column>::member_count))
    {
        copy_range_mt(a.template column_at<k>(), first, last,
                      soa_detail::member_values<SoaArray<T, Column>::member(k)>(src, n));
    }
}

/// The structs of [@p first, @p last) into [out, out + (last - first)) -- the
/// way back, member by member: a compressed chunk is a fill of its member.
/// @return @p out advanced past the last element written.
template <typename T, template <typename> class Column, std::random_access_iterator OutIt>
OutIt read_range(const SoaArray<T, Column>& a, std::size_t first, std::size_t last, OutIt out)
{
    const std::size_t n = soa_detail::clamped_count(first, last, a.size());
    if (n == 0)
    {
        return out;
    }
    template for (constexpr std::size_t k : std::views::iota(0uz, SoaArray<T, Column>::member_count))
    {
        std::size_t j = 0;
        for_each_span(a.template column_at<k>(), first, last,
                      soa_detail::MemberWriter<SoaArray<T, Column>::member(k), OutIt>{out, j});
    }
    return out + static_cast<std::ptrdiff_t>(n);
}

} // namespace shrink_array

'''),
    ('examples/soa_array/soa_array_validation.cpp',
     r'''//    annotation left alone -- gets those column types (compile time), reads
//    back as a std::vector after parallel writes, compound ops and set(), and
//    its compressed columns compress; two storage annotations on one member
//    are refused (compile time).

#include <cstdio>      // std::printf
#include <type_traits> // std::is_same_v, std::remove_cvref_t (the column types)
#include <utility>     // std::as_const
''',
     r'''//    annotation left alone -- gets those column types (compile time), reads
//    back as a std::vector after parallel writes, compound ops and set(), and
//    its compressed columns compress; two storage annotations on one member
//    are refused (compile time);
//  - the range operations, member by member: fill_range, copy_range (from a
//    std::vector<Particle>: AoS to SoA), their _mt forms from every thread of
//    a phase, read_range back into structs (a compressed chunk as a fill),
//    a range past the end clamped; resize() with and without a value,
//    push_back(); a uniform fill leaves its compressed columns' chunks
//    without a buffer.

#include <algorithm>   // std::fill, std::copy (the reference)
#include <cstdio>      // std::printf
#include <type_traits> // std::is_same_v, std::remove_cvref_t (the column types)
#include <utility>     // std::as_const
'''),
    ('examples/soa_array/soa_array_validation.cpp',
     r'''    return bad;
}

// What element i holds after the parallel write.
static Cell expected(int i, int n)
{
''',
     r'''    return bad;
}

static bool same_particle(const Particle& p, const Particle& q)
{
    return p.x == q.x && p.mat == q.mat && p.rho == q.rho && p.flag == q.flag;
}

static int ranges(int n)
{
    int                    bad = 0;
    std::vector<Particle>  v(static_cast<std::size_t>(n));
    sa::SoaArray<Particle> a(static_cast<std::size_t>(n));
    const auto             check = [&](const char* what)
    {
        std::vector<Particle> out(static_cast<std::size_t>(n), Particle{-1.0, -1, -1.0, true});
        const auto            end = read_range(a, 0, static_cast<std::size_t>(n), out.begin());
        int                   w   = end == out.end() ? 0 : 1;
        for (int i = 0; i < n; ++i)
        {
            w += same_particle(out[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i)]) &&
                         same_particle(std::as_const(a)[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i)])
                     ? 0
                     : 1;
        }
        if (w != 0)
        {
            std::printf("ranges: %s, %d wrong\n", what, w);
            ++bad;
        }
    };

    // Serial: a uniform fill over whole chunks, then a copy from structs.
    const Particle u{3.5, 7, 2.0, true};
    sa::fill_range(a, static_cast<std::size_t>(n / 8), static_cast<std::size_t>(n / 4), u);
    std::fill(v.begin() + n / 8, v.begin() + n / 4, u);
    check("fill_range");
    if (a.column<^^Particle::rho>().stats() > 1.0 || a.column<^^Particle::flag>().stats() > 1.0)
    {
        std::printf("ranges: a uniform fill left buffers in a compressed column\n");
        ++bad;
    }
    std::vector<Particle> src(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] = expected_particle(i, n);
    }
    sa::copy_range(a, static_cast<std::size_t>(n / 2), static_cast<std::size_t>(n) + 100, src.begin()); // clamped
    std::copy(src.begin(), src.begin() + (n - n / 2), v.begin() + n / 2);
    check("copy_range");

    // _mt: every thread its blocks, in a phase.
    const int block = 4096;
    sa::parallel_phase(
        [&]
        {
#pragma omp for schedule(static)
            for (int b = 0; b < n / block; ++b)
            {
                const auto lo = static_cast<std::size_t>(b) * block;
                if (b % 3 == 0)
                {
                    sa::fill_range_mt(a, lo, lo + block, Particle{1.0 * b, b % 5, 0.5, false});
                }
                else
                {
                    sa::copy_range_mt(a, lo, lo + block, src.begin() + static_cast<std::ptrdiff_t>(lo));
                }
            }
        });
    for (int b = 0; b < n / block; ++b)
    {
        const auto lo = static_cast<std::size_t>(b) * block;
        for (std::size_t i = lo; i < lo + block; ++i)
        {
            v[i] = b % 3 == 0 ? Particle{1.0 * b, b % 5, 0.5, false} : src[i];
        }
    }
    check("fill_range_mt, copy_range_mt");

    // resize, push_back.
    a.resize(static_cast<std::size_t>(n) + 1000, u);
    v.resize(static_cast<std::size_t>(n) + 1000, u);
    a.push_back(src[5]);
    v.push_back(src[5]);
    n = static_cast<int>(v.size());
    if (a.size() != v.size())
    {
        std::printf("ranges: size %zu after resize and push_back, expected %zu\n", a.size(), v.size());
        ++bad;
    }
    check("resize(n, v), push_back");
    a.resize(static_cast<std::size_t>(n / 2));
    v.resize(static_cast<std::size_t>(n / 2));
    n = static_cast<int>(v.size());
    check("resize(n)");
    std::printf("ranges: %d wrong\n", bad);
    return bad;
}

// What element i holds after the parallel write.
static Cell expected(int i, int n)
{
'''),
    ('examples/soa_array/soa_array_validation.cpp',
     r'''    } // the columns release their chunks here

    bad += particles(1 << 20);

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
''',
     r'''    } // the columns release their chunks here

    bad += particles(1 << 20);
    bad += ranges(1 << 20);

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
'''),
]

NEW_FILES = {
}

def tokens(text):
    # comment line stars are layout, not text
    return re.sub(r"(?m)^[ \t]*\*(?=[ \t]|$)", " ", text).split()


def fuzzy(text):
    sep = r"\s+(?:\*\s+)*"  # no nested quantifier: linear on a miss
    return re.compile(sep.join(re.escape(t) for t in tokens(text)))


def core(old, new):
    """new without the leading and trailing layout (spaces, comment stars)
    old shares with it: the fuzzy match starts and ends on a token."""
    # the tokens of old, as tokens() sees them: a lone star at a line start is layout
    toks = [m for m in re.finditer(r"\S+", old)
            if not (m.group() == "*" and old[old.rfind("\n", 0, m.start()) + 1:m.start()].strip() == "")]
    lead = toks[0].start()
    trail = len(old) - toks[-1].end()
    if new[:lead] == old[:lead]:
        new = new[lead:]
    if trail and new[-trail:] == old[-trail:]:
        new = new[:-trail]
    return new


SENTINEL = "read_range(a, lo, hi, out)"  # in every file once changed, in none before
SENTINELS = {"chunk/public/chunk/soa_array.h": "struct MemberWriter",
             "examples/soa_array/soa_array_validation.cpp": "static int ranges(int n)"}



errors = []
texts = {}
orig = {}
for p, old, new in HUNKS:
    if p not in texts:
        if not os.path.exists(os.path.join(ROOT, p)):
            if f"{p}: missing" not in errors:
                errors.append(f"{p}: missing")
            continue
        with open(os.path.join(ROOT, p), encoding="utf-8") as f:
            texts[p] = orig[p] = f.read()
    s = texts[p]
    if SENTINELS.get(p, SENTINEL) in orig[p]:
        continue  # this file already has the change
    if s.count(old) == 1:
        texts[p] = s.replace(old, new)
        continue
    # Not found as is: the same text with other spaces, line breaks or
    # comment reflow (another clang-format) -- matched token by token.
    found = list(fuzzy(old).finditer(s))
    if len(found) != 1:
        errors.append(f"{p}: anchor found {len(found)} times: {old.strip().splitlines()[0][:70]!r}")
        continue
    m = found[0]
    texts[p] = s[:m.start()] + core(old, new) + s[m.end():]

for p, content in NEW_FILES.items():
    full = os.path.join(ROOT, p)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            if f.read() != content:
                errors.append(f"{p}: exists with other content")
    elif not os.path.isdir(os.path.dirname(full)):
        errors.append(f"{p}: no directory {os.path.dirname(p)}")
    texts[p] = content

if errors:
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    print("nothing written", file=sys.stderr)
    sys.exit(1)
written = 0
for p, s in texts.items():
    full = os.path.join(ROOT, p)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            if f.read() == s:
                continue
    with open(os.path.join(ROOT, p), "w", encoding="utf-8") as f:
        f.write(s)
    print("written:", p)
    written += 1
if not written:
    print("already applied: nothing to do")
