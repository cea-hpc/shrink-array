#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
soa_binding_packs.py -- SoaArray: C++26 structured binding packs over the
columns (operator[], shrink_to_fit, resize) in place of an index_sequence
lambda and std::apply. soa_array.h is built only with reflection (GCC 16),
so it may use them without a fallback.

Run from the root of the shrink-array tree (branch cpp26):  python3 soa_binding_packs.py
Idempotent; writes nothing unless every hunk applies.
"""
import os
import re
import sys

ROOT = os.getcwd()

HUNKS = [
    ('chunk/public/chunk/soa_array.h',
     r'''    #include <span>        // std::span (read_range's visitor)
    #include <string>      // std::string (the column names, consteval)
    #include <string_view> // std::string_view (the member names)
    #include <tuple>       // std::tuple, std::get, std::apply
    #include <type_traits> // std::is_class_v
    #include <utility>     // std::index_sequence, std::declval, std::as_const
    #include <vector>      // std::vector (the member lists, consteval; a soa::plain column)
''',
     r'''    #include <span>        // std::span (read_range's visitor)
    #include <string>      // std::string (the column names, consteval)
    #include <string_view> // std::string_view (the member names)
    #include <tuple>       // std::tuple, std::get (a structured binding pack over the columns)
    #include <type_traits> // std::is_class_v
    #include <utility>     // std::index_sequence, std::declval, std::as_const
    #include <vector>      // std::vector (the member lists, consteval; a soa::plain column)
'''),
    ('chunk/public/chunk/soa_array.h',
     r'''    /// @p n elements, every value T{}'s (no buffer allocated; a soa::plain
    /// column is a std::vector of n V{}).
    explicit SoaArray(size_type n)
        : columns_([n]<std::size_t... K>(std::index_sequence<K...>)
                   { return Columns{std::tuple_element_t<K, Columns>(static_cast<int>(n))...}; }(
                       std::make_index_sequence<n_columns_>{}))
    {
    }

''',
     r'''    /// @p n elements, every value T{}'s (no buffer allocated; a soa::plain
    /// column is a std::vector of n V{}).
    explicit SoaArray(size_type n)
        : columns_(
              [n]<std::size_t... K>(std::index_sequence<K...>) {
                  return Columns{std::tuple_element_t<K, Columns>(static_cast<int>(n))...};
              }(std::make_index_sequence<n_columns_>{}))
    {
    }

'''),
    ('chunk/public/chunk/soa_array.h',
     r'''    /// The write side: T's members, the columns' proxies.
    [[nodiscard]] Ref operator[](size_type i)
    {
        return [&]<std::size_t... K>(std::index_sequence<K...>) -> Ref
        {
            return Ref{std::get<K>(columns_)[i]...}; // brace elision fills the nested parts in order
        }(std::make_index_sequence<n_columns_>{});
    }

    /// The read side: the element, gathered from the columns.
''',
     r'''    /// The write side: T's members, the columns' proxies.
    [[nodiscard]] Ref operator[](size_type i)
    {
        auto& [... cols] = columns_; // C++26 structured binding pack
        return Ref{cols[i]...};      // brace elision fills the nested parts in order
    }

    /// The read side: the element, gathered from the columns.
'''),
    ('chunk/public/chunk/soa_array.h',
     r'''    /// Sequential point: fold the chunks that became constant, every column.
    void shrink_to_fit()
    {
        std::apply([](auto&... c) { (c.shrink_to_fit(), ...); }, columns_);
    }

    /// Sequential point: @p n elements, every column; the new ones T{}'s.
    void resize(size_type n)
    {
        std::apply([n](auto&... c) { (c.resize(static_cast<decltype(c.size())>(n)), ...); }, columns_);
    }

    /// Sequential point: @p n elements, the new ones @p v (std::vector's semantics).
''',
     r'''    /// Sequential point: fold the chunks that became constant, every column.
    void shrink_to_fit()
    {
        auto& [... cols] = columns_;
        (cols.shrink_to_fit(), ...);
    }

    /// Sequential point: @p n elements, every column; the new ones T{}'s.
    void resize(size_type n)
    {
        auto& [... cols] = columns_;
        (cols.resize(static_cast<decltype(cols.size())>(n)), ...);
    }

    /// Sequential point: @p n elements, the new ones @p v (std::vector's semantics).
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


SENTINEL = "auto& [... cols] = columns_;"  # in every file once changed, in none before
SENTINELS = {}



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
