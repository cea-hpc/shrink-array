#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
deleted_reasons.py -- `= delete("reason")` (P2573, C++26) on the deliberate
restrictions of the public API: a proxy has no address (operator&), no
proxy-to-proxy operator=, ArrayChunkBase::data(), ChunkElems(),
ParamChunk::set_second_value. Branch cpp26 (GCC 16, -std=c++26): no fallback.

If the earlier version of this script (CHUNK_DELETED macro) was applied, it is
converted: the macro uses become delete("..."), the macro leaves config.h.

Run from the root of the shrink-array tree (branch cpp26):  python3 deleted_reasons.py
Idempotent; writes nothing unless every hunk applies.
"""
import os
import re
import sys

ROOT = os.getcwd()

HUNKS = [
    ('chunk/public/chunk/param_chunk.h',
     r'''     */
    template <typename T>
    static void set_second_value(T second_value) = delete;

    /**
''',
     r'''     */
    template <typename T>
    static void set_second_value(T second_value) = delete("not implemented: see zero_one_value / zero_any");

    /**
'''),
    ('chunk/public/chunk/internal/chunk_size_policy.h',
     r'''{
  public:
    ChunkElems() = delete;
    /// @brief Wrap an element count.
    /// @param count The element count.
''',
     r'''{
  public:
    ChunkElems() = delete("every chunk element count must be explicit: ChunkElems{n}");
    /// @brief Wrap an element count.
    /// @param count The element count.
'''),
    ('chunk/public/chunk/internal/proxy_base.h',
     r'''    ///        the proxy, or copy the value? The derived classes define what
    ///        `a[i] = b[j]` means.
    const Derived& operator=(const Derived& from) = delete;

    /// @brief Deleted: there is no stable address to take. The chunk may be
    ///        compressed, or reallocate under the proxy.
    T* operator&() = delete;

    /// @brief Deleted, as the non-const overload: no stable address to take.
    const T* operator&() const = delete;

    /// @brief Copying a proxy is fine: it rebinds to the same element.
''',
     r'''    ///        the proxy, or copy the value? The derived classes define what
    ///        `a[i] = b[j]` means.
    const Derived& operator=(const Derived& from) = delete("ambiguous: rebind or copy the value?");

    /// @brief Deleted: there is no stable address to take. The chunk may be
    ///        compressed, or reallocate under the proxy.
    T* operator&() = delete("no stable address: the chunk may be compressed or moved");

    /// @brief Deleted, as the non-const overload: no stable address to take.
    const T* operator&() const = delete("no stable address: the chunk may be compressed or moved");

    /// @brief Copying a proxy is fine: it rebinds to the same element.
'''),
    ('chunk/public/chunk/internal/zero_proxy.h',
     r'''    // No address, and do not add operator T&(): `tab[nc] = val` would route
    // through it and materialise blocks for nothing. Hence foo(T&) rejects a[i].
    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
''',
     r'''    // No address, and do not add operator T&(): `tab[nc] = val` would route
    // through it and materialise blocks for nothing. Hence foo(T&) rejects a[i].
    T*       operator&()       = delete("no stable address: the chunk may be compressed or moved");
    const T* operator&() const = delete("no stable address: the chunk may be compressed or moved");

    /**
'''),
    ('chunk/public/chunk/internal/zero_one_value_proxy.h',
     r'''    Proxy(const Proxy& other) = default;

    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
''',
     r'''    Proxy(const Proxy& other) = default;

    T*       operator&()       = delete("no stable address: the chunk may be compressed or moved");
    const T* operator&() const = delete("no stable address: the chunk may be compressed or moved");

    /**
'''),
    ('chunk/public/chunk/internal/zero_any_proxy.h',
     r'''     * @brief Address-of operator is disabled to prevent unsafe access.
     */
    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
''',
     r'''     * @brief Address-of operator is disabled to prevent unsafe access.
     */
    T*       operator&()       = delete("no stable address: the chunk may be compressed or moved");
    const T* operator&() const = delete("no stable address: the chunk may be compressed or moved");

    /**
'''),
    ('chunk/public/chunk/internal/array_chunk_base.h',
     r'''    using reverse_iterator       = const_reverse_iterator;                ///< Same as const_reverse_iterator.

    T* data() = delete;
    /**
     * @brief Set the runtime chunk geometry.
''',
     r'''    using reverse_iterator       = const_reverse_iterator;                ///< Same as const_reverse_iterator.

    T* data() = delete("elements are not contiguous: use read_range or for_each_span");
    /**
     * @brief Set the runtime chunk geometry.
'''),
]

# The earlier version (CHUNK_DELETED macro): convert it in place first.
MACRO_BLOCK = re.compile(r'\n/// @brief `= CHUNK_DELETED\("reason"\)`.*?\n#endif\n', re.S)
MACRO_FILES = sorted({h[0] for h in HUNKS}) + ["chunk/public/chunk/config.h"]
pre = {}
for p in MACRO_FILES:
    full = os.path.join(ROOT, p)
    if not os.path.exists(full):
        continue
    with open(full, encoding="utf-8") as f:
        s = f.read()
    t = MACRO_BLOCK.sub("", s) if p.endswith("config.h") else s.replace("CHUNK_DELETED(", "delete(")
    if t != s:
        pre[p] = t

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


SENTINEL = '= delete("'  # in every file once changed, in none before
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
        if p in pre:
            texts[p] = orig[p] = pre[p]
        else:
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

for p, s in pre.items():
    texts.setdefault(p, s)

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
