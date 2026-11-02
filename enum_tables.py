#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
enum_tables.py -- the name tables a parser walks, checked against their enum
by C++26 reflection: CHUNK_LOG's channel and level names (log_core.h), the
CHUNK_RECHUNK_SCHEDULE kinds (a table, where an if-chain was).

Run from the root of the shrink-array tree (branch cpp26):  python3 enum_tables.py
Independent of the soa_*.py scripts.
Idempotent; writes nothing unless every hunk applies.

  fast_allocators/public/fast_allocators/internal/log_core.h   names_match, two static_asserts
  chunk/public/chunk/internal/enum_name.h                      names_match<E>(names, prefix, first)
  chunk/public/chunk/internal/array_chunk_rechunk_inline.h     sched_names, its check, the parser loop
  CHANGELOG.md                                                 Unreleased (cpp26)
"""
import os
import re
import sys

ROOT = os.getcwd()

HUNKS = [
    ('CHANGELOG.md',
     r'''  `CHUNK_STATS_JSON {"variant": "chunk2a", "alloc": ..., ...}`, which
  `python/bench_runner/parsing.py` reads first (the text blocks for the other
  builds; both give the same figures).

### Added (cpp23)

''',
     r'''  `CHUNK_STATS_JSON {"variant": "chunk2a", "alloc": ..., ...}`, which
  `python/bench_runner/parsing.py` reads first (the text blocks for the other
  builds; both give the same figures).
- The name tables that a parser walks are checked against their enum by
  reflection (a `static_assert` in the reflection builds; the tables stay for
  the others): `channel_names` and `level_names`, which `CHUNK_LOG` is parsed
  against, against `logging::Channel` and `Level` (`log_core.h`; a channel
  added without its name was an empty string, two names swapped a wrong
  channel, both silent); the `CHUNK_RECHUNK_SCHEDULE` kinds against
  `ChunkPmpSched`, the if-chain that parsed them now a loop over that table
  (`sched_names`, `array_chunk_rechunk_inline.h`). `enum_name.h` gains
  `names_match<E>(names, prefix, first)` for it.

### Added (cpp23)

'''),
    ('chunk/public/chunk/internal/enum_name.h',
     r''' *          - upper_enum_name(en, prefix): the name of the enumerator @p en,
 *            a string with static storage, usable at run time;
 *          - table_matches<E>(table, prefix): true when @p table gives, for
 *            every enumerator of E, that same name.
 *
 *          Only where the compiler has reflection (GCC 16, -freflection):
 *          SHRINK_ARRAY_HAS_ENUM_NAME says so. A caller keeps its switch for
''',
     r''' *          - upper_enum_name(en, prefix): the name of the enumerator @p en,
 *            a string with static storage, usable at run time;
 *          - table_matches<E>(table, prefix): true when @p table gives, for
 *            every enumerator of E, that same name;
 *          - names_match<E>(names, prefix, first): true when @p names, an
 *            array of strings, lists E's enumerators in order, as spelled
 *            (lower case, without @p prefix), enumerator k of value
 *            first + k -- for a table that a parser walks (string to enum).
 *
 *          Only where the compiler has reflection (GCC 16, -freflection):
 *          SHRINK_ARRAY_HAS_ENUM_NAME says so. A caller keeps its switch for
'''),
    ('chunk/public/chunk/internal/enum_name.h',
     r'''
#if SHRINK_ARRAY_HAS_ENUM_NAME

    #include <string>      // std::string (consteval only: the name being built)
    #include <string_view> // std::string_view

''',
     r'''
#if SHRINK_ARRAY_HAS_ENUM_NAME

    #include <cstddef>     // std::size_t
    #include <string>      // std::string (consteval only: the name being built)
    #include <string_view> // std::string_view

'''),
    ('chunk/public/chunk/internal/enum_name.h',
     r'''    return true;
}

} // namespace shrink_array

#endif // SHRINK_ARRAY_HAS_ENUM_NAME
''',
     r'''    return true;
}

/**
 * @brief True when @p names lists the enumerators of E, in declaration
 *        order: names[k] is the identifier of enumerator k without the
 *        leading @p prefix, and its value is @p first + k; E has exactly
 *        names.size() enumerators.
 */
template <typename E, typename Names>
consteval bool names_match(const Names& names, std::string_view prefix, long long first = 0)
{
    const auto ens = std::meta::enumerators_of(^^E);
    if (ens.size() != names.size())
    {
        return false;
    }
    for (std::size_t k = 0; k < ens.size(); ++k)
    {
        std::string_view id = std::meta::identifier_of(ens[k]);
        if (id.starts_with(prefix))
        {
            id.remove_prefix(prefix.size());
        }
        if (id != std::string_view(names[k]) ||
            static_cast<long long>(std::meta::extract<E>(ens[k])) != first + static_cast<long long>(k))
        {
            return false;
        }
    }
    return true;
}

} // namespace shrink_array

#endif // SHRINK_ARRAY_HAS_ENUM_NAME
'''),
    ('chunk/public/chunk/internal/array_chunk_rechunk_inline.h',
     r'''#include <cstddef>     // std::size_t
#include <format>      // std::format_to_n (CHUNK_RECHUNK_SCHEDULE text)
#include <cstring>     // std::memcpy / std::memcmp (refill bulk read)
#include <format>      // std::format_to_n (CHUNK_RECHUNK_SCHEDULE text)
#include <cstdlib>     // std::getenv (CHUNK_RECHUNK_RECYCLE, CHUNK_RECHUNK_SCHEDULE)
#include <optional>    // the affinity scope around tmp
#include <type_traits> // std::is_same_v, the zero bool guard
''',
     r'''#include <cstddef>     // std::size_t
#include <format>      // std::format_to_n (CHUNK_RECHUNK_SCHEDULE text)
#include <cstring>     // std::memcpy / std::memcmp (refill bulk read)
#include <cstdlib>     // std::getenv (CHUNK_RECHUNK_RECYCLE, CHUNK_RECHUNK_SCHEDULE)
#include <optional>    // the affinity scope around tmp
#include <type_traits> // std::is_same_v, the zero bool guard
'''),
    ('chunk/public/chunk/internal/array_chunk_rechunk_inline.h',
     r'''#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size / set_active_chunk_size
#include "chunk/bit.h"                          // is_bool_like_v (refill bulk read gate)
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED (refill bulk read check)

namespace shrink_array::internal
{
''',
     r'''#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size / set_active_chunk_size
#include "chunk/bit.h"                          // is_bool_like_v (refill bulk read gate)
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED (refill bulk read check)
#include "chunk/internal/enum_name.h"           // names_match (sched_names, reflection builds)

namespace shrink_array::internal
{
'''),
    ('chunk/public/chunk/internal/array_chunk_rechunk_inline.h',
     r'''    std::array<char, 32> text{"guided,16"};
};

/// @brief The rebuild loop's schedule, parsed once from CHUNK_RECHUNK_SCHEDULE ("kind[,chunk]").
/// @return The parsed schedule, or the default guided,16 when the variable is unset or not understood.
inline const RebuildSchedule& rebuild_schedule() noexcept
''',
     r'''    std::array<char, 32> text{"guided,16"};
};

/// @brief The kinds CHUNK_RECHUNK_SCHEDULE names, in ChunkPmpSched's order
///        (sched_static = 1 first): the enumerators without "sched_".
inline constexpr std::array<std::string_view, 4> sched_names{"static", "dynamic", "guided", "auto"};
#if SHRINK_ARRAY_HAS_ENUM_NAME
static_assert(names_match<ChunkPmpSched>(sched_names, "sched_", 1),
              "sched_names must list ChunkPmpSched's enumerators, in order, without \"sched_\"");
#endif

/// @brief The rebuild loop's schedule, parsed once from CHUNK_RECHUNK_SCHEDULE ("kind[,chunk]").
/// @return The parsed schedule, or the default guided,16 when the variable is unset or not understood.
inline const RebuildSchedule& rebuild_schedule() noexcept
'''),
    ('chunk/public/chunk/internal/array_chunk_rechunk_inline.h',
     r'''        }
        if (ok)
        {
            if (kv == "static")
            {
                k = ChunkPmpSched::sched_static;
            }
            else if (kv == "dynamic")
            {
                k = ChunkPmpSched::sched_dynamic;
            }
            else if (kv == "guided")
            {
                k = ChunkPmpSched::sched_guided;
            }
            else if (kv == "auto")
            {
                k = ChunkPmpSched::sched_auto;
            }
            else
            {
                ok = false;
            }
        }
        if (!ok)
''',
     r'''        }
        if (ok)
        {
            ok = false;
            for (std::size_t i = 0; i < sched_names.size() && !ok; ++i)
            {
                ok = kv == sched_names[i];
                k  = ok ? static_cast<ChunkPmpSched>(i + 1) : k;
            }
        }
        if (!ok)
'''),
    ('fast_allocators/public/fast_allocators/internal/log_core.h',
     r''' * - Not here: the fatal level. An invariant failure is CHUNK_INVARIANT_FAILED
 *   (chunk/internal/invariant_check.h), which no setting can turn off.
 */

#include <array>           // std::array (the thresholds, the names)
#include <atomic>          // std::atomic
''',
     r''' * - Not here: the fatal level. An invariant failure is CHUNK_INVARIANT_FAILED
 *   (chunk/internal/invariant_check.h), which no setting can turn off.
 */

// __cpp_lib_reflection comes with <meta>: included first where the language
// has reflection (the name tables are then checked against the enums).
#if defined(__cpp_impl_reflection)
    #include <meta> // std::meta (the names_match check)
#endif
#if defined(__cpp_impl_reflection) && defined(__cpp_lib_reflection)
    #define FAST_ALLOC_LOG_HAS_REFLECTION 1
#else
    #define FAST_ALLOC_LOG_HAS_REFLECTION 0
#endif

#include <array>           // std::array (the thresholds, the names)
#include <atomic>          // std::atomic
'''),
    ('fast_allocators/public/fast_allocators/internal/log_core.h',
     r'''    "compress", "recycle", "ledger", "ompt",  "affinity", "advisor", "memory",
};
inline constexpr std::array<std::string_view, n_levels> level_names{"fatal", "error", "warn", "info", "debug", "trace"};

/// @brief The name CHUNK_LOG spells channel @p c with.
[[nodiscard]] constexpr std::string_view name(Channel c) noexcept
''',
     r'''    "compress", "recycle", "ledger", "ompt",  "affinity", "advisor", "memory",
};
inline constexpr std::array<std::string_view, n_levels> level_names{"fatal", "error", "warn", "info", "debug", "trace"};

#if FAST_ALLOC_LOG_HAS_REFLECTION
/// @brief True when @p names spells E's enumerators, in order -- and E has
///        no other but, when @p with_count, a last one (count_). C++26
///        reflection: the tables above, which CHUNK_LOG is parsed against,
///        cannot drift from the enums (a forgotten name was an empty string,
///        a swapped one a wrong channel, both silent).
template <typename E, std::size_t N>
consteval bool names_match(const std::array<std::string_view, N>& names, bool with_count)
{
    const auto ens = std::meta::enumerators_of(^^E);
    if (ens.size() != N + (with_count ? 1 : 0))
    {
        return false;
    }
    for (std::size_t k = 0; k < N; ++k)
    {
        if (std::meta::identifier_of(ens[k]) != names[k] || std::meta::extract<E>(ens[k]) != static_cast<E>(k))
        {
            return false;
        }
    }
    return true;
}
static_assert(names_match<Channel>(channel_names, true), "channel_names must spell Channel's enumerators, in order");
static_assert(names_match<Level>(level_names, false), "level_names must spell Level's enumerators, in order");
#endif

/// @brief The name CHUNK_LOG spells channel @p c with.
[[nodiscard]] constexpr std::string_view name(Channel c) noexcept
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


SENTINEL = "names_match"  # in every file once changed, in none before
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
