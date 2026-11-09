#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The public interface is the one scripts/public_api.txt freezes.

Two checks, against the tree:

1. Headers: the [chunk/...] sections of public_api.txt are exactly the headers
   of chunk/public/chunk/ outside internal/. A header added there, moved under
   internal/ or removed fails until the list says so -- an interface change,
   which CHANGELOG.md must record.
2. Names: every name of a section is still reachable from its header: its last
   component (`Array` for zero::Array, `init_indexes` for
   ParamChunk::init_indexes) is a word of the header or of a header it
   includes from chunk/ or fast_allocators/, directly or not. A public name
   removed or renamed fails. The check does not compile anything: a name that
   moved to another header it includes still passes, which is what a program
   sees too.

The [not public] section lists the members of the public classes that name the
library's machinery; they must not appear in a header's section.

3. Examples: the programs under examples/ are what a user copies, so they
   include public headers only -- none under chunk/internal/ or
   fast_allocators/internal/. A program that needs one shows a name missing
   from the interface. INTERNAL_CLIENTS are the exceptions: the harnesses and
   the benchmarks, which look at the machinery on purpose.

Usage: scripts/check_public_api.py [repo root]   (exit 1 on a failure)
"""

import pathlib
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*"((?:chunk|fast_allocators)/[^"]+)"', re.M)

# The programs of examples/ allowed to include internal headers (check 3).
INTERNAL_CLIENTS = (
    "examples/validation/",                       # the harnesses: white-box tests
    "examples/shrink_bench/",                     # the benchmark: counters, pools, ring log
    "examples/access_cost/access_cost_pattern.cpp",  # the motif census walks the chunk list
    "examples/multi-mat/",                        # in_parallel_region, print_info_chunk: not public yet
)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INCLUDE_DIRS = ("chunk/public", "fast_allocators/public")


def parse(path):
    sections, current = {}, None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            current = m.group(1)
            if current in sections:
                raise SystemExit("%s:%d: section [%s] twice" % (path, n, current))
            sections[current] = []
            continue
        if current is None:
            raise SystemExit("%s:%d: a name before the first section" % (path, n))
        sections[current] += line.split()
    return sections


def resolve(root, include):
    for d in INCLUDE_DIRS:
        p = root / d / include
        if p.exists():
            return p
        if p.with_name(p.name + ".in").exists():   # configure_file: build_options.h
            return p.with_name(p.name + ".in")
    return None


def reachable_words(root, header, cache):
    """The words of `header` and of every chunk/ or fast_allocators/ header it
    includes, transitively."""
    seen, todo, words = set(), [header], set()
    while todo:
        inc = todo.pop()
        if inc in seen:
            continue
        seen.add(inc)
        p = resolve(root, inc)
        if p is None:
            continue
        if p not in cache:
            text = p.read_text(encoding="utf-8", errors="replace")
            cache[p] = (set(WORD.findall(text)), INCLUDE.findall(text))
        w, incs = cache[p]
        words |= w
        todo += incs
    return words


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parent.parent).resolve()
    sections = parse(root / "scripts" / "public_api.txt")
    not_public = set(sections.pop("not public", []))
    bad = []

    on_disk = {"chunk/" + p.name for p in (root / "chunk/public/chunk").glob("*.h")}
    listed = set(sections)
    for h in sorted(on_disk - listed):
        bad.append("%s: a public header (outside internal/) that public_api.txt does not list" % h)
    for h in sorted(listed - on_disk):
        bad.append("%s: listed in public_api.txt, but not a header of chunk/public/chunk/" % h)

    cache = {}
    for header in sorted(listed & on_disk):
        words = reachable_words(root, header, cache)
        for name in sections[header]:
            last = name.split("::")[-1]
            if name in not_public or last in not_public:
                bad.append("%s: %s is in [not public]" % (header, name))
            elif last not in words:
                bad.append("%s: %s is not declared there any more" % (header, name))

    examples = root / "examples"
    for path in sorted(examples.rglob("*")):
        if path.suffix not in (".cpp", ".h", ".hpp") or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(INTERNAL_CLIENTS):
            continue
        for inc in INCLUDE.findall(path.read_text(encoding="utf-8", errors="replace")):
            if "/internal/" in inc:
                bad.append("%s: includes %s, an internal header (examples use the public ones)" % (rel, inc))

    total = sum(len(v) for v in sections.values())
    if bad:
        print("check_public_api: %d problem(s)" % len(bad))
        for b in bad:
            print("  " + b)
        print("  (an interface change: update scripts/public_api.txt and say so in CHANGELOG.md)")
        return 1
    print("check_public_api: OK, %d public headers, %d names" % (len(listed), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
