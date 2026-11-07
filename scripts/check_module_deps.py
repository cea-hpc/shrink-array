#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The module unit's global module fragment includes every external header the
library's headers include.

chunk/modules/shrink_array.cppm reads the library's own headers in its purview
(chunk/internal/module_export.h). Their external headers -- standard, system,
OpenMP, third party -- must be read before, in the global module fragment: an
external header first included in the purview would be the module's (clang
says so, GCC does not). So every `#include <x>` and every quoted include of a
vendored header (chunk/vendor/...) in chunk/public and fast_allocators/public
must appear in the fragment; one the fragment includes and no header does is
reported too (a stale line). The conditions around them are not compared: the
fragment repeats them by hand.

Usage: scripts/check_module_deps.py [repo root]   (exit 1 on a problem)
"""

import pathlib
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*(<[^>]+>|"chunk/vendor/[^"]+")')
HEADER_DIRS = ("chunk/public", "fast_allocators/public")
MODULE_UNIT = "chunk/modules/shrink_array.cppm"
# Included by a header, deliberately absent from the fragment.
EXEMPT = {
    "<intrin.h>": "MSVC only (chunk/internal/cpu_pause.h); the module builds with GCC and clang",
    "<oneapi/tbb/blocked_range.h>": "chunk/tbb_adaptor.h, which the module does not read",
    "<oneapi/tbb/parallel_for.h>": "chunk/tbb_adaptor.h, which the module does not read",
}


def includes(path, text):
    """The external #includes of @p text, with their line numbers."""
    for number, line in enumerate(text.splitlines(), 1):
        match = INCLUDE.match(line)
        if match:
            yield match.group(1), number


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    unit = (root / MODULE_UNIT).read_text()
    fragment = unit[unit.index("module;") : unit.index("export module")]
    in_fragment = {name for name, _ in includes(MODULE_UNIT, fragment)}

    used = {}
    for directory in HEADER_DIRS:
        for header in sorted((root / directory).rglob("*.h")):
            rel = header.relative_to(root)
            for name, number in includes(rel, header.read_text()):
                used.setdefault(name, f"{rel}:{number}")

    problems = []
    for name, where in sorted(used.items()):
        if name not in in_fragment and name not in EXEMPT:
            problems.append(f"{where}: {name} is not in {MODULE_UNIT}'s global module fragment")
    for name in sorted(in_fragment - used.keys()):
        problems.append(f"{MODULE_UNIT}: {name} is in the global module fragment, but no header includes it")

    if problems:
        print(f"check_module_deps: {len(problems)} problem(s)")
        for problem in problems:
            print("  " + problem)
        return 1
    print(f"check_module_deps: OK ({len(in_fragment)} external headers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
