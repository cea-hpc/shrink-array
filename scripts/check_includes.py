#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every #include says why it is there, and system and project headers are not mixed.

Two rules, over chunk/, fast_allocators/ and examples/:

1. Every `#include` line carries a trailing `// <why>` comment: what the file
   uses from that header (`#include <vector> // std::vector`,
   `#include "chunk/config.h" // build_config::use_openmp`). An include whose
   reason is not written down cannot be told from a leftover.
2. A block of consecutive `#include` lines (no blank line, no preprocessor
   directive between them) holds either system headers (`<...>`) or project
   headers (`"..."`), not both: system headers first, a blank line, then the
   project's.

Usage: scripts/check_includes.py [repo root] [--list]   (exit 1 on a violation)
"""

import os
import pathlib
import re
import subprocess
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])')


DIRS = ("chunk", "fast_allocators", "examples")
SUFFIXES = (".h", ".hpp", ".cpp")


def files(root: pathlib.Path):
    """The C++ files of DIRS: git's list (tracked and new, ignored ones
    excluded) when there is one, else a walk that skips the build trees."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", *DIRS],
                             cwd=root, capture_output=True, text=True, check=True).stdout
        found = [root / f for f in out.split("\n") if f]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for d in DIRS:
            for top, dirs, names in os.walk(root / d):
                dirs[:] = [x for x in dirs if not x.startswith(".") and not x.startswith("cmake-build")
                           and x != "build"]
                found += [pathlib.Path(top) / n for n in names]
    return sorted(f for f in found if f.suffix in SUFFIXES and f.is_file())


def check(path: pathlib.Path, root: pathlib.Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    rel = path.relative_to(root)
    bad = []
    block = []  # (line number, kind) of the current run of consecutive includes
    def close():
        kinds = {k for _, k in block}
        if len(kinds) > 1:
            bad.append(f"{rel}:{block[0][0]}: block mixes system <...> and project \"...\" headers")
    for i, line in enumerate(lines, 1):
        m = INCLUDE.match(line)
        if m:
            block.append((i, m.group(1)))
            if "//" not in line:
                bad.append(f"{rel}:{i}: no reason given: {line.strip()}")
            continue
        if block:
            close()
            block = []
    if block:
        close()
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parent.parent)
    bad = []
    for f in files(root):
        bad += check(f, root)
    if bad:
        print(f"check_includes: {len(bad)} problem(s)")
        for b in bad if "--list" in sys.argv else bad[:40]:
            print("  " + b)
        if len(bad) > 40 and "--list" not in sys.argv:
            print(f"  ... ({len(bad) - 40} more, --list for all)")
        return 1
    print("check_includes: OK, every include has a reason and no block mixes system and project headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
