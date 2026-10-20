#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Constants are snake_case; UPPER_CASE is for macros.

Over chunk/, fast_allocators/ and examples/, the name a `constexpr` variable
declares (namespace scope, static member or local) is snake_case, as the
standard library's are (`std::dynamic_extent`, `std::is_same_v`):
`cst_buffer_size`, not `kCstBufferSize` nor `CST_BUFFER_SIZE`. So is a `const`
object named in the k-prefixed style (`const Case kCases[]`). An UPPER_CASE
name reads as a macro (CHUNK_SIZE, CHUNK_DYN_MAX_SIZE are macros) and may meet
one; a k-prefix is another project's convention.

A line with `[[deprecated` is left alone: the former name of a constant, kept
as an alias for the programs that still spell it (fast_alloc::SIZE_BLOCK).

Usage: scripts/check_naming.py [repo root] [--list]   (exit 1 on a violation)
"""

import os
import pathlib
import re
import subprocess
import sys

DIRS = ("chunk", "fast_allocators", "examples")
SUFFIXES = (".h", ".hpp", ".cpp")

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
# The name a constexpr declaration introduces: the last identifier before its
# initialiser (=, {) or array bound ([), with no parenthesis on the way (a
# constexpr function or constructor declares no variable).
CONSTEXPR_VAR = re.compile(r"\bconstexpr\b(?!\s*\()([^=;{(\[]*?)\b([A-Za-z_]\w*)\s*(=|\{|\[)")
K_CONST = re.compile(r"\bconst\b[^=;(]*?\b(k[A-Z]\w*)\s*(=|\{|\[)")
COMMENT = re.compile(r"//.*$")


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
    return sorted(f for f in found if f.suffix in SUFFIXES and f.is_file() and "/vendor/" not in str(f))


def check(path: pathlib.Path, root: pathlib.Path):
    rel = path.relative_to(root)
    bad = []
    in_block_comment = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line
        if in_block_comment:
            if "*/" not in code:
                continue
            code = code.split("*/", 1)[1]
            in_block_comment = False
        code = re.sub(r"/\*.*?\*/", "", code)
        if "/*" in code:
            code, in_block_comment = code.split("/*", 1)[0], True
        code = COMMENT.sub("", code)
        if "[[deprecated" in code or "if constexpr" in code:
            continue
        names = [m.group(2) for m in CONSTEXPR_VAR.finditer(code)]
        names += [m.group(1) for m in K_CONST.finditer(code)]
        for name in names:
            if not SNAKE.match(name):
                bad.append(f"{rel}:{i}: constant `{name}` is not snake_case")
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parent.parent)
    bad = []
    for f in files(root):
        bad += check(f, root)
    if bad:
        print(f"check_naming: {len(bad)} problem(s)")
        for b in bad if "--list" in sys.argv else bad[:40]:
            print("  " + b)
        if len(bad) > 40 and "--list" not in sys.argv:
            print(f"  ... ({len(bad) - 40} more, --list for all)")
        return 1
    print("check_naming: OK, every constant is snake_case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
