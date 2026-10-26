#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every C++ source is as clang-format leaves it (.clang-format).

The branches were formatted whole with clang-format 23 (2026-10-18), and a
file is formatted whole when it is touched; nothing kept it so: an include
added with its comment out of column, a member renamed in an aligned block.
This runs clang-format on every tracked .h, .hpp and .cpp of chunk/,
fast_allocators/ and examples/ and lists the files it would change. Not
formal/: the GenMC models are C, with their own comment columns.

Each file goes through stdin with --assume-filename: clang-format guesses
the language of a .h, and a header with C++26 reflection (`^^T`) reads to it
as Objective-C, which .clang-format (Language: Cpp) then refuses -- the file
would be skipped with a mere message. A .h is named .hpp for the guess.

The output depends on the clang-format version: 23 is required (CLANG_FORMAT
names another binary).

Usage: scripts/check_format.py [repo root] [--fix]   (exit 1 on a problem)
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CXX_DIRS = ("chunk", "fast_allocators", "examples")
CXX_SUFFIXES = (".h", ".hpp", ".cpp")
REQUIRED_MAJOR = 23


def clang_format_binary():
    for name in (os.environ.get("CLANG_FORMAT"), f"clang-format-{REQUIRED_MAJOR}", "clang-format"):
        if name and shutil.which(name):
            out = subprocess.run([name, "--version"], capture_output=True, text=True, check=True).stdout
            version = re.search(r"version (\d+)\.", out)
            if version and int(version.group(1)) == REQUIRED_MAJOR:
                return name
    return None


def formatted(binary, root, rel):
    path = root / rel
    assumed = path.with_name(path.name + ".hpp") if path.suffix == ".h" else path
    text = path.read_bytes()
    out = subprocess.run([binary, f"--assume-filename={assumed}"], input=text, capture_output=True, check=False)
    if out.returncode != 0:
        return rel, None, out.stderr.decode(errors="replace").strip()
    return rel, out.stdout, text


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fix = "--fix" in sys.argv[1:]
    root = pathlib.Path(args[0] if args else ".").resolve()
    binary = clang_format_binary()
    if binary is None:
        print(f"check_format: clang-format {REQUIRED_MAJOR} not found (CLANG_FORMAT, clang-format-{REQUIRED_MAJOR}, "
              "clang-format)")
        return 1
    listed = subprocess.run(["git", "ls-files", *CXX_DIRS], cwd=root, capture_output=True, text=True,
                            check=True).stdout.split("\n")
    files = sorted(f for f in listed if f.endswith(CXX_SUFFIXES))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(lambda rel: formatted(binary, root, rel), files))
    problems = 0
    for rel, new, old in results:
        if new is None:
            print(f"{rel}: clang-format failed: {old}")
            problems += 1
        elif new != old:
            problems += 1
            if fix:
                (root / rel).write_bytes(new)
                print(f"{rel}: reformatted")
            else:
                print(f"{rel}: not formatted (scripts/check_format.py --fix, or clang-format -i)")
    print(f"check_format: {len(files)} files, {problems} problem(s) ({binary})")
    return 1 if problems and not fix else 0


if __name__ == "__main__":
    sys.exit(main())
