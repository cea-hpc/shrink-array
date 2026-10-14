#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every file the sources name is tracked by git.

A file written next to the others but never `git add`ed builds on the machine
that has it and nowhere else: a clean clone fails to configure, to compile or
to import. It happened four times (a validation harness, two documentation
pages, a C++ header pair, a Python module), each found on another machine.
What the tracked sources name:

1. a quoted `#include "x"` of chunk/, fast_allocators/ and examples/ (looked up
   next to the including file, then under chunk/public, fast_allocators/public
   and third_party);
2. a relative import of the Python packages under python/ (`from . import a`,
   `from .a import b`, `from ..a import b`);
3. a source path written in a CMakeLists.txt (relative to it, or after
   ${PROJECT_SOURCE_DIR} / ${CMAKE_SOURCE_DIR} / ${CMAKE_CURRENT_SOURCE_DIR}),
   and the validation harnesses, built as `<name>.cpp` from the names
   examples/validation/CMakeLists.txt lists;
4. a relative link of a Markdown file (`[text](other.md)`, `(../examples/x.cpp)`).

A named file that exists but is untracked is an error (`git add` it); one that
exists nowhere is an error too, unless a tracked `<name>.in` generates it (a
CMake *-config-version.cmake is generated; words inside a CMake string are not
paths).

Usage: scripts/check_tracked.py [repo root]   (exit 1 on a problem)
"""

import pathlib
import re
import subprocess
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
PY_REL_FROM = re.compile(r"^\s*from\s+(\.+)([\w.]*)\s+import\s+(.+?)\s*(?:#.*)?$")
CMAKE_PATH = re.compile(r'(?:\$\{(?:PROJECT_SOURCE_DIR|CMAKE_SOURCE_DIR|CMAKE_CURRENT_SOURCE_DIR)\}/)?'
                        r'[A-Za-z0-9_./-]+\.(?:h|hpp|cpp|c|py|cmake|in)\b')
MD_LINK = re.compile(r"\]\(([^)#\s:]+)(?:#[^)]*)?\)")
CMAKE_HARNESS = re.compile(r"^\s*(?:set\(_rt_harnesses|set\(_rt_dyn_harnesses|list\(APPEND _group)\s+([^)]*)\)?")
INCLUDE_ROOTS = ("chunk/public", "fast_allocators/public", "third_party")
CXX_DIRS = ("chunk", "fast_allocators", "examples")
CXX_SUFFIXES = (".h", ".hpp", ".cpp", ".in")


def git_files(root, *args):
    out = subprocess.run(["git", "ls-files", *args], cwd=root, capture_output=True, text=True, check=True).stdout
    return {f for f in out.split("\n") if f}


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    try:
        tracked = git_files(root, "--cached")
    except (OSError, subprocess.CalledProcessError):
        print("check_tracked: not a git checkout, nothing to check")
        return 0
    deleted = git_files(root, "--deleted")
    tracked -= deleted
    untracked = git_files(root, "--others", "--exclude-standard")
    problems = []

    def verdict(where, named, candidates):
        """The first candidate that exists decides; none: missing (unless generated)."""
        for c in candidates:
            if c in tracked:
                return
            if c in untracked:
                problems.append(f"{where}: {named} -> {c} exists but is not tracked (git add it)")
                return
        if any(c + ".in" in tracked for c in candidates):
            return
        problems.append(f"{where}: {named} not found (looked for {', '.join(candidates[:3])})")

    # 1. quoted includes
    for f in sorted(tracked):
        if not f.startswith(CXX_DIRS) or not f.endswith(CXX_SUFFIXES) or "/vendor/" in f:
            continue
        text = (root / f).read_text(encoding="utf-8", errors="replace").split("\n")
        here = str(pathlib.PurePosixPath(f).parent)
        for n, line in enumerate(text, 1):
            m = INCLUDE.match(line)
            if not m:
                continue
            inc = m.group(1)
            cands = [str(pathlib.PurePosixPath(here, inc))] + [f"{r}/{inc}" for r in INCLUDE_ROOTS]
            cands = [str(pathlib.PurePosixPath(*[p for p in pathlib.PurePosixPath(c).parts])) for c in cands]
            if any(c in tracked or c in untracked or c + ".in" in tracked for c in cands):
                verdict(f"{f}:{n}", f'#include "{inc}"', cands)
            elif inc.startswith(("chunk/", "fast_allocators/")) or "/" not in inc:
                # a project header, or a sibling: it must exist somewhere
                verdict(f"{f}:{n}", f'#include "{inc}"', cands)

    # 2. relative Python imports
    for f in sorted(tracked):
        if not f.startswith("python/") or not f.endswith(".py"):
            continue
        pkg = pathlib.PurePosixPath(f).parent
        for n, line in enumerate((root / f).read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            m = PY_REL_FROM.match(line)
            if not m:
                continue
            base = pkg
            for _ in range(len(m.group(1)) - 1):
                base = base.parent
            mod = m.group(2)
            if mod:
                p = base.joinpath(*mod.split("."))
                verdict(f"{f}:{n}", f"from {m.group(1)}{mod}", [f"{p}.py", f"{p}/__init__.py"])
            else:
                for name in m.group(3).strip("()").split(","):
                    name = name.strip().split(" as ")[0].strip()
                    if name and name != "*":
                        p = base / name
                        verdict(f"{f}:{n}", f"from {m.group(1)} import {name}", [f"{p}.py", f"{p}/__init__.py"])

    # 3. CMake source paths and the validation harnesses
    for f in sorted(t for t in tracked if t.endswith("CMakeLists.txt")):
        here = pathlib.PurePosixPath(f).parent
        for n, line in enumerate((root / f).read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            code = line.split("#", 1)[0]
            # Words in a string (an option's description, a message) name
            # nothing to build; *-config-version.cmake is CMake's output.
            code = re.sub(r'"[^"]*"', '""', code)
            for m in CMAKE_PATH.finditer(code):
                if m.group(0).endswith("-config-version.cmake"):
                    continue
                tok = m.group(0)
                if tok.startswith("${CMAKE_CURRENT_SOURCE_DIR}/"):
                    p = here / tok.split("}/", 1)[1]
                elif tok.startswith("${"):
                    p = pathlib.PurePosixPath(tok.split("}/", 1)[1])
                elif m.start() > 0 and code[m.start() - 1] in "}$/":
                    continue  # glued to a variable we do not expand
                else:
                    p = here / tok
                verdict(f"{f}:{n}", tok, [str(p)])
            if f == "examples/validation/CMakeLists.txt":
                h = CMAKE_HARNESS.match(code)
                if h:
                    for name in h.group(1).split():
                        if not name.startswith("$"):
                            verdict(f"{f}:{n}", f"harness {name}", [f"{here}/{name}.cpp"])

    # 4. relative Markdown links
    for f in sorted(t for t in tracked if t.endswith(".md")):
        here = pathlib.PurePosixPath(f).parent
        in_fence = False
        for n, line in enumerate((root / f).read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for m in MD_LINK.finditer(re.sub(r"`[^`]*`", "``", line)):  # not inside code
                target = m.group(1)
                if target.startswith(("http", "mailto", "/")) or "{" in target or not re.search(r"[./]", target):
                    continue
                p = pathlib.PurePosixPath(*(here / target).parts)
                parts = []
                for part in p.parts:
                    if part == "..":
                        parts.pop() if parts else None
                    elif part != ".":
                        parts.append(part)
                path = "/".join(parts)
                if (root / path).is_dir() or any(t.startswith(path.rstrip("/") + "/") for t in tracked):
                    continue
                verdict(f"{f}:{n}", f"link {target}", [path])

    for p in problems:
        print(p)
    print(f"check_tracked: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
