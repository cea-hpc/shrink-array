#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every file the sources name is tracked by git.

A file written next to the others but never `git add`ed builds on the machine
that has it and nowhere else: a clean clone fails to configure, to compile or
to import. It keeps happening (a validation harness, two documentation pages,
C++ headers, a Python module, GenMC models), each found on another machine.
What the tracked sources name:

1. a quoted `#include "x"` of chunk/, fast_allocators/ and examples/ (looked up
   next to the including file, then under chunk/public, fast_allocators/public
   and third_party);
2. an import of the Python tree under python/: relative (`from . import a`,
   `from .a import b`, `from ..a import b`), or absolute when its first name
   is a module or package of ours -- next to the importing file (a script's
   directory is on its path: `import numa_launch`, a test's `from conftest
   import x`) or under python/ (`from shrink_gui.config import x`, `import
   bench_runner.pool`). The standard library and the installed packages are
   not ours and are left alone; a name imported FROM a module is looked at
   only when a file of that name sits there untracked (it may be an attribute);
3. a source path written in a CMakeLists.txt (relative to it, or after
   ${PROJECT_SOURCE_DIR} / ${CMAKE_SOURCE_DIR} / ${CMAKE_CURRENT_SOURCE_DIR}),
   the validation harnesses, built as `<name>.cpp` from the names
   examples/validation/CMakeLists.txt lists, and the GenMC models, built as
   `<name>.c` from the names formal/CMakeLists.txt lists;
4. a relative link of a Markdown file (`[text](other.md)`, `(../examples/x.cpp)`).

5. the reverse, for the libraries' headers: every tracked header under
   chunk/public/ and fast_allocators/public/ is listed in its library's
   CMakeLists.txt (CHUNK_*_HEADERS, ALLOCATORS_*_HEADERS: the target_sources
   that put it in an IDE's project, and the statement of what is API and what
   is detail). The install copies the whole tree either way, so an unlisted
   header still ships; it is only outside every target.
6. a std-alloc build (CHUNK_STD_ALLOC) installs no fast_allocators, only the
   handful of its headers that chunk's headers include: chunk/CMakeLists.txt's
   `if (CHUNK_STD_ALLOC)` install lists every fast_allocators header reached,
   transitively, from chunk/public outside the `#if CHUNK_OPT_FAST_ALLOC` /
   `#if !CHUNK_OPT_STD_ALLOC` blocks. A header missing from that list breaks
   every consumer of such an install, and no CI job builds one.
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
PY_ABS_FROM = re.compile(r"^\s*from\s+([A-Za-z_]\w*(?:\.\w+)*)\s+import\s+(.+?)\s*(?:#.*)?$")
PY_IMPORT = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*(?:\s+as\s+\w+)?(?:\s*,\s*[A-Za-z_][\w.]*(?:\s+as\s+\w+)?)*)\s*(?:#.*)?$")
CMAKE_PATH = re.compile(r'(?:\$\{(?:PROJECT_SOURCE_DIR|CMAKE_SOURCE_DIR|CMAKE_CURRENT_SOURCE_DIR)\}/)?'
                        r'[A-Za-z0-9_./-]+\.(?:h|hpp|cpp|c|py|cmake|in)\b')
MD_LINK = re.compile(r"\]\(([^)#\s:]+)(?:#[^)]*)?\)")
CMAKE_HARNESS = re.compile(r"^\s*(?:set\(_rt_harnesses|set\(_rt_dyn_harnesses|list\(APPEND _group)\s+([^)]*)\)?")
CMAKE_MODELS = re.compile(r"^\s*set\(_formal_models\s+([^)]*)\)?")
# CMakeLists.txt that build a source from a name it lists: (list pattern, suffix)
CMAKE_NAME_LISTS = {
    "examples/validation/CMakeLists.txt": (CMAKE_HARNESS, ".cpp", "harness"),
    "formal/CMakeLists.txt": (CMAKE_MODELS, ".c", "model"),
}
# examples/common: the include directory of shrink_examples_common (examples/CMakeLists.txt).
INCLUDE_ROOTS = ("chunk/public", "fast_allocators/public", "third_party", "examples/common")
CXX_DIRS = ("chunk", "fast_allocators", "examples")
CXX_SUFFIXES = (".h", ".hpp", ".cpp", ".in")


FAST_ALLOC_ONLY = re.compile(r"(?<![!\w])CHUNK_OPT_FAST_ALLOC\b|!\s*CHUNK_OPT_STD_ALLOC\b")
STD_ALLOC_ONLY = re.compile(r"!\s*CHUNK_OPT_FAST_ALLOC\b|(?<![!\w])CHUNK_OPT_STD_ALLOC\b")
FA_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](fast_allocators/[^>"]+)[>"]')
STD_ALLOC_INSTALL = re.compile(r"^if \(CHUNK_STD_ALLOC\)\n(.*?)^endif\(\)", re.M | re.S)


def std_alloc_includes(text):
    """The fast_allocators headers a file includes in a std-alloc build: its
    `#include "fast_allocators/..."` lines, minus those inside a block that only
    a fast-alloc build compiles (`#if CHUNK_OPT_FAST_ALLOC`,
    `#if !CHUNK_OPT_STD_ALLOC`, or the `#else` of the opposite test). Other
    conditions count as taken: a superset is the safe side."""
    stack = []  # per open #if: is the current branch fast-alloc only?
    found = set()
    for line in text.split("\n"):
        d = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if d:
            kw, cond = d.group(1), d.group(2)
            if kw in ("if", "ifdef", "ifndef"):
                stack.append([bool(FAST_ALLOC_ONLY.search(cond)), bool(STD_ALLOC_ONLY.search(cond))])
            elif kw == "elif" and stack:
                stack[-1] = [bool(FAST_ALLOC_ONLY.search(cond)), False]
            elif kw == "else" and stack:
                stack[-1] = [stack[-1][1], False]
            elif kw == "endif" and stack:
                stack.pop()
            continue
        m = FA_INCLUDE.match(line)
        if m and not any(fast for fast, _ in stack):
            found.add(m.group(1))
    return found


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

    # 2. Python imports: relative, and absolute ones of our own modules
    known = tracked | untracked

    def ours(base, first):
        """Is @p first a module or package of ours in directory @p base?"""
        prefix = f"{base}/{first}"
        return f"{prefix}.py" in known or any(k.startswith(prefix + "/") for k in known)

    def absolute_import(where, here, module, names):
        """`import module` / `from module import names`: checked when the
        module's first name is ours, next to the importing file or under
        python/."""
        parts = module.split(".")
        for base in dict.fromkeys((str(here), "python")):
            if not ours(base, parts[0]):
                continue
            p = pathlib.PurePosixPath(base).joinpath(*parts)
            verdict(where, f"import of {module}", [f"{p}.py", f"{p}/__init__.py"])
            for name in names:
                for c in (f"{p}/{name}.py", f"{p}/{name}/__init__.py"):
                    if c in untracked:
                        problems.append(f"{where}: from {module} import {name} -> {c} exists but is not tracked "
                                        "(git add it)")
            return

    for f in sorted(tracked):
        if not f.startswith("python/") or not f.endswith(".py"):
            continue
        pkg = pathlib.PurePosixPath(f).parent
        for n, line in enumerate((root / f).read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            m_abs = PY_ABS_FROM.match(line)
            if m_abs:
                names = [x.strip().split(" as ")[0].strip() for x in m_abs.group(2).strip("()").split(",")]
                absolute_import(f"{f}:{n}", pkg, m_abs.group(1), [x for x in names if x and x != "*"])
                continue
            m_imp = PY_IMPORT.match(line)
            if m_imp:
                for mod in m_imp.group(1).split(","):
                    absolute_import(f"{f}:{n}", pkg, mod.strip().split(" as ")[0].strip(), [])
                continue
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

    # 3. CMake source paths, and the sources built from the names a CMakeLists.txt lists
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
            if f in CMAKE_NAME_LISTS:
                pattern, suffix, kind = CMAKE_NAME_LISTS[f]
                h = pattern.match(code)
                if h:
                    for name in h.group(1).split():
                        if not name.startswith("$"):
                            verdict(f"{f}:{n}", f"{kind} {name}", [f"{here}/{name}{suffix}"])

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

    # 5. every tracked library header is listed in its library's CMakeLists.txt
    for lib in ("chunk", "fast_allocators"):
        listed = set(re.findall(r"public/[A-Za-z0-9_/]+\.h\b", (root / lib / "CMakeLists.txt").read_text(encoding="utf-8")))
        for f in sorted(t for t in tracked if t.startswith(lib + "/public/") and t.endswith(".h")):
            if f[len(lib) + 1:] not in listed:
                problems.append(f"{f}: tracked, but in no header list of {lib}/CMakeLists.txt (add it to "
                                f"{'CHUNK' if lib == 'chunk' else 'ALLOCATORS'}_INTERNAL_HEADERS or _PUBLIC_HEADERS)")

    # 6. the std-alloc install ships every fast_allocators header chunk's headers reach
    cml = (root / "chunk" / "CMakeLists.txt").read_text(encoding="utf-8")
    block = STD_ALLOC_INSTALL.search(cml)
    if not block:
        problems.append("chunk/CMakeLists.txt: no `if (CHUNK_STD_ALLOC)` install block found (rule 6)")
    else:
        installed = set(re.findall(r"fast_allocators/public/(fast_allocators/[A-Za-z0-9_/]+\.h)\b", block.group(1)))
        todo = set()
        for f in sorted(t for t in tracked if t.startswith("chunk/public/") and t.endswith(".h")):
            todo |= std_alloc_includes((root / f).read_text(encoding="utf-8", errors="replace"))
        reached = set()
        while todo:
            h = todo.pop()
            if h in reached:
                continue
            reached.add(h)
            src = root / "fast_allocators" / "public" / h
            if src.is_file():
                todo |= std_alloc_includes(src.read_text(encoding="utf-8", errors="replace"))
        for h in sorted(reached - installed):
            problems.append(f"chunk/CMakeLists.txt: the CHUNK_STD_ALLOC install misses fast_allocators/public/{h}, "
                            f"which chunk's headers include in a std-alloc build")

    for p in problems:
        print(p)
    print(f"check_tracked: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
