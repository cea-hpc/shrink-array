#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The naming convention of docs/contributing.md section 3, as far as a line
can show it.

Over chunk/, fast_allocators/ and examples/:

- Constants are snake_case; UPPER_CASE is for macros. The name a `constexpr`
  variable declares (namespace scope, static member or local) is snake_case,
  as the standard library's are (`std::dynamic_extent`, `std::is_same_v`):
  `cst_buffer_size`, not `kCstBufferSize` nor `CST_BUFFER_SIZE`. So is a
  `const` object named in the k-prefixed style (`const Case kCases[]`), and
  so is an enumerator (`ev_flush_begin`, not `EV_FLUSH_BEGIN`). An UPPER_CASE
  name reads as a macro (CHUNK_SIZE, CHUNK_DYN_MAX_SIZE are macros) and may
  meet one; a k-prefix is another project's convention.
- Functions, methods, variables and members are snake_case: no camelCase
  name (`getInstance`), and no capital inside a snake_case one
  (`add_TCTypeAllocator_inst`, `chunk2A_...`: the variant is `chunk2a` there).
- Types are PascalCase: a class or struct definition, a type alias, a
  template type parameter (`ChunkT`, not `CHUNK_T`). Standard-library shapes
  keep their spelling: the member types of the std protocols (value_type,
  iterator, ...) and any `xxx_type`, the traits and tags (is_..., has_...,
  ..._t, ..._tag, rebind).
- No identifier of the implementation's reserved `__` form, but the
  compiler's and the runtime's (__builtin_..., __cxa_..., __kmp_...).

What a line cannot show -- the `_` that ends a private or protected data
member, a parameter's case -- clang-tidy's readability-identifier-naming
checks (docs/contributing.md section 3).

A line with `[[deprecated` is left alone: the former name, kept as an alias
for the programs that still spell it (fast_alloc::SIZE_BLOCK, getChunkList).

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
PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
# The name a constexpr declaration introduces: the last identifier before its
# initialiser (=, {) or array bound ([), with no parenthesis on the way (a
# constexpr function or constructor declares no variable).
CONSTEXPR_VAR = re.compile(r"\bconstexpr\b(?!\s*\()([^=;{(\[]*?)\b([A-Za-z_]\w*)\s*(=|\{|\[)")
K_CONST = re.compile(r"\bconst\b[^=;(]*?\b(k[A-Z]\w*)\s*(=|\{|\[)")
COMMENT = re.compile(r"//.*$")
LITERAL = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])+'")

CAMEL = re.compile(r"(?<![\w:.>])([a-z][a-z0-9]*[A-Z][A-Za-z0-9]*)\b")
SNAKE_WITH_CAPITAL = re.compile(r"(?<![\w:.>])([a-z][a-z0-9]*(?:_[a-z0-9]+)*_?[a-z0-9]*[A-Z]\w*)\b")
ALIAS = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=")
TYPE_DEF = re.compile(r"\b(?:class|struct|union)\s+([A-Za-z_]\w*)\s*(?:final\b\s*)?(?=\{|:(?!:)|$)")
TEMPLATE_PARAM = re.compile(r"\b(?:typename|class)\s+([A-Z][A-Z_]*[A-Z_][A-Z0-9_]*)\b")  # T, T2, A0 pass
RESERVED = re.compile(r"(?<![\w])(__[a-z]\w*)")
RESERVED_OK = re.compile(r"__(cpp_|has_|attribute__|builtin_|int128|asm__|volatile__|cxa_|sanitizer_|alloc_token_|"
                         r"kmp_|lsan_|asan_|tsan_|msan_|ubsan_|forceinline|declspec|restrict|typeof|extension__)")
ENUM_HEAD = re.compile(r"\benum\b(?:\s+(?:class|struct))?(?:\s+\w+)?\s*(?::\s*[\w:]+\s*)?(\{|$)")
ENUMERATOR = re.compile(r"(?:^|[{,])\s*([A-Za-z_]\w*)\s*(?==|,|\}|$)")

# The std protocols' member types, spelt as the standard spells them.
STD_TYPE_NAMES = {
    "value_type", "size_type", "difference_type", "reference", "const_reference", "pointer",
    "const_pointer", "iterator", "const_iterator", "reverse_iterator", "const_reverse_iterator",
    "iterator_category", "iterator_concept", "element_type", "allocator_type", "key_type",
    "mapped_type", "other", "type", "is_always_equal", "is_transparent", "rebind", "hash",
    "formatter", "get_base", "modulo",
}


def std_shaped(name):
    """A type name the standard library's style allows: a protocol member type,
    an `xxx_type`, a trait or a tag."""
    return (name in STD_TYPE_NAMES or name.endswith(("_type", "_tag", "_t"))
            or name.startswith(("is_", "has_")) or name.startswith("propagate_on_"))


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


def code_lines(path: pathlib.Path):
    """(line number, the code of the line): comments removed, string and
    character literals emptied, preprocessor lines skipped."""
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
        code = COMMENT.sub("", LITERAL.sub('""', code))
        if code.lstrip().startswith("#"):
            continue
        yield i, code


def check(path: pathlib.Path, root: pathlib.Path):
    rel = path.relative_to(root)
    bad = []
    in_enum = False
    enum_pending = False
    after_deprecated = False
    for i, code in code_lines(path):
        # The attribute may stand on the line before the declaration it marks.
        if "[[deprecated" in code or after_deprecated:
            after_deprecated = "[[deprecated" in code and not code.rstrip().endswith((";", "}", "{"))
            continue
        where = f"{rel}:{i}"

        if "if constexpr" not in code:
            names = [m.group(2) for m in CONSTEXPR_VAR.finditer(code)]
            names += [m.group(1) for m in K_CONST.finditer(code)]
            for name in names:
                if not SNAKE.match(name):
                    bad.append(f"{where}: constant `{name}` is not snake_case")

        # Enumerators: from the enum's `{` (on its line or the next) to its `}`.
        body = None
        head = ENUM_HEAD.search(code)
        if head and not in_enum:
            if head.group(1) == "{":
                in_enum, body = True, code[head.end():]
            else:
                enum_pending = not code.rstrip().endswith(";")
        elif enum_pending and code.strip().startswith("{"):
            enum_pending, in_enum, body = False, True, code.split("{", 1)[1]
        elif in_enum:
            body = code
        if body is not None:
            text = body.split("}", 1)[0]
            for m in ENUMERATOR.finditer(text):
                name = m.group(1)
                if not SNAKE.match(name.rstrip("_") or "_"):
                    bad.append(f"{where}: enumerator `{name}` is not snake_case")
            if "}" in body:
                in_enum = False

        for m in CAMEL.finditer(code):
            bad.append(f"{where}: `{m.group(1)}` is camelCase, not snake_case")
        for m in SNAKE_WITH_CAPITAL.finditer(code):
            if not CAMEL.fullmatch(m.group(1)):
                bad.append(f"{where}: `{m.group(1)}` has a capital inside a snake_case name")

        for m in ALIAS.finditer(code):
            name = m.group(1)
            if not PASCAL.match(name) and not std_shaped(name):
                bad.append(f"{where}: type alias `{name}` is not PascalCase")
        if "template" not in code and "friend" not in code:
            for m in TYPE_DEF.finditer(code):
                name = m.group(1)
                if not PASCAL.match(name) and not std_shaped(name):
                    bad.append(f"{where}: type `{name}` is not PascalCase")
        if "template" in code:
            for m in TEMPLATE_PARAM.finditer(code):
                bad.append(f"{where}: template parameter `{m.group(1)}` is UPPER_CASE, not PascalCase")

        for m in RESERVED.finditer(code):
            if not RESERVED_OK.match(m.group(1)):
                bad.append(f"{where}: `{m.group(1)}` is a reserved identifier (leading __)")
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
    print("check_naming: OK, the names follow the convention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
