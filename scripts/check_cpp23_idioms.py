#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""cpp23 branch: what a merge of gen-dev-31 must not bring back.

gen-dev-31 is C++20 and is merged into cpp23 regularly. Its new code arrives
in the C++20 idioms the cpp23 branch has retired, and nothing failed when it
did: a std::cout << std::fixed in bench_stats.cpp put back the number-format
drift 14721603 had removed, and a merge committed an editor backup
(array_chunk_base_inline.h.sa) full of derived() calls. This check makes the
next merge say so.

Rules, on chunk/, fast_allocators/ and examples/ (.h, .hpp, .cpp), comment and
#include lines excepted:

1. No iostream output: std::cout, std::cerr, std::clog, std::endl, and the
   stream-state manipulators std::fixed / std::setprecision / std::setw.
   std::println / std::print instead (fprintf stays where nothing may
   allocate). Exceptions, as decided when the library moved to println
   (def53af7): ring_log_debug (a SEGV-hunt tool), the std::cout reports of
   param_chunk.h and tab_cst_val2_holder.h, and print_compression_stats'
   `std::ostream& os = std::cout` default argument.
2. No derived(): the CRTP helper is gone, members take `this Derived& self`.
3. static_cast<Derived&>(*this) only where a comment says why (the policies
   called as Base:: from sibling policies, whose *this is not a Derived):
   chunk_base.h and array_chunk_base.h.
4. No editor or merge leftovers tracked anywhere: *.sa, *.orig, *.rej, *.bak,
   *~.
5. No C stdout printing: printf / std::printf / vprintf. std::println instead;
   std::fprintf(stderr, ...) stays (the paths where nothing may allocate).
6. No sprintf / snprintf: std::format, or std::format_to_n into a buffer that
   must stay a fixed char array.
7. No std::ostringstream / std::stringstream to build text: std::format.
   Exceptions: the rule-1 files, and element_ref_validation.cpp, which checks
   that `os << a[i]` compiles.

Usage: scripts/check_cpp23_idioms.py [repo root]   (exit 1 on a violation)
"""

import pathlib
import re
import subprocess
import sys

DIRS = ("chunk", "fast_allocators", "examples")
SUFFIXES = (".h", ".hpp", ".cpp")

IOSTREAM = re.compile(r"\bstd::(cout|cerr|clog|endl|fixed|setprecision|setw)\b")
IOSTREAM_FILES = {
    "chunk/private/ring_log_debug.cpp",
    "chunk/public/chunk/ring_log_debug.h",
    "chunk/public/chunk/param_chunk.h",
    "chunk/public/chunk/internal/tab_cst_val2_holder.h",
}
IOSTREAM_LINE_OK = re.compile(r"std::ostream&\s*\w+\s*=\s*std::cout\b")

DERIVED_CALL = re.compile(r"\bderived\(\)")
DOWNCAST = re.compile(r"static_cast<\s*(const\s+)?Derived\s*&>\(\s*\*this\s*\)")
DOWNCAST_FILES = {
    "chunk/public/chunk/internal/chunk_base.h",
    "chunk/public/chunk/internal/array_chunk_base.h",
}

C_PRINTF = re.compile(r"(?<![\w:])(std::)?v?printf\s*\(")
C_SPRINTF = re.compile(r"(?<![\w:])(std::)?v?s(n)?printf\s*\(")
SSTREAM = re.compile(r"\bstd::(ostringstream|stringstream)\b")
SSTREAM_FILES = IOSTREAM_FILES | {"examples/validation/element_ref_validation.cpp"}

LEFTOVER = re.compile(r"(\.sa|\.orig|\.rej|\.bak|~)$")

COMMENT_OR_INCLUDE = re.compile(r"^\s*(//|\*|/\*|#\s*include\b)")


def tracked(root: pathlib.Path):
    out = subprocess.run(["git", "ls-files"], cwd=root, stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split()


def code_part(line: str) -> str:
    """The line without a trailing // comment (good enough: no rule pattern
    contains //, and a string literal holding one is not a concern here)."""
    return line.split("//", 1)[0]


def check(root: pathlib.Path):
    bad = []
    files = tracked(root)
    for f in files:
        if LEFTOVER.search(f):
            bad.append(f"{f}: editor/merge leftover tracked by git")
    for f in files:
        if not f.startswith(DIRS) or not f.endswith(SUFFIXES):
            continue
        try:
            lines = (root / f).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(lines, 1):
            if COMMENT_OR_INCLUDE.match(line):
                continue
            code = code_part(line)
            if (IOSTREAM.search(code) and f not in IOSTREAM_FILES and not IOSTREAM_LINE_OK.search(code)):
                bad.append(f"{f}:{n}: iostream output ({IOSTREAM.search(code).group(0)}): use std::println")
            if C_PRINTF.search(code):
                bad.append(f"{f}:{n}: C printf to stdout: use std::println")
            if C_SPRINTF.search(code):
                bad.append(f"{f}:{n}: snprintf / sprintf: use std::format (std::format_to_n for a fixed buffer)")
            if SSTREAM.search(code) and f not in SSTREAM_FILES:
                bad.append(f"{f}:{n}: {SSTREAM.search(code).group(0)}: use std::format")
            if DERIVED_CALL.search(code):
                bad.append(f"{f}:{n}: derived(): take `this Derived& self` instead")
            if DOWNCAST.search(code) and f not in DOWNCAST_FILES:
                bad.append(f"{f}:{n}: static_cast<Derived&>(*this): take `this Derived& self` instead")
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else ".").resolve()
    bad = check(root)
    if bad:
        print(f"check_cpp23_idioms: {len(bad)} problem(s)")
        for b in bad:
            print(f"  {b}")
        return 1
    print("check_cpp23_idioms: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
