#!/usr/bin/env python3
"""Every C++ file opens with the project's license header, and only that.

Over chunk/, fast_allocators/ and examples/ (third_party/ keeps its own):

1. Line 1 starts the Apache 2.0 block below, word for word, followed by a
   single blank line, with the years as
   `Copyright <first>-<last> CEA (www.cea.fr)` or `Copyright <year> CEA ...`
   (first < last <= the current year, and last no older than the file's last
   commit when git knows it).
2. That block appears once, and no IDE stamp (`// Created by <user> on <date>`)
   is left in the file.

--fix rewrites the files that break a rule: the block goes to line 1, its text
is the canonical one, the stamps and any other form of the license (truncated,
`//` comments, SPDX line) are removed. The first year is the oldest of the old
copyright and the stamp, else the year of the file's first commit; the last is
the current year. Whatever stood above the old license (`#pragma once`, an
include) moves just below the new one.

Usage: scripts/check_headers.py [repo root] [--list] [--fix]   (exit 1 on a violation)
"""

import datetime
import os
import pathlib
import re
import subprocess
import sys

DIRS = ("chunk", "fast_allocators", "examples")
SUFFIXES = (".h", ".hpp", ".cpp")
THIS_YEAR = datetime.date.today().year

LICENSE_BODY = """ *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""
HEADER = re.compile(r"/\*\n \* Copyright (\d{4})(?:-(\d{4}))? CEA \(www\.cea\.fr\)\n" + re.escape(LICENSE_BODY) + r"\n(?!\n)")

# The forms an old license can take, for --fix: a /* */ block (full or
# truncated), a run of // lines, a // copyright + SPDX pair.
OLD_BLOCK = re.compile(r"/\*\s*\n\s*\*\s*Copyright[^\n]*CEA[^\n]*\n(?:(?!\*/).)*?Apache(?:(?!\*/).)*?\*/[ \t]*\n", re.S)
OLD_LINES = re.compile(r"// Copyright[^\n]*CEA[^\n]*\n(?://[^\n]*\n)*?// limitations under the License\.\n")
OLD_SPDX = re.compile(r"// Copyright[^\n]*CEA[^\n]*\n// SPDX-License-Identifier: Apache-2\.0\n")
STAMP = re.compile(r"(?://[ \t]*\n)?// Created by \S+ on \d+/\d+/(\d+)\.?[ \t]*\n(?://[ \t]*\n)?")
YEAR = re.compile(r"Copyright (\d{4})")


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


def check(text: str, last_commit=None):
    """The rules `text` breaks, as short reasons (empty: conforming)."""
    bad = []
    m = HEADER.match(text)
    if not m:
        bad.append("does not open with the license header" if "Licensed under the Apache" in text
                   or YEAR.search(text[:4000]) else "no license header")
    else:
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        if not first <= last <= THIS_YEAR or (m.group(2) and first == last):
            bad.append(f"bad copyright years {m.group(1)}{'-' + m.group(2) if m.group(2) else ''}")
        elif last_commit and last < last_commit:
            bad.append(f"copyright ends in {last}, file last committed in {last_commit}")
    if text.count("Licensed under the Apache") > 1:
        bad.append("license present more than once")
    if STAMP.search(text):
        bad.append("IDE 'Created by' stamp")
    return bad


def cut(text: str, a: int, b: int):
    """text without text[a:b], at most one blank line left at the junction, none at the top."""
    left, right = text[:a], text[b:]
    nl = (len(left) - len(left.rstrip("\n"))) + (len(right) - len(right.lstrip("\n")))
    left, right = left.rstrip("\n"), right.lstrip("\n")
    return right if not left else left + "\n" * min(max(nl, 1), 2) + right


def commit_years(root: pathlib.Path):
    """{relative path: year of its last commit}, from one git log (empty outside git)."""
    try:
        out = subprocess.run(["git", "log", "--format=@%ad", "--date=format:%Y", "--name-only", "--", *DIRS],
                             cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    years, year = {}, None
    for line in out.split("\n"):
        if line.startswith("@"):
            year = int(line[1:])
        elif line:
            years.setdefault(line, year)  # newest first: the first seen is the last commit
    return years


def first_commit_year(path: pathlib.Path, root: pathlib.Path):
    try:
        out = subprocess.run(["git", "log", "--follow", "--format=%ad", "--date=format:%Y", "--", str(path)],
                             cwd=root, capture_output=True, text=True, check=True).stdout.split()
        return int(out[-1]) if out else THIS_YEAR
    except (OSError, subprocess.CalledProcessError):
        return THIS_YEAR


def fixed(text: str, path: pathlib.Path, root: pathlib.Path):
    years = []
    m = STAMP.search(text)
    if m:
        y = int(m.group(1))
        years.append(y + 2000 if y < 100 else y)
        text = cut(text, m.start(), m.end())
    while True:  # every old license, however many there are
        for rx in (OLD_BLOCK, OLD_LINES, OLD_SPDX):
            mm = rx.search(text[:4000])
            if mm:
                break
        else:
            break
        ym = YEAR.search(mm.group(0))
        if ym:
            years.append(int(ym.group(1)))
        text = cut(text, mm.start(), mm.end())
    first = min(years) if years else first_commit_year(path, root)
    span = str(THIS_YEAR) if first >= THIS_YEAR else f"{first}-{THIS_YEAR}"
    return f"/*\n * Copyright {span} CEA (www.cea.fr)\n{LICENSE_BODY}\n{text.lstrip(chr(10))}"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parent.parent)
    bad = []
    n_fixed = 0
    last_commit = commit_years(root)
    for f in files(root):
        text = f.read_text(encoding="utf-8")
        last = last_commit.get(f.relative_to(root).as_posix())
        reasons = check(text, last)
        if reasons and "--fix" in sys.argv:
            f.write_text(fixed(text, f, root), encoding="utf-8")
            n_fixed += 1
            reasons = check(f.read_text(encoding="utf-8"), last)
        bad += [f"{f.relative_to(root)}: {r}" for r in reasons]
    if n_fixed:
        print(f"check_headers: {n_fixed} file(s) rewritten")
    if bad:
        print(f"check_headers: {len(bad)} problem(s)")
        for b in bad if "--list" in sys.argv else bad[:40]:
            print("  " + b)
        if len(bad) > 40 and "--list" not in sys.argv:
            print(f"  ... ({len(bad) - 40} more, --list for all)")
        return 1
    print("check_headers: OK, every file opens with the license header and carries no IDE stamp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
