#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every source file carries its license: the full Apache block for C++,
SPDX lines for the scripts and build files, the upstream license for the
vendored headers.

C++ files of chunk/, fast_allocators/ and examples/:

1. Line 1 starts the Apache 2.0 block below, word for word, followed by a
   single blank line, with the years as
   `Copyright <first>-<last> CEA (www.cea.fr)` or `Copyright <year> CEA ...`
   (first < last <= the current year, and last no older than the file's last
   commit when git knows it).
2. That block appears once, and no IDE stamp (`// Created by <user> on <date>`)
   is left in the file.

Scripts and build files (*.py, *.sh, *.cmake, *.cmake.in, CMakeLists.txt,
outside third_party/):

3. Right after the `#!` line and a `coding:` line, if any, the two lines
       
       # SPDX-License-Identifier: Apache-2.0
   with the years as in rule 1, and no other CEA copyright in the file.

Vendored headers (third_party/chunk/vendor/, installed with chunk):

4. Line 1 is `// SPDX-License-Identifier: <id>`, <id> the upstream license
   their provenance comment names (MIT, BSL-1.0) -- never Apache-2.0.

--fix rewrites the files that break a rule: the block goes to line 1, its text
is the canonical one, the stamps and any other form of the license (truncated,
`//` comments, SPDX line) are removed. The first year is the oldest of the old
copyright and the stamp, else the year of the file's first commit; the last is
the current year. Whatever stood above the old license (`#pragma once`, an
include) moves just below the new one. For the scripts, the two lines replace
any older CEA license comment; for the vendored headers, the SPDX line goes on
top.

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

SCRIPT_NAMES = ("CMakeLists.txt",)
SCRIPT_SUFFIXES = (".py", ".sh", ".cmake")
SCRIPT_PROLOGUE = re.compile(r"(#![^\n]*\n)?(#[^\n]*coding[:=][^\n]*\n)?")
SCRIPT_HEADER = re.compile(r"# Copyright (\d{4})(?:-(\d{4}))? CEA \(www\.cea\.fr\)\n"
                           r"# SPDX-License-Identifier: Apache-2\.0\n")
# Older CEA license comments in a script, for --fix: the full Apache text as
# # lines, or a copyright line on its own.
SCRIPT_OLD = re.compile(r"^# Copyright[^\n]*CEA[^\n]*\n(?:#[^\n]*\n)*?# limitations under the License\.\n"
                        r"|^# Copyright[^\n]*CEA[^\n]*\n(?:# SPDX-License-Identifier:[^\n]*\n)?", re.M)
# A CEA license comment line, anywhere in a script (rule 3: only the header's).
SCRIPT_LICENSE_LINE = re.compile(r"^# (?:Copyright[^\n]*CEA|Licensed under the Apache)", re.M)

VENDOR_DIR = "third_party/chunk/vendor"
# The provenance comment's "license :" line -> its SPDX identifier.
VENDOR_LICENSES = {"MIT": "MIT", "Boost Software License 1.0": "BSL-1.0"}
VENDOR_LICENSE_LINE = re.compile(r"^//\s+license\s*:\s*(.+?)\s*(?:\(|$)", re.M)


def all_files(root: pathlib.Path):
    """Every file of the repository: git's list (tracked and new, ignored ones
    excluded) when there is one, else a walk that skips the build trees."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=root, capture_output=True, text=True, check=True).stdout
        found = [root / f for f in out.split("\n") if f]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for top, dirs, names in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".") and not x.startswith("cmake-build")
                       and x != "build"]
            found += [pathlib.Path(top) / n for n in names]
    return sorted(f for f in found if f.is_file())


def families(root: pathlib.Path):
    """(C++ files, scripts and build files, vendored headers)."""
    cpp, scripts, vendor = [], [], []
    for f in all_files(root):
        rel = f.relative_to(root).as_posix()
        if rel.startswith(VENDOR_DIR + "/"):
            if f.suffix in SUFFIXES:
                vendor.append(f)
        elif rel.startswith("third_party/"):
            continue
        elif f.suffix in SUFFIXES:
            if rel.split("/", 1)[0] in DIRS:
                cpp.append(f)
        elif f.name in SCRIPT_NAMES or f.suffix in SCRIPT_SUFFIXES or f.name.endswith(".cmake.in"):
            scripts.append(f)
    return cpp, scripts, vendor


def years_reason(first: int, last, last_commit):
    """Why a `Copyright first[-last]` is wrong, or None."""
    end = int(last or first)
    if not first <= end <= THIS_YEAR or (last and first == end):
        return f"bad copyright years {first}{'-' + last if last else ''}"
    if last_commit and end < last_commit:
        return f"copyright ends in {end}, file last committed in {last_commit}"
    return None


def check(text: str, last_commit=None):
    """The rules `text` breaks, as short reasons (empty: conforming)."""
    bad = []
    m = HEADER.match(text)
    if not m:
        bad.append("does not open with the license header" if "Licensed under the Apache" in text
                   or YEAR.search(text[:4000]) else "no license header")
    else:
        why = years_reason(int(m.group(1)), m.group(2), last_commit)
        if why:
            bad.append(why)
    if text.count("Licensed under the Apache") > 1:
        bad.append("license present more than once")
    if STAMP.search(text):
        bad.append("IDE 'Created by' stamp")
    return bad


def check_script(text: str, last_commit=None):
    """The rules a script or build file breaks (rule 3)."""
    pro = SCRIPT_PROLOGUE.match(text).end()
    m = SCRIPT_HEADER.match(text, pro)
    if not m:
        return ["no SPDX header (# Copyright ... CEA / # SPDX-License-Identifier: Apache-2.0)"
                + (" where expected" if "SPDX-License-Identifier" in text or YEAR.search(text) else "")]
    bad = []
    why = years_reason(int(m.group(1)), m.group(2), last_commit)
    if why:
        bad.append(why)
    if len(SCRIPT_LICENSE_LINE.findall(text)) > 1:
        bad.append("another CEA license comment besides the SPDX header")
    return bad


def vendor_spdx(text: str):
    """The SPDX identifier the provenance comment's license line calls for, or None."""
    m = VENDOR_LICENSE_LINE.search(text[:2000])
    return VENDOR_LICENSES.get(m.group(1)) if m else None


def check_vendor(text: str, last_commit=None):
    """The rules a vendored header breaks (rule 4)."""
    want = vendor_spdx(text)
    if want is None:
        return ["provenance comment names no known license (see VENDOR_LICENSES)"]
    if not text.startswith(f"// SPDX-License-Identifier: {want}\n"):
        return [f"line 1 is not '// SPDX-License-Identifier: {want}'"]
    return []


def cut(text: str, a: int, b: int):
    """text without text[a:b], at most one blank line left at the junction, none at the top."""
    left, right = text[:a], text[b:]
    nl = (len(left) - len(left.rstrip("\n"))) + (len(right) - len(right.lstrip("\n")))
    left, right = left.rstrip("\n"), right.lstrip("\n")
    return right if not left else left + "\n" * min(max(nl, 1), 2) + right


def commit_years(root: pathlib.Path):
    """{relative path: year of its last commit}, from one git log (empty outside git)."""
    try:
        out = subprocess.run(["git", "log", "--format=@%ad", "--date=format:%Y", "--name-only"],
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


def fixed_script(text: str, path: pathlib.Path, root: pathlib.Path):
    pro = SCRIPT_PROLOGUE.match(text).end()
    head, body = text[:pro], text[pro:]
    years = []
    while True:  # every older CEA license comment
        mm = SCRIPT_OLD.search(body[:6000])
        if not mm:
            break
        ym = YEAR.search(mm.group(0))
        if ym:
            years.append(int(ym.group(1)))
        body = cut(body, mm.start(), mm.end())
    first = min(years) if years else first_commit_year(path, root)
    span = str(THIS_YEAR) if first >= THIS_YEAR else f"{first}-{THIS_YEAR}"
    return f"{head}# Copyright {span} CEA (www.cea.fr)\n# SPDX-License-Identifier: Apache-2.0\n{body}"


def fixed_vendor(text: str, path: pathlib.Path, root: pathlib.Path):
    want = vendor_spdx(text)
    if want is None:
        return text  # nothing safe to write: the check keeps reporting it
    text = re.sub(r"\A// SPDX-License-Identifier:[^\n]*\n", "", text)
    return f"// SPDX-License-Identifier: {want}\n{text}"


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
    cpp, scripts, vendor = families(root)
    for group, check_fn, fix_fn in ((cpp, check, fixed), (scripts, check_script, fixed_script),
                                    (vendor, check_vendor, fixed_vendor)):
        for f in group:
            text = f.read_text(encoding="utf-8")
            last = last_commit.get(f.relative_to(root).as_posix())
            reasons = check_fn(text, last)
            if reasons and "--fix" in sys.argv:
                new = fix_fn(text, f, root)
                if new != text:
                    f.write_text(new, encoding="utf-8")
                    n_fixed += 1
                    reasons = check_fn(new, last)
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
    print(f"check_headers: OK, {len(cpp)} C++ files with the license header, {len(scripts)} scripts and "
          f"build files with SPDX lines, {len(vendor)} vendored headers with their upstream SPDX line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
