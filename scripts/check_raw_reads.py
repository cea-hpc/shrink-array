#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every raw read of a chunk's buffer must say why it is safe.

A chunk's `values` pointer names a buffer that a compression pass may release
(and the allocator hand out again) at any time, or a zero_any identifier slot
that holds nothing. Reading an element straight through it is only safe in a
few situations: inside the write session (the write/compress handshake keeps
compressors off), on the deferred-replay path after materialisation, inside a
read_reclaim::ReadSection, in the compressor's own scan of a chunk it claimed,
in a serial build, or when handing out an address the caller must keep valid.
Three bugs came from reads that were none of those (de9309d8, 64289081,
8f444c17).

This script finds every such dereference in the library and fails on any line
that does not carry a `raw-read: <why>` comment, on the line itself or on the
line just above (a run of consecutive raw reads shares one). The marker is the
review: a new raw read has to state which of the cases above it is in.

Usage: scripts/check_raw_reads.py [repo root]   (exit 1 when a line is unmarked)
"""

import pathlib
import re
import sys

# Names the library gives a loaded values pointer.
_PTR = r"(?:values|vals|values_loc|live|values_before|values_after)"

PATTERNS = [
    re.compile(r"\b" + _PTR + r"\s*\["),                                   # values[num]
    re.compile(r"\bvalues_load\(\)\s*\["),                                   # values_load()[i]
    re.compile(r"\belem_(?:get|update)\(\s*(?:static_cast<const T\*>\()?" + _PTR + r"\b"),
    re.compile(r"\bload_at<[^>]*>\(\s*(?:vals|values_load)\b"),
]
MARKER = "raw-read:"
COMMENT = re.compile(r"^\s*(?://|\*|/\*)")


def scan(root: pathlib.Path):
    files = sorted(
        [p for p in (root / "chunk" / "public").rglob("*.h")]
        + [p for p in (root / "chunk" / "private").rglob("*.cpp")]
    )
    unmarked, marked = [], 0
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        covered_prev = False  # the line just above is a raw read with a reason
        for i, line in enumerate(lines):
            if COMMENT.match(line):
                covered_prev = False
                continue
            code = line.split("//", 1)[0]
            if not any(p.search(code) for p in PATTERNS):
                covered_prev = False
                continue
            prev = lines[i - 1] if i > 0 else ""
            # A run of consecutive raw reads shares the reason given above it.
            if MARKER in line or MARKER in prev or covered_prev:
                marked += 1
                covered_prev = True
            else:
                unmarked.append(f"{path.relative_to(root)}:{i + 1}: {line.strip()}")
                covered_prev = False
    return unmarked, marked


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent)
    unmarked, marked = scan(root)
    if unmarked:
        print(f"check_raw_reads: {len(unmarked)} raw read(s) of a chunk buffer without a `{MARKER}` reason:")
        for entry in unmarked:
            print("  " + entry)
        print("Say which safe case it is in (see this script's header), or read through "
              "element_at / read_values_outside_session.")
        return 1
    print(f"check_raw_reads: OK, {marked} raw read(s), each with a reason")
    return 0


if __name__ == "__main__":
    sys.exit(main())
