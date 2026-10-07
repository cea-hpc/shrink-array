#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Turn the tokens of an ALLOC_TOKEN census back into type names.

    python3 python/alloc_token_names.py [--dwarf] <binary> [<log>]    (log defaults to stdin)

The census (chunk/alloc_token_rt.cpp, SANITIZER=token builds) prints one line
per token, and a token is Clang's hash of the allocated type's name:
SipHash-2-4 with LLVM's fixed "stable hash" key, reduced to the lower half of
the 64-bit space, the upper half being for types that contain pointers
(-falloc-token-mode=typehashpointersplit, the default). The hash cannot be
inverted, so this script hashes CANDIDATE names and keeps the ones that match.

Candidates come from the binary's own symbols: `nm -C` prints every class
whose method, vtable or typeinfo was emitted, with its template arguments,
which is where allocated types live. Clang spells a type slightly differently
from the demangler (`const int` vs `int const`, `>>` vs `> >`), so each
candidate is normalised to Clang's spelling before hashing. Fundamental types
are added by hand: `new double[n]` names no class.

Best effort: a token that stays unresolved is one whose type left no symbol
(an aggregate with nothing out-of-line, say). --dwarf adds every struct /
class / union of the debug info as a candidate, through llvm-dwarfdump; that
covers the aggregates but is slow on a large binary. A token still unresolved
keeps its hex value.
"""

import re
import shutil
import struct
import subprocess
import sys

M64 = (1 << 64) - 1
HALF = M64 // 2  # SIZE_MAX / 2: the pointer / no-pointer boundary

# llvm/lib/Support/SipHash.cpp: the key behind getStableSipHash().
KEY = bytes([0xB5, 0xD4, 0xC9, 0xEB, 0x79, 0x10, 0x4A, 0x79, 0x6F, 0xEC, 0x8B, 0x1B, 0x42, 0x87, 0x81, 0xD4])

FUNDAMENTAL = [
    "bool", "char", "signed char", "unsigned char", "wchar_t", "char8_t", "char16_t", "char32_t",
    "short", "unsigned short", "int", "unsigned int", "long", "unsigned long",
    "long long", "unsigned long long", "__int128", "unsigned __int128",
    "float", "double", "long double", "void", "void *", "char *", "const char *",
]


def _rotl(x, b):
    return ((x << b) | (x >> (64 - b))) & M64


def siphash24(key, data):
    """Reference SipHash-2-4, 64-bit output."""
    k0, k1 = struct.unpack("<QQ", key)
    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    def rnd():
        nonlocal v0, v1, v2, v3
        v0 = (v0 + v1) & M64; v1 = _rotl(v1, 13); v1 ^= v0; v0 = _rotl(v0, 32)
        v2 = (v2 + v3) & M64; v3 = _rotl(v3, 16); v3 ^= v2
        v0 = (v0 + v3) & M64; v3 = _rotl(v3, 21); v3 ^= v0
        v2 = (v2 + v1) & M64; v1 = _rotl(v1, 17); v1 ^= v2; v2 = _rotl(v2, 32)

    n = len(data)
    b = (n << 56) & M64
    for i in range(0, n - n % 8, 8):
        m = struct.unpack("<Q", data[i:i + 8])[0]
        v3 ^= m
        rnd(); rnd()
        v0 ^= m
    for i, ch in enumerate(data[n - n % 8:]):
        b |= ch << (8 * i)
    v3 ^= b
    rnd(); rnd()
    v0 ^= b
    v2 ^= 0xFF
    rnd(); rnd(); rnd(); rnd()
    return (v0 ^ v1 ^ v2 ^ v3) & M64


def reduced(name):
    """The token of `name` minus its pointer-half offset."""
    return siphash24(KEY, name.encode()) % HALF


# ── Candidate names ──────────────────────────────────────────────────────────

def split_top(s, sep=","):
    """Split on `sep` outside <> and ()."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


# Enum template arguments: the demangler and DWARF print the value, Clang the
# enumerator. Only the one seen so far; extend as tokens stay unresolved.
ENUM_ARGS = {
    "(__gnu_cxx::_Lock_policy)0": "__gnu_cxx::_S_single",
    "(__gnu_cxx::_Lock_policy)1": "__gnu_cxx::_S_mutex",
    "(__gnu_cxx::_Lock_policy)2": "__gnu_cxx::_S_atomic",
}


def normalise(s):
    """Demangler spelling -> Clang spelling: `T const` -> `const T`, `> >` -> `>>`."""
    s = s.strip()
    s = ENUM_ARGS.get(s, s)
    if s.endswith(" const"):
        return "const " + normalise(s[:-6])
    if s.endswith("const*") or s.endswith("const&"):
        return "const " + normalise(s[:-6]) + s[-1]
    i = s.find("<")
    if i < 0 or not s.endswith(">"):
        return s
    base, args = s[:i], s[i + 1:-1]
    return base + "<" + ", ".join(normalise(a) for a in split_top(args)) + ">"


def template_args(s):
    """Every template argument of `s`, recursively (they are types too)."""
    i = s.find("<")
    if i < 0 or not s.endswith(">"):
        return []
    out = []
    for a in split_top(s[i + 1:-1]):
        a = a.strip()
        out.append(a)
        out.extend(template_args(a))
    return out


def candidates_from_binary(binary):
    names = set(FUNDAMENTAL)
    nm = subprocess.run(["nm", "-C", "--defined-only", binary], capture_output=True, text=True, check=True).stdout
    for line in nm.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        sym = parts[2]
        # "vtable for X", "typeinfo for X", "typeinfo name for X", "VTT for X", ...
        m = re.match(r"^(?:vtable|typeinfo|typeinfo name|VTT|construction vtable) for (.+)$", sym)
        if m:
            scopes = [m.group(1)]
        else:
            # A function: drop its parameter list and qualifiers, keep the
            # qualified name, then drop the member to get the class.
            depth, cut = 0, len(sym)
            for i, ch in enumerate(sym):
                if ch in "<(":
                    if ch == "(" and depth == 0:
                        cut = i
                        break
                    depth += 1
                elif ch in ">)":
                    depth -= 1
            qual = sym[:cut].strip()
            comps = split_top(qual, ":")  # "a::b::c" -> ["a", "", "b", "", "c"]
            comps = [c for c in comps if c != ""]
            scopes = ["::".join(comps[:k]) for k in range(1, len(comps))]
        for sc in scopes:
            if not sc:
                continue
            names.add(sc)
            names.update(template_args(sc))
    return {normalise(n) for n in names if n}


def candidates_from_dwarf(binary):
    """Every named struct / class / union of the debug info, namespace-qualified.

    llvm-dwarfdump indents a DIE by its depth, and DW_AT_name follows the
    DW_TAG line before any child, so a stack of (indent, tag, name) frames is
    enough to qualify a type by its enclosing namespaces and classes.
    """
    names = set()
    tool = next((t for t in map(shutil.which, ("llvm-dwarfdump", "llvm-dwarfdump-22", "llvm-dwarfdump-23")) if t), None)
    if tool is None:
        print("[alloc_token_names] --dwarf: no llvm-dwarfdump on PATH, skipped", file=sys.stderr)
        return names
    proc = subprocess.Popen([tool, "--debug-info", binary], stdout=subprocess.PIPE, text=True, errors="replace")
    stack = []  # [indent, tag, name]
    scoping = {"DW_TAG_namespace", "DW_TAG_structure_type", "DW_TAG_class_type", "DW_TAG_union_type"}
    tag_re = re.compile(r"^0x[0-9a-fA-F]+:(\s+)(DW_TAG_\w+)")
    name_re = re.compile(r'^\s+DW_AT_name\s+\("(.*)"\)')
    for line in proc.stdout:
        m = tag_re.match(line)
        if m:
            indent = len(m.group(1))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append([indent, m.group(2), None])
            continue
        m = name_re.match(line)
        if m and stack and stack[-1][2] is None:
            stack[-1][2] = m.group(1)
            if stack[-1][1] in scoping and stack[-1][1] != "DW_TAG_namespace":
                parts = []
                for _, tag, name in stack:
                    if tag not in scoping:
                        continue
                    parts.append(name if name is not None else "(anonymous namespace)")
                qual = "::".join(parts)
                names.add(qual)
                names.update(template_args(qual))
    proc.wait()
    return {normalise(n) for n in names if n}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    use_dwarf = "--dwarf" in argv
    if not args:
        sys.exit(__doc__)
    binary = args[0]
    log = open(args[1]).read() if len(args) > 1 else sys.stdin.read()

    candidates = candidates_from_binary(binary)
    if use_dwarf:
        candidates |= candidates_from_dwarf(binary)
    table = {}
    for name in candidates:
        table.setdefault(reduced(name), name)

    tok_re = re.compile(r"^(\s*0x([0-9a-fA-F]{16})\s.*?)$", re.M)
    seen, resolved = 0, 0

    def annotate(m):
        nonlocal seen, resolved
        seen += 1
        token = int(m.group(2), 16)
        name = table.get(token - HALF if token >= HALF else token)
        if token == 0:
            return m.group(1)
        if name is None:
            return m.group(1) + "  ?"
        resolved += 1
        return m.group(1) + "  " + name

    sys.stdout.write(tok_re.sub(annotate, log))
    print(f"[alloc_token_names] {resolved}/{seen} tokens resolved from {len(table)} candidate names", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
