#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Move the long in-class method bodies of the library headers to their `_inline.h` companion.

Rule: a member function whose body exceeds THRESHOLD non-blank lines is declared in the
class and defined out of class, in `<header>_inline.h` (created, with its `#include` at the
end of the header, when the header has none). The Doxygen comment stays on the declaration.
A header with no companion yet keeps its long bodies in class while it has fewer than
MIN_BODIES of them: a companion file for one or two functions costs more to read than it
saves. A header that has a companion moves every long body into it.

Two entry points, one file:
    python3 apply_inline_bodies.py [--dry-run]         # from the repository root: move the bodies,
                                                       # install scripts/check_inline_bodies.py (this
                                                       # file), register it in CMake and contributing.md
    python3 scripts/check_inline_bodies.py --check     # the ctest: lists the bodies that break the
                                                       # rule, exit 1 if any
    python3 scripts/check_inline_bodies.py --apply     # move them again after new code

What moves: the body, as it is. The declaration keeps its specifiers (static, virtual,
explicit, override, attributes), the definition gets the template heads of its enclosing
classes, the qualified name (a class-local return type becomes `Class<T>::Name`, a name of
the class's namespace gets that namespace) and `inline` for a non-template class. What
stays in the class, and is listed as "kept": friends, constexpr members a static_assert of
the header evaluates (and every constexpr member of a non-template class), members of a
class at global scope. Idempotent: a second run moves nothing. Run clang-format after it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

THRESHOLD = 5

# Fewest long bodies for which a header without an _inline.h gets one (see the rule above).
MIN_BODIES = 3

# Headers to scan: (directory, include prefix used by the companions)
SCOPES = [
    ("chunk/public/chunk", "chunk"),
    ("fast_allocators/public/fast_allocators", "fast_allocators"),
]

# Bodies the rule deliberately leaves in the class (why, in the report):
#   "Class::name" -> reason
KEEP: dict[str, str] = {}

SPECIFIERS = {"static", "virtual", "explicit", "inline", "constexpr", "consteval", "friend",
              "CHUNK_FORCE_INLINE", "CHUNK_FLATTEN", "FAST_ALLOC_COLD", "extern"}
DROP_IN_DEF = {"static", "virtual", "explicit", "friend"}
KEYWORDS_NOT_NAMES = {"decltype", "noexcept", "sizeof", "alignas", "alignof", "requires",
                      "static_assert", "__attribute__", "typeid", "if", "for", "while",
                      "switch", "catch", "return", "throw", "new", "delete", "explicit"}


# --------------------------------------------------------------------------- text helpers
def blank_comments_and_strings(src: str) -> str:
    """Same length as src; comments and literals replaced by spaces (newlines kept)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif c == "'" or c == '"':
            # raw strings are not used in these headers; plain literals only
            q = c
            j = i + 1
            while j < n and src[j] != q and src[j] != "\n":
                if src[j] == "\\":
                    j += 1
                j += 1
            j = min(n, j + 1)
            out.append(c + re.sub(r"[^\n]", " ", src[i + 1:j - 1]) + (q if j - 1 < n else ""))
            i = j
        else:
            out.append(c)
            i += 1
    s = "".join(out)
    assert len(s) == len(src)
    return s


def blank_pp_lines(src: str) -> str:
    """Blank the preprocessor lines (same length)."""
    lines = src.split("\n")
    out = []
    cont = False
    for ln in lines:
        if cont or ln.lstrip().startswith("#"):
            out.append(" " * len(ln))
            cont = ln.rstrip().endswith("\\")
        else:
            out.append(ln)
    return "\n".join(out)


def match_bracket(s: str, i: int) -> int:
    """Index of the bracket matching s[i] (one of ( [ { <)."""
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    o = s[i]
    c = pairs[o]
    depth = 0
    j = i
    while j < len(s):
        if s[j] == o:
            depth += 1
        elif s[j] == c:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def split_top(s: str, sep: str = ",") -> list[str]:
    """Split s on sep at bracket depth 0 (over (), [], {}, <>)."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def strip_defaults(params: str) -> str:
    """Remove `= default` from each top-level parameter."""
    out = []
    for p in split_top(params):
        depth = 0
        cut = None
        for k, ch in enumerate(p):
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == "=" and depth == 0 and p[k - 1:k] not in "=!<>" and p[k + 1:k + 2] != "=":
                cut = k
                break
        out.append(p[:cut].rstrip() if cut is not None else p)
    return ",".join(out)


def template_param_names(head: str) -> str:
    """'template <typename T, bool B = x, typename... Ts>' -> 'T, B, Ts...'."""
    inner = head[head.index("<") + 1: match_bracket(head, head.index("<"))]
    names = []
    for p in split_top(inner):
        p = p.strip()
        if "=" in p:
            p = strip_defaults(p)
        pack = "..." in p
        p_clean = p.replace("...", " ")
        m = re.search(r"([A-Za-z_]\w*)\s*$", p_clean)
        name = m.group(1) if m else p_clean.strip()
        names.append(name + ("..." if pack else ""))
    return ", ".join(names)


def strip_template_defaults(head: str) -> str:
    lt = head.index("<")
    gt = match_bracket(head, lt)
    inner = head[lt + 1:gt]
    return head[:lt + 1] + strip_defaults(inner) + head[gt:]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- model
@dataclass
class ClassScope:
    name: str
    template_head: str | None   # "template <typename T, bool B = x>" (raw) or None
    spec_args: str | None       # "<T, Derived, true>" for a partial specialisation
    requires_clause: str | None
    namespaces: list[str]       # enclosing namespaces (outermost first)
    body_start: int
    local_names: set[str] = field(default_factory=set)

    def qualifier(self) -> str:
        if self.spec_args:
            return self.name + self.spec_args
        if self.template_head:
            return f"{self.name}<{template_param_names(self.template_head)}>"
        return self.name


@dataclass
class Method:
    cls_chain: list[ClassScope]
    head_start: int        # raw index where the declaration starts (after template heads)
    head: str              # stripped head text (no template heads), normalised
    member_template: str | None
    member_requires: str | None
    brace: int             # index of '{'
    end: int               # index of matching '}'
    nb_lines: int
    name: str = ""
    outer_ns: str = ""     # the namespace block the definition goes in


CLASS_RE = re.compile(
    r"\b(class|struct|union)\b\s+((?:alignas\([^)]*\)\s*|\[\[[^\]]*\]\]\s*|[A-Z_]+_API\s+)*)([A-Za-z_]\w*)\s*(<)?")


def parse_file(raw: str):
    """Return (classes, methods) found in raw."""
    src = blank_pp_lines(blank_comments_and_strings(raw))
    n = len(src)
    stack: list[tuple[str, object]] = []   # (kind, payload)
    ns_stack: list[str] = []
    methods: list[Method] = []
    ns_spans: list[tuple[int, int, tuple[str, ...]]] = []   # (start, end, namespaces)
    ns_decls: list[tuple[tuple[str, ...], str]] = []        # (namespaces, name) of namespace-scope classes
    class_spans: list[tuple[int, int]] = []
    i = 0
    last_boundary = 0
    while i < n:
        c = src[i]
        if c == ";" and not any(k == "func" for k, _ in stack):
            last_boundary = i + 1
        elif c == "{":
            head = src[last_boundary:i]
            # drop access specifiers (same length)
            head = re.sub(r"\b(public|private|protected)\s*:", lambda mm: " " * len(mm.group(0)), head)
            head_s = head.strip()
            top = stack[-1][0] if stack else "ns"
            kind, payload = "other", None
            mns = re.match(r"^(inline\s+)?namespace\s+([A-Za-z_][\w:]*)?\s*$", head_s)
            if mns and top == "ns":
                kind, payload = "ns", (mns.group(2) or "")
            elif top in ("ns", "class"):
                tmpl, req, rest = split_template_heads(head_s)
                mc = CLASS_RE.match(rest)
                if mc and not rest.startswith("enum") and len(tmpl) <= 1:
                    spec = None
                    if mc.group(4):
                        lt = mc.end() - 1
                        gt = match_bracket(rest, lt)
                        spec = norm_ws(rest[lt:gt + 1])
                    kind = "class"
                    payload = ClassScope(mc.group(3), tmpl[0] if tmpl else None, spec, req,
                                         list(ns_stack), i + 1)
                elif top == "class":
                    m = parse_method_head(head)
                    if m is not None:
                        kind, payload = "func", (m, last_boundary)
            if kind == "other" and top in ("ns", "class") and \
                    (re.search(r"(=|\(|,|\[|<|>)\s*$", head_s) or
                     (re.search(r"[A-Za-z_0-9]\s*$", head_s) and "(" not in head_s)):
                kind = "init"   # brace initialiser inside a declaration: the head goes on
            stack.append((kind, payload))
            if kind == "ns":
                parts = [p for p in payload.split("::") if p] or [""]
                ns_stack.extend(parts)
                stack[-1] = ("ns", (parts, i + 1))
            if kind == "class" and not any(k == "class" for k, _ in stack[:-1]):
                ns_decls.append((tuple(ns_stack), payload.name))
            if kind != "init":
                last_boundary = i + 1
        elif c == "}":
            if stack:
                kind, payload = stack.pop()
                if kind == "init":
                    i += 1
                    continue          # the enclosing declaration goes on
                if kind == "ns":
                    parts, start = payload
                    ns_spans.append((start, i, tuple(ns_stack)))
                    del ns_stack[len(ns_stack) - len(parts):]
                elif kind == "func":
                    (head_s, tmpl, req, head_offset), lb = payload
                    chain = [p for k, p in stack if k == "class"]
                    brace = find_open_brace(src, i)
                    l0 = src.count("\n", 0, brace)
                    l1 = src.count("\n", 0, i)
                    body_lines = [ln for ln in raw.split("\n")[l0 + 1:l1] if ln.strip()]
                    methods.append(Method(chain, lb + head_offset, head_s, tmpl, req, brace, i,
                                          len(body_lines)))
                elif kind == "class":
                    class_spans.append((payload.body_start, i))
            last_boundary = i + 1
        i += 1
    # local names per class (for the return types)
    for m in methods:
        for cs in m.cls_chain:
            if not cs.local_names:
                cs.local_names = collect_local_names(src, cs.body_start)
    # other namespace-scope declarations: aliases, concepts, enums, variables
    for mm in re.finditer(r"\b(?:using|concept|enum(?:\s+class)?)\s+([A-Za-z_]\w*)\s*[=;:{]", src):
        inner = None
        for start, end, nss in ns_spans:
            if start <= mm.start() < end and (inner is None or start > inner[0]):
                inner = (start, end, nss)
        if inner is not None and not any(a <= mm.start() < b for a, b in class_spans):
            ns_decls.append((inner[2], mm.group(1)))
    parse_file.last_ns_decls = ns_decls
    return methods


def find_open_brace(src: str, close: int) -> int:
    depth = 0
    j = close
    while j >= 0:
        if src[j] == "}":
            depth += 1
        elif src[j] == "{":
            depth -= 1
            if depth == 0:
                return j
        j -= 1
    return -1


def collect_local_names(src: str, body_start: int) -> set[str]:
    end = match_bracket(src, body_start - 1)
    body = src[body_start:end]
    names = set()
    for m in re.finditer(r"\busing\s+([A-Za-z_]\w*)\s*=", body):
        names.add(m.group(1))
    for m in re.finditer(r"\busing\s+[\w:<>, ]+::([A-Za-z_]\w*)\s*;", body):
        names.add(m.group(1))
    for m in re.finditer(r"\btypedef\b[^;]*?\b([A-Za-z_]\w*)\s*;", body):
        names.add(m.group(1))
    for m in re.finditer(r"\b(?:class|struct|union|enum(?:\s+class)?)\s+([A-Za-z_]\w*)", body):
        names.add(m.group(1))
    for m in re.finditer(r"\bstatic\s+(?:constexpr|const|inline)[^;=(]*?\b([A-Za-z_]\w*)\s*(?:=|\{)", body):
        names.add(m.group(1))
    return names


def split_template_heads(head: str):
    """Leading 'template <...>' groups (and a requires-clause) -> (heads, requires, rest)."""
    heads = []
    req = None
    s = head.lstrip()
    while s.startswith("template"):
        lt = s.index("<")
        gt = match_bracket(s, lt)
        heads.append(norm_ws(s[:gt + 1]))
        s = s[gt + 1:].lstrip()
        if s.startswith("requires"):
            # requires-clause: up to the next 'template', class/struct keyword, or the declaration.
            m = re.search(r"\b(class|struct|union|template|typename|void|auto|static|constexpr|inline|\[\[)", s[8:])
            cut = 8 + m.start() if m else len(s)
            # a requires clause ending in a type name is ambiguous: keep to the known keyword
            req = norm_ws(s[:cut])
            s = s[cut:].lstrip()
    return heads, req, s


def parse_method_head(head: str):
    """A function head (no template heads) ending right before '{' -> (head, tmpl, req, offset) or None.
    `head` keeps the raw positions (same length as the source slice); offset is where the
    declaration proper starts in it (after the template heads)."""
    tmpl, req, rest = split_template_heads(head.strip())
    if not rest:
        return None
    if re.match(r"^(if|for|while|switch|do|else|try|catch)\b", rest):
        return None
    if re.match(r"^[^(]*=\s*\[", rest):   # `int x = [](...)`: a lambda initialiser
        return None
    if ")" not in rest:
        return None
    # the declaration must END with a parameter list + qualifiers (not `= value` nor a brace init)
    tail = norm_ws(rest)
    if not re.search(r"\)(\s*(const|volatile|noexcept(\([^{}]*\))?|override|final|&&?|->\s*[^{]+|requires\b[^{]*|:\s*[^{]+))*\s*$", tail):
        return None
    probe = rest[:30]
    offset = head.find(probe)
    if offset < 0:
        offset = 0
    return (tail, tmpl[-1] if tmpl else None, req, offset)


# --------------------------------------------------------------------------- signature parsing
def find_name(rest: str):
    """-> (name_start, name_end, params_start, params_end, name) in rest, or None."""
    depth = 0
    i = 0
    n = len(rest)
    while i < n:
        ch = rest[i]
        if rest.startswith("operator", i) and (i == 0 or not (rest[i - 1].isalnum() or rest[i - 1] == "_")) \
                and depth == 0:
            j = i + 8
            while j < n and rest[j] == " ":
                j += 1
            if rest.startswith("()", j):
                k = j + 2
            elif rest.startswith("[]", j):
                k = j + 2
            elif re.match(r"(new|delete)\s*\[\]", rest[j:]):
                k = j + re.match(r"(new|delete)\s*\[\]", rest[j:]).end()
            elif re.match(r"(new|delete)\b", rest[j:]):
                k = j + re.match(r"(new|delete)\b", rest[j:]).end()
            elif re.match(r"\"\"\s*[A-Za-z_]\w*", rest[j:]):
                k = j + re.match(r"\"\"\s*[A-Za-z_]\w*", rest[j:]).end()
            else:
                mo = re.match(r"(<=>|->\*|->|\+\+|--|<<=|>>=|<<|>>|[-+*/%^&|~!=<>]=|&&|\|\||[-+*/%^&|~!=<>,])", rest[j:])
                if mo:
                    k = j + mo.end()
                else:
                    # conversion operator: operator T (...)
                    mo = re.match(r"[^(]+", rest[j:])
                    k = j + mo.end()
                    k = k - (len(rest[j:k]) - len(rest[j:k].rstrip()))
            while k < n and rest[k] == " ":
                k += 1
            if k < n and rest[k] == "(":
                pe = match_bracket(rest, k)
                return i, k, k, pe, norm_ws(rest[i:k])
            return None
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif depth == 0 and (ch.isalpha() or ch == "_"):
            m = re.match(r"[A-Za-z_]\w*", rest[i:])
            word = m.group(0)
            j = i + len(word)
            while j < n and rest[j] == " ":
                j += 1
            if j < n and rest[j] == "(" and word not in KEYWORDS_NOT_NAMES and word not in SPECIFIERS:
                # ensure the word is not part of a qualified type followed by '(' e.g. decltype
                pe = match_bracket(rest, j)
                if pe < 0:
                    return None
                return i, i + len(word), j, pe, word
            i = j
            continue
        i += 1
    return None


def split_specifiers(prefix: str):
    """prefix before the name -> (attrs, specifiers, return_type)."""
    attrs, specs = [], []
    s = prefix.strip()
    changed = True
    while changed and s:
        changed = False
        if s.startswith("[["):
            e = s.index("]]") + 2
            attrs.append(s[:e])
            s = s[e:].lstrip()
            changed = True
            continue
        if s.startswith("__attribute__"):
            lp = s.index("(")
            e = match_bracket(s, lp) + 1
            attrs.append(s[:e])
            s = s[e:].lstrip()
            changed = True
            continue
        m = re.match(r"(\w+)\b", s)
        if m and m.group(1) in SPECIFIERS:
            specs.append(m.group(1))
            s = s[m.end():].lstrip()
            changed = True
            continue
        if m and m.group(1) == "explicit" and s[m.end():].lstrip().startswith("("):
            lp = s.index("(")
            e = match_bracket(s, lp) + 1
            specs.append("explicit")
            s = s[e:].lstrip()
            changed = True
    return attrs, specs, s.strip()


def parse_suffix(suffix: str, is_ctor: bool):
    """Text after the parameter list -> (quals_for_decl, quals_for_def, init_list)."""
    s = suffix.strip()
    init = None
    if is_ctor:
        # a depth-0 ':' that is not '::' starts the mem-initializer list
        depth = 0
        for k, ch in enumerate(s):
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == ":" and depth == 0 and s[k - 1:k] != ":" and s[k + 1:k + 2] != ":":
                init = s[k:].strip()
                s = s[:k].strip()
                break
    decl = s
    # definition: drop override / final, and the attributes (the declaration carries them)
    d = re.sub(r"\b(override|final)\b", " ", s)
    d = re.sub(r"\[\[[^\]]*\]\]", " ", d)
    while "__attribute__" in d:
        k = d.index("__attribute__")
        lp = d.index("(", k)
        d = d[:k] + d[match_bracket(d, lp) + 1:]
    d = norm_ws(d)
    return decl, d, init


# --------------------------------------------------------------------------- rewriting
def indent_of_line(raw: str, pos: int) -> str:
    ls = raw.rfind("\n", 0, pos) + 1
    m = re.match(r"[ \t]*", raw[ls:])
    return m.group(0)


def build_definition(m: Method, raw: str):
    """-> (definition text, new in-class declaration tail) or (None, reason)."""
    rest = m.head
    found = find_name(rest)
    if not found:
        return None, "signature not understood"
    ns_, ne, ps, pe, name = found
    prefix = rest[:ns_]
    params = rest[ps + 1:pe]
    suffix = rest[pe + 1:]
    attrs, specs, ret = split_specifiers(prefix)
    m.name = f"{'::'.join(c.name for c in m.cls_chain)}::{name.lstrip('~') if ret.endswith('~') else name}"
    if ret.endswith("~"):
        m.name = m.name.replace("::" + name, "::~" + name)
    if "friend" in specs:
        return None, "friend"
    if {"constexpr", "consteval"} & set(specs):
        is_template = any(c.template_head and norm_ws(c.template_head) != "template <>"
                          for c in m.cls_chain) or m.member_template
        if not is_template:
            return None, "constexpr member of a non-template class: must be visible for constant evaluation"
        stripped_all = blank_comments_and_strings(raw)
        if re.search(rf"static_assert\s*\([^;]*\b{re.escape(find_name(rest)[4])}\s*\(", stripped_all, re.S):
            return None, "constexpr member used in a static_assert of the header"
    if "#" in rest:
        return None, "preprocessor directive in the signature"
    cls = m.cls_chain[-1]
    is_ctor = ret == "" and (name == cls.name or name == "~" + cls.name or name.startswith("operator "))
    if ret == "" and name != cls.name and not name.startswith("~") and not name.startswith("operator"):
        return None, "no return type"
    if ret.strip() == "" and not is_ctor:
        return None, "no return type"
    # ret may still hold a leading '~' for destructors: name regex catches '~'?
    if ret.endswith("~"):
        ret = ret[:-1].strip()
        name = "~" + name
        is_ctor = True
    decl_quals, def_quals, init = parse_suffix(suffix, is_ctor)
    if m.name in KEEP:
        return None, KEEP[m.name]
    # template heads: enclosing classes (outermost first) then the member template
    heads = []
    reqs = []
    for c in m.cls_chain:
        if c.template_head and norm_ws(c.template_head) == "template <>":
            continue   # explicit specialisation: its members are defined like those of a plain class
        if c.template_head:
            heads.append(strip_template_defaults(c.template_head))
            if c.requires_clause:
                heads.append(c.requires_clause)
        elif c.spec_args:
            return None, "specialisation without template head"
    if m.member_template:
        heads.append(strip_template_defaults(m.member_template))
        if m.member_requires:
            heads.append(m.member_requires)
    # namespaces: the outermost one is the block the definition goes in, the rest qualify the name
    nss = list(cls.namespaces)
    if not nss:
        return None, "class at global namespace scope"
    m.outer_ns = nss[0]
    qual = "::".join([*nss[1:], *[c.qualifier() for c in m.cls_chain]])
    # return type: outside the class scope, a class-local name is written Class<T>::Name
    # (C++20: no `typename` needed) and a name of the class's namespace gets that namespace
    trailing = None
    if ret and ret not in ("auto", "decltype(auto)"):
        ret = qualify_return_type(ret, m, nss)
    has_inline = bool({"constexpr", "consteval", "inline", "CHUNK_FORCE_INLINE"} & set(specs))
    # a non-template class needs `inline`; so does an always_inline function defined out of
    # class (GNU: "always_inline function might not be inlinable", see def_force_inline.h)
    needs_inline = not has_inline and (not heads or "always_inline" in rest)
    def_specs = [s for s in specs if s not in DROP_IN_DEF]
    if needs_inline:
        def_specs.insert(0, "inline")
    lead = " ".join([*attrs, *def_specs])
    lead = (lead + " ") if lead else ""
    sig = f"{lead}{ret + ' ' if ret else ''}{qual}::{name}({strip_defaults(params)})"
    if def_quals:
        # a trailing return already present in the suffix: keep order ... -> R requires
        sig += " " + def_quals
    if trailing:
        # insert before a requires-clause if any
        if " requires " in (" " + def_quals):
            k = sig.rfind(" requires ")
            sig = sig[:k] + f" -> {trailing}" + sig[k:]
        else:
            sig += f" -> {trailing}"
    sig = wrap_signature(sig)
    if init:
        sig += "\n    " + init
    # body
    body_raw = raw[m.brace + 1:m.end]
    base_indent = indent_of_line(raw, m.head_start)
    lines = body_raw.split("\n")
    out_lines = []
    for ln in lines:
        if ln.startswith(base_indent):
            out_lines.append(ln[len(base_indent):])
        else:
            out_lines.append(ln.lstrip() if ln.strip() == "" else ln)
    body = "\n".join(out_lines).strip("\n")
    body = body.rstrip() if body.strip() else ""
    text = "\n".join(heads + [sig, "{", body, "}"]) if body else "\n".join(heads + [sig, "{", "}"])
    # the declaration keeps its text up to the end of the suffix (minus the init list)
    return text, (decl_quals, init)


COLUMN_LIMIT = 120

NS_INDEX: dict[str, set[tuple[str, ...]]] = {}   # name -> namespaces declaring it (whole tree)
TYPE_KEYWORDS = {"const", "volatile", "typename", "auto", "void", "bool", "char", "int", "long", "short",
                 "unsigned", "signed", "float", "double", "decltype", "noexcept", "sizeof", "true", "false",
                 "nullptr", "std", "size_t", "constexpr", "static", "inline"}


def template_param_set(m: "Method") -> set[str]:
    names = set()
    for c in m.cls_chain:
        if c.template_head:
            names |= {x.rstrip(".") for x in template_param_names(c.template_head).split(", ")}
    if m.member_template:
        names |= {x.rstrip(".") for x in template_param_names(m.member_template).split(", ")}
    return names


def qualify_return_type(ret: str, m: "Method", nss: list[str]) -> str:
    cls = m.cls_chain[-1]
    local = set()
    for c in m.cls_chain:
        local |= c.local_names
    tparams = template_param_set(m)
    class_qual = "::".join([*nss[1:], *[c.qualifier() for c in m.cls_chain]])
    ns_inner = tuple(nss)

    is_template = any(c.template_head and norm_ws(c.template_head) != "template <>" for c in m.cls_chain)

    def repl(mm):
        word = mm.group(0)
        before = ret[:mm.start()]
        if before.rstrip().endswith("::") or before.rstrip().endswith("."):
            return word
        if word in TYPE_KEYWORDS or word in tparams:
            return word
        # inside a template argument list, a dependent qualified name needs `typename`
        in_args = before.count("<") > before.count(">")
        tn = "typename " if (is_template and in_args) else ""
        # the injected-class-name of one of the enclosing classes
        followed_by_args = ret[mm.end():].lstrip().startswith("<")
        for k in range(len(m.cls_chain) - 1, -1, -1):
            if word == m.cls_chain[k].name:
                outer = [c.qualifier() for c in m.cls_chain[:k]]
                last = m.cls_chain[k].name if followed_by_args else m.cls_chain[k].qualifier()
                return tn + "::".join([*nss[1:], *outer, last])
        if word in local:
            return f"{tn}{class_qual}::{word}"
        # a name of the class's own (inner) namespace, or of an enclosing one below the outer
        for depth in range(len(ns_inner), 1, -1):
            if ns_inner[:depth] in NS_INDEX.get(word, ()):
                return "::".join(ns_inner[1:depth]) + "::" + word
        return word

    return re.sub(r"[A-Za-z_]\w*", repl, ret)


def wrap_signature(sig: str) -> str:
    """One parameter per line, aligned after '(', when the signature exceeds the column limit
    (.clang-format: ColumnLimit 120, BinPackParameters false, AlignAfterOpenBracket Align)."""
    if len(sig) <= COLUMN_LIMIT:
        return sig
    found = find_name(sig)
    if not found:
        return sig
    _, _, ps, pe, _ = found
    params = [p.strip() for p in split_top(sig[ps + 1:pe])]
    if len(params) <= 1:
        return sig
    head = sig[:ps + 1]
    tail = sig[pe:]
    col = len(head)
    if col + max(len(p) for p in params) + 1 > COLUMN_LIMIT:
        # too far right: continuation indent of 4 instead
        return head + "\n    " + (",\n    ".join(params)) + tail
    pad = " " * col
    return head + (",\n" + pad).join(params) + tail


def companion_path(header: str, scope_dir: str) -> str:
    """chunk/public/chunk/x.h -> chunk/public/chunk/internal/x_inline.h; internal/x.h stays in internal/."""
    d, f = os.path.split(header)
    base = f[:-2]
    if os.path.basename(d) == "internal":
        return os.path.join(d, base + "_inline.h")
    return os.path.join(d, "internal", base + "_inline.h")


def include_path(path: str, scope_dir: str, prefix: str) -> str:
    rel = os.path.relpath(path, scope_dir)
    return f"{prefix}/{rel}".replace(os.sep, "/")


def outer_namespace_close(text: str, ns: str) -> int | None:
    """Index of the '}' closing the LAST top-level `namespace ns { ... }` block of text, or None."""
    s = blank_pp_lines(blank_comments_and_strings(text))
    depth = 0
    pos = None
    opened_at_depth0 = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            if depth == 0:
                head = s[max(s.rfind(";", 0, i), s.rfind("}", 0, i)) + 1:i].strip()
                opened_at_depth0 = re.match(rf"^namespace\s+{re.escape(ns)}\s*$", head) is not None
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and opened_at_depth0:
                pos = i
        i += 1
    return pos


def license_block(raw: str) -> str:
    m = re.match(r"\s*/\*.*?\*/\n", raw, re.S)
    return m.group(0) if m else ""


def migrate(root: str, threshold: int, dry_run: bool, verbose: bool):
    moved, kept = [], []
    # pass 1: index the namespace-scope names of every header (return-type qualification)
    parsed: dict[str, list[Method]] = {}
    for scope_dir, prefix in SCOPES:
        sdir = os.path.join(root, scope_dir)
        for dp, _, fs in os.walk(sdir):
            for f in sorted(fs):
                if f.endswith(".h"):
                    h = os.path.join(dp, f)
                    raw = open(h, encoding="utf-8").read()
                    ms = parse_file(raw)
                    for nss, name in parse_file.last_ns_decls:
                        NS_INDEX.setdefault(name, set()).add(nss)
                    if not f.endswith("_inline.h"):
                        parsed[h] = ms
    # pass 2: move
    for scope_dir, prefix in SCOPES:
        sdir = os.path.join(root, scope_dir)
        for h in sorted(x for x in parsed if x.startswith(sdir + os.sep)):
            raw = open(h, encoding="utf-8").read()
            methods = [m for m in parsed[h] if m.nb_lines > threshold]
            if not methods:
                continue
            comp = companion_path(h, sdir)
            exists = os.path.exists(comp)
            comp_text = open(comp, encoding="utf-8").read() if exists else ""
            defs = []
            edits = []  # (start, end, replacement) on raw
            for m in methods:
                text, tail = build_definition(m, raw)
                if text is None:
                    kept.append((os.path.relpath(h, root), m.name or m.head[:60], m.nb_lines, tail))
                    continue
                decl_quals, init = tail
                # in-class: cut from the end of the parameter list's suffix (before init list) to '}'
                src_head_end = m.brace
                # locate the init list start / the text to keep: recompute on raw via stripped text
                stripped = blank_pp_lines(blank_comments_and_strings(raw))
                seg = stripped[m.head_start:m.brace].replace("\n", " ").replace("\t", " ")
                if init:
                    # the depth-0 ':' after the parameter list starts the mem-initializer list
                    found = find_name(seg)
                    k = found[3] if found else seg.rfind(")")
                    depth = 0
                    pos = None
                    for q in range(k + 1, len(seg)):
                        ch = seg[q]
                        if ch in "([{<":
                            depth += 1
                        elif ch in ")]}>":
                            depth -= 1
                        elif ch == ":" and depth == 0 and seg[q - 1] != ":" and seg[q + 1:q + 2] != ":":
                            pos = q
                            break
                    if pos is None:
                        kept.append((os.path.relpath(h, root), m.name, m.nb_lines, "init list not located"))
                        continue
                    cut_from = m.head_start + pos
                    # back over whitespace
                    while raw[cut_from - 1] in " \t\n":
                        cut_from -= 1
                    edits.append((cut_from, m.end + 1, ";"))
                else:
                    cut_from = m.brace
                    while raw[cut_from - 1] in " \t\n":
                        cut_from -= 1
                    edits.append((cut_from, m.end + 1, ";"))
                defs.append((m, text))
                moved.append((os.path.relpath(h, root), m.name, m.nb_lines, os.path.relpath(comp, root)))
            if not defs:
                continue
            if not exists and len(defs) < MIN_BODIES:
                # too few for a companion of their own: they stay in the class
                del moved[len(moved) - len(defs):]
                for m, _ in defs:
                    kept.append((os.path.relpath(h, root), m.name, m.nb_lines,
                                 f"{len(defs)} long bod{'y' if len(defs) == 1 else 'ies'}, "
                                 f"under {MIN_BODIES}: no _inline.h for so few"))
                continue
            if dry_run:
                continue
            # apply edits to the header (from the end)
            new_raw = raw
            for s, e, rep in sorted(edits, reverse=True):
                new_raw = new_raw[:s] + rep + new_raw[e:]
            # group the definitions by outer namespace, then by class
            groups: dict[str, list[str]] = {}
            last_cls = None
            for m, text in defs:
                blocks = groups.setdefault(m.outer_ns, [])
                key = "::".join(c.qualifier() for c in m.cls_chain)
                if key != last_cls:
                    blocks.append(f"// ---------------------------------------------------------------------------\n"
                                  f"// {m.outer_ns}::{key}\n"
                                  f"// ---------------------------------------------------------------------------\n")
                    last_cls = key
                blocks.append(text + "\n")
            if exists:
                new_comp = comp_text
            else:
                inc = include_path(h, sdir, prefix)
                new_comp = (license_block(raw) +
                            "\n#pragma once\n"
                            f'#include "{inc}" // the class these bodies belong to\n')
            for ns, blocks in groups.items():
                new_defs = "\n".join(blocks)
                pos = outer_namespace_close(new_comp, ns)
                if pos is not None:
                    ls = new_comp.rfind("\n", 0, pos) + 1
                    new_comp = new_comp[:ls].rstrip("\n") + "\n\n" + new_defs + "\n" + new_comp[ls:]
                else:
                    new_comp = (new_comp.rstrip("\n") + "\n\n" +
                                f"namespace {ns}\n{{\n\n" + new_defs + f"\n}} // namespace {ns}\n")
            if not exists:
                inc_c = include_path(comp, sdir, prefix)
                line = f'#include "{inc_c}" // the template bodies\n'
                stripped_h = new_raw.rstrip()
                if stripped_h.endswith("#endif") or re.search(r"#endif[^\n]*$", stripped_h):
                    k = stripped_h.rfind("#endif")
                    new_raw = stripped_h[:k].rstrip("\n") + "\n\n" + line + "\n" + stripped_h[k:] + "\n"
                else:
                    new_raw = stripped_h + "\n\n" + line
            open(h, "w", encoding="utf-8").write(new_raw)
            open(comp, "w", encoding="utf-8").write(new_comp)
    return moved, kept


CHECK_SCRIPT = "scripts/check_inline_bodies.py"

CMAKE_ANCHOR = "    set_property(TEST check_naming PROPERTY LABELS validation)\n"
CMAKE_BLOCK = """    # A member function body longer than 5 lines is defined out of class, in the
    # header's _inline.h companion (scripts/check_inline_bodies.py; apply moves them).
    add_test(NAME check_inline_bodies
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_inline_bodies.py --check ${PROJECT_SOURCE_DIR})
    set_property(TEST check_inline_bodies PROPERTY LABELS validation)
"""

DOC_EN_TABLE_ANCHOR = "| `check_naming` |"
DOC_EN_TABLE_ROW = ("| `check_inline_bodies` | no member function body longer than 5 lines inside a class of "
                    "`chunk/` or `fast_allocators/`: the body is in the header's `<header>_inline.h` companion |\n")
DOC_EN_RULE_ANCHOR = "- **No preprocessor where `if constexpr` works.**"
DOC_EN_RULE = """- **Long bodies go to the `<header>_inline.h` companion.** A member function whose body exceeds 5
  lines is declared in the class and defined in `internal/<header>_inline.h`, included at the
  end of the header; the Doxygen comment stays on the declaration. Accessors of a few lines
  stay in the class. `check_inline_bodies` enforces it;
  `scripts/check_inline_bodies.py --apply` moves what it lists (then clang-format the files).
"""
DOC_FR_TABLE_ANCHOR = "| `check_naming` |"
DOC_FR_TABLE_ROW = ("| `check_inline_bodies` | aucun corps de méthode de plus de 5 lignes dans une classe de "
                    "`chunk/` ou `fast_allocators/` : le corps est dans le compagnon `<en-tête>_inline.h` de l'en-tête |\n")
DOC_FR_RULE_ANCHOR = "- **Pas de préprocesseur"
DOC_FR_RULE = """- **Les corps longs vont dans le compagnon `<en-tête>_inline.h`.** Une méthode dont le corps dépasse
  5 lignes est déclarée dans la classe et définie dans `internal/<en-tête>_inline.h`, inclus en
  fin d'en-tête ; le commentaire Doxygen reste sur la déclaration. Les accesseurs de quelques
  lignes restent dans la classe. `check_inline_bodies` le vérifie ;
  `scripts/check_inline_bodies.py --apply` déplace ce qu'il liste (puis clang-format sur les fichiers).
"""


def insert_once(path: str, anchor: str, text: str, after_line_of_anchor: bool, marker: str) -> str:
    """Insert text at the anchor (after the anchor's line, or before it) unless marker is present."""
    if not os.path.exists(path):
        return f"{path}: not found, not patched"
    content = open(path, encoding="utf-8").read()
    if marker in content:
        return f"{path}: already patched"
    k = content.find(anchor)
    if k < 0:
        return f"{path}: anchor not found ({anchor[:40]!r}), not patched"
    if after_line_of_anchor:
        k = content.index("\n", k) + 1
    content = content[:k] + text + content[k:]
    open(path, "w", encoding="utf-8").write(content)
    return f"{path}: patched"


def install_check(root: str) -> list[str]:
    """Write scripts/check_inline_bodies.py (this file), register it in CMake and the docs."""
    notes = []
    dst = os.path.join(root, CHECK_SCRIPT)
    me = open(os.path.abspath(__file__), encoding="utf-8").read()
    if not os.path.exists(dst) or open(dst, encoding="utf-8").read() != me:
        open(dst, "w", encoding="utf-8").write(me)
        os.chmod(dst, 0o755)
        notes.append(f"{CHECK_SCRIPT}: written")
    else:
        notes.append(f"{CHECK_SCRIPT}: up to date")
    notes.append(insert_once(os.path.join(root, "CMakeLists.txt"), CMAKE_ANCHOR, CMAKE_BLOCK, True,
                             "check_inline_bodies"))
    notes.append(insert_once(os.path.join(root, "docs/contributing.md"), DOC_EN_TABLE_ANCHOR, DOC_EN_TABLE_ROW,
                             True, "`check_inline_bodies` |"))
    notes.append(insert_once(os.path.join(root, "docs/contributing.md"), DOC_EN_RULE_ANCHOR, DOC_EN_RULE, False,
                             "Long bodies go to"))
    notes.append(insert_once(os.path.join(root, "docs/contributing_fr.md"), DOC_FR_TABLE_ANCHOR, DOC_FR_TABLE_ROW,
                             True, "`check_inline_bodies` |"))
    notes.append(insert_once(os.path.join(root, "docs/contributing_fr.md"), DOC_FR_RULE_ANCHOR, DOC_FR_RULE, False,
                             "Les corps longs vont"))
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="repository root (default: current directory)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="list the bodies that break the rule, exit 1 if any")
    mode.add_argument("--apply", action="store_true", help="move them (the default when run as apply_*.py)")
    mode.add_argument("--dry-run", action="store_true", help="say what --apply would move")
    ap.add_argument("--threshold", type=int, default=THRESHOLD, help="bodies longer than this move (default 5)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if not os.path.isdir(os.path.join(root, SCOPES[0][0])):
        sys.exit(f"{a.root}: not the repository root ({SCOPES[0][0]} not found)")
    check = a.check or (not a.apply and not a.dry_run and
                        os.path.basename(sys.argv[0]).startswith("check_"))
    moved, kept = migrate(root, a.threshold, dry_run=(check or a.dry_run), verbose=a.verbose)
    if check:
        if moved:
            print(f"{len(moved)} member function bodies longer than {a.threshold} lines are defined in their class;")
            print(f"run `python3 {CHECK_SCRIPT} --apply` from the repository root to move them:")
            for h, name, nb, comp in moved:
                print(f"  {h}: {name} ({nb} lines)")
            sys.exit(1)
        print(f"check_inline_bodies: no member function body longer than {a.threshold} lines in a class")
        return
    by_file: dict[str, list] = {}
    for h, name, nb, comp in moved:
        by_file.setdefault(h, []).append((name, nb, comp))
    new_files = []
    for h, lst in by_file.items():
        comp = lst[0][2]
        print(f"{h}: {len(lst)} bodies -> {comp}")
        if a.verbose:
            for name, nb, _ in lst:
                print(f"    {nb:4d}  {name}")
        if not a.dry_run and comp not in new_files and \
                subprocess_untracked(root, comp):
            new_files.append(comp)
    print(f"{'would move' if a.dry_run else 'moved'}: {len(moved)}; left in class: {len(kept)}")
    for h, name, nb, why in kept:
        print(f"  kept  {h}: {name} ({nb} lines): {why}")
    if a.dry_run:
        return
    for note in install_check(root):
        print(note)
    if subprocess_untracked(root, CHECK_SCRIPT):
        new_files.append(CHECK_SCRIPT)
    if new_files:
        print("new files to add:\n  git add " + " ".join(new_files))
    print("then: python3 scripts/check_format.py --fix   (clang-format 23) and rebuild.")


def subprocess_untracked(root: str, rel: str) -> bool:
    """True if rel is not tracked by git (or git is unavailable)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", rel],
                           capture_output=True, text=True)
        return r.returncode != 0
    except OSError:
        return True


if __name__ == "__main__":
    main()
