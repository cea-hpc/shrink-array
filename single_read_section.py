#!/usr/bin/env python3
"""Reads: one read section per element read, not two.

Run from the repository root:  python3 single_read_section.py
Idempotent: a second run reports every edit as already done.

ChunkBase::operator[](int) const -- the read of every variant -- was
    CheckNaN<do_check>::maybe(element_at(i), i);
    return element_at(i);
Outside check builds maybe() is empty, but its argument is still evaluated:
element_at() opens a read_reclaim::ReadSection (sequence store, light fence,
acquire load of `values`, release store of the sequence), and that ordering
cannot be dropped by the compiler. Every read paid two sections; perf annotate
of access_cost on Grace shows them (the first one's `values` load discarded,
`ldapr xzr`), and the release store at their end as the hottest instruction
of the read loop.

Now one element_at(), its value checked (check builds) and returned -- the
check looks at the value actually returned, not at a second read of it.
"""
import pathlib
import sys

NEW_FILES = {}

EDITS = [
    ('chunk/public/chunk/internal/chunk_base.h',
     '        chunk_config::CheckNaN<chunk_config::do_check>::maybe(element_at(i), static_cast<long>(i));\n\n        return element_at(i);\n    }',
     '        // ONE element_at(): maybe() is a no-op outside check builds, but its\n        // argument is still evaluated -- a whole read section (two sequence\n        // stores, one of them a release, and an acquire load of `values`) whose\n        // ordering the compiler must keep. Every read paid two (perf annotate,\n        // Grace: the release store of the second section held the read).\n        const T v = element_at(i);\n        chunk_config::CheckNaN<chunk_config::do_check>::maybe(v, static_cast<long>(i));\n        return v;\n    }',
     1),
]


def main() -> int:
    root = pathlib.Path.cwd()
    if not (root / "chunk" / "public" / "chunk").is_dir():
        print("error: run this from the repository root", file=sys.stderr)
        return 1
    for path, text in NEW_FILES.items():
        p = root / path
        if p.exists() and p.read_text() == text:
            print("unchanged  " + path)
        else:
            p.write_text(text)
            print("written    " + path)
    texts, applied, done, skipped = {}, 0, 0, []
    for path, old, new, count in EDITS:
        if path not in texts:
            texts[path] = (root / path).read_text()
        t = texts[path]
        # Done when the new text is there -- or, for an insertion (old kept
        # inside new), when the inserted text is there: another script may
        # since have inserted its own entry at the same anchor.
        added = new.replace(old, "", 1) if old in new else ""
        if (new and new in t) or (added.strip() and added in t) or (not new and old not in t):
            done += 1
        elif t.count(old) == count:
            texts[path] = t.replace(old, new)
            applied += 1
        else:
            skipped.append((path, old))
    for path, t in texts.items():
        (root / path).write_text(t)
    print("applied %d/%d edits, %d already done" % (applied, len(EDITS), done))
    for path, old in skipped:
        print("SKIPPED in %s: block starting %r" % (path, old.split("\n", 1)[0][:80]), file=sys.stderr)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
