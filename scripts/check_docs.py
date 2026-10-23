#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The documentation still matches the code, and the two languages each other.

Five checks, over README.md, CHANGELOG.md and docs/*.md, and over the
comments of the sources:

1. Links: every relative link to a file ([text](path), path without a scheme)
   points to a file or directory that exists.
2. Names: every `code span` that looks like a code name -- a macro, a
   variable, a function, a type, a file -- is found in the source tree
   (chunk/, fast_allocators/, examples/, scripts/, python/, cmake/ and the
   CMakeLists.txt). A renamed or removed name leaves its documentation
   behind; this is how it is found. A span made of several words, a command
   line or a value (`0`, `-DOMP=ON`, `256;1024`) is not a name and is not
   checked; ALLOWED lists the names that are real but live outside the tree
   (standard library, OpenMP, Linux, tools).
3. Pairs: a document and its French version (x.md / x_fr.md, or the explicit
   pairs below, and README.md / README_fr.md) have the same skeleton: the same number of headings at each
   level, of table rows and of code fences. A section, a table row or an
   example added in one language only shows up here.
4. History: no development-history marker in a comment of the C++ sources,
   the scripts or the CMake files (fwNN, "SEGV hunt", "BUG FIX", "FIX (", a
   date such as "Sept '26" or 2026-09-14): comments state the rule a fix
   taught, not the story of the fix.
5. Paths: a file of the tree cited in such a comment (chunk/..., scripts/...,
   docs/...) exists, from the repository root or from an include directory.
   A file moved or removed leaves its mentions behind; this is how they are
   found.

Usage: scripts/check_docs.py [repo root] [--names-verbose]   (exit 1 on a failure)
"""

import os
import pathlib
import re
import subprocess
import sys

SOURCE_DIRS = ["chunk", "fast_allocators", "examples", "scripts", "python", "cmake", "formal"]
SOURCE_EXT = {".h", ".hpp", ".cpp", ".cxx", ".c", ".in", ".py", ".sh", ".txt", ".cmake"}

# English document -> French document, where the names differ.
PAIRS = {
    "variants.md": "variantes_fr.md",
    "adaptive_rechunk.md": "rechunk_adaptatif_fr.md",
    "environment_variables.md": "variables_environnement_fr.md",
    "developer_settings.md": "reglages_developpeur_fr.md",
    "performance_measurement.md": "mesures_performance_fr.md",
}

# Documents with no French version, and why.
ENGLISH_ONLY = {
    # the threads branch's working plan, status and decisions: it goes with the
    # branch, not into a release
    "threading_backends.md",
    # the plan of the pattern::Array variant: replaced by algo_ documents when it exists
    "pattern_design.md",
}

# Real names that are not defined in this tree.
ALLOWED = {
    # removed names that CHANGELOG.md records as removed
    "FAST_ALLOC_MAX_CONST_BLOCK_SIZE", "internal/static_const_array_define.h",
    "chunk_config::have_aggressive_compression",  # now chunk_config::aggressive_compression()
    # renamed to PascalCase (ChunkTemplateOf, ForEachTypeImpl, ...), which CHANGELOG.md records
    "index_type", "cst_val1_type", "chunk_template_of", "for_each_type_impl",
    # former class names the CHANGELOG cites (now SecondValue*Base, zero_any::ChunkOmpPolicy, ...)
    "Chunk2Base", "Proxy2Base", "ArrayChunk2Base", "Chunk2AOmpPolicy", "RechunkHooks2", "Chunk2Def",
    # files a shrink_gui run writes in its directory on the cluster (docs/shrink_gui.md)
    "run.sh", "job.sh", "launch.sh",
    # C++ standard library, compiler, language
    "std", "size_t", "int64_t", "uint32_t", "operator", "noexcept", "constexpr", "seq_cst",
    "memory_order_seq_cst", "memory_order_relaxed", "memory_order_acquire", "memory_order_release",
    "atomic_thread_fence", "atomic_signal_fence", "call_once", "once_flag", "unique_ptr", "make_unique",
    "lower_bound", "binary_search", "bit_cast", "as_const", "has_unique_object_representations",
    "emplace_back", "push_back", "pop_back", "shrink_to_fit", "begin", "end", "rbegin", "rend",
    "sort", "data", "capacity", "reserve", "resize", "clear", "assign", "swap", "front", "back", "at",
    "size", "empty", "complex", "array", "vector", "deque", "string", "operator==", "operator<",
    "operator<=>", "_GLIBCXX_ASSERTIONS", "__FAST_MATH__", "_OPENMP", "__GLIBC__", "__GNUC__",
    "__NVCOMPILER", "auto", "const", "static_assert",
    # OpenMP, OMPT
    "omp_get_max_threads", "omp_get_num_procs", "omp_get_thread_num", "omp_get_thread_limit",
    "omp_get_ancestor_thread_num", "omp_set_num_threads", "omp_in_parallel", "ompt_start_tool",
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_TOOL", "OMP_TOOL_LIBRARIES",
    "OMP_TOOL_VERBOSE_INIT", "gomp_barrier_wait_end", "parallel_begin",
    # oneTBB
    "task_scheduler_observer",
    # Linux, glibc, tools
    "membarrier", "MEMBARRIER_CMD_PRIVATE_EXPEDITED", "MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED",
    "madvise", "MADV_DONTNEED", "MADV_HUGEPAGE", "MADV_COLLAPSE", "mincore", "mprotect", "mmap",
    "munmap", "msync", "posix_memalign", "malloc", "malloc_trim", "svfree", "getrusage", "ru_maxrss",
    "VmRSS", "statm", "smaps_rollup", "sched_yield", "pthread_create", "mfence", "dmb", "ldar",
    "ldr", "str", "stlr", "pmap", "perf", "strace", "nm", "ldd", "strings", "buddyinfo",
    "libgomp", "libomp", "libimf", "libamath", "libmvec", "libarcher",
    "CMAKE_BUILD_TYPE", "CMAKE_INSTALL_PREFIX", "CMAKE_PREFIX_PATH", "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMakeCache.txt", "find_package", "target_link_libraries", "add_definitions", "add_test",
    "cmake", "ninja", "ctest", "spack", "doxygen", "Doxygen",
    # Words of the documents that look like names, and pseudo-code
    "dirty", "written", "tick", "first_value", "second_value", "values", "flag_A", "flag_B", "_fr.md",
}

# Names the build GENERATES, so that they appear nowhere whole in the sources:
# the variant libraries (chunk_zero_any_256, chunk::chunk_zero_any_dyn, ...) and the
# per-variant executables (shrink_bench_zero_any_256, validation_zero_dyn, ...).
GENERATED = re.compile(r"^(chunk::)?chunk_(zero|zero_one_value|zero_any|v1|v2|v2a)(_(\d+|dyn))?$"
                       r"|^(shrink_bench|validation|variant_compare|rechunk_validation|dyn_geometry_validation)"
                       r"_(zero|zero_one_value|zero_any|chunk|chunk2|chunk2a|vector)(_\w+)?$")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\]\(([^)\s]+)\)")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_~][A-Za-z0-9_]*)*(\(\))?$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_./-]+\.(h|hpp|cpp|py|sh|md|txt|in|cmake|supp)$")
HISTORY = re.compile(r"\bfw\d+[a-z]?\b|SEGV hunt|BUG FIX|\bFIX \(|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.? '\d\d\b|\b20\d\d-\d\d-\d\d\b")
CITED_PATH = re.compile(r"(?<![\w/.-])((?:chunk|fast_allocators|examples|scripts|python|docs|cmake)/"
                        r"[A-Za-z0-9_./-]*[A-Za-z0-9_]\.(?:h|hpp|cpp|py|sh|md|txt|cmake))\b")


def tracked(root: pathlib.Path):
    """The files of the tree: git's list (tracked and new, ignored ones excluded)
    when there is one, else a walk."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True).stdout
        files = [root / f for f in out.split("\n") if f]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and not x.startswith("cmake-build") and x != "build"]
        found += [pathlib.Path(d) / n for n in names]
    return found


def docs(root, files):
    return sorted(f for f in files
                  if f.suffix == ".md" and (f.parent == root / "docs" or f.parent == root))


def check_links(root, md_files):
    bad = []
    for f in md_files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for target in LINK.findall(CODE_SPAN.sub("", line)):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if path and not (f.parent / path).exists():
                    bad.append("%s:%d: broken link %s" % (f.relative_to(root), n, target))
    return bad


def source_words(root, files):
    words, names = set(), set()
    for f in files:
        rel = f.relative_to(root)
        if rel.parts[0] not in SOURCE_DIRS and rel.name != "CMakeLists.txt":
            continue
        names.add(rel.name)
        names.add(str(rel))
        if f.suffix not in SOURCE_EXT and f.name != "CMakeLists.txt":
            continue
        try:
            words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass
    return words, names


def check_names(root, md_files, words, names, verbose):
    bad = []
    for f in md_files:
        in_fence = False
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for span in CODE_SPAN.findall(line):
                span = span.strip()
                if FILE_NAME.match(span):
                    base = span.rsplit("/", 1)[-1]
                    if base in names or span in names or (root / span).exists() or span in ALLOWED:
                        continue
                    if any(generated_or_file(root / d / span) for d in INCLUDE_DIRS):
                        continue
                    if f.name.endswith(".md") and (f.parent / span).exists():
                        continue
                    bad.append("%s:%d: file `%s` not found in the tree" % (f.relative_to(root), n, span))
                    continue
                if not NAME.match(span):
                    continue
                last = span.removesuffix("()").split("::")[-1].lstrip("~")
                # Short or all-lowercase single words are prose more often than code.
                if len(last) < 4 or (last.islower() and "_" not in last and "::" not in span):
                    continue
                if last in words or last in ALLOWED or span in ALLOWED or GENERATED.match(span):
                    continue
                bad.append("%s:%d: `%s` not found in the sources" % (f.relative_to(root), n, span))
    if not verbose:
        # One line per (document, name): a name repeated in a document is one finding.
        seen, out = set(), []
        for b in bad:
            key = (b.split(":", 1)[0], b.split("`")[1] if "`" in b else b)
            if key not in seen:
                seen.add(key)
                out.append(b)
        bad = out
    return bad


def skeleton(path):
    heads, rows, fences = {}, 0, 0
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fences += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6}) ", line)
        if m:
            heads[len(m.group(1))] = heads.get(len(m.group(1)), 0) + 1
        elif line.startswith("|"):
            rows += 1
    return heads, rows, fences


def check_pairs(root):
    bad = []
    d = root / "docs"
    for en in sorted(d.glob("*.md")):
        if en.name.endswith("_fr.md") or en.name == "README.md" or en.name in ENGLISH_ONLY:
            continue
        fr = d / PAIRS.get(en.name, en.name[:-3] + "_fr.md")
        if not fr.exists():
            bad.append("docs/%s: no French version (%s)" % (en.name, fr.name))
            continue
        se, sf = skeleton(en), skeleton(fr)
        if se != sf:
            bad.append("docs/%s vs %s: headings %s / %s, table rows %d / %d, code fences %d / %d"
                       % (en.name, fr.name, se[0], sf[0], se[1], sf[1], se[2], sf[2]))
    for fr in sorted(d.glob("*_fr.md")):
        stem = fr.name[:-6] + ".md"
        en = next((e for e, f in PAIRS.items() if f == fr.name), stem)
        if not (d / en).exists():
            bad.append("docs/%s: no English version (%s)" % (fr.name, en))
    # The top-level README and its French version.
    en, fr = root / "README.md", root / "README_fr.md"
    if en.exists() and fr.exists():
        se, sf = skeleton(en), skeleton(fr)
        if se != sf:
            bad.append("README.md vs README_fr.md: headings %s / %s, table rows %d / %d, code fences %d / %d"
                       % (se[0], sf[0], se[1], sf[1], se[2], sf[2]))
    elif en.exists():
        bad.append("README.md: no French version (README_fr.md)")
    return bad


COMMENT_DIRS = ("chunk", "fast_allocators", "examples", "scripts")
# third_party/: the vendored headers, included as "chunk/vendor/...".
INCLUDE_DIRS = ("", "chunk/public", "fast_allocators/public", "third_party")


def comment_lines(root, files):
    """(path, line number, line) for the comment lines of the C++ sources, the
    scripts and the CMake files of COMMENT_DIRS and of the root CMakeLists.txt."""
    for f in files:
        rel = f.relative_to(root)
        if rel.parts[0] not in COMMENT_DIRS and rel.name != "CMakeLists.txt":
            continue
        if f.suffix in (".h", ".hpp", ".cpp"):
            is_comment = lambda line: "//" in line or line.lstrip().startswith("*")
        elif f.suffix in (".sh", ".py", ".cmake") or f.name == "CMakeLists.txt":
            is_comment = lambda line: line.lstrip().startswith("#") and not line.startswith("#!")
        else:
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if is_comment(line):
                yield f, n, line


def generated_or_file(path):
    """The file exists, or CMake generates it from path + ".in" (configure_file:
    chunk/build_options.h from chunk/public/chunk/build_options.h.in)."""
    return path.exists() or path.with_name(path.name + ".in").exists()


def check_history(root, lines):
    return ["%s:%d: history marker in a comment: %s" % (f.relative_to(root), n, line.strip()[:100])
            for f, n, line in lines if HISTORY.search(line)]


def check_cited_paths(root, lines):
    bad = []
    for f, n, line in lines:
        for path in CITED_PATH.findall(line):
            if not any(generated_or_file(root / d / path) for d in INCLUDE_DIRS) and not (f.parent / path).exists():
                bad.append("%s:%d: no file %s" % (f.relative_to(root), n, path))
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose = "--names-verbose" in sys.argv
    root = pathlib.Path(args[0] if args else ".").resolve()
    files = tracked(root)
    md_files = docs(root, files)
    words, names = source_words(root, files)
    comments = list(comment_lines(root, files))
    sections = [
        ("links", check_links(root, md_files)),
        ("names", check_names(root, md_files, words, names, verbose)),
        ("French / English pairs", check_pairs(root)),
        ("history markers", check_history(root, comments)),
        ("paths in comments", check_cited_paths(root, comments)),
    ]
    failed = 0
    for title, bad in sections:
        print("%-24s %s" % (title, "OK" if not bad else "%d problem(s)" % len(bad)))
        for b in bad:
            print("    " + b)
        failed += len(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
