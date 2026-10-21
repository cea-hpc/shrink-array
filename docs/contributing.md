# Contributing to shrink-array

> French version: [contributing_fr.md](contributing_fr.md).

> October 2026. How to build, test, measure and document a change. Read
> [algo_chunk2.md](algo_chunk2.md) first: most of the library is the machinery it describes.

***

## 1. Build

A development build of one configuration:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DOMP=ON -DCHUNK_STATIC_SIZES=256
cmake --build build
```

`-DCHUNK_STATIC_SIZES=256` builds one static size instead of two, which saves much of the build
time; `-DCHUNK_VARIANTS=chunk2a` builds one variant instead of three, with its examples and
harnesses only. A Debug build, `-DCHECK_OPTIM=ON` or a sanitizer also builds the validation harnesses
(`examples/validation`).

The whole matrix (GNU, LLVM, oneAPI, AOCC, NVHPC, sanitizers, statistics, std-alloc...) is
`scripts/compile_all.sh`. Its options are in [developer_settings.md](developer_settings.md)
§7. The ones used every day:

| option | use |
|---|---|
| `--target=<glob>` | build only some configurations (a shell glob, quoted) |
| `--exec=<glob>` | build only some executables in each configuration |
| `--fresh-cache` | after a change to a `CMakeLists.txt` or to an option, so no stale cached value is used |
| `--jobs=<n>` | fewer configurations at once when the node runs out of memory (`CHUNK_MEM_PER_JOB_MB` sets the memory per compiler) |
| `--with-ledger` | add the Debug build with the block ledger (`CHUNK_FORCE_LEDGER=ON`), for allocator work |

## 2. Test

`ctest` in a build directory runs the harnesses and the source checks; `ctest -L validation`
only those. `python/bench_and_test.py` runs them over a set of builds, with the benchmark in
check mode and the performance sweep (see `docs/example_bench_test.txt`).

The defects of this library are mostly **rare races**. A change to a concurrent path is not
tested until it has run under ThreadSanitizer, at several thread counts, and many times: use
`python/multi_launch_stress.py` (see `docs/example_multi_launch_stress.txt`).

What to run, by what the change touches:

| change | run at least |
|---|---|
| write engine, Dekker handshake, compression manager, deferred writes | `deferred_path_validation`, `read_reclaim_validation` (also with `CHUNK_TEST_READ_DELAY_EVERY`, and under ASan in a std-alloc build), the TSan builds, `multi_launch_stress.py` |
| compression marking, the passes | `mark_path_validation`, `compress_roundtrip`, the benchmark in check mode |
| allocators, reuse lists, slabs | `pool_accounting`, `slab_geometry_validation`, `blk_allocator_validation`, the ledger build (`--with-ledger`), ASan in a std-alloc build |
| re-chunk, dynamic geometry | `rechunk_validation`, `dyn_geometry_validation`, the benchmark with `CHUNK_FORCE_RECHUNK_TOGGLE=1` |
| range operations | `fill_range_mt_validation`, `fill_range_mt_v1v2_validation`, `range_counters_validation` |
| thread ranks, regions | `thread_growth_validation`, `nested_region_validation` |
| container interface | `iterator_validation`, `swap_move_validation`, `stale_proxy_validation`, `open_types_validation`, `index64_validation` |
| documentation | `check_docs` |

The source checks run with the others (`ctest -R check_`):

| test | rule |
|---|---|
| `check_includes` | every `#include` says why it is there; system and project headers are not mixed in one block |
| `check_naming` | the naming convention of section 3, as far as a line shows it: snake_case constants, enumerators, functions and variables, PascalCase types, aliases and template parameters, no reserved `__` name |
| `check_headers` | every C++ file opens with the Apache license header (years up to date) and carries no IDE `Created by` stamp; scripts and build files (`.py`, `.sh`, `.cmake`, `CMakeLists.txt`) carry `# Copyright ... CEA` and `# SPDX-License-Identifier: Apache-2.0`; the vendored headers, their upstream SPDX line; `scripts/check_headers.py --fix` repairs |
| `check_raw_reads` | every raw read of a chunk buffer in the library says why it is safe |
| `check_tracked` | every file the sources name (a quoted `#include`, a relative Python import, a CMake source or harness, a Markdown link) is tracked by git, not only present on the disk |
| `check_docs` | links, code names cited in the documents, English/French pairs, no history markers in comments, the files cited in comments exist |

## 3. Code rules

- **C++20.** The `cpp23` branch carries the same code with C++23 required.
- **English** for code, comments and commit messages.
- **No fixup commit is pushed.** A `fixup!`, `squash!` or `amend!` commit is squashed into its
  target before the push (`git rebase -i --autosquash origin/<branch>`): once published it can
  only stay. The CI refuses a push that brings one.
- **clang-format** with the repository's `.clang-format`, on the lines you touch.
- **Names.** One convention, the standard library's for what it names and PascalCase for the
  types:

  | what | case | example |
  |---|---|---|
  | functions, methods, variables, parameters, namespaces | snake_case | `get_chunk_list`, `num_tab` |
  | private and protected data members | snake_case ending in `_` | `chunk_list_`, `num_` |
  | constants (`constexpr`) and enumerators | snake_case | `cst_buffer_size`, `ev_flush_begin` |
  | classes, structs, concepts, type aliases, template parameters | PascalCase | `ArrayChunk2A`, `AllocT`, `ChunkT` |
  | macros | UPPER_CASE | `CHUNK_SIZE` |

  UPPER_CASE is for macros only: a constant or an enumerator spelled like one reads as one, and
  may meet one. The standard library's shapes keep their spelling: the member types of its
  protocols and the `xxx_type` ones (`value_type`, `chunk_type`), the traits and tags (`is_cplx`,
  `rebind`). Inside a snake_case name the variant is `chunk2a`; file names (`array_chunk2A.h`)
  stay as they are. Globals take a `g_` prefix; no name starts with `__` (reserved). A public name
  that changes keeps its former spelling as a `[[deprecated]]` alias until 1.0.
  `check_naming` enforces what a line shows; clang-tidy's `readability-identifier-naming`
  checks the rest (the member suffix, the parameters) with `scripts/clang_tidy_naming.yaml`.
- **No preprocessor where `if constexpr` works.** The rules that decide which `#ifdef` can go,
  and the ones that must stay, are written at the top of `examples/shrink_bench/bench_config.h`.
- **Comments state the rule, not the story.** Write why the code is the way it is, and what
  breaks if it changes. Do not write how the bug was found, which version had it, run counts,
  dates or ticket names: that belongs in the commit message. `check_docs` rejects the usual
  markers (`fwNN`, `SEGV hunt`, `BUG FIX`, `FIX (`, a date like `Sept '26` or `2026-09-14`), in
  the C++ sources, the scripts and the CMake files.
- **Doxygen** (`@brief`, `@param`, `@return`, `@warning`) on everything a user calls: the
  classes and functions of [api.md](api.md). `cmake --build build --target doc` must stay
  free of warnings (`ctest -R check_doxygen` when Doxygen is installed).
- **Lock-free code**: follow the rules of [synchronisation_dekker.md](synchronisation_dekker.md)
  §6, and write the ordering argument next to every atomic that needs one.
  When the change touches a handshake, run `formal/run_genmc.sh` too
  ([formal_models.md](formal_models.md)).

## 4. Performance claims

A change made for speed or memory comes with a measurement taken the way
[performance_measurement.md](performance_measurement.md) describes: memory with the exact
counter, time on the same binary, the modes alternated, at least three rounds, and only a gap
well above the noise (10% on the 64-thread dynamic benchmarks) is a result.

## 5. Documentation and changelog

- A user-visible change updates the documents it touches, **in both languages**. English is the
  reference; the French version follows. `check_docs` fails when the two drift apart in
  structure.
- Add an entry under _Unreleased_ in `CHANGELOG.md` (Added, Changed, Deprecated, Removed,
  Fixed). The versioning policy at the top of that file says which number the next release
  bumps.
- A new environment variable, macro or CMake option goes into
  [environment_variables.md](environment_variables.md) if a user of the library has to set it,
  into [developer_settings.md](developer_settings.md) otherwise. A switch that keeps an old
  behaviour for an A/B gets its removal date there when it is added.
