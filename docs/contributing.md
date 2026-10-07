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
time. A Debug build, `-DCHECK_OPTIM=ON` or a sanitizer also builds the validation harnesses
(`examples/validation`).

The whole matrix (GNU, LLVM, oneAPI, AOCC, NVHPC, sanitizers, statistics, std-alloc...) is
`scripts/compile_all.sh`. Its options are in [environment_variables.md](environment_variables.md)
§8. The ones used every day:

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
| `check_headers` | every C++ file opens with the Apache license header (years up to date) and carries no IDE `Created by` stamp; scripts and build files (`.py`, `.sh`, `.cmake`, `CMakeLists.txt`) carry `# Copyright ... CEA` and `# SPDX-License-Identifier: Apache-2.0`; the vendored headers, their upstream SPDX line; `scripts/check_headers.py --fix` repairs |
| `check_raw_reads` | every raw read of a chunk buffer in the library says why it is safe |
| `check_docs` | links, code names cited in the documents, English/French pairs, no history markers in comments, the files cited in comments exist |

## 3. Code rules

- **C++20.** The `cpp23` branch carries the same code with C++23 required.
- **English** for code, comments and commit messages.
- **clang-format** with the repository's `.clang-format`, on the lines you touch.
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
  [environment_variables.md](environment_variables.md).
