# Changelog

All notable changes to shrink-array. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the policy below.

## Versioning policy

Two version numbers, both `MAJOR.MINOR.PATCH`:

| component | where it is set | installed as |
|---|---|---|
| the chunk library | `chunk/CMakeLists.txt` (`CHUNK_MAJOR_VERSION`, `CHUNK_MINOR_VERSION`, `CHUNK_PATCH_VERSION`) | the `chunk` CMake package (`find_package(chunk X.Y)`) |
| fast_allocators | `fast_allocators/CMakeLists.txt` (`ALLOCATORS_*_VERSION`) | the `fast_allocators` package |

The **public interface** that the version describes is:

- the headers of `chunk/public/chunk/` that are not in `internal/`, and what they declare;
- the installed CMake targets (`chunk::chunk_v2a_256`, `chunk::chunk_v2a`, ...) and the
  variables `find_package(chunk)` sets;
- the CMake options and the environment variables documented as user settings
  (`docs/environment_variables.md` §1 and §7);
- the documented behaviour: what is compressed, the thread contract, the sequential points.

Everything under `internal/`, the benchmark, the harnesses and the scripts can change in any
release.

| bump | when | examples |
|---|---|---|
| **MAJOR** | a change that can break a correct consumer: a public name, target, option or variable removed or renamed with no compatibility path, a changed contract | removing `ArrayChunk2::set_second_value`; dropping `chunk::chunk_v1` |
| **MINOR** | an addition, or a change with a compatibility path (the old name kept, deprecated) | new targets installed; `HAVE_AGRESSIVE_COMPRESSION` renamed with the old spelling still accepted |
| **PATCH** | a fix that changes no interface | a race fixed; a wrong value in a trace |

The installed package declares `SameMajorVersion` compatibility: `find_package(chunk 2.2)`
accepts any 2.x at or above 2.2. A deprecated name stays for at least one MINOR release and is
removed only by the next MAJOR.

**fast_allocators** follows the same rules for its own interface. The chunk library pins the
exact fast_allocators version it was built with (`CHUNK_FAST_ALLOCATORS_VERSION` in
`chunk/CMakeLists.txt`, checked at configure time): bumping fast_allocators means bumping that
pin too, and at least the PATCH of chunk.

**Releasing**: move the entries of *Unreleased* under the new version and date, set the
numbers in the two `CMakeLists.txt`, tag `vX.Y.Z` (the Spack recipe downloads tags), and give
the tag's checksum to the recipe.

## Unreleased

### Added

- Every variant library is installed and exported under `chunk::`: `chunk::chunk_v1_<size>`,
  `chunk::chunk_v2_<size>`, `chunk::chunk_v2a_<size>`, the `_dyn` ones, the bare
  `chunk::chunk_v1` / `chunk::chunk_v2` / `chunk::chunk_v2a` (default size) and
  `chunk::chunk_vector`. Before, `find_package(chunk)` provided `chunk::chunk` alone, with no
  variant.
- `find_package(chunk)` sets `chunk_STATIC_SIZES`, `chunk_DEFAULT_SIZE`, `chunk_HAS_DYN`,
  `chunk_OPENMP`, `chunk_OMPT` and `chunk_STD_ALLOC`, and finds Threads and OpenMP itself.
- `examples/getting_started`: a first program, built in the tree (ctest `getting_started`) and
  against an install.
- `docs/getting_started.md` (and its French version): install, link from CMake, first program,
  OpenMP rules, pitfalls.
- English versions of every design document, and a bilingual docs index (`docs/README.md`).
- `docs/synchronisation_dekker.md`: the Dekker handshake, symmetric and asymmetric
  (`membarrier`), and its three uses in the library.
- `scripts/check_docs.py` (ctest `check_docs`): links, code names cited in the documents,
  French/English document pairs, history markers in the comments of the sources, scripts
  and CMake files, and the files those comments cite.
- `docs/api.md`: the API by task, on one page; `docs/contributing.md` and `CONTRIBUTING.md`:
  build, tests, code rules, measurements, docs (both with a French version).
- `scripts/check_doxygen.py` (ctest `check_doxygen`, when Doxygen is found): the
  documentation builds with no warning. The HTML draws the mermaid diagrams.
- `compile_all.sh` / `compile_all_gnu.sh`: `--with-ledger`.
- `scripts/check_headers.py` (ctest `check_headers`): every C++ file opens with the Apache
  license header, its years up to date, and carries no IDE `Created by` stamp; `--fix` repairs.

### Changed

- The variant libraries carry C++20, OpenMP (when built with it) and `HAVE_OMPT` to their
  consumers (PUBLIC). An OMPT build links its consumers with `-rdynamic`, so the OpenMP runtime
  finds the library's OMPT tool in the executable.
- `HAVE_AGRESSIVE_COMPRESSION` is renamed `HAVE_AGGRESSIVE_COMPRESSION` (option, macro,
  `chunk_config::have_aggressive_compression`, build directory names). CMake still accepts the
  old option name, with a deprecation message.
- The `cmake-build-omp-gnu-debug-ledger` build is no longer part of the default
  `compile_all.sh` run (`--with-ledger`, or a `--target` that selects it).
- The `[RECHUNK-POOL]` trace labels key 0 `(shared pool)`; it said `never swept`, which was
  wrong.
- Source comments rewritten throughout: the development history is gone, the rules it taught
  are kept. Comments that contradicted the code are fixed.
- `docs/`: refreshed; the variables, options and build-script options that were missing are
  documented.
- Vendored ankerl::unordered_dense updated from 5.0.1 to 5.3.1, its two local patches (nvc++)
  carried over: faster insert / try_emplace hits, lookups in an empty table without a test, a
  64-byte map instead of 80.
- The vendored hash tables install under `include/chunk/vendor/` instead of the root of
  `include/`, where they overwrote, or were shadowed by, a user's own copy of the same
  library. In the tree they moved to `third_party/chunk/vendor/` (for instance
  `third_party/chunk/vendor/ankerl/unordered_dense.h`), and chunk's headers include them as
  `"chunk/vendor/..."`.
- `NOTICE` (new) lists the third-party code and its licenses; it is installed with `LICENSE` and
  the license texts under `share/doc/chunk/`. `third_party/README.md` gives each vendored
  file's upstream, version and local patches.

### Fixed

- `scripts/check_includes.py` works outside a git checkout (a source archive, a Spack
  build) and checks files not yet added to git.
- Comments naming headers that moved to `internal/`, or a script that no longer exists.
- License headers: 29 files had none, others had it truncated, below `#pragma once` or with
  stale years; the IDE `Created by` stamps are gone.

### Deprecated

- The CMake option `HAVE_AGRESSIVE_COMPRESSION`: use `HAVE_AGGRESSIVE_COMPRESSION`.

## 2.2.8 and earlier

See the history of the repository.

