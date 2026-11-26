# Changelog

All notable changes to shrink-array. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the policy below.

## Versioning policy

Two version numbers, both `MAJOR.MINOR.PATCH`:

| component | where it is set | installed as |
|---|---|---|
| the chunk library | the top-level `CMakeLists.txt` (`project(shrink_array VERSION X.Y.Z)`) | the `shrink_array` CMake package (`find_package(shrink_array X.Y)`) |
| fast_allocators | `fast_allocators/CMakeLists.txt` (`ALLOCATORS_*_VERSION`) | the `fast_allocators` package |

The **public interface** that the version describes is:

- the headers of `chunk/public/chunk/` that are not in `internal/`, and the names of
  `docs/api.md` they declare: `scripts/public_api.txt` lists them, and `check_public_api` fails
  when one is added, moved, removed or renamed without the list. The members of the public
  classes that name the library's machinery (`ManagerType`, `Manager`, `ChunkType`,
  `ChunkListType`, `ProxyType`, `get_chunk_list()`) are not part of it;
- the installed CMake targets (`shrink_array::zero_any_256`, `shrink_array::zero_any`, ...) and
  the variables `find_package(shrink_array)` sets;
- the CMake options and the environment variables of `docs/environment_variables.md` (the user
  settings); those of `docs/developer_settings.md` are development tools, which can change in any
  release;
- the documented behaviour: what is compressed, the thread contract, the sequential points.

Everything under `internal/`, the benchmark, the harnesses and the scripts can change in any
release.

**Experimental until 1.0:** `chunk/mdspan.h` and `chunk/mdspan_tile_ops.h` (the `md::` views and
tile operations). Their names and signatures may change in a MINOR release, the change then listed
under Changed; the behaviour of the arrays behind a view keeps the policy above.

| bump | when | examples |
|---|---|---|
| **MAJOR** | a change that can break a correct consumer: a public name, target, option or variable removed or renamed with no compatibility path, a changed contract | removing `zero_one_value::Array::set_second_value`; dropping `chunk::chunk_zero` |
| **MINOR** | an addition, or a change with a compatibility path (the old name kept, deprecated) | new targets installed; a user option renamed with the old spelling still accepted |
| **PATCH** | a fix that changes no interface | a race fixed; a wrong value in a trace |

The installed package declares `SameMajorVersion` compatibility: `find_package(shrink_array 1.2)`
accepts any 1.x at or above 1.2. A deprecated name stays for at least one MINOR release and is
removed only by the next MAJOR.

**Before 1.0** (0.x versions), as SemVer allows, a MINOR release plays the role of a MAJOR one
(it may break) and PATCH covers additions and fixes; the package then declares
`SameMinorVersion`: `find_package(shrink_array 0.3)` accepts 0.3.x only.

**fast_allocators** follows the same rules for its own interface. The chunk library pins the
exact fast_allocators version it was built with (`CHUNK_FAST_ALLOCATORS_VERSION` in
`chunk/CMakeLists.txt`, checked at configure time): bumping fast_allocators means bumping that
pin too, and at least the PATCH of chunk.

**Releasing**: move the entries of *Unreleased* under the new version and date, set the
numbers in the two `CMakeLists.txt`, tag `vX.Y.Z` (the Spack recipe downloads tags), and give
the tag's checksum to the recipe.

## Unreleased

### Added

- `scripts/compile_all.sh --pgo=generate|use` (and `compile_all_gnu.sh`), CMake option `CHUNK_PGO`
  (`OFF`, `GENERATE`, `USE`) and `CHUNK_PGO_DIR`: profile-guided optimisation of the C++ code, in two
  passes on the same build directories (the optimised ones, as `--lto`). GCC, Clang and IntelLLVM
  (whose `.profraw` files are merged by `llvm-profdata` at configure time). On the development
  machine, shrink_bench zero_any trained on another size: 15 to 20 % off the total time.
- `scripts/compile_all.sh --lto` (and `compile_all_gnu.sh --lto`): link-time optimisation
  (`CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`) on the optimised builds, not on the Debug, sanitizer
  and nvhpc ones; every other configure gets it OFF. With a Clang-based compiler and LTO, the
  top-level CMakeLists links with lld instead of mold: mold and ld.bfd read LLVM bitcode only
  through LLVMgold.so, which some LLVM installs lack.
- `transform_ranges_mt(dst, first, last, op, a, b, ...)` (`chunk/range_ops.h`): `dst[i] = op(a[i], b[i], ...)`
  computed chunk by chunk. The sources -- chunked arrays of any variant, chunk size and element
  type, or contiguous containers -- are read by blocks, `op` runs in a plain loop over contiguous
  values, and each chunk span is stored with one write session; where every source is compressed
  on the span, `op` is called once and the span is filled with its result, so that the target
  chunk is not materialised. Callable from several threads on their own indices. A loop over two
  arrays written element by element, `c[i] = a[i] + k * b[i]`, costs 13 to 23 times a
  `std::vector`'s on the development machine (2 cores, GCC 13, `zero_any`, 2^25 doubles); through
  `transform_ranges_mt` 1.7 times on dense data, and a fifth to a quarter of the vector's time
  when nine chunks in ten are compressed. A source that is also the target is not supported.
  Behind it, `read_range_or_constant()`, the bulk read that reports a compressed chunk instead of
  copying it. New harness transform_ranges_validation.
- differential_validation, a harness that draws its operations: random sequences of container
  operations (writes, compound operations, `resize`, `push_back`, `assign`, `swap`, copies and
  moves, `compress`, `shrink_to_fit`, the range operations, `alloc_chunk`, export / import,
  parallel phases, `rechunk_to` at dyn) applied to an array and to a `std::vector`, the two read
  back in full after each. A difference prints the seed and the last operations. In the grouped
  executables, static and dyn, at 1 and 8 threads.

- The start of a core layer shared with a future GPU backend (docs/gpu_design.md, stage 1):
  `CHUNK_HD` (`chunk/internal/host_device.h`) is `__host__ __device__` in a CUDA or HIP
  translation unit and nothing otherwise. It marks `same_value`, `identity_less`,
  `ValueIdentity::same`, `log2int`, `StaticChunkSize` and `ChunkElems`, and the uniformity test
  of the compression scans, now the free function `internal::walk_constant`
  (`chunk/internal/constant_walk.h`) that `ChunkBase::walk_constant` calls. CPU builds are
  unchanged. `-DCHUNK_DEVICE_CORE_CHECK=ON` builds device_core_validation, which runs these
  functions in a CUDA kernel and compares them with the host, signed zeros and NaN payloads
  included, for float, double, `std::complex` and `std::array` (nvcc 13.1 and 13.3, clang 23).
  `same_value` and `identity_less` read the parts of a `std::complex` or a `std::array` through
  `internal::part()`, since `real()`, `imag()` and `std::array::operator[]` are host functions
  to nvcc. nvcc only warns about such a call, so the check builds with `-Werror=all-warnings`.

### Fixed

- `swap()` between two arrays of different chunk sizes (a run-time geometry build, one of them
  rechunked by `rechunk_to()` or the decider) aborted a check build ("mismatched chunk
  geometry") while a release build did it right, and while a move -- the same exchange, geometry
  included -- was accepted in both. The geometry goes with the chunks, as in a move: the check
  is gone, the documentation says so, and `swap_move_validation` is built at the run-time
  geometry too, with a swap after a rechunk.
- Serial builds, `zero` and `zero_or_value`: a value stored through the raw reference
  `operator()(i)` (after `alloc_chunk()`) could be lost. These builds decide to fold a chunk
  from per-chunk counters of its constants, kept by the writes through `operator[]` alone; the
  raw store left them claiming "all constant", and the next `operator[]` write of the constant
  into that chunk folded it, the raw value with it. The buffer is now counted before a fold:
  when it disagrees, the chunk stays and the counters are set from the buffer. One scan of a
  chunk per fold. Found by `differential_validation` (`bool`, seed 6); `alloc_chunk_validation`
  has the case (`mixed`). OpenMP builds were not affected (no counters).

## 0.4.0 -- 2026-11-23

A breaking release: every compatibility path for a former name is gone (see Removed).
fast_allocators is 2.5.0.

### Changed

- cpp26: the deleted copy / move members added under `chunk/` since 0.2.1 (the re-chunk refill,
  the registries' write lock, the stable table, two scope guards) give their reason again
  (`= delete("...")`), and so do the examples' (`multi_mat`, `shrink_bench`). `fast_allocators`
  is built as C++23 and its headers stay usable from C++23: its deletes keep the plain form.
- `ParamChunk::set_decalage_chunk()` is `set_chunk_shift()` (a no-op kept for old call sites, as
  before); the former name is not kept (see Removed). Internally ChunkSizePolicy's decalage() is
  `shift()`, and `print_info_chunk()` prints "chunk shift" instead of "decalage chunk".
- `shrink_gui` 0.18, "cette machine": the launcher defaults to
  `systemd-run --user --scope -q -p MemoryMax=90% -p MemorySwapMax=0` where `systemd-run --user`
  works, so that a run out of memory is killed alone instead of bringing WSL down (a launcher
  saved before stays as it was).
- `CHUNK_NATIVE_TUNING` defaults to OFF in a project that consumes the library (`add_subdirectory`,
  FetchContent): the library's objects no longer get `-march=native` / `-mcpu=native` while the
  project's do not; the project's own target flags apply. It sets `CHUNK_NATIVE_TUNING=ON` before
  including the library to have the former behaviour. Top level: unchanged (ON).
- The first and second values have the names the documentation and the statistics give them:
  the `const_val1` / `const_val2` (and `const2`) names of the chunks, proxies and registry
  follow. `is_const_val1()` / `is_const_val2()` are `is_first_value()` / `is_second_value()`,
  `get_const_val1()` / `get_const_val2()` are `first_value_ptr()` / `second_value_ptr()`,
  `ArrayConstVal2Holder` is `SecondValueRegistry` (`internal/second_value_registry.h`),
  `find_or_set_idx_const_val2()` is `find_or_register_second_value()`, and the others alike
  (all internal). The developer settings follow: `CHUNK_SECOND_VALUE_ID_LIMIT`,
  `CHUNK_SECOND_VALUE_RECLAIM`, and the compile-time `CHUNK_SECOND_VALUE_REGION_BYTES`,
  `CHUNK_SECOND_VALUE_MAX_IDS`, `CHUNK_SECOND_VALUE_LOOKUP_LOG2`; their `CHUNK_CONST_VAL2_*` /
  `CHUNK_CONST2_RECLAIM` names are no longer read (see Removed). The harnesses
  `const2_registry_validation` and `const2_reclaim_validation` are
  `second_value_registry_validation` and `second_value_reclaim_validation`, the
  `rechunk_validation` cases `val2-*` are `second-value-*`.

### Removed

- python/multi_launch_stress_gdb.py: referenced nowhere (no script, document or `shrink_gui`
  program).
- `CHUNK_CST_VAL2_ID_LIMIT` and `CHUNK_CST2_RECLAIM`, the names these two settings had before
  0.3.0, as 0.3.0 announced: `CHUNK_SECOND_VALUE_ID_LIMIT` and `CHUNK_SECOND_VALUE_RECLAIM`
  above.
- **Every compatibility path for a former name** (breaking, hence 0.4.0). The
  names deprecated in 0.3.0 and those kept "until 1.0" since 0.1.3 / 0.2.0 are gone; a former
  name no longer compiles, configures or is read:
  - the headers `chunk/manage_ompt.h`, `chunk/param_chunk.h`, `chunk/array_chunk_bits.h`,
    `chunk/array_chunk_range_ops.h`, `chunk/zero_one_value_array.h` (include `chunk/phases.h`,
    `chunk/runtime.h`, `chunk/bit_array.h`, `chunk/range_ops.h`, `chunk/zero_or_value_array.h`);
  - the `[[deprecated]]` names `ParamChunk`, `Runtime::init_indexes()` (both forms),
    `Runtime::set_decalage_chunk()`, `destroy_block_chunk()`, `ArrayChunkBits`,
    `build_config::has_not_ompt`, `chunk_config::use_zero_one_value`, the namespace
    `zero_one_value`, `get_second_value()`, `get_nb_chunk_written()` (use `Runtime`,
    `Runtime::init()`, `set_chunk_shift()`, `Runtime::shutdown()`, `BitArray`,
    `build_config::needs_explicit_phases`, `chunk_config::use_zero_or_value`, `zero_or_value`,
    `second_value()`, `written_chunk_count()`), and the macro `CHUNK_VARIANT_ZERO_ONE_VALUE`
    (`CHUNK_VARIANT_ZERO_OR_VALUE`); `docs/api.md` keeps the table of the 0.3 renames;
  - the CMake option names without `CHUNK_` (`OMP`, `SANITIZER`, `CHECK_OPTIM`, `USE_LIBCXX`,
    ...: 0.3.0 lists them), no longer read at the top level; the `CHUNK_VARIANTS` spelling
    `zero_one_value`, now refused with the new name as `chunk`, `chunk2`, `chunk2a` are; the
    targets `chunk::chunk_zero_one_value*`, `chunk_zero_one_value*` and
    `shrink_array::zero_one_value*` (link `shrink_array::zero_or_value*`);
  - the compile-time `CHUNK_CONST_VAL2_REGION_BYTES`, `CHUNK_CONST_VAL2_MAX_IDS`,
    `CHUNK_CONST_VAL2_INDEX_LOG2` and the variables `CHUNK_CONST_VAL2_ID_LIMIT`,
    `CHUNK_CONST2_RECLAIM` (the `CHUNK_SECOND_VALUE_*` names above);
  - the switches of before `CHUNK_LOG`: `FAST_ALLOC_TRACE`, `CHUNK_ARENA_REPORT`,
    `CHUNK_LEDGER_REPORT`, `CHUNK_OMPT_STATS`, `CHUNK_CST2_PROBE` (`CHUNK_LOG=alloc=trace`,
    `arena=info`, `ledger=info`, `ompt=info`, `compress=info`). fast_allocators is 2.5.0
    (`CHUNK_FAST_ALLOCATORS_VERSION` follows);
  - in the scripts, `CHUNK_BUILD_REP` (`CHUNK_BUILD_ROOT`) and `python/omp_speedup.py`
    (`python/thread_speedup.py`). The data recorded under old names (variant labels in perf
    databases and summaries, `shrink_gui` runs of `omp_speedup.py`) are still read.
  `find_package(chunk)` stays, deprecated: `find_package(shrink_array)` loads it.

### Fixed

- Clang or oneAPI with an OpenMP runtime that never loads an OMPT tool defined in the
  executable -- Ubuntu 24.04's `libomp5-18` binds its own weak `ompt_start_tool` at link time
  and searches only the objects loaded after it, so `-rdynamic` does not help -- built the
  library with its tool anyway: the regions ran with their phase boundaries missing (chunks
  left materialised, the concurrent-teams invariants broken; 40 validation tests of the nightly
  clang 18 job). The configure step now runs a probe (`cmake/ompt_probe.cpp`) and, when the
  runtime does not call it, builds without the tool (`CHUNK_OPT_OMPT` 0, `shrink_array_OMPT`
  OFF) with a warning. The verdict is cached in `CHUNK_OMPT_TOOL_LOADS`, which can also be set
  to skip the probe; it is not run when cross-compiling without an emulator.
- OMPT builds, a region opened inside a parallel region of one thread (`OMP_NUM_THREADS=1`, an
  inactive region: `omp_in_parallel()` is false there) -- the teardown census of an array
  destroyed under `omp single` -- passed for the end of a phase: its compression pass emptied the
  other arrays' candidate lists in the middle of the region. A region now ends the phase only if
  it began at the top level; the one-thread region itself still ends it. multi_mat's own OMPT
  tool had the same test and is fixed alike. `zero_or_value_shared_state_validation`
  (bool-construct, 1 thread) and `nested_region_validation` cover it.
- `zero_or_value`, the other arrays of a type when one array goes (destroyed, cleared, copy- or
  move-assigned, swapped):
  - the record of each second value (its constant block's index) was a pointer to the array
    that declared the value last. Once that array was destroyed, or took another value, the
    other arrays declaring the value were no longer drained by the passes -- or a pass followed
    a dangling pointer. The type now keeps its own copy of each record;
  - the pass that `clear()` runs at a sequential point emptied the other arrays' candidate lists
    without compressing them, so a temporary destroyed between two phases left their chunks
    materialised. It now compresses them, and runs only outside any region, an inactive
    one-thread region included.
  The new `bystander_validation` (every variant, 1 and 8 threads) checks that copy/move-assign,
  swap, `clear()` and `shrink_to_fit()` of one array leave two bystander arrays -- one compressed,
  one materialised -- as they were; `zero_or_value_shared_state_validation` gains the destroyed,
  cleared, assigned and swapped owner cases.
- `zero_or_value`: `set_second_value(T{})` on an array that had another second value took the
  new-value path (T{}'s sentinel index reads as "not registered"): it gave T{} a block of its
  own and replaced the T{} key every array without a second value is drained under. Called at
  a sequential point, the T{} record took a real index; inside a region, the T{} candidates
  other arrays had queued were dropped, their chunks left materialised. The call now puts the
  array back to "no second value". `zero_or_value_shared_state_validation` (zero-key case).
- Smaller findings of the rescan of 2026-11-20:
  - `RechunkDecider`: with `max_coarsen_factor` at 1 or 0 ("no cap", `--rechunk_max_coarsen`),
    an array banned from coarsening went on being coarsened;
  - `ChunkSizeAdvisor::analyze()` with `Options::max_mat_scans` at 0 divided by it when it
    sampled the materialised chunks (read from the code, not reproduced: at least one chunk
    is read now);
  - `zero::Chunk`'s copy-assignment named a member that does not exist (nothing instantiated it);
  - a shared library's soname carried the major version alone while two 0.x minor versions are
    declared incompatible: `lib<name>.so.0.<minor>` at 0.x;
  - `find_package(shrink_array)` through `shrink_array_DIR` alone did not find the
    fast_allocators installed with it, and could take another installation's;
  - a former option name (`-DOMP=OFF`, `-DSANITIZER=...`) given to a tree already configured
    printed its deprecation and was ignored;
  - fast_allocators built alone with the std backend stopped on a static_assert
    (`ALLOCATORS_THREADED` followed `ALLOCATORS_OMP` only);
  - the Spack recipe accepted `threads=std` and `+tbb` at 0.1.3 and `+stats` up to 0.1.1,
    versions that ignore those options (serial libraries installed for `threads=std`);
  - scripts/check_tracked.py followed relative Python imports only: an untracked module of
    `shrink_gui`, whose imports are absolute, passed the check;
  - docs: `zero_any::Array::compress()` reports and compresses nothing, and `shrink_to_fit()` is
    refused inside a region (`docs/api.md`, `docs/getting_started.md` said otherwise).
- `resize()` (and what goes through it: `reserve()`, `assign()`, a growing `push_back()`) of a
  shared array inside a region, from one thread while the team waits (`omp single`), as
  `docs/getting_started.md` section 5 allows: it purged the calling thread's candidate list
  only, the other ranks kept pointers into the chunk list it reallocates and flags in the row
  it replaces (an invariant failure in the check builds, a use after free in the others). An
  array other ranks have queued chunks of has its candidates drained on every rank; a work
  array of the calling thread, on its own rank as before (the three variants).
- `zero_or_value`, serial builds, an array without a declared second value: `+=`, `-=`, `++`,
  `--` and the bitwise operators on a zero element moved one of the chunk's two constant
  counters only, so the element still counted as zero; the next plain write that zeroed the
  rest of the chunk folded it, and the element read 0. Each counter now moves on its own
  (`Proxy::count_constant_transitions()`), as in a plain assignment.
- `zero`, OpenMP: a compression pass triggered from the write path took no writer handshake
  when `omp_get_max_threads()` was 1 -- the size of the next region, also inside a team opened
  with `num_threads(4)` under `OMP_NUM_THREADS=1`: chunks were folded under their writers
  (200 to 300 writes lost per run of the new case of nested_region_validation). The pass
  looks at the ranks that may hold work (`chunk_rank_bound()`).
- `a[i] = b[j]` between arrays of two element types (a `double` array from an `int` one): a
  deferred write kept a copy of `b`'s proxy and read `b[j]` when it was replayed, at the end
  of the phase -- the value by then, or freed storage after a resize of `b`. The value is
  read when the statement runs, as between two arrays of one type (the three variants).
- `alloc_chunk()` (the three variants), found by the new alloc_chunk_validation:
  - on a chunk compressed to a second value (`zero_or_value`, `zero_any`, bool's `true`) it
    installed a buffer filled with the first constant: the chunk read `T{}` from then on, in
    every build. It keeps the constant the chunk held; a serial `zero_or_value` also records
    the chunk's count of it;
  - in a dynamic build the buffer had the chunk size the calling thread used last (another
    array's, or the process-wide one), not the array's: the stores that followed overran it.
    The array's geometry and allocator set are pinned first, as the write proxies do.
- `operator()(i)`, the raw reference, did not compile on a `zero_or_value::Array` of a non-bool
  type in the check builds (the chunk's "materialised?" check instantiated a bool-only member):
  the member is constrained, and the check is made by the array, which has the second-value
  index.
- `shrink_gui` 0.19, "Supprimer" and "Arrêter" (the cluster-side helper now checks what the
  window assumed):
  - a run's own output directory was removed with the outputs of later runs written below it
    (a run given the default base as its output directory): refused when another run announces
    a directory in it, or when it holds files written after the run ended;
  - in a shared output directory, the files written after the run ended are no longer taken
    for its own (a later run deleted without its outputs, or launched by hand, lost them);
  - a run still active, or whose state cannot be known from here (started from another login
    node, no squeue), could be deleted: the helper refuses the first and wants a second
    confirmation for the other;
  - "Arrêter" on a local or detached run that had ended sent SIGTERM to whatever process group
    had its PID by then: refused once the run has an exit code, and the PID is tied to the run
    by its start time and the boot id, so that a PID reused after a restart of WSL is neither
    shown "en cours" nor signalled;
  - "Arrêter" marked a batch run "arrêté" before trying to cancel it, also when the cancel
    failed or the run had already ended with its exit code: marked once cancelled only.
- `shrink_gui`, "Supprimer" with the run's own output directory: the directory the log announces
  is removed whole only when every file in it was written after the run was submitted, it holds
  no git repository and is no mount point; the former guard (absolute, not `/` nor the home,
  three levels deep) let a log naming e.g. `~/Documents` have it removed.
- A relocatable install with the optimised libm: `SHRINK_INTEL_LIBM` (on by default, when oneAPI's
  `libimf` is found) and `SHRINK_ARM_LIBM` linked it into every target of the directory with an
  absolute path and rpath, the libraries included: the installed chunk-targets.cmake and
  fast_allocators-targets.cmake named `/opt/intel/.../libimf.so`. Only the project's executables
  (`examples/`) link it now.
- The installed fast_allocators-config.cmake calls `find_dependency(Threads)`: with the std backend its target
  links `Threads::Threads`, and a project that found fast_allocators alone (without chunk) failed
  to link.
- `zero_any`: every `a[i] += x` ran `ensure_writable_storage()`, which re-armed the per-chunk
  pointers the write engine's `set_info_on_chunk()` had just armed and tested both constants,
  on a chunk already materialised as well: about 44 instructions per operation. It runs only on
  a constant chunk now (a materialised one stays so for the write session); `pattern::Array`,
  whose motif chunks are neither constant, keeps the full path. Instructions per `+=` (1
  thread, steady state): 308 -> 259.
- `zero_or_value` and `zero_any` (and `pattern::Array`): `a[i] += x` cost more than a read plus a
  write (16.4 ns on Grace), and `zero_any`'s `a[i] = v` paid the same stall. Every write runs
  `set_info_on_chunk()`, which was called out of line: it stored into the proxy, on the caller's
  stack, with 8- and 4-byte stores that the caller's 16-byte reloads of the proxy spanned -- a
  store the core cannot forward, so each write waited for it to reach the cache. It is forced
  inline. `access_cost`, x86, 1 thread: `rmw` 13.2 -> 6.0 ns (`zero_or_value`), 17.1 -> 9.8 ns
  (`zero_any`), `zero_any`'s `write-varied` 14.5 -> 8.9 ns; 14 threads, `zero_or_value`'s `rmw`
  1.60 -> 0.85 ns. Not measured on Grace yet.
- Check builds, `zero_any` and `pattern::Array`: the destructor (noexcept) runs `clear()`, whose
  chunk-size prediction allocates a summary and formats a log line; a `std::bad_alloc` there
  would have called `std::terminate()`. The prediction, a log line only, is skipped on an
  exception (Coverity 661905, 661906).
- Compound operations with an operand that is not clean for the element (`int` `*= 1.5`,
  `int` `%= 4294967299L`; concept_is_clean_assignment.h): the rule -- the operand converted to `T`
  first, with an `HPC WARNING` -- held for `+=` / `-=` of `zero_or_value` and `zero_any` only.
  `zero`'s `+=` / `-=` computed in the common type on the direct path and converted on the
  deferred one (`-1 += 0.5`: 0 or -1); `*=` / `/=` / `%=` of every variant took the operand as it
  came, without warning, and `zero_or_value` gave 3 for `2 *= 1.5` in a serial build, 2 in an OpenMP
  one. Every compound operator of every variant now has the two overloads `+=` had: a clean operand
  computes as a plain `T` does, any other is converted to `T` at the operator, with the warning, so
  every path below gets a `T`. `docs/api.md` §4 states the rule; `mixed_operand_compound_validation`
  checks it, direct and deferred, in every variant.
- `zero_or_value`: a copy-assignment `a = b`, `b` declaring another second value than `a`,
  wrote `b`'s elements into the constant block of `a`'s old second value -- a block every array
  declaring that value shares: their compressed chunks read `b`'s elements afterwards.
  `operator=` switched the value before copying, so `a`'s chunks still on the old block passed as
  materialised. It now releases `a`'s chunks first when the second value changes, as the
  move-assignment does. `zero_or_value_shared_state_validation` checks it.
- `zero_or_value::Array<bool>` (and the bit arrays): building one installed a fresh, empty `true`
  candidate list on every rank, dropping the chunks the other live bool arrays had queued -- never
  queued again, never compressed by the passes -- and freeing the list under a thread of another
  bool array writing into it, for one built inside a region. The list is installed once.
  `zero_or_value_shared_state_validation` checks it.
- Every variant, a parallel region of one thread (`OMP_NUM_THREADS=1`): the teardown of a work
  array built in the region took the region for a sequential point -- one thread's region is
  inactive, `omp_in_parallel()` is false there -- and ran the pass over every type's candidate
  lists that `clear()` runs at one, a walk that empties them without compressing: what every array
  had queued in the region was never compressed. The own-rank test (`teardown_on_own_rank()`)
  counts an inactive region as a region. `zero_or_value_shared_state_validation` checks it.
- Install of a std-alloc build (`CHUNK_STD_ALLOC`, Spack's `~fast_alloc`): the handful of
  fast_allocators headers it installs with chunk missed three that the others include --
  `internal/append_text.h` (from `log_core.h` and `fatal.h`), `internal/pmp_loops_inline.h`
  and `internal/pmp_std_inline.h` -- so no consumer of such an install compiled. They are
  installed, and `check_tracked` (rule 6) recomputes the list from chunk's includes and fails
  when one is missing.

## 0.3.0 -- 2026-11-18

### Added

- The `shrink_array` C++ module, experimental and off by default: `-DCHUNK_MODULES=ON` (CMake 4.2,
  Ninja, GCC 16, `import std`) builds `chunk_<variant>_module` over each variant library
  (`chunk/modules/shrink_array.cppm`: the library read in the module's purview, what its headers
  mark `SHRINK_ARRAY_EXPORT` exported) and installs them as `shrink_array::<variant>_module`.
  `getting_started`, `bool_arrays`, `advisor_demo`, `mdspan_tiles`, `simple_array` and
  `vectorisation_demo` then `import shrink_array;` instead of including the headers, and
  `module_api_<variant>` checks the exported names; `.github/workflows/modules.yml` builds it.
  The phase hooks moved to `chunk/internal/phase_hooks.h` (`chunk/phases.h` includes it).
  This was the `cpp23-modules` branch, folded into cpp26.
- `scripts/compile_all.sh --list-targets`, `--list-execs`, `--list`: the configurations
  `--target` accepts (configured here or not, x86-64 or ARM only), the targets `--exec` accepts
  in each configured build directory, or both, without building; they honour `--target=<glob>`
  and `--exec=<glob>`.
- `scripts/compile_all.sh` builds with Arm Compiler for Linux on ARM (the `acfl` family:
  `omp-acfl`, `std-acfl`, `omp-acfl-debug`; `--only=acfl`, `--no-acfl`), the ARM counterpart
  of the x86-only `oneapi` and `aocc`. `armclang++` comes from `CHUNK_ACFL_ROOT` (an install
  prefix, as `spack location -i acfl` prints it), from the `PATH` (`spack load acfl`), from
  the newest `$CCCWORKDIR/opt/*/acfl-*` (a Spack install tree there) or from an `acfl` module;
  none found, the family is skipped. `bench_runner` and `build_arm.txt` list the two
  optimised builds.
- `shrink_array::tbb::use_tbb_for_library_teams()` (`chunk/tbb_adaptor.h`), called by every
  adaptor call: under the std backend, the library's own teams (the flush and compression pass of
  `finalize_parallel_phase()`, a re-chunk, a parallel initialisation) take their workers from TBB
  instead of the std backend's pool, so that a TBB program no longer runs two sets of threads
  competing for the cores. A team is made of the TBB tasks that started within
  `CHUNK_PMP_JOIN_US` (developer setting, 200 us): TBB does not promise to start them at once,
  and a later task returns without joining. TBB bench, 24 threads: computation -9.4%,
  involuntary context switches 16.3 M -> 0.6 M. Underneath, `pmp::set_team_launcher()` (any
  runtime that can start tasks).
- `shrink_gui` 0.17: "Nettoyer les builds…", beside "Builds up to date". Lists every configured
  build of the build root (every architecture) with what cleaning it frees, and removes from
  the checked ones what the next build re-creates: the object files (about half of a build),
  their dependency files, the precompiled headers, the static libraries. The executables, the
  shared libraries and the build files stay: the next `compile_all.sh` recompiles everything,
  with no new configuration. The build check of `run.sh` knows a cleaned build: until its next
  build it is up to date if `ninja -n` said so just before the clean (`.shrink_gui_cleaned`)
  and no source, header or CMake file of the repository changed since
  (`docs/shrink_gui.md` §5).
- CI (P3 of the audit): four gaps closed. Per push, `fetch-content`: `examples/fetch_content`
  (new) consumes the checkout with `FetchContent`, and the job checks that the consumer keeps
  its build type, flags, `doc` target and tests, and installs nothing of the library. Nightly:
  `oldest-compilers` (GCC 13 and Clang 18, the minimum the README and the recipe state, never
  built by the CI), `x86-openmp-ubsan` (`SANITIZER=undefined`, new: UBSan, the first report
  stops the program; 256 and `_dyn`), and `spack` (the recipe installed from the checkout for
  `target=x86_64_v3`, no target flag from the project, no AVX-512 instruction in the
  libraries, `getting_started` built against the install).
- The four `seq_cst` fences that closed the phase functions (`finalize_parallel_phase()`, its
  `only_flush` / `after_flush` halves, `finalize_barrier_phase()`;
  `chunk/private/type_state_registry.cpp`) are gone, and `CHUNK_PHASE_FENCES` with them. They
  were kept as "empirically load-bearing" since one unexplained checksum in August 2026; no
  handshake needs them as far as the code reads (each has its own barrier), and several defects
  that lost deferred writes at phase boundaries were fixed since. Settled by measure: several tens
  of thousands of `shrink_bench --mode check` runs on AArch64, the weaker memory model, with the
  fences off (`CHUNK_PHASE_FENCES=0`), not one checksum different from `std::vector`'s.
- `shrink_gui` 0.16: an environment field under the extra arguments (`NAME=value` words),
  exported in `run.sh` and `node.sh` before the command -- `CHUNK_LLVM_ROOT=~/llvm-22` for
  `compile_all.sh`, a run-time switch of the library for a bench. Exported before
  `scripts/site_env.sh` and again after it (the run's value wins); a leading `~/` is the home
  directory of the machine the run is on. Kept per program, written to `meta.json` (`env`),
  taken back by "Reprendre ses options", shown before the command and in the run's log.
- `shrink_gui`: the left panel ("Où et comment", "Quoi") scrolls when the window is shorter
  than it, and "Soumettre" stays below it, always in view: with one more field the button was
  pushed out of a laptop screen.
- SonarQube Cloud (project `cea-hpc_shrink-array`): `.github/workflows/sonar.yml` analyses
  the tree on every push to develop and main, and on demand, with the rules of SonarQube for IDE
  (SonarLint) -- the C++ from CMake's compilation database, one entry per source file, nothing
  compiled; `sonar-project.properties` says what is analysed. Needs the `SONAR_TOKEN` secret
  (without it the job stops, green). `scripts/sonar_export.py` writes every issue of a branch
  to one JSON or CSV file (`--hotspots` adds the security hotspots), through the Web API and
  the shell's Kerberos proxy, as `coverity_submit.sh` does; past the API's 10,000 issues per
  query it splits the query by language, type, severity, rule or directory.
- `docs/design.md` (`docs/conception_fr.md` in French), the design document: requirements and
  non-goals, the parts and their scope, static polymorphism, build-time configuration, the
  access paths, the concurrency model, memory, dynamic geometry, backends, the decisions with
  the alternatives rejected, known limits. Linked from `docs/README.md` and both READMEs;
  `check_docs.py` checks the pair.
- `shrink_bench --advise_bool`: hands `array_bool` and `array_bool_bits` to the re-chunk decider
  (off by default: the two bool containers stay at their birth geometry, to be compared as
  they are). `rechunk_validation` gains `bool-sliced`: `Array<bool>` through the sliced
  rebuild, for the three variants.
- `docs/gpu_design.md`, the English version of the GPU design document
  (`docs/conception_gpu_fr.md`); `check_docs.py` checks the pair like the others.
- `multi_launch_stress.py --peak_threads N [--peak_share F]`: most thread counts drawn near N,
  the top of the speedup curve, the others still uniform over the range. In `shrink_gui` too.
- `diag_zero_rechunk_slowdown.py`: the resident memory against the static run, the churn per array
  after the last re-chunk and, with `--perf`, the samples' share in the compression passes.
- **Three additions to the public interface**, what the examples needed from `internal/`:
  `chunk/threads.h` (new): `pmp::max_threads()`, `pmp::team_size()`, `pmp::thread_rank()`, the
  threads of whatever backend the build has (OpenMP, std, serial), instead of
  `CHUNK_PMP_MAX_THREADS`; `chunk/print.h` (was `chunk/internal/print.h`): `print` / `println`;
  `set_active_chunk_size()`, `set_thread_chunk_size()` (through `chunk/chunk_size_advisor.h`)
  and `zero_any::Chunk` (through `chunk/zero_any_array.h`, for `sizeof` in the advisor's
  options), and the arrays' `set_chunk_size()` (an empty array's geometry). 18 public headers. `simple_array`, `vectorisation_demo`, `variant_compare`,
  `access_cost`, `advisor_demo`, `getting_started` and `bool_arrays` include public headers
  only, and `check_public_api` fails when an example outside the harnesses and the benchmarks
  includes an internal one.
- `CHUNK_USE_CCACHE` (`AUTO`, `ON`, `OFF`): the build compiles through ccache (>= 4.8) when it
  finds it, on the `PATH` or with `-DCCACHE_EXECUTABLE=...`. A branch switch or a pull that leaves
  the library headers alone gets the unity units of `shrink_bench` and the validation back from
  the cache (a rebuild after `ninja clean`: 93 s to 0 s). The cache is split by host CPU, so
  `-march=native` objects never go from one kind of node to another; `coverity_submit.sh`
  configures with it `OFF`.

### Changed

- fast_allocators is 2.4.0: it had changed since 0.2.1 (the team launcher of the std backend,
  among others) with its number left at 2.3.0.
- The variant `zero_one_value` is `zero_or_value`: the name read "zero-one value", an array
  of 0s and 1s, when the second value is any value. Namespace, header
  (`chunk/zero_or_value_array.h`), `chunk_config::use_zero_or_value`,
  `CHUNK_VARIANT_ZERO_OR_VALUE`, the value of `CHUNK_VARIANTS`, the targets and library files
  (`chunk_zero_or_value_256`), the executables and tests (`validation_zero_or_value_256`), the
  documents (`docs/algo_zero_or_value.md`), the Spack recipe and the Python tools. The former
  name still works where a program or a script may say it -- the header, the namespace and the
  flag (deprecated), the macro, `CHUNK_VARIANTS` and the target names (with a deprecation
  message), the labels of the perf databases recorded before -- until the next minor version.
  Not the library files: `libchunk_zero_one_value_256.a` has no former copy.
- The statistics line of an array says "N second value", the words of the documents; it was
  "N type-2".
- French names and abbreviations in English, in the sources, the examples and the documents
  (internal names; nothing of the public interface): `nb` (nombre) is `n` (`n_val_eq_zero_`,
  `n_blocks`, `n_threads`), the increm and decrem prefixes are inc and dec (the statistics
  counters: `inc_chunk_write`, ...), `somme` is `total` (`total_chunk_create`), `raz` (remise à zéro) is
  `reset`, `recup` is `reclaim` (`reclaim_value`), `kdeb` / `kfin` are `kbegin` / `kend`
  (examples/multi_mat), `nbmat` is `n_materials`, `freq_conso_mem` is `freq_mem_report`.
- `CHUNK_BUILD_REP` (répertoire) is `CHUNK_BUILD_ROOT`, the name the Python tools read already:
  `compile_all.sh` and `scripts/site_env.sh` take their builds there. The former name is still
  read when the new one is unset, with a deprecation line; it goes at 0.3.
- `ind` (indice) is `idx` in the names (`get_idx_second_value`, `set_info_second_value_idx`, a
  bare `ind` is `idx`), and the arrays' `IndType` is `IndexType` (`Ind`, `IndT`: `Index`, `IndexT`).
- `cst` (constante) is `const` in the names: `const_val1`, `const_val2`, `is_const_val2`,
  `ConstSlot` in `chunk/internal/const_slot.h`, `const_buffer_size`, a bare `cst` is `constant`;
  the harnesses `const2_reclaim_validation` and `const2_registry_validation`. The variables
  `CHUNK_CONST2_RECLAIM` and `CHUNK_CONST_VAL2_ID_LIMIT` (`CHUNK_CST2_RECLAIM`,
  `CHUNK_CST_VAL2_ID_LIMIT`: still read when the new name is unset, with a warning; they go at
  0.3), the build-time `CHUNK_CONST_VAL2_MAX_IDS`, `CHUNK_CONST_VAL2_REGION_BYTES`,
  `CHUNK_CONST_VAL2_INDEX_LOG2`. `CHUNK_CST2_PROBE` keeps its name: it is a former variable
  `CHUNK_LOG` reads for compatibility.
- `elt` (élément) is `elem` in the allocator's names (`InfoFreeElem`, `free_elem_pow2`,
  `clear_info_free_elem`, ...), as everywhere else; `CalcType` (calcul) is `ComputeType`.
- `num` (numéro), the offset of an element in its chunk, is `elem_offset` (`elem_offset_`,
  `get_elem_offset()`), and `num_compress_value` (which of the two constants) `const_index`.
  examples/multi_mat: `MatArrayBase`, `MatZeroArray`, `MatZeroOrValueArray`, `MatVectorArray`
  (`VectCom`, `VectChunk`, `VectZeroOrValue`, `Vect`).
- `tab` (French: tableau) is `array` in the names of the sources, examples and documents:
  `array_int`, `array_bool`, ... of `shrink_bench` (`tab_int`, ...), `arr1` ... `arr5` and `arr`
  (`tab1`, `tab`), the library's `array_is_inside_chunk_list_to_compress`,
  `get_array_second_value`, `ArrayConstVal2Holder` in `chunk/internal/array_const_val2_holder.h`
  (`tab_cst_val2_holder.h`), ... The statistics name an array `[array_0 double]`
  (`[tab_0 double]`), and `shrink_bench`'s reference files are `data_array_<name>_<k>.bin`
  (`data_tab_*`): an oracle directory made before is made again by the next campaign.
- Internal names, none of them part of the interface. Headers of `chunk/internal/`:
  `concept_is_clean_assignment.h` (its name was misspelt), `pmp_chunk.h` and `chunk_tuples.h`
  (`PMP_CHUNK.h` and `TChunk.h` were the two files not in snake_case), `pow2.h` (`func_p2.h`)
  (`crp_helper.h` is gone here: deducing this replaced it). The French abbreviations: an array's number is `array_id`
  (`num_tab`), their count `array_count` (`nb_tab`), `ArrayUsers` (`TabUsers`); a chunk's number
  is `chunk_index` (`num_chunk`), their count `chunk_count_` (`nb_chunk_`). The registry of the
  arrays is `array_registry` (`share_info_array`). Examples:
  `examples/multi_mat` and its executables `multi_mat`, `multi_mat_embedded` (the one directory
  with a hyphen), `examples/vectorisation_demo` (`test_vecto_chunk`: an example, not a test);
  the harnesses `compress_roundtrip_validation` and `pool_accounting_validation` take the
  suffix of the others, and their ctest names with it.
- Std backend with the library's teams on TBB: `CHUNK_PMP_JOIN_US`, how long such a team waits
  for its TBB tasks to start, is 5 ms by default instead of 200 us. On a 144-thread AArch64 node
  the TBB bench (`shrink_bench_tbb_zero_any_int64_dyn --multiply 3 --mode perf`) ran 79.5 s with
  200 us and 50.9 s with 5 ms: the finalize 21 s -> 13.9 s, and the program's own regions
  52.5 s -> 31.1 s, since a task that missed its team was taken and dropped by a TBB worker
  during the next region (`BENCH_RANK_TIMING`'s launch share 13% -> 2.2%). Only TBB's teams
  wait so: the OpenMP backend and the std backend's own pool are untouched.
- The CMake package is `shrink_array`, the project's name: `find_package(shrink_array)`, and
  targets without the repeated word -- `shrink_array::zero_any`, `shrink_array::zero_any_256`,
  `shrink_array::zero_any_dyn` (`zero` and `zero_or_value` alike), `shrink_array::vector`,
  `shrink_array::chunk` -- installed and in the build tree (`add_subdirectory()`,
  `FetchContent`). Its variables are `shrink_array_STATIC_SIZES`, `shrink_array_DEFAULT_SIZE`,
  `shrink_array_HAS_DYN`, `shrink_array_OPENMP`, `shrink_array_THREADS`, `shrink_array_OMPT`,
  `shrink_array_STD_ALLOC`. The project had three names (`shrink-array`, `shrink_array`,
  `chunk`), and `chunk` is one another package may take in a shared prefix.
  `find_package(chunk)` and the `chunk::chunk_…` targets still work, with a deprecation
  message, until the next minor version. Unchanged: the `#include "chunk/…"` paths, the
  `CHUNK_` options, the names of the library files.
- CMake options without a prefix have one: `OMP` is `CHUNK_OPENMP`, `SANITIZER` is
  `CHUNK_SANITIZER`, `CHECK_OPTIM` is `CHUNK_CHECK_OPTIM`, and likewise `CHUNK_USE_LIBCXX`,
  `CHUNK_USE_MOLD`, `CHUNK_USE_GPERF`, `CHUNK_ENABLE_IWYU`, `CHUNK_GCC_TOOLCHAIN_DIR`,
  `CHUNK_INTEL_ICPX`, `CHUNK_INTEL_ICPX_GLOB`, `CHUNK_INTEL_LIBM_DIR`, `CHUNK_ONEAPI_FAST_FP`,
  `CHUNK_ONEAPI_NO_INTEL_LIBS`, `CHUNK_ONEAPI_XHOST`, `CHUNK_ARMPL_LIBM_DIR`;
  `HAVE_AGGRESSIVE_COMPRESSION` is `CHUNK_AGGRESSIVE_COMPRESSION_DEFAULT`. A project that took
  the library in with `add_subdirectory()` or `FetchContent` shared those names with it in its
  cache. The former names are still read, with a deprecation message, when the project is
  built on its own (`-DOMP=ON` sets `CHUNK_OPENMP`); as a subproject it ignores them.
- Public names, before the interface is published more widely. `ParamChunk` is `Runtime`
  (`chunk/runtime.h`), `ParamChunk::init_indexes()` is `Runtime::init()` and
  `destroy_block_chunk()` is `Runtime::shutdown()`: the façade's name said neither what it is
  nor what the two calls do. The phase calls are in `chunk/phases.h` (`chunk/manage_ompt.h`
  named the one build that does not need them) and `build_config::has_not_ompt` is
  `build_config::needs_explicit_phases`. `ArrayChunkBits` is `BitArray` (`chunk/bit_array.h`);
  `chunk/array_chunk_range_ops.h` is `chunk/range_ops.h`. `get_second_value()` is
  `second_value()` and `get_nb_chunk_written()` is `written_chunk_count()`, as `size()` and
  `chunk_size()` have no `get_`. `max_threads()`, `team_size()` and `thread_rank()` are
  reachable from `shrink_array::` (they were `shrink_array::pmp::` only). Every former name
  still compiles: the four headers include the new ones, the others are `[[deprecated]]` and
  forward. They go at the next minor version; `docs/api.md` has the table.
- `AllTypeChunk` is now `TypeStateRegistry`, and its two policies
  `TypeStateRegistryParallelInterface` and `TypeStateRegistryDynInterface`: the class is the
  process-wide registry of the per-type managers (`TypeState`), not a chunk. Its files
  follow: `type_state_registry.h`, `type_state_registry_decl.h`,
  `type_state_registry_inline.h` (`chunk/internal/`) and `chunk/private/type_state_registry.cpp`.
  Internal: no public header names it.
- `ChunkSizeAdvisor::analyze()` was one function of 837 lines; it is now 38, and its counting
  passes are the member functions of `internal::ChunkSizeScan` (`chunk/internal/chunk_size_scan.h`,
  bodies in `chunk_size_scan_inline.h`): the candidate sizes, the classification, the runs of
  constant chunks per part and their serial merge, the sampling, the ambiguous windows of a
  picked chunk, the motif forecast of `pattern::Array` -- one function each, the longest of 57
  lines. The table's rows come from `internal::advisor_row()`. Same interface, same report:
  the old and the new `analyze()` were run in one process on the same arrays of the four
  containers (`double`, `int`, `long`, `bool`, `Bit`; six sets of options; 1, 2 and 4 threads)
  and compared field by field, 5,040 reports without a difference, in the same time.
- `USE_LIBCXX` is `OFF` by default: a Clang build uses the standard library the compiler picks
  by itself (libstdc++ on Linux), `-DUSE_LIBCXX=ON` asks for libc++. The default was `ON`, and
  the first configure failed on a Clang installed without libc++; the build scripts, the CI
  jobs and the Spack recipe all passed `OFF` already.
- `CITATION.cff` at the root (GitHub's "Cite this repository"), and a "Citing" section in the
  READMEs.
- Spack recipe: `maintainers`, and the form the builtin repository's style check accepts (lines
  of 99 columns at most, the test of the `sizes` values as a function), so that the file can be
  submitted there unchanged.
- std backend: `Team::barrier()` spins for `CHUNK_PMP_SPIN_US` (default 1000 us) before sleeping,
  as OpenMP's blocktime does, instead of a `std::barrier` that slept on a futex almost at once;
  `single()` is lock-free (one CAS on a team-wide mark, no mutex, no map slot per construct); the
  pool's hand-offs spin for the same budget. At 160 threads the re-chunk's slice-boundary drain
  took 2.45 s in the TBB bench against 0.24 s with OpenMP, the time in the team's
  synchronisation. `CHUNK_PMP_SPIN_US=0` sleeps at once (the former behaviour, for A/B).
  `team_sync_validation` (and `team_sync_validation_nospin`) checks barriers, singles and many
  teams in a row; `--bench` prints their cost.
- The end-of-life census line of a zero array says what the variant compresses: "few all-zero
  chunks — zero compresses only all-zero chunks" instead of "compression marginal vs
  std::vector", which read as a failure on an array zero cannot help. `shrink_bench` says
  before `array_conn`'s line, in zero builds, that a connectivity is pattern::Array's case.
- `fill_range_mt` skips a chunk already at the fill value with one test, as `fill_range` does,
  instead of arming it element by element; `fill_range` does the same on its `bool` path.
- **The public interface, narrowed and frozen** (breaking). The public headers are what
  `docs/api.md` documents; `pmp_chunk.h`, `type_state_registry.h`, `chunk_traits.h`,
  `global_stats_chunk_list.h`, `ring_log_debug.h`, `singleton.h`, `term_color.h`,
  `type_name.h`, `zero_chunk.h`, `zero_or_value_chunk.h`, `zero_any_chunk.h`, `pattern_chunk.h`,
  `zero_chunk_def.h`, `zero_or_value_chunk_def.h` and `zero_any_chunk_def.h` are under
  `chunk/internal/` (a program that included one includes the array's header instead).
  `release_chunk_pools()` and `destroy_block_chunk()` come with `chunk/param_chunk.h` (they
  were in `chunk/zero_chunk.h`). `ManagerType`, `Manager`, `ChunkType`, `ChunkListType`,
  `ProxyType` and `get_chunk_list()` are not part of the interface any more. The interface is
  `scripts/public_api.txt`, checked by `check_public_api` (ctest). On cpp23 it also lists
  `chunk/mdspan.h` and `chunk/mdspan_tile_ops.h`, experimental until 1.0; on cpp26
  `chunk/soa_array.h` too, experimental.
- **`ManagerChunk` is `TypeState`**: the shared state of one (variant, element type) -- reuse
  pools, compression candidates, per-rank allocators, deferred flush -- named after what it is.
  `ManagerChunkInt` is `TypeStateInterface`, `ManagerChunkDynInterface` /
  `ManagerChunkParallelInterface` / `ManagerChunkDynImpl` / `ManagerChunkParallelImpl` are
  `TypeStateDynInterface` / `TypeStateParallelInterface` / `TypeStateDynImpl` /
  `TypeStateParallelImpl`, and `internal/manager_chunk*.h` are `internal/type_state*.h`
  (`chunk/private/manager_chunk.cpp`: `type_state.cpp`). All internal: no interface change.
- Re-chunk: `zero_or_value` and `zero_any` (and `pattern`) free each old chunk as soon as the
  rebuild has copied it, as `zero` did, and rebuild in 16 slices with the fast allocator under
  threads: each slice boundary recycles the slabs the slice emptied into the next slice's
  allocations (`array_chunk_rechunk_refill.h`; the chunks' new `release_st()`), `bool`
  arrays included; `ArrayChunkBits` (bit-packed) keeps one pass. `zero_or_value_dyn`, x86-64, 2 threads, 10^8 elements: run peak 1,772 ->
  1,724 MB, end +0.9% (the recycled slabs stay resident, as for `zero`); `zero_any`'s peak
  on this benchmark is the initialisation's, unchanged. `rechunk_validation` gains
  `sliced-val2-grow` and `sliced-val2-uneven`: sliced rebuilds mixing zero, second-value and
  live sources.
- With gcc 16.2 and later, `shrink_bench.cpp` is compiled inside the unity unit of its
  executable again: the `-fno-tree-vectorize` that dodged gcc 16's AVX2 ICE stays for gcc 16.0
  and 16.1 only. Compiled apart, the driver instantiated the whole library a second time: about
  30 % less compilation per bench executable (113 s to 79 s for `shrink_bench_omp_zero_any_256`).
- `CHUNK_DEBUG_INFO=auto` gives `lines` (`-g1`) to the address, thread and undefined sanitizer
  builds too, with `CHUNK_CHECK_OPTIM` or without: their reports need only the function names
  and the line tables (the same stacks, inlined frames included, with gcc and Clang), and the
  optimised sanitizer builds are the slowest to compile: the unity unit of
  `shrink_bench_omp_zero_any_256` under ThreadSanitizer takes 128 s and 3.5 GB instead of 162 s
  and 5.0 GB (gcc 16.2). `full` stays for `CHUNK_CHECK_OPTIM` without a sanitizer and for the
  `token` sanitizer; `-DCHUNK_DEBUG_INFO=full` brings the locals back for a debugger.

### Removed

- `CHUNK_CRTP_USE_DEDUCING_THIS` and `CHUNK_CRTP_FORCE_DEDUCING_THIS`, and the deducing-this
  variant of `add_to_compression_list_async()` they selected: develop is C++20, where it was
  never compiled; the cpp23 branch has that form alone.
- `python/run_test.py`: it ran a `test/test1` that no longer exists, under a path left as
  `????` when the tree went public; `bench_runner` averages the memory over runs.
- `chunk/debug.h`, which nothing included (`print_backtrace` is
  `fast_alloc::logging::print_backtrace`).
- **The other compatibility paths kept until 1.0** (breaking). The variant macros
  `CHUNK_VARIANT_V1`, `CHUNK_VARIANT_V2`, `CHUNK_VARIANT_V2A` (use `CHUNK_VARIANT_ZERO`,
  `CHUNK_VARIANT_ZERO_OR_VALUE`, `CHUNK_VARIANT_ZERO_ANY`); the `CHUNK_VARIANTS` spellings
  `chunk`, `chunk2`, `chunk2a`, now refused with the new name; the targets `chunk::chunk_v1*`,
  `chunk::chunk_v2*`, `chunk::chunk_v2a*` (link `chunk::chunk_zero*`, `chunk::chunk_zero_or_value*`,
  `chunk::chunk_zero_any*`). In the scripts, an old variant label on a command line (`chunk2a`,
  `chunk2a@dyn`, `tbb_chunk2`...) is refused with its new name, and `bench_and_test.py`'s
  hidden `--mem-chunk2a-vs-chunk2-max` is gone; the data recorded before the rename (perf
  databases, summaries, logs) are still read, their old labels as the new names.
- **The former C++ names kept until 1.0** (breaking: the next release is 0.3.0). The 45
  `[[deprecated]]` aliases: `ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A` (use `zero::Array`,
  `zero_or_value::Array`, `zero_any::Array`), `Chunk2`, `Chunk2A`, `Proxy2`, `Proxy2A` and the
  former `Chunk` / `Proxy` (use the variant's namespace), `use_chunk`, `use_chunk2`,
  `use_chunk2a`, `use_chunk2or2a` (use `use_zero`, `use_zero_or_value`, `use_zero_any`,
  `use_second_value`), the arrays' `chunk_type`, `manager_type`, `proxy_type`, `ind_type`,
  `chunk_list_type` (use `ChunkType`, `ManagerType`, `ProxyType`, `IndexType`, `ChunkListType`),
  `getChunkList()` (use `get_chunk_list()`), `ManagerChunk2`, `ManagerChunk2A` (use `Manager`),
  `pattern::Array::install_motif_for_test()` (use `install_motif()`),
  `ParamChunk::ChunkStats()` (use `chunk_stats()`), `ParamChunk::check_pow2()` (use
  `check_chunk_size()`), `ParamChunk::get_decalage_chunk()` and `chunk_decalage()` (use
  `i / chunk_size()`), `factor_div_somme_chunk_buffer_resuse`; in fast_allocators,
  `addAllocatedSize()`, `add_TCTypeAllocator_inst()`, `remove_TCTypeAllocator_inst()`,
  `SIZE_BLOCK` (use `add_allocated_size()`, `add_tc_type_allocator_inst()`,
  `remove_tc_type_allocator_inst()`, `size_block`). And the headers of the former names:
  `chunk/array_chunk.h`, `chunk/array_chunk2.h`, `chunk/array_chunk2A.h`, `chunk/chunk_def.h`,
  `chunk/chunk2_def.h`, `chunk/chunk2A_def.h`, `chunk/chunk_std.h`, `chunk/chunk2_std.h`,
  `chunk/chunk2A_std.h` (include `chunk/zero_array.h`, `chunk/zero_chunk.h`, ... instead). The
  proxies' `HPC WARNING` diagnostics on mixed-type operators stay.

### Fixed

- The Spack recipe built the released versions wrong since the CMake options took their
  `CHUNK_` prefix: 0.1.0 to 0.2.1 know `OMP`, `USE_LIBCXX`, `USE_MOLD` and `CHECK_OPTIM` only,
  ignored the new names and took their defaults -- serial libraries with `threads=openmp`,
  libc++ with Clang, no `+checks`. The recipe gives these versions their own names.
- TBB: the library's own teams run in the program's arena. `use_tbb_for_library_teams()`
  launched them (the flush and compression pass of `finalize_parallel_phase()`, a re-chunk) in
  a `task_arena` of its own, so TBB's workers left the program's arena for that one and came
  back at every phase. On a 144-thread AArch64 node the TBB bench took about three times the
  OpenMP one: per compute region, 12.7 ms for 3.2 ms of work per rank, a rank waiting 2.5 ms
  for its task to start, and 23 s in `finalize_parallel_phase()` over the run. The tasks now go
  to the calling thread's arena (`oneapi::tbb::this_task_arena::enqueue`, oneTBB 2021.6 and
  later), and `use_tbb_for_library_teams(arena)` names the arena for a program that runs its
  regions in an explicit `task_arena`; `shrink_bench` does. `BENCH_RANK_TIMING` printed the
  time of `finalize_parallel_phase()` as a part of the serial time, which it is not, and left
  it out of the thread-time it splits: its report gives the three wall times apart, and a
  finalize share.
- `multi_mat` built with gcc on x86-64: the compute loop (`LoopsSimd::omp_foreach_simd_red_sum`,
  `cos + sin + tan` under `omp simd`) never vectorised. glibc declares its vector `cos`, `sin`
  and `tan` (libmvec) to gcc only under `__FAST_MATH__`, which the flags of
  `chunk_enable_fast_math` do not define; gcc reported "function is not vectorizable" (the
  "loop vectorized" it printed at the same pragma was the reduction across the lanes). The
  target now adds `-ffast-math` with gcc on x86-64 Linux, except in a `CHUNK_DO_CHECK` build
  (its NaN check would fold away). That was enough with gcc 13.3 (8 lanes with AVX-512, the
  run on the `std::vector` materials about 2.3 times faster), not with gcc 14 to 16 on Intel
  tunings (alderlake, generic, skylake): they no longer emit hardware gathers (the
  Downfall/GDS mitigation), and the emulated gathers gave up on the load of the indirection's
  data pointer, left inside the `omp simd reduction` loop ("no vectype for stmt ...
  _M_start"). The loop now reads the indirection through `ll.data()`, taken once: gcc 14, 15
  and 16 vectorise it on every tuning (libmvec's _ZGVdN4v_cos, _sin and _tan with AVX2; gcc 16,
  i9-12900H, 2 M cells, 8 materials, 100 iterations, 2 threads: the whole run 24.3-27.6 s
  -> 21.0-22.5 s). `shrink_bench` is left as it was. The comment of
  `chunk_enable_fast_math` says it, and that clang knows libmvec's `tan` only since LLVM 19.
- Check builds (`CHECK_OPTIM`, and the sanitizer and Debug builds) were several times slower
  than the others under OpenMP: the cross-check of a proxy's `array_id` against the registry,
  done on every write (`zero_any`) and every replayed op (`zero`), took the registries' shared
  lock, and every thread bounced its reader count (`pthread_rwlock_rdlock` + `unlock` were 91 %
  of `shrink_bench_omp_zero_any_dyn`). A thread now skips a check it already made on the same
  array and `array_id` while no registry has changed: a generation counter that
  `RegistryWriteLock` bumps, and a per-thread memo of 16 sets of 4 ways (a direct-mapped one
  let two arrays written in turn evict each other in about one run out of two, depending on
  where ASLR put them). The bench at 14 threads: 7–8 s → 0.6–1.1 s.
- Check builds: the count of second-value lookups in flight (`GlobalVecHolder::LookupGuard`,
  which lets a reclaim detect a lookup running alongside it) was one atomic shared by every
  thread, two locked RMWs per lookup: about 90 % of `find_or_set_ind_cst_val2()`, and 13 % of
  the bench on a many-core machine. One counter per thread on a cache line of its own, summed
  by the reclaim; the seq_cst handshake is unchanged.
- Check builds, `zero_or_value` (and `ArrayChunkBits`): the comparison of a proxy's armed
  second-value index and `array_id` with the registry, on every write
  (`ProxySpec::armed_inf_array()`), took the registries' shared lock too: 95 % of
  `shrink_bench_omp_zero_or_value_1024` at 160 threads. It goes through the same per-thread memo.
  At 14 threads, `--mode perf --divisor 10 --iter_max 100`: 82–89 s → 1.7–1.8 s.
- `TCSlab::alloc_block()` tested its capacity after computing it: `block_allocator -
  sizeof(void*)` is unsigned and wraps for a slab smaller than the back-pointer, and the
  quotient, cast to `int`, could come out positive (a slab of 1 byte and elements of 3: a
  capacity of 1,431,655,763) and pass the "does not fit" test. The sizes are now compared before
  the subtraction, and an element size below 1 is refused with them. No size the library asks
  for reaches it (Coverity 661091).
- `vectorisation_demo` and `multi_mat`: `main()` catches `std::exception` and says it, as the
  other examples do, instead of ending in `std::terminate()` (Coverity 661907, 661959, 661960).
- Documentation brought back in line with the sources, in both languages.
  `docs/algo_pattern_fr.md` §5 lacked the paragraph on the OMPT pass
  (`Chunk::compress_to_motif()`). `docs/developer_settings.md` lists the CMake options it
  missed (`SHRINK_BENCH_INT64_SIZES`, `CHUNK_VALIDATION_UNITY_BATCH`, `GCC_TOOLCHAIN_DIR`,
  `USE_MOLD`, `CHUNK_VEC_REPORT`, `GETTING_STARTED_LIB`, `BOOL_ARRAYS_LIB`) and, in a new §8,
  the variables the tools and the harnesses read (`CHUNK_BUILD_ROOT`, `CHUNK_BENCH_EVENTS`,
  `CHUNK_POOL_REF_WAIT`, `CHUNK_POOL_POLL`, `CATCH_CRASH_EXE`, `CLANG_FORMAT`, `FK_SECONDS`,
  `BLK_HUGE`, `CHUNK_VALIDATION_CHILD`); `docs/environment_variables.md` §2 has
  `SHRINK_ARRAY_BUILD_EXAMPLES` and `SHRINK_ARRAY_BUILD_TESTS`. `docs/contributing.md` names
  `check_public_api`, `check_format` and `check_doxygen` in its table of checks;
  `docs/formal_models.md` says the native models keep their assertions (`-UNDEBUG`);
  `docs/synchronisation_dekker.md` §4.3 says a `membarrier` that fails once registered stops the
  program. `docs/README.md` no longer claims every document has a French version
  (`docs/threading_backends.md` is in English only) and its code table names `pattern_array.h`,
  `threads.h`, the `fetch_content` example and the other formal models. The Doxygen comments of
  the constructors and of `resize()` state what a thread may do inside a region.
- `resize()` inside a parallel region compressed chunks of other arrays under their writers.
  Its drain ran the compression pass, which takes no writer handshake there, on the calling
  thread's WHOLE candidate list: right for the array being resized, which only that thread
  writes, wrong for a shared array the thread had also written -- another thread could be
  writing that chunk (ThreadSanitizer: 9 races on a probe; a lost or torn value in principle).
  The copy constructor goes through the same drain. The pass now runs on the resized array's
  candidates only, the others set aside for its duration (`internal/own_candidates.h`,
  `drain_own_candidates()` in the three variants). For zero_or_value the same call also
  sweeps, on the calling rank, that array's candidates queued under an earlier second value,
  which the in-region drain left pointing into the reallocated chunk list.
  `region_storage_validation`'s whole-team case now grows each thread's array with chunks of
  the shared one in its list.
- OpenMP without OMPT (libgomp): a thread whose rank cache was left over from another team --
  `begin_parallel_phase()` forgotten, then a team that changed -- used another thread's per-rank
  structures (deferred queue, candidate list, flush lock), which that thread uses without a
  lock: hangs, crashes and aborts in about half the runs of `phase_protocol_validation`'s new
  "stale rank cache, no begin" case. Nothing clears the cache there (no implicit-task callback),
  so the builds without checks now re-read the rank on every use (`get_thr_rank_fast()`: one
  `omp_get_level()` and one `omp_get_thread_num()`; 2 to 5 % of `shrink_bench`'s computation
  time at 8 threads); check builds keep stopping on the stale cache. 45 runs out of 45 pass.
- Packaging (P1 and P4 of the audit). **Spack**: the package was built for the build host
  whatever `target=` said -- the project's `-march=native` (`-mcpu=native`, `-tp=host`) came
  after the wrapper's flag and won; `CHUNK_NATIVE_TUNING=OFF` (new, default ON) adds no target
  flag and the recipe sets it. The recipe also declares what it used without saying: the C
  compiler (`formal/`), `bzip2` for `+examples`, Python for the tests. **`-DNDEBUG`**: the
  project's own Release and RelWithDebInfo flags dropped CMake's `-DNDEBUG`, so a release
  library carried its `assert()`s; they are compiled out again, except in the builds that are
  there to check (`CHECK_OPTIM`, sanitizers, Debug). The reverse for `formal/`: its native models
  state their property with `assert()` and were built with CMake's C Release flags, `-DNDEBUG`
  included -- the six `formal_native_*` tests passed whatever the model did; they now build
  with `-UNDEBUG`. **Workflows**: the 21 `uses:` are pinned by commit, the release in a comment,
  and `.github/dependabot.yml` proposes the updates. The `doc` target is defined at the top
  level only (a consumer's own `doc` target stopped its configure).
- std backend: a thread building, re-chunking or destroying a large array outside a region could
  deadlock with another thread's team. It held the cold-path lock and waited for the pool (one
  team at a time) while a member of the team holding the pool waited for that lock to build its
  own array (ThreadSanitizer's lock-order inversion in `bulk_write_team_validation`). The pool
  no longer makes a team wait: when another thread's team holds it, the new team runs as a team
  of one on its caller, as a nested or application-phase team already does.
- Building or destroying arrays from several threads of a region at once corrupted the heap
  (a crash in most runs: 15 of 16 with a work array per thread and per iteration). The
  constructor grew the per-array tables that every write of every other thread indexes, and
  wrote the identity registry and the manager's totals with no lock. Building and destroying an
  array now take a cold-path lock (`internal/cold_path_lock.h`); the tables indexed by
  `array_id` no longer move their rows (`internal/stable_table.h`); the two thresholds the write
  path reads are atomics; the identity registries have a reader lock. A work array destroyed
  inside a region is torn down on its thread's rank alone (`internal/array_users.h`: no walk of
  the other ranks' lists, no pass over the other arrays, no allocator set of its own) after one
  `read_reclaim` grace period. Arrays another rank has used keep the sequential-point teardown.
  `resize()`, copies and `rechunk_to()` stay sequential points. `region_storage_validation`
  gains the whole-team case; `docs/getting_started.md` states the contract.
- A deferred enqueue raised its writer flag on the chunk it was deferred for -- a chunk the
  compressor had marked -- so a pass parked in `start()` (whose window is "no writer on a chunk
  I marked") waited for the very writers its mark had sent to the slow path, on its yield
  budget, and could give up. The enqueue publishes a tag instead of the chunk: the flushers
  still see a raised flag, the compressor's window skips it. `formal/writer_compressor.c`
  already lowered the flag before the capture.
- `wait_before_materialise()` (`zero_any`, `pattern`) waited for the compressor's mark with a
  pause and no yield, against the library's rule (1000 pauses, then a yield): with more threads
  than cores (the std backend, TBB) a thread could spin a whole time slice while the pass that
  holds the mark waited for a core. It yields after 1000 pauses now.
- `dekker::heavy_barrier()` ignored the return of `membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED)`.
  Under the asymmetric handshake the writers announce with a compiler barrier only, so a failed
  call -- after a successful registration it cannot fail but under a seccomp filter installed
  later -- would let a pass free a buffer a writer is in, silently. It stops now, in every
  build, with the errno.
- `zero_any` (and `pattern`): a pass handed its candidates' queue flags back after all its
  scans, while the scan had already lowered the mark of a candidate it dropped. A writer that
  found that mark down and made the chunk constant met the flag still "queued" and gave up; the
  pass then released the flag, and the chunk stayed constant and materialised in no list until
  its next write: memory not reclaimed, no wrong value -- F6's window, open again for those
  chunks (`formal/queue_handback.c`'s `MUT_HANDBACK_AFTER_UNMARK`). Each candidate's flag is
  handed back in the scan, before its mark can go down, as `zero` and `zero_or_value` do.
  `docs/formal_models.md` section 8.7.
- A second `ParamChunk::init_indexes()` with the other argument (or `init_indexes(false)` after
  the first array had run `init_indexes(true)`) changed the allocation block size: the
  allocators registered afterwards cut their slabs to another size than the earlier ones of the
  same element type, against the one-slab-size-per-type invariant `slab_of()` relies on
  (`tc_allocator.h`). Out of contract (`docs/api.md`: once), now enforced: a call that would
  change the block size is ignored with a warning; the same size again is a harmless repeat.
  `pool_accounting_validation` checks both.
- `a.push_back(a(i))` and `a.resize(n, a(i))` past the capacity, with an element of more than 16
  bytes (passed by reference): `resize()` ran first, and its compression pass could release the
  buffer `a(i)` points into, which the write of the new element could then reuse, filled with the
  constant, before the value was read. Both copy the value before `resize()`, as `assign()`
  already did. `open_types_validation` checks both on a 24-byte struct (`zero` and `zero_any`:
  `zero_or_value` has no `a(i)` on a non-`bool` element).
- `docs/api.md` section 7 said `set_thread_chunk_size(n)` gives "the chunk size of the arrays the
  calling thread builds next"; a new array still took the process-wide size. It routes the
  calling thread's chunk allocations, and the library sets it before each of them: the table
  and the function's comment say so, and point to `set_active_chunk_size()` and
  `a.set_chunk_size()` for a new array's geometry.
- `import_array(b, export_array(a))` returned `false` whenever `a` had another chunk count than a
  fresh array of its size: after `a.reserve(10000)` (40 chunks for 10 elements), or for an array
  grown by 256 `push_back` (1 chunk, against 2 for a fresh array of 256). A checkpoint could
  refuse to load. `import_array()` imports the chunks that hold elements, the first
  `ceil(size / chunk_size)`, and refuses only another chunk size or a state with fewer chunks
  than its size needs. `chunk_exchange_validation` imports both kinds of array.
- A negative size was accepted: `a.resize(-1)` threw nothing and left `size() == -1`,
  `a.reserve(-1)` raised the capacity to 4,294,967,296, `Array(-1)` built an array. They throw
  `std::length_error` now, as `std::vector`, and leave the array as it was. `shrink_bench
  --test_stl` checks the three, against `std::vector` too.
- `a.set_chunk_size(n)` on an array that had chunks changed its geometry without moving them:
  on 100,000 written elements, 64 -> 1024 left the 1563 chunks where they were, and `a[99000]`
  read another element; the size also became the birth size of every array built afterwards.
  The documentation said "the geometry of an EMPTY array", nothing enforced it. Another size on
  an array that has chunks is now refused with a warning, geometry kept (`rechunk_to()` moves
  the elements); the same size stays accepted. `advisor_demo`, which built an `Array(1)` then
  set another size, builds an `Array(0)`. `rechunk_validation`'s refusal case checks it.
- `rechunk_to()` corrupted the heap on an array ending less than a chunk below `INT_MAX` (an `int`
  index, `_dyn` build): a `zero_any::Array<int8_t>` of `INT_MAX` elements, its last ten written,
  re-chunked 4096 -> 2048, stopped on a glibc `malloc.c` assertion. The end of the last chunk,
  `base + chunk_size`, was computed in the index type and wrapped negative, and `hi - lo` became a
  huge `memcpy` length. Computed in `long long`, as `write_range_by_chunks` already did.
  `index64_validation`'s `int-top-rechunk` case re-chunks such an array to the other geometry and
  back.
- The sign of a zero was lost: a `-0.0` written to an array read back `+0.0` (`1.0 / x` went
  from `-inf` to `+inf`, `std::signbit`, `std::copysign`, `std::atan2` changed), for `float`,
  `double` and `std::complex`, in the three variants -- a write onto a chunk compressed to zero,
  a whole chunk written in parallel, a materialised chunk rewritten then compressed, a single
  element onto a `+0.0`, `fill_range`/`fill_range_mt`. Every "already there" / "the run is the
  constant" decision compared with `==`, and `-0.0 == +0.0`. They now go through
  `same_value()` (`internal/same_value.h`): the bits for `float` and `double` (a run of `-0.0`
  is no longer compressed to the zero constant -- in `zero_any` it compresses to its own
  constant; a run of one NaN payload now compresses and reads back bit for bit), `==` and the
  sign bit for `long double`, part by part for `std::complex` and `std::array`, `==` for the
  other types. An aggregate of floating-point members keeps comparing them with its `==`
  unless it specialises `shrink_array::ValueIdentity<T>`. `zero_or_value` orders its second
  values to match (`ValueLess`): `-0.0` can be a second value. `signed_zero_validation` checks
  every case, bit for bit, in each variant.
- `ChunkSizeAdvisor` counted a `pattern::Array`'s motif chunks as materialised, at every size:
  its model of an array of motifs was off by two orders of magnitude, and blind to what a
  coarsening loses. With the decider applying, a `pattern::Array<int>` of 2^26 elements (8
  local motifs, irregular offsets) was re-chunked 256 -> 1024, from 5.25 MB to 270 MB, and
  kept there. The advisor now counts them (`ChunkSizeEstimate::motif`, new, appended; `mat`
  no longer includes them; `mem_bytes` adds one entry per distinct motif) and forecasts them
  at the other sizes from the motif pointers and offsets. `pattern_validation` gains S.
- CI: the `check_*` step ran 8 of the 10 `scripts/check_*.py`; `check_public_api` (the frozen
  interface) and `check_inline_bodies` ran under ctest only. Both are in the loop.
- `README.md` and `README_fr.md` gave the version as 0.1.
- `check_doxygen.py` run by hand with no Doxygen installed ended on a
  `FileNotFoundError` traceback: it says Doxygen is not found and exits 77 (skipped).

## 0.2.1 -- 2026-11-05

### Added

- `CHUNK_X86_MARCH` (GCC and Clang on x86-64, default `native`): the `-march=` of the optimised
  builds, for a binary that must not depend on the machine (`docs/developer_settings.md`). The
  nightly's ASan job sets `x86-64-v3`: with `-march=native` on Zen 5, GCC 14 aligns slots of the
  frame on 64 bytes, ASan's fake stack can leave them 32 bytes off, and an aligned `vmovdqa64`
  faults (\#GP). The two SEGV of 2026-11-02 in `read_reclaim_validation` of `zero_one_value`
  were this, not the library: on EPYC 9V45 the same commit fails 3 times of 3 at `native` and
  passes 2 times of 2 at `x86-64-v3`.

- The check builds' `FLUSH_PROBE` also rejects non-canonical op pointers and vptrs, and looks at
  `flush_deferred` too. On a failure the nightly's ASan job prints the CPU, the stacks of every
  thread, and keeps the core dumps and the binaries as artifacts.

- `thread_speedup.py --merge DIR [DIR ...] --out OUTDIR` merges the output
  directories of several runs -- the jobs of a `--pool` over several nodes,
  each of which only draws the executables it claimed -- into one, with no
  measurement: the rows of every `summary.csv` and `results.csv`, a
  `meta.json` naming the directories merged, and `plot.gp` and `speedup.png`
  drawn by the functions a run uses, titled with the hosts merged. An
  executable found in several directories is taken from the first one given;
  that and any difference of CPU model, target, `--mode`, `--multiply`,
  `--metric`, `--numa`, executable arguments or thread list are warnings on
  stderr. shrink_gui's Graphique tab has an « Enregistrer l'image… » ("Save
  image…") button on each of its pages: the chart as shown (a pool's gathered
  curves included), as PNG or SVG.

- `dyn_geometry_validation` checks the refused chunk sizes: wider than
  `cst_buffer_size` or neither a power of two nor a multiple of 64,
  `chunk_size_error` says why, and `set_active_chunk_size` and `rechunk_to`
  keep the geometry with one warning on the rechunk channel; `pool_accounting`
  checks that `need_load_balance_list_block_free()` is false once the pools
  are released. Both were at 0 % in the coverage report.

- `bit_words_validation` checks an empty bitmap (`nbits = 0`: `fill_bits`
  writes nothing, `scan_words` says all_false, `count_ones` 0), and
  `log_validation` a `std::complex` and a type with no formatter on the
  message path (not only in the fatal children), still with no allocation:
  branches the coverage report had at 0 %.

- Two header-only harnesses for helpers the coverage report had at 0 % or
  close (`ctest -L validation`): `env_parse_validation` (every refusal of
  `parse_number`, `env_number`'s fallback with its warning on the env
  channel, `arg_number`, `env_flag`) and `chunk_index_sample_validation` (the
  chunk sample of the compression-residual messages: the text, "(+N more)",
  the cut at the stack buffer's size, a 64-bit element index).

- Coverage builds: with `-DCHUNK_GCOV_DUMP` (the nightly's gcov job sets it)
  the fatal report writes the gcov counters before `abort()`, which skips the
  exit handler that writes them: the paths the death tests take (the fatal
  report, the invariant and bad-pointer checks) are counted, where they read
  as never run.

- The TBB bench's 64-bit twins, `shrink_bench_tbb_<variant>_int64_<1024|dyn>`
  (std builds with oneTBB and `SHRINK_BENCH_INT64`, as the std benches have
  theirs): the TBB regions past INT_MAX elements. `thread_speedup.py
  --all-exec --int64` takes them too.

### Added (cpp26)

- The `cpp26` branch, made from `cpp23`: every target is built as C++26
  (`cxx_std_26`, or `-std=gnu++26` when CMake does not know it). The code still
  needs only C++23 outside the reflection parts, so GCC 14 and Clang 18 keep
  building it.
- C++26 static reflection, CMake `CHUNK_REFLECTION` (`AUTO` by default, `ON`,
  `OFF`): on where the compiler has it -- GCC 16, `-freflection`, PUBLIC on the
  libraries; Clang 23 has none. Recorded in `chunk/build_options.h`
  (`CHUNK_OPT_REFLECTION`, `build_config::has_reflection`), which stops a
  consumer compiled without it.
- `shrink_array::SoaArray<T>` (`chunk/soa_array.h`, experimental): a struct
  stored as one compressed array per member, the columns and the `a[i]` proxy
  (members named as in `T`) generated by reflection, so each member compresses
  on its own runs. `a[i].m = x`, `a[i].m += x`, `a.get(i)`, `a.set(i, v)`,
  `a.column<^^T::m>()`, `for_each_column()`, `shrink_to_fit()`. Checked by
  `examples/soa_array/soa_array_validation` (ctest, reflection builds only).
- `SoaArray<T>`: a member picks its own storage with a C++26 annotation
  (P3394, in GCC 16): `[[=shrink_array::soa::plain]]` stores it in a
  `std::vector` (a member that never repeats, where chunks only cost),
  `[[=shrink_array::soa::column<C>]]` in a `C<M>` (`zero::Array`,
  `zero_one_value::Array`, `zero_any::Array`, an alias), the other members
  keeping the array's `Column`; `a[i].m`, `get`, `set`, `column<^^T::m>()`
  follow. Other libraries' annotations are left alone; two storage
  annotations on one member stop the build. `soa_array.h` and
  `cst2_hash_validation` now name the arrays `zero_any::Array` and so on
  (they used the deprecated `ArrayChunk2A`: a warning in every reflection
  build).
  `soa_array_validation` checks the column types, the values after parallel
  writes, compound ops and `set()`, and that the annotated compressed columns
  compress.
- `SoaArray<T>`: the range operations, member by member, each column
  keeping its own semantics (a chunk filled with one value takes it as its
  constant, a `soa::plain` column is a `std::fill` / `std::copy`):
  `fill_range(a, lo, hi, v)`, `copy_range(a, lo, hi, src)` from structs (the
  AoS to SoA transposition, through a projection of each member),
  `fill_range_mt`, `copy_range_mt` (every thread of a phase on its own
  range), `read_range(a, lo, hi, out)` back into structs (a compressed chunk
  a fill of its member); `resize(n)`, `resize(n, v)`, `push_back(v)`; and
  `column_count`, `column_name(k)`, `column_at<k>()`, `field<k>(v)` to walk
  the columns by index. The file comment no longer announces a `compress()` it did not
  have. `soa_array_validation` checks them against a `std::vector`, the
  `_mt` forms from a phase, a range past the end clamped.
- `SoaArray<T>`: a member that is a struct -- with no `operator==`, so no
  `ArrayChunkElement` -- or an array is stored part by part, one column per
  scalar it holds (`pos.x`, `pos.y`, `w[1]`, `in.q[0]`), at any depth; a
  struct that is an `ArrayChunkElement` stays one column unless marked
  `[[=shrink_array::soa::split]]`. The generated `a[i]` mirrors it:
  `a[i].pos.y = 1`, `a[i].w[1] += 2`, nested aggregates and arrays of
  proxies (brace elision builds them from the flat list of columns).
  `column<^^Cell::pos, ^^Vec3::y>()` finds a column by its path of members;
  a storage annotation on a struct member goes to every part unless a part
  has its own. `soa_array_validation` checks a `Cell3` of 16 columns --
  names, types, the nested writes, `get`/`set`, `copy_range`/`read_range`
  against a `std::vector` -- and that a nested struct with a base class or
  a private member is refused.

### Changed (cpp26)

- GCC 16 only: the branch uses C++26 contracts as such, and CMake refuses a
  compiler without them; `compile_all.sh` defaults to C++26 there
  (`CHUNK_CXX_STANDARD=26`), so it builds the gnu configurations only.
- Contracts: `pre` on `operator[]` and `element_proxy` (`0 <= index <
  size()`), `swap()` (same chunk geometry), the chunks' `clear_st()` /
  `release_st()` (the chunk is materialised); `pre` / `post` on `resize()`,
  `reserve()`, `assign()`, `clear()`, `push_back()`, `emplace_back()`;
  `contract_assert` in the iterator comparisons (one array). `CHUNK_CONTRACTS`
  sets the evaluation semantic (AUTO: enforce in a check build, ignore
  otherwise). `contracts_validation` provokes each kind of violation.
- `chunk/mdspan.h` uses `std::mdspan` and `std::submdspan` only: the
  kokkos/mdspan fallback, `SHRINK_ARRAY_HAS_MDSPAN`,
  `SHRINK_ARRAY_HAS_SUBMDSPAN` and `md::mdspan_source` are gone, and the
  mdspan harnesses no longer skip (exit 77).
- C++26 static reflection is required: the `CHUNK_REFLECTION` option,
  `CHUNK_OPT_REFLECTION`, `build_config::has_reflection` and the
  `SHRINK_ARRAY_HAS_*` macros of `enum_name.h`, `reflect_fields.h`,
  `soa_array.h` and `value_hash.h` are gone, with what the builds without
  reflection used: the hand-written name tables (`event_name_table`, the
  switch of `bstate_name`), the two lists of `ChunkStatsThr` counters, and the
  checks that kept them in step (`table_matches`, `field_count`).
  `enum_name.h` gains `name_of()` and `upper_name_of()`, an enumerator's
  name from its value.
- `CHUNK_RECHUNK_SCHEDULE` reads its kind with `enum_from_name()` and lists
  the kinds it accepts with `enum_names_joined()` (`enum_name.h`, by
  reflection): `sched_names` and `names_match()` are gone. Tests
  `rechunk_schedule_env_<geometry>` and `rechunk_schedule_env_unknown_<geometry>`.
- `for_each_type<TypeList>()` walks the template arguments of `TypeList` by
  reflection (`ForEachTypeImpl` is gone); a tiled `submdspan`'s slices go
  through `template for` and pack indexing (`slices...[k]`) instead of an
  `index_sequence` lambda.
- The atomic maxima (the thread-rank high-water mark, the re-chunk slice
  timings, the RSS peak) use `std::atomic::fetch_max` (C++26) instead of a
  compare-exchange loop; the rank's is still written only when it grows.
- Every deleted copy / move member under `chunk/` gives its reason
  (`= delete("...")`, C++26): copying `AllTypeChunk`, a scope guard or a
  manager now says why in the compiler error. No behaviour change.
- The log and the invariant messages (`fmt_arg`, `log_core.h`; `value_repr()`
  of the `operator<<` diagnostics) print a value with no `std::formatter`
  member by member, by C++26 reflection, where they printed its bytes in hex:
  `{id: 4, z: (1-2i), at: {x: 0.5, y: -3}, secret: 9}` -- bases' members
  first, private ones included, each member by the same rules, an array of
  structs as `[...]`. Still written straight into the stack buffer: no
  allocation, cut with `...`. A union, a class holding one (an anonymous union
  included) and an enum stay in hex; a `std::formatter` of the type's own still
  comes first. Checked by `log_validation` (check 10).

### Fixed (cpp26)

- Chunk2A's second-value index could exhaust the memory: once its probe budget
  ran out, every new distinct value doubled the index again, and the superseded
  tables are kept until the last array of the type goes (6 GB and an OOM kill
  for 300 values of a `struct {int mat; double rho;}`). The index now doubles
  once, then degrades as it always did (`index_publish()`).
- The index hashed only the first 8 bytes of a value: for a wider type -- a
  `Point {x, y, z}` with one `x`, a `{int mat; double rho}` whose first 8
  bytes are `mat` and padding -- every value landed on one slot and the index
  degraded (O(N) lookups, reclaims refused). `value_hash()`
  (`chunk/internal/value_hash.h`) now hashes the whole value; a scalar of at
  most 8 bytes keeps the former formula, a floating zero normalised. With C++26
  reflection, a class whose `operator==` is the defaulted member one is hashed
  member by member, so equal values with different padding, or `-0.0` against
  `+0.0`, share one identifier (without reflection they may take two: memory,
  not a wrong value). Checked by `examples/cst2_hash/cst2_hash_validation`.
- The ring log (`CHUNK_RING_LOG_DEBUG` / `_OP`) printed 9 of its 27 events as
  `UNKNOWN` -- the cst_val transitions, the flushes, the async lambdas,
  `store_value`: `event_name()` had no case for them, hidden from `-Wswitch`
  by a `default:`. The names are now the enumerators without `ev_`,
  upper-cased (`EV_END_SLOW_PATH` becomes `END_SLOW_PATH`), derived by C++26
  reflection where the compiler has it (`chunk/internal/enum_name.h`); the
  switch, kept for the other builds, has no `default:` any more and a
  `static_assert` checks it against the enumerators. `bstate_name()` (pool
  ledger) gets the same check.
- CHUNK_STATS: the 35 counters of `ChunkStatsThr` are summed by reflection
  (`chunk/internal/reflect_fields.h`: `add_fields`), where they were listed by
  hand twice -- the totals of `ParamChunk::chunk_stats()` had already lost
  `released`. The hand lists stay for the builds without reflection, and a
  `static_assert` on the field count stops a reflection build when a counter
  is added without them. A variant is skipped only when every counter is zero
  (it tested 16 of the 35: a variant with only `still compressed (at exit)`
  lost its block), and `still compressed (at exit)` is no longer printed twice.
  A reflection build adds, after each block, one line with every counter:
  `CHUNK_STATS_JSON {"variant": "chunk2a", "alloc": ..., ...}`, which
  `python/bench_runner/parsing.py` reads first (the text blocks for the other
  builds; both give the same figures).
- The name tables that a parser walks are checked against their enum by
  reflection (a `static_assert` in the reflection builds; the tables stay for
  the others): `channel_names` and `level_names`, which `CHUNK_LOG` is parsed
  against, against `logging::Channel` and `Level` (`log_core.h`; a channel
  added without its name was an empty string, two names swapped a wrong
  channel, both silent); the `CHUNK_RECHUNK_SCHEDULE` kinds against
  `ChunkPmpSched`, the if-chain that parsed them now a loop over that table
  (`sched_names`, `array_chunk_rechunk_inline.h`). `enum_name.h` gains
  `names_match<E>(names, prefix, first)` for it.

### Added (cpp23)

- `std::expected` forms that say why they refuse, beside the existing calls
  (unchanged): `try_rechunk_to(n)`, `try_set_chunk_size(n)` and
  `try_set_active_chunk_size(n)` return `std::expected<void, ChunkSizeError>`
  (`not_pow2_or_x64`, `exceeds_buffer`, `fixed_geometry` in a static build);
  `ArrayChunk2A<T>::try_reclaim_second_values()` returns
  `std::expected<std::size_t, ReclaimRefusal>` (`parallel_region`,
  `rechunk_in_progress`, `degraded_index`). They write no message. The
  re-chunk decider uses `try_rechunk_to()`: a refused size is a warning and
  no event, where it was recorded as performed. `rechunk_to(0)` is now a
  refusal with its warning, where it returned silently.
- A fourth GenMC model (`formal/`) checks the in-place constant switch of `store_range_mt`
  (`fill_constant_st`) against the pass.
- `multi_mat_md`, the multi-material example in 2D through `mdspan`
  (`examples/multi-mat`): every material occupies a box of an `nx x ny` mesh
  at each iteration; its arrays are seen through `md::tiled_view` (chunk
  arrays, a tile per chunk) or a standard `mdspan` (`std::vector`), and one
  loop body serves both. `--chunk-mode tiles` (default) walks the chunk
  arrays tile by tile (`md::update_tiles`, `md::fill_box`, four tiles read in
  one go for the reduction), `elements` cell by cell in the same layout,
  `rows` cell by cell row-major. 1600 x 1600, 50 chunk materials, 2 threads:
  663 MiB peak in tiles, 1847 MiB row-major, 3912 MiB all `std::vector`;
  the same result in all four. The team splits the cells of all the boxes
  at once, tile by tile (row by row in `rows`): at 64 threads the most
  loaded thread writes 1.01 times the mean, where an `omp for` per box left
  49 threads idle (7.8 times). `--verify` replays the run serially and
  compares every array and the result; tests `multi_mat_md_tiles`,
  `multi_mat_md_elements`, `multi_mat_md_rows`, `multi_mat_md_balance`
  (`--max-imbalance`), and `multi_mat_md_embedded` with an OMPT tool.
- `md::mdspan_source` (`chunk/mdspan.h`): which `mdspan` the build uses and
  whether it has `submdspan` (`"std::mdspan, no submdspan"`,
  `"kokkos/mdspan, submdspan"`, ...); `mdspan_validation` and `mdspan_tiles`
  print it.
- `md::copy_box_out(f, lo, hi, dst)` and `md::copy_box_in(f, lo, hi, src)`
  (`chunk/mdspan_tile_ops.h`, experimental): a box of a view copied to or
  from a contiguous row-major buffer of its extents -- an MPI halo, an HDF5
  hyperslab, a library that wants a `T*`. Tiled view: tile by tile (a tile
  read in one go, a uniform one a fill; a tile the box covers written in one
  go with no read, as one constant when uniform), with `[first, last)` tile
  range overloads for an `omp for`. Strided view: a range op per row. 1024 x
  1024 box of a 2048 x 2048 tiled field, against `f[i, j]`: out x 3.9 to 4.4,
  in x 8.5 to 9.2 (`mdspan_tiles`). Harness `mdspan_copy_box_validation`
  (every variant, 1 and 8 threads).
- `md::copy_box_out_mt(f, lo, hi, dst)` and `md::copy_box_in_mt(f, lo, hi,
  src)` (`chunk/mdspan_tile_ops.h`, experimental): called by every thread of
  a region (`copy_box_in_mt` in a phase), they share the box themselves, with no
  `omp for` and no communication -- a tiled view's tiles of the box dealt by
  their cells inside it, a strided view's cells split on chunk boundaries.
  A band of a 2048 x 2048 field, against the tile range overloads in an
  `omp for`: x 1.7 on 1 thread (one call per thread, not per tile), x 3.0
  to 3.3 on 2 (the band no longer falls to one thread).
- `md::for_each_tile_mt`, `md::update_tiles_mt` (over every tile, or over the
  tiles a box touches) and `md::fill_box_mt` (tiled and strided views)
  (`chunk/mdspan_tile_ops.h`, experimental): called by every thread of a
  region (the writing ones in a phase), they deal the tiles -- or a strided
  box's cells, on chunk boundaries -- themselves, as `copy_box_*_mt` do, with
  no `omp for`, no communication and no barrier. A band of a 2048 x 2048
  field filled: x 7.4 to 7.8 against `fill_box(f, lo, hi, v, k, k + 1)` in an
  `omp for`; over the whole view the same time as the `omp for`. The serial
  forms now run on the same code (`detail::read_tiles`, `write_tiles`,
  `fill_tiles`, `fill_strided`), behaviour unchanged.
- `md::stencil_tiles(src, dst, h, boundary, kernel, on_uniform)`, its tile
  range form and `md::stencil_tiles_mt` (`chunk/mdspan_tile_ops.h`,
  experimental): a stencil tile by tile, the kernel reading the tile and a
  halo of `h` cells (`md::halo_block`, `in[i - 1, j]`) gathered in one go
  from the neighbour tiles, the border `md::constant_boundary<T>{v}` or
  `md::clamp_boundary{}`, the result tile of another array written back in
  one go (as one constant when uniform); `on_uniform` computes a tile whose
  neighbourhood is one value in O(1). Any rank. One diffusion step on a
  2048 x 2048 field: x 7 to 10 against the element loop, x 11 to 23 with
  `on_uniform`. Harness `mdspan_stencil_validation` (every variant, 1 and 8
  threads) and `mdspan_stencil_alias` (a stencil in place refused).
- `md::as_const(f)` (`chunk/mdspan.h`, experimental): the read-only view of
  any view of this library -- row-major, tiled, a sub-view -- over `const
  Array`, `f[i, j]` the value; `std::as_const(f)` leaves an mdspan's element
  type writable. Replaces `md::view_type<const Array, Layout>(f)` in the
  examples and the harnesses. `md::tiled_view_type<Array, T0, T1, ...>`, the
  type of `md::tiled_view<T0, T1, ...>()`, as `md::tiled_dyn_view_type` is
  `md::tiled_view_dyn()`'s; `md::tiled_view` checks it has one tile extent
  per dimension.

### Changed (cpp23)

- `RechunkDecider::print_rechunk_log(FILE*)` writes its table with
  `std::println(out, ...)` instead of `std::fprintf`: same columns, same
  output.

- `copy_range_mt` and `transform_range_mt` (and the tile writes of
  `md::update_tiles` inside a region) store a chunk span in one block copy and
  compare its values in a loop the compiler vectorises: the copy into the
  span buffer tested `same && v == v0` element by element, which kept both
  loops scalar. 1.7 times faster in cache (4 arrays of 64 K doubles,
  1 thread), about 1.3 times out of cache (2.56 M doubles, v1).
- `fill_range_mt` and `copy_range_mt` on Chunk2A switch a compressed chunk
  that a uniform span covers whole to the span's value in place, as
  `fill_range` does serially (`fill_constant_st`'s compressed branch, within
  the write session: no buffer allocated, filled, queued and scanned by the
  end-of-phase pass). A materialised chunk is still written and left to the
  pass, which waits out the readers before freeing it. `md::update_tiles` and
  `md::fill_box` inside a region inherit it: `mdspan_tiles 2048`, 2 threads,
  the update by tile 3.1-4.3 -> 1.1-1.25 ms, the 64 rectangles 5.4-7.0 ->
  2.9-3.2 ms. `constant_switch_validation`: the same switches from every
  thread, with the chunks materialised counted inside the region (64 per
  step before, 0 now).
- `copy_range` on a v1 or Chunk2 chunk compressed at a constant c skips the
  source elements equal to c before arming the chunk: a span equal to c
  writes nothing, a span that starts with c materialises the chunk from its
  first other value. They went through the proxy one by one (every write a
  no-op): 2.56 M doubles, 1 thread, a source equal to the chunk's constant
  3.5-7.4 -> 0.7 ns per element (v1 at first_value, Chunk2 at either
  constant). `internal::held_constant()` gives the constant of a compressed
  v1 / Chunk2 chunk (`holds_constant()` is built on it).
  `copy_range_constant_validation`: cases `same-first`, `same-part`,
  `same-second` and `prefix`, the materialised chunks checked on every
  variant.
- `fill_range` on Chunk2 compresses a chunk it covers whole with one of the
  array's constants (first_value or its second value) at once, as v1 does for
  first_value: a materialised chunk releases its buffer through the pass's own
  clear (`clear_st(num, ind)`), with nothing stored, scanned or queued; a
  chunk compressed at the other constant is armed, then compressed rather than
  queued. 2.56 M doubles over materialised chunks: 0.85-1.1 -> 0.08-0.15 ns
  per element, and no chunk left materialised (100 % before, until the next
  pass). `fill_range_constant_validation`: Chunk2 checked after a fill of a
  constant (the old code leaves 40 of 40 chunks materialised), new scenario
  `second-value`.
- `chunk/mdspan.h` and `chunk/mdspan_tile_ops.h` are declared experimental
  until 1.0: their names and signatures may change in a MINOR release (the
  versioning policy above, `docs/mdspan.md`, the headers). Checked on Clang 19
  with libc++ 19, whose `std::mdspan` has no `submdspan`: every view and tile
  harness passes, the `submdspan` ones are skipped.

### Fixed (cpp23)

- `set_chunk_size(n)` / `try_set_chunk_size(n)` refuse a size wider than
  `cst_buffer_size` (`exceeds_buffer`), as `rechunk_to()` does: they accepted
  it on an empty array, whose chunks then read the shared constant buffers
  out of bounds.
- docs/mdspan(_fr).md, `chunk/mdspan.h`: `std::mdspan` comes with
  libstdc++ 16, not 15 (libstdc++ 15 has no `<mdspan>`: a GCC 15 build takes
  kokkos/mdspan, or none). The checked toolchains now include GCC 15 (with
  kokkos/mdspan, and alone: the `mdspan` harnesses skipped) and GCC 16 in
  C++23 (`std::mdspan`) and C++26 (`std::submdspan`, which the tiled layouts
  serve as kokkos's does).
- `fill_range` on v1 compresses a whole chunk filled with first_value at once
  again (its buffer released), and passes over a chunk already at first_value:
  the guard of that shortcut asked for both spellings of `Chunk::clear_st`,
  and since the OpenMP / serial policy split each build has only one, so it
  never ran -- every element went through the proxy. The guard is now the
  variant (`internal::is_v1_v`), with a `static_assert` on the spelling of the
  build. `fill_range` and `fill_range_mt` also pass over a v1 or Chunk2 chunk
  already compressed at the value, whole or in part (`fill_range_mt`: with no
  deferred write pending on it): 2 to 9 ns per element before, about 0.01 ns
  now (2.56 M doubles, 2 threads). `fill_range_constant_validation` checks v1
  after a fill of first_value, and a new scenario `constant` (a fill of the
  constant the chunks hold, serial and from every thread: nothing
  materialised).
- The library did not compile with libc++ 19: `chunk_size_advisor.h` used
  `std::views::enumerate`, which libc++ has from 20 only (an index loop now).
- A harness that must stop, run where it has nothing to check (exit 77), was
  reported failed: `expect_stop.cmake` reports it skipped
  (`mdspan_submdspan_misaligned` without `submdspan`).

### Changed

- The internal names that kept an old variant label (`chunk2`, `chunk2a`,
  `V1`/`V2`/`V2A`) say the variant: the candidate lists are
  `zero_candidates(_thr)`, `zero_one_value_candidates_thr`,
  `zero_one_value_candidates_map_thr`, `zero_any_candidates(_thr)` (were
  `chunk_to_compress_potentially(_thr)`, `chunk2_...`, `map_chunk2_...`,
  `chunk2a_...`), with their accessors and `remove_from_<variant>_candidates_*`;
  their types `ZeroComp`, `ZeroCompThr`, `ZeroOneValueComp`,
  `ZeroOneValueCompThr`, `ZeroOneValueCompMap`, `ZeroOneValueCompByValue`,
  `ZeroAnyComp`, `ZeroAnyCompThr`, `ZeroAnyCompOf` (were `TChunkComp`,
  `TChunk2Comp`, `MapTypeChunk2P`, `TChunk2AComp`...); the concepts
  `IsZeroOneValue`, `IsSecondValueFamily` (were `IsChunk2`,
  `IsChunk2OrZeroAnyFamily`); the class aliases `SecondValueBase`, `Manager`
  (were `Base2`, `ManagerChunk2`, `ManagerChunk2A`), `CandidateTuples`,
  `CandidateValueTuples` (were `VecTupleChunk*pInt2`);
  `ZeroOneValueInfArrayPolicy`, `CHUNK_LINK_CHECK_ZERO/_ZERO_ONE_VALUE/_ZERO_ANY`
  and the multi-mat example's `VectZeroOneValue`; the traits
  `is_zero_array_v`, `is_zero_or_value_array_v`, `is_zero_any_family_array_v`
  (were `is_v1_v`, `is_v2_v`, `is_v2a_family_v`), the link-check symbol
  `variant_zero_*_zero_or_value_*_zero_any_*_size_*_dyn_*` (was
  `variant_v1_*_v2_*_v2a_*_...`: a library built before it does not link with a
  program built after) and the `PointerMap` of each variant in messages and
  docs (was `PointerMapChunk`, `PointerMapChunk2A`). Generic `chunk` names (the
  block of CHUNK_SIZE values, common to the variants: `ChunkBase`,
  `ManagerChunk`, `ParamChunk`...) stay.

- The `std::initializer_list` constructor of every array is implicit, as `std::vector`'s:
  `Array<T> a = {1, 2, 3};`, and a braced list as an argument or a return value. It was explicit
  in `zero_one_value` and `pattern` only.

### Deprecated

- `zero_one_value::Array::ManagerChunk2` and `zero_any::Array::ManagerChunk2A`:
  use `Manager`. Kept until 1.0.

### Fixed

- `make_block_compression()` inside a parallel region is refused: check builds stop, the others
  log an error and skip the pass. Run with no proxy, it took no writer handshake and, in `zero`
  and `zero_one_value`, no reader grace period: it compressed chunks other threads were writing
  or reading (stale reads, seen with a `read_reclaim` harness). It is a sequential point; in a
  region the writers' own passes compress. `resize()`, which an owning thread may call in a
  region, drains through `drain_candidates()`, its released buffers held for a grace period.

- `multi-mat-embedded`: its OMPT tool skips the initial task and the barrier that closes a region,
  as the library's does. It hung at start-up under libomp.

- The copy of a `zero_one_value` proxy no longer reads indeterminate values: `second_value_` and
  `ind_` have default initialisers (`ProxyBase`'s `ptr_on_array_` and `num_chunk_` too, Coverity
  CID 661511). No measurable cost in `access_cost` or `shrink_bench`.

- `scripts/coverity_submit.sh` goes on when the compiler module is missing, with `CXX` or the
  `g++` on `PATH`; it stopped before the capture.

- `pattern::Array` compresses to motifs in an OMPT build (clang, Intel: `CHUNK_HAVE_OMPT`).
  There the pass at each region's end is `compress_chunk_to_compress_in_par()`, chunk by chunk
  through `compress()`, which only knew the two constants: no chunk ever became a motif
  (`pattern_validation` failed under clang, the nightly's new clang job). A chunk `compress()`
  leaves materialised now goes through `pattern::Chunk::compress_to_motif()`: the admission rule
  and the `block_clear_motif()` order of the claimed pass (docs/algo_pattern.md, section 5).
  `chunk_exchange_validation` accepts, for pattern, an imported materialised chunk found
  compressed to a motif since (its copy is the motif's second sighting).

## 0.2.0 -- 2026-10-30

### Added

- `access_cost`: a `read-bulk` case, the varied array read through `for_each_value()` (each
  thread its slice): one read gate per chunk and one read section per block instead of both per
  element. On a 2-core x86 VM, 1.10 times the vector against 3.2 for the element read
  (`read-varied`); `python/access_cost.py` lists it.
- `CHUNK_READ_DEP_ORDER=1` (experimental, AArch64, a macro in `CMAKE_CXX_FLAGS`): a pure read
  (`ChunkBase::element_at`) loads the chunk's `values` relaxed, ordered with the loads through it
  -- the element, the second value -- by their address dependency instead of an acquire. On
  Grace every read costs 2.4 ns whatever the chunk's state (x86: 1.1 to 1.8): the hypothesis is
  that each acquire load (LDAPR) holds the next element's loads, and the loop runs at the
  latency of its two acquires (the read gate's and this one). Off by default, under evaluation
  with `access_cost`.
- `examples/bool_arrays`: the two boolean containers of a `zero_one_value` build side by side --
  `zero_one_value::Array<bool>` (a byte per element) and `ArrayChunkBits` (a bit per element, the
  layout of `std::vector<bool>`, neighbouring elements writable from different threads) -- the
  same mask written from every thread, read back, and the data each keeps (`ctest -L examples`).
  `docs/variants.md` gets the section they lacked ("Booleans: two containers": what a
  materialised chunk holds, the write, the threads, when to choose which), and
  `zero_one_value::Array`'s comment names both.
- `CHUNK_BREAK_ON_FAILURE`: under a debugger (gdb, cuda-gdb, gdbserver, lldb, lldb-server, read
  from `TracerPid` and the tracer's name, without allocating), an invariant failure or a
  `FAST_ALLOC_FATAL`, its report written, raises `SIGTRAP`: the debugger stops on the failing
  thread with its frames live, and continuing aborts as before. strace and the other tracers,
  TotalView and DDT do not stop; without a debugger nothing changes. `0` never stops, `1` always
  (`docs/environment_variables.md`). `log_break_on_failure` checks it under gdb and strace.
- shrink_gui 0.15: the partition list also offers the partitions the machine lists at the
  connection check (`[batch] partition_info`: `ccc_mpinfo` on inti, ``sinfo -o '%P %c %D %a'`` in
  `slurm_example.toml`), beside the site file's `[partitions]`, which win for a name in both. Their
  directives, launcher and detached options come from the site's new `[discovered]` templates
  (`{name}`, `{cores}` per node), their architecture from a pattern on the name (`arm`); `skip`
  leaves some out. The table read is remembered for the next start.
  `python3 -m shrink_gui.partitions` shows, on the cluster, what the window reads from it.
- `formal/queue_handback.c`: the end of a compression pass against the candidate marking of a
  writer -- F6 (`docs/formal_models.md`, section 8.8, fixed: see Fixed).
- `thread_speedup.py --pool=<id>` and shrink_gui's "Nodes" field for it: N one-node batch jobs
  share the executable list, as `compile_all.sh --pool` shares its configurations; each job
  measures the executables it claims first (an atomic `mkdir` in
  `<out base>/.pools/<id>/`), in its own output directory. The Chart and Summary tabs of any
  of the jobs gather the whole submission; "Compare with" lists such a submission once.
- `python/ab_settings_campaign.py`: the A/B campaign of the developer settings still read from the
  environment: each value against the defaults, computation time and peak / end memory of
  `shrink_bench` on the three variants at dyn, resumable, with a report of the deviations and
  of the noise (`docs/developer_settings.md` §1). Its first campaign fixed seven settings (see
  Removed); its points left, for Grace, are `CHUNK_RECLAIM_THREADS` and
  `CHUNK_DEFERRED_POOL_KB`.
- `CHUNK_LOG=rechunk=debug`: a `teardown split` line after each re-chunk's phases divides the
  `tmp teardown` phase (the old storage's destruction) into its steps, for zero_any and pattern;
  `scripts/rechunk_trace_summary.py` sums it. Measured on Milan, this phase was 56 % of the
  re-chunk time of `shrink_bench_zero_any_dyn`.
- Fast allocator: a sweep (`trim_reuse_pools()`, a re-chunk, a destruction) also gives back the
  free pages of the value slabs still in use (`TCSlab::advise_free_pages`), as `malloc_trim`
  does inside glibc's heaps: a slab no longer stays resident whole for its last few live buffers.
  The value slabs keep their free-list links aside, so a free element is never written to; the
  last page, the back-pointer's, is kept. The `CHUNK_ARENA_REPORT` `partial` line counts it.
  Measured on Milan (64 threads, three runs each): without it, the end memory is 7.5% higher for
  `zero_any` and 1% for `zero_one_value`, at the same time.
- fast_allocators 2.3.0 (the one above); `CHUNK_FAST_ALLOCATORS_VERSION` follows.
- shrink_gui runs on PySide6 6.2 and 6.3: QFormLayout's setRowVisible() / isRowVisible() came
  with Qt 6.4, and the window stopped at start-up without them (AttributeError in
  machine_panel). `widgets.RowVisibleFormLayout` provides them on every 6.x -- Qt's own from
  6.4, the row's widgets hidden one by one before (a hidden row then keeps its vertical spacing).
- shrink_gui: the Log pane shows, as soon as a job is submitted, the run directory, the
  program's command line and the submission command (`cd <run> && ccc_msub job.sh`, with the
  batch system's answer); until now it stayed empty while the job waited in the queue, since
  run.sh prints its `[gui] command:` line on the node. `submit.log` starts with the command too.
- CI: the per-push workflow builds and validates the std backend with oneTBB, and runs it under ThreadSanitizer.
- CI on AArch64: the per-push workflow runs the OpenMP check build, the examples, the install
  test and the native GenMC runs on GitHub's `ubuntu-24.04-arm` runner too -- the weak memory
  model the handshakes were modelled for, which x86-64 cannot exercise. A nightly workflow
  (`.github/workflows/nightly.yml`, develop every night, any branch on demand) runs on AArch64
  what is too long for a push: the std backend with oneTBB under ThreadSanitizer, and the
  OpenMP `_dyn` build with its re-chunk harnesses.
- OpenMP is no longer required: the library runs its own parallel work on a
  backend chosen at build time, CMake `CHUNK_THREADS=openmp|std|serial`
  (empty, the default: from `OMP`, as before). `std` runs the teams on
  `std::jthread` workers of the library's own, standard C++ only. Recorded
  in the installed package as `chunk_THREADS`; the targets of a std build
  carry `FAST_ALLOC_PMP_STD=1`, and a file compiled without it is refused.
- Under the std backend, the arrays can be written from the program's own
  threads -- `std::thread`, TBB, a pool of its own. Each takes a dense rank
  from a registry (given back when the thread exits); the phase contract is
  the serial one: `begin_parallel_phase()` on every thread that writes,
  `finalize_parallel_phase()` once, after they stopped
  ([docs/getting_started.md](docs/getting_started.md) §5). New harness
  `std_thread_validation`.
- `shrink_array::tbb::parallel_for` (`chunk/tbb_adaptor.h`, header-only):
  `oneapi::tbb::parallel_for` with the phase calls made for it, for a TBB
  program on the std backend. The library never links TBB. CMake
  `CHUNK_WITH_TBB` (off) builds what needs oneTBB: the harness
  `tbb_validation` and the TBB bench `shrink_bench_tbb_<variant>_<size>`.
- shrink_bench runs on every backend from one source
  (`examples/shrink_bench/bench_par.h`): OpenMP builds keep the same
  worksharing pragmas (the reductions are per-thread partials, in every
  build); std builds, which compiled the bench without OpenMP and so ran
  its loops on one thread, now run them on the std backend; the TBB bench runs them
  as TBB tasks.
- The test tooling runs the std backend and the TBB bench. `compile_all.sh` builds
  `std-llvm`, `std-llvm-check` and a ThreadSanitizer build, next to the `std-gnu*` builds of
  `compile_all_gnu.sh` (the builds take longer; ``--target='omp-*'`` leaves them out), and says at
  the start whether oneTBB is used. `bench_and_test.py` runs the std builds by default and the
  TBB bench as the variants `tbb_chunk`, `tbb_chunk2`, `tbb_chunk2a`. `multi_launch_stress.py`
  runs `tbb_chunk2` next to `chunk2` by default, the six variants with `--all_exec`, and its
  build lists include the std builds.
- Spack: variant `threads=openmp|std|serial` (default `openmp`), and `+tbb`
  for the TBB examples (with `+examples threads=std`).
- `compile_all.sh --pool=<id>` (and `compile_all_gnu.sh`): runs given the same id share the
  configurations, each built by the first run that claims it. `shrink_gui` 0.14 uses it for
  its "Nodes" field: `compile_all.sh` over N one-node batch jobs, and "Start: after" waits for
  all of them (`docs/shrink_gui.md`, section 3).

### Changed

- `zero_any`'s second-value table (`GlobalVecHolder<T>::value_by_id`, the value behind each
  identifier) is no longer a static array of 4 Mi values per element type but a pointer to an
  `mmap(MAP_NORESERVE)` reservation made at the type's first array, next to the identifier
  region, sized by `CHUNK_CST_VAL2_ID_LIMIT` when set, its pages given back at the last clear.
  The static array was 332 MiB of bss in the benchmark (32 MiB for a `double`, 96 for a 24-byte
  type): free while untouched for a trivial type, but an element type whose default constructor
  is not `constexpr` (which `ArrayChunkElement` allows) had it dynamically initialised at
  startup -- 101 MB of RSS before `main` for one 24-byte type, 3 MB now -- and every type counted
  against `RLIMIT_AS` and, under strict overcommit, the commit charge. bss of
  `shrink_bench_omp_zero_any_256`: 348 MB -> 0.27 MB. The price: reading a second value back
  loads the table's pointer first (one L1 hit, independent of the identifier's computation).
  Below measurement on Grace (aarch64, 1 thread, spread 0-1%): `access_cost`'s new
  `read-second` case 2.413 ns per element against 2.409 for `read-constant`, which does not
  touch the table, and 2.415 for `read-varied`. On a 2-core x86 VM, within the noise as well
  (`read-second` +1.8% at the minimum; the benchmark at 2 threads -6%, sd 8-10%, HWM
  unchanged). `region_init()` is now serialised by a mutex and
  publishes `region_begin` last: a thread that lost the race could find it set before
  `region_end` and `id_limit`.
- `access_cost`: a `read-second` case, an array all at 1.5 (zero_any: every chunk compressed to a
  second value, read back through the value table); `python/access_cost.py` lists it.
- The version is 0.2.0. Before 1.0 the installed package declares `SameMinorVersion`:
  `find_package(chunk 0.1)` does not find it, ask for `find_package(chunk 0.2)` (the README,
  getting_started and examples/getting_started do).

- The decider's "[RECHUNK-LEARN]" line, written to stdout when a coarsening
  freed less than the model predicted and is disabled for that array, is a
  log message like its neighbours: `CHUNK_LOG(info, rechunk, "learn ...")`,
  shown by default (rechunk=info), on the log's stream (stderr, the sink or
  CHUNK_LOG_FILE) rather than stdout.

- `SecondValueProxyBase::is_cst_val_new_value_is_other_cst_val()` is now
  `passes_mark_pre_check()` (internal): it has dispatched to the variant's
  `pre_check_mark_for_compression()` since September, and its name and comment still said the
  narrow test it replaced ("the chunk at one constant, the other one written"), which invited a
  fix back to the bug. `docs/algo_zero_one_value.md`: an array without a second value is queued
  under `T{}`, not never (`has_candidate_list` is a guard before the flag's CAS).
- The `shrink_bench` executables carry the threading backend of their build in their name:
  `shrink_bench_<backend>_<variant>_<size>` with `<backend>` `omp`, `std` or `serial`
  (`CHUNK_THREADS_BACKEND`), e.g. `shrink_bench_omp_zero_any_256`, `shrink_bench_std_zero_dyn`,
  `shrink_bench_serial_zero_one_value_1024`; likewise `shrink_bench_<backend>_<variant>_int64_<size>`,
  `shrink_bench_<backend>_vector_check` / `_vector_perf` / `_vector_int64_perf` and
  `shrink_bench_<backend>_std_alloc_<size>`. A binary of a `cmake-build-std-*` build now says
  that it runs without OpenMP. The ctest names follow
  (`shrink_bench_<backend>_<variant>_check_<size>`, `shrink_bench_<backend>_vector_check`); the
  TBB bench keeps `shrink_bench_tbb_<variant>_<size>`, and the validation harnesses and the
  fixture names do not change. The old names are gone: `bench_and_test.py`,
  `thread_speedup.py`, `multi_launch_stress.py` and the other scripts look for the new ones
  only (each of the three backend names, a build directory holding one;
  `bench_runner/settings.py`'s `bench_exe_names`), never for the old ones, which a build
  directory rebuilt since still holds, stale. Rebuild to get the new names. shrink_gui still
  reads the old names in the results of past runs. `compile_all.sh --exec=` matches the real
  target names: ``--exec='shrink_bench_*_zero_256'`` for every backend.
- shrink_gui: the options form lists the options in alphabetical order (dashes and case aside)
  rather than in the script's declaration order; the positional arguments stay first, in the
  script's order, since the command line keeps them in it.
- `thread_speedup.py --all-exec` takes the TBB bench too (`shrink_bench_tbb_<variant>_<size>`,
  after the other executables, in the std builds made with oneTBB; none with `--int64`, it has
  no 64-bit declination). shrink_gui's speedup chart reads its names: the colour of its variant,
  the shape of its size, a dash-dot line (dash-dot-dot for B) -- they were grey stars.
- `python/omp_speedup.py` is `python/thread_speedup.py`: its curves are not OpenMP's only, the
  std::thread backend's (`std-*` targets) and the TBB bench's (`shrink_bench_tbb_*`) too. Its
  default output directory is `$CCCWORKDIR/thread_speedup/` (else `~/thread_speedup/`).
  shrink_gui offers it under the new name and reads the runs recorded under the old one as its
  own (reuse, outputs, Chart).
- `zero` re-chunk: the recycled slabs the last slice of the rebuild did not take stay resident
  for the next allocations, instead of being given back to the system (the former
  `CHUNK_RECHUNK_KEEP_RECYCLED=1`, now the only behaviour). Milan, `zero_dyn`, 64 threads, three
  runs each: 11.7 s against 13.2 s of computation, for +0.1% of memory at the end of the run and
  +0.3% at the peak.
- The re-chunk decider reads no environment variable: its thresholds and its test hooks are
  `RechunkPolicy` fields only, which a program sets. New fields: `advisor_grid_bits` (the
  advisor's grid between the powers of two, 2), and the forced re-chunk of the tests,
  `force_size`, `force_at`, `force_toggle`, `force_alt`. `shrink_bench` exposes them as options:
  `--rechunk_dry_run`, `--rechunk_max_coarsen`, `--rechunk_min_gain_mb`, `--rechunk_grid_bits`,
  `--rechunk_coarsen_min_meta_mb`, `--rechunk_coarsen_min_meta_rate`, `--rechunk_backoff_max`,
  `--force_rechunk`, `--force_rechunk_at`, `--force_rechunk_toggle`, `--force_rechunk_alt`, plus
  `--advisor_period` and `--advise_five` for its own schedule. `bench_and_test.py`'s re-chunk
  phase and `diag_zero_rechunk_slowdown.py` pass them.
- The library's reports are log levels, no longer environment variables: the buffer arena's
  summary (with its `partial` and `freepath` lines and the free-path counters) is
  `CHUNK_LOG=arena=info`, the block registry's report at exit `ledger=info`, the OMPT
  statistics at exit `ompt=info`, and `shrink_bench`'s report of zero_any's second-value
  registry at each advisor pass `compress=info` (its heading, `cst2-gc @ iter <k>`, is a log line
  now). A report is switched on by its channel's level and then written in full; a bare
  `CHUNK_LOG=info` switches them all on. `diag_zero_rechunk_slowdown.py` asks for `arena=info`,
  and reads the `freepath` lines as the log writes them: its `[FREEPATH]` pattern matched
  nothing since the log.
- The decider's two per-array traces are log levels, no longer environment variables: the churn
  line (materialisations per pass, the transient they imply) is `CHUNK_LOG=rechunk=debug`
  (it was `rechunk=info`, switched on by `CHUNK_RECHUNK_CHURN`), and the advisor's full table at
  each pass is `CHUNK_LOG=advisor=debug`, written to the log (`CHUNK_LOG_FILE`) instead of
  stdout. The former `=<name>` form is the new `RechunkPolicy::trace_array` (empty: every array;
  `shrink_bench --rechunk_trace_array <name>`). `ChunkSizeAdvisor<T>::log()` writes a report to
  the log as `print()` does to stdout. The decider reads no environment variable any more.
- `shrink_bench`: `tab_conn` holds about `sz / 32` ints by default, `--conn_scale <x>` sets it
  (`x × sz`, whole rows). At `sz` its dense gather and renumbering took ~87% of a run (zero_any,
  8 threads, `--divisor 2`: 64.8 s with it, 8.4 s without with GCC 16, 140 s against 19 s with
  icpx) and hid every other phase -- what looked like a regression of the bench.
- The re-chunks are on screen again by default: the `rechunk` log channel
  starts at `info` (`CHUNK_LOG` unset means `warn,rechunk=info`), so each
  re-chunk the decider applies writes its ``decide <array>: re-chunk S=..
  -> S'=.. [APPLIED]`` line, as `[RECHUNK-DECIDE]` did before the log. Since
  the log, the channel started at `warn` with the others and nothing was
  shown. `CHUNK_LOG=warn` (a bare level sets every channel) hides it.
- Fast allocator: the value slabs are no longer floored at sixteen pages. The floor capped the
  last page an emptied slab used to keep; a sweep now gives the slab back whole, and the floor
  only made the slabs larger on 64 KiB-page nodes -- 1 MiB at 64 threads instead of 256 KiB --
  hence a larger partly-used slab per (rank, element type, array, size), and more memory kept
  resident by a slab's last few live buffers. Each element type's slab is floored at one page
  and at its largest buffer instead (`value_slab.h`). Unchanged on 4 KiB pages.
- The three array classes live in a namespace each, named after what they compress:
  `zero::Array` (was `ArrayChunk`), `zero_one_value::Array` (was `ArrayChunk2`),
  `zero_any::Array` (was `ArrayChunk2A`). The chunk and proxy classes, the CMake targets and
  the package keep their names for now.
- The headers follow: `chunk/zero_array.h`, `chunk/zero_one_value_array.h` and
  `chunk/zero_any_array.h` (were `chunk/array_chunk.h`, `array_chunk2.h`, `array_chunk2A.h`),
  and so do the files of `chunk/internal/` and `chunk/private/` that belong to one of the three
  classes (`zero_any_array_inline.h`, `zero_one_value_array_parallel.h`, `zero_array.cpp`, ...).
  The files shared by several of them (`array_chunk_base.h`, `array_chunk_core.h`, ...) keep
  their names.
- The chunk and proxy classes of each variant move into its namespace: `zero::Chunk`,
  `zero_one_value::Chunk`, `zero_any::Chunk` (were `Chunk`, `Chunk2`, `Chunk2A`), `zero::Proxy`,
  `zero_one_value::Proxy`, `zero_any::Proxy` (were `Proxy`, `Proxy2`, `Proxy2A`), and so do their
  definitions, specs, policies, pointer maps and array mixins (`zero_any::ChunkThreadPolicy` was
  `Chunk2AOmpPolicy`, `zero_one_value::RechunkHooks` was `RechunkHooks2`, ...). The bases shared by
  `zero_one_value` and `zero_any` become `SecondValueChunkBase`, `SecondValueProxyBase` and
  `SecondValueArrayBase` (were `Chunk2Base`, `Proxy2Base`, `ArrayChunk2Base`).
- The thread-model policies are named after `build_config::threaded`, which selects their
  specialisation under OpenMP and under the std backend alike: `ChunkThreadPolicy` in each
  variant's namespace (was `ChunkOmpPolicy`), `CompressionManagerThreadPolicy` (was
  `CompressionManagerOmpPolicy`).
- Their files follow: `chunk/zero_chunk.h`, `zero_one_value_chunk.h`, `zero_any_chunk.h` (were
  `chunk_std.h`, `chunk2_std.h`, `chunk2A_std.h`), `zero_chunk_def.h`, ... (were `chunk_def.h`, ...),
  `chunk/internal/zero_any_proxy.h`, ... (were proxy.h, proxy2.h, proxy2A.h), and
  `second_value_chunk_base.h`, `second_value_proxy_base.h`, `second_value_array_base.h` (were
  chunk2_base.h, proxy2_base.h, array_chunk2_base.h).
- The build names the variants after their namespace too: `CHUNK_VARIANTS` takes `zero`,
  `zero_one_value`, `zero_any` (was `chunk`, `chunk2`, `chunk2a`); the libraries are
  `chunk_zero_<size>`, `chunk_zero_one_value_<size>`, `chunk_zero_any_<size>`, `..._dyn` and the bare
  `chunk::chunk_zero_any`, ... (were `chunk_v1_*`, `chunk_v2_*`, `chunk_v2a_*`); the macros
  `CHUNK_VARIANT_ZERO`, `CHUNK_VARIANT_ZERO_ONE_VALUE`, `CHUNK_VARIANT_ZERO_ANY` (were
  `CHUNK_VARIANT_V1`, `_V2`, `_V2A`); `chunk_config::use_zero`, `use_zero_one_value`, `use_zero_any`,
  `use_second_value` (were `use_chunk`, `use_chunk2`, `use_chunk2a`, `use_chunk2or2a`); and the
  executables and tests, `shrink_bench_zero_any_256`, `validation_zero_dyn`, ... (were
  `shrink_bench_chunk2a_256`, ...).
- The python scripts use the new labels too: `bench_and_test.py` (`bench_runner`),
  `multi_launch_stress.py`, `omp_speedup.py`, `access_cost.py`, `perf_db.py` and `shrink_gui` take
  and show `zero`, `zero_one_value`, `zero_any`, `tbb_zero`, ... (`--variants`, `--exec`,
  `--stats-mark-ratio-min zero_any=0.4`, ...), and so does the CHUNK_STATS tag the library prints.
  The old labels (chunk, chunk2a@256, tbb_chunk2, ...) are still read as their new names until
  1.0: on the command line, with a note on stderr, and in the perf databases, bench_summary.csv
  files, logs, CHUNK_STATS tags and executable names (shrink_bench_chunk2a_dyn) written before.
  `--mem-zero-any-vs-zero-one-value-max` was --mem-chunk2a-vs-chunk2-max (still accepted, hidden),
  and `diag_zero_rechunk_slowdown.py` was diag_v1_rechunk_slowdown.py.
- The documentation and the README name the variants by their namespace, `zero`,
  `zero_one_value`, `zero_any` (were Chunk or v1, Chunk2, Chunk2A); `docs/algo_zero.md`,
  `docs/algo_zero_one_value.md`, `docs/algo_zero_any.md` and their French versions (`_fr.md`)
  were algo_chunk.md, algo_chunk2.md, algo_chunk2a.md.
- Types are PascalCase, with no `xxx_type` exception: the member aliases of the arrays
  (`chunk_type`, `manager_type`, `proxy_type`, `ind_type`, `chunk_list_type`) and the
  internal ones (`index_type`, `cst_val1_type`) become `ChunkType`, `ManagerType`, `ProxyType`,
  `IndType`, `ChunkListType`, `IndexType`, `CstVal1Type`; `chunk_template_of` and
  `for_each_type_impl` become `ChunkTemplateOf` and `ForEachTypeImpl`. The template parameters
  that carried those names are now `ChunkT`, `ProxyT`, `IntType`, `IndexT`. The former spellings of
  the five public aliases stay on `ArrayChunk`, `ArrayChunk2` and `ArrayChunk2A` as
  `[[deprecated]]` aliases until 1.0. Only the standard library's own names (`value_type`,
  `size_type`, ...) keep their shape; `scripts/check_naming.py` and
  `scripts/clang_tidy_naming.yaml` now say so.
- The library's `omp critical` sections are `std::mutex`es and its
  `omp atomic` updates `std::atomic_ref`s; every OpenMP construct sits in one
  file of the backend (`fast_allocators/internal/pmp_loops.h`). The OpenMP
  build runs the same regions, with the same clauses.
- Spack: the variant `openmp` is replaced by `threads` (`+openmp` is now
  `threads=openmp`, `~openmp` `threads=serial`).
- Vendored ankerl::unordered_dense updated from 5.3.1 to 5.3.3, its two local patches (nvc++)
  carried over: `__has_cpp_attribute` is probed with `defined()` first (MSVC 19.16); no other
  change.

### Deprecated

- `CHUNK_ARENA_REPORT`, `CHUNK_LEDGER_REPORT`, `CHUNK_OMPT_STATS` and `CHUNK_CST2_PROBE`: they set
  `CHUNK_LOG=arena=info`, `ledger=info`, `ompt=info` and `compress=info`, with a warning (see
  Changed); set means non-empty and not starting with `0`, for the two that any value used to
  switch on. Removal planned for 2026-12-31.
- `FAST_ALLOC_TRACE` (it sets `CHUNK_LOG=alloc=trace`, with a warning): removal planned for
  2026-12-31.
- `python/omp_speedup.py`: runs `thread_speedup.py` with the same arguments (one line on stderr
  says so), for the commands and job scripts that call it; to be removed in a later release.
- `ArrayChunk`, `ArrayChunk2` and `ArrayChunk2A`: alias templates of the classes above, with a
  `[[deprecated]]` attribute, until 1.0.
- `chunk/array_chunk.h`, `chunk/array_chunk2.h` and `chunk/array_chunk2A.h`: they include the
  new header and emit a `#warning`, until 1.0.
- `Chunk`, `Chunk2`, `Chunk2A`, `Proxy`, `Proxy2` and `Proxy2A`: `[[deprecated]]` alias templates of
  the classes above, until 1.0.
- `chunk/chunk_std.h`, `chunk2_std.h`, `chunk2A_std.h`, `chunk_def.h`, `chunk2_def.h` and
  `chunk2A_def.h`: they include the new header and emit a `#warning`, until 1.0.
- The former `CHUNK_VARIANTS` values (a CMake deprecation message), library names (INTERFACE
  targets with a `DEPRECATION` property, installed), `CHUNK_VARIANT_V*` macros (defined from the new
  ones, and the other way round) and `chunk_config::use_chunk*` constants (`[[deprecated]]`), until 1.0.
- `zero_any::ChunkDef::factor_div_somme_chunk_buffer_resuse`, a misspelling: now
  `factor_div_somme_chunk_buffer_reuse`, the old name a `[[deprecated]]` constant until 1.0.

### Removed

- The levels 1 to 3 of `CHUNK2A_COMPRESS_LEVEL` (zero_any's compression thresholds), never built
  or tested, and the macro with them: the thresholds are constants of `zero_any_chunk_def.h`
  (500 and 50, the former level 0). Their removal was planned for 2026-12-31.
- The CMake option `CHUNK_RECLAIM_EMPTY_SLABS` (and `FAST_ALLOC_OPT_RECLAIM_EMPTY_SLABS`,
  `CHUNK_OPT_RECLAIM_EMPTY_SLABS`, `alloc_config::reclaim_empty_slabs`,
  `build_config::reclaim_empty_slabs`): the sweeps always give emptied slabs back, as the default
  did; `OFF` was measured costlier in memory for no time saved, and its removal was planned for
  2026-12-31. A cache that still sets it `OFF` gets a configure warning. The fast allocator's
  start-up line no longer prints it.
- `scripts/diag_zero_rechunk_slowdown.py`, a copy of `python/diag_zero_rechunk_slowdown.py`
  that had drifted from it; the one kept gains the copy's check of the missing executables.
- The developer settings fixed by the first `ab_settings_campaign.py` campaign (Milan,
  `shrink_bench` int64 dyn, `zero`, `zero_any` and `zero_one_value`, 64 threads, three
  interleaved runs per point; noise 0.3%, 1.1% and 2.5% of the time):
  - `CHUNK_ALLOC_AFFINITY_EAGER`: the mode is fixed per variant, as its default was -- eager for
    `zero` (lazy: +9.8% of time, +14.6% at the peak, +11.8% at the end), lazy for `zero_any`
    (eager: +7.4% at the peak) and `zero_one_value` (the same either way).
  - `CHUNK_RECLAIM_PARTIAL`: the free pages of the slabs in use are always given back (without
    it, +7.5% of end memory for `zero_any`, +1% for `zero_one_value`, no time saved).
  - `CHUNK_SLAB_FULLEST_FIRST`, with the re-chaining of the slabs by occupancy
    (`TCTypeCAllocator::order_fullest_first`): no effect on time or memory, on any variant.
  - `CHUNK_RECLAIM_PHASE_KEEP`, with the sweep after every compression pass it switched on
    (`ManagerChunk::sweep_empty_slabs_int`): +74% to +251% of time for no memory saved, whatever
    the number of slabs kept.
  - `CHUNK_REUSE_SHARE_DIV` (50), `CHUNK_REUSE_FLOOR_KB` (500 buffers, 100 for an array with
    its own allocators) and `CHUNK_RECLAIM_MIN_SLABS` (0): within the noise at 25 and 100, at no
    floor and 4 MiB per rank, at 16 slabs.

  `ManagerChunk::reuse_floor_buffers()` goes with `CHUNK_REUSE_FLOOR_KB`, and the `keep_empty`
  argument of `ManagerChunk::sweep_empty_slabs()` with the per-pass sweep.
  `diag_zero_rechunk_slowdown.py` loses its `dyn_lazy` run. `CHUNK_RECLAIM_THREADS` stays
  (1) until it is measured on Grace: `0` was 3.5% faster for `zero_one_value` on Milan.
- Developer settings measured on Milan (64 threads, `zero`, `zero_any` and `zero_one_value`
  dyn, three runs each) and fixed at the value that won or that changed nothing:
  `CHUNK_RECHUNK_RECYCLE` (recycling is always on: 29% slower without),
  `CHUNK_RECHUNK_KEEP_RECYCLED` and `CHUNK_RECHUNK_KEEP_RECYCLED_MB` (see Changed),
  `CHUNK_RECHUNK_SCHEDULE` (`guided,16`: `static` and `dynamic,64` within 1% except
  `zero_one_value`, where `static` is 7.6% slower), `CHUNK_THP` (the kernel's THP mode decides:
  `never` and `1` changed neither time nor memory), `CHUNK_SLAB_MIN_PAGES` (no measurable
  effect; its removal was planned) and `CHUNK_RECHUNK_SLICES` (16: with the recycled slabs kept,
  32, 64 and 128 slices are 1.3%, 5.2% and 6.5% slower at 64 threads, five runs each; 8 is 1.8%
  faster for +0.5% of memory). `alloc_config::thp_advise()`, `thp_never()` and
  `advise_thp_region()` go with `CHUNK_THP`. `scripts/bench_recycle.sh` loses its recycling
  dimension, and `diag_zero_rechunk_slowdown.py` its `dyn_static1` and `dyn_norecycle` runs.
- `CHUNK_RECHUNK_DRY_RUN` (a user setting of `docs/environment_variables.md`): set
  `RechunkPolicy::dry_run` (`shrink_bench --rechunk_dry_run`).
- The decider's developer variables `CHUNK_RECHUNK_MAX_COARSEN`, `CHUNK_ADVISOR_MIN_GAIN_MB`,
  `CHUNK_ADVISOR_GRID_BITS`, `CHUNK_ADVISOR_BACKOFF_MAX`, `CHUNK_COARSEN_MIN_META_MB`,
  `CHUNK_COARSEN_MIN_META_RATE`, `CHUNK_FORCE_RECHUNK`, `CHUNK_FORCE_RECHUNK_AT`,
  `CHUNK_FORCE_RECHUNK_TOGGLE` and `CHUNK_FORCE_RECHUNK_ALT`, and the benchmark's
  `CHUNK_ADVISOR_PERIOD` and `SHRINK_BENCH_ADVISE_ALL`: the `RechunkPolicy` fields and
  `shrink_bench` options of Changed. Set, they are ignored without a message.
- `ChunkSizeAdvisor<T>::default_grid_bits()` (it read `CHUNK_ADVISOR_GRID_BITS`):
  `Options::grid_bits` defaults to 2.
- The deprecated trace switches `CHUNK_RECHUNK_TRACE`, `CHUNK_RECHUNK_TRACE_SLICES`,
  `CHUNK_RECHUNK_MEMTRACE`, `CHUNK_RECHUNK_TRACE_MEM`, `CHUNK_AFFINITY_TRACE` and
  `CHUNK_RECYCLE_TRACE`: `CHUNK_LOG` (`rechunk=debug`, `rechunk=trace`, `memory=info`,
  `memory=debug`, `affinity=debug`, `recycle=trace`). `FAST_ALLOC_TRACE` stays, deprecated.
- `CHUNK_RECHUNK_CHURN` and `CHUNK_ADVISOR_TABLE`: `CHUNK_LOG=rechunk=debug` and
  `CHUNK_LOG=advisor=debug`, with `RechunkPolicy::trace_array` for a single array (see Changed).
  Set, they are ignored without a message.
- `CHUNK_RECHUNK_REUSE_LIST` and the chunk list buffer a re-chunk kept for the next one
  (`ArrayChunkCore::spare_chunk_list_`): see Fixed.
- Two internal headers nothing included: `chunk/internal/diag_mutex.h` (the lock of the
  diagnostic reports, which are `CHUNK_LOG` lines now) and `chunk/internal/nano_sleeper.h`.

### Fixed

- `CHUNK_DO_CHECK=ON` builds compile again. The proxies' write paths called
  `global_check_write_nan()`, defined nowhere; they now go through
  `chunk_config::CheckNaN<do_check>::maybe()`, as the reads do, so `CheckNaN::update(false)`
  turns the write check off too. And `CheckNaN<true>::maybe()` (config.h) called
  `global_check_nan()` before any declaration of it: for a `double` argument-dependent lookup
  finds nothing, so every TU that saw config.h before global_check.h failed (GCC 13). The
  overloads are now declared in config.h, under `CHUNK_OPT_DO_CHECK`.
- The compression passes of `zero`, `zero_one_value` and `zero_any` (threaded builds): a write that
  made a chunk constant while its pass handed it back -- between the pass's `steal_unmark()` and
  the reset of its "queued" flag -- met the flag still up and did not queue it, and the reset
  then dropped the chunk from every list: it stayed materialised until written again (memory,
  no wrong value). The flag now has a "handed back" state the pass sets before it lifts its
  marks; a writer that finds it notes itself ("written meanwhile") and the pass queues the
  chunk again (`chunk/internal/queue_flag.h`; F6, `docs/formal_models.md` section 8.8, model
  `formal/queue_handback.c`). Cost: one compare more on a refused marking, an exchange and a
  CAS per candidate in a pass.
- `fill_range_mt()`, `copy_range_mt()`, `transform_range_mt()` on `zero_any` elements with relaxed
  access (`double`, `int`, ...): two threads filling the two ends of one chunk could both read the
  other's edge before its store (store buffering, allowed on x86 and AArch64) and neither queue
  the chunk they left uniform. The filter now looks again after a seq_cst fence when its first
  look says "not uniform". The check builds' audit counted 6 to 52 such chunks per
  `shrink_bench` run at 16 to 128 threads on Milan. Model: `formal/range_edges.c`
  (`docs/formal_models.md`, section 8.9, F7).
- `fill_range_mt()`, `copy_range_mt()`, `transform_range_mt()` on a `zero_any` array of an element
  type without relaxed element access (`std::complex<double>`, the aggregates): a span that did not
  cover its whole chunk -- the ends of a range off a chunk boundary -- never queued the chunk, even
  when it completed it uniform, and the chunk stayed materialised for the run. Any uniform span
  queues it now; the pass drops it if the rest of the chunk differs. The check builds' audit
  ("never queued for compression") counted 34 to 140 such chunks of `tab_complex` per run of
  `shrink_bench` at `--divisor 10 --iter_max 1000` on Milan, at every thread count, 1 included;
  none now. `fill_range_mt_validation` checks the case.
- `compile_all.sh` no longer loads a hard-coded `gcc/15.2.0`: it loads `gcc/15.2.0` where it exists,
  else the newest `gcc/<ver>` module, for llvm, oneapi, nvhpc and aocc. On a machine without
  15.2.0 the gcc load failed, then the llvm module, which needs a gcc, failed too, and the llvm
  builds silently used the system `clang++`. The gcc is now loaded before llvm, and a failed
  `module load llvm/<ver>` stops the script with what the module requires (`module show`).
- `compile_all_gnu.sh` on aarch64 loads `binutils/2.43` where it exists, else the newest binutils
  module, and stops if the assembler is still the system one (RHEL 9: 2.35, too old for gcc 16
  `-mcpu=native`): a missing or conflicting binutils module used to go unnoticed until the build
  died on unknown instructions. The llvm and nvhpc sections of `compile_all.sh` print the `as` /
  `ld` they use and warn on aarch64 when it is the system one.
- Serial builds (no OpenMP, no std backend): `finalize_parallel_phase()` did not drain the
  `zero_any` family's compression candidates. The in-loop pass only runs past half the list
  maximum, so a backlog below it waited to the end of the run, and chunks written for the last
  time stayed materialised. The end-of-run census hid it for the constants, which it classifies
  on content, but not for `pattern`'s motifs: `shrink_bench_zero_any_256` (`--divisor 100
  --iter_max 2000`, seed 42) left 206 of `tab_conn`'s 1217 chunks materialised in serial, and
  0 under OpenMP, and the seq/omp compare of bench_and_test.py reported it as
  `SEQ_OMP_COMPARE_MISMATCH` (539706 against 539912). The phase end now runs the same pass as
  the in-loop trigger.
- bench_and_test.py's scalability guard: an OOM kill ends its curve, since the next thread
  counts have arrays of the same size and would be killed the same way. For the `vector`
  reference it is a note rather than an error: it holds the uncompressed arrays, so it is the
  first not to fit a job sized for the compressed ones. Before, `vector` at `--multiply 2`
  logged four `SCALING_RUN_FAIL` per config on a node without the memory for it.
- `thread_speedup.py` (formerly `omp_speedup.py`) takes `CHUNK_NUM_THREADS` out of the runs'
  environment: the std backend and the TBB bench read it before `OMP_NUM_THREADS`, so a value
  left in the shell pinned every point of a `std-*` run to one thread count.
- shrink_gui: the legends of the Graphique tab are readable again. Set in one row at the top,
  they cut every label short -- with `thread_speedup.py --all-exec` (a dozen executables, twice
  over in a comparison) no label was left. They sit on the right now, one entry a line, with the
  full label on hover; the speedup chart has one entry per executable, its marker (colour: the
  variant, shape: the size), B's dashed twin of an A curve adding none.
- The teardown verbs (`release_chunk_pools()`, `destroy_block_chunk()`, `delete_chunk_allocator()`,
  `clear_all_list_block_free_thr()`, `delete_list_block_chunk_free_itself()`) and
  `shrink_to_fit()`'s pool trim refused a call from inside a parallel region in check builds
  only, and four of them not even there. In a release build every thread of the team emptied the
  same pools at once (double frees, broken lists). All six refuse it in every build now: a check
  build aborts, the others print one line on stderr and skip the call.
- The per-chunk vectors (the chunk list, `zero_any`'s second-value and constant-type rows, the
  candidate flags, the dirty bitset) are mapped by their allocator (`NoInitAllocator`) from
  1 MiB on, and unmapped when freed. They came from glibc's heap below its mmap threshold, which
  grows up to 32 MiB each time a mapped block is freed: after a re-chunk freed one list, the
  next ones went to the heap, which kept them resident once freed. `zero_any_dyn`
  (`--divisor 10`, default glibc settings): 1387 MB at iteration 1990, 815 MB now; peak 1389 MB,
  1076 MB now. At 10^9 elements the lists exceed 32 MiB and were already mapped.
- Re-chunk: the old geometry's chunk list came back to the system. `CHUNK_RECHUNK_REUSE_LIST`
  (on by default) kept a dying storage's chunk list buffer for the next re-chunk of the same
  chunk type, and took it whenever it was at most 4x too large: a coarsening 64 -> 256 kept the
  64 list resident whole for a list a quarter of it, and the last spare of each type was never
  freed (16 bytes per chunk: 250 MB per 10^9-element array at S=64). `zero_any_dyn`
  (`--mode perf --divisor 10`) ended at 1165 MB against 761 MB for static 256; 807 MB now
  (64 threads, full size: 7895 MB -> 5138 MB, static 256: 5088 MB). The reuse is gone, setting
  and code (see Removed).
- `shrink_to_fit()` and `ParamChunk::force_compression()` left uniform `zero_any` chunks
  materialised in a serial build (OpenMP off): a serial `zero_any` queues its compression
  candidates and folds them in batches of half the list from the write path, and both swept the
  last, partial batch only in the OpenMP builds. Up to that many chunks kept a buffer -- right
  values, memory not given back (the cpp26 branch's second-value registry test saw 1750 second
  values for 2000). They sweep in every build now; `shrink_to_fit_validation` checks it In the threaded builds,
  `shrink_to_fit()` also folds every rank's candidates now, not the calling thread's alone: with the
  end-of-phase pass off (`CHUNK_AGGRESSIVE_COMPRESSION=1`), 44 of 100 uniform chunks kept their
  buffer at 8 threads (`shrink_to_fit_validation_*_aggressive_8thr`).
- `compile_all.sh`: the slot pool (`.build-slots.<host>.<pid>`) and the log directory
  (`logs/<date>-<host>-<pid>`) of a run carry the host name. In a `$CHUNK_BUILD_REP` shared by
  several nodes, two runs with the same pid used the same slot pool.
- Chunk2A: the deferred replay of `=` stored the element with a plain store
  while another writer's uniform-chunk scan read it as a relaxed atomic -- a
  data race (found by TSan on the std backend, present under OpenMP too).
  Same machine code on x86 and AArch64, so no value was wrong.
- Build: the flags of one C++ compiler (icpx's `-fp-model=precise` and
  `-fimf-use-svml=true`, nvc++'s `-tp=host`, `-xHost`, `-no-intel-lib`, the
  mold link options) apply to C++ only. With `CXX=icpx` alone the C targets of
  `formal/` build with the system cc, and GCC stopped on them.
- Std backend: `finalize_parallel_phase()` closes the application phase
  again. Its early return for a nested region tested `in_parallel()`, which
  the application phase makes true: the phase never ended, and the check
  builds stopped at `release_chunk_pools()`. It now tests an active team.
  A `parallel_phase()` opened during an application phase is nested in it:
  it no longer ends that phase, and its begin is not a second begin.

## 0.1.3 -- 2026-10-22

### Added

- `examples/getting_started/getting_started_ompt.cpp`: the first program with
  plain `#pragma omp parallel for` loops and no `parallel_phase()`, for an OMPT
  build (Clang, oneAPI), checked element by element against the same function
  run on a `std::vector` (ctest `getting_started_ompt`, OMPT builds only).
- Reduced models of the three lock-free handshakes -- writer / compressor,
  compressor / flush, readers / buffer reclamation -- checked by GenMC
  (`formal/`, `docs/formal_models.md`): the protocols of the source verified
  under RC11 and IMM, twelve mutations refused (six of them the library
  as it once was, the lost replay without the compressor's second look
  among them) and four findings: three fixed (see Fixed: F2 to F4), one
  closed by removing the build it was in (F1, see Removed). New ctest label
  `formal`.
- One log for both libraries (`chunk/log.h`, built on
  `fast_allocators/internal/log_core.h`): every message has a level and a
  channel, `CHUNK_LOG` (default `warn`) says which are written, e.g.
  `CHUNK_LOG=warn,rechunk=debug`, and `CHUNK_LOG_FILE` where (stderr by
  default). A program can set the levels with `shrink_array::logging::configure()`
  and receive every message in its own logger with
  `shrink_array::logging::set_sink()`. A message is formatted on the stack (no
  allocation) and written as one line, `[shrink_array][channel][level][tid] text`.
  New harness `log_validation`. The library's other stderr messages move to it
  in the next steps.

### Changed

- Aggressive compression is a run-time switch, `CHUNK_AGGRESSIVE_COMPRESSION=1`
  (the CMake option `HAVE_AGGRESSIVE_COMPRESSION` only sets its default): every
  site that reads it is a phase boundary, a teardown or a check-build audit.
  `compile_all*.sh` no longer builds `omp-gnu-have-aggressive-compression`,
  `omp-llvm-have-aggressive-compression` and `omp-gnu-debug-have-aggressive-compression`;
  `bench_and_test.py` and `multi_launch_stress.py` run those names as the base
  build with the variable set. The two optimised TSan builds lose the switch from
  their name: `omp-gnu-relwithdebinfo-check-thread-sanitizer`,
  `omp-llvm-relwithdebinfo-check-thread-sanitizer` (run them with the variable for
  the same coverage).
- The thread capacity (the ranks the per-thread structures are sized for) is at
  least 256, besides `omp_get_max_threads()` and the cores: a team larger than
  the node's cores -- a `num_threads()` clause, a component that picks its own
  team, a job granted two cores -- no longer stops the program. About 1 KiB per
  element type and per rank, built once; `CHUNK_THREAD_CAPACITY` still raises it.
- Less compilation, the same measured binaries. The optimised builds carry
  function names and line tables only (`-g1`), full `-g` where a debugger reads
  locals (`CHECK_OPTIM`, sanitizers): `CHUNK_DEBUG_INFO` (auto, full, lines,
  none). Same machine code; 20 to 27 % less compilation and 40 % less peak
  memory on the bench and validation units. And `CHUNK_STATIC_SIZES` defaults
  to `auto`: `256;1024`, but 256 alone (with dyn) in a Debug or sanitizer build
  of this project, a third fewer libraries and executables there.
  `bench_and_test.py` skips the sizes a configuration does not build. Existing
  build directories keep their cached list: `compile_all.sh --fresh-cache`.
- Constants are snake_case everywhere, as the standard library's; UPPER_CASE is left to
  macros. 84 names change, all internal but one: `fast_alloc::SIZE_BLOCK` is now
  `fast_alloc::size_block`, the former name kept as a deprecated alias. `check_naming` (a
  source check, in ctest and the CI) keeps it so.
- One naming convention over the whole tree ([docs/contributing.md](docs/contributing.md)
  section 3): snake_case functions, variables, constants and enumerators, a `_` at the end of
  private and protected data members, PascalCase types, aliases, concepts and template
  parameters, `chunk2a` inside a snake_case name, no reserved `__` name. About 400 names
  change, all internal but those listed under Deprecated. `check_naming` now checks the
  enumerators, the camelCase names, the types and template parameters and the reserved names;
  `scripts/clang_tidy_naming.yaml` gives clang-tidy the rest.
- An invariant failure is reported by one thread only (when a race is seen by
  several threads at once, their reports no longer interleave), as one block
  on stderr followed by the backtrace, and its message is formatted without
  allocating; it then reaches the program's sink (`set_sink()`) or
  `CHUNK_LOG_FILE`, as a record of level fatal.
- A double free caught at teardown, a drain that does not settle and deferred
  writes about to be lost stop a check build (as before) and, in the other
  builds, are errors of the log (channels `pool`, `drain`) instead of raw
  stderr lines.
- The re-chunk traces go through the log, with a new channel `memory` for the
  probes that cost: `CHUNK_LOG=rechunk=debug` (phase times, rebuild schedule,
  slice split; formerly `CHUNK_RECHUNK_TRACE`), `rechunk=trace` (memory per
  slice; `CHUNK_RECHUNK_TRACE_SLICES`), `memory=info` (memory split at each
  advisor tick; `CHUNK_RECHUNK_MEMTRACE`), `memory=debug` (exact memory at
  each re-chunk step, census; `CHUNK_RECHUNK_TRACE_MEM`). The former variables
  still work, with a deprecation warning; `0` now means unset for them too.
- The re-chunk decider's lines are no longer written by default:
  `rechunk=info` writes its moves (``decide <array>: re-chunk S=.. -> S'=..
  ... [APPLIED]``), `rechunk=debug` its waits too. `CHUNK_RECHUNK_DRY_RUN`,
  `CHUNK_RECHUNK_CHURN` and `CHUNK_FORCE_RECHUNK` switch `rechunk=info` on.
  python/bench_runner and multi_launch_stress.py set
  `CHUNK_LOG=warn,rechunk=info` unless it is set; their parser,
  rechunk_trace_summary.py and diag_v1_rechunk_slowdown.py read the new and
  the former format.
- A refused `rechunk_to()`, `set_chunk_size()` or `set_active_chunk_size()` is
  a warning of channel `rechunk` (written by default, as before); the notices
  of `CHUNK_RECHUNK_KEEP_RECYCLED(_MB)` are of level info (no longer written
  by default); check builds' chunk-size prediction is `advisor=info`, and its
  scan is skipped when not written.
- Manager, compression audits, pool and ledger reports, OMPT go through the
  log: a drain left with deferred operations at teardown is an error of
  channel `drain` (the first 16); the check builds' compression audits and the
  compression residual of `compress()` are `compress=info` (no longer written
  by default; ArrayChunk2's per-chunk detail at `compress=debug`); the
  per-array allocator sets are `affinity=debug` (formerly
  `CHUNK_AFFINITY_TRACE`, deprecated); the OMPT tool's warnings are of channel
  `ompt`, and with `CHUNK_REQUIRE_OMPT=1` an invariant failure. Reports asked
  for explicitly -- `print_all_reuse_pool_breakdown()`, `CHUNK_LEDGER_REPORT`,
  `CHUNK_OMPT_STATS`, `compress_int(true)`, the `CHUNK_RECHUNK_LEDGERTRACE`
  build -- are written whatever `CHUNK_LOG` says (`CHUNK_LOG_ALWAYS`), through
  the sink.
- fast_allocators goes through the log too, and its fatal failures (an
  allocation or a `mmap` that fails, a slab handed out twice, an element that
  does not fit a slab) through the same fatal report as an invariant failure
  (`FAST_ALLOC_FATAL`, `fast_allocators/internal/fatal.h`, which also holds
  `print_backtrace()` now -- `chunk/debug.h` forwards to it): one block on
  stderr headed `[CRITICAL_FAILURE]`, the backtrace, the program's sink, one
  reporting thread. A slab freed or recycled twice is a warning of channel
  `arena`, an invalid `CHUNK_RECLAIM_MIN_SLABS` / `CHUNK_THP` a warning of
  `env` (the `CHUNK_THP` notices are of level info), a failed trim or
  `posix_memalign` an `alloc` message; `CHUNK_ARENA_REPORT` is a report
  (written whatever `CHUNK_LOG` says); the slab evictions are `alloc=trace`
  (formerly `FAST_ALLOC_TRACE`, deprecated).
- The library's remaining messages go through the log: the warnings of
  `set_second_value()` and of a static build's ignored setters (`ParamChunk`)
  and the one-page caution (`warn`, channel `general`); a compound operator a
  bool element has not (`error`); the second-value registry's warnings
  (identifiers past half used, degraded index, refused reclaim: `warn`,
  channel `compress`); a deferred replay retried past its cap outside check
  builds (`drain`); `gc_mark_report()` and the `verbose` build's candidate-list
  lines (reports, written whatever `CHUNK_LOG` says); the foreign blocks the
  deferred-write pool drops (`recycle=debug`, every one at `recycle=trace`,
  formerly `CHUNK_RECYCLE_TRACE`, deprecated; no longer written by default in
  check builds); a lookup of an unknown array key, before its invariant failure
  (`error`).
- The warnings on an environment variable or an argument that is not a valid
  number go through the log: `[shrink_array][env][W][tid] NAME="value" is not a
  valid value: ...` instead of `[env] NAME=...`.
- fast_allocators 2.2.0 (the log, the snake_case names with their deprecated aliases);
  `CHUNK_FAST_ALLOCATORS_VERSION` follows.

### Deprecated

- The trace switches of before the log, replaced by `CHUNK_LOG` levels: `CHUNK_RECHUNK_TRACE`
  (`rechunk=debug`), `CHUNK_RECHUNK_TRACE_SLICES` (`rechunk=trace`), `CHUNK_RECHUNK_MEMTRACE`
  (`memory=info`), `CHUNK_RECHUNK_TRACE_MEM` (`memory=debug`), `CHUNK_AFFINITY_TRACE`
  (`affinity=debug`), `CHUNK_RECYCLE_TRACE` (`recycle=trace`), `FAST_ALLOC_TRACE`
  (`alloc=trace`). Still accepted -- each sets its level, with a warning -- and to be removed in
  0.3.0. They are developer settings ([docs/developer_settings.md](docs/developer_settings.md)),
  outside the versioning policy; the deprecation period is a courtesy to the scripts that set them.
- `getChunkList()` (ArrayChunk, ArrayChunk2, ArrayChunk2A) and `ParamChunk::ChunkStats()`,
  renamed `get_chunk_list()` and `ParamChunk::chunk_stats()`: functions and methods are
  snake_case. The former names stay, `[[deprecated]]`, until 1.0.
- fast_allocators (`fast_allocators/tc_allocator.h`): `VTCTypeAllocator::addAllocatedSize()`,
  `TCTypeAllocatorEBC::addAllocatedSize()`, `add_TCTypeAllocator_inst()` and
  `remove_TCTypeAllocator_inst()`, renamed `add_allocated_size()`, `add_tc_type_allocator_inst()`
  and `remove_tc_type_allocator_inst()`. The former names stay, `[[deprecated]]`, until
  fast_allocators 3.0; a class derived from `VTCTypeAllocator` overrides the new name.

### Removed

- The `SHRINK_SMALL_CHUNK` build (a `-D` compiler define, no CMake option): 8-byte chunks with
  no per-chunk pending counter and no steal mark, hence a coarse read gate (any pending op
  anywhere made a dirty-chunk read flush) and a global write gate on `hard_inflight_`. No test
  or CI job built it, and it was a second write protocol to model-check. Its gate also had a
  hole, up to 0.1.2: it read `hard_inflight_` relaxed, so on AArch64 a writer that saw a pass
  over could still load the buffer that pass had just returned and write into it (F1 in
  `docs/formal_models.md`, found by the GenMC model). An OpenMP chunk is 16 bytes; an 8-byte
  chunk with the per-chunk gates is not achievable cleanly (`chunk/config.h`).

### Fixed

- Two top-level OpenMP teams writing arrays at once -- opened by two host
  threads, two components each on its own `std::thread` -- shared ranks 0..n-1
  of the per-thread structures: one team's phase end replayed the other's
  writes, and in a release build the values came out wrong, silently. The
  second team now stops the program with a message, in every build once an
  array exists: under OMPT when it begins (a `parallel_begin` callback counts
  the top-level regions), without OMPT at its `begin_parallel_phase()`. Nested
  regions are not concerned. New harness `concurrent_teams_validation`.
- A parallel region nested in a phase -- a solver or BLAS call opening its own
  region, a component calling `parallel_phase()` inside the program's -- ended
  the phase at its own end: under OMPT the runtime reports every region's end,
  and the library flushed every rank's deferred writes while the enclosing team
  was still writing them (writes lost, crashes); without OMPT the nested
  `parallel_phase()`'s `finalize_parallel_phase()` did the same. Only the
  outermost region's end is a phase boundary now, and `finalize_parallel_phase()`
  does nothing inside a region. `nested_region_validation` has the two cases,
  with writes deferred when the nested regions end.
- `set_chunk_size()` accepted, on an empty array of a `_dyn` build, a size wider than
  `cst_buffer_size` (e.g. twice it): the array's constant chunks then read the shared constant
  buffers out of bounds. It checked only that the size is a power of two or a multiple of 64;
  `rechunk_to()` and `set_active_chunk_size()` also checked the width. The three setters and
  `ParamChunk::check_chunk_size()` now share one rule, `chunk_size_error()`, and refuse such a
  size with a warning, geometry kept. Present in 0.1.2.
- A writer materialising a chunk that a pass marked right after the writer's
  gate check spun on the mark while the pass, parked, waited for that
  writer's flag: both stalled for the whole `CHUNK_COMPRESS_MAX_YIELDS`
  budget, then the pass was abandoned. A fast-path writer no longer waits
  before materialising: the pass holding the mark is parked on its flag and
  cannot steal (`wait_before_materialise()` takes the caller's rank;
  `WriteCompressSyncPolicy::is_thread_writing_on()`). Found by the model.
- The odd entry store of a read section was relaxed. The reclaimer, having
  seen a slot odd, waits for it to move, and may then read the next read's
  entry; by C++20 that relaxed store synchronised with nothing, so only the
  AArch64 and x86 mappings kept the previous read's load before the free of
  the buffer. It is a release store: +1.9 % on access_cost's worst read case
  at 1 thread on Grace, within the spread, nothing measurable at 64. Found
  by the GenMC model (`formal/`).
- A reader past `CHUNK_READ_RECLAIM_SLOTS` announced itself with a seq_cst
  increment alone; by C11 that is not an announce (store buffering against
  the pass's acquire load of the counter), though the AArch64 and x86
  mappings make it one. It is followed by the light fence of the slot path.
  Found by the GenMC model (`formal/`).
- A "wait" line of the re-chunk decider ended with a stray `0` when the array
  had no back-off.
- An invariant failure inside a program's log sink, while the log was reading
  its configuration, blocked instead of stopping the program.
- The Spack recipe pinned, for 0.1.0, 0.1.1 and 0.1.2, commits of before a history rewrite that
  are not on GitHub: an install of one of those versions could not fetch. It now pins what each
  tag points to.
- A cache entry declared by a subdirectory no longer hides the list the top level resolved
  (policy CMP0126 NEW, CMake 3.21+): an older `chunk/CMakeLists.txt`, which declared
  `CHUNK_STATIC_SIZES` in the cache, configured over a cached `auto` saw `auto` and stopped on
  `CHUNK_DEFAULT_SIZE=256 is not in CHUNK_STATIC_SIZES`.

## 0.1.2 -- 2026-10-16

### Added

- `chunk/mdspan.h`: a C++23 `mdspan` view over an array, `f[i, j]`, whose accessor
  goes through the array's `operator[]` (writes through the proxy, compression
  unchanged). `md::view` is row-major; `md::tiled_view<16, 16>` lays the field out
  in tiles of one chunk each, so a uniform 2D block is a uniform chunk: on a
  2048 x 2048 field of materials, 15 % of the chunks stay materialised against
  45 % row-major (`examples/mdspan_tiles`, chunk2A 256). `std::mdspan` when the
  standard library has it, else kokkos/mdspan if on the include path, else
  nothing. Harness `mdspan_validation_<variant>` (skipped without `mdspan`). See
  [docs/mdspan.md](docs/mdspan.md).
- `chunk/mdspan_tile_ops.h`: `md::for_each_tile` and `md::update_tiles` read and
  update a tiled view tile by tile -- a block copy into a static-extent
  `mdspan`, or one call for a uniform tile -- and write a changed tile back in
  one go (one constant when it came out uniform; an unchanged uniform tile is
  not written). Range overloads for `#pragma omp for` over the tiles; in a
  parallel region a tile must be whole chunks (checked). `mdspan_tiles`,
  2048 x 2048, against `f[i, j]`: the sum x 8 to x 15, `f = 0.5 f + 1` x 15 to
  x 20. Harness `mdspan_tile_ops_validation_<variant>`.
- `md::fill_box` (`chunk/mdspan_tile_ops.h`): set a box of a tiled or a
  row-major view to one value, one range op per run of covered tiles (tiled)
  or per contiguous run (row-major), crossed tiles updated as by
  `update_tiles`; a tile-range overload for a parallel phase. `mdspan_tiles`,
  64 rectangles on 2048 x 2048 against `f[i, j]`: x 10 to x 12.
- Fast allocator: `CHUNK_REUSE_SHARE_DIV` and `CHUNK_REUSE_FLOOR_KB` (developer
  settings) set the reuse budget -- how many freed buffers a rank keeps parked
  for its next allocations -- whose default counts every chunk created,
  compressed ones included, so at 99 % compression it parks about twice the
  buffers actually materialised. `CHUNK_RECLAIM_PHASE_KEEP=K` gives the empty
  slabs back to the system after every compression pass, keeping K per
  allocator instance (off by default: it costs page faults).
- `ArrayChunk2A<T>::second_value_stats()`: where the type's second-value
  identifiers stand -- in use against the capacity (running out stops the run),
  free, the reclaims run and what they freed, the reclaims refused by cause
  (parallel region, re-chunk, degraded index), and the degradation. A refused
  `reclaim_second_values()` returns 0 as one with nothing to free does; this
  tells them apart. Relaxed reads, callable at any time. shrink_bench prints it
  per type with `CHUNK_CST2_RECLAIM=N`.
- Validation coverage: the round-trip harnesses run at every static size
  built (`CHUNK_STATIC_SIZES`, so 1024 too), not the default one only;
  `index64_validation_*_dyn_np2` starts at 384, a size that is not a power of
  two (the magic-number index split, a re-chunk 384 -> 768, past 2^31 at 768);
  `cst2_reclaim_validation` runs at dyn as well and reclaims across a re-chunk;
  `compile_all.sh` builds `omp-llvm-check-sv-indexed`
  (`CHUNK_SECOND_VALUE_INDEXED`, CHECK_OPTIM, so with its harnesses; Chunk2A
  only), also in the `multi_launch_stress.py` build lists.
- CMake option `CHUNK_VARIANTS` (default `chunk;chunk2;chunk2a`): the variants
  built. The libraries, examples, benchmarks, harnesses and tests of a variant
  left out are not built nor registered.
- `md::submdspan` (the `submdspan` of the mdspan in use, with
  `md::full_extent`): sub-views of the views of `chunk/mdspan.h`. The data
  handle is now an `md::handle` -- the array and the index of the view's first
  element; `*data_handle()` is still the array -- and the accessor has
  `offset()`. A row-major view takes any slices; a tiled view takes
  rank-preserving slices aligned on its tiles (a range starts on a tile and
  ends on one or at the extent; anything else stops the program) and stays
  tiled, in its parent's tile grid (`layout_tiled::mapping` gains tile strides
  and a `submdspan_mapping`). `for_each_tile`, `update_tiles` and `fill_box`
  work on a tiled sub-view (a tile's position comes from the mapping, not from
  its number); `fill_box` takes any strided view. Checked against
  kokkos/mdspan before and after its canonical slices. Harness
  `mdspan_submdspan_validation`.
- `md::layout_tiled_dyn<Rank>`: `layout_tiled` with the tile given at run time,
  for the `_dyn` libraries (the same storage as `layout_tiled` for the same
  tile; the tile operations, `submdspan` and `fill_box` take it alike).
  `md::tile_shape<Rank>(n)`: a tile of exactly `n` elements, as square as the
  divisors allow (384 -> 16 x 24); `md::tiled_view_dyn(a, ext...)` takes a tile
  of `a.chunk_size()` elements, `md::tiled_size_dyn` sizes the array. The two
  layouts share one implementation (`detail::TiledMappingBase`); the tile
  operations read the tile from the mapping, their buffer inline up to 8 KiB.
  2048 x 2048, 1 thread: `f[i, j]` as fast as with `layout_tiled`,
  `update_tiles` about 10 % slower. Harness `mdspan_tiled_dyn_validation`
  (default size and dyn).

### Changed

- `fill_range` on Chunk2A sets a chunk it covers whole to the value by the
  chunk's state alone, whatever the value (`Chunk2A::fill_constant_st`): a
  materialised chunk releases its buffer through the compression's own clear
  (`clear_int_st`, no scan); a compressed one switches its constant in place
  (second-value slot, type, values pointer) -- no buffer allocated, filled,
  scanned and freed, nothing queued for a pass. 2^25 doubles, 1 thread: over
  chunks compressed at 1.0, filled with 2.0, 38-48 ms before this release,
  15-17 ms when the chunk was first materialised then compressed at once,
  1.5-1.7 ms now; over materialised chunks 34-41 ms -> 2.7 ms. The constant a chunk leaves stays
  registered until `reclaim_second_values()`, as when a write materialises
  it. Serial contract unchanged; `fill_range_mt` unchanged (another thread
  may share the chunk). Harnesses `fill_range_constant_validation`,
  `constant_switch_validation`. md::update_tiles and md::fill_box outside a
  parallel region inherit it (`mdspan_tiles 2048`, 1 thread: `update_tiles`
  by tile 3.1-3.5 -> 1.1-1.4 ms).
- `md::layout_tiled` computes its offsets in unsigned arithmetic (shifts, not
  corrected signed divisions): about a tenth of a tiled `f[i, j]` read.
- `copy_range` on Chunk2A reads the source span of a chunk it covers whole
  first (first, middle and last elements, then all of it): uniform, the chunk
  takes that constant as `fill_range` does, with no buffer; mixed, it is
  copied and no longer queued (the copy is mixed too). 2^25 doubles, 1
  thread, a source uniform per chunk: over compressed chunks 83-87 ms ->
  24-26 ms (48-54 ms when the copied chunk was compressed at once), over
  materialised ones 53-59 ms -> 30-32 ms, and 2545 -> 0 chunks
  left materialised; a mixed source unchanged (73-77 ms), and so is a span
  uniform but for one element (read whole, then copied: 76-83 ms). The
  rechunk rebuilds through `copy_range`, on each thread's own chunks: its
  fold then finds them constant (rebuild 64 -> 256 unchanged). Harness
  `copy_range_constant_validation`.

### Fixed

- Chunk2A: a chunk made uniform at a third value by writes over values that
  are neither constant (a region painted over another one) was never queued
  for compression and stayed materialised for good. The marking predicate's
  clause for a third value required the write to leave the chunk's second
  value; it now takes any change to a value other than the second value, under
  the same `may_become_uniform` filter. `mdspan_tiles`, 64 rectangles painted
  element by element: 6239 chunks left materialised -> 3109, as with
  `md::fill_box`. Cost: the filter's two loads on varied writes into chunks
  that have a second value, which chunks without one already paid (about
  +1 ns per write, +8 %, in a probe at 1 and 2 threads). `mark_path_validation`
  scenario D.
- Fast allocator: `trim_reuse_pools` (hence `shrink_to_fit()` and
  `compress_now_and_trim_pools()`) was a no-op. It now returns the parked
  buffers to their slabs and gives the emptied slabs back to the system, the
  fast allocator's `malloc_trim`: the memory a compression frees no longer stays
  resident until the next re-chunk. shrink_bench calls it at its checkpoints
  ("after init", "end") with `malloc_trim` (`SHRINK_BENCH_KEEP_POOLS=1`: not).
  `shrink_bench_chunk2a_int64_1024 --multiply 1`, 16 threads, current RSS after
  init: 3039 MB -> 567 MB (std allocator: 568 MB).

### Changed

- fast_allocators 2.1.10 (the reuse budget and the real `trim_reuse_pools` above);
  `CHUNK_FAST_ALLOCATORS_VERSION` follows.

## 0.1.1 -- 2026-10-12

### Added

- `ArrayChunk2A<T>::reclaim_second_values()`: at a sequential point, frees the second values
  no chunk of any `ArrayChunk2A<T>` refers to any more; the registry reuses their identifiers
  before carving new ones, so a workload whose constants drift no longer runs out of them
  (`CHUNK_CST_VAL2_MAX_IDS`). The benchmark calls it every N iterations under `CHUNK_CST2_RECLAIM=N`;
  harness `cst2_reclaim_validation`.
- `shrink_array::parallel_phase([n,] body)` (`chunk/manage_ompt.h`): a parallel region with
  `begin_parallel_phase()` on every thread and `finalize_parallel_phase()` after it, both left
  out under OMPT. Without OMPT, nothing left for the program to forget.
- Without OMPT, a `begin_parallel_phase()` with no `finalize_parallel_phase()` since the
  previous region stops the program, in every build (one comparison per thread and per region).
  It was not caught before (the message of the deferred-write check claimed it was); with a
  team size changing between regions it replays a freed deferred op. Check builds (Debug,
  `CHECK_OPTIM`) also stop on `release_chunk_pools()` and `destroy_block_chunk()` called inside
  a parallel region. Harness `phase_protocol_validation`.
- Chunk sizes may be multiples of 64 as well as powers of two (`CHUNK_SIZE`, `CHUNK_STATIC_SIZES`,
  `set_chunk_size`, `rechunk_to`): 192, 384, 576... `ParamChunk::check_chunk_size()` tells a valid
  size. `rechunk_to` moves between sizes that do not divide each other (256 -> 384).
- `ChunkSizeAdvisor` evaluates multiples of 64 as well (`Options::grid_bits`, set by
  `CHUNK_ADVISOR_GRID_BITS`, 2 by default: every C whose C / 64 has at most 2 significant bits --
  192, 384, 768... -- between S / 4 and S * 16; 3 adds 320, 448, 640...). A multiple of the current size (S * 3) is exact, like S * 2; the others are
  read from the sampled data across chunk boundaries. `advisor_demo` checks them against arrays
  built at those sizes.

### Changed

- The index split is a division (`SizePolicy::chunk_of` / `offset_of`): an unsigned division by
  the constant in static builds (a shift and a mask for a power of two, as before), a
  magic-number multiplication in `_dyn` ones (a power of two keeps the shift; about +1% on
  `shrink_bench_chunk2a_dyn`'s compute loop).
- fast_allocators: `free_elt_pow2` shifts only when the slab's element size is a power of two.
- fast_allocators: the macro `FAST_ALLOC_MAX_CONST_BLOCK_SIZE` is now `StaticConstArray::max_const_block_size`, a
  `static constexpr int` checked against the ladder's top bucket;
  `internal/static_const_array_define.h` is removed.
- A large shrink (`resize()` down) releases its chunks on a team, as `clear()` does, instead
  of one thread.
- Chunk2A: reading a chunk at its second value turns the slot pointer into the value's
  identifier with a shift (`region_stride` is a constant), not a 64-bit division.
- fast_allocators 2.1.9 (its two changes above); `CHUNK_FAST_ALLOCATORS_VERSION` follows.

### Deprecated

- `ParamChunk::check_pow2()` (use `check_chunk_size()`), `ArrayChunkBase::chunk_decalage()` and
  `ParamChunk::get_decalage_chunk()` (a shift means nothing for 384: use `i / chunk_size()`).

### Removed

- The A/B switches CHUNK_RECLAIM_COALESCE (sweeps always merge adjacent slabs) and
  CHUNK_RECHUNK_ADVISE_EACH_SLICE (the recycled slabs a v1 refill leaves are advised once, at
  its end), with their code paths.
- The CMake options CHUNK_PERF (the Chunk2 audit scan `compress()` now always runs when called)
  and CHUNK_DEBUG_SET_SIZE, and the old spelling HAVE_AGRESSIVE_COMPRESSION.
- `docs/environment_variables.md` keeps the user settings only (`CHUNK_THREAD_CAPACITY`,
  `CHUNK_REQUIRE_OMPT`, `CHUNK_RECHUNK_DRY_RUN`, and eight CMake options); every other switch
  moves to `docs/developer_settings.md`, outside the versioning policy. There,
  `CHUNK_RECHUNK_RECYCLE`, `CHUNK_RECHUNK_REUSE_LIST`, `CHUNK_THP`, `CHUNK_RECLAIM_EMPTY_SLABS`
  and the `CHUNK2A_COMPRESS_LEVEL` levels 1 to 3 carry a removal date (2026-12-31).

## 0.1.0 -- 2026-10-11

The first public release. The numbering starts again here: the earlier versions, up to 2.2.8,
were internal.

### Added
- A Spack recipe (Spack 1.x), in the repository: `spack/spack_repo/shrink_array`
  (`spack repo add` it, then `spack install shrink-array`). Variants `openmp`, `sizes`, `dyn`,
  `fast_alloc`, `stats`, `checks`, `second_value_indexed`, `examples`; versions `main`,
  `cpp23` and `develop`, from the branches. Checked: installed with GCC 16 and
  with Clang 23 (OpenMP, OMPT), then used by a `find_package(chunk)` project.

- `copy_range_mt(a, first, last, src_first)` and `transform_range_mt(a, first, last, src_first, op)`
  (`array_chunk_range_ops.h`, and plain-container overloads in `vector_range_ops.h`): the bulk
  copy and `std::transform`, callable by several threads at once on disjoint ranges, with one
  write session per chunk span like `fill_range_mt`. The three share one engine
  (`internal::store_range_mt`); a span's values are computed into a per-thread buffer before its
  session, so the caller's iterator or operation never runs inside it.
  `bits::store_bit_range_atomic` for `ArrayChunkBits`. Measured in `fill_range_mt_validation N
  perf` (Release, 2 cores): copy x6 to x12 at 1 thread, x4 to x7 at 2, over the element loop.
- `shrink_array::for_each_span(a, [first, last,] f)` (`array_chunk_range_ops.h`): the elements
  as `std::span<const T>` blocks, so the loop over them can be vectorised; a compressed chunk
  is passed as `f(v, count)` in O(1) when `f` accepts it. On a contiguous container
  (`std::vector`), one span.
- `element_ref_validation_<variant>` and `element_ref_misuse_<1..8>` (ctest, wherever the validation harnesses are built):
  every `std::vector`-style use of `a[i]` compiles and holds, and each of eight uses of a
  named element proxy fails to compile with `ElementRef`'s message.
- Every build: the non-const `operator[]`, `at()`, `front()` and `back()` return
  `ElementRef<proxy_type, T>` (`internal/element_ref.h`), the write proxy usable as a temporary
  only. `auto v = a[i];` then any use of `v` (conversion, assignment, compound operator,
  comparison, arithmetic) does not compile, with a message pointing to `T v = a[i];`. Its
  operations are force-inlined: the optimised code is the bare proxy's (clang 23 and GCC 16 at
  -O3 on `shrink_bench`; without the forced inlining GCC kept a complex `*=` out of line).
  `-DCHUNK_LOOSE_ELEMENT_REF` returns the bare proxy, for code not yet fixed.
- `chunk_hardening` (linked PUBLIC by every variant, exported) adds `-Werror=conditionally-supported`
  for GCC: `std::printf("%f", a[i])` passed the write proxy through `...` and printed garbage
  (`9223372036854775808.000000` for 1.5) with only a `-Wformat` warning; it is now an error, as
  clang already made it. `element_ref_misuse_9` (ctest) checks it. `ArrayChunkBase::element_proxy(i)` returns
  the proxy in every build, for code that keeps it on purpose.
- Usable as a subproject (`add_subdirectory`, `FetchContent`): `SHRINK_ARRAY_BUILD_EXAMPLES`
  and `SHRINK_ARRAY_BUILD_TESTS`, ON at the top level and OFF otherwise, so a consuming
  project builds the libraries only; the `chunk::` names exist in the build tree too
  (ALIAS targets), as after an install.
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
- `compile_all.sh`: `--with-token-sanitizer`.
- `scripts/check_headers.py` (ctest `check_headers`): every C++ file opens with the Apache
  license header, its years up to date, and carries no IDE `Created by` stamp; the scripts and
  build files carry `# Copyright ... CEA` / `# SPDX-License-Identifier: Apache-2.0`, the vendored
  headers the SPDX line of their upstream license; `--fix` repairs.

### Changed

- Fatal checks of the library report through `CHUNK_INVARIANT_FAILED` (location, message,
  backtrace) instead of a message on `stderr` / `std::cerr` and `std::abort()`: chunk size 0 at
  `create`, a manager registered before `init_indexes`, clearing an unwritten chunk, the
  `use_mm_protect` and second-value region `mmap` failures (with `strerror(errno)`), second-value
  identifiers exhausted, `[SLAB MISALIGN]`, `[LEDGER_BADPTR]`, `[CHUNK2]` missing candidate list,
  NaN (`CHUNK_DO_CHECK`), and the bare `std::abort()` of the bool / non-bool and build-mode
  branches that cannot be reached. Three messages that went on now stop, as broken invariants:
  v1's debug clear of an unwritten chunk (ChunkBase's same check already aborted), a failed
  identity swap in `rechunk_to` (check builds), and, in check builds, deferred writes about to be
  discarded by `flush_all_deferred` (`[DRAIN] ... LOST`; release builds still only print).
- Deferred operations: `AllocatorRegistry::get_or_create<T>()` caches its answer per thread
  and per type instead of scanning the registry at every allocation and release of an op or
  a context (`reset()` invalidates the cache); the virtual `slab_addr()` is no longer called
  when the pool ledger is off. In v1, the table number of a deferred write is no longer
  looked up in the registry (a linear scan over the live arrays) at enqueue and again at
  replay: the proxy's armed `num_tab` is captured in the context, and the replay reads it
  back (`Proxy::get_num_tab_cached`, `get_num_tab_async`); check builds compare it with the
  registry.
- `chunk_list_` (the chunks of an array) no longer has its new chunks value-initialised by
  `std::vector::resize` on the calling thread: its allocator (`NoInitAllocator`,
  `internal/array_chunk_core.h`) leaves them unconstructed, and the init loops of `create`
  and `resize` construct each chunk (`construct_chunk`) on the thread that initialises it.
  One serial pass less at each creation and re-chunk (the pages were already faulted in by
  a team), and a growing `resize()` no longer writes its new chunks from one thread.
  `getChunkList()` returns `const chunk_list_type&` (a `std::vector` with that allocator).
- Benchmark baseline (`shrink_bench_vector_*`): the arrays are a `std::vector` with an
  allocator that does not zero them (`examples/shrink_bench/baseline_vector.h`), zeroed by a
  parallel loop instead: the creation is parallel, and each page lands on the NUMA node of
  the thread that works on it. Creation times of the baseline are not comparable with
  earlier runs.
- CMake 3.20 is required (the documented minimum; the files said 3.12). The top-level
  project is `shrink_array`, and carries the version (`project(shrink_array VERSION ...)`),
  from which the chunk library takes its own. Consumed by another project, the build no
  longer writes the build type or the per-configuration flags into that project's cache.
- The variant libraries carry C++20 and OpenMP (when built with it) to their consumers
  (PUBLIC). An OMPT build links its consumers with `-rdynamic`, so the OpenMP runtime
  finds the library's OMPT tool in the executable.
- HAVE_AGRESSIVE_COMPRESSION is renamed `HAVE_AGGRESSIVE_COMPRESSION` (option,
  `chunk_config::have_aggressive_compression`, build directory names). CMake still accepts the
  old option name, with a deprecation message.
- The `cmake-build-omp-gnu-debug-ledger` build is no longer part of the default
  `compile_all.sh` run (`--with-ledger`, or a `--target` that selects it).
- The `cmake-build-omp-llvm-debug-std-alloc-token-sanitizer` build is no longer part of the
  default `compile_all.sh` run (`--with-token-sanitizer`, or a `--target` that selects it).
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
- `ParamChunk::init_indexes()` is optional: the first array of the program runs
  `init_indexes(true)` when it was not called. `release_chunk_pools()` and
  `destroy_block_chunk()` are optional too, and safe with arrays still alive: they then only
  empty the free lists.
- Global and `static` arrays work, for every variant, with OpenMP or not, fast or standard
  allocator, `MM_PROTECT` or not. The library's static state that an array reaches
  (per-type registries, pools, maps, the allocator ladder) is built at its first use or
  constant-initialised, and never destroyed (`chunk/internal/immortal.h`): a global array used
  to read it before its dynamic initialisation and after its destruction.
- No limit on the number of (chunk template, element type) pairs: the manager registry grows
  (it aborted past 256). Registering a manager no longer uses RTTI (dynamic_cast).
- **The build options are recorded, not passed as macros.** CMake generates
  `chunk/build_options.h` and `fast_allocators/build_options.h` (`CHUNK_OPT_*`,
  `FAST_ALLOC_OPT_*`), installed with the libraries; the headers read them there, never from the
  command line of the file that includes them. An installed consumer used to compile the headers
  with its own value of every option the targets did not carry (MM_PROTECT,
  `CHUNK_SECOND_VALUE_INDEXED`, CHUNK_NO_RECLAIM_EMPTY_SLABS, DEBUG, the test hooks...): two
  definitions of the same templates in one program, silently. A file compiled with OpenMP
  against a library built without it, or the reverse, is refused with an `#error`. The
  libraries no longer export HAVE_OMPT, `CHECK_OPTIM`, `CHUNK_STATS`, `CHUNK_STD_ALLOC`,
  `CHUNK_POOL_NO_REUSE`, `CHUNK_ALIGN_CHECK`, `HAVE_AGGRESSIVE_COMPRESSION`, `CHUNK_FORCE_LEDGER`,
  `CHUNK_DEFERRED_DRAIN_THRESHOLD`, `CHUNK_FORCE_DEFER_EVERY` as definitions.
- **A program compiled with other per-library settings than the library it links does not link.**
  The variant, `CHUNK_SIZE` and `CHUNK_DYNAMIC` are spelled into the name of one function each
  variant library defines (`chunk/internal/link_check.h`); the undefined symbol names the
  settings the program was compiled with.
- Renamed, without the old names: the variant macros CHUNK, CHUNK2, CHUNK2A are
  `CHUNK_VARIANT_V1`, `CHUNK_VARIANT_V2`, `CHUNK_VARIANT_V2A`. The bare `-D` switches DEBUG,
  PERF, DO_CHECK, DEBUG_SET_SIZE, RECHUNK_LEDGERTRACE, RING_LOG_DEBUG, RING_LOG_OP are
  CMake options `CHUNK_DEBUG` (on with a Debug build), CHUNK_PERF, `CHUNK_DO_CHECK`,
  CHUNK_DEBUG_SET_SIZE, `CHUNK_RECHUNK_LEDGERTRACE`, `CHUNK_RING_LOG_DEBUG`,
  `CHUNK_RING_LOG_OP`; MM_PROTECT is the option `CHUNK_MM_PROTECT` alone. fast_allocators:
  STD_ALLOC and ALLOC_ALIGN are `FAST_ALLOC_STD_ALLOC` and `FAST_ALLOC_ALIGN`; the platform
  macros MIC, KNL, ARM, ATOS_7, BSD are gone (no build defined them; `MAP_ANON` is used
  where `MAP_ANONYMOUS` is missing). chunk/chunk_define.h and
  chunk/internal/define_mm_protect.h are removed, and build_config::has_bzip2 moved to the
  benchmark (`have_bzip2`).
- **No name in the global namespace.** fast_allocators is in namespace `fast_alloc`
  (`fast_alloc::Singleton`, `fast_alloc::StaticConstArray`, `fast_alloc::TCTypeAllocator`,
  `fast_alloc::SIZE_BLOCK`, `fast_alloc::alloc_config`, the `sv*` functions...); the chunk
  library's last global names are in `shrink_array`: `shrink_array::build_config`,
  `shrink_array::chunk_config`, `shrink_array::g_ompt_suspended`, `type_name`, the term colours,
  `get_thr_rank_fast`, the chunk definitions (`Chunk2Def`...). A program that wrote
  build_config:: or chunk_config:: without `using namespace shrink_array` adds the namespace
  (or a namespace alias, as the benchmark does). complex_double is no longer defined by
  chunk/chunk_std.h (nothing used it; the benchmark has its own).
- Renamed, without the old names, the macros a header defines: `CHUNK_FORCE_INLINE`,
  `CHUNK_FLATTEN`, `CHUNK_ASSUME`, `CHUNK_CPU_PAUSE`, `CHUNK_INVARIANT_FAILED`,
  `CHUNK_INHERIT_CTOR_TYPENAME`, `CHUNK_RING_LOG_ANY`,
  `CHUNK_TUNING_LEVEL` (was FORCE_INLINE, SA_FLATTEN, SA_ASSUME, CPU_PAUSE, INVARIANT_FAILED,
  SA_INHERIT_CTOR_TYPENAME, RING_LOG_ANY, SHRINK_LEVEL; CRTP_USE_DEDUCING_THIS and
  CRTP_FORCE_DEDUCING_THIS are gone with the CRTP helper, every mixin using deducing this);
  fast_allocators: `FAST_ALLOC_DBG_ASSERT`, `FAST_ALLOC_LSAN_IGNORE`, `FAST_ALLOC_SL_LIST_CHECK`,
  `FAST_ALLOC_SIMD_SZ`, `FAST_ALLOC_MAX_CONST_BLOCK_SIZE`, `FAST_ALLOC_PMP_THREAD_RANK`,
  `FAST_ALLOC_PMP_MAX_THREADS`, `FAST_ALLOC_PMP_IN_PARALLEL_REGION` (was DBG_ASSERT,
  SA_LSAN_IGNORE, SL_LIST_CHECK, SIMD_SZ, MAX_CONST_BLOCK_SIZE, PMP_THREAD_RANK,
  PMP_MAX_THREADS, PMP_IN_PARALLEL_REGION). The fallback definitions of the membarrier(2)
  commands are constants of `shrink_array`, no longer macros. is_power_of_tow is
  `is_power_of_two`.
- Element write path (OpenMP builds): three acquire loads of the chunk header per write
  instead of five. Check #1 reuses the gate probe's value (reloaded only after waiting out a
  compressor), and the WAW guard reloads `pending_ops_` only when the entry read found
  something pending; check #2, the writer half of the Dekker handshake after
  `announce_write`, stays a fresh load. Landed in commit 083a0e54, whose message does not
  mention it.

### Fixed

- Deferred writes LOST at a phase end when the region that queued them had more threads than
  `omp_get_max_threads()` (`#pragma omp parallel num_threads(n)` above `OMP_NUM_THREADS`, or a
  team that shrank since): the serial `flush_all_deferred` drained only the ranks of its own
  default-sized team, then `storage.destroy()` discarded the others' ops
  (`nested_region_validation_*_1thr` under `CHUNK_FORCE_DEFER_EVERY`). It now sweeps the ranks
  beyond its team serially, as the compression passes do. Same mop-up added to v1's
  `Chunk::compress_chunk_to_compress`, whose candidates of those ranks were never swept (and
  which now keeps to its own rank inside a region, as Chunk2A's does).
- `a[i] = b[j]` compiles in every variant and copies the value (gated read of `b[j]`, then the
  full write path): v1's `Proxy::operator=(const Proxy&)` took `&from` past the deleted
  `operator&`, and v2 / v2A deleted the assignment.
- Chunk2, serial builds: `x *= 0` (or `/=`) on an element holding the second value left it
  counted as one, so a later second-value write could fold the chunk to that constant with the
  element still different, and the element was lost. `%=` did not count a zero result as the
  first value (a missed compression, no value lost).
- `ParamChunk::init_indexes<Types>()` compiles: it called a private member of `AllTypeChunk`.
- `scripts/check_includes.py` works outside a git checkout (a source archive, a Spack
  build) and checks files not yet added to git.
- Comments naming headers that moved to `internal/`, or a script that no longer exists.
- License headers: 29 files had none, others had it truncated, below `#pragma once` or with
  stale years; the IDE `Created by` stamps are gone. The 61 scripts and build files without a
  license (`.py`, `.sh`, `.cmake`, the CMake config templates, `CMakeLists.txt`) carry SPDX lines; the three
  vendored headers, installed with chunk, the SPDX line of their own license (MIT, BSL-1.0).

### Deprecated

- The CMake option HAVE_AGRESSIVE_COMPRESSION: use `HAVE_AGGRESSIVE_COMPRESSION`.

## Before 0.1.0

Internal versions, up to 2.2.8: see the history of the repository.

