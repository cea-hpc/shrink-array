# Changelog

All notable changes to shrink-array. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the policy below.

## Versioning policy

Two version numbers, both `MAJOR.MINOR.PATCH`:

| component | where it is set | installed as |
|---|---|---|
| the chunk library | the top-level `CMakeLists.txt` (`project(shrink_array VERSION X.Y.Z)`) | the `chunk` CMake package (`find_package(chunk X.Y)`) |
| fast_allocators | `fast_allocators/CMakeLists.txt` (`ALLOCATORS_*_VERSION`) | the `fast_allocators` package |

The **public interface** that the version describes is:

- the headers of `chunk/public/chunk/` that are not in `internal/`, and the names of
  `docs/api.md` they declare: `scripts/public_api.txt` lists them, and `check_public_api` fails
  when one is added, moved, removed or renamed without the list. The members of the public
  classes that name the library's machinery (`ManagerType`, `Manager`, `ChunkType`,
  `ChunkListType`, `ProxyType`, `get_chunk_list()`) are not part of it;
- the installed CMake targets (`chunk::chunk_zero_any_256`, `chunk::chunk_zero_any`, ...) and the
  variables `find_package(chunk)` sets;
- the CMake options and the environment variables of `docs/environment_variables.md` (the user
  settings); those of `docs/developer_settings.md` are development tools, which can change in any
  release;
- the documented behaviour: what is compressed, the thread contract, the sequential points.

Everything under `internal/`, the benchmark, the harnesses and the scripts can change in any
release.

| bump | when | examples |
|---|---|---|
| **MAJOR** | a change that can break a correct consumer: a public name, target, option or variable removed or renamed with no compatibility path, a changed contract | removing `zero_one_value::Array::set_second_value`; dropping `chunk::chunk_zero` |
| **MINOR** | an addition, or a change with a compatibility path (the old name kept, deprecated) | new targets installed; a user option renamed with the old spelling still accepted |
| **PATCH** | a fix that changes no interface | a race fixed; a wrong value in a trace |

The installed package declares `SameMajorVersion` compatibility: `find_package(chunk 1.2)`
accepts any 1.x at or above 1.2. A deprecated name stays for at least one MINOR release and is
removed only by the next MAJOR.

**Before 1.0** (0.x versions), as SemVer allows, a MINOR release plays the role of a MAJOR one
(it may break) and PATCH covers additions and fixes; the package then declares
`SameMinorVersion`: `find_package(chunk 0.1)` accepts 0.1.x only.

**fast_allocators** follows the same rules for its own interface. The chunk library pins the
exact fast_allocators version it was built with (`CHUNK_FAST_ALLOCATORS_VERSION` in
`chunk/CMakeLists.txt`, checked at configure time): bumping fast_allocators means bumping that
pin too, and at least the PATCH of chunk.

**Releasing**: move the entries of *Unreleased* under the new version and date, set the
numbers in the two `CMakeLists.txt`, tag `vX.Y.Z` (the Spack recipe downloads tags), and give
the tag's checksum to the recipe.

## Unreleased

### Added

- CI (P3 of the audit): four gaps closed. Per push, `fetch-content`: `examples/fetch_content`
  (new) consumes the checkout with `FetchContent`, and the job checks that the consumer keeps
  its build type, flags, `doc` target and tests, and installs nothing of the library. Nightly:
  `oldest-compilers` (GCC 13 and Clang 18, the minimum the README and the recipe state, never
  built by the CI), `x86-openmp-ubsan` (`SANITIZER=undefined`, new: UBSan, the first report
  stops the program; 256 and `_dyn`), and `spack` (the recipe installed from the checkout for
  `target=x86_64_v3`, no target flag from the project, no AVX-512 instruction in the
  libraries, `getting_started` built against the install).
- `CHUNK_PHASE_FENCES=0` runs without the four `seq_cst` fences that close the phase functions
  (`chunk/private/all_type_chunk.cpp`), with a warning. They were kept as "empirically
  load-bearing" since one unexplained checksum in August 2026; no handshake needs them as far
  as the code reads, and several defects that lost deferred writes at phase boundaries have been
  fixed since. The switch is there to settle it by measure before they are removed; the default
  keeps them (`docs/developer_settings.md` §1).
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
- `shrink_bench --advise_bool`: hands `tab_bool` and `tab_bool_bits` to the re-chunk decider
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
  options), and the arrays' `set_chunk_size()` (an empty array's geometry). 18 public headers. `simple_array`, `test_vecto_chunk`, `variant_compare`,
  `access_cost`, `advisor_demo`, `getting_started` and `bool_arrays` include public headers
  only, and `check_public_api` fails when an example outside the harnesses and the benchmarks
  includes an internal one.

### Changed

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
  before `tab_conn`'s line, in zero builds, that a connectivity is pattern::Array's case.
- `fill_range_mt` skips a chunk already at the fill value with one test, as `fill_range` does,
  instead of arming it element by element; `fill_range` does the same on its `bool` path.
- **The public interface, narrowed and frozen** (breaking). The public headers are what
  `docs/api.md` documents; `PMP_CHUNK.h`, `all_type_chunk.h`, `chunk_traits.h`,
  `global_stats_chunk_list.h`, `ring_log_debug.h`, `singleton.h`, `term_color.h`,
  `type_name.h`, `zero_chunk.h`, `zero_one_value_chunk.h`, `zero_any_chunk.h`, `pattern_chunk.h`,
  `zero_chunk_def.h`, `zero_one_value_chunk_def.h` and `zero_any_chunk_def.h` are under
  `chunk/internal/` (a program that included one includes the array's header instead).
  `release_chunk_pools()` and `destroy_block_chunk()` come with `chunk/param_chunk.h` (they
  were in `chunk/zero_chunk.h`). `ManagerType`, `Manager`, `ChunkType`, `ChunkListType`,
  `ProxyType` and `get_chunk_list()` are not part of the interface any more. The interface is
  `scripts/public_api.txt`, checked by `check_public_api` (ctest).
- **`ManagerChunk` is `TypeState`**: the shared state of one (variant, element type) -- reuse
  pools, compression candidates, per-rank allocators, deferred flush -- named after what it is.
  `ManagerChunkInt` is `TypeStateInterface`, `ManagerChunkDynInterface` /
  `ManagerChunkParallelInterface` / `ManagerChunkDynImpl` / `ManagerChunkParallelImpl` are
  `TypeStateDynInterface` / `TypeStateParallelInterface` / `TypeStateDynImpl` /
  `TypeStateParallelImpl`, and `internal/manager_chunk*.h` are `internal/type_state*.h`
  (`chunk/private/manager_chunk.cpp`: `type_state.cpp`). All internal: no interface change.
- Re-chunk: `zero_one_value` and `zero_any` (and `pattern`) free each old chunk as soon as the
  rebuild has copied it, as `zero` did, and rebuild in 16 slices with the fast allocator under
  threads: each slice boundary recycles the slabs the slice emptied into the next slice's
  allocations (`array_chunk_rechunk_refill.h`; the chunks' new `release_st()`), `bool`
  arrays included; `ArrayChunkBits` (bit-packed) keeps one pass. `zero_one_value_dyn`, x86-64, 2 threads, 10^8 elements: run peak 1,772 ->
  1,724 MB, end +0.9% (the recycled slabs stay resident, as for `zero`); `zero_any`'s peak
  on this benchmark is the initialisation's, unchanged. `rechunk_validation` gains
  `sliced-val2-grow` and `sliced-val2-uneven`: sliced rebuilds mixing zero, second-value and
  live sources.

### Removed

- `python/run_test.py`: it ran a `test/test1` that no longer exists, under a path left as
  `????` when the tree went public; `bench_runner` averages the memory over runs.
- `chunk/debug.h`, which nothing included (`print_backtrace` is
  `fast_alloc::logging::print_backtrace`).
- **The other compatibility paths kept until 1.0** (breaking). The variant macros
  `CHUNK_VARIANT_V1`, `CHUNK_VARIANT_V2`, `CHUNK_VARIANT_V2A` (use `CHUNK_VARIANT_ZERO`,
  `CHUNK_VARIANT_ZERO_ONE_VALUE`, `CHUNK_VARIANT_ZERO_ANY`); the `CHUNK_VARIANTS` spellings
  `chunk`, `chunk2`, `chunk2a`, now refused with the new name; the targets `chunk::chunk_v1*`,
  `chunk::chunk_v2*`, `chunk::chunk_v2a*` (link `chunk::chunk_zero*`, `chunk::chunk_zero_one_value*`,
  `chunk::chunk_zero_any*`). In the scripts, an old variant label on a command line (`chunk2a`,
  `chunk2a@dyn`, `tbb_chunk2`...) is refused with its new name, and `bench_and_test.py`'s
  hidden `--mem-chunk2a-vs-chunk2-max` is gone; the data recorded before the rename (perf
  databases, summaries, logs) are still read, their old labels as the new names.
- **The former C++ names kept until 1.0** (breaking: the next release is 0.3.0). The 45
  `[[deprecated]]` aliases: `ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A` (use `zero::Array`,
  `zero_one_value::Array`, `zero_any::Array`), `Chunk2`, `Chunk2A`, `Proxy2`, `Proxy2A` and the
  former `Chunk` / `Proxy` (use the variant's namespace), `use_chunk`, `use_chunk2`,
  `use_chunk2a`, `use_chunk2or2a` (use `use_zero`, `use_zero_one_value`, `use_zero_any`,
  `use_second_value`), the arrays' `chunk_type`, `manager_type`, `proxy_type`, `ind_type`,
  `chunk_list_type` (use `ChunkType`, `ManagerType`, `ProxyType`, `IndType`, `ChunkListType`),
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

- `TCSlab::alloc_block()` tested its capacity after computing it: `block_allocator -
  sizeof(void*)` is unsigned and wraps for a slab smaller than the back-pointer, and the
  quotient, cast to `int`, could come out positive (a slab of 1 byte and elements of 3: a
  capacity of 1,431,655,763) and pass the "does not fit" test. The sizes are now compared before
  the subtraction, and an element size below 1 is refused with them. No size the library asks
  for reaches it (Coverity 661091).
- `test_vecto_chunk` and `multi-mat`: `main()` catches `std::exception` and says it, as the
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
  `drain_own_candidates()` in the three variants). For zero_one_value the same call also
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
  `num_tab` no longer move their rows (`internal/stable_table.h`); the two thresholds the write
  path reads are atomics; the identity registries have a reader lock. A work array destroyed
  inside a region is torn down on its thread's rank alone (`internal/tab_users.h`: no walk of
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
  handed back in the scan, before its mark can go down, as `zero` and `zero_one_value` do.
  `docs/formal_models.md` section 8.7.
- A second `ParamChunk::init_indexes()` with the other argument (or `init_indexes(false)` after
  the first array had run `init_indexes(true)`) changed the allocation block size: the
  allocators registered afterwards cut their slabs to another size than the earlier ones of the
  same element type, against the one-slab-size-per-type invariant `slab_of()` relies on
  (`tc_allocator.h`). Out of contract (`docs/api.md`: once), now enforced: a call that would
  change the block size is ignored with a warning; the same size again is a harmless repeat.
  `pool_accounting` checks both.
- `a.push_back(a(i))` and `a.resize(n, a(i))` past the capacity, with an element of more than 16
  bytes (passed by reference): `resize()` ran first, and its compression pass could release the
  buffer `a(i)` points into, which the write of the new element could then reuse, filled with the
  constant, before the value was read. Both copy the value before `resize()`, as `assign()`
  already did. `open_types_validation` checks both on a 24-byte struct (`zero` and `zero_any`:
  `zero_one_value` has no `a(i)` on a non-`bool` element).
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
  unless it specialises `shrink_array::ValueIdentity<T>`. `zero_one_value` orders its second
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
  faults (#GP). The two SEGV of 2026-11-02 in `read_reclaim_validation` of `zero_one_value`
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
  and the multi-mat example's `VectZeroOneValue`. Generic `chunk` names (the
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
  connection check (`[batch] partition_info`: `ccc_mpinfo` on inti, `sinfo -o '%P %c %D %a'` in
  `slurm_example.toml`), beside the site file's `[partitions]`, which win for a name in both. Their
  directives, launcher and detached options come from the site's new `[discovered]` templates
  (`{name}`, `{cores}` per node), their architecture from a pattern on the name (`arm`); `skip`
  leaves some out. The table read is remembered for the next start.
  `python3 -m shrink_gui.partitions` shows, on the cluster, what the window reads from it.
- `formal/queue_handback.c`: the end of a compression pass against the candidate marking of a
  writer -- F6 (`docs/formal_models.md`, section 8.7, fixed: see Fixed).
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
  `compile_all_gnu.sh` (the builds take longer; `--target='omp-*'` leaves them out), and says at
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
  target names: `--exec='shrink_bench_*_zero_256'` for every backend.
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
  re-chunk the decider applies writes its `decide <array>: re-chunk S=..
  -> S'=.. [APPLIED]` line, as `[RECHUNK-DECIDE]` did before the log. Since
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
  chunk again (`chunk/internal/queue_flag.h`; F6, `docs/formal_models.md` section 8.7, model
  `formal/queue_handback.c`). Cost: one compare more on a refused marking, an exchange and a
  CAS per candidate in a pass.
- `fill_range_mt()`, `copy_range_mt()`, `transform_range_mt()` on `zero_any` elements with relaxed
  access (`double`, `int`, ...): two threads filling the two ends of one chunk could both read the
  other's edge before its store (store buffering, allowed on x86 and AArch64) and neither queue
  the chunk they left uniform. The filter now looks again after a seq_cst fence when its first
  look says "not uniform". The check builds' audit counted 6 to 52 such chunks per
  `shrink_bench` run at 16 to 128 threads on Milan. Model: `formal/range_edges.c`
  (`docs/formal_models.md`, section 8.8, F7).
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
  `rechunk=info` writes its moves (`decide <array>: re-chunk S=.. -> S'=..
  ... [APPLIED]`), `rechunk=debug` its waits too. `CHUNK_RECHUNK_DRY_RUN`,
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

### Fixed

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
  `CHUNK_CRTP_USE_DEDUCING_THIS`, `CHUNK_CRTP_FORCE_DEDUCING_THIS`, `CHUNK_INHERIT_CTOR_TYPENAME`, `CHUNK_RING_LOG_ANY`,
  `CHUNK_TUNING_LEVEL` (was FORCE_INLINE, SA_FLATTEN, SA_ASSUME, CPU_PAUSE, INVARIANT_FAILED,
  CRTP_USE_DEDUCING_THIS, CRTP_FORCE_DEDUCING_THIS, SA_INHERIT_CTOR_TYPENAME, RING_LOG_ANY, SHRINK_LEVEL);
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

