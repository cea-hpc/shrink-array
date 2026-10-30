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

- the headers of `chunk/public/chunk/` that are not in `internal/`, and what they declare;
- the installed CMake targets (`chunk::chunk_v2a_256`, `chunk::chunk_v2a`, ...) and the
  variables `find_package(chunk)` sets;
- the CMake options and the environment variables of `docs/environment_variables.md` (the user
  settings); those of `docs/developer_settings.md` are development tools, which can change in any
  release;
- the documented behaviour: what is compressed, the thread contract, the sequential points.

Everything under `internal/`, the benchmark, the harnesses and the scripts can change in any
release.

| bump | when | examples |
|---|---|---|
| **MAJOR** | a change that can break a correct consumer: a public name, target, option or variable removed or renamed with no compatibility path, a changed contract | removing `ArrayChunk2::set_second_value`; dropping `chunk::chunk_v1` |
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

- shrink_gui 0.15: the partition list also offers the partitions the machine lists at the
  connection check (`[batch] partition_info`: `ccc_mpinfo` on inti, `sinfo -o '%P %c %D %a'` in
  `slurm_example.toml`), beside the site file's `[partitions]`, which win for a name in both. Their
  directives, launcher and detached options come from the site's new `[discovered]` templates
  (`{name}`, `{cores}` per node), their architecture from a pattern on the name (`arm`); `skip`
  leaves some out. The table read is remembered for the next start.
  `python3 -m shrink_gui.partitions` shows, on the cluster, what the window reads from it.
- `formal/queue_handback.c`: the end of a compression pass against the candidate marking of a
  writer. It finds F6 (`docs/formal_models.md`, section 8.7), open: a chunk made constant
  between `steal_unmark()` and the reset of its "queued" flag is dropped from every list and
  stays materialised until written again -- memory, no wrong value. The model accepts a fix,
  the flag handed back (1 -> 2, a writer's 3, re-queued by the pass), not applied yet.
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

