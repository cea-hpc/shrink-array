# Developer settings

> French version: [reglages_developpeur_fr.md](reglages_developpeur_fr.md).

> Status: November 2026. The switches for the library's developers: tuning of the decider and of
> the allocators, traces, forced rechunks, the benchmark, compilation macros, debugging and build
> options, the build scripts. They are **development tools**: not covered by the versioning policy
> of [CHANGELOG.md](../CHANGELOG.md), they can change or disappear in any release. What a user of
> the library sets is in [environment_variables.md](environment_variables.md).
>
> Unless stated otherwise, a variable is read only once, and `0` or an empty value is the same as
> leaving it unset -- except for the traces of §2 marked *set*, on as soon as they are defined
> (even at `0`), and for the variables whose row says what `0` does. A variable that is not read
> is ignored without any message: when in doubt, check with
> ``grep -rn '"NAME"' chunk fast_allocators examples``.
>
> **Removal dates.** A switch that keeps an old behaviour, or a choice no build tests, carries a
> removal date in its row; at that date it goes, with its code path, unless a measurement written
> down here justifies keeping it. A new switch of that kind gets its date when it is added.

***

## 1. Tuning of the rebuild and the allocators

The decider's thresholds (`RechunkPolicy`) are not environment variables: a program sets the
fields, and `shrink_bench` exposes them as options (§4). See
[adaptive_rechunk.md](adaptive_rechunk.md) §3.

| variable | default | effect |
|---|---|---|
| `CHUNK_RECLAIM_THREADS` | 1 | threads issuing a sweep's `madvise` calls; `0`: `omp_get_max_threads()`. On Milan (64 threads), `0` is 3.5% faster for `zero_or_value` and the same for the other variants; it stays at 1 until it is measured on Grace, where 288 threads would issue them |
| `CHUNK_SECOND_VALUE_ID_LIMIT` | unset | test hook: lowers the number of identifiers of `zero_any`'s second-value registry (the region's capacity otherwise), and sizes the value table to match, so that a harness reaches the registry's limit with a few values (`second_value_registry_validation`, `second_value_reclaim_validation`); can only lower it |
| `CHUNK_DEFERRED_POOL_KB` | 64 | slab size of the deferred-write pools, in KiB (the operation slabs are never given back: larger means fewer allocator calls but more resident memory) |
| `CHUNK_PMP_JOIN_US` | 5000 | std backend with the library's teams on TBB (`shrink_array::tbb::use_tbb_for_library_teams()`): how long a team waits for its TBB tasks to start, in microseconds; the tasks started by then make the team (one thread, the caller, at worst), a later one returns at once. 5 ms rather than 200 us: a late task is otherwise taken and dropped by a TBB worker during the program's next region (144 threads: 79.5 s -> 50.9 s on the TBB bench) |
| `CHUNK_PMP_SPIN_US` | 1000 | std backend: how long a thread waiting at a team's barrier, at its end or for the next team spins before it sleeps, in microseconds (OpenMP's blocktime); `0`: sleep at once. Spinning: at 160 threads the re-chunk's slice-boundary drain went from 2.45 s to the order of the OpenMP build's 0.24 s |

`python/ab_settings_campaign.py` measures such a setting against the defaults (computation
time, peak and end memory) on `zero`, `zero_any` and `zero_or_value` at dyn, three runs per
point, interleaved; run it inside a job allocation (`--help` lists the points), and again with
the same `--out` to resume an interrupted campaign. Its first campaign (Milan, October 2026)
settled the allocator settings formerly read here: the affinity mode per variant, the
partial reclaim of the slabs in use, the reuse budget, and no sweep after the compression
passes (CHANGELOG.md, Removed).

## 2. Traces and diagnostics

The traces are messages in the library's log, chosen by `CHUNK_LOG` (see
[environment_variables.md](environment_variables.md)). The variables of the former traces
(`CHUNK_RECHUNK_TRACE`, `CHUNK_RECHUNK_TRACE_SLICES`, `CHUNK_RECHUNK_MEMTRACE`,
`CHUNK_RECHUNK_TRACE_MEM`, `CHUNK_AFFINITY_TRACE`, `CHUNK_RECYCLE_TRACE`, `FAST_ALLOC_TRACE`) and
of the former reports (`CHUNK_ARENA_REPORT`, `CHUNK_LEDGER_REPORT`, `CHUNK_OMPT_STATS`,
`CHUNK_CST2_PROBE`) are no longer read. A report is switched on by the level of
its channel, then written in full; a bare level (`CHUNK_LOG=info`) switches them all on, and the
arena's and OMPT's counters with them, which cost a little time.

| setting | what it prints |
|---|---|
| `CHUNK_LOG=rechunk=info` | the decider's moves (``decide <array>: re-chunk S=.. -> S'=.. ... [APPLIED]``, or `[dry-run] would re-chunk`) |
| `CHUNK_LOG=rechunk=debug` (formerly `CHUNK_RECHUNK_TRACE`, and `CHUNK_RECHUNK_CHURN` for the churn) | in addition, the verdicts that wait, for each rechunk the duration of the phases and of the slices, and at each pass, per array analysed, the materialisations since the previous pass and an estimate of the transient cost they imply (`churn` line); for zero_any (and pattern), a `teardown split` line divides each rechunk's `tmp teardown` phase into its steps (flush, candidates, terminal sweep, park region, registry release, metadata tables, key drain, registry erase, `chunk_list_` unmap, rest) |
| `CHUNK_LOG=rechunk=trace` (formerly `CHUNK_RECHUNK_TRACE_SLICES`) | in addition, memory at each slice of the `zero` rebuild |
| `CHUNK_LOG=memory=info` (formerly `CHUNK_RECHUNK_MEMTRACE`) | at each advisor pass: exact memory, including the resident part of the buffer arena |
| `CHUNK_LOG=memory=debug` (formerly `CHUNK_RECHUNK_TRACE_MEM`) | in addition, the memory side of the rechunk trace: exact memory at each step (`rss` line), occupancy of the new geometry, chunks folded, refused and installed directly, allocator census. Costly (seconds per run at tens of GB): hence a channel separate from the timings |
| `CHUNK_LOG=advisor=info` | check builds: at an array's destruction, the chunk size that would have used the least memory, when it saves at least 10 % |
| `CHUNK_LOG=advisor=debug` (formerly `CHUNK_ADVISOR_TABLE`) | in addition, the full advisor table of each array analysed, at each pass |
| `CHUNK_LOG=affinity=debug` (formerly `CHUNK_AFFINITY_TRACE`) | each per-array set of allocators created (key, size, budget) |
| `CHUNK_LOG=compress=info` | check builds: the blocks left to compress after the final flush, and `zero_any`'s constant blocks never queued; the compression residual of an audit (`compress()` on zero_or_value::Array / zero_any::Array); `compress=debug` adds zero_or_value::Array's per-chunk detail |
| `CHUNK_LOG=arena=info` (formerly `CHUNK_ARENA_REPORT`) | the summary of the buffer arena at each drain point (slabs handed out, given back to the system; its `partial` line, the free pages given back from the slabs still in use; its `freepath` lines, counted only at this level) |
| `CHUNK_LOG=recycle=debug` / `recycle=trace` (formerly `CHUNK_RECYCLE_TRACE`) | check builds: the foreign blocks the deferred-write pool drops, on a doubling schedule / one line per block |
| `CHUNK_LOG=ledger=info` (formerly `CHUNK_LEDGER_REPORT`) | the report of the block registry (builds with the registry), at exit |
| `CHUNK_LOG=ompt=info` (formerly `CHUNK_OMPT_STATS`) | statistics of the synchronisation regions (OMPT builds), at exit |
| `CHUNK_LOG=compress=info` in `shrink_bench` (formerly `CHUNK_CST2_PROBE`) | `zero_any`: in addition, the report of the second-value registry at each advisor pass (`gc_mark_report()`) |
| `CHUNK_SECOND_VALUE_RECLAIM` | `zero_any`, `N`: every N iterations, frees the second values no chunk uses any more (`reclaim_second_values()`) and prints how many, then each type's identifiers in use against the capacity, free, reclaims and refusals (`second_value_stats()`) (benchmark, any build) |
| `ALLOC_TOKEN_REPORT` | `CHUNK_SANITIZER=token` builds always print, on exit, the allocations per token (the 40 largest); this sets how many: `0` none, `N` the N largest, `all` every token |
| `CHUNK_LOG=alloc=trace` (formerly `FAST_ALLOC_TRACE`) | one `munmap` line per emptied slab that a pool's reserve does not keep and that is freed |

The per-array traces (the `churn` line, the advisor table) cover every array the decider
analyses; `RechunkPolicy::trace_array` (`shrink_bench --rechunk_trace_array <name>`) restricts them
to one, as the former `=name` values of `CHUNK_RECHUNK_CHURN` and `CHUNK_ADVISOR_TABLE` did.

## 3. Forced rechunks (tests)

`RechunkPolicy::force_size` and its companions; in `shrink_bench`, the options:

| option | effect |
|---|---|
| `--force_rechunk <size>` | forces a rechunk to this size, even if the decider would not do it |
| `--force_rechunk_at <iteration>` | delays this forced rechunk until this iteration |
| `--force_rechunk_toggle` | rechunk at each pass, alternating with `--force_rechunk_alt <size>` (64 by default) |

## 4. The `shrink_bench` benchmark

| variable or option | effect |
|---|---|
| `--advisor_period <n>` | period of the advisor passes, in iterations (default `max(10, iter_max / 20)`; 0 disables it) |
| `--advise_five` | only the five historical arrays are handed to the decider (default: all arrays except `array_bool` and `array_bool_bits`) |
| `--advise_bool` | `array_bool` and `array_bool_bits` are handed to the decider too (default: they stay at their birth geometry, so that the two bool containers are compared as they are) |
| `--rechunk_dry_run` | the decider analyses and logs, but applies no rechunk (`RechunkPolicy::dry_run`) |
| `--rechunk_max_coarsen <n>` | maximum factor of a coarsening (`max_coarsen_factor`, 4; 1 = no limit) |
| `--rechunk_min_gain_mb <x>` | absolute threshold for the data-side gain, in MB (`min_gain_bytes`, 8; 0 removes it) |
| `--rechunk_grid_bits <n>` | the advisor's candidate sizes between the powers of two: the multiples of 64 whose C / 64 has at most this many significant bits (`advisor_grid_bits`, 2: 192, 384, 768...; 3: 320, 448, 640 too; 1: powers of two only). See [adaptive_rechunk.md](adaptive_rechunk.md) §2 |
| `--rechunk_coarsen_min_meta_mb <x>` | minimum metadata gain of a coarsening, in MB (`coarsen_min_meta_gain_bytes`, 1; 0 removes it) |
| `--rechunk_coarsen_min_meta_rate <x>` | minimum metadata gain per second of estimated rebuild time, in MB/s (`coarsen_min_meta_rate`, 250; 0 removes it) |
| `--rechunk_backoff_max <n>` | longest back-off of a settled array, in passes (`max_backoff_ticks`, 8; 0 or 1 disables it) |
| `--rechunk_trace_array <name>` | the decider's per-array traces (`churn` line of `CHUNK_LOG=rechunk=debug`, advisor table of `CHUNK_LOG=advisor=debug`) for this array only (`trace_array`; default: every array) |
| `SHRINK_BENCH_EXACT_RSS` | `1`: all memory lines are exact (expensive) |
| `SHRINK_BENCH_RSS_SAMPLE_MS` | sampling period of the exact peak, in ms (disabled by default) |
| `CHUNK_BENCH_MEM_PERIOD` | period of the memory line, in iterations |
| `CHUNK_BENCH_KSUM_PERIOD` | period of the checksum line, in iterations |
| `SHRINK_BENCH_AUDIT_FREQ` | period of the compression audit (verification builds) |
| `CHUNK_BENCH_ULONG_BANDS` | number of distinct second values of the `unsigned long` array (4,096; 0 = no limit) |
| `CHUNK_MALLOC_ARENAS` | number of glibc `malloc` arenas (0 = unchanged) |
| `SHRINK_BENCH_MALLOC_TRIM` | `1`: `malloc_trim` at each memory line, and not only at the checkpoints ("after init", "end"); distorts the curve and costs time in the timed loop |
| `SHRINK_BENCH_KEEP_POOLS` | `1`: no `trim_reuse_pools` at the checkpoints (the parked buffers, and with the fast allocator the emptied slabs, stay resident) |
| `BENCH_SCHED_BLOCK` | size of the scheduling blocks of the benchmark phases (64; 0 = contiguous slices) |
| `BENCH_RANK_TIMING` | `1`: every bench backend (OpenMP, std, TBB) times every region per rank (per virtual rank under TBB) and prints at exit, per call site, the split of the threads' time into busy / imbalance / launch / tail, and the serial time between regions (stderr, `[rank-timing]` lines). Under TBB the end of the phase (`finalize_parallel_phase`) is timed apart; elsewhere it falls in the tail (OMPT's, at each region's end) or in the serial time (the bench's own, between regions) |
| `BENCH_TBB_PARALLEL_PHASE` | `1`: TBB bench only. The bench's arena opens one oneTBB parallel phase for the whole run (`task_arena::start_parallel_phase()`, a oneTBB 2022.x preview, `TBB_PREVIEW_PARALLEL_PHASE` set by CMake). It only changes oneTBB's leave policy: a worker out of work may then stay in the arena, spinning, for 1 ms at most -- which it already may except on hybrid CPUs (P- and E-cores), where TBB makes it leave at once by default. Useful there only; no effect measured on a 72-thread AArch64 node. A line on stderr says whether it is on; ignored, and said so, with a oneTBB without the preview |
| `BENCH_TBB_LINGER_US` | `<us>`: TBB bench only. How long a region's task, its rank done, waits (spinning) for the other ranks before returning to TBB, so the worker stays in the arena past TBB's 1 ms when the ranks are imbalanced, for the finalize's team and the next region. It waits only once every rank has started, never longer than the budget. `0` (default): it returns at once |
| `BENCH_TBB_PIN` | `1`: TBB bench only, Linux. The TBB counterpart of `OMP_PROC_BIND=close`: a thread entering the bench's arena (the main thread included) is pinned to the CPU of its slot, slot `s` to the `s`-th CPU of the process's affinity mask at the arena's creation (cgroup, `taskset`), so that with the static partitioner rank `k` runs on the same core region after region. Leaving the arena, a thread gets the CPUs of the arena's slots (the first `max_concurrency()` of the mask), not every CPU of the process: the main thread's serial code between regions stays on the arena's socket. For a machine the run has to itself: a pinned thread cannot flee a busy core and the team waits for it at every join (15-30x slower beside a 3-job build, 14 threads). A line on stderr gives the CPUs used; ignored, and said so, elsewhere than on Linux |
| `BENCH_TBB_FINALIZE_INSIDE` | `1`: TBB bench only. Once every rank of a region is done, the first task to see it runs `finalize_parallel_phase()` itself, and the region's other tasks serve the library's teams (they take the team's join at once) instead of TBB tasks being enqueued one by one and waited for -- 0.19 ms + 0.33 ms per finalize on a 72-thread AArch64 node, for 0.19 ms of work. Only when no task of the region returned to TBB before (otherwise the finalize runs after the tasks, as without it); waits up to `BENCH_TBB_LINGER_US` (20 ms when unset) for the ranks. A line on stderr says it is on |
| `NO_IO`, `COMPRESS` | equivalents of `--no_io` and `--compress` |
| `--no_mask`, `--mask_scale <x>` | without `array_mask`; or its size, `x × sz` (default 1/32) |
| `--no_conn`, `--conn_scale <x>` | without `array_conn`, the mesh connectivity (`pattern::Array` in the `zero_or_value` and `zero_any` builds); or its size, about `x × sz` in whole rows (default 1/32) |
| `--no_struct` | without the arrays of structures (`Point3`, `Cell`) |

## 5. Compilation macros

Tuning values for which the headers give a default, and test hooks. The variant and geometry macros,
and how the shared options reach the headers, are in
[environment_variables.md](environment_variables.md) §3.

| macro | effect |
|---|---|
| `CHUNK_DYN_MAX_SIZE=<n>` | dynamic geometry: largest chunk size a rechunk can reach, a power of two (4096); sizes the shared buffers of the constant values, `const_buffer_size × sizeof(T)` per distinct second value |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<bytes>` | metadata threshold above which an array can have its own allocators (16 MiB; 0 = never) |
| `CHUNK_COMPRESS_MAX_YIELDS=<n>` | how long a parked compressor waits before giving up its pass (10,000 *yields*; 0: give up at the first writer seen). See [synchronisation_dekker.md](synchronisation_dekker.md) |
| `CHUNK_WRITE_WAIT_MAX_YIELDS=<n>` | how long a writer facing a compressor waits before deferring its write (64 *yields*) |
| `CHUNK_READ_RECLAIM_SLOTS=<n>` | number of reader threads tracked individually by the buffer reclamation (1,024) |
| `CHUNK_STATS_MAX_THR=<n>` | number of ranks of the `CHUNK_STATS` counters (1,024) |
| `CHUNK_READ_DEP_ORDER=1` | **experimental**, AArch64 only, off under TSan: a pure read loads the chunk's `values` pointer relaxed, ordered with the element load by their address dependency instead of an acquire (`chunk_config::read_dep_order`, config.h). Under evaluation on Grace (`access_cost`): pass it in `CMAKE_CXX_FLAGS` of a build of its own |
| `CHUNK_TEST_READ_DELAY_EVERY=<n>`, `CHUNK_TEST_READ_DELAY_US=<µs>` | test hook: one read in N by a thread that requested it waits (50 µs), to widen the reader / reclamation window (`read_reclaim_validation`) |
| `CHUNK_TEST_PASS_DELAY_EVERY=<n>`, `CHUNK_TEST_PASS_DELAY_US=<µs>` | test hook: a delay (200 µs) in one compression pass in N |
| `CHUNK_GCOV_DUMP` | coverage builds (`--coverage`): the fatal report writes the gcov counters before `abort()`, so the paths the death tests take are counted; set by the nightly coverage job |

The thresholds of `zero_any`'s compression passes are constants of `zero_any_chunk_def.h` (the levels
1 to 3 of the former `CHUNK2A_COMPRESS_LEVEL` were never built or tested).

## 6. CMake options

The options of [environment_variables.md](environment_variables.md) §2 are not repeated.

| option | default | effect |
|---|---|---|
| `SHRINK_BENCH_DYN_SIZE` | 64 | starting size of the `_dyn` targets |
| `SHRINK_BENCH_INT64` | `OFF` | targets with 64-bit indices (`_int64`) |
| `SHRINK_BENCH_INT64_SIZES` | 1024 | with `SHRINK_BENCH_INT64`: the static chunk sizes that get an `_int64` twin, a subset of `CHUNK_STATIC_SIZES` (the `_dyn` targets always get one) |
| `SHRINK_INTEL_LIBM` | `ON` | x86-64: link Intel's libm if it is found |
| `SHRINK_ARM_LIBM` | `OFF` | aarch64: link ArmPL's `libamath` (no gain measured on Grace) |
| `CHUNK_ICPX_O0_WORKAROUND` | `OFF` | workaround for an icpx internal error on three files |
| `CHUNK_MM_PROTECT` | `OFF` | `zero`, `zero_or_value`: shared buffers of the constant values made read-only (`mprotect`); a stray write into a compressed chunk causes a segmentation fault instead of corrupting the value (debugging build) |
| `CHUNK_LIB_UNITY` | `ON` | each variant archive (`chunk_zero_any_256`, ...) compiled as a single unit (unity build, 4 to 6 times less compilation); a file that has its own compile options stays separate |
| `SHRINK_BENCH_PRINT_MEM` | `ON` | `shrink_bench` prints its memory lines (after init, every freq_mem_report iterations, after a compression); `OFF` to time a run without the RSS reads behind them |
| `SHRINK_BENCH_UNITY` | `ON` | each `shrink_bench` executable compiled as a single unit (unity build, 3.5 to 4 times less compilation); `OFF` to compare timings with earlier campaigns, since inlining across benchmark files can change them; `SHRINK_BENCH_UNITY_BATCH` files per unit (0 = all) |
| `CHUNK_DEBUG_INFO` | `auto` | debug information of the optimised builds: `full` (`-g`), `lines` (`-g1`: function names and line tables, enough for perf and gdb backtraces, not the locals), `none`; `auto`: `full` with `CHUNK_CHECK_OPTIM` and no sanitizer, or with the `token` sanitizer; `lines` otherwise, the address, thread and undefined sanitizers included, `CHUNK_CHECK_OPTIM` or not (their reports need only the line tables). The machine code is the same; `lines` compiles 20 to 27 % faster with 40 % less peak memory. Debug builds always get `-g` |
| `CHUNK_VALIDATION_UNITY` | `ON` | the validation harnesses grouped into one executable per variant and geometry (unity build); `OFF`: one executable per harness |
| `CHUNK_VALIDATION_UNITY_BATCH` | 0 | with `CHUNK_VALIDATION_UNITY`: harnesses per translation unit of such an executable; 0: all those of a group in one unit. A small value (3) lowers the compiler's memory peak on a machine with little memory |
| `CHUNK_BUILD_VALIDATION` | depends on the build | compile the harnesses of `examples/validation`, and with them the test hooks (`CHUNK_TEST_HOOKS`); by default `ON` in Debug, with `CHUNK_CHECK_OPTIM` or a sanitizer, `OFF` elsewhere (not cached: follows the current configuration) |
| `FAST_ALLOC_CHECK_LISTS` | `OFF` | stronger debug level for the slab lists: in Debug, full check of the list at each insertion and removal (slow); an ordinary Debug build keeps the light checks |
| `CHUNK_AGGRESSIVE_COMPRESSION_DEFAULT` | `OFF` | default of the "aggressive" compression, a run-time switch: `CHUNK_AGGRESSIVE_COMPRESSION=1` (or `0`) sets it for a run. The pass between two phases is skipped, and all compression happens during the parallel regions, at unpredictable times. Slower, but it exercises the concurrent paths: a validation mode, which no longer needs a build of its own |
| `CHUNK_FORCE_LEDGER` | `OFF` | keeps the block registry (`pool_ledger.h`) active under OpenMP, where it is off by default: its global mutex serialises every allocation and free. Detects double frees and illegal returns to the reserve |
| `CHUNK_DEFERRED_DRAIN_THRESHOLD` | empty (4,096) | threshold at which a rank flushes its own queue |
| `CHUNK_FORCE_DEFER_EVERY` | 0 | one write in N through the deferred path, across the whole tree (validation; 32 threads at most) |
| `CHUNK_DEBUG` | depends on the build | the library's debug checks; `ON` with `CMAKE_BUILD_TYPE=Debug` (not cached: follows the current configuration) |
| `CHUNK_POOL_NO_REUSE` | `OFF` | deferred-op pools without free list: every block a real `new` / `delete` (ASan, TSan) |
| `CHUNK_DO_CHECK` | `OFF` | every written value checked for NaN |
| `CHUNK_RECHUNK_LEDGERTRACE` | `OFF` | trace every block-registry transition |
| `CHUNK_RING_LOG_DEBUG`, `CHUNK_RING_LOG_OP` | `OFF` | debug ring log of every event (heavy); of the op-lifecycle events only (light) |
| `CHUNK_GLIBCXX_ASSERTIONS` | `OFF` | libstdc++ assertions (`_GLIBCXX_ASSERTIONS`) in the library and its consumers |
| `CHUNK_UBSAN` | `OFF` | UndefinedBehaviorSanitizer (to pair with ASan, not with TSan) |
| `CHUNK_ALIGN_CHECK` | `OFF` | abort if a slab block is misaligned for its type |
| `CHUNK_NO_IPA_ICF` | `OFF` | disables `-fipa-icf` (otherwise GCC merges identical functions, which scrambles call stacks) |
| `CHUNK_WARNINGS`, `CHUNK_WERROR` | `ON`, `OFF` | `-Wall -Wextra` on all builds; warnings as errors in Debug |
| `CHUNK_HEAVY_JOB_POOL` | empty | Ninja job pool for the executables compiled as a single unit (limits compilation memory) |
| `CHUNK_X86_MARCH` | `native` | GCC, Clang on x86-64: `-march=` of the optimised builds; a fixed level gives a binary independent of the machine (`x86-64-v3` in the nightly's ASan job, `native` faults there on Zen 5) |
| `CHUNK_NATIVE_TUNING` | ON (OFF in a project that consumes the library: add_subdirectory, FetchContent) | OFF: no target flag from the project at all (`-march=`/`-mcpu=native`, `-tp=host`, `-xHost`): the target is the caller's. The Spack recipe sets it OFF, so that the spec's `target=` is the one compiled for; the project's flag came after the wrapper's and won |
| `CHUNK_USE_LIBCXX` | `OFF` | Clang builds: `ON` for libc++ (`-stdlib=libc++`) instead of the compiler's default standard library, libstdc++ on Linux |
| `CHUNK_GCC_TOOLCHAIN_DIR` | depends on the machine | Clang builds: the GCC installation whose libstdc++ and runtime clang uses (`--gcc-toolchain=`); empty: the driver's own choice. A path that does not exist is ignored, with a warning |
| `CHUNK_USE_CCACHE` | `AUTO` | compile through ccache (>= 4.8): `AUTO` (when it is found, on `PATH` or with `-DCCACHE_EXECUTABLE=...`), `ON` (an error otherwise), `OFF`. The cache is split by host CPU, so `-march=native` objects never cross nodes; a `CMAKE_CXX_COMPILER_LAUNCHER` already given is kept |
| `CHUNK_USE_MOLD` | `AUTO` | link with mold: `AUTO` (when it is found and the build supports it), `ON` (an error otherwise), `OFF` |
| `CHUNK_VEC_REPORT` | `OFF` | fast-math targets of the examples: the compiler reports the loops it did not vectorise |
| `GETTING_STARTED_LIB`, `BOOL_ARRAYS_LIB` | `shrink_array::zero_any`, `shrink_array::zero_or_value` | `examples/getting_started` and `examples/bool_arrays` built on their own, against an installed library: the library target they link |
| `CHUNK_USE_GPERF`, `CHUNK_ENABLE_IWYU` | `OFF` | gperftools profiling; include-what-you-use during compilation |
| `CHUNK_ONEAPI_XHOST`, `CHUNK_ONEAPI_NO_INTEL_LIBS`, `CHUNK_ONEAPI_FAST_FP` | `OFF` | icpx: Intel-only binary (`-xHost`); no Intel runtime library; keep `-fp-model=fast` |

For clang builds on x86-64, `CHUNK_LIBMVEC_LINKS` (in the cache, computed at
configure time) tells whether the glibc vector library provides the math
functions of the benchmarks; otherwise, their loops stay scalar (glibc < 2.35).

## 7. The build scripts

`scripts/compile_all.sh` configures and builds the whole matrix (GNU, LLVM, oneAPI, AOCC,
NVHPC, sanitizers); it delegates the GNU builds to `scripts/compile_all_gnu.sh`, passing it
the common options. `--help` gives the full list.

| option | effect |
|---|---|
| `--dryrun` | print the commands without running them |
| `--build` | without reconfiguring (rebuild only) |
| `--fresh-cache` | delete the `CMakeCache.txt` of each selected build before configuring |
| `--debug-only` | only the Debug builds |
| `--only=<c>`, `--no-<c>` | a single compiler, or without it (`gnu`, `oneapi`, `nvhpc`, `llvm`, `aocc`, `acfl`; `oneapi` and `aocc` on x86-64 only, `acfl` on ARM only) |
| `--list-targets`, `--list-execs`, `--list` | list the configurations `--target` accepts (configured here or not, x86-64 or ARM only), the targets `--exec` accepts in each configured build directory, or both; nothing is built. They honour `--target=<glob>` and `--exec=<glob>` |
| `--with-nvhpc` | add NVHPC, disabled by default (about 1 h 30 per build on x86, with no gain) |
| `--no-sanitizers` | without the LLVM sanitizer builds |
| `--with-ledger` | add `cmake-build-omp-gnu-debug-ledger` (OpenMP Debug with `CHUNK_FORCE_LEDGER=ON`), disabled by default; a `--target` that selects it also builds it |
| `--with-token-sanitizer` | add `cmake-build-omp-llvm-debug-std-alloc-token-sanitizer` (`CHUNK_SANITIZER=token`), disabled by default; a `--target` that selects it also builds it; `--no-sanitizers` skips it |
| `--target=<glob>` | only the configurations whose name matches (`omp-llvm`, or a quoted shell glob); a pattern that matches nothing is an error |
| `--exec=<glob>` | only the matching Ninja targets in each build (`shrink_bench_omp_zero_256`, or a quoted shell glob: ``'shrink_bench_*_zero_256'`` matches it regardless of the backend) |
| `--jobs=<n>` | number of configurations compiled at the same time (4 by default); the core budget is shared between them (a jobserver with ninja 1.13 or later, see below) |
| `--no-hardening` | without `CHUNK_GLIBCXX_ASSERTIONS` or `CHUNK_UBSAN` on the Debug and sanitizer builds |
| `--llvm-cs=<n>` | add an LLVM build with `CHUNK_SIZE=n` |
| `--pool=<id>` | share the configurations with the other runs given the same `<id>` (one batch job per node, `shrink_gui`'s "Nodes" field): each builds those it claims first; `<id>` is a name under `$CHUNK_BUILD_ROOT/.pools/` or a directory all the nodes share |

Where the compilers come from, beside the modules the script loads: `CHUNK_LLVM_ROOT=<dir>`
for the llvm builds (`<dir>/bin/clang++`, LLVM's release tarball extracted, e.g. where the llvm
module has no OpenMP runtime); on ARM, `CHUNK_ACFL_ROOT=<prefix>` for the acfl family
(`spack location -i acfl`; else `armclang++` in `PATH`, the newest `$CCCWORKDIR/opt/*/acfl-*`,
an acfl module). oneTBB for the std builds' TBB targets: `TBBROOT`, `CHUNK_TBB_DIR` or
`$CCCWORKDIR/tbb-<arch>`; `CHUNK_WITH_TBB=0` leaves them out, `CHUNK_TBB_TSAN_DIR` is the
oneTBB built with TSan of the std TSan build.

`compile_all_gnu.sh` also accepts `--no-ipa-icf` (`CHUNK_NO_IPA_ICF=ON`). The compilation
memory per job is set by `CHUNK_MEM_PER_JOB_MB` (3,072 MB by default, twice that for
sanitizer builds).

The compile jobs of the configurations built at the same time come from one budget, the cores
or what the memory holds. With ninja 1.13 or later (the ninja a build dir is configured with,
its `CMAKE_MAKE_PROGRAM`), the scripts share it through a GNU make jobserver: a configuration
that ends leaves its cores to the ones still running, and the last one finishes on the whole
node. The sanitizer builds take part too: when the memory is what limits the run, their compile
jobs weigh `CHUNK_MEM_PER_JOB_MB_SANITIZER / CHUNK_MEM_PER_JOB_MB` tokens each, and
`scripts/jobserver_broker.sh` gives their ninja a pool of its own, one token of it for that many
of the run's. With an older ninja each configuration keeps the share it was given at its start
(`cores / --jobs`), and the last ones finish on that share alone. The scripts take the newest
ninja of `PATH` that runs on the node, ahead of the module's (a `~/.local/bin/ninja` 1.13 rather
than the module's 1.11), and when it is 1.13 or later they set `CHUNK_NINJA` to it.
`CHUNK_NINJA=<ninja>` configures the build dirs with that ninja (each dir that changes ninja
rebuilds in full once: the two versions do not read each other's `.ninja_log`); set by hand, it
wins (``CHUNK_NINJA=<the module's ninja>`` keeps the old one). `CHUNK_JOBSERVER=off` keeps the
fixed shares.

The std threading backend (`CHUNK_THREADS=std`) has its own builds: `std-gnu`, `std-gnu-check`
and a ThreadSanitizer one from `compile_all_gnu.sh`, `std-llvm`, `std-llvm-check` and a
ThreadSanitizer one from `compile_all.sh`. They lengthen the compilation (``--target='omp-*'``
leaves them out, ``--target='std-*'`` builds only them). Their bench executables say so in their
name, `shrink_bench_std_*` (`shrink_bench_omp_*` in the OpenMP builds, `shrink_bench_serial_*`
in the serial ones): none of them uses OpenMP. The TBB bench (`shrink_bench_tbb_*`) and
`tbb_validation` need oneTBB in those builds: the scripts add `CHUNK_WITH_TBB=ON` when `TBBROOT`
is set (by oneAPI's environment script or `spack load intel-oneapi-tbb`) or when `CHUNK_WITH_TBB=1`, and say
so at the start; `CHUNK_WITH_TBB=0` leaves TBB out. The ThreadSanitizer std builds take TBB
only from `CHUNK_TBB_TSAN_DIR`, the CMake package directory of a oneTBB built with TSan.

## 8. Variables of the tools and of the tests

Read by the Python tools, the check scripts and a few harnesses, not by the library.

| variable | default | effect |
|---|---|---|
| `CHUNK_BUILD_ROOT` | current directory | where the `cmake-build-<target>` directories are: default of `--build-root` in `bench_and_test.py`, `multi_launch_stress.py` and `thread_speedup.py`; `compile_all.sh` takes its builds there, `scripts/site_env.sh` sets it per site. `shrink_gui` exports it in the scripts it submits |
| `CHUNK_BENCH_EVENTS` | unset | file the campaigns append their events to, one JSON record per line (default of `--events-file`); `shrink_gui` sets it and follows the file |
| `CHUNK_POOL_REF_WAIT` | 21,600 | `--pool` campaigns: seconds a job waits for the reference another job generates, before skipping what needs it |
| `CHUNK_POOL_POLL` | 2 | `--pool` campaigns: seconds between two looks at what the other jobs did |
| `CATCH_CRASH_EXE` | unset | default of `--exe` in `python/catch_crash.py` |
| `CLANG_FORMAT` | unset | the clang-format binary `scripts/check_format.py` runs, before `clang-format-23` and `clang-format` on the PATH (version 23 is required). `ctest` sets it to the one CMake found |
| `FK_SECONDS` | 2 | duration, in seconds, of the stress of `zero_or_value_foreign_key_validation` |
| `BLK_HUGE` | unset | set: `blk_allocator_validation` also allocates its blocks past 2 GiB and 4 GiB (skipped otherwise) |
| `CHUNK_VALIDATION_CHILD` | unset | set by a harness on the child processes it runs to watch them fail: the child disables its core dump. Not to be set by hand |
