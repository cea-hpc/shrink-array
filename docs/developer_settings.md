# Developer settings

> French version: [reglages_developpeur_fr.md](reglages_developpeur_fr.md).

> Status: October 2026. The switches of the library's developers: tuning of the decider and of
> the allocators, traces, forced rechunks, the benchmark, compilation macros, debugging and build
> options, the build scripts. They are **development tools**: not covered by the versioning policy
> of [CHANGELOG.md](../CHANGELOG.md), they can change or disappear in any release. What a user of
> the library sets is in [environment_variables.md](environment_variables.md).
>
> Unless stated otherwise, a variable is read only once, and `0` or an empty value is the same as
> leaving it unset -- except for the traces of §2 marked *set*, on as soon as they are defined
> (even at `0`), and for the variables whose row says what `0` does. A variable that is not read
> is ignored without any message: when in doubt, check with
> `grep -rn '"NAME"' chunk fast_allocators examples`.
>
> **Removal dates.** A switch that keeps an old behaviour, or a choice no build tests, carries a
> removal date in its row; at that date it goes, with its code path, unless a measurement written
> down here justifies keeping it. A new switch of that kind gets its date when it is added.

***

## 1. Tuning of the decider, the rebuild and the allocators

| variable | default | effect |
|---|---|---|
| `CHUNK_ALLOC_AFFINITY_EAGER` | depends on the variant | `1`: each large array has its own set of allocators from its creation (default for `zero`); `0`: only from its first rechunk (default for `zero_one_value` and `zero_any`). See [adaptive_rechunk.md](adaptive_rechunk.md) §5 |
| `CHUNK_RECHUNK_MAX_COARSEN` | 4 | maximum factor of a coarsening (1 = no limit) |
| `CHUNK_ADVISOR_MIN_GAIN_MB` | 8 | absolute threshold for the data-side gain, in MB (0 removes it) |
| `CHUNK_ADVISOR_GRID_BITS` | 2 | the advisor's candidate sizes between the powers of two: the multiples of 64 whose C / 64 has at most this many significant bits (2: 192, 384, 768...; 3: 320, 448, 640 too; 1: powers of two only). See [adaptive_rechunk.md](adaptive_rechunk.md) §2 |
| `CHUNK_COARSEN_MIN_META_MB` | 1 | minimum metadata gain of a coarsening, in MB |
| `CHUNK_COARSEN_MIN_META_RATE` | 250 | minimum metadata gain per second of estimated rebuild time, in MB/s |
| `CHUNK_RECHUNK_SCHEDULE` | `guided,16` | OpenMP schedule of the rebuild (`kind[,chunk]`) |
| `CHUNK_RECHUNK_SLICES` | 16 | number of slices in the `zero` rebuild |
| `CHUNK_RECLAIM_MIN_SLABS` | 0 | a sweep gives empty slabs back to the system only if there are at least this many |
| `CHUNK_RECLAIM_THREADS` | 1 | threads issuing a sweep's `madvise` calls; `0`: `omp_get_max_threads()` |
| `CHUNK_RECHUNK_RECYCLE` | on | `zero` rebuild: the slabs a slice has just emptied go to the next slice instead of back to the system; `0` turns it off. *Removal planned: 2026-12-31.* |
| `CHUNK_RECHUNK_KEEP_RECYCLED_MB` | 0 | `zero` rebuild: at its end, up to this many MiB per arena of the recycled slabs no slice took stay resident (split evenly over the ranks) for the next allocations, instead of going back to the system; `0`: all go back |
| `CHUNK_RECHUNK_KEEP_RECYCLED` | off | `1`: the same, with no cap (the RSS keeps every leftover until reused) |
| `CHUNK_RECHUNK_REUSE_LIST` | on | a rechunk reuses the previous rechunk's chunk list buffer (pages already resident); `0`: a new one each time. *Removal planned: 2026-12-31.* |
| `CHUNK_THP` | 0 | unset: the kernel's THP mode decides. `never`: no transparent huge pages on the slab arenas (`MADV_NOHUGEPAGE`) -- under the mode `always` a slab otherwise becomes resident whole at its first touch; compare the two with the RSS and AnonHugePages of `smaps_rollup`. Experimental: `1` requests them (`MADV_HUGEPAGE`) on the slab arenas and the cst2 region. Ignored, with a message, when the huge page is larger than 2 MiB (kernel with 64 KiB pages: 512 MiB). Cost: a slab that has been started becomes fully resident. *Removal planned: 2026-12-31.* |
| `CHUNK_REUSE_SHARE_DIV` | 50 | fast allocator: a rank keeps parked for reuse up to chunks / (this × threads) freed buffers per element type (per array for an array with its own allocators); the chunks counted include the compressed ones |
| `CHUNK_REUSE_FLOOR_KB` | unset | floor of that budget in KiB per rank, converted to buffers at the chunk size; unset: 500 buffers (100 for an array with its own allocators), whatever their size; `0`: no floor |
| `CHUNK_RECLAIM_PHASE_KEEP` | unset | `K`: after every compression pass, give the empty slabs back to the system (`madvise`), keeping `K` per allocator instance; unset: only at a re-chunk, a destruction, `shrink_to_fit()` |
| `CHUNK_DEFERRED_POOL_KB` | 64 | slab size of the deferred-write pools, in KiB (the operation slabs are never given back: larger means fewer allocator calls but more resident memory) |

## 2. Traces and diagnostics

The traces are messages of the library's log, chosen by `CHUNK_LOG` (see
[environment_variables.md](environment_variables.md)). The variables of the former traces
still work: each sets the level it stands for, with a deprecation warning.

| setting | what it prints |
|---|---|
| `CHUNK_LOG=rechunk=info` | the decider's moves (`decide <array>: re-chunk S=.. -> S'=.. ... [APPLIED]`, or `[dry-run] would re-chunk`) |
| `CHUNK_LOG=rechunk=debug` (formerly `CHUNK_RECHUNK_TRACE`) | in addition, the verdicts that wait, and for each rechunk the duration of the phases and of the slices |
| `CHUNK_LOG=rechunk=trace` (formerly `CHUNK_RECHUNK_TRACE_SLICES`) | in addition, memory at each slice of the `zero` rebuild |
| `CHUNK_LOG=memory=info` (formerly `CHUNK_RECHUNK_MEMTRACE`) | at each advisor pass: exact memory, including the resident part of the buffer arena |
| `CHUNK_LOG=memory=debug` (formerly `CHUNK_RECHUNK_TRACE_MEM`) | in addition, the memory side of the rechunk trace: exact memory at each step (`rss` line), occupancy of the new geometry, chunks folded, refused and installed directly, allocator census. Costly (seconds per run at tens of GB): a channel apart from the timings |
| `CHUNK_LOG=advisor=info` | check builds: at an array's destruction, the chunk size that would have used the least memory, when it saves at least 10 % |
| `CHUNK_ADVISOR_TABLE` | `1` or `=name`: the full advisor table at each pass (all arrays or a single one) |
| `CHUNK_RECHUNK_CHURN` | `1` or `=name`: materialisations per pass and an estimate of the transient cost they imply (level info of channel rechunk, which it switches on) |
| `CHUNK_LOG=affinity=debug` (formerly `CHUNK_AFFINITY_TRACE`) | each per-array set of allocators created (key, size, budget) |
| `CHUNK_LOG=compress=info` | check builds: the blocks left to compress after the final flush, and `zero_any`'s constant blocks never queued; the compression residual of an audit (`compress()` on zero_one_value::Array / zero_any::Array); `compress=debug` adds zero_one_value::Array's per-chunk detail |
| `CHUNK_ARENA_REPORT` | the summary of the buffer arena (slabs handed out, given back to the system); a report: written whatever `CHUNK_LOG` says (channel `arena`) |
| `CHUNK_LOG=recycle=debug` / `recycle=trace` (formerly `CHUNK_RECYCLE_TRACE`) | check builds: the foreign blocks the deferred-write pool drops, on a doubling schedule / one line per block |
| `CHUNK_LEDGER_REPORT` (*set*) | the report of the block registry (builds with the registry), at exit; a report: written whatever `CHUNK_LOG` says (channel `ledger`) |
| `CHUNK_OMPT_STATS` | statistics of the synchronisation regions (OMPT builds), at exit; a report: written whatever `CHUNK_LOG` says (channel `ompt`) |
| `CHUNK_CST2_PROBE` (*set*) | `zero_any`: report of the second-value registry at each pass (benchmark) |
| `CHUNK_CST2_RECLAIM` | `zero_any`, `N`: every N iterations, frees the second values no chunk uses any more (`reclaim_second_values()`) and prints how many, then each type's identifiers in use against the capacity, free, reclaims and refusals (`second_value_stats()`) (benchmark, any build) |
| `ALLOC_TOKEN_REPORT` | `SANITIZER=token` builds always print, on exit, the allocations per token (the 40 largest); this sets how many: `0` none, `N` the N largest, `all` every token |
| `CHUNK_LOG=alloc=trace` (formerly `FAST_ALLOC_TRACE`) | one `munmap` line per emptied slab that a pool's reserve does not keep and that is freed |

## 3. Forced rechunks (tests)

| variable | effect |
|---|---|
| `CHUNK_FORCE_RECHUNK=<size>` | forces a rechunk to this size, even if the decider would not do it |
| `CHUNK_FORCE_RECHUNK_AT=<iteration>` | delays this forced rechunk until this iteration |
| `CHUNK_FORCE_RECHUNK_TOGGLE=1` | rechunk at each pass, alternating with `CHUNK_FORCE_RECHUNK_ALT` (64 by default) |

## 4. The `shrink_bench` benchmark

| variable or option | effect |
|---|---|
| `CHUNK_ADVISOR_PERIOD` | period of the advisor passes, in iterations (default `max(10, iter_max / 20)`; 0 disables it) |
| `SHRINK_BENCH_ADVISE_ALL` | default: all arrays except `tab_bool` are handed to the decider; `0`: the five historical arrays |
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
| `NO_IO`, `COMPRESS` | equivalents of `--no_io` and `--compress` |
| `--no_mask`, `--mask_scale <x>` | without `tab_mask`; or its size, `x × sz` (default 1/32) |
| `--no_conn` | without `tab_conn`, the mesh connectivity (`pattern::Array` in the `zero_one_value` and `zero_any` builds) |
| `--no_struct` | without the arrays of structures (`Point3`, `Cell`) |

## 5. Compilation macros

Tuning values the headers give a default, and test hooks. The variant and geometry macros,
and how the shared options reach the headers, are in
[environment_variables.md](environment_variables.md) §3.

| macro | effect |
|---|---|
| `CHUNK_DYN_MAX_SIZE=<n>` | dynamic geometry: largest chunk size a rechunk can reach, a power of two (4096); sizes the shared buffers of the constant values, `cst_buffer_size × sizeof(T)` per distinct second value |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<bytes>` | metadata threshold above which an array can have its own allocators (16 MiB; 0 = never) |
| `CHUNK_COMPRESS_MAX_YIELDS=<n>` | wait of a parked compressor before giving up its pass (10,000 *yields*; 0: give up at the first writer seen). See [synchronisation_dekker.md](synchronisation_dekker.md) |
| `CHUNK_WRITE_WAIT_MAX_YIELDS=<n>` | wait of a writer facing a compressor before deferring its write (64 *yields*) |
| `CHUNK_READ_RECLAIM_SLOTS=<n>` | number of reader threads tracked individually by the buffer reclamation (1,024) |
| `CHUNK_STATS_MAX_THR=<n>` | number of ranks of the `CHUNK_STATS` counters (1,024) |
| `CHUNK_TEST_READ_DELAY_EVERY=<n>`, `CHUNK_TEST_READ_DELAY_US=<µs>` | test hook: one read in N by a thread that requested it waits (50 µs), to widen the reader / reclamation window (`read_reclaim_validation`) |
| `CHUNK_TEST_PASS_DELAY_EVERY=<n>`, `CHUNK_TEST_PASS_DELAY_US=<µs>` | test hook: a delay (200 µs) in one compression pass in N |

And one macro specific to `zero_any`: `CHUNK2A_COMPRESS_LEVEL=<0|1|2|3>` (0 by default) selects the
set of trigger thresholds for the compression passes (`zero_any_chunk_def.h`). Only level 0 is built and
tested. *Removal of levels 1 to 3 planned: 2026-12-31.*

## 6. CMake options

The options of [environment_variables.md](environment_variables.md) §2 are not repeated.

| option | default | effect |
|---|---|---|
| `SHRINK_BENCH_DYN_SIZE` | 64 | starting size of the `_dyn` targets |
| `SHRINK_BENCH_INT64` | `OFF` | targets with 64-bit indices (`_int64`) |
| `SHRINK_INTEL_LIBM` | `ON` | x86-64: link Intel's libm if it is found |
| `SHRINK_ARM_LIBM` | `OFF` | aarch64: link ArmPL's `libamath` (no gain measured on Grace) |
| `CHUNK_ICPX_O0_WORKAROUND` | `OFF` | workaround for an icpx internal error on three files |
| `CHUNK_MM_PROTECT` | `OFF` | `zero`, `zero_one_value`: shared buffers of the constant values made read-only (`mprotect`); a stray write into a compressed chunk causes a segmentation fault instead of corrupting the value (debugging build) |
| `CHUNK_LIB_UNITY` | `ON` | each variant archive (`chunk_zero_any_256`, ...) compiled as a single unit (unity build, 4 to 6 times less compilation); a file that has its own compile options stays separate |
| `SHRINK_BENCH_PRINT_MEM` | `ON` | `shrink_bench` prints its memory lines (after init, every freq_conso_mem iterations, after a compression); `OFF` to time a run without the RSS reads behind them |
| `SHRINK_BENCH_UNITY` | `ON` | each `shrink_bench` executable compiled as a single unit (unity build, 3.5 to 4 times less compilation); `OFF` to compare timings with earlier campaigns, since inlining across benchmark files can change them; `SHRINK_BENCH_UNITY_BATCH` files per unit (0 = all) |
| `CHUNK_DEBUG_INFO` | `auto` | debug information of the optimised builds: `full` (`-g`), `lines` (`-g1`: function names and line tables, enough for perf and gdb backtraces, not the locals), `none`; `auto`: `full` with `CHECK_OPTIM` or a sanitizer, `lines` otherwise. The machine code is the same; `lines` compiles 20 to 27 % faster with 40 % less peak memory. Debug builds always get `-g` |
| `CHUNK_VALIDATION_UNITY` | `ON` | the validation harnesses grouped into one executable per variant and geometry (unity build); `OFF`: one executable per harness |
| `CHUNK_BUILD_VALIDATION` | depends on the build | compile the harnesses of `examples/validation`, and with them the test hooks (`CHUNK_TEST_HOOKS`); by default `ON` in Debug, with `CHECK_OPTIM` or a sanitizer, `OFF` elsewhere (not cached: follows the current configuration) |
| `CHUNK_RECLAIM_EMPTY_SLABS` | `ON` | give emptied slabs back to the system (`madvise`) during explicit sweeps (rechunk, destruction); `OFF` keeps them resident. *Removal planned: 2026-12-31.* |
| `FAST_ALLOC_CHECK_LISTS` | `OFF` | stronger debug level for the slab lists: in Debug, full check of the list at each insertion and removal (slow); an ordinary Debug build keeps the light checks |
| `HAVE_AGGRESSIVE_COMPRESSION` | `OFF` | default of the "aggressive" compression, a run-time switch: `CHUNK_AGGRESSIVE_COMPRESSION=1` (or `0`) sets it for a run. The pass between two phases is skipped, and all compression happens during the parallel regions, at unpredictable times. Slower, but it exercises the concurrent paths: a validation mode, which no longer needs a build of its own |
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
| `USE_LIBCXX` | `ON` | Clang builds: libc++ instead of libstdc++ |
| `USE_GPERF`, `ENABLE_IWYU` | `OFF` | gperftools profiling; include-what-you-use during compilation |
| `ONEAPI_XHOST`, `ONEAPI_NO_INTEL_LIBS`, `ONEAPI_FAST_FP` | `OFF` | icpx: Intel-only binary (`-xHost`); no Intel runtime library; keep `-fp-model=fast` |

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
| `--only=<c>`, `--no-<c>` | a single compiler, or without it (`gnu`, `oneapi`, `nvhpc`, `llvm`, `aocc`) |
| `--with-nvhpc` | add NVHPC, disabled by default (about 1 h 30 per build on x86, with no gain) |
| `--no-sanitizers` | without the LLVM sanitizer builds |
| `--with-ledger` | add `cmake-build-omp-gnu-debug-ledger` (OpenMP Debug with `CHUNK_FORCE_LEDGER=ON`), disabled by default; a `--target` that selects it also builds it |
| `--with-token-sanitizer` | add `cmake-build-omp-llvm-debug-std-alloc-token-sanitizer` (`SANITIZER=token`), disabled by default; a `--target` that selects it also builds it; `--no-sanitizers` skips it |
| `--target=<glob>` | only the configurations whose name matches (`omp-llvm`, or a quoted shell glob); a pattern that matches nothing is an error |
| `--exec=<glob>` | only the matching Ninja targets in each build (`shrink_bench_zero_64`, or a quoted shell glob) |
| `--jobs=<n>` | number of configurations compiled at the same time (4 by default); the core budget is shared between them |
| `--no-hardening` | without `CHUNK_GLIBCXX_ASSERTIONS` or `CHUNK_UBSAN` on the Debug and sanitizer builds |
| `--llvm-cs=<n>` | add an LLVM build with `CHUNK_SIZE=n` |

`compile_all_gnu.sh` also accepts `--no-ipa-icf` (`CHUNK_NO_IPA_ICF=ON`). The compilation
memory per job is set by `CHUNK_MEM_PER_JOB_MB` (3,072 MB by default, twice that for
sanitizer builds).

The std threading backend (`CHUNK_THREADS=std`) has its own builds: `std-gnu`, `std-gnu-check`
and a ThreadSanitizer one from `compile_all_gnu.sh`, `std-llvm`, `std-llvm-check` and a
ThreadSanitizer one from `compile_all.sh`. They lengthen the compilation (`--target='omp-*'`
leaves them out, `--target='std-*'` makes them alone). The TBB bench (`shrink_bench_tbb_*`) and
`tbb_validation` need oneTBB in those builds: the scripts add `CHUNK_WITH_TBB=ON` when `TBBROOT`
is set (by oneAPI's environment script or `spack load intel-oneapi-tbb`) or when `CHUNK_WITH_TBB=1`, and say
so at the start; `CHUNK_WITH_TBB=0` leaves TBB out. The ThreadSanitizer std builds take TBB
only from `CHUNK_TBB_TSAN_DIR`, the CMake package directory of a oneTBB built with TSan.
