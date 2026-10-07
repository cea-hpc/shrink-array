# Environment variables, macros and build options

> French version: [variables_environnement_fr.md](variables_environnement_fr.md).

> Status: early October 2026. The list was collected from the code; a variable that is not
> read is ignored without any message, so when in doubt, check with
> `grep -rn '"NAME"' chunk fast_allocators examples`. Unless stated otherwise, a variable is
> read only once, and `0` or an empty value is the same as leaving it unset.

***

## 1. Library settings

| variable | default | effect |
|---|---|---|
| `CHUNK_ALLOC_AFFINITY_EAGER` | depends on the variant | `1`: each large array has its own set of allocators from its creation (default for Chunk, v1); `0`: only from its first rechunk (default for Chunk2 and Chunk2A). See [adaptive_rechunk.md](adaptive_rechunk.md) §5 |
| `CHUNK_RECHUNK_MAX_COARSEN` | 4 | maximum factor of a coarsening (1 = no limit) |
| `CHUNK_ADVISOR_MIN_GAIN_MB` | 8 | absolute threshold for the data-side gain, in MB (0 removes it) |
| `CHUNK_COARSEN_MIN_META_MB` | 1 | minimum metadata gain of a coarsening, in MB |
| `CHUNK_COARSEN_MIN_META_RATE` | 250 | minimum metadata gain per second of estimated rebuild time, in MB/s |
| `CHUNK_RECHUNK_DRY_RUN` | — | `1`: the decider analyses and logs, but applies no rechunk |
| `CHUNK_RECHUNK_SCHEDULE` | `guided,16` | OpenMP schedule of the rebuild (`kind[,chunk]`) |
| `CHUNK_RECHUNK_SLICES` | 16 | number of slices in the v1 rebuild |
| `CHUNK_RECLAIM_MIN_SLABS` | 0 | a sweep gives empty slabs back to the system only if there are at least this many |
| `CHUNK_THP` | 0 | unset: the kernel's THP mode decides. `never`: no transparent huge pages on the slab arenas (`MADV_NOHUGEPAGE`) -- under the mode `always` a slab otherwise becomes resident whole at its first touch; compare the two with the RSS and AnonHugePages of `smaps_rollup`. Experimental: `1` requests them (`MADV_HUGEPAGE`) on the slab arenas and the cst2 region. Ignored, with a message, when the huge page is larger than 2 MiB (kernel with 64 KiB pages: 512 MiB). Cost: a slab that has been started becomes fully resident |
| `CHUNK_THREAD_CAPACITY` | number of cores | number of ranks the per-thread structures are sized for (increase it in case of oversubscription) |
| `CHUNK_DEFERRED_POOL_KB` | 64 | slab size of the deferred-write pools, in KiB (the operation slabs are never given back: larger means fewer allocator calls but more resident memory) |
| `CHUNK_REQUIRE_OMPT` | — | OMPT builds: `1` aborts the program if the OMPT tool was not loaded by the runtime (otherwise, a message on stderr), since the end-of-phase events would then never happen |

## 2. Comparison switches

There are none left: the ones that restored an old behaviour were
removed once the new one was validated. The settings useful to compare two runs are
in section 1 and in [performance_measurement.md](performance_measurement.md) §2.

## 3. Traces and diagnostics

| variable | what it prints |
|---|---|
| `CHUNK_RECHUNK_TRACE` | for each rechunk: duration of the steps, fallbacks and direct installations, memory at each step (`[RECHUNK-RSS]`) |
| `CHUNK_RECHUNK_TRACE_SLICES` | memory at each slice of the v1 rebuild |
| `CHUNK_RECHUNK_MEMTRACE` | at each advisor pass: exact memory, including the resident part of the buffer arena |
| `CHUNK_ADVISOR_TABLE` | `1` or `=nom`: the full advisor table at each pass (all arrays or a single one) |
| `CHUNK_RECHUNK_CHURN` | `1` or `=nom`: materialisations per pass and an estimate of the transient cost they imply |
| `CHUNK_AFFINITY_TRACE` | each per-array set of allocators created (key, size, budget) |
| `CHUNK_ARENA_REPORT` | the summary of the buffer arena (slabs handed out, given back to the system) |
| `CHUNK_RECYCLE_TRACE` | one line per recycled block of the deferred-write pool (instead of a summary) |
| `CHUNK_LEDGER_REPORT` | the report of the block registry (builds with the registry) |
| `CHUNK_OMPT_STATS` | statistics of the synchronisation regions (OMPT builds) |
| `CHUNK_CST2_PROBE` | Chunk2A: report of the second-value registry at each pass (benchmark) |
| `ALLOC_TOKEN_REPORT` | on exit, the count of allocations per token |
| `FAST_ALLOC_TRACE` | one `[MUNMAP]` line per emptied slab that a pool's reserve does not keep and that is freed (silent by default) |

## 4. Forced rechunks (tests)

| variable | effect |
|---|---|
| `CHUNK_FORCE_RECHUNK=<size>` | forces a rechunk to this size, even if the decider would not do it |
| `CHUNK_FORCE_RECHUNK_AT=<iteration>` | delays this forced rechunk until this iteration |
| `CHUNK_FORCE_RECHUNK_TOGGLE=1` | rechunk at each pass, alternating with `CHUNK_FORCE_RECHUNK_ALT` (64 by default) |

## 5. The `shrink_bench` benchmark

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
| `BENCH_SCHED_BLOCK` | size of the scheduling blocks of the benchmark phases (64; 0 = contiguous slices) |
| `NO_IO`, `COMPRESS` | equivalents of `--no_io` and `--compress` |
| `--no_mask`, `--mask_scale <x>` | without `tab_mask`; or its size, `x × sz` (default 1/32) |
| `--no_struct` | without the arrays of structures (`Point3`, `Cell`) |

## 6. Compilation macros

The options shared by every library of a build (§7) are not macros: CMake records them in the
generated `chunk/build_options.h` and `fast_allocators/build_options.h`, installed with the
libraries, and the headers read them there, whatever a program defines itself. A file compiled
with OpenMP against a library built without it, or the reverse, does not compile. The macros
below are those a library target carries (the variant, the geometry) and the tuning values the
headers give a default; a program compiles with the library's. A program spelling the variant or
the geometry differently from the library it links does not link: the undefined symbol
(`shrink_array::link_check::variant_...`) names the settings it was compiled with.

| macro | effect |
|---|---|
| `CHUNK_VARIANT_V1=1`, `CHUNK_VARIANT_V2=1`, `CHUNK_VARIANT_V2A=1` | the variant |
| `CHUNK_DYNAMIC`, `DYN_CHUNK_SIZE=<n>` | dynamic geometry, and starting size |
| `CHUNK_DYN_MAX_SIZE=<n>` | dynamic geometry: largest chunk size a rechunk can reach, a power of two (4096); sizes the shared buffers of the constant values, `cst_buffer_size × sizeof(T)` per distinct second value |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<bytes>` | metadata threshold above which an array can have its own allocators (16 MiB; 0 = never) |
| `CHUNK_COMPRESS_MAX_YIELDS=<n>` | wait of a parked compressor before giving up its pass (10,000 *yields*; 0: give up at the first writer seen). See [synchronisation_dekker.md](synchronisation_dekker.md) |
| `CHUNK_WRITE_WAIT_MAX_YIELDS=<n>` | wait of a writer facing a compressor before deferring its write (64 *yields*) |
| `CHUNK_READ_RECLAIM_SLOTS=<n>` | number of reader threads tracked individually by the buffer reclamation (1,024) |
| `CHUNK_STATS_MAX_THR=<n>` | number of ranks of the `CHUNK_STATS` counters (1,024) |
| `CHUNK_TEST_READ_DELAY_EVERY=<n>`, `CHUNK_TEST_READ_DELAY_US=<µs>` | test hook: one read in N by a thread that requested it waits (50 µs), to widen the reader / reclamation window (`read_reclaim_validation`) |
| `CHUNK_TEST_PASS_DELAY_EVERY=<n>`, `CHUNK_TEST_PASS_DELAY_US=<µs>` | test hook: a delay (200 µs) in one compression pass in N |

## 7. CMake options

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | static sizes built (targets `_<size>`) |
| `SHRINK_BENCH_DYN_SIZE` | 64 | starting size of the `_dyn` targets |
| `SHRINK_BENCH_INT64` | `OFF` | targets with 64-bit indices (`_int64`) |
| `SHRINK_INTEL_LIBM` | `ON` | x86-64: link Intel's libm if it is found |
| `SHRINK_ARM_LIBM` | `OFF` | aarch64: link ArmPL's `libamath` (no gain measured on Grace) |
| `CHUNK_ICPX_O0_WORKAROUND` | `OFF` | workaround for an icpx internal error on three files |
| `CHUNK_MM_PROTECT` | `OFF` | Chunk, Chunk2: shared buffers of the constant values made read-only (`mprotect`); a stray write into a compressed chunk causes a segmentation fault instead of corrupting the value (debugging build; `compile_all.sh` builds it so that its branches keep compiling) |
| `CHUNK_LIB_UNITY` | `ON` | each variant archive (`chunk_v2a_256`, ...) compiled as a single unit (unity build, 4 to 6 times less compilation); a file that has its own compile options stays separate |
| `SHRINK_BENCH_UNITY` | `ON` | each `shrink_bench` executable compiled as a single unit (unity build, 3.5 to 4 times less compilation); `OFF` to compare timings with earlier campaigns, since inlining across benchmark files can change them; `SHRINK_BENCH_UNITY_BATCH` files per unit (0 = all) |
| `CHUNK_VALIDATION_UNITY` | `ON` | the validation harnesses grouped into one executable per variant and geometry (unity build); `OFF`: one executable per harness |
| `CHUNK_BUILD_VALIDATION` | depends on the build | compile the harnesses of `examples/validation`, and with them the test hooks (`CHUNK_TEST_HOOKS`); by default `ON` in Debug, with `CHECK_OPTIM` or a sanitizer, `OFF` elsewhere (not cached: follows the current configuration) |
| `CHUNK_RECLAIM_EMPTY_SLABS` | `ON` | give emptied slabs back to the system (`madvise`) during explicit sweeps (rechunk, destruction); `OFF` keeps them resident |
| `FAST_ALLOC_CHECK_LISTS` | `OFF` | stronger debug level for the slab lists: in Debug, full check of the list at each insertion and removal (slow); an ordinary Debug build keeps the light checks |
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | Chunk2A: the second-value slot holds a 4-byte identifier instead of the value (less memory, +3% write time) |
| `CHUNK_STD_ALLOC` | `OFF` | standard allocator for chunk buffers instead of `fast_allocators` (useful under ASan: each buffer is a real `malloc`) |
| `CHUNK_BUILD_DYN` | `ON` | also build a dynamic-geometry library (`_dyn`) per variant |
| `CHUNK_STATS` | `OFF` | instrumentation counters and the `print_chunk_stats()` report |
| `CHECK_OPTIM` | `OFF` | invariant checks in an optimised build |
| `HAVE_AGGRESSIVE_COMPRESSION` | `OFF` | "aggressive" compression: the pass between two phases is skipped, and all compression happens during the parallel regions, at unpredictable times. Slower, but it exercises the concurrent paths (the old spelling `HAVE_AGRESSIVE_COMPRESSION` is still accepted, with a warning) |
| `CHUNK_FORCE_LEDGER` | `OFF` | keeps the block registry (`pool_ledger.h`) active under OpenMP, where it is off by default: its global mutex serialises every allocation and free. Detects double frees and illegal returns to the reserve |
| `CHUNK_DEFERRED_DRAIN_THRESHOLD` | empty (4,096) | threshold at which a rank flushes its own queue |
| `CHUNK_FORCE_DEFER_EVERY` | 0 | one write in N through the deferred path, across the whole tree (validation; 32 threads at most) |
| `CHUNK_DEBUG` | depends on the build | the library's debug checks; `ON` with `CMAKE_BUILD_TYPE=Debug` (not cached: follows the current configuration) |
| `CHUNK_POOL_NO_REUSE` | `OFF` | deferred-op pools without free list: every block a real `new` / `delete` (ASan, TSan) |
| `CHUNK_PERF`, `CHUNK_DO_CHECK` | `OFF` | Chunk2 skips its diagnostic scan of missed compressions; every written value checked for NaN |
| `CHUNK_DEBUG_SET_SIZE`, `CHUNK_RECHUNK_LEDGERTRACE` | `OFF` | print the compression thresholds and the lists' memory estimate; trace every block-registry transition |
| `CHUNK_RING_LOG_DEBUG`, `CHUNK_RING_LOG_OP` | `OFF` | debug ring log of every event (heavy); of the op-lifecycle events only (light) |
| `SANITIZER` | `none` | `thread`, `address` or `token` (AllocToken, Clang ≥ 22), for all build types |
| `CHUNK_GLIBCXX_ASSERTIONS` | `OFF` | libstdc++ assertions (`_GLIBCXX_ASSERTIONS`) in the library and its consumers |
| `CHUNK_UBSAN` | `OFF` | UndefinedBehaviorSanitizer (to pair with ASan, not with TSan) |
| `CHUNK_ALIGN_CHECK` | `OFF` | abort if a slab block is misaligned for its type |
| `CHUNK_NO_IPA_ICF` | `OFF` | disables `-fipa-icf` (otherwise GCC merges identical functions, which scrambles call stacks) |
| `CHUNK_WARNINGS`, `CHUNK_WERROR` | `ON`, `OFF` | `-Wall -Wextra` on all builds; warnings as errors in Debug |
| `CHUNK_HEAVY_JOB_POOL` | empty | Ninja job pool for the executables compiled as a single unit (limits compilation memory) |
| `USE_LIBCXX` | `ON` | Clang builds: libc++ instead of libstdc++ |
| `USE_GPERF`, `ENABLE_IWYU` | `OFF` | gperftools profiling; include-what-you-use during compilation |
| `ONEAPI_XHOST`, `ONEAPI_NO_INTEL_LIBS`, `ONEAPI_FAST_FP` | `OFF` | icpx: Intel-only binary (`-xHost`); no Intel runtime library; keep `-fp-model=fast` |

And one macro specific to Chunk2A: `CHUNK2A_COMPRESS_LEVEL=<0|1|2|3>` (0 by default) selects the
set of trigger thresholds for the compression passes (`chunk2A_def.h`).

For clang builds on x86-64, `CHUNK_LIBMVEC_LINKS` (in the cache, computed at
configure time) tells whether the glibc vector library provides the math
functions of the benchmarks; otherwise, their loops stay scalar (glibc < 2.35).

## 8. The build scripts

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
| `--target=<glob>` | only the configurations whose name matches (`omp-llvm`, or a quoted shell glob); a pattern that matches nothing is an error |
| `--exec=<glob>` | only the matching Ninja targets in each build (`shrink_bench_chunk_64`, or a quoted shell glob) |
| `--jobs=<n>` | number of configurations compiled at the same time (4 by default); the core budget is shared between them |
| `--no-hardening` | without `CHUNK_GLIBCXX_ASSERTIONS` or `CHUNK_UBSAN` on the Debug and sanitizer builds |
| `--llvm-cs=<n>` | add an LLVM build with `CHUNK_SIZE=n` |

`compile_all_gnu.sh` also accepts `--no-ipa-icf` (`CHUNK_NO_IPA_ICF=ON`). The compilation
memory per job is set by `CHUNK_MEM_PER_JOB_MB` (3,072 MB by default, twice that for
sanitizer builds).
