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

### Fixed (cpp23)

- `set_chunk_size(n)` / `try_set_chunk_size(n)` refuse a size wider than
  `cst_buffer_size` (`exceeds_buffer`), as `rechunk_to()` does: they accepted
  it on an empty array, whose chunks then read the shared constant buffers
  out of bounds.

### Added

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
- `multi-mat-md`, the multi-material example in 2D through `mdspan`
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

### Changed

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

### Deprecated

- The trace switches of before the log, replaced by `CHUNK_LOG` levels: `CHUNK_RECHUNK_TRACE`
  (`rechunk=debug`), `CHUNK_RECHUNK_TRACE_SLICES` (`rechunk=trace`), `CHUNK_RECHUNK_MEMTRACE`
  (`memory=info`), `CHUNK_RECHUNK_TRACE_MEM` (`memory=debug`), `CHUNK_AFFINITY_TRACE`
  (`affinity=debug`), `CHUNK_RECYCLE_TRACE` (`recycle=trace`), `FAST_ALLOC_TRACE`
  (`alloc=trace`). Still accepted -- each sets its level, with a warning -- and to be removed in
  0.3.0. They are developer settings ([docs/developer_settings.md](docs/developer_settings.md)),
  outside the versioning policy; the deprecation period is a courtesy to the scripts that set them.

### Fixed

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
- A "wait" line of the re-chunk decider ended with a stray `0` when the array
  had no back-off.
- An invariant failure inside a program's log sink, while the log was reading
  its configuration, blocked instead of stopping the program.

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
  layouts share one implementation (`detail::tiled_mapping_base`); the tile
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
- `chunk_list` (the chunks of an array) no longer has its new chunks value-initialised by
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

