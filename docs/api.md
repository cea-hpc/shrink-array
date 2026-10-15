# The API at a glance

> French version: [api_fr.md](api_fr.md).

> October 2026. One page for what a program calls, grouped by task. The details (parameters, preconditions, complexity) are in the Doxygen reference
> (`cmake --build build --target doc`), searchable by these names. Usage in context:
> [getting_started.md](getting_started.md).

***

All names are in namespace `shrink_array`, `build_config` and `chunk_config` included; the allocator library's are in `fast_alloc`. The macros start with CHUNK (CHUNK_, CHUNK2A_) or FAST_ALLOC_.

## 1. Starting and stopping the library

| call | header | when |
|---|---|---|
| `ParamChunk::init_indexes(true)` | `chunk/param_chunk.h` | optional, once, before the first array, to choose the blocks (`true`: large allocation blocks); without it the first array runs `init_indexes(true)` |
| `release_chunk_pools()` | `chunk/chunk_std.h` | optional, once, at the end; while an array is alive (a global or `static` one) it only empties the free lists |
| `destroy_block_chunk()` | `chunk/chunk_std.h` | optional, right after `release_chunk_pools()`; does nothing while an array is alive |
| `ParamChunk::force_compression()` | `chunk/param_chunk.h` | to compress every pending candidate now, for a measurement or a checkpoint |
| `ParamChunk::ChunkStats()` | `chunk/param_chunk.h` | prints the `CHUNK_STATS` counters (no-op in other builds) |

## 2. The containers

| class | header | compresses chunks uniform at… |
|---|---|---|
| `ArrayChunk<T, ind_type = int>` | `chunk/array_chunk.h` | `T{}` |
| `ArrayChunk2<T, ind_type = int>` | `chunk/array_chunk2.h` | `T{}` or the array's second value |
| `ArrayChunk2A<T, ind_type = int>` | `chunk/array_chunk2A.h` | `T{}` or any value, per chunk |
| `ArrayChunkBits` | `chunk/array_chunk_bits.h` | `bool`, stored one bit per element (Chunk2 builds) |

`ind_type` is the index type: `long` for more than 2³¹ elements. The variant compiled in is the
one of the library linked (`chunk::chunk_v2a`, ...); `chunk_config::use_chunk`,
`chunk_config::use_chunk2` and `chunk_config::use_chunk2a` tell which, at compile time.

`chunk/mdspan.h` puts a multidimensional view, `f[i, j]`, on any of them (C++23 `mdspan`,
row-major or in tiles aligned on the chunks), and `chunk/mdspan_tile_ops.h` reads and updates such
a view tile by tile, a uniform tile in one call, and fills a box of it by runs: see
[mdspan.md](mdspan.md).

## 3. Container operations

What a `std::vector` has, with the differences that matter. **(S)** marks a sequential point:
outside any parallel region, and it invalidates the array's proxies.

| operation | members | notes |
|---|---|---|
| construct | `(n)`, `(first, last)`, `{a, b, c}`, copy (deep), move | a new array reads `T{}` and allocates nothing |
| size | `size()`, `empty()`, `capacity()` | |
| read | `operator[](i) const`, `at(i) const`, `front()`, `back()`, `operator()(i) const` | return a `T` |
| write | `operator[](i)`, `at(i)` | return a write proxy (§4) |
| raw reference | `operator()(i)` (non-const) | a `T&` into the buffer: **only on a materialised chunk**, bypasses the library |
| grow, shrink **(S)** | `resize(n)`, `reserve(n)`, `clear()`, `shrink_to_fit()` | `shrink_to_fit()` also folds every chunk that became constant |
| append | `push_back(v)`, `emplace_back(...)`, `pop_back()` | **(S)** only when it needs a new chunk |
| replace **(S)** | `assign(n, v)`, `assign(first, last)`, `swap(other)`, `operator=` | |
| iterate | `begin()`, `end()`, `cbegin()`, `rbegin()`, `rend()` | const random-access iterators, by value |
| compress **(S)** | `compress()` | a full compression pass now (`ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A`) |
| free second values **(S)** | `reclaim_second_values()` (static) | frees the second values no chunk of any `ArrayChunk2A<T>` uses any more, so that their identifiers are reused; returns how many (`ArrayChunk2A`) |
| statistics | `stats()`, `get_nb_chunk_written()` | percentage and number of chunks holding a buffer (each array also prints one line when it is destroyed) |
| geometry | `chunk_size()`, `rechunk_to(n)` **(S)** | `rechunk_to()` only in the `_dyn` libraries (a no-op in static ones) |

Chunk2 only: `set_second_value(v)` **(S)**, before the writes, and `get_second_value()`.

Not there: `data()`, mutable iterators, `insert` / `erase` in the middle, an in-place sort.

## 4. The write proxy

`operator[](i)` on a non-const array returns a proxy, not a `T&`:

| use | effect |
|---|---|
| `T v = a[i];` | reads the value |
| `a[i] = v;` | writes it (may materialise the chunk, or be deferred while a compressor holds it) |
| `a[i] = b[j];` | copies the value: reads `b[j]`, then writes `a[i]` as above |
| `a[i] += v;` `-=` `*=` `/=` `%=` `&=` `|=` `^=` `<<=` `>>=` | compound operations, for the types that have them; return a proxy |
| `++a[i]`, `--a[i]` | return a proxy; `a[i]++` returns the old value as a `T` |
| `a[i] == v`, `a[i] != v` | compare the current value; other comparisons go through the conversion to `T` |
| `std::printf("%f", a[i])` | refused: `...` would get the proxy, not the value (GCC: `-Werror=conditionally-supported`, set by the library; clang: always). Write `double(a[i])` |

`auto x = a[i];` keeps the proxy, which follows the element. A proxy must not outlive the
array's storage: every **(S)** operation invalidates it. In every build,
`operator[]`, `at()`, `front()` and `back()` return the proxy wrapped in an `ElementRef`
(`internal/element_ref.h`) that only a temporary may use: `auto x = a[i];` then any use of `x`
does not compile, with a message. A proxy kept on purpose: `std::move(x) = v;`, or
`a.element_proxy(i)`. The wrapper costs nothing at run time (force-inlined).
`-DCHUNK_LOOSE_ELEMENT_REF` turns the check off.

## 5. Range operations

`chunk/array_chunk_range_ops.h`. One decision per chunk instead of one per element.

| call | context |
|---|---|
| `fill_range(a, first, last, v)` | sequential: `[first, last)` set to `v` |
| `copy_range(a, first, last, src_first)` | sequential: `[first, last)` copied from an iterator |
| `fill_range_mt(a, first, last, v)` | callable by several threads at once, on disjoint ranges (a chunk may be shared) |
| `copy_range_mt(a, first, last, src_first)` | `copy_range`, callable as `fill_range_mt` is |
| `transform_range_mt(a, first, last, src_first, op)` | element `i` gets `op(*(src_first + (i - first)))`, callable as `fill_range_mt` is; `op` must not touch `a` |
| `for_each_value(a, first, last, f)` | read: `f(v)` on each element, one read section per block |
| `for_each_span(a, first, last, f)` | read: `f(std::span<const T>)` per block (a vectorisable loop); a compressed chunk as `f(v, count)` when `f` accepts it. A `std::vector` is one span |
| `read_range(a, first, last, out)` | read: `[first, last)` copied to `out` |

## 6. OpenMP phases

`chunk/manage_ompt.h`. Needed only when `build_config::has_not_ompt` (GCC's libgomp); no-ops
otherwise, and in serial builds.

| call | where |
|---|---|
| `begin_parallel_phase()` | inside a parallel region, by every thread, before its first access |
| `finalize_parallel_phase()` | after the region, once |
| `finalize_parallel_phase_only_flush()` + `finalize_parallel_phase_after_flush()` | the same, split: the first inside the region (it emits a barrier), the second after it |
| `finalize_barrier_phase()` | inside a region, after an explicit `omp barrier`: makes the writes before it visible after it |
| `parallel_phase([n,] body)` | instead of the first two: a parallel region (`n` threads, or the runtime's) running `body` on every thread, with `begin_parallel_phase()` before it and `finalize_parallel_phase()` after the region; the worksharing (`omp for`) goes in `body` |

A `begin_parallel_phase()` with no `finalize_parallel_phase()` since the previous region stops
the program, in every build. Check builds (Debug, `CHECK_OPTIM`) also stop on a deferred write
in a region by a thread that did not call `begin_parallel_phase()`.

## 7. Adaptive chunk size (`_dyn` libraries)

| call | header | what it does |
|---|---|---|
| `ChunkSizeAdvisor<T>::analyze(a)` | `chunk/chunk_size_advisor.h` | estimates the array's memory at each candidate size; read-only, at a quiescent point |
| `RechunkDecider` with `RechunkPolicy` | `chunk/rechunk_decider.h` | decides, tick after tick, whether a re-chunk is worth it |
| `decider.consider<T>(a, tick, iter, "name")` | | analyses the array and applies the re-chunk if the policy agrees **(S)** |
| `decider.print_rechunk_log()` | | the table of the re-chunks performed |
| `a.rechunk_to(n)` **(S)** | | re-chunk one array to `n` elements per chunk |

The policy, its defaults and the measured results: [adaptive_rechunk.md](adaptive_rechunk.md).

## 8. Compile-time configuration

| name | meaning |
|---|---|
| `build_config::use_openmp` | built with OpenMP |
| `build_config::has_ompt`, `build_config::has_not_ompt` | built with the OMPT tool, or not |
| `build_config::chunk_dynamic` | dynamic geometry (`_dyn` library) |
| `chunk_config::use_chunk`, `chunk_config::use_chunk2`, `chunk_config::use_chunk2a` | the variant |
| `chunk_config::stats` | `CHUNK_STATS` build |

Environment variables and build options: [environment_variables.md](environment_variables.md).
