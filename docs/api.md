# The API at a glance

> French version: [api_fr.md](api_fr.md).

> October 2026. One page for what a program calls, grouped by task. The details (parameters, preconditions, complexity) are in the Doxygen reference
> (`cmake --build build --target doc`), searchable by these names. Usage in context:
> [getting_started.md](getting_started.md).

***

All names are in namespace `shrink_array`, except `build_config` and `chunk_config` (global).

## 1. Starting and stopping the library

| call | header | when |
|---|---|---|
| `ParamChunk::init_indexes(true)` | `chunk/param_chunk.h` | once, before the first array (`true`: large allocation blocks) |
| `release_chunk_pools()` | `chunk/chunk_std.h` | once, at the end, after the last array |
| `destroy_block_chunk()` | `chunk/chunk_std.h` | right after `release_chunk_pools()` |
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
| statistics | `stats()`, `get_nb_chunk_written()`, `print_compression_stats()` | percentage and number of chunks holding a buffer; one line per array |
| geometry | `chunk_size()`, `rechunk_to(n)` **(S)** | `rechunk_to()` only in the `_dyn` libraries (a no-op in static ones) |

Chunk2 only: `set_second_value(v)` **(S)**, before the writes, and `get_second_value()`.

Not there: `data()`, mutable iterators, `insert` / `erase` in the middle, an in-place sort.

## 4. The write proxy

`operator[](i)` on a non-const array returns a proxy, not a `T&`:

| use | effect |
|---|---|
| `T v = a[i];` | reads the value |
| `a[i] = v;` | writes it (may materialise the chunk, or be deferred while a compressor holds it) |
| `a[i] += v;` `-=` `*=` `/=` `%=` `&=` `|=` `^=` `<<=` `>>=` | compound operations, for the types that have them; return a proxy |
| `++a[i]`, `--a[i]` | return a proxy; `a[i]++` returns the old value as a `T` |
| `a[i] == v`, `a[i] != v` | compare the current value; other comparisons go through the conversion to `T` |

`auto x = a[i];` keeps the proxy, which follows the element. A proxy must not outlive the
array's storage: every **(S)** operation invalidates it.

## 5. Range operations

`chunk/array_chunk_range_ops.h`. One decision per chunk instead of one per element.

| call | context |
|---|---|
| `fill_range(a, first, last, v)` | sequential: `[first, last)` set to `v` |
| `copy_range(a, first, last, src_first)` | sequential: `[first, last)` copied from an iterator |
| `fill_range_mt(a, first, last, v)` | callable by several threads at once, on disjoint ranges (a chunk may be shared) |

## 6. OpenMP phases

`chunk/manage_ompt.h`. Needed only when `build_config::has_not_ompt` (GCC's libgomp); no-ops
otherwise, and in serial builds.

| call | where |
|---|---|
| `begin_parallel_phase()` | inside a parallel region, by every thread, before its first access |
| `finalize_parallel_phase()` | after the region, once |
| `finalize_parallel_phase_only_flush()` + `finalize_parallel_phase_after_flush()` | the same, split: the first inside the region (it emits a barrier), the second after it |
| `finalize_barrier_phase()` | inside a region, after an explicit `omp barrier`: makes the writes before it visible after it |

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
