# The API at a glance

> French version: [api_fr.md](api_fr.md).

> November 2026. One page for what a program calls, grouped by task. The details (parameters, preconditions, complexity) are in the Doxygen reference
> (`cmake --build build --target doc`), searchable by these names. Usage in context:
> [getting_started.md](getting_started.md).

***

All names are in namespace `shrink_array`, `build_config` and `chunk_config` included; the allocator library's are in `fast_alloc`. The macros start with CHUNK_ or FAST_ALLOC_.

## 1. Starting and stopping the library

| call | header | when |
|---|---|---|
| `Runtime::init(true)` | `chunk/runtime.h` | optional, once, before the first array, to choose the blocks (`true`: large allocation blocks); without it the first array runs `init(true)` |
| `release_chunk_pools()` | `chunk/runtime.h` | optional, once, at the end; while an array is alive (a global or `static` one) it only empties the free lists |
| `Runtime::shutdown()` | `chunk/runtime.h` | optional, right after `release_chunk_pools()`; does nothing while an array is alive |
| `Runtime::force_compression()` | `chunk/runtime.h` | to compress every pending candidate now, for a measurement or a checkpoint |
| `Runtime::chunk_stats()` | `chunk/runtime.h` | prints the `CHUNK_STATS` counters (no-op in other builds) |

## 2. The containers

| class | header | compresses chunks uniform at… |
|---|---|---|
| `zero::Array<T, IntType = int>` | `chunk/zero_array.h` | `T{}` |
| `zero_or_value::Array<T, IntType = int>` | `chunk/zero_or_value_array.h` | `T{}` or the array's second value |
| `zero_any::Array<T, IntType = int>` | `chunk/zero_any_array.h` | `T{}` or any value, per chunk |
| `BitArray` | `chunk/bit_array.h` | `bool`, stored one bit per element (`zero_or_value` builds) |

`IntType` is the index type: `long` for more than 2³¹ elements. The variant compiled in is
that of the linked library (`shrink_array::zero_any`, ...); `chunk_config::use_zero`,
`chunk_config::use_zero_or_value` and `chunk_config::use_zero_any` tell which, at compile time.

## 3. Container operations

What a `std::vector` has, with the differences that matter. **(S)** marks a sequential point:
outside any parallel region, or inside one from a single thread while the others leave the arrays
alone ([getting started, §5](getting_started.md)); it invalidates the array's proxies.

| operation | members | notes |
|---|---|---|
| construct | `(n)`, `(first, last)`, `{a, b, c}`, copy (deep), move | a new array reads `T{}` and allocates nothing |
| size | `size()`, `empty()`, `capacity()` | |
| read | `operator[](i) const`, `at(i) const`, `front()`, `back()`, `operator()(i) const` | return a `T` |
| write | `operator[](i)`, `at(i)` | return a write proxy (§4) |
| raw reference | `operator()(i)` (non-const) | a `T&` into the buffer: **only on a materialised chunk**, bypasses the library |
| preallocate | `alloc_chunk(i)`, `alloc_chunk(i, rank)` | materialises the chunk holding element `i` if it is compressed, keeping what it reads (`T{}`, or the second value it was compressed to); the loop that follows can then store through `operator()(i)`. Callable from a parallel region, each thread on chunks of its own (`rank`: the calling thread's, its buffer pool) |
| grow, shrink **(S)** | `resize(n)`, `reserve(n)`, `clear()`, `shrink_to_fit()` | `shrink_to_fit()` also folds every chunk that became constant; a negative `n` (here and in the constructor) throws `std::length_error`, as `std::vector` |
| append | `push_back(v)`, `emplace_back(...)`, `pop_back()` | **(S)** only when it needs a new chunk |
| replace **(S)** | `assign(n, v)`, `assign(first, last)`, `swap(other)`, `operator=` | |
| iterate | `begin()`, `end()`, `cbegin()`, `rbegin()`, `rend()` | const random-access iterators, by value |
| compress **(S)** | `compress()` | a full compression pass now (`zero::Array`, `zero_or_value::Array`). `zero_any::Array`'s compresses nothing: it reports the chunks that are constant and still materialised (`CHUNK_LOG=compress=info`); `shrink_to_fit()` is the pass there |
| free second values **(S)** | `reclaim_second_values()` (static) | frees the second values no chunk of any `zero_any::Array<T>` uses any more, so that their identifiers are reused; returns how many (`zero_any::Array`); a sequential point over **every** array of the type, see below |
| statistics | `stats()`, `written_chunk_count()` | percentage and number of chunks holding a buffer (each array also prints one line when it is destroyed) |
| geometry | `chunk_size()`, `rechunk_to(n)` **(S)** | `rechunk_to()` only in the `_dyn` libraries (a no-op in static ones) |

`zero_or_value` only: `set_second_value(v)` **(S)**, before the writes, and `second_value()`.

**`reclaim_second_values()`: a precondition over the whole type.** The other **(S)** operations
concern one array; this one is static and frees second values for every `zero_any::Array<T>` of the
program, regardless of their index type. While it runs, **no thread may write to or compress any
`zero_any::Array<T>`**, not only the arrays the caller owns: in a program where several modules keep
their own `zero_any::Array<double>`, the sequential point must hold for all of them.

- *Outside the contract, the error is silent*: a chunk compressed while the reclaim runs can keep an
  identifier that the reclaim frees and then hands to another value; that chunk then reads the
  other value, with no crash and no message.
- *What is refused, in every build*: a call inside a parallel region of the calling thread,
  during a re-chunk, or while the second-value index is degraded (it returns 0). Threads the
  call cannot see are not detected: a `std::thread` or TBB pool outside the phase contract, or
  an OpenMP team started by another host thread. Under the std backend the application's
  threads are seen between `begin_parallel_phase()` and `finalize_parallel_phase()`: the call is
  refused there.
- *What detects it*: check builds (Debug, `CHUNK_CHECK_OPTIM`) stop on any second-value lookup or
  registration that overlaps a reclaim. Run the application once in a check build with its
  reclaim calls in place; release builds pay nothing and check nothing.
- *Watching the identifiers*: `zero_any::Array<T>::second_value_stats()` (static, any time, any
  thread) returns the identifiers in use against the capacity -- running out stops the run --,
  the free ones, the reclaims run and what they freed, the reclaims refused by cause (a refused
  call returns 0 as one with nothing to free does) and whether the index is degraded (then every
  reclaim is refused and the identifiers only grow). Sampled at the application's checkpoints,
  `in_use()` / `capacity` is the early warning.

Not provided: `data()`, mutable iterators, `insert` / `erase` in the middle, an in-place sort.

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

An operand of another type than the element: a plain `T` computes in the common type, and so does
the proxy for the *clean* operands -- the same type, an integer into a floating or complex `T`, the
scalar of a complex `T`, an `int` into `short`, `char` or `unsigned char`. Any other operand is
converted to `T` first, with an `HPC WARNING` (`[[deprecated]]`), the same in every variant and
build, deferred or not: an `int` element `*= 1.5` is multiplied by 1, `/= 0.5` divided by 0.

## 5. Range operations

`chunk/range_ops.h`. One decision per chunk instead of one per element.

| call | context |
|---|---|
| `fill_range(a, first, last, v)` | sequential: `[first, last)` set to `v` |
| `copy_range(a, first, last, src_first)` | sequential: `[first, last)` copied from an iterator |
| `fill_range_mt(a, first, last, v)` | callable by several threads at once, on disjoint ranges (a chunk may be shared) |
| `copy_range_mt(a, first, last, src_first)` | `copy_range`, callable as `fill_range_mt` is |
| `transform_range_mt(a, first, last, src_first, op)` | element `i` gets `op(*(src_first + (i - first)))`, callable as `fill_range_mt` is; `op` must not touch `a` |
| `transform_ranges_mt(a, first, last, op, b, c, ...)` | element `i` gets `op(b[i], c[i], ...)`, computed chunk by chunk: the sources (chunked arrays or contiguous containers) are read by blocks, `op` runs in a loop the compiler can vectorise, and a span where every source is compressed is filled with one result, without a buffer; callable as `fill_range_mt` is. No source may be `a`, nor be written meanwhile |
| `for_each_value(a, first, last, f)` | read: `f(v)` on each element, one read section per block |
| `for_each_span(a, first, last, f)` | read: `f(std::span<const T>)` per block (a vectorisable loop); a compressed chunk as `f(v, count)` when `f` accepts it. A `std::vector` is one span |
| `read_range(a, first, last, out)` | read: `[first, last)` copied to `out` |

### 5.1 Chunk states: export and import

`chunk/chunk_exchange.h`. An array read and rebuilt chunk by chunk without decompressing it --
the base of a transfer (host to device) or a compressed checkpoint. A `ChunkState<T>` says what a
chunk holds: `first_value` or `second_value` (`value`, no buffer), `motif` (pattern: element `k`
is `data[k] + value`), or `materialised` (`data`, its `chunk_size()` elements). It does not
depend on how the library encodes the chunk inside.

| call | context |
|---|---|
| `export_chunk(a, c)` | sequential: chunk `c`'s state; `data` points into `a`, valid until `a` is written |
| `import_chunk(a, c, state)` | sequential: chunk `c` takes `state`, a constant without a buffer; `false` if the chunk was written |
| `export_array(a)` | sequential: size, `chunk_size()`, zero_or_value's second value, every chunk's state |
| `import_array(a, state)` | sequential: `a` becomes the array `state` describes, whatever it held; `false` (and `a` unchanged) at another chunk size; the chunks past `state`'s size (a `reserve()`) are not imported |

Every variant and element type but `Bit` (BitArray). Call them after the phase, as
`fill_range`.

## 6. OpenMP phases

`chunk/phases.h`. Needed only when `build_config::needs_explicit_phases` (GCC's libgomp); no-ops
otherwise, and in serial builds.

| call | where |
|---|---|
| `begin_parallel_phase()` | inside a parallel region, by every thread, before its first access |
| `finalize_parallel_phase()` | after the region, once |
| `finalize_parallel_phase_only_flush()` + `finalize_parallel_phase_after_flush()` | the same, split: the first inside the region (it emits a barrier), the second after it |
| `finalize_barrier_phase()` | inside a region, after an explicit `omp barrier`: makes the writes before it visible after it |
| `parallel_phase([n,] body)` | instead of the first two: a parallel region (`n` threads, or the runtime's) running `body` on every thread, with `begin_parallel_phase()` before it and `finalize_parallel_phase()` after the region; the worksharing (`omp for`) goes in `body` |
| `tbb::parallel_for(...)` | a TBB program, std or serial backend (`chunk/tbb_adaptor.h`): the arguments of `oneapi::tbb::parallel_for`, with `begin_parallel_phase()` before every piece and `finalize_parallel_phase()` after the call |

A `begin_parallel_phase()` with no `finalize_parallel_phase()` since the previous region stops
the program, in every build. Check builds (Debug, `CHUNK_CHECK_OPTIM`) also stop on a deferred write
in a region by a thread that did not call `begin_parallel_phase()`.

The application's own threads (`std::thread`, TBB) need the std backend (`CHUNK_THREADS=std`):
every thread that writes calls `begin_parallel_phase()` before its first write, and one thread
calls `finalize_parallel_phase()` once they have stopped. `shrink_array::tbb::parallel_for`
makes both calls for a TBB loop; the header is installed with the others, and the program links
TBB itself (the library does not). Its first call also gives the library's own teams (the flush
and compression pass of `finalize_parallel_phase()`, a re-chunk, a parallel initialisation) to
TBB's threads instead of the std backend's pool, so that the two do not compete for the cores;
a TBB program that writes the arrays without the adaptor calls
`shrink_array::tbb::use_tbb_for_library_teams()` itself, once. Those teams run in the arena of
the thread that opens them, the one its TBB algorithms run in; a program whose regions run in a
`task_arena` of its own while it calls `finalize_parallel_phase()` outside it passes that arena:
`use_tbb_for_library_teams(arena)`. Otherwise TBB's threads change arena twice per region.

The threads, whatever the backend (`chunk/threads.h`): size per-thread data and number the
threads with these rather than with `omp_get_*`, so that the program builds with every backend.

| call | OpenMP | std backend | serial |
|---|---|---|---|
| `shrink_array::max_threads()` | `omp_get_max_threads()` | the team size of the next region | 1 |
| `shrink_array::team_size()` | `omp_get_num_threads()` | the size of the innermost team | 1 |
| `shrink_array::thread_rank()` | `omp_get_thread_num()` | the rank in the innermost team (the registered rank outside one) | 0 |

## 7. Adaptive chunk size (`_dyn` libraries)

| call | header | what it does |
|---|---|---|
| `ChunkSizeAdvisor<T>::analyze(a)` | `chunk/chunk_size_advisor.h` | estimates the array's memory at each candidate size; read-only, at a quiescent point |
| `RechunkDecider` with `RechunkPolicy` | `chunk/rechunk_decider.h` | decides, tick after tick, whether a re-chunk is worth it |
| `decider.consider<T>(a, tick, iter, "name")` | | analyses the array and applies the re-chunk if the policy agrees **(S)** |
| `decider.print_rechunk_log()` | | the table of the re-chunks performed |
| `a.rechunk_to(n)` **(S)** | | re-chunk one array to `n` elements per chunk |
| `a.set_chunk_size(n)` | | the geometry of an EMPTY array, before its first `resize()`; refused (warning) on an array that has chunks -- `rechunk_to()` moves the elements; a no-op in a static build |
| `set_active_chunk_size(n)` | `chunk/chunk_size_advisor.h` | the process-wide chunk size, before `Runtime::init()`: the first geometry |
| `set_thread_chunk_size(n)` | `chunk/chunk_size_advisor.h` | the size the calling thread's next chunk allocations are routed to; the library sets it before each allocation. Not a new array's geometry: arrays are born at `set_active_chunk_size()`, `a.set_chunk_size(n)` sets an empty one's |
| `sizeof(zero_any::Chunk<T>)` | `chunk/zero_any_array.h` | the metadata of one chunk, for the advisor's options |

The policy, its defaults and the measured results: [adaptive_rechunk.md](adaptive_rechunk.md).

## 8. Compile-time configuration

| name | meaning |
|---|---|
| `build_config::use_openmp` | built with OpenMP |
| `build_config::threaded` | the library runs threads (`CHUNK_THREADS` `openmp` or `std`) |
| `build_config::has_ompt`, `build_config::needs_explicit_phases` | built with the OMPT tool, or not |
| `build_config::chunk_dynamic` | dynamic geometry (`_dyn` library) |
| `chunk_config::use_zero`, `chunk_config::use_zero_or_value`, `chunk_config::use_zero_any` | the variant |
| `chunk_config::stats` | `CHUNK_STATS` build |

Environment variables and build options: [environment_variables.md](environment_variables.md).

## 9. The log

Every message of both libraries goes through one log (`chunk/log.h`, namespace
`shrink_array::logging`): a level (`error`, `warn`, `info`, `debug`, `trace`) and a channel
(`general`, `env`, `alloc`, `arena`, `pool`, `rechunk`, `drain`, `compress`, `recycle`, `ledger`,
`ompt`, `affinity`, `advisor`, `memory`). By default the warnings and errors are written, one line
each, to stderr:

```
[shrink_array][rechunk][W][12345] rechunk_to(48) refused: a chunk size must be a power of two or a multiple of 64
```

| call or variable | what it does |
|---|---|
| `CHUNK_LOG=warn,rechunk=info` | the levels: a bare level for every channel, `channel=level` for one, left to right; `off` writes nothing but the fatal failures |
| `CHUNK_LOG_FILE=path` | the file the lines are appended to instead of stderr (`%p`: the process id) |
| `logging::configure("warn,rechunk=debug")` | the same specification from the code, on top of `CHUNK_LOG`; returns the number of items not understood |
| `logging::set_sink(fn, user)` | hands every record (level, channel, message, source location) to `fn` -- the program's own logger -- instead of stderr; `nullptr` restores stderr. A record can arrive at exit: the sink must outlive it, or be removed first |

What a program asks for explicitly -- `print_all_reuse_pool_breakdown()`, `gc_mark_report()` -- is
written regardless of what `CHUNK_LOG` says, through the same sink. The library's reports are levels of
their channels: `arena=info` (the buffer arena), `ledger=info` (the block registry, at exit),
`ompt=info` (the synchronisation regions, at exit). A broken invariant or a fatal allocator failure always stops the program: a block on
stderr headed `[CRITICAL_FAILURE]` (`[DEBUG_INVARIANT_FAILURE]` in a debug build), the
backtrace, then the record (level fatal) to the sink or `CHUNK_LOG_FILE`. The channels'
contents: [developer_settings.md](developer_settings.md), section 2.

## 10. Printing

`chunk/print.h`: `shrink_array::print` and `println`, the C++23 `std::print` / `std::println` in
every build -- the standard ones where the library has them, the same signatures over
`std::format` otherwise. `println("{} chunks", n)` to stdout, `println(stderr, ...)` to a stream.
The examples print through them.

## 11. GPU arrays (`chunk/device/`, experimental)

One device implementation, `device::Array<T, S, Variant>` (`chunk/device/array.h`), under the
names of the host variants. Header-only, compiled by the program's CUDA or HIP compiler; the
CMake target `shrink_array::device` gives the include path and C++20, the program enables its
device language and links its runtime (`CUDA::cudart`, `hip::host`). Design and measurements:
[gpu_design.md](gpu_design.md), [gpu_assign.md](gpu_assign.md).

| header | names | folds a chunk back to |
|---|---|---|
| `chunk/device/zero_array.h` | `zero::DeviceArray<T, S = 1024>`, `zero::DeviceView`, `zero::DeviceWriter` | `T{}` |
| `chunk/device/zero_any_array.h` | `zero_any::DeviceArray`, `DeviceView`, `DeviceWriter` | any value |
| `chunk/device/zero_or_value_array.h` | `zero_or_value::DeviceArray`, `DeviceView`, `DeviceWriter` | `T{}` or `set_second_value(v)` |
| `chunk/device/kokkos_adaptor.h` | `zero::KokkosArray`, `zero_any::KokkosArray`, `zero_or_value::KokkosArray` | as above, for Kokkos codes |

S = 1024 is the GPU geometry (section 13 of gpu_design.md). What the host arrays share with the
device ones: `DeviceArray(n)` (a first arena of one buffer per 64 chunks, grown by the phases),
`size()`, `chunk_size()`, `stats()` and `written_chunk_count()` (the chunks holding a buffer),
`compress()`. The phases: a write kernel takes `writer()` (`w.assign(i, v)`), then `finish()`;
`parallel_for_chunks(lo, hi, f)` is the library's own write loop, a phase by itself; a read
kernel takes `view()` (`v(i)`), between write phases. To and from a host array, without
decompressing: `DeviceArray(export_array(host))` or `assign(state)`, and `image()`, whose
`state` `import_array()` takes (`chunk/chunk_exchange.h`, in a host translation unit: the device
headers do not include the host arrays'). Not on the device: `resize`, `push_back`, iterators,
element access from the host (`to_host()` copies every element).

## Names changed in 0.3

The former names compiled until 0.3, deprecated; 0.4 removed them. A program written against
0.2 replaces them as follows (the `pmp::` spellings of the three thread functions remain).

| former name | name |
|---|---|
| `chunk/manage_ompt.h` | `chunk/phases.h` |
| `chunk/param_chunk.h` | `chunk/runtime.h` |
| `chunk/array_chunk_bits.h` | `chunk/bit_array.h` |
| `chunk/array_chunk_range_ops.h` | `chunk/range_ops.h` |
| `ParamChunk` | `Runtime` |
| `ParamChunk::init_indexes()` | `Runtime::init()` |
| `destroy_block_chunk()` | `Runtime::shutdown()` |
| `build_config::has_not_ompt` | `build_config::needs_explicit_phases` |
| `ArrayChunkBits<>` | `BitArray<>` |
| `get_second_value()` | `second_value()` |
| `get_nb_chunk_written()` | `written_chunk_count()` |
| `pmp::max_threads()`, `pmp::team_size()`, `pmp::thread_rank()` | `max_threads()`, `team_size()`, `thread_rank()` |
| `zero_one_value::Array`, `chunk/zero_one_value_array.h` | `zero_or_value::Array`, `chunk/zero_or_value_array.h` |
| `chunk_config::use_zero_one_value`, `CHUNK_VARIANT_ZERO_ONE_VALUE` | `chunk_config::use_zero_or_value`, `CHUNK_VARIANT_ZERO_OR_VALUE` |
