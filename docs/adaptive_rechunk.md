# Dynamic geometry and adaptive rechunk

> French version: [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md).

> Status: end of September 2026, for the three variants (`zero`, `zero_one_value`, `zero_any`). The figures
> come from `shrink_bench` on Grace (64 threads, `OMP_PROC_BIND=close`,
> `OMP_PLACES=cores`), with memory measured exactly and times compared on the same
> binary: see [performance_measurement.md](performance_measurement.md). When in doubt,
> the code is the reference.

***

## 1. The principle

In **static geometry** (`-DCHUNK_SIZE=…`, or the `_64`, `_256`, `_1024` targets of the benchmark),
the chunk size is a compile-time constant. In **dynamic geometry**
(`-DCHUNK_DYNAMIC`, `_dyn` targets), each array has its own size, which it can change
during the run through a **rechunk**: the complete rebuild of its storage at another
size.

The point: start from **small chunks** (64 elements in the benchmark, `SHRINK_BENCH_DYN_SIZE`),
which compress finely, then **grow** when the metadata outweighs what the
fine granularity gains. The goals, reached for `zero_one_value` and `zero_any`:

- **at the start**, use as much memory as the static build at the same size;
- **at the end**, use less than the best static build.

Three actors: the **advisor** estimates the array's memory at each size, the **decider**
chooses whether to change and to which size, and `rechunk_to()` does the rebuild.

## 2. The advisor: estimating memory at each size

<tt>ChunkSizeAdvisor\<T\>\::analyze(arr)</tt> (`chunk_size_advisor.h`) produces a table: for each
candidate size, the number of chunks, the number of materialised chunks, and the estimated
memory (metadata + materialised data).

The candidates: the powers-of-two ratios (S×2 to S×16, S÷2 and S÷4), and between those
bounds every multiple of 64 whose C / 64 has at most `Options::grid_bits` significant bits
(`RechunkPolicy::advisor_grid_bits`, `--rechunk_grid_bits` in the benchmark; 2 by default: 192, 384, 768, 1536...: steps of 50% at most; 3 adds
320, 448, 640, 896...: steps of 25%).
Each row cuts the array into windows of C elements, read from the current chunks:

- a window **inside a run of chunks compressed to the same constant** is compressed: counted
  exactly, from the chunk states alone;
- a window that **covers a whole materialised chunk** is materialised;
- a window that **overlaps a materialised chunk in part** (a finer or a misaligned size) is
  settled by reading the data, on a **sample** of the materialised chunks (65,536 at most):
  it is compressed if it holds one value, and a representable one. For `zero` and `zero_one_value` that
  is the first or the second value; for **`zero_any`** any value, since each chunk records its
  own second value.

When S divides C (S×2, S×3, S×4...) the last case never occurs: the row is an **exact
folding** of the actual state of the chunks. `advisor_demo` checks every row against arrays
built at those sizes, from S = 256 and S = 1024.

Why 2 and not 3 (shrink_bench zero_any_dyn, `--divisor 4`, 16 threads, memory at the end of the
run): 1 bit (powers of two only) 2320 MB, 2 bits 2317 MB, 3 bits 2396 MB. The predictions are as
accurate at 3 bits, but the cheapest size is then often a small step (256 -> 384) whose
metadata gain the decider's rate gate (`coarsen_min_meta_rate`) refuses; the decider does not
fall back to the next best size, and two arrays stayed at 256.

Chunk classification is done on the fly, with no work array: until
September 2026, each analysis allocated one byte and one pointer per chunk (140 MB for an
array of 15.6 million chunks), and the analyses cost 1 to 2 s on a 6.5 s run.
They now cost ~0.7 s on an 8.4 s `zero_one_value` run.

`CHUNK_LOG=advisor=debug` writes the full table of each array analysed at each tick (only one
array's with `RechunkPolicy::trace_array`, `--rechunk_trace_array <name>` in the benchmark).

## 3. The decider: when and where to

`RechunkDecider::consider()` (`rechunk_decider.h`, `internal/rechunk_decider_inline.h`) is
called at each **tick**: every `--advisor_period` iterations in the benchmark
(by default `max(10, iter_max / 20)`, i.e. 100 for 2,000 iterations; 0 disables it). It
takes the best size from the table, then applies the policy (`RechunkPolicy`). The library reads
none of these settings from the environment: a program sets the fields; the benchmark exposes
them as options (in brackets):

| setting | default | role |
|---|---|---|
| `stability_ticks` | 3 | the best size must stay the same for 3 ticks |
| `cooldown_ticks` | 5 | at least 5 ticks between two rechunks of the same array |
| `anti_reversal_ticks` | 30 | no return to a size left less than 30 ticks ago |
| `max_coarsen_factor` | **4** | a coarsening multiplies the size by 4 at most (`--rechunk_max_coarsen`) |
| `min_gain_bytes`, `min_gain_frac` | 8 MB, 15% | data-side gain thresholds (`--rechunk_min_gain_mb`) |
| `coarsen_gain_multiple` | 3 | a coarsening judged on data must gain 3 times the threshold |
| `coarsen_min_meta_gain_bytes` | 1 MB | minimum metadata-side gain (`--rechunk_coarsen_min_meta_mb`) |
| `coarsen_min_meta_rate` | 250 MB/s | metadata gain per second of estimated rebuild time (`--rechunk_coarsen_min_meta_rate`) |
| `coarsen_trust_ratio` | 0.5 | learning: a coarsening that returned less than half of its forecast forbids the following ones |
| `dry_run` | `true` | apply nothing (the benchmark sets it to `false`; `--rechunk_dry_run` keeps it) |
| `max_backoff_ticks` | 8 | an array whose verdict is a settled "no move" (already optimal, gain below the threshold) is analysed again after 2, 4, then up to 8 ticks; any other verdict brings it back to every tick (`--rechunk_backoff_max`, 0 or 1 disables) |
| `advisor_grid_bits` | 2 | the advisor's candidate sizes between the powers of two, see §2 (`--rechunk_grid_bits`) |
| `trace_array` | empty | the per-array traces (`churn` line of `CHUNK_LOG=rechunk=debug`, advisor table of `CHUNK_LOG=advisor=debug`) for this array only; empty: every array (`--rechunk_trace_array`) |
| `force_size`, `force_at`, `force_toggle`, `force_alt` | 0, 0, off, 64 | tests: re-chunk every array to `force_size` from iteration `force_at`, whatever the advisor says; `force_toggle` alternates with `force_alt` at every tick (`--force_rechunk`, `--force_rechunk_at`, `--force_rechunk_toggle`, `--force_rechunk_alt`) |

A coarsening is accepted if **either** of two branches passes: the
**metadata** branch (metadata gain above its floor, and enough throughput given
the cost estimated by `rechunk_cost_ms()`), or the **data** branch (total gain above the
multiplied threshold). Since this "gain per cost" criterion, `min_gain_bytes` is no longer
enough to block coarsenings: to measure the cost of the analyses alone, use
`dry_run` (`--rechunk_dry_run`).

Learning measures the gain of a coarsening **at unchanged data**: at the next
tick, the row of the old size in the table (finer rows) gives the memory
the array would have had without the rechunk, on its current data. Simply comparing
before and after mixed the rechunk with the evolution of the data over a period.

**Why a factor of 4.** Measured on `zero_any` with all arrays handed to the decider:

| factor | rechunks | end memory | gain at unchanged data / forecast |
|---|---|---|---|
| 2 | 44 | 6,729-6,737 MB | 3,655 / 3,693 MB (99%) |
| **4** | **33** | **6,539-6,542 MB** | 3,885 / 3,934 MB (98.8%) |
| 8 | 20 | 6,721-6,724 MB | (not measurable: the old size drops out of the table) |

The limit of 2 dated from two bad large jumps measured before the
September 2026 fixes (skewed `VmRSS` measurement, leak of `zero_any` rows, folding by copy).
At 4, the old size stays within the two finer levels of the table, which keeps the
measurement at unchanged data and the ability to go back.

## 4. The rebuild: `rechunk_to()`

`internal/array_chunk_rechunk_inline.h`. The steps, with their name in the
`CHUNK_LOG=rechunk=debug` trace:

1. **prologue**: at the old geometry, flush the array's deferred writes, then
   compress its pending candidates before removing them from the lists;
2. **tmp create**: build the new storage, at the new size;
3. **refill**: fill it in parallel (schedule `guided,16`).
   A new chunk whose sources are all at the first value is skipped. A new
   chunk whose sources were all compressed is **folded** right away. For
   **`zero_any`**, if all its sources are at the **same** second constant, that constant is
   **installed directly**, with no buffer and no copy (copying followed by folding left
   up to 640 MB of touched pages per array);
4. **flush+swap**: swap identities between the old and the new storage;
5. **tmp teardown**: destroy the old storage;
6. **old-pool drain**: drain the pools of the old size and return the
   slabs that have become empty to the system.

**`zero`** rebuilds **in slices** (16) and frees
each old chunk as soon as it has been copied. Between two slices, the pools of the old
size are drained and the emptied slabs returned to the system, including in the **shared
pool** when the old storage lives there: without this, the memory of the old storage
stayed held until the end of the rebuild (on `rechunk_validation`, rebuild peak
78 → 22 MB). Measured on `zero` at 64 threads: peak of +6.9 GB in one pass,
+1.7 GB with 16 slices.

## 5. Per-array allocators (affinity)

An array whose metadata exceeds 16 MiB (`CHUNK_ALLOC_AFFINITY_MIN_BYTES`) can
have its **own set of allocators** (an affinity key), per rank and per size. Its
buffers then share no slab with any other array: when it is rebuilt or
destroyed, its slabs empty completely and are returned to the system.

The price: one partly used slab per (rank, array, size) triple. With the 1 MiB slabs of the
former sixteen-page floor (64 KiB pages), this was ~780 MiB over the 16 arrays of the benchmark at
64 threads; the slabs are 256 KiB there now, a quarter of it.

Hence two modes, fixed per variant:

| mode | when an array gets its key | default for | why |
|---|---|---|---|
| **lazy** | at its first rechunk | `zero_one_value`, `zero_any` | they are mostly made of metadata: a start without per-array sets uses exactly as much as the static build (`zero_any`: 11,590 MB after initialisation, against 12,435 MB in eager mode) |
| **eager** | at its creation | `zero` | `zero` is mostly made of data, and its slice-by-slice rebuild only returns memory progressively if the old storage has its own slabs. In lazy mode, the first rechunk starts from shared slabs, which mix the blocks of several arrays and do not empty: +6.0 GB on 39.3 GB. `zero` peak: 41.2 GB in eager mode, 45.6 GB in lazy mode, for the same end memory |

The choice was measured against the other mode (Milan, `shrink_bench` int64 dyn, 64 threads,
three runs each, October 2026): `zero` in lazy mode takes +9.8% of time, +14.6% at the peak and
+11.8% at the end; `zero_any` in eager mode +7.4% at the peak; `zero_one_value` is the same either
way. The mode is no longer an environment variable (`CHUNK_ALLOC_AFFINITY_EAGER` is gone).

## 6. What must disappear with the old storage

A rechunk builds the new storage under a **new array identifier**
(`num_tab`) and destroys the old one. Everything indexed by the old identifier must be
**freed**, not just emptied. Two leaks of this kind were fixed in September 2026:

- **`zero_any`**: the per-chunk metadata rows (constant type, 1 byte; second
  value, `sizeof(T)` bytes by default) were emptied by `clear()`, which keeps the capacity. On the
  benchmark with the 15 rechunked arrays: 10,276 → 6,730 MB at the end of the run;
- **all three variants**: the "in compression list" indicator (1 byte per chunk)
  was never freed. `zero_any` −0.52 GB, `zero` −0.75 GB.

In addition, the **compression pass trigger threshold** and the **pool
budget**, computed from the total number of chunks, are now also recomputed when
a storage is destroyed, and no longer only when it is created. No measurable effect on time, but the thresholds
match the actual geometry again.

## 7. Results

Memory (GB) on `shrink_bench`, 64 threads, all arrays handed to the decider (the
benchmark default; `--advise_five` goes back to the five historical arrays):

| variant | static 64 (end) | dynamic without advisor (end) | **dynamic with advisor (end)** | static 256 (end) |
|---|---|---|---|---|
| `zero_any` | 12.7 | 12.7 | **6.0** | 7.3 |
| `zero_one_value` | 21.4 | 21.4 | **16.3** | 17.3 |
| `zero` | 43.6 | 44.6 | **40.3** | 40.2 |

Time cost of the advisor, measured on **the same binary** (averages over 3 rounds):

| dynamic `zero_one_value` | time | cost |
|---|---|---|
| without advisor (`--advisor_period 0`) | 8.38 s | — |
| analyses only (`--rechunk_dry_run`) | 9.07 s | +0.69 s |
| analyses and rechunks (30) | 11.43 s | +2.36 s |

For `zero_any`, the cost of the rechunks (33) ranges from 1 to 3.5 s depending on the measurement series. For
`zero_one_value`, the intervals that follow the rechunks are as fast as those before: the cost
is limited to the rebuilds.

## 8. Known limitations

- **`zero` slows down after its rechunk.** After switching to 1,024, dynamic `zero` runs 1.5 to
  2 times slower per iteration than static 1,024 (about 40 busy cores at an IPC of
  3.65, against 52 cores at 4.24). A dynamic build **started** at 1,024 runs as fast as the
  static one, and `zero_one_value` does not slow down after its rechunks: the state left by the
  `zero` rebuild is to blame. Neither the scheduling of the rebuild nor the
  compression thresholds explain the gap. Cause still unknown.
  Measured again in October 2026 (inti, aarch64, LLVM, 64 threads, `--mode perf`,
  2,000 iterations, `diag_zero_rechunk_slowdown.py`): +17% per
  iteration after the switch to 1,024. Unchanged by one slice, a static rebuild
  schedule, no slab recycling, no `chunk_list_` reuse, lazy affinity; the NUMA
  placement is balanced like the static run's. Most of it was the advisor: its
  analyses went on at every tick once the geometry had settled, and their serial
  parts kept the other threads waiting (__kmp_fork_barrier). With
  `max_backoff_ticks` (same node, `OMP_PROC_BIND=close`, 4,000 iterations, last
  re-chunk at iteration 1,700): +22% -> **+7%**. What remains is compression
  churn the static run does not have (~15,000 buffers freed per iteration
  against none), not yet explained.
- **The model is static.** The advisor judges on the present state of the chunks, not on
  future writes. On this benchmark, the transient materialisations measured do not matter
  (the `churn` line of `CHUNK_LOG=rechunk=debug`), but a workload that writes a lot into constant regions
  could make a coarsening regrettable: this is the role of learning
  (`coarsen_trust_ratio`) and of the anti-reversal rule.
- **Rechunking boolean arrays** is validated for `ArrayChunkBits` only
  (`bits_container_validation`, `zero_one_value` at dyn); `zero_one_value::Array<bool>` and `zero_any::Array<bool>`
  have no harness, and the benchmark does not hand `tab_bool` to the decider.
- **The peak remains the starting one**: a dynamic run that starts from 64 first reaches the
  memory of static 64, then goes down.
