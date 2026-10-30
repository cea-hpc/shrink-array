# The three variants: `zero`, `zero_one_value`, `zero_any`

> French version: [variantes_fr.md](variantes_fr.md).

> Status: end of September 2026. The figures come from `shrink_bench` on Grace (64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), with memory measured exactly (see
> [performance_measurement.md](performance_measurement.md)). When in doubt, the code is the
> reference.

> Names in the code: each variant is named after its namespace, `zero::Array`, `zero_one_value::Array`,
> `zero_any::Array`. Older material calls them Chunk (or v1), Chunk2 and Chunk2A (the former class
> names `ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A` are deprecated aliases until 1.0).

***

## 1. The common principle

An array is split into **chunks** of `S` consecutive elements. A chunk is either
**materialised** (it owns a buffer of `S` elements) or **compressed**: it stores only
a constant, and then costs only its metadata. A read from a compressed chunk
returns the constant; a write of a different value first **materialises** it. Compression
passes then give the buffer back from chunks that have become uniform again.

The three variants share this machinery: the proxies that intercept reads and
writes, the deferred writes while a compression is in progress, the block compression
passes, the per-rank allocators (`fast_allocators`), and the dynamic
geometry. They differ in **what they can compress**.

## 2. What each one can compress

| variant | library (macro it defines) | a chunk is compressed if it is uniform at… | cost of a constant |
|---|---|---|---|
| **`zero`** | `chunk::chunk_zero` (`CHUNK_VARIANT_ZERO`) | **zero** (`T{}`) | nothing: one shared constant buffer |
| **`zero_one_value`** | `chunk::chunk_zero_one_value` (`CHUNK_VARIANT_ZERO_ONE_VALUE`) | zero, **or** a **second value** specific to the array (`set_second_value`) | nothing beyond the array |
| **`zero_any`** | `chunk::chunk_zero_any` (`CHUNK_VARIANT_ZERO_ANY`) | zero, **or any other value**, chosen chunk by chunk | one array from the second-value registry (`tab_cst_val2_holder.h`) per distinct value, shared by the whole process |

Boolean arrays in `zero_one_value` builds can also use `ArrayChunkBits`, which
packs 8 booleans per byte in a materialised chunk.

`zero_any` builds also provide **`pattern::Array`**, for integers: a member of the `zero_any`
family that compresses, besides its constants, a chunk equal to a **shared motif plus an offset**
-- the shape of a mesh connectivity, compressed to about 1 % of a vector on a structured mesh.
See [algo_pattern.md](algo_pattern.md).

## 3. What a chunk costs

Every chunk costs its **metadata**, whether it is compressed or not. For all three variants,
the chunk object itself is 16 bytes in an OpenMP build (the pointer, then the per-chunk read and
write gates). On top of that, per chunk:

- one "already in compression list" byte (block compression);
- two bytes for the array's *dirty bitset* (one bit per chunk, in 64-bit words each padded to
  128 bytes against false sharing), and a share of the tracking lists (the advisor counts
  about 3 bytes for all of these);
- for **`zero_any`**: one constant-type byte and a slot for the chunk's second value,
  of `sizeof(T)` bytes (8 for a `double`), or of 4 bytes with the CMake option
  `CHUNK_SECOND_VALUE_INDEXED`.

This cost is what makes **small chunks** expensive in memory (4 times more metadata at
64 elements than at 256), and what justifies the dynamic geometry: start with small chunks, which
compress finely, then grow them when the metadata dominates.

## 4. Which one to choose

| the data is… | variant |
|---|---|
| mostly zero, with areas of arbitrary data | **`zero`**: the simplest, and no second-value metadata |
| zero or equal to **one** other dominant value, known to the array | **`zero_one_value`** |
| made of many **areas uniform at different values** | **`zero_any`** |

Order of magnitude on `shrink_bench` (arrays of 10⁹ elements, 64 threads), memory at the end
of the run:

| variant | static 256 | dynamic (start at 64, all arrays handed to the decider) |
|---|---|---|
| `zero_any` | 7.26 GB | **6.0 GB** |
| `zero_one_value` | 17.3 GB | **16.3 GB** |
| `zero` | 40.2 GB | 40.3 GB (and 39.7 GB in static 1,024, the size it reaches) |
| `std::vector` (`shrink_bench_omp_vector_perf`) | 125.9 GB | — |

The dynamic geometry gives a clear gain for `zero_one_value` and `zero_any`. For `zero`, whose memory is
mostly data, it breaks even with static 256: see the known limits
in [adaptive_rechunk.md](adaptive_rechunk.md).

## 5. Where to read next

| topic | document |
|---|---|
| read, write, parallelism, in-flight compression, range operations | [algo_zero_one_value.md](algo_zero_one_value.md): describes `zero_one_value`, but the machinery of its §5 to §8 and §10 is common to all three variants |
| what `zero` and `zero_any` do differently | [algo_zero.md](algo_zero.md), [algo_zero_any.md](algo_zero_any.md) |
| dynamic geometry and rechunk, for all three variants | [adaptive_rechunk.md](adaptive_rechunk.md) |
| lock-free synchronisation (asymmetric Dekker) | [synchronisation_dekker.md](synchronisation_dekker.md) |
| settings | [environment_variables.md](environment_variables.md) |
| documentation index | [README.md](README.md) |
