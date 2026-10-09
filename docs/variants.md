# The three variants: Chunk, Chunk2, Chunk2A

> French version: [variantes_fr.md](variantes_fr.md).

> Status: end of September 2026. The figures come from `shrink_bench` on Grace (64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), with memory measured exactly (see
> [performance_measurement.md](performance_measurement.md)). When in doubt, the code is the
> reference.

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
| **Chunk** (v1) | `chunk::chunk_v1` (`CHUNK_VARIANT_V1`) | **zero** (`T{}`) | nothing: one shared constant buffer |
| **Chunk2** | `chunk::chunk_v2` (`CHUNK_VARIANT_V2`) | zero, **or** a **second value** specific to the array (`set_second_value`) | nothing beyond the array |
| **Chunk2A** | `chunk::chunk_v2a` (`CHUNK_VARIANT_V2A`) | zero, **or any other value**, chosen chunk by chunk | one array from the second-value registry (`tab_cst_val2_holder.h`) per distinct value, shared by the whole process |

Boolean arrays in Chunk2 builds can also use `ArrayChunkBits`, which
packs 8 booleans per byte in a materialised chunk.

## 3. What a chunk costs

Every chunk costs its **metadata**, whether it is compressed or not. For all three variants,
the chunk object itself is 16 bytes. On top of that, per chunk:

- one "already in compression list" byte (block compression);
- two bytes for the array's *dirty bitset* (one bit per chunk, in 64-bit words each padded to
  128 bytes against false sharing), and a share of the tracking lists (the advisor counts
  about 3 bytes for all of these);
- for **Chunk2A**: one constant-type byte and a slot for the chunk's second value,
  of `sizeof(T)` bytes (8 for a `double`), or of 4 bytes with the CMake option
  `CHUNK_SECOND_VALUE_INDEXED`.

This cost is what makes **small chunks** expensive in memory (4 times more metadata at
64 elements than at 256), and what justifies the dynamic geometry: start with small chunks, which
compress finely, then grow them when the metadata dominates.

## 4. Which one to choose

| the data is… | variant |
|---|---|
| mostly zero, with areas of arbitrary data | **Chunk**: the simplest, and no second-value metadata |
| zero or equal to **one** other dominant value, known to the array | **Chunk2** |
| made of many **areas uniform at different values** | **Chunk2A** |

Order of magnitude on `shrink_bench` (arrays of 10⁹ elements, 64 threads), memory at the end
of the run:

| variant | static 256 | dynamic (start at 64, all arrays handed to the decider) |
|---|---|---|
| Chunk2A | 7.26 GB | **6.0 GB** |
| Chunk2 | 17.3 GB | **16.3 GB** |
| Chunk (v1) | 40.2 GB | 40.3 GB (and 39.7 GB in static 1,024, the size it reaches) |

The dynamic geometry gives a clear gain for Chunk2 and Chunk2A. For v1, whose memory is
mostly data, it breaks even with static 256: see the known limits
in [adaptive_rechunk.md](adaptive_rechunk.md).

## 5. Where to read next

| topic | document |
|---|---|
| read, write, parallelism, in-flight compression, range operations | [algo_chunk2.md](algo_chunk2.md): describes Chunk2, but the machinery of its §5 to §8 and §10 is common to all three variants |
| what Chunk (v1) and Chunk2A do differently | [algo_chunk.md](algo_chunk.md), [algo_chunk2a.md](algo_chunk2a.md) |
| dynamic geometry and rechunk, for all three variants | [adaptive_rechunk.md](adaptive_rechunk.md) |
| lock-free synchronisation (asymmetric Dekker) | [synchronisation_dekker.md](synchronisation_dekker.md) |
| settings | [environment_variables.md](environment_variables.md) |
| documentation index | [README.md](README.md) |
