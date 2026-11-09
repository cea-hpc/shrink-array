# The three variants: `zero`, `zero_one_value`, `zero_any`

> French version: [variantes_fr.md](variantes_fr.md).

> Status: end of September 2026. The figures come from `shrink_bench` on Grace (64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), with memory measured exactly (see
> [performance_measurement.md](performance_measurement.md)). When in doubt, the code is the
> reference.

> Names in the code: each variant is named after its namespace, `zero::Array`, `zero_one_value::Array`,
> `zero_any::Array`. Older material calls them Chunk (or v1), Chunk2 and Chunk2A (the former class
> names `ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A` were removed before 1.0, see CHANGELOG.md).

***

## 1. The common principle

An array is split into **chunks** of `S` consecutive elements. A chunk is either
**materialised** (it owns a buffer of `S` elements) or **compressed**: it stores only
a constant, and then costs only its metadata. A read from a compressed chunk
returns the constant; a write of a different value first **materialises** it. Compression
passes then take the buffer back from chunks that have become uniform again.

The three variants share this machinery: the proxies that intercept reads and
writes, the deferred writes while a compression is in progress, the block compression
passes, the per-rank allocators (`fast_allocators`), and the dynamic
geometry. They differ in **what they can compress**.

## 2. What each one can compress

| variant | library (macro it defines) | a chunk is compressed if all its elements equal… | cost of a constant |
|---|---|---|---|
| **`zero`** | `chunk::chunk_zero` (`CHUNK_VARIANT_ZERO`) | **zero** (`T{}`) | nothing: one shared constant buffer |
| **`zero_one_value`** | `chunk::chunk_zero_one_value` (`CHUNK_VARIANT_ZERO_ONE_VALUE`) | zero, **or** a **second value** specific to the array (`set_second_value`) | nothing beyond the array |
| **`zero_any`** | `chunk::chunk_zero_any` (`CHUNK_VARIANT_ZERO_ANY`) | zero, **or any other value**, chosen chunk by chunk | one array from the second-value registry (`tab_cst_val2_holder.h`) per distinct value, shared by the whole process |

### Booleans: two containers

A `zero_one_value` build has two containers for booleans, chosen by type, array by array. Both
are the `zero_one_value` machinery: an all-`false` chunk is compressed to the first value, an
all-`true` chunk to the second, which is `true` by construction (no `set_second_value()`), and
a compressed chunk costs no data in either. They differ in what a **materialised** chunk holds:

| container | header | a materialised chunk of S elements | a write | neighbouring writes from two threads |
|---|---|---|---|---|
| `zero_one_value::Array<bool>` | `chunk/zero_one_value_array.h` | S bytes, one per `bool` | a plain byte store | safe, as for any element |
| `ArrayChunkBits<>` (`zero_one_value::Array<Bit>`) | `chunk/array_chunk_bits.h` | S/8 bytes, 8 `bool` per byte -- the layout of `std::vector<bool>` | an atomic read-modify-write of the 64-bit word holding the bit (OpenMP builds) | safe: the word is updated atomically, no write is lost |
| `std::vector<bool>`, for comparison | `<vector>` | the whole array, bit-packed, no compression | a read-modify-write of a word, not atomic | **a data race**: two writes in one word may lose one |

`ArrayChunkBits` takes 8 times less memory per element of a materialised chunk, for a costlier
write; use it
for large boolean arrays with many mixed areas (masks, flags over a mesh) that are written less
often than read. `zero_one_value::Array<bool>` is the one for booleans written in hot loops, or
mostly uniform: its constant chunks cost nothing either. Unlike `std::vector<bool>`,
`ArrayChunkBits` is not a specialisation that changes `Array<bool>`'s behaviour behind its back:
it is a type of its own, and `zero_one_value::Array<bool>` stays a byte store. The element reads
as a `Bit`, which converts to and from `bool`; as for every proxy container here, an element has
no address. In a dynamic-geometry build an `ArrayChunkBits` is born with chunks 8 times the
process-wide size (`packed_bit_birth_factor`), so that a chunk holds as many bytes as a `bool`
one.

```cpp
shrink_array::zero_one_value::Array<bool> flags(n); // one byte per element in a materialised chunk
shrink_array::ArrayChunkBits<>            mask(n);  // one bit per element, 8 per byte
flags[i] = true;
mask[i]  = true;              // two threads may write neighbouring bits at once
const bool b = std::as_const(mask)[i];
```

The program [examples/bool_arrays](../examples/bool_arrays/bool_arrays.cpp) writes both from every
thread, neighbouring elements included, checks every value and prints the data each one keeps.

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
| made of many **uniform areas, each with its own value** | **`zero_any`** |

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
