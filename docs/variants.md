# The three variants: `zero`, `zero_or_value`, `zero_any`

> French version: [variantes_fr.md](variantes_fr.md).

> Status: end of September 2026 for the host (sections 1 to 4: `shrink_bench` on Grace, 64
> threads, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, memory measured exactly, see
> [performance_measurement.md](performance_measurement.md)); end of November 2026 for the GPU
> (section 5: `kokkos_shrink_bench` on a GH200 and an MI300A). When in doubt, the code is the
> reference.

> Names in the code: each variant is named after its namespace, `zero::Array`, `zero_or_value::Array`,
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
| **`zero`** | `shrink_array::zero` (`CHUNK_VARIANT_ZERO`) | **zero** (`T{}`) | nothing: one shared constant buffer |
| **`zero_or_value`** | `shrink_array::zero_or_value` (`CHUNK_VARIANT_ZERO_OR_VALUE`) | zero, **or** a **second value** specific to the array (`set_second_value`) | nothing beyond the array |
| **`zero_any`** | `shrink_array::zero_any` (`CHUNK_VARIANT_ZERO_ANY`) | zero, **or any other value**, chosen chunk by chunk | one array from the second-value registry (`second_value_registry.h`) per distinct value, shared by the whole process |

### Booleans: two containers

A `zero_or_value` build has two containers for booleans, chosen by type, array by array. Both
are the `zero_or_value` machinery: an all-`false` chunk is compressed to the first value, an
all-`true` chunk to the second, which is `true` by construction (no `set_second_value()`), and
a compressed chunk costs no data in either. They differ in what a **materialised** chunk holds:

| container | header | a materialised chunk of S elements | a write | neighbouring writes from two threads |
|---|---|---|---|---|
| `zero_or_value::Array<bool>` | `chunk/zero_or_value_array.h` | S bytes, one per `bool` | a plain byte store | safe, as for any element |
| `BitArray<>` (`zero_or_value::Array<Bit>`) | `chunk/bit_array.h` | S/8 bytes, 8 `bool` per byte -- the layout of `std::vector<bool>` | an atomic read-modify-write of the 64-bit word holding the bit (OpenMP builds) | safe: the word is updated atomically, no write is lost |
| `std::vector<bool>`, for comparison | `<vector>` | the whole array, bit-packed, no compression | a read-modify-write of a word, not atomic | **a data race**: two writes in one word may lose one |

`BitArray` takes 8 times less memory per element of a materialised chunk, for a costlier
write; use it
for large boolean arrays with many mixed areas (masks, flags over a mesh) that are written less
often than read. `zero_or_value::Array<bool>` is the one for booleans written in hot loops, or
mostly uniform: its constant chunks cost nothing either. Unlike `std::vector<bool>`,
`BitArray` is not a specialisation that changes `Array<bool>`'s behaviour behind its back:
it is a type of its own, and `zero_or_value::Array<bool>` stays a byte store. The element reads
as a `Bit`, which converts to and from `bool`; as for every proxy container here, an element has
no address. In a dynamic-geometry build an `BitArray` is born with chunks 8 times the
process-wide size (`packed_bit_birth_factor`), so that a chunk holds as many bytes as a `bool`
one.

```cpp
shrink_array::zero_or_value::Array<bool> flags(n); // one byte per element in a materialised chunk
shrink_array::BitArray<>            mask(n);  // one bit per element, 8 per byte
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
| zero or equal to **one** other dominant value, known to the array | **`zero_or_value`** |
| made of many **uniform areas, each with its own value** | **`zero_any`** |

Order of magnitude on `shrink_bench` (arrays of 10⁹ elements, 64 threads), memory at the end
of the run:

| variant | static 256 | dynamic (start at 64, all arrays handed to the decider) |
|---|---|---|
| `zero_any` | 7.26 GB | **6.0 GB** |
| `zero_or_value` | 17.3 GB | **16.3 GB** |
| `zero` | 40.2 GB | 40.3 GB (and 39.7 GB in static 1,024, the size it reaches) |
| `std::vector` (`shrink_bench_omp_vector_perf`) | 125.9 GB | — |

The dynamic geometry gives a clear gain for `zero_or_value` and `zero_any`. For `zero`, whose memory is
mostly data, it breaks even with static 256: see the known limits
in [adaptive_rechunk.md](adaptive_rechunk.md).

## 5. On the GPU

The same three variants run on the GPU (CUDA or HIP, header-only, the CMake target
`shrink_array::device`): `zero::DeviceArray<T, S = 1024>`, `zero_or_value::DeviceArray`,
`zero_any::DeviceArray` (`chunk/device/zero_array.h`, ...), and their Kokkos facades
`zero::KokkosArray`, ... ([api.md](api.md) section 11). A chunk folds back by the same rule as on
the host; what differs is the machinery: a descriptor per chunk that carries its constant, writes
that materialise a chunk once, and the compression pass at the end of a write phase
(`finish()`, `compress()`) instead of during it ([gpu_design.md](gpu_design.md)). S = 1024 is
the GPU geometry.

`kokkos_shrink_bench` (`examples/kokkos`): shrink_bench's access pattern on five arrays of
2^30 `double`, written by Kokkos kernels through the facade, against five `Kokkos::View<double*>`;
2000 iterations of windows of up to 107374 elements (small windows) or 200 of up to 10^7 (large
windows), `compress()` every 10. GH200 (H100, nvcc 13.3) and MI300A (ROCm 10.2), November 2026
(675d0d701); the checksums equal the flat run's everywhere.

| | `zero_any` | `zero_or_value` | `zero` |
|---|---|---|---|
| memory at the end, small windows (flat: 42.95 GB) | **1.51 GB (3.5 %)** | 4.97 GB (11.6 %) | 19.17 GB (44.6 %) |
| memory at the end, large windows | **8.08 GB (18.8 %)** | 11.35 GB (26.4 %) | 25.89 GB (60.3 %) |
| loop / flat, small windows, GH200 / MI300A | 1.49x / 1.50x | 1.47x / 1.49x | **1.36x / 1.38x** |
| loop / flat, large windows, GH200 / MI300A | 1.42x / 1.57x | 1.30x / 1.49x | **1.24x / 1.40x** |
| allocation + fill / flat, GH200 / MI300A | 0.65x / 0.05x | 0.74x / 0.13x | 1.06x / 0.52x |

The memory held is the same to the byte on both devices: it depends on what the benchmark writes,
not on the device (the arenas also keep 0.3 to 2.3 GB of free buffers for the next phases). The
loop's ratio is that of flat arrays that are already allocated and filled; the allocation and fill
of the compressed arrays cost less than the flat views' (on the MI300A, 20 times less for
`zero_any`, the views' 43 GB being mapped at their allocation). The 20 read passes of each
iteration run at flat's speed (0.93x to 1.02x); what the loop pays is the writes into the window
and around it, and `compress()` (about 0.6 ms a pass for the five arrays).

The same work on Grace's 72 cores (`host_twin_bench`, the host arrays, OpenMP), `zero_or_value`:
the GH200's loop is 3.1x faster on the small windows and 25x on the large ones (the GPU is
bandwidth-bound there and the reads are flat's), [gpu_design.md](gpu_design.md) section 13.

Which one on the GPU: the same rule as on the host (section 4). `zero_any` holds the least
memory for 9 to 15 % more time in the loop than `zero`; `zero` is the fastest and holds the most;
`zero_or_value` sits between them when one value besides zero dominates. Write regular loops
with the chunk-aligned path (`parallel_for_chunks`) where you can; `Writer::assign` (Kokkos's
`w(i) = v`) is the general path ([gpu_assign.md](gpu_assign.md)).

## 6. Where to read next

| topic | document |
|---|---|
| the GPU arrays: design, write path, measurements | [gpu_design.md](gpu_design.md), [gpu_assign.md](gpu_assign.md) |
| read, write, parallelism, in-flight compression, range operations | [algo_zero_or_value.md](algo_zero_or_value.md): describes `zero_or_value`, but the machinery of its §5 to §8 and §10 is common to all three variants |
| what `zero` and `zero_any` do differently | [algo_zero.md](algo_zero.md), [algo_zero_any.md](algo_zero_any.md) |
| dynamic geometry and rechunk, for all three variants | [adaptive_rechunk.md](adaptive_rechunk.md) |
| lock-free synchronisation (asymmetric Dekker) | [synchronisation_dekker.md](synchronisation_dekker.md) |
| settings | [environment_variables.md](environment_variables.md) |
| documentation index | [README.md](README.md) |
