# shrink-array: compressed arrays for C++

shrink-array provides arrays that are used like a `std::vector` and take less memory when large
parts of them hold the same value -- zero, or a few repeated values. It is written for HPC codes:
C++20, OpenMP, arrays of 10⁹ elements, up to a few hundred threads.

An array is cut into **chunks** of `S` consecutive elements (`S` a power of two). A chunk whose
elements all hold one constant keeps no buffer of its own: it points to a shared buffer of that
constant and costs a few bytes of metadata. Writing another value into it gives it a buffer
again, and while the program runs, the chunks that have become uniform are compressed again and
their buffers go back to the allocator. Reads and writes keep the `[]` syntax.

It pays off on data that is mostly zero, or mostly made of runs of a few values: sparse fields,
material masks, multi-material quantities, flags. It does not on data where every chunk holds
varied values: there it costs a little more memory than a `std::vector`, and every access is
slower.

## Three variants

| variant | class | a chunk is compressed when it is uniform at… |
|---|---|---|
| Chunk (v1) | `ArrayChunk<T>` | `T{}` (zero) only |
| Chunk2 | `ArrayChunk2<T>` | `T{}`, or **one** second value per array, declared with `set_second_value()` |
| Chunk2A | `ArrayChunk2A<T>` | `T{}`, or **any** value, chunk by chunk, with nothing to declare |

The variant and the chunk size are chosen by the library you link (`chunk::chunk_v2a_256`,
`chunk::chunk_v2a_dyn`, ...); the headers are the same for all of them. Chunk2A compresses the
most and is the one to try first. As an order of magnitude, on the `shrink_bench` benchmark
(arrays of 10⁹ elements, 64 threads), memory at the end of the run:

| variant | static size 256 | dynamic size (adaptive re-chunk) |
|---|---|---|
| Chunk2A | 7.26 GB | 6.0 GB |
| Chunk2 | 17.3 GB | 16.3 GB |
| Chunk (v1) | 40.2 GB | 40.3 GB |

[docs/variants.md](docs/variants.md) explains what each one costs and which one to choose.
`ArrayChunkBits` stores `bool` one bit per element (Chunk2 builds).

## Requirements

- **Linux**, x86-64 or aarch64. The library relies on Linux system calls (`membarrier`,
  `madvise`, `mincore`) and on `/proc`.
- A **C++20** compiler: GCC 13 or 14 and Clang 18 are tested; the build also covers oneAPI
  (`icpx`), AOCC and NVHPC. With Clang the build links libc++ by default (`-DUSE_LIBCXX=OFF`
  for libstdc++).
- **CMake 3.20** or later.
- **OpenMP**, optional (`-DOMP=ON`). A program that uses the arrays from OpenMP regions needs a
  library built with it. Thread identity is the OpenMP rank: use the arrays from OpenMP
  threads, not from `std::thread`, and set `OMP_NUM_THREADS` (or `CHUNK_THREAD_CAPACITY`) to the
  largest team before the first array.
- **OMPT**: with Clang or oneAPI the OpenMP runtime has the OMPT interface, and the library hooks
  the start and end of the parallel regions itself. GCC's `libgomp` has none: the program then
  marks its regions -- `begin_parallel_phase()` by every thread at the top of a region,
  `finalize_parallel_phase()` once after it.
- Python 3, for the source checks run by `ctest` (optional).

## Build and install

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOMP=ON -DCMAKE_INSTALL_PREFIX=$HOME/opt/shrink-array
cmake --build build
ctest --test-dir build
cmake --install build
```

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | the compile-time chunk sizes built, one library per size and variant |
| `CHUNK_BUILD_DYN` | `ON` | also build the libraries whose chunk size is chosen, and can change, at run time (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | allocate the chunk buffers with the standard allocator instead of the bundled `fast_allocators` (useful under AddressSanitizer) |
| `SHRINK_ARRAY_BUILD_EXAMPLES` | `ON` at the top level | the examples, the benchmark and the validation harnesses (the harnesses in Debug, `CHECK_OPTIM` or sanitizer builds, or with `-DCHUNK_BUILD_VALIDATION=ON`) |
| `SHRINK_ARRAY_BUILD_TESTS` | `ON` at the top level | register the tests with `ctest` |

Every option, macro and environment variable: [docs/environment_variables.md](docs/environment_variables.md).

## Use it from CMake

After an install:

```cmake
find_package(chunk 2.2 REQUIRED)            # -DCMAKE_PREFIX_PATH=<install prefix>
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

Or from the sources, as part of your own build. Only the libraries are built then -- no
examples, no tests -- and nothing is written into your cache:

```cmake
add_subdirectory(shrink-array)              # or FetchContent_Declare(shrink_array GIT_REPOSITORY <url> GIT_TAG <tag>)
                                            #    FetchContent_MakeAvailable(shrink_array)
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

`chunk::chunk_v2a` is Chunk2A at the first static size; `chunk::chunk_v2a_1024` and
`chunk::chunk_v2a_dyn` pick another size, `chunk::chunk_v1` and `chunk::chunk_v2` another
variant.

## A first program

```cpp
#include <cstdio>
#include "chunk/array_chunk2A.h"   // shrink_array::ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // build_config::has_not_ompt
#include "chunk/manage_ompt.h"     // begin_parallel_phase, finalize_parallel_phase
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

namespace sa = shrink_array;

int main()
{
    sa::ParamChunk::init_indexes(true);       // optional: the first array runs it otherwise
    {
        const int                n = 1 << 24;
        sa::ArrayChunk2A<double> a(n);        // every element 0.0, no buffer yet

#pragma omp parallel
        {
            if constexpr (build_config::has_not_ompt) sa::begin_parallel_phase();
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
                a[i] = (i < n / 8) ? 1.0 + i % 7 : 3.5;   // the 3.5 run compresses
        }
        if constexpr (build_config::has_not_ompt) sa::finalize_parallel_phase();

        a.shrink_to_fit();                    // fold the chunks that became constant
        std::printf("chunks holding a buffer: %.1f %%\n", a.stats());
    }                                         // the array releases its chunks here
    sa::release_chunk_pools();                // optional: give the pools back before the end
    sa::destroy_block_chunk();
}
```

The complete version, which also checks what it reads back, is
[examples/getting_started](examples/getting_started/getting_started.cpp), and
[docs/getting_started.md](docs/getting_started.md) walks through it.

## Using the arrays

### Element types

An array holds any type that is **trivially copyable**, **equality comparable** and **default
constructible**: every arithmetic type, `bool`, `std::complex`, enums, `std::array` of those and
any aggregate of them. Nothing is declared up front and nothing needs calling first: each array registers its element
type when the first one is constructed, global and `static` arrays included.

```cpp
struct Point { double x, y, z; auto operator<=>(const Point&) const = default; };
shrink_array::ArrayChunk2A<Point> pts(n);   // compresses runs of {0,0,0} and of any repeated point
```

A type owning a resource (`std::string`, anything whose destructor matters) is refused at
compile time, with the contract in the error (`ArrayChunkElement`). The chunks are raw memory,
copied, parked and replaced by a shared constant buffer without ever running a constructor or
a destructor. Chunk2 orders its second values and asks a floating-point aggregate for an
`operator<` (a defaulted `operator<=>` provides it).

### What of `std::vector` is there

- **There:** `operator[]`, `at`, `front`, `back`; `size`, `empty`, `capacity`, `reserve`;
  `resize`, `clear`, `assign`, `swap`, `shrink_to_fit`; `push_back`, `emplace_back`,
  `pop_back`; const random-access iterators (`begin`/`end`, `rbegin`/`rend`, range-for,
  `std::lower_bound`); the `(first, last)` and `{a, b, c}` constructors; deep copy and move.
  `shrink_to_fit()` also folds every chunk that has become constant, and `compress()` runs a
  full compression pass on demand.
- **Not there, by construction:** `data()` (a compressed chunk has no buffer), mutable iterators
  and `T&` on an element, `insert` / `erase` in the middle, and hence an in-place `std::sort`
  (sort a copy, then `assign` it back).
- The index type is `int` by default: `ArrayChunk2A<T, long>` for more than 2³¹ elements.

[docs/api.md](docs/api.md) lists every call, by task, on one page.

### The write proxy

The non-const `operator[]` returns a *proxy*, not a `T&`: a handle on the element that reads it
on conversion and routes every write through the compression machinery.

- `T v = a[i];` takes the value; `auto v = a[i];` keeps the proxy, which then follows the
  element. Code ported from `std::vector` wants `T`, not `auto`.
- `a[i] += x;` and the other compound operators return a proxy; discarding it, as here, lets the
  library defer the operation at no cost. `a[i]++` returns the old value as a `T`.
- A proxy holds references into the array's chunks: anything that rebuilds them -- `resize`,
  `reserve`, `shrink_to_fit`, `clear`, `assign`, `swap`, an assignment to the array, a re-chunk
  -- invalidates it. Keep proxies short-lived.

### Threads

Under OpenMP the contract is `std::vector`'s, element by element:

- **Different elements** need no synchronisation, even when they share a chunk.
- **The same element** accessed from several threads must be ordered by the program (a barrier,
  a `critical` section, an atomic flag with release/acquire). Ordered, it behaves as on a
  `std::vector`: a write the library deferred is applied before any later access the program
  ordered after it.
- **Everything that rebuilds the storage** -- the operations that invalidate proxies, re-chunks
  included -- is a sequential point: call it outside any parallel region.

The write path takes no lock: writers and compressors exclude each other with an asymmetric
Dekker handshake (a plain store for the writer, a `membarrier()` call for the rare compressor),
and a write that meets a compressor on its chunk is deferred and replayed, never lost
([docs/synchronisation_dekker.md](docs/synchronisation_dekker.md)).

### Filling ranges

`chunk/array_chunk_range_ops.h` decides once per chunk instead of once per element:
`fill_range(a, first, last, v)`, `copy_range(a, first, last, src_first)`, and `fill_range_mt`
for several threads filling disjoint ranges at once.

### Adaptive chunk size

In the `_dyn` libraries the chunk size of an array is chosen, and can change, at run time:
`a.rechunk_to(n)` re-chunks one array, and a `RechunkDecider` (`chunk/rechunk_decider.h`),
driven by `ChunkSizeAdvisor` estimates, decides tick after tick whether a re-chunk is worth it.
See [docs/adaptive_rechunk.md](docs/adaptive_rechunk.md).

## Examples and tools

- [examples/getting_started](examples/getting_started): the first program above, complete.
- [examples/simple_array](examples/simple_array): the same computation on a `std::vector` and on
  a compressed array, to compare time and memory.
- [examples/multi-mat](examples/multi-mat): multi-material fields, each material in compressed
  arrays or in `std::vector`s, with loops written once for both
  ([Loops.h](examples/multi-mat/Loops.h)).
- [examples/shrink_bench](examples/shrink_bench): the reference benchmark -- thirteen arrays of
  different element types under a compute loop, `--mode check` against `std::vector` reference
  files, `--mode perf` for timing.
- [examples/validation](examples/validation): one harness per property (compression round trip,
  deferred writes, re-chunk, read/reclaim, pool accounting, nested regions...), run by `ctest`.
- [examples/advisor_demo](examples/advisor_demo) and
  [examples/variant_compare](examples/variant_compare): the chunk-size advisor against real sizes,
  and Chunk2 against Chunk2A on one workload.

## Documentation

| I want to… | read |
|---|---|
| install, link from CMake, write a first program, follow the OpenMP rules | [docs/getting_started.md](docs/getting_started.md) |
| find a call | [docs/api.md](docs/api.md) |
| choose a variant | [docs/variants.md](docs/variants.md) |
| understand reads, writes, in-flight compression, deferred writes | [docs/algo_chunk2.md](docs/algo_chunk2.md) |
| use dynamic chunk sizes | [docs/adaptive_rechunk.md](docs/adaptive_rechunk.md) |
| tune the library or the build | [docs/environment_variables.md](docs/environment_variables.md) |
| measure memory and time reliably | [docs/performance_measurement.md](docs/performance_measurement.md) |
| contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

Every document also exists in French; the index is [docs/README.md](docs/README.md). What changed
from one version to the next, and the versioning policy: [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0, see [LICENSE](LICENSE).

Copyright CEA
