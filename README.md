# shrink-array: compressed arrays for C++

## Goals

This C++ library compresses arrays in memory when large parts of them hold the same value -- zero, or a few repeated values.

These compressed arrays are used the same way as a `std::vector`. Compression management is done by overloading `operator[]`.

The syntax for reading and writing data in these arrays is identical, using `[]`.
However, the implementation of `operator[]` overload is not the same:
 - in the case of reading there are just two indirections and some bitwise computations, 
 - in the case of writing there is also the possible creation of a block via dynamic allocation.

## Implementation

An array is cut into **chunks** of `S` consecutive elements, `S` a power of two: a compile-time constant (`CHUNK_STATIC_SIZES`, 256 and 1024 by default), or, in the dynamic-geometry libraries (`_dyn`), a per-array size that can change at run time (`rechunk_to`, or the adaptive `RechunkDecider`). A chunk whose elements all hold the same constant is **compressed**: it keeps no buffer of its own and points to a shared constant one. Three variants differ in what they can compress:

| variant | a chunk is compressed when it is uniform at… |
|---|---|
| `ArrayChunk` (v1) | `T{}` |
| `ArrayChunk2` | `T{}`, or one second value per array (`set_second_value`) |
| `ArrayChunk2A` | `T{}`, or any value, chosen chunk by chunk |

Compression happens **while the program runs**: a write that turns an element into one of the constants queues its chunk as a candidate, and compression passes give the buffers of the chunks found uniform back to per-thread reuse lists. `compress()` and `shrink_to_fit()` run a full pass on demand.

Under OpenMP the write path takes **no lock**. Writers and compressors exclude each other with an asymmetric Dekker handshake: a writer announces itself with a plain store, and the rare compressor pays a `membarrier()` system call instead. A write that meets a compressor on its chunk is deferred and replayed later, never lost. See [docs/synchronisation_dekker.md](docs/synchronisation_dekker.md).

Chunk buffers come from the bundled `fast_allocators` (per-thread slab arenas and reuse lists), which reduce memory consumption and allocation cost compared with the C library allocator. `-DCHUNK_STD_ALLOC=ON` switches back to the standard allocator (useful under AddressSanitizer).

The design documents, in English and French, are indexed in [docs/README.md](docs/README.md).

## Requirements

- **Linux**, x86-64 or aarch64. The library relies on Linux system calls (`membarrier`, `madvise`, `mincore`) and on `/proc`.
- A **C++20** compiler: GCC 13 or 14 and Clang 18 are tested; the build also covers oneAPI (`icpx`), AOCC and NVHPC. With Clang the build links libc++ by default (`-DUSE_LIBCXX=OFF` for libstdc++).
- **CMake 3.20** or later.
- **OpenMP**, optional (`-DOMP=ON`): a program that uses the arrays from OpenMP regions needs a library built with it. The thread identity is the OpenMP rank: use the arrays from OpenMP threads, not from `std::thread`. Set `OMP_NUM_THREADS` (or `CHUNK_THREAD_CAPACITY`) to the largest team before the first array.
- **OMPT**: with Clang or oneAPI the OpenMP runtime has the OMPT interface, and the library hooks the start and end of the parallel regions itself. GCC's `libgomp` has no OMPT: the program then marks its regions, `begin_parallel_phase()` by every thread at the top of a region and `finalize_parallel_phase()` once after it (see the program below).
- Python 3, for the source checks run by `ctest` (optional).

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
    sa::ParamChunk::init_indexes(true);       // once, before the first array
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
    sa::release_chunk_pools();                // at the end, once every array is gone
    sa::destroy_block_chunk();
}
```

The complete version, which also checks what it reads back, is [examples/getting_started](examples/getting_started/getting_started.cpp); [docs/getting_started.md](docs/getting_started.md) walks through it.

## Using it from CMake

The library you link chooses the variant (`chunk_v1`, `chunk_v2`, `chunk_v2a`) and the chunk size (`_256`, `_1024`, `_dyn` for a size chosen at run time; the bare name is the first static size).

After an install (`cmake --install`):

```cmake
find_package(chunk 2.2 REQUIRED)            # -DCMAKE_PREFIX_PATH=<install prefix>
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

Or from the sources, as part of your build -- only the libraries are built then, no examples and no tests:

```cmake
add_subdirectory(shrink-array)              # or FetchContent_Declare(shrink_array GIT_REPOSITORY <url> GIT_TAG <tag>)
                                            #    FetchContent_MakeAvailable(shrink_array)
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

## Element types and initialisation

An array holds any type that a `std::vector` holds and that is also **trivially copyable**, **equality comparable** and **default constructible**: the chunks are raw memory, copied, parked and replaced by a shared constant buffer without ever running a constructor or a destructor; compression is "every element `==` the constant", with the type's own `operator==` (member-wise for a defaulted one, so padding bytes are not part of the value); and `T{}` is the first constant, the value a fresh array reads everywhere. Every arithmetic type, `bool`, `std::complex`, enums, `std::array` of those and any aggregate of them qualify:

```cpp
struct Point { double x, y, z; auto operator<=>(const Point&) const = default; };
shrink_array::ArrayChunk2A<Point> pts(n);   // compresses runs of {0,0,0} and of any repeated point
```

A type owning a resource (`std::string`, anything with a destructor that matters) is refused at compile time, with the contract in the error (`concept ArrayChunkElement`, `chunk/internal/concept_array_chunk_element.h`). The compound operators of the element proxies (`+=`, `|=`, `++` ...) exist for the types that have them; a struct gets assignment, reads and comparison. The chunk2 variant orders second values and asks a floating-point aggregate for an `operator<` (a defaulted `operator<=>` does); see `chunk/internal/value_less.h`.

No type is declared up front. A program calls, once, before its first array:

```cpp
shrink_array::ParamChunk::init_indexes(true);   // block size, statistics; true = big allocation blocks
```

and each array registers its element type when it is first constructed, from any context (a magic static per type; the per-rank structures and allocators of the type are built then, from the constructing thread). `init_indexes<std::tuple<double, int>>(true)` still registers a list of types up front, for whoever wants every manager in place before the first parallel region. The elements are compared and compressed at `T{}`; chunk2 / chunk2a also collapse runs of a second value, per array (chunk2, `set_second_value`) or per chunk (chunk2a, found by the pass).

### What of `std::vector`'s interface is there

Positional access (`operator[]`, `at`, `front`, `back`), `size`/`empty`/`capacity`/`reserve`, `resize`/`clear`/`assign`/`swap`, `push_back`/`emplace_back`/`pop_back`, const random-access iterators (`begin`/`end`, `rbegin`/`rend`, range-for, `it[n]`, `std::lower_bound`/`binary_search` in O(log n)), the `(first, last)` and `{a, b, c}` constructors, a deep copy (copy constructor and copy-assignment), `shrink_to_fit` -- which, for a container that compresses, folds every chunk that has become constant and gives the parked buffers back. A `push_back` within the capacity is one increment and one write; the one that needs a new chunk goes through `resize`, a sequential point, so `reserve` first when filling incrementally.

What is not there, by construction: `data()` (no contiguous storage: a compressed chunk has no buffer), mutable iterators and `T&` on an element (the non-const `operator[]` returns a write proxy: read, build, assign -- `tab[i].x = 1` does not compile), `insert`/`erase` in the middle, and hence an in-place `std::sort` (it swaps through `T&`): sort a copy -- `std::vector<T> v(a.begin(), a.end())`, `std::sort`, `a.assign(v.begin(), v.end())`. A default-constructed array is empty, as a `std::vector`.

### The write proxy

The non-const `operator[]` returns a *proxy*: a handle on the element that reads it on conversion and routes every write through the compression machinery. It behaves like a value in expressions, with two differences worth knowing.

It is a handle, not a copy. `T v = tab[i];` takes the value; `auto v = tab[i];` keeps the proxy, which then follows the element -- unlike `std::vector`, where `auto` copies: code ported from it wants `T`, not `auto`. The compound operators (`+=` ... `>>=`) and the prefix `++`/`--` return the same kind of proxy: `tab[i] += x;` discards it -- the operation can then be deferred at no cost -- while `T y = (tab[i] += x);` reads the result. The postfix `tab[i]++` returns the old value, as a `T`.

It must not outlive the storage it points into. A proxy holds references into the array's chunks, so it is invalidated by anything that rebuilds or releases them: `resize` (a `push_back` beyond the capacity included), `reserve`, `shrink_to_fit`, `clear`, `assign`, `swap`, an assignment to the array, its destruction -- and a re-chunk, whether you call `rechunk_to` or a `RechunkDecider` decides it for you. Keep proxies short-lived, and take a `T` whenever a value has to survive.

### Threads

Under OpenMP the contract is `std::vector`'s, element by element:

- Different elements need no synchronisation, even when they share a chunk: compression, materialisation and the writes a compressor in flight makes the library defer are the library's business.
- Accesses to the same element from different threads must be ordered by the program, by any means -- a barrier, a `critical` section, a lock, an atomic flag with release/acquire. Ordered, they behave exactly as on a `std::vector`: a deferred write is applied before any later access the program has ordered after it, on whichever thread. Unordered, it is a data race, as it is on a `std::vector`; do not rely on which write wins.
- Everything that rebuilds the storage -- the operations that invalidate proxies above, re-chunks included -- is a sequential point: call it outside any parallel region.

## Compilation

```shell
cmake -S . -B build_dir -DCMAKE_BUILD_TYPE=Release -DOMP=ON
cmake --build build_dir
ctest --test-dir build_dir          # after the build
```

The options a user may want:

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | the compile-time chunk sizes built, one library per size and variant |
| `CHUNK_BUILD_DYN` | `ON` | also build the libraries whose chunk size is chosen at run time (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | the standard allocator for the chunk buffers instead of the bundled `fast_allocators` |
| `SHRINK_ARRAY_BUILD_EXAMPLES` | `ON` at the top level | the examples, benchmarks and validation harnesses (the harnesses in Debug, `CHECK_OPTIM` or sanitizer builds, or with `-DCHUNK_BUILD_VALIDATION=ON`) |
| `SHRINK_ARRAY_BUILD_TESTS` | `ON` at the top level | register the tests with `ctest` |

[scripts/compile_all.sh](./scripts/compile_all.sh) builds the whole set of configurations used during development.

To install the library and use it from your own CMake project, start with [docs/getting_started.md](docs/getting_started.md) (in French: [docs/getting_started_fr.md](docs/getting_started_fr.md)). What changed from one version to the next, and the versioning policy, are in [CHANGELOG.md](CHANGELOG.md). The API on one page: [docs/api.md](docs/api.md); to contribute: [CONTRIBUTING.md](CONTRIBUTING.md).
The library and examples require C++23. They build with g++ (14 or later), LLVM clang (18 or later) and Intel oneAPI (icpx). nvhpc and AOCC 5 have no usable C++23 and are no longer built by default (`compile_all.sh --with-nvhpc` / `--with-aocc`).

## Examples

The [examples](./examples) directory has a `CMakeLists.txt` that shows how to consume the library.

Two examples show how to use the library:
 - The [first](examples/simple_array) compares an implementation with a plain `std::vector` and one with compressed arrays (`ArrayChunk`).
    We can compare the execution speed and memory consumption between the two modes.

 - The [second](examples/multi-mat) is more advanced: it shows the use of this compression on multidimensional arrays. Each material is stored either in compressed arrays or in `std::vector`s: the program asks for the split at start-up (`Option`: 1 all compressed, 2 all `std::vector`, 3 a random draw, 4 a random draw at a given rate; see [InputData.cpp](examples/multi-mat/InputData.cpp)).
   We also show a vectorization technique on these data structures: every loop is written once against a loop policy, scalar for the compressed arrays and `omp simd` for the `std::vector` ones -- see [Loops.h](examples/multi-mat/Loops.h) and [ComputeLoops.h](examples/multi-mat/ComputeLoops.h).

The other directories are tools for developing the library:
 - [shrink_bench](examples/shrink_bench): the reference benchmark, thirteen arrays of different element types under a compute loop, with a `--mode check` that compares every value against `std::vector` reference files and a `--mode perf` for timing;
 - [validation](examples/validation): one harness per property (compression round trip, deferred writes, re-chunk, read/reclaim, nested regions, thread growth...), registered with `ctest`;
 - [advisor_demo](examples/advisor_demo) and [variant_compare](examples/variant_compare): the chunk-size advisor's predictions against real sizes, and Chunk2 against Chunk2A on the same workload.

`python/bench_and_test.py` runs the test and performance campaigns over a set of builds.


## License

This library is under the Apache 2.0 license. See [LICENSE](./LICENSE) for more details.

It includes two header-only hash tables from other projects, under their own licenses:
[ankerl::unordered_dense](https://github.com/martinus/unordered_dense) (MIT) and
[ska::flat_hash_map](https://github.com/skarupke/flat_hash_map) (Boost Software License 1.0).
See [NOTICE](./NOTICE) and [third_party/README.md](./third_party/README.md).

Copyright CEA
