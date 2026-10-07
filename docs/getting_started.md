# Getting started with shrink-array

> October 2026. This guide is about **using** the library. How it works inside is described
> in the design documents, indexed in [README.md](README.md). A French version of
> this guide is [getting_started_fr.md](getting_started_fr.md).

***

## 1. What it is

shrink-array provides arrays that behave like a `std::vector` and take less memory when
large parts of them hold the same value. An array is cut into **chunks** of `S` consecutive
elements (`S` a power of two, 256 by default). A chunk whose elements all hold one constant
keeps no buffer of its own. It points to a shared buffer of that constant and costs only a
few bytes of metadata. Writing another value into it gives it a buffer again. While the
program runs, chunks that have become uniform are compressed again, and their buffers go
back to the allocator.

It pays off on data that is mostly zero, or mostly made of runs of a few values: sparse
fields, material masks, multi-material quantities, flags. It does not on data where every
chunk holds varied values. There it costs a little more memory than a `std::vector`, and
every access is slower.

Three variants, chosen when you link:

| variant | class | a chunk is compressed when it is uniform at… |
|---|---|---|
| Chunk (v1) | `ArrayChunk<T>` | `T{}` (zero) only |
| Chunk2 | `ArrayChunk2<T>` | `T{}`, or **one** second value per array, declared with `set_second_value()` |
| Chunk2A | `ArrayChunk2A<T>` | `T{}`, or **any** value, chunk by chunk, with nothing to declare |

Chunk2A compresses the most and is the one to try first. Chunk2 has lighter metadata when
the data has a single dominant non-zero value, and Chunk (v1) is the simplest when only zeros
matter. [variants.md](variants.md) compares them on a real workload.

## 2. Install

Spack recipe: to come (`spack install shrink-array`). Meanwhile, from the sources, with a
C++20 compiler (tested with GCC 13 and 14 and Clang 18; the build scripts also cover
oneAPI, AOCC and NVHPC) and CMake ≥ 3.20:

```sh
cmake -S shrink-array -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOMP=ON \
      -DCMAKE_INSTALL_PREFIX=$HOME/opt/shrink-array
cmake --build build
cmake --install build
```

The options that matter to a user:

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP. A program that uses the arrays from OpenMP regions needs a library built with it |
| `CHUNK_STATIC_SIZES` | `256;1024` | the compile-time chunk sizes built, one library per size and variant |
| `CHUNK_BUILD_DYN` | `ON` | also build the libraries whose chunk size is chosen at run time, and can change (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | allocate the chunk buffers with the standard allocator instead of the bundled `fast_allocators` (useful under AddressSanitizer) |

The first build compiles the examples and the benchmark too, which takes a while. To build
only the libraries, configure with `-DSHRINK_ARRAY_BUILD_EXAMPLES=OFF`, or give
`cmake --build` their targets (`--target chunk_v2a_256 ...`) before `cmake --install`. Taken
into another project with `add_subdirectory` or `FetchContent`, the libraries alone are built.

With Clang or oneAPI, an OpenMP build also uses the runtime's **OMPT** interface, which spares
your program the phase calls of §5.

## 3. Link it from CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(chunk 2.2 REQUIRED)        # -DCMAKE_PREFIX_PATH=<install prefix>

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

The library you link **chooses the variant and the chunk size**. The headers are the same
for all of them, and the library brings the compile definitions that select its own
(`CHUNK_VARIANT_V2A=1`, `CHUNK_SIZE=256` or `CHUNK_DYNAMIC`), together with C++20,
`-fopenmp` when it was built with OpenMP, and its dependencies. The other options it was built
with are in the installed `chunk/build_options.h`, which the headers read: a program compiles them
as the library did.

| target | variant | chunk size |
|---|---|---|
| `chunk::chunk_v1`, `chunk::chunk_v2`, `chunk::chunk_v2a` | v1, Chunk2, Chunk2A | the first of `CHUNK_STATIC_SIZES` (256) |
| `chunk::chunk_v2a_1024`, … | | that compile-time size |
| `chunk::chunk_v2a_dyn`, … | | chosen at run time, per array (can be changed with `rechunk_to()` or by the adaptive decider) |
| `chunk::chunk_vector` | none: the arrays are `std::vector` | — |

Link **one** variant library per program. `chunk::chunk_vector` builds the same program on
plain `std::vector`s, which is the easiest way to compare memory and time.

`find_package(chunk)` also sets `chunk_STATIC_SIZES`, `chunk_DEFAULT_SIZE`, `chunk_HAS_DYN`,
`chunk_OPENMP`, `chunk_OMPT` and `chunk_STD_ALLOC`, for a project that wants to check what the
installation holds.

## 4. A first program

This is [`examples/getting_started/getting_started.cpp`](../examples/getting_started/getting_started.cpp),
shortened. The directory also has the `CMakeLists.txt` of §3: it builds against an install as
well as inside the tree.

```cpp
#include <cstdio>
#include <utility>                 // std::as_const

#include "chunk/array_chunk2A.h"   // shrink_array::ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // sa::build_config::has_not_ompt
#include "chunk/manage_ompt.h"     // begin_parallel_phase, finalize_parallel_phase
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

namespace sa = shrink_array;

static void begin_phase() { if constexpr (sa::build_config::has_not_ompt) sa::begin_parallel_phase(); }
static void end_phase()   { if constexpr (sa::build_config::has_not_ompt) sa::finalize_parallel_phase(); }

int main()
{
    sa::ParamChunk::init_indexes(true);          // (1) once, before the first array

    {
        const int n = 1 << 24;                   // 16 M doubles: 128 MiB as a std::vector
        sa::ArrayChunk2A<double> a(n);           // (2) all 0.0, nothing allocated yet

#pragma omp parallel
        {
            begin_phase();                       // (3) every thread, top of the region
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
                a[i] = i < n / 8 ? 1.0 + i % 7   //     varied values: these chunks keep a buffer
                                 : 3.5;          //     one constant: these chunks compress
        }
        end_phase();                             // (3) once, after the region

        const auto& ca = std::as_const(a);       // (4) read through a const reference
        double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < n; ++i)
            sum += ca[i];

        a.shrink_to_fit();                       // (5) fold every chunk that became constant
        std::printf("sum = %.1f, chunks with a buffer: %.1f %%\n", sum, a.stats());
    }                                            // (6) the array releases its chunks

    sa::release_chunk_pools();                   // (7) shut the library down
    sa::destroy_block_chunk();
}
```

1. **`init_indexes(true)`**, once, before the first array: it sets up the allocation blocks
   (`true`: large blocks, the usual choice). Optional: without it, the first array of the
   program runs `init_indexes(true)` itself. Element types need no declaration: each one is
   registered the first time an array of it is built.
2. **A new array** reads `T{}` everywhere and allocates no buffer.
3. **The phase calls**, needed only without OMPT (§5). They compile to nothing when the
   library was built with OMPT, and in a serial build.
4. **Reading**: through a `const` array, `operator[]` returns a `T`. On a non-const array it
   returns a *proxy* (§6).
5. **`shrink_to_fit()`** runs a full compression pass, outside any parallel region. You do not
   need to call it: chunks are compressed as the program runs. It is there for when you want
   the memory back at a precise point.
6. **Destroying the array** gives its buffers back. The library prints one line per array at
   that point, with its compression ratio.
7. **The shutdown pair**, at the end of the program, after the last array. Optional too: the
   process can simply exit. With an array still alive (a global or `static` one, destroyed
   after `main` returns), the pair keeps what that array will need and only empties the free
   lists.

Expected output: about 12 % of the chunks hold a buffer. A `std::vector` would hold all of
them.

## 5. OpenMP: the rules

**What you can do without synchronisation.** As with a `std::vector`, different threads can
read and write **different elements** at the same time, even elements of the same chunk:
compression and materialisation are the library's business. Two accesses to the **same
element** from different threads must be ordered by your program (a barrier, a `critical`
section, a lock, release/acquire atomics), exactly as on a `std::vector`.

**Phases, without OMPT.** When a writer meets a compression in progress on its chunk, the
library **defers** its write and replays it later. It also compresses between parallel regions.
With OMPT (Clang, oneAPI builds) the OpenMP runtime tells the library where regions begin and
end. Without it (GCC's libgomp), your program does:

```cpp
#pragma omp parallel
{
    shrink_array::begin_parallel_phase();     // every thread, before its first access
    // ... omp for loops on the arrays ...
}
shrink_array::finalize_parallel_phase();      // once, after the region
```

Forget them and nothing fails at once, but deferred writes are only applied when the element
is next read, and the compression passes between regions do not run: memory stays higher than
it should. Writing them behind `if constexpr (sa::build_config::has_not_ompt)`, as above, keeps
one source for both runtimes.

With OMPT, if the runtime did not load the library's tool (`OMP_TOOL=disabled`, another tool
found first), the phases never happen. The library prints a warning on stderr;
`CHUNK_REQUIRE_OMPT=1` makes it stop the program instead.

**Sequential points.** Everything that rebuilds an array's storage must be called outside any
parallel region: `resize()`, `reserve()`, `push_back()` beyond the capacity, `clear()`,
`assign()`, `swap()`, assignment, `shrink_to_fit()`, `compress()`, `rechunk_to()`, and, for
Chunk2, `set_second_value()`.

**Threads.** The per-thread structures are sized once for `max(omp_get_max_threads(), number
of cores)` threads. A program that runs more threads than cores sets
`CHUNK_THREAD_CAPACITY=<n>`; a thread beyond the capacity stops the program with a message.

## 6. Pitfalls

- **`auto` keeps a proxy.** On a non-const array, `tab[i]` returns a write proxy. `T v = tab[i];`
  takes the value; `auto v = tab[i];` keeps the proxy, which follows the element if it changes
  later. In code ported from `std::vector`, replace `auto` with the element type.
- **No `T&` on an element.** `tab[i].x = 1` does not compile; read the element, change it,
  write it back. `data()`, mutable iterators and an in-place `std::sort` do not exist either:
  sort a `std::vector` copy and `assign()` it back.
- **A proxy must not outlive the storage.** Any of the sequential operations of §5 invalidates
  the proxies of that array. Keep proxies short-lived.
- **Global and `static` arrays work**: built before `main`, destroyed after it returns. The
  library's own state is built at its first use and never destroyed, so the order of the
  static initialisations does not matter.
- **Indices are `int` by default.** For more than 2³¹ elements, use the second template
  parameter: `ArrayChunk2A<double, long>`.
- **Element types**: anything trivially copyable, `==`-comparable and default-constructible
  (arithmetic types, `bool`, `std::complex`, enums, `std::array` and aggregates of those). A
  type that owns a resource (`std::string`) is refused at compile time. Chunk2 also needs an
  order on the second value: a struct of doubles declares `auto operator<=>(const T&) const =
  default;`.
- **Chunk2 compresses only to its declared second value.** Without `set_second_value()`, a
  Chunk2 array compresses to zero only.
- **Measure memory with the right tool.** `VmRSS` lags behind on many threads; see
  [performance_measurement.md](performance_measurement.md).

## 7. Going further

| for… | read |
|---|---|
| the full container interface | the main [README.md](../README.md), and the Doxygen documentation (`cmake --build build --target doc`) |
| choosing a variant on your data | [variants.md](variants.md), and `examples/variant_compare` |
| letting the chunk size adapt at run time | [adaptive_rechunk.md](adaptive_rechunk.md) (`_dyn` libraries) |
| every tuning variable and build option | [environment_variables.md](environment_variables.md) |
| how it works inside | [algo_chunk2.md](algo_chunk2.md), [synchronisation_dekker.md](synchronisation_dekker.md) |
| a larger example: multi-material fields, loops written once for both containers | `examples/multi-mat` |
