# Getting started with shrink-array

> October 2026. This guide is about **using** the library. How it works inside is described
> in the design documents, indexed in [README.md](README.md). A French version of
> this guide is [getting_started_fr.md](getting_started_fr.md).

***

## 1. What it is

shrink-array provides arrays that behave like a `std::vector` and take less memory when
large parts of them hold the same value. An array is cut into **chunks** of `S` consecutive
elements (`S` a power of two or a multiple of 64, 256 by default). A chunk whose elements all hold one constant
keeps no buffer of its own. It points to a shared buffer of that constant and costs only a
few bytes of metadata. Writing another value into it gives it a buffer again. While the
program runs, chunks that have become uniform are compressed again, and their buffers go
back to the allocator.

It pays off on data that is mostly zero, or mostly made of runs of a few values: sparse
fields, material masks, multi-material quantities, flags. It does not on data where every
chunk holds varied values. There it costs a little more memory than a `std::vector`, and
every access is slower.

Three variants, chosen when you link:

| variant | class | a chunk is compressed when all its elements equal… |
|---|---|---|
| `zero` | `zero::Array<T>` | `T{}` (zero) only |
| `zero_one_value` | `zero_one_value::Array<T>` | `T{}`, or **one** second value per array, declared with `set_second_value()` |
| `zero_any` | `zero_any::Array<T>` | `T{}`, or **any** value, chunk by chunk, with nothing to declare |

`zero_any` compresses the most and is the one to try first. `zero_one_value` has lighter metadata when
the data has a single dominant non-zero value, and `zero` is the simplest when only zeros
matter. [variants.md](variants.md) compares them on a real workload.

## 2. Install

With Spack, the recipe is in the repository: see [the README](../README.md#with-spack)
(`spack install shrink-array`). From the sources, with a C++20 compiler (tested with GCC 13
and 14 and Clang 18; the build scripts also cover oneAPI, AOCC and NVHPC; the `cpp23`
branch needs C++23, GCC 14 or later; the `cpp26` branch, GCC 16 and no other compiler) and
CMake ≥ 3.20:

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
| `CHUNK_THREADS` | empty | threading backend of the library: `openmp`, `std` (`std::jthread` workers of its own, no OpenMP) or `serial`; empty: `openmp` with `OMP=ON`, `serial` otherwise. Set, it sets `OMP` to match |
| `CHUNK_STATIC_SIZES` | `auto` | the compile-time chunk sizes built, one library per size and variant; `auto`: `256;1024`, but `256` alone in a Debug or sanitizer build of this project (not of a project that includes it) |
| `CHUNK_BUILD_DYN` | `ON` | also build the libraries whose chunk size is chosen at run time, and can change (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | allocate the chunk buffers with the standard allocator instead of the bundled `fast_allocators` (useful under AddressSanitizer) |

The first build compiles the examples and the benchmark too, which takes a while. To build
only the libraries, configure with `-DSHRINK_ARRAY_BUILD_EXAMPLES=OFF`, or give
`cmake --build` their targets (`--target chunk_zero_any_256 ...`) before `cmake --install`. Pulled
into another project with `add_subdirectory` or `FetchContent`, it builds only the libraries.

With Clang or oneAPI, an OpenMP build also uses the runtime's **OMPT** interface, which spares
your program the phase calls of §5.

## 3. Link it from CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(chunk 0.1 REQUIRED)        # -DCMAKE_PREFIX_PATH=<install prefix>

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE chunk::chunk_zero_any)
```

The library you link **chooses the variant and the chunk size**. The headers are the same
for all of them, and the library brings the compile definitions that select its own
(`CHUNK_VARIANT_ZERO_ANY=1`, `CHUNK_SIZE=256` or `CHUNK_DYNAMIC`), together with C++20,
`-fopenmp` when it was built with OpenMP, and its dependencies. The other options it was built
with are in the installed `chunk/build_options.h`, which the headers read: a program compiles them
as the library did.

| target | variant | chunk size |
|---|---|---|
| `chunk::chunk_zero`, `chunk::chunk_zero_one_value`, `chunk::chunk_zero_any` | `zero`, `zero_one_value`, `zero_any` | the first of `CHUNK_STATIC_SIZES` (256) |
| `chunk::chunk_zero_any_1024`, … | | that compile-time size |
| `chunk::chunk_zero_any_dyn`, … | | chosen at run time, per array (can be changed with `rechunk_to()` or by the adaptive decider) |
| `chunk::chunk_vector` | none: the arrays are `std::vector` | — |

Link **one** variant library per program, and write **that variant's class**:
`zero_any::Array<T>` with `chunk::chunk_zero_any`. The class you write is what you get -- a program
naming `zero_any::Array` keeps it regardless of what it links -- and an array of a variant other than the
library's is not supported: depending on the calls, it does not compile or it crashes.
To let the link choose, write the arrays through an alias for the variant the library
defines (`examples/shrink_bench` and `examples/simple_array` do so):

```cpp
#include <type_traits>
#include <vector>
#include "chunk/zero_array.h"
#include "chunk/zero_one_value_array.h"
#include "chunk/zero_any_array.h"
#include "chunk/config.h"

namespace sa = shrink_array;
template <typename T>
using Array = std::conditional_t<sa::chunk_config::use_zero_any, sa::zero_any::Array<T>,
              std::conditional_t<sa::chunk_config::use_zero_one_value, sa::zero_one_value::Array<T>,
              std::conditional_t<sa::chunk_config::use_zero, sa::zero::Array<T>, std::vector<T>>>>;
```

With it, `chunk::chunk_vector` builds the same program on plain `std::vector`s, which is the
easiest way to compare memory and time.

`find_package(chunk)` also sets `chunk_STATIC_SIZES`, `chunk_DEFAULT_SIZE`, `chunk_HAS_DYN`,
`chunk_OPENMP`, `chunk_OMPT` and `chunk_STD_ALLOC`, for a project that wants to check what the
installation holds.

## 4. A first program, parallel and multi-threaded

This is [`examples/getting_started/getting_started.cpp`](../examples/getting_started/getting_started.cpp),
shortened. The directory also has the `CMakeLists.txt` of §3: it builds against an install as
well as inside the tree.

```cpp
#include <cstdio>
#include <utility>                 // std::as_const

#include "chunk/zero_any_array.h"   // shrink_array::zero_any::Array
#include "chunk/zero_chunk.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

namespace sa = shrink_array;

int main()
{
    sa::ParamChunk::init_indexes(true);          // (1) once, before the first array

    {
        const int n = 1 << 24;                   // 16 M doubles: 128 MiB as a std::vector
        sa::zero_any::Array<double> a(n);           // (2) all 0.0, nothing allocated yet

        sa::parallel_phase([&] {                 // (3) a parallel region, phases marked
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
                a[i] = i < n / 8 ? 1.0 + i % 7   //     varied values: these chunks keep a buffer
                                 : 3.5;          //     one constant: these chunks compress
        });

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
3. **`parallel_phase()`** opens a parallel region and runs the lambda on every thread. Without
   OMPT (§5) it also marks the phase for the library: `begin_parallel_phase()` on every thread
   before the lambda, `finalize_parallel_phase()` after the region. With OMPT, and in a serial
   build, those two calls compile to nothing.
4. **Reading**: through a `const` array, `operator[]` returns a `T`. On a non-const array it
   returns a *proxy* (§6).
5. **`shrink_to_fit()`** runs a full compression pass, outside any parallel region. You do not
   need to call it: chunks are compressed as the program runs. It is there for when you want
   the memory back at a specific point.
6. **Destroying the array** gives its buffers back. The library prints one line per array at
   that point, with its compression ratio.
7. **The shutdown pair**, at the end of the program, after the last array. Optional too: the
   process can simply exit. With an array still alive (a global or `static` one, destroyed
   after `main` returns), the pair keeps what that array will need and only empties the free
   lists.

Expected output: about 12 % of the chunks hold a buffer. A `std::vector` would hold all of
them.

## 5. Threads: the rules

**What you can do without synchronisation.** As with a `std::vector`, different threads can
read and write **different elements** at the same time, even elements of the same chunk:
compression and materialisation are the library's business. Two accesses to the **same
element** from different threads must be ordered by your program (a barrier, a `critical`
section, a lock, release/acquire atomics), exactly as on a `std::vector`.

**With OMPT, the `std::vector` loop as it is.** With a library built with OMPT (Clang, oneAPI),
the same OpenMP loop fills a `std::vector` and a compressed array: no `parallel_phase()`, no
phase call. Linked as in §3, with `OpenMP::OpenMP_CXX` too:

```cpp
#include <cstdio>
#include <utility>                 // std::as_const
#include <vector>

#include "chunk/zero_any_array.h"   // shrink_array::zero_any::Array

namespace sa = shrink_array;

template <typename Array>
void fill(Array& t, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n / 2; ++i)
        t[i] = (i < n / 8) ? 1.0 + i % 7 : 3.5;
}

int main()
{
    const int n = 1 << 24;

    std::vector<double>      v(n);
    sa::zero_any::Array<double> a(n);
    fill(v, n);                          // the reference
    fill(a, n);                          // the same loop: no parallel_phase(), no phase calls

    // Read through a const view: on a non-const array, a[i] is a write proxy.
    const auto& ca  = std::as_const(a);
    long        bad = 0;
    for (int i = 0; i < n; ++i)
        bad += (ca[i] != v[i]);

    std::printf("%s: %ld element(s) differ out of %d\n", bad == 0 ? "OK" : "FAIL", bad, n);
    return bad == 0 ? 0 : 1;
}
```

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

Forget them and nothing fails right away. With the same team from region to region the values
still come out right, but the compression passes between regions do not run: memory stays
higher than it should. When the team size changes from one region to the next, a forgotten
`finalize_parallel_phase()` replays a deferred write that was already freed, and with both
calls forgotten, values come out wrong, silently. The program stops, in every build, on a
`begin_parallel_phase()` with no `finalize_parallel_phase()` since the previous region; check
builds (Debug, `CHECK_OPTIM`) also stop on a deferred write by a thread that skipped
`begin_parallel_phase()`.
Writing them behind `if constexpr (sa::build_config::has_not_ompt)`, as above, keeps one
source for both runtimes.

`shrink_array::parallel_phase()` writes the region and both calls for you, and leaves them out
under OMPT:

```cpp
shrink_array::parallel_phase([&] {            // or parallel_phase(n_threads, [&] { ... })
#pragma omp for
    for (int i = 0; i < n; ++i)
        a[i] = f(i);
});
```

With OMPT, if the runtime did not load the library's tool (`OMP_TOOL=disabled`, another tool
found first), the phases never happen. The library prints a warning on stderr;
`CHUNK_REQUIRE_OMPT=1` makes it stop the program instead.

With OMPT you may also write no phase at all: plain `#pragma omp parallel for` loops, the same
code as on a `std::vector`.
[`examples/getting_started/getting_started_ompt.cpp`](../examples/getting_started/getting_started_ompt.cpp)
is the program of §4 written that way: one function fills a `std::vector<double>` and a
`zero_any::Array<double>`, and the program checks that they read back the same. It refuses to
compile against a library without OMPT; its test runs with `CHUNK_REQUIRE_OMPT=1`.

**Sequential points.** Everything that rebuilds an array's storage must be called outside any
parallel region: `resize()`, `reserve()`, `push_back()` beyond the capacity, `clear()`,
`assign()`, `swap()`, assignment, `shrink_to_fit()`, `compress()`, `rechunk_to()`, and, for
`zero_one_value`, `set_second_value()`.

**Threads.** The per-thread structures are sized once for `max(omp_get_max_threads(), number
of cores, 256)` threads (in a std build, the team size -- `CHUNK_NUM_THREADS`, else
`OMP_NUM_THREADS` -- stands for `omp_get_max_threads()`; about 1 KiB per element type and per
thread). A program that runs
more threads than that sets `CHUNK_THREAD_CAPACITY=<n>`; a thread beyond the capacity stops the
program with a message.

**Next to other components.** The library keeps its state per process, not per array: the
element-type registries, the allocators, `zero_any`'s second-value registry, the per-thread
structures and their capacity, the log settings, and the phase. Two parts of a program that use
it without knowing about each other -- your code and a library built on shrink-array, say -- share
all of it. What that asks of them:

- **One phase at a time.** The per-thread queues are indexed by the rank in the team, and the
  end of a phase drains every rank's queue. Two top-level OpenMP teams writing arrays at the
  same time -- opened by two host threads, two components each on its own `std::thread` --
  share ranks 0 to n-1, and one team's phase end replays the other's writes in the middle of
  its region. Once an array exists, the second team stops the program with a message (OMPT:
  when it begins; without OMPT: at its `begin_parallel_phase()`). Run such parts one after the
  other, or in one team.
- **Nested regions are fine.** A solver or a BLAS call that opens its own region inside a
  phase, a component that calls `parallel_phase()` inside yours: only the outermost region's
  end is a phase boundary, and `finalize_parallel_phase()` does nothing inside an active
  region. The arrays may be written in a nested region while it is serialised
  (`OMP_MAX_ACTIVE_LEVELS=1`, the default); with two active levels a write stops the program.
- **The first array fixes the thread capacity** (above): at least 256 threads, or the cores, or
  `OMP_NUM_THREADS`; a component that opens larger teams needs `CHUNK_THREAD_CAPACITY` set
  before the first array.
- **Process-wide calls.** `ParamChunk::init_indexes()` sets the allocation block size of the
  whole process: one call, at the start, by the program (a later one leaves the arrays already
  built intact). `release_chunk_pools()` and `destroy_block_chunk()` empty everyone's free
  lists (an array still alive keeps what it needs): once, at the end of the program, or not at
  all. `reclaim_second_values()` covers every array of a type; `CHUNK_LOG` and
  `logging::set_sink()` are one setting for all.

**Without OpenMP: your own threads, TBB.** A library built with `CHUNK_THREADS=std` runs its own
work on a pool of `std::jthread`s and takes writes from any thread of the program: `std::thread`,
TBB, a thread pool of your own. The phases are then always explicit: every thread that writes
calls `begin_parallel_phase()` before its first write, and **one** thread calls
`finalize_parallel_phase()` once every writer has stopped (after the `join()`):

```cpp
std::vector<std::thread> workers;
for (int t = 0; t < nt; ++t)
    workers.emplace_back([&, t] {
        sa::begin_parallel_phase();               // every thread that writes
        for (long i = t; i < n; i += nt)
            a[i] = f(i);
    });
for (auto& w : workers)
    w.join();
sa::finalize_parallel_phase();                    // once, after the join
```

A TBB program uses `shrink_array::tbb::parallel_for` (`chunk/tbb_adaptor.h`): the arguments of
`oneapi::tbb::parallel_for`, both calls made for you. The header comes with the library; the
program links TBB itself (`TBB::tbb`), the library does not:

```cpp
#include "chunk/tbb_adaptor.h"
sa::tbb::parallel_for(oneapi::tbb::blocked_range<long>(0, n), [&](const auto& r) {
    for (long i = r.begin(); i != r.end(); ++i)
        a[i] = f(i);
});
```

Between `begin` and `finalize` the library counts the program's threads as a parallel region
(the sequential points above must not be called there, as inside an OpenMP region), and each of them
holds one of the per-thread slots while it is alive. An OpenMP build numbers only OpenMP's
threads: for the others, use the std backend.

## 6. Pitfalls

- **`auto` keeps a proxy.** On a non-const array, `tab[i]` returns a write proxy. `T v = tab[i];`
  takes the value; `auto v = tab[i];` keeps the proxy, which follows the element if it changes
  later. In code ported from `std::vector`, replace `auto` with the element type. Every build
  refuses to compile a use of such a `v`, and says so.
- **Convert an element before handing it on.** `std::printf("%f", tab[i])` would pass the proxy,
  not the value: refused (GCC through `-Werror=conditionally-supported`, which the library sets;
  clang always). `std::format("{}", tab[i])`, `std::max(tab[i], 2.0)` and a generic function
  that takes `tab[i]` by value are refused too. Write `double(tab[i])`, or `T(tab[i])`.
- **No `T&` on an element.** `tab[i].x = 1` does not compile; read the element, change it,
  write it back. `data()`, mutable iterators and an in-place `std::sort` do not exist either:
  sort a `std::vector` copy and `assign()` it back.
- **A proxy must not outlive the storage.** Any of the sequential operations of §5 invalidates
  the proxies of that array. Keep proxies short-lived.
- **Global and `static` arrays work**: built before `main`, destroyed after it returns. The
  library's own state is built at its first use and never destroyed, so the order of the
  static initialisations does not matter.
- **Indices are `int` by default.** For more than 2³¹ elements, use the second template
  parameter: `zero_any::Array<double, long>`.
- **Element types**: anything trivially copyable, `==`-comparable and default-constructible
  (arithmetic types, `bool`, `std::complex`, enums, `std::array` and aggregates of those). A
  type that owns a resource (`std::string`) is refused at compile time. `zero_one_value` also needs an
  order on the second value: a struct of doubles declares `auto operator<=>(const T&) const =
  default;`.
- **`zero_one_value` compresses only to its declared second value.** Without `set_second_value()`, a
  `zero_one_value` array compresses to zero only.
- **Measure memory with the right tool.** `VmRSS` lags behind on many threads; see
  [performance_measurement.md](performance_measurement.md).

## 7. Going further

| for… | read |
|---|---|
| the full container interface | the main [README.md](../README.md), and the Doxygen documentation (`cmake --build build --target doc`) |
| choosing a variant on your data | [variants.md](variants.md), and `examples/variant_compare` |
| letting the chunk size adapt at run time | [adaptive_rechunk.md](adaptive_rechunk.md) (`_dyn` libraries) |
| every tuning variable and build option | [environment_variables.md](environment_variables.md) |
| how it works inside | [algo_zero_one_value.md](algo_zero_one_value.md), [synchronisation_dekker.md](synchronisation_dekker.md) |
| a larger example: multi-material fields, loops written once for both containers | `examples/multi-mat` |
