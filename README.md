# Chunk: Compressing Sparse Arrays

## Goals

This c++ library allows you to compress arrays in memory when they contain a significant proportion of zero.

These compressed arrays can used the same way as `std::vector`. Compression management is done by overloading `operator[]`.

The syntax for reading and writing data in these arrays is identical, using `[]`.
However, the implementation of `operator[]` overload is not the same:
 - in the case of reading there are just two indirections and some bitwise computations, 
 - in the case of writing there is also the possible creation of a block via dynamic allocation.

## Implementation

The implementation of this library requires that the number of contiguous zeros be a power of two.
In the [examples](./examples), we use the value of 64, e.g. if there are 64 contiguous zeros, the 64 memory boxes will be transformed into a single one.

There is a `compress` method that scans all elements of the array and transforms contiguous areas from zero (64 in the examples) into a single memory box.

This library supports OpenMP multi-threading. In this case, inserting (writing) elements uses `compare and swap atomic` operations to ensure correctness, which can also be expansive in some cases.

This library is associated with a library allocation a priori a little faster than the standard type allocators of libc. You can activate this libray with the cmake variable CHUNK_STD_ALLOC. Using this library reduces memory consumption, on the example test1 we decrease it by about 15%

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

The compilation is done via `CMake`:

```shell
cmake -S . -B build_dir
cmake --build build_dir
```

See [compile_all.sh](./scripts/compile_all.sh) for more details.
The library and examples require C++23. They build with g++ (14 or later), LLVM clang (18 or later) and Intel oneAPI (icpx). nvhpc and AOCC 5 have no usable C++23 and are no longer built by default (`compile_all.sh --with-nvhpc` / `--with-aocc`).

## Examples

In the [examples](./examples) directory we have a CMakeList.txt file that shows how to consume the library (chunk).

Two examples are proposed:
 - The [first](examples/simple_array) allows to compare an implementation with classical `std::vector` and one with compress arrays (ArrayChunk).
    We can compare the execution speed and memory consumption between the two modes.

 - The [second](examples/multi-mat) is more advanced: it shows the use of this compression on multidimensional arrays. Each material is stored either in compressed arrays or in `std::vector`s: the program asks for the split at start-up (`Option`: 1 all compressed, 2 all `std::vector`, 3 a random draw, 4 a random draw at a given rate; see [InputData.cpp](examples/multi-mat/InputData.cpp)).
   We also show a vectorization technique on these data structures: every loop is written once against a loop policy, scalar for the compressed arrays and `omp simd` for the `std::vector` ones -- see [Loops.h](examples/multi-mat/Loops.h) and [ComputeLoops.h](examples/multi-mat/ComputeLoops.h).


## License

This library is under the Apache 2.0 license. See [LICENSE](./LICENSE) for more details.

Copyright CEA
