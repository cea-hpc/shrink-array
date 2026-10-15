# A multidimensional view: mdspan

> French version: [mdspan_fr.md](mdspan_fr.md).

> October 2026. `chunk/mdspan.h` puts a C++23 `mdspan` on top of an array of the library. The
> array keeps its role, the drop-in replacement for `std::vector`; the view adds `f[i, j]`.

***

## 1. What it is

The array stays the owner: it holds the elements, compresses them, re-chunks them and keeps its
whole interface. The view is an `mdspan` whose accessor goes through the array's `operator[]`, so
`f[i, j] = v` is `a[k] = v` with `k` given by the layout: the same write proxy, the same
compression, the same rules.

```cpp
#include "chunk/array_chunk2A.h"
#include "chunk/mdspan.h"

namespace sa = shrink_array;

sa::ArrayChunk2A<double> a(nx * ny);
auto f = sa::md::view(a, nx, ny);         // row-major: f[i, j] is a[i * ny + j]
f[i, j] = 1.0;
f[i, j] += 2.0;                           // compound operators, through the proxy
double x = f[i, j];                       // a value, not auto (see section 5)

const auto& ca = a;
auto cf = sa::md::view(ca, nx, ny);       // read-only: f[i, j] is a double
```

| call | gives |
|---|---|
| `md::view(a, e0, e1, ...)` | an `mdspan` in `layout_right` (row-major) of extents `e0, e1, ...`; their product must not exceed `a.size()` |
| `md::tiled_view<T0, T1, ...>(a, e0, e1, ...)` | an `mdspan` in `md::layout_tiled<T0, T1, ...>`: tiles of `T0 x T1 x ...` elements stored one after the other |
| `md::tiled_size<T0, T1, ...>(e0, e1, ...)` | the size the array needs for that tiled view, padding of the last tiles included |
| `md::view_type<Array, Layout, Rank>` | the type of these views; `view_type<const Array, ...>` is the read-only one, and converts from the writable one |
| `md::accessor<Array>`, `md::layout_tiled<...>` | the accessor and the layout, for building an `mdspan` by hand |

## 2. Tiles: why the layout matters for compression

A chunk is a run of consecutive elements (256 in a `_256` library). In `layout_right` it is a
piece of a row: a uniform 2D region only gives uniform chunks where it is at least 256 cells
wide. With `layout_tiled<16, 16>`, tile `k` is chunk `k`: a 16 x 16 square inside one material
is a uniform chunk, whatever the width of the material. Choose the tile so that its element
count is the chunk size: 16 x 16 for 256, 32 x 32 for 1024, 8 x 8 x 4 for 256 in 3D.

`examples/mdspan_tiles` writes the same field (uniform background, disks and narrow bands of one
material each, and a noisy corner that compresses in neither layout) through both views of a
`chunk::chunk_v2a_256` array, then counts the chunks left materialised after the compression
pass:

| field 2048 x 2048 doubles (32 MiB) | chunks | materialised | values held |
|---|---|---|---|
| row-major (`layout_right`) | 16385 | 7449 (45.5 %) | 14.5 MiB |
| tiles 16 x 16 (`layout_tiled<16, 16>`) | 16385 | 2464 (15.0 %) | 4.8 MiB |

1521 of the 2464 tiles left are the noisy corner; the others straddle a boundary between
materials. The figure depends on the field: a field made of layers wider than a chunk gains
nothing from tiles, a field of small blobs gains most.

## 3. Tile by tile: `for_each_tile`, `update_tiles`, `fill_box`

`f[i, j]` pays an `operator[]` per element. In a tiled view a tile is a run of consecutive
elements -- one chunk when it has `chunk_size()` elements -- so `chunk/mdspan_tile_ops.h` reads
it in one go (a block copy, or the constant of a compressed chunk) and writes it back in one go.

```cpp
#include "chunk/mdspan_tile_ops.h"

auto f = sa::md::tiled_view<16, 16>(a, nx, ny);
double s = 0;
sa::md::for_each_tile(std::as_const(f),
    [&](const auto& t, auto b) {                  // a tile of several values
        for (long i = 0; i < t.extent[0]; ++i)
            for (long j = 0; j < t.extent[1]; ++j) s += b[i, j];
    },
    [&](const auto& t, double v) {                // a uniform tile: one call
        s += v * double(t.extent[0] * t.extent[1]);
    });

sa::md::update_tiles(f,                           // f = 0.5 f + 1
    [](const auto&, auto b) {
        for (long i = 0; i < 16; ++i)
            for (long j = 0; j < 16; ++j) b[i, j] = 0.5 * b[i, j] + 1.0;
    },
    [](const auto&, double& v) { v = 0.5 * v + 1.0; return true; });
```

| call | does |
|---|---|
| `md::for_each_tile(f, on_block, on_uniform)` | `on_block(t, b)` for a tile of several values, `b` an `mdspan<const T, extents<16, 16>>` over a copy of the tile; `on_uniform(t, v)` for a uniform tile. Without `on_uniform`, `on_block` gets every tile |
| `md::update_tiles(f, on_block, on_uniform)` | the same, `b` writable; the tile is written back only if it changed, as one constant if it came out uniform. `on_uniform(t, v)` changes `v` and returns true, or returns false to get the tile as a block filled with `v` |
| `md::for_each_tile(f, first, last, ...)`, `md::update_tiles(f, first, last, ...)` | tiles `[first, last)` in storage order, for a loop `#pragma omp for` over `[0, md::tile_count(f))` |
| `md::fill_box(f, lo, hi, v)` | sets the box `[lo, hi)`, clipped to the extents, to `v`. Tiled view: a tile the box covers is filled whole, padding included, one range op per run of consecutive covered tiles; a tile it crosses is updated as by `update_tiles` (left alone if already uniform at `v`). Row-major view (`md::view`): one range op per contiguous run, several rows at once when the box spans whole rows |
| `md::fill_box(f, lo, hi, v, first, last)` | tiled view, tiles `[first, last)` only: each thread its own slice of the tiles, in a phase |
| `t.number`, `t.origin`, `t.extent` | the tile's rank in storage order, the multi-index of its first element, the part of it inside the view (smaller than the tile on the padded edges) |

The block covers the whole tile, padding included: loop to `t.extent` where the padding must not
count. A uniform tile left unchanged by `update_tiles` is not written, so a kernel that leaves the
background alone does not materialise it. A changed tile is written with `fill_range()` when it
is uniform, else `copy_range()` (the serial range ops outside a parallel region, the `_mt` ones
inside). Outside a parallel region, on Chunk2A, a tile of whole chunks that comes out uniform is
compressed at once, one buffer at a time (`fill_range`). Otherwise a compressed chunk that changes
is materialised for the write and queued: the end of the phase, or the next compression pass,
compresses it again.

In a parallel region each thread works on its own tiles, inside a phase for `update_tiles`, and a
tile must be made of whole chunks: tile elements a multiple of `chunk_size()`, checked (a tile
sharing a chunk with another thread's stops the program). Outside a region any chunk size works.

`examples/mdspan_tiles`, same 2048 x 2048 field in 16 x 16 tiles (85 % of them uniform), Release
build, GCC 14, on a 2-core container (orders of magnitude, not a benchmark):

| kernel | threads | element by element, `f[i, j]` | tile by tile | ratio |
|---|---|---|---|---|
| sum of the field | 1 | 23.5 ms | 1.6 ms | x 14.6 |
| sum of the field | 2 | 11.4 ms | 1.35 ms | x 8.5 |
| `f = 0.5 f + 1` | 1 | 125.8 ms | 6.3 ms | x 19.9 |
| `f = 0.5 f + 1` | 2 | 64.1 ms | 4.4 ms | x 14.7 |
| 64 rectangles painted, `md::fill_box` | 1 | 103.3 ms | 8.8 ms | x 11.7 |
| 64 rectangles painted, `md::fill_box` | 2 | 72.0 ms | 7.0 ms | x 10.4 |

Each pair leaves the same field and the same materialised chunks: 2464 after the sum and the
update, 3109 after the painting.

The gain is in skipping the per-element access, not in the multi-index, which costs little: the
same field summed element by element, 1 thread, reads at 2.8 ns per element through `a[k]`,
3.3 ns through `f[i, j]` row-major and 3.5 ns through `f[i, j]` in tiles (the tiled mapping
divides by its constant tile extents in unsigned arithmetic: shifts and masks).

## 4. Where `mdspan` comes from

| available | used |
|---|---|
| `<mdspan>` with `__cpp_lib_mdspan` (libstdc++ 15, libc++ 17 and later) | `std::mdspan` |
| otherwise, `<mdspan/mdspan.hpp>` on the include path (the reference implementation, kokkos/mdspan) | its `mdspan`, in the namespace it declares |
| neither | nothing: the header only defines `SHRINK_ARRAY_HAS_MDSPAN` to 0 |

`f[i, j]` needs the multidimensional subscript of C++23 (GCC 12, Clang 15 and later); the
library already requires C++23. The harness and the demo exit 77 without an `mdspan`, which ctest
reports as skipped.

## 5. Rules and limits

- The view does not own the array: it must not outlive it. After a `resize()`, make a new view;
  a re-chunk keeps it correct, but a tiled layout is aligned on the chunks only while the chunk
  size is the tile's element count.
- The rules of the array apply unchanged: writes in parallel inside an OpenMP phase
  (`parallel_phase()`, see [getting_started.md](getting_started.md)), no element proxy kept
  across a sequential point. In check builds `auto x = f[i, j];` does not compile, as for
  `a[i]`: write `double x = f[i, j];`.
- It is not a pointer: `data_handle()` is the array's address, not a `T*`, and the accessor has
  no `offset()` -- no `submdspan`, no handing the data to code that expects contiguous memory.
- `layout_tiled` is not strided (`is_strided()` is false) and is exhaustive only when every
  extent is a multiple of its tile; code that requires `layout_right` or a strided layout takes
  the row-major view.

## 6. Files

| file | contents |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::layout_tiled`, `md::view`, `md::tiled_view`, `md::tiled_size` |
| `examples/validation/mdspan_validation.cpp` | writes through the view and linear read-back, compound operators, rank 3, read-only views, the tiled mapping (bijection, padding), a uniform tile is a uniform run; every variant (one ctest each, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile`, `md::update_tiles`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocks, tile extents on padded edges, uniform tiles, the update against the element formula, untouched tiles stay compressed, the parallel update; every variant at 1 and 8 threads, and `mdspan_tile_ops_misaligned` (a tile smaller than a chunk refused in a parallel region) |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | the comparisons of sections 2 and 3, and the cost of the multi-index; `mdspan_tiles 4096` for a larger field |
