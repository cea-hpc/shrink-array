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
double x = f[i, j];                       // a value, not auto (see section 4)

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

## 3. Where `mdspan` comes from

| available | used |
|---|---|
| `<mdspan>` with `__cpp_lib_mdspan` (libstdc++ 15, libc++ 17 and later) | `std::mdspan` |
| otherwise, `<mdspan/mdspan.hpp>` on the include path (the reference implementation, kokkos/mdspan) | its `mdspan`, in the namespace it declares |
| neither | nothing: the header only defines `SHRINK_ARRAY_HAS_MDSPAN` to 0 |

`f[i, j]` needs the multidimensional subscript of C++23 (GCC 12, Clang 15 and later); the
library already requires C++23. The harness and the demo exit 77 without an `mdspan`, which ctest
reports as skipped.

## 4. Rules and limits

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

## 5. Files

| file | contents |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::layout_tiled`, `md::view`, `md::tiled_view`, `md::tiled_size` |
| `examples/validation/mdspan_validation.cpp` | writes through the view and linear read-back, compound operators, rank 3, read-only views, the tiled mapping (bijection, padding), a uniform tile is a uniform run; every variant (one ctest each, `mdspan_validation_<variant>`) |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | the comparison of section 2; `mdspan_tiles 4096` for a larger field |
