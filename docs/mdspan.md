# A multidimensional view: mdspan

> French version: [mdspan_fr.md](mdspan_fr.md).

> October 2026. `chunk/mdspan.h` puts a C++23 `mdspan` on top of an array of the library. The
> array keeps its role, the drop-in replacement for `std::vector`; the view adds `f[i, j]`.

> **Experimental until 1.0.** The names and signatures of `chunk/mdspan.h` and
> `chunk/mdspan_tile_ops.h` may change in a MINOR release -- an exception to the versioning policy
> of [CHANGELOG.md](../CHANGELOG.md). The behaviour of the array behind a view does not: a view
> reads and writes through the array's `operator[]` and range operations, whose contract is
> stable.

***

## 1. What it is

The array stays the owner: it holds the elements, compresses them, re-chunks them and keeps its
whole interface. The view is an `mdspan` whose accessor goes through the array's `operator[]`, so
`f[i, j] = v` is `a[k] = v` with `k` given by the layout: the same write proxy, the same
compression, the same rules.

```cpp
#include "chunk/zero_any_array.h"
#include "chunk/mdspan.h"

namespace sa = shrink_array;

sa::zero_any::Array<double> a(nx * ny);
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
| `md::view_type<Array, Layout, Rank>`, `md::tiled_view_type<Array, T0, T1, ...>` | the type of these views; `view_type<const Array, ...>` is the read-only one, and converts from the writable one |
| `md::as_const(f)` | the read-only view of `f`: same extents, layout and offset over `const Array`, `f[i, j]` the value -- row-major, tiled or a sub-view. `std::as_const(f)` is not that: an `mdspan` is const on the surface only, its element type stays writable |
| `md::accessor<Array>`, `md::layout_tiled<...>` | the accessor and the layout, for building an `mdspan` by hand |

## 2. Tiles: why the layout matters for compression

A chunk is a run of consecutive elements (256 in a `_256` library). In `layout_right` it is a
piece of a row: a uniform 2D region only gives uniform chunks where it is at least 256 cells
wide. With `layout_tiled<16, 16>`, tile `k` is chunk `k`: a 16 x 16 square inside one material
is a uniform chunk, whatever the width of the material. Choose the tile so that its element
count is the chunk size: 16 x 16 for 256, 32 x 32 for 1024, 8 x 8 x 4 for 256 in 3D.

`examples/mdspan_tiles` writes the same field (uniform background, disks and narrow bands of one
material each, and a noisy corner that compresses in neither layout) through both views of a
`chunk::chunk_zero_any_256` array, then counts the chunks left materialised after the compression
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
sa::md::for_each_tile(sa::md::as_const(f),
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
| `md::for_each_tile_mt(f, ...)`, `md::update_tiles_mt(f, ...)` | called by every thread of a region (`update_tiles_mt` in a phase), no `omp for`: the tiles dealt to the team, thread t of T taking the tiles `[n t / T, n (t + 1) / T)` in storage order |
| `md::for_each_tile_mt(f, lo, hi, ...)`, `md::update_tiles_mt(f, lo, hi, ...)` | the same over the tiles the box `[lo, hi)` touches (clipped to the extents), one unit each; the callbacks get whole tiles |
| `md::fill_box_mt(f, lo, hi, v)` | called by every thread of a region, in a phase: tiled view, the box's tiles dealt by their cells inside it; row-major view, the box's cells split on chunk boundaries (as `copy_box_in_mt`) |
| `t.number`, `t.origin`, `t.extent` | the tile's rank in storage order, the multi-index of its first element, the part of it inside the view (smaller than the tile on the padded edges) |

The block covers the whole tile, padding included: loop to `t.extent` where the padding must not
count. A uniform tile left unchanged by `update_tiles` is not written, so a kernel that leaves the
background alone does not materialise it. A changed tile is written with `fill_range()` when it
is uniform, else `copy_range()` (the serial range ops outside a parallel region, the `_mt` ones
inside). On zero_any, a compressed chunk whose part of the tile comes out uniform switches its
constant in place, without a buffer, in a parallel region too (`fill_range_mt`, `copy_range_mt`,
within the write session). Outside a region a materialised chunk that comes out uniform is
compressed at once as well; inside, it is written and queued -- its buffer is freed by the end of
the phase or the next compression pass, which wait out the readers. Otherwise a compressed chunk
that changes is materialised for the write and queued, and the same pass compresses it again.

In a parallel region each thread works on its own tiles, inside a phase for `update_tiles`, and a
tile must be made of whole chunks: tile elements a multiple of `chunk_size()`, checked (a tile
sharing a chunk with another thread's stops the program). Outside a region any chunk size works.

The `_mt` forms share the work themselves: every thread computes the same deal, with no
communication, and a tile goes to one thread. Outside a region (or in a team of one) they do the
whole job serially. They end with no barrier: what one call writes is complete at the region's
next barrier, so two calls on boxes that overlap need an `omp barrier` between them -- the next
deal may give a shared tile to another thread. For many overlapping boxes in a row, a fixed slice
of the tiles per thread and the `[first, last)` overloads (each tile always on the same thread, no
barrier) remain the right idiom. `examples/mdspan_tiles`, 2048 x 2048 field, Release, GCC 14,
2-core container:

| kernel | threads | `omp for`, one tile per call | `_mt` | ratio |
|---|---|---|---|---|
| a band (rows `[n/16, 5n/16)`) filled, `fill_box_mt` | 1 | 0.58 to 0.88 ms | 0.08 to 0.11 ms | x 7.4 to 7.8 |
| a band filled, `fill_box_mt` | 2 | 0.48 ms | 0.06 to 0.07 ms | x 7.4 to 7.6 |
| sum of the field, `for_each_tile_mt` | 1 | 1.18 to 1.40 ms | 1.16 to 1.22 ms | x 1.0 to 1.1 |
| sum of the field, `for_each_tile_mt` | 2 | 0.91 to 1.19 ms | 0.85 to 1.05 ms | x 1.1 |
| 64 rectangles, against a slice of the tiles per thread | 1, 2 | 3.6 to 4.4 ms | 3.6 to 5.5 ms (a barrier per rectangle) | none |

The gain is where the `omp for` loses: on a box, whose tiles it gives to the threads that hold
those rows of tiles (all of the band to thread 0 at 2 threads), one call per tile of the whole
view; over the whole view the deal is the `omp for`'s, and so is the time.

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

### Stencils: `md::stencil_tiles`

A stencil reads a cell's neighbours, which near a tile's edge lie in another tile. `md::stencil_tiles`
hands the kernel the tile and `h` cells around it, read in one go from the neighbour tiles (each
gathered once along a row of tiles, a uniform one costing its constant), and writes the result
tile of another array back in one go.

```cpp
auto src = sa::md::tiled_view<16, 16>(a, nx, ny);   // read
auto dst = sa::md::tiled_view<16, 16>(b, nx, ny);   // written: another array
sa::parallel_phase([&] {
    sa::md::stencil_tiles_mt(sa::md::as_const(src), dst, 1, sa::md::clamp_boundary{},
        [](const auto& t, const auto& in, auto out) {    // in[i, j], i in [-1, 17)
            for (long i = 0; i < t.extent[0]; ++i)
                for (long j = 0; j < t.extent[1]; ++j)
                    out[i, j] = in[i, j] + 0.2 * (in[i - 1, j] + in[i + 1, j] + in[i, j - 1]
                                                  + in[i, j + 1] - 4.0 * in[i, j]);
        },
        [](const auto&, const double& v, double& out) { out = v; return true; });  // optional
});
```

| call | does |
|---|---|
| `md::stencil_tiles(src, dst, h, boundary, on_block, on_uniform)` | for each tile: `on_block(t, in, out)`, `in` a `md::halo_block` (`in[i, j]` for `i` in `[-h, te + h)`, `(0, 0)` the tile's first cell), `out` an `mdspan<T>` over the tile of `dst`, starting as a copy of the source tile. The tile is written back as one constant when its cells inside the view are one value, else copied |
| `on_uniform(t, v, out) -> bool` | optional: for a tile whose cells, halo and border are all `v`, in O(1) -- set `out` and return true, the tile of `dst` becomes `out` (no halo read, no kernel); false gets `on_block` |
| `md::constant_boundary<T>{v}`, `md::clamp_boundary{}` | the halo cells outside the view: `v` (Dirichlet), or the nearest cell of the view (zero gradient) |
| `md::stencil_tiles(src, dst, h, boundary, first, last, ...)` | tiles `[first, last)`, for an `omp for` or a slice per thread |
| `md::stencil_tiles_mt(src, dst, h, boundary, ...)` | called by every thread of a region, in a phase: the tiles dealt as `for_each_tile_mt` deals them (consecutive: a thread's neighbour tiles are mostly its own), no `omp for`, no barrier -- the next step, which reads `dst`, needs the region's next barrier |

`src` and `dst` have the same extents and tiles and are two arrays, and `0 <= h <=` a tile's extent
in every dimension: checked, a mismatch stops the program (a stencil in place would read cells its
other tiles have already written). Any rank: `in[i, j, k]` in 3D. Inside a region the rules of the
tile operations hold: a tile whole chunks, `dst` written in a phase, `src` written by no one.

`examples/mdspan_tiles`, one explicit diffusion step (5 points, clamped border) on the 2048 x 2048
field (85 % of its tiles uniform) into another array, against the element loop (`f[i, j]` and its
four neighbours, the border by `min` / `max`), Release, GCC 14, 2-core container, several runs:

| threads | element by element | `stencil_tiles_mt` | with `on_uniform` |
|---|---|---|---|
| 1 | 119 to 181 ms | 15.8 to 18.2 ms (x 7.4 to 10.4) | 7.4 to 7.9 ms (x 15.9 to 22.9) |
| 2 | 63 to 92 ms | 9.7 to 12.6 ms (x 6.5 to 7.3) | 5.3 to 8.5 ms (x 10.6 to 12.0) |

All three leave the same field, with the same 2561 materialised chunks.

### A box in and out of contiguous memory: `md::copy_box_out`, `md::copy_box_in`

An MPI halo, an HDF5 hyperslab, a library that wants a `T*`: a box of the view copied to or from
a contiguous row-major buffer of the box's extents, without an `operator[]` per element.

```cpp
std::vector<double> buf((hi[0] - lo[0]) * (hi[1] - lo[1]));
md::copy_box_out(f, lo, hi, buf.data());   // buf[(i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1])] = f[i, j]
// ... MPI_Sendrecv(buf, ...) ...
md::copy_box_in(f, lo, hi, buf.data());    // and back
```

| call | does |
|---|---|
| `md::copy_box_out(f, lo, hi, dst)` | tiled view: tile by tile, a tile read in one go, a uniform one a fill of its part of the box. Strided view (`md::view`, a `submdspan`): one read per row (`for_each_span`), element by element when the last stride is not 1 |
| `md::copy_box_in(f, lo, hi, src)` | tiled view: a tile the box covers is written in one go with no read -- `fill_range` when its values are one value (a chunk already at it is left alone, a compressed one switches, a materialised one is released where the variant can), else `copy_range` --; a tile it crosses is updated as by `update_tiles`. Strided view: one `copy_range` per row |
| `md::copy_box_out(f, lo, hi, dst, first, last)`, `md::copy_box_in(f, lo, hi, src, first, last)` | tiled view, tiles `[first, last)` only, for a loop `#pragma omp for` over `[0, md::tile_count(f))`; a strided view is split by the caller, each thread its own rows |
| `md::copy_box_out_mt(f, lo, hi, dst)`, `md::copy_box_in_mt(f, lo, hi, src)` | called by every thread of a region (`copy_box_in_mt` in a phase), no `omp for`: the box shared by the functions themselves. Tiled view: the tiles of the box, in storage order, each weighing its cells inside the box, dealt in T consecutive shares of about W / T cells. Strided view: the box's cells in row-major order split in T shares, each bound moved up to the next chunk boundary when the rows are contiguous, so that a chunk inside a row is written by one thread. Every thread computes the same split, with no communication; outside a region, the whole box. No barrier: the buffer is complete at the region's next barrier |

The box must lie inside the view's extents (a box outside stops the program: clipped, it would
change the buffer's shape). Inside a parallel region the rules of the tile operations hold:
`copy_box_in` in a phase, a tile made of whole chunks.

`examples/mdspan_tiles`, a box of 1024 x 1024 in a 2048 x 2048 tiled field, against the element
loop (`f[i, j]`, rows shared by the team), Release, GCC 14, 2-core container:

| | threads | element by element | `copy_box_*` | ratio |
|---|---|---|---|---|
| out | 1 | 7.96 ms | 2.03 ms | x 3.9 |
| out | 2 | 3.79 ms | 0.87 ms | x 4.4 |
| in | 1 | 22.79 ms | 2.48 ms | x 9.2 |
| in | 2 | 10.42 ms | 1.23 ms | x 8.5 |

The `omp for` over `[0, md::tile_count(f))` gives a thread the tiles of its rows of tiles, in the box
or not: a box that is a band of the field (rows `[n/16, 5n/16)`, every column) falls to thread 0
alone at 2 threads. The `_mt` forms share the band's tiles, and make one call per thread instead of
one per tile (`mdspan_tiles`, 2048 x 2048 field):

| | threads | `omp for`, one tile per call | `copy_box_*_mt` | ratio |
|---|---|---|---|---|
| out | 1 | 1.15 ms | 0.69 ms | x 1.7 |
| out | 2 | 1.01 ms | 0.31 ms | x 3.3 |
| in | 1 | 1.99 ms | 1.19 ms | x 1.7 |
| in | 2 | 1.72 ms | 0.58 ms | x 3.0 |

### Sub-views: `md::submdspan`

`md::submdspan(f, slices...)` is `std::submdspan`, with `md::full_extent`. The view's data
handle is an `md::handle`: the array and the index in it of the sub-view's first element
(`f.data_handle().base`); `*f.data_handle()` is still the array.

```cpp
auto f   = md::view(a, nx, ny);
auto blk = md::submdspan(f, std::pair{3, 20}, std::pair{5, 40}); // blk[0, 0] is f[3, 5]
auto row = md::submdspan(f, 7, md::full_extent);                 // rank 1

auto g   = md::tiled_view<16, 16>(b, nx, ny);
auto sub = md::submdspan(g, std::pair{16, 64}, std::pair{32, ny}); // tiled again
md::update_tiles(sub, on_block, on_uniform);                       // the tiles of sub only
```

- A row-major view takes any slices (the standard layouts' own `submdspan_mapping`: the result is
  `layout_right` or `layout_stride`). `fill_box` takes any strided view: one range op per row of
  the box when the last stride is 1, one per element otherwise.
- A tiled view takes rank-preserving slices of stride 1 (`full_extent`, `{first, last}`) aligned
  on its tiles: each range starts on a tile and ends on one or at the view's extent; anything
  else stops the program. The sub-view is a `layout_tiled` again, in its parent's tile grid: its
  tiles are the parent's, each still one chunk, and the padding of its last tiles is the
  parent's padding. `for_each_tile`, `update_tiles` and `fill_box` work on it as on a whole view
  -- they read and write whole tiles, and touch nothing outside the sub-view --, in a parallel
  phase too (`tile_count(sub)` tiles).

`examples/validation/mdspan_submdspan_validation.cpp`; checked against kokkos/mdspan before and
after its canonical slices (May 2025, October 2026).

### The tile at run time: `layout_tiled_dyn`

A `_dyn` library knows its chunk size at run time only (`DYN_CHUNK_SIZE`, a re-chunk), so a tile
of one chunk cannot be a template argument. `md::layout_tiled_dyn<Rank>` takes the tile at run
time and lays the array out exactly as `layout_tiled` does for the same tile (the two views of one
array agree); the tile operations, `submdspan` and `fill_box` take it as they take
`layout_tiled`.

```cpp
zero_any::Array<double> a(md::tiled_size_dyn(md::tile_shape<2, long>(chunk_size), std::array<long, 2>{nx, ny}));
auto f = md::tiled_view_dyn(a, nx, ny);                          // a tile of a.chunk_size() elements
auto g = md::tiled_view_dyn(a, std::array<long, 2>{16, 24}, nx, ny); // or any tile
```

- `md::tile_shape<Rank>(n)` gives a tile of exactly `n` elements, as square as the divisors of
  `n` allow, the longer extents last: 256 -> 16 x 16, 384 -> 16 x 24, 512 -> 16 x 32, 1024 ->
  32 x 32, 256 in 3D -> 4 x 8 x 8. Every chunk size of the library (a power of two or a multiple
  of 64) has such a tile; `tiled_view_dyn(a, ext...)` uses it with `a.chunk_size()`.
- Cost, 2048 x 2048, 1 thread: a read through `f[i, j]` as with `layout_tiled` (2.75 ns per
  element: the run-time division hides behind the element access), `update_tiles` about 10 %
  slower (the block's extents are run-time values too).
- After a re-chunk nothing moves: the view still reads and writes the right elements, and the
  tile operations still work serially. But a tile is no longer a chunk -- at twice the chunk
  size, a chunk holds two consecutive tiles -- so in a parallel region the tile operations stop
  (two threads would share a chunk) until the field is copied into a view of the new tile.

### A whole program: `multi-mat-md`

`examples/multi-mat` exists in 2D: `multi-mat-md` is the same mock (every material carries three
fields and a presence fraction over the whole mesh, occupies a random part of it at each iteration,
is written there and read back by the same gadget computation), on an `nx x ny` mesh where a
material occupies a box. Every array is seen through a 2D `mdspan`: `md::view` or `md::tiled_view`
for the chunk arrays (zero for the fields, zero_one_value for the fraction), a standard `mdspan`
over `data()` for the `std::vector` ones. The loop body is written once against the views; the
policy of the material's storage (`LoopsMd.h`) says how a box is written, zeroed and read, as
`Loops.h` does in 1D -- and the indirection list of the 1D version is gone: the box is the loop.

| `--chunk-mode` | chunk arrays | box loops |
|---|---|---|
| `tiles` (default) | `layout_tiled`, a tile per chunk | tile by tile: `update_tiles` (a tile of the interior already at 1 is left alone), `fill_box` for the reset, four tiles read in one go each for the reduction |
| `elements` | `layout_tiled` | cell by cell, `f[i, j]` |
| `rows` | row-major `md::view` | cell by cell, `f[i, j]` |

The team splits the cells of all the boxes at once, by weight, tile by tile (row by row of a
box in `rows`, where a chunk is a piece of a row); every thread computes the same split, with no
communication, and a tile goes to one thread only. Split box by box, an `omp for` per material,
a box of 113 rows has 8 rows of tiles: at 64 threads only 15 threads wrote and the most loaded
did 7.8 times the mean (35 times at 288 threads). Split at once: 1.01 at 64 threads, 1.07 at
288 (3200 x 3200, 50 materials, counted on 2 cores). The run prints that ratio;
`--max-imbalance` makes it a check (test `multi_mat_md_balance`).

1600 x 1600 mesh, 50 materials all in chunk arrays, 100 iterations, boxes up to 113 x 113
(`--pc-chunk 0.5`), the same boxes for every line (`--seed 3`); Release, GCC 14, kokkos/mdspan,
2 threads on a 2-core container (orders of magnitude: ± 20 % from one run to the next there):

| | chunks written, fields / fraction | peak memory | time | of which writes / reads |
|---|---|---|---|---|
| `--chunk-mode tiles` | 16.7 % / 12.5 % | 663 MiB | 2.29 s | 0.35 s / 1.82 s |
| `--chunk-mode elements` | 16.7 % / 12.5 % | 659 MiB | 4.36 s | 0.68 s / 2.36 s |
| `--chunk-mode rows` | 46.1 % / 46.1 % | 1847 MiB | 4.14 s | 1.01 s / 1.96 s |
| all in `std::vector` (`--option 2`) | -- | 3912 MiB | 1.90 s | 0.12 s / 1.55 s |

All four print the same result. Row-major, a box narrower than a chunk leaves every chunk it
crosses written, and the fraction never compresses inside it; in tiles only the tiles along the
sides stay. The time left after the writes and the reads is the initialisation, which zeroes
every array over the whole mesh as the 1D version does: in tiles a `fill_box` of chunks already
at 0, which writes nothing (0.01 s); cell by cell (`elements`, `rows`), a write per cell
(about 1.2 s). `--verify` replays the run serially on plain vectors and compares every array cell by
cell and the result (exit 1 on a difference); the tests `multi_mat_md_tiles`, `multi_mat_md_elements` and `multi_mat_md_rows` run it small.
The options are listed at the top of `main_md.cpp`.

## 4. Where `mdspan` comes from

The standard library's: `std::mdspan` (C++23) and `std::submdspan` (C++26), both in GCC 16's
libstdc++, the compiler this branch builds with. `md::submdspan` and `md::full_extent` are
`std::submdspan` and `std::full_extent`; the tiled layouts give it their `submdspan_mapping`.
`chunk/mdspan.h` stops with an `#error` where `__cpp_lib_submdspan` is missing. (`cpp23` also takes
the reference implementation, kokkos/mdspan, and builds without an `mdspan`.)

## 5. Rules and limits

- The view does not own the array: it must not outlive it. After a `resize()`, make a new view;
  a re-chunk keeps it correct, but a tiled layout is aligned on the chunks only while the chunk
  size is the tile's element count.
- The rules of the array apply unchanged: writes in parallel inside an OpenMP phase
  (`parallel_phase()`, see [getting_started.md](getting_started.md)), no element proxy kept
  across a sequential point. In check builds `auto x = f[i, j];` does not compile, as for
  `a[i]`: write `double x = f[i, j];`.
- It is not a pointer: `data_handle()` is an `md::handle` (the array and an index), not a `T*`
  -- no handing the data to code that expects contiguous memory.
- `layout_tiled` is not strided (`is_strided()` is false) and is exhaustive only when every
  extent is a multiple of its tile; code that requires `layout_right` or a strided layout takes
  the row-major view.

## 6. Files

| file | contents |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::handle`, `md::layout_tiled` (and its `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::as_const`, `md::view_type`, `md::tiled_view_type`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | writes through the view and linear read-back, compound operators, rank 3, read-only views (`md::as_const` of a row-major, a tiled and a sub-view), the tiled mapping (bijection, padding), a uniform tile is a uniform run; every variant (one ctest each, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile(_mt)`, `md::update_tiles(_mt)`, `md::fill_box(_mt)`, `md::copy_box_out(_mt)`, `md::copy_box_in(_mt)`, `md::stencil_tiles(_mt)`, `md::halo_block`, `md::constant_boundary`, `md::clamp_boundary`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocks, tile extents on padded edges, uniform tiles, the update against the element formula, untouched tiles stay compressed, the parallel update; the `_mt` forms (each tile of the view or of a box visited once and no other, a thread's tiles within one of the others' count, `fill_box_mt` on tiled and row-major views, in a phase and outside any region); every variant at 1 and 8 threads, and `mdspan_tile_ops_misaligned` (a tile smaller than a chunk refused in a parallel region) |
| `examples/validation/mdspan_copy_box_validation.cpp` | `copy_box_out` / `copy_box_in` against `f[i, j]` and a reference: boxes aligned, unaligned, on the padded edges, one row, one column, the whole view, empty; tiled (a tile per chunk, 16 x 16), row-major, rank 3, a `submdspan` and a column; whole tiles copied in uniform left compressed; tile by tile and row slice by row slice in a parallel phase; the `_mt` forms on every box and view, and their deal (each tile of the box to one thread, no thread above W / T plus a tile, strided shares a partition with their bounds on chunk boundaries); every variant at 1 and 8 threads |
| `examples/validation/mdspan_stencil_validation.cpp` | `stencil_tiles` / `stencil_tiles_mt` against the stencil through `f[i, j]`: 9 points (halo 1) and 13 (halo 2), constant and clamped borders, tiles of 16 x 16 and of a chunk (padded edges), a halo as wide as a tile, rank 3; the same result with and without `on_uniform`, the uniform tiles of the result holding no buffer, the source untouched; serial, per tile in an `omp for`, from every thread of a phase and outside any region; every variant at 1 and 8 threads, and `mdspan_stencil_alias` (a stencil from an array into itself refused) |
| `examples/validation/mdspan_submdspan_validation.cpp` | sub-views of a row-major view (block, row, column, read-only, `fill_box`) and of a tiled view (aligned, ending at the extent, a sub-view of a sub-view; the tile operations on it, serially and in a parallel phase); every variant at 1 and 8 threads, and `mdspan_submdspan_misaligned` (a tiled slice off the tiles stops) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, every multiple of 64 to 4096), `layout_tiled_dyn` against `layout_tiled` index by index and through their sub-views, `tiled_view_dyn`'s tile per chunk, the tile operations on a tile per chunk, a 3 x 5 tile and one beyond the inline buffer; every variant at the default size and at dyn (a re-chunk under the view), 1 and 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | the comparisons of sections 2 and 3, and the cost of the multi-index; `mdspan_tiles 4096` for a larger field |
| `examples/multi-mat/main_md.cpp`, `LoopsMd.h`, `ComputeLoopsMd.h`, `SimuMd.h` | `multi-mat-md`, the multi-material mock in 2D (section 3); tests `multi_mat_md_tiles`, `multi_mat_md_elements`, `multi_mat_md_rows` (`--verify`), `multi_mat_md_balance` |
