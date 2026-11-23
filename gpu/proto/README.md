# GPU prototype, stage 0

Stage 0 of [docs/gpu_design.md](../../docs/gpu_design.md) (section 13): a standalone `zero_any`
array on GPU, out of the library's tree, to decide by measurement whether the chunk format has a
GPU future. Target applications: Kokkos and Arcane codes (the facade comes after this core).

| file | contents |
|---|---|
| `device_intrinsics.h` | the device primitives (ballot, shuffle, scoped atomics, acquire/release) mapped on CUDA and HIP |
| `zero_any_device.h` | `ChunkDesc` (buffer or constant), `View` (read, write proxy), `DeviceArray` (arenas, free stack, append-buffer, `compress()`, `finalize()`) |
| `proto_test.cu` | correctness against a host mirror, for race, claim and the chunk-aligned path: random writes from many warps, fold-backs, arena exhaustion and replay, append overflow |
| `proto_bench.cu` | the stage-0 measurements against a flat array, for the three coordinations |

## Build and run

```
cmake -S gpu/proto -B build-proto -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=<llvm>/bin/clang++ -DCMAKE_CXX_COMPILER=<llvm>/bin/clang++ -DCUDAToolkit_ROOT=<cuda>
ninja -C build-proto && build-proto/proto_test && build-proto/proto_bench 27 10
```

nvcc works too (`-DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=<a g++ the toolkit
accepts>`: CUDA 12.4 takes GCC 13 at most). With clang, `-O3` matters: device code is not
optimised by default (the CMakeLists sets it). `compute-sanitizer --tool synccheck` and
`--tool memcheck` report no error on `proto_test`.

## Measurements (2026-11-23, laptop RTX A1000, sm_86, 4 GB, 1 GiB of doubles)

| | S = 256 | S = 1024 | criterion |
|---|---|---|---|
| read, every chunk materialised | 1.04x flat | 1.23x flat | <= 1.5x: met |
| write into materialised chunks | 0.82x | 1.08x | <= 2x: met |
| `compress()` of 10 GB, scaled to 3.35 TB/s (fraction of stream bandwidth) | 4.1 ms | 3.0 ms | <= 10 ms: met |
| first write over the array, race | 5.5x | 25x | -- |
| first write over the array, claim | 2.2x | 3.4x | -- |
| first write over the array, chunk-aligned | 1.9x | 1.2x | -- |
| random range rewrites, write + `finalize()`, race | 4.4x | 4.1x | -- |
| random range rewrites, write + `finalize()`, claim | 3.7x | 2.1x | -- |
| random range rewrites, write + `finalize()`, chunk-aligned | 4.2x | 1.9x | -- |

The laptop gives ratios, not absolute times; the GH200 and the MI300A give those. The
`compress()` line is the case where every chunk folds back (a full read); a scan where nothing
folds back stops at the first differing element and is faster.

## GH200 (2026-11-23)

**GH200 (H100, 132 SMs, HBM3 at 3.7 TB/s measured), 8 GiB of doubles**, first run (the
prototype at 91d7af328):

| | S = 256 | S = 1024 | criterion |
|---|---|---|---|
| read, every chunk materialised | 1.04x | 1.04x | <= 1.5x: met |
| write into materialised chunks (acquire load) | 1.46x | 1.75x | <= 2x: met, but close |
| `compress()`, every chunk folds back: 10 GB extrapolated from 8 GiB | 7.7 ms | 4.6 ms | <= 10 ms: met |
| first write, race / claim / chunk-aligned | 15.6x / 17.7x / 5.1x | 21.8x / 14.4x / 1.8x | -- |
| range rewrites + `finalize()`, race / claim / chunk-aligned | 7.5x / 6.9x / 6.1x | 5.4x / 3.4x / 2.7x | -- |

What the large GPU shows, and the laptop hid behind its 170 GB/s:

- **The write path is bound by one descriptor load in flight per warp.** A store into a
  buffer may alias the descriptors, so the compiler cannot issue the next write's descriptor
  load before the current store: each warp waits ~1 us per write. A second run measured the
  acquire, relaxed and plain loads (`WriteLoad`) equal (1.48x to 1.52x): the barrier was not
  the cause. The chunk-aligned path, one descriptor load per block, costs 1.12x at S = 1024;
  the bench now also measures four descriptors loaded ahead per thread (what a facade loop
  can do).
- **A materialisation is a chain of round trips, on two global counters** (the free stack's
  top and the bump), paid once per grid-stride step: claim is no better than race at
  S = 256. Two shortcuts tried in between made it worse and were reverted: a look at the free
  stack before its atomic (the bench's first write follows a fold, so the stack is full and
  the look is one more round trip) and one device fence by the leader after the warp barrier
  instead of one per lane (the fences then run one after the other); first write 5.1x ->
  9.0x (S = 256), 1.8x -> 2.5x (S = 1024). The counters are now **sharded**: each arena is
  split into 128 shards, each with its own bump and free-stack segment; a chunk allocates in
  its home shard (its index modulo 128), then the next ones, and a buffer goes back to the
  shard of its address.
- **`compress()`**: one warp per chunk is now the default (GH200: 83% of the stream bandwidth
  at S = 1024, against 66% per block). At S = 256 both reach only 36-39% when every chunk
  folds back: the cost is the free-stack push of each fold on one counter (a scan without
  folds runs in 0.96 ms against 6.4 ms), which the shards spread.
- **S = 1024 is the GPU geometry**: equal or better on every line but two (the race's first
  write, and `assign()` into materialised chunks).
- **A race-path bug found on the way**: the race grouping loop, which each lane left once its
  own group was done, ran later shuffles with changing masks while other lanes were inside
  their group's work. With clang-CUDA and the slower sharded allocation, some lanes then
  pre-filled the wrong buffer, and published buffers kept never-written elements (one test
  run in two). The grouping is now warp-uniform, as the claim path's already was
  (`group_by_chunk`), and 16 runs out of 16 pass.

## What the prototype changed in the design

- **A chunk spans S/32 warps** (S/64 wavefronts): warp aggregation (6.2) gives one
  materialisation per warp, not per chunk. On a grid-stride first write, 5.5 (S = 256) and 22.8
  (S = 1024) warps per chunk lose the CAS, each after pre-filling a whole buffer for nothing:
  the race line above. Hence the per-chunk coordination below.
- **Lost buffers** (race): given up until `compress()` (6.3, first form), they exhausted the arena
  on the first full write. They are kept in a per-warp spare slot (an atomic exchange, no wait,
  no ABA) and reused by the warp's next materialisation; `compress()` takes the spares back.
- **The leader looks again** at the descriptor before allocating: a chunk published meanwhile
  costs nothing (lost CAS divided by 11 on scattered writes).
- **Writers read `buf` with acquire** (6.2 says so; 6.1's sketch had a relaxed load): without
  it a writer may be overwritten by the winner's pre-fill still in flight.
- **Arena growth** in `finalize()`: one buffer per chunk not yet materialised at most, not one
  per parked write.

## Coordination per chunk

`View::coord` selects how `assign()` coordinates the warps that need one compressed chunk;
`block_write` / `DeviceArray::parallel_for_chunks` is the facade's chunk-aligned path.

- **race** (6.2 as written): every warp pre-fills, the CAS elects one. No wait; S/32 pre-fills
  per chunk on a first write, and an arena sized for the losers.
- **claim**: the first warp's CAS installs a busy mark (`busy_mark()`, a misaligned address), it
  pre-fills and publishes; the others wait for the publication. One pre-fill per chunk, no lost
  buffer, an arena of exactly one buffer per materialised chunk. The wait ends because the
  holder is resident and waits for nothing between its CAS and its publication -- provided a
  warp **publishes before it waits**. The first version let one group of a warp hold a claim
  while another group of the same warp waited (a warp over two chunks, on unaligned ranges):
  with SIMT reconvergence the holder can be held back behind its waiting sibling, and two warps
  in that state wait for each other. Measured: 591 waits that only the spin bound ended, a
  random-range rewrite 80x slower. Now every group of the warp first makes one attempt without
  waiting, the groups that claimed pre-fill and publish, and only then do the others wait,
  never claiming. A bounded spin remains, then the append-buffer: two `assign()` calls on
  divergent paths of one AMD wavefront (no independent thread scheduling) could still wait for
  each other; that costs a replay, not a deadlock. This revises the "nobody waits in a kernel"
  rule of section 3 into "nobody waits while holding a claim".
- **chunk-aligned** (7.3, mitigation 2): one block per chunk votes (`__syncthreads_or`), takes
  one buffer and writes the values and the constant in one pass: no pre-fill of the elements
  written, f evaluated once per element. The CAS stays for two ranges that share a chunk. Best
  at S = 1024 (1.2x on a first write); at S = 256 one 256-thread block per chunk pays four
  barriers per 2 KiB, and claim is as good.

Conclusion for the facade: the Kokkos / Arcane `parallel_for` over an array takes the
chunk-aligned path; arbitrary user kernels go through `assign()` with claim. Race stays as the
reference.
