# GPU prototype, stage 0

Stage 0 of [docs/gpu_design.md](../../docs/gpu_design.md) (section 13): a standalone `zero_any`
array on GPU, out of the library's tree, to decide by measurement whether the chunk format has a
GPU future. Target applications: Kokkos and Arcane codes (the facade comes after this core).

| file | contents |
|---|---|
| `device_intrinsics.h` | the device primitives (ballot, shuffle, scoped atomics, acquire/release) mapped on CUDA and HIP |
| `zero_any_device.h` | `ChunkDesc` (buffer or constant), `View` (read, write proxy), `DeviceArray` (arenas, free stack, append-buffer, `compress()`, `finalize()`) |
| `proto_test.cu` | correctness against a host mirror: random writes from many warps, fold-backs, arena exhaustion and replay, append overflow |
| `proto_bench.cu` | the stage-0 measurements against a flat array |

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

## First measurements (2026-11-23, laptop RTX A1000, sm_86, 4 GB, 1 GiB of doubles)

| | S = 256 | S = 1024 | criterion |
|---|---|---|---|
| read, every chunk materialised | 1.02x flat | 1.22x flat | <= 1.5x: met |
| write into materialised chunks | 0.82x | 1.08x | <= 2x: met |
| `compress()` of 10 GB, scaled to 3.35 TB/s (fraction of stream bandwidth) | 4.0 ms | 3.0 ms | <= 10 ms: met |
| first write over the array (materialises every chunk) | 5.5x | 25x | -- |
| random range rewrites, write + `compress()` | 4.4x | 4.2x | -- |

The laptop gives ratios, not absolute times; the GH200 and the MI300A give those.

## What the prototype changed in the design

- **A chunk spans S/32 warps** (S/64 wavefronts): warp aggregation (6.2) gives one
  materialisation per warp, not per chunk. On a grid-stride first write, 5.5 (S = 256) and 22.8
  (S = 1024) warps per chunk lose the CAS. Each loser has pre-filled a whole buffer for nothing:
  that is the 5.5x and 25x above.
- **Lost buffers**: given up until `compress()` (6.3, first form), they exhausted the arena on
  the first full write. They are now kept in a per-warp spare slot (an atomic exchange, no
  wait, no ABA) and reused by the warp's next materialisation; `compress()` takes the spares
  back. The memory waste is bounded by the slots; the wasted pre-fill work is not.
- **The leader looks again** at the descriptor before allocating: a chunk published meanwhile
  costs nothing (lost CAS divided by 11 on scattered writes).
- **Writers read `buf` with acquire** (6.2 says so; 6.1's sketch had a relaxed load): without
  it a writer may be overwritten by the winner's pre-fill still in flight.
- **Arena growth** in `finalize()`: one buffer per chunk not yet materialised at most, not one
  per parked write.

## Open: coordination per chunk

The first-write cost needs one pre-fill per chunk, not per warp. Candidates, to measure:

1. **Claim, then fill**: a CAS installs a "busy" mark; the winner allocates, pre-fills and
   publishes; a warp that finds the mark waits for it. The waited-for warp is resident and
   waits for nothing, so the wait ends -- but it breaks the "nobody waits in a kernel" rule of
   section 3, to be argued.
2. **Library-split kernels** (7.3, mitigation 2): the `parallel_for` of the Kokkos facade aligns
   teams on chunks and votes per block before materialising: one writer per chunk, no CAS race.
   Covers the facade's loops, not arbitrary user kernels.
3. **Losers defer** to the append-buffer when the chunk is being claimed: no wait, but a replay
   for most first writes.
