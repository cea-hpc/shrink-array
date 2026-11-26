# GPU port: first design document

> Version française : [conception_gpu_fr.md](conception_gpu_fr.md).

> Status: written in October 2026 before any prototype; revised in November 2026 after the
> stage-0 prototype ([gpu/proto](../gpu/proto/README.md): a laptop RTX A1000, then a GH200).
> This document sets out what a GPU backend of shrink-array should be, what it takes from the
> CPU version and what it leaves behind, and the plan for deciding by measurement. Section 13
> gives the stage-0 figures, laptop, GH200 and MI300A. The
> decisions marked **(decided by the prototype)** were open in the first version. Stage 1 has
> started (end of section 13).

***

## 1. Goal and scope

**Goal.** Offer on GPU (NVIDIA and AMD) an array that is read and written in a kernel like a
flat array, `a(i)` / `a(i) = v`, and that does not store its uniform chunks -- the same data
format and the same guarantees (lossless, in-place writes, any repeated value compressed by
`zero_any`) as the CPU version. The gain sought is first **memory**: 80 to 192 GB of HBM is
where "20 times less" changes what a code can handle.

**In the scope of this first design:**

- the three variants `zero`, `zero_or_value`, `zero_any`, and `pattern` (section 8);
- fixed geometry (S a compile-time constant) first, dynamic geometry next (section 9.3);
- an execution model **by phases**, aligned on the kernels (section 3);
- interoperability with the CPU version: same chunk format, transfers both ways, hybrid mode
  on coherent-memory machines (section 11).

**Out of scope:**

- running the existing CPU code on GPU: the execution machinery (section 2) is not portable
  and does not need to be;
- writes concurrent with a compression **inside** a kernel;
- bit-packed `bool` (`BitArray`);
- a SYCL/Intel backend (possible through the portability layer, not targeted).

## 2. What ports, what does not

The CPU library has two layers of very different kinds:

| layer | contents | on GPU |
|---|---|---|
| **format** | table of chunks; a chunk = private buffer or constant; S a power of 2 or a multiple of 64; index `i >> shift`, `i & mask`; per-chunk metadata | **ports as is**, and is even better suited there than on CPU (section 4) |
| **access semantics** | reading a compressed chunk = the constant; writing a compressed chunk's constant does nothing; writing anything else materialises, pre-filled with the constant left; compression of chunks that became uniform again; rechunk by `copy_range` | **ports**: it is the definition of the container |
| **execution machinery** | asymmetric Dekker and `membarrier`, per-thread deferred writes, read barrier and dirty bitset, hazard pointers, OMPT, `thread_local`, `fast_allocators` (per-rank slabs, `mmap` arenas), per-thread reuse lists, TBB adaptor, lock-free second-value registry | **does not port**, and is replaced by the model of section 3 |

The precise reasons why the third layer does not port:

- **no forward-progress guarantee between blocks**: a GPU thread waiting for another warp may
  wait for a warp that will never be scheduled while it holds the slot. Any bounded
  inter-thread wait (`wait_out_compressor`, the compressor waiting on the flags) is forbidden
  in a kernel;
- **no equivalent of `membarrier`** (no inter-processor interrupt on the device side); the
  symmetric `seq_cst` shape would exist, but without a progress guarantee it is of no use;
- `std::atomic`, `thread_local`, `mmap` and system calls do not exist in device code (nvcc);
  device atomics have **scopes** (block, device, system) and, under Kokkos, are *relaxed* by
  definition;
- the GenMC models of `formal/` check the C11 model; the scoped memory models (PTX, HSA) have
  no practical checker. A fine-grained acquire/release protocol in a kernel could not be
  "checked, not only tested", which goes against the library's rule.

## 3. Execution model: one role per kernel

The three handshakes of the CPU version exist because an OpenMP region runs writers, a
compression pass and flushes **at the same time**, plus pure readers against buffer reuse. On
GPU the exclusion is provided by the **kernel boundary**: a kernel ends, `fence`, the next one
starts. The design rests on a single invariant:

> **Invariant.** During a kernel, the chunks of an array are touched by **one role** only:
> readers, or writers, or the compression pass, or the rechunk rebuild. No buffer is reclaimed
> during a write or read kernel.

| phase | role | possible conflict | resolution |
|---|---|---|---|
| user kernel, reading | readers | none | -- |
| user kernel, writing | writers | several warps materialise the same chunk | a claim on the descriptor (CAS to a busy mark), one pre-fill, publication; the others wait for it (6.2) |
| `compress()` (end of phase) | the pass | none: no writer, no reader | per-block vote + compaction (7) |
| `rechunk_to()` | host + kernels | none | sequence of phases (9.3) |

Consequences:

- **nobody waits while holding a claim**: the first version said "nobody ever waits", with a
  CAS race where every warp pre-fills its own buffer and the loser reads the winner's. The
  prototype measured that race at 5.5x to 25x a flat write on a first write (6.2), and replaced
  it by a claim: the warps that find a chunk being claimed wait for its publication. The wait
  ends because the holder is resident and waits for nothing between its claim and its
  publication; a warp publishes its own claims before any of its lanes waits (6.2);
- compression is no longer "in flight" but **at the end of the phase**: the memory peak of a
  phase is bounded by what that phase materialises (section 7.3 for the mitigations);
- the phase API already exists on the CPU side (`parallel_phase`, `begin_parallel_phase`,
  `finalize_parallel_phase`); on the GPU side, the phase is the kernel, and `finalize` is
  `compress()` after the `fence` (section 12).

The word "phase" covers one kernel or a sequence of kernels with the same role: several read
kernels, or several write kernels, can follow one another with no compression in between.

## 4. The device format

### 4.1 The chunk descriptor

On the CPU side, a chunk is an atomic `values` pointer (plus the read/write gates under
OpenMP), which points to a private buffer or to a shared constant buffer. On GPU the
**constant is stored in the descriptor**: there is no constant buffer any more, and no
second-value registry on the hot path.

```cpp
template <typename T>
struct ChunkDesc                 // 16 bytes for a T of 8 bytes or less
{
    T*       buf;                // nullptr: compressed; otherwise a buffer of S elements
    T        constant;                // the constant of the compressed chunk (or the one left: 6.2)
    // zero: constant is always T{}; zero_or_value: T{} or the array's second value;
    // zero_any: any value.  pattern: see 8.4.
};
```

For a 16-byte `T` (`std::complex<double>`) the descriptor is 24 bytes, aligned to 32
**(to be decided: 32 aligned bytes, or a tagged pointer + side table; measure)**. The
materialisation CAS is on `buf` alone (8 bytes): `constant` is written only by the compression pass
(section 7), never by a writer, so never concurrently with a reader.

The descriptor is a **trivially copyable** object, with no atomic in its type: atomic accesses
are made through a device-scope `atomic_ref` on `buf`, at the only places that change it. It
can therefore be shared with the host (section 11).

### 4.2 Geometry and alignment

S is a power of 2, a **multiple of 64** (the size of an AMD wavefront; 32 at NVIDIA).
Consecutive threads of a warp read consecutive indices: with S ≥ 64 and aligned buffers, a
warp always falls **within a single chunk**. Three properties follow, which make the format
better suited to the GPU than to the CPU:

- the descriptor is read **once per warp** (same address for every lane: L1 broadcast), not
  once per thread;
- the `buf == nullptr` test is **uniform per warp**: no divergence;
- the buffer accesses are **coalesced**, as on a flat array.

Starting size: 256 or 1024 **(to be decided)**. The GPU would want larger chunks to amortise
the descriptor and the atomics (one materialisation per chunk), fine-grained compression would
want them small; it is the same trade-off as the CPU dynamic geometry (9.3).

### 4.3 Per-array metadata

| data | size | where |
|---|---|---|
| `ChunkDesc[n_chunks]` | 16 or 24 B per chunk | device (or coherent memory, 11) |
| the arena's bump counter, top of the free-buffer stack | 2 words | device |
| append-buffer of the writes not applied (6.4) | bounded, set at creation | device |
| statistics (`stats()`, materialised chunks, arena failures) | a few words | device, read back by the host at the end of the phase |

There is **no** dirty bitset, no candidate list, no "already listed" line: the compression
pass looks at every materialised chunk (7.1), which on GPU costs less than maintaining lists.

## 5. Reading

```cpp
__device__ T operator()(IndexT i) const noexcept
{
    const auto& d = desc_[i >> shift_];
    return d.buf ? d.buf[i & mask_] : d.constant;     // branch uniform per warp
}
```

No read barrier: there is no deferred write to see, since a read phase has no writer, and
`compress()` and the application of the append-buffer (6.4) are finished before the phase.

Expected: 1.1 to 1.5 times the time of a flat array on a bandwidth-bound kernel (one
descriptor read per warp, latency hidden by occupancy), against 1.9 measured on the CPU side
(`examples/access_cost`). **To be measured first** (section 13): it is the figure that
decides everything.

## 6. Writing

### 6.1 The device proxy

As on CPU, `a(i) = v` goes through a proxy that knows the array, the chunk and the position,
and that **sees the value before deciding**. The device proxy comes down to three cases:

```cpp
__device__ void assign(T v)
{
    ChunkDesc<T>& d = desc_[nc];
    T* buf = atomic_load_acquire(d.buf);          // device scope; busy mark: see 6.2
    if (buf == nullptr)
    {
        if (v == d.constant) return;                   // 1. the constant onto its constant: nothing
        buf = materialise(d);                     // 2. materialise (6.2), then write
    }
    buf[pos] = v;                                 // 3. materialised chunk: direct write
}
```

Cases 1 and 3 are the hot path, uniform per warp, with no atomic read-modify-write. The load
of `buf` is an **acquire** (the first sketch had a relaxed load): without it a writer may write
before the pre-fill of the buffer it sees has landed, and be overwritten by it. Its cost is not
measurable (write into a materialised chunk: 0.82x and 1.08x a flat write). The rules of
`zero_or_value` ("the second value onto an element that already holds it") and of `zero_any`
("discover", "leave", `may_become_uniform`) fed the candidate lists: they **disappear**, the
end-of-phase pass needs no candidates. What remains to handle is `+=` and the other compound
operators, which read before writing: same cases, with `v` computed from `d.constant` or
`buf[pos]`.

### 6.2 Materialisation

1. **Warp aggregation.** The lanes of the warp whose chunk is the same (in practice all of
   them: 4.2) elect one lane (`__match_any_sync` / `__ballot`, `__activemask`); the others
   wait for the elected lane's result through a shuffle. One materialisation per warp and per
   chunk, not per thread.
2. **Allocation**: the elected lane takes a buffer from the **free-buffer stack** (returned
   by the previous compressions, 7.2), otherwise by **bump** on the array's arena
   (`atomicAdd` on the counter, one per buffer). If the arena is exhausted: 6.4.
3. **Pre-filling** of the buffer with `d.constant`: by **the whole warp** (S/32 elements per lane,
   coalesced), not by the elected lane alone.
4. **Publication by CAS** of `nullptr` → `buf` on `d.buf`, device scope, `release` after the
   pre-fill (a `__threadfence()` before the CAS, or `atomic_ref::compare_exchange` with
   `memory_order_release`). The loser **does not retry**: it reads the winning `buf`
   (`acquire`) and keeps its buffer for later (6.3). With the claim described below, the
   publication is a release store, because the claim's CAS has already elected the writer.
5. Every lane writes its value into `buf`.

Why pre-filling before the CAS is safe: as long as the CAS has not happened, the buffer is
visible to nobody; after it, any writer that sees it has read `buf` with `acquire` and sees the
complete pre-fill. Two warps that materialise the same chunk each pre-fill their own buffer;
only one is published. No reader exists during this phase.

These are exactly the mechanisms of GPU hash tables (bump allocation, insertion by CAS, warp
aggregation): they are well known and portable to CUDA and HIP.

**What the prototype found: a chunk spans S/32 warps** (S/64 wavefronts). Warp aggregation gives
one materialisation per warp, not per chunk: on a grid-stride first write, 5.5 (S = 256) and
22.8 (S = 1024) warps per chunk lose the CAS, each after pre-filling a whole buffer for nothing
(first write 5.5x and 25x a flat write), and their buffers exhausted an arena sized at one per
chunk. Hence a coordination per chunk, between warps **(decided by the prototype)**:

- **Claim** (the default of the proxy): the elected lane installs a **busy mark** in `d.buf` (a
  misaligned address, never a buffer) by CAS from `nullptr`; the winner takes a buffer, its
  group pre-fills it, it publishes it (a release store). A lane that finds the mark waits for
  the publication with a backoff. One pre-fill per chunk, no lost buffer, an arena of one
  buffer per materialised chunk. First write: 2.2x (S = 256) and 3.4x (S = 1024). Case 1 of 6.1
  still holds under the mark: the constant does not change before the publication, and the
  pre-fill writes it.
- **Publish before waiting.** A warp over two chunks (an unaligned range) has two groups; in
  the first version one group held a claim while the other waited, and with SIMT reconvergence
  the holder could be held back behind its waiting sibling: two warps in that state wait for
  each other (measured: waits that only the bound ended, a range rewrite 80x slower). The warp
  now goes through three phases: every group makes one attempt without waiting; the groups that
  claimed pre-fill and publish; only then do the others wait, and they never claim (a released
  claim, arena exhausted, sends the write to the append-buffer).
- **A bounded wait.** Past a number of backoff rounds the write goes to the append-buffer
  (6.4). What the bound covers: two `assign()` calls on divergent paths of one AMD wavefront,
  which has no independent thread scheduling. It costs a replay, not a deadlock. On the MI300A
  no wait was given up in `proto_test` or the bench; a test of that exact case is still to
  write.
- **Race** (the scheme of steps 1 to 4) stays in the prototype as the reference. Its losers keep
  their buffer in a per-warp **spare slot** (an atomic exchange) for their next
  materialisation; `compress()` takes the spares back.
- **Chunk-aligned writes** (7.3, mitigation 2) avoid the question for the library's loops: one
  block per chunk.

### 6.3 Free-buffer stack

A LIFO stack of pointers, atomic top, **with no ABA** because no pop happens while a push is
possible: pushes only happen in `compress()` (7.2). **(Decided by the prototype)**: no push
during a write kernel. With the claim (6.2) no buffer is lost; the bump or pop that a claim
makes is always used. A buffer taken for nothing still happens on the race reference and when
two chunk-aligned blocks share a chunk (7.3). It waits in a per-warp spare slot, which the next
materialisation of that warp reuses. A spare slot that is already occupied sends its buffer to a
lost list. Both the slots and the list are given back by `compress()`.

### 6.4 Arena exhaustion: the append-buffer

When the bump goes past the arena, the write is not lost: the lane writes `(i, v)` into a
per-array **append-buffer** (`atomicAdd` on its counter, size set at creation) and raises a
failure flag. It is the GPU counterpart of the CPU version's deferred writes, with one
difference: **nothing** applies them during the kernel. At the end of the phase, the host reads
the flag, runs `compress()`, **grows the arena** (9.1), then launches the kernel that applies
the append-buffer: one write per entry, through the same proxy, with an arena that is now
large enough.

If the append-buffer itself overflows, the write is lost and the array knows it: it is an
**error** reported by `finalize()`, not a silent behaviour. The size of the append-buffer is a
setting of the array (proposed default: 1% of the arena's size).

Two writes of the same element by two threads in a kernel are a data race in the user
program, on GPU as on a `std::vector`: the library does not order them, including between a
direct write and an append-buffer entry.

## 7. End-of-phase compression

### 7.1 The pass

`compress()` is launched by the host after the write phase (after the `fence`). One **block
per materialised chunk** (or one warp per chunk for S ≤ 1024, **to be decided by
measurement**; the prototype's block per chunk reaches 73% (S = 256) and 99% (S = 1024) of the
stream read bandwidth when every chunk folds back):

1. each lane reads its S/32 elements and compares them with `buf[0]`;
2. `__all_sync` vote (`&&` reduction within the block);
3. if uniform at a value the variant can compress (`zero`: `T{}`; `zero_or_value`: `T{}` or
   the second value; `zero_any`: any value): `d.constant = buf[0]`, `d.buf = nullptr`, and the
   buffer is returned (7.2).

The pass is bandwidth-bound: re-reading every materialised buffer of a 10 GB array takes a few
milliseconds at 3 TB/s. It **replaces** at once in-flight compression, the candidate lists,
the per-chunk counters of the serial builds and `shrink_to_fit()`.

To avoid re-reading what has not changed, a **"touched" bit per chunk** (set `relaxed` by the
first writer of the chunk in the phase, cleared by the pass) can restrict the pass to the
chunks written during the phase. This bit is not a dirty bitset in the CPU sense (it gates no
read); **to be decided** by measurement on a benchmark where few chunks change per phase.

### 7.2 Returning the buffers

The buffers of the compressed chunks are pushed onto the free stack by one `atomicExch` of the
top per buffer, or better, **compacted** first (`cub::DeviceSelect` / `hipcub`) then added in
one block. A returned buffer is reused by the next materialisation (6.2, step 2); the arena
therefore only grows if the number of chunks materialised **at the same time** grows.

### 7.3 The memory peak of a phase, and its mitigations

Without in-flight compression, a phase that writes `second_value` over a whole array at `T{}`
materialises **everything** before `compress()` folds everything back. Three mitigations, in
order of preference:

1. **Kernels are short.** A GPU code already splits its work into kernels; the phase
   granularity is the kernel, not a 2,000-iteration OpenMP region. It is up to the user to put
   `compress()` where they would put a `fence` anyway.
2. **Vote before materialising.** When a block covers a whole chunk (the case of regular
   loops: S = 256, blocks of 256 threads aligned on the chunks, which the `parallel_for` of
   section 12 guarantees when it can), the block votes "all my values are equal to `v`"
   **before** materialising: if so, it writes `d.constant = v` and nothing else. This is in-flight
   compression, with no concurrency, because the block is the chunk's only writer. This
   optimisation only holds for the kernels whose split is made by the library; a user kernel
   with arbitrary indices does not have it.

   The prototype (`block_write`, `parallel_for_chunks`) does more than the vote. The block
   computes its values once, votes with `__syncthreads_or` whether one of them differs from the
   constant, and takes one buffer. It then writes the values and the constant around them in one
   pass, so the elements written are never pre-filled. The CAS stays, for two ranges that share
   a chunk. A first write costs 1.9x (S = 256) and 1.2x (S = 1024) a flat write, and a random
   range rewrite with `finalize()` costs 1.9x at S = 1024. At S = 256 a 256-thread block per
   2 KiB chunk pays its barriers, and the claim is as good. **For the facade**: the Kokkos or
   Arcane `parallel_for` over an array takes this path, and arbitrary kernels go through the
   proxy with the claim.
3. **Memory pressure forces the phase.** When the arena is exhausted, the append-buffer (6.4)
   ends the phase in effect: the host compresses, grows, replays. The arena is therefore also
   a **budget**: setting it small forces frequent compressions, at the price of replays.

## 8. The variants

### 8.1 `zero`

`constant` is always `T{}`; the pass only folds to `T{}`. Nothing else.

Implemented (stage 1) as `device::Array<T, S, Variant::zero>`: the descriptor is the pointer
alone (8 bytes, the size of the host zero variant's chunk; 16 for a `zero_any` double), so a
compressed chunk reads `T{}` without loading a constant. A host state converted into it
(`assign()`) gives a buffer to a chunk compressed to another value.

### 8.2 `zero_or_value`

`constant` is `T{}` or the array's second value (a field of the array, copied into the descriptor
by the pass). `set_second_value()` is a host operation between two phases. On GPU,
`zero_or_value` has no interest of its own over `zero_any` any more (the registry that
justified it no longer exists): it is kept for interoperability and for its semantics (compress
only to a declared value), **(to be decided)** whether it deserves separate code or an
`if constexpr` in `zero_any`.

### 8.3 `zero_any`

The descriptor carries the constant: it is the natural shape of `zero_any`, and the variant to
prototype first. What disappears: the second-value registry (`second_value_registry.h`), its
segments, its hash index, `reclaim_second_values()`, the `array_type_const_val` and
`array_second_value` lines, `CHUNK_SECOND_VALUE_INDEXED`. The cost per chunk becomes uniform
again: the descriptor.

### 8.4 `pattern`

A motif chunk = a shared motif (S integers) + an offset. The descriptor gains a shape: `buf`
points to the motif, the **low bit of `buf`** (or a `kind` field) says "motif", and `constant`
serves as the offset:

```cpp
return is_motif(d) ? motif(d)[i & mask_] + d.offset : (d.buf ? d.buf[i & mask_] : d.constant);
```

The read stays uniform per warp. **Writing** into a motif chunk materialises it, pre-filled
with `motif[j] + offset` (the pre-fill of 6.2 with a computation instead of a copy). The
**detection** of a motif is done in `compress()`: a materialised non-uniform chunk is compared
with each known motif up to an offset (`buf[j] - motif[j]` constant); the motif registry is
maintained by the **host** between phases (a motif is admitted at its second occurrence, as on
CPU), with a device search over a small table of candidate motifs. **(To be decided)** after
the `zero_any` prototype: the cost of detection against the gain on a connectivity.

## 9. Memory

### 9.1 Arenas

An array has **one arena** of S-element buffers (device allocation in one block, `cudaMalloc`
/ `hipMalloc`, or through Kokkos's `MemorySpace`), a bump counter and a free stack. Growth is
done by the host between two phases: a new arena is added to a **list of arenas** (the
existing buffers do not move, the descriptors' pointers stay valid), never by
reallocate-and-copy. The initial size is a setting of the array (proposed default: the number
of chunks the CPU variant materialises on the same benchmark, otherwise 10% of the chunks).

No `fast_allocators`: no per-rank slab, no per-type arena. A device allocator inside the
kernel (device `malloc`) is ruled out: too slow, unbounded.

### 9.2 Release

An array returns its arenas at its destruction, on the host. There is no global buffer pool
shared between arrays in this first design **(to be decided later)**: the arenas of a
destroyed array do not serve another one, unless a host pool layer is added, which is trivial
since it is outside any kernel.

### 9.3 Dynamic geometry and rechunk

The CPU rechunk (advisor, decider, rebuild by sliced `copy_range`, slab affinity) translates
into a sequence of host phases: `compress()`; per-chunk statistics (count of materialised
chunks and of sub-blocks uniform at the target size: a reduction kernel, the same as the
pass); decision on the host; rebuild by a `copy_range` kernel into a new array (one block per
target chunk, which votes on the uniformity of its sources and allocates only otherwise: the
"direct installation of the constants" of `zero_any`); swap; release of the old arenas.
Nothing concurrent: it is the part of the CPU version that ports best, because it was already
made of phases.

It comes **after** fixed geometry in the plan (section 13).

## 10. Memory model and device code rules

- **No inter-thread wait** in a kernel: no spin, no bounded wait. A lost CAS is resolved by
  reading the winner, never by retrying.
- The only atomics: `atomicAdd` (bump, append-buffer, counters), `atomicCAS` on `d.buf`,
  `atomicExch` of the stack top, one `atomicOr` of the "touched" bit. All with **device
  scope** (`cuda::atomic_ref<…, cuda::thread_scope_device>`; `__hip_atomic_*` with
  `__HIP_MEMORY_SCOPE_AGENT`).
- A single non-relaxed ordering: `release` on the publication of the pre-filled buffer,
  `acquire` on its read by the loser. Everything else is `relaxed`, and the phase boundaries
  are host `fence`s (`cudaDeviceSynchronize` / stream sync / `Kokkos::fence()`).
- Under Kokkos, where atomics are *relaxed*, the release/acquire pair is replaced by
  `Kokkos::memory_fence()` before the CAS and after reading the winner; it is the only place
  where the Kokkos backend and the native backend differ.
- No `std::atomic`, no `thread_local`, no exception, no allocation in device code. The device
  code is header-only (`__host__ __device__` / `KOKKOS_INLINE_FUNCTION`), C++20, with no
  module.
- No GenMC model for the device: the protocols come down to "one role per kernel" plus a
  publication CAS, whose argument fits in one paragraph (6.2). If a finer protocol seems
  necessary, it is the sign that the design has drifted.

## 11. Interoperability with the CPU version

- **Common format**: the device descriptor `{buf, constant}` is also a valid CPU format, with two
  differences: the CPU version points to a shared constant buffer where the device puts
  `nullptr`, and carries the OpenMP gates in the chunk object. A **converter** both ways is
  simple: per chunk, `nullptr` ↔ `first_value_buffer_` / registry buffer, and the materialised buffers
  are copied as they are (same S, same layout). It fits in one kernel plus a host loop, and
  makes the host ↔ device transfer **without decompressing**: only the materialised buffers
  and the descriptor table are transferred.
- **Coherent-memory machines** (Grace-Hopper through NVLink-C2C, MI300A): the descriptor table
  and the arenas can live in **shared** memory (`cudaMallocManaged`, system memory with ATS,
  or the MI300A's unified HBM). The host then does the orchestration (end-of-phase compression
  on the CPU, if the GPU is busy; rechunk; `pattern` registries), and the GPU the bulk reads
  and writes. The CPU version, in **phase mode** (no in-flight compression: the serial
  variant, or a region in which no pass triggers), can read and write the same array between
  two kernels. It is a second-stage goal, but it already justifies a descriptor that is
  trivially copyable and has no atomic in its type.
- The current Grace cluster (with no GPU) is used for the CPU measurements and the converter;
  the device measurements need an H100 or GH200 on the NVIDIA side, an MI250X or MI300A on the
  AMD side.

## 12. Portability layer and API

### 12.1 Languages

- **CUDA**: NVIDIA only. **HIP**: AMD, and NVIDIA by delegation to nvcc; an API almost 1:1
  with CUDA. **Kokkos**: a C++ portability library (not a language), backends CUDA, HIP, SYCL,
  OpenMP, Serial; *relaxed* atomics; does not expose the warp intrinsics (`__match_any_sync`,
  `__ballot`) portably, but leaves them reachable in `#if defined(KOKKOS_ENABLE_CUDA)` /
  `KOKKOS_ENABLE_HIP` islands.
- **Proposed decision**: the core (sections 5 to 7) is written once in **common device C++**
  -- a small abstraction header `chunk/device/intrinsics.h` that maps about ten primitives
  (scoped atomics, ballot/match/shuffle, fence, `all_sync`) onto CUDA and onto HIP, the way
  `fast_allocators` does for the CPU platforms. The **prototype is in HIP** (compiles for AMD
  and NVIDIA) or in CUDA, depending on the machine available. **Kokkos is a facade** shipped
  on top (`chunk/device/kokkos_adaptor.h`, like `tbb_adaptor.h`): an equivalent of
  `Kokkos::View` whose `operator()` is the proxy of section 6, and a `parallel_for` that marks
  the phases. This facade is what Kokkos codes use; native CUDA/HIP codes use the native
  headers.
- Compilers: nvcc ≥ 12 (C++20; not C++23, no modules; slow on heavy CRTP); hipcc/clang (C++20
  and 23). The `cpp26` branch is not a GPU base; the applications' C++20 target is the right
  one. **(To be decided)**: clang as the single CUDA compiler, which avoids nvcc and its
  limits, at the price of CUDA support one version behind.
- CMake: a `CHUNK_DEVICE=none|cuda|hip` option (and `CHUNK_DEVICE_KOKKOS=ON` for the facade),
  targets `shrink_array::zero_any_device_256`, …; Spack: a `device` variant.

### 12.2 API sketch

```cpp
namespace sa = shrink_array;

sa::zero_any::DeviceArray<double> a(n, {.chunk = 256, .arena_chunks = n / 256 / 10});

// write phase: a user kernel, or the library's parallel_for (which aligns its blocks on
// the chunks and gets the vote before materialising, 7.3)
sa::device::parallel_for(0, n, [=] __device__ (long i) { a(i) = field(i); });

a.finalize();        // fence; arena failure flag? → compress, grow, replay the append-buffer
                     // then compress(): per-block vote, compaction, buffers returned

// read phase: any kernel, no barrier
reduce_kernel<<<…>>>(a.view(), out);

a.stats();           // % of materialised chunks, arena, failures
auto h = a.to_host();                 // CPU zero_any::Array<double>, no decompression
sa::zero_any::DeviceArray<double> b = sa::device::from_host(h);
```

`a.view()` is a trivially copyable object (descriptors, shift, mask, counters) passed by value
to the kernels; the write proxy holds only `view`, `nc`, `pos`. `finalize()` is the
equivalent of `finalize_parallel_phase()` and `shrink_to_fit()` together. A write kernel
launched without `finalize()` before a read kernel violates the invariant of section 3: it is
the only usage trap, and it is documented the way the thread rule is on the CPU side
(`docs/getting_started.md` §5).

## 13. Staged plan and criteria

| stage | contents | criterion to move on |
|---|---|---|
| **0. Prototype** (1 to 2 weeks) | out of tree, HIP or CUDA, `zero_any` alone, S = 256 and 1024 fixed, three kernels (read; write with bump + warp aggregation + CAS; `compress()` by vote + CUB/hipCUB compaction); benchmark: streaming read, rewrite by random ranges as in `shrink_bench`, against a flat array | read ≤ 1.5× the flat array; write into a materialised chunk ≤ 2×; `compress()` of 10 GB materialised ≤ 10 ms. **If the read goes past 2×, the design is to be reviewed before going further** |
| 1. Core in the tree | `chunk/public/chunk/device/`: descriptor, view, proxy, arena, append-buffer, `compress()`, `intrinsics.h` CUDA + HIP; `zero_any` and `zero`; host ↔ device converter; validation tests against the CPU version on the same write sequences; CI on a GPU runner if available, otherwise compilation only | the validations pass identically on CPU and GPU; same final memory as the CPU variant at equal geometry |
| 2. Phases and benchmark | aligned `parallel_for` and vote before materialising; device `shrink_bench` (the benchmark's phases against `DeviceArray`); measurements on H100 / MI250X or MI300A; documentation (this document updated with the figures) | figures published in `variants.md` |
| 3. Kokkos facade | `kokkos_adaptor.h`, Kokkos `parallel_for` marking the phases, `memory_fence` instead of release/acquire | a Kokkos example; same validations |
| 4. `zero_or_value`, `pattern` | 8.2, 8.4 | validations; measured gain of `pattern` on a connectivity |
| 5. Dynamic geometry | 9.3 | device `rechunk_validation` |
| 6. Hybrid | coherent memory (GH200, MI300A), phase mode of the CPU version | an array read and written alternately by CPU and GPU with no copy |

**Stage 0, done on a laptop GPU (RTX A1000, sm_86, November 2026)**, 1 GiB of doubles, ratios
against a flat array (details and the bench in [gpu/proto](../gpu/proto/README.md)):

| | S = 256 | S = 1024 | criterion |
|---|---|---|---|
| read, every chunk materialised | 1.04x | 1.23x | <= 1.5x: met |
| write into materialised chunks | 0.82x | 1.08x | <= 2x: met |
| `compress()` of 10 GB, scaled to 3.35 TB/s | 4.1 ms | 3.0 ms | <= 10 ms: met |
| first write, claim / chunk-aligned (race) | 2.2x / 1.9x (5.5x) | 3.4x / 1.2x (25x) | -- |
| range rewrites + `finalize()`, claim / chunk-aligned (race) | 3.7x / 4.2x (4.4x) | 2.1x / 1.9x (4.1x) | -- |

The criteria hold on the laptop, on the GH200 and on the MI300A (below; on the MI300A the
write with the relaxed load only). The prototype uses neither
CUB nor hipCUB: `compress()` is one block per chunk with a `__syncthreads_and` vote, and the
buffers go back to the free stack with no compaction pass.

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

**Third GH200 run** (eb29b07df: 128 shards, home shard = chunk index modulo 128), 8 GiB:

| | S = 256 | S = 1024 |
|---|---|---|
| write into materialised chunks, acquire / 4 loads ahead | 1.85x / 1.24x | 1.79x / 1.21x |
| write into materialised chunks, chunk-aligned | 1.71x | 1.43x |
| first write, race / claim / chunk-aligned | 15.1x / 14.8x / 5.0x | 22.1x / 14.3x / 2.3x |
| range rewrites + `finalize()`, race / claim / chunk-aligned | 5.9x / 4.5x / 4.5x | 5.1x / 2.7x / 2.4x |
| `compress()` by warp, every chunk folds back | 62% | 67% |

- Four descriptor loads ahead bring the write down to 1.2x: the one-load-in-flight
  explanation holds, and the facade's loop will batch its descriptor loads.
- The shards helped where the counters were contended (folds at S = 256: 36% -> 62%; claim
  range rewrites), but writes into materialised chunks (1.48x -> 1.85x), the chunk-aligned
  paths and the folds at S = 1024 (83% -> 67%) got slower. Not the buffers' placement: a home
  shard per run of 1, 64 or 1024 chunks measured the same (and runs of 64 concentrated the
  allocations on one 16-buffer shard: 9x more lost CAS, a roomy arena exhausted; dropped).
  Nor the register count: the inlined slow path took the write kernel from 32 to 38
  registers (48 resident warps per SM instead of 64), but forcing 32 registers with
  `__launch_bounds__(256, 8)` gave 1.92x, and the 4-loads-ahead kernel, which then spilled
  its pointers, went from 1.24x to 2.46x; reverted. The 1.5x to 1.9x of `assign()` into a
  materialised chunk moves with unrelated code changes (1.75x at S = 1024 on the first run);
  an Nsight Compute profile of the write kernel (occupancy, stall reasons) is the next step
  instead of more guesses. What is robust: four descriptor loads ahead per thread give 1.2x,
  and the facade's loops will be written that way.
- The first write by `assign()` stays at 15x to 22x: its cost is the latency chain of a
  materialisation more than the counters; the chunk-aligned path stays the facade's answer.

**Nsight Compute profile on the GH200** (S = 256, 1 GiB, nvcc 12.4):

| kernel | registers, occupancy (theoretical / achieved) | DRAM throughput | main stall |
|---|---|---|---|
| `write_ramp<Flat>` | 32, 100% / 82% | 90% | LG queue full (bandwidth-bound) |
| `write_ramp<View>`, materialised chunks | 38, 75% / 66% | 46% | long scoreboard 69% (waits for the descriptor load) |
| `write_ramp<View>`, first write (materialises) | 38, 75% / 74% | 7% | memory barrier 46%, then long scoreboard 31%; 26 active threads per warp |

- The write into a materialised chunk is latency-bound on its descriptor load (one in flight
  per warp), with the register-limited occupancy second: four loads ahead (1.2x) address the
  first; the facade's loops will batch their descriptor loads, and `assign()` itself is not
  tuned further.
- The first write waits on the pre-fill's fences (one `__threadfence` per lane). The
  publication now uses one release CAS (exchange) by the publishing lane after the warp or
  block barrier (`PROTO_PUBLISH=1`, the default; `0` keeps the fence per lane, to compare).
  The earlier leader-fence attempt was measured together with the free-stack look and
  reverted with it, so this is its first measurement alone.

**Publication, same session** (8 GiB): one release CAS instead of a fence per lane gives the
chunk-aligned first write 4.95x -> 4.59x (S = 256) and 2.32x -> 2.17x (S = 1024), its range
rewrites 4.51x -> 4.28x and 2.40x -> 2.35x; neutral on `assign()` (race and claim within
2%). Kept as the default.

**Stage 0 on NVIDIA: done.** The three criteria hold on the GH200 (read 1.04x, write into a
materialised chunk under 2x, `compress()` of 10 GB in 4.4 to 4.8 ms). The conclusions that
carry into stage 1:

- S = 1024 is the GPU geometry.
- The facade's loops take the chunk-aligned path (first write 2.2x, range rewrites with
  `finalize()` 2.4x at S = 1024) and batch their descriptor loads (1.2x for writes into
  materialised chunks).
- `assign()` in arbitrary kernels uses the claim, with a warp-uniform grouping, and is a
  correct fallback (first write 14x, range rewrites 2.7x at S = 1024), not a fast path.
- Arena counters sharded per arena (folds at S = 256: 36% -> 62%); `compress()` one warp
  per chunk.

**MI300A** (2026-11-24, gfx942, 228 CUs in six XCDs, HBM3, 8 GiB of doubles, 57b489d70).
`proto_test` passes every case: wavefronts of 64, the claim (no wait given up), the relaxed
and plain loads, the chunk-aligned path, the replays.

| | S = 256 | S = 1024 | criterion |
|---|---|---|---|
| read, every chunk materialised | 1.47x | 1.45x | <= 1.5x: met, barely |
| write into materialised chunks, acquire / relaxed load | 66x / 1.40x | 64x / 1.34x | <= 2x: met with the relaxed load only |
| write into materialised chunks, 4 relaxed loads ahead / chunk-aligned | 1.11x / 16.7x | 1.09x / 4.2x | -- |
| `compress()` by warp, every chunk folds back: 10 GB at 3.35 TB/s | 3.9 ms | 2.8 ms | <= 10 ms: met |
| first write, race / claim, acquire load | 366x / 504x | 393x / 603x | -- |
| first write, race / claim, relaxed load | 306x / 436x | 323x / 538x | -- |
| first write, chunk-aligned | 48x | 15x | -- |
| range rewrites + `finalize()`, race / claim, acquire load | 113x / 136x | 103x / 136x | -- |
| range rewrites + `finalize()`, race / claim, relaxed load | 64x / 84x | 39x / 64x | -- |
| range rewrites + `finalize()`, chunk-aligned | 37x | 14x | -- |

- **An agent-scope acquire or release is a cache operation, not an ordering of one
  address.** The L2 of gfx942 belongs to one XCD and is not coherent with the five others:
  an acquire invalidates it (`buffer_inv sc1`), a release writes its dirty lines back
  (`buffer_wbl2 sc1`). The assembly of both programs holds them (88 and 196 occurrences,
  `-save-temps`). Their cost is per operation, whatever the address: one acquire per block and
  chunk, and no release, already makes the chunk-aligned write into materialised chunks 16.7x
  (S = 256, four times the chunks) and 4.2x (S = 1024); the first write adds the release of
  each publication (48x and 15x, against 5.0x and 2.3x on the GH200).
- **The relaxed load is the write path on AMD** (1.34x to 1.40x, 1.1x four loads ahead), and
  `proto_test` validates it there. It does not save the first write by `assign()` (-15%):
  that cost is in the materialisation, its acquires, its spins and its release.
- **Race on AMD**: ~12 lost CAS per chunk at S = 1024 (5 on the GH200) exhausted a roomy
  arena; the first run lost writes to a full append-buffer (the test, which stopped at the
  first error, now runs every case and gives the roomy cases n entries).

Stage 0 on the MI300A: the three criteria hold, the write with the relaxed load; the
materialisation does not, 15x at best. Open for stage 1, to measure before writing the
device arena: no agent-scope acquire on the write path (a kernel boundary already orders what
earlier kernels published; only the chunks published in the running kernel need more), and
no release per materialisation -- for instance a chunk-aligned path that publishes with a
relaxed CAS and appends, instead of writing into it, to a buffer that another block of the
same kernel published (rare on aligned ranges), or publications batched per block.

**Relaxed publication on the MI300A** (e1b224191, `set_relaxed_publish(true)`: see
[gpu/proto](../gpu/proto/README.md), "Coordination per chunk"). `proto_test` passes, its three
relaxed cases included. The chunk-aligned path, before -> after:

| | S = 256 | S = 1024 | GH200, acquire/release (S = 256 / 1024) |
|---|---|---|---|
| first write | 47x -> 13.9x | 15x -> 3.4x | 5.0x / 2.3x |
| write into materialised chunks | 16.6x -> 1.53x | 4.2x -> 0.79x | 1.71x / 1.43x |
| range rewrites + `finalize()` | 36x -> 7.6x | 13.6x -> 3.0x | 4.5x / 2.4x |

Without acquire or release in the kernel, the MI300A at S = 1024 comes within 1.5x of the
GH200 with them.

**And on the GH200** (same commit; `proto_test` passes), before -> after: first write 4.72x ->
4.20x (S = 256) and 2.17x -> 1.83x (S = 1024); writes into materialised chunks 1.83x -> 1.81x
and 1.12x -> 1.16x; range rewrites + `finalize()` 5.23x -> 5.05x and 2.09x -> 2.21x. It costs
nothing on NVIDIA and gains on the first write: **the relaxed publication is the chunk-aligned
path of stage 1, on both vendors (decided by the prototype).**

**Stage 1, started (November 2026).** The functions the host and device versions share are in
the library, compiled for both:

- `CHUNK_HD` (`chunk/internal/host_device.h`) is `__host__ __device__` in a CUDA or HIP
  translation unit and nothing otherwise. It marks the geometry (`StaticChunkSize`,
  `ChunkElems`, `log2int`), the value identity (`same_value`, `identity_less`,
  `ValueIdentity::same`, the parts of `std::complex` and `std::array` included) and the
  uniformity test of a chunk, `internal::walk_constant` (`chunk/internal/constant_walk.h`),
  which the CPU's compression scans call and the device `compress()` will call.
- `device_core_validation` (`-DCHUNK_DEVICE_CORE_CHECK=ON`) runs them in a CUDA kernel and
  compares them with the host, signed zeros and NaN payloads included. It builds with
  `-Werror=all-warnings` under nvcc, which only warns when such a function calls a host one.
  It passes with nvcc 13.1 (T1000, sm_75), on the GH200 (sm_90) and with clang 23.

In `chunk/public/chunk/device/`, from the prototype (`gpu/proto/zero_any_device.h`):

- `chunk_desc.h`, `view.h`: the chunk descriptor and the read path (`device_core_validation`).
- `backend.h`, `atomics.h`: CUDA or HIP, chosen by the compiler; the host calls (allocate,
  copy, synchronize, throwing on failure) and the device atomics, all relaxed (13, MI300A).
- `arena.h`: `DeviceArena`, the arenas of buffers with 128 shards each (a bump and a
  free-stack segment per shard), taken by the write kernels, given back by the compression
  pass, grown between phases. `device_arena_validation` checks concurrent takes (each buffer
  once, exhaustion, no overlap), give-backs, growth and home shards; it builds for CUDA or HIP
  (`-DCHUNK_DEVICE_LANGUAGE=HIP`) and passes on the T1000, the GH200 and the MI300A.

- `warp.h`, `block_write.h`, `array.h`: the chunk-aligned write with the relaxed publication
  (13). `Writer::write` (one block per call, a part of one chunk): descriptor loaded relaxed,
  vote, a buffer from the arena published by a relaxed CAS, tagged fresh when the block
  writes only part of the chunk (`is_fresh`, `chunk_desc.h`); every block marks its elements
  in the chunk's `written` bitmap, and `settle_fresh` fills the others with the constant
  after the kernel. `device::Array` owns the descriptors, the arena and the lists:
  `parallel_for_chunks(lo, hi, f)`, or `writer()` in a kernel of the caller's and
  `finish(f)` after it. When the arena runs out, a block writes nothing in that chunk and
  records its span; `finish` grows the arena and writes the span again with the same f (no
  append-buffer: f must be a function of i). A lost CAS parks its buffer in a list given back
  at `finish`. `device_write_validation` checks it against a host mirror (ranges shared by
  two blocks, two kernels before one `finish`, writes of the constant only, small arenas,
  tail chunks, a non-zero constant), and fails when the settle or the rewrite of the failed
  spans is broken. It passes on the T1000, the GH200 and the MI300A.

- `compress.h`: the compression pass, `Array::compress()`. One warp per chunk, grid-stride;
  a materialised chunk whose elements below n are one value (`same_value`) takes it as its
  constant and gives its buffer back to the arena (after `normalise()`, no take meanwhile);
  the tail of the last chunk is not compared. `device_write_validation` runs it after every
  round against a host model (folds counted, elements, materialised chunks and buffers in use
  checked again), and fails when the tail is compared or a buffer is not given back. It
  passes on the T1000, the GH200 and the MI300A.

- The host <-> device conversion: `Array::assign(state)` loads an `ArrayState` (the
  `export_array()` of a host array, `chunk_exchange.h`), `Array::image()` gives one back for
  `import_array()`. Only the materialised chunks move (a motif expanded), in batches of 64 MiB;
  a compressed chunk is its descriptor. nvcc cannot compile the host arrays' headers (it
  rejects their inline static data members of templates, section 14), so a program converts
  in two translation units: `export_array()` / `import_array()` in a C++ one, `assign()` /
  `image()` in the device one, the state passed between. The state's types moved to
  `chunk/internal/chunk_state.h`, which both include (`chunk_exchange.h`'s interface is
  unchanged). `device_convert_validation` is built that way: a host `zero_any::Array` with
  first-value, second-value and materialised chunks goes to the device, is written and
  compressed there (ten chunks fold to 2.5, a second value again on the host), and comes back
  equal to a host mirror. It passes on the T1000, the GH200 and the MI300A. With HIP, both translation
  units must use one libstdc++: ROCm's clang picks a GCC of its own (gcc-toolset-12 on the
  MI300A node, against GCC 14.1 for the C++ one, and the link failed), so the build passes
  `-DCMAKE_HIP_FLAGS=--gcc-install-dir=<the C++ compiler's>`; the configure skips the test
  with that hint otherwise.

- `examples/device_bench`: the measurements of the prototype's bench on `device::Array`
  (read, chunk-aligned write, range rewrites, `compress()`, conversion), against a flat
  array, to check the port. T1000 (0.5 GiB, S = 256 / 1024): read 1.02x / 0.96x, first write
  1.59x / 0.86x, write into materialised chunks 0.92x / 0.80x, range rewrites + `compress()`
  4.8x / 2.0x, `compress()` of 10 GB 2.5 / 2.4 ms when everything folds; `assign()` and
  `image()` 3 GB/s (pageable host memory, a per-element staging loop: not tuned).
  On the big GPUs (8 GiB of doubles), library against prototype (relaxed publication):

  | S = 1024 | GH200 | GH200 proto | MI300A | MI300A proto |
  |---|---|---|---|---|
  | read, every chunk materialised | 1.04x | 1.03x | 1.32x | 1.45x |
  | first write, chunk-aligned (+ `finish()`) | 1.56x | 1.83x | 1.60x | 3.4x |
  | write into materialised chunks | 1.17x | 1.16x | 1.21x | 0.79x |
  | range rewrites + `compress()` | 2.41x | 2.21x | 3.31x | 2.99x |
  | `compress()` of 10 GB, every chunk folds | 3.5 ms | 3.5 ms | 3.7 ms | 2.9 ms |

  At S = 256: read 1.04x / 1.33x, first write 3.50x / 4.33x, write into materialised chunks
  1.67x / 2.02x, `compress()` 4.0 / 4.7 ms (GH200 / MI300A). The port lost nothing on the
  GH200; on the MI300A the first write is twice as fast as the prototype's (which counted
  every materialisation on one global counter) and the write into materialised chunks
  slower (1.21x against 0.79x at S = 1024, 2.02x at S = 256: at the criterion), to be looked
  at. The conversion: 12 to 18 GB/s on the GH200 (not tuned).

- `Writer::assign` / `load`, `Array::finish()`: the writes at arbitrary indices without
  device-scope acquire or release, designed and measured in [gpu_assign.md](gpu_assign.md);
  `device_assign_validation` passes on the T1000 and the MI300A. The warp aggregation of the
  claims followed (gpu_assign.md, section 5: on the MI300A no wait given up, scattered writes
  1.05x to 1.69x the flat scattered write, the grid-stride first write 57x / 42x at S = 256 /
  1024).

  MI300A rerun of the whole bench (491a05a87, 8 GiB): against the 2026-11-24 table, the
  chunk-aligned first write 5.07x / 1.75x (was 4.33x / 1.60x) and the write into materialised
  chunks 2.15x / 1.27x (was 2.02x / 1.21x; past the criterion at S = 256), read 1.33x / 1.31x,
  `compress()` of 10 GB 4.7 / 3.6 ms. Measured against 407674ad5 (before `assign()`) in one
  slot, part of that gap is the node's day to day (407674ad5 itself: 4.46x / 1.64x, and 2.08x
  for the write into materialised chunks at S = 256, already past the criterion); the rest was
  real on the materialising writes, +13% / +7% on the first write, from one more device-scope
  atomic per take (the spare slot tried before the arena). With the arena first (fb14a408d):
  9.64 / 3.47 ms against 9.41 / 3.44 ms before `assign()`, the range rewrites equal; the write
  into materialised chunks keeps +2% to +3.5% (the check for an `assign()` claim). The
  conversion: 3.5 GB/s for `assign()`, 1.5 GB/s for `image()` (0.25 GiB; at 1 GiB the job ran
  out of host memory), not tuned.

- The `zero` variant (8.1): `Variant` (`chunk_desc.h`), a template parameter of `ChunkDesc`,
  `View`, `Writer` and `Array`, `zero_any` by default. `ChunkDesc<T, Variant::zero>` holds the
  pointer alone and `constant_of(d)` reads `T{}` for it (`zero_value<T>()`: from zero bytes for
  `std::complex` and `std::array` in device code, whose constructors are host functions to
  nvcc); the pass folds only to `T{}`; the constructor refuses another constant; `assign(state)`
  materialises a chunk compressed to another value than `T{}`. The validations run both variants:
  `device_core_validation` (the view, the descriptor's size), `device_write_validation` and
  `device_assign_validation` (folds to +0.0 only, -0.0 is another value), and
  `device_convert_validation` (a host `zero_any` array into a device `zero` one: its ten
  second-value chunks get a buffer and come back materialised). They pass on the T1000 and the
  MI300A (no claim wait given up there either). `device_bench` measures both variants (a fourth
  argument picks one). T1000, 0.5 GiB: the same within the noise everywhere but the scattered
  `assign()` into materialised chunks, 1.67x against 1.97x at S = 1024 (2.14x against 2.22x at
  S = 256): a random access loads its descriptor, half the size. The read of a compressed array
  does not change there (0.45x): the descriptors stay in the cache at that size.
  MI300A, 8 GiB (bb4207da2), zero against zero_any at S = 256 / 1024: the read of a compressed
  array 0.38x / 0.35x the flat read against 0.65x / 0.62x (-42% / -44%: the read is a descriptor
  load, 32 MB of them against 64 MB at S = 256), `compress()`'s scan -24% / -16%, the
  chunk-aligned write into materialised chunks 2.09x / 1.09x against 2.15x / 1.27x, its first
  write 4.70x / 1.65x against 5.07x / 1.75x, the range rewrites with `compress()` -8%; the read
  of a materialised array, `+=` and the scattered writes into materialised chunks do not move
  (their buffers dominate). At S = 256 the write into materialised chunks stays past the 2x
  criterion in both variants.

- The criterion "same final memory as the CPU variant at equal geometry":
  `device_memory_validation_zero_any` and `_zero` run the same six rounds of writes on a host
  array and on a device array of the same variant (by `Writer::write` per span and by
  `Writer::assign` per element: 0.0 and -0.0, a constant per chunk, NaN with a payload, the
  partial last chunk, zeros everywhere), and compare the layouts chunk by chunk after every
  `shrink_to_fit()` / `compress()`: kinds, constants bit for bit, elements. They are equal, but
  for one known difference: in `zero_any`, a partial last chunk uniform over its valid elements
  folds on the device and keeps its buffer on the host. The host tests a chunk over its
  `chunk_size()` elements, the tail past `size()` included (it holds the constant of the
  chunk's materialisation); the device compares the elements below n. One buffer more on the
  host, at most one per array. **CPU defect, to fix on develop if wanted**: pass the valid
  count of the last chunk through `ChunkElems` (the write proxy's three enqueues, the arrays'
  sweeps, the rechunk fold), with a count that a `resize()` between the enqueue and the pass
  cannot leave stale, and the deferred writes' replay. Not in `zero` (its tail is 0, its only
  fold value). Both pass on the T1000 and the MI300A, with the same layouts.

- CI (`.github/workflows/device.yml`, on the `gpu` branch): compilation only, the hosted runners
  have no GPU. nvcc 13.1 for sm_75 and sm_90, and ROCm 7.2.4 for gfx942, build every device
  validation and `device_bench` (first run green, 2026-11-25).

**Stage 3, started (November 2026): the Kokkos facade.** The benchmarks target Kokkos and Arcane
codes, Kokkos first. `chunk/device/kokkos_adaptor.h`: `device::kokkos::Array<T, S, V>` owns a
`device::Array`; `write()` gives a write access (`w(i) = v`, `w(i) += v`, `-=`, `*=`, `/=`,
`T x = w(i)`: `Writer::assign` / `load` from a `KOKKOS_LAMBDA`), `end_write()` ends the phase
(fence, `finish()`), `read()` gives the `View` of a read phase, `parallel_for_chunks(lo, hi, f)` is
the library's chunk-aligned loop (its own kernels: Kokkos's CUDA teams sit on `threadIdx.y`),
`compress()`, `bytes()` / `arena_bytes()`. `-DCHUNK_DEVICE_KOKKOS=ON -DCHUNK_KOKKOS_ARCH=TURING75 |
HOPPER90 | AMD_GFX942_APU` fetches Kokkos 5.3.0 (SHA-256 pinned; `CHUNK_KOKKOS_URL` takes a local
tarball where CMake's curl cannot reach GitHub), built for CUDA or HIP as a CMake language, without
rocThrust. On the MI300A the HIP compiler must not be pointed at GCC 14.1, whose `<format>` it
rejects (`++this` in `bits/unicode.h`; Kokkos includes `<chrono>`): GCC 15 or 16.
`kokkos_validation` passes on the T1000 and the MI300A.

Plugging Kokkos in found four faults of the library, now fixed: `Writer::assign` / `load` were
`__device__` only (now `CHUNK_HD`, a host call aborts; a `KOKKOS_IF_ON_DEVICE` branch in the
caller's lambda failed to launch); `lane_id()` was `threadIdx.x % warp_size`, 0 for every lane of a
Kokkos `RangePolicy` block (its threads on y), and the warp aggregation never ended (now the
hardware lane); the arena grew by small steps until `max_arenas` (now by a quarter at least); and
a write of the chunk's constant after an append of the same element by the same thread was
dropped, the replay then applied the older value (MI300A, wrong checksums from run to run;
`device_assign_validation` step 0 reproduces it on the T1000).

`examples/kokkos/kokkos_shrink_bench`: shrink_bench's access pattern on five
`Kokkos::View<double*>` and on five facade arrays, the same kernels: the init bands
(`bench_phase_fill.cpp`, by `parallel_for_chunks`), then per iteration a random window
(`bench_phase_compute.cpp`: the writes before and after it, the window rewritten, 20 read passes,
2 read-modify-write passes), a reset every iters/10, `compress()` every N. MI300A, 5 x 2^30
doubles, 2000 iterations, windows of 10^5 (871d5a370 -> a777dfc35, the listed settle below):

| | zero_any | zero |
|---|---|---|
| memory held at the end / the views' 42.95 GB | 1.51 GB (3.5%) | 19.2 GB (44.6%) |
| with the arenas' free buffers | 1.98 GB (4.6%) | 22.3 GB (52%) |
| init (chunk-aligned) | 0.55x -> 0.54x | 47.5x -> 46.6x |
| allocation + init (c4dad808f) | -- | 0.64x |
| loop | 2.84x -> 2.09x | 2.26x -> 1.86x |
| 20 read passes / 2 read-modify-write passes | 1.00x / 4.44x -> 1.01x / 4.15x | 1.01x / 4.14x -> 1.01x / 4.04x |
| writes around the window / window rewritten | 36x / 39x -> 15.2x / 21.7x | 24x / 19x -> 11.6x / 10.6x |
| claim waits (a777dfc35) | 3.6 million | 2.0 million |

Checksums equal to the views' in both runs, no wait given up. The init had been 23x and the arenas 20 GB until a compressed chunk
written whole with one value took it as its constant (`Writer::fold_whole`, 7.3). The small phases
(a window is about 100 chunks an array, 15 us flat) pay `end_write()`: a counter read, the settle,
fences. Since that run, `finish()` decides every step from one counter read, the settle visits
the chunks a phase published fresh only (a fresh list) without a synchronisation, and `write()`
no longer fences: T1000, 5 x 2^24, the loop from about 2.0x to 1.24x-1.46x (zero_any) and
1.07x-1.26x (zero), checksums equal; the MI300A from 2.84x to 2.09x and 2.26x to 1.86x (the
table). What the loop pays now, zero_any: the reads (1.21 s of 2.88, at the views' speed), the
writes around and in the window (about 1 s against 55 ms flat, 3.6 million claim waits: threads
waiting for another to materialise their chunk), the read-modify-write passes (0.5 s), compress()
(0.17 s). zero's init (47x) is its allocation: the views' (zero-initialised, 1.84 s) was not
timed, the arenas' growth falls in the fill (c4dad808f times both). Allocation and init together:
1.29 s, 0.64x the views'. An arena sized from the start (1/2 of the chunks) moves it: 0.92 s of
allocation, a fill of 0.24 s (the first touch of 18 GB of buffers), 1.16 s in all; the failed
first pass and the second cost about 50 ms. Nothing to change in the library.

The claim waits were not the cost of those writes: what the waiters waited for was. A take past
an empty home shard looked at the shards one by one, two dependent loads each, 29 on average
(T1000, small arenas); each arena now has a mask of its shards worth a try (`ArenaDesc::avail`,
cleared by a take that finds a shard empty, set by a give back), the next one found by ffs. And
the five arrays of a phase waited once each in `end_write()`, a synchronisation and a blocking
copy of the counters: `kokkos::end_write(a1, ..., a5)` (`device::finish_all`) enqueues the five
copies and waits once. T1000, 5 x 2^24, 400 iterations, interleaved runs: zero_any writes around
the window 76 -> 52 -> 43 ms, window rewritten 58 -> 49 -> 39 ms, loop 1.50x -> 1.39x -> 1.20x;
zero 38 -> 27 ms and 27 -> 17 ms, loop 1.21x -> 1.13x (the mask leaves zero as it was). MI300A,
5 x 2^30, 2000 iterations (7ee357b5e against a707bbcf3), checksums equal, no wait given up, the
memory unchanged:

| | zero_any | zero |
|---|---|---|
| loop | 2.08x -> 1.77x | 1.87x -> 1.54x |
| writes around the window | 440 -> 354 ms (12.6x) | 338 -> 255 ms (9.1x) |
| window rewritten | 529 -> 450 ms (18.2x) | 271 -> 171 ms (6.9x) |
| 2 read-modify-write passes | 499 -> 256 ms (2.1x) | 486 -> 240 ms (2.0x) |

The profile (rocprofv3, 5 x 2^26): a window's write phase took about 260 us, of which five
`hipMemset` of the fresh count after each settle (13 us of host time each), five asynchronous
copies of the counters (5.5 us and a blit kernel each), and the two bench kernels at 27 us each
against 3.4 us flat. `settle_listed` now empties the list itself and `finish_all` gathers the
counters of up to 16 arrays with one kernel (7381cc5ba): loop 1.77x -> 1.64x (zero_any) and
1.54x -> 1.45x (zero); writes around the window 354 -> 258 ms and 255 -> 193 ms, window rewritten
450 -> 380 ms and 171 -> 135 ms, read-modify-write 256 -> 220 ms and 240 -> 204 ms; checksums
equal.

Next: the write kernels themselves (27 us against 3.4); giving free arenas back; the GH200; then
Arcane.

## 14. Risks and open questions

- **Reads beyond 1.5×**: if the latency of the indirection is not hidden on a kernel that is
  already bandwidth-bound, the format has no GPU future as it stands; that is why stage 0
  measures the read first.
- **Kernels with arbitrary indices** (irregular gather/scatter): a warp then touches several
  chunks, warp aggregation degrades into several allocations, and the accesses are no longer
  coalesced -- but they would not be on a flat array either. To be measured separately.
- **Lost CAS**: not rare, as it turned out: S/32 warps per chunk on a regular loop (6.2). The
  claim removes them; what is left is the claim's wait (no wait given up on the MI300A).
- **Agent-scope acquire and release on AMD** (13, MI300A): an L2 invalidation or write-back
  each, which makes every materialisation 15x to 600x a flat write there. Stage 1 has to
  remove them from the write path, not tune them.
- **nvcc** on the library's template code: compile time and front-end bugs on concepts;
  clang-CUDA is the way out.
- **The motif registry** of `pattern` maintained by the host: detection is costly if the
  number of candidate motifs is large; to be bounded as on CPU (one motif per shape).
- **Which users**: Kokkos, Arcane, native CUDA/HIP codes? The answer decides whether the
  Kokkos facade is stage 3 or stage 1, and whether Arcane's accelerator layer should be a
  second facade.
- **The measurement machines**: a GH200 and an MI300A are available on the cluster for the
  absolute figures; the laptop's RTX A1000 gave the ratios of stage 0.
