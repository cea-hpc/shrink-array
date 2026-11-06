# GPU port: first design document

> Version française : [conception_gpu_fr.md](conception_gpu_fr.md).

> Status: October 2026, **before any prototype**. This document sets out what a GPU backend of
> shrink-array should be, what it takes from the CPU version and what it leaves behind, and the
> plan for deciding by measurement. The figures given are expectations, not results.
> The decisions marked **(to be decided)** wait for the prototype of section 13.

***

## 1. Goal and scope

**Goal.** Offer on GPU (NVIDIA and AMD) an array that is read and written in a kernel like a
flat array, `a(i)` / `a(i) = v`, and that does not store its uniform chunks -- the same data
format and the same guarantees (lossless, in-place writes, any repeated value compressed by
`zero_any`) as the CPU version. The gain sought is first **memory**: 80 to 192 GB of HBM is
where "20 times less" changes what a code can handle.

**In the scope of this first design:**

- the three variants `zero`, `zero_one_value`, `zero_any`, and `pattern` (section 8);
- fixed geometry (S a compile-time constant) first, dynamic geometry next (section 9.3);
- an execution model **by phases**, aligned on the kernels (section 3);
- interoperability with the CPU version: same chunk format, transfers both ways, hybrid mode
  on coherent-memory machines (section 11).

**Out of scope:**

- running the existing CPU code on GPU: the execution machinery (section 2) is not portable
  and does not need to be;
- writes concurrent with a compression **inside** a kernel;
- bit-packed `bool` (`ArrayChunkBits`);
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
| user kernel, writing | writers | two threads materialise the same chunk | CAS on the descriptor; the loser gives up its arena slot, reclaimed at the end of the phase (6.3) |
| `compress()` (end of phase) | the pass | none: no writer, no reader | per-block vote + compaction (7) |
| `rechunk_to()` | host + kernels | none | sequence of phases (9.3) |

Consequences:

- **nobody ever waits for anybody** in a kernel: the only atomic of the write path is a CAS
  with no wait loop (the loser does not retry, it reads the winner);
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
    T        cst;                // the constant of the compressed chunk (or the one left: 6.2)
    // zero: cst is always T{}; zero_one_value: T{} or the array's second value;
    // zero_any: any value.  pattern: see 8.4.
};
```

For a 16-byte `T` (`std::complex<double>`) the descriptor is 24 bytes, aligned to 32
**(to be decided: 32 aligned bytes, or a tagged pointer + side table; measure)**. The
materialisation CAS is on `buf` alone (8 bytes): `cst` is written only by the compression pass
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
| `ChunkDesc[nb_chunks]` | 16 or 24 B per chunk | device (or coherent memory, 11) |
| the arena's bump counter, top of the free-buffer stack | 2 words | device |
| append-buffer of the writes not applied (6.4) | bounded, set at creation | device |
| statistics (`stats()`, materialised chunks, arena failures) | a few words | device, read back by the host at the end of the phase |

There is **no** dirty bitset, no candidate list, no "already listed" line: the compression
pass looks at every materialised chunk (7.1), which on GPU costs less than maintaining lists.

## 5. Reading

```cpp
__device__ T operator()(IndT i) const noexcept
{
    const auto& d = desc_[i >> shift_];
    return d.buf ? d.buf[i & mask_] : d.cst;     // branch uniform per warp
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
    T* buf = atomic_load_relaxed(d.buf);          // device scope
    if (buf == nullptr)
    {
        if (v == d.cst) return;                   // 1. the constant onto its constant: nothing
        buf = materialise(d);                     // 2. materialise (6.2), then write
    }
    buf[pos] = v;                                 // 3. materialised chunk: direct write
}
```

Cases 1 and 3 are the hot path, uniform per warp, with no atomic. The rules of
`zero_one_value` ("the second value onto an element that already holds it") and of `zero_any`
("discover", "leave", `may_become_uniform`) fed the candidate lists: they **disappear**, the
end-of-phase pass needs no candidates. What remains to handle is `+=` and the other compound
operators, which read before writing: same cases, with `v` computed from `d.cst` or
`buf[pos]`.

### 6.2 Materialisation

1. **Warp aggregation.** The lanes of the warp whose chunk is the same (in practice all of
   them: 4.2) elect one lane (`__match_any_sync` / `__ballot`, `__activemask`); the others
   wait for the elected lane's result through a shuffle. One materialisation per warp and per
   chunk, not per thread.
2. **Allocation**: the elected lane takes a buffer from the **free-buffer stack** (returned
   by the previous compressions, 7.2), otherwise by **bump** on the array's arena
   (`atomicAdd` on the counter, one per buffer). If the arena is exhausted: 6.4.
3. **Pre-filling** of the buffer with `d.cst`: by **the whole warp** (S/32 elements per lane,
   coalesced), not by the elected lane alone.
4. **Publication by CAS** of `nullptr` → `buf` on `d.buf`, device scope, `release` after the
   pre-fill (a `__threadfence()` before the CAS, or `atomic_ref::compare_exchange` with
   `memory_order_release`). The loser **does not retry**: it reads the winning `buf`
   (`acquire`) and returns its buffer to the free stack (or gives it up until the end of the
   phase: **to be decided**, the free stack costs one more atomic, giving it up costs memory
   until `compress()`).
5. Every lane writes its value into `buf`.

Why pre-filling before the CAS is safe: as long as the CAS has not happened, the buffer is
visible to nobody; after it, any writer that sees it has read `buf` with `acquire` and sees the
complete pre-fill. Two warps that materialise the same chunk each pre-fill their own buffer;
only one is published. No reader exists during this phase.

These are exactly the mechanisms of GPU hash tables (bump allocation, insertion by CAS, warp
aggregation): they are well known, portable to CUDA and HIP, and wait-free.

### 6.3 Free-buffer stack

A LIFO stack of pointers, atomic top, **with no ABA** because no pop happens while a push is
possible: pushes only happen in `compress()` (7.2) and in case 4 of 6.2 (the loser of a CAS,
during a write kernel where pops are possible). **(To be decided)**: either the loser gives up
its buffer until the end of the phase (no push during the pops, the stack is trivial), or the
stack gets a generation counter. The first form is preferred for its simplicity; the cost is a
transient memory bounded by the number of lost CAS, rare with warp aggregation.

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
measurement**):

1. each lane reads its S/32 elements and compares them with `buf[0]`;
2. `__all_sync` vote (`&&` reduction within the block);
3. if uniform at a value the variant can compress (`zero`: `T{}`; `zero_one_value`: `T{}` or
   the second value; `zero_any`: any value): `d.cst = buf[0]`, `d.buf = nullptr`, and the
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
   **before** materialising: if so, it writes `d.cst = v` and nothing else. This is in-flight
   compression, with no concurrency, because the block is the chunk's only writer. This
   optimisation only holds for the kernels whose split is made by the library; a user kernel
   with arbitrary indices does not have it.
3. **Memory pressure forces the phase.** When the arena is exhausted, the append-buffer (6.4)
   ends the phase in effect: the host compresses, grows, replays. The arena is therefore also
   a **budget**: setting it small forces frequent compressions, at the price of replays.

## 8. The variants

### 8.1 `zero`

`cst` is always `T{}`; the pass only folds to `T{}`. Nothing else.

### 8.2 `zero_one_value`

`cst` is `T{}` or the array's second value (a field of the array, copied into the descriptor
by the pass). `set_second_value()` is a host operation between two phases. On GPU,
`zero_one_value` has no interest of its own over `zero_any` any more (the registry that
justified it no longer exists): it is kept for interoperability and for its semantics (compress
only to a declared value), **(to be decided)** whether it deserves separate code or an
`if constexpr` in `zero_any`.

### 8.3 `zero_any`

The descriptor carries the constant: it is the natural shape of `zero_any`, and the variant to
prototype first. What disappears: the second-value registry (`tab_cst_val2_holder.h`), its
segments, its hash index, `reclaim_second_values()`, the `tab_type_cst_val` and
`tab_second_value` lines, `CHUNK_SECOND_VALUE_INDEXED`. The cost per chunk becomes uniform
again: the descriptor.

### 8.4 `pattern`

A motif chunk = a shared motif (S integers) + an offset. The descriptor gains a shape: `buf`
points to the motif, the **low bit of `buf`** (or a `kind` field) says "motif", and `cst`
serves as the offset:

```cpp
return is_motif(d) ? motif(d)[i & mask_] + d.offset : (d.buf ? d.buf[i & mask_] : d.cst);
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

- **Common format**: the device descriptor `{buf, cst}` is also a valid CPU format, with two
  differences: the CPU version points to a shared constant buffer where the device puts
  `nullptr`, and carries the OpenMP gates in the chunk object. A **converter** both ways is
  simple: per chunk, `nullptr` ↔ `cst_val1` / registry buffer, and the materialised buffers
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
  targets `chunk::chunk_zero_any_device_256`, …; Spack: a `device` variant.

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
| 4. `zero_one_value`, `pattern` | 8.2, 8.4 | validations; measured gain of `pattern` on a connectivity |
| 5. Dynamic geometry | 9.3 | device `rechunk_validation` |
| 6. Hybrid | coherent memory (GH200, MI300A), phase mode of the CPU version | an array read and written alternately by CPU and GPU with no copy |

## 14. Risks and open questions

- **Reads beyond 1.5×**: if the latency of the indirection is not hidden on a kernel that is
  already bandwidth-bound, the format has no GPU future as it stands; that is why stage 0
  measures the read first.
- **Kernels with arbitrary indices** (irregular gather/scatter): a warp then touches several
  chunks, warp aggregation degrades into several allocations, and the accesses are no longer
  coalesced -- but they would not be on a flat array either. To be measured separately.
- **Lost CAS**: rare with warp aggregation on regular loops, but an irregular kernel that
  materialises the same chunk from thousands of warps wastes as many buffers until the end of
  the phase (6.3). A `cas_lost` counter in `stats()` will tell whether the free stack with a
  generation counter is needed.
- **nvcc** on the library's template code: compile time and front-end bugs on concepts;
  clang-CUDA is the way out.
- **The motif registry** of `pattern` maintained by the host: detection is costly if the
  number of candidate motifs is large; to be bounded as on CPU (one motif per shape).
- **Which users**: Kokkos, Arcane, native CUDA/HIP codes? The answer decides whether the
  Kokkos facade is stage 3 or stage 1, and whether Arcane's accelerator layer should be a
  second facade.
- **The measurement machines**: the Grace cluster has no GPU; stage 0 needs access to an
  H100/GH200 or an MI250X/MI300A, even a modest one (the prototype is small).
