# `assign()` on the device without device-scope acquire or release

> Version française : [gpu_assign_fr.md](gpu_assign_fr.md).

> Status: design note, November 2026, for review before any code. It replaces, for stage 1,
> the materialisation of [gpu_design.md](gpu_design.md) 6.1 and 6.2 (pre-fill, release
> publication, claim) by the deferred fill that the chunk-aligned path already uses
> (gpu_design.md, section 13, "relaxed publication").

## 1. The problem

The library writes a device array through the chunk-aligned path only: one block per call, a
part of one chunk (`Writer::write`). A kernel of the application that writes `a(i) = v` at
arbitrary indices, one thread per element, has nothing yet. The prototype's `assign()` did it with
the protocol of gpu_design.md 6.2:

- a device-scope **acquire** load of the descriptor on every write;
- on a compressed chunk, a buffer **pre-filled** with the constant by the warp, then published by
  a **release** CAS; with the claim, the other warps **wait** for the publication.

On the GH200 that cost 14x to 20x a flat write on a first write; on the MI300A 300x to 600x,
because an agent-scope acquire invalidates the L2 of an XCD and a release writes it back
(gpu_design.md, 13). The relaxed load alone brought writes into materialised chunks to 1.34x on
the MI300A, but the materialisation stayed at 300x and more.

## 2. What the chunk-aligned path already showed

The relaxed publication (gpu_design.md, 13) removed every acquire and release from the
chunk-aligned write, and its first write went from 15x to 1.6x a flat write on the MI300A (library,
S = 1024). Its idea carries over: **no pre-fill before the publication**. A buffer is published
empty, tagged fresh; every writer writes its own elements and sets their bits in the chunk's
`written` bitmap; after the kernel, `settle_fresh` writes the constant where no bit is set. Nobody
ever reads an element that another thread has not yet written, so nothing has to order one
thread's stores before another's. Nothing waits either.

## 3. The proposal

### 3.1 The write

```cpp
__device__ void Writer::assign(std::int64_t i, T v) const
{
    const std::int64_t c = i >> shift;
    const unsigned     e = static_cast<unsigned>(i & mask);
    ChunkDesc<T>&      d = desc[c];
    T* buf = load_ptr(&d.buf);                   // relaxed
    if (buf == nullptr)
    {
        if (same_value(v, d.constant)) return;   // 1. the constant onto its chunk: nothing
        buf = materialise(c, d);                 // 2. a fresh buffer, or nullptr (arena exhausted)
        if (buf == nullptr) { append(i, v); return; }
    }
    if (is_fresh(buf))                           // 3. fresh: the element, then its bit
    {
        untag(buf)[e] = v;
        fetch_or(&written[c * words + e / 32], 1U << (e % 32));
        return;
    }
    buf[e] = v;                                  // 4. a buffer of an earlier kernel
}
```

`materialise` looks at the descriptor again (a chunk published meanwhile costs nothing), takes a
buffer (the warp's spare slot, else the arena: `DeviceArena::take`), and publishes it tagged fresh
by a **relaxed** CAS from `nullptr`. The loser keeps its buffer in its spare slot and writes into
the winner's. No pre-fill, no claim, no wait, no fence. A buffer taken from the arena is never read
before its elements are written: its old contents do not matter.

Case 4 is the hot path: a relaxed load and a store, measured at 1.34x (MI300A) and 1.76x (GH200)
a flat write in the prototype, 1.1x to 1.2x with the descriptor loads issued ahead.

### 3.2 Reads in a write kernel

`a(i) += v` reads first. `Writer::load(i)` follows the same cases: the constant of a compressed
chunk; in a fresh buffer the element if its bit is set, else the constant; in an older buffer
the element. A thread always reads what it wrote itself (same thread, same address), and what a
thread of its block wrote before a `__syncthreads()` (block scope). Communication between blocks
inside a kernel needs device-scope fences in the application, as on a flat array; they order the
descriptor, the bitmap and the elements alike.

`View` (the read path) is for read kernels only and never sees a fresh tag: `finish()` settles.

### 3.3 Lost publications

On a grid-stride first write, the S/32 warps (S/64 wavefronts) that share a chunk all find it
compressed at once. Without a pre-fill a lost publication costs one take and one exchange into the
warp's spare slot, which its next materialisation reuses (the prototype's mechanism). The arena
pressure is bounded: one buffer per materialised chunk, plus one per resident warp in the spare
slots, plus the lost list when a slot is already held (given back at `finish()`).

Warp aggregation (one materialisation per group of lanes on the same chunk, gpu_design.md 6.2
step 1) is an optimisation, not a need: it is measured, not assumed (section 5).

### 3.4 Arena exhausted

`DeviceArena::take` returns `nullptr` and nothing else in the kernel can make room: the write
goes to the append-buffer (gpu_design.md 6.4), and `finish()` grows the arena and replays it.
The chunk-aligned path records a span and writes it again with its function; `assign()` has no
function to call again, so it keeps the value.

One case is new. A thread may append `(i, v1)` while the arena is exhausted, then write the same
element directly later in the kernel (another thread published a buffer for that chunk in
between). The replay must not overwrite the later value with the earlier one. Two facts make the
rule simple:

- a thread that has seen a buffer for a chunk never sees `nullptr` for it again in the kernel
  (coherence of one address, and buffers are only removed between kernels): **its appends to a
  chunk all come before its direct writes to that chunk**;
- the append counter gives one thread's appends in its program order.

So the replay keeps, for each element, the last append (the highest index), and skips it when the
element's bit is set: a direct write came after. The replay runs before `settle_fresh`, which
clears the bits. Two threads that write one element in a kernel are a data race of the
application, as on a `std::vector`; the library does not order them.

### 3.5 End of the phase

`finish()` without a function (the chunk-aligned path keeps `finish(f)` for its spans):

1. replay the append-buffer if it is not empty: dedup on the host (it is small: the arena ran
   out), grow the arena, a replay kernel through `assign()` that skips the elements whose bit is
   set; repeat if the replay appended again;
2. `settle_fresh`: the constant where no bit is set, bitmaps cleared, tags removed;
3. the lost list and the spare slots back to the arena.

## 4. Why it is correct

- **No element is read before it is written, except as the constant.** A fresh buffer's element
  is read (by `load`, by `settle_fresh`) only where its bit is set; the others read the constant,
  which does not change in a write kernel. The old contents of a reused buffer are never seen.
- **No store can be overwritten by a later fill.** The fill is `settle_fresh`, after the kernel,
  and only where no bit is set. Within the kernel nobody writes an element but its writers.
- **Every write lands or is replayed.** Case 1 changes nothing; cases 3 and 4 store; case 2 either
  publishes or finds the winner's buffer, or appends; the replay applies the appends that no later
  direct write superseded (3.4).
- **Nothing waits.** No spin, so no deadlock between lanes of one AMD wavefront on divergent paths
  (the case the claim's bound was there for).
- **The descriptor and the bitmap are only touched by atomics** (relaxed, device scope): they are
  coherent; the elements are plain stores of distinct threads to distinct elements, ordered for the
  next kernels by the kernel boundary.

## 5. What it should cost, and what to measure

| | prototype (GH200 / MI300A, S = 1024) | expected |
|---|---|---|
| write into a materialised chunk | 1.76x / 1.34x (relaxed load) | the same |
| first write, grid-stride, every chunk compressed | 14x / 433x to 541x | close to the chunk-aligned path (1.6x), plus the lost publications |
| write into a fresh chunk | -- | one relaxed atomic or per element (per 32 elements if aggregated) |
| random range rewrites, then `finish()` + `compress()` | 2.5x / 64x (claim, relaxed load) | between the two above |

Added to `device_bench`: grid-stride first write by `assign()`, scattered writes (a random
permutation of the indices), `+=` on a materialised array, range rewrites by `assign()`. Each
with and without warp aggregation, and with one atomic or per element against one per 32 lanes
when they write consecutive elements (a ballot).

## 6. Alternatives set aside

- **The prototype's protocol** (acquire load, pre-fill, release CAS, claim): 300x to 600x on the
  MI300A.
- **Claim without pre-fill** (a busy mark, the others wait for the buffer): the wait remains, and
  with it the divergent-wavefront case and its bound; the deferred fill makes the claim
  unnecessary.
- **Uncached or fine-grained memory for the buffers**, coherent across the XCDs without cache
  operations: it would slow down every read and write of a materialised chunk, the hot path.
- **A sentinel value instead of the bitmap** (an unwritten element holds a reserved value): there
  is no reserved value for an arbitrary `T`.

## 7. Risks and open points

- **The bitmap atomics under scattered writes**: one relaxed atomic per element written into a
  fresh chunk. GPUs execute atomics on distinct addresses at L2 rate; to measure, against the
  flat scattered write.
- **The first write's lost publications**: S/32 warps per chunk take a buffer each and all but
  one park it. Without a pre-fill the cost is two atomics per loser; the arena needs the spare
  slots on top (about 60 MB of buffers of 8 KiB on an MI300A at full occupancy).
- **A kernel without `finish()`**: a fresh tag would reach a read kernel. A check build can
  catch it in `View` (a tagged pointer is misaligned).
- **The append-buffer's size**: an overflow is an error (gpu_design.md 6.4). Default to
  discuss: a fraction of the arena, or of n.
- **Decisions asked for**: drop the claim and race variants from the library (the prototype
  keeps them as references); warp aggregation in the first version or after the measurements;
  the default size of the append-buffer.

## 8. Plan

1. `Writer::assign`, `Writer::load`, `materialise` without pre-fill, the spare slots, the
   append-buffer and `finish()` with its replay (3.4, 3.5), in `chunk/device/`.
2. `device_assign_validation` against a host mirror: scattered writes from many threads,
   grid-stride first writes, `+=` by the owner thread, one thread writing an element twice, a
   small arena (appends then direct writes to the same elements, the replay), two kernels before
   one `finish()`, `assign()` and chunk-aligned writes in separate kernels, two `assign()` on
   divergent paths of one wavefront.
3. The `device_bench` cases of section 5, on the T1000, the GH200 and the MI300A.
4. Then warp aggregation and the batched descriptor loads, if the measurements ask for them.
