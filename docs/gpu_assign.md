# `assign()` on the device without device-scope acquire or release

> Version française : [gpu_assign_fr.md](gpu_assign_fr.md).

> Status: design note, November 2026, implemented on the `gpu` branch
> (`Writer::assign`, `Writer::load`, `Array::finish()`, `device_assign_validation`). It
> replaces, for stage 1, the materialisation of [gpu_design.md](gpu_design.md) 6.1 and 6.2
> (pre-fill, release publication, claim with acquire loads) by the deferred fill that the
> chunk-aligned path already uses (gpu_design.md, section 13, "relaxed publication"), with a
> claim that keeps one buffer per chunk (3.3: the first version without it did not).

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
    if (buf == nullptr || buf == busy_mark<T>())
    {
        if (same_value(v, d.constant)) return;   // 1. the constant onto its chunk: nothing
        buf = materialise(c, d, buf);            // 2. claim, or wait for the claim (3.3)
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

`materialise` claims the chunk by a **relaxed** CAS from `nullptr` to the busy mark; its winner
alone takes a buffer (`DeviceArena::take`) and publishes it empty, tagged fresh, by a relaxed
exchange -- or puts `nullptr` back when the arena is exhausted. The other threads poll the
descriptor (relaxed loads, a short pause) until the publication (3.3). No pre-fill, no fence. A
buffer taken from the arena is never read before its elements are written: its old contents do
not matter.

Case 4 is the hot path: a relaxed load and a store, measured at 1.34x (MI300A) and 1.76x (GH200)
a flat write in the prototype, 1.1x to 1.2x with the descriptor loads issued ahead.

### 3.2 Reads in a write kernel

`a(i) += v` reads first. `Writer::load(i)` follows the same cases: the constant of a compressed
chunk; in a fresh buffer the element if its bit is set, else the constant; in an older buffer
the element. A thread always reads what it wrote itself (same thread, same address), and what a
thread of its block wrote before a `__syncthreads()` (block scope). That includes a write that
went to the append-buffer (3.4): once the arena has run out in the kernel, an element that would
read the constant is looked up in the append-buffer first, from its end -- slow, but only until
`finish()`; without it `a(i) += 1` twice by one thread lost an update. Communication between blocks
inside a kernel needs device-scope fences in the application, as on a flat array; they order the
descriptor, the bitmap and the elements alike.

`View` (the read path) is for read kernels only and never sees a fresh tag: `finish()` settles.

### 3.3 The claim before the buffer

The first version took the buffer **before** its CAS, as the prototype's race did, and counted
on a lost publication being cheap without a pre-fill. It was not the cost but the count: with
scattered writes some 32 threads of different warps meet each compressed chunk at once, each
took a buffer, and one wave took about 31 buffers too many per chunk. `device_assign_validation`
measured it: an arena of one buffer per chunk (4 096) grew to 26 817 buffers in 19 arenas
(242 000 lost publications), most writes went through the append-buffer, and the limit of 32
arenas was near.

So the claim comes first: a CAS from `nullptr` to the busy mark, then the winner alone takes a
buffer. The others wait, but the wait is short -- between its CAS and its publication the winner
makes one or two atomics on its shard, no pre-fill -- and bounded (`claim_spin_max` polls, then
the append-buffer): what the bound is for is a claim held by a lane of the same AMD wavefront on a
divergent path, which cannot progress while this lane polls. Measured on the T1000: the roomy
arenas do not grow, no wait reached the bound in millions.

A claimed chunk still reads its constant (case 1 holds under the mark); `load()` reads it as
compressed. The chunk-aligned path waits for an `assign()` claim of the same kernel the same way,
and records its span past the bound.

Warp aggregation (one claim per group of lanes on the same chunk, gpu_design.md 6.2 step 1) is an
optimisation: the grid-stride first write is its case (every lane of a warp polls the same
claim), measured in section 5.

**HIP only.** On CUDA the claim is per lane (`Writer::materialise_grouped`). Under independent
thread scheduling (Volta and later) the lanes that reach `assign()` need not be converged: those
on a materialised chunk write and go on to their next grid-stride iteration while their
neighbours are still in the grouped claim. Two subsets of one warp then computed
`__activemask()` masks that named each other and waited for each other, one in a `__syncwarp`,
the other in a shuffle, for ever. On the RTX A1000 (sm_86, nvcc 13.3) it happened on every run
of a grid-stride scatter of n/4 shuffled indices into a 2^20-element array whose chunks were
partly materialised (`device_assign_validation`, step 1b), whatever the arena. The validations
before it scattered into a compressed array only. In lockstep (AMD) the active mask is exactly
the lanes of the call, and the grouping stays. `write_marked` takes the full warp mask for the
same reason: the whole block calls it, and `f`'s divergence could narrow `active_lanes()`.

Cost on CUDA (RTX A1000, 0.5 GiB, `device_bench`): the grid-stride first write by `assign()`
goes from 5.0x-5.9x to 7.0x-9.9x the flat write (claim waits 9 M -> 334 M; not their counter:
without it, the same times); the scattered writes do not change (4.1x-4.5x). The chunk-aligned
path is not concerned.

GH200, 8 GiB, the same session (`fc2801768` against `adda2ea5e`, `scripts/device_check.sh cuda`,
the six device validations and ten `device_assign_validation` runs green): the grid-stride first
write goes from 21.0x / 20.3x to 27.4x / 26.8x at S = 256 (`zero_any` / `zero`) and from 14.1x /
13.8x to 19.9x / 19.3x at S = 1024, +30% / +40%; the claim waits from 0.27 G to 10.3 G (S = 256)
and 0.17 G to 7.7 G (S = 1024), one per element written. The scattered writes do not change
(2.1x-2.3x first write, 1.0x-1.3x into materialised chunks). MI300A (`adda2ea5e`, ROCm 10.2, the
HIP path unchanged but for `write_marked`): validations green, no wait given up; scattered writes
1.45x-1.70x and 1.03x-1.07x as at 491a05a87; the grid-stride first write 74.6x / 48.5x
(S = 256 / 1024) against 57x / 42x there. Against `fc2801768` on the same node and build (ROCm
10.2, 8 GiB): 75.1x / 74.8x and 44.0x / 42.8x (`zero_any` / `zero`, S = 256 / 1024) before,
74.8x / 74.5x and 46.6x / 45.6x at `ece8e73c1`, the same within noise. The S = 256 gap to 57x
predates the per-lane claim: a commit between 491a05a87 and fc2801768, to be bisected.

**The grouping's gain back, without a collective.** What the grouping saved was the CAS of 32
lanes on one descriptor (a grid-stride first write: about one CAS per element per lane against one
per warp), not the waits between warps, which stay. The CUDA path now lets the first active lane
claim first; the other lanes, in a block after it, load the descriptor again and claim only a chunk
still compressed. `active_lanes()` is a hint there: no sync depends on it, and a wrong one costs a
CAS, not a wait. RTX A1000, 0.5 GiB, the same session, grid-stride first write at S = 256
(`zero_any` / `zero`) and 1024: grouped 22.2 / 20.1 and 23.2 / 22.8 ms; per lane 31.0 / 29.9 and
30.2 / 27.9 ms; with the hint 22.8 / 21.2 and 23.4 / 23.4 ms. The scattered writes do not change
(147-160 ms), the stress that hung the grouping passes 30 runs of 30, the device validations and
ten `device_assign_validation` runs pass. The claim waits stay counted per lane (about one per
element, as per lane). GH200, 8 GiB (`7b835276b`, every device validation and ten
`device_assign_validation` runs green): 22.0x / 22.2x at S = 256 and 14.3x / 14.5x at S = 1024,
against 21.0x / 20.3x and 14.1x / 13.8x grouped and 27.4x / 26.8x and 19.9x / 19.3x per lane;
the scattered writes unchanged, no wait given up.

**CUDA: filled buffers, no `written` bit.** On CUDA the bit of each element written into a fresh
buffer was a device-scope `fetch_or` of its own (CUDA cannot aggregate it per warp, above): half
of a first write (RTX A1000, device_bench, 0.5 GiB: a grid-stride first write 22.8 ms with the
bits, 11.0 ms with none, the result then wrong). The claim's winner now fills the buffer with the
constant (one lane, 16-byte stores) and publishes it tagged *filled* (`filled_tag`, bit 2) by a
release exchange; a writer that sees the tag acquires the descriptor (the waiters and the losers of
the CAS too) and stores its element, no bit; the settle only removes the tag. A chunk that appended
in the phase appends again (no bit tells the replay that a direct write came after an append of the
same element: the last append wins); `load()` reads a filled buffer through the append-buffer the
same way. HIP keeps the fresh buffers and the wavefront's `fetch_or` (acquire and release cost the
MI300A 300x to 600x a flat write in the prototype's protocol). A literal T{} for the fill where the
constant is T{}: stores from the zero register; the same zeros from a register that holds them, or
1.0 as a literal, took the first write 60 % longer (S = 1024: 13.8 against 22.6 and 21.5 ms).

| RTX A1000, 0.5 GiB, zero_any / zero | before | filled |
|---|---|---|
| grid-stride first write, S = 256 | 22.8 / 21.2 ms | 11.8 / 11.4 ms |
| grid-stride first write, S = 1024 | 23.4 / 23.4 ms | 14.4 / 13.7 ms |
| scattered first write, S = 256 | 160 / 157 ms | 105 / 98 ms |
| scattered first write, S = 1024 | 152 / 148 ms | 89 / 73 ms |
| into materialised chunks (contiguous, scattered) | the same | the same |

The device validations, ten `device_assign_validation` runs and compute-sanitizer's memcheck pass.
To be measured on the GH200.

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

The same reasoning covers a write of the chunk's constant after an append of the element (a
direct write that does nothing): it is appended too, so that it is the last append. And a load of
an element that reads the constant may have to find an append first. Both concern only the
chunks that appended in the phase: `append()` sets the chunk's bit in `Writer::appended` before
writing its entry, and the constant's append and the look-up of the append-buffer happen only
under that bit (the replay clears the bits with the count). Conditioning them on the phase's
append count instead, as first done, made one wait given up anywhere turn every write of a
constant into an append and every such load into a scan of the whole buffer: on windows of 10^7
elements (GH200) the append-buffer overflowed, and the rewrite took 591x flat.

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
- **Every wait ends.** A claim's holder waits for nothing between its CAS and its publication;
  the polls are bounded, and a write that gives up goes to the append-buffer: two lanes of one
  AMD wavefront on divergent paths cost a replay, not a deadlock.
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

T1000, 0.5 GiB of doubles, S = 256 / 1024 (`device_bench`, `finish()` timed): `assign()` into
materialised chunks 0.91x / 0.72x, `a(i) = load(i) + 1` 1.08x / 1.18x, scattered into
materialised chunks 2.24x / 2.32x and as a first write 4.59x / 4.36x (against the flat scattered
write), grid-stride first write 7.31x / 6.22x: there every lane of a warp waits for one claim,
the case of warp aggregation.

GH200, 8 GiB, S = 256 / 1024, before aggregation: into materialised chunks 1.79x / 1.64x, `+=`
1.18x / 1.13x, scattered into materialised chunks 1.33x / 1.12x and as a first write 2.30x /
2.14x, but the grid-stride first write 23.7x / 18.9x (the prototype's claim: 14x at S = 1024),
with about one wait per element written. With warp aggregation (the active lanes on one chunk
elect a leader, which claims or waits; `group_by_key`, `warp.h`) the waits drop 35-fold on the
T1000 and its grid-stride first write goes from 7.31x / 6.22x to 4.80x / 5.04x; the scattered
writes do not change.

MI300A, 8 GiB, S = 256 / 1024, with aggregation and the three-phase claim (491a05a87, two runs
within 3%): into materialised chunks 1.78x / 1.60x, `+=` 0.97x / 0.94x, scattered into
materialised chunks 1.08x / 1.05x and as a first write 1.69x / 1.47x (against the flat scattered
write), the grid-stride first write 57x / 42x (the prototype's claim: 433x to 541x). No wait
given up (0.3% before the three phases) and no arena growth, in the bench as in
`device_assign_validation`. The grid-stride first write is still the hard case: about 3 waits
per chunk at S = 256 and 9 at S = 1024, the other wavefronts of the chunk waiting for the
winner's publication, which the cross-XCD atomics make long; warp aggregation does not group
distinct wavefronts. Regular loops take the chunk-aligned path (1.75x at S = 1024).

Added to `device_bench`: grid-stride first write by `assign()`, scattered writes (a random
permutation of the indices), `+=` on a materialised array, range rewrites by `assign()`. Each
with and without warp aggregation, and with one atomic or per element against one per 32 lanes
when they write consecutive elements (a ballot).

## 6. Alternatives set aside

- **The prototype's protocol** (acquire load, pre-fill, release CAS, claim): 300x to 600x on the
  MI300A.
- **No claim** (the buffer taken before the CAS, losers parked in spare slots): the first
  version; it took about 31 buffers too many per chunk under scattered writes (3.3).
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
- **Decisions taken** (2026-11-24): the prototype's claim (pre-fill, acquire loads) and race
  stay out of the library, as references in `gpu/proto`; the claim of 3.3 replaces them. Warp
  aggregation after the measurements. The append-buffer holds 1% of the arena's elements by
  default (4 096 at least), set at construction.

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
