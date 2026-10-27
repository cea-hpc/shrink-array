# pattern::Array: design note

> Status: design, October 2026, before any code. English only, like
> [threading_backends.md](threading_backends.md): it is a plan and it goes away, replaced by an
> `algo_` document, when the variant exists. When in doubt about `zero_any`, the code and
> [algo_zero_any.md](algo_zero_any.md) are the reference.

***

## 1. Goal

A container for **integer arrays with long runs of zeros and with series of motifs that repeat
shifted by a constant**: the typical case is a mesh connectivity, where cell `k` lists nodes
`n, n+1, n+nx+1, ...` and cell `k+1` lists the same motif plus one. `zero_any` already compresses a
chunk uniform at *any* value; this variant compresses a chunk that equals a **shared motif plus
one offset**.

Decisions taken with the author (October 2026):

| question | decision |
|---|---|
| what is a repeated motif | a **whole chunk**: `chunk[j] = P[j] + off`, with `P` shared |
| element types | **integers only** (`static_assert` otherwise) |
| when is a chunk examined | the **write heuristic** only (like `zero_any`), no explicit pass |
| where | a **new variant** built from `zero_any`, not a mode of `zero_any` |
| name | class **pattern::Array**: the array classes live in a namespace each, named after what they compress (zero, zero_one_value, zero_any, pattern); the CMake library follows the other variants |
| the uniform chunks | **option B**: `zero_any`'s second-value registry, **kept as it is**; the variant adds only a *motif* state (a pointer into a shared motif region + an offset), and the offset serves the motif chunks only |
| where the motif offset lives | **in the chunk's second-value slot** (§3): a motif chunk has no second value, so its slot is free; no byte more per chunk, one cache line more per read of a motif chunk |

Mathematically `zero_any` is the special case `P = 0` (a chunk uniform at V is the zero motif plus
the offset V), and an earlier draft of this note built on it: constants as offsets, no
second-value registry. That is not the design retained. `zero_any`'s zero and uniform chunks,
its registry, its `reclaim_second_values()` and its three modelled handshakes stay untouched
(the variant is built by parameterising `zero_any`'s code, not by cloning it); the motif is a
**fourth state**, next to them, with its own registry (§7) and its own handshake (§6). What the
variant adds is then bounded: it changes nothing a `zero_any` chunk does, and only a motif chunk
pays for the offset.

## 2. What a chunk can be

| state | `values` points to | offset | from |
|---|---|---|---|
| zero | the shared zero buffer | ignored | `zero_any`, as is |
| uniform at V | the constant buffer of V in the second-value registry | ignored | `zero_any`, as is |
| motif | a registered motif `P` in the **motif region** | `off` | **new** |
| materialised | a buffer owned by the chunk, holding the **absolute** values | ignored | `zero_any`, as is |

The zero, uniform and materialised states are `zero_any`'s, read and written by its code; the
motif state is the only addition, and the offset is read only when `values` points into the
motif region. A motif is **normalised**: `P[0] = 0`, `P[j] = A[j] - A[0]` (wrapping arithmetic,
on the unsigned type of the same width, so a signed overflow is never undefined behaviour). Two
chunks are equal up to an offset exactly when they have the same normalised motif. A chunk whose
normalised motif is all zero is a uniform chunk: it takes `zero_any`'s path (its constant goes to
the second-value registry), never the motif one, so the zero motif is never registered.

## 3. Layout

`zero_any` keeps 16 bytes per chunk (the pointer, a pending-operations count and the steal mark)
and a row of `sizeof(T)` bytes per chunk for its second value, plus two bytes of rows.
Here `zero_any`'s layout is kept as it is (the second-value row and the type rows serve the
uniform chunks exactly as before), and the offset takes **the chunk's slot in the second-value
row**: a motif chunk has no second value, so that slot -- `sizeof(T)` bytes, allocated for every
chunk already -- is free. Per chunk: 26 bytes (8-byte elements) or 22 (4-byte elements), exactly
`zero_any`'s.

This is a choice between memory and locality. The other place is the chunk object, next to the
pointer the reader loads anyway: 24 bytes instead of 16, hence +8 bytes per chunk for every
chunk of the array (34 or 30 bytes), but the offset of a motif chunk in the line already loaded.
The slot costs nothing in memory, and one more cache line per read of a **motif** chunk only:
the row is another array, indexed by the chunk's array (`num_tab`) and position. The reads of
zero, uniform and materialised chunks -- `zero_any`'s -- do not touch it. The +8 bytes are
therefore not the price of option B but of the offset in the chunk; the slot is retained, and
the other option stays an open point (§11), to measure with `access_cost`.

Two consequences. A reader must know the chunk's index to find the slot: the chunk object
alone does not (§4). And the slot must hold a `T`: under `CHUNK_SECOND_VALUE_INDEXED` a slot
of a type wider than 4 bytes holds a 4-byte identifier, which an offset does not fit, so
`pattern::Array` refuses that configuration (a `static_assert`) for those types.

## 4. The read

```
p = values (acquire)
v = p[j]
if p is in the motif region:  v += offset           // wrapping
```

- The motif region is **one reserved virtual range** (mmap), distinct from the second-value
  registry's, so "in the region" is one subtract-compare against two constants, not a table
  lookup. `zero_any` already does this test for its registry (`region_contains`); the cost to
  measure is that compare plus the add on the **materialised, zero and uniform** reads, which are
  `zero_any`'s reads and must not slow down by more than the noise (the access cost bench,
  `access_cost`, decides).
- A read of a motif chunk costs one more add and one more cache line than a `zero_any` read:
  the slot (§3).
- Every place that reads a chunk goes through the one element accessor of the chunk base
  (`element_at`, behind `read_value`, `read_value_scan`, the scans, `copy_from`, the comparison,
  the range operations, the advisor); the offset is added there and nowhere else. Direct
  `values[i]` reads (about 25 sites, all listed by grep) are each checked once.
- The chunk does not know where its slot is: the accessor takes it from its caller, which
  knows the chunk's index -- the array's accessors, its iterator and bulk reads (`num_tab`
  and the chunk number), the proxy (its armed slot). A read of a motif chunk without its slot
  is an invariant failure, except the marking filter's relaxed scan, which only takes a hint.

## 5. The write

A write into a motif chunk **materialises** it first, like `zero_any` materialises a constant chunk
pre-filled with its constant: the buffer receives `P[j] + off` for every `j`, **absolute**
values, then the write goes in. The offset is left in the slot (it is ignored while the pointer
is not in the region), so the pointer switch is the single publication point and a reader never
combines a new pointer with a wrong offset. Until the chunk next compresses, `zero_any`'s write
path sees that offset where it expects the chunk's second value: its marking heuristics take a
wrong hint at worst, and its one shortcut on it (a write of the second value onto an element
that already holds it is skipped) stays right. The fast path (a write into an already
materialised chunk) is `zero_any`'s, unchanged.

## 6. The offset and the readers (the hard point)

A reader loads the pointer and then the offset; they are two words. The sequences:

- **buffer -> motif** (compression): the compressor stores the offset in the slot first
  (release), then switches the pointer by CAS (acq_rel). A reader that sees the new pointer sees
  the new offset; one that sees the old (buffer) pointer ignores the slot.
- **motif -> buffer** (materialisation): the buffer is filled, then the pointer is published.
  `offset` is not touched.
- The remaining hazard: a reader that loaded an **old motif pointer** `Q` and reads the slot
  only after the chunk went `Q -> buffer -> (slot rewritten) -> Q'`. It would add the new value
  of the slot to `Q`. With the offset in the slot, two writes rewrite it: a compression to a new
  motif (a new offset), and `zero_any`'s compression of the chunk to a constant (its second
  value, or 0). Both happen in the compression pass, **before** its CAS; its grace period comes
  after the CAS and protects the buffer it frees, not the slot. This note first argued the
  hazard away on that grace period; the reduced model (`formal/motif_offset.c`,
  `docs/formal_models.md` section 7) found it in two executions (F5, section 8.6 there).
- **The rule, as implemented.** The reader loads the slot with **acquire**, then the pointer
  **again** (`Chunk::motif_offset()`): unchanged, the offset belongs to the state the two loads
  bracket; moved, it starts over. Every rewrite of the slot of a chunk of a member with the
  motif state is a **release** atomic store (never the plain store `zero_any` keeps: a reader
  may load the slot meanwhile), made after the pass saw the materialised buffer: a reader that
  sees the new slot cannot see the old pointer again. The ABA -- a pass compressing back to the
  same motif -- is harmless: pointer and slot then describe the new state together. Checked
  with up to 3 readers and 3 reads, for every target, under RC11 and IMM; each part of the rule
  is load-bearing (the model's mutations). Cost: one acquire load and a second load of the
  pointer, from the same cache line, on the read of a motif chunk only.

## 7. The motif registry

- Entries are chunk-sized arrays of `T` in the motif region (`MotifRegion<T>`), one reserved
  virtual range per `T` whose pages are touched only as entries are written; an entry is filled
  **before** it is published (release). Finding a known motif is a lock-free probe of an
  open-addressed index; only a new motif takes the insertion lock. A grown index keeps its
  superseded tables until the next reclaim, so a reader never holds a freed table.
- A key is a 64-bit hash of the normalised motif, **confirmed by comparing the arrays** (a hash
  alone never decides equality).
- **Admission.** Registering every distinct motif would make random data cost *more* (one
  registered copy per chunk). A motif is registered the **second time** it is seen, **by another
  chunk**: the first sighting only records, at its hash in a probation table of 64 Ki bytes, an
  8-bit tag of the chunk that made it; the chunk of the first sighting stays materialised. The same
  chunk seen again with the same contents (a random chunk rewritten unchanged) stays on
  probation: a motif one chunk uses saves nothing. Collisions in the table or between tags cost a
  missed or a useless entry, never a wrong value. On a mesh, that is one chunk per distinct
  motif (one per thread at most when several passes see it at once).
- **Reclaim.** Modelled on `zero_any`'s `reclaim_second_values` (which stays, for the constants):
  at a sequential point, mark the motifs still referenced, free the others. The two registries
  are reclaimed by the same call, `pattern::Array<T>::reclaim_second_values()`, each its own
  entries: the motifs only outside a parallel region. A freed entry's identifier is handed out
  again; the index is rebuilt from the survivors and the probation table cleared.
- **Geometry.** A motif has the chunk's element count, so there is one registry per chunk size.
  First stage: the static sizes (`CHUNK_STATIC_SIZES`); the dynamic geometry and the rechunk
  come in a later stage (§10). Today one region holds one entry size per `T`: an entry of another
  size is an invariant failure.

## 8. When a chunk is examined

Without an explicit pass, the only signal is the writes. A connectivity is written once, often
in order. A chunk joins the compression list when a write **completes it**: the write of its
**last element**, or of its **first element**, whichever the writer order produces (the scan
tells whether the chunk is really a motif; a chunk examined too early is dropped and is examined
again at the next write of those positions). The scan computes the normalised motif, hashes it,
looks it up, and compresses on a match. A scan whose normalised motif is all zero has found a
zero or a uniform chunk: it hands the chunk to `zero_any`'s compression (the constant to the
second-value registry), the motif registry sees nothing of it.

As implemented (stage 2): the write path queues a chunk when its first or last element is
written, in addition to `zero_any`'s marking; the pass runs `zero_any`'s two constant scans first
and, for a chunk that is neither, a third classification (`Chunk::classify_motif`). A match keeps
the pass's claim on the chunk, and the chunk is compressed after the constants of the same pass
(`compress_motif_candidates`): the offset stored in the slot (release), the type slot set to
`CstSlot::motif`, the values pointer switched by CAS from the buffer to the motif, the buffer
released to the pool after the grace period, as `zero_any`'s compression to a constant. The rule
also catches the uniform chunks `zero_any` misses (a chunk whose last write lands on an element
whose old value was neither constant is not queued by `zero_any`): `pattern_validation` checks that
`pattern::Array` compresses every uniform chunk again where `zero_any` compresses at most as many.

Known limit: a chunk filled in a random order, whose first and last elements are written
before the rest, is examined too early and may stay materialised; the range operations
(`assign`, `fill`) can examine the chunks they have just filled themselves. A later
explicit entry point stays possible and costs nothing in the design.

## 9. What is reused from `zero_any`

| part | reused | to adapt |
|---|---|---|
| array container, copy/move, swap, registry of arrays | as is | the class name and the traits |
| write engine, proxies, deferred writes, barriers | as is | the read of a motif chunk; the materialisation pre-fill |
| compression lists, reservation, the pass, the Dekker handshakes | as is | classification (`fill_list_to_compress`): constant or motif |
| `clear_int_st` / CAS commit of a compression | the CAS | stores the offset in the slot first, installs the motif pointer |
| second-value registry, `reclaim_second_values` | as is, for the uniform chunks | nothing; the **motif** registry (§7) is a second registry built on the same machinery; a motif chunk's slot holds its offset (§3) |
| marking filter (`may_become_uniform`) | the idea | replaced by the completion rule of §8 |
| range operations, `copy_from`, comparisons | as is | read through the element accessor |
| rechunk and the advisor | later | a motif is chunk-size dependent (§7) |

## 10. Stages

1. **The chunk and the read**: the motif region, the offset in the second-value slot, the
   element accessor taking the slot from its caller,
   the static sizes, single-thread; `zero_any`'s zero and uniform chunks untouched (the
   parameterisation of `zero_any` -- a common trait and per-model hooks -- comes first, with
   `zero_any`'s machine code unchanged, checked by the existing harnesses); validation harness
   comparing with `std::vector` on zero-only, uniform, shifted-motif and random chunks.
2. **Compression and the registry**: scan, hash, admission, the CAS commit, materialisation on
   write; the OpenMP and std backends; the check-mode builds. *Done for the OpenMP backend at the
   static size* (`pattern_validation`, cases M to P, at 1 and 8 threads; `zero_any`'s library
   objects unchanged); the std and serial backends pass too.
3. **The model**: the GenMC handshake of §6 and its mutations; the findings are applied to the
   sources before the variant is announced. *Done*: `formal/motif_offset.c`, one finding (F5,
   the reader's re-check and the release stores of the slot, §6), applied.
4. **Dynamic geometry, rechunk, advisor**: motifs per size; the advisor counts a sub-block as
   compressible when it matches a motif.
5. **Bench and docs**: a mesh-connectivity case in `shrink_bench`, `access_cost` for the
   materialised reads, the `algo_` document in both languages (this note then goes away).

## 11. Open points

- **Overflow semantics** for signed `T`: arithmetic is wrapping on the unsigned twin, so reading
  gives back the exact bits; to confirm for `T = int` with negative offsets.
- **Cost of the read compare** on materialised chunks (stage 1 measures it first; if it is
  visible, the motif chunks can be recognised by a bit in the pending-operations word instead
  of the pointer range).
- **Where the offset lives** (§3). The slot is retained: no byte more per chunk, a cache line
  more per read of a motif chunk. The other option, the offset in the chunk object (+8 bytes per
  chunk, the offset in the line already loaded), is to be measured with `access_cost` on
  motif-heavy reads -- a connectivity read in its order -- before stage 5; it would also lift
  the indexed-slot restriction below.
- **`CHUNK_SECOND_VALUE_INDEXED`** is refused for element types wider than 4 bytes (their slot
  holds an identifier, §3): a 64-bit connectivity then pays `zero_any`'s full slot. To revisit
  if the indexed slots matter for those arrays.
- **Motifs shorter than a chunk** (a stride that does not divide the chunk size, or an offset
  that grows inside the chunk) are not covered: `chunk[j] = P[j mod s] + (j div s) * step` would
  be a second kind of motif. Measure on real connectivities first.
- **Naming.** Decided (October 2026): one namespace per variant, named after what it compresses,
  with the class called Array in each: zero::Array, zero_one_value::Array, zero_any::Array and,
  for this variant, pattern::Array. The CMake targets and the package keep the `chunk` names
  until the question of the package name is settled.
