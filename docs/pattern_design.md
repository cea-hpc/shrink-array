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

`zero_any` is the special case `P = 0`: a chunk uniform at V is the zero motif plus the offset V.
The new variant therefore needs **no second-value registry**: constants are offsets.

## 2. What a chunk can be

| state | `values` points to | offset |
|---|---|---|
| zero | the shared zero buffer, slot 0 of the motif region | 0 |
| motif | a registered motif `P` (slot > 0 of the motif region) | `off` |
| uniform at V | the zero motif | `V` |
| materialised | a buffer owned by the chunk, holding the **absolute** values | ignored |

A uniform chunk is a motif chunk whose motif is the zero one; there is no separate "second
value" state. A motif is **normalised**: `P[0] = 0`, `P[j] = A[j] - A[0]` (wrapping arithmetic, on
the unsigned type of the same width, so a signed overflow is never undefined behaviour). Two
chunks are equal up to an offset exactly when they have the same normalised motif.

## 3. Layout

`zero_any` keeps 16 bytes per chunk (the pointer, a pending-operations count and the steal mark)
and a row of `sizeof(T)` bytes per chunk for its second value, plus two bytes of rows.
Here the offset lives **in the chunk object**, next to the pointer the reader loads anyway:
24 bytes for 4- and 8-byte integers, a one-byte type row no longer needed (the type is the region
test of §4), the candidate flag row kept. Per chunk: 25 bytes, against 26 (8-byte
elements) or 22 (4-byte elements) for `zero_any`. The memory per chunk is the same order; the
read is cheaper because the offset is in the line already loaded.

## 4. The read

```
p = values (acquire)
v = p[j]
if p is in the motif region:  v += offset           // wrapping
```

- The motif region is **one reserved virtual range** (mmap), so "in the region" is one
  subtract-compare against two constants, not a table lookup. `zero_any` already does this
  test for its registry (`region_contains`); the cost to measure is that compare plus the add on
  the **materialised** reads, which must not slow down by more than the noise (the access
  cost bench, `access_cost`, decides).
- A read of a motif chunk costs one more add and no more cache line than a `zero_any` read.
- Every place that reads a chunk goes through the one element accessor of the chunk base
  (`element_at`, behind `read_value`, `read_value_scan`, the scans, `copy_from`, the comparison,
  the range operations, the advisor); the offset is added there and nowhere else. Direct
  `values[i]` reads (about 25 sites, all listed by grep) are each checked once.

## 5. The write

A write into a motif chunk **materialises** it first, like `zero_any` materialises a constant chunk
pre-filled with its constant: the buffer receives `P[j] + off` for every `j`, **absolute**
values, then the write goes in. `offset` is left as it is (it is ignored while the pointer is not
in the region), so the pointer switch is the single publication point and a reader never combines
a new pointer with a wrong offset. The fast path (a write into an already materialised chunk)
is `zero_any`'s, unchanged.

## 6. The offset and the readers (the hard point)

A reader loads the pointer and then the offset; they are two words. The sequences:

- **buffer -> motif** (compression): the compressor stores `offset` first (release), then
  switches the pointer by CAS (acq_rel). A reader that sees the new pointer sees the new offset;
  one that sees the old (buffer) pointer ignores the offset.
- **motif -> buffer** (materialisation): the buffer is filled, then the pointer is published.
  `offset` is not touched.
- The remaining hazard: a reader that loaded an **old motif pointer** `Q` and reads `offset`
  only after the chunk went `Q -> buffer -> (offset rewritten) -> Q'`. It would add the new offset
  to `Q`. The compression pass already waits for a **grace period** (readers announced before
  the pass have left their read section) before it recycles a buffer: the offset is rewritten
  only in that pass, after that wait, and only on chunks that were materialised **before the
  pass began**. A reader of the old motif pointer began before the materialisation, hence before
  the wait ends. This needs the same argument as the buffer reclaim handshake, and **a fourth
  reduced GenMC model** (`formal/`), with its mutation oracles, before the variant is called
  correct.

## 7. The motif registry

- Entries are chunk-sized arrays of `T` in the motif region, in chained segments of 64 slots like
  `zero_any`'s registry, filled **before** they are published (release). Finding a known motif is a
  lock-free hash probe; only a new motif takes the insertion lock.
- A key is a 64-bit hash of the normalised motif, **confirmed by comparing the arrays** (a hash
  alone never decides equality).
- **Admission.** Registering every distinct motif would make random data cost *more* (one
  registered copy per chunk). A motif is registered the **second time** it is seen: the first
  sighting only counts the hash in a small probation table (a counting filter, bounded memory);
  the chunk of the first sighting stays materialised. On a mesh, that is one chunk per distinct
  motif.
- **Reclaim.** As `zero_any`'s `reclaim_second_values`: at a sequential point, mark the motifs still
  referenced, free the others.
- **Geometry.** A motif has the chunk's element count, so there is one registry per chunk size.
  First stage: the static sizes (`CHUNK_STATIC_SIZES`); the dynamic geometry and the rechunk
  come in a later stage (§10).

## 8. When a chunk is examined

Without an explicit pass, the only signal is the writes. A connectivity is written once, often
in order. A chunk joins the compression list when a write **completes it**: the write of its
**last element**, or of its **first element**, whichever the writer order produces (the scan
tells whether the chunk is really a motif; a chunk examined too early is dropped and is examined
again at the next write of those positions). The scan computes the normalised motif, hashes it,
looks it up, and compresses on a match; zero and uniform chunks are found by the same scan (the
motif is all zero).

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
| `clear_int_st` / CAS commit of a compression | the CAS | stores the offset first, installs the motif pointer |
| second-value registry, `reclaim_second_values` | the machinery | becomes the **motif** registry (§7), the constants leave it |
| marking filter (`may_become_uniform`) | the idea | replaced by the completion rule of §8 |
| range operations, `copy_from`, comparisons | as is | read through the element accessor |
| rechunk and the advisor | later | a motif is chunk-size dependent (§7) |

## 10. Stages

1. **The chunk and the read**: the motif region, the offset in the chunk, the element accessor,
   the static sizes, single-thread; validation harness comparing with `std::vector` on
   zero-only, uniform, shifted-motif and random chunks.
2. **Compression and the registry**: scan, hash, admission, the CAS commit, materialisation on
   write; the OpenMP and std backends; the check-mode builds.
3. **The model**: the GenMC handshake of §6 and its mutations; the findings are applied to the
   sources before the variant is announced.
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
- **Motifs shorter than a chunk** (a stride that does not divide the chunk size, or an offset
  that grows inside the chunk) are not covered: `chunk[j] = P[j mod s] + (j div s) * step` would
  be a second kind of motif. Measure on real connectivities first.
- **Naming.** Decided (October 2026): one namespace per variant, named after what it compresses,
  with the class called Array in each: zero::Array, zero_one_value::Array, zero_any::Array and,
  for this variant, pattern::Array. The CMake targets and the package keep the `chunk` names
  until the question of the package name is settled.
