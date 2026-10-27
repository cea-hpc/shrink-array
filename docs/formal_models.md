# Formal models of the lock-free handshakes

> French version: [formal_models_fr.md](formal_models_fr.md).

[synchronisation_dekker.md](synchronisation_dekker.md) explains the three handshakes of the
library and argues their correctness on paper. `formal/` holds the same protocols as **reduced
models**: small C11 programs, one per handshake (and a fourth, pattern::Array's offset slot,
section 7), that keep nothing but the atomics of the source
with their exact memory orders, and that a model checker -- [GenMC](https://github.com/MPI-SWS/genmc)
-- explores exhaustively under the C11 memory model (RC11, and IMM as a second opinion). This
document says what the models are, how to run them, what they proved, and what they found.

## 1. What a model proves, and what it does not

A reduced model is a program of a few threads (up to 5 writers and one pass) and one chunk. The checker enumerates **every**
execution the memory model allows -- every interleaving, every read-from choice a weak processor
could make -- and reports a violated assertion, an access to freed memory, or (with
`--check-liveness`) a spin loop that never ends. Where the stress runs of
`multi_launch_stress.py` find a defect once in thousands of runs, the checker finds it in the
first execution that exhibits it, in a fraction of a second, and says "none" otherwise.

What "none" means has limits, listed in section 9: the model is a reduction (fidelity is the
risk), it is bounded (up to 5 writers), and C11 has no `membarrier()`. The way to trust a model
is to **calibrate** it: feed it the bugs the library really had, as mutations of the model, and
check that the checker refuses every one of them. Section 4 is that calibration.

## 2. Running the models

GenMC needs an LLVM 19 or later development install and a C++20 compiler. On Ubuntu 24.04:

```shell
sudo apt install llvm-19-dev clang-19 libclang-19-dev libffi-dev g++-14
git clone https://github.com/MPI-SWS/genmc.git && cd genmc
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
      -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm -DClang_DIR=/usr/lib/llvm-19/lib/cmake/clang -B build
cmake --build build -j && export PATH=$PWD/build/bin:/usr/lib/llvm-19/bin:$PATH
```

Without root, Spack provides the same (`spack install llvm@19 +clang`, then the `cmake` lines above
against its prefix). Then, from the repository:

```shell
formal/run_genmc.sh            # every configuration and mutation, RC11; exit 1 on a verdict not expected
formal/run_genmc.sh --imm      # the same under IMM
genmc --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT formal/writer_compressor.c   # one case by hand
```

With CMake, `ctest -L formal` runs `formal_genmc` (when `genmc` is on the PATH or
`-DGENMC_EXECUTABLE=` names it) and `formal_native_*`: each model compiled as an ordinary program
and run 300 times, which proves the model is a real program and is a weak oracle for the protocol.
Nothing in `formal/` is built into the library or installed.

## 3. The model of handshake 1: writer against compressor

`formal/writer_compressor.c` is sections 5.1 to 5.3 and 5.6 of the synchronisation document:
NWRITERS writers (2 by default) each write one element of one chunk once, while one compression
pass marks the chunk, opens its window, scans it and steals its buffer if it is all zero. Each
name of the model stands for one place of the source, with the same memory order:

| model | source (`chunk/public/chunk/internal/`) |
|---|---|
| `flag[r]`, `announce_write()`, `end_write()` | `write_compress_sync_policy.h`: `WritingFlag::v`, `announce_write()`, `end_write()` |
| `any_writing_on_marked()`, `any_writing()` | `write_compress_sync_policy.h`: `any_thread_writing_on_marked()`, `any_thread_writing()` |
| `hard_inflight`, `stealing_inflight`, `must_defer_write()`, `is_stealing()` | `compression_manager.h`: `hard_inflight_`, `stealing_inflight_`, `must_defer_write()`, `is_hard_compression_inflight()` |
| `compressor_start()`, `compressor_finish()`, `heavy_barrier()` | `compression_manager_inline.h`: `start()`, `finish()`; `dekker::heavy_barrier()` |
| `mark`, `steal_mark()`, `steal_marked()`, `steal_unmark()` | `chunk_base.h`: `steal_mark_` and its three verbs |
| `values`, `values_load()`, `cas_install()`, `steal()` | `chunk_base.h`: `values`, `values_load()`, `compare_and_swap_gen()`; `zero_any_chunk.h`: `try_mark_cst_and_get_old()` |
| `materialise()`, the scan, the element store | `zero_any_chunk_inline.h`: `create_value()`, `wait_before_materialise()`, `fill_list_to_compress()`; `relaxed_element.h` |
| `writer()` | `proxy_base_inline.h`: `write_with_optional_lock_impl()` |

Three properties are asserted: **P1**, a writer never stores into a buffer the pass has returned
(the model frees it, and GenMC reports any later access); **P2**, a write that took the direct
path is in the chunk once every thread is done, and a deferred one is accounted for -- a write is
never silently lost; **P3**, with `--check-liveness`, every spin loop terminates.

The model has the two **shapes** of section 4 of the synchronisation document. `SHAPE_SYMMETRIC`
(the default) is what the checker can verify: C11 has no `membarrier()`. `SHAPE_ASYMMETRIC_NAKED`
is the asymmetric shape with the writer's plain store and *no* `membarrier()` on the other side:
it must fail, and its failure is exactly what the system call repairs. The asymmetric-to-symmetric
reduction is the argument of section 4.2 and of `membarrier(2)`'s own contract, not a checker
result. The **configurations** are `INIT_CONSTANT` (the chunk starts compressed, the writers materialise it;
the default is an all-zero buffer, a candidate for the pass), `BOUNDED_COMPRESSOR_WAIT` (the
`CHUNK_COMPRESS_MAX_YIELDS` budget reduced to its effect: one look, then give up) and `NWRITERS`.

## 4. The calibration

Every line below is one `run_genmc.sh` case; the verdicts are those of GenMC 0.19 under RC11 and
under IMM (identical). "Executions" is the number of complete executions the checker explored.

| case | expected | verdict | executions |
|---|---|---|---|
| the protocol of the source, marks, buffer start | pass | pass | 21 |
| source, marks, constant start (the writers materialise) | pass | pass | 26 |
| source, marks, 3 writers; constant start, 3 writers | pass | pass | 120; 191 |
| source, liveness, **unbounded** compressor wait, buffer or constant start | pass | pass | 21; 26 |
| source, liveness, constant start, 3 writers; bounded wait (the library) | pass | pass | 191; 82 |
| source, liveness, 5 writers, buffer start; constant start | pass | pass | 3,938; 9,637 |
| `MUT_NO_CHECK2`: no look at the gate after the announce | fail | lost write | -- |
| `SHAPE_ASYMMETRIC_NAKED`: asymmetric shape without `membarrier()` | fail | access to freed memory | -- |
| `MUT_NO_FENCE_COMPRESSOR`: the compressor announces without its barrier | fail | access to freed memory | -- |
| `MUT_WAIT_ON_HARD`: materialisation waits on `hard_inflight_` (the triangle of 5.3) | fail | liveness violation | 5 |
| `MUT_RELAXED_MARK`: the mark relaxed instead of release/acquire (5.3) | fail | access to freed memory | 8 |
| `MUT_MATERIALISE_WAIT_UNDER_FLAG`: the materialisation wait before **F2**, unbounded compressor wait | fail | **liveness violation** | 4 |
| the same with the library's bounded wait: safe, a stall only | pass | pass | 116 |

Every mutation is refused, and the protocol of the source passes in every configuration. The
last two lines are the behaviour this work changed in the source (section 8, F2): the library as
it was before is refused, the library as it is passes. The verdict on
`MUT_RELAXED_MARK` is worth a word: section 5.3 says the mark "needs its own
release/acquire pair" and that a relaxed read was "confirmed by TSan" -- TSan does not model
fences, so that confirmation alone could have been an artefact of the symmetric build. The checker
settles it: the acquire *is* load-bearing, and the trace shows how. A writer reads the mark at
zero **after** `steal_unmark()`; with a relaxed read nothing orders it after the pass's CAS, so its
`values_load()` may return the old buffer -- the one the pass has just returned -- and the write
lands in freed memory. The acquire pairs with the release of `steal_unmark()` and carries the CAS
with it.

## 5. The model of handshake 2: compressor against flush

`formal/compressor_flush.c` adds section 5.4 on top of handshake 1. NWRITERS writers (1 by
default, up to 3 checked) each write one element once -- **deferred** when check no. 2 sees the
pass's mark, direct otherwise -- and a writer that deferred then flushes its own rank, as the
library does past its backlog threshold: `announce_flush()` (seq_cst increment, barrier), the wait
for the stealers on `stealing_inflight_`, then the replay exactly as `lbd_equal_asynch` does it: a
**plain** store ("flushes exclude the stealers"), the post-store confirmation, the retry loop. The
pass has its two looks at the flushers, the early-out after `hard_inflight_++` and the second look
after `stealing_inflight_++`. A fourth property joins the three of handshake 1: **P4**, the plain
replay store never races the pass's scan (GenMC's race detection).

| case | expected | verdict | executions |
|---|---|---|---|
| source, 1 writer; 2 writers; 3 writers (liveness checked) | pass | pass | 12; 403; 25,810 |
| `MUT_NO_SECOND_LOOK`: the bug of 5.4 (1 and 2 writers) | fail | **race**, and with race detection off the **lost replay** itself | 1 |
| `MUT_NO_SECOND_LOOK` + `MUT_NO_POSTCHECK` | fail | lost replay | 1 |
| `MUT_FLUSH_NO_DRAIN`: the flush replays without waiting the stealers out | fail | lost replay | 1 |
| `MUT_NO_FIRST_LOOK`: the early-out removed (1 and 2 writers) | pass | pass | 10; 357 |
| `MUT_NO_POSTCHECK`: the post-store confirmation removed (1 and 2 writers) | pass | pass | 12; 403 |

Three things the table says. The **second look** is the pair: without it the checker's first
counter-example is the history's bug, line for line -- the writer defers on the mark; the flush
announces, finds `stealing_inflight_` at zero (the pass is parked), replays, its post-store
confirmation passes; the pass then opens its window, enters stealing, scans the pre-replay zeros,
CASes the chunk back to the constant and returns the buffer; the replayed write is gone. The **first
look** "pairs with nothing", as section 5.4 says: removing it changes no verdict, it is an
optimisation. The **post-store confirmation** is a second line of defence, not the exclusion: with
the second look in place it is never needed, and without the second look it does not save the write
(the pass's CAS can land after the confirmation's load).

## 6. The model of handshake 3: readers against buffer reclamation

`formal/reader_reclaim.c` is section 5.5 (`internal/read_reclaim.h`). One chunk holding an
all-zero buffer; NREADERS readers (2 by default, 3 checked) each doing NREADS pure reads (1 by
default; 2 and 3 exercise the "wait for the sequence to move, not to be even" rule), through
`ReadSection` as `element_at()` does: the odd entry store and the light fence, `values_load()`,
the element load, the even exit store (release). One pass run inside a region steals the buffer
(CAS to the constant), waits the grace period out as `PassScope` does -- heavy barrier, then each
slot, odd ones until they move, then the overflow counter -- and recycles the block, modelled as a
poison store (the value another chunk would put there) followed by the free. `OVERFLOW_READER`
sends one reader down the shared-counter path of a program with more reader threads than
`CHUNK_READ_RECLAIM_SLOTS`. The properties: a reader never dereferences a freed buffer, never
reads a value the chunk never held, and the grace period terminates.

| case | expected | verdict | executions |
|---|---|---|---|
| source, 2 readers; 3 readers; one read each, liveness checked | pass | pass | 25; 125 |
| source, 2 reads per reader; 3 reads; 3 readers x 2 reads | pass | pass | 225; 1,089; 3,375 |
| source, overflow reader; with 2 reads | pass | pass | 15; 90 |
| `MUT_ENTER_RELAXED`, 2 reads: the odd entry store as it was -- **F3** | fail | **access to freed memory** | 19 |
| `MUT_ENTER_RELAXED`, 1 read: the hole needs back-to-back reads | pass | pass | 25 |
| `MUT_WAIT_FOR_EVEN`, 2 reads (the pass-side way out of F3, not taken) | pass | pass | 169 |
| `MUT_NO_GRACE`: the pass frees without waiting | fail | access to freed memory | -- |
| `MUT_NO_HEAVY_BARRIER`: the pass reads the slots without its barrier | fail | access to freed memory | -- |
| `MUT_LEAVE_RELAXED`: the even exit store relaxed instead of release | fail | access to freed memory | -- |
| `SHAPE_ASYMMETRIC_NAKED`: the reader without its fence (no `membarrier()`) | fail | access to freed memory | -- |
| `MUT_OVERFLOW_NO_FENCE`: the overflow announce as it was -- **F4** | fail | access to freed memory | -- |

The protocol holds, every mutation is refused, and the two lines marked F3 and F4 are the
library as it was before this work (section 8): refused as it was, passing as it is.

## 7. The model of handshake 4: motif readers against the rewrites of the offset slot

`formal/motif_offset.c` is section 6 of [algo_pattern.md](algo_pattern.md). A chunk of
`pattern::Array` compressed to a **motif** points at a shared, registered motif, and keeps its
**offset** in its second-value slot: element `j` is `motif[j] + offset`, two words, the pointer
and the slot, loaded one after the other. The slot is rewritten by a compression pass, before it
switches the pointer: to the new offset when the pass compresses the chunk to a motif, to
`zero_any`'s second value -- or reset to 0 -- when it compresses it to a constant. The model has
one chunk of two elements at motif A plus 10; NREADERS readers (1, 2 and 3 checked) each reading
element 1 NREADS times (up to 3) under a `ReadSection`, as `element_at(offset, slot)` does; one
writer materialising the chunk from its motif and writing both elements; one pass compressing the
buffer to its TARGET -- another motif, the same motif (the pointer comes back: an ABA), the first
or the second constant -- by the slot store then the CAS, then waiting the grace period out and
recycling the buffer. The writer and the pass never work on the chunk at once: that is handshake
1, taken as given (the pass waits for the writer's mark). The property: a reader returns a value
element 1 held at some point, and never accesses the slot through a racing non-atomic store; the
read sections and the buffer's free are checked as in section 6.

| model | source (`chunk/public/chunk/`) |
|---|---|
| `values`, `slot` | `internal/chunk_base.h` `ChunkBase::values_`; the chunk's `SecondValueRef<T>::slot` |
| `read_element()` | `internal/chunk_base_inline.h` `element_at_unguarded(offset, slot)`, `read_block()`, `read_block_or_constant()`; `internal/proxy_base_inline.h` `read_values_at()`; `pattern_chunk.h` `Chunk::motif_offset()` |
| `writer()` | `internal/zero_any_chunk_inline.h` `create_value()`, the motif branch |
| `pass()`, a motif | `internal/pattern_chunk_inline.h` `block_clear_motif()` |
| `pass()`, a constant | `internal/second_value_chunk_base_inline.h` `block_clear_common()` (`reset_release()`); `internal/zero_any_chunk_inline.h` `find_or_set_ind_cst_val2_release()` |

| case | expected | verdict | executions |
|---|---|---|---|
| source: another motif; the same motif; first constant; second constant | pass | pass | 23 each |
| source, 3 reads; 2 readers x 2 reads; 3 readers | pass | pass | 287; 10,201; 12,167 |
| `MUT_NO_RECHECK`: the reader as stage 2 left it -- **F5**, another motif; first constant; second constant | fail | **safety violation** (a value the chunk never held) | 2 |
| `MUT_NO_RECHECK`, the same motif: the slot then holds the new state's offset | pass | pass | 11 |
| `MUT_SLOT_STORE_PLAIN`: the pass's store as `zero_any`'s `reset()` and `set_from_array()` -- **F5** | fail | **non-atomic race** | -- |
| `MUT_SLOT_LOAD_RELAXED`: the re-check with a relaxed slot load | fail | safety violation | 17 |
| `MUT_SLOT_STORE_RELAXED`: the pass's slot store relaxed | fail | safety violation | 17 |
| `MUT_CAS_BEFORE_SLOT`: the pointer switched before the slot is rewritten | fail | safety violation | 6 |
| `MUT_NO_GRACE`: the buffer freed without the grace period (calibration) | fail | access to freed memory | 1 |

The same verdicts under IMM. The design note's argument for this hazard did not hold (section
8.6): the model found it in two executions, and the source was changed before the variant is
announced.

## 8. Findings

### 8.1 F1: the gate of the build without marks read `hard_inflight_` relaxed (closed: the build was removed)

The `SHRINK_SMALL_CHUNK` build -- 8-byte chunks, no per-chunk mark, a global gate -- had
`CompressionManager::must_defer_write()` as `hard_inflight_.load(relaxed) > 0`. The checker,
on the model's configuration of that build, found this execution: the pass runs to completion -- mark, `start()`, scan, CAS of
`values` to the constant, buffer returned, `finish()` -- and only then the writer's check no. 2
reads `hard_inflight_`, which is back to zero. The writer proceeds; its `values_load()`
(acquire) is not ordered after the pass's CAS, since nothing synchronised with it, and returns the
**old** pointer: the write goes into the returned buffer. The symmetric fences are not at fault:
they only promise that the writer sees the increment *or a later value* of the counter, and the
decrement is a later value. The per-chunk mark has no such hole because `steal_unmark()` is a
release store and `steal_marked()` an acquire load. On AArch64 the relaxed `ldr` of the counter
may be satisfied after a later `ldar` of `values`; on x86 the two loads stay in order and the hole
is closed by the hardware.

The fix was one word, `memory_order_acquire` on that load, and the checker accepted it with 2
and 3 writers. The build was then **removed** instead (CHANGELOG, *Removed*): a `-D` define no
test or build script exercised, failing its own harnesses, and a second write protocol to keep
correct. Every threaded build has the marks now (`chunk_config::use_chunk_steal_mark`), and the
model's configuration of the removed build went with it: this section is the record of the
finding, and `synchronisation_dekker.md` section 5.3 keeps a line on it.

### 8.2 F2: a materialising writer and a parked compressor waited for each other (fixed)

The comment of `wait_before_materialise()` said it was "a no-op in practice: a fast-path writer
never gets here with the chunk marked (the gate stopped it)". The model shows the window the
sentence skipped: the mark can land **between** check no. 2 and `wait_before_materialise()`. A writer passes
check no. 2 on a compressed chunk, finds `values` at the constant, and enters `create_value()`;
the pass marks the chunk and parks in `start()`, waiting for the flags that point to a marked
chunk -- this writer's. The writer spins on the mark, the compressor on the flag. In the library
the compressor's wait is bounded (`CHUNK_COMPRESS_MAX_YIELDS`, 10,000 yields of 1,000 pauses):
not a deadlock, but a stall of both threads for that whole budget, after which the pass is
abandoned (the gate counter of aborts on an overly long wait, in the `CHUNK_STATS` report, counts these). It needs a
chunk that is at a constant while sitting in a candidate list -- the case `fill_list_to_compress()`
guards against with `is_not_cst_val1()` -- so it is rare, not impossible.

The fix follows from the handshake itself: a fast-path writer holds its flag and has passed
check no. 2, so a mark it sees now belongs to a compressor that is **parked** on that flag;
nothing can steal the chunk before `end_write()`, and the wait protects nothing.
`wait_before_materialise()` now takes the caller's rank and returns at once when that rank's
flag is on the chunk (`WriteCompressSyncPolicy::is_thread_writing_on()`, a relaxed load of the
thread's own flag). The callers with no flag -- `alloc_chunk()`, the pre-allocation entry point
usable from a parallel region, and a flush replay -- keep the wait, which the model does not
cover (handshake 2). With the fix the checker proves safety and liveness even with an
**unbounded** compressor wait (26 and 191 executions); `MUT_MATERIALISE_WAIT_UNDER_FLAG` is the
wait as it was.

### 8.3 What the checker confirmed

The rules of section 6 of the synchronisation document hold in the model: the two counters must
stay two (`MUT_WAIT_ON_HARD` is the triangle, found as a liveness violation in 5 executions); the
compressor's barrier and the writer's look after its announce are each load-bearing; the
asymmetric shape is unsound without `membarrier()`; and the protocol as shipped is correct, with
the per-chunk marks, for 2 and 3 writers, whether the chunk starts compressed or materialised.

### 8.4 F3: back-to-back reads and the C++20 release sequence (fixed)

Having seen a slot odd, the pass waits for it to **move**, by design: waiting for it to be even
would stall behind a reader whose reads follow one another. The value it then reads may be the
**next read's odd entry**, a relaxed store. In C++20 a release store followed by a relaxed store of
the same thread forms no release sequence (P0982R1 removed that case), so reading the odd entry
synchronised with nothing: in the model, the previous read's element load and the pass's free
were unordered, and GenMC reports the access (`MUT_ENTER_RELAXED` with 2 reads; with 1 read there
is no next entry and the hole does not exist). The reasoning of `read_reclaim.h` -- release on
exit "lets the reclaimer trust an even value" -- was right for an even value and silent about an
odd one.

On the mappings the library runs on, the hole was closed anyway: on AArch64 the exit store is an
`stlr`, the entry store a later `str` to the same address, and an observer of the later store
observes the earlier one and everything the release ordered before it; on x86 every store is a
release. So the code was right on every target shrink-array has, and wrong by the standard. Two
ways to make it right by the standard, both accepted by the checker: the odd entry store release
instead of relaxed -- an `stlr` instead of an `str` on the read path of AArch64 -- or
`MUT_WAIT_FOR_EVEN`, the pass waiting for an even value, free on the read path and the stall the
design refused. The first was measured on Grace with `examples/access_cost`, two builds of the
same tree alternated over 5 rounds x 5 repetitions (the vector's times, the same code in both,
within 0.6 %): at 1 thread `read-varied` +1.9 % with a 2-3 % spread, `read-constant` +0.6 %,
`read-random` +0.0 %, the writes within their spread; at 64 threads nothing measurable. **Applied**:
the entry store is release, the library is right by the standard for at most two percent on the
worst read case, and `MUT_ENTER_RELAXED` is the store as it was.

### 8.5 F4: the overflow reader announced without its fence (fixed)

A reader past `CHUNK_READ_RECLAIM_SLOTS` announces with `g_overflow_readers.fetch_add(seq_cst)`
and then loads `values`; the pass, after its barrier, loads the counter with acquire. A seq_cst
RMW followed by an acquire load is not a Dekker announce in C11: nothing puts the RMW and the
pass's fence in one order with the two loads, and the checker finds the store-buffering execution
-- the reader loads the old buffer while the pass reads the counter at zero and frees it. On
AArch64 an `ldaddal` keeps a later `ldar` behind it and on x86 a locked RMW is a full barrier, so,
as F3, the hole is the standard's; unlike F3 the fix is free: `light_fence()` after the increment,
as on the slot path (a compiler barrier under `membarrier()`, a fence otherwise, on a path a
program takes only with more than a thousand reader threads). Applied; `MUT_OVERFLOW_NO_FENCE` is
the line as it was.

### 8.6 F5: a motif reader could add another state's slot to its motif (fixed before release)

A reader of a `pattern::Array` motif chunk loads the values pointer -- the motif A -- then the
offset from the slot. Between the two loads a writer can materialise the chunk (a buffer filled
from A and the offset, the slot untouched) and write it, and a compression pass can compress the
buffer again: it rewrites the slot first -- the offset of another motif B, or the second value
of a constant -- then switches the pointer. The reader then returns `A[j]` plus the new slot
value: a value the chunk never held. The checker found it in two executions (the reader reads
A, then the pass's offset 20, and returns 5 + 20 = 25 where the chunk held 15, then 27).
The design note of `pattern::Array` (its section 6, now [algo_pattern.md](algo_pattern.md) section 6) had argued the hazard away: the slot "is rewritten only in that
pass, after that wait". It is not -- the grace period of `PassScope` protects the buffer the
pass frees, after the CAS, not the slot it rewrites before it. A second defect came with it: for
the two constants the pass rewrites the slot through `SecondValueRef::reset()` and
`set_from_array()`, plain stores, while the reader loads it through an `atomic_ref`: a data race,
undefined behaviour whatever the value is used for (`MUT_SLOT_STORE_PLAIN`).

The fix, accepted by the checker for every target, up to 3 readers and 3 reads: the reader loads
the slot with **acquire** and the values pointer **again**; unchanged, the offset belongs to the
state the two loads bracket, and the value is `motif[j] + offset`; moved, it starts over
(`Chunk::motif_offset()`, used by `element_at_unguarded(offset, slot)`, `read_block()`,
`read_block_or_constant()` and `read_values_at()`). The pass's slot stores are **release** atomic
stores for a member with the motif state (`reset_release()`, `set_from_array_release()`, and the
store of `block_clear_motif()`, already release); they pair with the reader's acquire load and
carry the pass's own sight of the materialised buffer, so a reader that sees the new slot cannot
see the old motif pointer again (`MUT_SLOT_LOAD_RELAXED`, `MUT_SLOT_STORE_RELAXED`). The ABA --
the pass compressing to the same motif -- is harmless with or without the re-check: the pointer
and the slot then describe the new state together. The slot first, the pointer second (section 6
of the note) stays load-bearing (`MUT_CAS_BEFORE_SLOT`).

Cost: on the read path of a motif chunk only, one acquire load instead of a relaxed one and a
second load of the values pointer, from the cache line the first load brought; `zero_any`'s code
is untouched (`if constexpr` on the motif state; the machine code of its libraries, examples
and validation programs compared function by function). The materialisation reads the slot once,
relaxed: `wait_before_materialise()` keeps the pass away from the chunk (handshake 1), so the
slot is stable there.

## 9. Limits

- **Fidelity.** The model is a reduction written by hand from the source: a line the model
  simplifies is a line the checker does not see. The mapping table of section 3 is the thing to
  review when the source changes, and the mutations are the guard against a model that proves
  nothing.
- **Bounds.** Up to 5 writers (handshake 1; 3 for handshake 2, each with its own flush; 3
  readers and 3 reads each for handshake 3; 3 readers, or 2 readers of 2 reads, and one
  materialisation then one compression for handshake 4), one chunk, one pass, one write each (5 writers with
  materialisation: 9,637 executions, 7 s). A bug that needs two passes or
  a chunk recycled twice is outside.
- **`membarrier()`.** No language memory model has it. The checker verifies the symmetric shape;
  the asymmetric one rests on section 4.2 and on `membarrier(2)`'s contract. GenMC also treats
  `atomic_signal_fence()` as a thread fence, so the compiler barrier of the asymmetric shape is
  left out of the model (the IR GenMC explores keeps program order anyway).
- **Model, not hardware.** RC11 and IMM are the C11 model and a refinement of it, not AArch64's
  own model. An execution they forbid is forbidden on every conforming compiler and processor; one
  they allow may or may not be observable on Grace.

## 10. Next steps

- Dartagnan on the same three files, under the official AArch64 model: the last limit above
  falls for the machine the library targets.
