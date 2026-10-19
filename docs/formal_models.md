# Formal models of the lock-free handshakes

> French version: [formal_models_fr.md](formal_models_fr.md).

[synchronisation_dekker.md](synchronisation_dekker.md) explains the three handshakes of the
library and argues their correctness on paper. `formal/` holds the same protocols as **reduced
models**: small C11 programs, one per handshake, that keep nothing but the atomics of the source
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

What "none" means has limits, listed in section 6: the model is a reduction (fidelity is the
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
| `values`, `values_load()`, `cas_install()`, `steal()` | `chunk_base.h`: `values`, `values_load()`, `compare_and_swap_gen()`; `chunk2A_std.h`: `try_mark_cst_and_get_old()` |
| `materialise()`, the scan, the element store | `chunk2A_std_inline.h`: `create_value()`, `wait_before_materialise()`, `fill_list_to_compress()`; `relaxed_element.h` |
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
result. The **configurations** are `NO_MARKS` (the `SHRINK_SMALL_CHUNK` build: global gate on
`hard_inflight_`), `INIT_CONSTANT` (the chunk starts compressed, the writers materialise it;
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
| source, no marks (`SHRINK_SMALL_CHUNK`); constant start; 3 writers | pass | pass | 21; 82; 120 |
| source, liveness, **unbounded** compressor wait, buffer or constant start | pass | pass | 21; 26 |
| source, liveness, constant start, 3 writers; bounded wait (the library) | pass | pass | 191; 82 |
| source, liveness, 5 writers, buffer start; constant start | pass | pass | 3,938; 9,637 |
| `MUT_NO_CHECK2`: no look at the gate after the announce | fail | lost write | -- |
| `SHAPE_ASYMMETRIC_NAKED`: asymmetric shape without `membarrier()` | fail | access to freed memory | -- |
| `MUT_NO_FENCE_COMPRESSOR`: the compressor announces without its barrier | fail | access to freed memory | -- |
| `MUT_WAIT_ON_HARD`: materialisation waits on `hard_inflight_` (the triangle of 5.3) | fail | liveness violation | 5 |
| `MUT_RELAXED_MARK`: the mark relaxed instead of release/acquire (5.3) | fail | access to freed memory | 8 |
| `MUT_NO_MARKS_RELAXED_GATE`: the no-marks gate before **F1** (relaxed load) | fail | **access to freed memory** | 5 |
| `MUT_MATERIALISE_WAIT_UNDER_FLAG`: the materialisation wait before **F2**, unbounded compressor wait | fail | **liveness violation** | 4 |
| the same with the library's bounded wait: safe, a stall only | pass | pass | 116 |

Every mutation is refused, and the protocol of the source passes in every configuration. The
last three lines are the two behaviours this work changed in the source (section 5): the
library as it was before is refused, the library as it is passes. The verdict on
`MUT_RELAXED_MARK` is worth a word: section 5.3 says the mark "needs its own
release/acquire pair" and that a relaxed read was "confirmed by TSan" -- TSan does not model
fences, so that confirmation alone could have been an artefact of the symmetric build. The checker
settles it: the acquire *is* load-bearing, and the trace shows how. A writer reads the mark at
zero **after** `steal_unmark()`; with a relaxed read nothing orders it after the pass's CAS, so its
`values_load()` may return the old buffer -- the one the pass has just returned -- and the write
lands in freed memory. The acquire pairs with the release of `steal_unmark()` and carries the CAS
with it.

## 5. Findings

### 5.1 F1: the gate without marks read `hard_inflight_` relaxed (fixed)

In the `SHRINK_SMALL_CHUNK` build (no per-chunk mark, `chunk_config::use_chunk_steal_mark`
false), `CompressionManager::must_defer_write()` was `hard_inflight_.load(relaxed) > 0`. The
checker finds this execution: the pass runs to completion -- mark, `start()`, scan, CAS of
`values` to the constant, buffer returned, `finish()` -- and only then the writer's check no. 2
reads `hard_inflight_`, which is back to zero. The writer proceeds; its `values_load()`
(acquire) is not ordered after the pass's CAS, since nothing synchronised with it, and returns the
**old** pointer: the write goes into the returned buffer. The symmetric fences are not at fault:
they only promise that the writer sees the increment *or a later value* of the counter, and the
decrement is a later value. The per-chunk mark has no such hole because `steal_unmark()` is a
release store and `steal_marked()` an acquire load. On AArch64 the relaxed `ldr` of the counter
may be satisfied after a later `ldar` of `values`; on x86 the two loads stay in order and the hole
is closed by the hardware.

The fix is one word: `memory_order_acquire` on that load (`compression_manager.h`). It pairs
with the `seq_cst` RMW of `finish()`, so a writer that sees the pass over also sees the `values`
it installed, and takes the constant path (materialisation). The checker accepts it with 2 and 3
writers; `MUT_NO_MARKS_RELAXED_GATE` is the line as it was. The build is a developer setting
([developer_settings.md](developer_settings.md)), not a shipped library; the default build is not
affected.

### 5.2 F2: a materialising writer and a parked compressor waited for each other (fixed)

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

### 5.3 What the checker confirmed

The rules of section 6 of the synchronisation document hold in the model: the two counters must
stay two (`MUT_WAIT_ON_HARD` is the triangle, found as a liveness violation in 5 executions); the
compressor's barrier and the writer's look after its announce are each load-bearing; the
asymmetric shape is unsound without `membarrier()`; and the protocol as shipped is correct, with
the per-chunk marks, for 2 and 3 writers, whether the chunk starts compressed or materialised.

## 6. Limits

- **Fidelity.** The model is a reduction written by hand from the source: a line the model
  simplifies is a line the checker does not see. The mapping table of section 3 is the thing to
  review when the source changes, and the mutations are the guard against a model that proves
  nothing.
- **Bounds.** Up to 5 writers, one chunk, one pass, one write each (5 writers with
  materialisation: 9,637 executions, 7 s). A bug that needs two passes or
  a chunk recycled twice is outside.
- **`membarrier()`.** No language memory model has it. The checker verifies the symmetric shape;
  the asymmetric one rests on section 4.2 and on `membarrier(2)`'s contract. GenMC also treats
  `atomic_signal_fence()` as a thread fence, so the compiler barrier of the asymmetric shape is
  left out of the model (the IR GenMC explores keeps program order anyway).
- **Model, not hardware.** RC11 and IMM are the C11 model and a refinement of it, not AArch64's
  own model. An execution they forbid is forbidden on every conforming compiler and processor; one
  they allow may or may not be observable on Grace.

## 7. Next steps

- Handshake 2, compressor against the deferred-write flush (5.4): the "second look", with the
  lost-replay bug of the library's history as its oracle.
- Handshake 3, readers against buffer reclamation (5.5): the current `ReadSection`, and the
  grace period under design next to it, so that only a correct variant is measured.
