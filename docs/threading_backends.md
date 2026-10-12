# Threading backends — plan of the `threads` branch

Goal: OpenMP becomes **one** backend of the library's threading, no longer a
requirement. The library must work, and stay fast, whatever threads the
*application* uses (OpenMP, TBB, `std::thread`), and run its *own* parallel
work on a backend chosen at build time: OpenMP (the default, unchanged),
a `std::jthread` pool, or serial.

Each step below is one commit (or a few), with **no change of behaviour** in
the OpenMP build until step 7: the validations and the shrink_bench checksums
must stay identical at every step.

## 1. What OpenMP does in the library today

Inventory of the library sources (`chunk/`, `fast_allocators/`; examples and
benches excluded), October 2026: 80 `#pragma omp` in 20 files, 42 `omp_*`
calls, OMPT named in 23 files. Five different uses:

| | use | where | replaced by |
|---|---|---|---|
| **A** | runtime queries: rank, thread count, in-region, level | `PMP_CHUNK.h` (`CHUNK_PMP_THREAD_RANK`, `CHUNK_PMP_MAX_THREADS`, `CHUNK_PMP_IN_PARALLEL_REGION`, ...), `fast_allocators/internal/pmp_alloc.h`, `slab_arena.h` (`omp_get_max_threads`, `omp_in_parallel`), `manage_ompt.cpp` | the same macros, routed to the backend; rank of a non-OpenMP thread: dense, `thread_local`, given at its first use |
| **B** | one-shot data-parallel loops at a serial point (`parallel for` [`reduction`] [`if`]) | `array_chunk_base_inline.h:110`, `array_chunk_inline.h:117,524`, `array_chunk2_inline.h:450,487,535`, `array_chunk2A_inline.h:150,209`, `array_chunk2_base.h:175`, `parallel_storage_init.h:143,173`, `slab_arena.h:529,822` (dynamic) | `pmp::parallel_for(n, schedule, f)`, `pmp::parallel_reduce(n, init, f, op)` |
| **C** | team regions with worksharing and synchronisation (`parallel` + `for` / `single` / `masked` / `barrier` / `nowait`) | rechunk slices (`array_chunk_rechunk_inline.h:1139-1337`), compression passes (`array_chunk_inline.h:216-357`, `array_chunk2_inline.h:705-1068`, `array_chunk2A_inline.h:428-619`), team drain (`manager_chunk.h:1046-1300`, `manager_chunk_inline.h:414-525`), advisor (`chunk_size_advisor.h:392-722`), `all_type_chunk.cpp:305,520-545`, `array_chunk2_base.h:96`, `chunk*_std_inline.h`, `array_chunk.h:301`, `manage_ompt.h:171-176` | `pmp::team(nthreads, [&](pmp::Team& t) { t.for_static(...); t.barrier(); t.single(...); })` |
| **D** | mutual exclusion | `#pragma omp critical` (`array_chunk_inline.h:357`, `array_chunk2_inline.h:1009,1026,1068`, `array_chunk2A_inline.h:588,619`, `static_const_array_inline.h:83,145,154` named `fast_alloc_ladder`), `#pragma omp atomic` (`chunk_size_advisor.h:463,722`) | `std::mutex` / `std::atomic` — no backend involved |
| **E** | phase boundaries of the **application**: OMPT `parallel_begin` / `parallel_end` callbacks drive the deferred-op flush and the compression pass | `manage_ompt.cpp`, `manage_ompt.h`, `all_type_chunk.cpp:156` and `manage_ompt.cpp:295` (`#pragma omp barrier` run by **every** thread of the application's team) | the phase contract below; OMPT becomes its automatic implementation |

E is the only part with no mechanical replacement: no other runtime tells a
library that the application's parallel section ended (TBB's
`task_scheduler_observer` reports threads entering and leaving an arena, not the
end of a loop). The manual mode already exists for builds without OMPT
(`begin_parallel_phase()` / `finalize_parallel_phase()`, `getting_started`).

## 2. The phase contract (to settle before step 6)

Today the end of a phase is **collective**: every thread of the application's
team runs it, meets an OpenMP barrier, then drains its own deferred-op queue by
its rank. Outside OpenMP there is no team to meet in. Two shapes:

- **collective**: every application thread calls `shrink_array::end_phase()`;
  the library meets them on its own `std::barrier`, sized to the participants
  (registered at `begin_phase`). Closest to today; the application must call it
  on every thread.
- **serial**: the application calls `finalize_parallel_phase()` once, from one
  thread, after joining its workers; the library drains every rank's queue
  itself (the team drain already routes blocks by owner) on its own backend.
  Simplest for the application; the per-rank drain becomes a loop over ranks.

Proposal: serial for the non-OpenMP backends (one call, no barrier inside the
application's threads), collective kept for OpenMP + OMPT. A TBB application
gets an adaptor, `shrink_array::tbb::parallel_for(range, body)`, that calls
`tbb::parallel_for` and then `finalize_parallel_phase()`.

## 3. Steps

1. **This plan** and the inventory (this file).
2. **D**: `omp critical` -> `std::mutex`, `omp atomic` -> `std::atomic`. No
   backend involved; the OpenMP build behaves the same.
3. **B**: `pmp::parallel_for` / `pmp::parallel_reduce` in `PMP_CHUNK.h`,
   OpenMP implementation (same `schedule`, same `if` thresholds), the B sites
   converted. `fast_allocators` gets the same through `pmp_alloc.h`.
4. **C**: `pmp::team` and `pmp::Team` (`for_static`, `for_dynamic`, `barrier`,
   `single`, `masked`, `rank`, `size`), OpenMP implementation (a `parallel`
   region whose body receives the team; the constructs inside stay orphaned
   OpenMP constructs). Sites converted one by one, the rechunk refill last.
5. **A**: the queries through the backend; the `thread_local` rank registry for
   non-OpenMP threads (dense, bounded by the configured maximum, reused when a
   thread exits).
6. **E**: the phase contract (section 2) written down and implemented; OMPT
   becomes one way of calling it.
7. **Pool backend**: `std::jthread` workers, `std::barrier`, static partitioning
   (first touch where the static loops use the data); CMake
   `CHUNK_THREADS=openmp|pool|serial`; a validation harness driving arrays from
   `std::thread`s; the build in `compile_all.sh`.
8. (later, on demand) **TBB backend** for the library's own loops, inside the
   application's arena, `static_partitioner`; the adaptor of section 2.

## 4. Rules for the branch

- The OpenMP backend stays the default and the reference: at every step the
  OpenMP build gives the same results (validations, shrink_bench checksums)
  and the same timings within noise (`multi_launch_stress.py` before/after).
- No new dependency for the default build; TBB only behind an option.
- The library never starts threads of its own while the application's
  threads are active (oversubscription): its parallel work runs at serial
  points, on the backend of the application's runtime when there is one.
