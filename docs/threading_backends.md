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

*Settled (step 6)*: the serial shape is the contract of every backend --
`begin_parallel_phase()` on every thread that writes, before its first write,
then `finalize_parallel_phase()` once, from one thread, after the writers have
stopped. It already drained every rank's queue itself (the serial flush), so
nothing changes for an application that uses it. The collective shapes -- the
split end (`finalize_parallel_phase_only_flush()` in the region,
`finalize_parallel_phase_after_flush()` after it) and `finalize_barrier_phase()`
-- need a barrier of the application's team: OpenMP only. OMPT is the automatic
caller of the contract for OpenMP applications; `parallel_phase()` the manual
one, on a team of the backend. `manage_ompt.h` states the contract.

## 3. Steps

1. **This plan** and the inventory (this file).
2. **D**: `omp critical` -> `std::mutex`, `omp atomic` -> `std::atomic`. No
   backend involved; the OpenMP build behaves the same. *Done*: the six
   unnamed criticals (reports) hold `internal::diag_print_mutex()`
   (`diag_mutex.h`), the three `fast_alloc_ladder` ones
   `StaticConstArray::ladder_mutex()`, the advisor's two `omp atomic` are
   relaxed `std::atomic_ref::fetch_add`. Left for step 5: the ladder takes its
   lock only when `FAST_ALLOC_PMP_IN_PARALLEL_REGION`, an OpenMP query.
3. **B**: `pmp::parallel_for` and a reduction in `PMP_CHUNK.h`,
   OpenMP implementation (same `schedule`, same `if` thresholds), the B sites
   converted. `fast_allocators` gets the same through `pmp_alloc.h`. *Done*:
   one implementation, `fast_allocators/internal/pmp_loops.h` (namespace
   `fast_alloc::pmp`, included by `pmp_alloc.h`, re-exported by `PMP_CHUNK.h`
   as `shrink_array::pmp`): `parallel_for`, `parallel_for_local` (a per-thread
   state, the release of a shrink) and `parallel_sum<T, N>` (the only
   reduction the sites need: N counters, `reduction(+ : acc[:N])`); a general
   reduction waits for a site that needs another operator. Options:
   `use_team` (the `if` clause: false still opens a team of one, so the OMPT
   callbacks fire as before), `num_threads`, `schedule` (`static_blocks`,
   `static_chunks(c)`, `dynamic(c)`). The loops that had no `schedule` clause
   get `schedule(static)`, the default of libgomp and libomp. Converted: the
   exit-census counts of the three variants, `get_nb_chunk_written`, the
   Chunk2A size-prediction scan, `for_each_chunk`, `for_each_chunk_in`,
   `construct_range`, the two `madvise` sweeps of `slab_arena.h`.
4. **C**: `pmp::team` and `pmp::Team` (`for_static`, `for_dynamic`, `barrier`,
   `single`, `masked`, `rank`, `size`), OpenMP implementation (a `parallel`
   region whose body receives the team; the constructs inside stay orphaned
   OpenMP constructs). Sites converted one by one, the rechunk refill last.
   *4a done*: the API, in `pmp_loops.h` with the loops (one backend file):
   `team(opt, body)`, `team_sum<T, N>(opt, body)` (the `parallel
   reduction(+ : ...)` regions), `current_team()` (the innermost team: the
   library's, the caller's region, or a team of one) and `Team` (`rank`,
   `size`, `for_static`, `for_dynamic`, `barrier`, `single`, `masked`, each
   with `wait_mode::nowait` where OpenMP has it). Converted: the
   `compress()` / `compress_int()` passes of the three variants
   (`compress_int(pmp::Team&)`, the old signatures kept as forwarders on
   `current_team()`), their `clear_core()` teams, and the loop of
   `write_range_by_chunks`. *4b done*: `team_any(opt, body)` (the
   `parallel reduction(|| : ...)` region of the deferred flush); converted
   the per-rank sweeps of `compress_chunk_to_compress` (`chunk*_std_inline.h`),
   the manager drains (`manager_chunk.h`, `manager_chunk_inline.h`, its
   orphaned `single` on `current_team()`), the sweep and the teardown free
   team of `all_type_chunk.cpp`, and the size advisor's region. *4c done*:
   `Team::for_runtime` (`schedule(runtime)`: `CHUNK_RECHUNK_SCHEDULE`); the
   rechunk refill's slice team and `drain_reuse_list_for_size_team`, which now
   takes the team it runs on. Left after step 4, for step 6 (use E): the
   regions of `parallel_phase` (`manage_ompt.h`) and the `omp barrier` of
   `all_type_chunk.cpp`'s phase end, run by the application's team.
5. **A**: the queries through the backend; the `thread_local` rank registry for
   non-OpenMP threads (dense, bounded by the configured maximum, reused when a
   thread exits). *Done*: `fast_allocators/internal/pmp_runtime.h` (namespace
   `fast_alloc::pmp`, included by `pmp_alloc.h` and `PMP_CHUNK.h`), the one
   file that asks the runtime: `backend` / `threaded` (and
   `FAST_ALLOC_PMP_THREADED` for the declaration-level switches), `max_threads`,
   `num_procs`, `team_rank`, `team_size`, `in_parallel`, `level`,
   `active_level`, `team_size_at`, `ancestor_rank`, `set_num_threads`,
   `set_max_active_levels`, `get_runtime_schedule` / `set_runtime_schedule`;
   OpenMP: the `omp_*` call of the same meaning, serial: one thread. The
   `CHUNK_PMP_*` and `FAST_ALLOC_PMP_*` macros keep their names and route to
   it, the threaded branch of `PMP_CHUNK.h` (capacity, checked rank, library
   rank, rank bound, schedule) names no OpenMP call any more, and
   `slab_arena.h`'s sweeps and `manage_ompt.cpp` use the queries (no
   `#ifdef _OPENMP` left in `slab_arena.h`). The registry: `RankRegistry`
   (lowest free rank first, -1 when full, lock-free), the process-wide
   `thread_rank_registry()` (capacity fixed by
   `configure_thread_rank_registry` before its first use), `registered_rank()`
   (taken at the first call, given back by a `thread_local` destructor when
   the thread exits, or by `release_registered_rank()`); tested alone from
   `std::thread`s by `rank_registry_validation`. Nothing in the library calls
   it yet: the OpenMP backend numbers its threads itself, and the serial
   backend has one rank. Left for step 7: wiring it into the pool backend's
   rank (a full registry is the invariant failure of a rank past
   `chunk_thread_capacity()`), the meaning of `in_parallel()` there (the
   ladder of `static_const_array_inline.h` takes its lock only when it is
   true: with application threads no runtime sees, it must answer "other
   threads may be in the library"), and the declaration-level `#ifdef
   _OPENMP` of `tc_allocator.h`, `tc_allocator.cpp` and `chunk_base.h` (the
   serial allocators), which become `FAST_ALLOC_PMP_THREADED`.
6. **E**: the phase contract (section 2) written down and implemented; OMPT
   becomes one way of calling it. *Done*: the contract in section 2 and in
   `manage_ompt.h`; `parallel_phase()` opens its region with `pmp::team`
   (`num_threads` 0: the backend's default, OpenMP `omp_get_max_threads()`,
   where it used a region with no `num_threads` clause), and the barrier of
   `finalize_parallel_phase_only_flush()` is `pmp::current_team().barrier()`.
   No `#pragma omp` is left outside `pmp_loops.h`. The TBB adaptor of section 2
   waits for step 8.
7. **Pool backend**: `std::jthread` workers, `std::barrier`, static partitioning
   (first touch where the static loops use the data); CMake
   `CHUNK_THREADS=openmp|pool|serial`; a validation harness driving arrays from
   `std::thread`s; the build in `compile_all.sh`.
   In four parts: 7a the CMake option and `threaded` apart from `use_openmp`
   (no new backend), 7b the pool backend, 7c the rank registry and
   `in_parallel()` in it, 7d the `std::thread` harness and `compile_all.sh`.
   *7a done*: `CHUNK_THREADS` (`openmp` or `serial`; empty: from `OMP`, as
   before; `pool` refused until 7b), recorded as `CHUNK_OPT_THREADED` /
   `FAST_ALLOC_OPT_THREADED` in the generated build options
   (`chunk/public/chunk/build_options.h.in`) and as `chunk_THREADS` in
   `chunk/chunk-config.cmake.in`; `build_config::threaded` /
   `not_threaded` and `alloc_config::threaded`, each checked against
   `pmp::threaded` by a `static_assert`. Every library test of `use_openmp`
   is now one of `threaded` -- the per-rank machinery, the atomics, the
   deferred queues, the compression passes, `use_fast_block_compression` --
   except the three shapes of the phase contract only OpenMP has
   (`finalize_parallel_phase_only_flush`, `finalize_parallel_phase_after_flush`,
   `finalize_barrier_phase`), which keep `use_openmp`. The `*OmpPolicy`
   class names stay. The examples and harnesses, OpenMP programs, are left
   as they are.
   *7b done*: `CHUNK_THREADS=pool` (OMP off, `FAST_ALLOC_PMP_POOL=1` PUBLIC on
   the targets, checked by both generated build options against `CHUNK_OPT_POOL` /
   `FAST_ALLOC_OPT_POOL`). `pmp_pool.h` implements the API of `pmp_loops.h`
   with OpenMP's model: a team is the calling thread (rank 0) and workers
   1..n-1 of one process-wide pool of `std::jthread`s (worker k is always
   rank k, so successive static loops give a rank the same block: first
   touch); for_static splits as libgomp does, for_dynamic / single find
   their shared state by the count of constructs each thread met, barrier is
   a `std::barrier`, the end of a team a join; a nested team runs as a team
   of one (max_active_levels 1); one pool team at a time. `pmp_pool_state.h`
   holds what the queries read: each thread's stack of team frames and the
   settings (`set_num_threads`, the run-time schedule); the default team
   size is `CHUNK_NUM_THREADS`, else `OMP_NUM_THREADS`, else the hardware
   threads. The declaration-level `#ifdef _OPENMP` of `tc_allocator.h`,
   `tc_allocator.cpp` and `chunk_base.h` are `FAST_ALLOC_PMP_THREADED`. In
   7b the application drives the arrays from ONE thread; the library's own
   teams run on the pool. Application threads are 7c.
   *7c done*: the application's threads under the pool backend.
   - **Rank**: `pmp::thread_rank()` (`pmp_runtime.h`) is the rank in the
     innermost active team of the pool when the thread is in one
     (`pool::ThreadState::active_rank`, kept by `FrameGuard`), its
     `registered_rank()` otherwise -- the application's threads, and the
     thread that drives the library between teams. `chunk_pmp_library_rank()`
     returns it under the pool backend; a full registry (-1) is the invariant
     failure "no rank left", as a rank past `chunk_thread_capacity()` is.
     `chunk_thread_capacity()` configures the registry with its own capacity
     before the first rank; built first by the allocator, the registry takes
     the same rule (hardware threads, team size, `CHUNK_THREAD_CAPACITY`).
     `FAST_ALLOC_PMP_THREAD_RANK` is `thread_rank()` too (OpenMP: the same
     `omp_get_thread_num()` as before), and the ladder's batches refuse a
     rank of -1 as they refuse one past their size.
   - **The two numberings share the per-rank structures, apart in time**: the
     thread that opens a team is rank 0 in it whatever its registered rank,
     and worker k is rank k, which is also an application thread's registered
     rank. The contract keeps them apart: no team of the pool opens while
     application threads write (below), and the team drains and sweeps
     already mop up the ranks past the team (`CHUNK_PMP_RANK_BOUND`, which the
     registered ranks raise through `chunk_pmp_checked_rank`).
   - **Rank cache**: `chunk_rank_cache()` is `g_fast_omp_rank` (OpenMP,
     serial) or the pool's `ThreadState::rank_cache`, which `FrameGuard`
     clears on entering a team and restores on leaving it -- what OMPT's
     implicit-task callback does for `g_fast_omp_rank` -- and
     `release_registered_rank()` clears. `get_thr_rank_fast()` and
     `begin_parallel_phase()` go through it.
   - **Application phase** (`pool::Settings::application_phase`): opened by
     `begin_parallel_phase()` from a thread in no team of the pool
     (`pmp::enter_application_phase()`), closed first thing by
     `finalize_parallel_phase()` (`leave_application_phase()`). While open,
     `in_parallel()` is true on every thread -- the ladder takes its lock, the
     loops that take a team only at a serial point stay on the caller -- and
     `pool::open_team` runs every team as a team of one. `parallel_phase()`
     on the pool does not open it: its `begin_parallel_phase()` calls are made
     inside a team. A program that drives the arrays from ONE thread without
     calling `begin_parallel_phase()` (7b) sees no change.
   - **Phase guard**: `check_begin()` stamps a thread that begins in the
     application phase (level 0, phase open), as one of a region, so a
     deferred write of an unstamped thread still fails in check builds. A
     second `begin_parallel_phase()` of the same thread in the phase is
     allowed (same rank, same queue, writes in program order): the OpenMP
     harnesses run serially in a pool build call it once per loop.
   - **Op pools of exited threads** (`deferred_op_pool.h`): the serial
     contract joins the writers, then finalizes -- so an application thread
     exits with writes still queued on its rank. Its `thread_local` op / ctx
     byte allocator gave its slabs back to the Op / Ctx arena at the exit,
     where another thread's next slab took them and overwrote the queued ops
     (`[DEFERRED_LIFECYCLE] op magic corrupt/poisoned`, found by the harness
     below). Under the pool backend (`deferred_pools_outlive_thread`) the
     allocator lives on the heap, the exiting thread hands it to
     `detail::exited_thread_pools()`, and `finalize_parallel_phase()`
     destroys those after its flush (`retire_exited_thread_pools()`).
     OpenMP and serial keep the allocator inside the `thread_local`.
   - Harness: `std_thread_validation` (pool builds, every variant and
     geometry, 1 and 8 threads): fresh `std::thread`s every phase with the
     main thread writing too, then long-lived ones on a `std::barrier`, then
     work inside the phase -- each thread compresses its own array (a pass
     on the calling thread), rewrites part of it, writes its share of a
     shared array, one write in 7 deferred (`force_defer_every`); block and
     interleaved shares, compression at the serial points, the arrays
     against `std::vector`s after every phase, distinct ranks, ranks given
     back at thread exit, `in_parallel()` during the phase only. It was
     planned for 7d; 7d keeps the build in `compile_all.sh`.
   - Harnesses made backend-neutral (the 7b suite in pool): they failed in
     7b already, from OpenMP assumptions, not from the pool.
     `bulk_write_team_validation` created its in-region arrays under
     `#pragma omp critical` inside `parallel_phase()` -- a real team of the
     pool, where the pragma is ignored: eight arrays created at once, out of
     contract (TSan: `PointerMapChunk2A::insert`); a `std::mutex` now.
     `pool_accounting` expected parking only under `use_openmp` (now
     `threaded`, as the library since 7a) and materialised with an
     `omp parallel for`, one thread in pool, which takes back the parked
     buffers of one rank only; it now fills through `parallel_phase()` and
     `current_team().for_static`.
   - Not done: the application's threads must stop writing before
     `finalize_parallel_phase()` -- nothing detects a writer that did not call
     `begin_parallel_phase()` outside a check build (`phase_guard::check_write`
     does, there). `churn::Materialisations` stays sized by
     `CHUNK_PMP_MAX_THREADS` (statistics only; ranks past it are not counted).
     The harness does not tell `in_parallel()`'s application-phase answer
     from the old one (a mutation that drops it passes, TSan included): the
     paths it guards (the ladder's lock, the in-region frees) are not reached
     by its workload -- covered in 7d.
   *7d done*: the pool build in the build matrix, the harness completed.
   - `compile_all_gnu.sh`: `cmake-build-pool-gnu` (release, for the timings
     against OpenMP), `cmake-build-pool-gnu-check` (CHECK_OPTIM: the
     validations) and `cmake-build-pool-gnu-relwithdebinfo-thread-sanitizer`.
     Unlike libgomp the pool is standard C++, so TSan sees every
     synchronisation of the library's teams there.
   - `std_thread_validation`, two more cases. *Cross-rank frees*: each thread
     materialises its array, then folds its neighbour's (constants,
     `compress()`), while the neighbour does the same to the next one.
     *Maintenance refused in the phase* (Chunk2A): `reclaim_second_values()`
     from one application thread while the others write is refused as "in
     a parallel region" (`second_value_stats().refused_parallel`) and runs
     after `finalize_parallel_phase()`. The mutation that drops the
     application phase from `in_parallel()` now fails this case, where it
     passed every other one: in a threaded build a pass in the phase parks
     its buffers on the rank's reuse list (probe: neither
     `release_to_owner` nor the foreign free of `free_block_to_pool` ran in
     an application phase), so the frees `in_parallel()` routes are not
     reached by writes and passes; the maintenance calls (reclaim, trim,
     sweeps) are what it protects there.
   - Found by the harness under TSan (chunk2a, dyn, 8 threads): the deferred
     replay of `=` on Chunk2A stored the element with a plain store while
     another writer's `may_become_uniform` scan read it relaxed; Chunk2A's
     element stores must be relaxed atomics (`relaxed_element_stores_v`),
     as the compound ops' replay already was. Now
     `store_element_relaxed` (`proxy2A_inline.h`). Not specific to the pool:
     the replay runs in a region under OpenMP as well. Same machine code on
     x86 and Arm (an aligned store), so no value was wrong; a data race all
     the same.
   - `array_chunk2A.h`: `reclaim_second_values()`'s warning said that a
     `std::thread` pool is never detected; under the pool backend, within
     the phase contract, it is.
   - Left as they are: the `omp critical` of `deferred_path_validation` and
     `blk_allocator_validation` sit in `omp parallel` regions, serial in a
     pool build (the latter tests OpenMP's critical reentrancy itself).
   *7e done*: shrink_bench on the backend's teams. Compiled without OpenMP,
   the bench's pragmas were ignored: in a pool build the application side
   ran on ONE thread, and an OpenMP-vs-pool timing compared 64 threads with
   one. Its regions and loops now go through `examples/shrink_bench/bench_par.h`,
   one source for every backend:
   - a region is `bench_par::region(body)`, `body(Team& t)` on a team of
     `pmp::team` (OpenMP: a `#pragma omp parallel` calling the body);
   - a loop is a header macro, `BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e)`
     and its schedules (`_STATIC`, `_BLOCK` for `schedule(static,
     sched_blk)`, `_ROUND_ROBIN` for `schedule(static, 1)`), then the body.
     OpenMP: the pragma the loop had, unchanged, and
     `for (Int i = (b); i < (e); ++i)` -- the worksharing the compiler sees is
     the same orphaned `omp for`. Pool and serial: the calling thread's share,
     computed from `t.rank()` and `t.size()` as libgomp shares the same
     schedule out, so a rank gets the same iterations in every loop of a
     region (the nowait reasoning of bench_phase_compute.cpp) and from one
     region to the next (first touch);
   - reductions: the body adds to `BENCH_RED(v)`; OpenMP keeps its
     `reduction(+ : v)` clause on the shared `v`, the other builds add a
     per-thread partial into `v` atomically when the thread's share ends.
     The region-level `parallel for reduction` loops (census, checksum) are
     `pmp::parallel_sum`, the zeroing of the baseline vector
     `pmp::parallel_for`;
   - `omp single nowait` is `t.single(f, bench_par::nowait)`; the `sections`
     of the reference check are one `single nowait` each; `omp critical` is
     `bench_par::critical(f)` (one mutex, as an unnamed critical), `omp
     atomic` `bench_par::atomic_add` (`omp atomic` itself under OpenMP), the
     stress phase's `schedule(dynamic, grain)` `Team::for_dynamic`;
   - the phase ends: the split end (`only_flush` in the region,
     `after_flush` after it) stays under OpenMP; the other backends call
     `finalize_parallel_phase()` after the region (bench_phases.h), the
     serial shape, which drains every rank's queue itself. No library change.
   - The prealloc test and the compression dispatch passed the rank to
     `alloc_chunk` and drained the queued candidates only under
     `use_openmp`; now under `threaded`.
   Checked here (gcc 13, Release + CHECK_OPTIM, chunk2a_256): every array
   checksum of `--mode check --no_io` runs (with and without `--test_prealloc
   --test_stl`) is the same for the OpenMP build before the change, after it,
   and the pool build, at 1, 4 and 8 threads.
   *Pool, v1's in-write block pass*: `ChunkFastCompPolicy::make_block_compression`
   (chunk_std_inline.h) took its writer protection -- the steal marks and
   `start()` / `finish()` -- only when `CHUNK_PMP_MAX_THREADS > 1`: a team of
   one meant "no other writer". Under the pool backend the application's
   threads write around a team of one, so with `OMP_NUM_THREADS=1` and two
   application writers the pass folded a chunk another thread was writing
   (lost writes, `got=0 expected=3.5`, 2 runs in 5 under TSan; TSan's race on
   `is_constant_to`). The pass is now guarded when the team has more than one
   thread or an application phase is open (`pmp::in_application_phase()`,
   always false outside the pool backend: OpenMP unchanged), decided once for
   the `start()` and its `finish()`. The harness `std_thread_validation` (chunk, 256, 1 thread)
   under TSan: 6 runs of 6 clean, against 6 of 6 with races before. Chunk2
   and Chunk2A take the protection in every threaded build.
8. **TBB**: the adaptor of section 2 (8a), then, on demand, a TBB backend for
   the library's own loops, inside the application's arena,
   `static_partitioner` (8b).
   *8a done*: `chunk/tbb_adaptor.h`, header-only (the library does not link
   TBB; the program that includes it does).
   - `shrink_array::tbb::parallel_for` takes the forms of
     `oneapi::tbb::parallel_for` with the same arguments -- a range and a body,
     or first, last [, step] and a function of the index, then a partitioner
     and / or a `task_group_context` -- and runs them as one phase:
     `begin_parallel_phase()` before every piece of the range a thread runs
     (TBB has no hook at the start of a task; a thread that runs several
     pieces begins again, which the phase guard allows in an application
     phase), `finalize_parallel_phase()` once the TBB call returned, also
     when the body threw. The index forms run on a `blocked_range` of the
     iteration numbers: one begin per piece, not per index.
     `shrink_array::tbb::phase_body(body)` is the wrapper itself, for a
     `oneapi::tbb::parallel_for` nested in the body. A step that is not
     positive is refused with `std::invalid_argument`, as TBB does, before
     any phase opens. The adaptor's own calls are qualified: unqualified,
     argument-dependent lookup also finds `oneapi::tbb::parallel_for`
     (`blocked_range` is TBB's), which wins the overload resolution and
     skips both ends of the phase -- the harness's first run caught it in
     the index form.
   - Backends: pool (each TBB thread takes a rank of the registry, as the
     `std::thread`s of 7c; the flush and the pass of the finalize run on the
     pool) and serial (the body over the whole range on the calling thread,
     without TBB: the library is not thread-safe there). OpenMP refuses it
     at compile time, when it is called: OpenMP numbers only its own
     threads.
   - CMake `CHUNK_WITH_TBB` (OFF): `find_package(TBB)` and the harness,
     `tbb_validation` (pool and serial builds, every variant and geometry, 1
     and 8 threads, in a `task_arena` of that many threads): each
     partitioner (auto, simple with a small grain, static), the index form
     with steps 1 and 3, compression and forced deferrals in the phase, a
     nested `oneapi::tbb::parallel_for`, a throwing body, a narrower arena;
     the arrays against `std::vector`s after every phase, one rank per thread
     all phase long and distinct between threads, the application phase
     open in the pieces and closed after the call. Mutations: the adaptor
     without its finalize (the phase stays open) or without its begin (the
     pieces run outside the phase) fails it, at 1 and 8 threads.
   - TSan: the harness needs a oneTBB built with TSan (its CMake:
     `-DTBB_SANITIZE=thread`). With an uninstrumented libtbb TSan sees no
     hand-off between tasks, and it reports TBB's own scheduling and every
     value one task passes to another (hundreds of reports on the first
     loop). Against an instrumented oneTBB 2021.11: no report (Chunk2 and
     Chunk2A, 8 threads), while a plain counter incremented in the pieces is
     reported at once. `compile_all_gnu.sh` gives the pool TSan build TBB
     only from `CHUNK_TBB_TSAN_DIR`, the package directory of such a build.
   - The registry must hold every TBB thread alive at once and the calling
     thread: `chunk_thread_capacity()` (the hardware threads, the team size,
     `CHUNK_THREAD_CAPACITY`) covers TBB's default arena; a wider
     `task_arena` needs `CHUNK_THREAD_CAPACITY`. TBB's workers keep their rank
     until they exit (at the end of the program): the ranks stay dense.
   *8b, not planned*: a TBB backend for the library's own work. The B loops
   have no barrier inside and would map onto `oneapi::tbb::parallel_for`
   (`static_partitioner`) in the caller's arena. The C teams would not: TBB
   guarantees no concurrency between tasks, so a `Team::barrier()` among n
   tasks deadlocks when fewer than n threads come (a `global_control` limit,
   workers held by another arena), and the passes that matter (compression,
   rechunk, drains) are C teams. They would stay on the pool's
   `std::jthread`s, which run at serial points only, while TBB's workers are
   idle. So the library has two threaded backends, OpenMP and pool, and a TBB
   program uses the pool backend through the adaptor; the TBB bench (8c)
   measures what that costs, and this decision is revisited only if it does.
   *8c done*: the TBB bench, `shrink_bench_tbb_<variant>_<declination>` (pool
   builds with `CHUNK_WITH_TBB`), from the sources of shrink_bench (7e).
   - The bench's regions have no barrier: every loop and every single is
     nowait (the two loops of the prealloc test that ended with one, right
     before the end of their region, are nowait now). So a region can run as
     `n` TBB tasks, one per virtual rank, that never wait for each other: a
     task never blocks one that has not started, the way a barrier deadlocks
     TBB. `bench_par.h` under `BENCH_PAR_TBB`: `region()` is a
     `oneapi::tbb::parallel_for` over the ranks `[0, n)` (grain 1,
     `static_partitioner`) in an arena of `CHUNK_PMP_MAX_THREADS` threads (the
     backend's team size: `OMP_NUM_THREADS`, as the scripts set it); its
     `Team` is the bench's own -- rank, size, `single(nowait)` and
     `for_dynamic(nowait)` on per-region slots, no `barrier()`. The loop
     macros share by the virtual rank, as under the pool.
   - The phase protocol is the bench's own calls, unchanged: each task begins
     the phase (`begin_parallel_phase()`, again for a thread that runs a
     second rank, which the phase guard allows), and the serial
     `finalize_parallel_phase()` after the region.
   - The compression phase: a team region calls `compress()` on every thread,
     a team-wide pass. TBB tasks are no team of the library, so there each
     array is compressed once, at the serial point, where the pass runs on
     the library's own team (`bench_par::team_regions`). The per-thread
     checksum slots are indexed by the region's rank (`t.rank()`, a virtual
     rank under TBB), no longer by the library's.
   - `multi_launch_stress.py --exec tbb_chunk2a` runs it (by name only:
     `all` stays the three variants); `compile_all_gnu.sh` builds it in the
     release pool build when oneTBB is there.

## 4. Rules for the branch

- The OpenMP backend stays the default and the reference: at every step the
  OpenMP build gives the same results (validations, shrink_bench checksums)
  and the same timings within noise (`multi_launch_stress.py` before/after).
- No new dependency for the default build; TBB only behind an option
  (`CHUNK_WITH_TBB`).
- The library never starts threads of its own while the application's
  threads are active (oversubscription): its parallel work runs at serial
  points, on the backend of the application's runtime when there is one.
