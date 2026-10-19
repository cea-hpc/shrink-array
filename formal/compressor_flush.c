/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * formal/compressor_flush.c -- handshake 2 of docs/synchronisation_dekker.md: the compression
 * pass against the deferred-write flush (section 5.4), on top of handshake 1 (5.2, 5.3).
 *
 * A REDUCED MODEL. One chunk, one compression pass, NWRITERS writers (1 by default, -DNWRITERS=2)
 * each writing one element once: DEFERRED when check #2 sees the pass's mark, direct otherwise;
 * a writer that deferred then flushes its own rank, as the library does past the backlog
 * threshold: announce, drain the stealers, replay the op the way lbd_equal_asynch does -- a
 * PLAIN store, a post-store confirmation, a retry loop.
 * writer_compressor.c holds the mapping of everything handshake 1 contributes; the new names:
 *
 *   model                         source (chunk/public/chunk/internal/)
 *   ---------------------------   -------------------------------------------------------
 *   flushing                      write_compress_sync_policy.h  num_thread_flushing
 *   announce_flush(), end_flush() write_compress_sync_policy.h  announce_flush(), end_flush()
 *   any_flushing()                write_compress_sync_policy.h  any_thread_flushing() (seq_cst)
 *   first look / second look      compression_manager_inline.h  start(): the early-out, the re-check
 *   flush_deferred()              compression_manager_inline.h  flush_deferred()
 *   replay()                      proxy2A_inline.h              assign_impl(): lbd_equal_asynch
 *   queued op                     the rank's deferred map (one op here)
 *
 * PROPERTIES (asserted):
 *   P1  no store into a buffer the pass has returned (free: GenMC reports a later access);
 *   P2  when every thread is done every write IS in the chunk, direct or replayed: never lost;
 *   P3  (--check-liveness) every spin loop terminates;
 *   P4  (GenMC's race detection) the plain replay store never races the pass's scan.
 *
 * SHAPES: the two sides of this handshake are both rare, both seq_cst RMW + fence: only the
 * symmetric shape exists here; the writer's own announce keeps -DSHAPE_ASYMMETRIC_NAKED as in
 * writer_compressor.c (not the point of this model).
 *
 * MUTATIONS:
 *   -DMUT_NO_SECOND_LOOK       the compressor does not look at the flushers after stealing++
 *                              -- the bug of the library's history (a wrong second value at one
 *                              index, once in ~2,300 runs), section 5.4. Must FAIL.
 *   -DMUT_NO_FIRST_LOOK        the early-out at the top of start() removed. The document says it
 *                              "pairs with nothing": it must still PASS (it is an optimisation).
 *   -DMUT_NO_POSTCHECK         the replay's post-store confirmation removed. With the second look
 *                              in place the exclusion is complete: must still PASS (the check is
 *                              a second line of defence). Combined with MUT_NO_SECOND_LOOK: FAIL.
 *   -DMUT_FLUSH_NO_DRAIN       the flush replays without waiting stealing_inflight_ out: FAIL.
 *
 * Build for GenMC:   genmc [--check-liveness] -- compressor_flush.c
 * Build natively:    cc -O2 -pthread -DNATIVE compressor_flush.c && ./a.out
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef NWRITERS
    #define NWRITERS 1
#endif
#define NFLAGS NWRITERS
#define NELEMS NWRITERS

/* ───────────────────────── the shared state ───────────────────────── */

typedef struct
{
    _Atomic(int*) values;                               /* chunk_base.h: values (cst1 or a heap buffer) */
    atomic_uint   mark;                                 /* chunk_base.h: steal_mark_ */
} chunk_t;

static int     cst1[NELEMS];                            /* the shared constant buffer: never written */
static chunk_t chunk;

static _Atomic(chunk_t*) flag[NFLAGS];                  /* per_thread_writing[r].v */
static atomic_int        hard_inflight;                 /* compression_manager.h */
static atomic_int        stealing_inflight;
static atomic_int        flushing;                      /* write_compress_sync_policy.h num_thread_flushing */

/* each rank's deferred queue: one op at most (thread-private: written, then read, by its rank) */
static int  queued_v[NFLAGS];
static bool queued[NFLAGS];

static bool direct_done[NFLAGS];
static bool deferred_done[NFLAGS];

#ifdef NATIVE
static _Atomic(int*) returned_buffer;
#endif

/* ───────────────────────── elements: plain int, two access kinds ───────────────────────── */

/* Chunk2A's direct store / scan load: relaxed atomics on the element (relaxed_element.h). */
static void elem_store_relaxed(int* p, int v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
static int  elem_load_relaxed(const int* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

/* ───────────────────────── chunk primitives (as writer_compressor.c) ───────────────────────── */

static int* values_load(void) { return atomic_load_explicit(&chunk.values, memory_order_acquire); }
static bool steal_marked(void) { return atomic_load_explicit(&chunk.mark, memory_order_acquire) != 0; }
static void steal_mark(void) { atomic_store_explicit(&chunk.mark, 1, memory_order_release); }
static void steal_unmark(void) { atomic_store_explicit(&chunk.mark, 0, memory_order_release); }

static int* alloc_zero_buffer(void)
{
    int* b = malloc(sizeof(int) * NELEMS);
    for (int i = 0; i < NELEMS; ++i)
    {
        b[i] = 0;                                       /* filled BEFORE publication */
    }
    return b;
}

static void return_buffer(int* b)
{
#ifdef NATIVE
    atomic_store_explicit(&returned_buffer, b, memory_order_seq_cst);
#else
    free(b);
#endif
}

static bool cas_install(int* expected_cst, int* buffer)   /* compare_and_swap_gen() */
{
    if (atomic_load_explicit(&chunk.values, memory_order_relaxed) != expected_cst)
    {
        return false;
    }
    return atomic_compare_exchange_strong_explicit(&chunk.values, &expected_cst, buffer,
                                                   memory_order_acq_rel, memory_order_acquire);
}

/* create_value(): wait_before_materialise(chunk, thr_rank) -- no wait under the caller's own
   flag (F2); a caller with no flag (here: the flush replay) spins on the mark. */
static int* materialise(int r)
{
    const bool own_session = r >= 0 && atomic_load_explicit(&flag[r], memory_order_relaxed) == &chunk;
    if (!own_session)
    {
        while (steal_marked()) {}
    }
    int* b = alloc_zero_buffer();
    if (!cas_install(cst1, b))
    {
        free(b);
    }
    return values_load();
}

/* ───────────────────────── the policy ───────────────────────── */

static void announce_write(int r)                       /* symmetric shape */
{
    atomic_store_explicit(&flag[r], &chunk, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
}
static void end_write(int r) { atomic_store_explicit(&flag[r], NULL, memory_order_release); }

static void heavy_barrier(void) { atomic_thread_fence(memory_order_seq_cst); }

static bool any_writing_on_marked(void)
{
    for (int i = 0; i < NFLAGS; ++i)
    {
        chunk_t* c = atomic_load_explicit(&flag[i], memory_order_acquire);
        if (c != NULL && steal_marked())
        {
            return true;
        }
    }
    return false;
}

static bool must_defer_write(void) { return steal_marked(); }

static bool is_stealing(void)                           /* is_hard_compression_inflight(): seq_cst */
{
    return atomic_load_explicit(&stealing_inflight, memory_order_seq_cst) > 0;
}

static void announce_flush(void)                        /* begin_flush() + heavy_barrier() */
{
    atomic_fetch_add_explicit(&flushing, 1, memory_order_seq_cst);
    heavy_barrier();
}
static void end_flush(void) { atomic_fetch_sub_explicit(&flushing, 1, memory_order_release); }
static bool any_flushing(void) { return atomic_load_explicit(&flushing, memory_order_seq_cst) > 0; }

/* ───────────────────────── the flush and its replay ───────────────────────── */

/* lbd_equal_asynch: the replay of one deferred op, inside the flush's exclusion. */
static void replay(int idx, int v)
{
    while (true)
    {
        int* vals = values_load();
        if (vals == cst1)
        {
            vals = materialise(-1);                     /* no flag: waits on the mark */
        }
        int* values_before = values_load();
        if (values_before == cst1)                      /* pre-store guard: back through the loop */
        {
            continue;
        }
        values_before[idx] = v;                         /* raw store: "flushes exclude the stealers" (P4) */
#ifndef MUT_NO_POSTCHECK
        int* values_after = values_load();              /* post-store confirmation */
        if (values_after != values_before)
        {
            continue;
        }
#endif
        break;
    }
}

static void flush_deferred(int r)                       /* compression_manager_inline.h flush_deferred(thr) */
{
    if (!queued[r])
    {
        return;
    }
    announce_flush();                                   /* counter + heavy barrier, one Dekker verb */
#ifndef MUT_FLUSH_NO_DRAIN
    while (is_stealing()) {}                            /* drain the stealers; never hard_inflight_ */
#endif
    replay(r, queued_v[r]);                             /* FlushReplayGuard: gate bypassed */
    queued[r]        = false;
    deferred_done[r] = true;
    end_flush();
}

/* ───────────────────────── the writers ───────────────────────── */

static void write_element(int r, int idx, int v)        /* write_with_optional_lock_impl() */
{
    announce_write(r);
    if (!must_defer_write())                            /* check #2 */
    {
        int* vals = values_load();
        if (vals == cst1)
        {
            vals = materialise(r);
        }
#ifdef NATIVE
        assert(vals != atomic_load_explicit(&returned_buffer, memory_order_seq_cst));
#endif
        elem_store_relaxed(&vals[idx], v);
        end_write(r);
        direct_done[r] = true;
    }
    else
    {
        end_write(r);                                   /* slow path: the op goes to the rank's queue */
        queued_v[r] = v;
        queued[r]   = true;
    }
}

static void* writer(void* arg)                          /* rank r: writes element r, then drains its rank */
{
    const int r = (int)(long)arg;
    write_element(r, r, r + 1);
    flush_deferred(r);                                  /* the rank's own drain (threshold / barrier) */
    return NULL;
}

/* ───────────────────────── the compressor ───────────────────────── */

static bool compressor_start(void)
{
    atomic_fetch_add_explicit(&hard_inflight, 1, memory_order_seq_cst);     /* parked */
#ifndef MUT_NO_FIRST_LOOK
    if (any_flushing())                                 /* early-out: "defer compression, not the flush" */
    {
        atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
        return false;
    }
#endif
    heavy_barrier();
    while (any_writing_on_marked()) {}                  /* the writer window (unbounded here) */
    atomic_fetch_add_explicit(&stealing_inflight, 1, memory_order_seq_cst); /* parked -> stealing */
    atomic_thread_fence(memory_order_seq_cst);
#ifndef MUT_NO_SECOND_LOOK
    if (any_flushing())                                 /* THE SECOND LOOK: the Dekker pair of the flush */
    {
        atomic_fetch_sub_explicit(&stealing_inflight, 1, memory_order_seq_cst);
        atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
        return false;
    }
#endif
    return true;
}

static void compressor_finish(void)
{
    atomic_fetch_sub_explicit(&stealing_inflight, 1, memory_order_seq_cst);
    atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
}

static int* steal(void)                                 /* try_mark_cst_and_get_old() */
{
    int* old = values_load();
    if (old == cst1)
    {
        return NULL;
    }
    int* expected = old;
    if (!atomic_compare_exchange_strong_explicit(&chunk.values, &expected, cst1,
                                                 memory_order_acq_rel, memory_order_acquire))
    {
        return NULL;
    }
    return old;
}

static void* compressor(void* arg)
{
    (void)arg;
    steal_mark();
    if (!compressor_start())
    {
        steal_unmark();
        return NULL;
    }
    int* vals = values_load();
    if (vals != cst1)
    {
        bool all_first = true;
        for (int i = 0; i < NELEMS; ++i)
        {
            if (elem_load_relaxed(&vals[i]) != 0)       /* scan_constants: relaxed element loads */
            {
                all_first = false;
            }
        }
        if (all_first)
        {
            int* old = steal();
            if (old != NULL)
            {
                return_buffer(old);
            }
        }
    }
    steal_unmark();
    compressor_finish();
    return NULL;
}

/* ───────────────────────── main ───────────────────────── */

int main(void)
{
    atomic_init(&chunk.mark, 0);
    atomic_init(&chunk.values, alloc_zero_buffer());    /* an all-zero buffer: a candidate */
    for (int i = 0; i < NFLAGS; ++i)
    {
        atomic_init(&flag[i], NULL);
    }
    atomic_init(&hard_inflight, 0);
    atomic_init(&stealing_inflight, 0);
    atomic_init(&flushing, 0);
#ifdef NATIVE
    atomic_init(&returned_buffer, NULL);
#endif

    pthread_t tw[NWRITERS], tc;
    for (long r = 0; r < NWRITERS; ++r)
    {
        pthread_create(&tw[r], NULL, writer, (void*)r);
    }
    pthread_create(&tc, NULL, compressor, NULL);
    for (int r = 0; r < NWRITERS; ++r)
    {
        pthread_join(tw[r], NULL);
    }
    pthread_join(tc, NULL);

    /* P2: every write is in the chunk, direct or replayed; never lost */
    int* vals = atomic_load_explicit(&chunk.values, memory_order_seq_cst);
    assert(vals != cst1);
    for (int r = 0; r < NWRITERS; ++r)
    {
        assert(direct_done[r] || deferred_done[r]);
        assert(elem_load_relaxed(&vals[r]) == r + 1);
    }
    assert(atomic_load_explicit(&hard_inflight, memory_order_seq_cst) == 0);
    assert(atomic_load_explicit(&stealing_inflight, memory_order_seq_cst) == 0);
    assert(atomic_load_explicit(&flushing, memory_order_seq_cst) == 0);
#ifndef NATIVE
    free(vals);
#endif
    return 0;
}
