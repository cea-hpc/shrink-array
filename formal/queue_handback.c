/*
 * Copyright 2026 CEA (www.cea.fr)
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
 * formal/queue_handback.c -- the end of a compression pass against the candidate marking of a
 * writer: does a chunk that BECOMES constant always end in a candidate list?
 *
 * A REDUCED MODEL. One chunk of NELEMS elements, materialised, all first_value (0) but the last
 * one; it is a candidate in the list of one pass (its "queued" flag is up). The writer W writes
 * that last element to 0: after it, the chunk is all 0 -- compressible -- and the write is the
 * one that should queue it. With -DWITH_W2, a second writer stores a plain value into element 0
 * (the exclusion check: a write the pass must never lose, nor store into a returned buffer).
 * Handshake 1 is writer_compressor.c's, reduced to the symmetric shape; the names:
 *
 *   model                       source (chunk/public/chunk/internal/)
 *   -------------------------   ---------------------------------------------------------
 *   queued                      proxy_base.h      tab_is_inside_chunk_list_to_compress[t][c]
 *   mark_candidate()            proxy_base_inline.h  add_to_compression_list(): probe, CAS 0->1
 *   listed                      the entries of the chunk in every rank's candidate list
 *   pass()                      zero_any_chunk_inline.h  make_block_compression(): claim
 *                               (steal_mark), start(), scan (fill_list_to_compress), clear +
 *                               steal_unmark of the constant ones, then the hand-back loop:
 *                               steal_unmark, reset_tab_inside_list_to_compress (release)
 *   writer()                    proxy_base_inline.h  write_with_optional_lock_impl(): announce,
 *                               check 2 (must_defer_write = steal_marked), body (materialise,
 *                               store, mark for compression), end_write
 *
 * A deferred write is modelled as a retry once the mark is down: the library replays it later
 * still (after the pass's finish()), so the retry covers more interleavings, not fewer -- and a
 * writer arriving fresh in the same instant takes the direct path anyway.
 *
 * PROPERTIES (asserted when every thread is done):
 *   Q1  a materialised chunk whose content is constant is in a candidate list: the write that
 *       made it constant queued it, or the pass that dropped it queued it again;
 *   Q2  queued == 1 exactly when the chunk is listed, and it is listed once at most;
 *   P1  no store into a buffer the pass has returned (GenMC reports the access to freed memory);
 *   P2  (-DWITH_W2) W2's value is in the chunk.
 *
 * THE SOURCE (default): the flag is HANDED BACK (chunk/internal/queue_flag.h): once the scans
 * are done, every candidate's flag goes 1 -> 2 while its mark still keeps the writers out
 * (queue_flag::hand_back); a writer that finds 2 turns it into 3 ("written meanwhile") instead
 * of giving up (queue_flag::try_queue); the pass then resets 2 -> 0 by CAS, and on a 3 queues
 * the chunk again itself (queue_flag::release). Must PASS.
 *
 * MUTATIONS, each must FAIL Q1:
 *   -DMUT_OLD_HANDBACK           steal_unmark(), then the flag reset to 0 -- the source before
 *                                (F6, docs/formal_models.md section 8.7): between the two a writer
 *                                writes directly (no mark), its CAS 0->1 meets the flag still up
 *                                and gives up, then the reset drops the chunk.
 *   -DMUT_HANDBACK_AFTER_UNMARK  the flag goes 1 -> 2 after the mark is down: the window is back.
 *   -DMUT_WRITER_IGNORES_2       the writer gives up on a 2 as on a 1: nobody notes the write.
 *
 * VARIANT kept for the record (no verdict asserted on Q1 alone):
 *   -DORDER_FLAG_FIRST           the old reset BEFORE steal_unmark() -- the order the source's
 *                                comment forbids for what it does to ANOTHER pass's claim, which
 *                                this one-pass model does not have.
 *
 * Build for GenMC:   genmc -- [-DMUT_...] [-DWITH_W2] queue_handback.c
 * Build natively:    cc -O2 -pthread -DNATIVE queue_handback.c && ./a.out
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#define NELEMS 2

#if !defined(MUT_OLD_HANDBACK) && !defined(ORDER_FLAG_FIRST)
    #define HANDBACK 1 /* the source */
#endif
#define W2_VALUE 5

/* ───────────────────────── the shared state ───────────────────────── */

static int        cst1[NELEMS];          /* the shared constant buffer (all 0): never written */
static _Atomic(int*) values;             /* chunk_base.h: values */
static atomic_uint   mark;               /* chunk_base.h: steal_mark_ */

#define NWRITERS 2
static atomic_int flag[NWRITERS];       /* WritingFlag::v: 1 = writing THE chunk (the one chunk), 0 = null */
static atomic_int hard_inflight, stealing_inflight;

static atomic_int queued;                /* 0 free, 1 queued; 2 handed back, 3 written meanwhile (FIX) */
static atomic_int listed;                /* entries of the chunk across the candidate lists */

/* ───────────────────────── elements ───────────────────────── */

static void elem_store(int* p, int v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
static int  elem_load(const int* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

/* ───────────────────────── marking (add_to_compression_list) ───────────────────────── */

/* The writer's side of the queue flag; true if it pushed an entry. */
static bool mark_candidate(void)
{
#ifdef HANDBACK
    for (;;)
    {
        int v = atomic_load_explicit(&queued, memory_order_relaxed);
#ifdef MUT_WRITER_IGNORES_2
        if (v == 1 || v == 2 || v == 3)
            return false;
#else
        if (v == 1 || v == 3)
            return false;                                        /* listed, or already noted */
#endif
        if (v == 2)
        {
            if (atomic_compare_exchange_strong_explicit(&queued, &v, 3, memory_order_acq_rel,
                                                        memory_order_acquire))
                return false;                                    /* the pass queues it again */
            continue;
        }
        if (atomic_compare_exchange_strong_explicit(&queued, &v, 1, memory_order_acq_rel,
                                                    memory_order_acquire))
        {
            atomic_fetch_add_explicit(&listed, 1, memory_order_relaxed);
            return true;
        }
    }
#else
    int expected = 0;
    if (atomic_load_explicit(&queued, memory_order_relaxed) == 0 &&
        atomic_compare_exchange_strong_explicit(&queued, &expected, 1, memory_order_acq_rel,
                                                memory_order_acquire))
    {
        atomic_fetch_add_explicit(&listed, 1, memory_order_relaxed);
        return true;
    }
    return false;                                                /* flag already up: refused */
#endif
}

/* ───────────────────────── the writer (handshake 1, symmetric shape) ───────────────────────── */

/* One attempt: false if check 2 saw the pass's mark (the write is then deferred). */
static bool writer_attempt(int rank, int idx, int v)
{
    atomic_store_explicit(&flag[rank], 1, memory_order_seq_cst);                    /* announce */
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(&mark, memory_order_acquire) != 0)                    /* check 2 */
    {
        atomic_store_explicit(&flag[rank], 0, memory_order_release);               /* end_write */
        return false;
    }
    int* buf = atomic_load_explicit(&values, memory_order_acquire);
    if (buf == cst1)                                                               /* materialise */
    {
        int* nb = malloc(sizeof(int) * NELEMS);
        for (int i = 0; i < NELEMS; ++i)
            nb[i] = 0;
        int* exp = cst1;
        if (atomic_compare_exchange_strong_explicit(&values, &exp, nb, memory_order_acq_rel,
                                                    memory_order_acquire))
            buf = nb;
        else
        {
            free(nb);
            buf = exp;
        }
    }
    const int old = elem_load(&buf[idx]);
    elem_store(&buf[idx], v);
    if (v == 0 && old != 0)                         /* an element turned into the constant */
        mark_candidate();
    atomic_store_explicit(&flag[rank], 0, memory_order_release);                   /* end_write */
    return true;
}

/* The write, deferred if the mark is up: replayed once the mark is down (one pass: the second
 * attempt cannot see a mark again). Unrolled rather than a retry loop, which GenMC's spin-loop
 * detection would take for a loop with no progress. */
static void writer_store(int rank, int idx, int v)
{
    if (writer_attempt(rank, idx, v))
        return;
    while (atomic_load_explicit(&mark, memory_order_acquire) != 0)                 /* deferred */
        ;
    const bool done = writer_attempt(rank, idx, v);                                /* replay */
    assert(done);
}

static void* writer_w(void* arg)
{
    (void)arg;
    writer_store(0, NELEMS - 1, 0);
    return NULL;
}

#ifdef WITH_W2
static void* writer_w2(void* arg)
{
    (void)arg;
    writer_store(1, 0, W2_VALUE);
    return NULL;
}
#endif

/* ───────────────────────── the pass (make_block_compression) ───────────────────────── */

static bool any_writer_on_chunk(void)
{
    for (int r = 0; r < NWRITERS; ++r)
        if (atomic_load_explicit(&flag[r], memory_order_acquire) != 0)
            return true;
    return false;
}

static void unmark(void) { atomic_store_explicit(&mark, 0, memory_order_release); }

static void* pass(void* arg)
{
    (void)arg;
    atomic_store_explicit(&mark, 1, memory_order_release);                         /* claim */
    atomic_fetch_add_explicit(&hard_inflight, 1, memory_order_seq_cst);            /* start() */
    atomic_thread_fence(memory_order_seq_cst);                                     /* heavy barrier */
    while (any_writer_on_chunk())
        ;
    atomic_fetch_add_explicit(&stealing_inflight, 1, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);

    /* fill_list_to_compress: the scan */
    int* buf = atomic_load_explicit(&values, memory_order_acquire);
    bool constant = buf != cst1;
    if (constant)
        for (int i = 0; i < NELEMS; ++i)
            if (elem_load(&buf[i]) != 0)
                constant = false;

#if defined(HANDBACK) && !defined(MUT_HANDBACK_AFTER_UNMARK)
    /* every scan done, the marks still up: hand the flags back (1 -> 2), queue_flag::hand_back */
    atomic_exchange_explicit(&queued, 2, memory_order_acq_rel);
#endif

    /* the clear of the constant ones, then their steal_unmark() */
    bool unmarked = false;
    if (constant)
    {
        int* exp = buf;
        if (atomic_compare_exchange_strong_explicit(&values, &exp, cst1, memory_order_acq_rel,
                                                    memory_order_acquire))
            free(buf);
        unmark();
        unmarked = true;
    }

    /* the hand-back loop */
    atomic_fetch_sub_explicit(&listed, 1, memory_order_relaxed);                   /* list cleared */
#if defined(HANDBACK)
    if (!unmarked)
        unmark();
    #ifdef MUT_HANDBACK_AFTER_UNMARK
    atomic_exchange_explicit(&queued, 2, memory_order_acq_rel);
    #endif
    int two = 2;
    if (!atomic_compare_exchange_strong_explicit(&queued, &two, 0, memory_order_acq_rel,
                                                 memory_order_acquire))
    {
        /* 3: written while handed back -- queue it again, in this rank's (new) list */
        atomic_store_explicit(&queued, 1, memory_order_release);
        atomic_fetch_add_explicit(&listed, 1, memory_order_relaxed);
    }
#elif defined(ORDER_FLAG_FIRST)
    atomic_store_explicit(&queued, 0, memory_order_release);
    if (!unmarked)
        unmark();
#else
    if (!unmarked)
        unmark();
    atomic_store_explicit(&queued, 0, memory_order_release);
#endif

    atomic_fetch_sub_explicit(&stealing_inflight, 1, memory_order_seq_cst);        /* finish() */
    atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
    return NULL;
}

/* ───────────────────────── the run ───────────────────────── */

static int run_once(void)
{
    int* buf = malloc(sizeof(int) * NELEMS);
    for (int i = 0; i < NELEMS; ++i)
        buf[i] = 0;
    buf[NELEMS - 1] = 7;                         /* not constant: one element to go */
    atomic_init(&values, buf);
    atomic_init(&mark, 0);
    atomic_init(&queued, 1);                     /* a candidate in the pass's list */
    atomic_init(&listed, 1);
    atomic_init(&hard_inflight, 0);
    atomic_init(&stealing_inflight, 0);
    for (int r = 0; r < NWRITERS; ++r)
        atomic_init(&flag[r], 0);

    pthread_t tp, tw;
#ifdef WITH_W2
    pthread_t tw2;
#endif
    pthread_create(&tp, NULL, pass, NULL);
    pthread_create(&tw, NULL, writer_w, NULL);
#ifdef WITH_W2
    pthread_create(&tw2, NULL, writer_w2, NULL);
#endif
    pthread_join(tp, NULL);
    pthread_join(tw, NULL);
#ifdef WITH_W2
    pthread_join(tw2, NULL);
#endif

    int* end = atomic_load_explicit(&values, memory_order_acquire);
    const int q = atomic_load_explicit(&queued, memory_order_acquire);
    const int l = atomic_load_explicit(&listed, memory_order_relaxed);

    bool constant_content = true;
    for (int i = 0; i < NELEMS; ++i)
        if (end[i] != 0)
            constant_content = false;

    assert(l == 0 || l == 1);                                   /* Q2: never listed twice */
    assert((q == 1) == (l == 1));                               /* Q2: flag up <=> listed */
    assert(!(end != cst1 && constant_content) || l == 1);       /* Q1: no lost candidate */
#ifdef WITH_W2
    assert(end[0] == W2_VALUE);                                 /* P2 */
#endif
    if (end != cst1)
        free(end);
    return 0;
}

#ifdef NATIVE
int main(void)
{
    for (int k = 0; k < 20; ++k)
        run_once();
    return 0;
}
#else
int main(void) { return run_once(); }
#endif
