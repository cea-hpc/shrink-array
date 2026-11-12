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
 * writer: does a chunk that BECOMES constant always end in a candidate list? And, with a second
 * pass, does the end of the first one leave the second one's claim alone?
 *
 * A REDUCED MODEL. One chunk of NELEMS elements, materialised, all first_value (0) but the last
 * one; it is a candidate in the list of pass A, rank 0 (its "queued" flag is up). The writer W,
 * rank 1, writes that last element to 0: after it, the chunk is all 0 -- compressible -- and the
 * write is the one that should queue it. With -DWITH_W2, a second writer W2, rank 2, stores a
 * plain value into element 0 (the exclusion check: a write a pass must never lose, nor store into
 * a returned buffer). Handshake 1 is writer_compressor.c's, reduced to the symmetric shape; the
 * names:
 *
 *   model                       source (chunk/public/chunk/internal/)
 *   -------------------------   ---------------------------------------------------------
 *   queued                      proxy_base.h      tab_is_inside_chunk_list_to_compress[t][c]
 *   mark_candidate()            queue_flag.h      try_queue(), from add_to_compression_list()
 *   listed[r]                   the entries of the chunk in rank r's candidate list
 *   run_pass()                  zero_chunk_inline.h, zero_one_value_chunk_inline.h,
 *                               zero_any_chunk_inline.h  make_block_compression():
 *                                 claim             steal_mark() of every candidate
 *                                 start()           compression_manager start(): heavy barrier,
 *                                                   no writer on a marked chunk
 *                                 scan              fill_list_chunk_to_compress_in_par():
 *                                                   fill_list_to_compress, queue_flag::hand_back,
 *                                                   steal_unmark() of a chunk it drops
 *                                 clear             clear_int_st() (or its variant), then
 *                                                   steal_unmark() of the chunk
 *                                 hand-back loop    queue_flag::release_candidates(): for every
 *                                                   candidate steal_unmark() (again for the ones
 *                                                   already released), then queue_flag::release
 *   writer()                    proxy_base_inline.h  write_with_optional_lock_impl(): announce,
 *                               check 2 (must_defer_write = steal_marked), body (materialise,
 *                               store, mark for compression), end_write
 *
 * A deferred write is modelled as a retry once the mark is down: the library replays it later
 * still (after the pass's finish()), so the retry covers more interleavings, not fewer -- and a
 * writer arriving fresh in the same instant takes the direct path anyway.
 *
 * TWO PASSES (-DTWO_PASSES, implies -DWITH_W2): a pass runs on the rank whose list holds the
 * chunk, so when W's write queues it (into rank 1's list) W's thread then runs pass B on it, as
 * the library's in-loop trigger does. This is what the ORDER of the hand-back is for: each
 * candidate gives its mark back BEFORE its flag. While the flag is not free the chunk is in no
 * other list, no other pass can have marked it, and the unmark drops this pass's claim only;
 * flag first, and a writer can queue the chunk, its pass mark it, and this pass's late unmark --
 * the redundant one of release_candidates(), the chunk was released in the scan already -- erase
 * that claim: W2 then writes into a chunk pass B is clearing.
 *
 * PROPERTIES (asserted when every thread is done, except P1):
 *   Q1  a materialised chunk whose content is constant is in a candidate list: the write that
 *       made it constant queued it, or the pass that dropped it queued it again;
 *   Q2  queued == 1 exactly when the chunk is listed, and it is listed once at most;
 *   P1  no store into a buffer a pass has returned (GenMC reports the access to freed memory);
 *   P2  (-DWITH_W2) W2's value is in the chunk.
 *
 * THE SOURCE (default): the flag is HANDED BACK (chunk/internal/queue_flag.h): as soon as a
 * candidate is scanned, its flag goes 1 -> 2 while its mark still keeps the writers out
 * (queue_flag::hand_back); a writer that finds 2 turns it into 3 ("written meanwhile") instead
 * of giving up (queue_flag::try_queue); the pass then gives the mark back and resets 2 -> 0 by
 * CAS, and on a 3 queues the chunk again itself (queue_flag::release). Must PASS, one pass or two.
 *
 * MUTATIONS, each must FAIL:
 *   -DMUT_OLD_HANDBACK           steal_unmark(), then the flag reset to 0 -- the source before
 *                                (F6, docs/formal_models.md section 8.7): between the two a writer
 *                                writes directly (no mark), its CAS 0->1 meets the flag still up
 *                                and gives up, then the reset drops the chunk. Fails Q1.
 *   -DMUT_HANDBACK_AFTER_UNMARK  the flag goes 1 -> 2 after the mark is down: the window is back.
 *                                Fails Q1.
 *   -DMUT_WRITER_IGNORES_2       the writer gives up on a 2 as on a 1: nobody notes the write.
 *                                Fails Q1.
 *   -DMUT_RELEASE_BEFORE_UNMARK  the hand-back loop releases the flag, THEN gives the mark back
 *                                (the order release_candidates() forbids). With -DTWO_PASSES:
 *                                fails P1 or P2 (pass B's claim erased). With one pass there is
 *                                no other claim to erase: it passes, which is why the one-pass
 *                                model could not give the order a verdict.
 *
 * FORCED SCHEDULE (-DNATIVE -DDEMO_SCHEDULE, with -DTWO_PASSES): a native run cannot hit a window
 * a few instructions wide, so this one opens them with sleeps, in the order the erased claim
 * needs -- W replays after pass A's release, pass B scans after A's late unmark, W2 checks the
 * mark between the two and stores after B's clear. The source passes it (W's write is noted on
 * the handed-back flag, pass B never runs); MUT_RELEASE_BEFORE_UNMARK fails P2 in every run (and
 * stores into the freed buffer: -fsanitize=address reports it). An oracle for that one scenario,
 * not a substitute for GenMC.
 *
 * Build for GenMC:   genmc -- [-DTWO_PASSES] [-DMUT_...] [-DWITH_W2] queue_handback.c
 * Build natively:    cc -O2 -pthread -DNATIVE [-DTWO_PASSES] [-DDEMO_SCHEDULE] queue_handback.c && ./a.out
 */

#if defined(NATIVE) && defined(DEMO_SCHEDULE)
    #define _DEFAULT_SOURCE /* usleep() under -std=c11 */
#endif

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#if defined(NATIVE) && defined(DEMO_SCHEDULE)
    #include <unistd.h>
    #define DEMO_SLEEP_MS(ms) usleep((ms) * 1000)
#else
    #define DEMO_SLEEP_MS(ms) ((void)0)
#endif

#define NELEMS 2

#ifdef TWO_PASSES
    #ifndef WITH_W2
        #define WITH_W2 1
    #endif
#endif

#ifndef MUT_OLD_HANDBACK
    #define HANDBACK 1 /* the source */
#endif
#define W2_VALUE 5

/* ───────────────────────── the shared state ───────────────────────── */

static int        cst1[NELEMS];          /* the shared constant buffer (all 0): never written */
static _Atomic(int*) values;             /* chunk_base.h: values */
static atomic_uint   mark;               /* chunk_base.h: steal_mark_ */

#define NRANKS 3                         /* 0: pass A; 1: W (and pass B); 2: W2 */
static atomic_int flag[NRANKS];         /* WritingFlag::v: 1 = writing THE chunk (the one chunk), 0 = null */
static atomic_int hard_inflight, stealing_inflight;

static atomic_int queued;                /* 0 free, 1 queued; 2 handed back, 3 written meanwhile */
static atomic_int listed[NRANKS];        /* entries of the chunk in rank r's candidate list */

/* ───────────────────────── elements ───────────────────────── */

static void elem_store(int* p, int v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
static int  elem_load(const int* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

/* ───────────────────────── marking (queue_flag::try_queue) ───────────────────────── */

/* The writer's side of the queue flag; true if it pushed an entry into rank's list. */
static bool mark_candidate(int rank)
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
            atomic_fetch_add_explicit(&listed[rank], 1, memory_order_relaxed);
            return true;
        }
    }
#else
    int expected = 0;
    if (atomic_load_explicit(&queued, memory_order_relaxed) == 0 &&
        atomic_compare_exchange_strong_explicit(&queued, &expected, 1, memory_order_acq_rel,
                                                memory_order_acquire))
    {
        atomic_fetch_add_explicit(&listed[rank], 1, memory_order_relaxed);
        return true;
    }
    return false;                                                /* flag already up: refused */
#endif
}

/* ───────────────────────── the writer (handshake 1, symmetric shape) ───────────────────────── */

/* One attempt: 0 if check 2 saw a pass's mark (the write is then deferred), 1 if it wrote, 2 if
 * it wrote and queued the chunk into its rank's list. */
static int writer_attempt(int rank, int idx, int v)
{
    atomic_store_explicit(&flag[rank], 1, memory_order_seq_cst);                    /* announce */
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(&mark, memory_order_acquire) != 0)                    /* check 2 */
    {
        atomic_store_explicit(&flag[rank], 0, memory_order_release);               /* end_write */
        return 0;
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
    if (rank == 2)
        DEMO_SLEEP_MS(150);                         /* DEMO: W2 stores after pass B's clear */
    const int old = elem_load(&buf[idx]);
    elem_store(&buf[idx], v);
    bool pushed = false;
    if (v == 0 && old != 0)                         /* an element turned into the constant */
        pushed = mark_candidate(rank);
    atomic_store_explicit(&flag[rank], 0, memory_order_release);                   /* end_write */
    return pushed ? 2 : 1;
}

/* The write, deferred while a mark is up: replayed once it is down. Unrolled rather than a retry
 * loop, which GenMC's spin-loop detection would take for a loop with no progress. One retry per
 * pass that can hold a mark: one, or two with -DTWO_PASSES. Returns whether it queued the chunk. */
static bool writer_store(int rank, int idx, int v)
{
    int r = writer_attempt(rank, idx, v);
    if (r != 0)
        return r == 2;
    while (atomic_load_explicit(&mark, memory_order_acquire) != 0)                 /* deferred */
        ;
    DEMO_SLEEP_MS(10);                                 /* DEMO: replay after pass A's release */
    r = writer_attempt(rank, idx, v);                                              /* replay */
#ifdef TWO_PASSES
    if (r == 0)
    {
        while (atomic_load_explicit(&mark, memory_order_acquire) != 0)
            ;
        r = writer_attempt(rank, idx, v);
    }
#endif
    assert(r != 0);
    return r == 2;
}

/* ───────────────────────── a pass (make_block_compression) ───────────────────────── */

static bool any_writer_on_chunk(void)
{
    for (int r = 0; r < NRANKS; ++r)
        if (atomic_load_explicit(&flag[r], memory_order_acquire) != 0)
            return true;
    return false;
}

static void unmark(void) { atomic_store_explicit(&mark, 0, memory_order_release); }

/* queue_flag::release (or the old reset): the end of the hand-back of the one candidate. */
static void release_flag(int rank)
{
#ifdef HANDBACK
    int two = 2;
    if (!atomic_compare_exchange_strong_explicit(&queued, &two, 0, memory_order_acq_rel,
                                                 memory_order_acquire))
    {
        /* 3: written while handed back -- queue it again, in this rank's (new) list */
        atomic_store_explicit(&queued, 1, memory_order_release);
        atomic_fetch_add_explicit(&listed[rank], 1, memory_order_relaxed);
    }
#else
    (void)rank;
    atomic_store_explicit(&queued, 0, memory_order_release);
#endif
}

/* One pass over rank's list, which holds the chunk. */
static void run_pass(int rank)
{
    atomic_store_explicit(&mark, 1, memory_order_release);                         /* claim */
    atomic_fetch_add_explicit(&hard_inflight, 1, memory_order_seq_cst);            /* start() */
    atomic_thread_fence(memory_order_seq_cst);                                     /* heavy barrier */
    while (any_writer_on_chunk())
        ;
    atomic_fetch_add_explicit(&stealing_inflight, 1, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
    if (rank == 1)
        DEMO_SLEEP_MS(100);                     /* DEMO: pass B scans after A's late unmark */

    /* fill_list_to_compress: the scan */
    int* buf = atomic_load_explicit(&values, memory_order_acquire);
    bool constant = buf != cst1;
    if (constant)
        for (int i = 0; i < NELEMS; ++i)
            if (elem_load(&buf[i]) != 0)
                constant = false;

#if defined(HANDBACK) && !defined(MUT_HANDBACK_AFTER_UNMARK)
    /* scanned, the mark still up: hand the flag back (1 -> 2), queue_flag::hand_back */
    atomic_exchange_explicit(&queued, 2, memory_order_acq_rel);
#endif

    if (constant)
    {
        /* the clear (clear_int_st), then the chunk's steal_unmark() */
        int* exp = buf;
        if (atomic_compare_exchange_strong_explicit(&values, &exp, cst1, memory_order_acq_rel,
                                                    memory_order_acquire))
            free(buf);
        unmark();
    }
    else
    {
        /* dropped by the scan: steal_unmark() there, the writers may have it back */
        unmark();
    }

    /* the hand-back loop: release_candidates() */
    atomic_fetch_sub_explicit(&listed[rank], 1, memory_order_relaxed);             /* list cleared */
#ifdef MUT_HANDBACK_AFTER_UNMARK
    atomic_exchange_explicit(&queued, 2, memory_order_acq_rel);
#endif
#ifdef MUT_RELEASE_BEFORE_UNMARK
    release_flag(rank);
    DEMO_SLEEP_MS(50);
    unmark();                        /* the redundant unmark, late: may hit another pass's claim */
#else
    unmark();                        /* every candidate, the released ones again: idempotent here */
    DEMO_SLEEP_MS(50);
    release_flag(rank);
#endif

    atomic_fetch_sub_explicit(&stealing_inflight, 1, memory_order_seq_cst);        /* finish() */
    atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
}

/* ───────────────────────── the threads ───────────────────────── */

static void* pass_a(void* arg)
{
    (void)arg;
    run_pass(0);
    return NULL;
}

static void* writer_w(void* arg)
{
    (void)arg;
    DEMO_SLEEP_MS(5);                                  /* DEMO: first attempt under A's mark */
    const bool pushed = writer_store(1, NELEMS - 1, 0);
#ifdef TWO_PASSES
    if (pushed)
        run_pass(1);                 /* the chunk is in this rank's list: its pass, pass B */
#else
    (void)pushed;
#endif
    return NULL;
}

#ifdef WITH_W2
static void* writer_w2(void* arg)
{
    (void)arg;
    DEMO_SLEEP_MS(70);                       /* DEMO: check 2 after A's late unmark, before B's scan */
    (void)writer_store(2, 0, W2_VALUE);
    return NULL;
}
#endif

/* ───────────────────────── the run ───────────────────────── */

static int run_once(void)
{
    int* buf = malloc(sizeof(int) * NELEMS);
    for (int i = 0; i < NELEMS; ++i)
        buf[i] = 0;
    buf[NELEMS - 1] = 7;                         /* not constant: one element to go */
    atomic_init(&values, buf);
    atomic_init(&mark, 0);
    atomic_init(&queued, 1);                     /* a candidate in pass A's list */
    atomic_init(&hard_inflight, 0);
    atomic_init(&stealing_inflight, 0);
    for (int r = 0; r < NRANKS; ++r)
    {
        atomic_init(&flag[r], 0);
        atomic_init(&listed[r], 0);
    }
    atomic_init(&listed[0], 1);

    pthread_t tp, tw;
#ifdef WITH_W2
    pthread_t tw2;
#endif
    pthread_create(&tp, NULL, pass_a, NULL);
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
    int l = 0;
    for (int r = 0; r < NRANKS; ++r)
        l += atomic_load_explicit(&listed[r], memory_order_relaxed);

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
