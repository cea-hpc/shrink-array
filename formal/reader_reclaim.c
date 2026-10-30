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
 * formal/reader_reclaim.c -- handshake 3 of docs/synchronisation_dekker.md: pure readers
 * against the buffer reclamation of a compression pass run inside a region (section 5.5,
 * chunk/public/chunk/internal/read_reclaim.h).
 *
 * A REDUCED MODEL. One chunk holding an all-zero buffer; NREADERS readers (2 by default) each
 * doing NREADS pure reads (1 by default; 2 exercises the "wait for the sequence to MOVE, not to
 * be even" rule); one pass that steals the buffer (CAS to the constant), collects it, waits the
 * grace period out, then recycles the block -- modelled as a poison store, the value another
 * chunk would put there -- and frees it.
 *
 *   model                         source (chunk/public/chunk/internal/)
 *   ---------------------------   -------------------------------------------------------
 *   slot[r].seq                   read_reclaim.h  ReaderSlot::seq (odd while a read is in flight)
 *   overflow                      read_reclaim.h  g_overflow_readers (readers past slot_count)
 *   read_enter() / read_leave()   read_reclaim.h  ReadSection ctor / dtor
 *   light_fence()                 read_reclaim.h  light_fence(): the reader's side of the Dekker
 *   wait_for_readers()            read_reclaim.h  wait_for_readers(): heavy barrier, then the slots
 *   steal(), recycle              zero_any_chunk.h   try_mark_cst_and_get_old(); PassScope::~PassScope
 *   reader()                      chunk_base.h    element_at(): values_load() then the element load
 *
 * PROPERTIES (asserted):
 *   P1  a reader never dereferences a buffer the pass has freed (GenMC: access to freed memory,
 *       or a race with the free);
 *   P2  a reader never reads a value the chunk never held (the poison of a recycled block);
 *   P3  (--check-liveness) the grace period terminates.
 *
 * SHAPES (section 4): -DSHAPE_SYMMETRIC (default): the reader's light fence is a seq_cst fence,
 * what the checker can verify. -DSHAPE_ASYMMETRIC_NAKED: nothing on the reader's side (the
 * membarrier() shape without the membarrier): must FAIL.
 *
 * CONFIGURATIONS:
 *   -DNREADERS=n, -DNREADS=k   readers and reads per reader (default 2 and 1);
 *   -DOVERFLOW_READER          reader 0 takes the overflow path (no slot: the shared seq_cst
 *                              counter, as a program with more reader threads than
 *                              CHUNK_READ_RECLAIM_SLOTS).
 *
 * MUTATIONS -- each must be refused:
 *   -DMUT_NO_GRACE             the pass frees without waiting for the readers: the hazard itself.
 *   -DMUT_NO_HEAVY_BARRIER     the pass reads the slots without its barrier.
 *   -DMUT_LEAVE_RELAXED        the reader's even store relaxed instead of release (the document:
 *                              release "orders the element load before it, which is what lets the
 *                              reclaimer trust an even value").
 *   -DMUT_OVERFLOW_NO_FENCE    OVERFLOW_READER: the overflow announce without its light fence, as
 *                              the library was before the model (docs/formal_models.md, F4): a
 *                              seq_cst RMW followed by an acquire load is not a Dekker announce in
 *                              C11 (it is on the AArch64 and x86 mappings).
 *
 *   -DMUT_ENTER_RELAXED        the odd entry store relaxed, as the library was before the model
 *                              (docs/formal_models.md, F3). With NREADS=2 it FAILS: the pass,
 *                              having seen a slot odd, waits for it to MOVE, and the value it then
 *                              reads may be the next read's odd entry; in C++20 a release store
 *                              followed by a relaxed store of the same thread is no release
 *                              sequence, so nothing orders the previous read's element load before
 *                              the free (the AArch64 and x86 mappings do; the standard does not).
 *                              The entry store is release now: measured on Grace at +1.9 % on the
 *                              worst read case at 1 thread, within the spread, nothing elsewhere.
 *   -DMUT_WAIT_FOR_EVEN        the other way out of F3, on the pass's side: wait for an odd slot
 *                              to become EVEN. Passes (every even value is a release store), but a
 *                              reader doing back-to-back reads is odd most of the time: the stall
 *                              the design avoided. Not taken.
 *
 * Build for GenMC:   genmc [--check-liveness] -- reader_reclaim.c
 * Build natively:    cc -O2 -pthread -DNATIVE reader_reclaim.c && ./a.out
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef NREADERS
    #define NREADERS 2
#endif
#ifndef NREADS
    #define NREADS 1
#endif
#define NELEMS 1
#define POISON 77                                       /* a value the chunk never holds */

#if !defined(SHAPE_SYMMETRIC) && !defined(SHAPE_ASYMMETRIC_NAKED)
    #define SHAPE_SYMMETRIC 1
#endif

/* ───────────────────────── the shared state ───────────────────────── */

static _Atomic(atomic_int*) values;                     /* the chunk's values: cst1 or the buffer */
static atomic_int           cst1[NELEMS];               /* the shared constant, all zero */

typedef struct
{
    atomic_uint_fast64_t seq;                           /* ReaderSlot::seq: odd while reading */
} slot_t;
static slot_t     slot[NREADERS];
static atomic_long overflow;                            /* g_overflow_readers */

#ifdef NATIVE
static _Atomic(atomic_int*) freed_buffer;
#endif

/* ───────────────────────── the reader's side ───────────────────────── */

static void light_fence(void)                           /* read_reclaim.h light_fence() */
{
#ifdef SHAPE_ASYMMETRIC_NAKED
    /* the membarrier shape: a compiler barrier only (left out: GenMC would read it as a fence) */
#else
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

static bool uses_overflow(int r)
{
#ifdef OVERFLOW_READER
    return r == 0;
#else
    (void)r;
    return false;
#endif
}

static uint_fast64_t read_enter(int r)                  /* ReadSection::ReadSection() */
{
    if (uses_overflow(r))
    {
        atomic_fetch_add_explicit(&overflow, 1, memory_order_seq_cst);
#ifndef MUT_OVERFLOW_NO_FENCE
        light_fence();                                  /* F4: the announce needs its fence, a seq_cst RMW is not one */
#endif
        return 0;
    }
    const uint_fast64_t s = atomic_load_explicit(&slot[r].seq, memory_order_relaxed) + 1; /* odd */
#ifdef MUT_ENTER_RELAXED
    atomic_store_explicit(&slot[r].seq, s, memory_order_relaxed);  /* before F3 */
#else
    atomic_store_explicit(&slot[r].seq, s, memory_order_release);  /* F3: the previous read's loads are ordered before it */
#endif
    light_fence();
    return s;
}

static void read_leave(int r, uint_fast64_t s)          /* ReadSection::~ReadSection() */
{
    if (uses_overflow(r))
    {
        atomic_fetch_sub_explicit(&overflow, 1, memory_order_release);
        return;
    }
#ifdef MUT_LEAVE_RELAXED
    atomic_store_explicit(&slot[r].seq, s + 1, memory_order_relaxed);
#else
    atomic_store_explicit(&slot[r].seq, s + 1, memory_order_release);  /* even: the element load is done */
#endif
}

static void* reader(void* arg)                          /* element_at(), NREADS times */
{
    const int r = (int)(long)arg;
    for (int k = 0; k < NREADS; ++k)
    {
        const uint_fast64_t s    = read_enter(r);
        atomic_int* const   vals = atomic_load_explicit(&values, memory_order_acquire); /* values_load() */
#ifdef NATIVE
        assert(vals != atomic_load_explicit(&freed_buffer, memory_order_seq_cst));     /* P1 */
#endif
        const int v = atomic_load_explicit(&vals[0], memory_order_relaxed);            /* load_element_relaxed() */
        read_leave(r, s);
        assert(v == 0);                                 /* P2: never the poison of a recycled block */
    }
    return NULL;
}

/* ───────────────────────── the pass's side ───────────────────────── */

static void heavy_barrier(void)
{
#ifndef MUT_NO_HEAVY_BARRIER
    atomic_thread_fence(memory_order_seq_cst);          /* + membarrier() in the asymmetric shape */
#endif
}

static void wait_for_readers(void)                      /* read_reclaim.h wait_for_readers() */
{
    heavy_barrier();
    for (int i = 0; i < NREADERS; ++i)
    {
        const uint_fast64_t v = atomic_load_explicit(&slot[i].seq, memory_order_acquire);
        if ((v & 1U) == 0U)
        {
            continue;                                   /* even: finished before, or will see the constant */
        }
#ifdef MUT_WAIT_FOR_EVEN
        while ((atomic_load_explicit(&slot[i].seq, memory_order_acquire) & 1U) != 0U) {}
#else
        while (atomic_load_explicit(&slot[i].seq, memory_order_acquire) == v) {}   /* until it MOVES */
#endif
    }
    while (atomic_load_explicit(&overflow, memory_order_acquire) != 0) {}
}

static atomic_int* steal(void)                          /* try_mark_cst_and_get_old(): CAS to the constant */
{
    atomic_int* old = atomic_load_explicit(&values, memory_order_acquire);
    if (old == cst1)
    {
        return NULL;
    }
    atomic_int* expected = old;
    if (!atomic_compare_exchange_strong_explicit(&values, &expected, cst1,
                                                 memory_order_acq_rel, memory_order_acquire))
    {
        return NULL;
    }
    return old;
}

static void* pass(void* arg)                            /* a pass inside a region, under PassScope */
{
    (void)arg;
    atomic_int* old = steal();
    if (old != NULL)
    {
        /* PassScope::~PassScope(): the grace period, then the release of the batch */
#ifndef MUT_NO_GRACE
        wait_for_readers();
#endif
        /* release to the reuse list: another chunk may take the block at once and fill it */
        atomic_store_explicit(&old[0], POISON, memory_order_relaxed);   /* the recycled block's new content */
#ifdef NATIVE
        atomic_store_explicit(&freed_buffer, old, memory_order_seq_cst);
#else
        free(old);                                      /* GenMC: a later or racing access is reported (P1) */
#endif
    }
    return NULL;
}

/* ───────────────────────── main ───────────────────────── */

int main(void)
{
    atomic_init(&cst1[0], 0);
    atomic_int* b = malloc(sizeof(atomic_int) * NELEMS);
    atomic_init(&b[0], 0);
    atomic_init(&values, b);
    for (int i = 0; i < NREADERS; ++i)
    {
        atomic_init(&slot[i].seq, 0);
    }
    atomic_init(&overflow, 0);
#ifdef NATIVE
    atomic_init(&freed_buffer, NULL);
#endif

    pthread_t tr[NREADERS], tp;
    for (long i = 0; i < NREADERS; ++i)
    {
        pthread_create(&tr[i], NULL, reader, (void*)i);
    }
    pthread_create(&tp, NULL, pass, NULL);
    for (int i = 0; i < NREADERS; ++i)
    {
        pthread_join(tr[i], NULL);
    }
    pthread_join(tp, NULL);

    assert(atomic_load_explicit(&values, memory_order_seq_cst) == cst1);
    assert(atomic_load_explicit(&overflow, memory_order_seq_cst) == 0);
    for (int i = 0; i < NREADERS; ++i)
    {
        assert((atomic_load_explicit(&slot[i].seq, memory_order_seq_cst) & 1U) == 0U);
    }
    return 0;
}
