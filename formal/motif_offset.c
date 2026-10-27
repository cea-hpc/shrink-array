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
 * formal/motif_offset.c -- handshake 4: the readers of a pattern::Array motif chunk against the
 * rewrites of its second-value slot, which holds the chunk's offset (docs/pattern_design.md,
 * section 6; docs/formal_models.md, section 7).
 *
 * A REDUCED MODEL. One chunk of two elements compressed to motif A = {0, 5} with offset 10 in
 * its slot: it holds {10, 15}. NREADERS readers (1 by default) each read element 1 NREADS times
 * (1 by default) through a ReadSection, as element_at(offset, slot) does. One writer
 * materialises the chunk (a buffer filled from the motif and the slot, published by CAS) and
 * writes both elements; one compression pass then compresses the buffer to its TARGET -- the
 * slot rewritten first, then the values pointer switched by CAS -- waits the grace period out
 * and recycles the buffer (a poison store, then the free). The writer and the pass never work
 * on the chunk at once: handshake 1 (writer_compressor.c) is taken as given, modelled by the
 * pass waiting for the writer's mark.
 *
 * The hazard: a reader that loaded the motif pointer A and reads the slot only after the pass
 * rewrote it computes A[1] + the NEW slot value, a value the chunk never held. The grace period
 * does not help: it protects the buffer the pass frees, not the slot it rewrites.
 *
 *   model                         source (chunk/public/chunk/)
 *   ---------------------------   ----------------------------------------------------------
 *   values                        internal/chunk_base.h  ChunkBase::values_ (values_load(): acquire)
 *   slot                          the chunk's second-value slot (internal/tab_cst_val2_holder.h
 *                                 SecondValueRef<T>::slot): the offset of a motif chunk, the
 *                                 second value of a constant one
 *   motif_a, motif_b              internal/motif_region.h  MotifRegion<T> entries (never freed here)
 *   cst1, cst2                    zero_any's two constants: the value comes from the POINTER
 *                                 (second_value_from_ptr()), never from the slot
 *   reader()                      internal/chunk_base_inline.h  element_at(offset, slot) ->
 *                                 element_at_unguarded(offset, slot) -> Chunk::motif_value()
 *   read_enter/leave, wait        internal/read_reclaim.h, as in reader_reclaim.c
 *   writer()                      internal/zero_any_chunk_inline.h  create_value(), the motif
 *                                 branch: fill from motif + slot, CAS against the motif pointer
 *   pass(), TARGET_MOTIF(_SAME)   internal/pattern_chunk_inline.h  block_clear_motif()
 *   pass(), TARGET_CST1           internal/second_value_chunk_base_inline.h block_clear_common():
 *                                 ptr_second_value.reset(), then the CAS
 *   pass(), TARGET_CST2           internal/zero_any_chunk_inline.h  find_or_set_ind_cst_val2()
 *                                 (set_from_array() on the slot), then block_clear()
 *
 * PROPERTIES (asserted):
 *   P1  a reader never dereferences the freed buffer;
 *   P2  a reader returns a value element 1 held at some point: 15 (the motif, then the fresh
 *       buffer) or the value the writer wrote; never motif[1] plus a slot value of another
 *       state (no "torn" read), and the slot is never accessed by a racing non-atomic store;
 *   P3  a reader's retry is bounded (it retries at most once per change of `values`).
 *
 * TARGETS (what the writer leaves, hence what the pass compresses to):
 *   -DTARGET_MOTIF (default)  {20, 27}: another motif, B = {0, 7}, offset 20;
 *   -DTARGET_SAME_MOTIF       {20, 25}: motif A again, offset 20 (the pointer comes back: ABA);
 *   -DTARGET_CST1             {0, 0}: zero_any's first constant, the slot reset to 0;
 *   -DTARGET_CST2             {30, 30}: zero_any's second constant, the slot set to 30.
 *
 * THE SOURCE MODELLED (the fix of F5, docs/formal_models.md section 8.6):
 *   reader: load `values`; a motif: load the slot ACQUIRE, load `values` again; if it moved,
 *           start over (the value is that of the state the two loads bracket);
 *   pass:   every rewrite of the slot of a chunk of a member with the motif state is an atomic
 *           RELEASE store, before the CAS.
 *
 * MUTATIONS -- each must be refused, but where stated:
 *   -DMUT_NO_RECHECK           the reader as stage 2 left it: slot relaxed, no second load of
 *                              `values` (F5, before). Refused for TARGET_MOTIF, CST1, CST2;
 *                              passes for TARGET_SAME_MOTIF (the slot then holds the offset of
 *                              the same motif: the new state, a value the chunk holds).
 *   -DMUT_SLOT_LOAD_RELAXED    the re-check, the slot load relaxed: the second load of `values`
 *                              may still return the motif it rewrote.
 *   -DMUT_SLOT_STORE_RELAXED   the pass's slot store relaxed instead of release.
 *   -DMUT_SLOT_STORE_PLAIN     the pass's slot store non-atomic, as zero_any's reset() and
 *                              set_from_array() are (F5, before, for the two constants): a data
 *                              race with the reader's atomic load.
 *   -DMUT_CAS_BEFORE_SLOT      the pass switches the pointer before it rewrites the slot: the
 *                              order of section 6 is load-bearing.
 *   -DMUT_NO_GRACE             the pass frees the buffer without the grace period (calibration).
 *
 * CONFIGURATIONS: -DNREADERS=n, -DNREADS=k.
 *
 * Build for GenMC:   genmc -- motif_offset.c
 * Build natively:    cc -O2 -pthread -DNATIVE motif_offset.c && ./a.out
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef NREADERS
    #define NREADERS 1
#endif
#ifndef NREADS
    #define NREADS 1
#endif
#define NELEMS   2
#define J        1                                       /* the element the readers read */
#define POISON   77                                      /* a value the chunk never holds */
#define MAXTRY   (NELEMS + 2)                            /* P3: `values` changes twice at most */

#if !defined(TARGET_MOTIF) && !defined(TARGET_SAME_MOTIF) && !defined(TARGET_CST1) && !defined(TARGET_CST2)
    #define TARGET_MOTIF 1
#endif

/* What the writer leaves in the buffer, and what the pass compresses it to. */
#if defined(TARGET_MOTIF)
    #define W0 20
    #define W1 27
#elif defined(TARGET_SAME_MOTIF)
    #define W0 20
    #define W1 25
#elif defined(TARGET_CST1)
    #define W0 0
    #define W1 0
#else /* TARGET_CST2 */
    #define W0 30
    #define W1 30
#endif

/* ───────────────────────── the shared state ───────────────────────── */

static _Atomic(atomic_int*) values;                      /* ChunkBase::values_ */
static atomic_int           slot;                        /* the second-value slot (an atomic_ref in the source) */
#ifdef MUT_SLOT_STORE_PLAIN
static int*                 slot_plain;                  /* the same object, written non-atomically */
#endif
static atomic_int           motif_a[NELEMS];             /* {0, 5} */
static atomic_int           motif_b[NELEMS];             /* {0, 7} */
static atomic_int           cst1[NELEMS];                /* the first constant, 0 */
static atomic_int           cst2[NELEMS];                /* the second constant's registry entry, 30 */
static atomic_int           marked;                      /* the writer's mark: the pass may take the chunk */

typedef struct
{
    atomic_uint_fast64_t seq;                            /* ReaderSlot::seq: odd while reading */
} rslot_t;
static rslot_t rslot[NREADERS];

#ifdef NATIVE
static _Atomic(atomic_int*) freed_buffer;
#endif

static bool is_motif(atomic_int* p) { return p == motif_a || p == motif_b; } /* MotifRegion::contains() */
static bool is_cst(atomic_int* p) { return p == cst1 || p == cst2; }       /* is_cst_val2_ptr_of() and cst1 */

/* ───────────────────────── read sections (read_reclaim.h, as reader_reclaim.c) ───────────────────────── */

static uint_fast64_t read_enter(int r)
{
    const uint_fast64_t s = atomic_load_explicit(&rslot[r].seq, memory_order_relaxed) + 1;
    atomic_store_explicit(&rslot[r].seq, s, memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);           /* light_fence(), symmetric shape */
    return s;
}

static void read_leave(int r, uint_fast64_t s)
{
    atomic_store_explicit(&rslot[r].seq, s + 1, memory_order_release);
}

static void wait_for_readers(void)
{
    atomic_thread_fence(memory_order_seq_cst);           /* heavy_barrier() */
    for (int i = 0; i < NREADERS; ++i)
    {
        const uint_fast64_t v = atomic_load_explicit(&rslot[i].seq, memory_order_acquire);
        if ((v & 1U) == 0U)
        {
            continue;
        }
        while (atomic_load_explicit(&rslot[i].seq, memory_order_acquire) == v) {}
    }
}

/* ───────────────────────── the reader ───────────────────────── */

static int read_element(void)                            /* element_at_unguarded(offset, slot) */
{
    for (int t = 0; t < MAXTRY; ++t)
    {
        atomic_int* const p = atomic_load_explicit(&values, memory_order_acquire);     /* values_load() */
#ifdef NATIVE
        assert(p != atomic_load_explicit(&freed_buffer, memory_order_seq_cst));      /* P1 */
#endif
        if (is_cst(p))
        {
            return atomic_load_explicit(&p[0], memory_order_relaxed);               /* second_value_from_ptr() */
        }
        if (!is_motif(p))
        {
            return atomic_load_explicit(&p[J], memory_order_relaxed);               /* the buffer */
        }
#if defined(MUT_NO_RECHECK)
        const int off = atomic_load_explicit(&slot, memory_order_relaxed);          /* load_element_relaxed(slot) */
        return atomic_load_explicit(&p[J], memory_order_relaxed) + off;             /* motif_value() */
#else
    #ifdef MUT_SLOT_LOAD_RELAXED
        const int off = atomic_load_explicit(&slot, memory_order_relaxed);
    #else
        const int off = atomic_load_explicit(&slot, memory_order_acquire);          /* the offset, then... */
    #endif
        if (atomic_load_explicit(&values, memory_order_relaxed) == p)               /* ...the pointer again */
        {
            return atomic_load_explicit(&p[J], memory_order_relaxed) + off;
        }
        /* `values` moved between the two loads: the slot may be another state's; start over. */
#endif
    }
    assert(0 && "P3: the reader retried more often than `values` can change");
    return -1;
}

static void* reader(void* arg)
{
    const int r = (int)(long)arg;
    for (int k = 0; k < NREADS; ++k)
    {
        const uint_fast64_t s = read_enter(r);
        const int           v = read_element();
        read_leave(r, s);
        assert(v == 15 || v == W1);                      /* P2: a value element 1 held */
    }
    return NULL;
}

/* ───────────────────────── the writer: materialise, then write ───────────────────────── */

static void* writer(void* arg)
{
    (void)arg;
    /* create_value(), the motif branch; wait_before_materialise() is handshake 1 (given) */
    atomic_int* const motif = atomic_load_explicit(&values, memory_order_acquire);
    atomic_int*       buf   = malloc(sizeof(atomic_int) * NELEMS);
    for (int j = 0; j < NELEMS; ++j)
    {
        atomic_init(&buf[j], atomic_load_explicit(&motif[j], memory_order_relaxed) +
                                 atomic_load_explicit(&slot, memory_order_relaxed));
    }
    atomic_int* expected = motif;
    const bool  won = atomic_compare_exchange_strong_explicit(&values, &expected, buf, memory_order_acq_rel,
                                                              memory_order_acquire);
    assert(won);                                         /* the only materialiser of the model */
    /* the write itself, in place, then the mark that queues the chunk */
    atomic_store_explicit(&buf[0], W0, memory_order_relaxed);
    atomic_store_explicit(&buf[1], W1, memory_order_relaxed);
    atomic_store_explicit(&marked, 1, memory_order_release);
    return NULL;
}

/* ───────────────────────── the pass: compress the buffer to its target ───────────────────────── */

static void slot_store(int v)
{
#if defined(MUT_SLOT_STORE_PLAIN)
    *slot_plain = v;                                     /* SecondValueRef::reset() / set_from_array() */
#elif defined(MUT_SLOT_STORE_RELAXED)
    atomic_store_explicit(&slot, v, memory_order_relaxed);
#else
    atomic_store_explicit(&slot, v, memory_order_release);   /* std::atomic_ref<T>(*slot).store(v, release) */
#endif
}

static void* pass(void* arg)
{
    (void)arg;
    while (atomic_load_explicit(&marked, memory_order_acquire) == 0) {}  /* the chunk is queued, claimed */
    atomic_int* const old = atomic_load_explicit(&values, memory_order_acquire);
    const int         e0  = atomic_load_explicit(&old[0], memory_order_relaxed);   /* the scan */
    const int         e1  = atomic_load_explicit(&old[1], memory_order_relaxed);
    atomic_int*       target;
    int               slot_value;
    if (e0 == e1)
    {
        target     = (e0 == 0) ? cst1 : cst2;            /* zero_any's scan: a constant */
        slot_value = (e0 == 0) ? 0 : e0;                 /* reset(), or the second value */
    }
    else
    {
        target     = (e1 - e0 == 5) ? motif_a : motif_b; /* classify_motif(): find_or_admit() */
        slot_value = e0;                                 /* the offset: element 0 */
    }
#ifdef MUT_CAS_BEFORE_SLOT
    atomic_store_explicit(&values, target, memory_order_release);
    slot_store(slot_value);
#else
    slot_store(slot_value);                              /* the slot first (section 6)... */
    atomic_int* expected = old;
    const bool  won = atomic_compare_exchange_strong_explicit(&values, &expected, target, memory_order_acq_rel,
                                                              memory_order_acquire);  /* ...then the pointer */
    assert(won);                                         /* the chunk is claimed: the CAS cannot fail */
#endif
#ifndef MUT_NO_GRACE
    wait_for_readers();                                  /* PassScope::~PassScope() */
#endif
    atomic_store_explicit(&old[J], POISON, memory_order_relaxed);   /* the recycled block's new content */
#ifdef NATIVE
    atomic_store_explicit(&freed_buffer, old, memory_order_seq_cst);
#else
    free(old);
#endif
    return NULL;
}

/* ───────────────────────── main ───────────────────────── */

int main(void)
{
    atomic_init(&motif_a[0], 0);
    atomic_init(&motif_a[1], 5);
    atomic_init(&motif_b[0], 0);
    atomic_init(&motif_b[1], 7);
    atomic_init(&cst1[0], 0);
    atomic_init(&cst1[1], 0);
    atomic_init(&cst2[0], 30);
    atomic_init(&cst2[1], 30);
    atomic_init(&slot, 10);                              /* the offset of the motif chunk */
#ifdef MUT_SLOT_STORE_PLAIN
    slot_plain = (int*)&slot;
#endif
    atomic_init(&values, motif_a);                       /* the chunk holds {10, 15} */
    atomic_init(&marked, 0);
    for (int i = 0; i < NREADERS; ++i)
    {
        atomic_init(&rslot[i].seq, 0);
    }
#ifdef NATIVE
    atomic_init(&freed_buffer, NULL);
#endif

    pthread_t tr[NREADERS], tw, tp;
    for (long i = 0; i < NREADERS; ++i)
    {
        pthread_create(&tr[i], NULL, reader, (void*)i);
    }
    pthread_create(&tw, NULL, writer, NULL);
    pthread_create(&tp, NULL, pass, NULL);
    for (int i = 0; i < NREADERS; ++i)
    {
        pthread_join(tr[i], NULL);
    }
    pthread_join(tw, NULL);
    pthread_join(tp, NULL);
    return 0;
}
