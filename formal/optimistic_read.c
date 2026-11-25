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
 * formal/optimistic_read.c -- EXPERIMENT exp/optimistic-read, CHUNK_READ_SECTION_MODE=7: pure
 * readers that announce nothing, against the buffer reclamation of a pass run inside a region.
 * The optimistic counterpart of reader_reclaim.c (same reduced shape, same poison).
 *
 * A REDUCED MODEL. One chunk holding an all-zero buffer of NELEMS elements; NREADERS readers
 * (1 by default) each doing NREADS optimistic reads; one pass that steals the buffer (CAS to the
 * constant), bumps the epoch, and releases the block at once -- modelled as poison stores, what
 * the next materialisation puts there. The block is NOT freed: the value pools never unmap
 * (n_max_block_empty = -1, MADV_DONTNEED keeps the mapping), which is the premise of mode 7.
 *
 *   model                       source (chunk/public/chunk/internal/)
 *   -------------------------   ---------------------------------------------------------
 *   epoch                       read_reclaim.h  g_reclaim_epoch
 *   read_enter()                read_reclaim_inline.h  ReadSection ctor, mode 7 (acquire)
 *   recheck()                   read_reclaim_inline.h  retry_after() / retry() (+ fold())
 *   pass()                      read_reclaim.h  wait_for_readers()'s fetch_add, PassScope release
 *   reader()                    chunk_base_inline.h  element_at() / read_block(): the retry loop
 *
 * PROPERTY (asserted): a read the re-check ACCEPTS never returns the poison of a recycled block.
 *
 * SHAPES -- how the re-check is ordered after the element load(s):
 *   -DSHAPE_DEP (default)  the library's: the epoch's address depends on the value (one element)
 *                          or on the OR of the values (a block, NELEMS=2). An ordering the C11
 *                          model does not give (dependencies order nothing in RC11): checked
 *                          under IMM, which keeps address dependencies as AArch64 does; under
 *                          RC11 this shape is EXPECTED to fail -- the calibration that the
 *                          argument is the hardware's, not the standard's.
 *   -DSHAPE_FENCE          an acquire fence before a relaxed re-check (mode 7 before 95b417ae4
 *                          for the blocks): ordered in C11 itself, must pass under RC11 too.
 *
 * CONFIGURATIONS: -DNREADERS=n, -DNREADS=k, -DNELEMS=1|2 (2: the block read and its fold).
 *   -DPASS_RELEASE_FENCE   a release fence between the epoch bump and the release of the block
 *                          (the seqlock writer's fence): what IMM and C11 need, ARMv8 does not.
 *   -DRECYCLER_THREAD      the block is reused by ANOTHER thread, handed over through a flag
 *                          (release / acquire), as a materialisation on another rank would.
 *   -DENTRY_DEP            element_at's entry with CHUNK_READ_DEP_ORDER (ReadSection(DepEntry)):
 *                          the epoch loaded RELAXED and `values` loaded relaxed at an address
 *                          that depends on it (dep()), instead of two acquires.
 *
 * MUTATIONS -- each must be refused:
 *   -DMUT_NO_RECHECK           mode 3 without its retry: accept whatever was read.
 *   -DMUT_RECHECK_UNORDERED    a relaxed re-check with no dependency and no fence: the CPU may
 *                              take it before the element load.
 *   -DMUT_ENTER_RELAXED        the entry load of the epoch relaxed: the reader may see the bumped
 *                              epoch and still load the old pointer (no synchronisation with the
 *                              pass), then the poison, and accept it.
 *   -DMUT_BUMP_AFTER_RELEASE   the pass recycles the block before it bumps the epoch.
 *   -DMUT_ENTRY_NO_DEP         ENTRY_DEP with the values load relaxed but NOT depending on the
 *                              epoch: the old pointer may be read before the bumped epoch.
 *   -DMUT_FOLD_FIRST_ONLY      NELEMS=2: the block's dependency on element 0 only, as if fold()
 *                              were skipped for the others.
 *
 * VERDICTS (GenMC 0.20, 2026-11-25; formal/run_genmc.sh):
 *   ARMv8 (--arm8) and TSO (--tso): the source shape is correct, the block reused by the pass's
 *     own thread or by another one; every mutation refused under ARMv8.
 *   RC11 and IMM: the source shape FAILS -- the reader accepts the poison. Mode 7 is a hardware
 *     argument (AArch64, x86), not a C++ one, like rcu_dereference(): it needs the compiler to keep
 *     the dependency (the asm barrier of retry_after()). The C11 seqlock -- an acquire fence before
 *     the re-check (SHAPE_FENCE) AND a release fence after the bump (PASS_RELEASE_FENCE) -- passes
 *     RC11 and IMM only when the pass's own thread reuses the block; reused by another thread, its
 *     stores need a release as well (one per materialisation), and the reader's dmb ishld per
 *     element is what made mode 2 +50 % on Grace.
 *
 * Build for GenMC:   genmc --arm8 -- optimistic_read.c       (SHAPE_DEP)
 *                    genmc -- -DSHAPE_FENCE optimistic_read.c
 * Build natively:    cc -O2 -pthread -DNATIVE optimistic_read.c && ./a.out
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
#ifndef NELEMS
    #define NELEMS 1
#endif
#define POISON 77 /* a value the chunk never holds */
/* One pass bumps the epoch once: a read retries at most once. One more for the margin. */
#define MAX_TRIES 3

#if !defined(SHAPE_DEP) && !defined(SHAPE_FENCE)
    #define SHAPE_DEP 1
#endif

/* ───────────────────────── the shared state ───────────────────────── */

static _Atomic(atomic_int*) values;     /* the chunk's values: cst1 or the buffer */
static atomic_int           cst1[NELEMS]; /* the shared constant, all zero */
static atomic_uint_fast64_t epoch[1];   /* g_reclaim_epoch (an array: indexed by the dependency) */
#ifdef RECYCLER_THREAD
static _Atomic(atomic_int*) handed; /* the reuse list, reduced to one slot */
#endif

/* ───────────────────────── the reader's side ───────────────────────── */

#if (defined(SHAPE_DEP) && !defined(MUT_RECHECK_UNORDERED)) || defined(ENTRY_DEP)
/* The value's bits through a call the optimiser cannot see into: zero at run time, but an
 * address computed from it depends on the value (the asm trick of retry_after()). */
    #if defined(__clang__)
__attribute__((noinline, optnone))
    #else
__attribute__((noinline, noipa))
    #endif
static uintptr_t opaque(uintptr_t x)
{
    return x;
}
#endif

static uint_fast64_t read_enter(void) /* ReadSection::ReadSection(), mode 7 */
{
#if defined(MUT_ENTER_RELAXED) || defined(ENTRY_DEP)
    return atomic_load_explicit(&epoch[0], memory_order_relaxed);
#else
    return atomic_load_explicit(&epoch[0], memory_order_acquire);
#endif
}

/* retry_after() / retry(): true when a grace period began since the entry. */
static bool moved(uint_fast64_t seen, uintptr_t acc)
{
#if defined(MUT_RECHECK_UNORDERED)
    (void)acc;
    return atomic_load_explicit(&epoch[0], memory_order_relaxed) != seen;
#elif defined(SHAPE_FENCE)
    (void)acc;
    atomic_thread_fence(memory_order_acquire);
    return atomic_load_explicit(&epoch[0], memory_order_relaxed) != seen;
#else
    const uintptr_t zero = opaque(acc) ^ acc; /* 0, depending on every folded value */
    return atomic_load_explicit(&epoch[zero], memory_order_relaxed) != seen;
#endif
}

#if defined(ENTRY_DEP)
static uintptr_t dep_of(uint_fast64_t seen) /* ReadSection::dep(): 0, depending on the epoch */
{
    return (uintptr_t)opaque((uintptr_t)seen) ^ (uintptr_t)seen;
}
#endif

static void* reader(void* arg) /* element_at() / read_block(), NREADS times */
{
    (void)arg;
    for (int k = 0; k < NREADS; ++k)
    {
        uint_fast64_t seen     = read_enter();
        bool          accepted = false;
        int           got[NELEMS];
        for (int t = 0; t < MAX_TRIES && !accepted; ++t)
        {
#if defined(ENTRY_DEP)
            /* values_load_for_read(dep): relaxed, at an address depending on the epoch */
    #ifdef MUT_ENTRY_NO_DEP
            _Atomic(atomic_int*)* const slot = &values; /* relaxed, NOT depending on the epoch */
    #else
            _Atomic(atomic_int*)* const slot = (_Atomic(atomic_int*)*)((uintptr_t)&values + dep_of(seen));
    #endif
            atomic_int* const vals = atomic_load_explicit(slot, memory_order_relaxed);
#else
            atomic_int* const vals = atomic_load_explicit(&values, memory_order_acquire); /* values_load() */
#endif
            uintptr_t         acc  = 0;
            for (int j = 0; j < NELEMS; ++j)
            {
                got[j] = atomic_load_explicit(&vals[j], memory_order_relaxed); /* the plain vals[i] */
#ifdef MUT_FOLD_FIRST_ONLY
                if (j == 0)
#endif
                {
                    acc |= (uintptr_t)(unsigned)got[j]; /* fold() */
                }
            }
#ifdef MUT_NO_RECHECK
            (void)seen;
            (void)acc;
            accepted = true;
#else
            if (!moved(seen, acc))
            {
                accepted = true;
            }
            else
            {
                seen = atomic_load_explicit(&epoch[0], memory_order_acquire); /* note_retry() */
            }
#endif
        }
        assert(accepted); /* one pass: one retry at most */
        for (int j = 0; j < NELEMS; ++j)
        {
            assert(got[j] == 0); /* PROPERTY: never the poison of a recycled block */
        }
    }
    return NULL;
}

/* ───────────────────────── the pass's side ───────────────────────── */

static atomic_int* steal(void) /* try_mark_cst_and_get_old(): CAS to the constant */
{
    atomic_int* old = atomic_load_explicit(&values, memory_order_acquire);
    if (old == cst1)
    {
        return NULL;
    }
    atomic_int* expected = old;
    if (!atomic_compare_exchange_strong_explicit(&values, &expected, cst1, memory_order_acq_rel,
                                                 memory_order_acquire))
    {
        return NULL;
    }
    return old;
}

static void recycle(atomic_int* old) /* release to the reuse list; the next materialisation fills it */
{
    for (int j = 0; j < NELEMS; ++j)
    {
        atomic_store_explicit(&old[j], POISON, memory_order_relaxed);
    }
}

static void* pass(void* arg) /* a pass inside a region, under PassScope */
{
    (void)arg;
    atomic_int* old = steal();
    if (old != NULL)
    {
#ifdef MUT_BUMP_AFTER_RELEASE
        recycle(old);
        atomic_fetch_add_explicit(&epoch[0], 1, memory_order_seq_cst);
#else
        atomic_fetch_add_explicit(&epoch[0], 1, memory_order_seq_cst); /* wait_for_readers(): first */
    #ifdef PASS_RELEASE_FENCE
        atomic_thread_fence(memory_order_release); /* the bump before any reuse, for every observer */
    #endif
    #ifdef RECYCLER_THREAD
        atomic_store_explicit(&handed, old, memory_order_release); /* to the reuse list */
    #else
        recycle(old); /* then the release, at once */
    #endif
#endif
    }
    return NULL;
}

#ifdef RECYCLER_THREAD
static void* recycler(void* arg) /* a materialisation on another rank: pops the block, fills it */
{
    (void)arg;
    atomic_int* blk = atomic_load_explicit(&handed, memory_order_acquire);
    if (blk != NULL)
    {
        recycle(blk);
    }
    return NULL;
}
#endif

/* ───────────────────────── main ───────────────────────── */

int main(void)
{
    atomic_int* b = malloc(sizeof(atomic_int) * NELEMS);
    for (int j = 0; j < NELEMS; ++j)
    {
        atomic_init(&cst1[j], 0);
        atomic_init(&b[j], 0);
    }
    atomic_init(&values, b);
    atomic_init(&epoch[0], 0);
#ifdef RECYCLER_THREAD
    atomic_init(&handed, NULL);
    pthread_t tq;
    pthread_create(&tq, NULL, recycler, NULL);
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
#ifdef RECYCLER_THREAD
    pthread_join(tq, NULL);
#endif

    assert(atomic_load_explicit(&values, memory_order_seq_cst) == cst1);
    assert(atomic_load_explicit(&epoch[0], memory_order_seq_cst) == 1);
    free(b); /* after every thread: never during a read (the pools never unmap) */
    return 0;
}
