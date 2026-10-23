/*
 * Copyright 2026 CEA (www.cea.fr)
 * SPDX-License-Identifier: Apache-2.0
 *
 * switch_mt.c -- reduced GenMC model: a writer switching a compressed zero_any::Chunk chunk to
 * another constant in place (zero_any::Chunk::fill_constant_st, compressed branch, called by
 * internal::store_range_mt for a uniform span over the whole chunk) against a compression
 * pass that holds the same chunk in its candidate list (zero_any::Chunk::make_block_compression).
 *
 * The chunk is reduced to two words:
 *   values   the pointer: CST1 (the first-constant block), CST2A / CST2B (two second-value
 *            slots: identifiers in the registry region, NOT buffers) or BUF (a buffer);
 *   meta     the per-chunk metadata the library writes next to it, the type table and the
 *            second-value slot folded into one word: M1, M2A, M2B (a plain variable, as the
 *            library's `char` and slot: GenMC checks races on it).
 * and its steal mark. The writer's flag says which chunk it is writing.
 *
 * Writer (store_range_mt, zero_any::Chunk, a uniform span covering the chunk):
 *   announce_write(chunk)             -- flag = X; full barrier (symmetric shape)
 *   if (!must_defer_write(chunk)       -- the mark, acquire
 *       && !materialised(chunk))       -- values != BUF
 *       fill_constant_st(v)            -- meta = new, THEN values = new (release store)
 *   end_write()                        -- flag = null, release
 *
 * Compressor (make_block_compression with a proxy), as the source runs it:
 *   steal_mark(X)                      -- release
 *   start(): announce + full barrier, then wait until no writer flag names a marked chunk
 *   fill_list_to_compress(X):
 *     is_not_cst_val1()   -- load 1; a chunk at the first constant is skipped
 *     is_cst_val2()       -- load 2; a chunk at a second value is skipped
 *     scan                -- load 3: the chunk goes on the compact1 list (all first_value) or
 *                            the compact2 list (all one other value)
 *   clear_int_st -> block_clear_common (zero_any::Chunk):
 *     meta = the target's                -- BEFORE the CAS, as the source orders it
 *     old = try_mark_cst_and_get_old():  -- a chunk already at a constant gives nullptr, else
 *                                           CAS(values: old -> the constant), acq_rel
 *     if (!old) values = init_values_cst_val(meta)   -- the "critical re-sync"
 *     else free(old)
 *   steal_unmark(X)
 *
 * Properties, checked after the joins:
 *   P1  the switch the writer made is what the chunk holds (the pass did not overwrite it
 *       with a stale decision, e.g. through the re-sync);
 *   P2  values and meta agree (the type table describes the pointer installed);
 *   P3  the pass frees only a buffer, and a buffer installed is not lost (-DSECOND_MATERIALISER).
 *
 * The heavy barrier of the asymmetric shape (membarrier) is modelled by its symmetric
 * equivalent, a seq_cst fence on both sides, as in the other models of this directory.
 *
 * Configurations (default: the chunk starts at CST2A, queued earlier, and the writer switches
 * it to CST2B):
 *   -DTO_FIRST            the writer switches to CST1: the case where load 1 and load 2 of
 *                         the pass can disagree
 *   -DINIT_BUF            the chunk starts materialised: the writer must leave it alone, the
 *                         pass compresses it (the CAS path)
 *   -DSTRICT_SCAN         the stronger property of isolation: the pass never scans a chunk that is
 *                         not a buffer. The library does not need it for safety (the clear is
 *                         idempotent, see section 7 of docs/formal_models.md); it is what the
 *                         mark and the window buy by design, and what the mutations below break
 *   -DSWITCH_CAS          the switch is a CAS from the constant observed, not a store (an
 *                         alternative the source does not take)
 *
 * Mutations. With -DSTRICT_SCAN -DTO_FIRST the first four make GenMC report a violation;
 * under the properties P1-P3 alone they do NOT (that is the finding, docs/formal_models.md 7.6):
 *   -DMUT_NO_RECHECK      the writer switches without looking at the mark
 *   -DMUT_CHECK_FIRST     the writer looks at the mark BEFORE announcing
 *   -DMUT_NO_WINDOW       the pass does not wait for the writers
 *   -DMUT_NO_FENCE        no full barrier on either side (store; load)
 *   -DMUT_NO_BUF_CHECK    the writer switches without looking at `values` (materialised?): with
 *                         -DINIT_BUF it overwrites a buffer, which is refused under P1-P3
 *   -DMUT_RELAXED_VALUES  the switch is a relaxed store of `values` (readers are not modelled:
 *                         passes here, see docs/formal_models.md section 8)
 *   -DMUT_META_AFTER      the writer stores `values` BEFORE `meta` (passes here, same remark)
 *   -DSECOND_MATERIALISER a second thread materialises the chunk (CAS constant -> buffer) while
 *                         the writer switches: the span's precondition, broken. Must be refused
 *                         with the store, and passes with -DSWITCH_CAS.
 *
 * Build natively:    cc -O2 -pthread -DNATIVE switch_mt.c && ./a.out   (stress, weak oracle)
 * Usage under GenMC: genmc [--imm] [--disable-estimation] -- [-D...] switch_mt.c
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#define CST1 1
#define CST2A 2
#define CST2B 3
#define BUF 10

#define M1 1
#define M2A 2
#define M2B 3

#define CHUNK_X ((intptr_t)0x100)

#ifdef TO_FIRST
    #define NEW_CST CST1
    #define NEW_META M1
#else
    #define NEW_CST CST2B
    #define NEW_META M2B
#endif

#ifdef INIT_BUF
    #define INIT_VALUES BUF
#else
    #define INIT_VALUES CST2A
#endif

static atomic_int      values = INIT_VALUES;
static atomic_int      mark   = 0; /* steal mark of X */
static atomic_intptr_t wflag  = 0; /* the writer's flag: the chunk it writes, or 0 */
static atomic_int      comp   = 0; /* the pass's announce (start()) */

#ifdef NATIVE
static atomic_int meta = M2A;
    #define META_STORE(v) atomic_store_explicit(&meta, (v), memory_order_relaxed)
    #define META_LOAD() atomic_load_explicit(&meta, memory_order_relaxed)
#else
static int meta = M2A; /* plain: a data race on it is reported */
    #define META_STORE(v) (meta = (v))
    #define META_LOAD() (meta)
#endif

static int switched    = 0; /* written by the writer, read after the joins */
static int installed   = 0; /* written by the second materialiser */
static int freed       = 0; /* written by the pass */
static int freed_other = 0; /* the pass freed something that was not a buffer */

static int is_cst2(int v)
{
    return v == CST2A || v == CST2B;
}

static int values_of_meta(int m)
{
    return m == M1 ? CST1 : (m == M2A ? CST2A : CST2B);
}

static int meta_of_values(int v)
{
    return v == CST1 ? M1 : (v == CST2A ? M2A : M2B);
}

static void* writer(void* arg)
{
    (void)arg;
#ifdef MUT_CHECK_FIRST
    const int m = atomic_load_explicit(&mark, memory_order_acquire);
#endif
    /* announce_write(): the flag, then the full barrier */
#ifdef MUT_NO_FENCE
    atomic_store_explicit(&wflag, CHUNK_X, memory_order_relaxed);
#else
    atomic_store_explicit(&wflag, CHUNK_X, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
#endif
#if defined(MUT_NO_RECHECK)
    const int m = 0;
#elif !defined(MUT_CHECK_FIRST)
    const int m = atomic_load_explicit(&mark, memory_order_acquire); /* must_defer_write */
#endif
    if (!m)
    {
        int cur = atomic_load_explicit(&values, memory_order_acquire); /* materialised? */
#ifdef MUT_NO_BUF_CHECK
        if (1)
#else
        if (cur != BUF)
#endif
        {
            /* fill_constant_st, compressed branch: the slot and type, then the pointer */
#ifdef SWITCH_CAS
            META_STORE(NEW_META);
            if (atomic_compare_exchange_strong_explicit(&values, &cur, NEW_CST, memory_order_acq_rel,
                                                        memory_order_acquire))
            {
                switched = 1;
            }
#elif defined(MUT_META_AFTER)
            atomic_store_explicit(&values, NEW_CST, memory_order_release);
            META_STORE(NEW_META);
            switched = 1;
#else
            META_STORE(NEW_META);
    #ifdef MUT_RELAXED_VALUES
            atomic_store_explicit(&values, NEW_CST, memory_order_relaxed);
    #else
            atomic_store_explicit(&values, NEW_CST, memory_order_release);
    #endif
            switched = 1;
#endif
        }
    }
    atomic_store_explicit(&wflag, 0, memory_order_release); /* end_write */
    return NULL;
}

#ifdef SECOND_MATERIALISER
/* create_value(): CAS the constant observed -> a buffer (the element path of another thread) */
static void* materialiser(void* arg)
{
    (void)arg;
    int cur = atomic_load_explicit(&values, memory_order_acquire);
    if (cur != BUF && atomic_compare_exchange_strong_explicit(&values, &cur, BUF, memory_order_acq_rel,
                                                              memory_order_acquire))
    {
        installed = 1;
    }
    return NULL;
}
#endif

static void* compressor(void* arg)
{
    (void)arg;
    atomic_store_explicit(&mark, 1, memory_order_release); /* steal_mark */

    /* start(): announce, full barrier, then the window */
#ifdef MUT_NO_FENCE
    atomic_store_explicit(&comp, 1, memory_order_relaxed);
#else
    atomic_fetch_add_explicit(&comp, 1, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
#endif
#ifndef MUT_NO_WINDOW
    /* any_thread_writing_on_marked(): wait while the writer's flag names X */
    while (atomic_load_explicit(&wflag, memory_order_acquire) == CHUNK_X)
        ;
#endif

    /* fill_list_to_compress */
    int listed = 0;
    int target = CST1;
    const int v1 = atomic_load_explicit(&values, memory_order_acquire);
    if (v1 != CST1) /* is_not_cst_val1 */
    {
        const int v2 = atomic_load_explicit(&values, memory_order_acquire);
        if (!is_cst2(v2)) /* is_cst_val2 */
        {
            const int v3 = atomic_load_explicit(&values, memory_order_acquire); /* the scan */
            listed       = 1;
#ifdef STRICT_SCAN
            /* isolation: the pass scans what it is about to clear, a buffer */
            assert(v3 == BUF && "the pass scans a values word that is not a buffer");
#endif
            /* A buffer is all first_value in the model; a constant block scans as what it is. */
            target = is_cst2(v3) ? v3 : CST1;
        }
    }

    if (listed) /* clear_int_st -> block_clear_common */
    {
        META_STORE(meta_of_values(target)); /* 1. the metadata, BEFORE the CAS */
        int old = atomic_load_explicit(&values, memory_order_acquire); /* try_mark_cst_and_get_old */
        int ok  = 0;
        if (old != CST1 && !is_cst2(old))
        {
            ok = atomic_compare_exchange_strong_explicit(&values, &old, target, memory_order_acq_rel,
                                                         memory_order_acquire);
        }
        if (!ok)
        {
            /* init_values_cst_val(): the critical re-sync of values from the metadata */
            atomic_store_explicit(&values, values_of_meta(META_LOAD()), memory_order_release);
        }
        else
        {
            freed = 1;
            if (old != BUF)
            {
                freed_other = 1;
            }
        }
    }
    atomic_store_explicit(&mark, 0, memory_order_release); /* steal_unmark */
    return NULL;
}

int main(void)
{
    pthread_t w;
    pthread_t c;
#ifdef SECOND_MATERIALISER
    pthread_t m;
#endif
    pthread_create(&w, NULL, writer, NULL);
    pthread_create(&c, NULL, compressor, NULL);
#ifdef SECOND_MATERIALISER
    pthread_create(&m, NULL, materialiser, NULL);
#endif
    pthread_join(w, NULL);
    pthread_join(c, NULL);
#ifdef SECOND_MATERIALISER
    pthread_join(m, NULL);
#endif

    const int v = atomic_load_explicit(&values, memory_order_relaxed);

    /* P1: the switch, when made, is what the chunk holds (unless a buffer replaced it later) */
    if (switched && !installed)
    {
        assert(v == NEW_CST && "the switch was lost");
    }
    /* P2: the metadata describes the pointer */
    if (v != BUF)
    {
        assert(META_LOAD() == meta_of_values(v) && "values and metadata disagree");
    }
    /* P3: the pass frees buffers only, and a buffer owned is not lost */
    assert(!freed_other && "the pass freed a constant block as a buffer");
#ifdef INIT_BUF
    if (!freed)
    {
        assert(v == BUF && "the buffer the chunk started with was overwritten");
    }
#endif
#ifdef SECOND_MATERIALISER
    if (installed && !freed)
    {
        assert(v == BUF && "a buffer installed by a materialiser was overwritten");
    }
#endif
    return 0;
}
