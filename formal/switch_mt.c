/*
 * Copyright 2026 CEA (www.cea.fr)
 * SPDX-License-Identifier: Apache-2.0
 *
 * switch_mt.c -- reduced GenMC model: a writer switching a compressed Chunk2A
 * chunk to another constant in place (store_range_mt's in-place switch,
 * array_chunk_range_ops.h) against a compression pass that holds the same
 * chunk in its candidate list (Chunk2A::make_block_compression).
 *
 * The chunk is reduced to its `values` word:
 *   CST1  the first-constant block (a real block of zeros in the library),
 *   CST2A, CST2B  two second-value slots (identifiers in the registry region,
 *         NOT buffers: a pass that scans one and clears it frees a slot),
 *   BUF   a materialised buffer.
 * and its steal mark. The writer's flag says which chunk it is writing.
 *
 * Writer (store_range_mt, Chunk2A, a uniform span covering the chunk):
 *   announce_write(chunk)            -- flag = X; full barrier (symmetric shape)
 *   if (!must_defer_write(chunk)      -- the mark, acquire
 *       && !materialised(chunk))      -- values != BUF
 *       fill_constant_st(v)          -- values = the new constant, release
 *   end_write()                      -- flag = null, release
 *
 * Compressor (make_block_compression with a proxy):
 *   steal_mark(X)                    -- release
 *   start(): announce + full barrier, then wait until no writer flag names a
 *            marked chunk             -- the window (acquire loads)
 *   fill_list_to_compress(X):
 *     is_not_cst_val1()   -- load 1
 *     is_cst_val2()       -- load 2 (a second load of values)
 *     scan the buffer      -- load 3, must be BUF: the property
 *     clear_int_st         -- values = constant, the buffer freed
 *   steal_unmark(X)
 *
 * Property: the pass never scans and clears a `values` that is not a buffer
 * (a constant block or a second-value slot freed as a buffer: corruption),
 * and the switch the writer made is what the chunk holds at the end.
 *
 * The heavy barrier of the asymmetric shape (membarrier) is modelled by its
 * symmetric equivalent, a seq_cst fence on both sides, as in the other models
 * of this directory: the proof of announce_write() reduces one to the other.
 *
 * Mutations (each must make GenMC report a violation):
 *   -DMUT_NO_RECHECK   the writer switches without looking at the mark
 *   -DMUT_CHECK_FIRST  the writer looks at the mark BEFORE announcing
 *   -DMUT_NO_WINDOW    the pass does not wait for the writers
 *   -DMUT_NO_FENCE     no full barrier on either side (store; load)
 *
 * Usage: genmc [-- -DTO_FIRST] switch_mt.c        (TO_FIRST: switch to CST1,
 *        the case where load 1 and load 2 disagree; default: to CST2B)
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#define CST1 1
#define CST2A 2
#define CST2B 3
#define BUF 10

#define CHUNK_X ((intptr_t)0x100)

#ifdef TO_FIRST
    #define NEW_CST CST1
#else
    #define NEW_CST CST2B
#endif

static atomic_int      values = CST2A; /* compressed at a second value, queued earlier */
static atomic_int      mark   = 0;     /* steal mark of X */
static atomic_intptr_t wflag  = 0;     /* the writer's flag: the chunk it writes, or 0 */
static atomic_int      comp   = 0;     /* the pass's announce (start()) */

static int switched = 0; /* written by the writer, read after the joins */

static int is_cst2(int v)
{
    return v == CST2A || v == CST2B;
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
    if (!m && atomic_load_explicit(&values, memory_order_acquire) != BUF)
    {
        /* fill_constant_st, compressed branch: slot and type, then values */
        atomic_store_explicit(&values, NEW_CST, memory_order_release);
        switched = 1;
    }
    atomic_store_explicit(&wflag, 0, memory_order_release); /* end_write */
    return NULL;
}

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
    const int v1 = atomic_load_explicit(&values, memory_order_acquire);
    if (v1 != CST1) /* is_not_cst_val1 */
    {
        const int v2 = atomic_load_explicit(&values, memory_order_acquire);
        if (!is_cst2(v2)) /* is_cst_val2 */
        {
            const int v3 = atomic_load_explicit(&values, memory_order_acquire); /* the scan */
            assert(v3 == BUF && "the pass scans and clears a values word that is not a buffer");
            atomic_store_explicit(&values, CST1, memory_order_release); /* clear_int_st */
        }
    }
    atomic_store_explicit(&mark, 0, memory_order_release); /* steal_unmark */
    return NULL;
}

int main(void)
{
    pthread_t w;
    pthread_t c;
    pthread_create(&w, NULL, writer, NULL);
    pthread_create(&c, NULL, compressor, NULL);
    pthread_join(w, NULL);
    pthread_join(c, NULL);

    /* The switch, when made, is what the chunk holds: the pass did not
     * overwrite it with a stale decision. */
    if (switched)
    {
        assert(atomic_load_explicit(&values, memory_order_relaxed) == NEW_CST && "the switch was lost");
    }
    return 0;
}
