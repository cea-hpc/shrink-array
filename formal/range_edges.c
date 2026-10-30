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
 * formal/range_edges.c -- the queueing filter of fill_range_mt on zero_any's relaxed elements:
 * two threads fill the two halves of ONE chunk with the same constant; does one of them queue it?
 *
 * A REDUCED MODEL. A chunk of 4 elements, all 9 (materialised, not uniform at 0). Thread A
 * stores 0 into [0, 2), thread B into [2, 4), each through the relaxed element stores of
 * fill_span_relaxed_edges(); then each runs the filter of store_range_mt step 7,
 * may_be_uniform_relaxed(): elements 0, S/2 and S-1, relaxed loads; equal -> it queues the chunk.
 * The write sessions and the queue flag are left out: the question is only whether the filter
 * can say "not uniform" to both threads of a chunk they leave uniform.
 *
 *   model                      source (chunk/public/chunk/array_chunk_range_ops.h)
 *   ------------------------   -----------------------------------------------------
 *   fill_half()                fill_span_relaxed_edges() (store_element_relaxed)
 *   may_be_uniform()           may_be_uniform_relaxed() (load_element_relaxed)
 *   writer()                   store_range_mt(), steps 5 and 7, relaxed-element branch
 *
 * PROPERTY (asserted when both are done): the chunk is uniform at 0, so at least one thread
 * queued it.
 *
 * THE SOURCE (default): a look; if it says "not uniform", a seq_cst fence and a second look.
 * Two seq_cst fences order the two store->load pairs, so one thread's second look sees the
 * other's stores: must PASS.
 *
 * MUTATIONS, each must FAIL:
 *   -DMUT_ONE_LOOK          one look, no fence -- the source before (F7, docs/formal_models.md):
 *                           store buffering, each thread's stores sit in its buffer while it
 *                           loads the other's edge, and both read 9 (allowed by C11 relaxed,
 *                           x86-TSO and AArch64; reproduced natively on x86).
 *   -DMUT_FENCE_ONE_SIDE    the fence in thread A only: both sides need it.
 *   -DMUT_RECHECK_NO_FENCE  the second look without the fence.
 *
 * Build for GenMC:   genmc -- [-DMUT_...] range_edges.c
 * Build natively:    cc -O2 -pthread -DNATIVE range_edges.c && ./a.out
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

#define S 4

static int        elems[S];
static atomic_int queued;

static void store_relaxed(int i, int v) { __atomic_store_n(&elems[i], v, __ATOMIC_RELAXED); }
static int  load_relaxed(int i) { return __atomic_load_n(&elems[i], __ATOMIC_RELAXED); }

static bool may_be_uniform(void)
{
    const int v0 = load_relaxed(0);
    return v0 == load_relaxed(S / 2) && v0 == load_relaxed(S - 1);
}

static void writer(int lo, int hi, bool fence_side)
{
    for (int i = lo; i < hi; ++i)
        store_relaxed(i, 0);
    bool uniform = may_be_uniform();
#ifndef MUT_ONE_LOOK
    if (!uniform)
    {
    #ifndef MUT_RECHECK_NO_FENCE
        if (fence_side)
            atomic_thread_fence(memory_order_seq_cst);
    #endif
        uniform = may_be_uniform();
    }
#else
    (void)fence_side;
#endif
    if (uniform)
        atomic_fetch_add_explicit(&queued, 1, memory_order_relaxed);
}

static void* writer_a(void* arg)
{
    (void)arg;
    writer(0, S / 2, true);
    return NULL;
}

static void* writer_b(void* arg)
{
    (void)arg;
#ifdef MUT_FENCE_ONE_SIDE
    writer(S / 2, S, false);
#else
    writer(S / 2, S, true);
#endif
    return NULL;
}

static int run_once(void)
{
    for (int i = 0; i < S; ++i)
        elems[i] = 9;
    atomic_init(&queued, 0);
    pthread_t a, b;
    pthread_create(&a, NULL, writer_a, NULL);
    pthread_create(&b, NULL, writer_b, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    assert(atomic_load_explicit(&queued, memory_order_relaxed) >= 1);
    return 0;
}

#ifdef NATIVE
int main(void)
{
    for (int k = 0; k < 100000; ++k)
        run_once();
    return 0;
}
#else
int main(void) { return run_once(); }
#endif
