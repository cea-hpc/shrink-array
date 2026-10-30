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
 * formal/writer_compressor.c -- handshake 1 of docs/synchronisation_dekker.md:
 * the element writer against the compression pass (section 5.1 to 5.3, 5.6).
 *
 * A REDUCED MODEL, not the library: one chunk, NWRITERS writers each writing
 * one element once, one compression pass. Every atomic below carries the
 * memory order of the source it stands for; the table says which line.
 *
 *   model                         source (chunk/public/chunk/internal/)
 *   ---------------------------   -------------------------------------------------------
 *   flag[r]                       write_compress_sync_policy.h  WritingFlag::v
 *   announce_write()              write_compress_sync_policy.h  announce_write()
 *   end_write()                   write_compress_sync_policy.h  end_write()
 *   any_writing_on_marked()       write_compress_sync_policy.h  any_thread_writing_on_marked()
 *   any_writing()                 write_compress_sync_policy.h  any_thread_writing()
 *   hard_inflight, stealing       compression_manager.h         hard_inflight_, stealing_inflight_
 *   must_defer_write()            compression_manager.h         must_defer_write()
 *   is_stealing()                 compression_manager.h         is_hard_compression_inflight()
 *   compressor_start()/finish()   compression_manager_inline.h  start(), finish()
 *   heavy_barrier()               write_compress_sync_policy.h  dekker::heavy_barrier()
 *   mark / steal_mark*()          chunk_base.h                  steal_mark_, steal_mark(), steal_marked(), steal_unmark()
 *   values, values_load()         chunk_base.h                  values, values_load()
 *   materialise()                 zero_any_chunk_inline.h          create_value() + wait_before_materialise()
 *   cas_install()                 chunk_base.h                  compare_and_swap_gen()
 *   steal()                       zero_any_chunk.h                 try_mark_cst_and_get_old()
 *   scan                          zero_any_chunk_inline.h          fill_list_to_compress() / scan_constants
 *   element store / load          relaxed_element.h             store_element_relaxed / load_element_relaxed
 *   writer()                      proxy_base_inline.h           write_with_optional_lock_impl()
 *
 * PROPERTIES (asserted):
 *   P1  a writer never stores into a buffer the pass has returned (use after free);
 *   P2  a write that took the direct path is in the chunk when every thread is done
 *       (never silently lost); a deferred one is accounted as deferred;
 *   P3  (with --check-liveness) every spin loop terminates.
 *
 * SHAPES (section 4 of the document) -- pick one:
 *   -DSHAPE_SYMMETRIC          (default) writer: seq_cst store + seq_cst fence; compressor: fence.
 *                              What the checker can verify: C11 has no membarrier().
 *   -DSHAPE_ASYMMETRIC_NAKED   writer: relaxed store + compiler barrier, and NOTHING on the
 *                              compressor side but its fence. This is the asymmetric shape
 *                              WITHOUT the membarrier(): it must FAIL (store buffering), and
 *                              the failure is exactly what membarrier(2) repairs. The
 *                              asymmetric -> symmetric reduction is the argument of section
 *                              4.2 and of membarrier(2)'s own contract, not a checker result.
 *
 * CONFIGURATIONS:
 *   -DINIT_CONSTANT            the chunk starts at the constant (writers materialise it);
 *                              default: it starts with an all-zero buffer (a candidate).
 *   -DBOUNDED_COMPRESSOR_WAIT  the compressor gives up after one failed look (the
 *                              CHUNK_COMPRESS_MAX_YIELDS budget, reduced to its effect).
 *   -DNWRITERS=n               writers (default 2; NELEMS follows).
 *
 * MUTATIONS -- each is a bug the library had, or a line the document says is load-bearing.
 * The checker must REFUSE every one of them (that is the calibration of the model):
 *   -DMUT_NO_CHECK2            the writer does not look at the gate after its announce.
 *   -DMUT_RELAXED_MARK         steal_mark()/steal_marked() relaxed instead of release/acquire
 *                              (section 5.3, "Why the mark is read with acquire").
 *   -DMUT_WAIT_ON_HARD         materialisation waits on hard_inflight -- a PARKED compressor --
 *                              instead of its mark: the "triangle" the document forbids
 *                              (section 5.3, "Why two counters"). A liveness violation.
 *   -DMUT_NO_FENCE_COMPRESSOR  the compressor announces without its heavy barrier.
 *   -DMUT_MATERIALISE_WAIT_UNDER_FLAG
 *                              a fast-path writer spins on the mark before materialising even
 *                              though it holds its flag, as the library did before the model
 *                              (F2): with an unbounded compressor wait, a liveness violation.
 *
 * Build for GenMC:   genmc [--check-liveness] -- -DSHAPE_SYMMETRIC writer_compressor.c
 * Build natively:    cc -O2 -pthread -DNATIVE writer_compressor.c && ./a.out   (stress, weak oracle)
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef NWRITERS
    #define NWRITERS 2
#endif
#define NELEMS NWRITERS /* one element per writer: the writes never collide */

#if !defined(SHAPE_SYMMETRIC) && !defined(SHAPE_ASYMMETRIC_NAKED)
    #define SHAPE_SYMMETRIC 1
#endif

#ifdef MUT_RELAXED_MARK
    #define MO_MARK_STORE memory_order_relaxed
    #define MO_MARK_LOAD  memory_order_relaxed
#else
    #define MO_MARK_STORE memory_order_release /* steal_mark(): release */
    #define MO_MARK_LOAD  memory_order_acquire /* steal_marked(): acquire */
#endif

/* ───────────────────────── the shared state ───────────────────────── */

/* One chunk. `values` points to cst1 (compressed) or to a heap buffer. */
typedef struct
{
    _Atomic(atomic_int*) values;    /* chunk_base.h: atomic_ptr_t values */
    atomic_uint          mark;      /* chunk_base.h: steal_mark_ (0 clear, 1 marked) */
} chunk_t;

static atomic_int cst1[NELEMS];     /* the shared constant buffer, all zero, never written */
static chunk_t    chunk;

/* write_compress_sync_policy.h: per_thread_writing[r].v -- the chunk being written, or NULL */
static _Atomic(chunk_t*) flag[NWRITERS];

/* compression_manager.h: the two counters, signed on purpose */
static atomic_int hard_inflight;
static atomic_int stealing_inflight;

/* bookkeeping for the final check (plain: read after the joins only) */
static int  direct_done[NWRITERS];
static int  deferred[NWRITERS];
static bool compressor_gave_up;

#ifdef NATIVE
/* Natively, a returned buffer is not freed (free() would hide P1): it is flagged. */
static _Atomic(atomic_int*) returned_buffer;
#endif

/* ───────────────────────── chunk primitives ───────────────────────── */

static atomic_int* values_load(void)                   /* chunk_base.h values_load(): acquire */
{
    return atomic_load_explicit(&chunk.values, memory_order_acquire);
}

static bool steal_marked(void)                          /* chunk_base.h steal_marked() */
{
    return atomic_load_explicit(&chunk.mark, MO_MARK_LOAD) != 0;
}

static void steal_mark(void)                            /* chunk_base.h steal_mark() */
{
    atomic_store_explicit(&chunk.mark, 1, MO_MARK_STORE);
}

static void steal_unmark(void)                          /* chunk_base.h steal_unmark(): release */
{
    atomic_store_explicit(&chunk.mark, 0, memory_order_release);
}

static atomic_int* alloc_zero_buffer(void)
{
    atomic_int* b = malloc(sizeof(atomic_int) * NELEMS);
    for (int i = 0; i < NELEMS; ++i)
    {
        atomic_init(&b[i], 0);                          /* init_block(): filled BEFORE publication */
    }
    return b;
}

static void return_buffer(atomic_int* b)                /* free_block_to_pool(): the buffer leaves the chunk */
{
#ifdef NATIVE
    atomic_store_explicit(&returned_buffer, b, memory_order_seq_cst);
#else
    free(b);                                            /* GenMC: a later access is reported (P1) */
#endif
}

/* ───────────────────────── the Dekker policy ───────────────────────── */

static void announce_write(int r)                       /* write_compress_sync_policy.h announce_write() */
{
#ifdef SHAPE_ASYMMETRIC_NAKED
    atomic_store_explicit(&flag[r], &chunk, memory_order_relaxed);
    /* The source adds atomic_signal_fence(seq_cst) here: a COMPILER barrier. It is left out
       of the model on purpose: GenMC treats a signal fence as a thread fence (it would turn
       this shape into the symmetric one), and the IR GenMC explores keeps program order. */
#else
    atomic_store_explicit(&flag[r], &chunk, memory_order_seq_cst); /* begin_write(): seq_cst */
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

static void end_write(int r)                            /* write_compress_sync_policy.h end_write(): release */
{
    atomic_store_explicit(&flag[r], NULL, memory_order_release);
}

static void heavy_barrier(void)                         /* dekker::heavy_barrier(): fence (+ membarrier, not in C11) */
{
#ifndef MUT_NO_FENCE_COMPRESSOR
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

__attribute__((unused)) static bool any_writing_on_marked(void) /* any_thread_writing_on_marked(): acquire loads */
{
    for (int i = 0; i < NWRITERS; ++i)
    {
        chunk_t* c = atomic_load_explicit(&flag[i], memory_order_acquire);
        if (c != NULL && steal_marked())                /* c->steal_marked(): the one chunk here */
        {
            return true;
        }
    }
    return false;
}

__attribute__((unused)) static bool any_writing(void)   /* any_thread_writing(): acquire loads */
{
    for (int i = 0; i < NWRITERS; ++i)
    {
        if (atomic_load_explicit(&flag[i], memory_order_acquire) != NULL)
        {
            return true;
        }
    }
    return false;
}

static bool must_defer_write(void)                      /* compression_manager.h must_defer_write() */
{
    return steal_marked();
}

__attribute__((unused)) static bool is_stealing(void)   /* is_hard_compression_inflight(): seq_cst */
{
    return atomic_load_explicit(&stealing_inflight, memory_order_seq_cst) > 0;
}

/* ───────────────────────── the writer ───────────────────────── */

/* compare_and_swap_gen(): relaxed pre-load, then acq_rel CAS. Returns true on success. */
static bool cas_install(atomic_int* expected_cst, atomic_int* buffer)
{
    if (atomic_load_explicit(&chunk.values, memory_order_relaxed) != expected_cst)
    {
        return false;
    }
    return atomic_compare_exchange_strong_explicit(&chunk.values, &expected_cst, buffer,
                                                   memory_order_acq_rel, memory_order_acquire);
}

/* create_value(): wait_before_materialise(), allocate + fill, publish by one CAS. */
static atomic_int* materialise(int r)
{
#if defined(MUT_WAIT_ON_HARD)
    while (atomic_load_explicit(&hard_inflight, memory_order_seq_cst) > 0) {}   /* THE TRIANGLE */
#else
    /* wait_before_materialise(): a caller whose flag is on THIS chunk does not wait -- it
       passed check #2, so a mark it sees now belongs to a PARKED compressor waiting for that
       very flag; nothing can steal the chunk before end_write() (F2). A caller with no flag
       (alloc_chunk(), a flush replay) spins on the mark: not modelled here. */
    #ifdef MUT_MATERIALISE_WAIT_UNDER_FLAG
    const bool own_session = false;                     /* before F2: always wait */
    #else
    const bool own_session = atomic_load_explicit(&flag[r], memory_order_relaxed) == &chunk;
    #endif
    if (!own_session)
    {
        while (steal_marked()) {}                       /* this chunk's mark */
    }
#endif
    atomic_int* b = alloc_zero_buffer();
    if (!cas_install(cst1, b))
    {
        free(b);                                        /* recup_value(): the loser recycles its buffer */
    }
    return values_load();
}

static void* writer(void* arg)                          /* write_with_optional_lock_impl() */
{
    const int r   = (int)(long)arg;
    const int idx = r;                                  /* each writer its own element */
    const int v   = r + 1;                              /* never the constant 0 */

    /* Check #1 is a pure optimisation with no flag raised (a stale read is harmless):
       it is left out of the model on purpose. */

    announce_write(r);                                  /* (1)+(2) */

#ifdef MUT_NO_CHECK2
    const bool gate_closed = false;
#else
    const bool gate_closed = must_defer_write();        /* (3) check #2: a FRESH load after the announce */
#endif

    if (!gate_closed)
    {
        /* handle_no_bool_assignment(): ONE load of values inside the session */
        atomic_int* vals = values_load();
        if (vals == cst1)
        {
            vals = materialise(r);                      /* perform_core_assignment(): materialise then reload */
        }
#ifdef NATIVE
        assert(vals != atomic_load_explicit(&returned_buffer, memory_order_seq_cst)); /* P1 */
#endif
        atomic_store_explicit(&vals[idx], v, memory_order_relaxed);  /* store_element_relaxed() -- P1 under GenMC */
        end_write(r);
        direct_done[r] = 1;
    }
    else
    {
        end_write(r);                                   /* slow path: the write is captured, replayed by a flush */
        deferred[r] = 1;
    }
    return NULL;
}

/* ───────────────────────── the compressor ───────────────────────── */

static bool compressor_start(void)                      /* CompressionManagerThreadPolicy::start() */
{
    atomic_fetch_add_explicit(&hard_inflight, 1, memory_order_seq_cst);   /* parked */
    /* (flush early-out: handshake 2, not modelled here) */
    heavy_barrier();

    /* bounded wait for the writer window */
#define window_shut() any_writing_on_marked()
#ifdef BOUNDED_COMPRESSOR_WAIT
    if (window_shut())                                  /* the yield budget, reduced to "one look" */
    {
        atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
        return false;
    }
#else
    while (window_shut()) {}
#endif

    atomic_fetch_add_explicit(&stealing_inflight, 1, memory_order_seq_cst); /* parked -> stealing */
    atomic_thread_fence(memory_order_seq_cst);
    /* (second look at the flushers: handshake 2) */
    return true;
}

static void compressor_finish(void)                     /* CompressionManagerThreadPolicy::finish() */
{
    atomic_fetch_sub_explicit(&stealing_inflight, 1, memory_order_seq_cst);
    atomic_fetch_sub_explicit(&hard_inflight, 1, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
}

/* try_mark_cst_and_get_old(): acquire load, acq_rel CAS to the constant, the old buffer back */
static atomic_int* steal(void)
{
    atomic_int* old = values_load();
    if (old == cst1)
    {
        return NULL;
    }
    atomic_int* expected = old;
    if (!atomic_compare_exchange_strong_explicit(&chunk.values, &expected, cst1,
                                                 memory_order_acq_rel, memory_order_acquire))
    {
        return NULL;
    }
    return old;
}

static void* compressor(void* arg)                      /* make_block_compression_impl(), one candidate */
{
    (void)arg;
    steal_mark();                                       /* claim BEFORE start() */
    if (!compressor_start())
    {
        steal_unmark();
        compressor_gave_up = true;
        return NULL;
    }

    /* fill_list_to_compress(): the scan, relaxed element loads */
    atomic_int* vals = values_load();
    if (vals != cst1)
    {
        bool all_first = true;
        for (int i = 0; i < NELEMS; ++i)
        {
            if (atomic_load_explicit(&vals[i], memory_order_relaxed) != 0)
            {
                all_first = false;
            }
        }
        if (all_first)
        {
            atomic_int* old = steal();                  /* block_clear_common(): CAS, then free */
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

/* ───────────────────────── main: the check ───────────────────────── */

int main(void)
{
    for (int i = 0; i < NELEMS; ++i)
    {
        atomic_init(&cst1[i], 0);
    }
    atomic_init(&chunk.mark, 0);
#ifdef INIT_CONSTANT
    atomic_init(&chunk.values, cst1);
#else
    atomic_init(&chunk.values, alloc_zero_buffer());    /* a materialised, all-zero chunk: a candidate */
#endif
    for (int i = 0; i < NWRITERS; ++i)
    {
        atomic_init(&flag[i], NULL);
    }
    atomic_init(&hard_inflight, 0);
    atomic_init(&stealing_inflight, 0);
#ifdef NATIVE
    atomic_init(&returned_buffer, NULL);
#endif

    pthread_t tw[NWRITERS], tc;
    for (long i = 0; i < NWRITERS; ++i)
    {
        pthread_create(&tw[i], NULL, writer, (void*)i);
    }
    pthread_create(&tc, NULL, compressor, NULL);
    for (int i = 0; i < NWRITERS; ++i)
    {
        pthread_join(tw[i], NULL);
    }
    pthread_join(tc, NULL);

    /* P2: every direct write is in the chunk; every writer went one way or the other */
    atomic_int* vals = atomic_load_explicit(&chunk.values, memory_order_seq_cst);
    for (int r = 0; r < NWRITERS; ++r)
    {
        assert(direct_done[r] || deferred[r]);
        if (direct_done[r])
        {
            assert(vals != cst1);
            assert(atomic_load_explicit(&vals[r], memory_order_relaxed) == r + 1);
        }
    }
    /* the counters are back to zero */
    assert(atomic_load_explicit(&hard_inflight, memory_order_seq_cst) == 0);
    assert(atomic_load_explicit(&stealing_inflight, memory_order_seq_cst) == 0);
    (void)compressor_gave_up;
#ifndef NATIVE
    if (vals != cst1)
    {
        free(vals);                                     /* the array's destructor: no leak to report */
    }
#endif
    return 0;
}
