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

// ============================================================================
// zero_one_value foreign-key harness. zero_one_value only (zero_one_value::Array<T>), default static
// size, OpenMP builds (the passes it is about run inside a region).
// ============================================================================
//
// Guards ONE property: a chunk is only ever compressed to its OWN array's
// second value.
//
// THE QUESTION
//
//   A rank keeps one candidate list per second value. zero_one_value::Proxy's
//   perform_compression() runs the pass of every key of the rank and hands
//   each one the WRITER's proxy, and the pass ends by queueing that proxy's
//   chunk -- into the list of the key it just processed. If that key is not
//   the writer's array's second value, the chunk sits in a foreign list, and
//   a later pass of that key, finding it constant at the foreign value, would
//   compress it to the foreign constant array. Reads would still be right
//   (the array holds that value), but the owning array, which only recognises
//   its own constant, would take the chunk for materialised and WRITE INTO
//   the shared constant array: every chunk of every array compressed to that
//   value would change.
//
// WHAT IT DOES
//
//   `a` declares 1.0 and holds 2.0 everywhere: no chunk of `a` may ever be
//   compressed (2.0 is neither 0 nor its second value). `b` declares 2.0, is
//   compressed to it, and every thread keeps taking its own chunks of `b`
//   through materialise / constant again, so the key-2.0 passes run all the
//   time. In the same region every thread writes 2.0 into elements of `a`
//   (offsets of its own), so `a`'s proxies are the writers when those passes
//   are triggered.
//
//   Then: every chunk of `a` must still be materialised, and a write of 5.0
//   into each chunk of `a` must leave `b` reading 2.0 everywhere.
//
// EXIT STATUS: 0 when both hold and the passes ran, 1 otherwise; 77 when the
// build has no OpenMP.

#include <atomic>  // std::atomic, std::memory_order_relaxed
#include <chrono>  // std::chrono::steady_clock, std::chrono::duration
#include <cstdint> // std::uint64_t
#include <cstdlib> // std::getenv, std::atof

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"               // build_config::has_not_ompt, build_config::use_openmp
#include "chunk/PMP_CHUNK.h"            // CHUNK_PMP_MAX_THREADS, CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/manage_ompt.h"          // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"       // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::zero_one_value_foreign_key_validation
{

using namespace shrink_array;

namespace
{

using Arr = zero_one_value::Array<double>;
using Ind = Arr::IndType;

constexpr long   chunks_a = 512;  ///< chunks of `a`
constexpr long   chunks_b = 4096; ///< chunks of `b`
constexpr double val_a    = 1.0;  ///< `a`'s second value
constexpr double val_b    = 2.0;  ///< `b`'s second value, and what `a` holds

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

struct Xorshift
{
    std::uint64_t s;
    std::uint64_t next() noexcept
    {
        s ^= s << 13U;
        s ^= s >> 7U;
        s ^= s << 17U;
        return s;
    }
};

/// @brief Chunks of @p arr whose values pointer is @p cst: compressed to that
///        constant, whatever the array itself would call them.
long pointing_at(const Arr& arr, long n_chunks, const double* cst)
{
    long count = 0;
    for (long c = 0; c < n_chunks; ++c)
    {
        count += (arr.chunk_ref(static_cast<std::size_t>(c)).values_load() == cst) ? 1 : 0;
    }
    return count;
}

/// @brief Chunks of @p arr compressed (not materialised).
long compressed(const Arr& arr, long n_chunks, long chunk_sz)
{
    long count = 0;
    for (long c = 0; c < n_chunks; ++c)
    {
        count += arr.written(static_cast<Ind>(c * chunk_sz)) ? 0 : 1;
    }
    return count;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    if constexpr (!build_config::use_openmp)
    {
        println("[zero_one_value/foreign-key] SKIP: the passes this is about run inside a parallel region");
        return 77;
    }

    int fails = 0;
    {
        Arr        arr;
        Arr        other;
        const long chunk_sz = static_cast<long>(arr.chunk_size());
        arr.resize(static_cast<Ind>(chunks_a * chunk_sz));
        other.resize(static_cast<Ind>(chunks_b * chunk_sz));
        arr.set_second_value(val_a);
        other.set_second_value(val_b);
        const Arr& ca   = arr;
        const Arr& cb   = other;
        const int  team = CHUNK_PMP_MAX_THREADS;
        println("=== zero_one_value_foreign_key_validation  chunk_size={}  threads={}  a: {} chunks of {:g} (declares "
                "{:g})  b: {} chunks (declares {:g}) ===",
                chunk_sz, team, chunks_a, val_b, val_a, chunks_b, val_b);

        // `arr` = 2.0 everywhere, `other` = 2.0 everywhere and compressed to it.
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < chunks_a * chunk_sz; ++i)
            {
                arr[static_cast<Ind>(i)] = val_b;
            }
#pragma omp for schedule(static)
            for (long i = 0; i < chunks_b * chunk_sz; ++i)
            {
                other[static_cast<Ind>(i)] = val_b;
            }
        }
        end_phase();
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            other.compress();
        }
        end_phase();
        const long b_compressed0 = compressed(cb, chunks_b, chunk_sz);

        // The stress: own chunks of `other` toggled, own offsets of `arr` rewritten.
        const char*  env_s   = std::getenv("FK_SECONDS");
        const double seconds = env_s ? std::atof(env_s) : 2.0;
        // `other`'s constant: the pointer every compressed chunk of `other` holds.
        const double* const b_cst       = cb.chunk_ref(0).values_load();
        const long          a_on_b_cst0 = pointing_at(ca, chunks_a, b_cst);
        std::atomic<long>   toggles{0};
#pragma omp parallel
        {
            begin_phase();
            const int  me       = CHUNK_PMP_THREAD_RANK;
            const int  nthreads = CHUNK_PMP_NUM_THREADS;
            Xorshift   rng{0x2545F4914F6CDD1DULL ^ (static_cast<std::uint64_t>(me + 1) * 0x9E3779B97F4A7C15ULL)};
            const auto t0    = std::chrono::steady_clock::now();
            long       local = 0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
            {
                for (int r = 0; r < 256; ++r)
                {
                    // `other`: a chunk of this thread's, materialised then constant again.
                    const long cbk =
                        me + static_cast<long>(rng.next() % static_cast<std::uint64_t>(chunks_b / nthreads)) * nthreads;
                    const Ind b0 = static_cast<Ind>(cbk * chunk_sz);
                    other[b0]    = 3.0;
                    other[b0]    = val_b;
                    // `arr`: an element of this thread's, rewritten to the same 2.0.
                    const long cak = static_cast<long>(rng.next() % static_cast<std::uint64_t>(chunks_a));
                    const long off = me % chunk_sz;
                    arr[static_cast<Ind>(cak * chunk_sz + off)] = val_b;
                    ++local;
                }
            }
            toggles.fetch_add(local, std::memory_order_relaxed);
        }
        end_phase();

        const long a_compressed = pointing_at(ca, chunks_a, b_cst);
        const long b_compressed = compressed(cb, chunks_b, chunk_sz);
        println("[zero_one_value/foreign-key] {} toggles ({:.1f} s); b: {}/{} chunks compressed before, {} after; "
                "a: {} of {} chunks on b's constant before the stress, {} after",
                toggles.load(), seconds, b_compressed0, chunks_b, b_compressed, a_on_b_cst0, chunks_a, a_compressed);
        if (b_compressed0 == 0)
        {
            println("[zero_one_value/foreign-key] FAIL premise: `b` was not compressed to its second value");
            ++fails;
        }
        if (a_compressed != 0)
        {
            println("[zero_one_value/foreign-key] FAIL {} chunk(s) of `a` point at `b`'s constant array: compressed to "
                    "another array's second value",
                    a_compressed);
            ++fails;
        }

        // A write into every chunk of `arr` must not reach `other`.
        for (long c = 0; c < chunks_a; ++c)
        {
            arr[static_cast<Ind>(c * chunk_sz + (chunk_sz - 1))] = 5.0;
        }
        long bad_b = 0;
        long bad_a = 0;
        for (long i = 0; i < chunks_b * chunk_sz; ++i)
        {
            bad_b += (cb[static_cast<Ind>(i)] != val_b) ? 1 : 0;
        }
        for (long i = 0; i < chunks_a * chunk_sz; ++i)
        {
            const double want = (i % chunk_sz == chunk_sz - 1) ? 5.0 : val_b;
            bad_a += (ca[static_cast<Ind>(i)] != want) ? 1 : 0;
        }
        if (bad_b != 0 || bad_a != 0)
        {
            println("[zero_one_value/foreign-key] FAIL after writing 5.0 into each chunk of `a`: {} element(s) of `b` "
                    "and {} of `a` read wrong -- the write landed in a shared constant array",
                    bad_b, bad_a);
            ++fails;
        }
        else
        {
            println("[zero_one_value/foreign-key] OK: no chunk of `a` compressed to `b`'s value, writes stay in `a`");
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[zero_one_value] {}", fails ? "=== FOREIGN KEY FAILED ===" : "=== FOREIGN KEY OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::zero_one_value_foreign_key_validation
