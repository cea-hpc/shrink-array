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
// Read / reclaim harness. Variant-agnostic: zero / zero_one_value / zero_any follow the
// compile-time macro (CHUNK_VARIANT_ZERO*), like the other harnesses here.
// OpenMP builds only: the property is about two threads.
// ============================================================================
//
// Guards ONE property: a pure read never sees the buffer a concurrent
// compression has just released.
//
// THE WINDOW
//
//   A pure read (ArrayChunkBase::operator[] const -> ChunkBase::element_at)
//   loads the chunk's `values` pointer, then reads vals[i]. A compression pass
//   run INSIDE the region -- by a writer whose candidate list crossed the
//   threshold (trigger_compression_if_needed -> make_block_compression) --
//   CASes `values` to the constant and frees the old buffer at once
//   (clear_int_st -> free_block_to_pool). Nothing orders the two: the
//   write/compress handshake (start(), steal_mark) waits for WRITERS only.
//
//     CHUNK_STD_ALLOC    the buffer goes back to malloc: use-after-free. ASan
//                        reports it; without ASan, free()'s bin pointers sit in
//                        the first 16 bytes (elements 0 and 1 of a double).
//     fast allocator     the buffer is parked at the top of the rank's LIFO
//                        reuse list; the next materialisation pops it and
//                        overwrites it: the reader returns a value the chunk
//                        never held, silently.
//
//   The window is two instructions wide. The read-delay test hook
//   (test_hooks.h, switched on by this harness: one read in 16, 50 us) makes
//   the READER threads wait between the load and the dereference.
//
// WHAT IT DOES
//
//   One array, partitioned per writer into VICTIM chunks and POLLUTER chunks.
//   Everything is the first constant (0) unless said otherwise.
//
//   writers   sweep their victims: element 0 of each to 1.0 (materialise),
//             back to 0.0 (the chunk is constant again: a candidate). The
//             list crosses the variant's threshold, and the compression pass
//             runs inside the region, freeing victim buffers. After each
//             victim, polluter chunks taken round a ring are filled with 3.0
//             (4.0 at element 0: not constant) -- materialised from the reuse
//             list, i.e. from the buffers just freed -- and stay so for half a
//             ring before going back to 0.0 (a candidate, freed in turn).
//   readers   read random victim elements 1 .. S-1, alternately through the
//             const operator[] and through the write proxy's conversion
//             (`double v = a[i]` on a non-const array). Nobody writes those
//             elements: the answer is 0.0, always. Anything else is a read
//             from a released buffer.
//
//   Premise, checked: victim chunks were found compressed at the end of a
//   sweep, i.e. the pass did run inside the region. Without it the test
//   proves nothing and fails as such.
//
// EXIT STATUS: 0 = no stale read and the premise held, 1 otherwise,
// 77 = skipped (fewer than 2 threads).
//
// MODE `compound` (argument): the same window, on the WRITE side
//
//   `a[i] *= x` (and /=, %=) on zero_one_value / zero_any reads the element first
//   (zero_one_value::Proxy / zero_any::Proxy apply_compound_operation: v_old = read_values()), outside
//   the write session, and drops the operation when it reads first_value (0):
//   "0 * x is 0". If a pass in the region compressed the chunk and released
//   its buffer between that read's pointer load and its element load, and a
//   first-constant materialisation took the buffer back (zero-filled), the
//   read sees 0 and the multiplication is lost -- the element keeps its old
//   value.
//
//   victims   chunks uniform at the second value V: the writers take each
//             one through materialise (element 0 to 1.0) and back to V (a
//             candidate), so the passes keep compressing them inside the
//             region and releasing their buffers.
//   polluters first-constant chunks the writers materialise with element 0
//             only: the buffer they take from the reuse list is zero-filled.
//   RMW       threads (opted into the delay) apply `*= 2` then `*= 0.5` to
//             random victim elements 1 .. S-1, each thread on offsets of its
//             own. The two are exact on V, so every victim element ends at V;
//             one lost operation leaves V / 2 or 2 V.
//
//   zero has no second value: the mode is skipped there (77).
//
// MODE `remark` (argument): a pass must not erase another pass's claim
//
//   A compression pass marks its candidates (ChunkBase::steal_mark) so that
//   writers keep off them while it scans and steals, and hands each one back
//   at the end: its mark, and the dedup flag that lets a writer queue the
//   chunk again. Released flag first, mark second, another thread could queue
//   the chunk in between, start its own pass and mark it -- and the first
//   pass's unmark erased that claim: writers then wrote into a chunk the
//   second pass was scanning and stealing.
//
//   Every thread writes 1.0 then 0.0 to elements of a small pool of shared
//   chunks (offsets of its own), reading each 1.0 back; the 0.0 makes the
//   chunk constant again, a candidate, so passes run all the time on every
//   thread. Thread 0's passes wait at the hand-back point
//   (CHUNK_TEST_PASS_DELAY_*, test_hooks.h). Check builds stop on the
//   erasure itself (CHUNK_INVARIANT_FAILED in the pass); every build checks the
//   read-backs and that the pool ends at 0.

#include <atomic>      // std::atomic, std::memory_order
#include <chrono>      // std::chrono::steady_clock, std::chrono::duration
#include <cstdint>     // std::uint64_t
#include <cstdlib>     // std::getenv, std::atol
#include <string_view> // std::string_view
#include <type_traits> // std::conditional_t
#include <omp.h>       // omp_get_thread_num, omp_get_max_threads

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"               // chunk_config::use_zero_one_value*, build_config::use_openmp / has_not_ompt
#include "chunk/manage_ompt.h"          // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/test_hooks.h"  // test_hooks::read_delay_* / pass_delay_*
#include "chunk/internal/print.h"       // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::read_reclaim_validation
{

using namespace shrink_array;

namespace
{

static_assert(build_config::use_openmp, "read_reclaim_validation needs an OpenMP build");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

using Arr = Array<double>;
using Ind = Arr::IndType;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "zero_one_value";
    }
    else
    {
        return "zero";
    }
}

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

// Victims per writer: twice the largest in-region trigger floor (zero_one_value:
// 2000), so a single sweep crosses it on every variant.
constexpr long n_victims   = 4096;
constexpr long n_polluters = 512;
constexpr long per_writer  = n_victims + n_polluters;

constexpr double pollute_value = 3.0;

long env_long(const char* name, long dflt)
{
    const char* env   = std::getenv(name);
    const long  value = env ? std::atol(env) : 0;
    return value > 0 ? value : dflt;
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

// set_second_value only exists on zero_one_value / zero_any.
template <typename A, typename Val>
void maybe_set_second_value(A& arr, const Val& value)
{
    if constexpr (requires(A probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

constexpr double second_value = 2.0;

/// @brief The `compound` mode (see the file header). @return the exit status.
int run_compound(int team, int writers, double seconds, long pollute)
{
    if constexpr (!chunk_config::use_zero_one_value && !chunk_config::use_zero_any)
    {
        println("[{}/read-reclaim compound] SKIP: no second value on this variant", variant_name());
        return 77;
    }

    int rc = 0;
    {
        Arr        arr;
        const long chunk_sz = static_cast<long>(arr.chunk_size());
        const long total    = writers * per_writer * chunk_sz;
        arr.resize(static_cast<Ind>(total));
        maybe_set_second_value(arr, second_value);
        const Arr& ca = arr;

        println("=== read_reclaim_validation compound [variant = {}]  threads={} (writers {})  chunk_size={}  "
                "read_delay_every={} ({} us)  pollute={}  {:.0f} s ===",
                variant_name(), team, writers, chunk_sz, test_hooks::read_delay_every, test_hooks::read_delay_us,
                pollute, seconds);

        // Victims uniform at the second value, serially, before any region.
        for (long w = 0; w < writers; ++w)
        {
            for (long c = 0; c < n_victims; ++c)
            {
                for (long k = 0; k < chunk_sz; ++k)
                {
                    arr[static_cast<Ind>((w * per_writer + c) * chunk_sz + k)] = second_value;
                }
            }
        }

        std::atomic<int>  writers_left{writers};
        std::atomic<long> rmw_ops{0};
        std::atomic<long> compressed_seen{0};
        std::atomic<long> sweeps{0};

#pragma omp parallel num_threads(team)
        {
            begin_phase();
            const int me = omp_get_thread_num();

            if (me < writers)
            {
                const long base_v = me * per_writer;
                const long base_p = base_v + n_victims;
                long       next_p = 0;
                const auto t0     = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
                {
                    for (long c = 0; c < n_victims; ++c)
                    {
                        const Ind v0 = static_cast<Ind>((base_v + c) * chunk_sz);
                        arr[v0]      = 1.0;          // materialise (filled with V)
                        arr[v0]      = second_value; // uniform again: a candidate

                        // Polluters, as a ring: the chunk half a ring behind goes
                        // back to 0 (a candidate), the one at the head is
                        // materialised from the reuse list with element 0 only --
                        // a zero-filled buffer, which is what a delayed v_old read
                        // then sees.
                        for (long q = 0; q < pollute; ++q)
                        {
                            const Ind old0 =
                                static_cast<Ind>((base_p + (next_p + n_polluters / 2) % n_polluters) * chunk_sz);
                            arr[old0]    = 0.0;
                            const Ind p0 = static_cast<Ind>((base_p + next_p) * chunk_sz);
                            next_p       = (next_p + 1) % n_polluters;
                            arr[p0]      = pollute_value;
                        }
                    }
                    long count = 0;
                    for (long c = 0; c < n_victims; ++c)
                    {
                        count += ca.written(static_cast<Ind>((base_v + c) * chunk_sz)) ? 0 : 1;
                    }
                    compressed_seen.fetch_add(count, std::memory_order_relaxed);
                    sweeps.fetch_add(1, std::memory_order_relaxed);
                }
                writers_left.fetch_sub(1, std::memory_order_release);
            }
            else
            {
                // Each RMW thread owns the offsets j = 1 + rmw_rank + k * n_rmw:
                // two read-modify-writes of the SAME element from two threads
                // race in the library as in a std::vector (nothing serialises
                // writers between themselves), and would lose an operation
                // that has nothing to do with the window under test.
                test_hooks::read_delay_enabled = true;
                Xorshift   rng{0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(me) * 0xBF58476D1CE4E5B9ULL)};
                const long n_rmw     = team - writers;
                const long rmw_rank  = me - writers;
                const long n_own     = (chunk_sz - 1 - rmw_rank + n_rmw - 1) / n_rmw; // offsets of this thread
                long       local_ops = 0;
                while (n_own > 0 && writers_left.load(std::memory_order_acquire) > 0)
                {
                    const long writer = static_cast<long>(rng.next() % static_cast<std::uint64_t>(writers));
                    const long victim = static_cast<long>(rng.next() % static_cast<std::uint64_t>(n_victims));
                    const long kk     = static_cast<long>(rng.next() % static_cast<std::uint64_t>(n_own));
                    const long offset = 1 + rmw_rank + kk * n_rmw;
                    const Ind  idx    = static_cast<Ind>((writer * per_writer + victim) * chunk_sz + offset);
                    arr[idx] *= 2.0;
                    arr[idx] *= 0.5;
                    local_ops += 2;
                }
                test_hooks::read_delay_enabled = false;
                rmw_ops.fetch_add(local_ops, std::memory_order_relaxed);
            }
        }
        end_phase();

        // Every victim element must be back to V.
        long bad = 0;
        for (long w = 0; w < writers; ++w)
        {
            for (long c = 0; c < n_victims; ++c)
            {
                for (long k = 0; k < chunk_sz; ++k)
                {
                    const Ind    idx   = static_cast<Ind>((w * per_writer + c) * chunk_sz + k);
                    const double value = ca[idx];
                    if (value != second_value) [[unlikely]]
                    {
                        if (bad < 8)
                        {
                            println("[{}/read-reclaim compound] FAIL element {} (chunk {}, offset {}) = {:g}, "
                                    "expected {:g} -- a compound operation was lost{}",
                                    variant_name(), static_cast<long>(idx), w * per_writer + c, k, value, second_value,
                                    value == second_value / 2
                                        ? " (a *= 2 dropped)"
                                        : (value == second_value * 2 ? " (a *= 0.5 dropped)" : ""));
                        }
                        ++bad;
                    }
                }
            }
        }

        const long seen = compressed_seen.load();
        println("[{}/read-reclaim compound] {} compound ops, {} sweeps, {} victim chunks found compressed "
                "after a sweep",
                variant_name(), rmw_ops.load(), sweeps.load(), seen);
        if (seen == 0)
        {
            println("[{}/read-reclaim compound] FAIL premise: no victim chunk was compressed inside the region "
                    "-- the test proves nothing",
                    variant_name());
            rc = 1;
        }
        if (bad > 0)
        {
            println("[{}/read-reclaim compound] FAIL {} element(s) off V", variant_name(), bad);
            rc = 1;
        }
        else if (seen > 0)
        {
            println("[{}/read-reclaim compound] OK: every compound operation applied", variant_name());
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            rc ? "=== READ RECLAIM COMPOUND FAILED ===" : "=== READ RECLAIM COMPOUND OK ===");
    return rc;
}

/// @brief The `remark` mode (see the file header). @return the exit status.
int run_remark(int team, double seconds)
{
    constexpr long pool_chunks = 4096; // chunks shared by every thread

    int rc = 0;
    {
        Arr        arr;
        const long chunk_sz = static_cast<long>(arr.chunk_size());
        arr.resize(static_cast<Ind>(pool_chunks * chunk_sz));
        const Arr& ca = arr;
        test_hooks::pass_delay_us =
            static_cast<unsigned>(env_long("READ_RECLAIM_PASS_DELAY_US", CHUNK_TEST_PASS_DELAY_US));

        println("=== read_reclaim_validation remark [variant = {}]  threads={}  chunk_size={}  pool={} chunks  "
                "pass_delay_every={} ({} us, thread 0)  {:.0f} s ===",
                variant_name(), team, chunk_sz, pool_chunks, test_hooks::pass_delay_every, test_hooks::pass_delay_us,
                seconds);

        std::atomic<long> lost{0};
        std::atomic<long> writes{0};

#pragma omp parallel num_threads(team)
        {
            begin_phase();
            const int me = omp_get_thread_num();
            if (me == 0)
            {
                test_hooks::pass_delay_enabled = true;
            }
            // Offsets of my own, j = me + k * team: two threads never write
            // the same element.
            const long n_own = (chunk_sz - me + team - 1) / team;
            Xorshift   rng{0xD1B54A32D192ED03ULL ^ (static_cast<std::uint64_t>(me + 1) * 0x9E3779B97F4A7C15ULL)};
            long       local = 0;
            const auto t0    = std::chrono::steady_clock::now();
            while (n_own > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
            {
                for (int r = 0; r < 256; ++r)
                {
                    const long victim = static_cast<long>(rng.next() % static_cast<std::uint64_t>(pool_chunks));
                    const long offset = me + static_cast<long>(rng.next() % static_cast<std::uint64_t>(n_own)) * team;
                    const Ind  idx    = static_cast<Ind>(victim * chunk_sz + offset);
                    arr[idx]          = 1.0;
                    const double got  = static_cast<double>(arr[idx]); // read_pure: synchronised
                    if (got != 1.0) [[unlikely]]
                    {
                        const long slot = lost.fetch_add(1, std::memory_order_relaxed);
                        if (slot < 8)
                        {
                            println("[{}/read-reclaim remark] FAIL thread {}: element {} (chunk {}, offset {}) "
                                    "reads {:g} right after writing 1 -- the write was lost",
                                    variant_name(), me, static_cast<long>(idx), victim, offset, got);
                        }
                    }
                    arr[idx] = 0.0; // constant again: a candidate
                    ++local;
                }
            }
            test_hooks::pass_delay_enabled = false;
            writes.fetch_add(local, std::memory_order_relaxed);
        }
        end_phase();

        long nonzero = 0;
        for (long i = 0; i < pool_chunks * chunk_sz; ++i)
        {
            nonzero += (ca[static_cast<Ind>(i)] != 0.0) ? 1 : 0;
        }

        println("[{}/read-reclaim remark] {} write pairs, {} lost read-backs, {} elements left non-zero",
                variant_name(), writes.load(), lost.load(), nonzero);
        if (lost.load() > 0 || nonzero > 0)
        {
            println("[{}/read-reclaim remark] FAIL", variant_name());
            rc = 1;
        }
        else
        {
            println("[{}/read-reclaim remark] OK: no write lost", variant_name());
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), rc ? "=== READ RECLAIM REMARK FAILED ===" : "=== READ RECLAIM REMARK OK ===");
    return rc;
}

} // namespace

int run(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    // The test hooks this harness widens its windows with (test_hooks.h), on
    // for this process only: one pure read in 16 of a reader thread, one
    // candidate in 8 of thread 0's passes -- unless a whole-tree build fixed
    // other values. A build without the hooks (no CHUNK_TEST_HOOKS) runs the
    // same checks without the delays.
    if (test_hooks::read_delay_every == 0)
    {
        test_hooks::read_delay_every = 16;
    }
    if (test_hooks::pass_delay_every == 0)
    {
        test_hooks::pass_delay_every = 8;
    }

    const int team = omp_get_max_threads();
    if (team < 2)
    {
        println("[{}/read-reclaim] SKIP: needs at least 2 threads (OMP_NUM_THREADS={})", variant_name(), team);
        return 77;
    }
    const int writers = team >= 8 ? team / 4 : 1;
    // Tunables (environment): run time, polluter chunks filled per victim,
    // reader delay in microseconds.
    const double seconds      = static_cast<double>(env_long("READ_RECLAIM_SECONDS", 2));
    const long   pollute      = env_long("READ_RECLAIM_POLLUTE", 4);
    test_hooks::read_delay_us = static_cast<unsigned>(env_long("READ_RECLAIM_DELAY_US", CHUNK_TEST_READ_DELAY_US));

    if (argc > 1 && std::string_view(argv[1]) == "compound")
    {
        return run_compound(team, writers, seconds, pollute);
    }
    if (argc > 1 && std::string_view(argv[1]) == "remark")
    {
        return run_remark(team, seconds);
    }

    long fails = 0;
    int  rc    = 0;
    {
        Arr        arr;
        const long chunk_sz = static_cast<long>(arr.chunk_size());
        const long total    = writers * per_writer * chunk_sz;
        arr.resize(static_cast<Ind>(total));
        const Arr& ca = arr;

        println("=== read_reclaim_validation [variant = {}]  threads={} (writers {})  chunk_size={}  "
                "read_delay_every={} ({} us)  pollute={}  {:.0f} s ===",
                variant_name(), team, writers, chunk_sz, test_hooks::read_delay_every, test_hooks::read_delay_us,
                pollute, seconds);

        std::atomic<int>  writers_left{writers};
        std::atomic<long> bad{0};
        std::atomic<long> reads{0};
        std::atomic<long> compressed_seen{0}; // victims found compressed at the end of a sweep
        std::atomic<long> sweeps{0};

#pragma omp parallel num_threads(team)
        {
            begin_phase();
            const int me = omp_get_thread_num();

            if (me < writers)
            {
                // ── Writer ──────────────────────────────────────────────────
                const long base_v = me * per_writer;
                const long base_p = base_v + n_victims;
                long       next_p = 0;
                const auto t0     = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
                {
                    for (long c = 0; c < n_victims; ++c)
                    {
                        const Ind v0 = static_cast<Ind>((base_v + c) * chunk_sz);
                        arr[v0]      = 1.0; // materialise
                        arr[v0]      = 0.0; // constant again: a candidate

                        // Pollute, as a ring: the chunk half a ring behind
                        // goes back to 0 (a candidate: the pass frees it),
                        // the chunk at the head is materialised from the
                        // reuse list -- the buffers the pass just freed --
                        // and filled with pollute_value, element 0 apart so the
                        // chunk is NOT constant and stays materialised (and
                        // its buffer overwritten) for half a ring. A pollution
                        // undone at once would hand a late reader zeros again.
                        for (long q = 0; q < pollute; ++q)
                        {
                            const Ind old0 =
                                static_cast<Ind>((base_p + (next_p + n_polluters / 2) % n_polluters) * chunk_sz);
                            for (long k = 0; k < chunk_sz; ++k)
                            {
                                arr[old0 + static_cast<Ind>(k)] = 0.0;
                            }
                            const Ind p0 = static_cast<Ind>((base_p + next_p) * chunk_sz);
                            next_p       = (next_p + 1) % n_polluters;
                            arr[p0]      = pollute_value + 1.0;
                            for (long k = 1; k < chunk_sz; ++k)
                            {
                                arr[p0 + static_cast<Ind>(k)] = pollute_value;
                            }
                        }
                    }
                    long count = 0;
                    for (long c = 0; c < n_victims; ++c)
                    {
                        count += ca.written(static_cast<Ind>((base_v + c) * chunk_sz)) ? 0 : 1;
                    }
                    compressed_seen.fetch_add(count, std::memory_order_relaxed);
                    sweeps.fetch_add(1, std::memory_order_relaxed);
                }
                writers_left.fetch_sub(1, std::memory_order_release);
            }
            else
            {
                // ── Reader ──────────────────────────────────────────────────
                test_hooks::read_delay_enabled = true;
                Xorshift rng{0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(me) * 0xBF58476D1CE4E5B9ULL)};
                long     local_reads = 0;
                while (writers_left.load(std::memory_order_acquire) > 0)
                {
                    const long writer = static_cast<long>(rng.next() % static_cast<std::uint64_t>(writers));
                    const long victim = static_cast<long>(rng.next() % static_cast<std::uint64_t>(n_victims));
                    const long offset = 1 + static_cast<long>(rng.next() % static_cast<std::uint64_t>(chunk_sz - 1));
                    const Ind  idx    = static_cast<Ind>((writer * per_writer + victim) * chunk_sz + offset);
                    // Both read entries: the const operator[] and the write
                    // proxy's conversion (operator T -> read_pure), which a
                    // `double v = a[i]` on a non-const array goes through.
                    const double value = (local_reads & 1) ? ca[idx] : static_cast<double>(arr[idx]);
                    ++local_reads;
                    if (value != 0.0) [[unlikely]]
                    {
                        const long slot = bad.fetch_add(1, std::memory_order_relaxed);
                        if (slot < 8)
                        {
                            println("[{}/read-reclaim] FAIL stale read: element {} (chunk {}, offset {}) = {:g}, "
                                    "expected 0 -- read from a released buffer{}",
                                    variant_name(), static_cast<long>(idx), writer * per_writer + victim, offset, value,
                                    value == pollute_value ? " (reused and overwritten)" : "");
                        }
                    }
                }
                test_hooks::read_delay_enabled = false;
                reads.fetch_add(local_reads, std::memory_order_relaxed);
            }
        }
        end_phase();

        fails           = bad.load();
        const long seen = compressed_seen.load();
        println("[{}/read-reclaim] {} reads, {} sweeps, {} victim chunks found compressed after a sweep",
                variant_name(), reads.load(), sweeps.load(), seen);
        if (seen == 0)
        {
            println("[{}/read-reclaim] FAIL premise: no victim chunk was compressed inside the region -- the "
                    "pass never ran there, the test proves nothing",
                    variant_name());
            rc = 1;
        }
        if (fails > 0)
        {
            println("[{}/read-reclaim] FAIL {} stale read(s)", variant_name(), fails);
            rc = 1;
        }
        else if (seen > 0)
        {
            println("[{}/read-reclaim] OK: no read returned anything but the constant", variant_name());
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), rc ? "=== READ RECLAIM FAILED ===" : "=== READ RECLAIM OK ===");
    return rc;
}

} // namespace validation::read_reclaim_validation
