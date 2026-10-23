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
// The whole-array writes that take a team of their own outside a region
// (ArrayChunkBase::write_range_by_chunks): the copy constructor and copy
// assignment (copy_from), assign(count, val), assign(first, last) from a
// random-access and from a forward source, resize(n, val); a shrinking resize
// (release_chunk_range on a team); and zero_any::Array's operator== (on a team,
// stopping at a difference).
//
// The arrays have far more chunks than parallel_init_min_chunks, so outside a
// region every one of these runs on a team -- through parallel_phase(): in a
// check build without OMPT, a write deferred by a thread that did not begin
// the phase fails the phase protocol. Each result is compared element by
// element with a std::vector, before and after a compression pass. The same
// operations on a small array inside a region run on the calling thread, one
// thread at a time (array creation is not thread-safe).
//
// Exit 0 = every check holds. Any thread count (the tests run it at 1 and 8).
// ============================================================================

#include <forward_list> // std::forward_list (assign from a forward iterator)
#include <mutex>        // std::mutex, std::scoped_lock (the in-region arrays, one at a time)
#include <type_traits>  // std::conditional_t
#include <utility>      // std::as_const
#include <vector>       // std::vector

#include "chunk/chunk_traits.h"       // forward decls + default IntType = int
#include "chunk/array_chunk.h"        // zero::Array
#include "chunk/array_chunk2.h"       // zero_one_value::Array
#include "chunk/array_chunk2A.h"      // zero_any::Array
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/chunk_std.h"          // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"             // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // parallel_phase
#include "chunk/internal/env_parse.h" // shrink_array::arg_number
#include "chunk/internal/print.h"     // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::bulk_write_team_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "bulk_write_team_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

/// @brief Runs of zero, runs of 7 and varied values, by blocks of 1000.
template <typename T>
T value_at(long i)
{
    switch ((i / 1000) % 3)
    {
        case 0:
            return T{};
        case 1:
            return static_cast<T>(7);
        default:
            return static_cast<T>((i % 97) + 1);
    }
}

template <typename T>
int compare(const char* type, const char* step, const Array<T>& arr, const std::vector<T>& ref)
{
    if (static_cast<std::size_t>(arr.size()) != ref.size())
    {
        println("[{}/{}/{}] FAIL: size {}, expected {}", variant_name(), type, step, static_cast<long>(arr.size()),
                ref.size());
        return 1;
    }
    long bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if (!(std::as_const(arr)[static_cast<int>(i)] == ref[i]))
        {
            if (bad < 3)
            {
                println("[{}/{}/{}] FAIL: element {} differs", variant_name(), type, step, i);
            }
            ++bad;
        }
    }
    if (bad)
    {
        println("[{}/{}/{}] FAIL: {} element(s) differ", variant_name(), type, step, bad);
    }
    return bad ? 1 : 0;
}

/// @brief compare(), then again after a compression pass.
template <typename T>
int check(const char* type, const char* step, Array<T>& arr, const std::vector<T>& ref)
{
    int fails = compare(type, step, arr, ref);
    arr.compress();
    fails += compare(type, step, arr, ref);
    return fails;
}

template <typename T>
int run_type(const char* type, long n)
{
    int            fails = 0;
    std::vector<T> ref(static_cast<std::size_t>(n));
    Array<T>       a(static_cast<int>(n));
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = value_at<T>(i);
        a[static_cast<int>(i)]           = value_at<T>(i);
    }
    fails += check(type, "source", a, ref);

    // Copy constructor.
    Array<T> b(a);
    fails += check(type, "copy-construct", b, ref);

    // Copy assignment into an array of another size and content.
    Array<T> c(static_cast<int>(n / 3));
    for (long i = 0; i < n / 3; ++i)
    {
        c[static_cast<int>(i)] = static_cast<T>(3);
    }
    c = a;
    fails += check(type, "copy-assign", c, ref);

    if constexpr (chunk_config::use_chunk2a)
    {
        if (!(b == a) || !(c == a))
        {
            println("[{}/{}] FAIL: operator== on equal arrays", variant_name(), type);
            ++fails;
        }
        b[static_cast<int>(n - 2)] = static_cast<T>(55);
        if (b == a)
        {
            println("[{}/{}] FAIL: operator== misses a difference near the end", variant_name(), type);
            ++fails;
        }
        b[static_cast<int>(n - 2)] = ref[static_cast<std::size_t>(n - 2)];
        c[1]                       = static_cast<T>(55);
        if (c == a)
        {
            println("[{}/{}] FAIL: operator== misses a difference at the start", variant_name(), type);
            ++fails;
        }
    }

    // assign(count, val): a value, then the first constant.
    b.assign(static_cast<int>(n + n / 2), static_cast<T>(9));
    fails += check(type, "assign-9", b, std::vector<T>(static_cast<std::size_t>(n + n / 2), static_cast<T>(9)));
    b.assign(static_cast<int>(n), T{});
    fails += check(type, "assign-0", b, std::vector<T>(static_cast<std::size_t>(n), T{}));

    // assign(first, last): random access (a team), forward (in order).
    b.assign(ref.begin(), ref.end());
    fails += check(type, "assign-random-access", b, ref);
    std::forward_list<T> fl;
    for (long i = 999; i >= 0; --i)
    {
        fl.push_front(value_at<T>(i + 1000));
    }
    std::vector<T> ref_fl(fl.begin(), fl.end());
    b.assign(fl.begin(), fl.end());
    fails += check(type, "assign-forward", b, ref_fl);

    // resize(n, val): the new elements only.
    std::vector<T> ref_grow = ref;
    ref_grow.resize(static_cast<std::size_t>(n + n / 2), static_cast<T>(5));
    a.resize(static_cast<int>(n + n / 2), static_cast<T>(5));
    fails += check(type, "resize-grow-val", a, ref_grow);

    // A shrink large enough for its release to take a team, then a growth.
    ref_grow.resize(static_cast<std::size_t>(n / 4));
    a.resize(static_cast<int>(n / 4));
    fails += check(type, "resize-shrink", a, ref_grow);
    ref_grow.resize(static_cast<std::size_t>(n), T{});
    a.resize(static_cast<int>(n));
    fails += check(type, "resize-grow-again", a, ref_grow);

    // Inside a region: each thread its own small array, on the calling thread.
    // One thread at a time (a mutex, not `omp critical`: the region is a team
    // of the build's backend, the std backend's too, where an OpenMP pragma is
    // ignored): creating, resizing or destroying arrays from several threads
    // at once is outside the library's contract -- create_common() takes no
    // lock and the Chunk2A registry (PointerMapChunk2A) is not thread-safe
    // (see region_storage_validation).
    int        region_fails = 0;
    std::mutex one_at_a_time;
    parallel_phase(
        [&]
        {
            {
                const std::scoped_lock lock(one_at_a_time);
                const int              n_region = 3000;
                std::vector<T>         r(static_cast<std::size_t>(n_region));
                Array<T>               s(n_region);
                for (int i = 0; i < n_region; ++i)
                {
                    r[static_cast<std::size_t>(i)] = value_at<T>(i);
                    s[i]                           = value_at<T>(i);
                }
                Array<T> t(s);
                int      f = compare(type, "in-region-copy", t, r);
                t.assign(n_region, static_cast<T>(4));
                f += compare(type, "in-region-assign", t,
                             std::vector<T>(static_cast<std::size_t>(n_region), static_cast<T>(4)));
                region_fails += f; // under the lock
            }
        });
    fails += region_fails;

    if (fails == 0)
    {
        println("[{}/{}] OK: N={} copy, assign, resize(n, val), shrink{}", variant_name(), type, n,
                chunk_config::use_chunk2a ? ", operator==" : "");
    }
    return fails;
}

} // namespace

int run(int argc, char** argv)
{
    ParamChunk::init_indexes(true);
    const long n = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 20, "N") : (1L << 20);

    println("=== bulk_write_team_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    fails += run_type<double>("double", n);
    fails += run_type<int>("int", n);

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== BULK WRITES FAILED ===" : "=== BULK WRITES OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::bulk_write_team_validation
