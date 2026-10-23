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
// copy_range over whole chunks. Variant-agnostic: chunk / chunk2 / chunk2a
// follow the compile-time macro (CHUNK_VARIANT_ZERO / _V2 / _V2A), like the
// other harnesses here.
// ============================================================================
//
// The copy counterpart of fill_range_constant_validation. Guards:
//
//   1. every element reads back its source, whatever state the chunk was in
//      before (first constant, second constant, materialised) and whatever
//      the source holds (uniform per chunk, mixed, uniform but for one
//      element that the three-element guess does not look at);
//   2. on Chunk2A, a chunk the range covers whole and whose source is
//      uniform is compressed AT ONCE (Chunk2A::compress, apply = true):
//      right after the call, with no pass between, it no longer holds a
//      buffer. A chunk with a mixed source keeps its buffer (the scan
//      refuses it), and so does a chunk covered in part.
//
// The other variants queue the chunks for the pass: only property 1 is
// checked there, and
//
//   3. on v1 and Chunk2, a source equal to the constant a chunk is
//      compressed at writes nothing: the chunk stays compressed (whole or in
//      part, at the first constant or Chunk2's second); a source equal to it
//      on a prefix only materialises the chunk from the first other value.
//
// WHAT IT EXERCISES -- an array of n_chunks chunks plus a partial tail:
//
//   fresh       first constant <- chunk c uniform at 10 + c
//   mixed-dst   materialised   <- chunk c uniform at 10 + c
//   second      second value   <- the second value (1.0) everywhere
//   zero        materialised   <- the first constant (0.0) everywhere
//   mixed-src   first constant <- plain values: every chunk keeps its buffer
//   half        materialised   <- even chunks uniform, odd chunks plain
//   near        materialised   <- uniform but for element S/2 + 1 (the guess
//                                 passes, the scan refuses): all keep a buffer
//   partial     materialised   <- uniform, over [S/2, N - S/2)
//   tail        first constant <- 3.0 over the whole array, partial last chunk
//   same-first  first constant <- the first constant (0.0) everywhere, partial
//               last chunk included: nothing materialised on any variant
//   same-part   first constant <- 0.0 over [S/2, N - S/2): nothing materialised
//   same-second second value   <- 1.0 everywhere: nothing on Chunk2 and Chunk2A
//   prefix      first constant <- 0.0 on the first half of each chunk, plain
//               values after: every chunk materialised, every element right
//
// The setup writes go through operator[] in a region (the pass between two
// regions then runs as usual); the copies run outside any region
// (copy_range's serial contract).
//
// EXIT STATUS: 0 when every check holds, 1 otherwise. Any thread count.

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector (the sources)

#include "chunk/chunk_traits.h"          // forward decls + default ind_type = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h"          // zero_one_value::Array
#include "chunk/zero_any_array.h"         // zero_any::Array
#include "chunk/array_chunk_range_ops.h" // copy_range
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_zero*
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::copy_range_constant_validation
{

using namespace shrink_array;

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

constexpr int n_chunks = 64;

/// @brief Two values alternating: no chunk uniform.
static double plain(long i)
{
    return (i % 2) ? 2.5 : 3.5;
}

/// @brief One region writing [0, n) through operator[], then the pass.
template <typename Fn>
static void phase(Array<double>& arr, long n, Fn&& body)
{
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
        {
            begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            body(i);
        }
    }
    if constexpr (build_config::has_not_ompt)
    {
        finalize_parallel_phase();
    }
    if constexpr (chunk_config::use_zero_any && !build_config::use_openmp)
    {
        arr.make_block_compression(); // a serial chunk2a has no between-region pass
    }
}

/// @brief Chunks [c0, c1) still holding a buffer.
static int materialised(const Array<double>& arr, int c0, int c1)
{
    const long S = static_cast<long>(arr.chunk_size());
    int        n = 0;
    for (int c = c0; c < c1; ++c)
    {
        n += arr.written(static_cast<int>(c * S));
    }
    return n;
}

/// @brief Elements of [lo, hi) that do not read back @p expect(i).
template <typename Expect>
static long mismatches(const Array<double>& arr, long lo, long hi, Expect&& expect)
{
    long bad = 0;
    for (long i = lo; i < hi; ++i)
    {
        bad += (std::as_const(arr)[static_cast<int>(i)] != expect(i));
    }
    return bad;
}

/// @brief The checks of a case: the read-back, and the chunks materialised
///        when @p check_mat (Chunk2A always, the others for property 3).
static int report(const char* scase, long bad, int mat, int mat_expected, bool check_mat)
{
    int fails = 0;
    if (bad != 0)
    {
        println("[{}/{}] FAIL: {} element(s) do not read back their source", variant_name(), scase, bad);
        ++fails;
    }
    if (check_mat && mat != mat_expected)
    {
        println("[{}/{}] FAIL: {} chunk(s) materialised right after the copy, expected {}", variant_name(), scase, mat,
                mat_expected);
        ++fails;
    }
    if (fails == 0)
    {
        println("[{}/{}] ok    ({} materialised right after the copy)", variant_name(), scase, mat);
    }
    return fails;
}

/// @brief A source of @p n elements, element i = @p f(i).
template <typename F>
static std::vector<double> source(long n, F&& f)
{
    std::vector<double> v(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        v[static_cast<std::size_t>(i)] = f(i);
    }
    return v;
}

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    const long S    = static_cast<long>(Array<double>(1).chunk_size());
    const long N    = n_chunks * S;
    const long Ntot = N + S / 2; // a partial last chunk, never covered whole

    println("=== copy_range_constant_validation [variant = {}]  chunk_size={}  chunks={} ===", variant_name(), S,
            n_chunks);

    int        fails = 0;
    const auto sz    = [](long x) { return static_cast<std::size_t>(x); };

    const auto per_chunk = [S](long i) { return 10.0 + static_cast<double>(i / S); };
    const auto half      = [S](long i) { return ((i / S) % 2 == 0) ? 10.0 + static_cast<double>(i / S) : plain(i); };
    const auto near      = [S](long i) { return (i % S == S / 2 + 1) ? 9.5 : 4.0; };

    // Each case: setup of the destination, the source over [lo, hi), the
    // chunks expected to keep a buffer on Chunk2A.
    enum class Dst
    {
        fresh,
        materialised,
        second
    };
    struct Case
    {
        const char*         name;
        Dst                 dst;
        std::vector<double> src; // over [0, Ntot); copied over [lo, hi)
        long                lo;
        long                hi;
        int                 mat_expected;
        int                 count_to;    // chunks counted: [0, count_to)
        int                 mat_v1 = -1; // v1: chunks expected materialised, -1 unchecked
        int                 mat_v2 = -1; // Chunk2: likewise
    };

    // Half of each chunk at the first constant, then plain values.
    const auto prefix = [S](long i) { return (i % S < S / 2) ? 0.0 : plain(i); };

    const std::vector<Case> cases = {
        {"fresh", Dst::fresh, source(Ntot, per_chunk), 0, N, 0, n_chunks},
        {"mixed-dst", Dst::materialised, source(Ntot, per_chunk), 0, N, 0, n_chunks},
        {"second", Dst::second, source(Ntot, [](long) { return 1.0; }), 0, N, 0, n_chunks},
        {"zero", Dst::materialised, source(Ntot, [](long) { return 0.0; }), 0, N, 0, n_chunks},
        {"mixed-src", Dst::fresh, source(Ntot, plain), 0, N, n_chunks, n_chunks},
        {"half", Dst::materialised, source(Ntot, half), 0, N, n_chunks / 2, n_chunks},
        {"near", Dst::materialised, source(Ntot, near), 0, N, n_chunks, n_chunks},
        {"partial", Dst::materialised, source(Ntot, per_chunk), S / 2, N - S / 2, 2, n_chunks},
        {"tail", Dst::fresh, source(Ntot, [](long) { return 3.0; }), 0, Ntot, 1, n_chunks + 1},
        {"same-first", Dst::fresh, source(Ntot, [](long) { return 0.0; }), 0, Ntot, 0, n_chunks + 1, 0, 0},
        {"same-part", Dst::fresh, source(Ntot, [](long) { return 0.0; }), S / 2, N - S / 2, 0, n_chunks, 0, 0},
        {"same-second", Dst::second, source(Ntot, [](long) { return 1.0; }), 0, N, 0, n_chunks, -1, 0},
        {"prefix", Dst::fresh, source(Ntot, prefix), 0, N, n_chunks, n_chunks, n_chunks, n_chunks},
    };

    for (const Case& c : cases)
    {
        Array<double> a(static_cast<int>(Ntot));
        maybe_set_second_value(a, 1.0);
        switch (c.dst)
        {
            case Dst::fresh:
                break;
            case Dst::materialised:
                phase(a, N, [&](long i) { a[static_cast<int>(i)] = plain(i); });
                break;
            case Dst::second:
                phase(a, N, [&](long i) { a[static_cast<int>(i)] = 1.0; });
                break;
        }
        // What every element must read back: the source inside [lo, hi), the
        // setup outside it.
        const auto before = [&](long i) -> double
        {
            if (i >= N)
            {
                return 0.0;
            }
            switch (c.dst)
            {
                case Dst::fresh:
                    return 0.0;
                case Dst::materialised:
                    return plain(i);
                case Dst::second:
                    return 1.0;
            }
            return 0.0;
        };
        copy_range(a, sz(c.lo), sz(c.hi), c.src.begin() + c.lo);
        const long bad =
            mismatches(a, 0, Ntot, [&](long i)
                       { return (i >= c.lo && i < c.hi) ? c.src[static_cast<std::size_t>(i)] : before(i); });
        constexpr bool v1           = !chunk_config::use_zero_any && !chunk_config::use_zero_one_value;
        const int      mat_expected = chunk_config::use_zero_any ? c.mat_expected : v1 ? c.mat_v1 : c.mat_v2;
        fails += report(c.name, bad, materialised(a, 0, c.count_to), mat_expected, mat_expected >= 0);
        a.clear();
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            fails == 0 ? "=== COPY RANGE CONSTANT OK, TEARDOWN OK ===" : "=== COPY RANGE CONSTANT FAILURES ===");
    return fails ? 1 : 0;
}

} // namespace validation::copy_range_constant_validation
