// ============================================================================
// Marking-path harness. Variant-agnostic: chunk / chunk2 / chunk2a follow the
// compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the other harnesses here.
// ============================================================================
//
// Guards ONE property: a chunk whose elements are ALL written back to a
// constant is compressed by the library's own incremental path -- the write
// marks it a candidate, the pass between two regions compresses it -- with
// no compress() call from the application. compress() (compress_int) is the
// audit pass of the check builds: it scans every chunk and compresses what
// the incremental path left behind, so a program that relies on it sees no
// leak in a check build and a run-long leak in a perf build. This harness
// never calls it: what it counts after a region is what a perf build keeps.
//
// WHAT IT EXERCISES
//
//   Each scenario writes an array of 100 chunks, region after region, and
//   counts the chunks still materialised after each one (written(i) on the
//   first element of every chunk):
//
//   A  plain values everywhere (2.5 / 3.5 alternating), then the first constant (0)
//      everywhere. The chunk was materialised by a value that is NEITHER
//      constant (nor uniform: chunk2a would take a uniform run as its
//      second value), and no write ever moved it from one constant to the other.
//      Until Sept '26 the fast-block pre-check accepted only that
//      constant-to-constant transition, and this chunk stayed materialised
//      for the run: 100 out of 100 in the probe that found it.
//
//   B  the second constant everywhere (a fresh chunk goes cst1 -> cst2:
//      the transition the old pre-check did accept), then a plain value,
//      then the second constant again -- the case A for the second
//      constant -- then the first.
//
//   C  an array that has declared NO second value, after another array
//      declared 7.0: a plain value, then 0; then 7.0 -- the OTHER array's
//      second value, which must not compress here on chunk2 -- then 0, and a
//      read-back of every element. Until Sept '26 the first 0 written over
//      a materialised chunk of such an array aborted: its candidate list did
//      not exist.
//
//   What is expected depends on the variant: chunk (v1) has no second
//   constant, so its runs of 1.0 or 7.0 stay materialised; chunk2a picks a
//   second value per chunk, so its runs of 7.0 compress even with nothing
//   declared. The table is in expect_after().
//
// THE PHASE PROTOCOL
//
//   Same brackets as compress_roundtrip: begin_parallel_phase() by every
//   thread inside the region, finalize_parallel_phase() outside it -- that
//   last one is the pass this harness relies on in a GNU build (under OMPT
//   the callbacks run it, serially the counters compress inline). At 1
//   thread and at 8, as the other harnesses: the answer must not depend on
//   the team.
//
// EXIT STATUS: 0 when every count matches, 1 otherwise.

#include <cstdint> // std::uint8_t (the Kind enum's base)
#include <cstdio>
#include <type_traits>
#include <utility> // std::as_const
#include <unistd.h>

#include "chunk/chunk_traits.h" // forward decls + default ind_type = int
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/chunk_std.h"   // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"      // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"   // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h" // begin_parallel_phase / finalize_parallel_phase

// Defined by every executable of the library (the fast allocator reads it).
int __mem_page_size = 0;

using namespace shrink_array;

// The phase brackets, no-ops under OMPT (the callbacks do it) and serially.
static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

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

// set_second_value only exists on chunk2 / chunk2a; no-op elsewhere.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

constexpr int kChunks = 100;

/// @brief The "plain" content: two values alternating, so that no chunk is
///        uniform -- chunk2a takes any uniform run as a second constant of
///        its own, and a single plain value everywhere would compress there.
static double plain(int i)
{
    return (i % 2) ? 2.5 : 3.5;
}

/// @brief The chunks of @p arr still holding a buffer, out of the kChunks
///        written (the array has one more, empty, that no phase touches).
static int materialised(const Array<double>& arr)
{
    const int S = static_cast<int>(arr.chunk_size());
    int       n = 0;
    for (int j = 0; j < kChunks; ++j)
    {
        n += arr.written(j * S);
    }
    return n;
}

/// @brief One region writing every element, then the between-region pass.
///
/// A serial chunk2a has no between-region pass at all (there is no region):
/// its candidates queue until make_block_compression() -- the pass a writer
/// runs once its list is long enough, and resize() runs -- consumes them,
/// and a hundred candidates never reach that length. The harness runs that
/// pass itself there: it consumes the candidate lists, so a chunk the write
/// path failed to queue still shows as materialised, which is the property
/// here. Every OpenMP build compresses at finalize_parallel_phase(), chunk2a
/// included since Sept '26 (Chunk2A::compress_chunk_to_compress runs the
/// writers' pass rank by rank when no OMPT callback does it).
template <typename Fn>
static void phase(Array<double>& arr, int N, Fn&& body)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            body(i);
        }
    }
    end_phase();
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
}

/// @brief What a constant written everywhere must leave materialised, per
///        variant: `first` compresses everywhere; `declared` (the array's
///        declared second value) compresses on chunk2 / chunk2a; `other`
///        (a value the array never declared) compresses on chunk2a alone,
///        which discovers its second value per chunk.
enum class Kind : std::uint8_t
{
    first,
    declared,
    other,
    plain
};

static int expect_after(Kind k)
{
    switch (k)
    {
        case Kind::first:
            return 0;
        case Kind::declared:
            return chunk_config::use_chunk2or2a ? 0 : kChunks;
        case Kind::other:
            return chunk_config::use_chunk2a ? 0 : kChunks;
        case Kind::plain:
            return kChunks;
    }
    return -1;
}

static int check(const char* scenario, const char* step, const Array<double>& arr, Kind k)
{
    const int got = materialised(arr);
    const int exp = expect_after(k);
    if (got != exp)
    {
        printf("[%s/%s] FAIL after '%s': %d chunk(s) materialised, expected %d\n", variant_name(), scenario, step, got,
               exp);
        return 1;
    }
    printf("[%s/%s] ok    after '%-10s': %3d materialised\n", variant_name(), scenario, step, got);
    return 0;
}

int main()
{
    __mem_page_size = getpagesize();
    ParamChunk::init_indexes(true);

    // The geometry this build runs at, read off a throwaway array.
    const int S = static_cast<int>(Array<double>(1).chunk_size());
    const int N = kChunks * S;

    printf("=== mark_path_validation [variant = %s]  chunk_size=%d  chunks=%d  threads=%d ===\n", variant_name(), S,
           kChunks, CHUNK_PMP_MAX_THREADS);

    int fails = 0;

    // A: materialised by a plain value, then the first constant everywhere.
    {
        Array<double> a(N);
        maybe_set_second_value(a, 1.0);
        phase(a, N, [&](int i) { a[i] = plain(i); });
        fails += check("A", "plain", a, Kind::plain);
        phase(a, N, [&](int i) { a[i] = 0.0; });
        fails += check("A", "first", a, Kind::first);
        a.clear();
    }

    // B: the second constant, a plain value, the second constant, the first.
    {
        Array<double> b(N);
        maybe_set_second_value(b, 1.0);
        phase(b, N, [&](int i) { b[i] = 1.0; });
        fails += check("B", "second", b, Kind::declared);
        phase(b, N, [&](int i) { b[i] = plain(i); });
        fails += check("B", "plain", b, Kind::plain);
        phase(b, N, [&](int i) { b[i] = 1.0; });
        fails += check("B", "second", b, Kind::declared);
        phase(b, N, [&](int i) { b[i] = 0.0; });
        fails += check("B", "first", b, Kind::first);
        b.clear();
    }

    // C: no second value declared, next to an array that declared 7.0.
    {
        Array<double> other(N);
        maybe_set_second_value(other, 7.0);
        Array<double> c(N);
        phase(c, N, [&](int i) { c[i] = plain(i); });
        fails += check("C", "plain", c, Kind::plain);
        phase(c, N, [&](int i) { c[i] = 0.0; });
        fails += check("C", "first", c, Kind::first);
        phase(c, N, [&](int i) { c[i] = 7.0; });
        fails += check("C", "other", c, Kind::other);
        phase(c, N, [&](int i) { c[i] = 0.0; });
        fails += check("C", "first", c, Kind::first);

        long bad = 0;
        for (int i = 0; i < N; ++i)
        {
            bad += (std::as_const(c)[i] != 0.0);
        }
        if (bad)
        {
            printf("[%s/C] FAIL: %ld element(s) do not read back 0\n", variant_name(), bad);
            ++fails;
        }
        c.clear();
        other.clear();
    }

    // The library's own teardown, as in the other harnesses.
    release_chunk_pools();
    destroy_block_chunk();

    printf("\n[%s] %s\n", variant_name(),
           fails == 0 ? "=== MARK PATH OK, TEARDOWN OK ===" : "=== MARK PATH FAILURES ===");
    return fails ? 1 : 0;
}
