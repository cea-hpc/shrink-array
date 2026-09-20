// ============================================================================
// SEQUENTIAL re-chunk validation harness (path C, non-OMP) -- v3.
// Variant-agnostic: tests chunk / chunk2 / chunk2a depending on the
// compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like shrink_bench.cpp.
// ============================================================================
//
// Proves CONTENT INVARIANCE and, on failure, LOCATES the cause:
//   - first checks that the SNAPSHOT (read at the old geometry) is correct;
//   - traces the geometry before/after;
//   - shows the first mismatches along with the corresponding chunk boundary.
//
// The variant is selected by the macro passed by CMake (one executable per
// variant, linked against the matching dyn lib: chunk_v2_dyn / chunk_v2a_dyn /
// chunk_v1_dyn). No std::vector fallback: re-chunk is specific to the chunked
// variants, hence the static_assert.
//
// Chunk2 CONTRACT: the compression-candidate list is indexed by
// second_value (map_chunk2_to_compress_potentially_thr[thr][second_value]),
// populated ONLY by set_second_value. A second_value must therefore be
// declared BEFORE writing, otherwise get_vec_tuple_to_compress_potentially_thr
// asserts (vec == nullptr). This is what the bench does via
// maybe_set_second_value; we reproduce it here for every case. 2A / v1
// don't need this (no-op).
//
// Compile NON-OMP (or OMP), dyn, check. Exit 0 = every invariant holds.

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <unistd.h>

#include "chunk/chunk_traits.h"   // forward decls + default ind_type = int
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"         // chunk_config::use_chunk*
#include "chunk/array_chunk_range_ops.h" // fill_range / copy_range
#include "chunk/manage_ompt.h"   // finalize_parallel_phase (compression backlog drain)

using namespace shrink_array;

int __mem_page_size = 0;

namespace
{

// ── Compression backlog drain ────────────────────────────────────────────────
// A write only QUEUES its chunk as a compression candidate; the queue is
// drained at phase boundaries. When OMPT is compiled in, its callbacks do it;
// otherwise the application must, which is what shrink_bench.cpp does around
// each of its parallel phases.
//
// This file opens no parallel region and used to call neither, so under OpenMP
// the backlog was never drained: every all-zero chunk stayed materialised, and
// compress_int() -- which in embedded-compression mode only DIAGNOSES, it does
// not compress -- reported them round after round:
//
//   Should not have compress (0) values
//   Number of compress block : 494
//
// ~500 of the 1563 chunks, i.e. exactly the generator's zero runs. Values were
// always correct; the memory was simply never reclaimed, so the test was not
// exercising compression at all past the first round. No-op in serial builds
// (finalize_parallel_phase is `if constexpr (use_openmp)`) and skipped under
// OMPT, which already drains.
inline void drain_compression_backlog()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
}

} // namespace

// --- Active variant selection (same shape as shrink_bench.cpp) -------
static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "rechunk_validation must be built with CHUNK, CHUNK2 or CHUNK2A "
              "(re-chunk is chunk-only; no std::vector fallback here)");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a, ArrayChunk2A<T>,
        std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>,
            ArrayChunk<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a) return "chunk2a";
    else if constexpr (chunk_config::use_chunk2) return "chunk2";
    else return "chunk";
}

// set_second_value only exists on chunk2 / chunk2a: no-op elsewhere. REQUIRED
// on chunk2 before any write (see the contract at the top of the file); also
// exercises the val2 path (second_value preservation + tab_m_ptr_cst_val2
// queue on growth, for chunk2).
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
        a.set_second_value(v);
}

// sv_value: the value declared as second_value. Choosing a value present in
// the data makes val2 compression actually kick in (a stronger test); any
// value is enough to satisfy the Chunk2 contract (populate the map).
template <typename T>
int check_invariance(const char* label, long N, unsigned new_size, T (*gen)(long),
                     T sv_value = T(1), int post_rounds = 0)
{
    Array<T> arr(N);
    maybe_set_second_value(arr, sv_value); // no-op on chunk v1; required on chunk2

    fprintf(stderr, "[%s/%s] arr built: chunk_size=%u (g_active should be 512)\n",
            variant_name(), label, arr.chunk_size());

    std::vector<T> ref(N);
    for (long i = 0; i < N; ++i) { T v = gen(i); arr[i] = v; ref[i] = v; }

    const unsigned old_size = arr.chunk_size();

    // (a) pre-rechunk DIAGNOSTIC: is the content read at the old geometry correct?
    long pre_bad = 0, pre_first = -1;
    for (long i = 0; i < N; ++i)
        if (arr[i] != ref[i]) { if (pre_first < 0) pre_first = i; ++pre_bad; }
    if (pre_bad) {
        printf("[%s/%s] PRE-FAIL: %ld mismatches BEFORE rechunk (setup/read bug, not rechunk); "
               "first at i=%ld (chunk #%ld, offset %ld) got=%g expected=%g\n",
               variant_name(), label, pre_bad, pre_first, pre_first / (long) old_size,
               pre_first % (long) old_size, (double) arr[pre_first], (double) ref[pre_first]);
        return 1;
    }

    arr.rechunk_to(new_size);
    const unsigned got_size = arr.chunk_size();

    if (got_size != new_size) {
        printf("[%s/%s] FAIL: geometry %u -> %u (expected %u)\n",
               variant_name(), label, old_size, got_size, new_size);
        return 1;
    }
    if ((long)arr.size() != N) {
        printf("[%s/%s] FAIL: size %ld -> %ld\n", variant_name(), label, N, (long)arr.size());
        return 1;
    }
    long mismatches = 0, first = -1;
    for (long i = 0; i < N; ++i) {
        if (arr[i] != ref[i]) {
            if (first < 0) first = i;
            if (mismatches < 5)
                printf("[%s/%s] MISMATCH i=%ld got=%.17g exp=%.17g (new-chunk #%ld, offset %ld)\n",
                       variant_name(), label, i, (double)arr[i], (double)ref[i],
                       i / (long)new_size, i % (long)new_size);
            ++mismatches;
        }
    }
    if (mismatches) {
        printf("[%s/%s] FAIL: %ld/%ld mismatches after rechunk %u->%u (first at i=%ld)\n",
               variant_name(), label, mismatches, N, old_size, new_size, first);
        return 1;
    }
    // ── POST-RECHUNK PHASE: rewrite + compress in rounds, mimicking the bench.
    //    The perf-bench crash appears HERE (writes/compression touching the
    //    freshly re-chunked geometry), not in rechunk_to itself. Each round
    //    overwrites with changing data, forces a compression pass, and re-checks
    //    invariance against the ground-truth ref.
    for (int round = 0; round < post_rounds; ++round) {
        for (long i = 0; i < N; ++i) { T v = gen(i + (long)round * 7919); arr[i] = v; ref[i] = v; }
        // compress_int signature differs across variants: v1 (CHUNK) takes a
        // verbose bool; CHUNK2 / CHUNK2A take none. rechunk_validation.cpp is
        // compiled once per variant, so guard the call.
        // Hand the queued candidates to the compression pass before asking for
        // the result -- see drain_compression_backlog() above.
        drain_compression_backlog();
#if defined(CHUNK2A) || defined(CHUNK2)
        arr.compress_int();
#else
        arr.compress_int(false);
#endif
        long bad = 0, firstb = -1;
        for (long i = 0; i < N; ++i) if (arr[i] != ref[i]) { if (firstb < 0) firstb = i; ++bad; }
        if (bad) {
            printf("[%s/%s] POST-FAIL round %d: %ld/%ld mismatches after compress (first i=%ld, new-chunk #%ld off %ld)\n",
                   variant_name(), label, round, bad, N, firstb,
                   firstb / (long)new_size, firstb % (long)new_size);
            return 1;
        }
    }
    printf("[%s/%s] OK: %ld elements invariant, geometry %u->%u (second_value=%.17g)%s\n",
           variant_name(), label, N, old_size, new_size, (double)sv_value,
           post_rounds ? " [+post-rechunk compress rounds]" : "");
    return 0;
}

// ── COEXISTENCE CASE ─────────────────────────────────────────────────────────
// Every check above exercises ONE array at a time, so all live arrays always
// share a single geometry -- and they all pass. The perf bench does not: the
// advisor re-chunks the arrays it tracks and leaves the others at the original
// size, so arrays of DIFFERENT geometries are alive and written to in the same
// loop. That is what crashes (ASan: heap-buffer-overflow in store_value, a
// 64-element buffer written at index 76), on chunk AND on chunk2/chunk2a alike.
//
// This reproduces that state in the small: two arrays, one re-chunked and one
// left alone, written alternately. Passing means geometry coexistence is safe;
// failing pins the defect down to ~30 lines instead of a 1200-iteration bench.
template <typename T>
int check_coexistence(const char* label, long N, unsigned moved_size, int rounds)
{
    using Arr = Array<T>;

    Arr moved(static_cast<int>(N));   // will be re-chunked
    Arr fixed(static_cast<int>(N));   // stays at the ambient geometry

    for (long i = 0; i < N; ++i)
    {
        moved[i] = static_cast<T>((i % 7) ? 0 : i);
        fixed[i] = static_cast<T>((i % 5) ? 0 : i);
    }

    moved.rechunk_to(moved_size);

    // Alternate writes between the two geometries, the way the bench's loop does.
    for (int r = 0; r < rounds; ++r)
        for (long i = 0; i < N; ++i)
        {
            moved[i] = static_cast<T>((i + r) % 11 ? 0 : (i + r));
            fixed[i] = static_cast<T>((i + r) % 13 ? 0 : (i + r));
        }

    // Verify both arrays still read back what was last written.
    int bad = 0;
    const int r = rounds - 1;
    for (long i = 0; i < N && bad < 5; ++i)
    {
        const T em = static_cast<T>((i + r) % 11 ? 0 : (i + r));
        const T ef = static_cast<T>((i + r) % 13 ? 0 : (i + r));
        if (!(static_cast<T>(moved[i]) == em) || !(static_cast<T>(fixed[i]) == ef))
        {
            printf("[%s/%s] MISMATCH at %ld: moved=%g exp %g | fixed=%g exp %g\n",
                   variant_name(), label, i,
                   (double) static_cast<T>(moved[i]), (double) em,
                   (double) static_cast<T>(fixed[i]), (double) ef);
            ++bad;
        }
    }
    if (!bad)
        printf("[%s/%s] OK: %ld elements, geometries %u and ambient coexisting\n",
               variant_name(), label, N, moved_size);
    return bad ? 1 : 0;
}


// ── RANGE OPS ────────────────────────────────────────────────────────────────
// fill_range / copy_range take one decision per CHUNK where operator[] takes one
// per element. Correctness is checked against the element-by-element path on the
// SAME data, so any divergence -- a stale zero counter, a chunk wrongly left
// constant, a boundary off by one -- shows up as a mismatch rather than as a
// slow drift noticed three compression passes later.
template <typename T>
int check_range_ops(const char* label, long N)
{
    using Arr = Array<T>;

    Arr viaRange(static_cast<int>(N));
    Arr viaElem(static_cast<int>(N));

    auto expected = [](long i) -> T
    {
        // Long constant runs (so chunks really do compress) with scattered
        // non-constant values, which is what makes the two paths diverge if the
        // bookkeeping is wrong.
        if ((i / 100) % 3 == 0) return static_cast<T>(0);
        if ((i % 37) == 0)      return static_cast<T>(i % 1000);
        return static_cast<T>(0);
    };

    // 1) whole-array zero fill: the case where a correct implementation must
    //    allocate NOTHING (every chunk set constant outright).
    shrink_array::fill_range(viaRange, 0, static_cast<std::size_t>(N), static_cast<T>(0));
    for (long i = 0; i < N; ++i)
        viaElem[i] = static_cast<T>(0);

    // 2) scattered writes, both paths identical
    for (long i = 0; i < N; ++i)
    {
        const T e = expected(i);
        if (!(e == static_cast<T>(0)))
        {
            viaRange[i] = e;
            viaElem[i]  = e;
        }
    }

    // 3) partial fills straddling chunk boundaries, deliberately unaligned
    const unsigned S = viaRange.chunk_size();
    for (long off : {1L, (long) S - 1, (long) S + 1, (long) (2 * S) + 3})
    {
        const std::size_t a0 = static_cast<std::size_t>(off);
        const std::size_t b0 = static_cast<std::size_t>(off) + 5 * S + 7;
        if (b0 >= static_cast<std::size_t>(N))
            continue;
        shrink_array::fill_range(viaRange, a0, b0, static_cast<T>(7));
        for (std::size_t i = a0; i < b0; ++i)
            viaElem[static_cast<long>(i)] = static_cast<T>(7);
    }

    // 4) copy_range from a plain vector -- the generic/overloaded pair in use
    std::vector<T> src(static_cast<std::size_t>(N));
    for (long i = 0; i < N; ++i)
        src[static_cast<std::size_t>(i)] = static_cast<T>((i % 11) ? 0 : i);
    const std::size_t ca = 3, cb = static_cast<std::size_t>(N) - 3;
    shrink_array::copy_range(viaRange, ca, cb, src.begin() + 0);
    printf("[dbg] apres copy: range[3]=%g  S=%u  range[320]=%g elem[320]=%g src[317]=%g\n",
           (double) static_cast<T>(viaRange[3]), viaRange.chunk_size(),
           (double) static_cast<T>(viaRange[320]), (double) static_cast<T>(viaElem[320]),
           (double) src[317]);
    for (std::size_t i = ca; i < cb; ++i)
        viaElem[static_cast<long>(i)] = src[i - ca];

    int bad = 0;
    for (long i = 0; i < N && bad < 5; ++i)
    {
        const T r = static_cast<T>(viaRange[i]);
        const T e = static_cast<T>(viaElem[i]);
        if (!(r == e))
        {
            printf("[%s/%s] MISMATCH at %ld: range=%g element=%g\n",
                   variant_name(), label, i, (double) r, (double) e);
            ++bad;
        }
    }
    if (!bad)
        printf("[%s/%s] OK: %ld elements, range ops match the element path\n",
               variant_name(), label, N);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    __mem_page_size = getpagesize();

    // Scale repro: N and the 64->target geometry are configurable, to hunt the
    // size-dependent rechunk crash under ASan without the perf bench's weight.
    //   argv[1] = N (default 1e6)   argv[2] = target size (default 1024)
    const long     N_scale = argc > 1 ? atol(argv[1]) : 1000000L;
    const unsigned tgt     = argc > 2 ? (unsigned) atoi(argv[2]) : 1024u;
    const int      rounds  = argc > 3 ? atoi(argv[3]) : 200;

    ParamChunk::init_indexes(true);

    printf("=== rechunk_validation [variant = %s]  N_scale=%ld  64->%u ===\n",
           variant_name(), N_scale, tgt);

    int fails = 0;

    // ── SCALE CASE: start at 64, grow to tgt (reproduces the perf-bench crash) ──
    set_active_chunk_size(64);
    set_thread_chunk_size(64);
    // Declare the geometry to the allocators. set_active_chunk_size() only moves
    // the ARRAY-side geometry: without this, arrays report chunk_size()==64 but
    // draw their buffers from whatever size init_indexes() primed, and every read
    // past that width returns another chunk's data.
    ParamChunk::register_chunk_size(64);
    // Coexistence first: it is the cheapest and the one the bench actually hits.
    fails += check_coexistence<double>("coexistence-64-and-256", 200000L, 256u, 3);

    fails += check_range_ops<double>("range-ops", 200000L);

    fails += check_invariance<double>("scale-64-to-tgt", N_scale, tgt,
        [](long i)->double { return double((i / 5000) % 3); }, /*sv=*/2.0, /*post_rounds=*/rounds);

    // ── existing 512-start checks (unchanged) ──
    set_active_chunk_size(512);
    set_thread_chunk_size(512);
    ParamChunk::register_chunk_size(512); // see the note above the 64 registration

    const long N = 1000000;

    // Each case declares a second_value present in its data (value -> val2
    // compression actually kicks in). On chunk2 this is REQUIRED; on 2A / v1 it's a no-op.
    fails += check_invariance<double>("mostly-constant", N, 2048,
        [](long i)->double { return double((i / 5000) % 3); }, /*sv=*/2.0);
    fails += check_invariance<double>("sin-quantized", N, 4096,
        [](long i)->double { return std::floor(std::sin(double(i)*0.001)*4.0); }, /*sv=*/1.0);
    fails += check_invariance<double>("shrink-to-smaller", N, 256,
        [](long i)->double { return double((i/1000)%2); }, /*sv=*/1.0);
    // 4096, not 8192: rechunk_to() refuses anything above cst_buffer_size
    // (CHUNK_DYN_MAX_SIZE, 4096 by default) because the shared cst_val1/cst_val2
    // buffers are sized to that ceiling and a wider chunk would read them out of
    // bounds. Asking for 8192 made this case fail with a bare
    // "geometry 512 -> 512": the explanatory message in rechunk_to is gated on
    // build_config::any_check (is_debug || check_optim), which the -DCHECK
    // targets do NOT satisfy, so the refusal was silent. Raise the target here
    // only together with -DCHUNK_DYN_MAX_SIZE.
    fails += check_invariance<double>("all-same", N, 4096,
        [](long)->double { return 42.0; }, /*sv=*/42.0);
    fails += check_invariance<double>("all-different", 100000, 1024,
        [](long i)->double { return double(i); }, /*sv=*/1.0);

    // Growth 512 -> 2048 with val2: targets the tab_m_ptr_cst_val2 queue
    // (chunk2) + the second_value round-trip.
    fails += check_invariance<double>("val2-grow", N, 2048,
        [](long i)->double { return double((i / 5000) % 3); }, /*sv=*/2.0);
    // Shrink with val2 (queue never read past S', must pass).
    fails += check_invariance<double>("val2-shrink", N, 256,
        [](long i)->double { return double((i / 1000) % 2); }, /*sv=*/1.0);

    printf("\n[%s] %s\n", variant_name(),
           fails==0 ? "=== ALL INVARIANCE CHECKS PASSED ===" : "=== INVARIANCE FAILURES ===");
    return fails;
}
