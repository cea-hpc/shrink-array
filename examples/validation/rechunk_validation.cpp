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
//
// Each case runs twice by default (--mode both): SERIALLY, as it always did,
// then with its writes and checks in parallel regions (--mode parallel) --
// the candidate lists per rank, the deferred path and the ambient chunk size
// of worker threads around a re-chunk, which the serial pass never reaches.
// rechunk_to() itself is always called outside any region.
//
//   rechunk_validation [N [target [rounds]]] [--case NAME] [--mode serial|parallel|both] [--list]
//
// --case runs one case: CMake registers one test per case, so `ctest -j`
// spreads them over the cores (the cases share global re-chunk state -- the
// ambient chunk size, the registries -- so they run in separate processes,
// never side by side in one).

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <print>
#include <functional>
#include <string>
#include <vector>
#include <cmath>
#include <type_traits>
#include <unistd.h>

#include "chunk/chunk_traits.h" // forward decls + default ind_type = int
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/config.h"                     // chunk_config::use_chunk*
#include "chunk/array_chunk_range_ops.h"      // fill_range / copy_range
#include "chunk/manage_ompt.h"                // finalize_parallel_phase (compression backlog drain)
#include "chunk/internal/env_parse.h"         // env_number / arg_number (checked, not atoi)

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
void drain_compression_backlog()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
}

// ── Serial or parallel passes ────────────────────────────────────────────────
bool g_parallel = false; ///< --mode: the pass being run

/// @brief f(i) for i in [0, N): a plain loop, or a work-shared parallel
///        region (static schedule) followed by the phase-end drain.
template <typename F>
void for_each_index(long N, F&& f)
{
    if (!g_parallel)
    {
        for (long i = 0; i < N; ++i)
            f(i);
        return;
    }
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
            shrink_array::begin_parallel_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < N; ++i)
            f(i);
    }
    drain_compression_backlog();
}

/// @brief How many i in [0, N) satisfy bad(i) -- the checks, in the same
///        serial or parallel pass. Details are then printed by a serial scan.
template <typename F>
long count_if_index(long N, F&& bad)
{
    long n = 0;
    if (!g_parallel)
    {
        for (long i = 0; i < N; ++i)
            n += bad(i) ? 1 : 0;
        return n;
    }
#pragma omp parallel reduction(+ : n)
    {
        if constexpr (build_config::has_not_ompt)
            shrink_array::begin_parallel_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < N; ++i)
            n += bad(i) ? 1 : 0;
    }
    drain_compression_backlog();
    return n;
}

/// @brief The case label with the pass it runs in.
std::string pass_label(const char* label)
{
    return std::string(label) + (g_parallel ? "@par" : "@ser");
}

} // namespace

// --- Active variant selection (same shape as shrink_bench.cpp) -------
static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "rechunk_validation must be built with CHUNK, CHUNK2 or CHUNK2A "
              "(re-chunk is chunk-only; no std::vector fallback here)");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
        return "chunk2a";
    else if constexpr (chunk_config::use_chunk2)
        return "chunk2";
    else
        return "chunk";
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
int check_invariance(const char* label,
                     long        N,
                     unsigned    new_size,
                     T (*gen)(long),
                     T   sv_value    = T(1),
                     int post_rounds = 0)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    Array<T> arr(N);
    maybe_set_second_value(arr, sv_value); // no-op on chunk v1; required on chunk2

    fprintf(stderr, "[%s/%s] arr built: chunk_size=%u (g_active should be 512)\n", variant_name(), label,
            arr.chunk_size());

    std::vector<T> ref(N);
    for_each_index(N,
                   [&](long i)
                   {
                       T v    = gen(i);
                       arr[i] = v;
                       ref[i] = v;
                   });

    const unsigned old_size = arr.chunk_size();

    // (a) pre-rechunk DIAGNOSTIC: is the content read at the old geometry correct?
    const long pre_bad   = count_if_index(N, [&](long i) { return arr[i] != ref[i]; });
    long       pre_first = -1;
    for (long i = 0; pre_bad && pre_first < 0 && i < N; ++i)
        if (arr[i] != ref[i])
            pre_first = i;
    if (pre_bad)
    {
        std::println("[{}/{}] PRE-FAIL: {} mismatches BEFORE rechunk (setup/read bug, not rechunk); "
                     "first at i={} (chunk #{}, offset {}) got={:g} expected={:g}",
                     variant_name(), label, pre_bad, pre_first, pre_first / (long)old_size, pre_first % (long)old_size,
                     (double)arr[pre_first], (double)ref[pre_first]);
        return 1;
    }

    arr.rechunk_to(new_size);
    const unsigned got_size = arr.chunk_size();

    if (got_size != new_size)
    {
        std::println("[{}/{}] FAIL: geometry {} -> {} (expected {})", variant_name(), label, old_size, got_size,
                     new_size);
        return 1;
    }
    if ((long)arr.size() != N)
    {
        std::println("[{}/{}] FAIL: size {} -> {}", variant_name(), label, N, (long)arr.size());
        return 1;
    }
    const long mismatches = count_if_index(N, [&](long i) { return arr[i] != ref[i]; });
    long       first      = -1;
    for (long i = 0, shown = 0; mismatches && shown < 5 && i < N; ++i)
    {
        if (arr[i] != ref[i])
        {
            if (first < 0)
                first = i;
            std::println("[{}/{}] MISMATCH i={} got={:.17g} exp={:.17g} (new-chunk #{}, offset {})", variant_name(),
                         label, i, (double)arr[i], (double)ref[i], i / (long)new_size, i % (long)new_size);
            ++shown;
        }
    }
    if (mismatches)
    {
        std::println("[{}/{}] FAIL: {}/{} mismatches after rechunk {}->{} (first at i={})", variant_name(), label,
                     mismatches, N, old_size, new_size, first);
        return 1;
    }
    // ── POST-RECHUNK PHASE: rewrite + compress in rounds, mimicking the bench.
    //    The perf-bench crash appears HERE (writes/compression touching the
    //    freshly re-chunked geometry), not in rechunk_to itself. Each round
    //    overwrites with changing data, forces a compression pass, and re-checks
    //    invariance against the ground-truth ref.
    for (int round = 0; round < post_rounds; ++round)
    {
        for_each_index(N,
                       [&](long i)
                       {
                           T v    = gen(i + (long)round * 7919);
                           arr[i] = v;
                           ref[i] = v;
                       });
        // Hand the queued candidates to the compression pass before asking for
        // the result -- see drain_compression_backlog() above.
        drain_compression_backlog();
        // compress_int signature differs across variants: v1 (CHUNK) takes a
        // verbose bool; CHUNK2 / CHUNK2A take none.
        if constexpr (requires { arr.compress_int(false); })
            arr.compress_int(false);
        else
            arr.compress_int();
        const long bad    = count_if_index(N, [&](long i) { return arr[i] != ref[i]; });
        long       firstb = -1;
        for (long i = 0; bad && firstb < 0 && i < N; ++i)
            if (arr[i] != ref[i])
                firstb = i;
        if (bad)
        {
            std::println(
                "[{}/{}] POST-FAIL round {}: {}/{} mismatches after compress (first i={}, new-chunk #{} off {})",
                variant_name(), label, round, bad, N, firstb, firstb / (long)new_size, firstb % (long)new_size);
            return 1;
        }
    }
    std::println("[{}/{}] OK: {} elements invariant, geometry {}->{} (second_value={:.17g}){}", variant_name(), label,
                 N, old_size, new_size, (double)sv_value, post_rounds ? " [+post-rechunk compress rounds]" : "");
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

    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    Arr moved(static_cast<int>(N)); // will be re-chunked
    Arr fixed(static_cast<int>(N)); // stays at the ambient geometry

    for_each_index(N,
                   [&](long i)
                   {
                       moved[i] = static_cast<T>((i % 7) ? 0 : i);
                       fixed[i] = static_cast<T>((i % 5) ? 0 : i);
                   });

    moved.rechunk_to(moved_size);

    // Alternate writes between the two geometries, the way the bench's loop does.
    for (int r = 0; r < rounds; ++r)
        for_each_index(N,
                       [&](long i)
                       {
                           moved[i] = static_cast<T>((i + r) % 11 ? 0 : (i + r));
                           fixed[i] = static_cast<T>((i + r) % 13 ? 0 : (i + r));
                       });

    // Verify both arrays still read back what was last written.
    const int  r = rounds - 1;
    const long nbad =
        count_if_index(N,
                       [&](long i)
                       {
                           return !(static_cast<T>(moved[i]) == static_cast<T>((i + r) % 11 ? 0 : (i + r))) ||
                                  !(static_cast<T>(fixed[i]) == static_cast<T>((i + r) % 13 ? 0 : (i + r)));
                       });
    int bad = 0;
    for (long i = 0; nbad && i < N && bad < 5; ++i)
    {
        const T em = static_cast<T>((i + r) % 11 ? 0 : (i + r));
        const T ef = static_cast<T>((i + r) % 13 ? 0 : (i + r));
        if (!(static_cast<T>(moved[i]) == em) || !(static_cast<T>(fixed[i]) == ef))
        {
            std::println("[{}/{}] MISMATCH at {}: moved={:g} exp {:g} | fixed={:g} exp {:g}", variant_name(), label, i,
                         (double)static_cast<T>(moved[i]), (double)em, (double)static_cast<T>(fixed[i]), (double)ef);
            ++bad;
        }
    }
    if (!bad)
        std::println("[{}/{}] OK: {} elements, geometries {} and ambient coexisting", variant_name(), label, N,
                     moved_size);
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

    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    Arr viaRange(static_cast<int>(N));
    Arr viaElem(static_cast<int>(N));

    auto expected = [](long i) -> T
    {
        // Long constant runs (so chunks really do compress) with scattered
        // non-constant values, which is what makes the two paths diverge if the
        // bookkeeping is wrong.
        if ((i / 100) % 3 == 0)
            return static_cast<T>(0);
        if ((i % 37) == 0)
            return static_cast<T>(i % 1000);
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
    for (long off : {1L, (long)S - 1, (long)S + 1, (long)(2 * S) + 3})
    {
        const auto a0 = static_cast<std::size_t>(off);
        const auto b0 = static_cast<std::size_t>(off) + (5 * static_cast<std::size_t>(S)) + 7;
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
    for (std::size_t i = ca; i < cb; ++i)
        viaElem[static_cast<long>(i)] = src[i - ca];

    const long nbad =
        count_if_index(N, [&](long i) { return !(static_cast<T>(viaRange[i]) == static_cast<T>(viaElem[i])); });
    int bad = 0;
    for (long i = 0; nbad && i < N && bad < 5; ++i)
    {
        const T r = static_cast<T>(viaRange[i]);
        const T e = static_cast<T>(viaElem[i]);
        if (!(r == e))
        {
            std::println("[{}/{}] MISMATCH at {}: range={:g} element={:g}", variant_name(), label, i, (double)r,
                         (double)e);
            ++bad;
        }
    }
    if (!bad)
        std::println("[{}/{}] OK: {} elements, range ops match the element path", variant_name(), label, N);
    return bad ? 1 : 0;
}

namespace
{
/// @brief One case: its name (--case), the geometry it starts from, its body.
struct Case
{
    const char*          name;
    unsigned             start_size;
    std::function<int()> run;
};

/// @brief Declare @p size as the ambient geometry, to the arrays AND to the
///        allocators. set_active_chunk_size() only moves the ARRAY-side
///        geometry: without the registration, arrays report chunk_size() ==
///        size but draw their buffers from whatever size init_indexes()
///        primed, and every read past that width returns another chunk's data.
void set_start_geometry(unsigned size)
{
    set_active_chunk_size(size);
    set_thread_chunk_size(size);
    ParamChunk::register_chunk_size(size);
}
} // namespace

int main(int argc, char** argv)
{
    __mem_page_size = getpagesize();

    // Positional arguments (the scale repro: N and the 64->target geometry are
    // configurable, to hunt a size-dependent rechunk crash under ASan without
    // the perf bench's weight):  [N (1e6)] [target size (1024)] [rounds (200)]
    // Flags: --case NAME, --mode serial|parallel|both (both), --list.
    std::vector<const char*> pos;
    const char*              only = nullptr;
    const char*              mode = "both";
    bool                     list = false;
    for (int a = 1; a < argc; ++a)
    {
        if (!std::strcmp(argv[a], "--case") && a + 1 < argc)
            only = argv[++a];
        else if (!std::strcmp(argv[a], "--mode") && a + 1 < argc)
            mode = argv[++a];
        else if (!std::strcmp(argv[a], "--list"))
            list = true;
        else
            pos.push_back(argv[a]);
    }
    const long     N_scale = pos.size() > 0 ? shrink_array::arg_number<long>(pos[0], 1000000L, "N") : 1000000L;
    const unsigned tgt     = pos.size() > 1 ? shrink_array::arg_number<unsigned>(pos[1], 1024u, "target size") : 1024u;
    const int      rounds  = pos.size() > 2 ? shrink_array::arg_number<int>(pos[2], 200, "rounds") : 200;
    const bool     run_ser = std::strcmp(mode, "parallel") != 0;
    const bool     run_par = std::strcmp(mode, "serial") != 0;
    if (!run_ser && !run_par)
    {
        std::println("unknown --mode '{}' (serial, parallel or both)", mode);
        return 2;
    }

    const long N = 1000000;
    // Each case declares a second_value present in its data (value -> val2
    // compression actually kicks in). On chunk2 this is REQUIRED; on 2A / v1
    // it is a no-op. The first three start at 64 (the scale case grows it to
    // the target, as the perf bench does), the others at 512.
    const std::vector<Case> cases = {
        // Coexistence first: it is the cheapest and the one the bench actually hits.
        {"coexistence", 64, [] { return check_coexistence<double>("coexistence-64-and-256", 200000L, 256u, 3); }},
        {"range-ops", 64, [] { return check_range_ops<double>("range-ops", 200000L); }},
        {"scale-64-to-tgt", 64,
         [&]
         {
             return check_invariance<double>(
                 "scale-64-to-tgt", N_scale, tgt, [](long i) -> double { return double((i / 5000) % 3); },
                 /*sv_value=*/2.0, /*post_rounds=*/rounds);
         }},
        {"mostly-constant", 512,
         [&]
         {
             return check_invariance<double>(
                 "mostly-constant", N, 2048, [](long i) -> double { return double((i / 5000) % 3); },
                 /*sv_value=*/2.0);
         }},
        {"sin-quantized", 512,
         [&]
         {
             return check_invariance<double>(
                 "sin-quantized", N, 4096,
                 [](long i) -> double { return std::floor(std::sin(double(i) * 0.001) * 4.0); }, /*sv_value=*/1.0);
         }},
        {"shrink-to-smaller", 512,
         [&]
         {
             return check_invariance<double>(
                 "shrink-to-smaller", N, 256, [](long i) -> double { return double((i / 1000) % 2); },
                 /*sv_value=*/1.0);
         }},
        // 4096, not 8192: rechunk_to() refuses anything above cst_buffer_size
        // (CHUNK_DYN_MAX_SIZE, 4096 by default) because the shared cst_val1/cst_val2
        // buffers are sized to that ceiling and a wider chunk would read them out of
        // bounds. Asking for 8192 made this case fail with a bare
        // "geometry 512 -> 512": the explanatory message in rechunk_to is gated on
        // build_config::any_check (is_debug || check_optim), which the -DCHECK
        // targets do NOT satisfy, so the refusal was silent. Raise the target here
        // only together with -DCHUNK_DYN_MAX_SIZE.
        {"all-same", 512,
         [&]
         {
             return check_invariance<double>(
                 "all-same", N, 4096, [](long) -> double { return 42.0; },
                 /*sv_value=*/42.0);
         }},
        {"all-different", 512,
         []
         {
             return check_invariance<double>(
                 "all-different", 100000, 1024, [](long i) -> double { return double(i); }, /*sv_value=*/1.0);
         }},
        // Growth 512 -> 2048 with val2: targets the tab_m_ptr_cst_val2 queue
        // (chunk2) + the second_value round-trip.
        {"val2-grow", 512,
         [&]
         {
             return check_invariance<double>(
                 "val2-grow", N, 2048, [](long i) -> double { return double((i / 5000) % 3); }, /*sv_value=*/2.0);
         }},
        // Shrink with val2 (queue never read past S', must pass).
        {"val2-shrink", 512,
         [&]
         {
             return check_invariance<double>(
                 "val2-shrink", N, 256, [](long i) -> double { return double((i / 1000) % 2); },
                 /*sv_value=*/1.0);
         }},
    };

    if (list)
    {
        for (const Case& c : cases)
            std::println("{}", c.name);
        return 0;
    }
    if (only)
    {
        bool known = false;
        for (const Case& c : cases)
            known = known || !std::strcmp(c.name, only);
        if (!known)
        {
            std::println("unknown --case '{}' (--list gives the names)", only);
            return 2;
        }
    }

    ParamChunk::init_indexes(true);

    std::println("=== rechunk_validation [variant = {}]  N_scale={}  64->{}  case={}  mode={} ===", variant_name(),
                 N_scale, tgt, only ? only : "all", mode);

    int      fails    = 0;
    unsigned geometry = 0;
    for (const Case& c : cases)
    {
        if (only && std::strcmp(c.name, only) != 0)
            continue;
        if (c.start_size != geometry)
        {
            set_start_geometry(c.start_size);
            geometry = c.start_size;
        }
        if (run_ser)
        {
            g_parallel = false;
            fails += c.run();
        }
        if (run_par)
        {
            g_parallel = true;
            fails += c.run();
        }
    }

    std::println("\n[{}] {}", variant_name(),
                 fails == 0 ? "=== ALL INVARIANCE CHECKS PASSED ===" : "=== INVARIANCE FAILURES ===");
    return fails;
}
