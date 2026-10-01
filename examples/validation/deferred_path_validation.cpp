// ============================================================================
// Deferred-write path harness. Variant-agnostic: chunk / chunk2 / chunk2a by
// the compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property: a write that the engine refuses on the spot and PARKS
// in the writer's rank queue (to_be_queued_for_deferred_processing) lands in
// program order with every other write and read of the same element -- the
// direct writes before and after it, the read-modify-writes that need its
// value, the reads from another array, the replays at the P3 drain, at the
// read gate (reader_sync), at the end of the region, and the WAW guard that
// retires a superseded op before a direct store.
//
// WHY A SEPARATE HARNESS
//
//   Since the per-chunk marks and the bounded writer wait, a bench or a fuzz
//   replay defers next to nothing (`gate deferred = 0` in the stats): the
//   deferred path runs only when a compressor holds the chunk at the very
//   instant of a write, which the marks made a microsecond-sized window. The
//   code is live, it is just never exercised. This executable is built with
//   CHUNK_FORCE_DEFER_EVERY (see config.h): one write in N per thread takes
//   the deferred exit with no compressor in sight, alternating check #1 and
//   check #2, so the path runs a few hundred thousand times in a second on
//   an ordinary laptop, at 1 thread and at 8.
//
// WHAT IT EXERCISES
//
//   The operator chains of shrink_bench's compute phase, verbatim in shape:
//   plain assignment, compound arithmetic (+= -= *= /=), compound bitwise
//   (|= ^= <<= >>=), pre/post increment and decrement, "x = f(x)" read-then-
//   write, the constant written onto a constant chunk over a pending op (the
//   early-outs before the engine), reads of a second array written in the
//   previous region, a compression pass between regions (compress() by the
//   whole team, as the bench does it) so that deferred ops, replays and the
//   compressor coexist, and one region of three loops separated by plain
//   barriers with no phase bracket: reads of elements another thread wrote
//   before the barrier while its ops are still queued (the foreign-pending
//   read gate), and writes under a remapped schedule over elements whose ops
//   sit in another rank's queue (the cross-rank WAW drain). The same chain is
//   applied, by the same template, to a std::vector reference; every element
//   must compare equal at the end of each region. int, double and bool.
//
// WHAT IT DOES NOT COVER
//
//   Re-chunking and resize (sequential quiescent points: they drain first),
//   the array copy-assignment (a shallow alias of the chunks, no concurrent
//   contract), geometry sweeps. One geometry per executable.
//
// Exit 0 = every element matches the reference after every region, and, in a
// CHUNK_STATS build with the knob on, at least one write really was deferred.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <vector>
#include <unistd.h>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/chunk_std.h" // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"
#include "chunk/internal/env_parse.h" // env_number / arg_number (checked, not atoi)
#ifdef CHUNK_STATS
    #include "chunk/internal/chunk_stats.h"
#endif

int __mem_page_size = 0;

using namespace shrink_array;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::begin_parallel_phase();
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
}

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

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
        a.set_second_value(v);
}

// Runs of kRun elements alternate constant 0 / second constant / varied, so
// that the pass between regions has blocks to collapse.
constexpr long kRun = 2048;

// ── The chains, one per type, applied to the array AND to the reference ─────
// Every read of the container goes through T(c[i]) -- the proxy's conversion
// -- so the expression is the same one whether c is an Array<T> or a vector.

template <typename C>
static void chain_int(C& c, long i)
{
    using T = int;
    c[i]    = static_cast<T>(i & 0xF);
    c[i] |= 0x1;
    c[i] ^= 0x3;
    c[i] <<= 2;
    c[i] >>= 1;
    c[i] = (T(c[i]) & 0xF) | 0x2;
    c[i] %= 5; // operator%=: no other test uses it
    c[i] = ~T(c[i]);
    c[i] = 1;
    if (i % 3 == 0)
        c[i] = (T(c[i]) == 1) ? 2 : 0;
    ++c[i];
    c[i]--;
    c[i] += 5;
    c[i] -= 2;
    c[i] *= 3;
    c[i] = -T(c[i]);
    c[i] = std::abs(T(c[i]));
    if (i % 5 == 0)
        c[i] = 0; // a run of the first constant, here and there
}

template <typename C>
static void chain_double(C& c, long i)
{
    using T = double;
    c[i]    = static_cast<T>(i % 97) + 0.5;
    c[i] += 1.25;
    c[i] *= 2.0;
    c[i] -= 0.75;
    c[i] /= 4.0;
    c[i] = std::floor(T(c[i]) * 3.0);
    c[i] = T(c[i]) + 1.0;
    if (i % 4 == 0)
        c[i] = 0.0;
    else if (i % 4 == 1)
        c[i] = 3.5; // the second constant of the scenario
}

template <typename C>
static void chain_bool(C& c, long i)
{
    c[i] = (i % 3) == 0;
    c[i] = !bool(c[i]);
    c[i] = bool(c[i]) || (i % 7 == 0);
    c[i] = bool(c[i]) && (i % 11 != 0);
    if (i % 5 == 0)
        c[i] = true;
    if (i % 13 == 0)
        c[i] = false;
}

// "x[i] op= other[j]": the read of `other` is the read gate on an array whose
// ops were deferred -- in the previous region, or by another thread of this
// one when j is not i.
template <typename T, typename C, typename O>
static void mix_other(C& c, const O& o, long j, long i)
{
    if constexpr (std::is_same_v<T, bool>)
        c[i] = bool(c[i]) != bool(o[j]);
    else if constexpr (std::is_integral_v<T>)
        c[i] += T(o[j]) & 0x7;
    else
        c[i] += T(o[j]) * 0.5;
}

template <typename T>
struct Scenario
{
    T second;
    void (*chain)(Array<T>&, long);
    void (*chain_ref)(std::vector<T>&, long);
};

template <typename T>
static long compare(const char* label, const char* stage, const Array<T>& arr, const std::vector<T>& ref)
{
    const long N     = static_cast<long>(ref.size());
    long       bad   = 0;
    long       first = -1;
    T          got_first{};
    T          exp_first{};
    for (long i = 0; i < N; ++i)
    {
        const T got = arr[i];
        if (got != ref[i])
        {
            if (first < 0)
            {
                first     = i;
                got_first = got;
                exp_first = ref[i];
            }
            ++bad;
        }
    }
    if (bad)
    {
        const long csz = static_cast<long>(arr.chunk_size());
        printf("[%s/%s] FAIL after %s: %ld/%ld mismatches, first at i=%ld (chunk #%ld, offset %ld) "
               "got=%g expected=%g\n",
               variant_name(), label, stage, bad, N, first, first / csz, first % csz, (double)got_first,
               (double)exp_first);
    }
    return bad;
}

/**
 * @brief Run the chains on the array and on the reference, region by region,
 *        with a compression pass in the middle; compare after each region.
 * @return 0 if every element matched every time, 1 otherwise.
 */
template <typename T>
static int run(const char* label, long N, const Scenario<T>& sc)
{
    Array<T> arr(N);
    Array<T> other(N);
    maybe_set_second_value(arr, sc.second);
    maybe_set_second_value(other, sc.second);
    std::vector<T> ref(N);
    std::vector<T> ref_other(N);

    long fails = 0;

    // Region 1: fill both arrays with the run pattern. Every element is first
    // written a varied value, then the run's: on a still-constant chunk whose
    // first write was deferred, the second write is the constant the chunk
    // already holds -- the early-out before the engine, which must not fire
    // while that older op is pending.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
        {
            const long run = (i / kRun) % 3;
            const T    v   = run == 0 ? T(0) : run == 1 ? sc.second : static_cast<T>((i % 23) + 1);
            arr[i]         = static_cast<T>((i % 23) + 1);
            arr[i]         = v;
            other[i]       = static_cast<T>(i % 13);
        }
    }
    end_phase();
    for (long i = 0; i < N; ++i)
    {
        const long run = (i / kRun) % 3;
        ref[i]         = run == 0 ? T(0) : run == 1 ? sc.second : static_cast<T>((i % 23) + 1);
        ref_other[i]   = static_cast<T>(i % 13);
    }
    fails += compare(label, "fill", arr, ref);
    fails += compare(label, "fill(other)", other, ref_other);

    // Region 2: the chains, every element.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
            sc.chain(arr, i);
    }
    end_phase();
    for (long i = 0; i < N; ++i)
        sc.chain_ref(ref, i);
    fails += compare(label, "chains", arr, ref);

    // A pass between regions, the way the bench drives it: compress() by
    // every thread of the region (it workshares its scan).
#pragma omp parallel
    {
        begin_phase();
        arr.compress();
        other.compress();
    }
    end_phase();
    fails += compare(label, "compress", arr, ref);

    // Region 3: mix in the other array (its reads go through the gate), then
    // the chains again on the odd elements, so writes land on blocks the pass
    // just compressed and on blocks it left alone.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
        {
            mix_other<T>(arr, other, i, i);
            if (i & 1)
                sc.chain(arr, i);
        }
    }
    end_phase();
    for (long i = 0; i < N; ++i)
    {
        mix_other<T>(ref, ref_other, i, i);
        if (i & 1)
            sc.chain_ref(ref, i);
    }
    fails += compare(label, "mix+chains", arr, ref);

    // Region 4: one region, three loops, two plain barriers and NO phase
    // bracket around them. Loop A leaves ops pending in every rank's queue;
    // loop B, under another schedule, reads elements loop A wrote from OTHER
    // threads (the foreign-pending read gate, P1); loop C, same remapped
    // schedule, writes over elements whose loop-A ops sit in another rank's
    // queue (the cross-rank WAW drain before a direct write). The reference
    // runs the three loops in sequence, which is what the barriers promise.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < N; ++i)
            sc.chain(other, i);
        // (the omp for's own barrier)
#pragma omp for schedule(static, 37)
        for (long i = 0; i < N; ++i)
            mix_other<T>(arr, other, (i + (N / 2)) % N, i);
#pragma omp for schedule(static, 37)
        for (long i = 0; i < N; ++i)
            if (i % 3 != 1)
                sc.chain(other, i);
    }
    end_phase();
    for (long i = 0; i < N; ++i)
        sc.chain_ref(ref_other, i);
    for (long i = 0; i < N; ++i)
        mix_other<T>(ref, ref_other, (i + (N / 2)) % N, i);
    for (long i = 0; i < N; ++i)
        if (i % 3 != 1)
            sc.chain_ref(ref_other, i);
    fails += compare(label, "barriers", arr, ref);
    fails += compare(label, "barriers(other)", other, ref_other);

    // Region 5: `other` rewritten from `arr` -- the gate on arr this time.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
            other[i] = T(arr[i]);
    }
    end_phase();
    for (long i = 0; i < N; ++i)
        ref_other[i] = ref[i];
    fails += compare(label, "copy(other)", other, ref_other);

    if (!fails)
        printf("[%s/%s] OK: %ld elements, chunk_size=%ld\n", variant_name(), label, N,
               static_cast<long>(arr.chunk_size()));

    arr.clear();
    other.clear();
    return fails ? 1 : 0;
}

int main(int argc, char** argv)
{
    __mem_page_size = getpagesize();

    // 128 Ki elements: some 60 chunks per run at the default geometry, a
    // million deferred writes per run of the executable, and a Debug build
    // still under 20 s at 1 thread (each deferred op is a context, a map
    // entry and a replay -- forty times the cost of the write it stands for).
    const long N = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 17, "N") : (1L << 17);

    ParamChunk::init_indexes(true);

    printf("=== deferred_path_validation [variant = %s]  N=%ld  threads=%d  force_defer_every=%u ===\n", variant_name(),
           N, CHUNK_PMP_MAX_THREADS, build_config::force_defer_every);

    int fails = 0;

    const Scenario<int>    si{7, &chain_int<Array<int>>, &chain_int<std::vector<int>>};
    const Scenario<double> sd{3.5, &chain_double<Array<double>>, &chain_double<std::vector<double>>};
    const Scenario<bool>   sb{true, &chain_bool<Array<bool>>, &chain_bool<std::vector<bool>>};

    fails += run<int>("int", N, si);
    fails += run<double>("double", N, sd);
    fails += run<bool>("bool", N, sb);

#ifdef CHUNK_STATS
    if constexpr (build_config::force_defer_every > 0)
    {
        long deferred = 0;
        for (int v = 0; v < CHUNK_STATS_N_VARIANT; ++v)
            deferred += reduce_chunk_stats(v).gate_deferred;
        printf("[%s] writes deferred: %ld\n", variant_name(), deferred);
        if (deferred == 0)
        {
            printf("[%s] FAIL: the knob is on and nothing was deferred -- the path was not exercised\n",
                   variant_name());
            ++fails;
        }
    }
#endif

    release_chunk_pools();
    destroy_block_chunk();

    printf("\n[%s] %s\n", variant_name(), fails == 0 ? "=== DEFERRED PATH OK ===" : "=== DEFERRED PATH FAILURES ===");
    return fails;
}
