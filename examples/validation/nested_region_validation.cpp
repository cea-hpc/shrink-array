// ============================================================================
// Nested-region harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property: a thread keeps its own per-rank structures across a
// nested parallel region.
//
// The library indexes its per-thread structures (candidate lists, deferred-op
// queues) with a rank cached in thread-local storage (__fast_omp_rank). That
// cache used to be set at the start of every implicit task and never cleared,
// and to hold omp_get_thread_num() -- the rank in the INNERMOST team. A thread
// of the outer team that entered a nested region, even a serialised one (a
// team of one), came back with rank 0 and shared rank 0's structures with the
// real rank 0, unprotected. The rank is now the thread's rank in the active
// team (omp_get_ancestor_thread_num when nested), the cache is cleared at the
// end of each implicit task, and two ACTIVE levels are an invariant failure.
//
// What it does:
//   A. max-active-levels 1: an outer team of 4 writes the arrays; every thread
//      opens a nested region (serialised by that setting), writes in it, and
//      goes on writing in the outer region after it. Checked against a
//      std::vector; check builds also check each cached rank against the
//      thread's rank in the active team.
//   B. re-runs itself with `--nested-active`: max-active-levels 2 and a
//      2-thread inner team writing the arrays, which must stop with the
//      "nested ACTIVE" invariant failure.
// In a serial build there is no nesting: A runs single-threaded, B is skipped.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>
#include <sys/wait.h> // WIFEXITED, WEXITSTATUS

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"

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

/// The value element i holds after round `round` (zero runs of 2048 so the
/// chunks alternate constant and materialised).
static int expected_before_inner(long i, int round)
{
    return ((i / 2048) + round) % 3 == 0 ? 0 : static_cast<int>((i * 5 + round) % 89) + 1;
}

/// Scenario A (and B's body): each thread of the outer team writes its share,
/// opens a nested region that writes again, then writes after it.
static long run_nested(Array<int>& a, std::vector<int>& ref, int rounds)
{
    const long N = static_cast<long>(ref.size());
    for (int round = 0; round < rounds; ++round)
    {
#pragma omp parallel num_threads(4)
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < N; ++i)
                a[i] = expected_before_inner(i, round);

            // A nested region per thread, over this thread's own slice (the
            // static schedule gives the same slice as above).
#pragma omp for schedule(static)
            for (long b = 0; b < N / 1024; ++b)
            {
#pragma omp parallel num_threads(2)
                {
                    begin_phase();
#pragma omp single
                    for (long i = b * 1024; i < (b + 1) * 1024; ++i)
                        if (i % 7 == 0)
                            a[i] += 3;
                }
                // Back in the outer region: writes go to THIS thread's rank.
                for (long i = b * 1024; i < (b + 1) * 1024; ++i)
                {
                    a[i] *= 2;
                    if (i % 11 == 0)
                        a[i] = 0;
                }
            }
        }
        end_phase();
        for (long i = 0; i < N; ++i)
        {
            ref[i] = expected_before_inner(i, round);
            if (i % 7 == 0)
                ref[i] += 3;
            ref[i] *= 2;
            if (i % 11 == 0)
                ref[i] = 0;
        }
    }

    long fails = 0;
    for (long i = 0; i < N; ++i)
    {
        const int v = a[i];
        if (v != ref[i])
        {
            if (fails < 5)
                std::printf("  MISMATCH i=%ld  %d (ref %d)\n", i, v, ref[i]);
            ++fails;
        }
    }
    return fails;
}

/// Re-runs this executable with --nested-active and checks that it stopped on
/// the nested-active invariant.
[[maybe_unused]] static int check_nested_active(const char* self) // OpenMP builds only
{
    const std::string log    = "/tmp/nested_region_active_" + std::to_string(getpid()) + ".log";
    const std::string cmd    = std::string("\"") + self + "\" --nested-active > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    std::string       out;
    if (FILE* f = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
            out.append(buf, n);
        std::fclose(f);
    }
    std::remove(log.c_str());

    const bool stopped = status != -1 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
    const bool message = out.find("nested ACTIVE") != std::string::npos;
    std::printf("[nested-active] two active levels: %s, %s\n", stopped ? "stopped" : "RAN ON",
                message ? "invariant message printed" : "NO invariant message");
    if (!stopped || !message)
    {
        std::printf("---- child output ----\n%s----------------------\n", out.c_str());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    __mem_page_size          = getpagesize();
    const bool nested_active = argc > 1 && std::string(argv[1]) == "--nested-active";

    chunk_pmp_set_max_active_levels(nested_active ? 2 : 1);
    ParamChunk::init_indexes(true);

    const long       N = 1L << 16;
    Array<int>       a(N);
    std::vector<int> ref(N, 0);

    std::printf("=== nested_region_validation [variant = %s]  outer team 4, nested regions %s ===\n", variant_name(),
                nested_active ? "ACTIVE (--nested-active)" : "serialised");
    std::fflush(stdout);

    long fails = run_nested(a, ref, 3);

    if (nested_active)
    {
        std::printf("[nested-active] two active levels ran without an invariant failure\n");
        return 1;
    }

    if constexpr (build_config::use_openmp)
        fails += check_nested_active(argv[0]);

    if (fails)
    {
        std::printf("=== NESTED REGION FAILED (%ld) ===\n", fails);
        return 1;
    }
    std::printf("=== NESTED REGION OK ===\n");
    return 0;
}
