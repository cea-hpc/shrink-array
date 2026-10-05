// ============================================================================
// Thread-growth harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property: the OpenMP team may grow after the arrays are created.
//
// Every per-rank structure of the library (chunk allocators, compression-
// candidate lists, deferred-op maps, writer and flush flags) is indexed by the
// thread rank and sized once. It used to be sized by omp_get_max_threads() at
// that moment, so an omp_set_num_threads() raising the team afterwards indexed
// past the end: a silent out-of-bounds access, or a writer the compressor did
// not wait for. They are now sized for the thread capacity
// (shrink_array::chunk_thread_capacity(): max(omp_get_max_threads(),
// omp_get_num_procs()), or CHUNK_THREAD_CAPACITY), and a rank past it is an
// invariant failure.
//
// What it does:
//   1. creates the arrays under a 2-thread team;
//   2. raises the team to the capacity and runs several regions of writes,
//      compound operations, zero runs (compression candidates) and reads of
//      elements other threads wrote, checked against a std::vector;
//   3. re-runs itself with `--beyond`, which raises the team PAST the
//      capacity, and checks that the child stops with the invariant failure
//      message rather than running on.
// In a serial build there is no team to grow: step 2 runs single-threaded and
// step 3 is skipped.
// ============================================================================

#include <cstdio>      // std::printf, std::fopen, std::remove
#include <cstdlib>     // std::system
#include <string>      // std::string, std::to_string
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector
#include <unistd.h>    // getpid
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS

#include "chunk/chunk_traits.h"  // forward decls + default ind_type = int
#include "chunk/array_chunk.h"   // ArrayChunk
#include "chunk/array_chunk2.h"  // ArrayChunk2
#include "chunk/array_chunk2A.h" // ArrayChunk2A
#include "chunk/param_chunk.h"   // ParamChunk::init_indexes
#include "chunk/config.h"        // chunk_config::use_chunk2*, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"     // chunk_pmp_set_num_threads, CHUNK_PMP_THREAD_CAPACITY
#include "chunk/manage_ompt.h"   // begin_parallel_phase / finalize_parallel_phase

#include "chunk/chunk_std.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::thread_growth_validation
{

using namespace shrink_array;

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

static void set_team(int n)
{
    chunk_pmp_set_num_threads(n);
}

/// Runs of 2048 elements alternate zero / varied, so each region leaves
/// constant chunks for the passes and materialised ones for the writes.
static bool zero_run(long i, int round)
{
    return ((i / 2048) + round) % 3 == 0;
}

/// Regions under the current team: writes, compound operations, zero runs and
/// cross-thread reads (element i reads element N-1-i, written by another
/// thread under the static schedule), on the array and on the reference.
static long run_regions(Array<int>& a, std::vector<int>& ra, Array<double>& d, std::vector<double>& rd, int rounds)
{
    const long N = static_cast<long>(ra.size());
    for (int round = 0; round < rounds; ++round)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < N; ++i)
            {
                if (zero_run(i, round))
                {
                    a[i] = 0;
                    d[i] = 0.0;
                    continue;
                }
                a[i] = static_cast<int>(((i * 7) + round) % 97) + 1;
                a[i] += 3;
                a[i] *= 2;
                a[i] %= 101;
                d[i] = static_cast<double>(i % 53) + (0.5 * round);
                d[i] *= 1.5;
            }
        }
        end_phase();
        for (long i = 0; i < N; ++i)
        {
            if (zero_run(i, round))
            {
                ra[i] = 0;
                rd[i] = 0.0;
                continue;
            }
            ra[i] = static_cast<int>(((i * 7) + round) % 97) + 1;
            ra[i] += 3;
            ra[i] *= 2;
            ra[i] %= 101;
            rd[i] = static_cast<double>(i % 53) + (0.5 * round);
            rd[i] *= 1.5;
        }

        // Cross-thread reads: a[i] += a[N-1-i] (a snapshot of the reference
        // first, so both sides read the values before this region).
        const std::vector<int> snap = ra;
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < N / 2; ++i)
            {
                a[i] += static_cast<int>(a[N - 1 - i]);
            }
        }
        end_phase();
        for (long i = 0; i < N / 2; ++i)
        {
            ra[i] += snap[N - 1 - i];
        }
    }

    long fails = 0;
    for (long i = 0; i < N; ++i)
    {
        const int    ai = a[i];
        const double di = d[i];
        if (ai != ra[i] || di != rd[i])
        {
            if (fails < 5)
            {
                std::printf("  MISMATCH i=%ld  int %d (ref %d)  double %g (ref %g)\n", i, ai, ra[i], di, rd[i]);
            }
            ++fails;
        }
    }
    return fails;
}

/// Re-runs this executable with --beyond and checks that it stopped on the
/// capacity invariant.
[[maybe_unused]] static int check_beyond(const char* self) // OpenMP builds only
{
    const std::string log    = "/tmp/thread_growth_beyond_" + std::to_string(getpid()) + ".log";
    const std::string cmd    = validation::self_command(self) + " --beyond > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    std::string       out;
    if (FILE* f = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        {
            out.append(buf, n);
        }
        std::fclose(f);
    }
    std::remove(log.c_str());

    const bool stopped = status != -1 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
    const bool message = out.find("thread capacity") != std::string::npos;
    std::printf("[beyond] team past the capacity: %s, %s\n", stopped ? "stopped" : "RAN ON",
                message ? "capacity message printed" : "NO capacity message");
    if (!stopped || !message)
    {
        std::printf("---- child output ----\n%s----------------------\n", out.c_str());
        return 1;
    }
    return 0;
}

static int run_body(int argc, char** argv)
{
    const bool beyond = argc > 1 && std::string(argv[1]) == "--beyond";

    // 1. The arrays are created under a 2-thread team.
    set_team(2);
    ParamChunk::init_indexes(true);

    const long          N = 1L << 16;
    Array<int>          a(N);
    Array<double>       d(N);
    std::vector<int>    ra(N, 0);
    std::vector<double> rd(N, 0.0);

    long fails = run_regions(a, ra, d, rd, 1);

    // 2. The team grows: to the capacity, or past it with --beyond.
    const int capacity = CHUNK_PMP_THREAD_CAPACITY;
    const int grown    = beyond ? capacity + 4 : capacity;
    set_team(grown);
    std::printf("=== thread_growth_validation [variant = %s]  created at 2 threads, now %d (capacity %d)%s ===\n",
                variant_name(), grown, capacity, beyond ? "  --beyond" : "");
    std::fflush(stdout);

    fails += run_regions(a, ra, d, rd, 4);

    if (beyond)
    {
        std::printf("[beyond] the team ran past the capacity without an invariant failure\n");
        return 1;
    }

    // 3. Past the capacity: must stop with the invariant failure.
    if constexpr (build_config::use_openmp)
    {
        fails += check_beyond(argv[0]);
    }

    if (fails)
    {
        std::printf("=== THREAD GROWTH FAILED (%ld) ===\n", fails);
        return 1;
    }
    std::printf("=== THREAD GROWTH OK ===\n");
    return 0;
}

int run(int argc, char** argv)
{
    // The arrays live in run_body() and are destroyed when it returns, which
    // parks their buffers in the reuse lists; release_chunk_pools() frees them.
    // Without it they stay parked until exit, and LeakSanitizer reports every
    // one in the std-alloc builds (each buffer is a real malloc there).
    const int rc = run_body(argc, argv);
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}

} // namespace validation::thread_growth_validation
