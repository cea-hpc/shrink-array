// ============================================================================
// Chunk<T>::operator new[] / init_chunks_cst_val harness. Chunk v1 only: it is
// the one variant with a class-level array-new (see chunk_std.h).
// ============================================================================
//
// Guards ONE property: an array of chunks allocated through
// Chunk<T>::operator new[] comes back with EVERY chunk pointing at the first
// constant -- at every size, from a serial context and from inside a parallel
// region, at whatever thread count OMP_NUM_THREADS gives.
//
// WHY A HARNESS FOR THIS
//
//   The operator splits on a threshold (500 chunks) and on whether a parallel
//   region is open, and nothing in the library calls it: the arrays hold a
//   std::vector<chunk_type> and initialise it with their own loops, so a fault
//   here would surface in no other test. Two it would have caught, both found
//   in Sept '26 by a throwaway program:
//
//     - the sizes tested below bracket the threshold. The form before that
//       date read `if (n < 500) ... else if (n > 500)`, and at exactly 500
//       NOTHING was initialised;
//     - the "region" cases call the operator from an `omp single`, which is
//       what an allocation inside a parallel region looks like. Sharing the
//       initialisation with an orphaned `omp for` HANGS there (the caller
//       waits on the loop's barrier, the other threads on the single's); the
//       operator therefore parallelises only when no region is open. A
//       nested `omp parallel` is not an option either -- serialised by every
//       default runtime, and N x N/2 threads from an allocation when nesting
//       is enabled.
//
//   `new Chunk<T>[n]` is NOT one of the cases: the default constructors run
//   after the operator and overwrite what it wrote. The operator pays off for
//   an array built by placement new, which is the last case below.
//
// EXIT STATUS: 0 when every chunk of every case is at the first constant.

#include <cstdio>
#include <cstdlib>
#include <new>
#include <unistd.h>

#include "chunk/chunk_traits.h" // forward decls + default ind_type = int
#include "chunk/array_chunk.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"    // chunk_config::use_chunk
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS

static_assert(chunk_config::use_chunk, "chunk v1 only: operator new[] is Chunk's");
// ChunkOmpPolicy's serial specialisation declares no array-new, so there is
// nothing to test without OpenMP; CMake builds this target under OMP alone.
static_assert(build_config::use_openmp, "OpenMP builds only: the array-new is ChunkOmpPolicy's");

// Defined by every executable of the library (the fast allocator reads it).
int __mem_page_size = 0;

using namespace shrink_array;

namespace
{

using Ch = Chunk<double>;

/// @brief How many of the @p n chunks hold a buffer (none should).
int materialised(const Ch* v, std::size_t n)
{
    int bad = 0;
    for (std::size_t k = 0; k < n; ++k)
        bad += v[k].written();
    return bad;
}

/// @brief One case: allocate @p n chunks, serially or from an `omp single`
///        inside a region, and check every one of them.
int one_case(std::size_t n, bool in_region)
{
    void* p = nullptr;
    if (in_region)
    {
#pragma omp parallel
        {
#pragma omp single
            p = Ch::operator new[](n * sizeof(Ch));
        }
    }
    else
    {
        p = Ch::operator new[](n * sizeof(Ch));
    }

    auto*     v   = static_cast<Ch*>(p);
    const int bad = materialised(v, n);
    printf("[chunk/array-new] %-6s n=%5zu : %s\n", in_region ? "region" : "serial", n,
           bad ? "FAIL" : "ok, every chunk at the first constant");
    if (bad)
        printf("  %d of %zu chunk(s) not initialised\n", bad, n);
    Ch::operator delete[](p);
    return bad ? 1 : 0;
}

/// @brief The operator's real use: an array built by placement new, where no
///        constructor runs to overwrite the initialisation.
int placement_case(std::size_t n)
{
    void* p = Ch::operator new[](n * sizeof(Ch));
    auto* v = static_cast<Ch*>(p);

    const int bad = materialised(v, n);
    printf("[chunk/array-new] placement n=%5zu : %s\n", n,
           bad ? "FAIL" : "ok, every chunk at the first constant");
    Ch::operator delete[](p);
    return bad ? 1 : 0;
}

} // namespace

int main()
{
    __mem_page_size = getpagesize();
    ParamChunk::init_indexes(true);

    printf("=== chunk_array_new_validation  threads=%d ===\n", CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    // 1 and 5000 are the plain cases; 499 / 500 / 501 bracket the threshold.
    for (const std::size_t n : {std::size_t(1), std::size_t(499), std::size_t(500), std::size_t(501),
                                std::size_t(5000)})
    {
        fails += one_case(n, /*in_region=*/false);
        fails += one_case(n, /*in_region=*/true);
    }
    fails += placement_case(1000);

    printf("\n[chunk] %s\n", fails == 0 ? "=== ARRAY-NEW OK ===" : "=== ARRAY-NEW FAILURES ===");
    return fails ? 1 : 0;
}
