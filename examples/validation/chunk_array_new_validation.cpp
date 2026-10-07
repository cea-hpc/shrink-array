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
// operator new[] / init_chunks_cst_val harness. Variant-agnostic: chunk /
// chunk2 / chunk2a follow the compile-time macro, like the other harnesses
// here -- the three carry the same pair of operators, built on the one
// chunk_array_new::init_chunks (chunk_array_new.h).
// ============================================================================
//
// Guards ONE property: an array of chunks allocated through the variant's
// operator new[] comes back with EVERY chunk pointing at the first constant
// -- at every size, from a serial context and from inside a parallel region,
// at whatever thread count OMP_NUM_THREADS gives.
//
// WHY A HARNESS FOR THIS
//
//   The initialisation splits on a threshold (500 chunks) and on whether a
//   parallel region is open, and nothing in the library calls it: the arrays
//   hold a std::vector<chunk_type> and initialise it with their own loops, so
//   a fault here would surface in no other test. Two hazards:
//
//     - the sizes tested below bracket the threshold, where an off-by-one
//       (`if (n < 500) ... else if (n > 500)`) initialises NOTHING at 500;
//     - the "region" cases call the operator from an `omp single`, which is
//       what an allocation inside a parallel region looks like. Sharing the
//       initialisation with an orphaned `omp for` HANGS there (the caller
//       waits on the loop's barrier, the other threads on the single's); the
//       operator therefore parallelises only when no region is open. A
//       nested `omp parallel` is not an option either -- serialised by every
//       default runtime, and N x N/2 threads from an allocation when nesting
//       is enabled.
//
//   `new C[n]` is NOT one of the cases: the default constructors run after the
//   operator and overwrite what it wrote. The operator pays off for an array
//   built by placement new, which is the last case below.
//
// EXIT STATUS: 0 when every chunk of every case is at the first constant.

#include <cstdio>      // printf
#include <type_traits> // std::conditional_t

#include "chunk/chunk_traits.h"  // forward decls + default ind_type = int
#include "chunk/array_chunk.h"   // Chunk<T> complete (chunk_std.h): its operator new[]
#include "chunk/array_chunk2.h"  // Chunk2<T> complete (chunk2_std.h): its operator new[]
#include "chunk/array_chunk2A.h" // Chunk2A<T> complete (chunk2A_std.h): its operator new[]
#include "chunk/param_chunk.h"   // ParamChunk::init_indexes
#include "chunk/config.h"        // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"     // CHUNK_PMP_MAX_THREADS

// The *OmpPolicy serial specialisations declare no array-new, so there is
// nothing to test without OpenMP; CMake builds this target under OMP alone.
static_assert(shrink_array::build_config::use_openmp, "OpenMP builds only: the array-new is the *OmpPolicy's");

// Defined by every executable of the library (the fast allocator reads it).
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::chunk_array_new_validation
{

using namespace shrink_array;

namespace
{

/// The chunk type of this build, as the arrays pick theirs.
using Ch = std::conditional_t<chunk_config::use_chunk2a,
                              Chunk2A<double>,
                              std::conditional_t<chunk_config::use_chunk2, Chunk2<double>, Chunk<double>>>;

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

/// @brief How many of the @p n chunks hold a buffer (none should).
/// @note A TEMPLATE, so that the branch `if constexpr` discards is not
///       type-checked: chunk2's written() takes the second-value index (0 --
///       nothing else is registered here), v1's and chunk2a's take none, and
///       in a plain function both spellings would have to compile.
template <typename C>
int materialised(const C* v, std::size_t n)
{
    int bad = 0;
    for (std::size_t k = 0; k < n; ++k)
    {
        if constexpr (requires { v[k].written(0); })
        {
            bad += v[k].written(0);
        }
        else
        {
            bad += v[k].written();
        }
    }
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

    const auto* v   = static_cast<Ch*>(p);
    const int   bad = materialised(v, n);
    printf("[%s/array-new] %-6s n=%5zu : %s\n", variant_name(), in_region ? "region" : "serial", n,
           bad ? "FAIL" : "ok, every chunk at the first constant");
    if (bad)
    {
        printf("  %d of %zu chunk(s) not initialised\n", bad, n);
    }
    Ch::operator delete[](p);
    return bad ? 1 : 0;
}

/// @brief The operator's real use: an array built by placement new, where no
///        constructor runs to overwrite the initialisation.
int placement_case(std::size_t n)
{
    void*       p = Ch::operator new[](n * sizeof(Ch));
    const auto* v = static_cast<Ch*>(p);

    const int bad = materialised(v, n);
    printf("[%s/array-new] placement n=%5zu : %s\n", variant_name(), n,
           bad ? "FAIL" : "ok, every chunk at the first constant");
    Ch::operator delete[](p);
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    printf("=== chunk_array_new_validation [variant = %s]  threads=%d ===\n", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    // 1 and 5000 are the plain cases; 499 / 500 / 501 bracket the threshold.
    for (const std::size_t n :
         {std::size_t(1), std::size_t(499), std::size_t(500), std::size_t(501), std::size_t(5000)})
    {
        fails += one_case(n, /*in_region=*/false);
        fails += one_case(n, /*in_region=*/true);
    }
    fails += placement_case(1000);

    printf("\n[%s] %s\n", variant_name(), fails == 0 ? "=== ARRAY-NEW OK ===" : "=== ARRAY-NEW FAILURES ===");
    return fails ? 1 : 0;
}

} // namespace validation::chunk_array_new_validation
