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

// Parallel multi-threaded writes to a compressed array, written as for a
// std::vector: plain `#pragma omp parallel for` loops, no parallel_phase().
// The same function fills a std::vector<double> and an zero_any::Array<double>
// from many threads at once; exit status 0 when the two read back the same,
// element by element.
//
// OMPT builds only (Clang, oneAPI: CHUNK_OPT_OMPT). There the OpenMP runtime
// tells the library where each parallel region begins and ends, through the
// OMPT tool the library carries, so the loops need nothing of the library's
// around them. With GCC's libgomp (no OMPT) the phases would never be marked:
// write the regions with parallel_phase(), as getting_started.cpp does.

#include <utility> // std::as_const
#include <vector>  // std::vector, the reference

#include "chunk/zero_any_array.h"  // shrink_array::zero_any::Array
#include "chunk/config.h"         // build_config::has_ompt
#include "chunk/internal/print.h" // shrink_array::println

namespace sa = shrink_array;
using shrink_array::println;

static_assert(sa::build_config::not_use_openmp || sa::build_config::has_ompt,
              "getting_started_ompt needs a library built with the OMPT tool (Clang or oneAPI, OMP=ON); "
              "without OMPT, open the regions with parallel_phase() (getting_started.cpp)");

// Written once for both containers, run by all the threads of the region: the
// first eighth of the array gets varied values (those chunks keep a buffer),
// the rest of the first half one constant (those chunks compress), the second
// half is never written (compressed from the start).
template <typename Array>
static void fill(Array& t, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n / 2; ++i)
    {
        t[i] = i < n / 8 ? 1.0 + static_cast<double>(i % 7) : 3.5;
    }
}

int main()
{
    const int                   n = 1 << 24; // 16 M doubles: 128 MiB as a std::vector
    std::vector<double>         v(n);        // the reference: every element 0.0
    sa::zero_any::Array<double> a(n);        // every element 0.0, no buffer allocated yet

    fill(v, n);
    fill(a, n); // the same parallel loop: no parallel_phase(), no begin/finalize calls

    // Read back in parallel, through a const reference (operator[] then returns a double).
    const auto& ca   = std::as_const(a);
    int         diff = 0;
#pragma omp parallel for reduction(+ : diff)
    for (int i = 0; i < n; ++i)
    {
        if (ca[i] != v[i])
        {
            ++diff;
        }
    }

    println("n = {} elements, {} different from the std::vector, chunks holding a buffer: {:.1f} %", n, diff,
            a.stats());
    return diff == 0 ? 0 : 1;
}
