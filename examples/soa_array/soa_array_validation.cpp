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

// SoaArray<Cell> (chunk/soa_array.h) against a std::vector<Cell>, the same
// writes on both. Cell's members repeat on their own: the material id is
// constant over large regions, the temperature everywhere but in a hot zone,
// the flag mostly false, while the density varies in a quarter of the domain.
// An ArrayChunk2A<Cell> would compress only where the whole Cell repeats; the
// SoaArray compresses each column on its own runs.
//
// Checks, exit status 0 when all hold:
//  - every member of every element reads back as in the std::vector, after
//    parallel writes (parallel_phase), compound ops (+=), set() and a
//    shrink_to_fit();
//  - each column compresses as its own data allows (mat and flag: almost no
//    buffer; rho: at most the varied quarter);
//  - the generated Ref carries T's member names (a compile-time check).

#include <cstdio>   // std::printf
#include <utility>  // std::as_const
#include <vector>   // std::vector, the reference

#include "chunk/chunk_std.h"   // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h" // parallel_phase
#include "chunk/soa_array.h"   // shrink_array::SoaArray

#if !SHRINK_ARRAY_HAS_SOA_ARRAY
    #error "soa_array_validation needs C++26 static reflection (GCC 16, -freflection): CMake builds it only then"
#endif

namespace sa = shrink_array;

struct Cell
{
    double rho;
    double temp;
    int    mat;
    bool   flag;
};

// The Ref of a[i] has Cell's member names, in Cell's order.
static_assert(std::meta::identifier_of(
                  std::meta::nonstatic_data_members_of(^^sa::SoaArray<Cell>::Ref, std::meta::access_context::unchecked())[2])
              == "mat");

// What element i holds after the parallel write.
static Cell expected(int i, int n)
{
    Cell c{};
    c.rho  = i < n / 4 ? 1.0 + 0.001 * static_cast<double>(i % 1000) : 1.0; // varied in the first quarter
    c.temp = (i >= n / 2 && i < n / 2 + n / 16) ? 900.0 : 300.0;            // a hot zone
    c.mat  = i < n / 3 ? 1 : (i < 2 * (n / 3) ? 2 : 3);                       // three material regions
    c.flag = (i % (n / 8)) == 0;                                              // a few isolated flags
    return c;
}

static bool same(const Cell& a, const Cell& b)
{
    return a.rho == b.rho && a.temp == b.temp && a.mat == b.mat && a.flag == b.flag;
}

int main()
{
    const int n = 1 << 22;
    int       bad = 0;
    {
        std::vector<Cell>  v(n);
        sa::SoaArray<Cell> a(n);

        // Parallel write: member by member through the generated Ref, and the
        // whole struct through set().
        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int i = 0; i < n; ++i)
                {
                    const Cell& c = expected(i, n);
                    v[i] = c;
                    if (i % 2 == 0)
                    {
                        a[i].rho  = c.rho;
                        a[i].temp = c.temp;
                        a[i].mat  = c.mat;
                        a[i].flag = c.flag;
                    }
                    else
                    {
                        a.set(i, c);
                    }
                }
            });

        // A compound op on one member, in another region.
        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int i = n / 2; i < n / 2 + n / 16; ++i)
                {
                    a[i].temp += 100.0;
                    v[i].temp += 100.0;
                }
            });

        a.shrink_to_fit(); // sequential point, every column

        const auto& ca = std::as_const(a);
#pragma omp parallel for reduction(+ : bad)
        for (int i = 0; i < n; ++i)
        {
            if (!same(ca[i], v[i]))
            {
                ++bad;
            }
        }

        // Per column: the share of chunks that hold a buffer.
        std::printf("n = %d elements, %d wrong\n", n, bad);
        a.for_each_column([](std::string_view name, const auto& col)
                          { std::printf("  column %-5.*s %5.1f %% of chunks hold a buffer\n",
                                        static_cast<int>(name.size()), name.data(), col.stats()); });

        const double mat_buf  = a.column<^^Cell::mat>().stats();
        const double flag_buf = a.column<^^Cell::flag>().stats();
        const double rho_buf  = a.column<^^Cell::rho>().stats();
        if (mat_buf > 1.0 || flag_buf > 1.0 || rho_buf > 26.0)
        {
            std::printf("compression below expectations: mat %.1f %%, flag %.1f %%, rho %.1f %%\n", mat_buf,
                        flag_buf, rho_buf);
            ++bad;
        }
    } // the columns release their chunks here

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
