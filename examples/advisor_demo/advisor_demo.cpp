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

// Step-0 validation harness for ChunkSizeAdvisor: build the SAME data at
// several REAL chunk sizes -- powers of two and multiples of 64 (192, 384,
// ...) -- then compare the advisor's per-size predictions (each set made from
// ONE reference size: 256, then 1024) against ground truth at each size.
#include <format>   // std::format
#include <iterator> // std::size
#include <vector>   // std::vector
#include <print>    // std::println

#include "chunk/array_chunk2A.h"              // ArrayChunk2A
#include "chunk/chunk2A_std.h"                // Chunk2A
#include "chunk/chunk_size_advisor.h"         // ChunkSizeAdvisor
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/param_chunk.h"                // ParamChunk::init_indexes

using namespace shrink_array;


static constexpr int N = 1 << 22; // 4M elements

// Mixed, size-sensitive pattern:
//  [0,50%)   : zeros                          -> cst1 at every size
//  [50,75%)  : constant 3.5                   -> cst2 at every size
//  [75,87.5%): 256-long alternating constants -> cst at S<=256, MAT above
//  [87.5,100%): pseudo-random                 -> MAT at every size
//  plus: one exceptional value every 4096 elements inside the zero region
//        -> pollutes 1 chunk in 64 at S=64, 1 in 4 at S=1024...
static double value_at(int i)
{
    const double f = static_cast<double>(i) / N;
    if (f < 0.5)
    {
        return (i % 4096 == 2048) ? 42.0 : 0.0;
    }
    if (f < 0.75)
    {
        return 3.5;
    }
    if (f < 0.875)
    {
        return ((i / 256) % 2) ? 7.25 : -1.5;
    }
    return 0.001 * ((i * 2654435761u) % 1000);
}

template <typename TAB>
static void fill_and_compress(TAB& tab)
{
    for (int i = 0; i < N; ++i)
    {
        tab[i] = value_at(i);
    }
    TAB::make_block_compression();
}

struct Truth
{
    long long chunks, mat;
    double    mem;
};

static Truth ground_truth(const ArrayChunk2A<double>& tab, double meta)
{
    Truth t{tab.nb_chunk, 0, 0.};
    for (long long j = 0; j < t.chunks; ++j)
    {
        const auto& c = tab.chunk_list[j];
        if (!c.is_cst_val1() && !c.is_cst_val2())
        {
            ++t.mat;
        }
    }
    t.mem = (static_cast<double>(t.chunks) * meta) + (double(t.mat) * tab.chunk_size() * sizeof(double));
    return t;
}

int main()
{
    std::println("=== ChunkSizeAdvisor step-0 validation (N={} doubles) ===", N);

    // Mandatory library init: the block size and the allocator indexes for
    // the first size. Element types register themselves at their first array.
    set_active_chunk_size(64);
    ParamChunk::init_indexes(true);

    // Ground truth at every real size
    // The powers of two, and the multiples of 64 the advisor's grid evaluates
    // from 256 and 1024 (Options::grid_bits = 3): misaligned with the reference
    // chunks, these rows are the ones that read data across chunk boundaries.
    const unsigned     sizes[] = {64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048};
    std::vector<Truth> truths;
    double             meta = 0.;
    for (const unsigned cs : sizes)
    {
        set_thread_chunk_size(cs);
        ArrayChunk2A<double> tab(1); // proven pattern (test_multisize):
        tab.set_chunk_size(cs);      // register per-array size FIRST,
        tab.resize(N);               // THEN size the geometry
        fill_and_compress(tab);
        if (meta == 0.)
        {
            const ChunkSizeAdvisor<double>::Options o;
            meta = sizeof(Chunk2A<double>) + o.meta_extra_bytes;
        }
        truths.push_back(ground_truth(tab, meta));
    }

    // Advisor predictions against ground truth, from two reference sizes:
    //  - 256: finer 128/64, coarser 512/1024, and the grid 192, 320, ...;
    //  - 1024: finer 512/256, coarser 2048, and the grid 384, 640, 1280, ... There the 256-long runs at
    //    -1.5 / 7.25 are materialised, so the finer rows must recognise them
    //    as compressible -- Chunk2A compresses a uniform chunk at ANY value.
    //    Gated on first_value/second value, the S=256 row read 4608
    //    materialised chunks against 2560 real (+80%).
    //  - 256 again, reading the data of 300 materialised chunks only (of
    //    ~2560): the sampled rows are extrapolated, within the 10% tolerance.
    struct Ref
    {
        unsigned ref_size;
        long     max_scans;
    };
    int fail      = 0;
    int grid_rows = 0; // rows at a size that is not a power of two, checked
    for (const Ref run : {Ref{256u, 65536}, Ref{1024u, 65536}, Ref{256u, 300}})
    {
        const unsigned ref_size = run.ref_size;
        set_thread_chunk_size(ref_size);
        ArrayChunk2A<double> ref(1);
        ref.set_chunk_size(ref_size);
        ref.resize(N);
        fill_and_compress(ref);
        ChunkSizeAdvisor<double>::Options opt;
        opt.coarser_levels = 2;
        opt.finer_levels   = 2;
        opt.max_mat_scans  = run.max_scans;
        opt.grid_bits      = 3; // the finest grid: 320, 448, 640... too
        const auto rep     = ChunkSizeAdvisor<double>::analyze(ref, opt);
        ChunkSizeAdvisor<double>::print(rep,
                                        std::format("ref S={}, {} scans at most", ref_size, run.max_scans).c_str());

        std::println("\n── prediction (from S={}) vs ground truth ──", ref_size);
        for (const auto& e : rep.table)
        {
            // find matching real size
            for (size_t k = 0; k < std::size(sizes); ++k)
            {
                if (sizes[k] != e.size)
                {
                    continue;
                }
                const auto&  t  = truths[k];
                const double dm = t.mat ? 100. * double(e.mat - t.mat) / double(t.mat) : (e.mat ? 100. : 0.);
                const bool   ok = e.exact ? (e.mat == t.mat && e.n_chunks == t.chunks) : (dm > -10. && dm < 10.);
                std::println(
                    "   S={:5} : pred mat={:9} {} | real mat={:9} | d={:+6.2f}%  mem pred={:8.2f} real={:8.2f} Mo  {}",
                    e.size, (long long)e.mat, e.exact ? "(exact)" : "(sampl)", t.mat, dm, e.mem_bytes / 1048576.,
                    t.mem / 1048576., ok ? "OK" : "FAIL");
                fail += !ok;
                grid_rows += (e.size & (e.size - 1)) != 0 ? 1 : 0;
            }
        }
        ref.clear();
    }
    if (grid_rows == 0)
    {
        std::println("\nFAIL: no prediction at a size that is not a power of two");
        ++fail;
    }
    std::println("\n=== {} ===", fail ? "ADVISOR VALIDATION FAILED" : "ADVISOR VALIDATED");
    return fail;
}
