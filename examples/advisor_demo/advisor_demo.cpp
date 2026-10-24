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
#include <cstdio>   // std::snprintf
#include <iterator> // std::size
#include <vector>   // std::vector

#include "chunk/zero_any_array.h"             // zero_any::Array
#include "chunk/zero_any_chunk.h"             // zero_any
#include "chunk/chunk_size_advisor.h"         // ChunkSizeAdvisor
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/param_chunk.h"                // ParamChunk::init_indexes
#include "chunk/internal/print.h"             // shrink_array::println

using namespace shrink_array;


static constexpr int n_elems = 1 << 22; // 4M elements

// Mixed, size-sensitive pattern:
//  [0,50%)   : zeros                          -> cst1 at every size
//  [50,75%)  : constant 3.5                   -> cst2 at every size
//  [75,87.5%): 256-long alternating constants -> cst at S<=256, MAT above
//  [87.5,100%): pseudo-random                 -> MAT at every size
//  plus: one exceptional value every 4096 elements inside the zero region
//        -> pollutes 1 chunk in 64 at S=64, 1 in 4 at S=1024...
static double value_at(int index)
{
    const double frac = static_cast<double>(index) / n_elems;
    if (frac < 0.5)
    {
        return (index % 4096 == 2048) ? 42.0 : 0.0;
    }
    if (frac < 0.75)
    {
        return 3.5;
    }
    if (frac < 0.875)
    {
        return ((index / 256) % 2) ? 7.25 : -1.5;
    }
    return 0.001 * ((index * 2654435761u) % 1000);
}

template <typename Tab>
static void fill_and_compress(Tab& tab)
{
    for (int i = 0; i < n_elems; ++i)
    {
        tab[i] = value_at(i);
    }
    Tab::make_block_compression();
}

struct Truth
{
    long long chunks, mat;
    double    mem;
};

static Truth ground_truth(const zero_any::Array<double>& tab, double meta)
{
    Truth truth{tab.nb_chunk_, 0, 0.};
    for (long long j = 0; j < truth.chunks; ++j)
    {
        const auto& chunk = tab.chunk_list_[j];
        if (!chunk.is_cst_val1() && !chunk.is_cst_val2())
        {
            ++truth.mat;
        }
    }
    truth.mem = (static_cast<double>(truth.chunks) * meta) + (double(truth.mat) * tab.chunk_size() * sizeof(double));
    return truth;
}

int main()
{
    println("=== ChunkSizeAdvisor step-0 validation (n_elems={} doubles) ===", n_elems);

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
        zero_any::Array<double> tab(1); // proven pattern (test_multisize):
        tab.set_chunk_size(cs);         // register per-array size FIRST,
        tab.resize(n_elems);            // THEN size the geometry
        fill_and_compress(tab);
        if (meta == 0.)
        {
            const ChunkSizeAdvisor<double>::Options opts;
            meta = sizeof(zero_any::Chunk<double>) + opts.meta_extra_bytes;
        }
        truths.push_back(ground_truth(tab, meta));
    }

    // Advisor predictions against ground truth, from two reference sizes:
    //  - 256: finer 128/64, coarser 512/1024, and the grid 192, 320, ...;
    //  - 1024: finer 512/256, coarser 2048, and the grid 384, 640, 1280, ... There the 256-long runs at
    //    -1.5 / 7.25 are materialised, so the finer rows must recognise them
    //    as compressible -- zero_any compresses a uniform chunk at ANY value.
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
        zero_any::Array<double> ref(1);
        ref.set_chunk_size(ref_size);
        ref.resize(n_elems);
        fill_and_compress(ref);
        ChunkSizeAdvisor<double>::Options opt;
        opt.coarser_levels = 2;
        opt.finer_levels   = 2;
        opt.max_mat_scans  = run.max_scans;
        opt.grid_bits      = 3; // the finest grid: 320, 448, 640... too
        const auto rep     = ChunkSizeAdvisor<double>::analyze(ref, opt);
        char       title[48];
        std::snprintf(title, sizeof title, "ref S=%u, %ld scans at most", ref_size, run.max_scans);
        ChunkSizeAdvisor<double>::print(rep, title);

        println("\n── prediction (from S={}) vs ground truth ──", ref_size);
        for (const auto& entry : rep.table)
        {
            // find matching real size
            for (size_t k = 0; k < std::size(sizes); ++k)
            {
                if (sizes[k] != entry.size)
                {
                    continue;
                }
                const auto&  truth = truths[k];
                const double dm =
                    truth.mat ? 100. * double(entry.mat - truth.mat) / double(truth.mat) : (entry.mat ? 100. : 0.);
                const bool ok =
                    entry.exact ? (entry.mat == truth.mat && entry.n_chunks == truth.chunks) : (dm > -10. && dm < 10.);
                println(
                    "   S={:5} : pred mat={:9} {} | real mat={:9} | d={:+6.2f}%  mem pred={:8.2f} real={:8.2f} Mo  {}",
                    entry.size, (long long)entry.mat, entry.exact ? "(exact)" : "(sampl)", truth.mat, dm,
                    entry.mem_bytes / 1048576., truth.mem / 1048576., ok ? "OK" : "FAIL");
                fail += !ok;
                grid_rows += (entry.size & (entry.size - 1)) != 0 ? 1 : 0;
            }
        }
        ref.clear();
    }
    if (grid_rows == 0)
    {
        println("\nFAIL: no prediction at a size that is not a power of two");
        ++fail;
    }
    println("\n=== {} ===", fail ? "ADVISOR VALIDATION FAILED" : "ADVISOR VALIDATED");
    return fail;
}
