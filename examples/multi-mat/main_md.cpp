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

/**
 * @file main_md.cpp
 * @brief multi-mat-md: multi-mat on a 2D mesh, through mdspan views.
 *
 * The mock of multi-mat (main.cpp) with an nx x ny mesh: at every iteration
 * each material occupies a random box, where its three fields and its
 * presence fraction are written and then read back by the gadget
 * computation. Every array is seen through a 2D mdspan (chunk/mdspan.h for
 * the chunk arrays, std mdspan over the data of the std::vectors), so one
 * loop body serves both storages (ComputeLoopsMd.h, LoopsMd.h).
 *
 * The chunk materials are laid out in tiles of one chunk (16 x 16 for a
 * chunk of 256) and walked tile by tile by default; --chunk-mode elements
 * keeps the tiles and walks cell by cell, --chunk-mode rows lays them out
 * row-major. The run prints the chunks written, the result, the time and
 * the peak memory; --verify replays it serially and compares.
 *
 * Usage: multi-mat-md [--nx N] [--ny N] [--mat N] [--iter N] [--option 1-4]
 *        [--chunk-rate P] [--pc-chunk P] [--pc-vector P] [--freq-rtz N]
 *        [--freq-compress N] [--interface N] [--chunk-mode tiles|elements|rows]
 *        [--seed S] [--verify] [--verbose]
 * Exit 0; 1 when --verify finds a difference; 2 on a bad argument; 77 when
 * no mdspan is available (ctest: skipped).
 */

#include <chrono>    // std::chrono::steady_clock
#include <cmath>     // std::round, std::sqrt
#include <cstdio>    // stderr
#include <cstdlib>   // std::abort
#include <print>     // std::println
#include <random>    // std::mt19937, std::uniform_int_distribution, std::random_device
#include <string>    // std::string, std::stoi, std::stod
#include <utility>   // std::as_const
#include <stdexcept> // std::exception (a bad number)

#include <sys/resource.h> // getrusage

#include "chunk/internal/chunk_size_config.h" // static_chunk_size
#include "chunk/param_chunk.h"                // ParamChunk

#include "config.h"         // multi_mat::max_threads, embedded_compression
#include "ComputeLoopsMd.h" // SimuMd::computeLoops, verify (SHRINK_ARRAY_HAS_MDSPAN)
#include "Dispatch.h"       // Dispatch

using namespace shrink_array;

/// @brief The storage of every material, read by the containers.
Dispatch dispatch;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace multi_mat::md
{

bool SimuMd::parse(int argc, char** argv)
{
    for (int a = 1; a < argc; ++a)
    {
        const std::string key = argv[a];
        const auto        val = [&]() -> std::string
        {
            if (a + 1 >= argc)
            {
                throw std::invalid_argument(key + " needs a value");
            }
            return argv[++a];
        };
        try
        {
            if (key == "--nx")
            {
                nx = std::stoi(val());
            }
            else if (key == "--ny")
            {
                ny = std::stoi(val());
            }
            else if (key == "--mat")
            {
                nbmat = std::stoi(val());
            }
            else if (key == "--iter")
            {
                iter_max = std::stoi(val());
            }
            else if (key == "--option")
            {
                option_chunk = std::stoi(val());
            }
            else if (key == "--chunk-rate")
            {
                chunk_rate = std::stoi(val());
            }
            else if (key == "--pc-chunk")
            {
                pc_chunk = std::stod(val());
            }
            else if (key == "--pc-vector")
            {
                pc_vector = std::stod(val());
            }
            else if (key == "--freq-rtz")
            {
                freq_rtz = std::stoi(val());
            }
            else if (key == "--freq-compress")
            {
                freq_compress = std::stoi(val());
            }
            else if (key == "--interface")
            {
                interface_width = std::stoi(val());
            }
            else if (key == "--seed")
            {
                seed = static_cast<unsigned>(std::stoul(val()));
            }
            else if (key == "--verify")
            {
                check = true;
            }
            else if (key == "--verbose")
            {
                verbose = true;
            }
            else if (key == "--chunk-mode")
            {
                const std::string m = val();
                if (m == "tiles")
                {
                    chunk_mode = ChunkMode::tiles;
                }
                else if (m == "elements")
                {
                    chunk_mode = ChunkMode::elements;
                }
                else if (m == "rows")
                {
                    chunk_mode = ChunkMode::rows;
                }
                else
                {
                    throw std::invalid_argument("--chunk-mode: tiles, elements or rows, not " + m);
                }
            }
            else
            {
                throw std::invalid_argument("unknown argument " + key);
            }
        }
        catch (const std::exception& e)
        {
            std::println(stderr, "multi-mat-md: {}", e.what());
            return false;
        }
    }
    const bool ok = nx > 0 && ny > 0 && nbmat > 0 && iter_max > 0 && option_chunk >= 1 && option_chunk <= 4 &&
                    freq_rtz > 0 && freq_compress > 0 && interface_width >= 0 && pc_chunk > 0 && pc_chunk <= 100 &&
                    pc_vector > 0 && pc_vector <= 100 &&
                    static_cast<long long>(storageSize()) <= static_cast<long long>(1 << 30);
    if (!ok)
    {
        std::println(stderr, "multi-mat-md: a parameter is out of range");
    }
    return ok;
}

std::size_t SimuMd::storageSize() const
{
    return chunk_mode == ChunkMode::rows ? static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
                                         : tiled_storage(nx, ny);
}

void SimuMd::drawStorage()
{
    std::mt19937 gen(seed != 0 ? seed : std::random_device{}());
    dis.assign(nbmat, option_chunk == 1);
    if (option_chunk >= 3)
    {
        std::uniform_int_distribution<> distrib(0, option_chunk == 3 ? 1 : 100);
        const int                       threshold = option_chunk == 3 ? 1 : chunk_rate;
        for (int m = 0; m < nbmat; ++m)
        {
            dis[m] = distrib(gen) < threshold;
        }
    }
    int nb_chunk = 0;
    for (const bool c : dis)
    {
        nb_chunk += c ? 1 : 0;
    }
    std::println("Number of chunk materials : {} on {}", nb_chunk, nbmat);
}

void SimuMd::createTabs()
{
    const int size = static_cast<int>(storageSize());
    for (auto& t : tabs)
    {
        t.create(nbmat, size);
    }
    frac.create(nbmat, size);
}

void SimuMd::computeDistribution()
{
    // The largest box of a kind covers pc % of the mesh's area: its sides are
    // sqrt(pc %) of the mesh's. Sides and corners uniform, as the intervals
    // of multi-mat.
    const auto side = [](int n, double pc)
    { return std::max(1, static_cast<int>(std::round(static_cast<double>(n) * std::sqrt(pc / 100.)))); };
    const int hc = side(nx, pc_chunk);
    const int wc = side(ny, pc_chunk);
    const int hv = side(nx, pc_vector);
    const int wv = side(ny, pc_vector);
    std::println("Largest box : chunk {} x {}, vector {} x {}", hc, wc, hv, wv);

    std::mt19937 rng(seed != 0 ? seed + 1 : std::random_device{}());
    box_store.resize(static_cast<std::size_t>(iter_max) * static_cast<std::size_t>(nbmat));
    const auto bx = mdns::mdspan(box_store.data(), mdns::dextents<int, 2>(iter_max, nbmat));
    for (int it = 0; it < iter_max; ++it)
    {
        for (int mat = 0; mat < nbmat; ++mat)
        {
            const bool c = dis[mat];
            const int  h = std::uniform_int_distribution(1, c ? hc : hv)(rng);
            const int  w = std::uniform_int_distribution(1, c ? wc : wv)(rng);
            Box b{std::uniform_int_distribution(0, nx - h)(rng), std::uniform_int_distribution(0, ny - w)(rng), h, w};
            bx[it, mat] = b;
            if (verbose)
            {
                std::println("Iteration {} material #{} : rows [{}, {}) columns [{}, {})", it, mat, b.i0, b.i1(), b.j0,
                             b.j1());
            }
        }
    }
}

/// @brief The chunk filling rates (written chunks, %) and the real occupation
///        of the chunk materials, averaged.
template <typename MM>
void print_stats(const SimuMd& s, const MM& mm_algo)
{
    double fields = 0;
    double frac   = 0;
    double real   = 0;
    int    n      = 0;
    mm_algo.loop(0, s.nbmat,
                 [&](int mat, const auto& get_mat_vec, const auto& loops)
                 {
                     if (!dispatch.isChunk(mat))
                     {
                         return;
                     }
                     const auto& t0 = get_mat_vec(s.tabs[0]);
                     using V        = std::remove_cvref_t<decltype(t0)>;
                     if constexpr (std::is_same_v<V, VectChunk<double>>)
                     {
                         fields += (t0.stats() + get_mat_vec(s.tabs[1]).stats() + get_mat_vec(s.tabs[2]).stats()) / 3;
                         frac += get_mat_vec(s.frac).stats();
                         const auto c0 = loops.view(std::as_const(t0), s.nx, s.ny);
                         long       nz = 0;
    #pragma omp parallel for reduction(+ : nz)
                         for (int i = 0; i < s.nx; ++i)
                         {
                             for (int j = 0; j < s.ny; ++j)
                             {
                                 const double v = c0[i, j];
                                 nz += v != 0 ? 1 : 0;
                             }
                         }
                         real += 100. * static_cast<double>(nz) / (static_cast<double>(s.nx) * s.ny);
                         ++n;
                     }
                 });
    if (n != 0)
    {
        std::println("\nChunk materials ({}), averages:", n);
        std::println("  fields (v1), chunks written     : {:.2f}%", fields / n);
        std::println("  fraction (Chunk2), chunks written : {:.2f}%", frac / n);
        std::println("  real filling (non-zero cells)   : {:.2f}%", real / n);
    }
}

/// @brief The run with the policies of @p mm_algo: the iterations in one
///        parallel region, then the stats and the replay.
template <typename MM>
int run(SimuMd& s, const MM& mm_algo, const char* chunk_policy)
{
    std::println("Chunk policy      : {}", chunk_policy);
    std::println("Vector policy     : {}\n", LoopsMdVector::name);

    auto t0 = std::chrono::steady_clock::now();
    s.createTabs();
    std::println("Elapsed time create arrays : {:g}s",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    double value = 0;
    t0           = std::chrono::steady_clock::now();
    #pragma omp parallel reduction(+ : value)
    {
        s.computeLoops(mm_algo, value);
    }
    multi_mat::end_parallel_phase_after_region(); // manual executable only
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    print_stats(s, mm_algo);
    std::println("\nResult = {:.6e}", value);
    std::println("\nElapsed time  : {:.2f}s (initialisation {:.2f}s, writes {:.2f}s, reads {:.2f}s, compression passes "
                 "{:.2f}s)\n",
                 elapsed, s.time_init, s.time_write, s.time_read, s.time_compress);

    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        std::println("Memory consumption : {:.2f} Mo\n", static_cast<double>(usage.ru_maxrss) / 1024.);
    }

    int rc = 0;
    if (s.check)
    {
        rc = s.verify(mm_algo, value) == 0 ? 0 : 1;
    }

    if constexpr (multi_mat::chunk_stats)
    {
        if (s.option_chunk != 2)
        {
            ParamChunk::ChunkStats();
        }
    }
    return rc;
}

} // namespace multi_mat::md

int main(int argc, char** argv)
{
    using namespace multi_mat::md;

    SimuMd simu;
    if (!simu.parse(argc, argv))
    {
        return 2;
    }

    constexpr int chunk_size = static_cast<int>(shrink_array::static_chunk_size);
    if constexpr (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
    {
        std::println(stderr, "The chunk size must be in [{}, {}]", ParamChunk::size_min, ParamChunk::size_max);
        std::abort();
    }

    std::println("Number of threads : {}", multi_mat::max_threads());
    std::println("Compression       : {}", multi_mat::embedded_compression
                                               ? "embedded (OMPT tool of manage_ompt.cpp, no call from here)"
                                               : "manual (explicit compress() on the field arrays)");
    std::println("Mesh              : {} x {} ({} cells), {} materials, {} iterations", simu.nx, simu.ny,
                 static_cast<long>(simu.nx) * simu.ny, simu.nbmat, simu.iter_max);
    std::println("Tile              : {} x {} (chunk size {})", tile_shape[0], tile_shape[1], chunk_size);

    ParamChunk::init_indexes(true);

    simu.drawStorage();
    dispatch.create(simu.nbmat);
    dispatch.set(simu.dis);
    simu.computeDistribution();

    switch (simu.chunk_mode)
    {
        case ChunkMode::tiles:
            return run(simu, MultiMatAlgo<LoopsMdVector, LoopsMdTiles>{}, LoopsMdTiles::name);
        case ChunkMode::elements:
            return run(simu, MultiMatAlgo<LoopsMdVector, LoopsMdTilesElements>{}, LoopsMdTilesElements::name);
        case ChunkMode::rows:
            return run(simu, MultiMatAlgo<LoopsMdVector, LoopsMdRows>{}, LoopsMdRows::name);
    }
    return 2;
}

#else

int main()
{
    std::println("multi-mat-md: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
