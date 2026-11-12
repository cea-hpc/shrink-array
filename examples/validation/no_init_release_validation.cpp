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
// The per-chunk vectors' allocator (parallel_storage_init.h): build, read back
// and release NoInitVector<long> below, around and above its thresholds --
// std::allocator under 1 MiB, mmap + munmap from 1 MiB, the release from a team
// (release_mapped: madvise per slice, then munmap) from 32 MiB outside a
// region. The large ones are released outside a region and inside one (the
// single-thread path there), and the resident memory must fall back after
// them. `--bench` times the release of a 512 MiB vector against a plain munmap
// of the same touched mapping.
// ============================================================================

#include <chrono>     // std::chrono::steady_clock (--bench)
#include <cstddef>    // std::size_t
#include <cstdio>     // std::FILE, std::fopen, std::fscanf
#include <cstring>    // std::strcmp, std::memset
#include <sys/mman.h> // mmap, munmap (--bench reference)
#include <unistd.h>   // sysconf

#include "chunk/internal/parallel_storage_init.h" // NoInitVector, make_filled
#include "chunk/internal/PMP_CHUNK.h"             // pmp::team
#include "chunk/print.h"                          // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::no_init_release_validation
{

using namespace shrink_array;

namespace
{

/// @brief Resident memory of the process, bytes (/proc/self/statm); 0 if unknown.
std::size_t resident_bytes()
{
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr)
    {
        return 0;
    }
    unsigned long size     = 0;
    unsigned long resident = 0;
    const int     got      = std::fscanf(f, "%lu %lu", &size, &resident);
    std::fclose(f);
    return got == 2 ? static_cast<std::size_t>(resident) * static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)) : 0;
}

/// @brief Build a vector of @p bytes, check it, release it (in a region when
///        @p in_region); for 32 MiB and more, check the resident memory fell back.
int one_case(std::size_t bytes, bool in_region)
{
    using parallel_storage_init::NoInitVector;
    const std::size_t count = bytes / sizeof(long);
    const std::size_t rss0  = resident_bytes();
    auto* vec = new NoInitVector<long>(parallel_storage_init::make_filled<long>(count, 7L)); // NOLINT: owned below
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < count; i += 509)
    {
        wrong += (*vec)[i] != 7L ? 1 : 0;
    }
    wrong += (*vec)[count - 1] != 7L ? 1 : 0;
    const std::size_t rss1 = resident_bytes();
    if (in_region)
    {
        pmp::team(pmp::Options{}, [&](pmp::Team& team) { team.masked([&] { delete vec; }); });
    }
    else
    {
        delete vec; // NOLINT(cppcoreguidelines-owning-memory)
    }
    const std::size_t rss2 = resident_bytes();
    // Large: the pages are back -- at most a quarter of the vector still resident.
    const bool large     = bytes >= (std::size_t{32} << 20U);
    const bool rss_known = rss0 != 0 && rss1 > rss2;
    const bool fell_back = !large || !rss_known || rss1 - rss2 >= bytes - bytes / 4;
    const bool ok        = wrong == 0 && fell_back;
    println("[no-init-release] {:>9} KiB {:<14} wrong {}  rss {} -> {} -> {} KiB  {}", bytes >> 10U,
            in_region ? "in a region" : "outside", wrong, rss0 >> 10U, rss1 >> 10U, rss2 >> 10U, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int bench()
{
    using parallel_storage_init::NoInitVector;
    constexpr std::size_t bytes = std::size_t{512} << 20U;
    auto* vec = new NoInitVector<long>(parallel_storage_init::make_filled<long>(bytes / sizeof(long), 1L)); // NOLINT
    auto  t0  = std::chrono::steady_clock::now();
    delete vec; // NOLINT(cppcoreguidelines-owning-memory)
    const double team_ms = ms_since(t0);

    void* raw = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED)
    {
        return 1;
    }
    pmp::parallel_for(0L, static_cast<long>(bytes / 4096), pmp::Options{},
                      [raw](long p) { static_cast<char*>(raw)[static_cast<std::size_t>(p) * 4096] = 1; });
    t0 = std::chrono::steady_clock::now();
    ::munmap(raw, bytes);
    const double munmap_ms = ms_since(t0);
    println("[no-init-release] 512 MiB, {} threads: release from a team {:.1f} ms | plain munmap {:.1f} ms",
            CHUNK_PMP_MAX_THREADS, team_ms, munmap_ms);
    return 0;
}

} // namespace

int run(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        return bench();
    }
    int fails = 0;
    for (const std::size_t kib : {std::size_t{64}, std::size_t{1024}, std::size_t{4096}, std::size_t{32768},
                                  std::size_t{33000}, std::size_t{131072}})
    {
        fails += one_case(kib << 10U, false);
    }
    fails += one_case(std::size_t{65536} << 10U, true);
    println("{}", fails ? "no_init_release_validation: FAILED" : "no_init_release_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::no_init_release_validation
