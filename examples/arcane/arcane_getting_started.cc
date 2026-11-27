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
 * @file arcane_getting_started.cc
 * @brief A compressed array in an Arcane code, from creation to reads: kokkos_getting_started
 *        with Arcane's accelerator API (StandaloneAcceleratorMng, RUNCOMMAND_LOOP1).
 *
 * @details One zero_any array of 2^26 doubles (512 MiB flat), every element 0.0 at first and
 *          no buffer allocated:
 *          1. a regular loop over the first half, by the library's chunk-aligned loop
 *             (parallel_for_chunks): varied values on the first eighth, 3.5 on the next three
 *             eighths -- a chunk written with one value takes it as its constant, no buffer;
 *          2. RUNCOMMAND_LOOP1 commands through write() and end_write(): 3.5 written element by
 *             element over the fifth eighth (its chunks get a buffer), one element in 1000 of
 *             the last quarter set to 2.0, and 1.0 added to every element of the first eighth;
 *          3. compress(), between phases: the chunks that became uniform -- the fifth
 *             eighth, all 3.5 -- fold back;
 *          4. a RUNCOMMAND_LOOP1 reading through read(): every element checked into a
 *             NumArray, summed on the host.
 *
 *          Built with -DCHUNK_DEVICE_ARCANE=ON (find_package(Arcane), an Arcane built with
 *          CUDA or HIP), compiled by arcane_accelerator_add_source_files. Run with
 *          `-A,AcceleratorRuntime=cuda` (or hip): the default queue is otherwise a host one,
 *          which the facade refuses. Exit status 0 when every element reads as expected.
 */

#include <cstdint>  // std::int64_t
#include <cstdio>   // std::printf
#include <iostream> // std::cerr

#include <arcane/launcher/ArcaneLauncher.h>          // ArcaneLauncher, StandaloneAcceleratorMng
#include <arcane/utils/Exception.h>                  // arcaneCallFunctionAndCatchException
#include <arcane/utils/NumArray.h>                   // NumArray
#include <arcane/accelerator/core/IAcceleratorMng.h> // IAcceleratorMng::defaultQueue
#include <arcane/accelerator/NumArrayViews.h>        // viewOut
#include <arcane/accelerator/RunCommandLoop.h>       // RUNCOMMAND_LOOP1
#include <arcane/accelerator/RunQueue.h>             // makeCommand

#include "chunk/device/arcane_adaptor.h" // zero_any::ArcaneArray

namespace sa = shrink_array;

namespace
{

constexpr std::int64_t n = std::int64_t{1} << 26;

/// @brief What element i holds after the phases below.
ARCCORE_HOST_DEVICE inline double expected(std::int64_t i)
{
    if (i < n / 8)
    {
        return 1.0 + static_cast<double>(i % 7) + 1.0; // step 1, then += 1.0 in step 2
    }
    if (i < 5 * n / 8)
    {
        return 3.5; // step 1 (one constant, no buffer), then step 2 element by element
    }
    if (i >= 3 * n / 4 && (i - 3 * n / 4) % 1000 == 0)
    {
        return 2.0; // step 2
    }
    return 0.0; // never written
}

int run()
{
    using namespace Arcane;
    using namespace Arcane::Accelerator;

    // An accelerator manager without a mesh; it must outlive every command.
    StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
    IAcceleratorMng*         acc_mng = launcher.acceleratorMng();
    auto                     queue   = acc_mng->defaultQueue();

    // n elements, every one 0.0, no buffer: a first arena of one buffer per chunk is
    // reserved (no phase can run out of it); ArcaneArray(n, arena_buffers) starts smaller.
    sa::zero_any::ArcaneArray<double> a(n);

    // 1. A regular loop: f(i) for i in [0, n / 2), one block per chunk. A phase by itself.
    a.parallel_for_chunks(queue, 0, n / 2, [=] ARCCORE_HOST_DEVICE(std::int64_t i) {
        return i < n / 8 ? 1.0 + static_cast<double>(i % 7) : 3.5;
    });

    // 2. Any commands between write() and end_write(): w(i) = v, w(i) += v, ...
    //    Two writes of one element by two threads in a phase are a race, as on a NumArray.
    {
        const auto w = a.write(queue);
        {
            auto command = makeCommand(queue);
            command << RUNCOMMAND_LOOP1(iter, n / 4 / 1000 + 1)
            {
                auto [k]             = iter();
                const std::int64_t i = 3 * n / 4 + 1000 * static_cast<std::int64_t>(k);
                if (i < n)
                {
                    w(i) = 2.0;
                }
            };
        }
        {
            auto command = makeCommand(queue);
            command << RUNCOMMAND_LOOP1(iter, n / 8)
            {
                auto [k]           = iter();
                w(n / 2 + k) = 3.5;
            };
        }
        {
            auto command = makeCommand(queue);
            command << RUNCOMMAND_LOOP1(iter, n / 8)
            {
                auto [i] = iter();
                w(i) += 1.0;
            };
        }
        a.end_write(queue);
    }

    // 3. Between phases: the chunks that became uniform give their buffer back (the fifth
    //    eighth: n / 8 / 1024 chunks at S = 1024).
    const unsigned long long folded = a.compress(queue);

    // 4. A read phase: read() in any command, until the next write().
    NumArray<Int32, MDDim1> bad(n);
    {
        auto       command = makeCommand(queue);
        const auto r       = a.read(queue);
        auto       out     = viewOut(command, bad);
        command << RUNCOMMAND_LOOP1(iter, n)
        {
            auto [i] = iter();
            out(i)   = r(i) == expected(i) ? 0 : 1;
        };
    }
    sa::device::arcane::barrier(queue); // before the host reads `bad`
    long wrong = 0;
    for (std::int64_t i = 0; i < n; ++i)
    {
        wrong += bad(static_cast<Int32>(i));
    }

    const double flat_gb = static_cast<double>(n) * sizeof(double) / 1e9;
    std::printf("arcane_getting_started: n = %lld doubles, %ld wrong\n", static_cast<long long>(n), wrong);
    std::printf("  %llu chunks folded by compress(); %llu of %lld chunks hold a buffer\n", folded, a.materialised(),
                static_cast<long long>((n + a.chunk_size() - 1) / a.chunk_size()));
    std::printf("  %.3f GB held against %.3f GB flat (%.1f %%)\n", static_cast<double>(a.bytes()) / 1e9, flat_gb,
                100.0 * static_cast<double>(a.bytes()) / 1e9 / flat_gb);
    return wrong == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    int  status = 1;
    auto func   = [&] {
        Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
        status = run();
    };
    const int caught = Arcane::arcaneCallFunctionAndCatchException(func);
    if (caught != 0)
    {
        std::cerr << "arcane_getting_started: an exception was caught\n";
        return caught;
    }
    return status;
}
