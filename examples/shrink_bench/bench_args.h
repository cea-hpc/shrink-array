/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file bench_args.h
 * @brief Command-line parsing, out of main().
 *
 * Nothing here touches the workload: it turns argv into a handful of settings
 * and stops. It was the first eighty lines of main() and the last thing a
 * reader looking for the benchmark wanted to wade through.
 */

#pragma once

#include "bench_int.h" // Int: the sizes this struct carries

/// @brief Everything the command line can set.
struct BenchArgs
{
    bool no_io_v = false; ///< --no_io
    /// --compress: bzip2 the reference files written by the vector baseline and
    /// read the .bz2 oracle back. Off by default: the check runs work on the
    /// plain .bin, which spares the compress / decompress round trip.
    bool compress_v  = false;
    bool write_trace = false; ///< --write_trace
    bool read_trace  = false; ///< --read_trace
    bool test_stl    = false; ///< --test_stl: run the STL-interface self-checks
    /// --test_prealloc: alloc_chunk() over a partial window, then the loop
    /// that uses it (shrink_bench.cpp, run_prealloc_checks).
    bool test_prealloc = false;
    /// --no_struct: leave out the two struct arrays (Point3, Cell; bench_struct.h).
    /// They add 32 bytes per element to the std::vector baseline (+35 % of its
    /// memory, +28 % of its wall at 6 threads) and ~9 % to a chunk build's
    /// wall: this is the flag for a run that must compare with the campaigns
    /// made before they existed. Their arrays are then of size 1 and their
    /// phases, checks, stats and checksums are skipped.
    bool no_struct = false;
    /// --no_mask: leave out tab_mask, the large bool array of random masks and
    /// scattered flags (bench_phase_mask.cpp) -- for a run that must compare
    /// with the campaigns made before it existed. Its array is then of size 1
    /// and its phases, check and compression are skipped.
    bool no_mask = false;

    double multiply_factor = 1.0; ///< --multiply (a real factor)
    Int    divisor_factor  = 1;   ///< --divisor

    /// --iter_max. 2000 by default, kept from the historical behaviour.
    Int iter_max_arg = 2000;
};

/**
 * @brief Parse argv into a BenchArgs, and select the bench mode.
 *
 * @details Moved VERBATIM out of main(), exit() calls included: --help and an
 *          unknown argument both end the process here, as they always did.
 *          set_bench_mode() is called from inside for --mode, before any
 *          get_ or have_ accessor is read.
 */
BenchArgs parse_bench_args(int argc, char* argv[]);
