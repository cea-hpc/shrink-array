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
    bool compress_v = false;
    /// --checksum_only: the vector baseline writes the "_check_sum" files only, no
    /// data. A shrink run then passes or fails on the checksum alone (bit-exact: the
    /// element fallback's 1e-10 / 1e-6 tolerance is gone) and aborts on a miss.
    bool checksum_only = false;
    bool write_trace   = false; ///< --write_trace
    bool read_trace    = false; ///< --read_trace
    bool test_stl      = false; ///< --test_stl: run the STL-interface self-checks
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
    /// --mask_scale: tab_mask holds mask_scale * sz bools (a real factor, like
    /// --multiply). Default 1/32 (bench_mask_default_scale, bench_array.h);
    /// 8 is the size tab_mask was created with.
    double mask_scale = 1.0 / 32.0;
    /// --conn_scale: tab_conn holds about conn_scale * sz ints, whole rows (a
    /// real factor). Default 1/32 (bench_conn_default_scale, bench_array.h); 1
    /// is the size tab_conn came in with.
    double conn_scale = 1.0 / 32.0;
    /// --no_conn: leave out tab_conn, the mesh connectivity (bench_phase_conn.cpp)
    /// -- for a run that must compare with the campaigns made before it existed.
    /// Its array is then of size 1 and its phases, check and compression are
    /// skipped.
    bool no_conn = false;

    double multiply_factor = 1.0; ///< --multiply (a real factor)
    double divisor_factor  = 1.0; ///< --divisor (a real factor, like --multiply)

    /// --iter_max. 2000 by default.
    Int iter_max_arg = 2000;

    /// @name Adaptive re-chunk (dyn builds; the others ignore them)
    /// The advisor's schedule and the RechunkPolicy fields the bench exposes.
    /// The library reads none of them from the environment: a program sets the
    /// policy, and these options are how this one does.
    /// @{
    /// --advisor_period: iterations between two advisor passes; 0: no advisor;
    /// -1 (not given): max(10, iter_max / 20).
    Int advisor_period = -1;
    /// --advise_five: hand the decider only the five historical arrays (tab1,
    /// tab2, tab_short, tab_complex, tab_point), for runs that must compare with
    /// the campaigns made before the others were advised.
    bool advise_five     = false;
    bool rechunk_dry_run = false; ///< --rechunk_dry_run: analyse and log, never move
    /// --rechunk_min_gain_mb: absolute deadband of the data-side gain, MB (0
    /// removes it); < 0 (not given): the policy's.
    double rechunk_min_gain_mb = -1.0;
    /// --rechunk_coarsen_min_meta_mb: floor of a coarsening's metadata gain, MB
    /// (0 removes it); < 0: the policy's.
    double rechunk_coarsen_min_meta_mb = -1.0;
    /// --rechunk_coarsen_min_meta_rate: metadata MB per second of estimated
    /// rebuild a coarsening must reach (0 removes it); < 0: the policy's.
    double rechunk_coarsen_min_meta_rate = -1.0;
    /// --rechunk_max_coarsen: largest coarsening factor (1: no cap); 0: the policy's.
    Int rechunk_max_coarsen = 0;
    /// --rechunk_grid_bits: the advisor's grid between powers of two; < 0: the policy's.
    Int rechunk_grid_bits = -1;
    /// --rechunk_backoff_max: longest back-off, in ticks (0 or 1: none); < 0: the policy's.
    Int rechunk_backoff_max = -1;
    /// --force_rechunk <size>: every advised array re-chunked to this size,
    /// whatever the advisor says (tests); 0: never.
    Int  force_rechunk        = 0;
    Int  force_rechunk_at     = 0;     ///< --force_rechunk_at: first iteration that forces
    Int  force_rechunk_alt    = 64;    ///< --force_rechunk_alt: --force_rechunk_toggle's other size
    bool force_rechunk_toggle = false; ///< --force_rechunk_toggle: force at every pass, alternating
    /// @}
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
