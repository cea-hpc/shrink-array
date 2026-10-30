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

// Central prelude first: fixes the chunk/ include order and derives the
// SHRINK_ARRAY / CHECK_COMPRESSION / NEED_COMPRESSION macros from the
// build-time variant selector.
#include "bench_prelude.h" // Int, tc, ComplexDouble, SHRINK_ARRAY

#include <algorithm> // std::min, std::max
#include <limits>    // std::numeric_limits (tab_mask's size)
#include <cfenv>     // feenableexcept, FE_DIVBYZERO
#include <chrono>    // std::chrono::steady_clock
#include <cmath>     // std::sin (the range-API checks)
#include <cstdio>    // std::fprintf
#include <cstdlib>   // std::getenv, std::exit, std::abort
#include <iomanip>   // std::setw, std::left
#include <iostream>  // std::cout, std::cerr, std::cin
#include <random>    // std::mt19937, std::uniform_int_distribution
#include <stdexcept> // std::invalid_argument (the INPUT_DATA prompt)
#include <string>    // std::string, std::to_string, std::getline
#include <unistd.h>  // getpid
#include <vector>    // std::vector

// The bulk range API (the chunked arrays' overloads or the std::vector
// fallback) comes with bench_array.h, through bench_phases.h below.

// Bench-mode configuration (mode_perf / mode_check / get_size_* / ...).
#include "bench_config.h" // mode_perf, get_size_data, get_iter_stop

// Utility templates and function declarations (checksums, correlations,
// seed I/O, help/suggest, memory report).
#include "bench_utils.h" // initialize_rng, memory_consumption

// The phases extracted from main() and their shared context: the array type,
// the second_value_* constants, the BenchArrays handle, and the OMPT phase
// wrappers (which lived here until the parallel regions left).
#include "bench_phases.h" // bench_phase_*, BenchArrays

// The command line (BenchArgs, parse_bench_args).
#include "bench_args.h" // parse_bench_args, BenchArgs


#include "chunk/ring_log_debug.h"     // ring_init/analyze_ring_sweep decls
#include "chunk/rechunk_decider.h"    // re-chunk decisions (acts in dyn builds only)
#include "chunk/internal/env_parse.h" // env_number / arg_number (checked, not atoi)
#include "chunk/log.h"                 // CHUNK_LOG, logging::enabled (the cst2 report at compress=info)

using namespace shrink_array;

namespace shrink_array
{

// ActiveArray now lives in bench_array.h, with the `Int` index type it hands
// to the library: this file carried a second, identical copy of the alias --
// harmless while both spelled the same thing, a redefinition as soon as one of
// them gained a template argument.

// chunk_dyn_size now lives in bench_array.h: bench_init.cpp needs it as much
// as main() does.

} // namespace shrink_array

/// @brief The chunk-size advisor and the re-chunk decider act in a dynamic
///        chunked build only. The objects exist in every build so the loop
///        reads the same; RechunkDecider::consider() is itself a no-op when
///        !chunk_dynamic, and this flag gates the rest with `if constexpr`.
constexpr bool has_advisor = chunk_config::is_shrink_enabled && build_config::chunk_dynamic;

template <typename T>
using Array = ActiveArray<T>;

// ── --test_prealloc: alloc_chunk() before a partial loop ─────────────────────
// alloc_chunk(i) materialises the chunk holding element i ahead of the loop
// that writes it: the loop then only stores into buffers that exist, instead
// of materialising chunks one write at a time inside the parallel region. It
// exists on the three variants (zero::Array, zero_one_value::Array, zero_any::Array) and had
// no caller. The window is PARTIAL (an eighth of the array, from its first
// quarter): the chunks around it must stay compressed, or preallocating would
// prove nothing. bool goes through its own materialiser on zero_one_value / zero_any.

/// @brief One element type: preallocate the window, check which chunks are
///        materialised, run the loop over the window, read everything back
///        against a std::vector. @return the number of failures.
template <typename T>
static long prealloc_check(const char* name, Int count, T (*value_at)(Int))
{
    Array<T>       arr(count);
    std::vector<T> ref(static_cast<std::size_t>(count), T{});
    const Int      beg   = count / 4;
    const Int      end   = beg + (count / 8);
    long           fails = 0;

    if constexpr (requires {
                      arr.alloc_chunk(Int{0});
                      arr.chunk_size();
                      arr.written(Int{0});
                  })
    {
        const Int step = static_cast<Int>(arr.chunk_size());
        bench_par::region(
            [&]([[maybe_unused]] bench_par::Team& team)
            {
                begin_parallel_phase_g();
                bench_par::for_static_vec(team, beg / step, ((end - 1) / step) + 1,
                                          [&](const Int c) __attribute__((always_inline))
                                          {
                                              if constexpr (build_config::threaded)
                                              {
                                                  arr.alloc_chunk(c * step, get_thr_rank_fast());
                                              }
                                              else
                                              {
                                                  arr.alloc_chunk(c * step);
                                              }
                                          });
                finalize_parallel_phase_only_flush_g();
            });
        finalize_parallel_phase_after_flush_g();

        // Materialised exactly over the window: every chunk it touches, and
        // not the chunks around it.
        long in  = 0;
        long out = 0;
        for (Int c = 0; c * step < count; ++c)
        {
            const bool in_window = c >= beg / step && c <= (end - 1) / step;
            const bool written   = arr.written(c * step);
            in += in_window && written;
            out += !in_window && written;
            if (written != in_window && fails < 3)
            {
                std::cout << "[prealloc] " << name << ": chunk " << c << (in_window ? " not" : "") << " materialised\n";
            }
            fails += written != in_window;
        }
        std::cout << "[prealloc] " << name << ": " << in << " chunks preallocated in [" << beg << ", " << end << "), "
                  << out << " outside\n";
    }

    // The loop over the window: stores into chunks that already exist.
    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            bench_par::for_static_vec(team, beg, end,
                                      [&](const Int i) __attribute__((always_inline)) { arr[i] = value_at(i); });
            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();
    for (Int i = beg; i < end; ++i)
    {
        ref[static_cast<std::size_t>(i)] = value_at(i);
    }

    // Read back through the read operator, window and surroundings.
    for (Int i = 0; i < count; ++i)
    {
        const T value = arr[i];
        if (!(value == ref[static_cast<std::size_t>(i)]))
        {
            if (fails < 5)
            {
                std::cout << "[prealloc] " << name << ": MISMATCH at " << i << "\n";
            }
            ++fails;
        }
    }
    return fails;
}

/// @brief The --test_prealloc checks; exits 1 on a failure.
static void run_prealloc_checks(Int count)
{
    long fails = 0;
    fails += prealloc_check<double>("double", count,
                                    [](Int i) { return i % 5 == 0 ? 0.0 : 0.5 * static_cast<double>(i % 97); });
    fails += prealloc_check<int>("int", count, [](Int i) { return static_cast<int>(i % 13); });
    fails += prealloc_check<bool>("bool", count, [](Int i) { return i % 3 != 0; });
    if (fails)
    {
        std::cout << "[prealloc] FAILED (" << fails << ")\n";
        std::exit(1);
    }
    std::cout << "[prealloc] ALL PASSED\n";
}

int main(int argc, char* argv[])
{
    // Optionally cap the glibc arenas, BEFORE any parallel region creates them.
    //
    // OFF by default, and the default is the point of this comment.
    //
    // glibc gives each contending thread its own arena and an arena never lends
    // a freed block to another, so capping them trades allocator contention
    // against retained memory. Which way that trade goes is NOT a property of
    // the library -- it depends on the variant, the thread count and how much
    // the workload allocates:
    //
    //   zero_any, 64 threads, --divisor 5: 5232 MB -> 2765 MB capped at 1 arena,
    //                                      no measurable time cost.
    //   zero dyn, 64 threads:              4.44s / 51501 MB at glibc's default
    //                                     19.63s / 51426 MB capped at 4
    //                                     17.87s / 51565 MB capped at 16
    //                                      4.36s / 51539 MB capped at 64
    //
    // On that second workload the memory does not move at all (0.3% across every
    // setting) and the time is 4.5x worse as soon as the arenas are capped below
    // one per thread. A default of 4 -- which this code shipped with, calibrated
    // on the first measurement alone -- made it four times slower for nothing.
    //
    // So: no default. Set CHUNK_MALLOC_ARENAS=<n> to experiment on a given
    // workload, and measure both numbers before keeping a value.
    //
    // Not settable via setenv() from here: glibc reads MALLOC_ARENA_MAX at the
    // first allocation, which happens in static initialisation, before main.
    // mallopt() is not too late -- it applies to arenas created afterwards.
    {
        const int arenas = shrink_array::env_number<int>("CHUNK_MALLOC_ARENAS", 0); // 0 = leave glibc alone
        if (arenas > 0)
        {
            set_max_malloc_arenas(arenas);
        }
    }

    // TSan tip: unbuffer stderr with setvbuf(stderr, nullptr, _IONBF, 0) so its
    // reports interleave correctly with ours.

    // Command-line parsing: bench_args.cpp.
    const BenchArgs bench_args      = parse_bench_args(argc, argv);
    const bool      no_io_v         = bench_args.no_io_v;
    const bool      compress_v      = bench_args.compress_v;
    const bool      checksum_only   = bench_args.checksum_only;
    const bool      write_trace     = bench_args.write_trace;
    const bool      read_trace      = bench_args.read_trace;
    const bool      test_stl        = bench_args.test_stl;
    const bool      no_struct       = bench_args.no_struct;
    const bool      no_mask         = bench_args.no_mask;
    const bool      no_conn         = bench_args.no_conn;
    const double    mask_scale      = bench_args.mask_scale; // tab_mask's size / sz
    const double    conn_scale      = bench_args.conn_scale; // tab_conn's size / sz
    const double    multiply_factor = bench_args.multiply_factor;
    const double    divisor_factor  = bench_args.divisor_factor;
    const Int       iter_max_arg    = bench_args.iter_max_arg;

    // Before sz is computed: a factor past the Int range would wrap here.
    check_size_limits(multiply_factor, divisor_factor, iter_max_arg);

    const Int sz       = get_size_data(multiply_factor, divisor_factor);
    const Int interval = get_size_interval(multiply_factor, divisor_factor);

    // Example behavior based on flags
    if (no_io_v)
    {
        std::cout << "I/O operations will be skipped.\n";
    }
    if (compress_v)
    {
        std::cout << "Compression is enabled.\n";
    }
    if (checksum_only)
    {
        std::cout << "Reference written as checksums only.\n";
    }
    if (write_trace)
    {
        std::cout << "Trace information will be written.\n";
    }
    if (read_trace)
    {
        std::cout << "Trace information will be read.\n";
    }

    // The second_value_* constants and zero_complex live in bench_array.h: the
    // extracted phases need them as much as main() does.

    if constexpr (build_config::is_debug)
    {
        feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
    }

    // Not const: the INPUT_DATA branch below re-reads it from std::cin. And no
    // longer constexpr either -- the mode is chosen at run time by --mode.
    Int io_res = input_data ? 0 : get_io_mode();

    const char* no_io = no_io_v ? "1" : std::getenv("NO_IO");
    // Compression is opt-in (--compress or COMPRESS=1): the I/O layer takes a
    // no_compress flag, so the default is the non-null value.
    const char* no_compress = (compress_v || std::getenv("COMPRESS")) ? nullptr : "1";

    if constexpr (input_data)
    {
        std::string data;

        if constexpr (chunk_config::is_shrink_enabled)
        {
            std::cout << "no io (default 0) / read data and check (2) : ";
        }
        else
        {
            std::cout << "no io (default 0) / write data (1) / read data and check (2) : ";
        }

        std::getline(std::cin, data);

        // Checks if the user entered data or pressed Enter (carriage return)
        if (data.empty())
        {
            std::cout << "default io_res: " << io_res << "\n";
        }
        else
        {
            try
            {
                io_res = std::stoi(data); // Convert string to integer
                std::cout << "You give : " << io_res << "\n";
            }
            catch (const std::invalid_argument&)
            {
                std::cerr << "Error: Conversion failed. The string is not a valid number" << '\n';
            }
        }
    }

    // Geometry validation and library start-up: bench_init.cpp.
    bench_init_library();

    const Int iter_max = iter_max_arg;
    std::cout << "iter_max = " << iter_max << "\n";

    const Int iter_stop = get_iter_stop(iter_max);

    // Every one of these is used as `k % freq == 0`, so it must never be zero.
    // iter_max / 10 and iter_max / 4 round down to 0 for a small --iter_max --
    // `--iter_max 6` raised SIGFPE at the k % freq_stats site. Clamping to 1
    // means "every iteration", which is the sensible reading of "every
    // iter_max/10" when iter_max < 10, and is what get_freq_raz() already does
    // with its own `if (ret == 0) ret = 1`.
    const Int freq_stats = have_stats() ? std::max<Int>(1, iter_max / 10) : 1000000000;

    const Int freq_check_sum = have_check_sum() ? std::max<Int>(1, iter_max / 4) : 1000000000;

    CONSTEXPR const Int freq_conso_mem = get_freq_conso_mem();

    // Cadence of the per-iteration checksum line. Defaults to the memory
    // line's, so the two land on the same iterations and a checksum can be read
    // next to the figure it goes with -- 2000 lines of checksum otherwise bury
    // the memory lines, the [RECHUNK-*] verdicts and the final summary.
    // CHUNK_BENCH_KSUM_PERIOD overrides it (=1 prints every iteration, which is
    // what a bisection wants).
    const Int freq_ksum = []() -> Int
    {
        if (const Int period = shrink_array::env_number<Int>("CHUNK_BENCH_KSUM_PERIOD", 0); period > 0)
        {
            return period;
        }
        return 0; // 0 = follow freq_conso_mem, resolved just below
    }();

    [[maybe_unused]] const Int freq_compress = need_compression ? 1 : std::max<Int>(1, iter_max / 10);

    // Period of the DEBUG audit of the inline compression ("Check compression",
    // zero::Array*::compress_int). Not freq_compress: that one is 1 whenever
    // need_compression holds, i.e. in the very builds where the audit exists.
    // Every iteration was a full sweep of every array plus a self-heal that
    // made debug compress more than the inline path alone. SHRINK_BENCH_AUDIT_FREQ
    // overrides (1 = every iteration, the former behaviour).
    [[maybe_unused]] const Int freq_audit = [iter_max]() -> Int
    {
        if (const char* env = std::getenv("SHRINK_BENCH_AUDIT_FREQ"); env != nullptr && *env != '\0')
        {
            if (Int value = 0; parse_number(env, value))
            {
                return std::max<Int>(1, value);
            }
            warn_invalid_env("SHRINK_BENCH_AUDIT_FREQ", env);
        }
        return std::max<Int>(1, iter_max / 10);
    }();

    const Int freq_raz = get_freq_raz(iter_max);

    // CHUNK_PMP_MAX_THREADS (PMP_CHUNK.h) is omp_get_max_threads() with OpenMP
    // and the literal 1 without, so no #ifdef is needed here any more.
    const Int max_thr = CHUNK_PMP_MAX_THREADS;

    CONSTEXPR const bool stats     = have_stats();
    CONSTEXPR const bool check_sum = have_check_sum();

    if (sz < 512)
    {
        std::cerr << "sz too small !\n";
        std::abort();
    }
    std::cout << "Number of threads : " << max_thr << "\n";
    std::cout << "Pid : " << getpid() << std::endl;

    if constexpr (chunk_config::use_zero)
    {
        std::cout << tc.bold << tc.magenta << "MODE: " << tc.reset << "zero (Single Value)\n";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        std::cout << tc.bold << tc.magenta << "MODE: " << tc.reset << "zero_one_value (Dual Value)\n";
    }
    else if constexpr (chunk_config::use_zero_any)
    {
        std::cout << tc.bold << tc.magenta << "MODE: " << tc.reset << "zero_any (Advanced Dual Value)\n";
    }
    else
    {
        std::cout << "VECTOR MODE\n";
    }

    if constexpr (build_config::use_fast_alloc)
    {
        std::cout << tc.bold << tc.green << "ALLOC: " << tc.reset << "FAST_ALLOC (Thread-Local Pools)\n";
    }
    else
    {
        std::cout << tc.bold << tc.yellow << "ALLOC: " << tc.reset << "STD_ALLOC (System Malloc)\n";
    }

    auto start = std::chrono::steady_clock::now();

    std::cout << "Initialisation ..." << std::endl;
    std::cout << "Creation of arrays ..." << std::endl;
    [[maybe_unused]] auto type_instances =
        std::tuple<double, int, bool, std::complex<double>, float, unsigned char, Point3, Cell>{};

    // Function to determine max type name length
    [[maybe_unused]] auto get_max_length = [&type_instances]<template <typename> typename ChunkType>() -> size_t
    {
        size_t max_len = 0;
        std::apply([&](auto... data_types)
                   { ((max_len = std::max(max_len, type_name<ChunkType<decltype(data_types)>>().size())), ...); },
                   type_instances);
        return max_len;
    };

    // Lambda template for printing `sizeof(ChunkType<T>)` with proper formatting

    [[maybe_unused]] const auto print_chunk_size =
        [&type_instances, &get_max_length]<template <typename> typename ChunkType>(const std::string& chunk_name)
    {
        // 1. Display section header in Bold Cyan
        std::cout << tc.bold << tc.cyan << "\n=== " << chunk_name << " ===\n" << tc.reset;

        const size_t max_width = get_max_length.operator()<ChunkType>();

        // 2. Iterate through each type stored in type_instances tuple
        std::apply(
            [&](auto... data_types)
            {
                (
                    [&](auto team)
                    {
                        // Construct the label string: "sizeof(ChunkType<T>)"
                        const std::string label = "sizeof(" + type_name<ChunkType<decltype(team)>>() + ")";

                        // Print the line with a dedicated color scheme:
                        // [Label in White] [= in Cyan] [Value in Bold Yellow] [Unit in Green]
                        std::cout << tc.white << std::left << std::setw(static_cast<int>(max_width + 10)) << label
                                  << tc.reset << tc.cyan << " = " << tc.reset << tc.bold << tc.yellow << std::setw(5)
                                  << sizeof(ChunkType<decltype(team)>) << tc.reset << tc.green << " bytes" << tc.reset
                                  << "\n";
                    }(data_types),
                    ...); // C++17 Fold expression
            },
            type_instances);
    };

    Array<double>        tab1(sz);
    Array<double>        tab2(sz);
    Array<double>        tab3(sz);
    Array<double>        tab4(sz);
    Array<double>        tab5(sz);
    Array<int>           tab_int(sz);
    Array<ComplexDouble> tab_complex(sz);

    Array<float>         tab_float(2 * sz);
    Array<short>         tab_short(sz / 2);
    Array<unsigned char> tab_char(2 * sz);
    Array<long>          tab_long(sz);
    Array<unsigned long> tab_ulong(sz);
    // Quantised field: no single second value to declare, it registers one per
    // level it lands on (bench_array.h::bench_double_bands).
    Array<double> tab_quant(sz);

    // Two aggregate element types (bench_struct.h): the library asks nothing
    // of them beyond the element contract, and their runs of null elements
    // and of a repeated value compress like the arithmetic arrays' 0 and
    // second value.
    // Size 1 under --no_struct: the phases skip them (BenchArrays::with_struct).
    Array<Point3> tab_point(no_struct ? 1 : sz);
    Array<Cell>   tab_cell(no_struct ? 1 : sz);

    maybe_set_second_value(tab1, second_value_d);
    maybe_set_second_value(tab2, second_value_d);
    maybe_set_second_value(tab3, second_value_d);

    maybe_set_second_value(tab4, second_value_d1);
    maybe_set_second_value(tab5, second_value_d2);

    maybe_set_second_value(tab_int, second_value_i);
    maybe_set_second_value(tab_complex, second_value_c);
    maybe_set_second_value(tab_float, second_value_f0);
    maybe_set_second_value(tab_short, second_value_s0);
    maybe_set_second_value(tab_char, second_value_c0);
    maybe_set_second_value(tab_long, second_value_l);
    maybe_set_second_value(tab_ulong, second_value_ul);
    maybe_set_second_value(tab_point, second_value_p);
    maybe_set_second_value(tab_cell, second_value_cell);

    if constexpr (chunk_config::use_zero_any)
    {
        print_chunk_size.operator()<zero_any::Chunk>("zero_any");
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        print_chunk_size.operator()<zero_one_value::Chunk>("zero_one_value");
    }
    else if constexpr (chunk_config::use_zero)
    {
        print_chunk_size.operator()<zero::Chunk>("zero");
    }

    // Array<bool> in a shrink build, std::vector<short> in the baseline
    // (bench_array.h says why): one alias, where four declarations under
    // #if defined CHUNK_VARIANT_ZERO* stood.
    BenchBoolArray tab_bool(sz);
    // tab_bool's twin, written with the same values at the same places:
    // ArrayChunkBits (8 bools per byte) in a zero_one_value build, tab_bool's own type
    // elsewhere (BenchBoolBitsArray). Compare the two in the stats, the
    // chunk filling rates and the pools of a single run.
    BenchBoolBitsArray tab_bool_bits(sz);

    // Large and mostly NOT constant: random masks and scattered flags between
    // whole constant segments (bench_phase_mask.cpp). ArrayChunkBits in a
    // zero_one_value build, std::vector<bool> in the baseline (BenchMaskArray).
    // Size 1 under --no_mask: its phases are skipped (BenchArrays::with_mask).
    // mask_scale * sz (--mask_scale, default 1/32): computed in 64 bits and
    // capped at the largest whole number of 1024-element blocks the build's Int
    // holds -- at --mask_scale 8, the size tab_mask was created with, 8 x 10^9
    // wrapped a 32-bit Int to -589934592 and zero_any::Array<bool, int> threw
    // std::length_error. The int64 builds take the full size.
    const long long mask_want = std::max(std::llround(mask_scale * static_cast<double>(sz)), 1LL);
    const Int       mask_sz = mask_want > static_cast<long long>(std::numeric_limits<Int>::max())
                                  ? static_cast<Int>(static_cast<long long>(std::numeric_limits<Int>::max()) & ~0x3FFLL)
                                  : static_cast<Int>(mask_want);
    if (!no_mask && static_cast<long long>(mask_sz) != mask_want)
    {
        std::cout << "Note: tab_mask capped at " << mask_sz << " elements (" << mask_want
                  << " requested; Int is 32-bit here, the int64 build takes the full size)\n";
    }
    BenchMaskArray tab_mask(no_mask ? 1 : mask_sz);
    if (!no_mask)
    {
        std::cout << "tab_mask: " << mask_sz << " elements (--mask_scale " << mask_scale << ")\n";
    }

    // A mesh connectivity (bench_phase_conn.cpp): pattern::Array<int> in the
    // zero_one_value and zero_any builds, zero::Array<int> in a zero build, a vector
    // in the baseline (BenchConnArray). About conn_scale * sz ints (--conn_scale,
    // default 1/32), whole rows of bench_conn_row, computed in 64 bits and capped
    // at the build's Int as tab_mask; size 1 under --no_conn
    // (BenchArrays::with_conn) or when that is less than one row.
    const long long conn_rows_want =
        std::llround(conn_scale * static_cast<double>(sz)) / static_cast<long long>(bench_conn_row);
    const long long conn_rows_max = static_cast<long long>(std::numeric_limits<Int>::max()) / bench_conn_row;
    const Int       conn_sz       = static_cast<Int>((conn_rows_want < conn_rows_max ? conn_rows_want : conn_rows_max) *
                                                     static_cast<long long>(bench_conn_row));
    const bool      no_tab_conn   = no_conn || conn_sz == 0;
    BenchConnArray  tab_conn(no_tab_conn ? 1 : conn_sz);
    if (!no_tab_conn)
    {
        std::cout << "tab_conn: " << conn_sz << " elements (" << conn_sz / bench_conn_row << " rows of "
                  << bench_conn_nx << " cells, --conn_scale " << conn_scale << ")\n";
    }

    // Handle on the nineteen arrays, passed to the extracted phases. References,
    // so main() stays the owner -- it builds them and destroys them.
    BenchArrays bench_arrays{tab1,          tab2,        tab3,        tab4,      tab5,     tab_int,  tab_bool,
                             tab_bool_bits, tab_complex, tab_float,   tab_short, tab_char, tab_long, tab_ulong,
                             tab_quant,     tab_point,   tab_cell,    tab_mask,  tab_conn, sz,       interval,
                             !no_struct,    !no_mask,    !no_tab_conn};

    // The end-of-life stats print these next to the table number.
    maybe_set_debug_name(tab1, "tab1");
    maybe_set_debug_name(tab2, "tab2");
    maybe_set_debug_name(tab3, "tab3");
    maybe_set_debug_name(tab4, "tab4");
    maybe_set_debug_name(tab5, "tab5");
    maybe_set_debug_name(tab_int, "tab_int");
    maybe_set_debug_name(tab_bool, "tab_bool");
    maybe_set_debug_name(tab_bool_bits, "tab_bool_bits");
    maybe_set_debug_name(tab_complex, "tab_complex");
    maybe_set_debug_name(tab_float, "tab_float");
    maybe_set_debug_name(tab_short, "tab_short");
    maybe_set_debug_name(tab_char, "tab_char");
    maybe_set_debug_name(tab_long, "tab_long");
    maybe_set_debug_name(tab_ulong, "tab_ulong");
    maybe_set_debug_name(tab_quant, "tab_quant");
    maybe_set_debug_name(tab_point, "tab_point");
    maybe_set_debug_name(tab_cell, "tab_cell");
    maybe_set_debug_name(tab_mask, "tab_mask");
    maybe_set_debug_name(tab_conn, "tab_conn");

    SingletonChk<GlobalStatsChunkList>::get_ref().print();

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    start                                         = end;

    std::cout << tc.white << "Elapsed time creation : " << tc.bold << elapsed_seconds.count() << "s" << tc.reset
              << "\n";
    // useless in chunk mode arrays are set to zero by default

    if (bench_args.test_prealloc)
    {
        run_prealloc_checks(sz);
    }

    // Initial fill: bench_phase_fill.cpp (see bench_phases.h).
    bench_phase_fill(bench_arrays, start, end, elapsed_seconds);
    bench_phase_struct_fill(bench_arrays);
    bench_phase_mask_fill(bench_arrays);
    bench_phase_conn_fill(bench_arrays);

    // Verification pass (writing or reading back the reference files):
    // lbd_check and lbd_check_all, in bench_checks.cpp.
    const BenchIoFlags bench_io_flags{no_io, no_compress, io_res, read_trace, write_trace, checksum_only};

    bench_check_all(bench_arrays, -1, bench_io_flags);

    tab_float.resize(sz);
    tab_short.resize(sz, 0);
    tab_char.resize(sz);

    // `if constexpr` rather than #ifdef: bench_config.h exposes print_mem for
    // exactly this, and memory_consumption is declared in every build. The body
    // therefore stays TYPE-CHECKED even when discarded, which a #ifdef does not
    // do -- dead code under a directive rots unnoticed, and this file has
    // already paid for that.
    if constexpr (print_mem)
    {
        std::cout << "\n";
        memory_consumption("after init");
    }
    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::cout << "\nElapsed time Initialisation to zero: " << elapsed_seconds.count() << "s\n";

    const Int min = 1;
    const Int max = sz;

    // Initialize the random number generator
    const std::string str            = std::to_string(iter_max);
    const std::string seed_file_name = "seed" + str + ".txt"; // File to store the seed
    std::mt19937      rng;                                    // Random number generator
    initialize_rng(rng, seed_file_name);

    // Create a uniform distribution for interval sizes
    std::uniform_int_distribution<Int> size_distribution(1, interval);

    // Create a uniform distribution for the starting points of the intervals
    std::uniform_int_distribution<Int> start_distribution(min, max);

    std::cout << "Iteration ..." << std::endl;

    std::vector<Int> tab_size(iter_max);
    std::vector<Int> tab_start(iter_max);
    std::vector<Int> tab_nba1(iter_max);
    std::vector<Int> tab_nba2(iter_max);

    // no parallel in OpenMP
    for (Int k = 0; k < iter_max; ++k)
    {
        const Int size   = size_distribution(rng);
        const Int start0 = start_distribution(rng);
        const Int end0   = start0 + size;
        tab_size[k]      = size;
        tab_start[k]     = start0;

        std::uniform_int_distribution uni1(start0, end0);
        tab_nba1[k] = uni1(rng);
        tab_nba2[k] = uni1(rng);
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::cout << "\nElapsed time distribution : " << elapsed_seconds.count() << "s\n";

    double        val_sum      = 0;
    ComplexDouble cv_val_sum   = zero_complex;
    long          long_val_sum = 0;
    double        point_sum    = 0; // the struct arrays' read sweep (bench_phase_struct.cpp)
    long          mask_count   = 0; // tab_mask's read sweep (bench_phase_mask.cpp)
    long          conn_sum     = 0; // tab_conn's gather (bench_phase_conn.cpp)
    double        v_real       = 0;
    double        v_im         = 0;

    // Checksums: bench_stats.cpp.

    // Value census and filling rates: lbd_fill_values, lbd_fill_chunk and
    // lbd_stats, in bench_stats.cpp.

    // Start iteration loop
    // #define TEST_CAPTUR_BUG

    // Random abort injection (TEST_CAPTUR_BUG). The draws sit inside the
    // `if constexpr`, so a normal build never touches std::random_device -- the
    // statement is compiled, never executed, and r stays -1, which the test at
    // the injection site below rejects.
    Int injection  = -1;
    Int cycle_alea = -1;
    if constexpr (test_captur_bug)
    {
        std::mt19937                       rng0(std::random_device{}());
        std::uniform_int_distribution<Int> dist0(0, 2);
        injection = dist0(rng0);

        std::uniform_int_distribution<Int> dist1(0, 31);
        cycle_alea = dist1(rng0);
    }

    // ── Chunk-size advisor (STEP 0: observation only, dyn+zero_any builds) ──
    // Every --advisor_period iterations (0 = disabled),
    // estimate each representative array's memory across candidate chunk
    // sizes and print the table + the analysis cost. Runs at this SERIAL
    // point of the loop (quiescent enough for observation; states may still
    // include not-yet-drained deferred ops). No mutation whatsoever.
    // Period derived from the run length, not fixed. An absolute default is
    // wrong at both ends: at 500 the first move lands at iteration 1000 and a
    // 300-iteration run never re-chunks at all; at 50 a 2000-iteration run
    // looks 40 times. Measured over 2000 iterations at 64 threads, end of run:
    //   no re-chunk             33847 MB
    //   period 500  ( 4 moves)  33357 MB   12.2 s
    //   period 100  (14 moves)  32795 MB   18.6 s   <- best
    //   period  50  (17 moves)  33059 MB   26.6 s
    // iter_max / 20 gives 100 here, and keeps the shape at any length: the
    // first move lands around a tenth of the way in, where the geometric gains
    // are largest, and the tick count stays bounded. The floor of 10 stops a
    // very short run from examining every other iteration.
    // --advisor_period overrides; 0 disables.
    const Int advisor_period =
        bench_args.advisor_period >= 0 ? bench_args.advisor_period : std::max<Int>(10, iter_max / 20);
    // Phase-3A decider (dry-run): persists across ticks, keyed by array address.
    // `advisor_call` increments once per advisor_tick so cooldown/stability count
    // in ticks, not iterations. Policy defaults: 3 stable ticks, 8 MB deadband,
    // 5-tick cooldown, dry_run=true (never mutates).
    // The library owns the whole decision: RechunkDecider::consider() runs the
    // advisor on the array, applies the memory + hysteresis policy, and re-chunks
    // it when worthwhile. dry_run=false makes it act (rechunk is now parallel and
    // cheap, so we gate purely on memory gain + hysteresis, not CPU cost).
    // The --rechunk_* / --force_rechunk* options set the policy's fields: the
    // library reads none of them from the environment.
    const RechunkPolicy rechunk_policy = [&bench_args]
    {
        constexpr double mib = 1024.0 * 1024.0;
        RechunkPolicy    pol;
        pol.dry_run = bench_args.rechunk_dry_run;
        if (bench_args.rechunk_min_gain_mb >= 0.0)
        {
            pol.min_gain_bytes = static_cast<std::size_t>(bench_args.rechunk_min_gain_mb * mib);
        }
        if (bench_args.rechunk_coarsen_min_meta_mb >= 0.0)
        {
            pol.coarsen_min_meta_gain_bytes = static_cast<std::size_t>(bench_args.rechunk_coarsen_min_meta_mb * mib);
        }
        if (bench_args.rechunk_coarsen_min_meta_rate >= 0.0)
        {
            pol.coarsen_min_meta_rate = bench_args.rechunk_coarsen_min_meta_rate;
        }
        if (bench_args.rechunk_max_coarsen > 0)
        {
            pol.max_coarsen_factor = static_cast<unsigned>(bench_args.rechunk_max_coarsen);
        }
        if (bench_args.rechunk_grid_bits >= 0)
        {
            pol.advisor_grid_bits = static_cast<int>(bench_args.rechunk_grid_bits);
        }
        if (bench_args.rechunk_backoff_max >= 0)
        {
            pol.max_backoff_ticks = static_cast<int>(bench_args.rechunk_backoff_max);
        }
        pol.trace_array  = bench_args.rechunk_trace_array;
        pol.force_size   = static_cast<unsigned>(bench_args.force_rechunk);
        pol.force_at     = static_cast<int>(bench_args.force_rechunk_at);
        pol.force_alt    = static_cast<unsigned>(bench_args.force_rechunk_alt);
        pol.force_toggle = bench_args.force_rechunk_toggle;
        return pol;
    }();
    RechunkDecider rechunk_decider{rechunk_policy};
    Int            advisor_call = 0;
    const auto     advisor_tick = [&](Int k_iter)
    {
        if constexpr (has_advisor)
        {
            if (advisor_period <= 0 || (k_iter % advisor_period) != 0)
            {
                return;
            }
            // consider() counts ticks and iterations in int; Int is `long` in
            // the int64 build, and both counts stay far below INT_MAX.
            const int tick = static_cast<int>(advisor_call++);
            const int iter = static_cast<int>(k_iter);
            rechunk_decider.consider<double>(tab1, tick, iter, "tab1");
            rechunk_decider.consider<double>(tab2, tick, iter, "tab2");
            rechunk_decider.consider<short>(tab_short, tick, iter, "tab_short");
            rechunk_decider.consider<ComplexDouble>(tab_complex, tick, iter, "tab_complex");
            if (!no_struct)
            {
                rechunk_decider.consider<Point3>(tab_point, tick, iter, "tab_point");
            }
            // Every other array too, advised by default: a dynamic run then
            // starts like the static S=64 build and ends below the static S=256
            // one (zero_any, 64 threads: 6.57 GB against 7.26 GB; with only the
            // five arrays above advised, the other ten stay at S=64, ~12.5 GB).
            // --advise_five advises only the five, for runs that must compare
            // with earlier campaigns. tab_bool and tab_bool_bits excepted: both
            // stay at their birth geometry, so the two bool containers are
            // compared as they are, not after a re-chunk.
            if (!bench_args.advise_five)
            {
                rechunk_decider.consider<double>(tab3, tick, iter, "tab3");
                rechunk_decider.consider<double>(tab4, tick, iter, "tab4");
                rechunk_decider.consider<double>(tab5, tick, iter, "tab5");
                rechunk_decider.consider<int>(tab_int, tick, iter, "tab_int");
                rechunk_decider.consider<float>(tab_float, tick, iter, "tab_float");
                rechunk_decider.consider<unsigned char>(tab_char, tick, iter, "tab_char");
                rechunk_decider.consider<long>(tab_long, tick, iter, "tab_long");
                rechunk_decider.consider<unsigned long>(tab_ulong, tick, iter, "tab_ulong");
                rechunk_decider.consider<double>(tab_quant, tick, iter, "tab_quant");
                if (!no_struct)
                {
                    rechunk_decider.consider<Cell>(tab_cell, tick, iter, "tab_cell");
                }
                if (!no_tab_conn)
                {
                    rechunk_decider.consider<int>(tab_conn, tick, iter, "tab_conn");
                }
            }
            // tab_mask: an ArrayChunkBits in a zero_one_value build, whose advisor
            // knows the packed layout (the other variants' Array<bool> is not
            // what this array is for).
            if constexpr (chunk_config::use_zero_one_value)
            {
                if (!no_mask)
                {
                    rechunk_decider.consider<BenchMaskArray::value_type>(tab_mask, tick, iter, "tab_mask");
                }
            }
            if constexpr (chunk_config::use_zero_any)
            {
                // CHUNK_LOG=compress=info: the registry's report at each pass.
                if (shrink_array::logging::enabled(shrink_array::logging::Channel::compress,
                                                   shrink_array::logging::Level::info))
                {
                    // STEP B — cst2 GC mark report (read-only) : zero_any::Array-only API
                    // (per-chunk cst_val2 pointer registry has no equivalent on zero /
                    // zero_one_value, which have no per-chunk GC-able registry to report on).
                    CHUNK_LOG(info, compress, "cst2-gc @ iter {}", k_iter);
                    // Spelled out rather than Array<T>: the enclosing
                    // `if constexpr` is in a plain lambda, so its discarded
                    // branch is still checked, and Array<T> is zero::Array<T> --
                    // which has no gc_mark_report -- in a zero/zero_one_value build. `Int`
                    // is what matters here: zero_any::Array<double> would take the
                    // default IndType (int), a DIFFERENT instantiation from the
                    // one the bench fills in an int64 build, and the report would
                    // be of an array nothing ever touched.
                    zero_any::Array<double, Int>::gc_mark_report();
                    zero_any::Array<short, Int>::gc_mark_report();
                    zero_any::Array<ComplexDouble, Int>::gc_mark_report();
                }
            }
        }
    };
    // CHUNK_CST2_RECLAIM=N: every N iterations, free the zero_any second values
    // no chunk uses any more, for every element type of the bench
    // (reclaim_second_values() covers all the arrays of a type, whatever their
    // index type). Static builds too, unlike the advisor. 0 or unset: never.
    const Int cst2_reclaim_period = []() -> Int
    {
        if (const char* env = std::getenv("CHUNK_CST2_RECLAIM"); env != nullptr && *env != '\0')
        {
            if (Int value = 0; parse_number(env, value))
            {
                return value;
            }
            warn_invalid_env("CHUNK_CST2_RECLAIM", env);
        }
        return 0;
    }();
    // [&], not [cst2_reclaim_period]: clang warns that the explicit capture is
    // not needed (-Wunused-lambda-capture) where the branch below is discarded.
    const auto cst2_reclaim_tick = [&](Int k_iter)
    {
        if constexpr (chunk_config::use_zero_any)
        {
            if (cst2_reclaim_period <= 0 || (k_iter % cst2_reclaim_period) != 0)
            {
                return;
            }
            // Timed: the pass is sequential, outside the write path, and its
            // cost (chunks + second-value slots + identifiers handed out) is
            // what decides how often it can run.
            const auto  t0    = std::chrono::steady_clock::now();
            std::size_t freed = 0;
            freed += zero_any::Array<double, Int>::reclaim_second_values();
            freed += zero_any::Array<float, Int>::reclaim_second_values();
            freed += zero_any::Array<int, Int>::reclaim_second_values();
            freed += zero_any::Array<long, Int>::reclaim_second_values();
            freed += zero_any::Array<unsigned long, Int>::reclaim_second_values();
            freed += zero_any::Array<short, Int>::reclaim_second_values();
            freed += zero_any::Array<unsigned char, Int>::reclaim_second_values();
            freed += zero_any::Array<ComplexDouble, Int>::reclaim_second_values();
            freed += zero_any::Array<Point3, Int>::reclaim_second_values();
            freed += zero_any::Array<Cell, Int>::reclaim_second_values();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[cst2-reclaim @ iter %ld] %zu second value(s) freed in %.3f ms\n",
                         static_cast<long>(k_iter), freed, ms);
            // Where each type's identifiers stand (second_value_stats()): the
            // trend of "in use" against the capacity is the early warning of the
            // exhaustion that stops the run; a refused reclaim freed nothing
            // because it could not run.
            const auto report = [](const char* name, const SecondValueStats& stats)
            {
                if (stats.handed_out == 0 && stats.refused() == 0)
                {
                    return;
                }
                std::fprintf(stderr, "  %-14s %zu in use / %zu (%.2f%%), %zu free, %zu reclaim(s), %zu refused%s\n",
                             name, stats.in_use(), stats.capacity,
                             stats.capacity != 0
                                 ? 100.0 * static_cast<double>(stats.in_use()) / static_cast<double>(stats.capacity)
                                 : 0.0,
                             stats.free, stats.reclaims, stats.refused(),
                             stats.degraded ? "  DEGRADED: reclaims refused" : "");
            };
            report("double", zero_any::Array<double, Int>::second_value_stats());
            report("float", zero_any::Array<float, Int>::second_value_stats());
            report("int", zero_any::Array<int, Int>::second_value_stats());
            report("long", zero_any::Array<long, Int>::second_value_stats());
            report("unsigned long", zero_any::Array<unsigned long, Int>::second_value_stats());
            report("short", zero_any::Array<short, Int>::second_value_stats());
            report("unsigned char", zero_any::Array<unsigned char, Int>::second_value_stats());
            report("complex_double", zero_any::Array<ComplexDouble, Int>::second_value_stats());
            report("Point3", zero_any::Array<Point3, Int>::second_value_stats());
            report("Cell", zero_any::Array<Cell, Int>::second_value_stats());
        }
    };

    advisor_tick(0); // post-init snapshot before the first iteration

    auto last = std::chrono::steady_clock::now();

    for (Int k = 0; k < iter_max; ++k)
    {
        if (k > 0)
        {
            advisor_tick(k);
            cst2_reclaim_tick(k);
        }
        if (k % 100 == 0)
        {
            const auto now = std::chrono::steady_clock::now();
            std::cout << "iter " << k << " : time : " << std::chrono::duration<double>(now - last).count() << std::endl;
            last = now;
        }

        // ring_init/analyze_ring_sweep are declared in ring_log_debug.h and
        // defined in every build of the library, so the guard need not be a
        // directive: the discarded branch still links.
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_init();
        }

        const Int size   = tab_size[k];
        const Int start0 = tab_start[k];

        Int end0 = start0 + size;

        end0 = std::min(end0, sz);

        if constexpr (test_captur_bug)
        {
            if (injection == 0 && k == cycle_alea)
            {
                std::cout << "Triggering abort()! cycle : " << k << std::endl;
                std::abort(); // triggers an abrupt program abort
            }
        }
        if (k % freq_raz == 0)
        {
            bench_phase_raz_sweep(bench_arrays, k, size, start0, end0);
            bench_phase_struct_raz(bench_arrays, start0, end0);
        }
        else
        {
            // Compute loop: bench_phase_compute.cpp (see bench_phases.h).
            bench_phase_compute(bench_arrays, k, start0, end0, tab_nba1, tab_nba2, val_sum, long_val_sum, v_real, v_im);
            // The struct arrays' window, its own region (bench_phase_struct.cpp).
            bench_phase_struct_compute(bench_arrays, k, start0, end0, tab_nba1, tab_nba2, point_sum);
            // tab_mask's update, its own region (bench_phase_mask.cpp).
            bench_phase_mask_compute(bench_arrays, k, mask_count);
            // tab_conn's renumbering, its own region (bench_phase_conn.cpp).
            bench_phase_conn_compute(bench_arrays, k, conn_sum);
        }

        // ─── STRESS TESTS ──────────────────────────────────────────────────
        // Extra work per k iteration, gated by disjoint k-modulos so each runs
        // ~1/5 of iterations — enough to surface rare bugs, not enough to
        // dominate baseline perf: (a) a short team and (b) a dynamic schedule,
        // parallel regions on a mid-array window (~interval/4 elements); (c)
        // the serial range API on a fixed window; (d) fill_range_mt from every
        // thread on ~interval/2 elements.

        // (a) n-1 threads — exercise the case where one thread never enters
        //     the parallel region. Rank-based bookkeeping that implicitly
        //     assumes "all threads participate" will surface here.
        //
        //     With ONE thread there is nobody to leave out, but the block
        //     still runs, on that thread: what the bench WRITES must not
        //     depend on the thread count. Every seed-pinned oracle (the fuzz
        //     sweep's sequential reference, the seq/omp compare) assumes so,
        //     and an OMP_NUM_THREADS=1 run that skipped this window had a
        //     different long_val_sum from every multi-thread run of the same
        //     seed -- a bench artefact reported as a content mismatch. The
        //     floor of 2 also covers the serial build (max_thr is 1 there,
        //     and the num_threads clause is ignored).

        if (k % 5 == 2)
        {
            const Int team  = std::max<Int>(max_thr, 2);
            const Int beg_s = std::max<Int>(start0 / 2, 0);
            const Int end_s = std::min(beg_s + (interval / 4), sz);

            bench_phase_stress_teams(bench_arrays, team, beg_s, end_s);
        }

        // (b) schedule(dynamic, grain) — scatters iterations so several
        //     threads land on the SAME chunk simultaneously. Stresses the
        //     CAS / Dekker paths that stay mostly idle under the default
        //     static schedule.
        //     Grain sizing (cs = EFFECTIVE chunk size — active_chunk_size():
        //     the runtime geometry in dynamic builds, == static_chunk_size in
        //     static builds where the seed is never changed):
        //       — grain = 1        → OMP dispatch dominates
        //       — grain = cs       → 1 task per chunk, no intra-chunk
        //                            contention
        //       — grain = cs/4     → 4 tasks per chunk, multiple threads
        //                            interleave inside the same chunk,
        //                            dispatch overhead still negligible.
        if (k % 5 == 3)
        {
            const Int beg_s = std::max<Int>(start0 / 2, 0);
            const Int end_s = std::min(beg_s + (interval / 4), sz);
            // NOT static_chunk_size: in dynamic builds that is the compile-time
            // default (64), while arrays run at chunk_dyn_size (512) — grain
            // would be 16, i.e. 32 tasks per 512-chunk instead of the 4
            // documented above.
            [[maybe_unused]] const Int grain = std::max<Int>(1, static_cast<Int>(active_chunk_size()) / 4);

            bench_phase_stress_dynamic(bench_arrays, grain, beg_s, end_s);
        }
        // (c) fill_range / copy_range -- the bulk range API, one decision per
        //     CHUNK where operator[] takes one per element. Serial contract
        //     (no synchronisation with concurrent writers or with a
        //     compression pass), so it runs OUTSIDE any parallel region, and
        //     therefore stays SMALL: a fixed window of 1029 elements -- a few
        //     chunks at any geometry, unaligned on both ends, and the SAME in
        //     every build (the vector reference has no chunk size to agree
        //     on) -- on the k-modulo slot the two stress tests above leave
        //     free. The phase end just flushed every deferred op, hence
        //     assume_drained: without it each call drains the WHOLE array
        //     (every chunk, every rank), which dwarfed the parallel work of
        //     the iteration. The same call sites compile against std::vector
        //     (vector_range_ops.h) and against the chunked arrays
        //     (array_chunk_range_ops.h): the reference data written by the
        //     vector build carry these writes, and the check builds compare
        //     them like any other.
        if (k % 5 == 4)
        {
            const auto r0 = static_cast<std::size_t>(std::max<Int>(start0 / 4, 0)) + 3;
            const auto r1 = std::min(r0 + 1029, static_cast<std::size_t>(sz));

            // DO NOT strip the  below: this file's `using
            // namespace shrink_array` is itself under #ifdef SHRINK_ARRAY, so it
            // is absent from the std::vector baseline build, where
            // assume_drained would then be invisible (ADL does not apply to a
            // variable). An IDE inspection already erased them once, silently
            // breaking that variant.
            // Constant fills, one per outcome on a chunked array: the second
            // value (a fully covered chunk goes constant outright on zero_any),
            // the first value (its buffer is released), a plain value
            // (materialised, then raw-stored), an Int and a bool (per-element
            // path: bool keeps its state in the proxy).
            fill_range(tab1, r0, r1, second_value_d, assume_drained);
            fill_range(tab2, r0, r1, 0., assume_drained);
            fill_range(tab3, r0, r1, 2.5, assume_drained);
            fill_range(tab_int, r0, r1, second_value_i, assume_drained);
            fill_range(tab_bool, r0, r1, true, assume_drained);
            fill_range(tab_bool_bits, r0, r1, true, assume_drained);

            // Copies from a plain vector: one dense, one mostly the second value
            // (so the copy lands on chunks that compress right after).
            std::vector<double> src(r1 - r0);
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                src[i] = std::sin(static_cast<double>(r0 + i) + static_cast<double>(k));
            }

            copy_range(tab4, r0, r1, src.begin(), assume_drained);
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                src[i] = (i % 97 == 0) ? static_cast<double>(i) : second_value_d2;
            }

            copy_range(tab5, r0, r1, src.begin(), assume_drained);
        }
        // (d) fill_range_mt -- the bulk range API from EVERY thread at once:
        //     one write session per chunk span (pending ops, compressor wait,
        //     Dekker announce, raw stores, queue). 64-element blocks dealt
        //     round-robin, so each chunk is shared by several threads; eight
        //     arrays, every constant outcome plus a plain value alternating
        //     between blocks. Larger than (c) -- interval/2 elements, from
        //     start0/3, unaligned -- since it is parallel and ~8x cheaper per
        //     element than operator[] on zero_any. Same writes in the
        //     std::vector build (vector_range_ops.h), so the check builds
        //     compare them like any other.
        if (k % 5 == 1)
        {
            const Int beg_s = std::max<Int>(start0 / 3, 0) + 7;
            const Int end_s = std::min(beg_s + (interval / 2), sz);

            bench_phase_range_mt(bench_arrays, k, beg_s, end_s);
        }
        // ─── END STRESS TESTS ──────────────────────────────────────────────

        cv_val_sum = ComplexDouble(v_real, v_im);

        // Always printed regardless of the cadence: the first and last
        // iterations, which bracket the run, and any non-finite checksum --
        // a divergence that first appears between two sampled iterations would
        // otherwise be reported against the wrong one.
        {
            const Int  ksum_period = freq_ksum > 0 ? freq_ksum : std::max<Int>(1, freq_conso_mem);
            const bool ksum_bad =
                !std::isfinite(val_sum) || !std::isfinite(cv_val_sum.real()) || !std::isfinite(cv_val_sum.imag());
            if (k % ksum_period == 0 || k == 0 || k == iter_max - 1 || ksum_bad)
            {
                std::cout << "k : " << k << " val_sum = " << val_sum << " long_val_sum = " << long_val_sum
                          << " cv_val_sum = " << cv_val_sum << " point_sum = " << point_sum
                          << " mask_count = " << mask_count << " conn_sum = " << conn_sum << std::endl;
            }
        }

        if constexpr (chunk_config::ring_log_debug)
        {
            if (shrink_array::analyze_ring_sweep())
            {
                std::abort();
            }
        }
        if constexpr (print_mem) // see the note on the first site
        {
            if (k % freq_conso_mem == 0)
            {
                const std::string msg_iter = "iteration : " + std::to_string(k);
                memory_consumption(msg_iter);
            }
        }

        if CONSTEXPR (check_sum)
        {
            if (k % freq_check_sum == 0)
            {
                bench_print_checksums(bench_arrays, check_sum);
            }
        }

        if CONSTEXPR (stats)
        {
            if (k % freq_stats == 0)
            {
                bench_stats(bench_arrays, stats);
            }
        }

        bench_check_all(bench_arrays, k, bench_io_flags);

        // ParamChunk::force_compression() and bench_phase_compress_all() are both
        // declared and defined in EVERY variant (the latter is a no-op in the
        // std::vector build), so the whole block can be a runtime-discarded
        // `if constexpr` rather than a directive.
        if constexpr (need_compression)
        {
            // to activate to check compression embededd
            // The `omp parallel` below is what needs the guard: without
            // OpenMP the pragma is ignored and force_compression() would run
            // once on the master instead of once per thread.
            if constexpr (chunk_config::use_fast_block_compression && chunk_config::has_ompt &&
                          build_config::use_openmp)
            {
                const bool have_to_compress = k % freq_compress == 0;
                if (have_to_compress)
                {
                    bench_par::region(
                        [&]([[maybe_unused]] bench_par::Team& team)
                        {
                            // Compiled in the std::vector build too: a no-op there.
                            ParamChunk::force_compression();
                        });
                }
            }


            // zero and zero_one_value always; zero_any only with fast block
            // compression (normally useless for zero_one_value and zero_any, whose writes
            // compress as they go).
            // Periodic, plus the last iteration so no end-of-run miss escapes.
            // A persisting miss stays visible at the next audit: a
            // materialised constant chunk stays so until it is written.
            const bool audit_now = (k % freq_audit == 0) || (k + 1 == iter_max);

            if constexpr (chunk_config::use_zero || chunk_config::use_zero_one_value ||
                          (chunk_config::use_zero_any && chunk_config::use_fast_block_compression))
            {
                if (audit_now)
                {
                    std::cout << "Check compression " << "\n";
                    bench_phase_compress_all(bench_arrays);
                }
            }

            if constexpr (print_mem) // see the note on the first site
            {
                if (audit_now) // "after compress" only means something right after one
                {
                    const std::string msg_iter_c = "iteration : " + std::to_string(k) + " after compress";
                    memory_consumption(msg_iter_c);
                }
            }
        }

        if (k > iter_stop)
        {
            std::cout << "\nStop by iter_stop\n";
            break;
        }
    }

    std::cout << "\nval_sum    = " << val_sum << "\n";
    std::cout << "\nlong_val_sum = " << long_val_sum << "\n";
    std::cout << "\ncv_val_sum = " << cv_val_sum << std::endl;
    std::cout << "\npoint_sum  = " << point_sum << "\n";
    std::cout << "mask_count = " << mask_count << "\n";
    std::cout << "conn_sum   = " << conn_sum << "\n";

    bench_print_checksums(bench_arrays, check_sum);

    if CONSTEXPR (stats)
    {
        bench_stats(bench_arrays, stats);
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "\nElapsed time computation : " << elapsed_seconds.count() << "s\n\n";

    memory_consumption("end");
    std::cout << "\n";

    // Number of committed re-chunks performed during the run (dynamic builds only;
    // static builds never re-chunk, so the line is omitted entirely there).
    if constexpr (has_advisor)
    {
        // Full re-chunk report: the advisor/decider runs in every dynamic-chunked
        // build (CHUNK_VARIANT_ZERO*), so rechunk_decider (and its per-event
        // log) exists here for all of them.
        std::cout << tc.bold;
        rechunk_decider.print_rechunk_log(stdout);
        std::cout << tc.reset << "\n";
    }
    else if constexpr (build_config::chunk_dynamic)
    {
        // A dynamic build with no advisor wired: just the committed count.
        std::cout << tc.bold << "Re-chunks performed: " << tc.reset << g_rechunk_count << "\n\n";
    }

    // Self-checks, array release, final report and library shutdown:
    // bench_teardown.cpp. The ORDER in there carries meaning, hence the single
    // block.
    bench_teardown(bench_arrays, test_stl);

    return 0;
}
