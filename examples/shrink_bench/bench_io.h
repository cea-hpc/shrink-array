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

#pragma once

#include <type_traits> // remove_cvref_t (diagnostic printing)

// NOTE (dynamic builds): every chunk-geometry computation in the diagnostic
// paths below MUST come from the array under inspection (data_check.chunk_size():
// k / chunk_size(), k % chunk_size()), never from the compile-time globals
// (static_chunk_size, ParamChunk::get_decalage_chunk()...). In CHUNK_DYNAMIC runs the
// effective size (chunk_dyn_size, e.g. 512) differs from the compile-time
// default (64): the static shift over-indexes get_chunk_list() (libstdc++ assert
// in the mismatch report) and the static loop bounds only scan the first 64
// values of a 512-wide chunk.


#include "bench_prelude.h" // Int, UInt, chunk_config::

#include <algorithm> // std::max
#include <cstddef>   // size_t
#include <cstdint>   // std::uint64_t, std::uintptr_t
#include <cstdlib>   // std::abort
#include <fstream>   // std::ifstream, std::ofstream
#include <iostream>  // std::cout, std::cerr
#include <sstream>   // std::ostringstream
#include <string>    // std::string, std::to_string, std::getline
#include <unistd.h>  // unlink
#include <format>    // std::format
#include <vector>    // std::vector

#include "bench_config.h"         // have_bzip2
#include "bench_utils.h"          // is_vector
#include "chunk/ring_log_debug.h" // analyze_ring_sweep / dump_ring decls

// -----------------------------------------------------------------------------
// BZip2 compress/decompress helpers, defined in bench_io.cpp when BZIP2 is on.
// The write_binary_file() template below dispatches to compress_file()
// through the `have_bzip2` compile-time flag (bench_config.h), so declaring
// them unconditionally is harmless (unused declarations don't link).
// -----------------------------------------------------------------------------
void compress_file(const std::string& source_file, const std::string& dest_file);
/// @brief Decompress the bzip2 file @p source_file into @p dest_file; aborts on an I/O or decompression error.
/// @param source_file Compressed input, starting with the original length.
/// @param dest_file Decompressed output.
void decompress_file(const std::string& source_file, const std::string& dest_file);

// -----------------------------------------------------------------------------
// write_binary_file<Vec> — dump a plain std::vector to disk as raw bytes
// alongside a companion checksum file. Compresses with bzip2 only when
// no_compress is null (--compress / COMPRESS=1).  Aborts on non-vector Vec
// (chunk arrays cannot be written directly, they need to be materialised
// first — not implemented).
// -----------------------------------------------------------------------------
// Compacted run-length dump of a value sequence. Runs longer than 2 elements
// are rendered as "v ...xN... v" (N = run length — keeping the count visible
// is deliberate: a 64-long run and a 512-long run must not look alike when
// hunting geometry bugs); short runs are printed element by element. `sink`
// receives std::string fragments (log_message, or a std::cerr wrapper) and
// `fmt` formats one value (the local format_val lambdas).
template <typename Sink, typename Fmt, typename T>
static void log_values_compacted(Sink&& sink, Fmt&& fmt, const T* values, UInt count)
{
    UInt p = 0;
    while (p < count)
    {
        const T    current_val = values[p];
        const UInt start_idx   = p;

        // Count how many identical consecutive values follow
        while (p < count && values[p] == current_val)
        {
            ++p;
        }

        const UInt        run_length = p - start_idx;
        const std::string val_str    = fmt(current_val);
        if (run_length > 2)
        {
            std::string line = val_str;
            line += " ...x";
            line += std::to_string(run_length);
            line += "... ";
            line += val_str;
            line += ' ';
            sink(line);
        }
        else
        {
            for (UInt i = 0; i < run_length; ++i)
            {
                sink(val_str + " ");
            }
        }
    }
    sink("\n");
}

/// @brief Write @p data to @p filename as raw bytes and @p checksum to "<filename>_check_sum", then, unless
///        @p no_compress is set, bzip2 it to "<filename>.bz2" (see the comment above log_values_compacted()).
/// @param filename Output file.
/// @param data std::vector to write; any other type aborts.
/// @param checksum Value written to the checksum file.
/// @param special_writing true, with bool or short elements: write each element as one bool.
/// @param no_compress Non-null: skip the compression.
/// @param checksum_only true (--checksum_only): write the checksum file alone. The data
///        file and its .bz2 are not written, and stale copies left by an earlier full
///        reference are removed, so a later checksum miss cannot be compared against
///        data from another oracle (read_check_binary_file then aborts).
template <typename Vec>
void write_binary_file(const std::string& filename,
                       const Vec&         data,
                       std::uint64_t      checksum,
                       bool               special_writing,
                       const char*        no_compress,
                       bool               checksum_only = false)
{
    using CV = std::remove_reference_t<Vec>;
    using V  = std::remove_const_t<CV>;

    const std::string filename_gz        = filename + ".bz2";
    const std::string filename_check_sum = filename + "_check_sum";

    if constexpr (is_vector<V>())
    {
        if (checksum_only)
        {
            std::ofstream file_check_sum(filename_check_sum, std::ios::binary);
            if (!file_check_sum)
            {
                std::cerr << std::format("Error open file : {}\n", filename_check_sum);
                std::abort();
            }
#pragma omp critical
            {
                std::cout << std::format("Write : {} (checksum only)\n", filename_check_sum);
            }
            file_check_sum.write(reinterpret_cast<const char*>(&checksum), sizeof(uint64_t));
            file_check_sum.close();
            unlink(filename.c_str());
            unlink(filename_gz.c_str());
            return;
        }

        using T = Vec::value_type;
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            std::cerr << std::format("Error open file : {}\n", filename);
            return;
        }

        std::ofstream file_check_sum(filename_check_sum, std::ios::binary);
        if (!file_check_sum)
        {
            std::cerr << std::format("Error open file : {}\n", filename_check_sum);
            return;
        }
        const size_t size = data.size();

#pragma omp critical
        {
            std::cout << std::format("Write : {}\n", filename_check_sum);
            std::cout << std::format("Write : {}\n", filename);
        }

        file_check_sum.write(reinterpret_cast<const char*>(&checksum), sizeof(uint64_t));
        file_check_sum.close();

        // Because we use short for bool...
        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, short>)
        {
            if (special_writing)
            {
                for (size_t k = 0; k < size; ++k)
                {
                    const bool val = data[k];
                    file.write(reinterpret_cast<const char*>(&val), sizeof(bool));
                }
            }

            else
            {
                if constexpr (!std::is_same_v<T, bool>)
                {
                    file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
                }
            }
        }
        else
        {
            // Write the content of vector
            file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
        }

        file.close();
    }
    else
    {
        std::cout << std::format("write is not avalaible on chunk vector\n");
        std::abort();
    }

    if (no_compress)
    {
        return;
    }

    if constexpr (have_bzip2)
    {
        compress_file(filename, filename_gz);
#pragma omp critical
        {
            std::cout << std::format("Compression : {} -> {}\n", filename, filename_gz);
        }
        unlink(filename.c_str());
    }
}

// -----------------------------------------------------------------------------
// read_check_binary_file<Vec> — read back a bench dump written by
// write_binary_file() and cross-check it against `data_check`, which may
// be either a std::vector or a chunk array.  When the checksum mismatches,
// the raw data is uncompressed and compared element-by-element; the first
// diverging element triggers a detailed dump of the chunk state (in a shrink
// build only -- the dump is `if constexpr`-guarded on a dependent object, not
// compiled out by the preprocessor) and calls std::abort().
//
// Trace mode:
//   - write_trace=true : dump the (ind, k) coordinate of the first divergence
//     to trace_err.txt so a follow-up run can jump straight to it.
//   - read_trace=true  : replay from trace_err.txt written by a previous run.
// -----------------------------------------------------------------------------
template <typename Vec>
void read_check_binary_file(Int                ind,
                            const std::string& tab_name,
                            const std::string& filename,
                            const Vec&         data_check,
                            std::uint64_t      lchs,
                            const char*        no_compress,
                            bool               read_trace,
                            bool               write_trace)
{
    using T = Vec::value_type;

    const std::string filename_gz        = filename + ".bz2";
    const std::string filename_check_sum = filename + "_check_sum";

    uint64_t lchs_read = 0;
#pragma omp critical
    {
        std::cout << std::format("Read : {}\n", filename_check_sum);
    }
    std::ifstream file_check_sum(filename_check_sum, std::ios::binary);
    if (!file_check_sum)
    {
        std::cerr << "Error open file  : " << filename_check_sum << '\n';
        std::abort();
    }

    file_check_sum.read(reinterpret_cast<char*>(&lchs_read), sizeof(uint64_t));

    if (lchs_read == lchs && !read_trace)
    {
#pragma omp critical
        {
            std::cout << tab_name << " Checksum OK !\n";
        }
        return;
    }
    else
    {
        if (!read_trace)
        {
            std::cerr << tab_name << "\nError checksum ! checksum read : " << lchs_read
                      << " checksum calculate : " << lchs;
        }

        if (!no_compress)
        {
            std::cerr << " --> we decompress and read the file : " << tab_name;
        }
        std::cerr << "\n";
    }

    // A reference written with --checksum_only has no data to compare against:
    // the miss cannot be located, so stop here with a clear message rather than
    // through a missing-file error further down.
    if (::access(filename.c_str(), F_OK) != 0 && ::access(filename_gz.c_str(), F_OK) != 0)
    {
#pragma omp critical
        {
            std::cerr << std::format(
                "{}: iteration {}: elements cannot be compared (checksum read {}, calculated {}): the reference "
                "has no data file ({} / {}): it was written with --checksum_only. Regenerate a "
                "full reference (vector baseline without --checksum_only) to locate the "
                "diverging element.\n",
                tab_name, ind, lchs_read, lchs, filename, filename_gz);
        }
        std::abort();
    }

    const Int sz = data_check.size();

    // Whether `filename` (.bin) is a decompression TEMPORARY we created here
    // (compressed regime) or the oracle's persistent copy (a baseline without
    // --compress, or without bzip2, keeps the plain .bin and never writes a .bz2). Only a
    // temporary may be unlinked after reading — deleting the persistent copy
    // would destroy the oracle's only instance for every later run.
    bool bin_is_temp = false;

    if constexpr (have_bzip2)
    {
        // Regime-agnostic read: prefer the .bz2 when it exists (compressed
        // oracle, .bin was unlinked at write time); otherwise fall back to a
        // plain .bin left by a baseline run without --compress. Both absent -> go
        // through decompress_file for its clear missing-source error.
        if (!no_compress && (::access(filename_gz.c_str(), F_OK) == 0 || ::access(filename.c_str(), F_OK) != 0))
        {
            decompress_file(filename_gz, filename);
            bin_is_temp = true;
        }
    }
#pragma omp critical
    {
        std::cout << "Read : " << filename << std::endl;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        // After a checksum miss: returning would let the run pass unchecked.
        std::cerr << "Error open file  : " << filename << '\n';
        std::abort();
    }

    // Helper lambda to format values as strings, handling char-to-Int conversion
    const auto format_val = [&](auto val) -> std::string
    {
        if constexpr (std::is_same_v<T, char> || std::is_same_v<T, unsigned char> || std::is_same_v<T, signed char>)
        {
            return std::to_string(static_cast<int>(val));
        }
        else
        {
            std::ostringstream oss;
            oss << val;
            return oss.str();
        }
    };

    // No directive: the whole lambda lives inside template <typename Vec>
    // read_check_binary_file, so every data_check.* call below is DEPENDENT and
    // the `if constexpr` inside it is genuinely not instantiated for a
    // std::vector Vec. It is unused there -- io_res == 2 is unreachable without
    // a shrink build -- hence [[maybe_unused]]. Capture-default rather than a
    // list for the same reason: format_val is only named in that discarded
    // branch, and an explicit capture that the std::vector instantiation
    // never reads draws -Wunused-lambda-capture from clang.
    [[maybe_unused]]
    const auto lbd_check_trace = [&](std::ifstream& trace_file, Int k, auto& value)
    {
        // The body as a whole, not statement by statement: everything below
        // reads data_check, whose type is the DEPENDENT template parameter, so
        // for a std::vector Vec none of it is instantiated.
        if constexpr (chunk_config::is_shrink_enabled)
        {
            std::string line;
            std::string current_tab;

            bool ok = false;

            std::getline(trace_file, line);

            std::getline(trace_file, line);
            // Detect tab_name
            if (line.find(" not equal !") != std::string::npos)
            {
                const size_t bracket_pos = line.find('[');
                if (bracket_pos != std::string::npos)
                {
                    current_tab = line.substr(0, bracket_pos);
                    if (current_tab == tab_name)
                    {
                        ok = true;
                    }
                }
            }

            if (!ok)
            {
                return;
            }

            std::getline(trace_file, line);
            std::getline(trace_file, line);

            const Int num_chunk = k / static_cast<Int>(data_check.chunk_size());
            const Int num       = k % static_cast<Int>(data_check.chunk_size());

            bool                  is_values   = false;
            bool                  is_cst_val1 = false;
            [[maybe_unused]] bool is_cst_val2 = false; // read only by the chunk2 / chunk2a traces
            while (std::getline(trace_file, line))
            {
                if (line.find("is_values") != std::string::npos)
                {
                    is_values = true;
                    break;
                }
                if (line.find("is_cst_val1") != std::string::npos)
                {
                    is_cst_val1 = true;
                    break;
                }
                // Pure text search on the trace line -- no variant API at all -- and the
                // enclosing lambda is generic, so the guard was never more than an
                // `if constexpr` written with the preprocessor.
                if constexpr (chunk_config::use_chunk2or2a)
                {
                    if (line.find("is_cst_val2") != std::string::npos)
                    {
                        is_cst_val2 = true;
                        break;
                    }
                }
            }

            const auto& chunk_t = data_check.get_chunk_list()[num_chunk];

            const bool is_val1 = chunk_t.is_cst_val1();

            // false everywhere but CHUNK2, exactly as the #else branch gave.
            bool is_val2 = false;
            if constexpr (chunk_config::use_chunk2)
            {
                static constexpr bool is_bool     = shrink_array::is_bool_like_v<T>; // bool, Bit
                static constexpr bool is_not_bool = !is_bool;

                if constexpr (is_not_bool)
                {
                    is_val2 = chunk_t.is_cst_val2(data_check.get_ind_second_value());
                }
                else
                {
                    is_val2 = chunk_t.is_cst_val2();
                }
            }

            const auto lbd_print = [&]
            {
                std::cerr << "value               : " << value << "\n";
                std::cerr << "chunk number        : " << num_chunk << "\n";
                std::cerr << "number inside chunk : " << num << "\n";
                std::cerr << "is values           : " << is_values << "\n";
                std::cerr << "is_cst_val1         : " << is_val1 << "\n";
                if constexpr (chunk_config::use_chunk2or2a)
                {
                    std::cerr << "is_cst_val2         : " << is_val2 << "\n";
                }

                if constexpr (chunk_config::ring_log_debug)
                {
                    shrink_array::analyze_ring_sweep();
                }
            };

            // Detect value category
            if (is_cst_val1)
            {
                if (is_val1)
                {
                    std::cerr << "\nwas cst_val1 it is\n";
                    lbd_print();
                    exit(0);
                }

                std::cerr << "\nwas cst_val1 is not!\n";
                lbd_print();
                exit(0);
            }

            if constexpr (chunk_config::use_chunk2or2a)
            {
                if (is_cst_val2)
                {
                    if (is_val2)
                    {
                        std::cerr << "\nwas cst_val2 it is\n";
                        lbd_print();
                        exit(0);
                    }

                    std::cerr << "\nwas cst_val2 is not!\n";
                    lbd_print();
                    exit(0);
                }
            }
            if (is_values)
            {
                if (!is_val1 && !is_val2)
                {
                    std::cerr << "\nwas values it is\n";
                    lbd_print();
                    constexpr T v{}; // T{}: the struct elements have no conversion from 0
                    bool        te = true;
                    for (UInt p = 0; p < data_check.chunk_size(); ++p)
                    {
                        if (chunk_t.values_[p] != v)
                        {
                            te = false;
                            break;
                        }
                    }

                    if (te)
                    {
                        std::cerr << "should be cst_val1\n";
                    }
                    else
                    {
                        bool       all_equal = true;
                        const auto first     = chunk_t.values_[0];

                        for (UInt p = 1; p < data_check.chunk_size(); ++p)
                        {
                            if (chunk_t.values_[p] != first)
                            {
                                all_equal = false;
                                break;
                            }
                        }

                        if (all_equal)
                        {
                            std::cerr << "All values of chunk are identical: " << first << "\n";
                        }
                        else
                        {
                            std::cerr << "All values of chunk (compacted):\n";
                            const T* vals = chunk_t.values_; // atomic<T*> implicit load or plain ptr
                            log_values_compacted([](const std::string& m) { std::cerr << m; }, format_val, vals,
                                                 data_check.chunk_size());
                        }
                    }

                    exit(0);
                }

                std::cerr << "\nwas values is not!\n";
                lbd_print();
                exit(0);
            }
        }
    };

    std::ofstream trace_file_write;
    std::ifstream trace_file_read;

    // Open the write-trace file once, up front, so both the bool and
    // non-bool branches below share the same guarded write path (tab_bool
    // errors included under --write_trace).
    if (write_trace)
    {
        trace_file_write.open("trace_err.txt");
        if (!trace_file_write.is_open())
        {
            std::cerr << "Failed to open trace_err.txt\n";
        }
    }

    if (read_trace)
    {
        trace_file_read.open("trace_err.txt");
        if (!trace_file_read.is_open())
        {
            std::cerr << "Failed to open trace_err.txt\n";
        }
    }
    Int ind_read = -1;

    if (read_trace)
    {
        trace_file_read >> ind_read;
    }
    bool ok_read_trace = false;
    if (read_trace)
    {
        if (ind_read == ind)
        {
            ok_read_trace = true;
        }
        else
        {
            trace_file_read.close();
        }
    }
    Int k_read = -1;
    if (ok_read_trace)
    {
        trace_file_read >> k_read;
    }
    else
    {
        trace_file_read.close();
    }


    if constexpr (std::is_same_v<T, bool>)
    {
        for (Int k = 0; k < sz; ++k)
        {
            T value;
            file.read(reinterpret_cast<char*>(&value), sizeof(T));

            const std::string full_tab_name = tab_name + "[" + std::to_string(k) + "]";

            const T value_check = data_check[k];

            if (read_trace && k_read == k)
            {
                lbd_check_trace(trace_file_read, k, value);
            }
            if (value != value_check)
            {
                // Lambda to log to std::cerr and optionally to the trace file
                const auto log_message = [&](const std::string& msg)
                {
                    std::cerr << msg;
                    if (write_trace && trace_file_write.is_open())
                    {
                        trace_file_write << msg;
                    }
                };

                if (write_trace && trace_file_write.is_open())
                {
                    trace_file_write << ind << "\n";
                    trace_file_write << k << "\n";
                }

                // Usage in your code
                log_message(full_tab_name + " not equal! read: " + std::to_string(value) +
                            " != " + std::to_string(data_check[k]) + "\n");

                std::cerr << "iteration: " + std::to_string(ind) + "\n";
                // One `if constexpr` over both former blocks: the CHUNK2 one below
                // reads chunk_t, declared here, so they cannot be split.
                if constexpr (chunk_config::is_shrink_enabled)
                {
                    const Int num_chunk = k / static_cast<Int>(data_check.chunk_size());

                    log_message("\nchunk number        : " + std::to_string(num_chunk) + "\n");
                    const Int num = k % static_cast<Int>(data_check.chunk_size());
                    log_message("number inside chunk : " + std::to_string(num) + "\n");
                    const auto& chunk_t = data_check.get_chunk_list()[num_chunk];
                    using ValueType     = decltype(chunk_t.values_); // infer the type

                    T* raw_ptr = nullptr;

                    if constexpr (is_atomic_pointer<ValueType>::value)
                    {
                        // It's a std::atomic<T*>
                        raw_ptr = chunk_t.values_.load();
                    }
                    else
                    {
                        // It's a raw pointer
                        raw_ptr = chunk_t.values_;
                    }
                    auto addr = reinterpret_cast<std::uintptr_t>(raw_ptr);

                    const std::string s0 = std::format("0x{:x}", addr);

                    log_message("address values       : " + s0 + "\n");
                    addr = reinterpret_cast<std::uintptr_t>(&raw_ptr[num]);
                    std::ostringstream oss1;
                    oss1 << "0x" << std::hex << addr;
                    const std::string s1  = oss1.str();
                    const Int         val = *reinterpret_cast<Int*>(addr);
                    log_message("value  = " + std::to_string(val) + "\n");
                    log_message("address values[" + std::to_string(num) + "] : " + s1 + "\n");

                    if constexpr (chunk_config::use_chunk2)
                    {
                        if (chunk_t.is_cst_val1())
                        {
                            log_message("is_cst_val1\n");
                        }
                        else if (chunk_t.is_cst_val2())
                        {
                            log_message("is_cst_val2\n");
                        }
                        else
                        {
                            log_message("is_values\n");
                        }
                    }
                }

                if (trace_file_write.is_open())
                {
                    trace_file_write.close();
                }

                std::abort();
            }
        }
    }
    else
    {
        std::vector<T> vec(sz);
        file.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));

        Int nn = 0;

        double ratio_max = 0;

        constexpr T zero{};

        constexpr bool is_complex = std::is_same_v<T, ComplexDouble>;

        for (Int k = 0; k < sz; ++k)
        {
            const T value       = vec[k];
            const T value_check = data_check[k];

            const bool ok1 = value == zero && value_check == zero;
            const bool ok2 = value == value_check;

            bool ok3 = false;

            double ratio = 0;

            if (read_trace && k_read == k)
            {
                lbd_check_trace(trace_file_read, k, value);
            }

            if constexpr (std::is_same_v<T, double> || is_complex)
            {
                if (!ok2 && value_check != zero)
                {
                    if constexpr (is_complex)
                    {
                        ratio = std::fabs((std::norm(value) - std::norm(value_check)) / std::norm(value_check));
                    }
                    else
                    {
                        ratio = std::fabs((abs(value) - abs(value_check)) / abs(value_check));
                    }
                    ratio_max = std::max(ratio, ratio_max);
                    ok3       = ratio < 1.e-10;
                }
            }

            if constexpr (std::is_same_v<T, float>)
            {
                if (!ok2 && value_check != zero)
                {
                    ratio = std::fabs((abs(value) - abs(value_check)) / abs(value_check));

                    ratio_max = std::max(ratio, ratio_max);
                    ok3       = ratio < 1.e-6;
                }
            }

            if (!ok1 && !ok2 && !ok3)
            {
                if (write_trace && trace_file_write.is_open())
                {
                    trace_file_write << ind << "\n";
                    trace_file_write << k << "\n";
                }

                const auto log_message = [&](const std::string& msg)
                {
                    std::cerr << msg;
                    if (write_trace && trace_file_write.is_open())
                    {
                        trace_file_write << msg;
                    }

                    if constexpr (chunk_config::ring_log_debug)
                    {
                        shrink_array::analyze_ring_sweep();
                    }
                };

                if constexpr (std::is_same_v<T, double>)
                {
                    log_message(std::string("")); // Reset stream flags handled below if needed
                }
                if constexpr (std::is_same_v<T, float>)
                {
                    log_message(std::string("")); // Same as above
                }

                // unsigned char goes through operator<< as a CHARACTER: a mismatch
                // on tab_char would print a control byte instead of the number.
                // Promote the narrow types to a wide integer for display only;
                // every other T prints as is.
                const auto printable = [](const auto& v)
                {
                    using V = std::remove_cvref_t<decltype(v)>;
                    if constexpr (std::is_same_v<V, unsigned char> || std::is_same_v<V, signed char> ||
                                  std::is_same_v<V, char>)
                    {
                        return static_cast<int>(v);
                    }
                    else
                    {
                        return v;
                    }
                };

                std::ostringstream oss;
                oss << tab_name << "[" << k << "]"
                    << " not equal ! read : " << printable(value) << " != " << printable(data_check[k]);
                log_message(oss.str() + "\n");

                if constexpr (chunk_config::is_shrink_enabled)
                {
                    const Int   num_chunk = k / static_cast<Int>(data_check.chunk_size());
                    const auto& chunk_t   = data_check.get_chunk_list()[num_chunk];
                    log_message("chunk number        : " + std::to_string(num_chunk) + "\n");
                    const Int num = k % static_cast<Int>(data_check.chunk_size());
                    log_message("number inside chunk : " + std::to_string(num) + "\n");

                    // Detect whether a type is an atomic of a pointer, regardless of the pointed-to type

                    T* raw_ptr = nullptr;

                    using ValueType = decltype(chunk_t.values_);

                    if constexpr (is_atomic_pointer<ValueType>::value)
                    {
                        // It's a std::atomic<T*>
                        raw_ptr = chunk_t.values_.load();
                    }
                    else
                    {
                        // It's a raw pointer
                        raw_ptr = chunk_t.values_;
                    }

                    auto               addr_values_member = reinterpret_cast<std::uintptr_t>(&chunk_t.values_);
                    std::ostringstream oss_member;
                    oss_member << "0x" << std::hex << addr_values_member;
                    const std::string s_member = oss_member.str();

                    log_message("address &values       : " + s_member + "\n");

                    auto               addr = reinterpret_cast<std::uintptr_t>(raw_ptr);
                    std::ostringstream oss0;
                    oss0 << "0x" << std::hex << addr;
                    const std::string s0 = oss0.str();

                    log_message("address values       : " + s0 + "\n");

                    addr = reinterpret_cast<std::uintptr_t>(&raw_ptr[num]);
                    std::ostringstream oss1;
                    oss1 << std::hex << &raw_ptr[num];
                    const std::string s1 = oss1.str();

                    log_message("address values[" + std::to_string(num) + "] : " + s1 + "\n");
                }

                if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float> || is_complex)
                {
                    if (value_check != zero)
                    {
                        log_message(" ratio = " + std::to_string(ratio) + "\n");
                    }
                }

                if constexpr (chunk_config::use_chunk2or2a)
                {
                    // num_chunk is recomputed rather than hoisted out of the block
                    // above: the ratio report sitting between the two must keep
                    // running in the std::vector build, so the two cannot be merged
                    // into a single `if constexpr`.
                    const Int num_chunk = k / static_cast<Int>(data_check.chunk_size());

                    if (const auto& chunk_t = data_check.chunk_list_[num_chunk]; chunk_t.is_cst_val1() && value != zero)
                    {
                        log_message("\nis_cst_val1 should be not !\n");
                        {
                            const T* vals = chunk_t.values_;
                            log_values_compacted(log_message, format_val, vals, data_check.chunk_size());
                        }
                    }
                    // IIFE: the two spellings of is_cst_val2 cannot each be an
                    // `else if` condition. We are inside a function template, so the
                    // discarded branch is not instantiated.
                    else if (
                        [&]
                        {
                            // bool / Bit: one second value, no index.
                            if constexpr (chunk_config::use_chunk2 && !shrink_array::is_bool_like_v<T>)
                            {
                                return chunk_t.is_cst_val2(data_check.get_ind_second_value());
                            }
                            else
                            {
                                return chunk_t.is_cst_val2();
                            }
                        }())
                    {
                        log_message("\n[Debug] is_cst_val2 should be not !\n");

                        constexpr T zero{};
                        bool        test_zero = true;

                        for (UInt p = 0; p < data_check.chunk_size(); p++)
                        {
                            if (chunk_t.values_[p] != zero)
                            {
                                test_zero = false;
                            }
                        }

                        if (!test_zero)
                        {
                            log_message("chunk values contain : ");
                            {
                                const T* vals = chunk_t.values_;
                                log_values_compacted(log_message, format_val, vals, data_check.chunk_size());
                            }
                        }
                        else
                        {
                            log_message("\n[Info] should be cst_val1 (all zero)!\n");
                        }

                        if constexpr (chunk_config::use_chunk2a)
                        {
                            const T* m_ptr_cst_values = chunk_t.values_load(); // a cst2 chunk reads from its array
                            log_message("\nm_ptr_cst_val2 contain : ");
                            log_values_compacted(log_message, format_val, m_ptr_cst_values, data_check.chunk_size());
                        }
                    }
                    else
                    {
                        bool       all_equal = true;
                        const auto first_v   = chunk_t.values_[0];

                        for (UInt i = 1; i < data_check.chunk_size(); ++i)
                        {
                            if (chunk_t.values_[i] != first_v)
                            {
                                all_equal = false;
                                break;
                            }
                        }

                        log_message("\nis_values\n");

                        if (all_equal)
                        {
                            log_message("All values of chunk are identical: " + format_val(first_v) + "\n");
                        }
                        else
                        {
                            log_message("All values of chunk (compacted):\n");
                            {
                                const T* vals = chunk_t.values_;
                                log_values_compacted(log_message, format_val, vals, data_check.chunk_size());
                            }
                        }
                    }
                }

                log_message("\n");
                trace_file_write.close();
                std::abort();
            }
            else if (!ok1 && ok3)
            {
                nn++;
            }
        }

        if (nn)
        {
            std::cerr << nn << " : values are not perfectly equal ! : error max = ";
            std::cerr << std::scientific << ratio_max << "\n";
            std::cerr << std::defaultfloat;
        }
        else if (!read_trace)
        {
            std::cout << "no data error (but checksum error...)\n";
        }

        file.close();
    }

    if (!no_compress && bin_is_temp)
    {
        unlink(filename.c_str());
    }
}
