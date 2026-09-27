/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include <type_traits> // remove_cvref_t (diagnostic printing)

// NOTE (dynamic builds): every chunk-geometry computation in the diagnostic
// paths below MUST come from the array under inspection (data_check.chunk_size()
// / chunk_decalage() / chunk_modulo()), never from the compile-time globals
// (__chunk_size, ParamChunk::get_decalage_chunk()...). In CHUNK_DYNAMIC runs the
// effective size (chunk_dyn_size, e.g. 512) differs from the compile-time
// default (64): the static shift over-indexes getChunkList() (libstdc++ assert
// in the mismatch report) and the static loop bounds only scan the first 64
// values of a 512-wide chunk.


#include "bench_prelude.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib> // std::abort
#include <fstream>
#include <sstream> // the diagnostic strings (format_val, the address dumps)
#include <string>
#include <unistd.h> // unlink
#include <format>
#include <print>
#include <vector>

#include "bench_utils.h"
#include "chunk/ring_log_debug.h" // SEGV hunt: analyze_ring_sweep / dump_ring decls

// -----------------------------------------------------------------------------
// BZip2 compress/decompress helpers, defined in bench_io.cpp when BZIP2 is on.
// The write_binary_file() template below dispatches to compress_file()
// through the `build_config::has_bzip2` compile-time flag, so declaring
// them unconditionally is harmless (unused declarations don't link).
// -----------------------------------------------------------------------------
void compress_file(const std::string& source_file, const std::string& dest_file);
void decompress_file(const std::string& source_file, const std::string& dest_file);

// -----------------------------------------------------------------------------
// write_binary_file<VEC> — dump a plain std::vector to disk as raw bytes
// alongside a companion checksum file. Compresses with bzip2 only when
// no_compress is null (--compress / COMPRESS=1).  Aborts on non-vector VEC
// (chunk arrays cannot be written directly, they need to be materialised
// first — not implemented).
// -----------------------------------------------------------------------------
// Compacted run-length dump of a value sequence. Runs longer than 2 elements
// are rendered as "v ...xN... v" (N = run length — keeping the count visible
// is deliberate: a 64-long run and a 512-long run must not look alike when
// hunting geometry bugs); short runs are printed element by element. `sink`
// receives std::string fragments (log_message, or a stderr wrapper) and
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

template <typename VEC>
void write_binary_file(const std::string& filename,
                       const VEC&         data,
                       std::uint64_t      checksum,
                       bool               special_writing,
                       const char*        no_compress)
{
    using CV = std::remove_reference_t<VEC>;
    using V  = std::remove_const_t<CV>;

    const std::string filename_gz        = filename + ".bz2";
    const std::string filename_check_sum = filename + "_check_sum";

    if constexpr (is_vector<V>())
    {
        using T = VEC::value_type;
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            std::println(stderr, "Error open file : {}", filename);
            return;
        }

        std::ofstream file_check_sum(filename_check_sum, std::ios::binary);
        if (!file_check_sum)
        {
            std::println(stderr, "Error open file : {}", filename_check_sum);
            return;
        }
        const size_t size = data.size();

#pragma omp critical
        {
            std::println("Write : {}", filename_check_sum);
            std::println("Write : {}", filename);
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
        std::println("write is not avalaible on chunk vector");
        std::abort();
    }

    if (no_compress)
        return;

    if constexpr (build_config::has_bzip2)
    {
        compress_file(filename, filename_gz);
#pragma omp critical
        {
            std::println("Compression : {} -> {}", filename, filename_gz);
        }
        unlink(filename.c_str());
    }
}

// -----------------------------------------------------------------------------
// read_check_binary_file<VEC> — read back a bench dump written by
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
template <typename VEC>
void read_check_binary_file(Int                ind,
                            const std::string& tab_name,
                            const std::string& filename,
                            const VEC&         data_check,
                            std::uint64_t      lchs,
                            const char*        no_compress,
                            bool               read_trace,
                            bool               write_trace)
{
    using T = VEC::value_type;

    const std::string filename_gz        = filename + ".bz2";
    const std::string filename_check_sum = filename + "_check_sum";

    uint64_t lchs_read = 0;
#pragma omp critical
    {
        std::println("Read : {}", filename_check_sum);
    }
    std::ifstream file_check_sum(filename_check_sum, std::ios::binary);
    if (!file_check_sum)
    {
        std::println(stderr, "Error open file  : {}", filename_check_sum);
        std::abort();
    }

    file_check_sum.read(reinterpret_cast<char*>(&lchs_read), sizeof(uint64_t));

    if (lchs_read == lchs && !read_trace)
    {
#pragma omp critical
        {
            std::println("{} Checksum OK !", tab_name);
        }
        return;
    }
    else
    {
        if (!read_trace)
            std::print(stderr, "{}\nError checksum ! checksum read : {} checksum calculate : {}", tab_name, lchs_read,
                       lchs);

        if (!no_compress)
        {
            std::print(stderr, " --> we decompress and read the file : {}", tab_name);
        }
        std::println(stderr, "");
    }

    const Int sz = data_check.size();

    // Whether `filename` (.bin) is a decompression TEMPORARY we created here
    // (compressed regime) or the oracle's persistent copy (a baseline without
    // --compress, or without bzip2, keeps the plain .bin and never writes a .bz2). Only a
    // temporary may be unlinked after reading — deleting the persistent copy
    // would destroy the oracle's only instance for every later run.
    bool bin_is_temp = false;

    if constexpr (build_config::has_bzip2)
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
        std::println("Read : {}", filename);
        std::fflush(stdout);
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::println(stderr, "Error open file  : {}", filename);
        return;
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

    // No directive: the whole lambda lives inside template <typename VEC>
    // read_check_binary_file, so every data_check.* call below is DEPENDENT and
    // the `if constexpr` inside it is genuinely not instantiated for a
    // std::vector VEC. It is unused there -- io_res == 2 is unreachable without
    // a shrink build -- hence [[maybe_unused]]. Capture-default rather than a
    // list for the same reason: format_val is only named in that discarded
    // branch, and an explicit capture that the std::vector instantiation
    // never reads draws -Wunused-lambda-capture from clang.
    [[maybe_unused]]
    const auto lbd_check_trace = [&](std::ifstream& trace_file, Int k, auto& value)
    {
        // The body as a whole, not statement by statement: everything below
        // reads data_check, whose type is the DEPENDENT template parameter, so
        // for a std::vector VEC none of it is instantiated.
        if constexpr (chunk_config::is_shrink_enabled)
        {
            std::string line;
            std::string current_tab;

            bool ok = false;

            std::getline(trace_file, line);

            std::getline(trace_file, line);
            // Detect tab_name
            if (line.contains(" not equal !"))
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

            const Int num_chunk = k >> data_check.chunk_decalage();
            const Int num       = k & data_check.chunk_modulo();

            bool                  is_values   = false;
            bool                  is_cst_val1 = false;
            [[maybe_unused]] bool is_cst_val2 = false; // read only by the chunk2 / chunk2a traces
            while (std::getline(trace_file, line))
            {
                if (line.contains("is_values"))
                {
                    is_values = true;
                    break;
                }
                if (line.contains("is_cst_val1"))
                {
                    is_cst_val1 = true;
                    break;
                }
                // Pure text search on the trace line -- no variant API at all -- and the
                // enclosing lambda is generic, so the guard was never more than an
                // `if constexpr` written with the preprocessor.
                if constexpr (chunk_config::use_chunk2or2a)
                {
                    if (line.contains("is_cst_val2"))
                    {
                        is_cst_val2 = true;
                        break;
                    }
                }
            }

            const auto& chunk_t = data_check.getChunkList()[num_chunk];

            const bool is_val1 = chunk_t.is_cst_val1();

            // false everywhere but CHUNK2, exactly as the #else branch gave.
            bool is_val2 = false;
            if constexpr (chunk_config::use_chunk2)
            {
                static constexpr bool is_bool     = std::is_same_v<T, bool>;
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
                std::println(stderr, "value               : {}", format_val(value));
                std::println(stderr, "chunk number        : {}", num_chunk);
                std::println(stderr, "number inside chunk : {}", num);
                std::println(stderr, "is values           : {:d}", is_values);
                std::println(stderr, "is_cst_val1         : {:d}", is_val1);
                if constexpr (chunk_config::use_chunk2or2a)
                {
                    std::println(stderr, "is_cst_val2         : {:d}", is_val2);
                }

                if constexpr (chunk_config::ring_log_debug)
                    shrink_array::analyze_ring_sweep();
            };

            // Detect value category
            if (is_cst_val1)
            {
                if (is_val1)
                {
                    std::println(stderr, "\nwas cst_val1 it is");
                    lbd_print();
                    exit(0);
                }

                std::println(stderr, "\nwas cst_val1 is not!");
                lbd_print();
                exit(0);
            }

            if constexpr (chunk_config::use_chunk2or2a)
            {
                if (is_cst_val2)
                {
                    if (is_val2)
                    {
                        std::println(stderr, "\nwas cst_val2 it is");
                        lbd_print();
                        exit(0);
                    }

                    std::println(stderr, "\nwas cst_val2 is not!");
                    lbd_print();
                    exit(0);
                }
            }
            if (is_values)
            {
                if (!is_val1 && !is_val2)
                {
                    std::println(stderr, "\nwas values it is");
                    lbd_print();
                    constexpr T v{}; // T{}: the struct elements have no conversion from 0
                    bool        te = true;
                    for (UInt p = 0; p < data_check.chunk_size(); ++p)
                    {
                        if (chunk_t.values[p] != v)
                        {
                            te = false;
                            break;
                        }
                    }

                    if (te)
                    {
                        std::println(stderr, "should be cst_val1");
                    }
                    else
                    {
                        bool       all_equal = true;
                        const auto first     = chunk_t.values[0];

                        for (UInt p = 1; p < data_check.chunk_size(); ++p)
                        {
                            if (chunk_t.values[p] != first)
                            {
                                all_equal = false;
                                break;
                            }
                        }

                        if (all_equal)
                        {
                            std::println(stderr, "All values of chunk are identical: {}", format_val(first));
                        }
                        else
                        {
                            std::println(stderr, "All values of chunk (compacted):");
                            const T* vals = chunk_t.values; // atomic<T*> implicit load or plain ptr
                            log_values_compacted([](const std::string& m) { std::print(stderr, "{}", m); }, format_val,
                                                 vals, data_check.chunk_size());
                        }
                    }

                    exit(0);
                }

                std::println(stderr, "\nwas values is not!");
                lbd_print();
                exit(0);
            }
        }
    };

    std::ofstream trace_file_write;
    std::ifstream trace_file_read;

    // Open the write-trace file once, up front, so both the bool and
    // non-bool branches below share the same guarded write path.
    // Previously the bool branch never opened it, silently discarding
    // trace output for tab_bool errors under --write_trace.
    if (write_trace)
    {
        trace_file_write.open("trace_err.txt");
        if (!trace_file_write.is_open())
        {
            std::println(stderr, "Failed to open trace_err.txt");
        }
    }

    if (read_trace)
    {
        trace_file_read.open("trace_err.txt");
        if (!trace_file_read.is_open())
        {
            std::println(stderr, "Failed to open trace_err.txt");
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
                // Lambda to log to stderr and optionally to the trace file
                const auto log_message = [&](const std::string& msg)
                {
                    std::print(stderr, "{}", msg);
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

                std::println(stderr, "iteration: {}", ind);
                // One `if constexpr` over both former blocks: the CHUNK2 one below
                // reads chunk_t, declared here, so they cannot be split.
                if constexpr (chunk_config::is_shrink_enabled)
                {
                    const Int num_chunk = k >> data_check.chunk_decalage();

                    log_message("\nchunk number        : " + std::to_string(num_chunk) + "\n");
                    const Int num = k & data_check.chunk_modulo();
                    log_message("number inside chunk : " + std::to_string(num) + "\n");
                    const auto& chunk_t = data_check.getChunkList()[num_chunk];
                    using ValueType     = decltype(chunk_t.values); // infer the type

                    T* raw_ptr = nullptr;

                    if constexpr (is_atomic_pointer<ValueType>::value)
                    {
                        // It's a std::atomic<T*>
                        raw_ptr = chunk_t.values.load();
                    }
                    else
                    {
                        // It's a raw pointer
                        raw_ptr = chunk_t.values;
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

        constexpr bool is_complex = std::is_same_v<T, complex_double>;

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
                    std::print(stderr, "{}", msg);
                    if (write_trace && trace_file_write.is_open())
                    {
                        trace_file_write << msg;
                    }

                    if constexpr (chunk_config::ring_log_debug)
                        shrink_array::analyze_ring_sweep();
                };

                if constexpr (std::is_same_v<T, double>)
                    log_message(std::string("")); // Reset stream flags handled below if needed
                if constexpr (std::is_same_v<T, float>)
                    log_message(std::string("")); // Same as above

                // FIX: unsigned char goes through operator<< as a CHARACTER, so a
                // mismatch on tab_char printed a control byte instead of the number
                // -- unreadable exactly when the message matters most. Promote the
                // narrow types to a wide integer for display only; every other T
                // prints as before.
                const auto printable = [](const auto& v)
                {
                    using V = std::remove_cvref_t<decltype(v)>;
                    if constexpr (std::is_same_v<V, unsigned char> || std::is_same_v<V, signed char> ||
                                  std::is_same_v<V, char>)
                        return static_cast<int>(v);
                    else
                        return v;
                };

                std::ostringstream oss;
                oss << tab_name << "[" << k << "]"
                    << " not equal ! read : " << printable(value) << " != " << printable(data_check[k]);
                log_message(oss.str() + "\n");

                if constexpr (chunk_config::is_shrink_enabled)
                {
                    const Int   num_chunk = k >> data_check.chunk_decalage();
                    const auto& chunk_t   = data_check.getChunkList()[num_chunk];
                    log_message("chunk number        : " + std::to_string(num_chunk) + "\n");
                    const Int num = k & data_check.chunk_modulo();
                    log_message("number inside chunk : " + std::to_string(num) + "\n");

                    // Detect whether a type is an atomic of a pointer, regardless of the pointed-to type

                    T* raw_ptr = nullptr;

                    using ValueType = decltype(chunk_t.values);

                    if constexpr (is_atomic_pointer<ValueType>::value)
                    {
                        // It's a std::atomic<T*>
                        raw_ptr = chunk_t.values.load();
                    }
                    else
                    {
                        // It's a raw pointer
                        raw_ptr = chunk_t.values;
                    }

                    auto               addr_values_member = reinterpret_cast<std::uintptr_t>(&chunk_t.values);
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
                    const Int num_chunk = k >> data_check.chunk_decalage();

                    if (const auto& chunk_t = data_check.chunk_list[num_chunk]; chunk_t.is_cst_val1() && value != zero)
                    {
                        log_message("\nis_cst_val1 should be not !\n");
                        {
                            const T* vals = chunk_t.values;
                            log_values_compacted(log_message, format_val, vals, data_check.chunk_size());
                        }
                    }
                    // IIFE: the two spellings of is_cst_val2 cannot each be an
                    // `else if` condition. We are inside a function template, so the
                    // discarded branch is not instantiated.
                    else if (
                        [&]
                        {
                            if constexpr (chunk_config::use_chunk2)
                                return chunk_t.is_cst_val2(data_check.get_ind_second_value());
                            else
                                return chunk_t.is_cst_val2();
                        }())
                    {
                        log_message("\n[Debug] is_cst_val2 should be not !\n");

                        constexpr T zero{};
                        bool        test_zero = true;

                        for (UInt p = 0; p < data_check.chunk_size(); p++)
                        {
                            if (chunk_t.values[p] != zero)
                            {
                                test_zero = false;
                            }
                        }

                        if (!test_zero)
                        {
                            log_message("chunk values contain : ");
                            {
                                const T* vals = chunk_t.values;
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
                        const auto first_v   = chunk_t.values[0];

                        for (UInt i = 1; i < data_check.chunk_size(); ++i)
                        {
                            if (chunk_t.values[i] != first_v)
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
                                const T* vals = chunk_t.values;
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
            std::println(stderr, "{} : values are not perfectly equal ! : error max = {:e}", nn, ratio_max);
        }
        else if (!read_trace)
        {
            std::println("no data error (but checksum error...)");
        }

        file.close();
    }

    if (!no_compress && bin_is_temp)
    {
        unlink(filename.c_str());
    }
}
