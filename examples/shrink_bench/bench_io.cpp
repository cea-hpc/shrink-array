/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include "bench_io.h"

// BZip2 helpers. Only present when BZIP2 is defined at build time.
// build_config::has_bzip2 (in chunk/config.h) must be consistent with
// this compile flag — otherwise write_binary_file() would call these
// symbols without them being linked.

#ifdef BZIP2

    #include <bzlib.h>
    #include <cstdlib> // std::abort
    #include <fstream>
    #include <cstdio>
    #include <print>
    #include <iterator>
    #include <string>
    #include <vector>

void compress_file(const std::string& source_file, const std::string& dest_file)
{
    std::ifstream in_file(source_file, std::ios_base::binary);
    if (!in_file)
    {
        std::println(stderr, "Error opening source file for compression");
        return;
    }

    std::ofstream out_file(dest_file, std::ios_base::binary);
    if (!out_file)
    {
        std::println(stderr, "Error opening destination file for compression");
        return;
    }

    std::vector        buffer((std::istreambuf_iterator(in_file)), std::istreambuf_iterator<char>());
    const unsigned int source_len = buffer.size();

    auto              dest_len = static_cast<unsigned int>(source_len * 1.01 + 600); // Estimate of the compressed size
    std::vector<char> dest_buffer(dest_len);

    const Int result = BZ2_bzBuffToBuffCompress(dest_buffer.data(), &dest_len, buffer.data(), source_len, 9, 0, 30);

    if (result != BZ_OK)
    {
        std::println(stderr, "Error during compression : {}", result);
        return;
    }

    out_file.write(reinterpret_cast<char*>(&dest_len), sizeof(dest_len));
    out_file.write(dest_buffer.data(), dest_len);
}

void decompress_file(const std::string& source_file, const std::string& dest_file)
{
    std::ifstream in_file(source_file, std::ios_base::binary);
    if (!in_file)
    {
        std::println(stderr, "Error when opening source file for decompression : {}", source_file);
        std::abort();
    }

    std::ofstream out_file(dest_file, std::ios_base::binary);
    if (!out_file)
    {
        std::println(stderr, "Error opening destination file for decompression: {}", dest_file);
        std::abort();
    }

    unsigned int originalLen = 0;
    in_file.read(reinterpret_cast<char*>(&originalLen), sizeof(originalLen)); // Read the original size
    if (!in_file)
    {
        std::println(stderr, "Error reading compressed data size.");
        std::abort();
    }

    std::vector        source_buffer((std::istreambuf_iterator(in_file)), std::istreambuf_iterator<char>());
    const unsigned int source_len = source_buffer.size();
    unsigned int       dest_len   = originalLen; // Initialize the size of the output buffer

    std::vector<char> dest_buffer(dest_len);

    Int result = BZ2_bzBuffToBuffDecompress(dest_buffer.data(), &dest_len, source_buffer.data(), source_len, 0, 0);

    while (result == BZ_OUTBUFF_FULL)
    {
        dest_len *= 2; // Dynamically increase the size of the output buffer
        dest_buffer.resize(dest_len);
        result = BZ2_bzBuffToBuffDecompress(dest_buffer.data(), &dest_len, source_buffer.data(), source_len, 0, 0);
    }

    if (result != BZ_OK)
    {
        std::println(stderr, "Error during decompression : {}", result);
        switch (result)
        {
            case BZ_CONFIG_ERROR:
                std::println(stderr, "Configuration error.");
                break;
            case BZ_PARAM_ERROR:
                std::println(stderr, "Parameter error.");
                break;
            case BZ_MEM_ERROR:
                std::println(stderr, "Memory error.");
                break;
            case BZ_DATA_ERROR:
                std::println(stderr, "Data integrity error.");
                break;
            case BZ_DATA_ERROR_MAGIC:
                std::println(stderr, "Data error, magic number mismatch.");
                break;
            default:
                std::println(stderr, "Unknown error.");
                break;
        }
        std::abort();
    }

    out_file.write(dest_buffer.data(), dest_len);
    std::println("Successful decompression : {} -> {}", source_file, dest_file);
    std::fflush(stdout);
}

#endif // BZIP2
