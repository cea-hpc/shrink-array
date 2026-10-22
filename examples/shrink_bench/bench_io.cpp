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

#include "bench_io.h" // the declarations defined here

// BZip2 helpers. Only present when BZIP2 is defined at build time.
// have_bzip2 (bench_config.h) mirrors this compile flag, so write_binary_file()
// calls these symbols only when they are linked.

#ifdef BZIP2

    #include <bzlib.h>  // BZ2_bzBuffToBuffCompress, BZ2_bzBuffToBuffDecompress
    #include <cstdlib>  // std::abort
    #include <fstream>  // std::ifstream, std::ofstream
    #include <iterator> // std::istreambuf_iterator
    #include <string>   // std::string
    #include <vector>   // std::vector
    #include <cstdio>   // stdout, stderr, std::fflush
    #include <print>    // std::println

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

    auto dest_len = static_cast<unsigned int>((source_len * 1.01) + 600); // Estimate of the compressed size
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

    unsigned int original_len = 0;
    in_file.read(reinterpret_cast<char*>(&original_len), sizeof(original_len)); // Read the original size
    if (!in_file)
    {
        std::println(stderr, "Error reading compressed data size.");
        std::abort();
    }

    std::vector        source_buffer((std::istreambuf_iterator(in_file)), std::istreambuf_iterator<char>());
    const unsigned int source_len = source_buffer.size();
    unsigned int       dest_len   = original_len; // Initialize the size of the output buffer

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
