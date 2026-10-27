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
 * @file shrink_to_fit_validation.cpp
 * @brief After shrink_to_fit(), no uniform chunk keeps a buffer, in every build.
 *
 * @details shrink_to_fit() folds every chunk that has become constant. A serial
 *          zero_any queues its compression candidates and folds them in batches
 *          of half its list, from the write path; shrink_to_fit() swept the last,
 *          partial batch only in the OpenMP builds, so a serial build left up to
 *          that many uniform chunks materialised -- right values, memory kept.
 *          Here a few uniform chunks (far below a batch), each its own constant
 *          in zero_any (the first one, or a second one per chunk; the first one
 *          in the other variants), then shrink_to_fit(): every chunk must be
 *          compressed and read back its value.
 */

#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/manage_ompt.h"          // parallel_phase
#include "chunk/config.h"               // chunk_config::use_zero*
#include "chunk/internal/print.h"       // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::shrink_to_fit_validation
{

using namespace shrink_array;

namespace
{

using Arr = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<double>,
    std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<double>, zero::Array<double>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "zero_one_value";
    }
    else
    {
        return "zero";
    }
}

/// The value of chunk @p chunk. zero_any: zero for the even ones (the first
/// constant), chunk + 1 for the odd ones (one second constant per chunk, the
/// queued kind). zero and zero_one_value: zero everywhere -- zero compresses
/// to that constant only, and zero_one_value to a second one only when the
/// array declares it.
double chunk_value(int chunk)
{
    if (!chunk_config::use_zero_any || chunk % 2 == 0)
    {
        return 0.0;
    }
    return static_cast<double>(chunk + 1);
}

/// @p n_chunks uniform chunks written, then shrink_to_fit(): the chunks
/// still holding a buffer, plus the elements that read back wrong.
int one_case(int n_chunks)
{
    Arr       arr(0);
    const int chunk_size = static_cast<int>(arr.chunk_size());
    const int count      = n_chunks * chunk_size;
    arr.resize(count);
    parallel_phase(
        [&]
        {
#pragma omp for
            for (int chunk = 0; chunk < n_chunks; ++chunk)
            {
                const double value = chunk_value(chunk);
                for (int offset = 0; offset < chunk_size; ++offset)
                {
                    arr[(chunk * chunk_size) + offset] = value;
                }
            }
        });
    arr.shrink_to_fit();

    const int   written = arr.get_nb_chunk_written();
    int         wrong   = 0;
    const auto& carr    = std::as_const(arr);
    for (int index = 0; index < count; ++index)
    {
        if (carr[index] != chunk_value(index / chunk_size))
        {
            ++wrong;
        }
    }
    println("[{}/shrink-to-fit] {:4} uniform chunks: {} still hold a buffer, {} wrong element(s)  {}", variant_name(),
            n_chunks, written, wrong, written == 0 && wrong == 0 ? "ok" : "FAIL");
    return written + wrong;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    int fails = 0;
    for (const int n_chunks : {1, 10, 100})
    {
        fails += one_case(n_chunks) != 0 ? 1 : 0;
    }
    println("{}", fails ? "shrink_to_fit_validation: FAILED" : "shrink_to_fit_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::shrink_to_fit_validation
