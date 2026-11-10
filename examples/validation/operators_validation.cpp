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
 * @file operators_validation.cpp
 * @brief operators_validation: every operator of an element, a[i], on every
 *        element type, against plain T lvalues -- run(), directly then with
 *        the deferred path forced. The cases: operators_validation_cases.h.
 */

#include <string> // std::string

#include "chunk/config.h"              // build_config::use_openmp
#include "chunk/param_chunk.h"         // ParamChunk::init_indexes
#include "chunk/print.h"               // shrink_array::println
#include "chunk/internal/test_hooks.h" // force_defer_every, force_defer_compiled

#include "operators_validation_cases.h" // Arr, n_chunks, only_type, run_small_integers, ...
#include "validation_harness.h"         // run() is called by validation_driver.cpp

namespace validation::operators_validation
{

namespace
{

/// @brief Every element type the variant takes, in one mode.
long run_types(long count, long chunk_sz, const char* mode, long& cases)
{
    long fails = 0;
    fails += run_small_integers(count, chunk_sz, mode, cases);
    fails += run_wide_integers(count, chunk_sz, mode, cases);
    fails += run_floating(count, chunk_sz, mode, cases);
    fails += run_bool_complex(count, chunk_sz, mode, cases);
    return fails;
}

} // namespace

int run(int argc, char** argv)
{
    if (argc > 2 && std::string(argv[1]) == "--type")
    {
        only_type = argv[2];
    }
    ParamChunk::init_indexes(true);
    const long chunk_sz = static_cast<long>(Arr<int>(1).chunk_size());
    const long count    = n_chunks * chunk_sz;
    println("=== operators_validation [variant = {}]  N={} ({} chunks of {}) ===", variant_name(), count, n_chunks,
            chunk_sz);

    long cases = 0;
    long fails = run_types(count, chunk_sz, "direct", cases);

    // One write in three down the deferred path: the replays must leave the
    // same values and return the same results.
    if constexpr (build_config::use_openmp && test_hooks::force_defer_compiled)
    {
        const unsigned int every      = test_hooks::force_defer_every;
        test_hooks::force_defer_every = 3U;
        fails += run_types(count, chunk_sz, "deferred", cases);
        test_hooks::force_defer_every = every;
    }
    else
    {
        println("[{}] note: no deferred path to force in this build", variant_name());
    }

    println("operators_validation [{}]: {} cases, {}", variant_name(), cases, fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}

} // namespace validation::operators_validation
