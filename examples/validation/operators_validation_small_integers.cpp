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
 * @file operators_validation_small_integers.cpp
 * @brief operators_validation, one translation unit of
 *        operators_validation_cases.h: run_type() on char, short and int.
 */

#include "operators_validation_cases.h" // run_type

namespace validation::operators_validation
{

long run_small_integers(long count, long chunk_sz, const char* mode, long& cases)
{
    long fails = 0;
    fails += run_type<char>(count, chunk_sz, mode, cases);
    fails += run_type<short>(count, chunk_sz, mode, cases);
    fails += run_type<int>(count, chunk_sz, mode, cases);
    return fails;
}

} // namespace validation::operators_validation
