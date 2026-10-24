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

#pragma once

/**
 * @file contracts.h
 * @brief C++26 contracts (P2900) where the compiler has them: CHUNK_PRE and
 *        CHUNK_CONTRACT_ASSERT.
 *
 * @details
 *   - CHUNK_PRE(cond): a precondition, `pre(cond)`, written after the
 *     declarator (after noexcept and a trailing return type). Without
 *     contracts it expands to nothing.
 *   - CHUNK_CONTRACT_ASSERT(cond): `contract_assert(cond);`. Without
 *     contracts, the check of a check build (build_config::any_check) with
 *     CHUNK_INVARIANT_FAILED, nothing otherwise.
 *
 *   The evaluation semantic is the compiler's, per translation unit: CMake's
 *   CHUNK_CONTRACTS passes -fcontract-evaluation-semantic= PUBLIC on the
 *   libraries (AUTO: enforce in a check build, ignore otherwise). A
 *   precondition of an inline function (operator[]) is then checked with the
 *   semantic of the program's translation unit, not the library's.
 *
 *   A violation goes to the program's ::handle_contract_violation, or to the
 *   default one: GCC 16's is in libstdc++exp (a weak symbol; the chunk
 *   libraries link that archive already, for std::stacktrace), one line with
 *   the function, file:line and predicate, then std::terminate under enforce.
 *   The library defines none: the handler is the program's to replace.
 *
 *   GCC 16.2 stops with an internal compiler error (check_noexcept_r) on a
 *   precondition that reads a parameter of a function template whose return
 *   type is deduced (`auto`, `decltype(auto)`): spell the return type there,
 *   or use CHUNK_CONTRACT_ASSERT in the body.
 *
 *   The cost of operator[]'s precondition (GCC 16.2, -O3, a serial loop over
 *   16 M elements of zero_any::Array<double>): none under ignore; enforce
 *   +35 % on a read, +4 % on a write; quick_enforce 5 times the read (GCC calls
 *   a terminate wrapper there, it does not trap): prefer enforce.
 *
 *   Only GCC 16 has them today (__cpp_contracts, on with -std=c++26); Clang 23
 *   has none.
 */

#if defined(__cpp_contracts)
    #define SHRINK_ARRAY_HAS_CONTRACTS 1
    #include <contracts> // std::contracts::contract_violation (the handlers)

    #define CHUNK_PRE(...) pre(__VA_ARGS__)
    #define CHUNK_CONTRACT_ASSERT(...) contract_assert(__VA_ARGS__)
#else
    #define SHRINK_ARRAY_HAS_CONTRACTS 0
    #include "chunk/config.h"                   // build_config::any_check
    #include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

    #define CHUNK_PRE(...)
    #define CHUNK_CONTRACT_ASSERT(...)                                             \
        do                                                                         \
        {                                                                          \
            if constexpr (::shrink_array::build_config::any_check)                 \
            {                                                                      \
                if (!(__VA_ARGS__)) [[unlikely]]                                   \
                {                                                                  \
                    CHUNK_INVARIANT_FAILED("contract violated: {}", #__VA_ARGS__); \
                }                                                                  \
            }                                                                      \
        } while (false)
#endif
