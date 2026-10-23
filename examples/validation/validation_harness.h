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

// ============================================================================
// What a grouped validation harness needs from the shared driver
// (validation_driver.cpp).
// ============================================================================
//
// Most harnesses of this directory are compiled together, one executable per
// variant and geometry (validation_<variant>_<geometry>, a unity build: the
// library's templates are instantiated once for all of them instead of once
// per harness -- four times less compilation). Each harness lives in its own
// namespace, validation::<harness>, and its entry point is
// validation::<harness>::run(argc, argv), called by the driver's main():
//
//     validation_zero_any_256 compress_roundtrip [args...]
//
// run() sees the same argv a standalone main() did: argv[0] the executable,
// then the arguments after the harness name.
// ============================================================================

#pragma once

#include <string> // std::string (self_command)

namespace validation
{

/// @brief The shell words that run this harness again: the executable, quoted,
///        then the harness name. For the harnesses that check a failure in a
///        child process (thread_growth, nested_region, stale_proxy,
///        cst2_registry).
/// @param argv0 the run()'s argv[0]
/// @return e.g. `"/path/validation_zero_any_256" stale_proxy_validation`
std::string self_command(const char* argv0);

} // namespace validation
