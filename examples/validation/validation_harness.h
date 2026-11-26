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
//     validation_zero_any_256 compress_roundtrip_validation [args...]
//
// run() sees the same argv a standalone main() did: argv[0] the executable,
// then the arguments after the harness name.
// ============================================================================

#pragma once

#include <string> // std::string (run_self, temp_path)
#include <vector> // std::vector (run_self's arguments)

#include "validation_process.h" // ProcessOptions, ProcessResult (run_self)

namespace validation
{

/// @brief Run this harness again as a child process and wait for it: the
///        executable, the harness name, then @p args -- no shell, each element
///        one argument -- with CHUNK_VALIDATION_CHILD=1 (main() makes the child
///        non-dumpable: most of them abort on purpose) besides @p opt's
///        variables. For the harnesses that check a failure in a child process
///        (thread_growth, nested_region, stale_proxy, second_value_registry...).
/// @param argv0 the run()'s argv[0]
/// @param args  the child's own arguments, e.g. {"--case", "2"}
/// @param opt   ProcessOptions; its output file, when empty, is a fresh
///              temp_path(): the child's stdout and stderr land in
///              ProcessResult::output either way
/// @return how the child ended, and what it printed
ProcessResult run_self(const char* argv0, const std::vector<std::string>& args, ProcessOptions opt = {});

/// @brief A path in this process's own temporary directory, for the logs of
///        the children a harness runs: the directory is made once, by
///        mkdtemp() under $TMPDIR (else /tmp), readable by this user only, and
///        removed with its content when the process exits. Unpredictable and
///        private, where a fixed /tmp/<name>_<pid> could be taken first by
///        another user of a shared node, or collide with another run.
/// @param name the file name, unique within this process
/// @return e.g. `/tmp/shrink_validation.Ab3xYz/phase_protocol_2.log`
std::string temp_path(const std::string& name);

} // namespace validation
