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
//     validation_chunk2a_256 compress_roundtrip [args...]
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
/// @return e.g. `"/path/validation_chunk2a_256" stale_proxy_validation`
std::string self_command(const char* argv0);

} // namespace validation
