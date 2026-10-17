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
 * @file log.h
 * @brief The library's log, for a program: shrink_array::logging::set_sink()
 *        hands every message to the program's own logger,
 *        shrink_array::logging::configure() sets the levels from the code
 *        (CHUNK_LOG's syntax: "warn,rechunk=debug").
 *
 * The messages are filtered by the environment variable CHUNK_LOG (warn by
 * default) and go to stderr, or to CHUNK_LOG_FILE. The log itself is
 * fast_allocators/internal/log_core.h, shared by both libraries.
 */

#include "fast_allocators/internal/log_core.h" // fast_alloc::logging, CHUNK_LOG

namespace shrink_array
{
/// @brief Not `log`: with a `using namespace shrink_array;`, a namespace log
///        would make every call to the function log() ambiguous.
namespace logging = fast_alloc::logging;
} // namespace shrink_array
