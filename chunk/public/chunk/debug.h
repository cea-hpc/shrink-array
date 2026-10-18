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

#pragma once

/**
 * @file debug.h
 * @brief print_backtrace(), kept here for the code that includes it: it lives
 *        in fast_allocators/internal/fatal.h, with the fatal report that
 *        prints it (shared by both libraries).
 */

#include "fast_allocators/internal/fatal.h" // fast_alloc::logging::print_backtrace

namespace shrink_array
{
/// @brief Print the calling thread's backtrace (up to 100 frames) on stderr, demangling names when possible.
using fast_alloc::logging::print_backtrace;
} // namespace shrink_array
