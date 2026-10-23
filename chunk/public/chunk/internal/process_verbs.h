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

#include <cstddef> // std::size_t

/**
 * @file process_verbs.h
 * @brief The process-wide verbs an array reaches without the manager headers.
 *
 * Declared here, once, because two facades need them and cannot include each
 * other: zero_chunk.h (where the definitions' file, chunk_std.cpp, gets them)
 * and array_chunk_base.h (whose shrink_to_fit() calls the first one).
 */
namespace shrink_array
{
/**
 * @brief The process-wide half of an array's shrink_to_fit(): fold every
 *        compression candidate now, then give the reuse pools' parked buffers
 *        back (all of them: every array of the process shares those pools).
 * @return The bytes the pools gave back.
 * @note Sequential context. Defined in chunk_std.cpp.
 */
std::size_t compress_now_and_trim_pools();
} // namespace shrink_array
