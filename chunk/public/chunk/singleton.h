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

/// @file singleton.h
/// @brief The chunk library's name for fast_allocators' fast_alloc::Singleton: one
///        implementation for both.

#include "fast_allocators/internal/singleton.h" // Singleton

namespace shrink_array
{

/// @brief Lazily created, never deleted, process-wide instance of T, built with
///        `new T()` on first use. fast_alloc::Singleton under the chunk library's name.
/// @note An alias: a class befriending it, or declaring it in advance, names
///       fast_alloc::Singleton itself (see AllTypeChunk).
template <typename T>
using SingletonChk = ::fast_alloc::Singleton<T>;

} // namespace shrink_array
