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

// CHUNK_STATS comes from the CMake option of that name, never from here: its
// counters cost thread performance.

#if !defined CHUNK_STD_ALLOC
    #define FAST_ALLOC
#endif

#if defined HAVE_OMPT
    #define HAVE_EMBEDDED_COMPRESSION_WITH_OMPT
#endif
