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

// The settings a program must share with its variant library (CHUNK_VARIANT_ZERO /
// _V2 / _V2A, CHUNK_SIZE, CHUNK_DYNAMIC), spelled into one function name. The
// library defines its own (chunk/private/link_check.cpp); the first array of
// each element type calls the program's (AllTypeChunk::ensure_manager). A
// mismatch, which would otherwise compile the templates differently in silence,
// fails to link, and the undefined symbol names the program's settings:
//
//     undefined reference to `shrink_array::link_check::variant_v1_0_v2_0_v2a_1_size_256_dyn_0()'
//
// The options shared by every library of a build are not here: they are in the
// generated chunk/build_options.h, which the program compiles with.

#include "chunk/internal/chunk_size_config.h" // CHUNK_SIZE (its default, when the build gives none)

// The 0/1 flags are pasted (##) into the symbol name below, so they must stay macros.
// NOLINTBEGIN(modernize-macro-to-enum)
#ifdef CHUNK_VARIANT_ZERO
    #define CHUNK_LINK_CHECK_V1 1
#else
    #define CHUNK_LINK_CHECK_V1 0
#endif
#ifdef CHUNK_VARIANT_ZERO_ONE_VALUE
    #define CHUNK_LINK_CHECK_V2 1
#else
    #define CHUNK_LINK_CHECK_V2 0
#endif
#ifdef CHUNK_VARIANT_ZERO_ANY
    #define CHUNK_LINK_CHECK_V2A 1
#else
    #define CHUNK_LINK_CHECK_V2A 0
#endif
#ifdef CHUNK_DYNAMIC
    #define CHUNK_LINK_CHECK_DYN 1
#else
    #define CHUNK_LINK_CHECK_DYN 0
#endif
// NOLINTEND(modernize-macro-to-enum)

#define CHUNK_LINK_CHECK_NAME_(v1, v2, v2a, size, dyn) variant_v1_##v1##_v2_##v2##_v2a_##v2a##_size_##size##_dyn_##dyn
#define CHUNK_LINK_CHECK_NAME(v1, v2, v2a, size, dyn) CHUNK_LINK_CHECK_NAME_(v1, v2, v2a, size, dyn)

/// The function of this TU's settings (see above).
#define CHUNK_LINK_CHECK                                                                              \
    CHUNK_LINK_CHECK_NAME(CHUNK_LINK_CHECK_V1, CHUNK_LINK_CHECK_V2, CHUNK_LINK_CHECK_V2A, CHUNK_SIZE, \
                          CHUNK_LINK_CHECK_DYN)

namespace shrink_array::link_check
{
/// @brief Does nothing; exists (or not) for the linker. See the top of this file.
void CHUNK_LINK_CHECK() noexcept;
} // namespace shrink_array::link_check
