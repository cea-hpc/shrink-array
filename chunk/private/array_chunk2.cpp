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

// Template method bodies now live in chunk/array_chunk2_inline.h, included
// transitively via chunk/array_chunk2.h.
//
// STEP 1 of the "remove static type list" refactor: the bodies have moved to
// the inline header (so any consumer TU can auto-instantiate ArrayChunk2<T>),
// but the explicit instantiations below are kept for now as a safety net so
// this build stays byte-for-byte compatible. Once the consumer side is
// confirmed to auto-instantiate every needed T, this list can be removed and
// the file reduced to just the include (see array_chunk2A.cpp / chunk27).

#include "chunk/array_chunk2.h"
