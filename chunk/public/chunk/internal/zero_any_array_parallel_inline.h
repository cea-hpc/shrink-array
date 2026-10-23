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

// Out-of-line bodies of zero_any::Array's OpenMP-only surface (ParallelSupport2A).
// Included at the end of zero_any_array.h, after the class and its own inline
// file, so the CRTP self() down-cast sees a complete Derived. The one body left,
// flush_pending_deferred_ops(), is the shared one.
#include "chunk/internal/zero_any_array_parallel.h"             // ParallelSupport2A, the declarations completed here
#include "chunk/internal/array_chunk_parallel_common_inline.h" // flush_pending_deferred_ops
