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

// Template method bodies now live in chunk/array_chunk_inline.h, included
// transitively via chunk/array_chunk.h.
//
// STEP 1 of the "remove static type list" refactor: bodies moved to the inline
// header (consumer TUs can auto-instantiate ArrayChunk<T>); the explicit
// instantiations have been removed (Stage 5b): each ArrayChunk<T> the
// consumer uses auto-instantiates at its use site (weak-symbol dedup).

#include "chunk/array_chunk.h" // its own header: ArrayChunk<T>, bodies in array_chunk_inline.h
