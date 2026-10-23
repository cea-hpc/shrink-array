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

// The template bodies are in array_chunk2_inline.h, through array_chunk2.h. No
// explicit instantiations: each zero_one_value::Array<T> is instantiated where it is used.

#include "chunk/array_chunk2.h" // its own header: zero_one_value::Array<T>, bodies in array_chunk2_inline.h
