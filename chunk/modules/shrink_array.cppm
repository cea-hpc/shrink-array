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

// The shrink_array module (prototype): the existing headers, included in the
// global module fragment, and the names a program of docs/getting_started.md
// uses, exported by using-declarations. Nothing is moved into the module
// purview: the headers stay the source of truth and `#include` keeps working.
//
// Built per variant library (chunk_zero_any_module links chunk_zero_any): the
// BMI carries that library's defines (CHUNK_VARIANT_ZERO_ANY, CHUNK_SIZE), so
// an importer must link the same variant. Macros (CHUNK_LOG, CHUNK_PMP_*) do not
// cross an import: a program that needs them still includes chunk/log.h.

module;

#include "chunk/zero_any_array.h" // zero_any::Array
#include "chunk/zero_chunk.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/param_chunk.h"    // ParamChunk
#include "chunk/internal/print.h" // print, println

#include <cstddef>  // std::size_t
#include <new>      // the placement operator new
#include <typeinfo> // std::type_info

export module shrink_array;

// For GCC 16: the templates the importer instantiates -- std::vector's
// relocation check (placement new), type_name.h's typeid -- find these two
// only where the importer can see them, not in the global module fragment
// (clang 23 looks there too, as [module.context] has it). Without the exports
// GCC stops in the importer ("must #include <typeinfo>", "no matching function
// for call to operator new(sizetype, void*)"); an importer that includes <new>
// and <typeinfo> itself does not need them.
export using ::operator new;
export namespace std
{
using std::type_info;
} // namespace std

export namespace shrink_array
{
using shrink_array::destroy_block_chunk;
using shrink_array::parallel_phase;
using shrink_array::ParamChunk;
using shrink_array::print;
using shrink_array::println;
using shrink_array::release_chunk_pools;

namespace zero_any
{
using shrink_array::zero_any::Array;
} // namespace zero_any
} // namespace shrink_array
