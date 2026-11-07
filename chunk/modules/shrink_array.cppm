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

// The shrink_array module (prototype): the names a program of
// docs/getting_started.md uses. Most headers are included in the global module
// fragment and their names exported by using-declarations; a few top headers
// are read in the purview and export their own declarations
// (chunk/internal/module_export.h). The headers stay the source of truth and
// `#include` keeps working.
//
// Built per variant library (chunk_zero_any_module links chunk_zero_any): the
// BMI carries that library's defines (CHUNK_VARIANT_ZERO_ANY, CHUNK_SIZE), so
// an importer must link the same variant. Macros (CHUNK_LOG, CHUNK_PMP_*) do not
// cross an import: a program that needs them still includes chunk/log.h.

module;

#include "chunk/zero_any_array.h" // zero_any::Array
#include "chunk/zero_chunk.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h"    // parallel_phase

// The #includes of the headers read in the purview below, and only those
// (chunk/internal/module_export.h).
#define SHRINK_ARRAY_MODULE_DEPS
#include "chunk/param_chunk.h"
#include "chunk/internal/print.h"
#undef SHRINK_ARRAY_MODULE_DEPS

export module shrink_array;

// The standard library, as a module. The importer's instantiations look up
// some std names from here ([module.context]): std::vector's relocation check
// (the placement operator new), type_name.h's typeid (std::type_info). So GCC
// 16 needs no `export using ::operator new;` and std::type_info exports, and
// an importer need not import std itself. The headers above still include
// their standard headers: both compilers merge them with the module.
import std;

// Headers whose declarations are the module's: those marked SHRINK_ARRAY_EXPORT
// are exported, the rest stays reachable but cannot be named by an importer.
// extern "C++" attaches them to the global module, as everywhere else.
#undef SHRINK_ARRAY_EXPORT
#define SHRINK_ARRAY_EXPORT export
extern "C++"
{
#include "chunk/param_chunk.h"    // ParamChunk
#include "chunk/internal/print.h" // print, println
}
#undef SHRINK_ARRAY_EXPORT

// The other headers' names, exported by using-declarations.
export namespace shrink_array
{
using shrink_array::destroy_block_chunk;
using shrink_array::parallel_phase;
using shrink_array::release_chunk_pools;

namespace zero_any
{
using shrink_array::zero_any::Array;
} // namespace zero_any
} // namespace shrink_array
