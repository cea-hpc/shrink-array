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

// A header that includes one of the purview's for convenience skips it here
// (chunk/internal/module_export.h).
#define SHRINK_ARRAY_MODULE_UNIT

// The array of this variant library: the module is built once per variant
// (chunk_<variant>_module), with the library's CHUNK_VARIANT_* define.
#if defined(CHUNK_VARIANT_ZERO_ANY)
    #include "chunk/zero_any_array.h" // zero_any::Array
    #include "chunk/pattern_array.h"  // pattern::Array, over zero_any's library
#elif defined(CHUNK_VARIANT_ZERO_ONE_VALUE)
    #include "chunk/zero_one_value_array.h" // zero_one_value::Array
#elif defined(CHUNK_VARIANT_ZERO)
    #include "chunk/zero_array.h" // zero::Array
#else
    #error "shrink_array.cppm: built with a variant library's CHUNK_VARIANT_* define"
#endif

// The #includes of the headers read in the purview below, and only those.
#define SHRINK_ARRAY_MODULE_DEPS
#include "chunk/param_chunk.h"
#include "chunk/internal/print.h"
#include "chunk/manage_ompt.h"
#include "chunk/shutdown.h"
#include "chunk/chunk_exchange.h"
#include "chunk/chunk_size_advisor.h"
#include "chunk/rechunk_decider.h"
#include "chunk/mdspan.h"
#include "chunk/mdspan_tile_ops.h"
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
#include "chunk/param_chunk.h"        // ParamChunk
#include "chunk/internal/print.h"     // print, println
#include "chunk/manage_ompt.h"        // parallel_phase
#include "chunk/shutdown.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/chunk_exchange.h"     // export_chunk, import_chunk, export_array, import_array, ChunkState, ...
#include "chunk/chunk_size_advisor.h" // ChunkSizeAdvisor, ChunkSizeReport, ChunkSizeEstimate
#include "chunk/rechunk_decider.h"    // RechunkDecider, RechunkPolicy
#include "chunk/mdspan.h"             // md::view, md::tiled_view, md::layout_tiled, ...
#include "chunk/mdspan_tile_ops.h"    // md::for_each_tile, md::fill_box, md::copy_box_in, ...
}
#undef SHRINK_ARRAY_EXPORT

// The other headers' names, exported by using-declarations.
export namespace shrink_array
{
// The phase hooks parallel_phase() calls, for a program that opens its regions
// itself (chunk/internal/phase_hooks.h).
using shrink_array::begin_parallel_phase;
using shrink_array::finalize_barrier_phase;
using shrink_array::finalize_parallel_phase;
using shrink_array::finalize_parallel_phase_after_flush;
using shrink_array::finalize_parallel_phase_only_flush;

// The variant's array: a using-declaration, since chunk_traits.h (in the global
// module fragment) declares it first, unexported.
#if defined(CHUNK_VARIANT_ZERO_ANY)
namespace zero_any
{
using shrink_array::zero_any::Array;
} // namespace zero_any
namespace pattern
{
using shrink_array::pattern::Array;
} // namespace pattern
#elif defined(CHUNK_VARIANT_ZERO_ONE_VALUE)
namespace zero_one_value
{
using shrink_array::zero_one_value::Array;
} // namespace zero_one_value
#elif defined(CHUNK_VARIANT_ZERO)
namespace zero
{
using shrink_array::zero::Array;
} // namespace zero
#endif
} // namespace shrink_array
