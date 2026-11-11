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

// The shrink_array module: the whole library, read in the purview, exports
// what its headers mark SHRINK_ARRAY_EXPORT -- the arrays, ParamChunk,
// parallel_phase, the teardown, print / println, the chunk exchange, the
// chunk-size advisor and decider, the mdspan views and tile operations, the
// build options (build_config, chunk_config), the pmp loops and queries
// (chunk/threads.h's among them), the chunk-size settings -- and hides the
// rest (chunk/internal/module_export.h).
// The headers stay the source of truth: the library's own sources and the
// white-box validation harnesses still #include them.
//
// Built per variant library (chunk_<variant>_module links chunk_<variant>; and
// chunk_vector_module, the std::vector mode, links chunk_vector): the BMI
// carries that library's defines (CHUNK_VARIANT_*, CHUNK_SIZE), so an importer
// must link the same library. Macros (CHUNK_LOG, CHUNK_PMP_*) do not
// cross an import: a program that needs them still includes chunk/log.h.

module;

// The library's external headers -- standard, system, OpenMP, third party --
// and only those, under the conditions of their #includes in the library's
// headers (scripts/check_module_deps.py compares the two). The library itself
// is read below, in the purview: its #includes of these then read nothing.
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <compare>
#include <complex>
#include <concepts>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ostream>
#include <print>
#include <ranges>
#include <ratio>
#include <shared_mutex>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <version>

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef CHUNK_HAVE_STACKTRACE
    #include <stacktrace>
#else
    #include <cxxabi.h>
    #include <execinfo.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
#if __has_include(<mdspan>)
    #include <mdspan>
#endif
#if !(defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L) && __has_include(<mdspan/mdspan.hpp>)
    #include <mdspan/mdspan.hpp>
#endif
#ifdef __has_feature
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h>
    #endif
#elifdef __SANITIZE_ADDRESS__
    #include <sanitizer/lsan_interface.h>
#endif

#include "chunk/vendor/flat_hash_map.hpp"
#include "chunk/vendor/ankerl/unordered_dense.h"

export module shrink_array;

// The standard library, as a module. The importer's instantiations look up
// some std names from here ([module.context]): std::vector's relocation check
// (the placement operator new), type_name.h's typeid (std::type_info). So GCC
// 16 needs no `export using ::operator new;` and std::type_info exports, and
// an importer need not import std itself. The headers above still include
// their standard headers: both compilers merge them with the module.
import std;

// The whole library, in the purview: what is marked SHRINK_ARRAY_EXPORT is
// exported, the rest stays reachable but cannot be named by an importer.
// extern "C++" attaches it to the global module, as in the compiled library and
// in any program that #includes the headers: the same entities, one set of
// pools. The array of this variant library: the module is built once per
// variant (chunk_<variant>_module), with the library's CHUNK_VARIANT_* define.
#define SHRINK_ARRAY_EXPORT export
extern "C++"
{
#if defined(CHUNK_VARIANT_ZERO_ANY)
    #include "chunk/zero_any_array.h" // zero_any::Array
    #include "chunk/pattern_array.h"  // pattern::Array, over zero_any's library
#elif defined(CHUNK_VARIANT_ZERO_ONE_VALUE)
    #include "chunk/zero_one_value_array.h" // zero_one_value::Array
    #include "chunk/array_chunk_bits.h"     // ArrayChunkBits: zero_one_value::Array<Bit>
#else
    // CHUNK_VARIANT_ZERO, or no variant: chunk_vector, the std::vector mode
    // (chunk_config::use_zero false), whose programs still name zero::Array.
    #include "chunk/zero_array.h" // zero::Array
#endif
#include "chunk/param_chunk.h"                // ParamChunk
#include "chunk/config.h"                     // build_config, chunk_config: the options the library was built with
#include "chunk/internal/PMP_CHUNK.h"         // pmp: the loops, the teams, max_threads, team_size, thread_rank
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/print.h"                      // print, println
#include "chunk/threads.h"                    // pmp::max_threads, team_size, thread_rank
#include "chunk/manage_ompt.h"                // parallel_phase
#include "chunk/internal/pool_release.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/chunk_exchange.h"             // export_chunk, import_chunk, export_array, import_array, ChunkState, ...
#include "chunk/chunk_size_advisor.h"         // ChunkSizeAdvisor, ChunkSizeReport, ChunkSizeEstimate
#include "chunk/rechunk_decider.h"            // RechunkDecider, RechunkPolicy
#include "chunk/mdspan.h"                     // md::view, md::tiled_view, md::layout_tiled, ...
#include "chunk/mdspan_tile_ops.h"            // md::for_each_tile, md::fill_box, md::copy_box_in, ...
}
#undef SHRINK_ARRAY_EXPORT

// GCC 16.2 emits an inline variable of the purview neither in the module's
// object nor in an importer whose instantiations read it into a local
// (`unsigned every = force_defer_every;` in test_hooks::forced_deferral()): the
// importer's link then fails on an undefined reference. A non-inline
// odr-use here emits it in the module's object. clang 23 does not need it.
[[gnu::used]] static const void* const gcc16_emit_inline_variables[] = {
    &shrink_array::test_hooks::force_defer_every,
};

// Names exported by using-declarations: their headers do not mark them.
export namespace shrink_array
{
// The phase hooks parallel_phase() calls, for a program that opens its regions
// itself (chunk/internal/phase_hooks.h).
using shrink_array::begin_parallel_phase;
using shrink_array::finalize_barrier_phase;
using shrink_array::finalize_parallel_phase;
using shrink_array::finalize_parallel_phase_after_flush;
using shrink_array::finalize_parallel_phase_only_flush;

// The variant's array: a using-declaration, since chunk_traits.h declares it
// first, unexported.
#if defined(CHUNK_VARIANT_ZERO_ANY)
namespace zero_any
{
using shrink_array::zero_any::Array;
using shrink_array::zero_any::Chunk; // sizeof(Chunk<T>): the chunk metadata, for the advisor's options
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
#else
namespace zero
{
using shrink_array::zero::Array;
} // namespace zero
#endif
} // namespace shrink_array
