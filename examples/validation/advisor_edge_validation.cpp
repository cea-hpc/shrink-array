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

// ============================================================================
// Edge cases of the chunk-size advisor and of the re-chunk decider. Dyn builds.
// ============================================================================
//
//   ban        RechunkDecider::capped_best_index(): an array banned from
//              coarsening (it paid for one bad coarsening) stays at its size
//              whatever max_coarsen_factor is -- 1, "no cap", returned the
//              coarser size before it looked at the ban;
//   no scans   ChunkSizeAdvisor::analyze() with Options::max_mat_scans = 0 on
//              an array that has a materialised chunk returns a report. The
//              sampling stride divided by that option; this array does not
//              reach the sampled branch (it passed before the guard too): the
//              case keeps the call alive, it does not prove the guard;
//   chunk =    zero::Chunk's copy-assignment compiles and copies the state
//              (it named a member that does not exist; nothing instantiated
//              it). zero builds only.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <type_traits> // std::conditional_t

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array, zero::Chunk
#include "chunk/zero_or_value_array.h"   // zero_or_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/chunk_size_advisor.h"    // ChunkSizeAdvisor, ChunkSizeReport
#include "chunk/rechunk_decider.h"       // RechunkDecider, RechunkPolicy
#include "chunk/runtime.h"               // Runtime::init
#include "chunk/config.h"                // chunk_config::use_zero*
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name

namespace validation::advisor_edge_validation
{

using namespace shrink_array;

namespace
{

using Arr = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<double>,
    std::conditional_t<chunk_config::use_zero_or_value, zero_or_value::Array<double>, zero::Array<double>>>;

int report(const char* what, bool ok)
{
    println("[{}/advisor-edge] {:<9} {}", variant_name(), what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

/// A report whose best size (512) is coarser than the current one (256).
ChunkSizeReport coarser_is_best()
{
    ChunkSizeReport rep;
    rep.table.resize(2);
    rep.table[0].size      = 256;
    rep.table[0].mem_bytes = 100.;
    rep.table[1].size      = 512;
    rep.table[1].mem_bytes = 50.;
    rep.cur                = 0;
    rep.best               = 1;
    return rep;
}

int ban_holds_without_a_cap()
{
    const ChunkSizeReport rep   = coarser_is_best();
    int                   fails = 0;
    for (const unsigned factor : {0U, 1U, 4U})
    {
        RechunkPolicy policy;
        policy.max_coarsen_factor = factor;
        const RechunkDecider decider(policy);
        // Banned: the current size. Not banned: the coarser one (within any cap here).
        fails += decider.capped_best_index(rep, /*banned=*/true) != rep.cur;
        fails += decider.capped_best_index(rep, /*banned=*/false) != rep.best;
    }
    return report("ban", fails == 0);
}

int analyze_without_scans()
{
    Arr array(1 << 16);
    array[3] = 1.0; // one materialised chunk
    ChunkSizeAdvisor<double>::Options options;
    options.max_mat_scans     = 0;
    const ChunkSizeReport rep = ChunkSizeAdvisor<double>::analyze(array, options);
    return report("no scans", rep.cur >= 0 && !rep.table.empty());
}

template <typename ChunkT = zero::Chunk<double>>
int chunk_copy_assignment()
{
    if constexpr (chunk_config::use_zero)
    {
        // The state of a materialised chunk and of a compressed one, copied
        // (the copies share the array's buffers and release nothing).
        Arr array(1 << 16);
        array[3] = 1.0;
        ChunkT from_written;
        ChunkT from_constant;
        from_written  = array.chunk_ref(0);
        from_constant = array.chunk_ref(1);
        return report("chunk =", from_written.written() && !from_constant.written());
    }
    else
    {
        return 0;
    }
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    Runtime::init(true);
    const int fails = ban_holds_without_a_cap() + analyze_without_scans() + chunk_copy_assignment();
    println("{}", fails ? "advisor_edge_validation: FAILED" : "advisor_edge_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::advisor_edge_validation
