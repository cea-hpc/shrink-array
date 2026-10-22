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
// Pool-accounting harness. Variant-agnostic: tests chunk / chunk2 / chunk2a
// depending on the compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the
// other harnesses here.
// ============================================================================
//
// Guards the manager's bookkeeping across an array's life: what it counts and
// what it parks must match what is actually alive.
//
// WHAT IT CHECKS
//
//   1. somme_chunk_create tracks the chunks ALIVE: it grows by an array's chunk
//      count at creation and comes back down by the same amount at clear().
//      That count feeds the compression threshold
//      (somme_chunk_create / (factor_div * threads), floored), so a count that
//      only ever grows -- v1's did, its clear_core() never decremented -- makes
//      the threshold climb with the program's history (390 -> ~42 000 on a v1
//      dyn run) until compression all but stops. After every array is gone
//      the recomputed threshold must be back at the floor.
//   2. The buffers a cleared array held are parked for reuse, in exact number
//      (every written chunk, none twice), and the next array takes them back.
//      The pools are per rank and an array is filled and cleared under two
//      different distributions of its chunks over the team, so a rank can run
//      dry and allocate fresh while another keeps a leftover: the residue is
//      allowed up to one buffer per thread (it is 0 at one thread). Serial
//      builds free instead of parking, so there the pools stay empty.
//   3. Dynamic geometry only: a rechunk_to() leaves nothing parked at the OLD
//      geometry once its drain has run, and the count of chunks alive follows
//      the new geometry.
//
// Then the library's own teardown, as compress_roundtrip does.
//
// Compile OMP or non-OMP, static or dyn, check. Exit 0 = every count matches.

#include <type_traits> // std::conditional_t

#include "chunk/chunk_traits.h"               // forward decls + default IndT = int
#include "chunk/array_chunk.h"                // ArrayChunk
#include "chunk/array_chunk2.h"               // ArrayChunk2
#include "chunk/array_chunk2A.h"              // ArrayChunk2A
#include "chunk/param_chunk.h"                // ParamChunk::init_indexes
#include "chunk/chunk_std.h"                  // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/chunk_size_policy.h" // cst_buffer_size
#include "chunk/config.h"                     // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS, pmp::current_team
#include "chunk/manage_ompt.h"                // parallel_phase
#include "chunk/chunk_def.h"                  // ChunkDef
#include "chunk/chunk2_def.h"                 // Chunk2Def
#include "chunk/chunk2A_def.h"                // Chunk2ADef
#include "chunk/internal/env_parse.h"         // env_number / arg_number (checked, not atoi)
#include "chunk/internal/print.h"             // shrink_array::println

// Defined by every executable of the library (the fast allocator reads it).
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::pool_accounting
{

using namespace shrink_array;

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

using VariantDef = std::conditional_t<chunk_config::use_chunk2a,
                                      Chunk2ADef,
                                      std::conditional_t<chunk_config::use_chunk2, Chunk2Def, ChunkDef>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

// The phase brackets: parallel_phase() (manage_ompt.h) places them.

// Serial builds free a cleared array's buffers outright; threaded ones
// (OpenMP, std) park them.
constexpr bool parks_on_clear = build_config::threaded;

using Manager = ArrayChunkTraits<Array<double>>::ManagerType;

static int  g_fails = 0;
static void expect(bool ok, const char* what, long got, long want)
{
    if (ok)
    {
        println("[{}] OK   {:<58} {}", variant_name(), what, got);
    }
    else
    {
        println("[{}] FAIL {:<58} got={} expected={}", variant_name(), what, got, want);
        ++g_fails;
    }
}

/// Materialise every chunk of @p arr with a value that is neither constant,
/// on a team of the build's backend (parallel_phase: OpenMP, or the std backend,
/// where an `omp parallel for` would run on one thread and take the parked
/// buffers of one rank only), static blocks as `omp for` gives.
static void materialise_all(Array<double>& arr, long n)
{
    parallel_phase([&]
                   { pmp::current_team().for_static(0L, n, [&](long i) { arr[i] = 1.0 + static_cast<double>(i); }); });
}

int run(int argc, char** argv)
{
    const long n = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 20, "N") : (1L << 20);

    ParamChunk::init_indexes(true);

    println("=== pool_accounting [variant = {}]  N={}  threads={} ===", variant_name(), n, CHUNK_PMP_MAX_THREADS);

    const long base = Manager::get_somme_chunk_create();
    expect(Manager::reuse_pool_blocks() == 0, "pools empty at start", (long)Manager::reuse_pool_blocks(), 0);

    // --- 1. count alive: two arrays, then one cleared, then both ------------
    long chunks_a = 0;
    long chunks_b = 0;
    {
        Array<double> a(n);
        maybe_set_second_value(a, 3.5);
        chunks_a = Manager::get_somme_chunk_create() - base;
        expect(chunks_a > 0, "somme_chunk_create grew by a's chunks", chunks_a, chunks_a);

        Array<double> b(n / 2);
        maybe_set_second_value(b, 3.5);
        chunks_b = Manager::get_somme_chunk_create() - base - chunks_a;
        expect(chunks_b > 0, "somme_chunk_create grew by b's chunks", chunks_b, chunks_b);

        materialise_all(a, n);
        materialise_all(b, n / 2);
        // The array keeps one chunk past the last element (N / csz + 1), so when
        // N is a multiple of the chunk size that spare chunk is never written.
        const long csz       = static_cast<long>(a.chunk_size());
        const long covering  = (n + csz - 1) / csz;
        const long written_a = a.get_nb_chunk_written();
        expect(written_a == covering, "a fully materialised (written == chunks covering N)", written_a, covering);

        // --- 2. parking: a's buffers go to the pools, exactly ----------------
        // Unless a owns its slabs (per-array affinity, alloc_key() != 0): then
        // clear() drains its lists straight back to its allocator set, whose
        // slabs empty, and NOTHING stays parked. The default threshold keeps
        // this harness's arrays below the line; -DCHUNK_ALLOC_AFFINITY_MIN_BYTES=1
        // puts every array above it, and this is what the check expects then.
        const bool a_owns_slabs = a.alloc_key() != 0;
        a.clear();
        expect(Manager::get_somme_chunk_create() == base + chunks_b,
               "somme_chunk_create back down by a's chunks after a.clear()", Manager::get_somme_chunk_create(),
               base + chunks_b);
        const long parked      = static_cast<long>(Manager::reuse_pool_blocks());
        const long want_parked = (parks_on_clear && !a_owns_slabs) ? written_a : 0;
        expect(parked == want_parked,
               a_owns_slabs     ? "a owned its slabs: nothing parked after a.clear()"
               : parks_on_clear ? "a's written buffers parked for reuse"
                                : "serial: nothing parked",
               parked, want_parked);

        // The next array takes them back.
        Array<double> c(n);
        maybe_set_second_value(c, 3.5);
        materialise_all(c, n);
        const long written_c    = c.get_nb_chunk_written();
        const long parked_after = static_cast<long>(Manager::reuse_pool_blocks());
        const long want_after   = parked > written_c ? parked - written_c : 0;
        const long slack        = static_cast<long>(CHUNK_PMP_MAX_THREADS); // per-rank pools, see the header
        expect(parked_after >= want_after && parked_after <= want_after + slack,
               "pools drained by what c materialised (+/- one per rank)", parked_after, want_after);

        // --- 3. re-chunk (dyn): nothing left parked at the old geometry ------
        if constexpr (build_config::chunk_dynamic)
        {
            const unsigned old_size = b.chunk_size();
            const unsigned new_size = old_size * 4 <= shrink_array::cst_buffer_size ? old_size * 4 : old_size / 2;
            if (new_size != old_size)
            {
                const long before = Manager::get_somme_chunk_create();
                b.rechunk_to(new_size);
                expect(b.chunk_size() == new_size, "b re-chunked to the new geometry", b.chunk_size(), new_size);
                const long at_old = static_cast<long>(Manager::reuse_pool_blocks_at(old_size));
                expect(at_old == 0, "nothing parked at the OLD geometry after rechunk_to", at_old, 0);
                // Chunks alive followed the geometry: fewer chunks at a larger size.
                const long delta = Manager::get_somme_chunk_create() - before;
                const long want  = static_cast<long>(b.get_nb_chunk_written()) - chunks_b;
                expect(delta <= 0 && delta >= want - 1, "somme_chunk_create follows the new chunk count", delta, want);
            }
        }
    } // b and c cleared here

    expect(Manager::get_somme_chunk_create() == base,
           "somme_chunk_create back to its start value once every array is gone", Manager::get_somme_chunk_create(),
           base);

    Manager::compute_size_max_number_of_chunk_to_check_compress();
    expect(
        Manager::get_size_max_number_of_chunk_to_check_compress() == VariantDef::size_min_list_chunk_to_check_compress,
        "compression threshold back at the floor with no array alive",
        Manager::get_size_max_number_of_chunk_to_check_compress(), VariantDef::size_min_list_chunk_to_check_compress);

    // The library's own teardown (see compress_roundtrip.cpp).
    // No #ifdef: release_chunk_pools() does whichever half this build needs --
    // the per-rank lists under OpenMP, the single allocator otherwise. See
    // ChunkPoolParallelInterface in chunk_std.h.
    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            g_fails == 0 ? "=== POOL ACCOUNTING OK, TEARDOWN OK ===" : "=== POOL ACCOUNTING FAILURES ===");
    return g_fails;
}

} // namespace validation::pool_accounting
