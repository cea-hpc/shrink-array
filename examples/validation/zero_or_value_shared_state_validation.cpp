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
// zero_or_value shared-state harness. zero_or_value only, default static size.
// ============================================================================
//
// Two things one array does that must leave the OTHER arrays of its type
// alone, both through state the arrays of a type share:
//
// 1. copy-assign (`a = b`, `b` declaring another second value). The constant
//    block of a second value is shared by every array that declares it. A
//    chunk counts as compressed to the second value when its values pointer
//    is the block of the array's CURRENT index, and operator= switches the
//    index (copy_second_value_from) before it copies the elements
//    (copy_from): the chunks of `a` still on the old block would pass as
//    materialised and the copy would write `b`'s elements INTO the old block,
//    read by every chunk of every other array compressed to that value.
//    operator= releases `a`'s chunks first when the value changes.
//    Checked: `c`, a bystander compressed to the same value, still reads it
//    everywhere; `a` reads `b`'s elements; no chunk of `a` is left on the old
//    block.
//
// 2. a new zero_or_value::Array<bool>. Each bool array's create() registers
//    `true` as its second value, and the `true` candidate list of a rank is
//    shared by all of them: a registration that installed a fresh copy would
//    drop the chunks the other bool arrays had queued (their dedup flags still
//    set, they would not be queued again, and the passes would never compress
//    them). The list is installed once.
//    Its destruction, in the region, must leave them queued too: at one
//    thread the region is inactive (omp_in_parallel() is false), and a
//    teardown that took it for a sequential point ran the pass that walks
//    every type's lists without compressing, emptying them.
//    Checked (OpenMP builds, at 1 and 8 threads): the chunks of `x`, rewritten
//    to `true` in a region, are still queued while another bool array built
//    in that region lives and once it is destroyed, and the region's end
//    compresses as many of them as it does without that array (the control
//    run).
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <cstddef> // std::size_t
#include <memory>  // std::make_unique (owner_run)
#include <utility> // std::as_const, std::move, std::pair

#include "chunk/zero_or_value_array.h" // zero_or_value::Array
#include "chunk/runtime.h"             // Runtime::init
#include "chunk/internal/zero_chunk.h" // release_chunk_pools, Runtime::shutdown
#include "chunk/config.h"              // build_config::use_openmp
#include "chunk/print.h"               // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, begin_phase / end_phase

namespace validation::zero_or_value_shared_state_validation
{

using namespace shrink_array;

namespace
{

constexpr long n_chunks = 16; ///< chunks of every array below

/// @brief Chunks of @p arr whose values pointer is @p block.
template <typename Arr>
long chunks_on(const Arr& arr, const void* block)
{
    long count = 0;
    for (long c = 0; c < n_chunks; ++c)
    {
        count += (static_cast<const void*>(arr.chunk_ref(static_cast<std::size_t>(c)).values_load()) == block) ? 1 : 0;
    }
    return count;
}

/// @brief Chunks of @p arr not materialised (at the first or the second value).
template <typename Arr>
long compressed(const Arr& arr, long chunk_sz)
{
    long count = 0;
    for (long c = 0; c < n_chunks; ++c)
    {
        count += arr.written(static_cast<typename Arr::IndexType>(c * chunk_sz)) ? 0 : 1;
    }
    return count;
}

// ── 1. copy-assign across second values ─────────────────────────────────────

int run_copy_assign()
{
    using Arr           = zero_or_value::Array<double>;
    using Index         = Arr::IndexType;
    constexpr double v1 = 1.0; ///< `a`'s and `c`'s second value
    constexpr double v2 = 2.0; ///< `b`'s second value
    constexpr double vb = 7.0; ///< what `b` holds

    Arr        c;
    Arr        a;
    Arr        b;
    const auto chunk_sz = static_cast<long>(a.chunk_size());
    const long count    = n_chunks * chunk_sz;
    c.resize(static_cast<Index>(count));
    a.resize(static_cast<Index>(count));
    b.resize(static_cast<Index>(count));
    c.set_second_value(v1);
    a.set_second_value(v1);
    b.set_second_value(v2);
    for (long i = 0; i < count; ++i)
    {
        c[static_cast<Index>(i)] = v1;
        a[static_cast<Index>(i)] = v1;
        b[static_cast<Index>(i)] = vb;
    }
    c.shrink_to_fit();
    a.shrink_to_fit();

    const Arr&  cc      = c;
    const Arr&  ca      = a;
    const void* block1  = cc.chunk_ref(0).values_load();
    const long  c_on_v1 = chunks_on(cc, block1);
    const long  a_on_v1 = chunks_on(ca, block1);
    println("[shared-state/copy-assign] before a = b: c has {}/{} chunks on {:g}'s block, a has {}", c_on_v1, n_chunks,
            v1, a_on_v1);
    if (c_on_v1 != n_chunks || a_on_v1 != n_chunks)
    {
        println("[shared-state/copy-assign] FAIL premise: `a` and `c` are not both compressed to {:g}", v1);
        return 1;
    }

    a = b;

    long bad_c = 0;
    long bad_a = 0;
    for (long i = 0; i < count; ++i)
    {
        bad_c += (cc[static_cast<Index>(i)] != v1) ? 1 : 0;
        bad_a += (ca[static_cast<Index>(i)] != vb) ? 1 : 0;
    }
    const long a_left = chunks_on(ca, block1);
    println("[shared-state/copy-assign] after a = b: {} element(s) of c not {:g}, {} of a not {:g}, {} chunk(s) of a "
            "still on {:g}'s block",
            bad_c, v1, bad_a, vb, a_left, v1);
    if (bad_c != 0 || bad_a != 0 || a_left != 0)
    {
        println("[shared-state/copy-assign] FAIL: the copy wrote into the block {:g} shares with `c`", v1);
        return 1;
    }
    println("[shared-state/copy-assign] OK");
    return 0;
}

// ── 2. a new bool array keeps the other bool arrays' queued candidates ──────

using BoolArr = zero_or_value::Array<bool>;

/// @brief Entries queued under key `true`, summed over the ranks.
long queued_true()
{
    long        total = 0;
    const auto& ranks = BoolArr::ManagerType::zero_or_value_candidates_map_thr;
    for (const auto& by_value : ranks)
    {
        const auto it = by_value.find(true);
        if (it != by_value.end() && it->second)
        {
            total += static_cast<long>(it->second->size());
        }
    }
    return total;
}

struct BoolRun
{
    long queued_before = 0; ///< `true` entries queued once `x` is rewritten
    long queued_after  = 0; ///< the same, after the other array's construction (or not)
    long queued_gone   = 0; ///< the same, once that array is destroyed
    long compressed    = 0; ///< chunks of `x` compressed after the region
};

/// @brief `x` (all `false`, compressed), every element rewritten to `true` in a
///        region; after a barrier, one thread counts the queued entries,
///        optionally builds another bool array, counts again.
BoolRun bool_run(bool build_other)
{
    using Index = BoolArr::IndexType;
    BoolRun    out;
    BoolArr    x;
    const auto chunk_sz = static_cast<long>(x.chunk_size());
    const long count    = n_chunks * chunk_sz;
    x.resize(static_cast<Index>(count));
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            x[static_cast<Index>(i)] = true;
        }
#pragma omp single
        {
            out.queued_before = queued_true();
            if (build_other)
            {
                const BoolArr other(static_cast<Index>(chunk_sz));
                out.queued_after = queued_true();
            }
            else
            {
                out.queued_after = queued_true();
            }
            out.queued_gone = queued_true();
        }
    }
    end_phase();
    out.compressed = compressed(std::as_const(x), chunk_sz);
    return out;
}

int run_bool_construct()
{
    if constexpr (!build_config::use_openmp)
    {
        println("[shared-state/bool-construct] skipped: serial build, no candidate lists");
        return 0;
    }
    const BoolRun control = bool_run(false);
    const BoolRun tested  = bool_run(true);
    println("[shared-state/bool-construct] control: {} queued, {} after, {} at the end, {}/{} chunks compressed",
            control.queued_before, control.queued_after, control.queued_gone, control.compressed, n_chunks);
    println("[shared-state/bool-construct] another bool array built: {} queued, {} while it lives, {} once destroyed, "
            "{}/{} chunks compressed",
            tested.queued_before, tested.queued_after, tested.queued_gone, tested.compressed, n_chunks);
    int fails = 0;
    if (tested.queued_after < tested.queued_before)
    {
        println("[shared-state/bool-construct] FAIL: building a bool array dropped {} queued `true` candidate(s) of "
                "another array",
                tested.queued_before - tested.queued_after);
        ++fails;
    }
    if (tested.queued_gone < tested.queued_before)
    {
        println("[shared-state/bool-construct] FAIL: destroying a bool array dropped {} queued `true` candidate(s) of "
                "another array",
                tested.queued_before - tested.queued_gone);
        ++fails;
    }
    if (tested.compressed < control.compressed)
    {
        println("[shared-state/bool-construct] FAIL: {} chunk(s) of `x` left materialised that the control run "
                "compresses",
                control.compressed - tested.compressed);
        ++fails;
    }
    if (fails == 0)
    {
        println("[shared-state/bool-construct] OK");
    }
    return fails;
}

// ── 3. the array a second value is registered under goes away ───────────────
//
// array_info_map_[v] points at the array_info member of the LAST array that
// declared v, and the passes drain only the values of that map. `c` and `a`
// both declare v1, `a` last; then `a` goes (destroyed, cleared, copy- or
// move-assigned from an array of v2). `c`, rewritten to v1 in a region, must
// still be compressed by the region's end as in the control run.

enum class Gone
{
    control, ///< `a` lives on
    destroyed,
    cleared,
    copy_assigned,
    move_assigned,
    swapped_then_destroyed, ///< a.swap(b), then b (now holding v1) destroyed
};

const char* gone_name(Gone g)
{
    switch (g)
    {
        case Gone::control:
            return "control";
        case Gone::destroyed:
            return "destroyed";
        case Gone::cleared:
            return "cleared";
        case Gone::copy_assigned:
            return "copy-assigned";
        case Gone::move_assigned:
            return "move-assigned";
        case Gone::swapped_then_destroyed:
            return "swapped, then destroyed";
    }
    return "?";
}

struct OwnerRun
{
    bool registered = false; ///< array_info_map_[v1] present and holding v1 before the region
    long compressed = 0;     ///< chunks of `c` compressed after the region
    long bad        = 0;     ///< elements of `c` not v1
};

OwnerRun owner_run(Gone gone)
{
    using Arr           = zero_or_value::Array<double>;
    using Index         = Arr::IndexType;
    constexpr double v1 = 3.0;
    constexpr double v2 = 5.0;

    OwnerRun   out;
    Arr        c;
    const auto chunk_sz = static_cast<long>(c.chunk_size());
    const long count    = n_chunks * chunk_sz;
    c.resize(static_cast<Index>(count));
    c.set_second_value(v1);

    auto a = std::make_unique<Arr>();
    a->resize(static_cast<Index>(count));
    a->set_second_value(v1);
    {
        Arr b;
        b.resize(static_cast<Index>(count));
        b.set_second_value(v2);
        switch (gone)
        {
            case Gone::control:
                break;
            case Gone::destroyed:
                a.reset();
                break;
            case Gone::cleared:
                a->clear();
                break;
            case Gone::copy_assigned:
                *a = b;
                break;
            case Gone::move_assigned:
                *a = std::move(b);
                break;
            case Gone::swapped_then_destroyed:
                a->swap(b);
                break; // b dies at the brace
        }
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto* info = zero_or_value::Proxy<double>::array_registry->find_array_info(v1);
        out.registered   = info != nullptr && info->second_value == v1;
    }
    else
    {
        out.registered = true;
    }

#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            c[static_cast<Index>(i)] = v1;
        }
    }
    end_phase();

    const Arr& cc  = c;
    out.compressed = compressed(cc, chunk_sz);
    for (long i = 0; i < count; ++i)
    {
        out.bad += (cc[static_cast<Index>(i)] != v1) ? 1 : 0;
    }
    return out;
}

int run_owner_gone()
{
    int            fails   = 0;
    const OwnerRun control = owner_run(Gone::control);
    println("[shared-state/owner-gone] control: v1 registered {}, {}/{} chunks of c compressed", control.registered,
            control.compressed, n_chunks);
    for (Gone g :
         {Gone::destroyed, Gone::cleared, Gone::copy_assigned, Gone::move_assigned, Gone::swapped_then_destroyed})
    {
        const OwnerRun r = owner_run(g);
        println("[shared-state/owner-gone] a {}: v1 registered {}, {}/{} chunks of c compressed, {} bad element(s)",
                gone_name(g), r.registered, r.compressed, n_chunks, r.bad);
        if (r.bad != 0 || !r.registered || r.compressed < control.compressed)
        {
            println("[shared-state/owner-gone] FAIL ({}): `c` declares v1 too and lost {}", gone_name(g),
                    r.bad != 0 ? "its values" : "the drain of its v1 candidates");
            ++fails;
        }
    }
    if (fails == 0)
    {
        println("[shared-state/owner-gone] OK");
    }
    return fails;
}

// ── 4. set_second_value(T{}) after another value keeps the T{} key ──────────
//
// The arrays with no second value are drained under the key T{}: its record
// in array_info_map_ (at the sentinel index) and a candidate list per rank
// (register_no_second_value_key). `y` declares 5, then T{}: T{} is now a
// real second value of `y`, and its registration must not replace the T{}
// key's list -- the chunks `x` (no second value) queued there, rewritten to
// T{} in the region, would be dropped, their dedup flags still set, and the
// region's end would leave them materialised.
// Twice: inside the region (outside set_second_value()'s contract, which asks
// for a sequential point, but where the lost candidates show), and before it
// (the documented use: nothing queued yet, but the T{} record took a real
// index -- arrays with no second value were then drained to a block of T{}).

using DblArr = zero_or_value::Array<double>;

/// @brief Entries queued under key T{} (0.0), summed over the ranks.
long queued_zero()
{
    long        total = 0;
    const auto& ranks = DblArr::ManagerType::zero_or_value_candidates_map_thr;
    for (const auto& by_value : ranks)
    {
        const auto it = by_value.find(0.0);
        if (it != by_value.end() && it->second)
        {
            total += static_cast<long>(it->second->size());
        }
    }
    return total;
}

struct ZeroKeyRun
{
    long queued_before = 0;    ///< T{} entries queued once `x` is rewritten
    long queued_after  = 0;    ///< the same, after `y`'s set_second_value(5), then (T{}) (or not)
    long compressed    = 0;    ///< chunks of `x` compressed after the region
    long bad_y         = 0;    ///< elements of `y` not read back
    int  zero_idx      = 0;    ///< index of the T{} record after the run (the sentinel expected)
    bool y_back_to_t0  = true; ///< y.second_value() is T{} after the redeclaration
};

enum class Redeclare
{
    no,
    in_region,
    before_region,
};

ZeroKeyRun zero_key_run(Redeclare redeclare)
{
    using Index = DblArr::IndexType;
    ZeroKeyRun out;
    DblArr     x;
    DblArr     y;
    const auto chunk_sz = static_cast<long>(x.chunk_size());
    const long count    = n_chunks * chunk_sz;
    x.resize(static_cast<Index>(count));
    y.resize(static_cast<Index>(count));
    for (long i = 0; i < count; ++i)
    {
        x[static_cast<Index>(i)] = 1.0 + static_cast<double>(i % 7); // materialised
    }
    if (redeclare == Redeclare::before_region)
    {
        y.set_second_value(5.0, false);
        y.set_second_value(0.0, false);
    }
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            x[static_cast<Index>(i)] = 0.0;
        }
#pragma omp single
        {
            out.queued_before = queued_zero();
            if (redeclare == Redeclare::in_region)
            {
                y.set_second_value(5.0, false);
                y.set_second_value(0.0, false);
            }
            out.queued_after = queued_zero();
        }
    }
    end_phase();
    out.compressed        = compressed(std::as_const(x), chunk_sz);
    const auto* zero_info = zero_or_value::Proxy<double>::array_registry->find_array_info(0.0);
    out.zero_idx          = zero_info != nullptr ? zero_info->idx : chunk_config::no_second_value_index;
    out.y_back_to_t0      = y.second_value() == 0.0;
    for (long i = 0; i < count; ++i)
    {
        y[static_cast<Index>(i)] = (i % 3 == 0) ? 0.0 : 2.0;
    }
    for (long i = 0; i < count; ++i)
    {
        out.bad_y += (std::as_const(y)[static_cast<Index>(i)] != ((i % 3 == 0) ? 0.0 : 2.0)) ? 1 : 0;
    }
    return out;
}

int run_zero_key()
{
    if constexpr (!build_config::use_openmp || !chunk_config::use_fast_block_compression)
    {
        println("[shared-state/zero-key] skipped: no candidate lists in this build");
        return 0;
    }
    const ZeroKeyRun control = zero_key_run(Redeclare::no);
    println("[shared-state/zero-key] control: {} queued, {} after, {}/{} chunks of x compressed", control.queued_before,
            control.queued_after, control.compressed, n_chunks);
    int fails = 0;
    for (const auto& [mode, where] :
         {std::pair{Redeclare::in_region, "in the region"}, std::pair{Redeclare::before_region, "before the region"}})
    {
        const ZeroKeyRun tested = zero_key_run(mode);
        println("[shared-state/zero-key] y declares 5 then T{{}} {}: {} queued, {} after, {}/{} chunks of x "
                "compressed, {} bad element(s) of y, T{{}} record at index {}, y back to T{{}}: {}",
                where, tested.queued_before, tested.queued_after, tested.compressed, n_chunks, tested.bad_y,
                tested.zero_idx, tested.y_back_to_t0);
        if (tested.queued_after < tested.queued_before)
        {
            println("[shared-state/zero-key] FAIL ({}): set_second_value(T{{}}) dropped {} queued T{{}} candidate(s) "
                    "of an array without second value",
                    where, tested.queued_before - tested.queued_after);
            ++fails;
        }
        if (tested.compressed < control.compressed)
        {
            println("[shared-state/zero-key] FAIL ({}): {} chunk(s) of `x` left materialised that the control run "
                    "compresses",
                    where, control.compressed - tested.compressed);
            ++fails;
        }
        if (tested.bad_y != 0)
        {
            println("[shared-state/zero-key] FAIL ({}): `y` (second value T{{}}) reads {} element(s) wrong", where,
                    tested.bad_y);
            ++fails;
        }
        if (tested.zero_idx != chunk_config::no_second_value_index)
        {
            println("[shared-state/zero-key] FAIL ({}): the T{{}} record took index {}, not the sentinel {}", where,
                    tested.zero_idx, chunk_config::no_second_value_index);
            ++fails;
        }
        if (!tested.y_back_to_t0)
        {
            println("[shared-state/zero-key] FAIL ({}): y.second_value() is not T{{}} after set_second_value(T{{}})",
                    where);
            ++fails;
        }
    }
    if (fails == 0)
    {
        println("[shared-state/zero-key] OK");
    }
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    Runtime::init(true);
    println("=== zero_or_value_shared_state_validation ===");
    int fails = 0;
    fails += run_copy_assign();
    fails += run_bool_construct();
    fails += run_owner_gone();
    fails += run_zero_key();
    release_chunk_pools();
    Runtime::shutdown();
    println("\n[zero_or_value] {}", fails ? "=== SHARED STATE FAILED ===" : "=== SHARED STATE OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::zero_or_value_shared_state_validation
