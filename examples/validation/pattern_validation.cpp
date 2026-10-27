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

/**
 * @file pattern_validation.cpp
 * @brief pattern::Array against std::vector (docs/algo_pattern.md).
 *
 * @details Built against the zero_any library (pattern::Array is a member of
 *          the zero_any family, built on its machinery), at the default static
 *          size and in the dynamic geometry, as its own executables:
 *          pattern_validation_zero_any_<size> and pattern_validation_zero_any_dyn.
 *
 *          For int, long, unsigned and short, an array of chunks of four
 *          kinds -- zero, uniform at a value of their own, a shifted motif
 *          (the same motif plus an offset per chunk) and random values -- is
 *          checked element by element against a std::vector holding the same
 *          values, after each of the steps below. The same steps run on a
 *          zero_any::Array, and the two must compress the same number of
 *          chunks to a constant at A and B: pattern::Array is zero_any's
 *          machinery with tables of its own. At C pattern::Array compresses
 *          every uniform chunk, zero_any at most as many (see C). The motif
 *          chunks are counted apart (the motif kind is compressed to its motif
 *          by pattern::Array only).
 *
 *            A  the first write of every element, then the between-region
 *               pass: the zero and uniform chunks are compressed (the
 *               family's machinery runs for pattern::Array: its own manager,
 *               candidate lists and tables);
 *            B  every fifth element of the uniform and motif chunks rewritten;
 *            C  the uniform chunks made uniform again at a new value. Not
 *               all of them compress, in zero_any as here: a chunk whose LAST
 *               write lands on an element B changed (old value neither
 *               constant) is not queued, the write-path marking only queues
 *               a write from or to one of the chunk's two constants.
 *               pattern::Array also queues a chunk whose first or last element
 *               is written (its completion rule, docs/algo_pattern.md,
 *               section 4): every uniform chunk is compressed again;
 *            D  a copy (equal, then unequal after one write in it);
 *            E  a move (the source left empty);
 *            F  a shrinking then a growing resize (std::vector semantics);
 *            G  assign(count, value).
 *
 *          Then H, the second-value registry shared with zero_any: a
 *          zero_any::Array<int> and a pattern::Array<int>, every chunk at a
 *          constant of its own, draw from the same GlobalVecHolder<int>. A
 *          reclaim_second_values() called on either type frees what NEITHER
 *          names any more, and only that: the values of the other array
 *          still read back after the freed identifiers are handed out again.
 *
 *          Then the motif state, chunks put in it by install_motif_for_test()
 *          (the compression to motifs is stage 2 of the note), every third chunk
 *          the same motif with an offset of its own (negative ones, and for
 *          short past the type's range: the arithmetic wraps), for int, long,
 *          unsigned and short:
 *
 *            I  one motif registered for all of them; every read path against
 *               std::vector -- const operator[], the iterator, read_range,
 *               for_each_value, a copy, operator== -- and no chunk materialised;
 *            J  one write in each motif chunk: materialised from its motif
 *               (motif + offset), the other elements unchanged;
 *            K  reads through the write proxy: right, and nothing materialised;
 *            L  a compound += on motif chunks (the old value read through the
 *               proxy), then the same comparison.
 *
 *          Then the compression to motifs, by the write path (a chunk is queued
 *          when its first or last element is written) and the pass (a motif is
 *          registered at its second sighting), for the same four types:
 *
 *            M  a connectivity -- every chunk one motif plus its own offset --
 *               written in order: every chunk compressed to the motif but the
 *               first sightings (one, or one per thread), one motif registered;
 *            N  one write in every other chunk: the motif ones materialised;
 *            O  random chunks, then the same rewritten unchanged: no motif
 *               admitted (a second sighting by the same chunk does not count);
 *            P  everything rewritten at random, reclaim_second_values(): every
 *               motif of the type freed; the connectivity again: compressed again;
 *            Q  (dynamic geometry) the connectivity rechunked to twice, half, four
 *               times and back to the chunk size: the values, and the chunks
 *               compressed again to motifs of the new size.
 *
 *          R  the chunk-size advisor on a pattern::Array and a zero_any::Array
 *             holding the same four kinds of chunks: the same table, row for row
 *             -- pattern's motif chunks are read as data (motif + offset), never
 *             as constants.
 *
 *          Usage: pattern_validation [chunks]
 */

#include <algorithm> // std::copy (the motif chunks)
#include <cstdint>   // std::uint64_t (the random values)
#include <utility>   // std::as_const, std::move, std::pair (the motif writes)
#include <vector>    // std::vector, the reference

#include "chunk/chunk_traits.h"       // forward decls + default IntType = int
#include "chunk/zero_any_array.h"     // zero_any::Array (H)
#include "chunk/pattern_array.h"      // pattern::Array, pattern::MotifRegion
#include "chunk/chunk_size_advisor.h" // ChunkSizeAdvisor (R)
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"         // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"             // chunk_config::use_zero_any, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)
#include "chunk/internal/print.h"     // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::pattern_validation
{

using namespace shrink_array;

static_assert(chunk_config::use_zero_any, "pattern_validation runs in the zero_any build: link chunk_zero_any_<size>");

static int fails = 0;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

/// @brief End a write phase. Serial zero_any has no between-region pass: run
///        the writers' pass of @p Arr.
template <typename Arr>
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
    if constexpr (!build_config::use_openmp)
    {
        Arr::make_block_compression();
    }
}

/// @brief Write @p ref[i] into @p arr[i] for every i in [@p lo, @p hi), in a
///        parallel phase, then the between-region pass.
template <typename Arr, typename T>
static void write_range(Arr& arr, const std::vector<T>& ref, long lo, long hi)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = lo; i < hi; ++i)
        {
            arr[i] = ref[static_cast<std::size_t>(i)];
        }
    }
    end_phase<Arr>();
}

/// @brief Every element of @p arr against @p ref, and the sizes.
template <typename Arr, typename T>
static void expect_same(const char* where, const Arr& arr, const std::vector<T>& ref)
{
    if (static_cast<std::size_t>(arr.size()) != ref.size())
    {
        println("  {:<44} size {} (expect {})  <-- WRONG", where, static_cast<long>(arr.size()),
                static_cast<long>(ref.size()));
        ++fails;
        return;
    }
    long bad   = 0;
    long first = -1;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if (std::as_const(arr)[static_cast<long>(i)] != ref[i])
        {
            first = first < 0 ? static_cast<long>(i) : first;
            ++bad;
        }
    }
    if (bad != 0)
    {
        println("  {:<44} {} element(s) wrong, first at {}  <-- WRONG", where, bad, first);
        ++fails;
    }
}

/// @brief A count against its expected value, printed either way.
static void expect_eq(const char* where, long got, long want)
{
    println("  {:<44} {} (expect {}){}", where, got, want, got == want ? "" : "  <-- WRONG");
    if (got != want)
    {
        ++fails;
    }
}

/// @brief A condition, printed either way.
static void expect_true(const char* where, bool ok)
{
    println("  {:<44} {}", where, ok ? "yes" : "NO  <-- WRONG");
    if (!ok)
    {
        ++fails;
    }
}

/// @brief The kind of chunk @p num_chunk: zero, uniform, motif, random, in turn.
enum class Kind
{
    zero,
    uniform,
    motif,
    random
};

static Kind kind_of(long num_chunk)
{
    return static_cast<Kind>(num_chunk % 4);
}

/// @brief Element @p j of chunk @p num_chunk of kind @p kind; @p round shifts
///        the constants and offsets from one round to the next.
template <typename T>
static T value_of(Kind kind, long num_chunk, long j, long round, std::uint64_t& rng)
{
    switch (kind)
    {
        case Kind::zero:
            return T{};
        case Kind::uniform:
            return static_cast<T>(num_chunk * 7 + 3 + round * 1000);
        case Kind::motif:
            // The same motif in every motif chunk, shifted by an offset of its own.
            return static_cast<T>((j * j) % 13 + j / 3 + num_chunk * 100 + round * 1000);
        case Kind::random:
        default:
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<T>(rng >> 33U);
    }
}

/// @brief Is chunk @p c of @p arr in the motif state? (false for a zero_any array)
template <typename Arr>
static bool is_motif_chunk(const Arr& arr, long c)
{
    using ChunkT = typename Arr::ChunkType;
    if constexpr (has_motif_state_v<ChunkT>)
    {
        return ChunkT::is_motif_ptr(arr.get_chunk_list()[static_cast<std::size_t>(c)].values_load());
    }
    else
    {
        return false;
    }
}

/// @brief How many chunks of @p arr are compressed to a constant (no buffer, no motif).
template <typename Arr>
static long compressed_chunks(const Arr& arr, long nch, long csz)
{
    long n = 0;
    for (long c = 0; c < nch; ++c)
    {
        n += (arr.written(c * csz) || is_motif_chunk(arr, c)) ? 0 : 1;
    }
    return n;
}

/// @brief How many chunks of @p arr are in the motif state.
template <typename Arr>
static long motif_chunks(const Arr& arr, long nch)
{
    long n = 0;
    for (long c = 0; c < nch; ++c)
    {
        n += is_motif_chunk(arr, c) ? 1 : 0;
    }
    return n;
}

/// @brief Cases A to G on ArrayT<T, long> (see the file comment).
/// @return The compressed-chunk count after each step that writes (A, B, C).
template <template <typename, typename> class ArrayT, typename T>
static std::vector<long> round_trip(const char* vname, const char* tname, long nch)
{
    using Arr      = ArrayT<T, long>;
    const long csz = static_cast<long>(Arr(0).chunk_size());
    const long n   = nch * csz;
    println("--- {}::Array<{}>  {} chunks of {}", vname, tname, nch, csz);
    std::vector<long> census;

    std::uint64_t  rng = 12345;
    std::vector<T> ref(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = value_of<T>(kind_of(i / csz), i / csz, i % csz, 0, rng);
    }

    Arr arr(n);

    // A: the first write, then the pass.
    write_range(arr, ref, 0, n);
    expect_same("A  written", arr, ref);
    long zero_or_uniform = 0;
    for (long c = 0; c < nch; ++c)
    {
        zero_or_uniform += (kind_of(c) == Kind::zero || kind_of(c) == Kind::uniform) ? 1 : 0;
    }
    census.push_back(compressed_chunks(arr, nch, csz));
    expect_eq("A  compressed chunks (zero and uniform)", census.back(), zero_or_uniform);

    // B: every fifth element of the uniform and motif chunks.
    for (long i = 0; i < n; i += 5)
    {
        const Kind k = kind_of(i / csz);
        if (k == Kind::uniform || k == Kind::motif)
        {
            ref[static_cast<std::size_t>(i)] = static_cast<T>(ref[static_cast<std::size_t>(i)] + 1);
        }
    }
    write_range(arr, ref, 0, n);
    expect_same("B  every fifth element rewritten", arr, ref);
    census.push_back(compressed_chunks(arr, nch, csz));

    // C: the uniform chunks uniform again, at a new value.
    for (long i = 0; i < n; ++i)
    {
        if (kind_of(i / csz) == Kind::uniform)
        {
            ref[static_cast<std::size_t>(i)] = value_of<T>(Kind::uniform, i / csz, i % csz, 1, rng);
        }
    }
    write_range(arr, ref, 0, n);
    expect_same("C  uniform chunks at a new value", arr, ref);
    census.push_back(compressed_chunks(arr, nch, csz));
    println("  {:<44} {} of {}", "C  compressed chunks (zero and uniform)", census.back(), zero_or_uniform);
    if constexpr (std::is_same_v<Arr, pattern::Array<T, long>>)
    {
        println("  {:<44} {}", "C  motif chunks", motif_chunks(arr, nch));
    }

    // D: a copy.
    {
        Arr copy(arr);
        expect_same("D  copy", copy, ref);
        expect_true("D  copy == original", copy == arr);
        std::vector<T> ref2 = ref;
        ref2[1]             = static_cast<T>(ref2[1] + 1);
        write_range(copy, ref2, 1, 2);
        expect_true("D  copy != original after a write", !(copy == arr));
        expect_same("D  original untouched", arr, ref);

        // E: a move.
        Arr moved(std::move(copy));
        expect_same("E  moved", moved, ref2);
        expect_eq("E  source left empty", static_cast<long>(copy.size()), 0); // NOLINT(bugprone-use-after-move)
    }

    // F: shrink, then grow (std::vector semantics: the new elements are T{}).
    const long shrunk = n / 2 + 17;
    arr.resize(shrunk);
    ref.resize(static_cast<std::size_t>(shrunk));
    expect_same("F  shrunk", arr, ref);
    arr.resize(n + 5);
    ref.resize(static_cast<std::size_t>(n + 5));
    expect_same("F  grown", arr, ref);

    // G: assign(count, value).
    arr.assign(n / 3, static_cast<T>(42));
    ref.assign(static_cast<std::size_t>(n / 3), static_cast<T>(42));
    expect_same("G  assign", arr, ref);
    return census;
}

/// @brief Cases A to G on pattern::Array<T> and on zero_any::Array<T>: the
///        same values, and the same compressed-chunk counts.
template <typename T>
static void against_zero_any(const char* tname, long nch)
{
    const std::vector<long> pc = round_trip<pattern::Array, T>("pattern", tname, nch);
    const std::vector<long> zc = round_trip<zero_any::Array, T>("zero_any", tname, nch);
    expect_true("pattern compresses as zero_any at A, B", pc[0] == zc[0] && pc[1] == zc[1]);
    expect_true("pattern compresses at least as zero_any at C", pc[2] >= zc[2]);
}

/// @brief Write @p value into @p arr[@p i] for each (i, value) of @p writes,
///        from one thread of a parallel phase, then the between-region pass.
template <typename Arr, typename T>
static void write_some(Arr& arr, const std::vector<std::pair<long, T>>& writes)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        for (const auto& [i, value] : writes)
        {
            arr[i] = value;
        }
    }
    end_phase<Arr>();
}

/// @brief Cases I to L on pattern::Array<T> (see the file comment).
template <typename T>
static void motif_state(const char* tname, long nch)
{
    using Arr      = pattern::Array<T, long>;
    using Ch       = pattern::Chunk<T>;
    const long csz = static_cast<long>(Arr(0).chunk_size());
    const long n   = nch * csz;
    println("--- pattern::Array<{}>  motif chunks", tname);

    // Every third chunk: the same motif shifted by an offset of its own -- negative
    // ones for a signed T, and past the range of a short (the arithmetic wraps).
    const auto motif_chunk = [&](long c, long shift)
    {
        const T        off = static_cast<T>((c - nch / 2) * 1000003L + shift);
        std::vector<T> vals(static_cast<std::size_t>(csz));
        for (long j = 0; j < csz; ++j)
        {
            vals[static_cast<std::size_t>(j)] = Ch::wrapping_add(static_cast<T>((j * j) % 13 + j / 3), off);
        }
        return vals;
    };
    const auto install = [&](Arr& arr, std::vector<T>& ref, long first, long shift)
    {
        long installed = 0;
        for (long c = first; c < nch; c += 3)
        {
            const std::vector<T> vals = motif_chunk(c, shift);
            if (!arr.install_motif_for_test(static_cast<int>(c), vals.data()))
            {
                println("  install_motif_for_test refused chunk {}  <-- WRONG", c);
                ++fails;
            }
            std::copy(vals.begin(), vals.end(), ref.begin() + c * csz);
            ++installed;
        }
        return installed;
    };
    const auto materialised = [&](const Arr& arr, long first)
    {
        long count = 0;
        for (long c = first; c < nch; c += 3)
        {
            count += arr.written(c * csz) ? 1 : 0;
        }
        return count;
    };

    const std::size_t entries_before = pattern::MotifRegion<T>::entries();
    std::vector<T>    ref(static_cast<std::size_t>(n), T{});
    Arr               arr(n);
    const long        n_motif = install(arr, ref, 0, 0);
    {
        // All the installed chunks point at the same entry: one motif, shared.
        const T* shared = nullptr;
        long     others = 0;
        for (long c = 0; c < nch; c += 3)
        {
            const T* const p = arr.get_chunk_list()[static_cast<std::size_t>(c)].values_load();
            shared           = shared == nullptr ? p : shared;
            others += (p != shared) ? 1 : 0;
        }
        expect_eq("I  installed chunks on another motif than the first", others, 0);
        expect_true("I  at most one motif registered for them",
                    pattern::MotifRegion<T>::entries() <= entries_before + 1);
    }
    expect_eq("I  motif chunks materialised", materialised(arr, 0), 0);

    // I: every read path.
    expect_same("I  const operator[]", arr, ref);
    {
        long        bad = 0;
        std::size_t i   = 0;
        for (const T v : std::as_const(arr))
        {
            bad += (v != ref[i++]) ? 1 : 0;
        }
        expect_eq("I  iterator: elements wrong", bad, 0);
    }
    {
        std::vector<T> out(static_cast<std::size_t>(n));
        arr.read_range(0, n, out.data());
        expect_true("I  read_range", out == ref);
    }
    {
        std::vector<T> got;
        arr.for_each_value(0, n, [&got](const T& v) { got.push_back(v); });
        expect_true("I  for_each_value", got == ref);
    }
    {
        Arr copy(arr);
        expect_same("I  copy", copy, ref);
        expect_true("I  copy == original", copy == arr);
    }

    // J: one write in each motif chunk materialises it from its motif.
    {
        std::vector<std::pair<long, T>> writes;
        for (long c = 0; c < nch; c += 3)
        {
            const long i                     = c * csz + c % csz;
            ref[static_cast<std::size_t>(i)] = static_cast<T>(ref[static_cast<std::size_t>(i)] + 7);
            writes.emplace_back(i, ref[static_cast<std::size_t>(i)]);
        }
        write_some(arr, writes);
        expect_same("J  one write per motif chunk", arr, ref);
        expect_eq("J  motif chunks materialised", materialised(arr, 0), n_motif);
    }

    // K and L on fresh motif chunks: a read through the write proxy, then a
    // compound operation (its old value read through the proxy too).
    {
        std::vector<T> ref2(static_cast<std::size_t>(n), T{});
        Arr            arr2(n);
        const long     n_motif2 = install(arr2, ref2, 1, 11);
        long           bad      = 0;
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            for (long i = 0; i < n; ++i)
            {
                const T v = arr2[i]; // the write proxy's read (read_pure)
                bad += (v != ref2[static_cast<std::size_t>(i)]) ? 1 : 0;
            }
        }
        end_phase<Arr>();
        expect_eq("K  proxy reads wrong", bad, 0);
        expect_eq("K  motif chunks materialised (reads only)", materialised(arr2, 1), 0);
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            for (long c = 1; c < nch; c += 3)
            {
                arr2[c * csz + 3] += static_cast<T>(5);
            }
        }
        end_phase<Arr>();
        for (long c = 1; c < nch; c += 3)
        {
            auto& e = ref2[static_cast<std::size_t>(c * csz + 3)];
            e       = static_cast<T>(e + 5);
        }
        expect_same("L  compound += on motif chunks", arr2, ref2);
        expect_eq("L  motif chunks materialised", materialised(arr2, 1), n_motif2);
    }
}

/// @brief Cases M to P on pattern::Array<T> (see the file comment): the
///        compression to motifs, by the write path and the pass.
template <typename T>
static void motif_compression(const char* tname, long nch)
{
    using Arr      = pattern::Array<T, long>;
    using Ch       = pattern::Chunk<T>;
    using Region   = pattern::MotifRegion<T>;
    const long csz = static_cast<long>(Arr(0).chunk_size());
    const long n   = nch * csz;
    println("--- pattern::Array<{}>  compression to motifs", tname);

    // A connectivity: chunk c is a motif plus c times a stride (every chunk the same motif,
    // shifted); not the motif of the cases I to L.
    const auto connectivity = [&](long stride)
    {
        std::vector<T> ref(static_cast<std::size_t>(n));
        for (long i = 0; i < n; ++i)
        {
            const long c = i / csz;
            const long j = i % csz;
            ref[static_cast<std::size_t>(i)] =
                Ch::wrapping_add(static_cast<T>((j % 7) * 3 + j / 8 + 1), static_cast<T>(c * stride));
        }
        return ref;
    };
    const auto random = [&](std::uint64_t seed)
    {
        std::vector<T> ref(static_cast<std::size_t>(n));
        for (auto& v : ref)
        {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            v    = static_cast<T>(seed >> 33U);
        }
        return ref;
    };
    // A first sighting leaves its chunk materialised: one per motif, one per thread at most
    // when the threads' passes see it at once.
    const long first_sightings = CHUNK_PMP_MAX_THREADS;

    // M: the connectivity written in order.
    const pattern::MotifStats before = Region::stats();
    std::vector<T>            ref    = connectivity(1000);
    Arr                       arr(n);
    write_range(arr, ref, 0, n);
    expect_same("M  written in order", arr, ref);
    const long nm = motif_chunks(arr, nch);
    println("  {:<44} {} of {}", "M  motif chunks", nm, nch);
    expect_true("M  every chunk but the first sightings", nm >= nch - first_sightings && nm <= nch - 1);
    expect_eq("M  motifs registered", static_cast<long>(Region::stats().registered - before.registered), 1);
    {
        std::vector<T> out(static_cast<std::size_t>(n));
        arr.read_range(0, n, out.data());
        expect_true("M  read_range", out == ref);
    }

    // N: one write in every other chunk materialises the motif ones from their motif.
    {
        std::vector<std::pair<long, T>> writes;
        long                            hit = 0;
        for (long c = 0; c < nch; c += 2)
        {
            hit += is_motif_chunk(arr, c) ? 1 : 0;
            const long i                     = c * csz + csz / 2;
            ref[static_cast<std::size_t>(i)] = static_cast<T>(ref[static_cast<std::size_t>(i)] + 1);
            writes.emplace_back(i, ref[static_cast<std::size_t>(i)]);
        }
        write_some(arr, writes);
        expect_same("N  one write in every other chunk", arr, ref);
        expect_eq("N  motif chunks left", motif_chunks(arr, nch), nm - hit);
    }

    // O: random chunks: no motif is admitted (each is a first sighting).
    {
        const pattern::MotifStats before_o = Region::stats();
        const auto                ref_r    = random(777);
        Arr                       arr_r(n);
        write_range(arr_r, ref_r, 0, n);
        expect_same("O  random chunks", arr_r, ref_r);
        expect_eq("O  motif chunks", motif_chunks(arr_r, nch), 0);
        expect_eq("O  motifs registered", static_cast<long>(Region::stats().registered - before_o.registered), 0);
        // The same chunks rewritten unchanged: each motif seen again, by the same chunk.
        write_range(arr_r, ref_r, 0, n);
        expect_same("O  rewritten unchanged", arr_r, ref_r);
        expect_eq("O  motifs registered, rewritten",
                  static_cast<long>(Region::stats().registered - before_o.registered), 0);
    }

    // P: the reclaim. Every chunk rewritten at random: no chunk points at a motif any more,
    // so reclaim_second_values() frees every motif of T. Then the connectivity again.
    {
        ref = random(4242);
        write_range(arr, ref, 0, n);
        expect_same("P  rewritten at random", arr, ref);
        expect_eq("P  motif chunks after the rewrite", motif_chunks(arr, nch), 0);
        const pattern::MotifStats before_p = Region::stats();
        (void)Arr::reclaim_second_values();
        expect_true("P  motifs freed", Region::stats().freed > before_p.freed);
        expect_eq("P  motifs left", static_cast<long>(Region::entries()), 0);
        ref = connectivity(3);
        write_range(arr, ref, 0, n);
        expect_same("P  the connectivity again", arr, ref);
        const long nm2 = motif_chunks(arr, nch);
        expect_true("P  compressed again", nm2 >= nch - first_sightings && nm2 <= nch - 1);
    }

    // Q (dynamic geometry): the connectivity rechunked to twice, half, four times and
    // back to the chunk size: the values read back, and the rebuilt chunks -- written
    // by the refill, queued by the completion rule -- are compressed again to motifs of
    // the NEW size (a size class of their own in the region), once the pass has run.
    if constexpr (build_config::chunk_dynamic)
    {
        for (const long size : {2 * csz, csz / 2, 4 * csz, csz})
        {
            arr.rechunk_to(static_cast<int>(size));
            const long nch_q = (n + size - 1) / size;
            expect_same("Q  rechunked", arr, ref);
            end_phase<Arr>();
            const long nm_q = motif_chunks(arr, nch_q);
            println("  {:<44} {} of {} at {}", "Q  motif chunks", nm_q, nch_q, size);
            // A chunk of half the size holds one of two motifs; back at the chunk
            // size the motif is registered already: no first sighting at all.
            const long distinct = size < csz ? csz / size : 1;
            expect_true("Q  compressed at the new size", nm_q >= nch_q - distinct * first_sightings && nm_q <= nch_q);
            expect_same("Q  after the pass", arr, ref);
        }
        expect_true("Q  one size class per chunk size", Region::stats().sizes >= 3);
    }
}

/// @brief R (see the file comment): the chunk-size advisor on pattern::Array<T>
///        and on zero_any::Array<T> holding the same values: the same table, row
///        for row (a motif chunk's data is read as zero_any reads a buffer).
template <typename T>
static void advisor_agrees(const char* tname, long nch)
{
    using PArr     = pattern::Array<T, long>;
    using ZArr     = zero_any::Array<T, long>;
    const long csz = static_cast<long>(PArr(0).chunk_size());
    const long n   = nch * csz;
    println("--- pattern::Array<{}>  the advisor", tname);
    std::uint64_t  rng = 99;
    std::vector<T> ref(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = value_of<T>(kind_of(i / csz), i / csz, i % csz, 0, rng);
    }
    PArr parr(n);
    ZArr zarr(n);
    write_range(parr, ref, 0, n);
    write_range(zarr, ref, 0, n);
    expect_true("R  pattern has motif chunks", motif_chunks(parr, nch) > 0);
    const ChunkSizeReport pr   = ChunkSizeAdvisor<T>::analyze(parr);
    const ChunkSizeReport zr   = ChunkSizeAdvisor<T>::analyze(zarr);
    bool                  same = pr.table.size() == zr.table.size() && !pr.table.empty();
    for (std::size_t i = 0; same && i < pr.table.size(); ++i)
    {
        const auto& a = pr.table[i];
        const auto& b = zr.table[i];
        same = a.size == b.size && a.n_chunks == b.n_chunks && a.mat == b.mat && a.cst == b.cst && a.exact == b.exact;
        if (!same)
        {
            println("  R  row {}: pattern {} chunks {} mat {} cst; zero_any {} chunks {} mat {} cst", a.size,
                    a.n_chunks, a.mat, a.cst, b.n_chunks, b.mat, b.cst);
        }
    }
    println("  {:<44} {} rows", "R  advisor table", pr.table.size());
    expect_true("R  the same table as zero_any", same);
}

/// @brief H (see the file comment): the registry shared with zero_any.
static void shared_registry(long nch)
{
    using Za       = zero_any::Array<int, long>;
    using Pa       = pattern::Array<int, long>;
    using Registry = zero_any::Chunk<int>::Holder::VecHolder;
    const long csz = static_cast<long>(Za(0).chunk_size());
    const long n   = nch * csz;
    println("--- H  zero_any::Array<int> and pattern::Array<int>: one second-value registry");

    // Chunk j whole at base + j.
    const auto constants = [&](int base)
    {
        std::vector<int> ref(static_cast<std::size_t>(n));
        for (long i = 0; i < n; ++i)
        {
            ref[static_cast<std::size_t>(i)] = base + static_cast<int>(i / csz);
        }
        return ref;
    };
    const auto entries = [] { return static_cast<long>(Registry::count_entries()); };

    Za za(n);
    Pa pa(n);
    write_range(za, constants(1000), 0, n);
    write_range(pa, constants(5000), 0, n);
    expect_eq("both filled: registered values", entries(), 2 * nch);

    write_range(za, constants(2000), 0, n);
    expect_eq("zero_any refilled: reclaim on zero_any frees", static_cast<long>(Za::reclaim_second_values()), nch);
    expect_eq("registered values after", entries(), 2 * nch);

    // The identifiers just freed go to new values: the pattern array must
    // not have lost its own to them.
    write_range(za, constants(3000), 0, n);
    expect_same("pattern values after zero_any's reclaim", pa, constants(5000));
    expect_same("zero_any values", za, constants(3000));

    write_range(pa, constants(6000), 0, n);
    expect_eq("both refilled: reclaim on pattern frees", static_cast<long>(Pa::reclaim_second_values()), 2 * nch);
    write_range(pa, constants(7000), 0, n);
    expect_same("zero_any values after pattern's reclaim", za, constants(3000));
    expect_same("pattern values", pa, constants(7000));
}

int run(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    const long nch = argc > 1 ? shrink_array::arg_number<long>(argv[1], 64, "chunks") : 64;
    println("=== pattern_validation  chunks={}  threads={} ===", nch, CHUNK_PMP_MAX_THREADS);

    against_zero_any<int>("int", nch);
    against_zero_any<long>("long", nch);
    against_zero_any<unsigned>("unsigned", nch);
    against_zero_any<short>("short", nch);
    shared_registry(nch / 4 > 0 ? nch / 4 : 1);
    motif_state<int>("int", nch);
    motif_state<long>("long", nch);
    motif_state<unsigned>("unsigned", nch);
    motif_state<short>("short", nch);
    motif_compression<int>("int", nch);
    motif_compression<long>("long", nch);
    motif_compression<unsigned>("unsigned", nch);
    motif_compression<short>("short", nch);
    advisor_agrees<int>("int", nch);
    advisor_agrees<long>("long", nch);

    release_chunk_pools();
    destroy_block_chunk();

    println("\n{}", fails == 0 ? "=== PATTERN OK ===" : "=== PATTERN FAILURES ===");
    return fails == 0 ? 0 : 1;
}

} // namespace validation::pattern_validation
