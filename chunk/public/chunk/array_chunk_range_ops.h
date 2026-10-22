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

#pragma once

/**
 * @file array_chunk_range_ops.h
 * @brief Bulk range operations: one decision per CHUNK instead of per element.
 *
 * @details WHY THESE EXIST
 *
 *          Every write through `arr[i] = v` builds a Proxy, arms ptr_on_array /
 *          num_chunk, runs the compression-candidacy bookkeeping and only then
 *          stores. For a bulk fill that is paid chunk_size times for a decision
 *          that is the same across the chunk: a bulk copy through these
 *          functions is about an order of magnitude faster than through the
 *          proxy (the re-chunk refill uses them).
 *
 *          Arm the chunk once with real proxy writes, write straight into its
 *          buffer, then restore the invariants the bulk write bypassed.
 *
 *          NOT MEMBER FUNCTIONS, on purpose: ArrayChunk does not promise
 *          std::vector's interface (no data(), no stable element address), but
 *          a caller should not care which container it holds. Free functions
 *          with a generic fallback send a std::vector to std::fill and an
 *          ArrayChunk to the fast path.
 *
 *          Filling a range with the compressible constant (first_value) through
 *          operator[] materialises every chunk and waits for a pass to reclaim
 *          them. Here a fully covered chunk is set constant directly: no
 *          allocation, and a materialised chunk releases its buffer at once.
 *
 * @warning Serial contract. These are bulk operations over an array the caller
 *          owns; they do not synchronise with concurrent writers or with a
 *          compression pass running on the same array. Call them from a
 *          quiescent point -- the same contract clear_core() and rechunk_to()
 *          already carry.
 */

#include <array>       // std::array (the shared edge offsets)
#include <algorithm>   // std::min (the range clamps)
#include <cstddef>     // std::size_t, std::ptrdiff_t
#include <type_traits> // std::is_same_v, std::conditional_t
#include <memory> // std::unique_ptr, std::make_unique_for_overwrite (the span buffer of copy_range_mt / transform_range_mt)

#include "chunk/bit.h"                     // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/element_access.h" // internal::as_words: Bit chunks are words of bits
#include "chunk/internal/bit_words.h" // bits::count_ones, bits::scan_words, bits::fill_bit_range(_atomic), bits::store_bit_range_atomic
#include "chunk/internal/in_t.h"                       // InT<T>: the element snapshots
#include "chunk/PMP_CHUNK.h"                           // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h"          // set_thread_chunk_size
#include "chunk/internal/conditional_counter.h"        // CounterValueT
#include "chunk/config.h"                              // build_config::threaded, chunk_config
#include "chunk/internal/chunk_stats.h"                // increm_chunk_* (used directly, not via compression_manager.h)
#include "chunk/chunk_traits.h"                        // Chunk / Chunk2 / Chunk2A: the type traits below
#include "chunk/vector_range_ops.h"                    // internal::ChunkedArray + the std::vector fallback
#include "chunk/internal/write_compress_sync_policy.h" // fill_range_mt: the writer side of the Dekker handshake
#include "chunk/internal/relaxed_element.h"            // fill_range_mt: Chunk2A's shared elements

namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkT>
struct CompressionManager;

// assume_drained_t / assume_drained: in vector_range_ops.h, so the tag form
// of the call compiles against std::vector too.

namespace internal
{

/**
 * @brief Can this variant be raw-filled, or must every element go through the
 *        proxy?
 *
 * @details bool only: its chunks are bit-packed. Chunk2 takes the raw paths
 *          under three conditions:
 *           - the materialised test asks written(ind2): the index-less
 *             written() is bool-only, false for every other T;
 *           - queue_for_compression() loads the second value before queueing
 *             (set_info_on_chunk): Chunk2's candidate lists are keyed by it,
 *             and an entry in the list of T{} is drained by no pass (the
 *             teardown of a rechunk's temporary array then fails);
 *           - in counter mode (serial builds) a raw store recounts Chunk2's
 *             two per-chunk counters (resync_counters), as v1's zero count --
 *             otherwise runs of the second value are lost to chunks
 *             compressed to the first one.
 */
template <typename T>
inline constexpr bool raw_fillable_v = !is_bool_like_v<T>;

/**
 * @brief Which chunk variant does array type @p Arr hold?
 *
 * @details Dispatch on the TYPE, never on chunk_config::use_chunk /
 *          use_chunk2 / use_chunk2a: those say which variants the BUILD
 *          contains, and a build may contain several (CHUNK_VARIANT_V1=1 with
 *          CHUNK_VARIANT_V2=1): testing the flags sends an ArrayChunk2 down
 *          v1's paths and an ArrayChunk down Chunk2's.
 */
template <typename Arr>
inline constexpr bool is_v1_v = std::is_same_v<typename Arr::ChunkType, Chunk<typename Arr::value_type>>;

/// @brief True when @p Arr holds Chunk2 chunks (see is_v1_v: dispatch on the type).
template <typename Arr>
inline constexpr bool is_v2_v = std::is_same_v<typename Arr::ChunkType, Chunk2<typename Arr::value_type>>;

/// @brief True when @p Arr holds Chunk2A chunks (see is_v1_v: dispatch on the type).
template <typename Arr>
inline constexpr bool is_v2a_v = std::is_same_v<typename Arr::ChunkType, Chunk2A<typename Arr::value_type>>;

/** @brief Chunk2 on a non-bool T: the variant with a second value (and written(ind)). */
template <typename Arr>
inline constexpr bool has_second_value_v =
    is_v2_v<Arr> && !is_bool_like_v<typename Arr::value_type>; // bool, Bit: no index

/** @brief Only v1 in counter mode carries a per-chunk zero count to restore. */
template <typename Arr>
inline constexpr bool track_zeros_v = is_v1_v<Arr> && chunk_config::use_compression_with_counter;

/** @brief Chunk2 in counter mode carries two: first-value and second-value counts. */
template <typename Arr>
inline constexpr bool track_chunk2_counts_v = is_v2_v<Arr> && chunk_config::use_compression_with_counter;

/**
 * @brief Recount the per-chunk element counters after a raw store into
 *        [c_begin, c_begin + S), which bypassed them.
 *
 * @details In counter mode (use_compression_with_counter, i.e. without fast
 *          block compression -- the serial builds) the compression of a chunk
 *          is decided by counters the proxy maintains INCREMENTALLY: v1's
 *          nb_val_eq_zero, Chunk2's nb_val_eq_first_value and
 *          nb_val_eq_second_value. A raw store moves none of them, and nothing
 *          detects the drift: the full tests compare to chunk_size() exactly,
 *          so a chunk that is NOT all constant compresses when a count lands
 *          there, and every read then returns the constant. Recount, don't
 *          adjust (as rechunk_refill). Chunk2's second-value count is taken
 *          against the array's second value, as Proxy2 counts it.
 */
template <typename Arr, typename Chunk, typename T>
void resync_counters(Arr& arr, Chunk& ch, std::size_t c_begin, const T* dst, unsigned int s)
{
    if constexpr (track_zeros_v<Arr>)
    {
        CounterValueT nz = 0;
        for (unsigned int k = 0; k < s; ++k)
        {
            if (dst[k] == static_cast<T>(Chunk::first_value))
            {
                ++nz;
            }
        }
        arr.element_proxy(static_cast<Arr::IndType>(c_begin)).resync_nb_val_eq_zero(nz);
    }
    else if constexpr (track_chunk2_counts_v<Arr> && requires { arr.get_second_value(); })
    {
        const T       sv = static_cast<T>(arr.get_second_value());
        CounterValueT n1 = 0;
        CounterValueT n2 = 0;
        for (unsigned int k = 0; k < s; ++k)
        {
            if (dst[k] == static_cast<T>(Chunk::first_value))
            {
                ++n1;
            }
            if (dst[k] == sv)
            {
                ++n2;
            }
        }
        ch.set_nb_val_eq_first_value(n1);
        ch.set_nb_val_eq_second_value(n2);
    }
}

/**
 * @brief Counter mode, Chunk2 and v1: fold a chunk the bulk store left uniform.
 *
 * @details In a serial build the compression of a Chunk2 chunk is the element
 *          write that brings one of its counters to S (manage_full_*): there
 *          is no pass, and queue_for_compression() does nothing. A raw store
 *          bypasses that write, so after resync_counters() a chunk whose count
 *          is S must be folded here, as that last write would have done --
 *          what packed_chunk_store() does for bit-packed chunks. Without it,
 *          every chunk a re-chunk rebuilds from second-value sources stays
 *          materialised (rechunk_refill()'s own fold-back is fast-block only).
 *          The second value is folded only when it is designated (differs from
 *          first_value).
 * @return true when the chunk was folded: its buffer is gone, the caller must
 *         not read it again.
 */
template <typename Arr, typename Chunk>
[[nodiscard]] bool fold_if_full(Arr& arr, Chunk& ch, std::size_t c_begin, unsigned int s)
{
    if constexpr (track_chunk2_counts_v<Arr> && has_second_value_v<Arr> && requires { arr.get_second_value(); })
    {
        using IndType     = decltype(arr.size());
        using T           = typename Arr::value_type;
        const bool first  = ch.is_full_first_value(s);
        const bool second = !first && ch.is_full_second_value(s) &&
                            !(static_cast<T>(arr.get_second_value()) == static_cast<T>(Chunk::first_value));
        if (!first && !second)
        {
            return false;
        }
        auto proxy = arr.element_proxy(static_cast<IndType>(c_begin));
        // element_proxy() leaves the proxy's second-value index unset; the
        // write path loads it first (assign_impl), and manage_full_*() pass it
        // to clear_st(): a stale index points the chunk at ANOTHER value's
        // constant array, which the next write then scribbles over.
        proxy.set_info_second_value_ind();
        if constexpr (build_config::chunk_dynamic)
        {
            // The ambient geometry and allocator set the free routes by, as the
            // write path pins them (Proxy2::execute_simple_operation).
            shrink_array::set_thread_chunk_size(s);
            shrink_array::set_thread_alloc_key(proxy.alloc_key());
        }
        if (first)
        {
            proxy.manage_full_first_value();
        }
        else
        {
            proxy.manage_full_second_value();
        }
        return true;
    }
    else if constexpr (track_zeros_v<Arr>)
    {
        // v1: one constant, zero (resync_counters() set the count).
        (void)ch;
        return arr.element_proxy(static_cast<decltype(arr.size())>(c_begin)).fold_if_full_zero();
    }
    else
    {
        (void)arr;
        (void)ch;
        (void)c_begin;
        (void)s;
    }
    return false;
}

/**
 * @brief Can the chunk whose buffer is @p dst (@p s elements) be uniform?
 * @details NECESSARY condition only -- first, middle and last element equal.
 *          A uniform chunk always passes; a chunk that fails cannot be
 *          compressed, so queueing it would only cost the next pass a full
 *          scan (and, in a parallel run, push the lists to "dangerous" and the
 *          scans onto the writers' slow path).
 */
template <typename T>
[[nodiscard]] bool may_be_uniform(const T* dst, unsigned int s) noexcept
{
    return s == 0 || (dst[0] == dst[s / 2] && dst[0] == dst[s - 1]);
}

/**
 * @brief may_be_uniform() on a buffer other threads are writing (fill_range_mt).
 * @details Same three elements, read as relaxed atomics: other writers of the
 *          chunk store them concurrently (relaxed_element.h).
 */
template <typename T>
[[nodiscard]] bool may_be_uniform_relaxed(const T* dst, unsigned int s) noexcept
{
    if (s == 0)
    {
        return true;
    }
    const T v0 = load_element_relaxed(dst);
    return v0 == load_element_relaxed(dst + (s / 2)) && v0 == load_element_relaxed(dst + s - 1);
}

/**
 * @brief Store @p v into offsets [@p o_lo, @p o_hi) of a chunk buffer other
 *        threads are writing and reading (fill_range_mt, Chunk2A).
 * @details Only five offsets are ever read by a thread that does not own them:
 *          0, 1, S-2, S-1 (the marking filter, Proxy2A::may_become_uniform)
 *          and 0, S/2, S-1 (may_be_uniform_relaxed). Those are stored as
 *          relaxed atomics; the runs between them stay plain stores, so the
 *          bulk of the fill still vectorises.
 */
template <typename T>
void fill_span_relaxed_edges(T* dst, std::size_t o_lo, std::size_t o_hi, std::size_t s, T v) noexcept
{
    // Ascending for S >= 4; for a smaller S the out-of-order or duplicate
    // entries are skipped by `edge < o`, and S - 2 wrapping for S == 1 by `edge >= o_hi`.
    const std::array<std::size_t, 5> shared = {0, 1, s / 2, s - 2, s - 1};
    std::size_t                      o      = o_lo;
    for (const std::size_t edge : shared)
    {
        if (edge < o)
        {
            continue;
        }
        if (edge >= o_hi)
        {
            break;
        }
        for (; o < edge; ++o)
        {
            dst[o] = v;
        }
        store_element_relaxed(dst + edge, v);
        o = edge + 1;
    }
    for (; o < o_hi; ++o)
    {
        dst[o] = v;
    }
}

/// @brief The array's second-value index for Chunk2 (non-bool), -1 otherwise.
///        Read once per range call: it is a registry lookup.
template <typename Arr>
[[nodiscard]] int second_value_index(const Arr& arr) noexcept
{
    if constexpr (has_second_value_v<Arr>)
    {
        return arr.get_ind_second_value();
    }
    else
    {
        return -1;
    }
}

/**
 * @brief Does this chunk own a buffer the bulk store may write into?
 *
 * @details NOT `!is_cst_val1()`. A Chunk2A parked on its SECOND constant is
 *          not cst_val1 either, and its values pointer is a second-value slot,
 *          not a buffer: stores through it are lost, and the chunk keeps
 *          reading its constant.
 *          written() is each variant's own "materialised" test -- v1: not
 *          cst_val1; Chunk2A: neither constant. Chunk2's index-less written()
 *          is bool-only (false for every other T), so it is asked with @p ind2.
 */
template <typename Arr, typename Chunk>
[[nodiscard]] bool materialised(const Arr& /*arr*/, const Chunk& ch, [[maybe_unused]] int ind2) noexcept
{
    if constexpr (has_second_value_v<Arr>)
    {
        return ch.written(ind2);
    }
    else
    {
        return ch.written();
    }
}

/**
 * @brief Queue the chunk holding element @p i for the next compression pass.
 *
 * @details The raw stores of the bulk paths below go straight into a
 *          materialised chunk's buffer, past the write path and its marking:
 *          a chunk they leave constant would otherwise stay materialised. This
 *          does what the span's last write would have done: the flag's CAS
 *          and, the first time, one entry in the caller's candidate list; the
 *          pass drops the chunk if it is not constant. Fast-block builds,
 *          and Chunk2A in every build: it has no counters, its serial build
 *          compresses through the same list and pass. The serial v1 / Chunk2
 *          counters are resynced by the caller instead (fold_if_full, when
 *          present). In a serial build the proxy is armed by
 *          set_info_on_chunk(): set_num_tab_ptr_is_inside_chunk() is a
 *          fast-block member.
 */
template <typename Arr>
void queue_for_compression(Arr& arr, std::size_t i)
{
    // Chunk2A: a candidate list in serial builds too (no counters there).
    if constexpr (chunk_config::use_fast_block_compression || is_v2a_v<Arr>)
    {
        // Chunk2: candidate lists are keyed by second value and exist only for
        // arrays that called set_second_value() -- skip the others, as the
        // element path's pre-check does (Proxy2::has_candidate_list).
        if constexpr (has_second_value_v<Arr> &&
                      requires { Arr::ProxyType::has_candidate_list(arr.get_second_value()); })
        {
            if (!Arr::ProxyType::has_candidate_list(arr.get_second_value()))
            {
                return;
            }
        }
        auto proxy = arr.element_proxy(static_cast<decltype(arr.size())>(i));
        // Not for Proxy2: it also has set_num_tab_ptr_is_inside_chunk() (from
        // the shared base), but that does not load the second value -- the key
        // of Chunk2's candidate list -- and the chunk would land in the list of
        // T{}, which no pass drains. set_info_on_chunk() loads (second_value,
        // ind, num_tab) and the flag pointer.
        constexpr bool is_proxy2 = std::is_same_v<typename Arr::ProxyType, Proxy2<typename Arr::value_type>>;
        if constexpr (!is_proxy2 && requires { proxy.set_num_tab_ptr_is_inside_chunk(); })
        {
            proxy.set_num_tab_ptr_is_inside_chunk(); // Proxy2A
        }
        else
        {
            proxy.set_info_on_chunk(); // Proxy2, Proxy (v1)
        }
        proxy.add_to_compression_list();
    }
}

/**
 * @brief ArrayChunkBits (T = Bit): store [@p lo, @p hi) of one chunk, bits
 *        packed 64 per word.
 *
 * @details Same shape as the byte path: real proxy writes (@p value_at) until
 *          the chunk materialises -- which allocates its buffer and registers
 *          the bookkeeping -- then the rest in one go through @p store_bits
 *          (word-wide stores, masked edges), under fill_range's serial
 *          contract. Then what the bypassed element writes would have done:
 *          the counters recounted (one popcount, serial builds), and the chunk
 *          queued if it came out uniform -- an exact word scan, not a
 *          three-element guess.
 */
template <typename Arr, typename Chunk, typename ValueAt, typename StoreBits>
void packed_chunk_store(Arr&         arr,
                        Chunk&       ch,
                        std::size_t  c_begin,
                        std::size_t  lo,
                        std::size_t  hi,
                        unsigned int s,
                        int          ind2,
                        ValueAt&&    value_at,
                        StoreBits&&  store_bits)
{
    using IndType = decltype(arr.size());
    std::size_t i = lo;
    while (i < hi && !materialised(arr, ch, ind2))
    {
        arr[static_cast<IndType>(i)] = value_at(i);
        ++i;
    }
    if (!materialised(arr, ch, ind2))
    {
        return; // the whole span was the chunk's constant
    }

    bits::WordT* w = shrink_array::internal::as_words(ch.values_load());
    store_bits(w, i - c_begin, hi - c_begin);

    if constexpr (track_chunk2_counts_v<Arr>)
    {
        // Counter mode (serial builds): the element write that brings a count
        // to S compresses the chunk on the spot (manage_full_*), and there is
        // no pass to queue for -- so do what that last write would have done.
        const auto ones = static_cast<CounterValueT>(bits::count_ones(w, s));
        ch.set_nb_val_eq_first_value(static_cast<CounterValueT>(s) - ones);
        ch.set_nb_val_eq_second_value(ones);
        if (ones == static_cast<CounterValueT>(s))
        {
            arr.element_proxy(static_cast<IndType>(c_begin)).manage_full_second_value();
        }
        else if (ones == 0)
        {
            arr.element_proxy(static_cast<IndType>(c_begin)).manage_full_first_value();
        }
    }
    else if (bits::scan_words(w, s) != bits::Uniform::mixed)
    {
        queue_for_compression(arr, c_begin);
    }
}

} // namespace internal

// ─────────────────────────────────────────────────────────────────────────────
// ArrayChunk overloads
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Bulk fill, one decision per chunk.
 *
 * @details Per chunk in the range:
 *          - fully covered and @p v is the compressible constant: make the
 *            chunk constant outright. A materialised chunk releases its buffer;
 *            an already-constant one costs nothing. No allocation either way.
 *          - otherwise: arm the chunk with real proxy writes until it
 *            materialises, then store the remainder straight into its buffer
 *            and restore the zero counter the bulk store bypassed.
 *
 *          bool (one byte per element, its second-value state in the proxy)
 *          takes the per-element path throughout -- still correct, just
 *          without the speedup. See internal::raw_fillable_v. ArrayChunkBits
 *          (Bit) stores 64 elements per word: internal::packed_chunk_store.
 *
 * @param arr   target array
 * @param first first index, inclusive (clamped to size())
 * @param last  last index, exclusive (clamped to size())
 * @param v     value to store
 * @param drain flush the array's deferred writes first (true); pass false, or
 *              use the assume_drained tag, when the caller already did
 */
template <internal::ChunkedArray Arr, typename U>
void fill_range(Arr& arr, std::size_t first, std::size_t last, const U& v, bool drain = true)
{
    using T         = Arr::value_type;
    using IndType   = decltype(arr.size());
    using ChunkType = Arr::ChunkType;

    const auto size = static_cast<std::size_t>(arr.size());
    if (first >= last || first >= size)
    {
        return;
    }
    last = std::min(last, size);

    // Drain this array's pending deferred writes FIRST: the bulk stores below
    // bypass the deferred machinery, and an op still queued on one of these
    // chunks would replay AFTER them and restore the old value. Forced: the
    // pending-ops counter can read 0 while ops are still queued, so do not
    // rely on the usual gate here.
    if (drain)
    {
        arr.flush_pending_deferred_ops(/*force=*/true);
    }

    const unsigned int s     = arr.chunk_size();
    const InT<T>       value = static_cast<T>(v);
    const int          ind2  = internal::second_value_index(arr); // Chunk2's written(ind)

    // Pin the ambient geometry once for the whole operation: every
    // materialisation below must size its buffer from THIS array's chunk_size,
    // not from whichever array wrote last (see Proxy::create_value).
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(s);
    }

    const bool value_is_cst1 = (value == static_cast<T>(ChunkType::first_value));

    const std::size_t first_chunk = first / s;
    const std::size_t last_chunk  = (last - 1) / s;

    for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
    {
        const std::size_t c_begin = ci * s;
        const std::size_t c_end   = c_begin + s;
        const std::size_t lo      = first > c_begin ? first : c_begin;
        const std::size_t hi      = last < c_end ? last : c_end;
        const bool        whole   = (lo == c_begin) && (hi == c_end);

        auto& ch = arr.chunk_ref(ci);

        // ── Whole chunk set to the compressible constant: no storage needed ──
        //
        // v1 only. Chunk2A's clear_st also takes the compression state, which
        // this loop must not synthesise (as rechunk_refill); the other variants
        // take the paths below, without the free-the-buffer shortcut.
        // Require BOTH spellings: Chunk2's clear_st(int num_compress_value, int
        // ind) accepts (thr, n) by conversion and would take a THREAD RANK as
        // the constant to compress to. Only v1 offers both.
        constexpr bool can_release = requires(ChunkType& chunk, int thr, unsigned int count) {
            chunk.clear_st(thr, count);
        } && requires(ChunkType& chunk) { chunk.clear_st(); };

        if constexpr (can_release)
        {
            if (whole && value_is_cst1)
            {
                if (!ch.is_cst_val1())
                {
                    // CHUNK_STATS: a compression reached without a scan -- count
                    // its (trivially successful) attempt with it, so the
                    // attempts/compress yield stays a yield.
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_compress_attempt<ChunkType>();
                    }
                    if constexpr (build_config::threaded)
                    {
                        ch.clear_st(get_thr_rank_fast(), s); // frees the buffer, marks cst1
                    }
                    else
                    {
                        ch.clear_st();
                    }
                }
                continue;
            }
        }

        // ── ArrayChunkBits: word-wide stores ─────────────────────────────────
        if constexpr (is_packed_bit_v<T>)
        {
            // Whole chunk already at that constant: nothing to write.
            if (whole && (value_is_cst1 ? ch.is_cst_val1() : ch.is_cst_val2()))
            {
                continue;
            }
            const bool bit = static_cast<bool>(value);
            internal::packed_chunk_store(
                arr, ch, c_begin, lo, hi, s, ind2, [&](std::size_t) { return static_cast<T>(value); },
                [bit](bits::WordT* w, std::size_t a, std::size_t b) { bits::fill_bit_range(w, a, b, bit); });
        }
        // ── Variants whose second-value state lives in the proxy ─────────────
        else if constexpr (!internal::raw_fillable_v<T>)
        {
            for (std::size_t i = lo; i < hi; ++i)
            {
                arr[static_cast<IndType>(i)] = value;
            }
        }
        else
        {
            // ── Arm, then bulk-store ────────────────────────────────────────
            // Real proxy writes until the chunk materialises: that allocates the
            // buffer and registers the compression bookkeeping. If the loop
            // walks the whole range without materialising, the chunk is
            // constant and nothing is left to do. "Materialises" is written(),
            // not !is_cst_val1(): see internal::materialised.
            std::size_t i = lo;
            while (i < hi && !internal::materialised(arr, ch, ind2))
            {
                arr[static_cast<IndType>(i)] = value;
                ++i;
            }
            if (!internal::materialised(arr, ch, ind2))
            {
                continue; // still constant: the range held only that constant
            }

            T* dst = ch.values_load();
            for (std::size_t k = i; k < hi; ++k)
            {
                dst[k - c_begin] = value;
            }

            // Restore the per-chunk counters the bulk store bypassed.
            internal::resync_counters(arr, ch, c_begin, dst, s);
            // Serial builds: the chunk is constant now -> folded here (no pass).
            if (internal::fold_if_full(arr, ch, c_begin, s))
            {
                continue;
            }

            // The chunk may be constant now: a fill of first_value, or on
            // chunk2a of any value (a uniform run is its second constant).
            bool compressible_value = value_is_cst1 || internal::is_v2a_v<Arr>;
            if constexpr (internal::is_v2_v<Arr> && requires { arr.get_second_value(); })
            {
                compressible_value = compressible_value || value == static_cast<T>(arr.get_second_value());
            }
            if (compressible_value && internal::may_be_uniform(dst, s))
            {
                internal::queue_for_compression(arr, c_begin);
            }
        }
    }
}

/**
 * @brief Bulk copy into [first, last) from a source iterator.
 *
 * @details Same per-chunk strategy as fill_range, minus the constant shortcut:
 *          the source values are not known to be uniform, so a chunk is set
 *          constant only when the arming loop finds every value to be the
 *          compressible constant.
 *
 * @param arr       target array
 * @param first     first target index, inclusive (clamped to size())
 * @param last      last target index, exclusive (clamped to size())
 * @param src_first source iterator; must not alias @p arr
 * @param drain     flush the array's deferred writes first (true); pass false,
 *                  or use the assume_drained tag, when the caller already did
 */
template <internal::ChunkedArray Arr, typename It>
void copy_range(Arr& arr, std::size_t first, std::size_t last, It src_first, bool drain = true)
{
    using T       = Arr::value_type;
    using IndType = decltype(arr.size());

    const auto size = static_cast<std::size_t>(arr.size());
    if (first >= last || first >= size)
    {
        return;
    }
    last = std::min(last, size);

    // Drain first, forced: see fill_range.
    if (drain)
    {
        arr.flush_pending_deferred_ops(/*force=*/true);
    }

    const unsigned int s    = arr.chunk_size();
    const int          ind2 = internal::second_value_index(arr); // Chunk2's written(ind)

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(s);
    }

    const std::size_t first_chunk = first / s;
    const std::size_t last_chunk  = (last - 1) / s;

    for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
    {
        const std::size_t c_begin = ci * s;
        const std::size_t c_end   = c_begin + s;
        const std::size_t lo      = first > c_begin ? first : c_begin;
        const std::size_t hi      = last < c_end ? last : c_end;

        auto& ch = arr.chunk_ref(ci);

        if constexpr (is_packed_bit_v<T>)
        {
            // ArrayChunkBits: the source read once per element, stored 64 at a time.
            const auto src_at = [&](std::size_t i)
            { return static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(i - first))); };
            internal::packed_chunk_store(arr, ch, c_begin, lo, hi, s, ind2, src_at,
                                         [&](bits::WordT* w, std::size_t a, std::size_t b)
                                         {
                                             bits::store_bit_range(w, a, b, [&](std::size_t bi)
                                                                   { return static_cast<bool>(src_at(c_begin + bi)); });
                                         });
        }
        else if constexpr (!internal::raw_fillable_v<T>)
        {
            for (std::size_t i = lo; i < hi; ++i)
            {
                arr[static_cast<IndType>(i)] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(i - first)));
            }
        }
        else
        {
            std::size_t i = lo;
            while (i < hi && !internal::materialised(arr, ch, ind2))
            {
                arr[static_cast<IndType>(i)] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(i - first)));
                ++i;
            }
            if (!internal::materialised(arr, ch, ind2))
            {
                continue; // the whole span was the chunk's constant
            }

            T* dst = ch.values_load();
            for (std::size_t k = i; k < hi; ++k)
            {
                dst[k - c_begin] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(k - first)));
            }

            internal::resync_counters(arr, ch, c_begin, dst, s); // as fill_range
            if (internal::fold_if_full(arr, ch, c_begin, s))     // serial builds: as fill_range
            {
                continue;
            }

            // The source is not known to be uniform: the pass decides -- for
            // the chunks that can be.
            if (internal::may_be_uniform(dst, s))
            {
                internal::queue_for_compression(arr, c_begin);
            }
        }
    }
}

/// @brief fill_range() on an array whose deferred queue the caller has already
///        drained (the assume_drained tag): no flush first.
template <internal::ChunkedArray Arr, typename U>
void fill_range(Arr& arr, std::size_t first, std::size_t last, const U& v, assume_drained_t /*tag*/)
{
    fill_range(arr, first, last, v, /*drain=*/false);
}

/// @brief copy_range() on an array whose deferred queue the caller has already
///        drained (the assume_drained tag): no flush first.
template <internal::ChunkedArray Arr, typename It>
void copy_range(Arr& arr, std::size_t first, std::size_t last, It src_first, assume_drained_t /*tag*/)
{
    copy_range(arr, first, last, src_first, /*drain=*/false);
}


// ─────────────────────────────────────────────────────────────────────────────
// fill_range_mt / copy_range_mt / transform_range_mt: the parallel forms --
// one write SESSION per chunk
// ─────────────────────────────────────────────────────────────────────────────

namespace internal
{

/**
 * @brief fill_span_relaxed_edges() for values that differ: offset o gets
 *        @p at(o). Same five shared offsets stored as relaxed atomics, the
 *        runs between them plain.
 */
template <typename T, typename At>
void store_span_relaxed_edges(T* dst, std::size_t o_lo, std::size_t o_hi, std::size_t s, At&& at) noexcept
{
    const std::array<std::size_t, 5> shared = {0, 1, s / 2, s - 2, s - 1};
    std::size_t                      o      = o_lo;
    for (const std::size_t edge : shared)
    {
        if (edge < o)
        {
            continue;
        }
        if (edge >= o_hi)
        {
            break;
        }
        for (; o < edge; ++o)
        {
            dst[o] = at(o);
        }
        store_element_relaxed(dst + edge, at(edge));
        o = edge + 1;
    }
    for (; o < o_hi; ++o)
    {
        dst[o] = at(o);
    }
}

/**
 * @brief The engine of fill_range_mt, copy_range_mt and transform_range_mt:
 *        element i of [first, last) gets @p value_at(i), with one write
 *        session per chunk span instead of one per element.
 *
 * @details Per chunk span:
 *            1. pending deferred ops on the chunk -> the span goes through the
 *               element path (it syncs foreign ops, erases/excludes own ones);
 *            2. a compressor on the chunk -> wait it out (bounded, before
 *               announcing, like the element path);
 *            3. announce this thread as a writer of the chunk (Dekker, the
 *               write side of WriteCompressSyncPolicy);
 *            4. re-check (compressor, pending ops, still materialised); any
 *               doubt -> retire and take the element path for the span;
 *            5. raw stores into the chunk's buffer;
 *            6. retire;
 *            7. queue the chunk as a compression candidate when it can be
 *               uniform (the pass drops it if it is not).
 *          A chunk that is not materialised is first materialised through the
 *          element path, outside any session (a proxy write inside a session
 *          would re-announce and then clear this thread's writer flag).
 *          A chunk is never RELEASED here: another thread may be writing other
 *          elements of the same chunk, into that very buffer.
 *
 *          @p value_at is the CALLER's code (an iterator, a user operation):
 *          it never runs inside a session, where a read of a chunked array or
 *          a long computation would hold the compressor off, and a write
 *          through a proxy would clear this thread's writer flag. Unless
 *          @p Constant, a span's values are computed into a buffer first, and
 *          the session only stores them. Each value_at(i) is called once.
 *
 *          The queueing rule only looks at the span's own values (reading the
 *          elements other threads write is a race, except Chunk2A's relaxed
 *          ones): v1 compresses only to first_value, Chunk2 to first_value or
 *          the array's second value, Chunk2A aggregates only a span covering
 *          the whole chunk, uniform. A fill is a uniform span of v, so the
 *          rule reduces to fill_range_mt's.
 *
 * @tparam Constant value_at returns one value for every index (fill_range_mt):
 *                  no buffer
 * @param arr       the array written
 * @param first     first index of the range
 * @param last      one past its last index
 * @param value_at  called as value_at(i), i the global index; returns a value
 *                  convertible to the element type
 */
template <bool Constant, ChunkedArray Arr, typename ValueAt>
void store_range_mt(Arr& arr, std::size_t first, std::size_t last, ValueAt&& value_at)
{
    using T         = Arr::value_type;
    using IndType   = decltype(arr.size());
    using ChunkType = Arr::ChunkType;

    const auto size = static_cast<std::size_t>(arr.size());
    if (first >= last || first >= size)
    {
        return;
    }
    last = std::min(last, size);

    constexpr bool session = (is_v2a_v<Arr> || is_v2_v<Arr> || is_v1_v<Arr>) && build_config::threaded &&
                             chunk_config::use_fast_block_compression && (raw_fillable_v<T> || is_packed_bit_v<T>);
    if constexpr (!session)
    {
        for (std::size_t i = first; i < last; ++i)
        {
            arr[static_cast<IndType>(i)] = static_cast<T>(value_at(i));
        }
    }
    else
    {
        using Policy = WriteCompressSyncPolicy<typename Arr::ProxyType>;
        using Mgr    = CompressionManager<T, ChunkType>;
        // The span's values, computed before the session (not for a constant).
        // bool, not Bit, for ArrayChunkBits: a Bit is a packed reference type.
        using BufValueT = std::conditional_t<is_packed_bit_v<T>, bool, T>;

        const unsigned int s    = arr.chunk_size();
        const int          ind2 = second_value_index(arr); // Chunk2's written(ind)
        if constexpr (build_config::chunk_dynamic)
        {
            set_thread_chunk_size(s);
        }
        const int thr = get_thr_rank_fast();

        // One buffer per thread and element type, reused across calls: callers
        // cut a range into small spans, and an allocation per call (malloc
        // contention, zeroing a chunk's worth) costs more than the copy. Not a
        // std::vector: std::vector<bool> (ArrayChunkBits) packs its bits and
        // has no data(). Not zeroed: a span writes each slot before reading it.
        BufValueT* buf = nullptr;
        if constexpr (!Constant)
        {
            thread_local std::unique_ptr<BufValueT[]> tl_buf;
            thread_local unsigned int                 tl_cap = 0;
            if (tl_cap < s)
            {
                tl_buf = std::make_unique_for_overwrite<BufValueT[]>(s);
                tl_cap = s;
            }
            buf = tl_buf.get();
        }

        const std::size_t first_chunk = first / s;
        const std::size_t last_chunk  = (last - 1) / s;

        for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
        {
            const std::size_t c_begin = ci * s;
            const std::size_t c_end   = c_begin + s;
            const std::size_t lo      = first > c_begin ? first : c_begin;
            const std::size_t hi      = last < c_end ? last : c_end;

            auto& ch = arr.chunk_ref(ci);

            // Materialise through the element path, outside any session.
            std::size_t i = lo;
            while (i < hi && !materialised(arr, ch, ind2))
            {
                arr[static_cast<IndType>(i)] = static_cast<T>(value_at(i));
                ++i;
            }
            if (i == hi)
            {
                continue;
            }

            // The span's values (offsets into the chunk), and whether they are
            // all one value: once, before the session.
            bool      same = true;
            BufValueT v0{};
            if constexpr (Constant)
            {
                v0 = static_cast<BufValueT>(static_cast<T>(value_at(i)));
            }
            else
            {
                v0               = static_cast<BufValueT>(static_cast<T>(value_at(i)));
                buf[i - c_begin] = v0;
                for (std::size_t k = i + 1; k < hi; ++k)
                {
                    const auto v     = static_cast<BufValueT>(static_cast<T>(value_at(k)));
                    buf[k - c_begin] = v;
                    same             = same && (v == v0);
                }
            }
            const auto at = [&](std::size_t o) -> BufValueT
            {
                if constexpr (Constant)
                {
                    static_cast<void>(o);
                    return v0;
                }
                else
                {
                    return buf[o];
                }
            };

            bool done    = false;
            bool uniform = false;  // may the chunk be uniform? (step 7)
            if (!ch.pending_any()) // 1. deferred ops on the chunk: element path
            {
                if (Mgr::must_defer_write(ch)) // 2. a compressor on it: wait (bounded)
                {
                    Mgr::wait_out_compressor(ch);
                }

                Policy::announce_write(thr, &ch);                                                   // 3.
                if (!Mgr::must_defer_write(ch) && !ch.pending_any() && materialised(arr, ch, ind2)) // 4.
                {
                    T* dst = ch.values_load(); // 5.
                    if constexpr (is_packed_bit_v<T>)
                    {
                        // ArrayChunkBits: other threads write the other bits of
                        // the span's edge words -- atomic updates there, a
                        // store for the words the span owns whole. false and
                        // true are both compressible.
                        if constexpr (Constant)
                        {
                            bits::fill_bit_range_atomic(as_words(dst), i - c_begin, hi - c_begin, v0);
                        }
                        else
                        {
                            bits::store_bit_range_atomic(as_words(dst), i - c_begin, hi - c_begin, at);
                        }
                        uniform = same;
                    }
                    else if constexpr (relaxed_element_stores_v<ChunkType, T>)
                    {
                        // Other writers of this chunk read some of these
                        // elements, and write the ones this span reads.
                        if constexpr (Constant)
                        {
                            fill_span_relaxed_edges(dst, i - c_begin, hi - c_begin, s, static_cast<T>(v0));
                        }
                        else
                        {
                            store_span_relaxed_edges(dst, i - c_begin, hi - c_begin, s, at);
                        }
                        uniform = may_be_uniform_relaxed(dst, s); // buffer pinned by the session
                    }
                    else
                    {
                        for (std::size_t k = i; k < hi; ++k)
                        {
                            dst[k - c_begin] = at(k - c_begin);
                        }
                        // Do not read elements this span does not own: other
                        // writers store them PLAIN here (only Chunk2A's are
                        // relaxed atomics), so a read is a data race. Queue on
                        // the span's values, as the element paths mark: v1
                        // compresses only to first_value, Chunk2 to first_value
                        // or the second value; Chunk2A aggregates (no lock-free
                        // atomic_ref): only a whole-chunk uniform span.
                        if constexpr (is_v1_v<Arr>)
                        {
                            uniform = same && static_cast<T>(v0) == static_cast<T>(ChunkType::first_value);
                        }
                        else if constexpr (is_v2_v<Arr> && requires { arr.get_second_value(); })
                        {
                            uniform = same && (static_cast<T>(v0) == static_cast<T>(ChunkType::first_value) ||
                                               static_cast<T>(v0) == static_cast<T>(arr.get_second_value()));
                        }
                        else
                        {
                            uniform = same && i == c_begin && hi == c_end;
                        }
                    }
                    done = true;
                }
                Policy::end_write(thr); // 6.
            }

            if (!done)
            {
                for (; i < hi; ++i)
                {
                    arr[static_cast<IndType>(i)] = static_cast<T>(at(i - c_begin));
                }
                continue; // the element path queued the chunk itself
            }

            if (uniform) // 7. only a chunk that can be uniform
            {
                queue_for_compression(arr, c_begin);
            }
        }
    }
}

} // namespace internal

/**
 * @brief Bulk fill callable concurrently from several threads, each on its own
 *        indices: one write session per chunk instead of one per element.
 *
 * @details fill_range() writes raw into a chunk's buffer under a SERIAL
 *          contract (no concurrent writer, no compression pass). This variant
 *          keeps the per-element write path's guarantees by taking its steps
 *          once per chunk span: see internal::store_range_mt. Unlike
 *          fill_range, a chunk is never RELEASED here (a whole-chunk fill of
 *          the constant leaves it materialised and queued).
 *
 *          Chunk2A, Chunk2 and Chunk under OpenMP with fast block compression
 *          (ArrayChunkBits too: one atomic update per word); elsewhere (bool,
 *          serial builds) the element loop -- correct, without the speed-up.
 *          Chunk qualifies because its zero counter does not exist with fast
 *          block compression, and the null sentinel its element path adds
 *          during the slow compression path is never reached (the flag is
 *          raised on WriteCompressSyncPolicy<Proxy<T>> but tested on
 *          WriteCompressSyncPolicy<Chunk<T>>).
 *
 * @pre   Called inside or outside a parallel region; concurrent callers write
 *        DISJOINT indices (any chunk may be shared). No rechunk_to() / resize /
 *        clear of @p arr meanwhile (the usual contract of those).
 * @param arr   target array
 * @param first first index, inclusive (clamped to size())
 * @param last  last index, exclusive (clamped to size())
 * @param v     value to store
 */
template <internal::ChunkedArray Arr, typename U>
void fill_range_mt(Arr& arr, std::size_t first, std::size_t last, const U& v)
{
    using T            = Arr::value_type;
    const InT<T> value = static_cast<T>(v);
    internal::store_range_mt<true>(arr, first, last, [&value](std::size_t) { return static_cast<T>(value); });
}

/**
 * @brief copy_range() callable concurrently from several threads, each on its
 *        own indices: [first, last) gets [src_first, src_first + (last - first)),
 *        with one write session per chunk span.
 *
 * @details fill_range_mt's protocol (internal::store_range_mt); a span's source
 *          values are read before its session, never inside it. Same variants
 *          and builds as fill_range_mt; elsewhere the element loop.
 *
 * @pre   fill_range_mt's. The source does not alias @p arr; it may be another
 *        chunked array's const iterator (read outside the sessions), provided
 *        nothing writes that array meanwhile.
 * @param arr       target array
 * @param first     first target index, inclusive (clamped to size())
 * @param last      last target index, exclusive (clamped to size())
 * @param src_first random-access source iterator: element i reads
 *                  *(src_first + (i - first))
 */
template <internal::ChunkedArray Arr, typename It>
void copy_range_mt(Arr& arr, std::size_t first, std::size_t last, It src_first)
{
    internal::store_range_mt<false>(arr, first, last, [&src_first, first](std::size_t i)
                                    { return *(src_first + static_cast<std::ptrdiff_t>(i - first)); });
}

/**
 * @brief std::transform into a chunked array, callable concurrently from
 *        several threads, each on its own indices: element i of [first, last)
 *        gets op(*(src_first + (i - first))), with one write session per chunk
 *        span.
 *
 * @details copy_range_mt with an operation: @p op runs before the span's
 *          session, once per element, in index order within a chunk span.
 *
 * @pre   copy_range_mt's. @p op does not touch @p arr (it runs between the
 *        element writes of the span's materialisation and the session).
 * @param arr       target array
 * @param first     first target index, inclusive (clamped to size())
 * @param last      last target index, exclusive (clamped to size())
 * @param src_first random-access source iterator
 * @param op        called as op(source value); returns a value convertible to
 *                  the element type
 */
template <internal::ChunkedArray Arr, typename It, typename Op>
void transform_range_mt(Arr& arr, std::size_t first, std::size_t last, It src_first, Op&& op)
{
    internal::store_range_mt<false>(arr, first, last, [&src_first, &op, first](std::size_t i)
                                    { return op(*(src_first + static_cast<std::ptrdiff_t>(i - first))); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bulk reads, chunked half (the generic one is in vector_range_ops.h).
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief for_each_value() on a chunked array: chunk by chunk, one read section
 *        per block of elements (ArrayChunkBase::for_each_value).
 * @warning No other thread may write @p arr during the call (the iterator's
 *          contract). Callable from a parallel region on disjoint slices.
 */
template <internal::ChunkedArray Arr, typename F>
void for_each_value(const Arr& arr, std::size_t first, std::size_t last, F&& f)
{
    using IndType   = decltype(arr.size());
    const auto size = static_cast<std::size_t>(arr.size());
    last            = std::min(last, size);
    if (first >= last)
    {
        return;
    }
    arr.for_each_value(static_cast<IndType>(first), static_cast<IndType>(last), std::forward<F>(f));
}

/// @brief for_each_value() over every element of a chunked array.
template <internal::ChunkedArray Arr, typename F>
void for_each_value(const Arr& arr, F&& f)
{
    arr.for_each_value(std::forward<F>(f));
}

/**
 * @brief for_each_span() on a chunked array: f(std::span<const T>) block by
 *        block, and f(v, count) on a compressed chunk when f accepts it
 *        (ArrayChunkBase::for_each_span).
 * @warning No other thread may write @p arr during the call (the iterator's
 *          contract). Callable from a parallel region on disjoint slices.
 */
template <internal::ChunkedArray Arr, typename F>
void for_each_span(const Arr& arr, std::size_t first, std::size_t last, F&& f)
{
    using IndType   = decltype(arr.size());
    const auto size = static_cast<std::size_t>(arr.size());
    last            = std::min(last, size);
    if (first >= last)
    {
        return;
    }
    arr.for_each_span(static_cast<IndType>(first), static_cast<IndType>(last), std::forward<F>(f));
}

/// @brief for_each_span() over every element of a chunked array.
template <internal::ChunkedArray Arr, typename F>
void for_each_span(const Arr& arr, F&& f)
{
    arr.for_each_span(std::forward<F>(f));
}

/**
 * @brief read_range() on a chunked array: into a T*, each block is read
 *        straight into the destination (ArrayChunkBase::read_range).
 * @return @p out advanced past the last element written.
 */
template <internal::ChunkedArray Arr, typename OutIt>
OutIt read_range(const Arr& arr, std::size_t first, std::size_t last, OutIt out)
{
    using IndType   = decltype(arr.size());
    const auto size = static_cast<std::size_t>(arr.size());
    last            = std::min(last, size);
    if (first >= last)
    {
        return out;
    }
    return arr.read_range(static_cast<IndType>(first), static_cast<IndType>(last), out);
}

} // namespace shrink_array
