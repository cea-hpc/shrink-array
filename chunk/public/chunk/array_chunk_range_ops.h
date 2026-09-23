#pragma once

/**
 * @file array_chunk_range_ops.h
 * @brief Bulk range operations: one decision per CHUNK instead of per element.
 *
 * @details WHY THESE EXIST
 *
 *          Every write through `arr[i] = v` builds a Proxy, arms ptr_on_array /
 *          num_chunk, runs the compression-candidacy bookkeeping and only then
 *          stores. That is the right price for an isolated write. For a bulk
 *          fill it is paid chunk_size times over for a decision that is the
 *          same across the whole chunk -- and it shows: rechunk_refill() grew a
 *          hand-written raw-fill precisely to escape it, with a measured
 *          108s -> 8s.
 *
 *          That raw-fill was a private fast path bypassing the proxy's
 *          invariants, and a fair share of this month's re-chunk debugging went
 *          into code that walked around it. These functions are the same idea
 *          made public and made correct: arm the chunk once, write straight
 *          into its buffer, then restore the invariants the bulk write bypassed.
 *
 *          NOT MEMBER FUNCTIONS, on purpose. ArrayChunk deliberately does not
 *          promise std::vector's interface (no data(), no stable element
 *          address -- a constant chunk has no storage at all), but the CALLERS
 *          of a bulk fill should not have to care which container they hold.
 *          Free functions with a generic fallback let the same call site take a
 *          std::vector or an ArrayChunk: the vector goes to std::fill, the
 *          ArrayChunk takes the fast path. Same shape the standard library uses
 *          to turn std::fill on char* into memset.
 *
 *          THE COMPRESSION WIN. Filling a range with the compressible constant
 *          (first_value, normally 0) through operator[] materialises every
 *          chunk it touches and waits for a later compression pass to reclaim
 *          them. Here a fully covered chunk is set constant directly: no
 *          allocation, and a materialised chunk hands its buffer back on the
 *          spot. On a zero-fill of a whole array that is the difference between
 *          allocating the entire array and allocating nothing.
 *
 * @warning Serial contract. These are bulk operations over an array the caller
 *          owns; they do not synchronise with concurrent writers or with a
 *          compression pass running on the same array. Call them from a
 *          quiescent point -- the same contract clear_core() and rechunk_to()
 *          already carry.
 */

#include <cstddef>
#include <type_traits>

#include "chunk/internal/in_t.h" // in_t<T>: the element snapshots
#include "chunk/PMP_CHUNK.h"           // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size
#include "chunk/internal/conditional_counter.h" // counter_value_t
#include "chunk/config.h"
#include "chunk/vector_range_ops.h" // detail::ChunkedArray + the std::vector fallback
#include "chunk/internal/write_compress_sync_policy.h" // fill_range_mt: the writer side of the Dekker handshake
#include "chunk/internal/compression_manager.h"      // fill_range_mt: must_defer_write / wait_out_compressor
#include "chunk/internal/relaxed_element.h"          // fill_range_mt: Chunk2A's shared elements

namespace shrink_array
{

// assume_drained_t / assume_drained: in vector_range_ops.h, so the tag form
// of the call compiles against std::vector too.

namespace detail
{

/**
 * @brief Can this variant be raw-filled, or must every element go through the
 *        proxy?
 *
 * @details bool only: its chunks are bit-packed. Chunk2 takes the raw paths
 *          since three fixes (Sept '26):
 *           - the materialised test must ask written(ind2), with the array's
 *             second-value index: the index-less written() is bool-only and
 *             false for every other T (materialised(a, ch, ind2));
 *           - queue_for_compression() must load the second value before
 *             queueing (set_info_on_chunk): Chunk2's candidate lists are keyed
 *             by it, and the Proxy2A-style arming it used put the chunks in the
 *             list of T{} -- a list no pass drains, whose entries then crashed
 *             the teardown of a rechunk's temporary array;
 *           - in counter mode (the serial builds) a raw store must recount
 *             Chunk2's two per-chunk counters (resync_counters), as it did v1's
 *             zero count. Without it rechunk_validation_chunk2_dyn lost runs of
 *             the second value to chunks compressed to the first one (POST-FAIL
 *             34819 / 1e6) and shrink_bench chunk2 --mode check read "is_cst_val2
 *             should be not". That was taken for a concurrency failure on Grace
 *             and Chunk2 was put back on the element path (5bc3ed1f); it fails
 *             identically on one core, serial build -- the configs it failed in
 *             were the serial ones, where the counters exist.
 */
template <typename T>
inline constexpr bool raw_fillable_v = !std::is_same_v<T, bool>;

/** @brief Only v1 in counter mode carries a per-chunk zero count to restore. */
inline constexpr bool track_zeros_v = chunk_config::use_chunk && chunk_config::use_compression_with_counter;

/** @brief Chunk2 in counter mode carries two: first-value and second-value counts. */
inline constexpr bool track_chunk2_counts_v = chunk_config::use_chunk2 && chunk_config::use_compression_with_counter;

/**
 * @brief Recount the per-chunk element counters after a raw store into
 *        [c_begin, c_begin + S), which bypassed them.
 *
 * @details In counter mode (use_compression_with_counter, i.e. without fast
 *          block compression -- the serial builds) the compression of a chunk
 *          is decided by counters the proxy maintains INCREMENTALLY: v1's
 *          nb_val_eq_zero, Chunk2's nb_val_eq_first_value and
 *          nb_val_eq_second_value. A raw store moves none of them, so they
 *          describe a state that no longer exists and nothing detects the
 *          drift: later writes only nudge them by one, and the full tests
 *          compare to chunk_size() exactly, so the chunk compresses the moment
 *          a count happens to land there -- a chunk that is NOT all constant,
 *          after which every read returns the constant. Recount, don't adjust
 *          (rechunk_refill documents the same trap).
 *
 *          Chunk2 used to be left out: its raw paths opened with v1's recount
 *          only, and rechunk_validation_chunk2_dyn (serial builds, the
 *          rebuild's copy_range) lost whole runs of the second value to
 *          chunks compressed to the first one (34819 of 1e6 elements).
 *          The second-value count is taken against the array's second value,
 *          exactly as Proxy2 counts it.
 */
template <typename Arr, typename Chunk, typename T>
void resync_counters(Arr& a, Chunk& ch, std::size_t c_begin, const T* dst, unsigned int S)
{
    if constexpr (track_zeros_v)
    {
        counter_value_t nz = 0;
        for (unsigned int k = 0; k < S; ++k)
            if (dst[k] == static_cast<T>(Chunk::first_value))
                ++nz;
        a[static_cast<typename Arr::ind_type>(c_begin)].resync_nb_val_eq_zero(nz);
    }
    else if constexpr (track_chunk2_counts_v && requires { a.get_second_value(); })
    {
        const T         sv = static_cast<T>(a.get_second_value());
        counter_value_t n1 = 0;
        counter_value_t n2 = 0;
        for (unsigned int k = 0; k < S; ++k)
        {
            if (dst[k] == static_cast<T>(Chunk::first_value))
                ++n1;
            if (dst[k] == sv)
                ++n2;
        }
        ch.set_nb_val_eq_first_value(n1);
        ch.set_nb_val_eq_second_value(n2);
    }
}

/**
 * @brief Does this chunk own a buffer the bulk store may write into?
 *
 * @details NOT `!is_cst_val1()`. A Chunk2A parked on its SECOND constant is
 *          not cst_val1 either, and its values pointer is the block shared by
 *          every chunk at that value: storing through it corrupted that block
 *          for all of them (shrink_bench, tab3 read as 0 where its second
 *          value 1 was expected, "[Debug] is_cst_val2 should be not!").
 *          written() is each variant's own "materialised" test -- v1: not
 *          cst_val1; Chunk2A: neither constant.
 */
template <typename Chunk>
[[nodiscard]] bool materialised(const Chunk& ch) noexcept
{
    return ch.written();
}

/**
 * @brief Can the chunk whose buffer is @p dst (@p S elements) be uniform?
 * @details NECESSARY condition only -- first, middle and last element equal.
 *          A uniform chunk always passes; a chunk that fails cannot be
 *          compressed, so queueing it would only cost the next pass a full
 *          scan (and, in a parallel run, push the lists to "dangerous" and the
 *          scans onto the writers' slow path).
 */
template <typename T>
[[nodiscard]] bool may_be_uniform(const T* dst, unsigned int S) noexcept
{
    return S == 0 || (dst[0] == dst[S / 2] && dst[0] == dst[S - 1]);
}

/**
 * @brief may_be_uniform() on a buffer other threads are writing (fill_range_mt).
 * @details Same three elements, read as relaxed atomics: other writers of the
 *          chunk store them concurrently (relaxed_element.h).
 */
template <typename T>
[[nodiscard]] bool may_be_uniform_relaxed(const T* dst, unsigned int S) noexcept
{
    if (S == 0)
        return true;
    const T v0 = load_element_relaxed(dst);
    return v0 == load_element_relaxed(dst + S / 2) && v0 == load_element_relaxed(dst + S - 1);
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
void fill_span_relaxed_edges(T* dst, std::size_t o_lo, std::size_t o_hi, std::size_t S, T v) noexcept
{
    // Ascending for S >= 4; for a smaller S the out-of-order or duplicate
    // entries are skipped by `s < o`, and S - 2 wrapping for S == 1 by `s >= o_hi`.
    const std::size_t shared[] = {0, 1, S / 2, S - 2, S - 1};
    std::size_t       o        = o_lo;
    for (const std::size_t s : shared)
    {
        if (s < o)
            continue;
        if (s >= o_hi)
            break;
        for (; o < s; ++o)
            dst[o] = v;
        store_element_relaxed(dst + s, v);
        o = s + 1;
    }
    for (; o < o_hi; ++o)
        dst[o] = v;
}

/// @brief The array's second-value index for Chunk2 (non-bool), -1 otherwise.
///        Read once per range call: it is a registry lookup.
template <typename Arr>
[[nodiscard]] int second_value_index(const Arr& a) noexcept
{
    if constexpr (chunk_config::use_chunk2 && !std::is_same_v<typename Arr::value_type, bool>)
        return a.get_ind_second_value();
    else
        return -1;
}

/// @brief materialised() for any variant: Chunk2's index-less written() is
///        bool-only (false for every other T), so it is asked with @p ind2.
template <typename Arr, typename Chunk>
[[nodiscard]] bool materialised(const Arr&, const Chunk& ch, [[maybe_unused]] int ind2) noexcept
{
    if constexpr (chunk_config::use_chunk2 && !std::is_same_v<typename Arr::value_type, bool>)
        return ch.written(ind2);
    else
        return ch.written();
}

/**
 * @brief Queue the chunk holding element @p i for the next compression pass.
 *
 * @details The raw stores of the bulk paths below go straight into a
 *          materialised chunk's buffer, past the write path and its marking:
 *          a chunk they leave constant (a fill of the constant over the whole
 *          chunk, a copy of a mostly-constant source) was never queued and
 *          stayed materialised for the run -- 56 of tab1's 360 chunks at the
 *          end of a chunk2a dyn check, all from its fill_range. This does what
 *          the last write of the span would have done: the flag's CAS and, on
 *          the first time, one entry in the caller's candidate list; the pass
 *          scans the chunk once and drops it if it is not constant. Fast-block
 *          builds only (the serial counters are resynced by the caller).
 */
template <typename Arr>
void queue_for_compression(Arr& a, std::size_t i)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        // Chunk2: candidate lists are keyed by second value and exist only for
        // arrays that called set_second_value() -- skip the others, as the
        // element path's pre-check does (Proxy2::has_candidate_list).
        if constexpr (chunk_config::use_chunk2 && !std::is_same_v<typename Arr::value_type, bool> &&
                      requires { Arr::proxy_type::has_candidate_list(a.get_second_value()); })
        {
            if (!Arr::proxy_type::has_candidate_list(a.get_second_value()))
                return;
        }
        auto p = a[static_cast<decltype(a.size())>(i)];
        // Chunk2 FIRST: Proxy2 also has set_num_tab_ptr_is_inside_chunk() (from
        // the shared base), but that one does not load the second value -- the
        // key of Chunk2's candidate list -- so the chunk landed in the list of
        // T{} (first_value), which no pass ever drains. set_info_on_chunk()
        // loads (second_value, ind, num_tab) and the flag pointer.
        if constexpr (std::is_same_v<typename Arr::proxy_type, Proxy2<typename Arr::value_type>>)
            p.set_info_on_chunk(); // Proxy2
        else if constexpr (requires { p.set_num_tab_ptr_is_inside_chunk(); })
            p.set_num_tab_ptr_is_inside_chunk(); // Proxy2A
        else
            p.set_info_on_chunk(); // Proxy (v1)
        p.add_to_compression_list();
    }
}

} // namespace detail

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
 *          bool (bit-packed chunks) takes the per-element path throughout --
 *          still correct, just without the speedup. See detail::raw_fillable_v.
 *
 * @param a     target array
 * @param first first index, inclusive (clamped to size())
 * @param last  last index, exclusive (clamped to size())
 * @param v     value to store
 */
template <detail::ChunkedArray Arr, typename U>
void fill_range(Arr& a, std::size_t first, std::size_t last, const U& v, bool drain = true)
{
    using T          = Arr::value_type;
    using ind_type   = decltype(a.size());
    using chunk_type = Arr::chunk_type;

    const auto n = static_cast<std::size_t>(a.size());
    if (first >= last || first >= n)
        return;
    if (last > n)
        last = n;

    // Drain this array's pending deferred writes FIRST.
    //
    // The bulk stores below go straight into the chunk buffer, bypassing the
    // deferred machinery. An op still queued on one of these chunks would be
    // replayed AFTER them and silently restore the old value -- measured: a
    // fill of 7 followed by a copy of 0 left 7 at index 320, on the chunk whose
    // op had not been flushed, while the neighbouring chunk was correct.
    // rechunk_refill() never hit this because rechunk_to() flushes before it
    // runs; a public bulk API has to do it itself.
    //
    // Forced: the pending-ops counter has been seen to read 0 while ops were
    // still queued, so the usual gate cannot be trusted here either.
    if (drain)
        a.flush_pending_deferred_ops(/*force=*/true);

    const unsigned int S     = a.chunk_size();
    const in_t<T>      value = static_cast<T>(v);
    const int          ind2  = detail::second_value_index(a); // Chunk2's written(ind)

    // Pin the ambient geometry once for the whole operation: every
    // materialisation below must size its buffer from THIS array's chunk_size,
    // not from whichever array wrote last (see Proxy::create_value).
    if constexpr (build_config::chunk_dynamic)
        set_thread_chunk_size(S);

    const bool value_is_cst1 = (value == static_cast<T>(chunk_type::first_value));

    const std::size_t first_chunk = first / S;
    const std::size_t last_chunk  = (last - 1) / S;

    for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
    {
        const std::size_t c_begin = ci * S;
        const std::size_t c_end   = c_begin + S;
        const std::size_t lo      = first > c_begin ? first : c_begin;
        const std::size_t hi      = last < c_end ? last : c_end;
        const bool        whole   = (lo == c_begin) && (hi == c_end);

        auto& ch = a.chunk_ref(ci);

        // ── Whole chunk set to the compressible constant: no storage needed ──
        //
        // Only where releasing the buffer is a two-argument clear_st(thr,
        // n_elems), i.e. v1. Chunk2A's clear_st also takes the compression state
        // (num_compress_value, ptr_type_cst, ptr_second_value), which this loop
        // has no business synthesising -- rechunk_refill draws the same line for
        // the same reason. Those variants fall through to the paths below, which
        // are correct, just without the free-the-buffer shortcut.
        // BOTH spellings, not either: the two-argument probe alone does NOT
        // discriminate. Chunk2 has clear_st(int num_compress_value, int ind),
        // which accepts (thr, n) by conversion while meaning something else
        // entirely -- the OpenMP branch below would then pass a THREAD RANK as
        // the constant to compress to, silently, and the serial branch would
        // fail to compile on the zero-argument call that Chunk2 lacks. Only v1
        // offers both, so requiring both is what actually names v1 here.
        constexpr bool can_release = requires(chunk_type& c, int thr, unsigned int n) { c.clear_st(thr, n); } &&
                                     requires(chunk_type& c) { c.clear_st(); };

        if constexpr (can_release)
        {
            if (whole && value_is_cst1)
            {
                if (!ch.is_cst_val1())
                {
                    if constexpr (build_config::use_openmp)
                        ch.clear_st(get_thr_rank_fast(), S); // frees the buffer, marks cst1
                    else
                        ch.clear_st();
                }
                continue;
            }
        }

        // ── Variants whose second-value state lives in the proxy ─────────────
        if constexpr (!detail::raw_fillable_v<T>)
        {
            for (std::size_t i = lo; i < hi; ++i)
                a[static_cast<ind_type>(i)] = value;
        }
        else
        {
            // ── Arm, then bulk-store ────────────────────────────────────────
            // Real proxy writes until the chunk materialises: that is what
            // allocates the buffer and registers the compression bookkeeping.
            // Writing a chunk's constant into it changes nothing, so this loop
            // can walk the whole range without materialising -- in which case
            // the chunk is genuinely constant and there is nothing left to do.
            // "Materialises" is written(), not !is_cst_val1(): see
            // detail::materialised for the Chunk2A second-constant trap.
            std::size_t i = lo;
            while (i < hi && !detail::materialised(a, ch, ind2))
            {
                a[static_cast<ind_type>(i)] = value;
                ++i;
            }
            if (!detail::materialised(a, ch, ind2))
                continue; // still constant: the range held only that constant

            T* dst = ch.values_load();
            for (std::size_t k = i; k < hi; ++k)
                dst[k - c_begin] = value;

            // Restore the per-chunk counters the bulk store bypassed.
            detail::resync_counters(a, ch, c_begin, dst, S);

            // The chunk may be constant now: a fill of first_value, or on
            // chunk2a of any value (a uniform run is its second constant).
            bool compressible_value = value_is_cst1 || chunk_config::use_chunk2a;
            if constexpr (chunk_config::use_chunk2 && requires { a.get_second_value(); })
                compressible_value = compressible_value || value == static_cast<T>(a.get_second_value());
            if (compressible_value && detail::may_be_uniform(dst, S))
                detail::queue_for_compression(a, c_begin);
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
 * @param a         target array
 * @param first     first target index, inclusive (clamped to size())
 * @param last      last target index, exclusive (clamped to size())
 * @param src_first source iterator; must not alias @p a
 */
template <detail::ChunkedArray Arr, typename It>
void copy_range(Arr& a, std::size_t first, std::size_t last, It src_first, bool drain = true)
{
    using T        = Arr::value_type;
    using ind_type = decltype(a.size());

    const auto n = static_cast<std::size_t>(a.size());
    if (first >= last || first >= n)
        return;
    if (last > n)
        last = n;

    // Drain this array's pending deferred writes FIRST.
    //
    // The bulk stores below go straight into the chunk buffer, bypassing the
    // deferred machinery. An op still queued on one of these chunks would be
    // replayed AFTER them and silently restore the old value -- measured: a
    // fill of 7 followed by a copy of 0 left 7 at index 320, on the chunk whose
    // op had not been flushed, while the neighbouring chunk was correct.
    // rechunk_refill() never hit this because rechunk_to() flushes before it
    // runs; a public bulk API has to do it itself.
    //
    // Forced: the pending-ops counter has been seen to read 0 while ops were
    // still queued, so the usual gate cannot be trusted here either.
    if (drain)
        a.flush_pending_deferred_ops(/*force=*/true);

    const unsigned int S    = a.chunk_size();
    const int          ind2 = detail::second_value_index(a); // Chunk2's written(ind)

    if constexpr (build_config::chunk_dynamic)
        set_thread_chunk_size(S);

    const std::size_t first_chunk = first / S;
    const std::size_t last_chunk  = (last - 1) / S;

    for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
    {
        const std::size_t c_begin = ci * S;
        const std::size_t c_end   = c_begin + S;
        const std::size_t lo      = first > c_begin ? first : c_begin;
        const std::size_t hi      = last < c_end ? last : c_end;

        auto& ch = a.chunk_ref(ci);

        if constexpr (!detail::raw_fillable_v<T>)
        {
            for (std::size_t i = lo; i < hi; ++i)
                a[static_cast<ind_type>(i)] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(i - first)));
        }
        else
        {
            std::size_t i = lo;
            while (i < hi && !detail::materialised(a, ch, ind2))
            {
                a[static_cast<ind_type>(i)] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(i - first)));
                ++i;
            }
            if (!detail::materialised(a, ch, ind2))
                continue; // the whole span was the chunk's constant

            T* dst = ch.values_load();
            for (std::size_t k = i; k < hi; ++k)
                dst[k - c_begin] = static_cast<T>(*(src_first + static_cast<std::ptrdiff_t>(k - first)));

            detail::resync_counters(a, ch, c_begin, dst, S); // as fill_range

            // The source is not known to be uniform: the pass decides -- for
            // the chunks that can be.
            if (detail::may_be_uniform(dst, S))
                detail::queue_for_compression(a, c_begin);
        }
    }
}

/// @copydoc fill_range
/// @note The assume_drained_t tag: caller guarantees the deferred queue is already drained
template <detail::ChunkedArray Arr, typename U>
void fill_range(Arr& a, std::size_t first, std::size_t last, const U& v, assume_drained_t)
{
    fill_range(a, first, last, v, /*drain=*/false);
}

/// @copydoc copy_range
/// @note The assume_drained_t tag: caller guarantees the deferred queue is already drained
template <detail::ChunkedArray Arr, typename It>
void copy_range(Arr& a, std::size_t first, std::size_t last, It src_first, assume_drained_t)
{
    copy_range(a, first, last, src_first, /*drain=*/false);
}


// ─────────────────────────────────────────────────────────────────────────────
// fill_range_mt: the parallel form -- one write SESSION per chunk
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Bulk fill callable concurrently from several threads, each on its own
 *        indices: one write session per chunk instead of one per element.
 *
 * @details fill_range() writes raw into a chunk's buffer under a SERIAL
 *          contract (no concurrent writer, no compression pass). This variant
 *          keeps the per-element write path's guarantees by taking its steps
 *          once per chunk span instead of once per element:
 *            1. pending deferred ops on the chunk -> this span goes through the
 *               element path (it syncs foreign ops, erases/excludes own ones);
 *            2. a compressor on the chunk -> wait it out (bounded, before
 *               announcing, like the element path);
 *            3. announce this thread as a writer of the chunk (Dekker, the
 *               write side of WriteCompressSyncPolicy);
 *            4. re-check (compressor, pending ops, still materialised); any
 *               doubt -> retire and take the element path for the span;
 *            5. raw stores into the chunk's buffer;
 *            6. retire;
 *            7. queue the chunk as a compression candidate (as the last element
 *               write would have: the pass drops it if it is not constant).
 *          A chunk that is not materialised is first materialised through the
 *          element path, outside any session (a proxy write inside a session
 *          would re-announce and then clear this thread's writer flag).
 *          Unlike fill_range, a chunk is never RELEASED here (a whole-chunk
 *          fill of the constant leaves it materialised and queued): another
 *          thread may be writing other elements of the same chunk, into that
 *          very buffer.
 *
 *          Chunk2A, Chunk2 and Chunk under OpenMP with fast block compression;
 *          everywhere else (bool, serial builds) it is the element loop
 *          -- correct in parallel, without the speed-up. Chunk: its zero
 *          counter, which a raw fill would have to resync from the whole chunk,
 *          does not exist with fast block compression; and the null sentinel
 *          its element path adds while the slow compression path is active is
 *          never reached (the flag is raised on WriteCompressSyncPolicy<Proxy<T>>
 *          but tested on WriteCompressSyncPolicy<Chunk<T>>).
 *
 * @pre   Called inside or outside a parallel region; concurrent callers write
 *        DISJOINT indices (any chunk may be shared). No rechunk_to() / resize /
 *        clear of @p a meanwhile (the usual contract of those).
 * @param a     target array
 * @param first first index, inclusive (clamped to size())
 * @param last  last index, exclusive (clamped to size())
 * @param v     value to store
 */
template <detail::ChunkedArray Arr, typename U>
void fill_range_mt(Arr& a, std::size_t first, std::size_t last, const U& v)
{
    using T          = Arr::value_type;
    using ind_type   = decltype(a.size());
    using chunk_type = Arr::chunk_type;

    const auto n = static_cast<std::size_t>(a.size());
    if (first >= last || first >= n)
        return;
    if (last > n)
        last = n;

    const in_t<T> value = static_cast<T>(v);

    constexpr bool session = (chunk_config::use_chunk2a || chunk_config::use_chunk2 || chunk_config::use_chunk) &&
                             build_config::use_openmp &&
                             chunk_config::use_fast_block_compression && detail::raw_fillable_v<T>;
    if constexpr (!session)
    {
        for (std::size_t i = first; i < last; ++i)
            a[static_cast<ind_type>(i)] = value;
    }
    else
    {
        using Policy = WriteCompressSyncPolicy<typename Arr::proxy_type>;
        using Mgr    = CompressionManager<T, chunk_type>;

        const unsigned int S    = a.chunk_size();
        const int          ind2 = detail::second_value_index(a); // Chunk2's written(ind)
        if constexpr (build_config::chunk_dynamic)
            set_thread_chunk_size(S);
        const int thr = get_thr_rank_fast();

        const std::size_t first_chunk = first / S;
        const std::size_t last_chunk  = (last - 1) / S;

        for (std::size_t ci = first_chunk; ci <= last_chunk; ++ci)
        {
            const std::size_t c_begin = ci * S;
            const std::size_t c_end   = c_begin + S;
            const std::size_t lo      = first > c_begin ? first : c_begin;
            const std::size_t hi      = last < c_end ? last : c_end;

            auto& ch = a.chunk_ref(ci);

            // Materialise through the element path, outside any session.
            std::size_t i = lo;
            while (i < hi && !detail::materialised(a, ch, ind2))
            {
                a[static_cast<ind_type>(i)] = value;
                ++i;
            }
            if (i == hi)
                continue;

            bool done    = false;
            bool uniform = false; // may_be_uniform, read inside the session
            if (!ch.pending_any()) // 1. deferred ops on the chunk: element path
            {
                if (Mgr::must_defer_write(ch)) // 2. a compressor on it: wait (bounded)
                    Mgr::wait_out_compressor(ch);

                Policy::announce_write(thr, &ch); // 3.
                if (!Mgr::must_defer_write(ch) && !ch.pending_any() && detail::materialised(a, ch, ind2)) // 4.
                {
                    T* dst = ch.values_load(); // 5.
                    if constexpr (detail::relaxed_element_stores_v<chunk_type, T>)
                    {
                        // Other writers of this chunk read some of these
                        // elements, and write the ones this span reads.
                        detail::fill_span_relaxed_edges(dst, i - c_begin, hi - c_begin, S, static_cast<T>(value));
                        uniform = detail::may_be_uniform_relaxed(dst, S); // buffer pinned by the session
                    }
                    else
                    {
                        for (std::size_t k = i; k < hi; ++k)
                            dst[k - c_begin] = value;
                        // No read of elements this span does not own: other
                        // writers of the chunk store them PLAIN here (only
                        // Chunk2A's are relaxed atomics), and reading them was a
                        // data race (TSan, bench_phase_range_mt). Queue on the
                        // value instead, as the element paths mark: v1 compresses
                        // only to first_value, Chunk2 only to first_value or the
                        // array's second value. Chunk2A aggregates (no lock-free
                        // atomic_ref): only a span covering the whole chunk,
                        // uniform by construction; a partial one is not queued.
                        if constexpr (chunk_config::use_chunk)
                            uniform = static_cast<T>(value) == static_cast<T>(chunk_type::first_value);
                        else if constexpr (chunk_config::use_chunk2 && requires { a.get_second_value(); })
                            uniform = static_cast<T>(value) == static_cast<T>(chunk_type::first_value) ||
                                      static_cast<T>(value) == static_cast<T>(a.get_second_value());
                        else
                            uniform = (i == c_begin && hi == c_end);
                    }
                    done    = true;
                }
                Policy::end_write(thr); // 6.
            }

            if (!done)
            {
                for (; i < hi; ++i)
                    a[static_cast<ind_type>(i)] = value;
                continue; // the element path queued the chunk itself
            }

            if (uniform) // 7. only a chunk that can be uniform
                detail::queue_for_compression(a, c_begin);
        }
    }
}

} // namespace shrink_array
