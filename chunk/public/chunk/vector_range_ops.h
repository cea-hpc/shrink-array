/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file vector_range_ops.h
 * @brief fill_range / copy_range for plain containers -- std::vector first.
 *
 * @details The generic half of the bulk range API. array_chunk_range_ops.h
 *          adds the overloads for the chunked arrays (one decision per chunk);
 *          this file carries the fallback for anything else, and nothing here
 *          depends on the chunk library. A program built on std::vector -- the
 *          bench's baseline, for one -- includes this header alone and calls
 *          shrink_array::fill_range / copy_range exactly as the chunked build
 *          does, so one call site serves both containers. Same shape the
 *          standard library uses to turn std::fill on char* into memset.
 *
 *          Kept in the shrink_array namespace on purpose: the point is that
 *          the call site does not change with the container.
 *
 * @see array_chunk_range_ops.h for the chunked overloads and the compression
 *      rationale.
 */

#include <algorithm>   // std::fill, std::copy
#include <concepts>    // std::convertible_to
#include <cstddef>     // std::size_t, std::ptrdiff_t
#include <iterator>    // std::next
#include <type_traits> // std::is_same_v (fill_range_mt's vector<bool> guard)
#include <vector>      // std::vector<bool> (same guard)

namespace shrink_array
{

/**
 * @brief Tag: the caller has already drained this array's deferred ops.
 *
 * @details The plain fill_range/copy_range on a CHUNKED array drain before
 *          touching a buffer, because a queued op replayed afterwards would
 *          silently restore the old value. That drain is a whole-array, serial
 *          operation (every chunk, every rank) -- expensive on a large array,
 *          wrong and pointless once per chunk from inside a parallel loop.
 *
 *          So callers that know the queues are empty pass this tag: a
 *          quiescent point right after a phase end (the bench), or
 *          rechunk_to(), which flushes before rechunk_refill() runs and then
 *          refills disjoint chunks concurrently. On a plain container the tag
 *          is accepted and ignored, so one call site serves both.
 *
 * @warning Passing it without having drained reintroduces exactly the bug the
 *          drain exists for, and it is silent: the write lands, then a replayed
 *          op overwrites it, and only some chunks are affected.
 */
struct assume_drained_t
{
    explicit assume_drained_t() = default;
};
inline constexpr assume_drained_t assume_drained{};

namespace detail
{

/**
 * @brief Is this one of the chunked arrays (Chunk, Chunk2, Chunk2A)?
 *
 * @details Detected structurally, on chunk_ref()/chunk_size(), rather than by
 *          naming the three class templates: they take a constrained element
 *          parameter, so a plain forward declaration does not match, and a trait
 *          keeps the overload open to any future variant exposing the same two
 *          entry points. Anything else -- std::vector first -- takes the generic
 *          version below, whose contiguous iterator arithmetic the chunked
 *          arrays cannot satisfy (their iterator is bidirectional).
 */
template <typename A>
concept ChunkedArray = requires(A& a, const A& ca, std::size_t i) {
    { a.chunk_ref(i) };
    { ca.chunk_size() } -> std::convertible_to<unsigned int>;
    { ca.size() };
};

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Generic fallback: anything with begin() and a random-access iterator,
// std::vector first among them. Kept as the primary template so an unforeseen
// container still compiles instead of failing overload resolution.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Set every element of [first, last) to @p v.
 * @param c     target container
 * @param first first index, inclusive (clamped to c.size())
 * @param last  last index, exclusive (clamped to c.size())
 * @param v     value to store
 */
template <typename C, typename U>
    requires(!detail::ChunkedArray<C>)
void fill_range(C& c, std::size_t first, std::size_t last, const U& v)
{
    const auto n = static_cast<std::size_t>(c.size());
    last         = std::min(last, n);
    if (first >= last)
        return;
    std::fill(std::next(c.begin(), static_cast<std::ptrdiff_t>(first)),
              std::next(c.begin(), static_cast<std::ptrdiff_t>(last)), static_cast<C::value_type>(v));
}

/**
 * @brief Copy [src_first, src_first + (last - first)) into [first, last).
 * @param c         target container
 * @param first     first target index, inclusive (clamped to c.size())
 * @param last      last target index, exclusive (clamped to c.size())
 * @param src_first iterator on the source range; must not alias @p c
 */
template <typename C, typename It>
    requires(!detail::ChunkedArray<C>)
void copy_range(C& c, std::size_t first, std::size_t last, It src_first)
{
    const auto n = static_cast<std::size_t>(c.size());
    last         = std::min(last, n);
    if (first >= last)
        return;
    std::copy(src_first, std::next(src_first, static_cast<std::ptrdiff_t>(last - first)),
              std::next(c.begin(), static_cast<std::ptrdiff_t>(first)));
}

/// @copydoc fill_range
/// @note The assume_drained_t tag: nothing to drain on a plain container; accepted so
///        the tag form of the call is the same on every container
template <typename C, typename U>
    requires(!detail::ChunkedArray<C>)
void fill_range(C& c, std::size_t first, std::size_t last, const U& v, assume_drained_t)
{
    fill_range(c, first, last, v);
}

/// @copydoc copy_range
/// @note The assume_drained_t tag: see fill_range's tag overload
template <typename C, typename It>
    requires(!detail::ChunkedArray<C>)
void copy_range(C& c, std::size_t first, std::size_t last, It src_first, assume_drained_t)
{
    copy_range(c, first, last, src_first);
}

/**
 * @brief fill_range_mt on a plain container: fill_range, callable from several
 *        threads at once on DISJOINT indices.
 * @details The chunked overload (array_chunk_range_ops.h) runs a per-chunk
 *          write session; a plain container needs nothing -- disjoint
 *          std::fill ranges do not race. NOT for std::vector<bool>, whose
 *          bit-packed words make neighbouring indices share storage.
 */
template <typename C, typename U>
    requires(!detail::ChunkedArray<C>)
void fill_range_mt(C& c, std::size_t first, std::size_t last, const U& v)
{
    static_assert(!std::is_same_v<C, std::vector<bool>>,
                  "fill_range_mt: std::vector<bool> packs neighbouring indices into one word");
    fill_range(c, first, last, v);
}

} // namespace shrink_array
