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
 * @file kokkos_adaptor.h
 * @brief The Kokkos facade of device::Array (docs/gpu_design.md, section 13 stage 3): what a
 *        Kokkos code holds, writes in its own parallel_for and reads in its own
 *        parallel_reduce, the phases marked by the facade's calls.
 *
 * @details Stage 3, experimental (see chunk_desc.h). A translation unit compiled for Kokkos's
 *          device backend (CUDA or HIP as a CMake language: Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE),
 *          Kokkos initialised. Usage:
 *          @code
 *          namespace sk = shrink_array::device::kokkos;
 *          sk::Array<double, 1024> a(n, arena_buffers);
 *          auto w = a.write();                         // a write phase
 *          Kokkos::parallel_for("f", n, KOKKOS_LAMBDA(std::int64_t i) { w(i) = f(i); w(i) += 1.0; });
 *          a.end_write();                              // fence, then device::Array::finish()
 *          auto r = a.read();                          // a read phase: any kernel, no barrier
 *          Kokkos::parallel_reduce("s", n, KOKKOS_LAMBDA(std::int64_t i, double& s) { s += r(i); }, sum);
 *          a.parallel_for_chunks(0, n, KOKKOS_LAMBDA(std::int64_t i) { return g(i); }); // aligned
 *          a.compress();
 *          @endcode
 *          The writes of a phase go through Writer::assign (any index, any thread); a write
 *          phase ends with end_write() before any read. parallel_for_chunks is the library's
 *          chunk-aligned loop (one block per chunk, the vote before materialising): it runs
 *          its own kernels, not a Kokkos TeamPolicy, whose CUDA teams sit on threadIdx.y where
 *          Writer::write expects threadIdx.x. The calls that run the library's kernels or
 *          read its counters fence Kokkos first (end_write, parallel_for_chunks, compress);
 *          write() and read() launch nothing and do not.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <memory>      // std::unique_ptr
#include <type_traits> // std::is_trivially_copyable_v

#include <Kokkos_Core.hpp> // Kokkos::fence, KOKKOS_INLINE_FUNCTION

#include "chunk/device/array.h"      // device::Array, Writer, View, Variant
#include "chunk/device/chunk_desc.h" // ChunkDesc (descriptor bytes)

namespace shrink_array::device::kokkos
{

/// @brief One element in a write phase: `w(i) = v`, `w(i) += v` (and -=, *=, /=), `T x = w(i)`.
template <typename T, unsigned int S, Variant V>
class Ref
{
  public:
    KOKKOS_INLINE_FUNCTION Ref(const Writer<T, S, V>& w, std::int64_t i) noexcept : w_(w), i_(i) {}

    KOKKOS_INLINE_FUNCTION const Ref& operator=(T v) const noexcept
    {
        w_.assign(i_, v);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION const Ref& operator+=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) + v);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION const Ref& operator-=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) - v);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION const Ref& operator*=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) * v);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION const Ref& operator/=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) / v);
        return *this;
    }

    /// @brief What this thread wrote, or the element as the phase found it (Writer::load).
    KOKKOS_INLINE_FUNCTION operator T() const noexcept
    {
        return w_.load(i_);
    }

  private:
    const Writer<T, S, V>& w_;
    std::int64_t           i_;
};

/// @brief The write access of a phase: trivially copyable, captured by value by a lambda.
template <typename T, unsigned int S, Variant V>
struct WriteAccess
{
    Writer<T, S, V> w;

    /// @brief Element @p i (0 <= i < size()).
    KOKKOS_INLINE_FUNCTION Ref<T, S, V> operator()(std::int64_t i) const noexcept
    {
        return Ref<T, S, V>(w, i);
    }
};

/**
 * @brief A device array for Kokkos codes: the device::Array it owns and the phases.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, 32 at least (1024 on GPU)
 * @tparam V the variant (chunk_desc.h)
 */
template <typename T, unsigned int S, Variant V = Variant::zero_any>
class Array
{
    static_assert(std::is_trivially_copyable_v<T>, "kokkos::Array: trivially copyable elements");

  public:
    /// @brief @p n elements reading @p constant (T{} in the zero variant); an arena of
    ///        @p arena_buffers buffers to start, grown by the phases that need more; an
    ///        append-buffer of @p append_capacity entries for the writes of a phase that finds
    ///        the arena exhausted (0: device::Array's default, 1% of the arena's elements).
    explicit Array(std::int64_t       n,
                   unsigned long long arena_buffers,
                   T                  constant        = T{},
                   unsigned long long append_capacity = 0)
        : a_(std::make_unique<device::Array<T, S, V>>(n, arena_buffers, constant, append_capacity))
    {
    }

    /// @brief The access of a write phase, for the caller's Kokkos kernels; end_write() after them.
    [[nodiscard]] WriteAccess<T, S, V> write() const
    {
        return WriteAccess<T, S, V>{a_->writer()}; // no fence: nothing runs on the device here
    }

    /// @brief The end of a write phase: fences Kokkos, replays the appends, settles the fresh
    ///        buffers (device::Array::finish()).
    void end_write()
    {
        Kokkos::fence("shrink_array::kokkos::Array::end_write");
        a_->finish();
    }

    /// @brief The access of a read phase (device::View): any kernel, until the next write phase.
    [[nodiscard]] View<T, S, V> read() const noexcept
    {
        return a_->view();
    }

    /// @brief a(i) = f(i) for i in [@p lo, @p hi) by the chunk-aligned loop, a phase of its own.
    ///        f: a KOKKOS_LAMBDA `T (std::int64_t)`, a function of i.
    template <typename F>
    void parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f)
    {
        Kokkos::fence("shrink_array::kokkos::Array::parallel_for_chunks");
        a_->parallel_for_chunks(lo, hi, f);
    }

    /// @brief The compression pass, between phases; returns the chunks folded back.
    unsigned long long compress()
    {
        Kokkos::fence("shrink_array::kokkos::Array::compress");
        return a_->compress();
    }

    /// @brief Elements.
    [[nodiscard]] std::int64_t size() const noexcept
    {
        return a_->size();
    }

    /// @brief Chunks holding a buffer (waits).
    [[nodiscard]] unsigned long long materialised() const
    {
        return a_->materialised();
    }

    /// @brief Bytes the array holds now: the buffers of its materialised chunks and its
    ///        descriptors (the arena's free buffers apart: see arena_bytes()).
    [[nodiscard]] std::size_t bytes() const
    {
        return static_cast<std::size_t>(materialised()) * S * sizeof(T) +
               static_cast<std::size_t>(a_->n_chunks()) * sizeof(ChunkDesc<T, V>);
    }

    /// @brief Bytes of buffers the arena has reserved, in use or free.
    [[nodiscard]] std::size_t arena_bytes() noexcept
    {
        return static_cast<std::size_t>(a_->arena().buffers()) * S * sizeof(T);
    }

    /// @brief The device::Array underneath (counters, conversion, the native API).
    [[nodiscard]] device::Array<T, S, V>& native() noexcept
    {
        return *a_;
    }

  private:
    std::unique_ptr<device::Array<T, S, V>> a_; ///< not movable itself: held by pointer
};

} // namespace shrink_array::device::kokkos
