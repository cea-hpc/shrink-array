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
 * @file arcane_adaptor.h
 * @brief The Arcane facade of device::Array (docs/gpu_design.md section 13, after stage 3): what an
 *        Arcane code holds, writes in its own RUNCOMMAND_LOOP and reads in another, the phases
 *        marked by the facade's calls on the code's RunQueue.
 *
 * @details Experimental (see chunk_desc.h). A translation unit compiled for Arcane's
 *          accelerator (arcane_accelerator_add_source_files, CUDA or HIP), the queue an
 *          accelerator one (`-A,AcceleratorRuntime=cuda`): the views read and write device
 *          memory, a host loop policy (Sequential, Thread) is refused. Usage:
 *          @code
 *          namespace sa = shrink_array;
 *          sa::zero_any::ArcaneArray<double> a(n);
 *          auto queue = acc_mng->defaultQueue();          // a RunQueue or a RunQueue*
 *          {
 *              auto command = makeCommand(queue);
 *              auto w = a.write(queue);                   // a write phase
 *              command << RUNCOMMAND_LOOP1(iter, n) { auto [i] = iter(); w(i) = f(i); w(i) += 1.0; };
 *          }
 *          a.end_write(queue);                            // barrier, then device::Array::finish()
 *          a.compress(queue);
 *          {
 *              auto command = makeCommand(queue);
 *              auto r = a.read(queue);                    // a read phase: any command, no barrier
 *              auto out = viewOut(command, numarray);
 *              command << RUNCOMMAND_LOOP1(iter, n) { auto [i] = iter(); out(i) = r(i); };
 *          }
 *          @endcode
 *          RUNCOMMAND_LOOP1's lambda captures by value ([=]), so write() and read() return
 *          trivially copyable accesses, as Kokkos's facade does. The library's kernels run on
 *          the device's default stream, Arcane's on the queue's: every facade call that runs
 *          them waits for the queue first (RunQueue::barrier()) and for its own kernels after
 *          (a device synchronisation), so the next commands of the queue see a settled array
 *          whatever the queue's stream flags. A loop index is an Int32 in RUNCOMMAND_LOOP1:
 *          an array of up to 2^31 - 1 elements per command.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <memory>      // std::unique_ptr
#include <stdexcept>   // std::runtime_error
#include <string>      // std::string (the error message)
#include <type_traits> // std::is_pointer_v, std::is_trivially_copyable_v, std::remove_cvref_t
#include <vector>      // std::vector (to_host)

#include "arcane/accelerator/RunQueue.h" // RunQueue::barrier(), executionPolicy(), eExecutionPolicy

#include "chunk/device/array.h"          // device::Array, Writer, View, Variant, ArrayState, ArrayImage
#include "chunk/device/backend.h"        // device::synchronize
#include "chunk/device/chunk_desc.h"     // ChunkDesc (descriptor bytes)
#include "chunk/internal/host_device.h"  // CHUNK_HD

namespace shrink_array::device::arcane
{

/// @brief One element in a write phase: `w(i) = v`, `w(i) += v` (and -=, *=, /=), `T x = w(i)`.
template <typename T, unsigned int S, Variant V>
class Ref
{
  public:
    CHUNK_HD Ref(const Writer<T, S, V>& w, std::int64_t i) noexcept : w_(w), i_(i) {}

    CHUNK_HD const Ref& operator=(T v) const noexcept
    {
        w_.assign(i_, v);
        return *this;
    }

    CHUNK_HD const Ref& operator+=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) + v);
        return *this;
    }

    CHUNK_HD const Ref& operator-=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) - v);
        return *this;
    }

    CHUNK_HD const Ref& operator*=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) * v);
        return *this;
    }

    CHUNK_HD const Ref& operator/=(T v) const noexcept
    {
        w_.assign(i_, w_.load(i_) / v);
        return *this;
    }

    /// @brief What this thread wrote, or the element as the phase found it (Writer::load).
    CHUNK_HD operator T() const noexcept
    {
        return w_.load(i_);
    }

  private:
    const Writer<T, S, V>& w_;
    std::int64_t           i_;
};

/// @brief The write access of a phase: trivially copyable, captured by RUNCOMMAND_LOOP's [=].
template <typename T, unsigned int S, Variant V>
struct WriteAccess
{
    Writer<T, S, V> w;

    /// @brief Element @p i (0 <= i < size()).
    CHUNK_HD Ref<T, S, V> operator()(std::int64_t i) const noexcept
    {
        return Ref<T, S, V>(w, i);
    }
};

namespace detail
{

/// @brief The RunQueue @p q designates: IAcceleratorMng::defaultQueue() gives a pointer in
///        some Arcane versions, queue() a RunQueue.
template <typename Q>
decltype(auto) queue_of(Q&& q) noexcept
{
    if constexpr (std::is_pointer_v<std::remove_cvref_t<Q>>)
    {
        return *q;
    }
    else
    {
        return static_cast<Q&&>(q);
    }
}

/// @brief Checks that @p q runs on the device.
/// @throws std::runtime_error if @p q's policy is a host one: the accesses would touch device
///         memory from the host.
template <typename Q>
void check_device(Q&& q, const char* what)
{
    const auto policy = queue_of(static_cast<Q&&>(q)).executionPolicy();
    if (policy != ::Arcane::Accelerator::eExecutionPolicy::CUDA &&
        policy != ::Arcane::Accelerator::eExecutionPolicy::HIP)
    {
        throw std::runtime_error(std::string("shrink_array::device::arcane::") + what +
                                 ": the RunQueue is not a CUDA or HIP one (run with -A,AcceleratorRuntime=cuda or hip)");
    }
}

/// @brief Waits for the commands of @p q, after check_device().
template <typename Q>
void barrier(Q&& q, const char* what)
{
    check_device(static_cast<Q&&>(q), what);
    queue_of(static_cast<Q&&>(q)).barrier();
}

} // namespace detail

/**
 * @brief A device array for Arcane codes: the device::Array it owns and the phases.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, 32 at least (1024 on GPU)
 * @tparam V the variant (chunk_desc.h)
 */
template <typename T, unsigned int S, Variant V = Variant::zero_any>
class Array
{
    static_assert(std::is_trivially_copyable_v<T>, "arcane::Array: trivially copyable elements");

  public:
    using value_type                 = T; ///< The element type, as the host arrays'.
    static constexpr Variant variant = V; ///< The variant (chunk_desc.h).

    /// @brief Elements per chunk: S (the host arrays' chunk_size()).
    [[nodiscard]] static constexpr unsigned int chunk_size() noexcept
    {
        return S;
    }

    /// @brief @p n elements reading @p constant (T{} in the zero variant); an arena of
    ///        @p arena_buffers buffers to start, grown by the phases that need more; an
    ///        append-buffer of @p append_capacity entries (0: device::Array's default).
    explicit Array(std::int64_t       n,
                   unsigned long long arena_buffers,
                   T                  constant        = T{},
                   unsigned long long append_capacity = 0)
        : a_(std::make_unique<device::Array<T, S, V>>(n, arena_buffers, constant, append_capacity))
    {
    }

    /// @brief @p n elements at T{}, device::Array's default first arena (default_arena(): a
    ///        buffer per chunk, no write lost).
    explicit Array(std::int64_t n) : a_(std::make_unique<device::Array<T, S, V>>(n))
    {
    }

    /// @brief The array @p state describes: export_array() of a host array at chunk size S, or
    ///        image().state of a device one (device::Array(state)).
    /// @throws std::runtime_error if @p state is at another chunk size.
    explicit Array(const ArrayState<T>& state) : a_(std::make_unique<device::Array<T, S, V>>(state))
    {
    }

    /// @brief The access of a write phase, for the caller's commands on @p queue (checked to be
    ///        a device one; launches nothing); end_write() after them.
    template <typename Q>
    [[nodiscard]] WriteAccess<T, S, V> write(Q&& queue) const
    {
        detail::check_device(static_cast<Q&&>(queue), "write");
        return WriteAccess<T, S, V>{a_->writer()};
    }

    /// @brief The end of a write phase: waits for @p queue, replays the appends, settles the
    ///        fresh buffers (device::Array::finish()), waits for them.
    template <typename Q>
    void end_write(Q&& queue)
    {
        detail::barrier(static_cast<Q&&>(queue), "end_write");
        a_->finish();
        device::synchronize();
    }

    /// @brief The access of a read phase (device::View) for commands on @p queue (checked to be
    ///        a device one): any command, until the next write phase.
    template <typename Q>
    [[nodiscard]] View<T, S, V> read(Q&& queue) const
    {
        detail::check_device(static_cast<Q&&>(queue), "read");
        return a_->view();
    }

    /// @brief a(i) = f(i) for i in [@p lo, @p hi) by the chunk-aligned loop, a phase of its own
    ///        (the library's kernels, not a RunCommand). f: a `__host__ __device__` callable
    ///        `T (std::int64_t)`, a function of i.
    template <typename Q, typename F>
    void parallel_for_chunks(Q&& queue, std::int64_t lo, std::int64_t hi, const F& f)
    {
        detail::barrier(static_cast<Q&&>(queue), "parallel_for_chunks");
        a_->parallel_for_chunks(lo, hi, f);
        device::synchronize();
    }

    /// @brief The compression pass, between phases; returns the chunks folded back.
    template <typename Q>
    unsigned long long compress(Q&& queue)
    {
        detail::barrier(static_cast<Q&&>(queue), "compress");
        const unsigned long long folded = a_->compress();
        device::synchronize();
        return folded;
    }

    /// @brief Makes this array the one @p state describes, between phases (waits for @p queue).
    /// @throws std::runtime_error if @p state has another size or chunk size.
    template <typename Q>
    void assign(Q&& queue, const ArrayState<T>& state)
    {
        detail::barrier(static_cast<Q&&>(queue), "assign");
        a_->assign(state);
        device::synchronize();
    }

    /// @brief This array as an ArrayState owning the materialised chunks' elements, for
    ///        import_array() on a host array, between phases (waits for @p queue).
    template <typename Q>
    [[nodiscard]] ArrayImage<T> image(Q&& queue) const
    {
        detail::barrier(static_cast<Q&&>(queue), "image");
        return a_->image();
    }

    /// @brief Every element, on the host, between phases (waits for @p queue).
    template <typename Q>
    [[nodiscard]] std::vector<T> to_host(Q&& queue) const
    {
        detail::barrier(static_cast<Q&&>(queue), "to_host");
        return a_->to_host();
    }

    /// @brief zero_or_value: the value other than T{} the chunks may fold to from now on (T{}:
    ///        none), between phases (device::Array::set_second_value).
    void set_second_value(T second) noexcept
        requires(V == Variant::zero_or_value)
    {
        a_->set_second_value(second);
    }

    /// @brief zero_or_value: the second value, T{} while none is declared.
    [[nodiscard]] T second_value() const noexcept
        requires(V == Variant::zero_or_value)
    {
        return a_->second_value();
    }

    /// @brief Elements.
    [[nodiscard]] std::int64_t size() const noexcept
    {
        return a_->size();
    }

    /// @brief Chunks holding a buffer (waits for the device, between phases).
    [[nodiscard]] unsigned long long materialised() const
    {
        return a_->materialised();
    }

    /// @brief Chunks holding a buffer, as the host arrays' written_chunk_count().
    [[nodiscard]] unsigned long long written_chunk_count() const
    {
        return a_->written_chunk_count();
    }

    /// @brief Percentage of the chunks holding a buffer, the host arrays' stats().
    [[nodiscard]] double stats() const
    {
        return a_->stats();
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

    /// @brief The device::Array underneath (counters, the native API).
    [[nodiscard]] device::Array<T, S, V>& native() noexcept
    {
        return *a_;
    }

  private:
    std::unique_ptr<device::Array<T, S, V>> a_; ///< not movable itself: held by pointer
};

/// @brief Waits for the commands of @p queue (a RunQueue or a RunQueue*, the one
///        IAcceleratorMng::defaultQueue() gives), before the host reads what they wrote.
template <typename Q>
void barrier(Q&& queue)
{
    detail::queue_of(static_cast<Q&&>(queue)).barrier();
}

/// @brief end_write() of every array of @p a after one wait for @p queue: one wait for all
///        (device::finish_all), for a write phase over several arrays.
template <typename Q, typename... A>
void end_write(Q&& queue, A&... a)
{
    detail::barrier(static_cast<Q&&>(queue), "end_write");
    device::finish_all(a.native()...);
    device::synchronize();
}

/// @brief compress() of every array of @p a after one wait for @p queue: two waits for all
///        (device::compress_all). Returns the chunks folded back.
template <typename Q, typename... A>
unsigned long long compress(Q&& queue, A&... a)
{
    detail::barrier(static_cast<Q&&>(queue), "compress");
    const unsigned long long folded = device::compress_all(a.native()...);
    device::synchronize();
    return folded;
}

} // namespace shrink_array::device::arcane

// The facade per variant, as the device arrays' (chunk/device/zero_array.h, ...).
namespace shrink_array::zero
{
/// @brief The Arcane facade of a zero device array (device::arcane::Array).
template <typename T, unsigned int S = 1024>
using ArcaneArray = device::arcane::Array<T, S, device::Variant::zero>;
} // namespace shrink_array::zero

namespace shrink_array::zero_any
{
/// @brief The Arcane facade of a zero_any device array (device::arcane::Array).
template <typename T, unsigned int S = 1024>
using ArcaneArray = device::arcane::Array<T, S, device::Variant::zero_any>;
} // namespace shrink_array::zero_any

namespace shrink_array::zero_or_value
{
/// @brief The Arcane facade of a zero_or_value device array (device::arcane::Array).
template <typename T, unsigned int S = 1024>
using ArcaneArray = device::arcane::Array<T, S, device::Variant::zero_or_value>;
} // namespace shrink_array::zero_or_value
