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
 * @file arena.h
 * @brief The buffers of a device array (docs/gpu_design.md, 6.3 and 9.1): arenas of
 *        buffers of S elements, each split into n_shards shards with their own bump and
 *        free-stack segment, taken by write kernels and given back by the compression pass.
 *
 * @details Stage 1, experimental (see chunk_desc.h), from the prototype
 *          (gpu/proto/zero_any_device.h), where sharding the counters took the folds at
 *          S = 256 from 36% to 62% of the stream bandwidth on the GH200.
 *
 *          - A write kernel takes (ArenaRef::take): in each arena, the newest first, the
 *            chunk's home shard (its index modulo n_shards) and then the next ones; in a
 *            shard its free stack, else its bump. Pops only: no ABA guard (6.3).
 *          - The compression pass gives back (ArenaRef::give_back): a buffer goes to the
 *            shard of its address, so a segment never overflows. Pushes only, after
 *            DeviceArena::normalise() (the failed pops of a write kernel leave a top below
 *            zero and a bump past the shard's length).
 *          - Between phases the host grows (DeviceArena::grow): a new arena, the old buffers
 *            do not move, the descriptors that point into them stay valid.
 *
 *          A write kernel and the compression pass never run together (one role per kernel,
 *          docs/gpu_design.md, section 3): the kernel boundary orders a push before the pop
 *          that reads it, and the atomics are relaxed (atomics.h).
 */

#include <cstddef>     // std::size_t
#include <stdexcept>   // std::runtime_error
#include <type_traits> // std::is_trivially_copyable_v
#include <vector>      // std::vector

#include "chunk/device/atomics.h"       // fetch_add, load
#include "chunk/device/backend.h"       // allocate, release, zero, copy, synchronize
#include "chunk/internal/host_device.h" // CHUNK_HD

namespace shrink_array::device
{

/// Shards per arena: as many bumps and free-stack tops, on as many cache lines.
inline constexpr int n_shards = 128;

/// Most arenas a DeviceArena may have (grow() adds one).
inline constexpr int max_arenas = 32;

/// @brief One shard's counters, alone on a 128-byte line.
struct alignas(128) Shard
{
    long long          top;  ///< entries of the shard's free-stack segment (below 0 after failed pops)
    unsigned long long bump; ///< next never-used buffer of the shard (past its length: exhausted)
};

/// @brief One arena as kernels see it: n buffers of S elements from @p base.
template <typename T, unsigned int S>
struct ArenaDesc
{
    T*                 base;
    unsigned long long n;            ///< buffers
    unsigned long long shard_size;   ///< buffers per shard (the last shards may hold fewer, or none)
    unsigned long long stack_offset; ///< first free-stack entry of this arena
    Shard*             shards;       ///< n_shards counters

    /// @brief Buffers of shard @p s.
    [[nodiscard]] CHUNK_HD unsigned long long shard_len(int s) const noexcept
    {
        const unsigned long long first = static_cast<unsigned long long>(s) * shard_size;
        return first >= n ? 0 : (n - first < shard_size ? n - first : shard_size);
    }

    /// @brief Whether @p buf is one of this arena's buffers.
    [[nodiscard]] CHUNK_HD bool owns(const T* buf) const noexcept
    {
        return buf >= base && buf < base + n * S;
    }
};

/// @brief What a kernel gets of the arenas: trivially copyable, passed by value.
template <typename T, unsigned int S>
struct ArenaRef
{
    const ArenaDesc<T, S>* arenas     = nullptr; ///< n_arenas descriptors, the newest last
    int                    n_arenas   = 0;
    T**                    free_stack = nullptr; ///< one segment per shard of every arena

    /// @brief A buffer for chunk @p home, or nullptr when every arena is exhausted. Write
    ///        kernels only. Its contents are whatever its last owner left.
    [[nodiscard]] __device__ T* take(long long home) const noexcept;

    /// @brief @p buf back to the free-stack segment of its shard. The compression pass
    ///        only, after DeviceArena::normalise(); no take() runs meanwhile.
    __device__ void give_back(T* buf) const noexcept;
};

/**
 * @brief The host side: owns the arenas, their shards and the free stack; hands ref() to
 *        kernels.
 * @tparam T element type, trivially copyable
 * @tparam S buffer size in elements (the chunk size)
 */
template <typename T, unsigned int S>
class DeviceArena
{
    static_assert(std::is_trivially_copyable_v<T>, "DeviceArena: buffers of trivially copyable elements");

  public:
    /// @brief A first arena of @p n_buffers buffers (none if 0: take() returns nullptr).
    explicit DeviceArena(unsigned long long n_buffers);
    ~DeviceArena();
    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&&)                 = delete;
    DeviceArena& operator=(DeviceArena&&)      = delete;

    /// @brief For a kernel; valid until the next grow().
    [[nodiscard]] ArenaRef<T, S> ref() const noexcept
    {
        return ref_;
    }

    /// @brief A new arena of @p more buffers, tried first by take(). Between phases only.
    void grow(unsigned long long more);

    /// @brief Every shard's top back to >= 0 and bump clamped to its length (a kernel):
    ///        before a kernel that gives back. Asynchronous.
    void normalise();

    /// @brief Buffers in every arena.
    [[nodiscard]] unsigned long long buffers() const noexcept
    {
        return total_;
    }

    /// @brief Arenas so far.
    [[nodiscard]] int arenas() const noexcept
    {
        return ref_.n_arenas;
    }

    /// @brief Buffers taken and not given back (normalises, waits for the device).
    [[nodiscard]] unsigned long long in_use();

  private:
    std::vector<T*>              bases_;
    std::vector<Shard*>          shards_;
    std::vector<ArenaDesc<T, S>> descs_;
    ArenaDesc<T, S>*             d_descs_ = nullptr; ///< max_arenas slots on the device
    unsigned long long           total_   = 0;
    ArenaRef<T, S>               ref_{};
};

namespace kernels
{

/// @brief DeviceArena::normalise(): one thread per shard of every arena.
template <typename T, unsigned int S>
__global__ void normalise_shards(ArenaRef<T, S> r)
{
    for (int a = 0; a < r.n_arenas; ++a)
    {
        const ArenaDesc<T, S>& ar = r.arenas[a];
        for (int s = static_cast<int>(threadIdx.x); s < n_shards; s += static_cast<int>(blockDim.x))
        {
            Shard&                   sh  = ar.shards[s];
            const unsigned long long len = ar.shard_len(s);
            sh.top                       = sh.top < 0 ? 0 : sh.top;
            sh.bump                      = sh.bump > len ? len : sh.bump;
        }
    }
}

} // namespace kernels

// ---------------------------------------------------------------------------
// ArenaRef

/**
 * @details The home shard is tried with its atomics directly (two at most, on counters that
 *          the neighbouring chunks do not share); the other shards only after a look, so
 *          that a nearly full arena costs loads, not atomics.
 */
template <typename T, unsigned int S>
[[nodiscard]] __device__ T* ArenaRef<T, S>::take(long long home) const noexcept
{
    const int s0 = static_cast<int>(static_cast<unsigned long long>(home) % n_shards);
    for (int a = n_arenas - 1; a >= 0; --a)
    {
        const ArenaDesc<T, S>& ar = arenas[a];
        for (int k = 0; k < n_shards; ++k)
        {
            const int                s   = (s0 + k) % n_shards;
            Shard&                   sh  = ar.shards[s];
            const unsigned long long len = ar.shard_len(s);
            if (k > 0 && load(&sh.top) <= 0 && load(&sh.bump) >= len)
            {
                continue;
            }
            const long long top = fetch_add(&sh.top, -1LL) - 1;
            if (top >= 0)
            {
                return free_stack[ar.stack_offset + s * ar.shard_size + static_cast<unsigned long long>(top)];
            }
            const unsigned long long b = fetch_add(&sh.bump, 1ULL);
            if (b < len)
            {
                return ar.base + (s * ar.shard_size + b) * S;
            }
        }
    }
    return nullptr;
}

template <typename T, unsigned int S>
__device__ void ArenaRef<T, S>::give_back(T* buf) const noexcept
{
    for (int a = 0; a < n_arenas; ++a)
    {
        const ArenaDesc<T, S>& ar = arenas[a];
        if (ar.owns(buf))
        {
            const unsigned long long idx = static_cast<unsigned long long>(buf - ar.base) / S;
            const unsigned long long s   = idx / ar.shard_size;
            const long long          top = fetch_add(&ar.shards[s].top, 1LL);
            free_stack[ar.stack_offset + s * ar.shard_size + static_cast<unsigned long long>(top)] = buf;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceArena

template <typename T, unsigned int S>
DeviceArena<T, S>::DeviceArena(unsigned long long n_buffers)
{
    d_descs_    = allocate<ArenaDesc<T, S>>(max_arenas);
    ref_.arenas = d_descs_;
    grow(n_buffers);
}

template <typename T, unsigned int S>
DeviceArena<T, S>::~DeviceArena()
{
    for (T* b : bases_)
    {
        release(b);
    }
    for (Shard* sh : shards_)
    {
        release(sh);
    }
    release(d_descs_);
    release(ref_.free_stack);
}

/// @details The free stack grows with the arenas: the old segments are copied at the same
///          offsets (they may hold buffers given back), the new arena's segment follows.
template <typename T, unsigned int S>
void DeviceArena<T, S>::grow(unsigned long long more)
{
    if (more == 0)
    {
        return;
    }
    if (descs_.size() >= static_cast<std::size_t>(max_arenas))
    {
        throw std::runtime_error("shrink_array::device::DeviceArena::grow: more than max_arenas arenas");
    }
    T*     base   = allocate<T>(static_cast<std::size_t>(more) * S);
    Shard* shards = allocate<Shard>(n_shards);
    zero(shards, n_shards);
    T** stack = allocate<T*>(static_cast<std::size_t>(total_ + more));
    if (ref_.free_stack != nullptr)
    {
        copy(stack, ref_.free_stack, static_cast<std::size_t>(total_));
        release(ref_.free_stack);
    }
    bases_.push_back(base);
    shards_.push_back(shards);
    descs_.push_back(ArenaDesc<T, S>{base, more, (more + n_shards - 1) / n_shards, total_, shards});
    total_ += more;
    copy(d_descs_, descs_.data(), descs_.size());
    ref_.free_stack = stack;
    ref_.n_arenas   = static_cast<int>(descs_.size());
}

template <typename T, unsigned int S>
void DeviceArena<T, S>::normalise()
{
    if (ref_.n_arenas > 0)
    {
        kernels::normalise_shards<<<1, n_shards>>>(ref_);
    }
}

template <typename T, unsigned int S>
[[nodiscard]] unsigned long long DeviceArena<T, S>::in_use()
{
    normalise();
    synchronize();
    unsigned long long used = 0;
    std::vector<Shard> h(n_shards);
    for (std::size_t a = 0; a < descs_.size(); ++a)
    {
        copy(h.data(), shards_[a], n_shards);
        for (int s = 0; s < n_shards; ++s)
        {
            used += h[static_cast<std::size_t>(s)].bump -
                    static_cast<unsigned long long>(h[static_cast<std::size_t>(s)].top);
        }
    }
    return used;
}

} // namespace shrink_array::device
