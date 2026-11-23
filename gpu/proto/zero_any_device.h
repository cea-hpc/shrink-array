// Copyright 2026 CEA (www.cea.fr)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file zero_any_device.h
 * @brief Stage-0 prototype of a GPU zero_any array (docs/gpu_design.md): chunk
 *        descriptors that carry their constant, reads with no barrier, writes that
 *        materialise a chunk once per warp and publish it by CAS, end-of-phase
 *        compression by a per-block vote.
 *
 * @details One role per kernel (section 3): a kernel only reads, or only writes, or
 *          is the compression pass. The host separates them (finalize()).
 *
 *          - Read (section 5): `d.buf ? d.buf[i & mask] : d.constant`.
 *          - Write (section 6): the constant onto its own chunk does nothing; otherwise
 *            the lanes of a warp that need the same chunk elect one lane, which takes a
 *            buffer (free stack, else bump on the arena), the group pre-fills it with
 *            the constant, and the elected lane publishes it by CAS (release). The loser
 *            does not retry: it writes into the winner's buffer and its own buffer goes
 *            to the lost list, given back by the next compress() (6.3, first form).
 *          - Arena exhausted (6.4): the write goes to the append-buffer and a flag is
 *            raised; finalize() compresses, grows the arena and replays.
 *          - compress() (section 7): one block per chunk; a uniform materialised chunk
 *            gets its value as constant and its buffer goes to the free stack.
 *
 *          Out of the library's tree on purpose (stage 0): no dependency on chunk/.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_intrinsics.h"

namespace shrink_array::device
{

#if defined(__HIP_PLATFORM_AMD__)
    #define SA_GPU(call) ::shrink_array::device::sa_check((call), #call, __FILE__, __LINE__)
inline void sa_check(hipError_t e, const char* what, const char* file, int line)
{
    if (e != hipSuccess)
    {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + what + ": " +
                                 hipGetErrorString(e));
    }
}
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemset hipMemset
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuGetLastError hipGetLastError
#else
    #define SA_GPU(call) ::shrink_array::device::sa_check((call), #call, __FILE__, __LINE__)
inline void sa_check(cudaError_t e, const char* what, const char* file, int line)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + what + ": " +
                                 cudaGetErrorString(e));
    }
}
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemset cudaMemset
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
#endif

/// @brief Same bits: what the CPU version calls same_value() (keeps -0.0 apart from 0.0,
///        a NaN equal to itself). The comparison of the write's case 1 and of the vote.
template <typename T>
__host__ __device__ inline bool same_bits(const T& a, const T& b) noexcept
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "the prototype handles 4- and 8-byte elements");
    if constexpr (sizeof(T) == 8)
    {
        unsigned long long x, y;
        std::memcpy(&x, &a, 8);
        std::memcpy(&y, &b, 8);
        return x == y;
    }
    else
    {
        unsigned int x, y;
        std::memcpy(&x, &a, 4);
        std::memcpy(&y, &b, 4);
        return x == y;
    }
}

/// @brief A chunk (section 4.1): its buffer, or nullptr and its constant. Trivially
///        copyable, no atomic in its type: the atomics go through the slot's address.
template <typename T>
struct ChunkDesc
{
    T* buf;      ///< nullptr: compressed; otherwise a buffer of S elements
    T  constant; ///< the compressed chunk's value (meaningless while buf is set)
};

/// @brief The device-side counters of an array, read back by the host at a phase end.
struct ArenaCounters
{
    unsigned long long bump;        ///< next arena buffer handed out (past capacity: exhausted)
    long long          free_top;    ///< entries of the free stack (may go below 0 under pops)
    unsigned long long lost;        ///< buffers of lost CAS, parked until compress()
    unsigned long long append_n;    ///< append-buffer entries written (past capacity: lost)
    unsigned long long arena_fail;  ///< writes that found the arena exhausted
    unsigned long long cas_lost;    ///< materialisations that lost the publication CAS
    unsigned long long materialise; ///< materialisations published
};

/**
 * @brief What a kernel gets of an array: a trivially copyable view, passed by value.
 * @tparam T element type (4 or 8 bytes)
 * @tparam S chunk size, a power of two and a multiple of the warp size
 */
template <typename T, int S>
struct View
{
    static_assert((S & (S - 1)) == 0 && S % warp_size == 0, "S: a power of two, a multiple of the warp size");
    static constexpr int  shift = __builtin_ctz(static_cast<unsigned>(S));
    static constexpr long mask  = S - 1;

    ChunkDesc<T>*      desc;          ///< n_chunks descriptors
    long               n;             ///< elements
    T*                 arena;         ///< the current arena: arena_chunks buffers of S elements
    unsigned long long arena_chunks;  ///< its capacity in buffers
    T**                free_stack;    ///< buffers given back by compress()
    T**                lost_list;     ///< buffers of lost CAS
    unsigned long long list_capacity; ///< capacity of each of the two lists
    T**                spare;         ///< one parked buffer per slot (a warp's lost CAS), see materialise()
    unsigned long long spare_slots;   ///< a power of two
    long*              append_index;  ///< append-buffer: element indices
    T*                 append_value;  ///< append-buffer: values
    unsigned long long append_capacity;
    ArenaCounters*     counters;

    /// @brief a(i), in a read kernel (section 5): no barrier, one descriptor per warp.
    __device__ T operator()(long i) const noexcept
    {
        const ChunkDesc<T>& d = desc[i >> shift];
        return d.buf != nullptr ? d.buf[i & mask] : d.constant;
    }

    /// @brief a(i) = v, in a write kernel (section 6.1).
    __device__ void assign(long i, T v) const noexcept;

  private:
    __device__ T*   take_buffer() const noexcept;
    __device__ T**  spare_slot() const noexcept;
    __device__ T*   materialise(long nc, ChunkDesc<T>& d) const noexcept;
    __device__ void append(long i, T v) const noexcept;
};

/// @brief A buffer for a materialisation (6.2 step 2): the free stack, else the arena's bump.
///        No push happens during a write kernel, so the pops need no ABA guard (6.3).
template <typename T, int S>
__device__ T* View<T, S>::take_buffer() const noexcept
{
    const long long top = fetch_add(&counters->free_top, -1LL) - 1;
    if (top >= 0)
    {
        return free_stack[top];
    }
    const unsigned long long b = fetch_add(&counters->bump, 1ULL);
    return b < arena_chunks ? arena + b * S : nullptr;
}

/// @brief This warp's spare slot: the buffer of a lost CAS waits there for the warp's next
///        materialisation (a grid-stride loop moves on to another chunk) instead of being
///        given up until compress(). Two warps sharing a slot only exchange atomically.
template <typename T, int S>
__device__ T** View<T, S>::spare_slot() const noexcept
{
    const unsigned long long thread = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    return spare + ((thread / warp_size) & (spare_slots - 1));
}

/// @brief Park (i, v) in the append-buffer (6.4): replayed by finalize().
template <typename T, int S>
__device__ void View<T, S>::append(long i, T v) const noexcept
{
    const unsigned long long k = fetch_add(&counters->append_n, 1ULL);
    if (k < append_capacity)
    {
        append_index[k] = i;
        append_value[k] = v;
    }
}

/**
 * @brief Materialise chunk @p nc for the lanes of the warp that call it (6.2): one
 *        buffer per chunk and per warp, pre-filled by the lanes that need it, published
 *        by one CAS. Returns the chunk's buffer (the winner's), or nullptr when the arena
 *        is exhausted and no other warp has published one.
 */
template <typename T, int S>
__device__ T* View<T, S>::materialise(long nc, ChunkDesc<T>& d) const noexcept
{
    const int lane    = lane_id();
    LaneMask  pending = active_lanes();
    T*        result  = nullptr;
    // A lane leaves as soon as its own group is done: the masks of the shuffles and
    // ballots below must always contain the calling lane.
    while (((pending >> lane) & 1) != 0)
    {
        const int      leader = first_lane(pending);
        const long     lnc    = shuffle(pending, nc, leader);
        const LaneMask group  = ballot(pending, nc == lnc);
        if (((group >> lane) & 1) != 0)
        {
            // The elected lane looks again before allocating: with scattered indices many
            // warps reach one chunk at once, and a chunk published meanwhile costs no buffer.
            T* existing = nullptr;
            T* mine     = nullptr;
            if (lane == leader)
            {
                existing = load_ptr_acquire(&d.buf);
                if (existing == nullptr)
                {
                    mine = exchange_ptr(spare_slot(), static_cast<T*>(nullptr));
                    mine = mine != nullptr ? mine : take_buffer();
                }
            }
            existing = shuffle_ptr(group, existing, leader);
            mine     = shuffle_ptr(group, mine, leader);
            if (existing != nullptr)
            {
                result = existing;
                sync_lanes(group); // the leader's acquire before the group's writes
            }
            else if (mine != nullptr)
            {
                // 3. pre-fill by the group, coalesced; 4. publish (release), the loser reads the winner.
                const T   c     = d.constant;
                const int rank  = popcount(group & ((LaneMask{1} << lane) - 1));
                const int width = popcount(group);
                for (int k = rank; k < S; k += width)
                {
                    mine[k] = c;
                }
                fence_device();
                sync_lanes(group);
                T* winner = nullptr;
                if (lane == leader)
                {
                    T* found = cas_ptr(&d.buf, static_cast<T*>(nullptr), mine);
                    if (found == nullptr)
                    {
                        winner = mine;
                        fetch_add(&counters->materialise, 1ULL);
                    }
                    else
                    {
                        winner = found;
                        fence_device(); // acquire: the winner's pre-fill before our writes
                        fetch_add(&counters->cas_lost, 1ULL);
                        // Kept for this warp's next chunk; a buffer already parked in the
                        // slot (another warp's) goes to the lost list until compress().
                        T* evicted = exchange_ptr(spare_slot(), mine);
                        if (evicted != nullptr)
                        {
                            const unsigned long long k = fetch_add(&counters->lost, 1ULL);
                            if (k < list_capacity)
                            {
                                lost_list[k] = evicted;
                            }
                        }
                    }
                }
                result = shuffle_ptr(group, winner, leader);
                sync_lanes(group);
            }
            else
            {
                // Arena exhausted: another warp may still have published this chunk.
                result = load_ptr_acquire(&d.buf);
                if (result == nullptr && lane == leader)
                {
                    fetch_add(&counters->arena_fail, 1ULL);
                }
            }
        }
        pending &= ~group;
    }
    return result;
}

template <typename T, int S>
__device__ void View<T, S>::assign(long i, T v) const noexcept
{
    const long    nc  = i >> shift;
    ChunkDesc<T>& d   = desc[nc];
    T*            buf = load_ptr_acquire(&d.buf);
    if (buf == nullptr)
    {
        if (same_bits(v, d.constant))
        {
            return; // 1. the constant onto its own chunk: nothing
        }
        buf = materialise(nc, d); // 2.
        if (buf == nullptr)
        {
            append(i, v); // 6.4
            return;
        }
    }
    buf[i & mask] = v; // 3.
}

// ---------------------------------------------------------------------------
// Kernels of the host operations
// ---------------------------------------------------------------------------

namespace kernels
{

/// @brief Before a compress(): the free stack's top back to >= 0 (pops past empty), and
///        the lost buffers onto the free stack (6.3: they wait for the phase end).
template <typename T, int S>
__global__ void reclaim_lost(View<T, S> v)
{
    const unsigned long long n_lost = v.counters->lost < v.list_capacity ? v.counters->lost : v.list_capacity;
    const long long          top    = v.counters->free_top < 0 ? 0 : v.counters->free_top;
    for (unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x; k < n_lost; k += gridDim.x * blockDim.x)
    {
        v.free_stack[top + static_cast<long long>(k)] = v.lost_list[k];
    }
}

/// @brief Counters after reclaim_lost: the stack holds the lost buffers, the lost list is empty.
__global__ void reclaim_lost_counters(ArenaCounters* c, unsigned long long list_capacity)
{
    const unsigned long long n_lost = c->lost < list_capacity ? c->lost : list_capacity;
    const long long          top    = c->free_top < 0 ? 0 : c->free_top;
    c->free_top                     = top + static_cast<long long>(n_lost);
    c->lost                         = 0;
}

/// @brief Before a compress(), after reclaim_lost_counters: every parked spare onto the free stack.
template <typename T, int S>
__global__ void reclaim_spares(View<T, S> v)
{
    for (unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x; k < v.spare_slots; k += gridDim.x * blockDim.x)
    {
        T* buf = v.spare[k];
        if (buf != nullptr)
        {
            v.spare[k]                                          = nullptr;
            v.free_stack[fetch_add(&v.counters->free_top, 1LL)] = buf;
        }
    }
}

/// @brief The pass (section 7.1): one block per chunk; a uniform materialised chunk takes
///        its value as constant and gives its buffer to the free stack (7.2).
template <typename T, int S>
__global__ void compress(View<T, S> v, long n_chunks)
{
    const long c = blockIdx.x;
    if (c >= n_chunks)
    {
        return;
    }
    ChunkDesc<T>& d   = v.desc[c];
    T*            buf = d.buf;
    if (buf == nullptr)
    {
        return;
    }
    const long valid   = (c + 1) * S <= v.n ? S : v.n - c * S; // the last chunk's tail is not compared
    const T    first   = buf[0];
    bool       uniform = true;
    for (long k = threadIdx.x; k < valid; k += blockDim.x)
    {
        uniform = uniform && same_bits(buf[k], first);
    }
    uniform = __syncthreads_and(uniform) != 0;
    if (uniform && threadIdx.x == 0)
    {
        d.constant          = first;
        d.buf               = nullptr;
        const long long top = fetch_add(&v.counters->free_top, 1LL);
        v.free_stack[top]   = buf;
    }
}

/// @brief Replay of a copy of the append-buffer (6.4), through the same write path: an
///        entry that finds the arena exhausted again goes back to the (emptied) append-buffer.
template <typename T, int S>
__global__ void replay(View<T, S> v, const long* index, const T* value, unsigned long long n_entries)
{
    for (unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x; k < n_entries; k += gridDim.x * blockDim.x)
    {
        v.assign(index[k], value[k]);
    }
}

/// @brief Every element into a flat buffer (to_host, the tests).
template <typename T, int S>
__global__ void expand(View<T, S> v, T* out)
{
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < v.n;
         i += static_cast<long>(gridDim.x) * blockDim.x)
    {
        out[i] = v(i);
    }
}

/// @brief Materialised chunks (stats()).
template <typename T>
__global__ void count_materialised(const ChunkDesc<T>* desc, long n_chunks, unsigned long long* out)
{
    unsigned long long mine = 0;
    for (long c = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; c < n_chunks;
         c += static_cast<long>(gridDim.x) * blockDim.x)
    {
        mine += desc[c].buf != nullptr ? 1 : 0;
    }
    atomicAdd(out, mine);
}

} // namespace kernels

/// @brief What finalize() and stats() report.
struct Stats
{
    long               n_chunks;
    unsigned long long materialised;     ///< chunks holding a buffer now
    unsigned long long arena_buffers;    ///< buffers of every arena
    unsigned long long cas_lost;         ///< since the array's creation
    unsigned long long arena_fail;       ///< since the array's creation
    unsigned long long replays;          ///< finalize() passes that replayed an append-buffer
    unsigned long long append_overflows; ///< writes lost: the append-buffer was full (an error)
};

/**
 * @brief Host side of a device zero_any array: owns the descriptors, the arenas, the
 *        lists and the counters; hands view() to kernels; finalize() ends a write phase.
 */
template <typename T, int S>
class DeviceArray
{
  public:
    /**
     * @param n elements
     * @param arena_chunks initial arena, in buffers of S elements (9.1)
     * @param append_capacity append-buffer entries (6.4)
     */
    DeviceArray(long n, unsigned long long arena_chunks, unsigned long long append_capacity);
    ~DeviceArray();
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    /// @brief The view for a kernel (valid until the next finalize(), which may grow the arena).
    [[nodiscard]] View<T, S> view() const noexcept
    {
        return view_;
    }

    /// @brief End of a write phase: replay the append-buffer if the arena ran out (compress,
    ///        grow, replay, as often as needed), then compress(). Throws if writes were lost.
    void finalize();

    /// @brief The compression pass alone (section 7), between two phases.
    void compress();

    [[nodiscard]] Stats stats() const;

    /// @brief Every element, flat (tests).
    [[nodiscard]] std::vector<T> to_host() const;

    [[nodiscard]] long n_chunks() const noexcept
    {
        return n_chunks_;
    }

  private:
    void                        grow(unsigned long long more_chunks);
    void                        resize_lists(unsigned long long capacity);
    [[nodiscard]] ArenaCounters read_counters() const;

    long               n_chunks_;
    View<T, S>         view_{};
    std::vector<T*>    arenas_;
    unsigned long long total_buffers_ = 0;
    unsigned long long replays_       = 0;
    unsigned long long overflows_     = 0;
};

template <typename T, int S>
DeviceArray<T, S>::DeviceArray(long n, unsigned long long arena_chunks, unsigned long long append_capacity)
    : n_chunks_((n + S - 1) / S)
{
    view_.n = n;
    SA_GPU(gpuMalloc(&view_.desc, sizeof(ChunkDesc<T>) * n_chunks_));
    SA_GPU(gpuMemset(view_.desc, 0, sizeof(ChunkDesc<T>) * n_chunks_)); // {nullptr, T{}} for arithmetic T
    view_.spare_slots = 1ULL << 16; // more than the warps resident on an H100 or an MI300A
    SA_GPU(gpuMalloc(&view_.spare, sizeof(T*) * view_.spare_slots));
    SA_GPU(gpuMemset(view_.spare, 0, sizeof(T*) * view_.spare_slots));
    SA_GPU(gpuMalloc(&view_.counters, sizeof(ArenaCounters)));
    SA_GPU(gpuMemset(view_.counters, 0, sizeof(ArenaCounters)));
    view_.append_capacity = append_capacity;
    SA_GPU(gpuMalloc(&view_.append_index, sizeof(long) * (append_capacity > 0 ? append_capacity : 1)));
    SA_GPU(gpuMalloc(&view_.append_value, sizeof(T) * (append_capacity > 0 ? append_capacity : 1)));
    grow(arena_chunks > 0 ? arena_chunks : 1);
}

template <typename T, int S>
DeviceArray<T, S>::~DeviceArray()
{
    for (T* a : arenas_)
    {
        (void)gpuFree(a);
    }
    (void)gpuFree(view_.desc);
    (void)gpuFree(view_.counters);
    (void)gpuFree(view_.append_index);
    (void)gpuFree(view_.append_value);
    (void)gpuFree(view_.free_stack);
    (void)gpuFree(view_.lost_list);
    (void)gpuFree(view_.spare);
}

/// @brief The free stack and the lost list hold at most every buffer: capacity follows the arenas.
template <typename T, int S>
void DeviceArray<T, S>::resize_lists(unsigned long long capacity)
{
    T** stack = nullptr;
    T** lost  = nullptr;
    SA_GPU(gpuMalloc(&stack, sizeof(T*) * capacity));
    SA_GPU(gpuMalloc(&lost, sizeof(T*) * capacity));
    if (view_.free_stack != nullptr)
    {
        const ArenaCounters c   = read_counters();
        const long long     top = c.free_top < 0 ? 0 : c.free_top;
        SA_GPU(gpuMemcpy(stack, view_.free_stack, sizeof(T*) * top, gpuMemcpyDeviceToDevice));
        (void)gpuFree(view_.free_stack);
        (void)gpuFree(view_.lost_list);
    }
    view_.free_stack    = stack;
    view_.lost_list     = lost;
    view_.list_capacity = capacity;
}

/// @brief A new arena of @p more_chunks buffers becomes the bump target (9.1): the old
///        buffers do not move. Called between phases, after a compress().
template <typename T, int S>
void DeviceArray<T, S>::grow(unsigned long long more_chunks)
{
    T* a = nullptr;
    SA_GPU(gpuMalloc(&a, sizeof(T) * S * more_chunks));
    arenas_.push_back(a);
    total_buffers_ += more_chunks;
    resize_lists(total_buffers_);
    view_.arena        = a;
    view_.arena_chunks = more_chunks;
    ArenaCounters c    = read_counters();
    c.bump             = 0;
    SA_GPU(gpuMemcpy(view_.counters, &c, sizeof(c), gpuMemcpyHostToDevice));
}

template <typename T, int S>
ArenaCounters DeviceArray<T, S>::read_counters() const
{
    ArenaCounters c{};
    SA_GPU(gpuMemcpy(&c, view_.counters, sizeof(c), gpuMemcpyDeviceToHost));
    return c;
}

template <typename T, int S>
void DeviceArray<T, S>::compress()
{
    SA_GPU(gpuDeviceSynchronize());
    kernels::reclaim_lost<<<64, 256>>>(view_);
    kernels::reclaim_lost_counters<<<1, 1>>>(view_.counters, view_.list_capacity);
    kernels::reclaim_spares<<<64, 256>>>(view_);
    constexpr int block = S < 256 ? S : 256;
    if (n_chunks_ > 0)
    {
        kernels::compress<<<static_cast<unsigned>(n_chunks_), block>>>(view_, n_chunks_);
    }
    SA_GPU(gpuGetLastError());
    SA_GPU(gpuDeviceSynchronize());
}

template <typename T, int S>
void DeviceArray<T, S>::finalize()
{
    SA_GPU(gpuDeviceSynchronize());
    for (;;)
    {
        ArenaCounters c = read_counters();
        if (c.append_n == 0)
        {
            break;
        }
        // 6.4: fold back what can be, give the arena room for every parked write, replay.
        const unsigned long long n_entries = c.append_n < view_.append_capacity ? c.append_n : view_.append_capacity;
        overflows_ += c.append_n - n_entries;
        compress();
        // At most one buffer per chunk not materialised yet, whatever the number of parked
        // writes; a quarter more for the CAS the replay itself may lose.
        const unsigned long long unmaterialised = static_cast<unsigned long long>(n_chunks_) - stats().materialised;
        const unsigned long long needed         = n_entries < unmaterialised ? n_entries : unmaterialised;
        grow(needed + needed / 4 + warp_size);
        c          = read_counters();
        c.append_n = 0;
        SA_GPU(gpuMemcpy(view_.counters, &c, sizeof(c), gpuMemcpyHostToDevice));
        // The replay may append again (to the emptied append-buffer): it reads from a copy.
        long* idx = nullptr;
        T*    val = nullptr;
        SA_GPU(gpuMalloc(&idx, sizeof(long) * n_entries));
        SA_GPU(gpuMalloc(&val, sizeof(T) * n_entries));
        SA_GPU(gpuMemcpy(idx, view_.append_index, sizeof(long) * n_entries, gpuMemcpyDeviceToDevice));
        SA_GPU(gpuMemcpy(val, view_.append_value, sizeof(T) * n_entries, gpuMemcpyDeviceToDevice));
        kernels::replay<<<256, 256>>>(view_, idx, val, n_entries);
        SA_GPU(gpuGetLastError());
        SA_GPU(gpuDeviceSynchronize());
        (void)gpuFree(idx);
        (void)gpuFree(val);
        ++replays_;
    }
    compress();
    if (overflows_ != 0)
    {
        throw std::runtime_error("DeviceArray::finalize: " + std::to_string(overflows_) +
                                 " writes lost (append-buffer full)");
    }
}

template <typename T, int S>
Stats DeviceArray<T, S>::stats() const
{
    unsigned long long* d_count = nullptr;
    SA_GPU(gpuMalloc(&d_count, sizeof(unsigned long long)));
    SA_GPU(gpuMemset(d_count, 0, sizeof(unsigned long long)));
    kernels::count_materialised<<<256, 256>>>(view_.desc, n_chunks_, d_count);
    unsigned long long materialised = 0;
    SA_GPU(gpuMemcpy(&materialised, d_count, sizeof(materialised), gpuMemcpyDeviceToHost));
    (void)gpuFree(d_count);
    const ArenaCounters c = read_counters();
    return Stats{n_chunks_, materialised, total_buffers_, c.cas_lost, c.arena_fail, replays_, overflows_};
}

template <typename T, int S>
std::vector<T> DeviceArray<T, S>::to_host() const
{
    std::vector<T> out(static_cast<std::size_t>(view_.n));
    T*             d_out = nullptr;
    SA_GPU(gpuMalloc(&d_out, sizeof(T) * view_.n));
    kernels::expand<<<1024, 256>>>(view_, d_out);
    SA_GPU(gpuGetLastError());
    SA_GPU(gpuMemcpy(out.data(), d_out, sizeof(T) * view_.n, gpuMemcpyDeviceToHost));
    (void)gpuFree(d_out);
    return out;
}

} // namespace shrink_array::device
