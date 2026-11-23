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
 *            the constant, and the elected lane publishes it by CAS (release). Between
 *            warps, Coordination::race lets every warp pre-fill and the CAS elect one
 *            (the loser parks its buffer and writes into the winner's);
 *            Coordination::claim lets the first warp claim the chunk and the others wait
 *            for its publication (bounded, then the append-buffer).
 *          - Chunk-aligned write (block_write, kernels::for_each_chunk): one block per
 *            chunk votes, takes one buffer and writes values and constant in one pass.
 *          - Arena exhausted (6.4): the write goes to the append-buffer and a flag is
 *            raised; finalize() compresses, grows the arena and replays.
 *          - compress() (section 7): one block per chunk; a uniform materialised chunk
 *            gets its value as constant and its buffer goes to the free stack.
 *
 *          Out of the library's tree on purpose (stage 0): no dependency on chunk/.
 */

#pragma once

#include <cstdint>
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
    T* buf;      ///< nullptr: compressed; busy_mark(): being claimed; otherwise a buffer of S elements
    T  constant; ///< the compressed chunk's value (meaningful until a buffer is published)
};

/// @brief The value of `buf` while a warp holds the claim of a chunk (Coordination::claim):
///        never a buffer address (misaligned). Only write kernels can see it: the claim
///        ends, by a publication or a release, before the kernel does.
template <typename T>
__host__ __device__ inline T* busy_mark() noexcept
{
    return reinterpret_cast<T*>(std::uintptr_t{1});
}

/// @brief How the warps that need one compressed chunk in a write kernel coordinate.
enum class Coordination : int
{
    /// Every warp pre-fills its own buffer, the CAS elects one (6.2): no wait, but S/32
    /// warps per chunk may each pre-fill a buffer for nothing on a first write.
    race,
    /// The first warp claims the chunk by a CAS to busy_mark(), pre-fills and publishes;
    /// the others wait for the publication (a bounded spin: past it, the write goes to the
    /// append-buffer, so a wait that cannot end -- lanes of one AMD wavefront on divergent
    /// paths -- costs a replay, not a deadlock). One pre-fill per chunk.
    claim,
};

/// @brief The device-side counters of an array, read back by the host at a phase end.
struct ArenaCounters
{
    unsigned long long lost;        ///< buffers of lost CAS, parked until compress()
    unsigned long long append_n;    ///< append-buffer entries written (past capacity: lost)
    unsigned long long arena_fail;  ///< writes that found the arena exhausted
    unsigned long long cas_lost;    ///< materialisations that lost the publication CAS
    unsigned long long materialise; ///< materialisations published
    unsigned long long claim_wait;  ///< Coordination::claim: groups that waited for another warp's claim
    unsigned long long claim_spin;  ///< Coordination::claim: waits given up (the write was appended)
};

/// @brief Number of shards of an arena: its bump and its free stack are split into that many
///        independent counters, so that the atomics of a phase spread over as many addresses
///        (one global counter serialised millions of materialisations on the GH200).
#ifndef PROTO_SHARDS
    #define PROTO_SHARDS 128
#endif
inline constexpr int n_shards = PROTO_SHARDS;


/// @brief One shard's counters, alone on a 128-byte line.
struct alignas(128) Shard
{
    long long          top;  ///< entries of the shard's free-stack segment (may go below 0 under pops)
    unsigned long long bump; ///< next never-used buffer of the shard (past its length: exhausted)
};

/// @brief An arena as kernels see it: its buffers, split into n_shards contiguous shards,
///        each owning the free-stack segment of its buffers (a buffer always goes back to the
///        shard of its address, so a segment never overflows).
template <typename T>
struct ArenaDesc
{
    T*                 base;
    unsigned long long n;            ///< buffers
    unsigned long long shard_size;   ///< buffers per shard (the last ones may hold fewer, or none)
    unsigned long long stack_offset; ///< first free-stack entry of this arena
    Shard*             shards;       ///< n_shards counters

    __host__ __device__ unsigned long long shard_len(int s) const noexcept
    {
        const unsigned long long first = static_cast<unsigned long long>(s) * shard_size;
        return first >= n ? 0 : (n - first < shard_size ? n - first : shard_size);
    }
};

/// @brief Most arenas an array may have (grow() adds one per replay).
inline constexpr int max_arenas = 32;

/// @brief How assign() reads `buf` before writing into a materialised chunk.
enum class WriteLoad : int
{
    /// Device-scope acquire (6.2 as written): formally right, but a compiler and memory
    /// barrier per write, so each write waits for its descriptor load (HBM latency) alone.
    acquire,
    /// Relaxed device-scope load, no barrier: the store into `buf[i & mask]` depends on the
    /// loaded address and cannot be performed before the load returns, and the publication
    /// of `buf` came after a device fence over its pre-fill: the store lands after the
    /// pre-fill (dependency ordering, as Linux's rcu_dereference; not stated by the PTX
    /// memory model, hence measured as an option).
    relaxed,
    /// A plain load, which L1 may serve: a stale nullptr or busy mark only sends the write
    /// to the slow path, which loads again with acquire; a stale buffer cannot exist (a
    /// buffer is published at most once per write kernel, and L1 starts empty per kernel).
    /// Same dependency argument as relaxed.
    plain,
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

    ChunkDesc<T>*      desc;   ///< n_chunks descriptors
    long               n;      ///< elements
    ArenaDesc<T>*      arenas; ///< n_arenas arenas, the newest last
    int                n_arenas;
    T**                free_stack;    ///< buffers given back by compress(), one segment per shard
    T**                lost_list;     ///< buffers of lost CAS
    unsigned long long list_capacity; ///< capacity of each of the two lists
    T**                spare;         ///< one parked buffer per slot (a warp's lost CAS), see materialise()
    unsigned long long spare_slots;   ///< a power of two
    long*              append_index;  ///< append-buffer: element indices
    T*                 append_value;  ///< append-buffer: values
    unsigned long long append_capacity;
    ArenaCounters*     counters;
    Coordination       coord;          ///< how assign() coordinates the warps of one chunk
    unsigned           claim_spin_max; ///< Coordination::claim: backoff rounds before giving up
    WriteLoad          write_load;     ///< how assign() reads `buf`

    /// @brief a(i), in a read kernel (section 5): no barrier, one descriptor per warp.
    __device__ T operator()(long i) const noexcept
    {
        const ChunkDesc<T>& d = desc[i >> shift];
        return d.buf != nullptr ? d.buf[i & mask] : d.constant;
    }

    /// @brief a(i) = v, in a write kernel (section 6.1).
    __device__ void assign(long i, T v) const noexcept;

    /**
     * @brief a(i) = f(i) for every i of [@p lo, @p hi), a part of chunk @p c, called by every
     *        thread of a block of @p BLOCK threads (the chunk-aligned loop of a facade,
     *        section 7.3 mitigation 2): one vote, one buffer, no pre-fill of the elements
     *        written, f evaluated once per element.
     *
     * @details The facade's loop gives each chunk to one block, so the publication CAS is
     *          not raced in the common case; it stays (two adjacent ranges may share a chunk)
     *          and the loser writes into the winner's buffer.
     */
    template <int BLOCK, typename F>
    __device__ void block_write(long c, long lo, long hi, const F& f) const noexcept;

    /// @brief @p buf back to the free-stack segment of its shard (compress() only: no pop runs).
    __device__ void give_back(T* buf) const noexcept;

  private:
    __device__ T*   take_buffer(long home) const noexcept;
    __device__ T**  spare_slot() const noexcept;
    __device__ void park(T* lost) const noexcept;
    __device__ T*   materialise(long nc, ChunkDesc<T>& d) const noexcept;
    __device__ void group_by_chunk(long nc, LaneMask warp, LaneMask& group, int& leader) const noexcept;
    __device__ T*   materialise_race(long nc, ChunkDesc<T>& d) const noexcept;
    __device__ T*   materialise_claim(long nc, ChunkDesc<T>& d) const noexcept;
    __device__ void prefill(T* buf, T c, LaneMask group, int leader, int lane) const noexcept;
    __device__ T*   take_spare() const noexcept;
    __device__ void append(long i, T v) const noexcept;
};

/**
 * @brief A buffer for the materialisation of chunk @p home (6.2 step 2): in each arena, the
 *        newest first, the chunk's home shard (home % n_shards) and then the next ones; in a
 *        shard, its free stack, else its bump. No push happens during a write kernel, so the
 *        pops need no ABA guard (6.3).
 *
 * @details The home shard is tried with its atomics directly (two at most, on counters that
 *          the neighbouring chunks do not share); the other shards only after a look, so that
 *          a nearly full arena costs loads, not atomics.
 */
template <typename T, int S>
__device__ T* View<T, S>::take_buffer(long home) const noexcept
{
    const int s0 = static_cast<int>(static_cast<unsigned long>(home) % n_shards);
    for (int a = n_arenas - 1; a >= 0; --a)
    {
        const ArenaDesc<T>& ar = arenas[a];
        for (int k = 0; k < n_shards; ++k)
        {
            const int                s   = (s0 + k) % n_shards;
            Shard&                   sh  = ar.shards[s];
            const unsigned long long len = ar.shard_len(s);
            if (k > 0 && load_relaxed(&sh.top) <= 0 &&
                *reinterpret_cast<const volatile unsigned long long*>(&sh.bump) >= len)
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

template <typename T, int S>
__device__ void View<T, S>::give_back(T* buf) const noexcept
{
    for (int a = 0; a < n_arenas; ++a)
    {
        const ArenaDesc<T>& ar = arenas[a];
        if (buf >= ar.base && buf < ar.base + ar.n * S)
        {
            const unsigned long long idx = static_cast<unsigned long long>(buf - ar.base) / S;
            const unsigned long long s   = idx / ar.shard_size;
            const long long          top = fetch_add(&ar.shards[s].top, 1LL);
            free_stack[ar.stack_offset + s * ar.shard_size + static_cast<unsigned long long>(top)] = buf;
            return;
        }
    }
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

/// @brief The buffer of a lost CAS: kept in this warp's spare slot for its next
///        materialisation; a buffer already parked there (another warp's) goes to the lost
///        list until compress().
template <typename T, int S>
__device__ void View<T, S>::park(T* lost) const noexcept
{
    T* evicted = exchange_ptr(spare_slot(), lost);
    if (evicted != nullptr)
    {
        const unsigned long long k = fetch_add(&counters->lost, 1ULL);
        if (k < list_capacity)
        {
            lost_list[k] = evicted;
        }
    }
}

/// @brief Step 3 of 6.2: the lanes of @p group fill @p buf with @p c, coalesced, then make
///        it visible device-wide before the publication (every lane fences its own stores,
///        concurrently: a fence by the leader alone after the warp barrier measured slower
///        on the GH200, the fences then running one after the other).
template <typename T, int S>
__device__ void View<T, S>::prefill(T* buf, T c, LaneMask group, int leader, int lane) const noexcept
{
    (void)leader;
    const int rank  = popcount(group & ((LaneMask{1} << lane) - 1));
    const int width = popcount(group);
    for (int k = rank; k < S; k += width)
    {
        buf[k] = c;
    }
    fence_device();
    sync_lanes(group);
}

/// @brief The buffer parked in this warp's spare slot, if any.
template <typename T, int S>
__device__ T* View<T, S>::take_spare() const noexcept
{
    return exchange_ptr(spare_slot(), static_cast<T*>(nullptr));
}

/**
 * @brief Coordination::claim, for the lanes of the warp that call it: the first warp to
 *        need a chunk claims it (CAS nullptr -> busy_mark()), pre-fills a buffer and
 *        publishes it; the others wait for the publication. Returns the chunk's buffer, or
 *        nullptr (arena exhausted, or the wait given up): the caller appends.
 *
 * @details Publish before waiting: every group of the warp first makes one attempt
 *          without waiting (phase 1), the groups that claimed pre-fill and publish (phase
 *          2), and only then do the others wait (phase 3), never claiming. A warp therefore
 *          never holds a claim while one of its lanes waits: with SIMT reconvergence, a
 *          claim holder blocked behind a waiting lane of its own warp closes a cycle with
 *          another warp in the symmetric state (a warp over two chunks, measured: waits
 *          that only the bound ended). The holder waits for nothing between its CAS and its
 *          publication and is resident: the waits end. The bound covers what is left, two
 *          assign() calls on divergent paths of one wavefront (AMD, no independent thread
 *          scheduling).
 *
 *          The grouping (group_by_chunk) is warp-uniform: the warp-wide shuffles use one
 *          mask, each lane reading its own leader.
 */
template <typename T, int S>
__device__ T* View<T, S>::materialise_claim(long nc, ChunkDesc<T>& d) const noexcept
{
    T* const       busy = busy_mark<T>();
    const int      lane = lane_id();
    const LaneMask warp = active_lanes();
    LaneMask       group;
    int            leader;
    group_by_chunk(nc, warp, group, leader);

    // Phase 1: one attempt per group, no wait.
    T* found = nullptr;
    T* mine  = nullptr;
    if (lane == leader)
    {
        found = load_ptr_acquire(&d.buf);
        if (found == nullptr)
        {
            found = cas_ptr(&d.buf, static_cast<T*>(nullptr), busy);
            if (found == nullptr)
            {
                mine = take_spare();
                mine = mine != nullptr ? mine : take_buffer(nc);
                if (mine == nullptr)
                {
                    exchange_ptr(&d.buf, static_cast<T*>(nullptr)); // arena exhausted: release
                    fetch_add(&counters->arena_fail, 1ULL);
                }
            }
        }
    }
    found = shuffle_ptr(warp, found, leader);
    mine  = shuffle_ptr(warp, mine, leader);

    // Phase 2: the claims of the warp published.
    if (mine != nullptr)
    {
        prefill(mine, d.constant, group, leader, lane);
        if (lane == leader)
        {
            exchange_ptr(&d.buf, mine); // after prefill()'s fence: a release store
            fetch_add(&counters->materialise, 1ULL);
        }
    }
    sync_lanes(warp);
    if (mine != nullptr)
    {
        return mine;
    }

    // Phase 3: wait for another warp's claim; a released claim (arena exhausted) is not
    // taken over: the write is appended.
    if (found == busy)
    {
        if (lane == leader)
        {
            unsigned spins = 0;
            do
            {
                if (++spins > claim_spin_max)
                {
                    found = nullptr;
                    fetch_add(&counters->claim_spin, 1ULL);
                    break;
                }
                backoff();
                found = load_ptr_acquire(&d.buf);
            } while (found == busy);
            fetch_add(&counters->claim_wait, 1ULL);
        }
        found = shuffle_ptr(group, found, leader);
    }
    sync_lanes(group); // the leader's acquire before the group's writes
    return found;
}

/**
 * @brief The lanes of the warp that call it, grouped by chunk: each lane gets its group and
 *        the group's leader (its lowest lane).
 *
 * @details Every lane goes through the loop as many times as the others (pending is
 *          uniform), so the warp-wide shuffles and ballots have one mask. In the first race
 *          version each lane left the loop once its own group was done, and the shuffles of
 *          the next iterations ran with masks that changed while other lanes sat inside
 *          their group's work.
 */
template <typename T, int S>
__device__ void View<T, S>::group_by_chunk(long nc, LaneMask warp, LaneMask& group, int& leader) const noexcept
{
    const int lane    = lane_id();
    LaneMask  pending = warp;
    group             = 0;
    leader            = lane;
    while (pending != 0)
    {
        const int      l   = first_lane(pending);
        const long     lnc = shuffle(warp, nc, l);
        const LaneMask g   = ballot(warp, ((pending >> lane) & 1) != 0 && nc == lnc);
        if (((g >> lane) & 1) != 0)
        {
            group  = g;
            leader = l;
        }
        pending &= ~g;
    }
}

/**
 * @brief Coordination::race (6.2 as written) for the lanes of the warp that call it: per
 *        group, one buffer, pre-filled, published by one CAS; the loser parks its buffer
 *        and writes into the winner's. Returns the chunk's buffer, or nullptr when the arena
 *        is exhausted and no other warp has published one.
 */
template <typename T, int S>
__device__ T* View<T, S>::materialise_race(long nc, ChunkDesc<T>& d) const noexcept
{
    const int      lane = lane_id();
    const LaneMask warp = active_lanes();
    LaneMask       group;
    int            leader;
    group_by_chunk(nc, warp, group, leader);

    // The elected lane looks again before allocating: with scattered indices many
    // warps reach one chunk at once, and a chunk published meanwhile costs no buffer.
    T* existing = nullptr;
    T* mine     = nullptr;
    if (lane == leader)
    {
        existing = load_ptr_acquire(&d.buf);
        if (existing == nullptr)
        {
            mine = take_spare();
            mine = mine != nullptr ? mine : take_buffer(nc);
        }
    }
    existing = shuffle_ptr(warp, existing, leader);
    mine     = shuffle_ptr(warp, mine, leader);

    T* winner = existing;
    if (existing == nullptr && mine != nullptr)
    {
        prefill(mine, d.constant, group, leader, lane);
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
                park(mine);
            }
        }
    }
    else if (existing == nullptr && lane == leader)
    {
        // Arena exhausted: another warp may still have published this chunk.
        winner = load_ptr_acquire(&d.buf);
        if (winner == nullptr)
        {
            fetch_add(&counters->arena_fail, 1ULL);
        }
    }
    winner = shuffle_ptr(warp, winner, leader);
    sync_lanes(warp); // the leaders' acquire before the writes of their groups
    return winner;
}

/// @brief Materialise chunk @p nc for the lanes of the warp that call it (6.2), as `coord`
///        says. Returns the chunk's buffer, or nullptr: the caller appends.
template <typename T, int S>
__device__ T* View<T, S>::materialise(long nc, ChunkDesc<T>& d) const noexcept
{
    return coord == Coordination::claim ? materialise_claim(nc, d) : materialise_race(nc, d);
}

template <typename T, int S>
__device__ void View<T, S>::assign(long i, T v) const noexcept
{
    const long    nc  = i >> shift;
    ChunkDesc<T>& d   = desc[nc];
    T*            buf = write_load == WriteLoad::acquire   ? load_ptr_acquire(&d.buf)
                        : write_load == WriteLoad::relaxed ? load_ptr_relaxed(&d.buf)
                                                           : d.buf;
    if (buf == nullptr || buf == busy_mark<T>())
    {
        // While claimed, the constant still holds and the claimer pre-fills it.
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

template <typename T, int S>
template <int BLOCK, typename F>
__device__ void View<T, S>::block_write(long c, long lo, long hi, const F& f) const noexcept
{
    static_assert(S % BLOCK == 0 && BLOCK % warp_size == 0, "BLOCK divides S, a multiple of the warp size");
    constexpr int K = S / BLOCK; // elements per thread
    __shared__ T* s_buf;
    ChunkDesc<T>& d    = desc[c];
    const long    base = c * S;
    const int     t    = static_cast<int>(threadIdx.x);

    __syncthreads(); // every thread has read s_buf of the previous call
    if (t == 0)
    {
        s_buf = load_ptr_acquire(&d.buf);
    }
    __syncthreads();
    T* buf = s_buf;
    if (buf == busy_mark<T>())
    {
        // Claimed by an assign() of the same kernel (outside the facade's contract): its path.
        for (int j = 0; j < K; ++j)
        {
            const long i = base + t + j * BLOCK;
            if (i >= lo && i < hi)
            {
                assign(i, f(i));
            }
        }
        return;
    }
    if (buf != nullptr)
    {
        for (int j = 0; j < K; ++j)
        {
            const long i = base + t + j * BLOCK;
            if (i >= lo && i < hi)
            {
                buf[i - base] = f(i);
            }
        }
        return;
    }
    // Compressed: one vote of the block before any buffer (the constant onto its own chunk
    // costs nothing), then the elements written and the constant elsewhere, in one pass.
    const T  constant = d.constant;
    T        vals[K];
    unsigned differs = 0; // bit j: element j written with another value than the constant
    for (int j = 0; j < K; ++j)
    {
        const long i = base + t + j * BLOCK;
        vals[j]      = constant;
        if (i >= lo && i < hi)
        {
            vals[j] = f(i);
            differs |= same_bits(vals[j], constant) ? 0U : 1U << j;
        }
    }
    if (__syncthreads_or(differs != 0) == 0)
    {
        return;
    }
    if (t == 0)
    {
        T* m  = take_spare();
        s_buf = m != nullptr ? m : take_buffer(c);
    }
    __syncthreads();
    T* const mine = s_buf;
    if (mine == nullptr)
    {
        if (t == 0)
        {
            fetch_add(&counters->arena_fail, 1ULL);
        }
        for (int j = 0; j < K; ++j)
        {
            if (((differs >> j) & 1U) != 0)
            {
                append(base + t + j * BLOCK, vals[j]);
            }
        }
        return;
    }
    for (int j = 0; j < K; ++j)
    {
        mine[t + j * BLOCK] = vals[j];
    }
    fence_device();
    __syncthreads();
    if (t == 0)
    {
        T* found = cas_ptr(&d.buf, static_cast<T*>(nullptr), mine);
        if (found == nullptr)
        {
            fetch_add(&counters->materialise, 1ULL);
        }
        else
        {
            // Another block published first (two ranges sharing the chunk): its buffer holds
            // the constant where we write; ours is parked.
            fence_device();
            fetch_add(&counters->cas_lost, 1ULL);
            park(mine);
            s_buf = found;
        }
    }
    __syncthreads();
    T* const winner = s_buf;
    if (winner != mine)
    {
        for (int j = 0; j < K; ++j)
        {
            const long i = base + t + j * BLOCK;
            if (i >= lo && i < hi)
            {
                winner[i - base] = vals[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels of the host operations
// ---------------------------------------------------------------------------

namespace kernels
{

/// @brief Before a compress(), first: every shard's top back to >= 0 (pops past empty) and
///        its bump clamped to its length, so that give_back() can push.
template <typename T, int S>
__global__ void normalise_shards(View<T, S> v)
{
    for (int a = 0; a < v.n_arenas; ++a)
    {
        const ArenaDesc<T>& ar = v.arenas[a];
        for (int s = threadIdx.x; s < n_shards; s += blockDim.x)
        {
            Shard& sh = ar.shards[s];
            sh.top    = sh.top < 0 ? 0 : sh.top;
            sh.bump   = sh.bump > ar.shard_len(s) ? ar.shard_len(s) : sh.bump;
        }
    }
}

/// @brief Before a compress(), after normalise_shards: the lost buffers and the parked spares
///        back to their shards (6.3: they wait for the phase end).
template <typename T, int S>
__global__ void reclaim_parked(View<T, S> v)
{
    const unsigned long long n_lost = v.counters->lost < v.list_capacity ? v.counters->lost : v.list_capacity;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x; k < n_lost; k += stride)
    {
        v.give_back(v.lost_list[k]);
    }
    for (unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x; k < v.spare_slots; k += stride)
    {
        T* buf = v.spare[k];
        if (buf != nullptr)
        {
            v.spare[k] = nullptr;
            v.give_back(buf);
        }
    }
}

/// @brief After reclaim_parked: the lost list is empty.
__global__ void clear_lost(ArenaCounters* c)
{
    c->lost = 0;
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
        d.constant = first;
        d.buf      = nullptr;
        v.give_back(buf);
    }
}

/// @brief The pass with one warp per chunk (grid-stride over the chunks): each lane reads
///        S/32 elements, a warp vote decides, lane 0 folds back.
template <typename T, int S>
__global__ void compress_warp(View<T, S> v, long n_chunks)
{
    const int  lane  = lane_id();
    const long warps = static_cast<long>(gridDim.x) * (blockDim.x / warp_size);
    const long first = (blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x) / warp_size;
    for (long c = first; c < n_chunks; c += warps)
    {
        ChunkDesc<T>& d   = v.desc[c];
        T*            buf = d.buf;
        if (buf == nullptr)
        {
            continue; // uniform per warp
        }
        const long valid   = (c + 1) * S <= v.n ? S : v.n - c * S;
        const T    head    = buf[0];
        bool       uniform = true;
        for (long k = lane; k < valid; k += warp_size)
        {
            uniform = uniform && same_bits(buf[k], head);
        }
        uniform = ballot(active_lanes(), !uniform) == 0;
        if (uniform && lane == 0)
        {
            d.constant = head;
            d.buf      = nullptr;
            v.give_back(buf);
        }
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

/**
 * @brief The chunk-aligned loop of a facade (section 7.3 mitigation 2): a(i) = f(i) for i in
 *        [lo, hi), one block of BLOCK threads per chunk (grid-stride over the chunks), so
 *        that each chunk has one writer and one vote. f: `T operator()(long) const`.
 */
template <int BLOCK, typename T, int S, typename F>
__global__ void __launch_bounds__(BLOCK) for_each_chunk(View<T, S> v, long lo, long hi, F f)
{
    const long first = lo / S;
    const long last  = (hi + S - 1) / S;
    for (long c = first + blockIdx.x; c < last; c += gridDim.x)
    {
        const long clo = c * S > lo ? c * S : lo;
        const long chi = (c + 1) * S < hi ? (c + 1) * S : hi;
        v.template block_write<BLOCK>(c, clo, chi, f);
    }
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
    unsigned long long claim_wait;       ///< groups that waited for a claim (Coordination::claim)
    unsigned long long claim_spin;       ///< waits given up, writes appended (Coordination::claim)
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

    /// @brief How assign() coordinates the warps of a compressed chunk (default: race).
    void set_coordination(Coordination c) noexcept
    {
        view_.coord = c;
    }

    /// @brief How assign() reads `buf` (default: acquire).
    void set_write_load(WriteLoad w) noexcept
    {
        view_.write_load = w;
    }

    /// @brief compress() with one warp per chunk instead of one block (default: per warp).
    void set_compress_per_warp(bool per_warp) noexcept
    {
        compress_per_warp_ = per_warp;
    }

    /// @brief a(i) = f(i) for i in [lo, hi), by the chunk-aligned loop (kernels::for_each_chunk).
    template <int BLOCK = 256, typename F>
    void parallel_for_chunks(long lo, long hi, const F& f, unsigned grid = 4096)
    {
        if (hi > lo)
        {
            const long chunks = (hi + S - 1) / S - lo / S;
            const auto blocks = static_cast<unsigned>(chunks < grid ? chunks : grid);
            kernels::for_each_chunk<BLOCK><<<blocks, BLOCK>>>(view_, lo, hi, f);
        }
    }

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

    long                      n_chunks_;
    View<T, S>                view_{};
    std::vector<T*>           arenas_;
    std::vector<Shard*>       shards_;
    std::vector<ArenaDesc<T>> arena_descs_;
    unsigned long long        total_buffers_     = 0;
    unsigned long long        replays_           = 0;
    unsigned long long        overflows_         = 0;
    bool                      compress_per_warp_ = true; // GH200: 83% of stream read at S = 1024, against 66% per block
};

template <typename T, int S>
DeviceArray<T, S>::DeviceArray(long n, unsigned long long arena_chunks, unsigned long long append_capacity)
    : n_chunks_((n + S - 1) / S)
{
    view_.n              = n;
    view_.coord          = Coordination::race;
    view_.claim_spin_max = 1U << 16; // ~4 ms of 64 ns pauses: far beyond one pre-fill
    view_.write_load     = WriteLoad::acquire;
    SA_GPU(gpuMalloc(&view_.desc, sizeof(ChunkDesc<T>) * n_chunks_));
    SA_GPU(gpuMemset(view_.desc, 0, sizeof(ChunkDesc<T>) * n_chunks_)); // {nullptr, T{}} for arithmetic T
    view_.spare_slots = 1ULL << 16; // more than the warps resident on an H100 or an MI300A
    SA_GPU(gpuMalloc(&view_.spare, sizeof(T*) * view_.spare_slots));
    SA_GPU(gpuMemset(view_.spare, 0, sizeof(T*) * view_.spare_slots));
    SA_GPU(gpuMalloc(&view_.arenas, sizeof(ArenaDesc<T>) * max_arenas));
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
    for (Shard* sh : shards_)
    {
        (void)gpuFree(sh);
    }
    (void)gpuFree(view_.arenas);
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
        // The old arenas keep their segments: the whole old stack, at the same offsets.
        SA_GPU(gpuMemcpy(stack, view_.free_stack, sizeof(T*) * view_.list_capacity, gpuMemcpyDeviceToDevice));
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
    if (arena_descs_.size() >= static_cast<std::size_t>(max_arenas))
    {
        throw std::runtime_error("DeviceArray::grow: more than max_arenas arenas");
    }
    T*     a      = nullptr;
    Shard* shards = nullptr;
    SA_GPU(gpuMalloc(&a, sizeof(T) * S * more_chunks));
    SA_GPU(gpuMalloc(&shards, sizeof(Shard) * n_shards));
    SA_GPU(gpuMemset(shards, 0, sizeof(Shard) * n_shards));
    arenas_.push_back(a);
    shards_.push_back(shards);
    ArenaDesc<T> ar{};
    ar.base         = a;
    ar.n            = more_chunks;
    ar.shard_size   = (more_chunks + n_shards - 1) / n_shards;
    ar.stack_offset = total_buffers_;
    ar.shards       = shards;
    arena_descs_.push_back(ar);
    total_buffers_ += more_chunks;
    resize_lists(total_buffers_);
    SA_GPU(gpuMemcpy(view_.arenas, arena_descs_.data(), sizeof(ArenaDesc<T>) * arena_descs_.size(),
                     gpuMemcpyHostToDevice));
    view_.n_arenas = static_cast<int>(arena_descs_.size());
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
    kernels::normalise_shards<<<1, n_shards>>>(view_);
    kernels::reclaim_parked<<<64, 256>>>(view_);
    kernels::clear_lost<<<1, 1>>>(view_.counters);
    constexpr int block = S < 256 ? S : 256;
    if (n_chunks_ > 0 && compress_per_warp_)
    {
        const long warps_per_block = 256 / warp_size;
        const long blocks          = (n_chunks_ + warps_per_block - 1) / warps_per_block;
        kernels::compress_warp<<<static_cast<unsigned>(blocks < 65536 ? blocks : 65536), 256>>>(view_, n_chunks_);
    }
    else if (n_chunks_ > 0)
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
    return Stats{n_chunks_, materialised, total_buffers_, c.cas_lost,  c.arena_fail,
                 replays_,  overflows_,   c.claim_wait,   c.claim_spin};
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
