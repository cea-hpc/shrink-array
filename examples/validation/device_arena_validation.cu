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
 * @file device_arena_validation.cu
 * @brief device::DeviceArena (chunk/device/arena.h) on the device: concurrent takes hand
 *        out each buffer once, exhaustion returns nullptr exactly past the last buffer,
 *        given-back buffers are taken again, growth adds an arena tried first, a take goes
 *        to its home shard first, and the buffers handed out do not overlap.
 *
 * @details Built with -DCHUNK_DEVICE_CORE_CHECK=ON, as device_core_validation. Every case
 *          launches more takers than buffers, so the shards run out under contention and
 *          the failed pops push the tops below zero (normalise() must repair them).
 */

#include <algorithm>     // std::sort, std::count, std::all_of, std::none_of
#include <cstdio>        // std::printf, std::snprintf
#include <exception>     // std::exception
#include <set>           // std::set
#include <vector>        // std::vector

#include "chunk/device/arena.h"   // DeviceArena, ArenaRef
#include "chunk/device/backend.h" // allocate, copy, synchronize

namespace sd = shrink_array::device;

namespace
{

constexpr unsigned int S = 1024;
using T                  = double;
using Arena              = sd::DeviceArena<T, S>;
using Ref                = sd::ArenaRef<T, S>;

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

/// @brief Taker t takes a buffer for home chunk t (+ @p home_offset) and fills it with t.
__global__ void take_kernel(Ref r, int n_takers, long long home_offset, T** out)
{
    const int t = static_cast<int>(blockIdx.x);
    if (t >= n_takers)
    {
        return;
    }
    __shared__ T* buf;
    if (threadIdx.x == 0)
    {
        buf    = r.take(t + home_offset);
        out[t] = buf;
    }
    __syncthreads();
    if (buf != nullptr)
    {
        for (unsigned int k = threadIdx.x; k < S; k += blockDim.x)
        {
            buf[k] = static_cast<T>(t);
        }
    }
}

__global__ void give_back_kernel(Ref r, T* const* bufs, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n && bufs[i] != nullptr)
    {
        r.give_back(bufs[i]);
    }
}

/// @brief @p n_takers concurrent takes (one block each); the buffers, nullptr where none.
std::vector<T*> take(Arena& arena, int n_takers, long long home_offset = 0)
{
    T** d_out = sd::allocate<T*>(static_cast<std::size_t>(n_takers));
    take_kernel<<<static_cast<unsigned>(n_takers), 128>>>(arena.ref(), n_takers, home_offset, d_out);
    sd::synchronize();
    std::vector<T*> out(static_cast<std::size_t>(n_takers));
    sd::copy(out.data(), d_out, out.size());
    sd::release(d_out);
    return out;
}

void give_back(Arena& arena, const std::vector<T*>& bufs)
{
    T** d_bufs = sd::allocate<T*>(bufs.size());
    sd::copy(d_bufs, bufs.data(), bufs.size());
    arena.normalise();
    give_back_kernel<<<static_cast<unsigned>((bufs.size() + 255) / 256), 256>>>(arena.ref(), d_bufs,
                                                                               static_cast<int>(bufs.size()));
    sd::synchronize();
    sd::release(d_bufs);
}

std::vector<T*> non_null(const std::vector<T*>& v)
{
    std::vector<T*> out;
    for (T* p : v)
    {
        if (p != nullptr)
        {
            out.push_back(p);
        }
    }
    return out;
}

bool distinct(std::vector<T*> v)
{
    std::sort(v.begin(), v.end());
    return std::adjacent_find(v.begin(), v.end()) == v.end();
}

/// @brief Every buffer of @p got lies in [@p base, @p base + n * S) on a buffer boundary.
bool in_arena(const std::vector<T*>& got, const T* base, unsigned long long n)
{
    return std::all_of(got.begin(), got.end(),
                       [&](const T* p) { return p >= base && p < base + n * S && (p - base) % S == 0; });
}

/// @brief Each buffer taken by taker t holds t everywhere: no two takers shared memory.
bool filled_by_owner(const std::vector<T*>& by_taker)
{
    std::vector<T> h(S);
    for (std::size_t t = 0; t < by_taker.size(); ++t)
    {
        if (by_taker[t] == nullptr)
        {
            continue;
        }
        sd::copy(h.data(), by_taker[t], S);
        if (std::count(h.begin(), h.end(), static_cast<T>(t)) != static_cast<long>(S))
        {
            return false;
        }
    }
    return true;
}

/// @brief The arena's first buffer: the lowest address handed out from a fresh arena.
const T* lowest(const std::vector<T*>& got)
{
    return *std::min_element(got.begin(), got.end());
}

void take_give_grow()
{
    std::printf("[take, give back, grow] S=%u, 1000 buffers, 4096 takers\n", S);
    Arena arena(1000);
    char  line[128];

    const std::vector<T*> first = take(arena, 4096);
    const std::vector<T*> got   = non_null(first);
    const T*              base  = lowest(got);
    std::snprintf(line, sizeof line, "exhaustion: %zu of 4096 takers served", got.size());
    expect(got.size() == 1000, line);
    expect(distinct(got), "each buffer handed out once");
    expect(in_arena(got, base, 1000), "every buffer in the arena, on a buffer boundary");
    expect(filled_by_owner(first), "no overlap: each buffer holds its taker's value");
    expect(arena.in_use() == 1000, "in_use() == 1000 (after failed pops: normalised)");

    // Every other buffer back, then 4096 takers again: exactly those, and nothing else.
    std::vector<T*> back;
    for (std::size_t k = 0; k < got.size(); k += 2)
    {
        back.push_back(got[k]);
    }
    give_back(arena, back);
    expect(arena.in_use() == 500, "500 given back: in_use() == 500");
    const std::vector<T*> again = non_null(take(arena, 4096));
    expect(again.size() == 500, "500 takers served from the free stacks");
    expect(std::set<T*>(again.begin(), again.end()) == std::set<T*>(back.begin(), back.end()),
           "the buffers given back, and only those");

    // Growth: a second arena, tried first.
    arena.grow(300);
    const std::vector<T*> grown = non_null(take(arena, 4096));
    expect(arena.arenas() == 2 && arena.buffers() == 1300, "grow(300): two arenas, 1300 buffers");
    expect(grown.size() == 300, "300 takers served after growth");
    const bool none_old = std::none_of(grown.begin(), grown.end(), [&](const T* p) { return p >= base && p < base + 1000 * S; });
    expect(in_arena(grown, lowest(grown), 300) && none_old, "all from the new arena");
    expect(arena.in_use() == 1300, "in_use() == 1300");

    // Everything back, then everything taken again: the free stacks of both arenas.
    std::vector<T*> all = got;
    all.insert(all.end(), grown.begin(), grown.end());
    give_back(arena, all);
    expect(arena.in_use() == 0, "all given back: in_use() == 0");
    const std::vector<T*> last     = take(arena, 4096);
    const std::vector<T*> last_got = non_null(last);
    expect(last_got.size() == 1300 && distinct(last_got), "1300 distinct buffers taken again");
    expect(filled_by_owner(last), "no overlap after a full cycle");
}

void release_free()
{
    std::printf("[release_free] S=%u, arenas of 1000, 300 and 2000 buffers\n", S);
    Arena arena(1000);
    arena.grow(300);
    arena.grow(2000);
    const std::vector<T*> got = non_null(take(arena, 4096)); // the three arenas exhausted
    char line[160];
    expect(got.size() == 3300 && distinct(got), "3300 distinct buffers over three arenas");
    expect(arena.release_free() == 0 && arena.arenas() == 3, "every arena in use: none freed");

    // Every buffer back but 100 of the arena at the lowest address: the two others are free,
    // the reserve (100 / 8) is met whichever of the three that is, both freed.
    std::vector<T*> keep, back;
    const T*        first_base = nullptr;
    for (T* p : got)
    {
        first_base = (first_base == nullptr || p < first_base) ? p : first_base;
    }
    for (T* p : got)
    {
        (p >= first_base && p < first_base + 1000 * S && keep.size() < 100 ? keep : back).push_back(p);
    }
    give_back(arena, back);
    const unsigned long long freed = arena.release_free();
    std::snprintf(line, sizeof line, "%llu buffers freed, %d arena(s) left, %llu buffers, %llu in use", freed,
                  arena.arenas(), arena.buffers(), arena.in_use());
    const bool one_left = arena.arenas() == 1 && arena.buffers() + freed == 3300 && arena.in_use() == 100;
    expect(one_left, line);

    // The arena left still serves: its 900 free buffers, none of the ones kept.
    const std::vector<T*> again = non_null(take(arena, 4096));
    const std::set<T*>    kept(keep.begin(), keep.end());
    const bool none_kept = std::none_of(again.begin(), again.end(), [&](T* p) { return kept.count(p) != 0; });
    std::snprintf(line, sizeof line, "after release: %zu buffers taken again, distinct, none of the 100 kept",
                  again.size());
    expect(again.size() == arena.buffers() - 100 && distinct(again) && none_kept, line);

    // A last arena never goes: everything back, release_free leaves one.
    std::vector<T*> all = keep;
    all.insert(all.end(), again.begin(), again.end());
    give_back(arena, all);
    expect(arena.release_free() == 0 && arena.arenas() == 1 && arena.in_use() == 0, "the last arena is kept");
}

void home_shards()
{
    std::printf("[home shard] S=%u, %d buffers (4 per shard), one taker per shard\n", S, 4 * sd::n_shards);
    Arena                 arena(4 * sd::n_shards);
    const std::vector<T*> got  = take(arena, sd::n_shards, 3 * sd::n_shards); // homes 384..511
    const T*              base = lowest(got);
    bool                  home = true;
    for (int t = 0; t < sd::n_shards; ++t)
    {
        home = home && got[static_cast<std::size_t>(t)] != nullptr &&
               (got[static_cast<std::size_t>(t)] - base) / S / 4 == t;
    }
    expect(home, "each take served by its home shard (home % n_shards)");
}

void small_arena()
{
    std::printf("[small arena] S=%u, 5 buffers (fewer than the shards), 256 takers\n", S);
    Arena                 arena(5);
    const std::vector<T*> first = take(arena, 256);
    const std::vector<T*> got   = non_null(first);
    expect(got.size() == 5 && distinct(got), "5 distinct buffers, then nullptr");
    expect(filled_by_owner(first), "no overlap");
    Arena empty(0);
    expect(non_null(take(empty, 64)).empty() && empty.in_use() == 0, "an arena of 0 buffers: nullptr only");
}

} // namespace

int main()
{
    std::printf("device_arena_validation: device::DeviceArena under concurrent takes\n");
    try
    {
        take_give_grow();
        release_free();
        home_shards();
        small_arena();
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
