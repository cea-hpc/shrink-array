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
 * @file alloc_token_rt.cpp
 * @brief The runtime behind -fsanitize=alloc-token: a per-type allocation
 *        census. Linked into every chunk variant when SANITIZER=token.
 *
 * @details Clang >= 22 with -fsanitize=alloc-token rewrites every heap
 *          allocation into `__alloc_token_<symbol>(<original args>, token)`,
 *          where the token is a hash of the allocated TYPE (mode
 *          typehashpointersplit: the top half of the id space is for types
 *          that contain pointers, the bottom half for those that do not; a C
 *          malloc with no type hint gets token 0). The instrumentation is all
 *          Clang provides -- no compiler-rt runtime, and the packaged Scudo
 *          does not export the entry points either -- so the fourteen
 *          functions below are what makes an instrumented program link.
 *
 *          Each one forwards to the real allocator and counts the call and
 *          its bytes in a lock-free table keyed by token. At exit the table
 *          is printed on stderr, biggest first. Nothing is stored in the
 *          blocks themselves, so untouched free() keeps working and the
 *          census is "who allocated how much", not a live-bytes tracker.
 *
 *          Tokens are hashes: python/alloc_token_names.py turns them back
 *          into type names from the binary's symbols.
 *
 * @note MUST be compiled with -fno-sanitize=alloc-token (chunk/CMakeLists.txt
 *       does so): instrumented, its own malloc / new calls would recurse into
 *       these very functions.
 *
 * @note Env: ALLOC_TOKEN_REPORT=0 silences the report, =N prints the N
 *       biggest tokens (default 40), =all prints them all.
 */

#include <array>     // std::array
#include <algorithm> // std::sort, the report
#include <atomic>    // std::atomic, std::memory_order_relaxed
#include <cstddef>   // std::size_t, std::ptrdiff_t
#include <cstdint>   // std::uint64_t
#include <cstdio>    // std::fprintf, std::fflush
#include <cstdlib>   // malloc & co, std::getenv
#include <cstring>   // std::strcmp
#include <malloc.h>  // memalign
#include <stdlib.h>  // ::posix_memalign: POSIX, not declared by <cstdlib>
#include <new>       // std::align_val_t, std::nothrow_t

#include "chunk/internal/env_parse.h" // env_number / env_flag (checked, not atoi)

namespace
{

/// @brief One token's counters. `key` is token + 1, so that 0 means "empty
///        slot" while token 0 (the untyped C allocations) stays representable.
struct Slot
{
    std::atomic<std::uint64_t> key{0};
    std::atomic<std::uint64_t> allocs{0};
    std::atomic<std::uint64_t> bytes{0};
};

constexpr unsigned    kSlotBits = 14;
constexpr std::size_t kSlots    = std::size_t{1} << kSlotBits; ///< 16384 distinct types
constexpr std::size_t kSlotMask = kSlots - 1;

std::array<Slot, kSlots> g_table; ///< Open addressing, linear probing, never resized.

std::atomic<std::uint64_t> g_dropped{0};       ///< Allocations that found the table full.
std::atomic<std::uint64_t> g_dropped_bytes{0}; ///< ... and their bytes.

/// @brief Record one allocation of @p bytes under @p token. Lock-free and
///        allocation-free: it runs INSIDE malloc, from every thread.
inline void note(std::uint64_t token, std::size_t bytes) noexcept
{
    const std::uint64_t key = token + 1;
    // Fibonacci hashing of the token to a slot: tokens are SipHash outputs,
    // already well spread, but token 0 and small ids must not all pile up.
    std::size_t i = static_cast<std::size_t>((token * 0x9E3779B97F4A7C15ULL) >> (64 - kSlotBits));
    for (std::size_t probe = 0; probe < kSlots; ++probe)
    {
        Slot&         slot = g_table[i];
        std::uint64_t k    = slot.key.load(std::memory_order_relaxed);
        if (k == 0 && slot.key.compare_exchange_strong(k, key, std::memory_order_relaxed))
        {
            k = key; // we claimed the slot; a lost race leaves the winner's key in k
        }
        if (k == key)
        {
            slot.allocs.fetch_add(1, std::memory_order_relaxed);
            slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
            return;
        }
        i = (i + 1) & kSlotMask;
    }
    g_dropped.fetch_add(1, std::memory_order_relaxed);
    g_dropped_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/// @brief The typehashpointersplit boundary: tokens >= this belong to types
///        that contain pointers. (SIZE_MAX / 2, as in LLVM's AllocToken pass.)
constexpr std::uint64_t kPointerHalf = UINT64_MAX / 2;

struct Entry
{
    std::uint64_t token;
    std::uint64_t allocs;
    std::uint64_t bytes;
};

std::array<Entry, kSlots> g_entries; ///< Static so the report allocates nothing either.

/// @brief Print the census on stderr. Runs after every C++ static destructor
///        (a .fini_array entry runs after the __cxa_atexit ones), so the
///        allocations of teardown are counted too.
__attribute__((destructor)) void report()
{
    std::size_t top = 40;
    if (const char* env = std::getenv("ALLOC_TOKEN_REPORT"); env != nullptr && *env != '\0')
    {
        if (std::strcmp(env, "all") == 0)
        {
            top = kSlots;
        }
        else
        {
            top = shrink_array::env_number<std::size_t>("ALLOC_TOKEN_REPORT", top);
        }
        if (top == 0)
        {
            return;
        }
    }

    std::size_t   count      = 0;
    std::uint64_t tot_allocs = 0;
    std::uint64_t tot_bytes  = 0;
    for (const Slot& s : g_table)
    {
        const std::uint64_t k = s.key.load(std::memory_order_relaxed);
        if (k == 0)
        {
            continue;
        }
        g_entries[count++] =
            Entry{k - 1, s.allocs.load(std::memory_order_relaxed), s.bytes.load(std::memory_order_relaxed)};
        tot_allocs += g_entries[count - 1].allocs;
        tot_bytes += g_entries[count - 1].bytes;
    }
    std::sort(g_entries.begin(), g_entries.begin() + static_cast<std::ptrdiff_t>(count),
              [](const Entry& lhs, const Entry& rhs) { return lhs.bytes > rhs.bytes; });

    std::fprintf(stderr, "\n===== ALLOC_TOKEN (per-type allocation census, -fsanitize=alloc-token) =====\n");
    std::fprintf(stderr, "  %zu distinct tokens, %llu allocations, %llu MiB requested", count,
                 static_cast<unsigned long long>(tot_allocs), static_cast<unsigned long long>(tot_bytes >> 20U));
    if (const auto dropped = g_dropped.load(std::memory_order_relaxed); dropped != 0)
    {
        std::fprintf(stderr, "  (table full: %llu allocations / %llu MiB NOT counted)",
                     static_cast<unsigned long long>(dropped),
                     static_cast<unsigned long long>(g_dropped_bytes.load(std::memory_order_relaxed) >> 20U));
    }
    std::fprintf(stderr, "\n  %-18s  %-3s  %14s  %18s\n", "token", "ptr", "allocs", "bytes");
    for (std::size_t i = 0; i < count && i < top; ++i)
    {
        const Entry& entry = g_entries[i];
        std::fprintf(stderr, "  0x%016llx  %-3s  %14llu  %18llu%s\n", static_cast<unsigned long long>(entry.token),
                     entry.token >= kPointerHalf ? "yes" : "no", static_cast<unsigned long long>(entry.allocs),
                     static_cast<unsigned long long>(entry.bytes),
                     entry.token == 0 ? "  untyped (C malloc/calloc/realloc)" : "");
    }
    if (count > top)
    {
        std::fprintf(stderr, "  ... %zu more (ALLOC_TOKEN_REPORT=all)\n", count - top);
    }
    std::fprintf(stderr, "  names: python3 python/alloc_token_names.py <binary> <this output>\n");
    std::fflush(stderr);
}

} // namespace

// ── The entry points ─────────────────────────────────────────────────────────
// Standard ABI: the original signature plus a trailing token. The C++ ones are
// named after the mangled operator they replace; being extern "C" they are
// plain identifiers here. Every one forwards to the real allocator, so the
// semantics (nothrow, alignment, ENOMEM) are exactly the original's.

extern "C"
{
    void* __alloc_token_malloc(std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return std::malloc(size);
    }

    void* __alloc_token_calloc(std::size_t nmemb, std::size_t size, std::uint64_t token)
    {
        note(token, nmemb * size);
        return std::calloc(nmemb, size);
    }

    void* __alloc_token_realloc(void* ptr, std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return std::realloc(ptr, size);
    }

    void* __alloc_token_aligned_alloc(std::size_t align, std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return std::aligned_alloc(align, size);
    }

    void* __alloc_token_memalign(std::size_t align, std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return ::memalign(align, size);
    }

    int __alloc_token_posix_memalign(void** out, std::size_t align, std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return ::posix_memalign(out, align, size);
    }

    // operator new(size_t)
    void* __alloc_token__Znwm(std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return ::operator new(size);
    }

    // operator new[](size_t)
    void* __alloc_token__Znam(std::size_t size, std::uint64_t token)
    {
        note(token, size);
        return ::operator new[](size);
    }

    // operator new(size_t, const std::nothrow_t&)
    void* __alloc_token__ZnwmRKSt9nothrow_t(std::size_t size, const std::nothrow_t& nt, std::uint64_t token)
    {
        note(token, size);
        return ::operator new(size, nt);
    }

    // operator new[](size_t, const std::nothrow_t&)
    void* __alloc_token__ZnamRKSt9nothrow_t(std::size_t size, const std::nothrow_t& nt, std::uint64_t token)
    {
        note(token, size);
        return ::operator new[](size, nt);
    }

    // operator new(size_t, std::align_val_t)
    void* __alloc_token__ZnwmSt11align_val_t(std::size_t size, std::align_val_t al, std::uint64_t token)
    {
        note(token, size);
        return ::operator new(size, al);
    }

    // operator new[](size_t, std::align_val_t)
    void* __alloc_token__ZnamSt11align_val_t(std::size_t size, std::align_val_t al, std::uint64_t token)
    {
        note(token, size);
        return ::operator new[](size, al);
    }

    // operator new(size_t, std::align_val_t, const std::nothrow_t&)
    void* __alloc_token__ZnwmSt11align_val_tRKSt9nothrow_t(std::size_t           size,
                                                           std::align_val_t      al,
                                                           const std::nothrow_t& nt,
                                                           std::uint64_t         token)
    {
        note(token, size);
        return ::operator new(size, al, nt);
    }

    // operator new[](size_t, std::align_val_t, const std::nothrow_t&)
    void* __alloc_token__ZnamSt11align_val_tRKSt9nothrow_t(std::size_t           size,
                                                           std::align_val_t      al,
                                                           const std::nothrow_t& nt,
                                                           std::uint64_t         token)
    {
        note(token, size);
        return ::operator new[](size, al, nt);
    }
}
