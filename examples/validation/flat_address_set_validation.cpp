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

// ============================================================================
// fast_alloc::detail::FlatAddressSet (fast_allocators/internal/flat_address_set.h),
// the set of SlabArena's live slabs, against std::unordered_set:
//
//   1. random inserts, erases and lookups of slab-aligned addresses (the
//      arena's case: 64 KiB alignment, low bits zero), every result compared;
//   2. the same with unaligned addresses (any non-zero key);
//   3. growth from an empty table, then erase everything, then refill: the
//      backward-shift erase must leave no chain broken;
//   4. dense runs of consecutive slabs (one region carved back to back).
//
// `--bench [n]`: insert then erase n slab addresses, both containers.
// Exit status 0 when every check passes.
// ============================================================================

#include <chrono>        // std::chrono::steady_clock (--bench)
#include <cstdint>       // std::uintptr_t, std::uint64_t
#include <cstdlib>       // std::atoll
#include <cstring>       // std::strcmp
#include <print>         // std::println
#include <random>        // std::mt19937_64
#include <unordered_set> // the reference
#include <vector>        // std::vector

#include "fast_allocators/internal/flat_address_set.h" // fast_alloc::detail::FlatAddressSet, under test

namespace
{

using fast_alloc::detail::FlatAddressSet;

int g_fails = 0;

void check(bool ok, const char* what)
{
    if (!ok)
    {
        ++g_fails;
        std::println("FAIL {}", what);
    }
}

/// @brief Random operations on keys drawn from a pool; every result against the reference.
void random_ops(std::uintptr_t align, std::size_t pool, std::size_t ops, std::uint64_t seed)
{
    std::mt19937_64             rng(seed);
    std::vector<std::uintptr_t> keys(pool);
    for (auto& k : keys)
    {
        k = ((rng() | 1U) & ~(align - 1)) | align; // non-zero, aligned
    }
    FlatAddressSet                     flat;
    std::unordered_set<std::uintptr_t> ref;
    std::size_t                        mismatches = 0;
    for (std::size_t n = 0; n < ops; ++n)
    {
        const std::uintptr_t key = keys[rng() % pool];
        switch (rng() % 3)
        {
            case 0:
                mismatches += flat.insert(key) != ref.insert(key).second ? 1 : 0;
                break;
            case 1:
                mismatches += flat.erase(key) != ref.erase(key) ? 1 : 0;
                break;
            default:
                mismatches += flat.contains(key) != ref.contains(key) ? 1 : 0;
                break;
        }
        mismatches += flat.size() != ref.size() ? 1 : 0;
    }
    for (const std::uintptr_t key : keys)
    {
        mismatches += flat.contains(key) != ref.contains(key) ? 1 : 0;
    }
    std::println("[flat-set] align {:6}, pool {:7}, {:8} ops: {} mismatch(es), {} keys left",
                 static_cast<std::size_t>(align), pool, ops, mismatches, ref.size());
    check(mismatches == 0, "random operations agree with std::unordered_set");
}

void grow_erase_refill()
{
    constexpr std::size_t    n     = 200000;
    constexpr std::uintptr_t align = 1U << 16U;
    FlatAddressSet           flat;
    bool                     ok = true;
    for (std::size_t i = 1; i <= n; ++i)
    {
        ok = ok && flat.insert(i * align * 7);
    }
    ok = ok && flat.size() == n;
    for (std::size_t i = 1; i <= n; i += 2)
    {
        ok = ok && flat.erase(i * align * 7) == 1;
    }
    for (std::size_t i = 1; i <= n; ++i)
    {
        ok = ok && flat.contains(i * align * 7) == (i % 2 == 0);
    }
    for (std::size_t i = 2; i <= n; i += 2)
    {
        ok = ok && flat.erase(i * align * 7) == 1;
    }
    ok = ok && flat.empty() && flat.erase(align) == 0 && !flat.contains(align);
    for (std::size_t i = 1; i <= n; ++i)
    {
        ok = ok && flat.insert(i * align * 3) && !flat.insert(i * align * 3);
    }
    ok = ok && flat.size() == n;
    std::println("[flat-set] grow, erase all, refill ({} keys): {}", n, ok ? "ok" : "FAIL");
    check(ok, "grow / erase all / refill");
}

void dense_runs()
{
    // One region carved back to back: consecutive multiples of the slab size.
    constexpr std::uintptr_t base = std::uintptr_t{1} << 40U;
    constexpr std::uintptr_t slab = 1U << 16U;
    constexpr std::size_t    n    = 100000;
    FlatAddressSet           flat;
    std::mt19937_64          rng(7);
    std::vector<char>        in(n, 0);
    std::size_t              mismatches = 0;
    for (std::size_t round = 0; round < 1000000; ++round)
    {
        const std::size_t i   = rng() % n;
        const auto        key = base + (i * slab);
        if (in[i] != 0)
        {
            mismatches += flat.erase(key) == 1 ? 0 : 1;
            in[i] = 0;
        }
        else
        {
            mismatches += flat.insert(key) ? 0 : 1;
            in[i] = 1;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        mismatches += flat.contains(base + (i * slab)) == (in[i] != 0) ? 0 : 1;
    }
    std::println("[flat-set] dense run of {} slabs, 1e6 toggles: {} mismatch(es)", n, mismatches);
    check(mismatches == 0, "dense runs");
}

double ns_per(std::chrono::steady_clock::time_point t0, std::size_t n)
{
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
           static_cast<double>(n);
}

int bench(std::size_t n)
{
    std::mt19937_64             rng(11);
    std::vector<std::uintptr_t> keys(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        keys[i] = (std::uintptr_t{1} << 40U) + (i * (std::uintptr_t{1} << 16U));
    }
    for (std::size_t i = n; i > 1; --i)
    {
        std::swap(keys[i - 1], keys[rng() % i]);
    }
    std::unordered_set<std::uintptr_t> ref;
    auto                               t0 = std::chrono::steady_clock::now();
    for (const auto k : keys)
    {
        ref.insert(k);
    }
    const double ref_ins = ns_per(t0, n);
    t0                   = std::chrono::steady_clock::now();
    for (const auto k : keys)
    {
        ref.erase(k);
    }
    const double   ref_era = ns_per(t0, n);
    FlatAddressSet flat;
    t0 = std::chrono::steady_clock::now();
    for (const auto k : keys)
    {
        flat.insert(k);
    }
    const double flat_ins = ns_per(t0, n);
    t0                    = std::chrono::steady_clock::now();
    for (const auto k : keys)
    {
        flat.erase(k);
    }
    const double flat_era = ns_per(t0, n);
    std::println("[flat-set] {} slab addresses: insert {:.1f} ns -> {:.1f} ns | erase {:.1f} ns -> {:.1f} ns "
                 "(std::unordered_set -> FlatAddressSet)",
                 n, ref_ins, flat_ins, ref_era, flat_era);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        return bench(argc > 2 ? static_cast<std::size_t>(std::atoll(argv[2])) : std::size_t{1000000});
    }
    random_ops(std::uintptr_t{1} << 16U, 5000, 2000000, 1);
    random_ops(std::uintptr_t{1} << 16U, 200000, 2000000, 2);
    random_ops(1, 3000, 2000000, 3);
    random_ops(8, 100000, 2000000, 4);
    grow_erase_refill();
    dense_runs();
    std::println("flat_address_set_validation: {}", g_fails == 0 ? "OK" : "FAILED");
    return g_fails == 0 ? 0 : 1;
}
