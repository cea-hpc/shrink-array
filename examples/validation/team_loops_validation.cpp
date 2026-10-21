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
// The static worksharing loops of pmp::Team (fast_allocators/internal/
// pmp_loops.h), alone: no array, no variant -- the backend of the build
// (OpenMP, pool or serial).
// ============================================================================
//
// For each loop -- for_static, for_static_simd, for_static_chunks(c),
// for_static_chunks_simd(c) --, teams of 1, 2, 3 and 8 threads, bounds that
// start at 0 or not (negative included), empty, shorter than the team, not a
// multiple of the team or of the block, int and long long indices:
//
//   1. every iteration runs exactly once;
//   2. on the rank OpenMP gives it: schedule(static), one contiguous block
//      per rank, the first (n % size) ranks one iteration more (libgomp's and
//      libomp's split); schedule(static, c), block k of c iterations from
//      `first` on rank k % size (the specification's);
//   3. two nowait loops of the same bounds and schedule in a row give a rank
//      the same iterations -- what a region that reads in its second loop
//      what it wrote in its first, with no barrier between, relies on (the
//      bench's phases);
//   4. the wait form and the nowait form share alike.
//
// Exit status 0 when every check passes.

#include <cstdio>  // std::printf
#include <vector>  // std::vector
#include <utility> // std::pair

#include "fast_allocators/internal/pmp_loops.h" // fast_alloc::pmp::team, Team, options, wait_mode

namespace pmp = fast_alloc::pmp;

namespace
{

int g_fails  = 0;
int g_checks = 0;

void check(bool ok, const char* what, const char* loop, int nt, long long first, long long n, int chunk)
{
    ++g_checks;
    if (!ok)
    {
        ++g_fails;
        std::printf("FAIL %s: %s, team %d, [%lld, %lld), chunk %d\n", what, loop, nt, first, first + n, chunk);
    }
}

/// @brief The rank schedule(static) gives iteration @p k (offset from first) of @p n.
int static_owner(long long k, long long n, int size)
{
    const long long q = n / size;
    const long long m = n % size;
    // ranks [0, m) have q + 1 iterations, [m, size) have q.
    if (k < m * (q + 1))
    {
        return static_cast<int>(k / (q + 1));
    }
    return static_cast<int>(m + ((k - (m * (q + 1))) / q));
}

/// @brief The rank schedule(static, c) gives iteration @p k (offset from first).
int chunks_owner(long long k, int c, int size)
{
    return static_cast<int>((k / c) % size);
}

enum class loop_kind
{
    static_blocks,
    static_blocks_simd,
    static_chunks,
    static_chunks_simd,
};

const char* name_of(loop_kind k)
{
    switch (k)
    {
        case loop_kind::static_blocks:
            return "for_static";
        case loop_kind::static_blocks_simd:
            return "for_static_simd";
        case loop_kind::static_chunks:
            return "for_static_chunks";
        case loop_kind::static_chunks_simd:
            return "for_static_chunks_simd";
    }
    return "?";
}

bool is_chunked(loop_kind k)
{
    return k == loop_kind::static_chunks || k == loop_kind::static_chunks_simd;
}

/// @brief One loop of kind @p k on @p t, the body writing what @p body writes.
template <typename Index, typename Body>
void run_loop(pmp::Team& t, loop_kind k, Index first, Index last, int chunk, pmp::wait_mode w, Body&& body)
{
    switch (k)
    {
        case loop_kind::static_blocks:
            t.for_static(first, last, body, w);
            return;
        case loop_kind::static_blocks_simd:
            t.for_static_simd(first, last, body, w);
            return;
        case loop_kind::static_chunks:
            t.for_static_chunks(first, last, chunk, body, w);
            return;
        case loop_kind::static_chunks_simd:
            t.for_static_chunks_simd(first, last, chunk, body, w);
            return;
    }
}

/**
 * @brief Checks 1 to 4 for one loop kind, team size, bounds and block size.
 * @details The bodies only store, each iteration into its own elements (no
 *          atomic, no call, no counter shared across iterations): a simd loop
 *          stays a loop the vectoriser may take, with no dependence for
 *          `simd` to break. An iteration run twice shows in hits[], one run
 *          on the wrong rank in owner[].
 */
template <typename Index>
void check_one(loop_kind k, int nt, Index first, Index n, int chunk)
{
    const Index      last = static_cast<Index>(first + n);
    const auto       un   = static_cast<std::size_t>(n);
    std::vector<int> owner_wait(un, -1);
    std::vector<int> owner_a(un, -1);
    std::vector<int> owner_b(un, -1);
    std::vector<int> hits_wait(un, 0);
    std::vector<int> hits_a(un, 0);
    int              size = 0;

    pmp::team(pmp::options{.num_threads = nt},
              [&](pmp::Team& t)
              {
                  const int r = t.rank();
                  int*      ow = owner_wait.data();
                  int*      oa = owner_a.data();
                  int*      ob = owner_b.data();
                  int*      hw = hits_wait.data();
                  int*      ha = hits_a.data();
                  t.single([&] { size = t.size(); }); // a barrier: size set for all

                  // 4: the wait form.
                  run_loop(t, k, first, last, chunk, pmp::wait_mode::wait,
                           [&](Index i)
                           {
                               ow[i - first] = r;
                               ++hw[i - first];
                           });
                  // 3: two nowait loops in a row, no barrier between.
                  run_loop(t, k, first, last, chunk, pmp::nowait,
                           [&](Index i)
                           {
                               oa[i - first] = r;
                               ++ha[i - first];
                           });
                  run_loop(t, k, first, last, chunk, pmp::nowait,
                           [&](Index i)
                           {
                               // Written by this rank in the loop above, if the
                               // shares agree: no barrier needed to read it.
                               ob[i - first] = oa[i - first] == r ? r : -2;
                           });
                  t.barrier();
              });

    const char* loop = name_of(k);
    const auto  lf   = static_cast<long long>(first);
    const auto  ln   = static_cast<long long>(n);
    check(size >= 1 && size <= nt, "team size in [1, asked]", loop, nt, lf, ln, chunk);
    if (size < 1)
    {
        return;
    }

    bool once_wait    = true;
    bool once_a       = true;
    bool on_rank_wait = true;
    bool on_rank_a    = true;
    bool same_share   = true;
    for (long long j = 0; j < ln; ++j)
    {
        const auto uj       = static_cast<std::size_t>(j);
        const int  expected = is_chunked(k) ? chunks_owner(j, chunk > 0 ? chunk : 1, size) : static_owner(j, ln, size);
        once_wait           = once_wait && hits_wait[uj] == 1;
        once_a              = once_a && hits_a[uj] == 1;
        on_rank_wait        = on_rank_wait && owner_wait[uj] == expected;
        on_rank_a           = on_rank_a && owner_a[uj] == expected;
        same_share          = same_share && owner_b[uj] == expected;
    }
    check(once_wait, "1: every iteration once (wait)", loop, nt, lf, ln, chunk);
    check(once_a, "1: every iteration once (nowait)", loop, nt, lf, ln, chunk);
    check(on_rank_wait, "2: the rank OpenMP gives each iteration (wait)", loop, nt, lf, ln, chunk);
    check(on_rank_a, "2: the rank OpenMP gives each iteration (nowait)", loop, nt, lf, ln, chunk);
    check(same_share, "3: two nowait loops in a row, the same share", loop, nt, lf, ln, chunk);
}

template <typename Index>
void check_all()
{
    // (first, n): empty, one, shorter than a team, offset, negative start,
    // multiples and not of the team and of the blocks.
    const std::vector<std::pair<long long, long long>> ranges = {
        {0, 0}, {0, 1}, {0, 5}, {7, 5}, {0, 100}, {13, 1001}, {-50, 333}, {0, 4096}, {1000, 8 * 64},
    };
    const loop_kind kinds[] = {loop_kind::static_blocks, loop_kind::static_blocks_simd, loop_kind::static_chunks,
                               loop_kind::static_chunks_simd};
    for (const int nt : {1, 2, 3, 8})
    {
        for (const auto& [first, n] : ranges)
        {
            for (const loop_kind k : kinds)
            {
                if (is_chunked(k))
                {
                    for (const int c : {0, 1, 3, 64})
                    {
                        check_one<Index>(k, nt, static_cast<Index>(first), static_cast<Index>(n), c);
                    }
                }
                else
                {
                    check_one<Index>(k, nt, static_cast<Index>(first), static_cast<Index>(n), 0);
                }
            }
        }
    }
}

} // namespace

int main()
{
    const char* backend = pmp::backend == pmp::backend_kind::openmp ? "openmp"
                        : pmp::backend == pmp::backend_kind::pool   ? "pool"
                                                                     : "serial";
    std::printf("team_loops_validation: backend %s\n", backend);
    check_all<int>();
    check_all<long long>();
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAILED");
    return g_fails == 0 ? 0 : 1;
}
