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

/// @file bench_stl_checks.cpp
/// @brief Bodies of the --test_stl self-checks. See bench_stl_checks.h for why
///        they live here rather than in main().

#include <algorithm>   // std::find
#include <cstddef>     // std::size_t
#include <iostream>    // std::cout
#include <numeric>     // std::accumulate -- STL self-checks
#include <stdexcept>   // std::out_of_range -- STL self-checks
#include <type_traits> // std::is_same_v -- STL self-checks
#include <utility>     // std::as_const -- STL self-checks
#include <vector>      // std::vector

#include "bench_array.h"      // Array, second_value_d1
#include "bench_stl_checks.h" // the declarations defined here
#include "bench_utils.h"      // maybe_set_second_value

// The terminal colours. Declared here rather than reached through the
// `using namespace shrink_array` of bench_init.cpp, which only leaks into this
// file in a unity build (SHRINK_BENCH_UNITY=OFF did not compile).
using shrink_array::tc;

void run_stl_self_checks([[maybe_unused]] bool test_stl)
{
    // Was a constexpr local of main(); the read-idempotence block below is its
    // only user here.
    [[maybe_unused]] constexpr double second_value_d1 = 2;

    // ─────────────────────────────────────────────────────────────────────────
    // STL-interface self-checks (--test_stl). Exercises the vector-like surface
    // added to zero::Array (typedefs, at/front/back, iterators, rbegin/rend,
    // assign/swap) against an independent oracle. Works in BOTH chunk and vector
    // modes: everything used here (const iteration, at, front/back, assign, swap)
    // exists on std::vector too, so the same code validates the "vector-like"
    // claim on the real std::vector baseline. Read-only iteration => no gate
    // interaction; assign/swap are serial and localised to this section.
    if (test_stl)
    {
        std::cout << tc.bold << tc.cyan << "\n=== STL interface self-checks ===\n" << tc.reset;
        Int  failures = 0;
        auto expect   = [&](bool ok, const char* what)
        {
            if (!ok)
            {
                ++failures;
                std::cout << "  [FAIL] " << what << "\n";
            }
            else
            {
                std::cout << "  [ ok ] " << what << "\n";
            }
        };

        // Fresh, deterministic array so the checks don't depend on the compute
        // loop's final state. Size spans several chunks to exercise boundaries.
        constexpr Int n_stl = 1000;
        Array<double> v(n_stl);
        for (Int i = 0; i < n_stl; ++i)
        {
            v[i] = static_cast<double>(i);
        }

        // Independent oracle.
        double oracle_sum = 0.0;
        for (Int i = 0; i < n_stl; ++i)
        {
            oracle_sum += static_cast<double>(i);
        }

        // --- member typedefs are present and sane -----------------------------
        static_assert(std::is_same_v<Array<double>::value_type, double>);
        static_assert(std::is_same_v<Array<double>::size_type, std::size_t>);
        expect(v.size() == static_cast<std::size_t>(n_stl), "size()");
        expect(!v.empty(), "empty() == false on non-empty");

        // --- forward const iteration (range-for) matches operator[] -----------
        {
            double it_sum = 0.0;
            Int    n      = 0;
            for (const double x : std::as_const(v))
            {
                it_sum += x;
                ++n;
            }
            expect(n == n_stl, "range-for visits every element");
            expect(it_sum == oracle_sum, "range-for sum == oracle");
        }

        // --- std::accumulate / std::find over const_iterator ------------------
        {
            const double acc = std::accumulate(v.begin(), v.end(), 0.0);
            expect(acc == oracle_sum, "std::accumulate == oracle");
            const auto it = std::find(v.begin(), v.end(), 512.0);
            expect(it != v.end() && *it == 512.0, "std::find locates value across a chunk boundary");
        }

        // --- reverse iteration visits elements in reverse ---------------------
        {
            double rev_sum       = 0.0;
            Int    n             = 0;
            double prev          = 1e300;
            bool   monotone_down = true;
            for (auto r = v.rbegin(); r != v.rend(); ++r)
            {
                if (*r > prev)
                {
                    monotone_down = false;
                }
                prev = *r;
                rev_sum += *r;
                ++n;
            }
            expect(n == n_stl, "rbegin/rend visit every element");
            expect(rev_sum == oracle_sum, "reverse sum == forward sum");
            expect(monotone_down, "reverse order is strictly descending");
        }

        // --- at() / front() / back() ------------------------------------------
        expect(v.at(0) == 0.0, "at(0)");
        expect(v.at(n_stl - 1) == double(n_stl - 1), "at(last)");
        expect(v.front() == 0.0, "front()");
        expect(v.back() == double(n_stl - 1), "back()");

        // at() out-of-range throws (conforming); front/back on empty throw.
        {
            bool threw = false;
            try
            {
                (void)v.at(n_stl);
            }
            catch (const std::out_of_range&)
            {
                threw = true;
            }
            expect(threw, "at(out-of-range) throws std::out_of_range");
        }
        // front()/back() on empty: our chunk container throws (safety choice);
        // std::vector leaves it UB, so only assert the throw in chunk mode.
        if constexpr (chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any)
        {
            Array<double> empty_arr(0);
            bool          threw = false;
            try
            {
                (void)empty_arr.front();
            }
            catch (const std::out_of_range&)
            {
                threw = true;
            }
            expect(threw, "front() on empty throws (chunk mode)");
        }

        // --- non-const at() returns a write proxy (assignment goes through it) -
        v.at(3) = -7.0;
        expect(v.at(3) == -7.0, "non-const at() assigns via proxy");
        v.at(3) = 3.0; // restore

        // --- assign(count, val) ----------------------------------------------
        {
            Array<double> a(5);
            a.assign(n_stl, 2.5);
            const std::size_t asz = a.size();
            bool              all = (asz == static_cast<std::size_t>(n_stl));
            for (std::size_t i = 0; all && i < asz; ++i)
            {
                all = (a[i] == 2.5);
            }
            expect(all, "assign(count,val) sizes and fills");
        }

        // --- swap() exchanges contents ----------------------------------------
        {
            Array<double> a(n_stl);
            Array<double> b(n_stl);
            for (Int i = 0; i < n_stl; ++i)
            {
                a[i] = 1.0;
                b[i] = 9.0;
            }
            a.swap(b);
            bool ok = true;
            for (Int i = 0; ok && i < n_stl; ++i)
            {
                ok = (a[i] == 9.0 && b[i] == 1.0);
            }
            expect(ok, "swap() exchanges element contents");
        }

        // --- capacity() / reserve() / push_back / emplace_back / pop_back ----
        // The growth by one: within the capacity a push_back is a size
        // increment and one write; reserve() moves the chunk allocation up
        // front. std::vector has the same words, so the baseline runs this too.
        {
            Array<double> g(1);
            g.reserve(n_stl);
            const std::size_t cap = g.capacity();
            expect(cap >= static_cast<std::size_t>(n_stl), "reserve(n): capacity() >= n");
            expect(g.size() == 1, "reserve() leaves size() alone");
            for (Int i = 1; i < n_stl; ++i)
            {
                g.push_back(static_cast<double>(i));
            }
            expect(g.size() == static_cast<std::size_t>(n_stl), "push_back grows size() by one each");
            expect(g.capacity() == cap, "push_back within the capacity does not change it");
            g.emplace_back(2.5);
            expect(g.size() == static_cast<std::size_t>(n_stl) + 1 && g.back() == 2.5, "emplace_back(args)");
            g.pop_back();
            expect(g.size() == static_cast<std::size_t>(n_stl) && g.back() == double(n_stl - 1), "pop_back");
            bool ok = true;
            for (Int i = 1; i < n_stl; ++i)
            {
                ok = ok && (std::as_const(g)[i] == static_cast<double>(i));
            }
            expect(ok, "push_back'd elements read back");
            // Past the capacity: the push_back that needs a new chunk.
            const std::size_t before = g.size();
            for (std::size_t i = g.size(); i < cap + 3; ++i)
            {
                g.push_back(-1.0);
            }
            expect(static_cast<std::size_t>(g.size()) == cap + 3 && g.capacity() > cap && g.back() == -1.0 &&
                       std::as_const(g)[static_cast<Int>(before) - 1] == double(n_stl - 1),
                   "push_back past the capacity grows the storage and keeps the contents");
            g.clear();
        }

        // --- (first, last) and {…} constructors, assign(first, last) ---------
        {
            std::vector<double> src(n_stl);
            for (Int i = 0; i < n_stl; ++i)
            {
                src[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i);
            }
            Array<double> r(src.begin(), src.end());
            bool          ok = static_cast<std::size_t>(r.size()) == src.size();
            for (Int i = 0; ok && i < n_stl; ++i)
            {
                ok = std::as_const(r)[i] == src[static_cast<std::size_t>(i)];
            }
            expect(ok, "(first, last) constructor copies the range");

            Array<double> il{1.0, 2.0, 3.0, 4.0};
            expect(il.size() == 4 && std::as_const(il)[0] == 1.0 && std::as_const(il)[3] == 4.0,
                   "initializer-list constructor");

            il.assign(src.begin() + 10, src.begin() + 20);
            expect(il.size() == 10 && std::as_const(il)[0] == 5.0 && std::as_const(il)[9] == 9.5,
                   "assign(first, last)");
            r.clear();
            il.clear();
        }

        // --- the copy is deep -------------------------------------------------
        // Copy constructor and copy-assignment: the copy reads back the source,
        // a write in the copy leaves the source alone (no aliasing of the
        // chunks), and a destination that held data is overwritten whole.
        {
            Array<double> c(v);
            bool          same = c.size() == v.size();
            for (Int i = 0; same && i < n_stl; ++i)
            {
                same = std::as_const(c)[i] == std::as_const(v)[i];
            }
            expect(same, "copy constructor: every element equal");
            c[5] = -99.0;
            expect(std::as_const(v)[5] == 5.0 && std::as_const(c)[5] == -99.0,
                   "a write in the copy leaves the source alone (deep, not an alias)");

            Array<double> d(3 * n_stl);
            for (Int i = 0; i < 3 * n_stl; ++i)
            {
                d[i] = 7.0;
            }
            d    = v;
            same = d.size() == v.size();
            for (Int i = 0; same && i < n_stl; ++i)
            {
                same = std::as_const(d)[i] == std::as_const(v)[i];
            }
            expect(same, "copy-assignment over a larger array: sized and equal");
            const auto& d_alias = d;       // through an alias: clang flags a literal `d = d`
            d                   = d_alias; // self-assignment: a no-op
            expect(std::as_const(d)[7] == 7.0, "self-assignment is a no-op");
            c.clear();
            d.clear();

            // chunk2: the array-level second value travels with the copy. A
            // generic lambda, so that the `if constexpr` really discards the
            // body on the arrays that have no get_second_value().
            [&]<typename A>(A*)
            {
                if constexpr (requires(A x) { x.get_second_value(); })
                {
                    A s2(n_stl);
                    maybe_set_second_value(s2, second_value_d1);
                    for (Int i = 0; i < n_stl; ++i)
                    {
                        s2[i] = (i % 3 == 0) ? second_value_d1 : 0.0;
                    }
                    A    t2(s2);
                    bool ok = t2.get_second_value() == second_value_d1;
                    for (Int i = 0; ok && i < n_stl; ++i)
                    {
                        ok = std::as_const(t2)[i] == std::as_const(s2)[i];
                    }
                    expect(ok, "copy constructor carries chunk2's second value and its runs");
                    s2.clear();
                    t2.clear();
                }
            }(static_cast<Array<double>*>(nullptr));
        }

        // --- shrink_to_fit(): folds the candidates, returns the parked buffers.
        // Observable here as: it returns, and the contents are unchanged.
        {
            Array<double> z(n_stl);
            for (Int i = 0; i < n_stl; ++i)
            {
                z[i] = (i < n_stl / 2) ? 0.0 : static_cast<double>(i);
            }
            z.shrink_to_fit();
            bool ok = z.size() == static_cast<std::size_t>(n_stl);
            for (Int i = 0; ok && i < n_stl; ++i)
            {
                ok = std::as_const(z)[i] == ((i < n_stl / 2) ? 0.0 : static_cast<double>(i));
            }
            expect(ok, "shrink_to_fit keeps size and contents");
            z.clear();
        }

        if (failures == 0)
        {
            std::cout << tc.bold << tc.green << "STL self-checks: ALL PASSED\n" << tc.reset;
        }
        else
        {
            std::cout << tc.bold << "STL self-checks: " << failures << " FAILURE(S)\n" << tc.reset;
        }
    }

    // === read-idempotence self-checks (--test_stl) ==========================
    // A const read must have NO observable side effect: two consecutive reads
    // of the same index return the same value, and the proxy comparison
    // operators (operator!=, operator==) agree with operator T() (read_pure()).
    // Under OpenMP the proxy's ctor-time values[] snapshot can go stale after a
    // chunk materialises or moves; a read op using that snapshot directly
    // instead of read_pure() would make "arr[i] != x" disagree with
    // "T v = arr[i]". Without OpenMP the reads are already pure, so it
    // trivially holds.
    if (test_stl)
    {
        std::cout << tc.bold << tc.cyan << "\n=== read-idempotence self-checks ===\n" << tc.reset;

        // Size spans many chunks (≈195 at chunk_size 512) so reads cross chunk
        // boundaries. Mixed data (mostly second_value, some distinct) exercises
        // both the cst2 read path and materialised reads — the two that can go
        // stale. maybe_set_second_value is not required for correctness: a
        // chunk2 array without it has T{} as second value (see
        // insert_only_num_tab), so it writes fine but only its zeros compress.
        // It is declared here, before the writes, so that the runs of 2.0
        // actually reach the cst2 read path this check is about (no-op on
        // chunk v1 / vector mode, per-chunk on chunk2A).
        const long    nr = 100000;
        Array<double> r(nr);
        maybe_set_second_value(r, second_value_d1); // second_value = 2.0
        for (long i = 0; i < nr; ++i)
        {
            r[i] = (i % 7 == 0) ? second_value_d1 : static_cast<double>(i);
        }

        long idem_bad = 0; // two consecutive reads disagree (read has a side effect)
        long ne_bad   = 0; // operator!= disagrees with operator T() (stale compare op)
        long first_i  = -1;
        for (long i = 0; i < nr; ++i)
        {
            const double a = r[i]; // operator T() -> read_pure (live)
            const double b = r[i]; // re-read: must equal a
            if (a != b)
            {
                if (first_i < 0)
                {
                    first_i = i;
                }
                ++idem_bad;
            }

            const double exp   = (i % 7 == 0) ? second_value_d1 : static_cast<double>(i);
            const bool   ne_op = (r[i] != exp); // operator!= path (was stale)
            const bool   ne_t  = (a != exp);    // T-based comparison (correct)
            if (ne_op != ne_t)
            {
                if (first_i < 0)
                {
                    first_i = i;
                }
                ++ne_bad;
            }
        }

        if (idem_bad == 0 && ne_bad == 0)
        {
            std::cout << tc.bold << tc.green << "read-idempotence: ALL PASSED\n" << tc.reset;
        }
        else
        {
            std::cout << tc.bold << "read-idempotence: FAILURE(S) — " << idem_bad << " non-idempotent reads, " << ne_bad
                      << " operator!=/operatorT disagreements"
                      << " (first at i=" << first_i << ")\n"
                      << tc.reset;
        }

        r.clear();
    }
}
