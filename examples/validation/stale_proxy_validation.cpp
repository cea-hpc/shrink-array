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
// Stale-proxy harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property, in check builds (build_config::any_check: Debug, or
// CHECK_OPTIM): a write proxy used after its array rebuilt its storage stops
// with an invariant failure instead of reading or writing released chunks.
//
// A proxy holds references into the array's chunks (README, "The write
// proxy"). `auto v = a[i];` keeps one -- where std::vector's auto copies --
// and every operation that rebuilds the storage leaves it dangling: resize,
// reserve, shrink_to_fit, clear, assign, swap, a copy or move assignment, a
// re-chunk. The array counts those rebuilds (storage_generation_), operator[]
// hands the count to the proxy, and every read and write entry of the proxy
// compares it (ProxyBase::check_not_stale).
//
// What it does: each case runs in a child (this executable re-run with
// --case <k>, a fresh process rather than a fork, the OpenMP runtime being
// what it is). The child takes `auto p = a.element_proxy(10);` (what `auto p = a[10];` keeps), uses it once, performs
// the case's operation, and uses p again. A rebuild case must stop on the invariant, with its message; a control case
// (nothing rebuilt, or a fresh proxy taken after the rebuild) must run to the end. Outside check builds the detection
// is compiled out and the harness only says so.
// ============================================================================

#include <cstdio>      // std::fopen, std::fclose, std::fread, FILE, std::remove
#include <cstdlib>     // std::system, std::atoi
#include <string>      // std::string, std::to_string
#include <type_traits> // std::conditional_t
#include <utility>     // std::move
#include <unistd.h>    // getpid
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS

#include "chunk/chunk_traits.h"   // forward decls + default IndT = int
#include "chunk/array_chunk.h"    // zero::Array
#include "chunk/array_chunk2.h"   // zero_one_value::Array
#include "chunk/array_chunk2A.h"  // zero_any::Array
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/chunk_std.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"         // chunk_config::use_chunk2*, build_config::chunk_dynamic / any_check
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::stale_proxy_validation
{

using namespace shrink_array;

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

constexpr long n_items = 4096;

struct Case
{
    const char* name;
    bool        must_stop; // a rebuild case; false: a control case
    bool        dyn_only;  // rechunk_to exists in dynamic-geometry builds only
};

const Case case_table[] = {
    {"control: no rebuild", false, false},
    {"control: push_back within the capacity", false, false},
    {"control: fresh proxy after resize", false, false},
    {"resize (grow), then write", true, false},
    {"resize (shrink), then write", true, false},
    {"resize, then read", true, false},
    {"resize, then +=", true, false},
    {"resize, then ++", true, false},
    {"reserve", true, false},
    {"clear", true, false},
    {"assign", true, false},
    {"swap", true, false},
    {"move assignment", true, false},
    {"copy assignment", true, false},
    {"shrink_to_fit", true, false},
    {"push_back beyond the capacity", true, false},
    {"rechunk_to", true, true},
};
constexpr int nb_cases = static_cast<int>(sizeof case_table / sizeof case_table[0]);

/// The child: one case. A template so that rechunk_to, which a static build
/// does not have, is only named in a discarded branch of a dependent call.
template <typename A>
static int run_case(int k)
{
    A    a(n_items);
    A    b(n_items);
    auto p   = a.element_proxy(10); // the proxy itself: a[10] refuses to be named in check builds
    p        = 1;                   // armed and valid
    int sink = 0;

    switch (k)
    {
        case 0:
            break;
        case 1:
            while (a.size() + 1 < static_cast<long>(a.capacity()))
            {
                a.push_back(0);
            }
            break;
        case 2:
            a.resize(2 * n_items);
            {
                auto q = a.element_proxy(10); // taken after the rebuild: valid
                q      = 2;
                sink += static_cast<int>(q);
            }
            println("[child] control done");
            return 0;
        case 3:
        case 5:
        case 6:
        case 7:
            a.resize(2 * n_items);
            break;
        case 4:
            a.resize(n_items / 2);
            break;
        case 8:
            a.reserve(static_cast<long>(a.capacity()) + 10 * n_items);
            break;
        case 9:
            a.clear();
            a.resize(n_items); // the element exists again; the proxy is still stale
            break;
        case 10:
            a.assign(n_items, 3);
            break;
        case 11:
            a.swap(b);
            break;
        case 12:
            a = std::move(b);
            a.resize(n_items);
            break;
        case 13:
            a = b;
            break;
        case 14:
            a.shrink_to_fit();
            break;
        case 15:
            while (a.size() < static_cast<long>(a.capacity()))
            {
                a.push_back(0);
            }
            a.push_back(0);
            break;
        case 16:
            if constexpr (build_config::chunk_dynamic)
            {
                a.rechunk_to(a.chunk_size() * 2);
            }
            break;
        default:
            return 2;
    }

    // The use after the operation: what a rebuild case must stop on.
    if (k == 5)
    {
        sink += static_cast<int>(p);
    }
    else if (k == 6)
    {
        p += 1;
    }
    else if (k == 7)
    {
        ++p;
    }
    else
    {
        p = 4;
    }

    println("[child] ran to the end (sink {})", sink);
    return 0;
}

static int run_child(int k)
{
    ParamChunk::init_indexes(true);
    const int rc = run_case<Array<int>>(k);
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}

/// Re-runs this executable on case @p k and checks the outcome.
static int check_case(const char* self, int k)
{
    const Case&       c   = case_table[k];
    const std::string log = "/tmp/stale_proxy_" + std::to_string(getpid()) + "_" + std::to_string(k) + ".log";
    const std::string cmd = validation::self_command(self) + " --case " + std::to_string(k) + " > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    std::string       out;
    if (FILE* f = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        {
            out.append(buf, n);
        }
        std::fclose(f);
    }
    std::remove(log.c_str());

    const bool ran_ok  = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    const bool message = out.find("used after the array rebuilt its storage") != std::string::npos;
    const bool pass    = c.must_stop ? (!ran_ok && message) : (ran_ok && !message);
    println("[{}] {:<40} {}{}", variant_name(), c.name,
            c.must_stop ? (ran_ok ? "RAN ON" : "stopped") : (ran_ok ? "ran" : "STOPPED"), pass ? "" : "   <-- FAIL");
    if (!pass)
    {
        println("---- child output ----\n{}----------------------", out.c_str());
        return 1;
    }
    return 0;
}

int run(int argc, char** argv)
{
    if (argc > 2 && std::string(argv[1]) == "--case")
    {
        return run_child(std::atoi(argv[2]));
    }

    println("=== stale_proxy_validation [variant = {}] ===", variant_name());
    if constexpr (!build_config::any_check)
    {
        println("[{}] not a check build: the stale-proxy check is compiled out, nothing to test", variant_name());
        return 0;
    }

    int fails = 0;
    for (int k = 0; k < nb_cases; ++k)
    {
        if (case_table[k].dyn_only && !build_config::chunk_dynamic)
        {
            continue;
        }
        fails += check_case(argv[0], k);
    }

    println("\n[{}] {}", variant_name(), fails == 0 ? "=== STALE PROXY OK ===" : "=== STALE PROXY FAILURES ===");
    return fails;
}

} // namespace validation::stale_proxy_validation
