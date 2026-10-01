#pragma once

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

/**
 * @file bench_rmw_check.h
 * @brief Checks the RETURN CONTRACT of the compound operators in --check mode.
 *
 * `y = (a[i] op= x)` must yield the value now stored. The compound operators
 * return the element's proxy: discarded, it keeps the operation deferrable;
 * converted, it synchronises the offset and reads the result. The compute phase
 * is where that is exercised for real -- parallel loops, large arrays,
 * compression in flight, so some of these writes ARE deferred -- which an
 * isolated unit test cannot force.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <utility>

#include "bench_config.h" // mode_check
#include "bench_int.h"    // Int

/**
 * @brief Take the result of `arr[idx] op= x` and, in --check mode, verify it.
 *
 * @param arr    the array the operation was applied to
 * @param idx    the element index
 * @param result what the compound operator returned
 * @param expr   the operation's text, for the diagnostic
 *
 * @details Perf mode reads nothing: the result is discarded unread, exactly as
 *          `arr[idx] op= x;` would, so the operation stays deferrable. Check
 *          mode converts the result (a synchronised read), re-reads the element
 *          through a fresh proxy, and compares the two BIT BY BIT -- memcmp, not
 *          !=, because a NaN (which `tab2 /= tab1` can produce) never compares
 *          equal to itself. It compares with a re-read rather than recomputing
 *          `old op x` here: a recomputation could round differently (FMA
 *          contraction), while the stored values are already checked against
 *          the reference files. What this adds is "the value returned is the
 *          value stored" -- the contract the operators used to break (the
 *          operand, or an unspecified value on the deferred path).
 * @note Sound only because the compute loops use an explicit static schedule:
 *       each thread re-reads only elements it wrote itself.
 */
template <typename Arr, typename Result>
void sb_check_compound(Arr& arr, Int idx, Result&& result, const char* expr)
{
    using T = std::decay_t<Arr>::value_type;
    static_assert(std::is_trivially_copyable_v<T>, "bit-wise comparison needs a trivially copyable element");
    if (!mode_check)
    {
        return;
    }
    const T returned = static_cast<T>(std::forward<Result>(result));
    const T stored   = arr[idx];
    if (std::memcmp(&returned, &stored, sizeof(T)) != 0)
    {
        std::cerr << "[CHECK] compound assignment returned a value that is not the stored one: " << expr
                  << "  (i = " << idx << ")\n";
        std::abort();
    }
}

/// @brief `SB_CHECKED_RMW(tab, i, += x)` performs `tab[i] += x` and checks its
///        result in --check mode (see sb_check_compound). Variadic so that
///        commas in the operand need no extra parentheses.
#define SB_CHECKED_RMW(arr, idx, ...) sb_check_compound(arr, idx, arr[idx] __VA_ARGS__, #arr "[" #idx "] " #__VA_ARGS__)
