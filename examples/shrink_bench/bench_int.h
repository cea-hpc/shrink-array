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

#pragma once
/**
 * @file bench_int.h
 * @brief The bench's index and size type, in one place: `Int` / `UInt`.
 *
 * @details `int` indices cap a run at INT_MAX elements -- and `--mode perf`
 *          already sits at 1e9, so `--multiply 2` needs 64-bit indices
 *          (check_size_limits() refuses it otherwise).
 *
 *          Every size/index/count in the bench is spelled `Int` (or `UInt`):
 *          `int` by default, `long` when the build defines BENCH_INT64 (a "big
 *          multiply" run). `Array<T>` passes `Int` to the library as its
 *          `ind_type`, so the two move together: there is no build where the
 *          bench indexes in 64 bits and the array in 32.
 *
 * @note ELEMENT types are NOT affected. `Array<int> tab_int` stays an array of
 *       `int` in both builds: it is one of the element types the bench covers
 *       (next to short, long, float, ...), not an index.
 *
 * @note The 64-bit build is a PERF instrument first: its targets are not
 *       registered with ctest. `--mode check` works on it, against the
 *       reference files (data_tab*) of the 32-bit vector_check of the same
 *       configuration -- element values, independent of Int -- so a check
 *       there asserts that 64-bit indices give the 32-bit results
 *       (python/bench_and_test.py runs a few, INT64_CHECK_VARIANTS).
 *
 * Enable with `cmake -DSHRINK_BENCH_INT64=ON`, which adds the int64 targets
 * (shrink_bench_<variant>_int64_<size>, for the sizes of
 * SHRINK_BENCH_INT64_SIZES and dyn) next to the ordinary ones (both are kept,
 * neither replaces the other).
 */

#ifdef BENCH_INT64
/// @brief The bench's signed index/size type: 64-bit in this build.
using Int = long;
/// @brief The bench's unsigned counterpart: 64-bit in this build.
using UInt = unsigned long;
#else
/// @brief The bench's signed index/size type: 32-bit in this build (default).
using Int = int;
/// @brief The bench's unsigned counterpart: 32-bit in this build (default).
using UInt = unsigned int;
#endif
