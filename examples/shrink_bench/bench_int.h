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
 * @details The bench used to compute every size, index and loop bound in
 *          `int`, and so did the library underneath it (`ind_type = int`, see
 *          chunk_traits.h). That caps a run at INT_MAX elements -- and
 *          `--mode perf` already sits at 1e9, so `--multiply 2` was refused by
 *          check_size_limits() for want of 64-bit indices, which is the whole
 *          reason this header exists.
 *
 *          Every size/index/count in the bench is spelled `Int` (or `UInt`)
 *          instead. It is `int` by default -- the 32-bit bench is bit-for-bit
 *          what it was -- and `long` when the build defines BENCH_INT64, which
 *          is what a "big multiply" run wants. `Array<T>` passes `Int` to the
 *          library as its `ind_type`, so the two move together: there is no
 *          build where the bench indexes in 64 bits and the array in 32.
 *
 * @note ELEMENT types are NOT affected. `Array<int> tab_int` stays an array of
 *       `int` in both builds: it is one of the element types the bench covers
 *       (next to short, long, float, ...), not an index.
 *
 * @note The 64-bit build is a PERF instrument only -- `--mode check` is not
 *       supported on it and its targets are not registered with ctest. The
 *       reference files a check compares against (data_tab*) are written by
 *       the 32-bit build, and nothing re-generates them at 64 bits; see
 *       examples/shrink_bench/CMakeLists.txt.
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
