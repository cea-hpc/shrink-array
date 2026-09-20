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
 * @file bench_stl_checks.h
 * @brief The --test_stl self-checks, in their own translation unit.
 */

#pragma once

/**
 * @brief Run the STL-interface and read-idempotence self-checks.
 *
 * @param test_stl the --test_stl flag; false makes this a no-op, exactly as
 *        the two `if (test_stl)` guards did when these sections were inline
 *        in main().
 *
 * @details Split out of shrink_bench.cpp for COMPILE TIME, not for tidiness.
 *          These ~215 lines instantiate a wide slice of the STL surface
 *          (iterators, reverse iterators, std::accumulate / std::find, at,
 *          front/back, assign, swap) over the element types, in a translation
 *          unit that the build compiles 20 times. On a machine with cores to
 *          spare the wall-clock floor of the whole build is the LONGEST single
 *          compile job, so shortening that job is what moves it -- splitting
 *          does nothing for a build already saturated at -j6, and costs a
 *          little there (each fragment re-reads the same headers).
 *
 *          Self-contained on purpose: the checks build their own arrays rather
 *          than reading main()'s workload state, which is what made them the
 *          first section that could leave.
 */
void run_stl_self_checks(bool test_stl);
