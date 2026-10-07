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
 * @file ompt_off.cpp
 * @brief The manual executable: no OMPT tool, the switch the library reads
 *        says so.
 *
 * The variant of the library this example links carries no tool of its own
 * (chunk_v1v2_<size>, NO_OMPT_TOOL), so the switch has to come from the
 * program: manage_ompt.cpp in the embedded executable, this file in the
 * manual one, where the program makes the phase calls itself (config.h).
 */

#include "chunk/config.h" // OmptSuspendCounter

/// @brief The suspension counter read by chunk_config::OMPTcc (chunk/config.h),
///        non-zero: no callback exists in this binary.
shrink_array::OmptSuspendCounter shrink_array::g_ompt_suspended{1};
