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

#pragma once

/**
 * @file module_export.h
 * @brief SHRINK_ARRAY_EXPORT: `export` where modules/shrink_array.cppm reads the
 *        library, nothing for an ordinary #include.
 *
 * The module unit includes the library's external headers (standard, system,
 * OpenMP, third party) in its global module fragment, then the library's own
 * headers in its purview, inside `extern "C++"`: every declaration is the
 * module's, still attached to the global module -- the same entities as in the
 * compiled library and in any program that #includes the headers: one
 * ParamChunk::block_size, one set of pools. What a header marks
 * SHRINK_ARRAY_EXPORT is exported; the rest stays reachable (an importer's
 * instantiations use it) but cannot be named by the importer.
 *
 * A name first declared elsewhere unmarked (a forward declaration) cannot be
 * exported by a later marked declaration: the module exports it by a
 * using-declaration instead (the variants' Array).
 */

#ifndef SHRINK_ARRAY_EXPORT
    #define SHRINK_ARRAY_EXPORT
#endif
