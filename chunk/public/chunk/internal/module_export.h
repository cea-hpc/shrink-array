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
 * @brief SHRINK_ARRAY_EXPORT: `export` where modules/shrink_array.cppm reads a
 *        header in the module's purview, nothing for an ordinary #include.
 *
 * A header whose declarations the module exports themselves (rather than
 * through a using-declaration in the .cppm) is read twice by the module unit:
 * - in the global module fragment, with SHRINK_ARRAY_MODULE_DEPS defined: its
 *   #includes only, so that they are global-module text there;
 * - in the purview, inside `extern "C++"`: its declarations, the exported ones
 *   marked SHRINK_ARRAY_EXPORT. extern "C++" keeps them attached to the global
 *   module, the same entities as in the compiled library and in any program
 *   that #includes the header: one ParamChunk::block_size, one set of pools.
 *
 * So such a header has, instead of #pragma once, a double guard: G for an
 * ordinary read (set on entry, as #pragma once would be), G_DEPS for the
 * #includes-only read -- an _inline.h tail includes its header back, and that
 * pass must not loop -- and its declarations sit under
 * #ifndef SHRINK_ARRAY_MODULE_DEPS. Its _inline.h tail, included after that
 * #endif, is read the same way.
 * Only a header that no other header of the module includes can be read so:
 * otherwise its declarations are already in the global module fragment. A
 * header that includes one for its users' convenience skips it when
 * SHRINK_ARRAY_MODULE_UNIT is defined (by the module unit, before any
 * #include): chunk/zero_chunk.h and chunk/shutdown.h.
 */

#ifndef SHRINK_ARRAY_EXPORT
    #define SHRINK_ARRAY_EXPORT
#endif
