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

#ifdef __INTEL_COMPILER

    #ifdef DEBUG
        #define FORCE_INLINE inline
    #else
        #define FORCE_INLINE __forceinline
    #endif

#elif defined(__GNUC__) || defined(__clang__)
    #ifdef DEBUG
        #define FORCE_INLINE inline
    #else
        // `inline` included, as in the DEBUG branch (and as __forceinline
        // implies): GNU warns "always_inline function might not be
        // inlinable" on a function not declared inline -- e.g. a member
        // template defined out of class -- and a caller cannot add `inline`
        // itself without a duplicate-specifier error in DEBUG.
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#else
    #define FORCE_INLINE inline
#endif

// ---------------------------------------------------------------------------
// SA_FLATTEN — inline the WHOLE call tree into the marked function. Powerful on
// a hot leaf like operator[], but it blows up compile time, so it is:
//   * opt-in, behind -DSHRINK_AGGRESSIVE_INLINE (off by default), and
//   * never emitted for nvc++ (__NVCOMPILER), whose optimizer is already very
//     slow to compile this heavily-templated code.
// GNU/clang only; a no-op elsewhere.
// ---------------------------------------------------------------------------
#if defined(SHRINK_AGGRESSIVE_INLINE) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER)
    #define SA_FLATTEN __attribute__((flatten))
#else
    #define SA_FLATTEN
#endif

// ---------------------------------------------------------------------------
// SA_ASSUME(expr) — C++23 [[assume(expr)]] (the library requires C++23). NOTE:
// expr MUST be side-effect free -- no function call either: clang drops an
// assumption it cannot prove side-effect free (-Wassume) -- and the compiler
// treats it as UB-if-false and may choose not to evaluate it.
// ---------------------------------------------------------------------------
#define SA_ASSUME(expr) [[assume(expr)]]
