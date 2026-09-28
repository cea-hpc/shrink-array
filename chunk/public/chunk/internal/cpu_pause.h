#pragma once

#include <thread> // std::this_thread::yield (the ARM and fallback CPU_PAUSE)

/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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
#ifdef __ARM_ARCH
    #if defined(__GNUC__) || defined(__clang__)
        #define CPU_PAUSE() asm volatile("yield" ::: "memory")
    #else
        #define CPU_PAUSE() std::this_thread::yield()
    #endif
#elif defined(__x86_64__) || defined(__i386__) // x86/x64, gcc and clang
    // Was `_M_IX86 || _M_X64`: MSVC's macros, never defined by gcc or clang,
    // so every x86 Linux build fell through to std::this_thread::yield() --
    // a sched_yield() SYSCALL per spin iteration, in every wait loop of the
    // library (rank flush locks, start()'s writer wait, the flush drains),
    // and the "yield after 1000 spins" discipline on top of it meant nothing.
    // Seen as 1-7% of the perf bench in __sched_yield at 8 threads on x86;
    // the AArch64 cluster always had the `yield` instruction above.
    #define CPU_PAUSE() __builtin_ia32_pause()
#elif defined(_M_IX86) || defined(_M_X64) // x86/x64, MSVC
    #include <intrin.h>
    #define CPU_PAUSE() _mm_pause()
#else
    #define CPU_PAUSE() std::this_thread::yield()
#endif