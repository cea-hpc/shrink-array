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
 * @file host_device.h
 * @brief CHUNK_HD: the functions of the core layer -- geometry, value identity, the
 *        constant walk -- compiled for the host and, in a CUDA or HIP translation unit,
 *        for the device too (docs/gpu_design.md, stage 1).
 *
 * @details A plain C++ build sees nothing: CHUNK_HD is empty. Under nvcc, clang -x cuda
 *          or clang -x hip it is `__host__ __device__`. A function marked CHUNK_HD only
 *          calls CHUNK_HD or constexpr functions, allocates nothing, throws nothing and
 *          touches no atomic, thread-local or global state.
 */

#if defined(__CUDACC__) || defined(__HIP__)
    #define CHUNK_HD __host__ __device__
#else
    #define CHUNK_HD
#endif
