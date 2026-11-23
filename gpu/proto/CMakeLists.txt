# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
#
# Stage-0 GPU prototype (docs/gpu_design.md section 13): a standalone project, out of the
# library's build. CUDA with nvcc or clang:
#
#   cmake -S gpu/proto -B build-proto -DCMAKE_CUDA_ARCHITECTURES=86 \
#         -DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=<g++ <= the toolkit's max>
#   cmake -S gpu/proto -B build-proto -DCMAKE_CUDA_ARCHITECTURES=86 \
#         -DCMAKE_CUDA_COMPILER=<llvm>/bin/clang++ -DCUDAToolkit_ROOT=<cuda>
#
# HIP (MI300A): -DPROTO_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942, with hipcc or clang as HIP compiler.

cmake_minimum_required(VERSION 3.25)
project(shrink_array_gpu_proto LANGUAGES CXX)

option(PROTO_HIP "Build for AMD with HIP instead of CUDA" OFF)

if (PROTO_HIP)
  enable_language(HIP)
  set(PROTO_LANG HIP)
else ()
  enable_language(CUDA)
  set(PROTO_LANG CUDA)
  find_package(CUDAToolkit REQUIRED)
endif ()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

foreach (prog proto_test proto_bench)
  add_executable(${prog} ${prog}.cu)
  set_source_files_properties(${prog}.cu PROPERTIES LANGUAGE ${PROTO_LANG})
  set_target_properties(${prog} PROPERTIES ${PROTO_LANG}_STANDARD 20 ${PROTO_LANG}_STANDARD_REQUIRED ON)
  target_include_directories(${prog} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # Device code optimised whatever the compiler: clang does not optimise it by default.
  target_compile_options(${prog} PRIVATE $<$<COMPILE_LANGUAGE:${PROTO_LANG}>:-O3>)
  if (NOT PROTO_HIP)
    target_link_libraries(${prog} PRIVATE CUDA::cudart_static) # no LD_LIBRARY_PATH to set
  endif ()
endforeach ()

enable_testing()
add_test(NAME proto_test COMMAND proto_test)
