# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
#
# The Kokkos facade of device::Array (chunk/device/kokkos_adaptor.h, docs/gpu_design.md
# stage 3): Kokkos fetched at a fixed release and built here, for CHUNK_DEVICE_LANGUAGE
# as a CMake language (Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE: the .cu sources compiled
# as CUDA or HIP, as the device validations are, not through nvcc_wrapper); Serial for the
# host space. Only the include path of a library target, not its compile options (host
# flags such as -march would reach the device compiler).

if (NOT CHUNK_DEVICE_LANGUAGE MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR "CHUNK_DEVICE_KOKKOS: CHUNK_DEVICE_LANGUAGE is CUDA or HIP, not '${CHUNK_DEVICE_LANGUAGE}'")
endif()
if (CHUNK_KOKKOS_ARCH STREQUAL "")
  message(FATAL_ERROR "CHUNK_DEVICE_KOKKOS: set CHUNK_KOKKOS_ARCH (TURING75, HOPPER90, AMD_GFX942_APU, ...: "
                      "Kokkos_ARCH_<it>; no detection, the configure may run where there is no GPU)")
endif()
if (CHUNK_DEVICE_LANGUAGE STREQUAL "CUDA" AND CMAKE_VERSION VERSION_LESS 3.25.2)
  message(FATAL_ERROR "CHUNK_DEVICE_KOKKOS: Kokkos with CUDA as a CMake language needs CMake 3.25.2")
endif()
enable_language(${CHUNK_DEVICE_LANGUAGE})

set(Kokkos_ENABLE_${CHUNK_DEVICE_LANGUAGE} ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_${CHUNK_KOKKOS_ARCH} ON CACHE BOOL "" FORCE)
# rocThrust (Kokkos's HIP sort) is not needed, and not in a ROCm without its libraries.
set(Kokkos_ENABLE_ROCTHRUST OFF CACHE BOOL "" FORCE)
include(FetchContent)
# The extracted files take the extraction's time, not the archive's (CMP0135, CMake 3.24).
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
# Pinned by its SHA-256: a release tarball is not supposed to change, a tag can move.
# CHUNK_KOKKOS_URL: the tarball fetched beforehand (a path), where CMake cannot reach
# GitHub -- a proxy whose authentication CMake's curl lacks, a compute node offline.
set(CHUNK_KOKKOS_URL "https://github.com/kokkos/kokkos/archive/refs/tags/5.3.0.tar.gz" CACHE STRING
    "Kokkos 5.3.0 release tarball: its URL, or a local path to it (same SHA-256)")
FetchContent_Declare(Kokkos
  URL ${CHUNK_KOKKOS_URL}
  URL_HASH SHA256=71590c348be05d83b12b3b21ccac36072be0006e23ada3ff4cb91f018cfbb042)
FetchContent_MakeAvailable(Kokkos)

foreach (_t IN ITEMS kokkos_validation kokkos_shrink_bench)
  add_executable(${_t} ${_t}.cu)
  set_source_files_properties(${_t}.cu PROPERTIES LANGUAGE ${CHUNK_DEVICE_LANGUAGE})
  set_target_properties(${_t} PROPERTIES ${CHUNK_DEVICE_LANGUAGE}_STANDARD 20 ${CHUNK_DEVICE_LANGUAGE}_STANDARD_REQUIRED ON)
  target_include_directories(${_t} PRIVATE
          $<TARGET_PROPERTY:chunk_zero_any,INTERFACE_INCLUDE_DIRECTORIES>
          ${PROJECT_SOURCE_DIR}/fast_allocators/public)
  target_link_libraries(${_t} PRIVATE Kokkos::kokkos)
endforeach()
add_test(NAME kokkos_validation COMMAND kokkos_validation)
set_property(TEST kokkos_validation PROPERTY LABELS validation)

# kokkos_getting_started: the facade as a program uses it, linked to the installable target
# shrink_array::device (chunk_device: the include path and C++20) and to Kokkos only.
add_executable(kokkos_getting_started kokkos_getting_started.cu)
set_source_files_properties(kokkos_getting_started.cu PROPERTIES LANGUAGE ${CHUNK_DEVICE_LANGUAGE})
set_target_properties(kokkos_getting_started PROPERTIES ${CHUNK_DEVICE_LANGUAGE}_STANDARD 20
                      ${CHUNK_DEVICE_LANGUAGE}_STANDARD_REQUIRED ON)
target_link_libraries(kokkos_getting_started PRIVATE chunk_device Kokkos::kokkos)
add_test(NAME kokkos_getting_started COMMAND kokkos_getting_started)
set_property(TEST kokkos_getting_started PROPERTY LABELS validation)
