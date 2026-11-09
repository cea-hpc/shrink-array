# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/bool_arrays: the two boolean containers of a zero_one_value build,
# zero_one_value::Array<bool> (a byte per element) and ArrayChunkBits (a bit per
# element), side by side (docs/variants.md, section 2).
#
# Builds two ways, as examples/getting_started:
#  - inside the shrink-array tree (examples/CMakeLists.txt adds it), against
#    the tree's chunk_zero_one_value library;
#  - on its own, against an INSTALLED shrink-array:
#        cmake -S examples/bool_arrays -B build-ba -DCMAKE_PREFIX_PATH=<prefix>
#        cmake --build build-ba && ./build-ba/bool_arrays
# It imports the shrink_array module: CMake 4.2, Ninja and import std, as
# examples/getting_started.
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # import std: CMake 4.2's gate value (it changes with CMake's version).
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
  project(bool_arrays LANGUAGES CXX)
  find_package(chunk 0.2 REQUIRED)
  set(BOOL_ARRAYS_MODULE chunk::chunk_zero_one_value_module CACHE STRING "The shrink-array module to import")
else()
  set(BOOL_ARRAYS_MODULE chunk_zero_one_value_module)
endif()

add_executable(bool_arrays bool_arrays.cpp)
# The variant's module carries, through its library, the compile definitions,
# include directories and -fopenmp (see examples/getting_started/CMakeLists.txt).
set_target_properties(bool_arrays PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_compile_features(bool_arrays PRIVATE cxx_std_23)
target_link_libraries(bool_arrays PRIVATE ${BOOL_ARRAYS_MODULE})

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME bool_arrays COMMAND bool_arrays)
  set_property(TEST bool_arrays PROPERTY LABELS validation examples)
  set_property(TEST bool_arrays PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
  set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS bool_arrays) # check builds it
endif()
