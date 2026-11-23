# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/bool_arrays: the two boolean containers of a zero_or_value build,
# zero_or_value::Array<bool> (a byte per element) and BitArray (a bit per
# element), side by side (docs/variants.md, section 2).
#
# Builds two ways, as examples/getting_started:
#  - inside the shrink-array tree (examples/CMakeLists.txt adds it), against
#    the tree's chunk_zero_or_value library;
#  - on its own, against an INSTALLED shrink-array:
#        cmake -S examples/bool_arrays -B build-ba -DCMAKE_PREFIX_PATH=<prefix>
#        cmake --build build-ba && ./build-ba/bool_arrays
# With -DCHUNK_MODULES=ON (experimental) it imports the shrink_array module
# instead of including the headers (see examples/getting_started/CMakeLists.txt).
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(CHUNK_MODULES "Import the shrink_array module (experimental: CMake 4.2, Ninja)" OFF)
  if (CHUNK_MODULES)
    # import std: CMake 4.2's gate value (it changes with CMake's version).
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
  endif()
  project(bool_arrays LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(shrink_array 0.4 REQUIRED)
  set(BOOL_ARRAYS_LIB shrink_array::zero_or_value CACHE STRING "The shrink-array library to link")
  set(BOOL_ARRAYS_MODULE shrink_array::zero_or_value_module CACHE STRING "The shrink-array module to import (CHUNK_MODULES)")
else()
  set(BOOL_ARRAYS_LIB chunk_zero_or_value)
endif()

add_executable(bool_arrays bool_arrays.cpp)
# The variant library carries its compile definitions, include directories and
# -fopenmp (see examples/getting_started/CMakeLists.txt); its module too.
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  chunk_use_library(bool_arrays ${BOOL_ARRAYS_LIB})
elseif (CHUNK_MODULES)
  set_target_properties(bool_arrays PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
  target_compile_features(bool_arrays PRIVATE cxx_std_23)
  target_compile_definitions(bool_arrays PRIVATE SHRINK_ARRAY_IMPORT=1)
  target_link_libraries(bool_arrays PRIVATE ${BOOL_ARRAYS_MODULE})
else()
  target_link_libraries(bool_arrays PRIVATE ${BOOL_ARRAYS_LIB})
endif()

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME bool_arrays COMMAND bool_arrays)
  set_property(TEST bool_arrays PROPERTY LABELS examples)
  set_property(TEST bool_arrays PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
  set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS bool_arrays) # check builds it
endif()
