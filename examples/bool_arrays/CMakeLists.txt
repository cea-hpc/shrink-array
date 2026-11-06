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
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(bool_arrays LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(chunk 0.2 REQUIRED)
  set(BOOL_ARRAYS_LIB chunk::chunk_zero_one_value CACHE STRING "The shrink-array library to link")
else()
  set(BOOL_ARRAYS_LIB chunk_zero_one_value)
endif()

add_executable(bool_arrays bool_arrays.cpp)
# The variant library carries its compile definitions, include directories and
# -fopenmp (see examples/getting_started/CMakeLists.txt).
target_link_libraries(bool_arrays PRIVATE ${BOOL_ARRAYS_LIB})

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME bool_arrays COMMAND bool_arrays)
  set_property(TEST bool_arrays PROPERTY LABELS examples)
  set_property(TEST bool_arrays PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
  set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS bool_arrays) # check builds it
endif()
