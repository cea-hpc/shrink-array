# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/getting_started: the program of docs/getting_started.md.
#
# Builds two ways:
#  - inside the shrink-array tree (examples/CMakeLists.txt adds it), against
#    the tree's chunk_v2a library;
#  - on its own, against an INSTALLED shrink-array, the way a user project
#    does it:
#        cmake -S examples/getting_started -B build-gs -DCMAKE_PREFIX_PATH=<prefix>
#        cmake --build build-gs && ./build-gs/getting_started
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(getting_started LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # Brings chunk::chunk_v1, chunk::chunk_v2, chunk::chunk_v2a (default size),
  # chunk::chunk_v2a_<size>, chunk::chunk_v2a_dyn, ... and their dependencies
  # (fast_allocators, OpenMP and Threads when the library was built with them).
  find_package(chunk 0.1 REQUIRED)
  set(GETTING_STARTED_LIB chunk::chunk_v2a CACHE STRING "The shrink-array library to link")
else()
  set(GETTING_STARTED_LIB chunk_v2a)
endif()

add_executable(getting_started getting_started.cpp)
# Linking the variant library is all it takes: it carries its compile
# definitions (CHUNK_VARIANT_V2A=1, CHUNK_SIZE or CHUNK_DYNAMIC), its include
# directories (with the generated chunk/build_options.h, the options it was
# built with) and -fopenmp when it was built with OpenMP.
target_link_libraries(getting_started PRIVATE ${GETTING_STARTED_LIB})

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME getting_started COMMAND getting_started)
endif()
