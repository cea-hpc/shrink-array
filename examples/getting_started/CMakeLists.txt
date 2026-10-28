# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/getting_started: the program of docs/getting_started.md.
#
# Builds two ways:
#  - inside the shrink-array tree (examples/CMakeLists.txt adds it), against
#    the tree's chunk_zero_any library;
#  - on its own, against an INSTALLED shrink-array, the way a user project
#    does it:
#        cmake -S examples/getting_started -B build-gs -DCMAKE_PREFIX_PATH=<prefix>
#        cmake --build build-gs && ./build-gs/getting_started
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(getting_started LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # Brings chunk::chunk_zero, chunk::chunk_zero_one_value, chunk::chunk_zero_any (default size),
  # chunk::chunk_zero_any_<size>, chunk::chunk_zero_any_dyn, ... and their dependencies
  # (fast_allocators, OpenMP and Threads when the library was built with them).
  find_package(chunk 0.1 REQUIRED)
  set(GETTING_STARTED_LIB chunk::chunk_zero_any CACHE STRING "The shrink-array library to link")
else()
  set(GETTING_STARTED_LIB chunk_zero_any)
endif()

add_executable(getting_started getting_started.cpp)
# Linking the variant library is all it takes: it carries its compile
# definitions (CHUNK_VARIANT_ZERO_ANY=1, CHUNK_SIZE or CHUNK_DYNAMIC), its include
# directories (with the generated chunk/build_options.h, the options it was
# built with) and -fopenmp when it was built with OpenMP.
target_link_libraries(getting_started PRIVATE ${GETTING_STARTED_LIB})

# getting_started_ompt.cpp: the same program with plain `omp parallel for`
# loops, no parallel_phase(), checked against a std::vector. Only with the
# OMPT tool (Clang, oneAPI): the runtime marks the phases. Its source refuses to
# compile without it (static_assert), and its test stops on a runtime that did
# not load the tool (CHUNK_REQUIRE_OMPT=1) instead of passing without phases.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_gs_ompt "${chunk_OMPT}")
else()
  set(_gs_ompt "${CHUNK_HAVE_OMPT}")
endif()
if (_gs_ompt)
  add_executable(getting_started_ompt getting_started_ompt.cpp)
  target_link_libraries(getting_started_ompt PRIVATE ${GETTING_STARTED_LIB})
  if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_test(NAME getting_started_ompt COMMAND getting_started_ompt)
    set_property(TEST getting_started_ompt PROPERTY ENVIRONMENT "CHUNK_REQUIRE_OMPT=1;OMP_NUM_THREADS=8")
  endif()
else()
  message(STATUS "getting_started_ompt: not built (no OMPT tool: Clang or oneAPI, OMP=ON)")
endif()

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME getting_started COMMAND getting_started)
  set_property(TEST getting_started PROPERTY LABELS examples)
  set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS getting_started) # check builds it
endif()
