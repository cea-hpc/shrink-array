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
#
# getting_started.cpp imports the shrink_array module (`import std;`,
# `import shrink_array;`): CMake 4.2, the Ninja generator and import std, whose
# experimental gate is set before project(), as in the library's own build.
cmake_minimum_required(VERSION 3.20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # import std: CMake 4.2's gate value (it changes with CMake's version).
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
  project(getting_started LANGUAGES CXX)
  # Brings chunk::chunk_zero, chunk::chunk_zero_one_value, chunk::chunk_zero_any (default size),
  # chunk::chunk_zero_any_<size>, chunk::chunk_zero_any_dyn, ..., the modules over the
  # default-size ones (chunk::chunk_zero_any_module, ...) and their dependencies
  # (fast_allocators, OpenMP and Threads when the library was built with them).
  find_package(chunk 0.2 REQUIRED)
  set(GETTING_STARTED_MODULE chunk::chunk_zero_any_module CACHE STRING "The shrink-array module to import")
else()
  set(GETTING_STARTED_MODULE chunk_zero_any_module)
endif()

add_executable(getting_started getting_started.cpp)
# Linking the variant's module is all it takes: through its library it carries
# the compile definitions (CHUNK_VARIANT_ZERO_ANY=1, CHUNK_SIZE or CHUNK_DYNAMIC),
# the include directories (with the generated chunk/build_options.h, the
# options it was built with) and -fopenmp when it was built with OpenMP. The
# tree keeps CMP0155 OLD (cmake_minimum_required 3.20): scan the source for its
# imports; import std needs C++23 asked as a feature.
set_target_properties(getting_started PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_compile_features(getting_started PRIVATE cxx_std_23)
target_link_libraries(getting_started PRIVATE ${GETTING_STARTED_MODULE})

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
  # It imports the module too, as getting_started.cpp.
  set_target_properties(getting_started_ompt PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
  target_compile_features(getting_started_ompt PRIVATE cxx_std_23)
  target_link_libraries(getting_started_ompt PRIVATE ${GETTING_STARTED_MODULE})
  if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_test(NAME getting_started_ompt COMMAND getting_started_ompt)
    set_property(TEST getting_started_ompt PROPERTY ENVIRONMENT "CHUNK_REQUIRE_OMPT=1;OMP_NUM_THREADS=8")
  endif()
else()
  message(STATUS "getting_started_ompt: not built (no OMPT tool: Clang or oneAPI, OMP=ON)")
endif()

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME getting_started COMMAND getting_started)
  set_property(TEST getting_started PROPERTY LABELS validation examples)
  set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS getting_started) # check builds it
endif()

# module_api.cpp: the names the module exports from its purview headers
# (chunk exchange, advisor, decider, mdspan), each used through the import,
# once per variant module (module_api_<variant>). Inside the tree only.
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  foreach(_mv IN LISTS CHUNK_VARIANTS)
    add_executable(module_api_${_mv} module_api.cpp)
    chunk_link_module(module_api_${_mv} chunk_${_mv})
    add_test(NAME module_api_${_mv} COMMAND module_api_${_mv})
    set_property(TEST module_api_${_mv} PROPERTY LABELS validation)
    set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS module_api_${_mv})
  endforeach()
endif()
