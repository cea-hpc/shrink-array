# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/fetch_content: shrink-array consumed by another project's build.
#
# What a user project does when it does not install the library: it brings the
# sources in with FetchContent and links a chunk:: target, as it would after
# find_package(chunk). Not part of the shrink-array build (examples/ does not
# add it): configured on its own, by the CI.
#
#     cmake -S examples/fetch_content -B build-fc
#     cmake --build build-fc && ./build-fc/consumer
#
# The sources come from this tree (SHRINK_ARRAY_SOURCE_DIR, two levels up by
# default). A real project gives the repository and a tag instead:
#
#     FetchContent_Declare(shrink_array
#         GIT_REPOSITORY https://github.com/cea-hpc/shrink-array.git
#         GIT_TAG        v0.2.1)
#
# It also checks what a consumer is owed: its own build type, flags and
# targets are left alone (a `doc` target of its own, no test of the library
# among its tests, nothing of the library in its install).
cmake_minimum_required(VERSION 3.20)
project(shrink_array_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
enable_testing()

# The consumer's own choices, made before the library is brought in.
set(_consumer_release_flags "${CMAKE_CXX_FLAGS_RELEASE}")
add_custom_target(doc COMMAND ${CMAKE_COMMAND} -E echo "the doc target of the consumer" VERBATIM)

# The library's options are set like any cache variable, before the fetch.
set(OMP ON CACHE BOOL "shrink-array: OpenMP")
set(CHUNK_STATIC_SIZES 256 CACHE STRING "shrink-array: one static chunk size")
set(CHUNK_BUILD_DYN OFF CACHE BOOL "shrink-array: no runtime-size library")

set(SHRINK_ARRAY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "The shrink-array sources")
include(FetchContent)
# EXCLUDE_FROM_ALL (CMake 3.28): only the libraries the consumer links are
# built, and `cmake --install` of the consumer installs nothing of the library.
# Without it every variant is built and the library's headers and archives go
# into the consumer's prefix.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    FetchContent_Declare(shrink_array SOURCE_DIR "${SHRINK_ARRAY_SOURCE_DIR}" EXCLUDE_FROM_ALL)
else()
    FetchContent_Declare(shrink_array SOURCE_DIR "${SHRINK_ARRAY_SOURCE_DIR}")
endif()
FetchContent_MakeAvailable(shrink_array)

# The program of docs/getting_started.md, from the fetched sources.
add_executable(consumer "${shrink_array_SOURCE_DIR}/examples/getting_started/getting_started.cpp")
target_link_libraries(consumer PRIVATE chunk::chunk_zero_any)
add_test(NAME consumer COMMAND consumer)
install(TARGETS consumer RUNTIME DESTINATION bin)

# What the library must not have touched.
if(NOT CMAKE_CXX_FLAGS_RELEASE STREQUAL _consumer_release_flags)
    message(FATAL_ERROR "shrink-array changed the consumer's CMAKE_CXX_FLAGS_RELEASE: "
                        "'${_consumer_release_flags}' -> '${CMAKE_CXX_FLAGS_RELEASE}'")
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "shrink-array changed the consumer's CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()
