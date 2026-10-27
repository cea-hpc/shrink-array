# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(fast_allocators LANGUAGES CXX)

# Built on its own only: inside shrink_array, or inside a consuming project, the
# build type is the top level's.
if(NOT CMAKE_BUILD_TYPE AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif ()

if (ALLOCATORS_OMP)
   find_package(OpenMP REQUIRED)
endif ()


add_library(fast_allocators)
add_library(fast_allocators::fast_allocators ALIAS fast_allocators)

set(ALLOCATORS_MAJOR_VERSION 2)  # inc. when incompatible change (reset to 0 others)
set(ALLOCATORS_MINOR_VERSION 3)  # inc. when compatible change (also reset patch vers.)
set(ALLOCATORS_PATCH_VERSION 0)  # inc. when just fixing bug

set(ALLOCATORS_VERSION
  ${ALLOCATORS_MAJOR_VERSION}.${ALLOCATORS_MINOR_VERSION}.${ALLOCATORS_PATCH_VERSION})

set(ALLOCATORS_PUBLIC_HEADERS
  public/fast_allocators/slab_arena.h
  public/fast_allocators/tc_allocator.h
)

# The implementation the two facades pull in: installed with them, not API.
set(ALLOCATORS_INTERNAL_HEADERS
  public/fast_allocators/internal/alloc_config.h
  public/fast_allocators/internal/fatal.h
  public/fast_allocators/internal/int_alloc.h
  public/fast_allocators/internal/lifetime.h
  public/fast_allocators/internal/log_core.h
  public/fast_allocators/internal/next_power_of_2.h
  public/fast_allocators/internal/pmp_alloc.h
  public/fast_allocators/internal/pmp_loops.h
  public/fast_allocators/internal/pmp_std.h
  public/fast_allocators/internal/pmp_std_state.h
  public/fast_allocators/internal/pmp_runtime.h
  public/fast_allocators/internal/raw_memory.h
  public/fast_allocators/internal/singleton.h
  public/fast_allocators/internal/static_const_array.h
  public/fast_allocators/internal/static_const_array_inline.h
  public/fast_allocators/internal/svalloc.h
  public/fast_allocators/internal/tc_allocator_inline.h
)

target_sources(fast_allocators PRIVATE
  private/tc_allocator.cpp
  private/static_const_array.cpp
  ${ALLOCATORS_PUBLIC_HEADERS}
  ${ALLOCATORS_INTERNAL_HEADERS}
)

set_target_properties(fast_allocators PROPERTIES
  VERSION ${ALLOCATORS_VERSION}
  SOVERSION ${ALLOCATORS_MAJOR_VERSION}
)

# C++23, required, as for the whole library. The -std flag directly when this
# CMake does not know cxx_std_23 for the compiler (NVHPC always; an older CMake
# with a newer compiler, e.g. IntelLLVM 2026.0 with CMake 3.20). Kept inline
# (not a shared root-level helper) so this file still configures when built
# standalone via its own project().
if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR NOT "cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  target_compile_options(fast_allocators PRIVATE -std=gnu++23)
else()
  target_compile_features(fast_allocators PRIVATE cxx_std_23)
endif()

# --- fast_allocators/build_options.h -------------------------------------
# The options the headers read, generated into the build tree and installed:
# an installed consumer compiles them exactly as the library was. Defaults for
# a standalone build of this directory: Debug checks with a Debug build, slab
# reclaim on, no list walking.
if (NOT DEFINED CHUNK_DEBUG)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CHUNK_DEBUG ON)
  else()
    set(CHUNK_DEBUG OFF)
  endif()
endif()
if (NOT DEFINED CHUNK_RECLAIM_EMPTY_SLABS)
  set(CHUNK_RECLAIM_EMPTY_SLABS ON)
endif()
# The library runs threads: every backend but serial (top level: CHUNK_THREADS).
if (NOT DEFINED ALLOCATORS_THREADED)
  set(ALLOCATORS_THREADED ${ALLOCATORS_OMP})
endif()
if ("${ALLOCATORS_THREADS_BACKEND}" STREQUAL "std")
  set(ALLOCATORS_STD_THREADS ON)
else()
  set(ALLOCATORS_STD_THREADS OFF)
endif()
foreach(_fa_opt OPENMP:ALLOCATORS_OMP THREADED:ALLOCATORS_THREADED STD_THREADS:ALLOCATORS_STD_THREADS DEBUG:CHUNK_DEBUG CHECK_LISTS:FAST_ALLOC_CHECK_LISTS
                RECLAIM_EMPTY_SLABS:CHUNK_RECLAIM_EMPTY_SLABS)
  string(REPLACE ":" ";" _fa_pair ${_fa_opt})
  list(GET _fa_pair 0 _fa_name)
  list(GET _fa_pair 1 _fa_var)
  if (${_fa_var})
    set(FAST_ALLOC_OPT_${_fa_name} 1)
  else()
    set(FAST_ALLOC_OPT_${_fa_name} 0)
  endif()
endforeach()
set(FAST_ALLOC_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(public/fast_allocators/build_options.h.in
  ${FAST_ALLOC_GENERATED_INCLUDE_DIR}/fast_allocators/build_options.h @ONLY)

target_include_directories(fast_allocators
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
  $<BUILD_INTERFACE:${FAST_ALLOC_GENERATED_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

if (ALLOCATORS_OMP)
  target_link_libraries(fast_allocators PUBLIC OpenMP::OpenMP_CXX)
endif ()
# The std backend (CHUNK_THREADS=std): selected in the headers by this
# definition (internal/pmp_runtime.h), PUBLIC as OpenMP is; its workers are
# std::jthreads.
if (ALLOCATORS_STD_THREADS)
  target_compile_definitions(fast_allocators PUBLIC FAST_ALLOC_PMP_STD=1)
  find_package(Threads REQUIRED)
  target_link_libraries(fast_allocators PUBLIC Threads::Threads)
endif ()

include(GNUInstallDirs)

# chunk-config.cmake pins the fast_allocators version it was built against
# (find_dependency). chunk/ is configured first and records it as a GLOBAL
# property: fail here, at configure time, if the two drifted apart.
get_property(_chunk_wants_fa GLOBAL PROPERTY CHUNK_FAST_ALLOCATORS_VERSION)
if (_chunk_wants_fa AND NOT _chunk_wants_fa VERSION_EQUAL ALLOCATORS_VERSION)
  message(FATAL_ERROR "chunk/CMakeLists.txt pins fast_allocators ${_chunk_wants_fa} in chunk-config.cmake, "
                      "this is ${ALLOCATORS_VERSION}: update CHUNK_FAST_ALLOCATORS_VERSION there.")
endif ()

install(TARGETS fast_allocators
  EXPORT fast_allocators-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT fast_allocators-runtime
  NAMELINK_COMPONENT fast_allocators-devel
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT fast_allocators-devel)

install(DIRECTORY public/fast_allocators
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT fast_allocators-devel
  FILES_MATCHING PATTERN "*.h")
install(FILES ${FAST_ALLOC_GENERATED_INCLUDE_DIR}/fast_allocators/build_options.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fast_allocators
  COMPONENT fast_allocators-devel)

include(CMakePackageConfigHelpers)

configure_package_config_file(fast_allocators-config.cmake.in fast_allocators-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/fast_allocators/cmake)

write_basic_package_version_file(fast_allocators-config-version.cmake
  VERSION ${ALLOCATORS_VERSION}
  COMPATIBILITY SameMajorVersion)

install(EXPORT fast_allocators-targets
  NAMESPACE fast_allocators::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/fast_allocators/cmake)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_allocators-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/fast_allocators-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/fast_allocators/cmake
  COMPONENT fast_allocators-devel
)
