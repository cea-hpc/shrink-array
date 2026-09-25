cmake_minimum_required(VERSION 3.12)
project(fast_allocators LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose de type of build" FORCE)
endif ()

if (ALLOCATORS_OMP)
   find_package(OpenMP REQUIRED)
endif ()


add_library(fast_allocators)
add_library(fast_allocators::fast_allocators ALIAS fast_allocators)

set(ALLOCATORS_MAJOR_VERSION 2)  # inc. when incompatible change (reset to 0 others)
set(ALLOCATORS_MINOR_VERSION 1)  # inc. when compatible change (also reset patch vers.)
set(ALLOCATORS_PATCH_VERSION 8)  # inc. when just fixing bug

set(ALLOCATORS_VERSION
  ${ALLOCATORS_MAJOR_VERSION}.${ALLOCATORS_MINOR_VERSION}.${ALLOCATORS_PATCH_VERSION})

set(ALLOCATORS_PUBLIC_HEADERS
  public/fast_allocators/slab_arena.h
  public/fast_allocators/tc_allocator.h
)

# The implementation the two facades pull in: installed with them, not API.
set(ALLOCATORS_INTERNAL_HEADERS
  public/fast_allocators/internal/alloc_config.h
  public/fast_allocators/internal/int_alloc.h
  public/fast_allocators/internal/next_power_of_2.h
  public/fast_allocators/internal/pmp_alloc.h
  public/fast_allocators/internal/singleton.h
  public/fast_allocators/internal/static_const_array.h
  public/fast_allocators/internal/static_const_array_define.h
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

target_include_directories(fast_allocators
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
  $<INSTALL_INTERFACE:include>)

if (ALLOCATORS_OMP)
  target_link_libraries(fast_allocators PUBLIC OpenMP::OpenMP_CXX)
endif ()

include(GNUInstallDirs)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/fast_allocators)

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

include(CMakePackageConfigHelpers)

configure_package_config_file(fast_allocators-config.cmake.in fast_allocators-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/fast_allocators/cmake
  PATH_VARS ALLOCATORS_VERSION)
  
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
