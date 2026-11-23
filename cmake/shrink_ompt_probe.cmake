# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Does the OpenMP runtime load the OMPT tool the library puts in the executable?
# cmake/ompt_probe.cpp says why it may not (Ubuntu 24.04's libomp5-18).
#
# shrink_check_ompt_tool_loads(<link options>): <link options> are those the
# library's targets give an executable for its tool (chunk/CMakeLists.txt);
# the probe links with the same. Sets CHUNK_HAVE_OMPT OFF in the caller's scope
# when the runtime does not call the tool. The verdict is cached in
# CHUNK_OMPT_TOOL_LOADS: setting it skips the probe, -UCHUNK_OMPT_TOOL_LOADS
# runs it again. Not checked (the tool kept) when cross-compiling without an
# emulator, or when the probe does not build.

function(shrink_check_ompt_tool_loads)
  if(NOT DEFINED CHUNK_OMPT_TOOL_LOADS)
    if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
      message(STATUS "OMPT: cross-compiling, the runtime's loading of the tool is not checked")
      return()
    endif()

    # The probe's run must not see a user's OMP_TOOL=disabled (ENV is global:
    # put it back).
    set(had_omp_tool OFF)
    if(DEFINED ENV{OMP_TOOL})
      set(had_omp_tool ON)
      set(omp_tool "$ENV{OMP_TOOL}")
      unset(ENV{OMP_TOOL})
    endif()
    try_run(probe_exit probe_compiles
      "${CMAKE_CURRENT_BINARY_DIR}/ompt_probe"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ompt_probe.cpp"
      LINK_LIBRARIES OpenMP::OpenMP_CXX
      LINK_OPTIONS ${ARGN}
      COMPILE_OUTPUT_VARIABLE probe_log)
    if(had_omp_tool)
      set(ENV{OMP_TOOL} "${omp_tool}")
    endif()
    # try_run caches its two result variables: copied, then dropped (the verdict
    # below is the one kept).
    set(compiles "${probe_compiles}")
    set(exit_code "${probe_exit}")
    unset(probe_exit CACHE)
    unset(probe_compiles CACHE)

    if(NOT compiles)
      # Not the runtime's verdict: keep the tool, the build will tell.
      message(STATUS "OMPT: the probe did not build, the runtime's loading of the tool is not checked")
      message(DEBUG "${probe_log}")
      return()
    endif()
    set(loads OFF)
    if(exit_code STREQUAL "0")
      set(loads ON)
    endif()
    set(CHUNK_OMPT_TOOL_LOADS ${loads} CACHE BOOL
      "The OpenMP runtime loads an OMPT tool defined in the executable (configure-time probe)")
    mark_as_advanced(CHUNK_OMPT_TOOL_LOADS)
  endif()

  if(CHUNK_OMPT_TOOL_LOADS)
    message(STATUS "OMPT: the OpenMP runtime loads the tool of the executable")
  else()
    set(CHUNK_HAVE_OMPT OFF PARENT_SCOPE)
    message(WARNING
      "OMPT: the OpenMP runtime does not load an OMPT tool defined in the executable: the "
      "library is built without its tool, and a parallel region that touches an array goes "
      "through parallel_phase() (or open_parallel_phase() / finalize_parallel_phase()), as "
      "with GCC. Ubuntu 24.04's libomp5-18 is such a runtime; the libomp of apt.llvm.org is "
      "not. -UCHUNK_OMPT_TOOL_LOADS checks again.")
  endif()
endfunction()
