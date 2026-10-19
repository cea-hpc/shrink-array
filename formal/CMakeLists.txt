# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# formal/: reduced models of the lock-free handshakes (docs/formal_models.md),
# checked by GenMC. Two kinds of tests, both under the label `formal`:
#
#   formal_native_<model>  each model compiled as an ordinary C program and run
#                          a few hundred times: proves it is a real program, a
#                          weak oracle for the protocol (needs a C compiler);
#   formal_genmc           formal/run_genmc.sh: every configuration and every
#                          mutation under GenMC, verdicts compared with the
#                          expected ones. Registered only when `genmc` is found
#                          (GENMC_EXECUTABLE, or on the PATH).
#
# Nothing here is built into the library or installed.
# ============================================================================

include(CheckLanguage)
check_language(C)
if(CMAKE_C_COMPILER)
    enable_language(C)
    set(_formal_models writer_compressor compressor_flush)
    foreach(_m IN LISTS _formal_models)
        add_executable(formal_native_${_m} ${_m}.c)
        target_compile_definitions(formal_native_${_m} PRIVATE NATIVE SHAPE_SYMMETRIC)
        set_target_properties(formal_native_${_m} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
        find_package(Threads REQUIRED)
        target_link_libraries(formal_native_${_m} PRIVATE Threads::Threads)
        # 300 runs: the stress is the point, one run explores one interleaving.
        add_test(NAME formal_native_${_m}
                 COMMAND ${CMAKE_COMMAND} -E env SHRINK_FORMAL_RUNS=300
                         ${CMAKE_CURRENT_SOURCE_DIR}/run_native.sh $<TARGET_FILE:formal_native_${_m}>)
        set_property(TEST formal_native_${_m} PROPERTY LABELS formal)
    endforeach()
else()
    message(STATUS "formal/: no C compiler, the native model tests are not registered")
endif()

find_program(GENMC_EXECUTABLE genmc DOC "GenMC model checker (formal/run_genmc.sh)")
if(GENMC_EXECUTABLE)
    add_test(NAME formal_genmc
             COMMAND ${CMAKE_COMMAND} -E env GENMC=${GENMC_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/run_genmc.sh)
    set_property(TEST formal_genmc PROPERTY LABELS formal)
    message(STATUS "formal/: genmc found (${GENMC_EXECUTABLE}), test formal_genmc registered")
else()
    message(STATUS "formal/: genmc not found, test formal_genmc not registered (docs/formal_models.md says how to install it)")
endif()
