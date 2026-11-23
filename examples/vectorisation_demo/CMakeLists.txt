# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
add_executable(vectorisation_demo vectorisation_demo.cpp)

if (CHUNK_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(vectorisation_demo PUBLIC OpenMP::OpenMP_CXX)
endif ()
if (CMAKE_CXX_COMPILER MATCHES "icx" OR CMAKE_CXX_COMPILER MATCHES "icpx")
    target_compile_options(vectorisation_demo PRIVATE -O3 -mavx2)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_options(vectorisation_demo PRIVATE -O3 -mavx2 -qopt-report=5 -qopt-report-file=vectorization_report.txt)
endif ()
chunk_require_cxx_std(vectorisation_demo PRIVATE)

# vectorisation_demo doesn't use CHUNK_VARIANT_ZERO* → chunk_vector.
# The zero (zero::Chunk<T>) variant at the default static size, not the bare
# `chunk_vector` target: that one carries no variant define (the std::vector
# shape of the library) while this example instantiates zero::Array<double>
# itself -- the same mismatch multi_mat had (its allocators and lists never
# set up: SIGSEGV in create_value from the first parallel write).
target_link_libraries(vectorisation_demo PRIVATE chunk_zero_${CHUNK_DEFAULT_SIZE} shrink_examples_common)

add_test(NAME vectorisation_demo COMMAND vectorisation_demo)
set_property(TEST vectorisation_demo PROPERTY LABELS examples)
# check builds it (cmake/shrink_build_stamp.cmake), as it builds the harnesses: ctest -L examples
# finds it.
set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS vectorisation_demo)

set_property(TEST vectorisation_demo
        PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
