# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
add_executable(simple_array_vector simple_array.cpp)
add_executable(simple_array_chunk  simple_array.cpp)

if (OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(simple_array_vector PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(simple_array_chunk  PUBLIC OpenMP::OpenMP_CXX)
endif()

chunk_require_cxx20(simple_array_vector PRIVATE)
chunk_require_cxx20(simple_array_chunk PRIVATE)

# Link the matching chunk variant so is_shrink_enabled sees the same value
# on both sides of the library boundary (see chunk/CMakeLists.txt header).
target_link_libraries(simple_array_vector PRIVATE chunk_vector)
target_link_libraries(simple_array_chunk  PRIVATE chunk_zero)

# CHUNK_VARIANT_ZERO=1 now comes from chunk_zero (PUBLIC); the line below is kept
# as a harmless redefinition to keep this executable self-explanatory in
# compile_commands.json.
target_compile_definitions(simple_array_chunk PRIVATE CHUNK_VARIANT_ZERO=1)

add_test(NAME simple_array_chunk COMMAND simple_array_chunk)
set_property(TEST simple_array_chunk PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
set_property(TEST simple_array_chunk PROPERTY LABELS examples)
# check builds it (cmake/shrink_build_stamp.cmake), as it builds the harnesses: ctest -L examples
# finds it.
set_property(GLOBAL APPEND PROPERTY SHRINK_CHECK_TARGETS simple_array_chunk)

# Not labelled `examples`: one std::vector of 10^9 doubles per array, more
# memory than a CI runner has (simple_array_chunk is the same program compressed).
add_test(NAME simple_array_vector COMMAND simple_array_vector)
set_property(TEST simple_array_vector PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
