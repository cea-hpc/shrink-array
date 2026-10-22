# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/cst2_hash: the second-value registry of ArrayChunk2A on struct
# element types -- one identifier per distinct value, no degraded index
# (chunk/internal/value_hash.h). Every build; the padding and signed-zero cases
# are exact only with C++26 reflection (CHUNK_HAVE_REFLECTION).

add_executable(cst2_hash_validation cst2_hash_validation.cpp)
target_link_libraries(cst2_hash_validation PRIVATE chunk_v2a)
chunk_require_cxx_std(cst2_hash_validation PRIVATE)
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME cst2_hash_validation COMMAND cst2_hash_validation)
  set_property(TEST cst2_hash_validation PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")
  set_property(TEST cst2_hash_validation PROPERTY LABELS validation)
endif()
