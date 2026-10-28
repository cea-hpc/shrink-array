# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# examples/cst2_hash: the second-value registry of zero_any::Array on struct
# element types -- one identifier per distinct value, no degraded index
# (chunk/internal/value_hash.h), the padding and signed-zero cases included
# (the member-wise hash, by C++26 reflection).

add_executable(cst2_hash_validation cst2_hash_validation.cpp)
target_link_libraries(cst2_hash_validation PRIVATE chunk_zero_any)
chunk_require_cxx_std(cst2_hash_validation PRIVATE)
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME cst2_hash_validation COMMAND cst2_hash_validation)
  set_property(TEST cst2_hash_validation PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")
  set_property(TEST cst2_hash_validation PROPERTY LABELS validation)
  # shrink-build-stamp: the executable prints the build stamp and is built by
  # `check` (cmake/shrink_build_stamp.cmake); its test carries the validation
  # label, so `check` runs it. Not standalone: the stamp lives in the project.
  shrink_stamp_directory_targets()
endif()
