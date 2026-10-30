# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ctest cleanup of fixture shrink_bench_oracle (see CMakeLists.txt): delete
# the reference files shrink_bench_<backend>_vector_check wrote in the build directory
# -- data_tab*.bin and their checksums, about 22 GB at the default sizes.
# Left behind, they stayed on disk after every ctest run.
#   cmake -DORACLE_DIR=<dir> -P remove_oracle_files.cmake
if (NOT ORACLE_DIR)
  message(FATAL_ERROR "remove_oracle_files.cmake: ORACLE_DIR is not set")
endif()
file(GLOB _oracle_files "${ORACLE_DIR}/data_tab*")
list(LENGTH _oracle_files _n)
if (_n GREATER 0)
  file(REMOVE ${_oracle_files})
endif()
message(STATUS "shrink_bench oracle: ${_n} file(s) removed from ${ORACLE_DIR}")
