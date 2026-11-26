# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# host_twin_bench: kokkos_shrink_bench's workload on the host arrays, the CPU
# against the GPU (host_twin_bench.cpp).
#
# One source, one executable per variant at chunk size 1024, the GPU's:
# host_twin_bench_zero_1024, host_twin_bench_zero_or_value_1024,
# host_twin_bench_zero_any_1024 (the variants of CHUNK_VARIANTS, if 1024 is in
# CHUNK_STATIC_SIZES). OpenMP only. A perf target: no test but a smoke run.
#
#   OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
#     ./host_twin_bench_zero_or_value_1024 30 2000 107374 10
#   against: kokkos_shrink_bench 30 2000 107374 10 zero_or_value
# ============================================================================
if (NOT CHUNK_OPENMP)
  return()
endif()
find_package(OpenMP REQUIRED)

foreach (_v IN ITEMS zero zero_or_value zero_any)
  if (NOT TARGET chunk_${_v}_1024)
    continue()
  endif()
  set(_tgt host_twin_bench_${_v}_1024)
  add_executable(${_tgt} host_twin_bench.cpp)
  target_link_libraries(${_tgt} PRIVATE chunk_${_v}_1024) # CHUNK_VARIANT_*, CHUNK_SIZE: PUBLIC
  target_link_libraries(${_tgt} PUBLIC OpenMP::OpenMP_CXX)
  chunk_require_cxx20(${_tgt} PRIVATE)
  add_test(NAME ${_tgt}_smoke COMMAND ${_tgt} 20 20 2000 5)
  set_property(TEST ${_tgt}_smoke PROPERTY ENVIRONMENT OMP_NUM_THREADS=4)
  set_property(TEST ${_tgt}_smoke PROPERTY LABELS examples)
  set_property(TEST ${_tgt}_smoke PROPERTY PASS_REGULAR_EXPRESSION "checksums equal to")
endforeach()

shrink_stamp_directory_targets()
