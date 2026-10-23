# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# access_cost: the time per element of six access patterns on a std::vector
# and on the variant's array, in the same binary (access_cost.cpp).
#
# One executable per variant at the default static size:
# access_cost_zero_<size>, access_cost_zero_one_value_<size>, access_cost_zero_any_<size>.
# A perf target -- no CHECK define. python/access_cost.py runs them over
# rounds and thread counts and prints the medians and the ratios.
#
#   OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=64 ./access_cost_zero_any_256
#
# The smoke tests run a small N with --verify at 1 and 8 threads, under the
# `validation` label: they keep the program building and correct, they
# measure nothing.
# ============================================================================

set(_ac_variant_names   zero    zero_one_value   zero_any)
set(_ac_variant_libbase chunk_zero chunk_zero_one_value chunk_zero_any)

list(LENGTH _ac_variant_names _ac_nvar)
math(EXPR _ac_nvar_last "${_ac_nvar} - 1")

foreach(_vi RANGE ${_ac_nvar_last})
  list(GET _ac_variant_names   ${_vi} _vname)
  list(GET _ac_variant_libbase ${_vi} _libbase)
  if (NOT TARGET ${_libbase}_${CHUNK_DEFAULT_SIZE})
    continue()
  endif()

  set(_tgt access_cost_${_vname}_${CHUNK_DEFAULT_SIZE})
  add_executable(${_tgt} access_cost.cpp)
  target_link_libraries(${_tgt} PRIVATE ${_libbase}_${CHUNK_DEFAULT_SIZE}) # variant macro, CHUNK_SIZE: PUBLIC
  if (OMP)
    target_link_libraries(${_tgt} PUBLIC OpenMP::OpenMP_CXX)
  endif ()
  chunk_require_cxx20(${_tgt} PRIVATE)

  foreach(_nthr 1 8)
    add_test(NAME ${_tgt}_smoke_${_nthr}thr COMMAND ${_tgt} --n 262144 --reps 2 --verify)
    set_property(TEST ${_tgt}_smoke_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
    set_property(TEST ${_tgt}_smoke_${_nthr}thr PROPERTY LABELS validation)
  endforeach()
endforeach()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
