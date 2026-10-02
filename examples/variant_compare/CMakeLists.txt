# ============================================================================
# variant_compare: Chunk2 against Chunk2A on one workload (variant_compare.cpp).
#
# One source, one executable per variant at the default static size (no dyn:
# a re-chunk would sit in the timings): variant_compare_chunk2_<size>,
# variant_compare_chunk2a_<size>. A perf target -- no CHECK define.
#
#   OMP_NUM_THREADS=64 ./variant_compare_chunk2_256  --constants 1
#   OMP_NUM_THREADS=64 ./variant_compare_chunk2a_256 --constants 1
#   (then --constants 4)
#
# The smoke tests run a small N with --verify (values against a std::vector)
# at 1 and 8 threads, under the `validation` label: they keep the program
# building and correct, they measure nothing.
# ============================================================================

set(_vc_variant_names   chunk2   chunk2a)
set(_vc_variant_libbase chunk_v2 chunk_v2a)

list(LENGTH _vc_variant_names _vc_nvar)
math(EXPR _vc_nvar_last "${_vc_nvar} - 1")

foreach(_vi RANGE ${_vc_nvar_last})
  list(GET _vc_variant_names   ${_vi} _vname)
  list(GET _vc_variant_libbase ${_vi} _libbase)
  if (NOT TARGET ${_libbase}_${CHUNK_DEFAULT_SIZE})
    continue()
  endif()

  set(_tgt variant_compare_${_vname}_${CHUNK_DEFAULT_SIZE})
  add_executable(${_tgt} variant_compare.cpp)
  target_link_libraries(${_tgt} PRIVATE ${_libbase}_${CHUNK_DEFAULT_SIZE}) # CHUNK2/CHUNK2A, CHUNK_SIZE: PUBLIC
  if (OMP)
    target_link_libraries(${_tgt} PUBLIC OpenMP::OpenMP_CXX)
  endif ()
  chunk_require_cxx20(${_tgt} PRIVATE)

  foreach(_nthr 1 8)
    add_test(NAME ${_tgt}_smoke_${_nthr}thr COMMAND ${_tgt} --n 262144 --iters 4 --constants 4 --verify)
    set_property(TEST ${_tgt}_smoke_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
    set_property(TEST ${_tgt}_smoke_${_nthr}thr PROPERTY LABELS validation)
  endforeach()
endforeach()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
