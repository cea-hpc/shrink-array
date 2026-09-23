# ============================================================================
# Validation harnesses: exit 0 / 1 programs registered with ctest, one
# executable per chunk variant, dyn-geometry builds only.
#
#   rechunk_validation      content invariance across rechunk_to(), and the
#                           location of the first mismatch when it fails
#   dyn_geometry_validation an array reads back what was written at EVERY
#                           registered runtime chunk size -- the pairing between
#                           set_active_chunk_size (array side) and
#                           ParamChunk::register_chunk_size (allocator side),
#                           which fails silently when only one of the two is
#                           done. No rechunk_to on purpose: when this one fails
#                           the cause is in the geometry/allocation pairing and
#                           nowhere else.
#   compress_roundtrip      every element reads back after a compression pass
#                           and a partial re-materialisation, and the library
#                           tears down cleanly -- at 1 thread and at 8, since
#                           the answer must not depend on the team.
#   pool_accounting         the manager's bookkeeping follows the arrays alive:
#                           chunk count up and down, buffers parked and taken
#                           back in exact number, nothing left at an old
#                           geometry after a re-chunk, threshold back at the
#                           floor when everything is gone.
#   open_types_validation   arrays of element types the library never listed
#                           (a struct of doubles, a struct with padding, an
#                           enum class, a std::array, a 4-byte pair) round-trip
#                           through writes, a compression pass and a partial
#                           rewrite like an array of double -- the element
#                           type contract: trivially copyable, ==, T{}.
#   deferred_path_validation a write parked in a rank queue (the deferred
#                           path: enqueue, replay, P3 drain, read gate, WAW
#                           guard) lands in program order with every other
#                           write and read of the element. The bench and the
#                           fuzz replays no longer reach that path (a
#                           compressor must hold the chunk at the instant of
#                           the write), so this executable alone is built
#                           with CHUNK_FORCE_DEFER_EVERY=7: one write in 7
#                           per thread is deferred with no compressor in
#                           sight, and the result must still equal a
#                           std::vector's, bit for bit.
#   chunk_array_new_validation  the variant's operator new[] hands back an
#                           array whose every chunk is at the first constant,
#                           at every size and from inside a parallel region as
#                           well as outside. Nothing else covers it: the
#                           library never array-news a chunk.
#   mark_path_validation    a chunk written back to a constant everywhere is
#                           compressed by the incremental path alone -- the
#                           write marks it, the between-region pass compresses
#                           it -- with no compress() call: what a perf build
#                           keeps. Plain-then-constant, second-constant
#                           round trips, and an array with no declared second
#                           value, per variant.
#
# All sources are variant-agnostic (the array type follows the CHUNK / CHUNK2
# / CHUNK2A macro) and depend on the library alone -- none of the shrink_bench
# bench files -- which is why they live here rather than next to the bench.
# Every test carries the `validation` label: `ctest -L validation` runs the
# lot, and that is what python/bench_and_test.py's ctest phase invokes.
# ============================================================================

# Runtime chunk size baked into the executables (-DDYN_CHUNK_SIZE). Shared
# with shrink_bench (a cache entry there); this is the same default.
set(SHRINK_BENCH_DYN_SIZE "64" CACHE STRING "Runtime chunk size for the _dyn executables.")

if (NOT DEFINED CHUNK_BUILD_DYN)
    set(CHUNK_BUILD_DYN ON)
endif()

# Finish wiring a harness target: OpenMP when on, C++20, check build.
function(validation_setup_target tgt)
    if (OMP)
        target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    chunk_require_cxx20(${tgt} PRIVATE)
    target_compile_definitions(${tgt} PRIVATE CHECK)
endfunction()

if (CHUNK_BUILD_DYN)

  set(_val_harnesses      rechunk_validation dyn_geometry_validation)
  set(_val_variant_names   chunk       chunk2      chunk2a)
  set(_val_variant_libbase chunk_v1    chunk_v2    chunk_v2a)
  set(_val_variant_macro   CHUNK       CHUNK2      CHUNK2A)

  list(LENGTH _val_variant_names _val_nvar)
  math(EXPR _val_nvar_last "${_val_nvar} - 1")

  foreach(_harness IN LISTS _val_harnesses)
    foreach(_vi RANGE ${_val_nvar_last})
      list(GET _val_variant_names   ${_vi} _vname)
      list(GET _val_variant_libbase ${_vi} _libbase)
      list(GET _val_variant_macro   ${_vi} _vmacro)

      set(_tgt ${_harness}_${_vname}_dyn)
      add_executable(${_tgt} ${_harness}.cpp)
      target_link_libraries(${_tgt} PRIVATE ${_libbase}_dyn)
      target_compile_definitions(${_tgt} PRIVATE
          ${_vmacro} CHUNK_DYNAMIC DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE}
          # rechunk_validation only: trace the reuse pools around a re-chunk.
          $<$<AND:$<BOOL:${RECHUNK_TRACE_POOL}>,$<STREQUAL:${_harness},rechunk_validation>>:RECHUNK_TRACE_POOL>)
      validation_setup_target(${_tgt})

      add_test(NAME ${_tgt} COMMAND ${_tgt})
      set_property(TEST ${_tgt} PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
      set_property(TEST ${_tgt} PROPERTY LABELS validation)
    endforeach()
  endforeach()

endif()

# --- compress_roundtrip and pool_accounting: one executable per harness,
# variant and geometry (the default static size and, when built, dyn). Two
# tests per executable: the properties must not depend on the team, so they
# are asserted at 1 thread and at 8.
set(_rt_harnesses       compress_roundtrip pool_accounting deferred_path_validation open_types_validation
                        mark_path_validation fill_range_mt_validation index64_validation)
set(_rt_variant_names   chunk       chunk2      chunk2a)
set(_rt_variant_libbase chunk_v1    chunk_v2    chunk_v2a)
set(_rt_variant_macro   CHUNK       CHUNK2      CHUNK2A)

set(_rt_geometries ${CHUNK_DEFAULT_SIZE})
if (CHUNK_BUILD_DYN)
  list(APPEND _rt_geometries dyn)
endif()

list(LENGTH _rt_variant_names _rt_nvar)
math(EXPR _rt_nvar_last "${_rt_nvar} - 1")

foreach(_harness IN LISTS _rt_harnesses)
foreach(_geom IN LISTS _rt_geometries)
  foreach(_vi RANGE ${_rt_nvar_last})
    list(GET _rt_variant_names   ${_vi} _vname)
    list(GET _rt_variant_libbase ${_vi} _libbase)
    list(GET _rt_variant_macro   ${_vi} _vmacro)

    set(_tgt ${_harness}_${_vname}_${_geom})
    add_executable(${_tgt} ${_harness}.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_geom})
    if (_geom STREQUAL "dyn")
      target_compile_definitions(${_tgt} PRIVATE
          ${_vmacro} CHUNK_DYNAMIC DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    else()
      target_compile_definitions(${_tgt} PRIVATE ${_vmacro}) # CHUNK_SIZE comes from the lib (PUBLIC)
    endif()
    validation_setup_target(${_tgt})
    # The forced-deferral knob, on this harness alone (the write engine is
    # header-only, instantiated in the executable: the library archive holds
    # none of it, so a target-scoped value is consistent). A global value set
    # on the cache (a whole-tree forced build) is not overridden.
    if (_harness STREQUAL "deferred_path_validation" AND "${CHUNK_FORCE_DEFER_EVERY}" STREQUAL "0")
      target_compile_definitions(${_tgt} PRIVATE CHUNK_FORCE_DEFER_EVERY=7)
    endif()

    foreach(_nthr 1 8)
      add_test(NAME ${_tgt}_${_nthr}thr COMMAND ${_tgt})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY LABELS validation)
    endforeach()
  endforeach()
endforeach()
endforeach()

# --- chunk_array_new_validation: the three variants, one executable per
# variant and geometry, at 1 and 8 threads like the others. OpenMP builds
# only: the operators are the *OmpPolicy's, and their serial specialisations
# declare nothing (there is no array-new to test there).
if (OMP)
foreach(_geom IN LISTS _rt_geometries)
  foreach(_vi RANGE ${_rt_nvar_last})
    list(GET _rt_variant_names   ${_vi} _vname)
    list(GET _rt_variant_libbase ${_vi} _libbase)
    list(GET _rt_variant_macro   ${_vi} _vmacro)

    set(_tgt chunk_array_new_validation_${_vname}_${_geom})
    add_executable(${_tgt} chunk_array_new_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_geom})
    if (_geom STREQUAL "dyn")
      target_compile_definitions(${_tgt} PRIVATE ${_vmacro} CHUNK_DYNAMIC DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    else()
      target_compile_definitions(${_tgt} PRIVATE ${_vmacro}) # CHUNK_SIZE comes from the lib (PUBLIC)
    endif()
    validation_setup_target(${_tgt})

    foreach(_nthr 1 8)
      add_test(NAME ${_tgt}_${_nthr}thr COMMAND ${_tgt})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY LABELS validation)
    endforeach()
  endforeach()
endforeach()
endif()
