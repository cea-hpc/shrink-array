# CMakeLists.txt for the shrink_bench bench component.
#
# NOTE: each executable now links against its matching chunk variant target
# (chunk_v1 / chunk_v2 / chunk_v2a / chunk_vector). The variant target sets
# CHUNK / CHUNK2 / CHUNK2A as PUBLIC, so:
#   - it's baked into libchunk_v<x> itself (fixing is_shrink=0 in
#     AllTypeChunk.cpp / manage_ompt.cpp)
#   - it propagates to the executable's own TUs
# This makes the executable-side -DCHUNK2=1 / -DCHUNK2A=1 defines below
# technically redundant, but we keep them for now to make each executable
# self-explanatory in compile_commands.json.

# chunk_enable_fast_math() and ARMPL_ROOT live in examples/CMakeLists.txt now:
# multi-mat uses them too.
add_link_options(-rdynamic)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17"
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(
        shrink_bench.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize"
    )
    message(STATUS
        "shrink_bench.cpp: disabling the vectorizer on g++-16 x86_64 to dodge "
        "the -O3 AVX2 expand-pass ICE (keeps -O3; driver TU only)")
endif()


set(SHRINK_BENCH_SOURCES
        bench_utils.cpp
        bench_args.cpp
        bench_init.cpp
        bench_teardown.cpp
        bench_io.cpp
        bench_phase_fill.cpp
        bench_phase_compute.cpp
        bench_phase_struct.cpp
        bench_phase_stress.cpp
        bench_phase_ranges.cpp
        bench_phase_compress.cpp
        bench_checks.cpp
        bench_stats.cpp
        bench_stl_checks.cpp
        shrink_bench.cpp
)
set(SHRINK_BENCH_HEADERS
        bench_struct.h
        bench_prelude.h
        bench_config.h
        bench_utils.h
        bench_io.h
        bench_args.h
        bench_array.h
        bench_phases.h
        bench_stl_checks.h
        crash_backtrace.h
)


# ============================================================================
# Multi-size executable generation.
#
# For each shrink variant (chunk/chunk2/chunk2a) and each mode (check/perf) we
# build one executable per declination:
#   _64, _256, _1024  -> static libs chunk_v<x>_<size> (compile-time geometry)
#   _dyn              -> dynamic lib chunk_v<x>_dyn (runtime geometry), with the
#                        runtime size baked via -DDYN_CHUNK_SIZE=<SHRINK_BENCH_DYN_SIZE>.
# Naming: shrink_bench_<variant>_<mode>_<declination>, e.g.
#   shrink_bench_chunk2_256, shrink_bench_chunk2a_dyn. Le mode check/perf est
#   choisi a l'execution par --mode.
#
# vector has no chunk geometry: it keeps its two plain targets (no declination).
# The size list mirrors chunk/CMakeLists.txt's CHUNK_STATIC_SIZES; keep them in
# sync (or trim both during development to cut build time).
# ============================================================================

find_package(BZip2 REQUIRED)
if (OMP)
    find_package(OpenMP REQUIRED)
endif ()

# Runtime size baked into the _dyn executables (the value seeded into
# g_active_chunk_size). Override with -DSHRINK_BENCH_DYN_SIZE=<n>.
set(SHRINK_BENCH_DYN_SIZE "64" CACHE STRING "Runtime chunk size for the _dyn executables.")

# Must match chunk/CMakeLists.txt CHUNK_STATIC_SIZES.
if (NOT DEFINED CHUNK_STATIC_SIZES)
    set(CHUNK_STATIC_SIZES "256;1024")
endif()
if (NOT DEFINED CHUNK_BUILD_DYN)
    set(CHUNK_BUILD_DYN ON)
endif()

# Helper: finish wiring a freshly created bench target (omp, c++20, bzip2).
#
# No mode argument any more, and no CHECK / PERF define: check and perf are one
# binary, selected at run time by --mode (see bench_config.h). They differed only
# by a handful of integers, and paying for a second compilation of a 2400-line
# translation unit -- 21 s and 2.2 GB apiece -- to bake them in was the single
# largest item in this directory's build time.
function(sa2_setup_target tgt)
    if (OMP)
        target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    chunk_require_cxx20(${tgt} PRIVATE)
    target_link_libraries(${tgt} PRIVATE BZip2::BZip2)
endfunction()

set(_sa2_variant_names   chunk       chunk2      chunk2a)
set(_sa2_variant_libbase chunk_v1    chunk_v2    chunk_v2a)

list(LENGTH _sa2_variant_names _sa2_nvar)
math(EXPR _sa2_nvar_last "${_sa2_nvar} - 1")

# One target per (variant, declination) -- no longer one per mode as well.
# Naming: shrink_bench_<variant>_<declination>, e.g. shrink_bench_chunk2_256,
# shrink_bench_chunk2a_dyn. Pick check or perf at run time with --mode.
foreach(_vi RANGE ${_sa2_nvar_last})
  list(GET _sa2_variant_names   ${_vi} _vname)
  list(GET _sa2_variant_libbase ${_vi} _libbase)

  # Static-size declinations.
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    set(_tgt shrink_bench_${_vname}_${_sz})
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_sz})
    sa2_setup_target(${_tgt})
  endforeach()

  # Dynamic declination.
  if (CHUNK_BUILD_DYN)
    set(_tgt shrink_bench_${_vname}_dyn)
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_dyn)
    target_compile_definitions(${_tgt} PRIVATE DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    sa2_setup_target(${_tgt})
  endif()
endforeach()

# --- vector: size-agnostic, and the ONE place that keeps two targets ---
#
# Deliberately NOT merged with the rest. chunk_enable_fast_math is applied to the
# perf target only (-funsafe-math-optimizations, -fno-signed-zeros, and the ArmPL
# vector libm on aarch64 clang), so the std::vector baseline is not handicapped
# in a timing comparison. check and perf therefore differ here by COMPILER FLAGS,
# not just by the run-time constants --mode now selects: a single binary would
# either change the baseline's numerics in check mode, or lose fast-math in perf
# mode and inflate every speedup this bench has ever reported. Two targets is the
# cheap, honest answer -- it is 2 of 26, and it is what keeps the measurement
# comparable to past campaigns.
add_executable(shrink_bench_vector_check ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
add_executable(shrink_bench_vector_perf  ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
target_link_libraries(shrink_bench_vector_check PRIVATE chunk_vector)
target_link_libraries(shrink_bench_vector_perf  PRIVATE chunk_vector)
sa2_setup_target(shrink_bench_vector_check)
sa2_setup_target(shrink_bench_vector_perf)
chunk_enable_fast_math(shrink_bench_vector_perf)
# The compute kernel is `omp for simd` over cos/sin/atan/tan: without the
# vector libm the perf target gets from chunk_enable_fast_math it stays scalar
# by design (its checksums must match a shrink build's, see bench_prelude.h),
# and clang reports the failed `simd` as -Wpass-failed on this target alone.
# Expected, not actionable: silenced here. The perf target keeps the warning,
# where it does mean a loop that should vectorise no longer does.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(shrink_bench_vector_check PRIVATE -Wno-pass-failed)
endif()

# --- std_alloc (optional): mirror the declination scheme on the v2a path ---
if (CHUNK_STD_ALLOC)
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    set(_tgt shrink_bench_std_alloc_${_sz})
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE chunk_v2a_${_sz})
    target_compile_definitions(${_tgt} PRIVATE STD_ALLOC=1)
    sa2_setup_target(${_tgt})
  endforeach()
  if (CHUNK_BUILD_DYN)
    set(_tgt shrink_bench_std_alloc_dyn)
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE chunk_v2a_dyn)
    target_compile_definitions(${_tgt} PRIVATE STD_ALLOC=1 DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    sa2_setup_target(${_tgt})
  endif()
endif ()

# rechunk_validation and dyn_geometry_validation moved to examples/validation:
# they depend on the library alone, not on the bench sources here.

if (CHUNK_BUILD_DYN)
  # --- Tests: register the check declinations of each variant ---
  foreach(_vname chunk chunk2 chunk2a)
    foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
      add_test(NAME shrink_bench_${_vname}_check_${_sz}
               COMMAND shrink_bench_${_vname}_check_${_sz})
      set_property(TEST shrink_bench_${_vname}_check_${_sz}
                   PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    endforeach()
  endforeach()

  add_test(NAME shrink_bench_vector_check COMMAND shrink_bench_vector_check)
  set_property(TEST shrink_bench_vector_check PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
endif()
