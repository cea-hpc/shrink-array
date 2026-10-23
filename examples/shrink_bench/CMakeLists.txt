# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# CMakeLists.txt for the shrink_bench bench component.
#
# NOTE: each executable links against its matching chunk variant target
# (chunk_zero / chunk_zero_one_value / chunk_zero_any / chunk_vector). The variant target sets
# CHUNK_VARIANT_ZERO* as PUBLIC, so:
#   - it's baked into libchunk_<variant> itself (fixing is_shrink=0 in
#     AllTypeChunk.cpp / manage_ompt.cpp)
#   - it propagates to the executable's own TUs

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
        bench_phase_mask.cpp
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
        bench_int.h
        bench_prelude.h
        bench_config.h
        bench_utils.h
        bench_io.h
        bench_args.h
        bench_array.h
        baseline_vector.h
        bench_phases.h
        bench_stl_checks.h
)


# ============================================================================
# Multi-size executable generation.
#
# For each shrink variant (zero/zero_one_value/zero_any) and each mode (check/perf) we
# build one executable per declination:
#   _64, _256, _1024  -> static libs chunk_<variant>_<size> (compile-time geometry)
#   _dyn              -> dynamic lib chunk_<variant>_dyn (runtime geometry), with the
#                        runtime size baked via -DDYN_CHUNK_SIZE=<SHRINK_BENCH_DYN_SIZE>.
# Naming: shrink_bench_<variant>_<mode>_<declination>, e.g.
#   shrink_bench_zero_one_value_256, shrink_bench_zero_any_dyn. The check / perf mode is
#   chosen at run time with --mode.
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
#
# Unity build (SHRINK_BENCH_UNITY, ON): the 16 sources of a bench executable
# compiled as one translation unit. Compiled apart, each re-instantiated the
# whole header-only library for itself: shrink_bench_zero_any_256 took 198 s
# of compilation in Debug and 312 s at -O2 -g, against 49 s and 90 s as one
# unit, for a peak of 2.9 / 3.4 GB instead of 2.3 / 2.8 (the checksums are the
# same). The sources needed no change. Two costs: the compiler may now inline
# across what were separate files, so the timings may differ from campaigns
# measured before (OFF to compare with them); and editing one bench file
# recompiles the whole unit (SHRINK_BENCH_UNITY_BATCH sources per unit, 0 =
# all, trades that back).
option(SHRINK_BENCH_UNITY "Compile each shrink_bench executable as one unity translation unit" ON)
set(SHRINK_BENCH_UNITY_BATCH "0" CACHE STRING "Bench sources per unity translation unit (0: all)")
function(sa2_setup_target tgt)
    if (OMP)
        target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    chunk_require_cxx_std(${tgt} PRIVATE)
    target_link_libraries(${tgt} PRIVATE BZip2::BZip2)
    if (SHRINK_BENCH_UNITY)
        set_target_properties(${tgt} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE ${SHRINK_BENCH_UNITY_BATCH})
        if (CHUNK_HEAVY_JOB_POOL)
            set_target_properties(${tgt} PROPERTIES JOB_POOL_COMPILE ${CHUNK_HEAVY_JOB_POOL})
        endif()
    endif()
endfunction()

# ============================================================================
# 64-bit index declination (int64).
#
# The bench spells every size and index `Int` (bench_int.h), and hands that same
# type to the library as `IntType` (bench_array.h). `Int` is `int` here and
# `long` under BENCH_INT64, so an int64 target is the SAME sources built with
# 64-bit indices -- which is what a run past INT_MAX elements needs: `--mode
# perf` already starts at 1e9, and `--multiply 2` is refused at 32 bits by
# check_size_limits().
#
# Naming: shrink_bench_<variant>_int64_<declination>, e.g.
# shrink_bench_zero_int64_1024, shrink_bench_zero_any_int64_dyn, and the
# baseline shrink_bench_vector_int64_perf.
#
# OFF by default: this directory is already the bulk of the build. Turn it on
# with -DSHRINK_BENCH_INT64=ON -- the compile scripts do so on the release
# OpenMP (perf) configurations only. Even then, only the static sizes listed
# in SHRINK_BENCH_INT64_SIZES (1024 by default: no 256, to keep the build time
# down) and dyn get a twin. The ordinary 32-bit targets are untouched either
# way, both live side by side.
#
# NOT registered with ctest: these are perf instruments first. --mode check
# works on them, against the data_tab* files of the 32-bit vector_check of the
# same configuration (element values, independent of Int);
# python/bench_and_test.py runs a few such checks.
# ============================================================================
option(SHRINK_BENCH_INT64 "Also build the 64-bit-index (int64) bench executables" OFF)
set(SHRINK_BENCH_INT64_SIZES "1024" CACHE STRING
    "Static chunk sizes that get an int64 bench twin (a subset of CHUNK_STATIC_SIZES); dyn always does")

# Helper: build the bench sources as int64 target <name>.
#   sa2_add_int64_target(<name> <lib> [extra compile definitions...])
function(sa2_add_int64_target name lib)
    if (NOT SHRINK_BENCH_INT64)
        return()
    endif()
    add_executable(${name} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${name} PRIVATE ${lib})
    target_compile_definitions(${name} PRIVATE BENCH_INT64=1 ${ARGN})
    sa2_setup_target(${name})
endfunction()

set(_sa2_variant_names   zero       zero_one_value      zero_any)
set(_sa2_variant_libbase chunk_zero    chunk_zero_one_value    chunk_zero_any)

list(LENGTH _sa2_variant_names _sa2_nvar)
math(EXPR _sa2_nvar_last "${_sa2_nvar} - 1")

# One target per (variant, declination) -- no longer one per mode as well.
# Naming: shrink_bench_<variant>_<declination>, e.g. shrink_bench_zero_one_value_256,
# shrink_bench_zero_any_dyn. Pick check or perf at run time with --mode.
foreach(_vi RANGE ${_sa2_nvar_last})
  list(GET _sa2_variant_names   ${_vi} _vname)
  list(GET _sa2_variant_libbase ${_vi} _libbase)
  if (NOT _vname IN_LIST CHUNK_VARIANTS)
    continue()
  endif()

  # Static-size declinations.
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    set(_tgt shrink_bench_${_vname}_${_sz})
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_sz})
    sa2_setup_target(${_tgt})
    if (_sz IN_LIST SHRINK_BENCH_INT64_SIZES)
      sa2_add_int64_target(shrink_bench_${_vname}_int64_${_sz} ${_libbase}_${_sz})
    endif()
  endforeach()

  # Dynamic declination.
  if (CHUNK_BUILD_DYN)
    set(_tgt shrink_bench_${_vname}_dyn)
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_dyn)
    target_compile_definitions(${_tgt} PRIVATE DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    sa2_setup_target(${_tgt})
    sa2_add_int64_target(shrink_bench_${_vname}_int64_dyn ${_libbase}_dyn DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
  endif()
endforeach()

# The TBB bench: the same sources, its regions run as TBB tasks (bench_par.h,
# BENCH_PAR_TBB) -- the arrays written from TBB's threads, as a TBB program
# would, the library's own passes on the std backend's std::jthreads at the serial
# points. Std builds with CHUNK_WITH_TBB (the TBB threads take ranks of the
# registry): shrink_bench_tbb_<variant>_<declination>, same --mode, same
# OMP_NUM_THREADS (CHUNK_NUM_THREADS) for the thread count.
if (CHUNK_WITH_TBB AND CHUNK_THREADS_BACKEND STREQUAL "std")
  foreach(_vi RANGE ${_sa2_nvar_last})
    list(GET _sa2_variant_names   ${_vi} _vname)
    list(GET _sa2_variant_libbase ${_vi} _libbase)
    if (NOT _vname IN_LIST CHUNK_VARIANTS)
      continue()
    endif()
    set(_decls ${CHUNK_STATIC_SIZES})
    if (CHUNK_BUILD_DYN)
      list(APPEND _decls dyn)
    endif()
    foreach(_sz IN LISTS _decls)
      set(_tgt shrink_bench_tbb_${_vname}_${_sz})
      add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
      target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_sz} TBB::tbb)
      target_compile_definitions(${_tgt} PRIVATE BENCH_PAR_TBB=1)
      if (_sz STREQUAL "dyn")
        target_compile_definitions(${_tgt} PRIVATE DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
      endif()
      sa2_setup_target(${_tgt})
    endforeach()
  endforeach()
endif()

# No chunk2bits targets: every shrink_bench_zero_one_value_* run carries tab_bool
# (zero_one_value::Array<bool>, one byte per bool) AND its twin tab_bool_bits
# (ArrayChunkBits, 8 per byte) on the same workload -- see BenchBoolBitsArray
# (bench_array.h). The zero / zero_any builds give the twin their own
# Array<bool>.

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

# The std::vector baseline gets an int64 target too: a 64-bit shrink run is
# only worth a number next to the baseline built the same way. Only the perf
# one -- the check target has no 64-bit meaning (no reference files).
sa2_add_int64_target(shrink_bench_vector_int64_perf chunk_vector)
if (SHRINK_BENCH_INT64)
  chunk_enable_fast_math(shrink_bench_vector_int64_perf)
endif()
# The compute kernel is `omp for simd` over cos/sin/atan/tan: without the
# vector libm the perf target gets from chunk_enable_fast_math it stays scalar
# by design (its checksums must match a shrink build's, see bench_prelude.h),
# and clang reports the failed `simd` as -Wpass-failed on this target alone.
# Expected, not actionable: silenced here. The perf target keeps the warning,
# where it does mean a loop that should vectorise no longer does.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(shrink_bench_vector_check PRIVATE -Wno-pass-failed)
endif()

# --- std_alloc (optional): mirror the declination scheme on the zero_any path ---
if (CHUNK_STD_ALLOC AND "zero_any" IN_LIST CHUNK_VARIANTS)
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    set(_tgt shrink_bench_std_alloc_${_sz})
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE chunk_zero_any_${_sz})
    sa2_setup_target(${_tgt})
  endforeach()
  if (CHUNK_BUILD_DYN)
    set(_tgt shrink_bench_std_alloc_dyn)
    add_executable(${_tgt} ${SHRINK_BENCH_SOURCES} ${SHRINK_BENCH_HEADERS})
    target_link_libraries(${_tgt} PRIVATE chunk_zero_any_dyn)
    target_compile_definitions(${_tgt} PRIVATE DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    sa2_setup_target(${_tgt})
  endif()
endif ()

# rechunk_validation and dyn_geometry_validation live in examples/validation:
# they depend on the library alone, not on the bench sources here.

if (CHUNK_BUILD_DYN)
  # --- Tests: register the check declinations of each variant ---
  #
  # The check runs compare against the reference files (data_tab*) that
  # shrink_bench_vector_check writes -- relative paths, in the working directory
  # every test here shares (this build dir). So:
  #  - vector_check is the SETUP of fixture shrink_bench_oracle and every check
  #    REQUIRES it: ctest runs it first, also under -R / -j, and marks the
  #    checks "Not Run" if it fails (without the fixture they would fail on a
  #    fresh build, the reference not written yet);
  #  - all of them hold RESOURCE_LOCK shrink_bench_oracle_files: they share the
  #    files, and a check may rewrite one (it decompresses a reference whose
  #    checksum misses), so they never overlap, even under -j;
  #  - shrink_bench_oracle_cleanup is the fixture's CLEANUP: ctest runs it once
  #    the last check is done and deletes the files (about 22 GB at the default
  #    sizes), which otherwise stayed in the build dir after every run. A
  #    manual `--mode check` run afterwards needs vector_check run again.
  # bench_and_test.py does not run these (no `validation` label): it keeps its
  # own oracles per configuration.
  add_test(NAME shrink_bench_vector_check COMMAND shrink_bench_vector_check)
  set_tests_properties(shrink_bench_vector_check PROPERTIES
                       ENVIRONMENT OMP_NUM_THREADS=8
                       FIXTURES_SETUP shrink_bench_oracle
                       RESOURCE_LOCK shrink_bench_oracle_files)
  add_test(NAME shrink_bench_oracle_cleanup
           COMMAND ${CMAKE_COMMAND} -DORACLE_DIR=${CMAKE_CURRENT_BINARY_DIR}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/remove_oracle_files.cmake)
  set_tests_properties(shrink_bench_oracle_cleanup PROPERTIES
                       FIXTURES_CLEANUP shrink_bench_oracle
                       RESOURCE_LOCK shrink_bench_oracle_files)

  foreach(_vname IN LISTS CHUNK_VARIANTS)
    foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
      # One binary per variant and size since the check/perf merge; the mode
      # is a run-time flag (the _check_ binaries these named no longer exist).
      add_test(NAME shrink_bench_${_vname}_check_${_sz}
               COMMAND shrink_bench_${_vname}_${_sz} --mode check)
      set_tests_properties(shrink_bench_${_vname}_check_${_sz} PROPERTIES
                           ENVIRONMENT OMP_NUM_THREADS=8
                           FIXTURES_REQUIRED shrink_bench_oracle
                           RESOURCE_LOCK shrink_bench_oracle_files)
      # The TBB bench against the same reference (std builds with
      # CHUNK_WITH_TBB): the arrays written from TBB's threads must hold what
      # the std::vector run wrote.
      if (TARGET shrink_bench_tbb_${_vname}_${_sz})
        add_test(NAME shrink_bench_tbb_${_vname}_check_${_sz}
                 COMMAND shrink_bench_tbb_${_vname}_${_sz} --mode check)
        set_tests_properties(shrink_bench_tbb_${_vname}_check_${_sz} PROPERTIES
                             ENVIRONMENT OMP_NUM_THREADS=8
                             FIXTURES_REQUIRED shrink_bench_oracle
                             RESOURCE_LOCK shrink_bench_oracle_files)
      endif()
    endforeach()
  endforeach()
endif()
