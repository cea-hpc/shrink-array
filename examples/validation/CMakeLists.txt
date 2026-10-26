# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Validation harnesses: exit 0 / 1 programs registered with ctest. Most of
# them share one executable per variant and geometry, validation_<variant>_
# <geometry> (see "The grouped harnesses" below); the test names are per
# harness all the same.
#
#   rechunk_validation      content invariance across rechunk_to(), and the
#                           location of the first mismatch when it fails;
#                           each case in a serial and a parallel pass, one
#                           ctest per case (--case) so `ctest -j` runs them
#                           side by side
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
#                           path: enqueue, replay, backlog drain, read gate, WAW
#                           guard) lands in program order with every other
#                           write and read of the element. The bench and the
#                           fuzz replays no longer reach that path (a
#                           compressor must hold the chunk at the instant of
#                           the write), so this harness switches on the
#                           forced-deferral test hook (test_hooks.h): one
#                           write in 7 per thread is deferred with no
#                           compressor in sight, and the result must still
#                           equal a std::vector's, bit for bit.
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
#   stale_proxy_validation  check builds: a write proxy used after its array
#                           rebuilt its storage (resize, reserve, clear,
#                           assign, swap, assignments, shrink_to_fit, a
#                           re-chunk) stops on an invariant failure; each case
#                           in a child, which re-runs this executable.
#   thread_growth_validation arrays created under a 2-thread team keep
#                           working when the team grows to the thread
#                           capacity afterwards (the per-rank structures are
#                           sized for it), and a team past the capacity stops
#                           with an invariant failure (the executable re-runs
#                           itself with --beyond to check it).
#   phase_protocol_validation the guard of the phase protocol
#                           (chunk/internal/phase_guard.h), team size
#                           changing between regions, deferrals forced: a
#                           forgotten finalize stops at the next begin (every
#                           build), a forgotten begin at the first deferred
#                           write and release_chunk_pools() in a region (check
#                           builds); the correct uses -- by hand,
#                           parallel_phase(), the split pair, a nested begin
#                           -- run and read back right. Each case in a child.
#   region_storage_validation an array created, grown (with and without a
#                           value), shrunk and regrown by ONE thread inside a
#                           parallel region (`omp single`, `omp masked`)
#                           reads back what std::vector would, and counts its
#                           written chunks right. The storage paths once fell
#                           back on an orphaned `omp for` there, and hung.
#   bulk_write_team_validation the whole-array writes that take a team of
#                           their own outside a region (copy construction and
#                           assignment, assign, resize(n, val), a shrinking
#                           resize's release) and zero_any's operator== equal a
#                           std::vector's, with the phase bracketed; inside a
#                           region they stay on the calling thread.
#   compound_ops_validation `*=`, `/=`, `%=` read back what std::vector would
#                           (results landing on zero included), `double /= 0.0`
#                           gives inf, set_second_value(0) declares nothing;
#                           check builds: an integer `/= 0` or `%= 0` stops
#                           the program (the executable re-runs itself with
#                           --child to check it).
#   nested_region_validation a thread that opens a nested (serialised) region
#                           keeps its own per-rank structures afterwards; two
#                           ACTIVE levels stop with an invariant failure
#                           (re-run with --nested-active).
#   fill_range_mt_zero_and_zero_one_value_validation  the range ops in a TWO-variant build
#                           (chunk_zero_and_zero_one_value): a zero::Array and a zero_one_value::Array side
#                           by side, fill_range_mt of each compressible value.
#                           The ops must dispatch on the array's type, not on
#                           the build flags -- otherwise a zero_one_value::Array
#                           filled with its second value stays materialised.
#                           The one harness that is not variant-agnostic.
#   cst2_registry_validation  zero_any's second-value registry is freed by the
#                           last array of every generation, not only the
#                           first (its reinstall took a phantom reference),
#                           and overlapping arrays free nothing early.
#                           zero_any only, default static size.
#   cst2_reclaim_validation  zero_any::Array::reclaim_second_values() frees
#                           exactly the second values nothing refers to,
#                           every element still reads back, and the
#                           identifiers are reused (a run past the
#                           identifier limit reaches its end), and across a
#                           re-chunk (dyn). zero_any only, default static
#                           size and dyn.
#   pattern_validation      pattern::Array against std::vector (zero, uniform,
#                           shifted-motif and random chunks; copy, move,
#                           resize, assign), and the second-value registry it
#                           shares with zero_any::Array (a reclaim on either
#                           keeps what both name). Its own executable
#                           (pattern_validation_zero_any_<size>), default static
#                           size (pattern::Array is static only).
#   std_thread_validation   std builds: arrays written from the application's
#                           own std::threads -- new ones every phase, then
#                           long-lived ones on a barrier -- each with a rank
#                           of the registry, begin_parallel_phase() on every
#                           writer and finalize_parallel_phase() once after
#                           the join; the array equals a std::vector after
#                           every phase, the ranks of a phase are distinct,
#                           in_parallel() is true during the phase only.
#   tbb_validation          CHUNK_WITH_TBB, std and serial builds: arrays
#                           written from a TBB program through
#                           shrink_array::tbb::parallel_for -- each
#                           partitioner, the index form, compression and
#                           deferred writes in the phase, a nested
#                           parallel_for, a body that throws, a narrower
#                           task_arena; the array equals a std::vector after
#                           every phase, every thread keeps one rank, distinct
#                           from the others', in_parallel() is true in the
#                           pieces only.
#   iterator_validation     the read-only const_iterator is random access:
#                           jumps (it + n, it[n], it2 - it1, orderings) on
#                           and around chunk boundaries, a ++/-- right after
#                           each, std::lower_bound / upper_bound /
#                           equal_range / binary_search and their ranges
#                           forms against a std::vector -- through folded
#                           chunks as well as materialised ones, and from
#                           every thread at once. At compile time: the
#                           concepts hold, the classic category is
#                           random_access (what std::lower_bound dispatches
#                           on), and std::sortable is false (no in-place
#                           sort, by construction).
#
# All sources are variant-agnostic (the array type follows the CHUNK_VARIANT_ZERO*
# macro) and depend on the library alone -- none of the shrink_bench
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

# --- The grouped harnesses: one executable per variant and geometry,
# validation_<variant>_<geometry>, a unity build. Compiled one by one, each
# harness re-instantiated the whole library for itself: 14 s apiece in a Debug
# build, against 0.4 s of link. Together they share the instantiations -- a
# pilot of six of them compiled in 13.8 s instead of 56.8 (Debug) and 28.4
# instead of 119.4 (-O2), for the same peak memory. Each harness is a
# namespace, validation::<harness>, with a run() the shared driver calls
# (validation_harness.h, validation_driver.cpp):
#
#     validation_zero_any_256 compress_roundtrip [args...]
#
# The ctest names are the ones of the one-executable-per-harness layout
# (compress_roundtrip_zero_any_256_8thr, rechunk_validation_zero_dyn_val2-grow,
# ...), so `ctest -R` and python/bench_and_test.py see no difference.
# CHUNK_VALIDATION_UNITY=OFF goes back to one executable per harness, named as
# before -- the one to debug a harness alone. CHUNK_VALIDATION_UNITY_BATCH caps
# the harnesses per unity TU (0: all of a group in one).
#
# deferred_path_validation and read_reclaim_validation are grouped too: their
# test hooks (forced deferrals, read and pass delays: chunk/internal/
# test_hooks.h) are compiled into the whole tree wherever the harnesses are
# built (CHUNK_TEST_HOOKS, top-level CMakeLists), off until a harness switches
# them on in its own process -- one definition of the header-only engine for
# every harness. Not grouped: those built against another library or none
# (fill_range_mt_v1v2, bit_words, blk_allocator, slab_geometry). RECHUNK_TRACE_POOL, a trace, applies to the whole group of
# rechunk_validation.
option(CHUNK_VALIDATION_UNITY "One unity-built executable per variant and geometry for the validation harnesses" ON)
set(CHUNK_VALIDATION_UNITY_BATCH "0" CACHE STRING "Harnesses per unity translation unit (0: all of a group)")

set(_rt_variant_names   zero       zero_one_value      zero_any)
set(_rt_variant_libbase chunk_zero    chunk_zero_one_value    chunk_zero_any)
set(_rt_variant_macro   CHUNK_VARIANT_ZERO CHUNK_VARIANT_ZERO_ONE_VALUE CHUNK_VARIANT_ZERO_ANY)

# Every static size built (CHUNK_STATIC_SIZES: 256;1024 by default), then dyn.
# A static size other than CHUNK_DEFAULT_SIZE gets the round-trip harnesses
# only (_rt_harnesses): what changes with the size is the chunk geometry they
# exercise, not the special cases below, which stay at the default size.
set(_rt_geometries ${CHUNK_DEFAULT_SIZE})
foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
  if (NOT _sz IN_LIST _rt_geometries)
    list(APPEND _rt_geometries ${_sz})
  endif()
endforeach()
if (CHUNK_BUILD_DYN)
  list(APPEND _rt_geometries dyn)
endif()

list(LENGTH _rt_variant_names _rt_nvar)
math(EXPR _rt_nvar_last "${_rt_nvar} - 1")

# The round-trip harnesses, at 1 and 8 threads. Only those of
# _rt_dyn_harnesses are built at dyn as well: the ones with code of their own
# there (rechunk_to, set_chunk_size, a chunk_dynamic branch). The others test
# nothing geometry-specific and are built at the default static size only --
# it exists in every configuration, CHUNK_BUILD_DYN or not.
set(_rt_harnesses       compress_roundtrip pool_accounting open_types_validation
                        mark_path_validation fill_range_mt_validation index64_validation
                        swap_move_validation range_counters_validation thread_growth_validation
                        nested_region_validation concurrent_teams_validation iterator_validation
                        stale_proxy_validation
                        region_storage_validation compound_ops_validation bulk_write_team_validation)
set(_rt_dyn_harnesses   pool_accounting open_types_validation index64_validation stale_proxy_validation)

# Compile definitions of a variant at a geometry.
function(validation_variant_defs tgt vmacro geom)
  if (geom STREQUAL "dyn")
    target_compile_definitions(${tgt} PRIVATE ${vmacro} CHUNK_DYNAMIC DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
  else()
    target_compile_definitions(${tgt} PRIVATE ${vmacro}) # CHUNK_SIZE comes from the lib (PUBLIC)
  endif()
endfunction()

# An executable of the shared driver and the harnesses of ARGN.
function(validation_harness_executable tgt libbase vmacro geom)
  add_executable(${tgt} validation_driver.cpp)
  set(_list "")
  foreach(_h IN LISTS ARGN)
    target_sources(${tgt} PRIVATE ${_h}.cpp)
    string(APPEND _list "VALIDATION_HARNESS(${_h})\n")
  endforeach()
  # The driver's table: rewritten only when it changes, so a re-configure does
  # not rebuild the driver.
  set(_hdr ${CMAKE_CURRENT_BINARY_DIR}/harness_lists/${tgt}.h)
  file(WRITE ${_hdr}.in "${_list}")
  configure_file(${_hdr}.in ${_hdr} COPYONLY)
  target_compile_definitions(${tgt} PRIVATE VALIDATION_HARNESS_LIST="${_hdr}")
  target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${tgt} PRIVATE ${libbase}_${geom})
  validation_variant_defs(${tgt} ${vmacro} ${geom})
  # tbb_validation: a TBB program (CHUNK_WITH_TBB); the library itself does not link TBB.
  if ("tbb_validation" IN_LIST ARGN)
    target_link_libraries(${tgt} PRIVATE TBB::tbb)
  endif()
  if (RECHUNK_TRACE_POOL AND "rechunk_validation" IN_LIST ARGN)
    target_compile_definitions(${tgt} PRIVATE RECHUNK_TRACE_POOL)
  endif()
  validation_setup_target(${tgt})
  if (CHUNK_VALIDATION_UNITY)
    set_target_properties(${tgt} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE ${CHUNK_VALIDATION_UNITY_BATCH})
    if (CHUNK_HEAVY_JOB_POOL)
      set_target_properties(${tgt} PROPERTIES JOB_POOL_COMPILE ${CHUNK_HEAVY_JOB_POOL})
    endif()
  endif()
endfunction()
set_source_files_properties(validation_driver.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# test name, executable, harness, thread counts ("" = one test, OMP_NUM_THREADS=8), args...
function(validation_add_test name exe harness nthreads)
  if (nthreads STREQUAL "")
    add_test(NAME ${name} COMMAND ${exe} ${harness} ${ARGN})
    set_property(TEST ${name} PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST ${name} PROPERTY LABELS validation)
  else()
    foreach(_nthr IN LISTS nthreads)
      add_test(NAME ${name}_${_nthr}thr COMMAND ${exe} ${harness} ${ARGN})
      set(_env OMP_NUM_THREADS=${_nthr})
      # nested_region_validation opens its outer team with num_threads(4),
      # whatever OMP_NUM_THREADS says. At 1 thread the per-thread capacity is
      # max(omp_get_max_threads(), omp_get_num_procs()): 1 on a node that grants
      # the job one core, and rank 3 is then (rightly) refused. The harness
      # declares the team it uses.
      if (harness STREQUAL "nested_region_validation")
        list(APPEND _env CHUNK_THREAD_CAPACITY=4)
      endif()
      set_property(TEST ${name}_${_nthr}thr PROPERTY ENVIRONMENT ${_env})
      set_property(TEST ${name}_${_nthr}thr PROPERTY LABELS validation)
    endforeach()
  endif()
endfunction()

foreach(_geom IN LISTS _rt_geometries)
  foreach(_vi RANGE ${_rt_nvar_last})
    list(GET _rt_variant_names   ${_vi} _vname)
    list(GET _rt_variant_libbase ${_vi} _libbase)
    list(GET _rt_variant_macro   ${_vi} _vmacro)
    if (NOT _vname IN_LIST CHUNK_VARIANTS)
      continue()
    endif()

    # The group: the round-trip harnesses, then the special cases below.
    set(_group "")
    foreach(_h IN LISTS _rt_harnesses)
      if (NOT _geom STREQUAL "dyn" OR _h IN_LIST _rt_dyn_harnesses)
        list(APPEND _group ${_h})
      endif()
    endforeach()
    set(_rt_group ${_group})
    # chunk_array_new_validation: the variant's operator new[] hands back an
    # array whose every chunk is at the first constant. OpenMP builds only: the
    # operators are the threaded *ThreadPolicy's (the serial specialisations
    # declare none), and the harness calls them from an `omp single`.
    if (OMP)
      list(APPEND _group chunk_array_new_validation)
    endif()
    # bits_container_validation: ArrayChunkBits (zero_one_value::Array<Bit>) against
    # std::vector<bool>. zero_one_value libraries only, every geometry.
    if (_vname STREQUAL "zero_one_value")
      list(APPEND _group bits_container_validation)
    endif()
    # cst2_registry_validation: zero_any's second-value registry lifecycle.
    # zero_any only, at the default static size.
    if (_vname STREQUAL "zero_any" AND NOT _geom STREQUAL "dyn")
      list(APPEND _group cst2_registry_validation)
    endif()
    # cst2_reclaim_validation: reclaim_second_values() frees what nothing
    # refers to and the identifiers are reused. zero_any only, at the default
    # static size and at dyn (where it reclaims across a re-chunk too).
    if (_vname STREQUAL "zero_any")
      list(APPEND _group cst2_reclaim_validation)
    endif()
    # zero_one_value_registry_growth_validation: zero_one_value's second-value registry must
    # not move the constant arrays chunks point at when a new value is
    # declared (the harness says why). zero_one_value only, default static size, at 1
    # and 4 threads.
    if (_vname STREQUAL "zero_one_value" AND NOT _geom STREQUAL "dyn")
      list(APPEND _group zero_one_value_registry_growth_validation)
    endif()
    # zero_one_value_foreign_key_validation: a chunk is only ever compressed to its own
    # array's second value (the harness says why). zero_one_value only, default static
    # size, at 8 threads; OpenMP builds only (the passes run in a region).
    if (OMP AND _vname STREQUAL "zero_one_value" AND NOT _geom STREQUAL "dyn")
      list(APPEND _group zero_one_value_foreign_key_validation)
    endif()
    # deferred_path_validation: every write path through the deferred queue,
    # forced by the test hook (test_hooks.h, switched on by the harness), at
    # 1 and 8 threads. Every variant and geometry: its replay path reads the
    # frozen geometry a dynamic build alone has.
    list(APPEND _group deferred_path_validation)
    # std_thread_validation: arrays driven from the application's own
    # std::threads (registered ranks, the application phase of the serial
    # contract), fresh threads every phase and long-lived ones. Std builds
    # only (an OpenMP build numbers only OpenMP's threads), every variant and
    # geometry, at 1 and 8 threads.
    if (CHUNK_THREADS_BACKEND STREQUAL "std")
      list(APPEND _group std_thread_validation)
    endif()
    # tbb_validation: arrays driven from a TBB program through
    # shrink_array::tbb::parallel_for (chunk/tbb_adaptor.h): every partitioner,
    # the index form, work in the phase, a nested parallel_for, a body that
    # throws. CHUNK_WITH_TBB builds, std and serial backends (an OpenMP build
    # refuses the adaptor), every variant and geometry, at 1 and 8 threads.
    if (CHUNK_WITH_TBB AND NOT CHUNK_THREADS_BACKEND STREQUAL "openmp")
      list(APPEND _group tbb_validation)
    endif()
    # read_reclaim_validation: a read or a compound op against an in-region
    # pass that releases the buffer, and a pass that must not erase another
    # pass's claim (the harness says why). OpenMP builds only, default static
    # size, at 8 threads (2 writers, 6 readers); 77 = skipped.
    if (OMP AND NOT _geom STREQUAL "dyn")
      list(APPEND _group read_reclaim_validation)
    endif()
    # phase_protocol_validation: the guard of the phase protocol (phase_guard.h)
    # stops a forgotten finalize / begin and lets the correct uses run, each
    # case in a child. OpenMP builds only, default static size, at 8 threads;
    # 77 = skipped (OMPT: the protocol is not manual there).
    if (OMP AND NOT _geom STREQUAL "dyn")
      list(APPEND _group phase_protocol_validation)
    endif()
    # rechunk_validation (content invariance across rechunk_to(), one test per
    # case) and dyn_geometry_validation (an array reads back what was written
    # at every registered runtime chunk size): dyn only, at 8 threads.
    if (_geom STREQUAL "dyn")
      list(APPEND _group rechunk_validation dyn_geometry_validation)
    endif()
    # Another static size: the round-trip harnesses alone.
    if (NOT _geom STREQUAL "dyn" AND NOT _geom STREQUAL CHUNK_DEFAULT_SIZE)
      set(_group ${_rt_group})
    endif()

    if (CHUNK_VALIDATION_UNITY)
      validation_harness_executable(validation_${_vname}_${_geom} ${_libbase} ${_vmacro} ${_geom} ${_group})
    endif()
    foreach(_h IN LISTS _group)
      set(_base ${_h}_${_vname}_${_geom})
      if (CHUNK_VALIDATION_UNITY)
        set(_exe validation_${_vname}_${_geom})
      else()
        validation_harness_executable(${_base} ${_libbase} ${_vmacro} ${_geom} ${_h})
        set(_exe ${_base})
      endif()
      if (_h STREQUAL "rechunk_validation")
        # One test per case (rechunk_validation --list), each in its serial and
        # parallel passes: `ctest -j` spreads them over the cores. The names
        # must match the case table of rechunk_validation.cpp -- an unknown
        # --case fails the test, so a drift shows up at once.
        foreach(_case coexistence range-ops scale-64-to-tgt mostly-constant sin-quantized
                      shrink-to-smaller all-same all-different val2-grow val2-shrink
                      invalid-size-refused active-size-refused multiple-of-64-chain index-split
                      sliced-grow sliced-shrink)
          validation_add_test(${_base}_${_case} ${_exe} ${_h} "" --case ${_case})
        endforeach()
      elseif (_h STREQUAL "dyn_geometry_validation")
        validation_add_test(${_base} ${_exe} ${_h} "")
      elseif (_h STREQUAL "index64_validation" AND _geom STREQUAL "dyn")
        validation_add_test(${_base} ${_exe} ${_h} "1;8")
        # _np2: from 384, not a power of two -- the magic-number index split,
        # a re-chunk 384 -> 768, and past 2^31 at 768.
        validation_add_test(${_base}_np2 ${_exe} ${_h} "1;8" --size 384)
      elseif (_h STREQUAL "bits_container_validation")
        validation_add_test(bits_container_validation_${_geom} ${_exe} ${_h} "1;8")
      elseif (_h STREQUAL "zero_one_value_registry_growth_validation")
        validation_add_test(zero_one_value_registry_growth_validation ${_exe} ${_h} "1;4")
      elseif (_h STREQUAL "zero_one_value_foreign_key_validation")
        validation_add_test(zero_one_value_foreign_key_validation ${_exe} ${_h} "8")
      elseif (_h STREQUAL "tbb_validation")
        if (SANITIZER STREQUAL "thread")
          # ThreadSanitizer: N = 2^16 elements instead of 2^18 -- every form,
          # partitioner and phase still runs, on four times fewer elements: at
          # 2^18, zero_one_value at 8 threads went past ctest's 1500 s on a
          # two-core runner. libtbb.so is not instrumented, and its
          # partitioners' bookkeeping shows up as races (tsan_tbb.supp says
          # why and what is left visible): appended to TSAN_OPTIONS, so the
          # CI's halt_on_error stays.
          validation_add_test(${_base} ${_exe} ${_h} "1;8" 65536)
          foreach(_t ${_base}_1thr ${_base}_8thr)
            set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                         "TSAN_OPTIONS=string_append: suppressions=${PROJECT_SOURCE_DIR}/tsan_tbb.supp")
            set_property(TEST ${_t} PROPERTY TIMEOUT 3000)
          endforeach()
        else()
          validation_add_test(${_base} ${_exe} ${_h} "1;8")
        endif()
      elseif (_h STREQUAL "deferred_path_validation")
        validation_add_test(${_base} ${_exe} ${_h} "1;8")
      elseif (_h STREQUAL "phase_protocol_validation")
        validation_add_test(${_base} ${_exe} ${_h} "8")
        set_property(TEST ${_base}_8thr PROPERTY SKIP_RETURN_CODE 77)
      elseif (_h STREQUAL "read_reclaim_validation")
        validation_add_test(${_base} ${_exe} ${_h} "8")
        validation_add_test(${_base}_compound ${_exe} ${_h} "8" compound)
        validation_add_test(${_base}_remark ${_exe} ${_h} "8" remark)
        foreach(_t ${_base}_8thr ${_base}_compound_8thr ${_base}_remark_8thr)
          set_property(TEST ${_t} PROPERTY SKIP_RETURN_CODE 77)
          set_property(TEST ${_t} PROPERTY TIMEOUT 120)
        endforeach()
      else()
        validation_add_test(${_base} ${_exe} ${_h} "1;8")
      endif()
    endforeach()
  endforeach()
endforeach()

# --- pattern_validation: pattern::Array, a member of the zero_any family,
# against std::vector, and the second-value registry it shares with
# zero_any::Array. Its own executable, outside the zero_any group: it
# instantiates pattern::Array and zero_any::Array for four element types (about
# 2 GB of compiler memory alone, at -O3 -g), which took the group's unity TU
# past 7 GB. At the default static size: pattern::Array is static only.
if (TARGET chunk_zero_any_${CHUNK_DEFAULT_SIZE})
  set(_tgt pattern_validation_zero_any_${CHUNK_DEFAULT_SIZE})
  validation_harness_executable(${_tgt} chunk_zero_any CHUNK_VARIANT_ZERO_ANY ${CHUNK_DEFAULT_SIZE}
                                pattern_validation)
  validation_add_test(${_tgt} ${_tgt} pattern_validation "1;8")
endif()

# --- element_ref_validation: what operator[] accepts and refuses
# (chunk/internal/element_ref.h), in every build that has the harnesses.
#   element_ref_validation_<variant>: every std::vector-style use of a[i]
#     compiles and leaves std::vector's values (default static size).
#   element_ref_misuse_<n>: one use of a NAMED element proxy, which must not
#     compile: the test builds the object target (not part of `all`) and passes
#     when the compiler output carries ElementRef's message. One at a time:
#     they share the build tree.
foreach(_vi RANGE ${_rt_nvar_last})
  list(GET _rt_variant_names   ${_vi} _vname)
  list(GET _rt_variant_libbase ${_vi} _libbase)
  list(GET _rt_variant_macro   ${_vi} _vmacro)
  if (TARGET ${_libbase}_${CHUNK_DEFAULT_SIZE})
    set(_tgt element_ref_validation_${_vname})
    add_executable(${_tgt} element_ref_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${CHUNK_DEFAULT_SIZE})
    validation_variant_defs(${_tgt} ${_vmacro} ${CHUNK_DEFAULT_SIZE})
    validation_setup_target(${_tgt})
    add_test(NAME ${_tgt} COMMAND ${_tgt})
    set_property(TEST ${_tgt} PROPERTY LABELS validation)
  endif()
endforeach()
if (TARGET chunk_zero_any_${CHUNK_DEFAULT_SIZE})
  foreach(_case RANGE 1 8)
    set(_tgt element_ref_misuse_${_case})
    add_library(${_tgt} OBJECT EXCLUDE_FROM_ALL element_ref_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE chunk_zero_any_${CHUNK_DEFAULT_SIZE})
    validation_variant_defs(${_tgt} CHUNK_VARIANT_ZERO_ANY ${CHUNK_DEFAULT_SIZE})
    target_compile_definitions(${_tgt} PRIVATE ELEMENT_REF_CASE=${_case})
    validation_setup_target(${_tgt})
    add_test(NAME ${_tgt} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${_tgt})
    # LC_ALL=C: the compiler's own words in English, whatever the locale (a
    # French GCC writes "est supporté conditionnellement" for case 9).
    set_tests_properties(${_tgt} PROPERTIES PASS_REGULAR_EXPRESSION "named element proxy is used"
                                            ENVIRONMENT LC_ALL=C
                                            RESOURCE_LOCK element_ref_misuse_build LABELS validation)
  endforeach()
  # 9: std::printf("%f", a[i]) passes the proxy through `...`: GCC with
  # chunk_hardening's -Werror=conditionally-supported ("... is conditionally
  # supported"), clang by default ("cannot pass non-trivial object ... to
  # variadic function").
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    set(_tgt element_ref_misuse_9)
    add_library(${_tgt} OBJECT EXCLUDE_FROM_ALL element_ref_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE chunk_zero_any_${CHUNK_DEFAULT_SIZE})
    validation_variant_defs(${_tgt} CHUNK_VARIANT_ZERO_ANY ${CHUNK_DEFAULT_SIZE})
    target_compile_definitions(${_tgt} PRIVATE ELEMENT_REF_CASE=9)
    validation_setup_target(${_tgt})
    add_test(NAME ${_tgt} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${_tgt})
    set_tests_properties(${_tgt} PROPERTIES
                         PASS_REGULAR_EXPRESSION "is conditionally supported|to variadic function"
                         ENVIRONMENT LC_ALL=C
                         RESOURCE_LOCK element_ref_misuse_build LABELS validation)
  endif()
endif()

# --- fill_range_mt_zero_and_zero_one_value_validation: the two-variant library (CHUNK_VARIANT_ZERO=1;CHUNK_VARIANT_ZERO_ONE_VALUE=1,
# default static size, no OMPT tool -- the harness defines the switch and makes
# the phase calls, as multi-mat does), zero::Array and zero_one_value::Array in one
# program. At 1 and 8 threads like the others.
if (TARGET chunk_zero_and_zero_one_value_${CHUNK_DEFAULT_SIZE})
  set(_tgt fill_range_mt_zero_and_zero_one_value_validation)
  add_executable(${_tgt} fill_range_mt_zero_and_zero_one_value_validation.cpp)
  target_link_libraries(${_tgt} PRIVATE chunk_zero_and_zero_one_value_${CHUNK_DEFAULT_SIZE}) # CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE, CHUNK_SIZE: PUBLIC
  validation_setup_target(${_tgt})
  foreach(_nthr 1 8)
    add_test(NAME ${_tgt}_${_nthr}thr COMMAND ${_tgt})
    set_property(TEST ${_tgt}_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
    set_property(TEST ${_tgt}_${_nthr}thr PROPERTY LABELS validation)
  endforeach()
endif()

# --- bit_words_validation: the bit-packed storage kernel of ArrayChunkBits
# alone (header only, no variant, no library): get/set against
# std::vector<bool>, the word scans, and concurrent writes to neighbouring
# bits of one word through the atomic read-modify-write.
find_package(Threads REQUIRED)
# SANITIZER=token: -fsanitize=alloc-token rewrites every allocation into
# __alloc_token_<name>(..., token), whose definitions (alloc_token_rt.cpp) are
# compiled INTO each chunk variant archive. The executables below link none, so
# they get their own copy -- uninstrumented, or its own malloc calls would
# recurse. A no-op for every other sanitizer.
function(chunk_add_alloc_token_rt target)
  if (SANITIZER STREQUAL "token")
    set(_rt ${PROJECT_SOURCE_DIR}/chunk/private/alloc_token_rt.cpp)
    target_sources(${target} PRIVATE ${_rt})
    set_source_files_properties(${_rt} DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                                PROPERTIES COMPILE_OPTIONS "-fno-sanitize=alloc-token")
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public # env_parse.h
                                                 ${PROJECT_SOURCE_DIR}/fast_allocators/public) # its log_core.h
  endif()
endfunction()

add_executable(bit_words_validation bit_words_validation.cpp)
chunk_add_alloc_token_rt(bit_words_validation)
target_include_directories(bit_words_validation PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public)
target_link_libraries(bit_words_validation PRIVATE Threads::Threads)
chunk_require_cxx20(bit_words_validation PRIVATE)
add_test(NAME bit_words_validation COMMAND bit_words_validation)
set_property(TEST bit_words_validation PROPERTY LABELS validation)

# --- rank_registry_validation: the rank registry of the threading backend
# (fast_allocators/internal/pmp_runtime.h) alone, header only, driven from
# std::threads: dense, unique, bounded ranks, given back at thread exit and
# reused; the serial backend's queries are those of one thread. Every build,
# with OpenMP when on (the backend the build selects).
add_executable(rank_registry_validation rank_registry_validation.cpp)
chunk_add_alloc_token_rt(rank_registry_validation)
target_include_directories(rank_registry_validation PRIVATE ${PROJECT_SOURCE_DIR}/fast_allocators/public)
target_link_libraries(rank_registry_validation PRIVATE Threads::Threads)
if (OMP)
  target_link_libraries(rank_registry_validation PRIVATE OpenMP::OpenMP_CXX)
endif ()
chunk_require_cxx20(rank_registry_validation PRIVATE)
add_test(NAME rank_registry_validation COMMAND rank_registry_validation)
set_property(TEST rank_registry_validation PROPERTY LABELS validation)

# --- log_validation: the library's log (fast_allocators/internal/log_core.h)
# alone -- CHUNK_LOG / CHUNK_LOG_FILE, the filter, a sink, the rate-limited
# macros, the cut of a long message, no allocation on the path of a message,
# whole lines from 8 threads -- and its fatal level, CHUNK_INVARIANT_FAILED
# and CHUNK_CHECK_FAILED, each case in a child (the executable re-runs
# itself with --fatal K). Header-only: no library, no variant.
add_executable(log_validation log_validation.cpp)
target_include_directories(log_validation PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public
                                                  ${PROJECT_BINARY_DIR}/chunk/generated # chunk/build_options.h
                                                  ${PROJECT_SOURCE_DIR}/fast_allocators/public)
target_link_libraries(log_validation PRIVATE Threads::Threads)
if (OMP)
  target_link_libraries(log_validation PRIVATE OpenMP::OpenMP_CXX) # build_options.h asks for the library's OpenMP
endif()
# ... and for the library's backend: the std backend's macro, which the chunk:: targets
# carry PUBLIC, but this target links none of them.
if (CHUNK_THREADS_BACKEND STREQUAL "std")
  target_compile_definitions(log_validation PRIVATE FAST_ALLOC_PMP_STD=1)
endif()
chunk_require_cxx20(log_validation PRIVATE)
add_test(NAME log_validation COMMAND log_validation)
set_property(TEST log_validation PROPERTY LABELS validation)

# --- team_loops_validation: pmp::Team's static loops (for_static,
# for_static_chunks and their _simd forms, fast_allocators/internal/
# pmp_loops.h) alone, on the build's backend: every iteration once, on the
# rank OpenMP gives it, the same share in two nowait loops in a row -- what
# the bench's barrier-free regions rely on. Header-only: no library, no variant.
add_executable(team_loops_validation team_loops_validation.cpp)
target_include_directories(team_loops_validation PRIVATE ${PROJECT_SOURCE_DIR}/fast_allocators/public)
target_link_libraries(team_loops_validation PRIVATE Threads::Threads)
if (OMP)
  target_link_libraries(team_loops_validation PRIVATE OpenMP::OpenMP_CXX)
endif()
if (CHUNK_THREADS_BACKEND STREQUAL "std")
  target_compile_definitions(team_loops_validation PRIVATE FAST_ALLOC_PMP_STD=1)
endif()
chunk_require_cxx20(team_loops_validation PRIVATE)
add_test(NAME team_loops_validation COMMAND team_loops_validation)
set_property(TEST team_loops_validation PROPERTY LABELS validation)

# --- blk_allocator_validation: fast_allocators' public size-dispatching front
# end (BlkAllocator) alone. Nothing in the library calls it, so this is what
# runs it: sizes on, between and past the buckets, many live blocks at once,
# every thread at once. Any build that has fast_allocators.
if (NOT CHUNK_STD_ALLOC)
  add_executable(blk_allocator_validation blk_allocator_validation.cpp)
  target_link_libraries(blk_allocator_validation PRIVATE fast_allocators)
  if (OMP)
    target_link_libraries(blk_allocator_validation PRIVATE OpenMP::OpenMP_CXX)
  endif ()
  chunk_add_alloc_token_rt(blk_allocator_validation)
  chunk_require_cxx20(blk_allocator_validation PRIVATE)
  # chunk/internal/print.h (println), header-only: the include path, not the library.
  target_include_directories(blk_allocator_validation PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public)
  foreach(_nthr 1 8)
    add_test(NAME blk_allocator_validation_${_nthr}thr COMMAND blk_allocator_validation)
    set_property(TEST blk_allocator_validation_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
    set_property(TEST blk_allocator_validation_${_nthr}thr PROPERTY LABELS validation)
  endforeach()
endif()

# --- slab_geometry_validation: the zeroed value allocator alone (no variant).
# Every allocator of one element type must cut its slabs at one size whatever
# the element size, or slab_of() through the ambient-size allocator misses the
# block's slab after a re-chunk (see the harness). TCTypeCAllocator exists
# under OpenMP only, and fast_allocators is not built with CHUNK_STD_ALLOC.
if (OMP AND NOT CHUNK_STD_ALLOC)
  add_executable(slab_geometry_validation slab_geometry_validation.cpp)
  target_link_libraries(slab_geometry_validation PRIVATE fast_allocators OpenMP::OpenMP_CXX)
  chunk_add_alloc_token_rt(slab_geometry_validation)
  chunk_require_cxx20(slab_geometry_validation PRIVATE)
  # chunk/internal/print.h (println), header-only: the include path, not the library.
  target_include_directories(slab_geometry_validation PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public)
  add_test(NAME slab_geometry_validation COMMAND slab_geometry_validation)
  set_property(TEST slab_geometry_validation PROPERTY LABELS validation)
endif()


# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
