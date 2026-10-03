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
#   nested_region_validation a thread that opens a nested (serialised) region
#                           keeps its own per-rank structures afterwards; two
#                           ACTIVE levels stop with an invariant failure
#                           (re-run with --nested-active).
#   fill_range_mt_v1v2_validation  the range ops in a TWO-variant build
#                           (chunk_v1v2): an ArrayChunk and an ArrayChunk2 side
#                           by side, fill_range_mt of each compressible value.
#                           The ops must dispatch on the array's type, not on
#                           the build flags -- an ArrayChunk2 filled with its
#                           second value used to stay materialised there.
#                           The one harness that is not variant-agnostic.
#   cst2_registry_validation  Chunk2A's second-value registry is freed by the
#                           last array of every generation, not only the
#                           first (its reinstall took a phantom reference),
#                           and overlapping arrays free nothing early.
#                           Chunk2A only, default static size.
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

# --- The grouped harnesses: one executable per variant and geometry,
# validation_<variant>_<geometry>, a unity build. Compiled one by one, each
# harness re-instantiated the whole library for itself: 14 s apiece in a Debug
# build, against 0.4 s of link. Together they share the instantiations -- a
# pilot of six of them compiled in 13.8 s instead of 56.8 (Debug) and 28.4
# instead of 119.4 (-O2), for the same peak memory. Each harness is a
# namespace, validation::<harness>, with a run() the shared driver calls
# (validation_harness.h, validation_driver.cpp):
#
#     validation_chunk2a_256 compress_roundtrip [args...]
#
# The ctest names are the ones of the one-executable-per-harness layout
# (compress_roundtrip_chunk2a_256_8thr, rechunk_validation_chunk_dyn_val2-grow,
# ...), so `ctest -R` and python/bench_and_test.py see no difference.
# CHUNK_VALIDATION_UNITY=OFF goes back to one executable per harness, named as
# before -- the one to debug a harness alone. CHUNK_VALIDATION_UNITY_BATCH caps
# the harnesses per unity TU (0: all of a group in one).
#
# Not grouped: the harnesses built with a knob of their own that changes
# header-only library code (deferred_path_validation, CHUNK_FORCE_DEFER_EVERY;
# read_reclaim_validation, CHUNK_TEST_READ_DELAY_*) -- in one executable with
# the others it would be an ODR violation -- and those built against another
# library or none (fill_range_mt_v1v2, bit_words, blk_allocator,
# slab_geometry). RECHUNK_TRACE_POOL, a trace, applies to the whole group of
# rechunk_validation.
option(CHUNK_VALIDATION_UNITY "One unity-built executable per variant and geometry for the validation harnesses" ON)
set(CHUNK_VALIDATION_UNITY_BATCH "0" CACHE STRING "Harnesses per unity translation unit (0: all of a group)")

set(_rt_variant_names   chunk       chunk2      chunk2a)
set(_rt_variant_libbase chunk_v1    chunk_v2    chunk_v2a)
set(_rt_variant_macro   CHUNK       CHUNK2      CHUNK2A)

set(_rt_geometries ${CHUNK_DEFAULT_SIZE})
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
                        nested_region_validation iterator_validation stale_proxy_validation)
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
  if (RECHUNK_TRACE_POOL AND "rechunk_validation" IN_LIST ARGN)
    target_compile_definitions(${tgt} PRIVATE RECHUNK_TRACE_POOL)
  endif()
  validation_setup_target(${tgt})
  if (CHUNK_VALIDATION_UNITY)
    set_target_properties(${tgt} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE ${CHUNK_VALIDATION_UNITY_BATCH})
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

    # The group: the round-trip harnesses, then the special cases below.
    set(_group "")
    foreach(_h IN LISTS _rt_harnesses)
      if (NOT _geom STREQUAL "dyn" OR _h IN_LIST _rt_dyn_harnesses)
        list(APPEND _group ${_h})
      endif()
    endforeach()
    # chunk_array_new_validation: the variant's operator new[] hands back an
    # array whose every chunk is at the first constant. OpenMP builds only: the
    # operators are the *OmpPolicy's, and their serial specialisations declare
    # nothing.
    if (OMP)
      list(APPEND _group chunk_array_new_validation)
    endif()
    # bits_container_validation: ArrayChunkBits (ArrayChunk2<Bit>) against
    # std::vector<bool>. Chunk2 libraries only, every geometry.
    if (_vname STREQUAL "chunk2")
      list(APPEND _group bits_container_validation)
    endif()
    # cst2_registry_validation: Chunk2A's second-value registry lifecycle.
    # Chunk2A only, at the default static size.
    if (_vname STREQUAL "chunk2a" AND NOT _geom STREQUAL "dyn")
      list(APPEND _group cst2_registry_validation)
    endif()
    # chunk2_registry_growth_validation: Chunk2's second-value registry must
    # not move the constant arrays chunks point at when a new value is
    # declared (the harness says why). Chunk2 only, default static size, at 1
    # and 4 threads.
    if (_vname STREQUAL "chunk2" AND NOT _geom STREQUAL "dyn")
      list(APPEND _group chunk2_registry_growth_validation)
    endif()
    # rechunk_validation (content invariance across rechunk_to(), one test per
    # case) and dyn_geometry_validation (an array reads back what was written
    # at every registered runtime chunk size): dyn only, at 8 threads.
    if (_geom STREQUAL "dyn")
      list(APPEND _group rechunk_validation dyn_geometry_validation)
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
                      non-pow2-refused)
          validation_add_test(${_base}_${_case} ${_exe} ${_h} "" --case ${_case})
        endforeach()
      elseif (_h STREQUAL "dyn_geometry_validation")
        validation_add_test(${_base} ${_exe} ${_h} "")
      elseif (_h STREQUAL "bits_container_validation")
        validation_add_test(bits_container_validation_${_geom} ${_exe} ${_h} "1;8")
      elseif (_h STREQUAL "chunk2_registry_growth_validation")
        validation_add_test(chunk2_registry_growth_validation ${_exe} ${_h} "1;4")
      else()
        validation_add_test(${_base} ${_exe} ${_h} "1;8")
      endif()
    endforeach()
  endforeach()
endforeach()

# --- deferred_path_validation: on its own, one executable per variant and
# geometry (both: its replay path reads the frozen geometry a dynamic build
# alone has), at 1 and 8 threads. The forced-deferral knob, on this harness
# alone (the write engine is header-only, instantiated in the executable: the
# library archive holds none of it, so a target-scoped value is consistent). A
# global value set on the cache (a whole-tree forced build) is not overridden.
foreach(_geom IN LISTS _rt_geometries)
  foreach(_vi RANGE ${_rt_nvar_last})
    list(GET _rt_variant_names   ${_vi} _vname)
    list(GET _rt_variant_libbase ${_vi} _libbase)
    list(GET _rt_variant_macro   ${_vi} _vmacro)
    set(_tgt deferred_path_validation_${_vname}_${_geom})
    add_executable(${_tgt} deferred_path_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${_geom})
    validation_variant_defs(${_tgt} ${_vmacro} ${_geom})
    validation_setup_target(${_tgt})
    if ("${CHUNK_FORCE_DEFER_EVERY}" STREQUAL "0")
      target_compile_definitions(${_tgt} PRIVATE CHUNK_FORCE_DEFER_EVERY=7)
    endif()
    foreach(_nthr 1 8)
      add_test(NAME ${_tgt}_${_nthr}thr COMMAND ${_tgt})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_nthr})
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY LABELS validation)
    endforeach()
  endforeach()
endforeach()

# --- fill_range_mt_v1v2_validation: the two-variant library (CHUNK=1;CHUNK2=1,
# default static size, no OMPT tool -- the harness defines the switch and makes
# the phase calls, as multi-mat does), ArrayChunk and ArrayChunk2 in one
# program. At 1 and 8 threads like the others.
if (TARGET chunk_v1v2_${CHUNK_DEFAULT_SIZE})
  set(_tgt fill_range_mt_v1v2_validation)
  add_executable(${_tgt} fill_range_mt_v1v2_validation.cpp)
  target_link_libraries(${_tgt} PRIVATE chunk_v1v2_${CHUNK_DEFAULT_SIZE}) # CHUNK, CHUNK2, CHUNK_SIZE: PUBLIC
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
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public) # env_parse.h
  endif()
endfunction()

add_executable(bit_words_validation bit_words_validation.cpp)
chunk_add_alloc_token_rt(bit_words_validation)
target_include_directories(bit_words_validation PRIVATE ${PROJECT_SOURCE_DIR}/chunk/public)
target_link_libraries(bit_words_validation PRIVATE Threads::Threads)
chunk_require_cxx20(bit_words_validation PRIVATE)
add_test(NAME bit_words_validation COMMAND bit_words_validation)
set_property(TEST bit_words_validation PROPERTY LABELS validation)

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
  add_test(NAME slab_geometry_validation COMMAND slab_geometry_validation)
  set_property(TEST slab_geometry_validation PROPERTY LABELS validation)
endif()

# --- read_reclaim_validation: a pure read against a compression pass that runs
# inside the region and releases the buffer the read is about to dereference
# (the harness says why). OpenMP builds only, default static size, the three
# variants. CHUNK_TEST_READ_DELAY_EVERY on this target alone (the read path is
# header-only, instantiated here -- same reasoning as CHUNK_FORCE_DEFER_EVERY):
# one read in 16 of the reader threads waits 50 us between the pointer load and
# the dereference. Run at 8 threads (2 writers, 6 readers); 77 = skipped.
if (OMP)
  foreach(_vi RANGE ${_rt_nvar_last})
    list(GET _rt_variant_names   ${_vi} _vname)
    list(GET _rt_variant_libbase ${_vi} _libbase)
    list(GET _rt_variant_macro   ${_vi} _vmacro)
    set(_tgt read_reclaim_validation_${_vname}_${CHUNK_DEFAULT_SIZE})
    add_executable(${_tgt} read_reclaim_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${CHUNK_DEFAULT_SIZE})
    target_compile_definitions(${_tgt} PRIVATE ${_vmacro} CHUNK_TEST_READ_DELAY_EVERY=16 CHUNK_TEST_READ_DELAY_US=50)
    validation_setup_target(${_tgt})
    add_test(NAME ${_tgt}_8thr COMMAND ${_tgt})
    set_property(TEST ${_tgt}_8thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST ${_tgt}_8thr PROPERTY LABELS validation)
    set_property(TEST ${_tgt}_8thr PROPERTY SKIP_RETURN_CODE 77)
    set_property(TEST ${_tgt}_8thr PROPERTY TIMEOUT 120)
    # The same window on the write side: a compound operation's v_old read
    # (the `compound` argument; the harness says why). 77 on chunk (v1).
    add_test(NAME ${_tgt}_compound_8thr COMMAND ${_tgt} compound)
    set_property(TEST ${_tgt}_compound_8thr PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST ${_tgt}_compound_8thr PROPERTY LABELS validation)
    set_property(TEST ${_tgt}_compound_8thr PROPERTY SKIP_RETURN_CODE 77)
    set_property(TEST ${_tgt}_compound_8thr PROPERTY TIMEOUT 120)
  endforeach()
endif()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
