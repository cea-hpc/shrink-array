# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# A variant library is compiled with one of CHUNK_VARIANT_V1 / _V2 / _V2A
# defined (or none, for the vector-mode build), PUBLIC: the macros drive
# `chunk_config::is_shrink_enabled` and friends in public/chunk/config.h, and
# they are read *inside* library-side .cpp files (AllTypeChunk.cpp,
# manage_ompt.cpp today; potentially others tomorrow). The options that are the
# same for every library of the build are not definitions: they are recorded
# in the generated chunk/build_options.h (below).
# ---------------------------------------------------------------------------

# The version is set once, in the top-level project(shrink_array VERSION ...).
# MAJOR: incompatible change; MINOR: compatible change; PATCH: fixes only
# (CHANGELOG.md, "Versioning policy").
set(CHUNK_MAJOR_VERSION ${shrink_array_VERSION_MAJOR})
set(CHUNK_MINOR_VERSION ${shrink_array_VERSION_MINOR})
set(CHUNK_PATCH_VERSION ${shrink_array_VERSION_PATCH})

set(CHUNK_VERSION
        ${CHUNK_MAJOR_VERSION}.${CHUNK_MINOR_VERSION}.${CHUNK_PATCH_VERSION})

if (CHUNK_OMP)
  find_package(OpenMP REQUIRED)
endif ()
# Threads::Threads, linked PUBLIC by every variant (see chunk_add_variant).
find_package(Threads REQUIRED)

option(CHUNK_STD_ALLOC "Use standard allocator for chunk." OFF)

# --- Multi-size declinations --------------------------------------------
# The compile-time chunk size (static_chunk_size) is frozen per library by
# -DCHUNK_SIZE. To benchmark several fixed sizes we therefore build ONE lib
# per (variant, size). CHUNK_STATIC_SIZES is the list of static sizes; each
# yields chunk_v<x>_<size>. CHUNK_BUILD_DYN adds a runtime-size (CHUNK_DYNAMIC)
# lib chunk_v<x>_dyn per variant. The list is resolved by the top-level
# CMakeLists.txt (auto: 256 alone in a Debug or sanitizer build); trim it during
# development to cut build time (e.g. -DCHUNK_STATIC_SIZES=256). Not a cache
# entry here: it would hide the resolved list from this directory.
if(NOT DEFINED CHUNK_STATIC_SIZES OR CHUNK_STATIC_SIZES STREQUAL "auto")
  set(CHUNK_STATIC_SIZES "256;1024")
endif()
option(CHUNK_BUILD_DYN "Also build a CHUNK_DYNAMIC (runtime size) lib per variant." ON)
message(STATUS "CHUNK_STATIC_SIZES = ${CHUNK_STATIC_SIZES}; CHUNK_BUILD_DYN = ${CHUNK_BUILD_DYN}")
# The variants built: chunk (ArrayChunk, chunk_v1_*), chunk2 (ArrayChunk2,
# chunk_v2_*), chunk2a (ArrayChunk2A, chunk_v2a_*). The examples, benchmarks
# and harnesses of a variant left out are not built nor registered as tests. A
# build that checks one variant's option (CHUNK_SECOND_VALUE_INDEXED is
# Chunk2A's) needs only that one; trim it during development too.
set(CHUNK_VARIANTS "chunk;chunk2;chunk2a" CACHE STRING
    "Semicolon list of the variants to build: chunk, chunk2, chunk2a.")
foreach(_v IN LISTS CHUNK_VARIANTS)
  if (NOT _v MATCHES "^(chunk|chunk2|chunk2a)$")
    message(FATAL_ERROR "CHUNK_VARIANTS: unknown variant '${_v}' (chunk, chunk2, chunk2a)")
  endif()
endforeach()
if (NOT CHUNK_VARIANTS)
  message(FATAL_ERROR "CHUNK_VARIANTS is empty: name at least one of chunk, chunk2, chunk2a")
endif()
message(STATUS "CHUNK_VARIANTS = ${CHUNK_VARIANTS}")

# --- Aggressive vs release compression scheduling ------------------------
# A run-time switch (chunk_config::aggressive_compression(), environment
# CHUNK_AGGRESSIVE_COMPRESSION=1 / 0); the option only sets its default.
option(HAVE_AGGRESSIVE_COMPRESSION "Default of the run-time aggressive-compression switch (CHUNK_AGGRESSIVE_COMPRESSION)" OFF)
if (HAVE_AGGRESSIVE_COMPRESSION)
  message(STATUS "HAVE_AGGRESSIVE_COMPRESSION=ON → aggressive scheduling by default (CHUNK_AGGRESSIVE_COMPRESSION=0 turns it off)")
endif()

# --- Read-side deferred-op sync -----------------------------------------
# Permanent: pending ops are synchronised at the read entry points, not
# flushed by the write operators (see the read-side visibility note in
# config.h). Nothing to define here for it.
#
# Backlog bound. Empty = use the header default (4096).
set(CHUNK_DEFERRED_DRAIN_THRESHOLD "" CACHE STRING
    "Own-rank deferred-backlog drain threshold (empty = header default 4096).")
if (NOT "${CHUNK_DEFERRED_DRAIN_THRESHOLD}" STREQUAL "")
  message(STATUS "CHUNK_DEFERRED_DRAIN_THRESHOLD = ${CHUNK_DEFERRED_DRAIN_THRESHOLD}")
endif()

# Test knob: force one write in N down the deferred path (enqueue / replay /
# backlog drain / read gate) with no compressor in flight; 0 = off. A build
# with, say, 7 must pass the shrink_bench file cross-check bit for bit. See
# config.h.
set(CHUNK_FORCE_DEFER_EVERY "0" CACHE STRING
    "Force one write in N onto the deferred path (test knob, 0 = off).")
if (NOT "${CHUNK_FORCE_DEFER_EVERY}" STREQUAL "0")
  message(STATUS "CHUNK_FORCE_DEFER_EVERY = ${CHUNK_FORCE_DEFER_EVERY} (TEST build: forced deferrals)")
endif()

# --- DIAGNOSTIC toggles --------------------------------------------------
#set(CHUNK_POOL_GUARD_BYTES "0" CACHE STRING "Trailing pad bytes per pooled element (OOB isolation)")
option(CHUNK_ALIGN_CHECK "Abort if a slab block is misaligned for its type" OFF)
#if (NOT CHUNK_POOL_GUARD_BYTES STREQUAL "0")
#  message(STATUS "CHUNK_POOL_GUARD_BYTES=${CHUNK_POOL_GUARD_BYTES} → padding each pooled element (OOB isolation)")
#endif()
if (CHUNK_ALIGN_CHECK)
  message(STATUS "CHUNK_ALIGN_CHECK=ON → aborting on misaligned slab block")
endif()

# Diagnostic switches (they were bare -D macros: PERF, DO_CHECK, ...).
option(CHUNK_DO_CHECK "Check every written value for NaN (global_check_nan)" OFF)
option(CHUNK_RECHUNK_LEDGERTRACE "Trace every pool-ledger state transition on stderr" OFF)
option(CHUNK_RING_LOG_DEBUG "Ring-log every event, hot path included (heavy)" OFF)
option(CHUNK_RING_LOG_OP "Ring-log the op-lifecycle events only (light)" OFF)

# --- chunk/build_options.h ------------------------------------------------
# The options every library of this build shares, generated into the build
# tree and installed: the headers read them from there, not from the command
# line, so an installed consumer compiles them exactly as the library was.
macro(_chunk_opt name value)
  if (${value})
    set(${name} 1)
  else()
    set(${name} 0)
  endif()
endmacro()
_chunk_opt(CHUNK_OPT_OPENMP CHUNK_OMP)
if (NOT DEFINED CHUNK_THREADED)
  set(CHUNK_THREADED ${CHUNK_OMP})
endif()
_chunk_opt(CHUNK_OPT_THREADED CHUNK_THREADED)
if ("${CHUNK_THREADS_BACKEND}" STREQUAL "std")
  set(CHUNK_OPT_STD_THREADS 1)
else()
  set(CHUNK_OPT_STD_THREADS 0)
endif()
_chunk_opt(CHUNK_OPT_STD_ALLOC CHUNK_STD_ALLOC)
if (CHUNK_OPT_STD_ALLOC)
  set(CHUNK_OPT_FAST_ALLOC 0)
else()
  set(CHUNK_OPT_FAST_ALLOC 1)
endif()
_chunk_opt(CHUNK_OPT_OMPT CHUNK_HAVE_OMPT)
_chunk_opt(CHUNK_OPT_DEBUG CHUNK_DEBUG)
_chunk_opt(CHUNK_OPT_CHECK_OPTIM CHECK_OPTIM)
_chunk_opt(CHUNK_OPT_STATS CHUNK_STATS)
_chunk_opt(CHUNK_OPT_MM_PROTECT CHUNK_MM_PROTECT)
_chunk_opt(CHUNK_OPT_SECOND_VALUE_INDEXED CHUNK_SECOND_VALUE_INDEXED)
_chunk_opt(CHUNK_OPT_RECLAIM_EMPTY_SLABS CHUNK_RECLAIM_EMPTY_SLABS)
_chunk_opt(CHUNK_OPT_AGGRESSIVE_COMPRESSION HAVE_AGGRESSIVE_COMPRESSION)
_chunk_opt(CHUNK_OPT_POOL_NO_REUSE CHUNK_POOL_NO_REUSE)
_chunk_opt(CHUNK_OPT_ALIGN_CHECK CHUNK_ALIGN_CHECK)
_chunk_opt(CHUNK_OPT_FORCE_LEDGER CHUNK_FORCE_LEDGER)
_chunk_opt(CHUNK_OPT_TEST_HOOKS CHUNK_TEST_HOOKS)
_chunk_opt(CHUNK_OPT_DO_CHECK CHUNK_DO_CHECK)
_chunk_opt(CHUNK_OPT_RECHUNK_LEDGERTRACE CHUNK_RECHUNK_LEDGERTRACE)
_chunk_opt(CHUNK_OPT_RING_LOG_DEBUG CHUNK_RING_LOG_DEBUG)
_chunk_opt(CHUNK_OPT_RING_LOG_OP CHUNK_RING_LOG_OP)
if ("${CHUNK_DEFERRED_DRAIN_THRESHOLD}" STREQUAL "")
  set(CHUNK_OPT_DEFERRED_DRAIN_THRESHOLD 4096)
else()
  set(CHUNK_OPT_DEFERRED_DRAIN_THRESHOLD ${CHUNK_DEFERRED_DRAIN_THRESHOLD})
endif()
set(CHUNK_OPT_FORCE_DEFER_EVERY ${CHUNK_FORCE_DEFER_EVERY})
set(CHUNK_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(public/chunk/build_options.h.in ${CHUNK_GENERATED_INCLUDE_DIR}/chunk/build_options.h @ONLY)

option(CHUNK_NO_IPA_ICF "Disable -fipa-icf (deferred pool aliasing diagnosis)" OFF)
if (CHUNK_NO_IPA_ICF AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "CHUNK_NO_IPA_ICF=ON → adding -fno-ipa-icf (deferred pool aliasing diagnosis)")
endif()

# --- HARDENING toggles ---------------------------------------------------
# Both are carried by one INTERFACE target, linked PUBLIC by every variant, so
# that the library AND every consumer TU see exactly the same definitions and
# instrumentation. That matters here: this code base is template-heavy and a
# define that differs between the lib's .cpp and the example's .cpp gives two
# incompatible definitions of the same class (ODR) -- the kind of divergence
# we spent a day chasing. Propagating PUBLIC removes that class of mistake.
#
# CHUNK_GLIBCXX_ASSERTIONS: libstdc++ precondition checks (vector::operator[]
#   bounds, iterator validity, empty front()/back()...). ABI-compatible (only
#   adds checks inside inline functions), so it is safe to mix and cheap enough
#   for every Debug/ASan config. This is what exposed the out-of-bounds
#   operator[] in ChunkAllocPolicy::allocate_chunk_internal -- UBSan cannot see
#   it, the check lives in the library, not the compiler. Do NOT confuse with
#   _GLIBCXX_DEBUG, which changes container layout and must never be mixed.
#
# CHUNK_UBSAN: -fsanitize=undefined. Catches what ASan does not: signed
#   overflow in index arithmetic, misaligned access (alignas(128) on aarch64),
#   invalid vptr in the CRTP hierarchies, out-of-range enums, null deref.
#   Combinable with ASan (same build), NOT with TSan -- guarded below.
#   -fno-sanitize-recover=all turns every report into a hard failure so a run
#   cannot exit 0 after an error (bench_and_test.py also greps the markers).
#   gcc/clang only; nvc++ has no UBSan runtime, the option is ignored there.
option(CHUNK_GLIBCXX_ASSERTIONS "Enable libstdc++ assertions (_GLIBCXX_ASSERTIONS) in lib and consumers" OFF)
option(CHUNK_UBSAN "Enable UndefinedBehaviorSanitizer in lib and consumers (gcc/clang; pair with ASan, not TSan)" OFF)

add_library(chunk_hardening INTERFACE)

if (CHUNK_GLIBCXX_ASSERTIONS)
  target_compile_definitions(chunk_hardening INTERFACE _GLIBCXX_ASSERTIONS)
  message(STATUS "CHUNK_GLIBCXX_ASSERTIONS=ON → _GLIBCXX_ASSERTIONS on lib and consumers")
endif()

if (CHUNK_UBSAN)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # TSan is selected either by the project's -DSANITIZER=thread or by a raw
    # -fsanitize=thread in the flags; refuse both.
    if ("${SANITIZER}" MATCHES "thread"
        OR "${CMAKE_CXX_FLAGS}" MATCHES "-fsanitize=[^ ]*thread"
        OR "${CMAKE_EXE_LINKER_FLAGS}" MATCHES "-fsanitize=[^ ]*thread")
      message(FATAL_ERROR
        "CHUNK_UBSAN=ON cannot be combined with ThreadSanitizer "
        "(SANITIZER=thread or -fsanitize=thread requested). "
        "UBSan pairs with ASan; TSan must stay in its own build.")
    endif()
    target_compile_options(chunk_hardening INTERFACE
            -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    target_link_options(chunk_hardening INTERFACE -fsanitize=undefined)
    message(STATUS "CHUNK_UBSAN=ON → -fsanitize=undefined -fno-sanitize-recover=all on lib and consumers")
  else()
    message(WARNING "CHUNK_UBSAN=ON ignored: no UBSan runtime for ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

# Exported alongside `chunk` so an installed consumer linking chunk::chunk
# inherits the same hardening flags (install(EXPORT chunk-targets) below).
# Frame pointers for ANY sanitizer build. -O0 keeps them anyway, but the
# RelWithDebInfo sanitizer flavours are compiled at -O2, which drops them on
# aarch64 -- and a TSan/ASan report without a usable stack is worthless. Kept
# separate from CHUNK_UBSAN (which sets it too) so a TSan build gets it as well.
if (SANITIZER AND NOT SANITIZER STREQUAL "none"
    AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  target_compile_options(chunk_hardening INTERFACE -fno-omit-frame-pointer)
endif()

# -Wno-tsan on the ThreadSanitizer builds (gcc). The compiler warns that
# `atomic_thread_fence` is not modelled by -fsanitize=thread. True, and a KNOWN
# limit of the tool, not a defect of the code: TSan cannot reason about fences,
# hence possible false positives on the library's Dekker paths. The warning
# fires on every template instantiation concerned -- dozens of ten-line blocks
# that drown the real diagnostics, which is exactly what made a build failure
# unreadable (see __report_job in scripts/functions_for_compile.sh).
if (SANITIZER MATCHES "thread" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(chunk_hardening INTERFACE -Wno-tsan)
endif()

# An element passed through `...` -- std::printf("%f", a[i]) -- is the write
# proxy, an object, not the value: GCC accepts it ("conditionally supported",
# a -Wformat warning at most) and printf then prints garbage; clang refuses
# it. An error with GCC too, for the library and every consumer (installed
# ones included): write double(a[i]).
target_compile_options(chunk_hardening INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Werror=conditionally-supported>)

install(TARGETS chunk_hardening EXPORT chunk-targets)

# --- WARNINGS (every build type) -----------------------------------------
# -Wall -Wextra on every build, for the library and the in-tree consumers
# alike (PUBLIC, BUILD_INTERFACE only: an installed consumer of chunk::chunk
# does not inherit our warning policy). Restricted to gcc/clang family: nvc++
# accepts the flags but drowns real diagnostics in noise, and the example
# CMakeLists already skip it for gcc/clang switches for that reason. Was
# Debug-only: -Wmaybe-uninitialized, -Wstringop-overflow, -Warray-bounds only
# fire from the optimisation passes, so a Debug-only policy never saw the
# warnings that matter in the build that ships. -Werror stays Debug-only
# (CI): a benign warning must not block a perf build. Opt out with
# -DCHUNK_WARNINGS=OFF; -DCHUNK_WERROR=ON promotes them to errors (Debug/CI).
option(CHUNK_WARNINGS "Enable -Wall -Wextra on every build type (gcc/clang family)" ON)
option(CHUNK_WERROR   "Treat warnings as errors on Debug builds when CHUNK_WARNINGS is on" OFF)

add_library(chunk_warnings INTERFACE)
if (CHUNK_WARNINGS)
  set(_chunk_warn_compilers "$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>")
  set(_chunk_warn_when "${_chunk_warn_compilers}")
  set(_chunk_werror_when "$<AND:$<CONFIG:Debug>,${_chunk_warn_compilers}>")
  # Without -fopenmp every `#pragma omp` (101 of them in chunk/) trips
  # -Wunknown-pragmas, which -Wall enables. The code is correct there -- the
  # loops just run serially -- so silence that one warning on non-OMP builds
  # only; OMP builds keep it, where an unknown pragma IS a typo worth seeing.
  target_compile_options(chunk_warnings INTERFACE
          "$<BUILD_INTERFACE:$<${_chunk_warn_when}:-Wall;-Wextra>>"
          "$<BUILD_INTERFACE:$<$<AND:${_chunk_warn_when},$<NOT:$<BOOL:${CHUNK_OMP}>>>:-Wno-unknown-pragmas>>"
          "$<BUILD_INTERFACE:$<$<AND:${_chunk_werror_when},$<BOOL:${CHUNK_WERROR}>>:-Werror>>")
  unset(_chunk_warn_compilers)
  unset(_chunk_warn_when)
  unset(_chunk_werror_when)
  if (CHUNK_WERROR)
    message(STATUS "CHUNK_WARNINGS=ON → -Wall -Wextra on every build, -Werror on Debug (gcc/clang family)")
  else()
    message(STATUS "CHUNK_WARNINGS=ON → -Wall -Wextra on every build (gcc/clang family)")
  endif()
endif()
install(TARGETS chunk_warnings EXPORT chunk-targets)

# --- Shared file lists ---------------------------------------------------
set(CHUNK_INTERNAL_HEADERS
        public/chunk/internal/TChunk.h
        public/chunk/internal/all_type_chunk_decl.h
        public/chunk/internal/all_type_chunk_inline.h
        public/chunk/internal/arithmetic_context.h
        public/chunk/internal/array_chunk2A_inline.h
        public/chunk/internal/array_chunk2A_parallel.h
        public/chunk/internal/array_chunk2A_parallel_inline.h
        public/chunk/internal/array_chunk2_base.h
        public/chunk/internal/array_chunk2_inline.h
        public/chunk/internal/array_chunk2_parallel.h
        public/chunk/internal/array_chunk2_parallel_inline.h
        public/chunk/internal/array_chunk2_rechunk_hooks.h
        public/chunk/internal/array_chunk2_rechunk_hooks_inline.h
        public/chunk/internal/array_chunk_base.h
        public/chunk/internal/array_chunk_base_inline.h
        public/chunk/internal/array_chunk_dirty_bitset.h
        public/chunk/internal/array_chunk_for.h
        public/chunk/internal/array_chunk_inline.h
        public/chunk/internal/array_chunk_parallel.h
        public/chunk/internal/array_chunk_parallel_common.h
        public/chunk/internal/array_chunk_parallel_common_inline.h
        public/chunk/internal/array_chunk_parallel_inline.h
        public/chunk/internal/array_chunk_rechunk.h
        public/chunk/internal/array_chunk_rechunk_hooks.h
        public/chunk/internal/array_chunk_rechunk_hooks_inline.h
        public/chunk/internal/array_chunk_rechunk_inline.h
        public/chunk/internal/bad_ptr_check.h
        public/chunk/internal/chunk2A_std_inline.h
        public/chunk/internal/chunk2_base.h
        public/chunk/internal/chunk2_base_inline.h
        public/chunk/internal/chunk2_std_inline.h
        public/chunk/internal/chunk_base.h
        public/chunk/internal/chunk_base_inline.h
        public/chunk/internal/chunk_func.h
        public/chunk/internal/chunk_size_config.h
        public/chunk/internal/chunk_size_policy.h
        public/chunk/internal/chunk_size_predictor.h
        public/chunk/internal/chunk_stats.h
        public/chunk/internal/chunk_stats_printer.h
        public/chunk/internal/chunk_std_inline.h
        public/chunk/internal/complex_less.h
        public/chunk/internal/compression_manager.h
        public/chunk/internal/compression_manager_inline.h
        public/chunk/internal/concept_array_chunk_element.h
        public/chunk/internal/concept_chunk.h
        public/chunk/internal/concept_composite_arithmetic.h
        public/chunk/internal/concept_is_clean_assignement.h
        public/chunk/internal/conditional_counter.h
        public/chunk/internal/cpu_pause.h
        public/chunk/internal/crp_helper.h
        public/chunk/internal/def_force_inline.h
        public/chunk/internal/deferred_lifecycle_check.h
        public/chunk/internal/deferred_op_pool.h
        public/chunk/internal/deferred_ops_storage.h
        public/chunk/internal/diag_mutex.h
        public/chunk/internal/for_each_type.h
        public/chunk/internal/freepath_counters.h
        public/chunk/internal/func_p2.h
        public/chunk/internal/global_check.h
        public/chunk/internal/immortal.h
        public/chunk/internal/in_t.h
        public/chunk/internal/invariant_check.h
        public/chunk/internal/link_check.h
        public/chunk/internal/live_arrays.h
        public/chunk/internal/manager_chunk.h
        public/chunk/internal/manager_chunk_inline.h
        public/chunk/internal/map_helper.h
        public/chunk/internal/nano_sleeper.h
        public/chunk/internal/phase_guard.h
        public/chunk/internal/pointer_map_chunk2.h
        public/chunk/internal/pointer_map_linear.h
        public/chunk/internal/pool_ledger.h
        public/chunk/internal/proxy.h
        public/chunk/internal/proxy2.h
        public/chunk/internal/proxy2A.h
        public/chunk/internal/proxy2A_inline.h
        public/chunk/internal/proxy2_base.h
        public/chunk/internal/proxy2_base_inline.h
        public/chunk/internal/proxy2_inline.h
        public/chunk/internal/proxy_base.h
        public/chunk/internal/proxy_base_inline.h
        public/chunk/internal/proxy_inline.h
        public/chunk/internal/proxy_operators.h
        public/chunk/internal/rechunk_cost_model.h
        public/chunk/internal/rechunk_decider_inline.h
        public/chunk/internal/tab_cst_val2_holder.h
        public/chunk/internal/write_compress_sync_policy.h
)

set(CHUNK_PUBLIC_HEADERS
        public/chunk/PMP_CHUNK.h
        public/chunk/all_type_chunk.h
        public/chunk/array_chunk.h
        public/chunk/array_chunk2.h
        public/chunk/array_chunk2A.h
        public/chunk/array_chunk_range_ops.h
        public/chunk/chunk2A_def.h
        public/chunk/chunk2A_std.h
        public/chunk/chunk2_def.h
        public/chunk/chunk2_std.h
        public/chunk/chunk_def.h
        public/chunk/chunk_size_advisor.h
        public/chunk/chunk_std.h
        public/chunk/chunk_traits.h
        public/chunk/config.h
        public/chunk/debug.h
        public/chunk/global_stats_chunk_list.h
        public/chunk/manage_ompt.h
        public/chunk/param_chunk.h
        public/chunk/rechunk_decider.h
        public/chunk/ring_log_debug.h
        public/chunk/singleton.h
        public/chunk/term_color.h
        public/chunk/type_name.h
        public/chunk/vector_range_ops.h
)

set(CHUNK_SOURCES
        private/array_chunk.cpp
        private/array_chunk2.cpp
        private/array_chunk2A.cpp
        private/manager_chunk.cpp
        private/all_type_chunk.cpp
        private/param_chunk.cpp
        private/chunk_std.cpp
        private/manage_ompt.cpp
        private/ring_log_debug.cpp
        private/link_check.cpp
)
# Optimisation: the library's .cpp files inherit the project's
# CMAKE_CXX_FLAGS_RELEASE / _RELWITHDEBINFO (-O3 and CHUNK_DEBUG_INFO's level),
# like every other target.
# If the icpx EmitCtorList ICE comes back, CHUNK_ICPX_O0_WORKAROUND=ON builds
# only the three TUs it hit at -O0, under icpx only.
option(CHUNK_ICPX_O0_WORKAROUND "icpx only: compile array_chunk2A.cpp, array_chunk2.cpp and manager_chunk.cpp at -O0 (EmitCtorList ICE)" OFF)
if (CHUNK_ICPX_O0_WORKAROUND AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set_property(SOURCE private/array_chunk2A.cpp private/array_chunk2.cpp private/manager_chunk.cpp
               APPEND PROPERTY COMPILE_OPTIONS "-O0;-fno-use-init-array")
  message(STATUS "CHUNK_ICPX_O0_WORKAROUND: array_chunk2A.cpp, array_chunk2.cpp, manager_chunk.cpp at -O0 (icpx)")
endif()

# --- Variant factory -----------------------------------------------------
# chunk_add_variant(<target> <variant_define> <size> <is_dyn>)
#   size   : compile-time chunk size (ignored/omitted for the dyn lib, which is
#            runtime-sized) — pass empty "" for dyn.
#   is_dyn : TRUE  -> add CHUNK_DYNAMIC (runtime size), no CHUNK_SIZE
#            FALSE -> add CHUNK_SIZE=<size> (compile-time fixed geometry)
#
# CHUNK_LIB_UNITY (ON): each variant archive compiled as one unity translation
# unit. Its nine sources each included the whole header-only engine:
# chunk_v2a_256 took 19.4 s of compilation in Debug and 21.3 s at -O2 -g
# apart, 3.0 s and 5.0 s as one unit -- for twelve archives per configuration.
# A source with options of its own (CHUNK_ICPX_O0_WORKAROUND's -O0,
# alloc_token_rt.cpp's -fno-sanitize) is left out of the unit by CMake and
# keeps them. What changes is the order in which the archive's globals are
# initialised: the order of the sources in the unit, where it was the
# unspecified one of separate objects.
option(CHUNK_LIB_UNITY "Compile each chunk variant archive as one unity translation unit" ON)
function(chunk_add_variant target_name variant_define size is_dyn)
  add_library(${target_name})
  if (CHUNK_LIB_UNITY)
    set_target_properties(${target_name} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 0)
  endif()

  # NO_OMPT_TOOL (extra argument): the variant is built without the OMPT tool
  # of private/manage_ompt.cpp, for a program that ships its own (the symbols
  # ompt_start_tool and g_ompt_suspended are then the program's to define).
  set(_variant_sources ${CHUNK_SOURCES})
  if ("NO_OMPT_TOOL" IN_LIST ARGN)
    list(REMOVE_ITEM _variant_sources private/manage_ompt.cpp)
  endif()
  target_sources(${target_name} PRIVATE
          ${_variant_sources}
          ${CHUNK_INTERNAL_HEADERS}
          ${CHUNK_PUBLIC_HEADERS}
  )

  if (NOT "${variant_define}" STREQUAL "")
    target_compile_definitions(${target_name} PUBLIC ${variant_define})
  endif()

  # The OMPT tool (chunk/build_options.h, CHUNK_OPT_OMPT).
  if (CHUNK_HAVE_OMPT)
    # The OpenMP runtime finds the tool with dlsym("ompt_start_tool") in the
    # EXECUTABLE: its symbols must be in the dynamic symbol table, which an
    # executable does not export by default. Without this the runtime silently
    # runs without the tool (manage_ompt.h's warning says so at start-up).
    target_link_options(${target_name} INTERFACE -rdynamic)
  endif()

  # Per-declination geometry: dynamic (runtime size) vs fixed compile-time size.
  if (is_dyn)
    target_compile_definitions(${target_name} PUBLIC CHUNK_DYNAMIC)
  elseif (NOT "${size}" STREQUAL "")
    target_compile_definitions(${target_name} PUBLIC CHUNK_SIZE=${size})
  endif()

  # CHECK_OPTIM, CHUNK_STATS, CHUNK_STD_ALLOC and the other options shared by
  # every library of the build are not definitions any more: they are in the
  # generated chunk/build_options.h, on the include path below.

  if (CHUNK_STD_ALLOC)
    # No fast_allocators library here, but some of its headers -- svalloc.h,
    # singleton.h, log_core.h, fatal.h and the threading backend (pmp_loops.h,
    # pmp_runtime.h, pmp_std.h, pmp_std_state.h), standard library (and
    # <omp.h>) only -- are chunk's too (one copy, see them): their directory
    # stays on the include path. Installed below.
    target_include_directories(${target_name}
            PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/fast_allocators/public>)
  else()
    target_link_libraries(${target_name} PUBLIC fast_allocators::fast_allocators)
  endif()

  # SANITIZER=token (umbrella CMakeLists): -fsanitize=alloc-token rewrites every
  # allocation into __alloc_token_<symbol>(..., token) and Clang ships no
  # runtime for those, so alloc_token_rt.cpp provides them -- a per-type
  # allocation census printed at exit. It goes INTO each variant archive rather
  # than into a library of its own so the linker meets the definitions in the
  # first archive it scans, whatever the order of the ones that follow.
  # Compiled uninstrumented: instrumented, its own malloc calls would recurse.
  if (SANITIZER STREQUAL "token")
    target_sources(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private/alloc_token_rt.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/private/alloc_token_rt.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-sanitize=alloc-token")
  endif()

  if (CHUNK_NO_IPA_ICF AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target_name} PUBLIC -fno-ipa-icf)
  endif()

  # Hardening (libstdc++ assertions, UBSan) and Debug warnings: PUBLIC so
  # consumers compile and link with the same flags as the library -- see the
  # HARDENING / WARNINGS toggles above.
  target_link_libraries(${target_name} PUBLIC chunk_hardening chunk_warnings)
  
  # PUBLIC: the headers are C++20; a consumer gets the standard it needs by
  # linking the library.
  chunk_require_cxx20(${target_name} PUBLIC)
  # PUBLIC: the headers branch on _OPENMP (build_config::use_openmp), so a
  # consumer must be compiled with OpenMP exactly when the library was.
  if (CHUNK_OMP)
    target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
  endif ()
  # PUBLIC, for the same reason: the std backend is selected in the headers
  # (fast_allocators/internal/pmp_runtime.h) by this definition, which
  # build_options.h checks against the library's.
  if (CHUNK_OPT_STD_THREADS)
    target_compile_definitions(${target_name} PUBLIC FAST_ALLOC_PMP_STD=1)
  endif ()
  # The library and its headers use threads (std::thread, std::call_once:
  # exact_rss.h's RSS sampler, the flush waits, the ring log), OpenMP or not.
  # On a glibc older than 2.34 (RHEL 8) pthread_create is in libpthread,
  # which only OpenMP's runtime brought in: a serial icpx build of
  # shrink_bench failed to link. PUBLIC, so every consumer links it.
  target_link_libraries(${target_name} PUBLIC Threads::Threads)

  target_include_directories(${target_name}
          PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
          $<BUILD_INTERFACE:${CHUNK_GENERATED_INCLUDE_DIR}>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party>
          $<INSTALL_INTERFACE:include>)

  set_target_properties(${target_name} PROPERTIES
          VERSION ${CHUNK_VERSION}
          SOVERSION ${CHUNK_MAJOR_VERSION}
  )
endfunction()

# --- Instantiate variants × sizes ---------------------------------------
# chunk_vector has no chunk geometry (plain std::vector), so it is size-agnostic
# and built once.
chunk_add_variant(chunk_vector "" "" FALSE)

# The three shrink variants, each in every static size + an optional dyn lib.
# Naming: chunk_v1_64, chunk_v1_256, ..., chunk_v2a_dyn.
# Only those of CHUNK_VARIANTS.
set(_chunk_variant_names   "")
set(_chunk_variant_defines "")
foreach(_pair "chunk:chunk_v1:CHUNK_VARIANT_V1=1" "chunk2:chunk_v2:CHUNK_VARIANT_V2=1"
              "chunk2a:chunk_v2a:CHUNK_VARIANT_V2A=1")
  string(REPLACE ":" ";" _f "${_pair}")
  list(GET _f 0 _short)
  if (_short IN_LIST CHUNK_VARIANTS)
    list(GET _f 1 _lib)
    list(GET _f 2 _def)
    list(APPEND _chunk_variant_names ${_lib})
    list(APPEND _chunk_variant_defines ${_def})
  endif()
endforeach()

list(LENGTH _chunk_variant_names _nvar)
math(EXPR _nvar_last "${_nvar} - 1")

foreach(_vi RANGE ${_nvar_last})
  list(GET _chunk_variant_names   ${_vi} _vname)
  list(GET _chunk_variant_defines ${_vi} _vdef)

  # Static compile-time sizes.
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    chunk_add_variant(${_vname}_${_sz} "${_vdef}" ${_sz} FALSE)
  endforeach()

  # Dynamic (runtime size) lib.
  if (CHUNK_BUILD_DYN)
    chunk_add_variant(${_vname}_dyn "${_vdef}" "" TRUE)
  endif()
endforeach()

# --- Backwards-compatible bare-name targets ------------------------------
# The examples link the bare names chunk_v1 / chunk_v2 / chunk_v2a; forward
# them to the default static size. NOTE: a raw (unknown) name given to
# target_link_libraries is silently treated by CMake as a system library
# (-lchunk_v1), so no usage requirement (-I, PUBLIC defines) propagates and
# the failure only shows up at compile time — hence these explicit wrappers.
# INTERFACE wrappers (not ALIASes) so usage requirements propagate reliably
# across directory scopes.
list(GET CHUNK_STATIC_SIZES 0 _chunk_first_size)
set(CHUNK_DEFAULT_SIZE ${_chunk_first_size} CACHE STRING
    "Static size bound to the bare chunk_v<x> targets.")
message(STATUS "CHUNK_DEFAULT_SIZE = ${CHUNK_DEFAULT_SIZE} (bare chunk_v<x> targets)")

foreach(_vname IN LISTS _chunk_variant_names)
  if (NOT TARGET ${_vname}_${CHUNK_DEFAULT_SIZE})
    message(FATAL_ERROR
        "CHUNK_DEFAULT_SIZE=${CHUNK_DEFAULT_SIZE} is not in CHUNK_STATIC_SIZES=${CHUNK_STATIC_SIZES}")
  endif()
  add_library(${_vname} INTERFACE)
  target_link_libraries(${_vname} INTERFACE ${_vname}_${CHUNK_DEFAULT_SIZE})
endforeach()

# --- Two families in one program ------------------------------------------
# examples/multi-mat keeps its field arrays in ArrayChunk<T> (v1) and its
# presence fraction in ArrayChunk2<T>: one library with both defines, at the
# default static size only. The registry allows it (all_type_chunk_inline.h:
# "nothing forbids a build from selecting more than one"); what the second
# define costs v1 is the raw fill of its range ops (raw_fillable_v). Without
# the OMPT tool: the example carries its own (examples/multi-mat/manage_ompt.cpp).
if ("chunk" IN_LIST CHUNK_VARIANTS AND "chunk2" IN_LIST CHUNK_VARIANTS)
  chunk_add_variant(chunk_v1v2_${CHUNK_DEFAULT_SIZE} "CHUNK_VARIANT_V1=1;CHUNK_VARIANT_V2=1" ${CHUNK_DEFAULT_SIZE} FALSE
                    NO_OMPT_TOOL)
endif()

# A REAL default library (not just an ALIAS), so the include directories reach the executables.
# Default library: no variant define, default compile-time size (static_chunk_size=64), static.
chunk_add_variant(chunk "" "" FALSE)

# --- Install: every variant library, under the chunk:: namespace ---------
# chunk::chunk_v2a_256, chunk::chunk_v2a_dyn, ..., the bare chunk::chunk_v1 /
# chunk::chunk_v2 / chunk::chunk_v2a (default static size), chunk::chunk_vector
# (std::vector mode) and chunk::chunk. Not chunk_v1v2_*: built for
# examples/multi-mat, without the OMPT tool.
include(GNUInstallDirs)
set(_chunk_install_targets chunk chunk_vector)
foreach(_vname IN LISTS _chunk_variant_names)
  foreach(_sz IN LISTS CHUNK_STATIC_SIZES)
    list(APPEND _chunk_install_targets ${_vname}_${_sz})
  endforeach()
  if (CHUNK_BUILD_DYN)
    list(APPEND _chunk_install_targets ${_vname}_dyn)
  endif()
  list(APPEND _chunk_install_targets ${_vname})
endforeach()

# The installed names in the build tree too: a project that takes this one
# through add_subdirectory() or FetchContent links chunk::chunk_v2a, as it
# would after find_package(chunk).
foreach(_t IN LISTS _chunk_install_targets)
  if (NOT TARGET chunk::${_t})
    add_library(chunk::${_t} ALIAS ${_t})
  endif()
endforeach()

# The fast_allocators version chunk-config.cmake asks for (find_dependency).
# fast_allocators/CMakeLists.txt, configured after this file, checks it against
# its own ALLOCATORS_VERSION through this global property.
set(CHUNK_FAST_ALLOCATORS_VERSION 2.2.0)
set_property(GLOBAL PROPERTY CHUNK_FAST_ALLOCATORS_VERSION ${CHUNK_FAST_ALLOCATORS_VERSION})

install(TARGETS ${_chunk_install_targets}
        EXPORT chunk-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT chunk-runtime
        NAMELINK_COMPONENT chunk-devel
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT chunk-devel)

# The whole public/ tree, not CHUNK_PUBLIC_HEADERS alone: the facades in that
# list include the CHUNK_INTERNAL_HEADERS (a template library ships its
# implementation headers), so an install of the 40 facades could not compile a
# single consumer. The two lists remain the statement of which headers are the
# API and which are implementation detail.
install(DIRECTORY public/chunk
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT chunk-devel
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CHUNK_GENERATED_INCLUDE_DIR}/chunk/build_options.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chunk
        COMPONENT chunk-devel)

# A std-alloc build installs no fast_allocators, yet chunk's headers include
# its svalloc.h, singleton.h, log_core.h, fatal.h and the backend's pmp_*.h:
# install those with chunk then.
if (CHUNK_STD_ALLOC)
  install(FILES ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/svalloc.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/singleton.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/log_core.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/fatal.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/pmp_loops.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/pmp_runtime.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/pmp_std.h
                ${PROJECT_SOURCE_DIR}/fast_allocators/public/fast_allocators/internal/pmp_std_state.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fast_allocators/internal
          COMPONENT chunk-devel)
endif()

# And the vendored third-party headers those include as "chunk/vendor/..."
# (ankerl's unordered_dense.h, skarupke's flat_hash_map.hpp): under
# include/chunk/vendor/, never at the root of include/, where they would
# overwrite, or be shadowed by, a user's own copy of the same library.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/chunk/vendor
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chunk
        COMPONENT chunk-devel
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# The licenses: the library's, the NOTICE listing the third-party code, and
# that code's license texts (third_party/README.md). With the runtime, which
# carries code compiled from those headers.
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/NOTICE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/chunk
        COMPONENT chunk-runtime)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/licenses/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/chunk/third_party
        COMPONENT chunk-runtime)

include(CMakePackageConfigHelpers)

configure_package_config_file(chunk-config.cmake.in chunk-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake)

# 0.x (SemVer): a MINOR release may break, so only the same 0.MINOR is
# compatible; from 1.0, the same MAJOR (CHANGELOG.md, "Versioning policy").
if (PROJECT_VERSION_MAJOR EQUAL 0)
  set(_chunk_compatibility SameMinorVersion)
else()
  set(_chunk_compatibility SameMajorVersion)
endif()
write_basic_package_version_file(
        chunk-config-version.cmake
        VERSION ${CHUNK_VERSION}
        COMPATIBILITY ${_chunk_compatibility})

install(EXPORT chunk-targets NAMESPACE chunk:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/chunk-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/chunk-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake
        COMPONENT chunk-devel
)
