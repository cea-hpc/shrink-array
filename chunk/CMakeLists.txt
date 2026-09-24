# ---------------------------------------------------------------------------
# The `chunk` library must be compiled with one of CHUNK / CHUNK2 / CHUNK2A
# defined (or none, for the vector-mode build). These macros drive
# `chunk_config::is_shrink_enabled` and friends in public/chunk/config.h, and
# they are read *inside* library-side .cpp files (AllTypeChunk.cpp,
# manage_ompt.cpp today; potentially others tomorrow).
# ---------------------------------------------------------------------------

set(CHUNK_MAJOR_VERSION 2)  # inc. when incompatible change (reset to 0 others)
set(CHUNK_MINOR_VERSION 2)  # inc. when compatible change (also reset patch vers.)
set(CHUNK_PATCH_VERSION 8)  # inc. when just fixing bug

set(CHUNK_VERSION
        ${CHUNK_MAJOR_VERSION}.${CHUNK_MINOR_VERSION}.${CHUNK_PATCH_VERSION})

if (CHUNK_OMP)
  find_package(OpenMP REQUIRED)
endif ()

option(CHUNK_STD_ALLOC "Use standard allocator for chunk." OFF)

# --- Multi-size declinations --------------------------------------------
# The compile-time chunk size (__chunk_size) is frozen per library by
# -DCHUNK_SIZE. To benchmark several fixed sizes we therefore build ONE lib
# per (variant, size). CHUNK_STATIC_SIZES is the list of static sizes; each
# yields chunk_v<x>_<size>. CHUNK_BUILD_DYN adds a runtime-size (CHUNK_DYNAMIC)
# lib chunk_v<x>_dyn per variant. Trim CHUNK_STATIC_SIZES during development to
# cut build time (e.g. -DCHUNK_STATIC_SIZES=256).
set(CHUNK_STATIC_SIZES "256;1024" CACHE STRING
    "Semicolon list of compile-time chunk sizes to build (powers of two).")
option(CHUNK_BUILD_DYN "Also build a CHUNK_DYNAMIC (runtime size) lib per variant." ON)
message(STATUS "CHUNK_STATIC_SIZES = ${CHUNK_STATIC_SIZES}; CHUNK_BUILD_DYN = ${CHUNK_BUILD_DYN}")

# --- Aggressive vs release compression scheduling ------------------------
option(HAVE_AGRESSIVE_COMPRESSION "Enable aggressive-compression scheduling mode." OFF)
if (HAVE_AGRESSIVE_COMPRESSION)
  message(STATUS "HAVE_AGRESSIVE_COMPRESSION=ON → aggressive scheduling (historical behavior)")
endif()

# --- Read-side deferred-op sync (A/B experiment) -------------------------
# OFF (default) is byte-identical to the historical build: the non-const proxy
# operator[] flushes pending ops before every write. ON moves that toll to the
# read entry points (see build_config::sync_on_read in config.h). This is now the
# permanent behaviour — always on, no toggle. The bool is hard-coded in config.h;
# there is nothing to define here for it.
# P3 backlog bound. Empty = use the header default (4096).
set(CHUNK_DEFERRED_DRAIN_THRESHOLD "" CACHE STRING
    "Own-rank deferred-backlog drain threshold (empty = header default 4096).")
if (NOT "${CHUNK_DEFERRED_DRAIN_THRESHOLD}" STREQUAL "")
  message(STATUS "CHUNK_DEFERRED_DRAIN_THRESHOLD = ${CHUNK_DEFERRED_DRAIN_THRESHOLD}")
endif()

# Test knob: force one write in N down the deferred path (enqueue / replay /
# P3 / read gate) with no compressor in flight; 0 = off. A build with, say, 7
# must pass the shrink_bench file cross-check bit for bit. See config.h.
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

# -Wno-tsan sur les builds ThreadSanitizer (gcc). Le compilateur avertit que
# `atomic_thread_fence` n'est pas modelise par -fsanitize=thread. C'est exact et
# c'est une limite CONNUE de l'outil, pas un defaut du code : TSan ne sait pas
# raisonner sur les barrieres, d'ou d'eventuels faux positifs sur les chemins
# Dekker de la bibliotheque. L'avertissement se declenche sur chaque
# instanciation de template concernee, soit des dizaines de blocs de dix lignes
# qui noient les vrais diagnostics -- c'est precisement ce qui a rendu un echec
# de build illisible (voir __report_job dans scripts/functions_for_compile.sh).
if (SANITIZER MATCHES "thread" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(chunk_hardening INTERFACE -Wno-tsan)
endif()

install(TARGETS chunk_hardening EXPORT chunk-targets)

# --- WARNINGS (Debug only) -----------------------------------------------
# -Wall -Wextra on every Debug build, for the library and the in-tree
# consumers alike (PUBLIC, BUILD_INTERFACE only: an installed consumer of
# chunk::chunk does not inherit our warning policy). Restricted to gcc/clang
# family: nvc++ accepts the flags but drowns real diagnostics in noise, and the
# example CMakeLists already skip it for gcc/clang switches for that reason.
# Release/RelWithDebInfo untouched. Opt out with -DCHUNK_WARNINGS=OFF;
# -DCHUNK_WERROR=ON promotes them to errors (CI).
option(CHUNK_WARNINGS "Enable -Wall -Wextra on Debug builds (gcc/clang family)" ON)
option(CHUNK_WERROR   "Treat warnings as errors when CHUNK_WARNINGS is on" OFF)

add_library(chunk_warnings INTERFACE)
if (CHUNK_WARNINGS)
  set(_chunk_warn_compilers "$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>")
  set(_chunk_warn_when "$<AND:$<CONFIG:Debug>,${_chunk_warn_compilers}>")
  # Without -fopenmp every `#pragma omp` (101 of them in chunk/) trips
  # -Wunknown-pragmas, which -Wall enables. The code is correct there -- the
  # loops just run serially -- so silence that one warning on non-OMP builds
  # only; OMP builds keep it, where an unknown pragma IS a typo worth seeing.
  target_compile_options(chunk_warnings INTERFACE
          "$<BUILD_INTERFACE:$<${_chunk_warn_when}:-Wall;-Wextra>>"
          "$<BUILD_INTERFACE:$<$<AND:${_chunk_warn_when},$<NOT:$<BOOL:${CHUNK_OMP}>>>:-Wno-unknown-pragmas>>"
          "$<BUILD_INTERFACE:$<$<AND:${_chunk_warn_when},$<BOOL:${CHUNK_WERROR}>>:-Werror>>")
  unset(_chunk_warn_compilers)
  unset(_chunk_warn_when)
  if (CHUNK_WERROR)
    message(STATUS "CHUNK_WARNINGS=ON → -Wall -Wextra -Werror on Debug builds (gcc/clang family)")
  else()
    message(STATUS "CHUNK_WARNINGS=ON → -Wall -Wextra on Debug builds (gcc/clang family)")
  endif()
endif()
install(TARGETS chunk_warnings EXPORT chunk-targets)

# --- Shared file lists ---------------------------------------------------
set(CHUNK_INTERNAL_HEADERS
        public/chunk/internal/TChunk.h
        public/chunk/internal/all_type_chunk_inline.h
        public/chunk/internal/arithmetic_context.h
        public/chunk/internal/array_chunk2A_inline.h
        public/chunk/internal/array_chunk2A_parallel.h
        public/chunk/internal/array_chunk2A_parallel_inline.h
        public/chunk/internal/array_chunk2A_rechunk_hooks.h
        public/chunk/internal/array_chunk2A_rechunk_hooks_inline.h
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
        public/chunk/internal/common_types.h
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
        public/chunk/internal/define_mm_protect.h
        public/chunk/internal/for_each_type.h
        public/chunk/internal/freepath_counters.h
        public/chunk/internal/func_p2.h
        public/chunk/internal/global_check.h
        public/chunk/internal/in_t.h
        public/chunk/internal/invariant_check.h
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
        public/chunk/internal/svalloc.h
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
        public/chunk/chunk_define.h
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
)
# Optimisation: the library's .cpp files inherit the project's
# CMAKE_CXX_FLAGS_RELEASE / _RELWITHDEBINFO (-O3 -g), like every other target.
# (Until Sept '26 an icpx 2026.0 ICE workaround stripped -O3 here and re-added
# it per source under names that matched no source file -- leaving every
# chunk/private/*.cpp at -O0 in optimised builds, on every compiler.)
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
function(chunk_add_variant target_name variant_define size is_dyn)
  add_library(${target_name})

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

  # Per-declination geometry: dynamic (runtime size) vs fixed compile-time size.
  if (is_dyn)
    target_compile_definitions(${target_name} PUBLIC CHUNK_DYNAMIC)
  elseif (NOT "${size}" STREQUAL "")
    target_compile_definitions(${target_name} PUBLIC CHUNK_SIZE=${size})
  endif()

  target_compile_definitions(${target_name} PUBLIC
          $<$<BOOL:${CHECK_OPTIM}>:CHECK_OPTIM>
          $<$<BOOL:${CHUNK_POOL_NO_REUSE}>:CHUNK_POOL_NO_REUSE>
          $<$<BOOL:${CHUNK_ALIGN_CHECK}>:CHUNK_ALIGN_CHECK>
          $<$<BOOL:${HAVE_AGRESSIVE_COMPRESSION}>:HAVE_AGRESSIVE_COMPRESSION>
          $<$<BOOL:${CHUNK_STATS}>:CHUNK_STATS>
          $<$<BOOL:${CHUNK_FORCE_LEDGER}>:CHUNK_FORCE_LEDGER>
  )

  # Read-side sync backlog threshold (only when explicitly set).
  if (NOT "${CHUNK_DEFERRED_DRAIN_THRESHOLD}" STREQUAL "")
    target_compile_definitions(${target_name} PUBLIC
            CHUNK_DEFERRED_DRAIN_THRESHOLD=${CHUNK_DEFERRED_DRAIN_THRESHOLD})
  endif()
  if (NOT "${CHUNK_FORCE_DEFER_EVERY}" STREQUAL "0")
    target_compile_definitions(${target_name} PUBLIC
            CHUNK_FORCE_DEFER_EVERY=${CHUNK_FORCE_DEFER_EVERY})
  endif()

  if (CHUNK_STD_ALLOC)
    target_compile_definitions(${target_name} PUBLIC CHUNK_STD_ALLOC)
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
  
  chunk_require_cxx20(${target_name} PRIVATE)
  if (CHUNK_OMP)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif ()

  target_include_directories(${target_name}
          PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
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
set(_chunk_variant_names   chunk_v1 chunk_v2 chunk_v2a)
set(_chunk_variant_defines "CHUNK=1" "CHUNK2=1" "CHUNK2A=1")

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
chunk_add_variant(chunk_v1v2_${CHUNK_DEFAULT_SIZE} "CHUNK=1;CHUNK2=1" ${CHUNK_DEFAULT_SIZE} FALSE NO_OMPT_TOOL)

# VRAIE LIBRAIRIE PAR DÉFAUT (au lieu d'un simple ALIAS) pour éviter la perte des includes dans les sous-exécutables
# Default library: no variant define, default compile-time size (__chunk_size=64), static.
chunk_add_variant(chunk "" "" FALSE)

# --- Install (vector variant only, as the canonical installed copy) -----
include(GNUInstallDirs)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/chunk)

install(TARGETS chunk
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

# And the vendored third-party headers those include (ankerl/unordered_dense.h
# from pool_ledger.h and four others): they are on the build-tree include path
# only, so the installed tree was missing them.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT chunk-devel
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

include(CMakePackageConfigHelpers)

configure_package_config_file(chunk-config.cmake.in chunk-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake
        PATH_VARS CHUNK_VERSION INCLUDE_INSTALL_DIR)

write_basic_package_version_file(
        chunk-config-version.cmake
        VERSION ${CHUNK_VERSION}
        COMPATIBILITY SameMajorVersion)

install(EXPORT chunk-targets NAMESPACE chunk:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/chunk-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/chunk-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/chunk/cmake
        COMPONENT chunk-devel
)
