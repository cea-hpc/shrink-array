# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(shrink_array VERSION 0.1.2 LANGUAGES CXX)

# Built on its own, or consumed by another project (add_subdirectory,
# FetchContent)? PROJECT_IS_TOP_LEVEL says it from CMake 3.21; this is the same
# test for 3.20. A consumer gets the libraries only: no examples, no tests, and
# nothing written into its cache (build type, per-configuration flags).
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(SHRINK_ARRAY_IS_TOP_LEVEL ON)
else()
    set(SHRINK_ARRAY_IS_TOP_LEVEL OFF)
endif()
option(SHRINK_ARRAY_BUILD_EXAMPLES "Build the examples, benchmarks and validation harnesses" ${SHRINK_ARRAY_IS_TOP_LEVEL})
option(SHRINK_ARRAY_BUILD_TESTS "Register the tests (include(CTest))" ${SHRINK_ARRAY_IS_TOP_LEVEL})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CHECK_OPTIM "Enable runtime invariant checks in release builds" OFF)
option(CHUNK_FORCE_LEDGER "Keep the pool ledger active under OpenMP (off by default there: its global mutex serialises every alloc/free)" OFF)

# An emptied slab would otherwise keep its resident pages: the allocator never
# unmaps a value slab (see the op/value aliasing note in tc_allocator.h), so the
# peak of a re-chunk -- where the old and the new geometry are both materialised
# -- would stay in RSS afterwards. This madvise()s those pages away while
# keeping the slab mapped and reusable.
#
# ON by default since it was measured to be free. rechunk_validation_chunk_dyn
# 200000 128 3, peak RSS, best of 3: 129.9 -> 88.3 MB at one thread and
# 104.8 -> 87.9 MB at four, with the wall time unchanged (16.79 -> 16.60 s and
# 16.22 -> 16.06 s). The syscall cost does not register: 99 madvise calls over
# the whole run, 29 microseconds in total, because a slab only empties at a
# geometry change or at teardown -- never in steady state.
#
# It is safe by construction rather than by care: the only call site is
# TCTypeCAllocator::reclaim_empty_slabs_now(), the VALUE family, on a set the
# caller knows abandoned (a re-chunk's old geometry, a teardown). Its free list
# lives in the heap-allocated tab_firstFree and its metadata in the slab header
# object -- both outside the pages being dropped -- and it makes no promise
# about element contents (callers initialise what they get), so the zero pages
# MADV_DONTNEED brings back are as good as any. The op/ctx family
# threads its free list THROUGH the freed elements and would be corrupted by
# this; its on_kept is an empty lambda and it is never madvised.
#
# Turn it OFF if a profile ever shows page faults from slabs that empty and
# refill in a loop; "strace -c -e trace=madvise" gives the call count directly.
#
# Recorded, like every option the headers read, in the generated
# chunk/build_options.h and fast_allocators/build_options.h, installed with the
# libraries: a project using the installed package compiles the inline
# allocator code with the library's value, whatever it defines itself.
option(CHUNK_RECLAIM_EMPTY_SLABS "Release an emptied slab's pages with madvise instead of keeping them resident" ON)

# Reinforced debug level for fast_allocators' slab lists: on a Debug build,
# every insert / remove also walks the whole list (sl_check, sl_contains) --
# linear per operation, quadratic over a run, so off unless asked for. A plain
# Debug build keeps the cheap assertions. See alloc_config::check_slab_lists.
option(FAST_ALLOC_CHECK_LISTS "Debug builds: check every slab list in full on each insert / remove (slow)" OFF)

# The write/compress handshake (chunk_config::asymmetric_dekker, config.h) is
# asymmetric wherever it can be, with no option: the write announces itself
# with a plain store, the compressor / flush side pays a membarrier() for
# everyone. The symmetric shape (a seq_cst store + fence on every element
# write) remains for TSan builds, non-Linux, and any run whose membarrier
# registration fails; config.h has the numbers behind the choice.

# Chunk2A's per-chunk second-value table holds a 4-byte identifier instead of
# the value for types wider than 4 bytes (double, long, complex<double>):
# 4 bytes per chunk instead of 8 or 16, -3.8% RSS on the 50 M-chunk reference
# run, for one more dependent load on every write of a materialised chunk --
# +3% CPU on the perf bench. Off by default; see SecondValueTraits in
# chunk/internal/tab_cst_val2_holder.h.
option(CHUNK_SECOND_VALUE_INDEXED "Chunk2A: store second values as 4-byte identifiers (less memory, +3% write CPU)" OFF)
# CHECK_OPTIM (option above) turns build_config::any_check on in a release
# build (chunk/build_options.h).

# CHUNK_MM_PROTECT: Chunk/Chunk2's shared constant buffers (cst_val1, the cst_val2
# region) are mprotect'ed read-only, so a stray write into a compressed chunk
# faults at the culprit instead of corrupting a value shared by thousands of
# chunks. A debugging build: one mprotect per buffer, and page-rounded slots.
# Its code sits in `if constexpr (build_config::use_mm_protect)` branches of
# class templates, which no other configuration instantiates -- compile_all.sh
# builds it so they keep compiling.
option(CHUNK_MM_PROTECT "Read-only (mprotect) constant buffers: a stray write to a compressed chunk faults" OFF)

# CHUNK_STATS: compression statistics (chunk_stats.h), printed by ParamChunk::ChunkStats().
option(CHUNK_STATS "Collect compression statistics (ParamChunk::ChunkStats())" OFF)

# ── Sanitizer option ──────────────────────────────────────────────────────────
set(SANITIZER "none" CACHE STRING "Sanitizer to enable: none, thread, address, token")
set_property(CACHE SANITIZER PROPERTY STRINGS none thread address token)

message(STATUS "Sanitizer: ${SANITIZER}")

# ── OpenMP ────────────────────────────────────────────────────────────────────
# Read by every target of the tree; the scripts pass -DOMP=ON. OFF (as an
# undeclared variable read) builds the serial library.
option(OMP "Build with OpenMP (the parallel library and examples)" OFF)
if(OMP)
    find_package(OpenMP 4.5 REQUIRED)
    set(CMAKE_LIBRARY_PATH /usr/local/lib/x86_64-unknown-linux-gnu ${CMAKE_LIBRARY_PATH})
endif()

# ── Options ───────────────────────────────────────────────────────────────────

set(CHUNK_OMP ${OMP})

if(SHRINK_ARRAY_BUILD_TESTS)
    include(CTest)
else()
    set(BUILD_TESTING OFF) # this subtree only: the parent's own setting is left alone
endif()

# ── Build type ────────────────────────────────────────────────────────────────
# Top level only: a consumer's build type is its own.
if(SHRINK_ARRAY_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# ── OMPT detection (Clang/IntelLLVM only) ────────────────────────────────────
if(OMP)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # TSan is incompatible with OMPT instrumentation
        if(NOT SANITIZER STREQUAL "thread")
            # Recorded in chunk/build_options.h (CHUNK_OPT_OMPT); also read
            # for the targets that exist only with the OMPT tool
            # (examples/multi-mat's embedded-compression executable).
            set(CHUNK_HAVE_OMPT ON)
        endif()
        message(STATUS "Clang/IntelLLVM compiler detected")
    else()
        message(STATUS "Non-Clang compiler detected")
    endif()
endif()

# ── Compiler-specific flags ───────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
    add_compile_options(-Kieee)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    option(USE_LIBCXX "Use Clang libc++ instead of libstdc++" ON)

    if(USE_LIBCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
    else()
        # The CEA default only where it exists: on a workstation an empty
        # GCC_TOOLCHAIN_DIR leaves clang its own libstdc++ (the system one).
        if(EXISTS "/ccc/products/gcc-15.2.0")
            set(_chunk_default_gcc_toolchain "/ccc/products/gcc-15.2.0")
        else()
            set(_chunk_default_gcc_toolchain "")
        endif()
        set(GCC_TOOLCHAIN_DIR "${_chunk_default_gcc_toolchain}" CACHE PATH "Path to GCC toolchain (empty: the driver's default)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
        if(GCC_TOOLCHAIN_DIR AND NOT EXISTS "${GCC_TOOLCHAIN_DIR}")
            # A cache made before this check (or copied from the CEA) would
            # otherwise hide every standard header.
            message(WARNING "GCC_TOOLCHAIN_DIR=${GCC_TOOLCHAIN_DIR} does not exist: using the driver's default")
        elseif(GCC_TOOLCHAIN_DIR)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_TOOLCHAIN_DIR}")
        endif()
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
endif()

# ── Sanitizer flags ───────────────────────────────────────────────────────────
if(SANITIZER STREQUAL "thread")
    message(STATUS "Thread Sanitizer enabled")
    set(SANITIZER_FLAGS "-fsanitize=thread")
    # TSan requires position-independent code
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")

elseif(SANITIZER STREQUAL "address")
    message(STATUS "Address Sanitizer enabled")
    set(SANITIZER_FLAGS "-fsanitize=address")

elseif(SANITIZER STREQUAL "token")
    # AllocToken (Clang >= 22): every malloc / calloc / realloc / operator new
    # is rewritten into __alloc_token_<name>(..., token), the token being a
    # hash of the allocated TYPE. It is instrumentation only: neither
    # compiler-rt nor the packaged Scudo ships the __alloc_token_* entry
    # points, so the program has to provide them or the link fails.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 22)
        message(FATAL_ERROR "SANITIZER=token needs Clang >= 22 (-fsanitize=alloc-token); "
                            "got ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    message(STATUS "Alloc Token instrumentation enabled")
    set(SANITIZER_FLAGS "-fsanitize=alloc-token")

elseif(NOT SANITIZER STREQUAL "none")
    message(FATAL_ERROR "Unknown sanitizer '${SANITIZER}'. Valid values: none, thread, address, token")
endif()

# Apply the sanitizer flags for EVERY build type.
#
# Not only into CMAKE_CXX_FLAGS_DEBUG below, which CMake reads only when
# CMAKE_BUILD_TYPE=Debug: a RelWithDebInfo build with -DSANITIZER=address would
# then configure cleanly, print "Address Sanitizer enabled", and produce an
# UNinstrumented binary (`nm -C <exe> | grep -c __asan` returns 1 instead of
# thousands).
# Compile AND link flags are set here, together: TSan's linker flags were
# commented out, so a thread build could compile instrumented yet miss its
# runtime.
#
# -fno-omit-frame-pointer: at -O0 the compiler keeps frame pointers anyway, but
# the optimised sanitizer flavours are -O3 and drop them on aarch64, which makes
# ASan/TSan stack traces unusable.
if(SANITIZER_FLAGS)
    set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS} -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
endif()

# ── GPerftools option ─────────────────────────────────────────────────────────
option(USE_GPERF "Enable gperftools CPU profiling" OFF)

if(USE_GPERF)
    message(STATUS "GPerftools profiling enabled")
    # The 'profiler' library
    find_library(GPERF_PROFILER profiler)
    if(GPERF_PROFILER)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GPERF_PROFILER}")
    else()
        message(FATAL_ERROR "gperftools profiler library not found")
    endif()
endif()

# ── Build type flags ──────────────────────────────────────────────────────────
# NOTE: no ${SANITIZER_FLAGS} here -- they are applied above, for every build
# type. Putting them in this Debug-only variable is what silently produced
# uninstrumented RelWithDebInfo sanitizer binaries.
# The library's debug checks (build_config::is_debug, the slab-list assertions):
# on in a Debug build, recorded in the generated build_options.h headers rather
# than passed as -DDEBUG -- a generic macro every program may define its own
# way. Deliberately not an option(): the default follows the build type of THIS
# configure; -DCHUNK_DEBUG=ON / OFF forces it.
if(NOT DEFINED CHUNK_DEBUG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CHUNK_DEBUG ON)
    else()
        set(CHUNK_DEBUG OFF)
    endif()
endif()

# Top level: in the cache, as before. Consumed by another project: plain
# variables of this subtree -- the cache is the consumer's.
if(SHRINK_ARRAY_IS_TOP_LEVEL)
    set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -g"
        CACHE STRING "Debug flags" FORCE)

    set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -g"
        CACHE STRING "Release flags" FORCE)

    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "-O3 -g"
        CACHE STRING "RelWithDebInfo flags" FORCE)
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
endif()

# ── Runtime RPATH: embed the compiler's own runtime libraries ─────────────────
# Binaries built with a module compiler (llvm/20.1.8, gcc/15.2.0) depend on that
# toolchain's runtime libs — a newer libstdc++.so.6 / libgomp.so.1 / libomp.so —
# which live under the module's lib dir, NOT in the system linker path. At
# runtime, without the module loaded, the loader fails with e.g.
#   "libstdc++.so.6: version GLIBCXX_3.4.32 not found"  or  "libomp.so: not found".
# We bake the compiler's runtime lib dir(s) into each executable's RUNPATH via
# -Wl,-rpath, so binaries are self-contained regardless of loaded modules.
# Works for gnu and clang because the path is derived from the *actual* compiler.
set(CMAKE_SKIP_BUILD_RPATH            FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH    FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(_rt_probe_libs "")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # This project links clang with -stdlib=libstdc++ (--gcc-toolchain=...) when
    # USE_LIBCXX=OFF, so the runtime C++ lib is gcc's libstdc++. libc++ names are
    # probed harmlessly in case a libc++ build is ever re-enabled.
    list(APPEND _rt_probe_libs libstdc++.so.6 libomp.so
                               libc++.so.1 libc++abi.so.1 libunwind.so.1)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND _rt_probe_libs libstdc++.so.6 libgomp.so.1 libgcc_s.so.1)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
	#add_compile_options($<$<CONFIG:Release,RelWithDebInfo>:-fno-vectorize>)
    list(APPEND _rt_probe_libs libstdc++.so.6 libiomp5.so libsvml.so)
endif ()
include(CheckCXXCompilerFlag)

# icpx and a non-Intel CPU.
#
#   "This executable is built with target processor feature specified, it can
#    only be run on Intel(R) processors."
#
# is printed by __intel_cpu_features_init_x, from Intel's runtime helper
# library, before main -- which is what an AMD node answers, after the OMPT
# callbacks have already printed their lines. -x<target> is what asks for that
# vendor dispatch, so the default here is -march=native: the clang spelling
# icpx also takes, same instruction set for the machine it is built on, no
# -x. ONEAPI_XHOST=ON restores -xHost for a binary that will only ever run on
# the Intel machine that built it.
#
# If the message persists with -march=native (the feature requirement is
# recorded either way; only the run on the AMD machine can tell), the last
# resort is ONEAPI_NO_INTEL_LIBS=ON: it links NO Intel runtime library at all.
# That is what it takes -- dropping libirc alone does not link, since libsvml
# itself needs it (undefined __intel_cpu_features_init_x) -- and the price is
# the vectorised math: -fveclib=none, so sin/cos go through libm one element
# at a time.
option(ONEAPI_XHOST "icpx: build an Intel-only binary (-xHost) instead of one that also runs on AMD" OFF)
option(ONEAPI_NO_INTEL_LIBS "icpx: link no Intel runtime library (no libirc, no libsvml: no vector math) -- the last resort when the vendor check still fires on a non-Intel CPU" OFF)

# icpx floating-point model. Unlike GCC and Clang, icpx defaults to
# -fp-model=fast (-ffast-math): reassociation, and SVML for the vectorised
# math calls while the scalar ones go to libm. The vector baseline is
# vectorised and the chunk variants are not, so they no longer compute the
# same bits and the correctness check (checksums against the vector oracle)
# fails by 1e-13 (double) to 6e-7 (float). -fp-model=precise is what GCC and
# Clang do by default; -fimf-use-svml=true sends the scalar calls to SVML too,
# so scalar and vector paths call the same routines (not under
# ONEAPI_NO_INTEL_LIBS: no SVML there, everything goes to libm already).
# ONEAPI_FAST_FP=ON restores icpx's default, for timings comparable with
# campaigns measured before -- the check is then expected to fail.
option(ONEAPI_FAST_FP "icpx: keep the default -fp-model=fast (the vector oracle check then fails)" OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND NOT ONEAPI_FAST_FP)
    add_compile_options(-fp-model=precise)
    if(NOT ONEAPI_NO_INTEL_LIBS)
        add_compile_options(-fimf-use-svml=true)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        add_compile_options(-tp=host)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        if(ONEAPI_XHOST)
            add_compile_options(-xHost)
        else()
            add_compile_options(-march=native)
        endif()
        if(ONEAPI_NO_INTEL_LIBS)
            add_compile_options(-no-intel-lib -fveclib=none)
            add_link_options(-no-intel-lib)
        endif()
    else()
        # GCC et Clang
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm)")
            add_compile_options(-mcpu=native)
        else()
            add_compile_options(-march=native)
        endif()
    endif()
endif()

set(_rt_rpath_dirs "")
# The probe must see the same flags as the real build — in particular clang's
# --gcc-toolchain=... (in CMAKE_CXX_FLAGS) — otherwise -print-file-name would
# resolve libstdc++ from the wrong gcc. Split the flags string into a list.
separate_arguments(_cxx_flags_list UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
foreach (_lib IN LISTS _rt_probe_libs)
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" ${_cxx_flags_list} -print-file-name=${_lib}
        OUTPUT_VARIABLE _lib_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    # -print-file-name echoes the input verbatim when it can't resolve it, so
    # only keep absolute paths that actually exist. REALPATH canonicalises the
    # dir (resolves symlinks / .. ) to avoid duplicate rpath entries.
    if (_lib_path AND IS_ABSOLUTE "${_lib_path}" AND EXISTS "${_lib_path}")
        get_filename_component(_lib_dir "${_lib_path}" DIRECTORY)
        get_filename_component(_lib_dir "${_lib_dir}" REALPATH)
        list(APPEND _rt_rpath_dirs "${_lib_dir}")
    endif ()
endforeach ()

list(REMOVE_DUPLICATES _rt_rpath_dirs)

if (_rt_rpath_dirs)
    message(STATUS "Runtime RPATH dirs: ${_rt_rpath_dirs}")
    list(APPEND CMAKE_BUILD_RPATH ${_rt_rpath_dirs})
    foreach (_dir IN LISTS _rt_rpath_dirs)
        add_link_options("LINKER:-rpath,${_dir}")
    endforeach ()
else ()
    message(WARNING
        "Could not resolve any compiler runtime lib dir for RPATH. Binaries may "
        "need the toolchain module loaded at runtime (LD_LIBRARY_PATH).")
endif ()

# ── Linker: use mold when available ──────────────────────────────────────────
# This tree relinks ~15 executables per debug iteration across ~20 build dirs;
# mold cuts each link from seconds to tenths. AUTO = use it if found and the
# toolchain supports -fuse-ld=mold: GNU >= 12.1, Clang, IntelLLVM (clang-based).
# NVHPC/PGI drive their own link step — never forced there. Sanitizer builds
# are fine with mold. Force with -DUSE_MOLD=ON/OFF.
set(USE_MOLD "AUTO" CACHE STRING "Link with mold: AUTO, ON, OFF")
set_property(CACHE USE_MOLD PROPERTY STRINGS AUTO ON OFF)

if(NOT USE_MOLD STREQUAL "OFF")
    # CHUNK_MOLD_PATH is set by scripts/compile_all.sh, which resolves mold in a
    # subshell: on x86 the mold module conflicts with the gnu version the build
    # uses, so its path is extracted without letting the module state leak here.
    # Falls back to a normal search when the variable is absent.
    if(DEFINED ENV{CHUNK_MOLD_PATH} AND EXISTS "$ENV{CHUNK_MOLD_PATH}")
        set(MOLD_EXECUTABLE "$ENV{CHUNK_MOLD_PATH}" CACHE FILEPATH "mold linker" FORCE)
    else()
        find_program(MOLD_EXECUTABLE mold)
    endif()
    set(_mold_ok FALSE)
    if(MOLD_EXECUTABLE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
            set(_mold_ok TRUE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
               AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.1")
            set(_mold_ok TRUE)
        endif()
    endif()

    if(_mold_ok)
        # How to reach mold, in order of preference.
        #
        # -fuse-ld=mold makes the GNU driver look for an executable named
        # ld.mold on its search path -- NOT for `mold`. A module that installs
        # only `mold` therefore passes find_program() and still fails at link
        # with the driver's real complaint: "collect2: fatal error: cannot find
        # 'ld'" (it fell back to plain ld, which the module environment had
        # displaced). Seen on x86 with gcc 15.2 + mold/2.42.0.
        #
        # So: use -fuse-ld=mold only when ld.mold actually exists. Otherwise
        # build a tiny directory holding an `ld` symlink to mold and hand it to
        # the driver with -B, which works with any gcc and touches nothing else
        # in the toolchain's search path.
        find_program(MOLD_LD_EXECUTABLE ld.mold)
        if(MOLD_LD_EXECUTABLE)
            set(_mold_link_opts "-fuse-ld=mold")
        else()
            set(_mold_shim "${CMAKE_BINARY_DIR}/mold-ld")
            file(MAKE_DIRECTORY "${_mold_shim}")
            file(CREATE_LINK "${MOLD_EXECUTABLE}" "${_mold_shim}/ld" SYMBOLIC)
            set(_mold_link_opts "-B${_mold_shim}")
            message(STATUS "Linker: ld.mold not found — routing mold through "
                           "${_mold_shim}/ld")
        endif()

        # Confirm the driver accepts it (module toolchains can surprise). The
        # cached answer is per mold version: an upgraded mold is tried again.
        execute_process(COMMAND "${MOLD_EXECUTABLE}" --version
                        OUTPUT_VARIABLE _mold_ver
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(MAKE_C_IDENTIFIER "${_mold_ver}" _mold_ver_id)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS "${_mold_link_opts}")
        check_cxx_source_compiles("int main(){return 0;}" _mold_link_works_${_mold_ver_id})
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
        set(_mold_link_works "${_mold_link_works_${_mold_ver_id}}")
        # GCC >= 16 links every program with libatomic_asneeded.so, the linker
        # script `INPUT ( AS_NEEDED ( -latomic ) )`, and mold 2.40.4 (Ubuntu
        # 26.04) rejects a group that opens with AS_NEEDED: "library not found:
        # AS_NEEDED", every link fails. The same libatomic on the command line
        # instead: -fno-link-libatomic drops the script, and -latomic under
        # --as-needed, at the end of the line, is recorded only when used.
        set(_mold_atomic_libs "")
        if(NOT _mold_link_works AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_REQUIRED_LINK_OPTIONS "${_mold_link_opts};-fno-link-libatomic")
            set(CMAKE_REQUIRED_LIBRARIES "-Wl,--push-state,--as-needed;-latomic;-Wl,--pop-state")
            check_cxx_source_compiles("int main(){return 0;}" _mold_link_atomic_works_${_mold_ver_id})
            unset(CMAKE_REQUIRED_LINK_OPTIONS)
            unset(CMAKE_REQUIRED_LIBRARIES)
            if(_mold_link_atomic_works_${_mold_ver_id})
                set(_mold_link_works TRUE)
                list(APPEND _mold_link_opts "-fno-link-libatomic")
                set(_mold_atomic_libs "-Wl,--push-state,--as-needed -latomic -Wl,--pop-state")
            endif()
        endif()
        if(_mold_link_works)
            set(_mold_note "")
            if(_mold_atomic_libs)
                set(_mold_note ", libatomic as ${_mold_atomic_libs} (mold rejects libatomic_asneeded.so)")
            endif()
            string(JOIN " " _mold_opts_text ${_mold_link_opts})
            message(STATUS "Linker: mold (${_mold_ver}) — ${MOLD_EXECUTABLE} "
                           "via ${_mold_opts_text}${_mold_note}")
            add_link_options(${_mold_link_opts})
            if(_mold_atomic_libs)
                string(APPEND CMAKE_CXX_STANDARD_LIBRARIES " ${_mold_atomic_libs}")
            endif()
        else()
            message(STATUS "Linker: mold found but ${_mold_link_opts} rejected by "
                           "${CMAKE_CXX_COMPILER_ID} — using the default linker")
        endif()
    elseif(USE_MOLD STREQUAL "ON")
        message(FATAL_ERROR "USE_MOLD=ON but mold is absent or unsupported by "
                            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    else()
        message(STATUS "Linker: default (mold not found or not supported by "
                       "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
endif()

# ── C++20 requirement (portable, with NVHPC workaround) ───────────────────────
# CMake's compile-feature table for nvc++ (NVHPC 26.x) doesn't advertise
# cxx_std_20, so target_compile_features(<t> ... cxx_std_20) hard-errors with
# "feature cxx_std_20 is not known to CXX compiler NVHPC" even though nvc++ fully
# supports C++20. Centralise the requirement: pass the -std flag directly for
# NVHPC, keep the portable feature request for every other compiler. Defined
# before the first add_subdirectory(), so it is visible in chunk/ and examples/.
# fast_allocators keeps its own inline equivalent because it is installable
# standalone (own project()) and must not depend on this umbrella-level helper.
# Drop the NVHPC branch once the baseline CMake knows nvc++'s C++20 support.
function(chunk_require_cxx20 target scope)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(${target} ${scope} -std=gnu++20)
  else()
    target_compile_features(${target} ${scope} cxx_std_20)
  endif()
endfunction()

# Compile-time chunk geometry (see chunk/public/chunk/internal/chunk_size_config.h).
# Global so every TU sees the same value (chunk_size_config.h is inline → ODR).
if(DEFINED CHUNK_SIZE)
    add_compile_definitions(CHUNK_SIZE=${CHUNK_SIZE})
    message(STATUS "chunk geometry: CHUNK_SIZE=${CHUNK_SIZE} (overriding default 64)")
else()
    message(STATUS "chunk geometry: CHUNK_SIZE=64 (default; pass -DCHUNK_SIZE=N to override)")
endif()

# ── include-what-you-use (optional) ──────────────────────────────────────────
# Ninja job pool for the unity-built executables (shrink_bench, validation):
# each of their translation units peaks at ~3 GB under GCC, so two at once can
# exhaust a 16 GB machine. Empty (default): they share CMAKE_JOB_POOL_COMPILE
# like every other compile. On a small machine, declare a pool of depth 1 and
# name it here, e.g.
#   -DCMAKE_JOB_POOLS="compile=2;link=1;heavy=1" -DCHUNK_HEAVY_JOB_POOL=heavy
set(CHUNK_HEAVY_JOB_POOL "" CACHE STRING "Ninja job pool for the unity-built shrink_bench and validation executables (empty: the default compile pool)")

# Opt-in header hygiene. -DENABLE_IWYU=ON runs include-what-you-use alongside
# each C++ compile, using the project mapping file iwyu.imp (routes *_inline.h
# implementations to their public facade, ankerl internals to
# unordered_dense.h, plus the STL bits mappings). IWYU must be on PATH; if it is
# absent we warn and continue (the build is never blocked). IWYU is slow and
# noisy, so it stays OFF by default — turn it on for a dedicated hygiene pass:
#   cmake -B build-iwyu -DENABLE_IWYU=ON ...   &&   cmake --build build-iwyu
# Must be set before the first add_subdirectory() so every target inherits it.
option(ENABLE_IWYU "Run include-what-you-use during the build (needs iwyu on PATH)" OFF)
if(ENABLE_IWYU)
    find_program(IWYU_EXE NAMES include-what-you-use iwyu)
    if(IWYU_EXE)
        # IWYU parses with its OWN embedded clang, not the project compiler. If
        # that clang is older than the libstdc++/C++ features this tree uses, it
        # fails to parse (e.g. clang 17 chokes on libstdc++-16's
        # __builtin_popcountg, a clang-19 builtin). Detect the embedded clang and
        # warn loudly rather than dumping a wall of parse errors. Threshold 19
        # covers the current headers; raise it if newer builtins are required.
        execute_process(COMMAND "${IWYU_EXE}" --version
                        OUTPUT_VARIABLE _iwyu_ver ERROR_VARIABLE _iwyu_ver
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX MATCH "clang version ([0-9]+)" _m "${_iwyu_ver}")
        set(_iwyu_clang "${CMAKE_MATCH_1}")
        if(_iwyu_clang AND _iwyu_clang LESS 19)
            message(WARNING
                "IWYU at ${IWYU_EXE} is built on clang ${_iwyu_clang}; this "
                "project needs an IWYU built on clang >= 19 to parse its C++23 + "
                "modern libstdc++ headers. It will likely fail with parse errors "
                "(e.g. '__builtin_popcountg'). Build IWYU against your project's "
                "clang and pass -DIWYU_EXE=<path>, or run the IWYU pass on a host "
                "whose toolchain matches. Continuing, but IWYU output may be junk.")
        endif()
        set(_iwyu_map "${CMAKE_CURRENT_SOURCE_DIR}/iwyu.imp")
        if(NOT EXISTS "${_iwyu_map}")
            message(WARNING "ENABLE_IWYU=ON but mapping file not found: ${_iwyu_map} "
                            "— running IWYU without a mapping file")
            set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE
                "${IWYU_EXE}" "-Xiwyu" "--max_line_length=100")
        else()
            set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE
                "${IWYU_EXE}"
                "-Xiwyu" "--mapping_file=${_iwyu_map}"
                "-Xiwyu" "--max_line_length=100")
            message(STATUS "IWYU: enabled (${IWYU_EXE}), mapping ${_iwyu_map}")
        endif()
    else()
        message(WARNING "ENABLE_IWYU=ON but include-what-you-use was not found on "
                        "PATH — continuing without it")
    endif()
endif()

# ── Intel libm for GNU / Clang builds on x86-64 ─────────────────────────────
# RHEL 8's glibc libm is old and slow on cos/sin/atan/tan, which the benches'
# compute phases live in: AMD Milan, GNU, shrink_bench_chunk2a_256, 64 threads,
# 15.70 s -> 12.41 s with libimf (same seed, same long_val_sum). icpx links
# libimf itself; other compilers get it here when it is found. Every
# executable of the build uses the same libm, so check-mode comparisons within
# a build are unaffected.
option(SHRINK_INTEL_LIBM "x86-64, non-icpx: link the executables with Intel's libm (libimf) when found" ON)
set(INTEL_LIBM_DIR "" CACHE PATH "Directory holding libimf.so and libintlc.so (default: $CMPLR_ROOT/lib, then next to icpx)")
if(SHRINK_INTEL_LIBM AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
   AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(_imf_hints "")
    if(INTEL_LIBM_DIR)
        list(APPEND _imf_hints "${INTEL_LIBM_DIR}")
    endif()
    if(DEFINED ENV{CMPLR_ROOT})
        list(APPEND _imf_hints "$ENV{CMPLR_ROOT}/lib")
    endif()
    find_program(_shrink_icpx icpx)
    # An icpx installed but not loaded: the llvm builds of compile_all.sh
    # reset the compiler modules, so neither icpx nor CMPLR_ROOT is visible
    # there, and the omp-llvm build on Milan silently linked glibc's libm
    # (34.6% of the chunk2a_256 profile). INTEL_ICPX names one; otherwise the
    # newest match of INTEL_ICPX_GLOB is taken.
    set(INTEL_ICPX "" CACHE FILEPATH "icpx whose libimf to link when icpx is not in PATH (default: newest match of INTEL_ICPX_GLOB)")
    set(INTEL_ICPX_GLOB "/ccc/products/icx-*/system/default/*/bin/icpx" CACHE STRING
        "Where to look for an installed icpx when it is not in PATH (SHRINK_INTEL_LIBM)")
    set(_shrink_icpx_used "${_shrink_icpx}")
    if(NOT _shrink_icpx)
        if(INTEL_ICPX AND EXISTS "${INTEL_ICPX}")
            set(_shrink_icpx_used "${INTEL_ICPX}")
        elseif(INTEL_ICPX_GLOB)
            file(GLOB _icpx_candidates LIST_DIRECTORIES false ${INTEL_ICPX_GLOB})
            if(_icpx_candidates)
                if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
                    list(SORT _icpx_candidates COMPARE NATURAL ORDER DESCENDING)
                else()
                    list(SORT _icpx_candidates)
                    list(REVERSE _icpx_candidates)
                endif()
                list(GET _icpx_candidates 0 _shrink_icpx_used)
            endif()
        endif()
    endif()
    if(_shrink_icpx_used)
        execute_process(COMMAND "${_shrink_icpx_used}" -print-file-name=libimf.so
                        OUTPUT_VARIABLE _imf_path OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_imf_path AND IS_ABSOLUTE "${_imf_path}" AND EXISTS "${_imf_path}")
            get_filename_component(_imf_dir0 "${_imf_path}" DIRECTORY)
            list(APPEND _imf_hints "${_imf_dir0}")
        endif()
        # The oneAPI layouts: <prefix>/bin/icpx with the runtime in <prefix>/lib
        # (or <prefix>/compiler/lib in older trees).
        get_filename_component(_icpx_prefix "${_shrink_icpx_used}" DIRECTORY)
        get_filename_component(_icpx_prefix "${_icpx_prefix}" DIRECTORY)
        list(APPEND _imf_hints "${_icpx_prefix}/lib" "${_icpx_prefix}/compiler/lib")
        if(NOT _shrink_icpx)
            message(STATUS "Intel libm: icpx not in PATH, looking next to ${_shrink_icpx_used}")
        endif()
    endif()
    if(_imf_hints)
        find_library(SHRINK_LIBIMF NAMES imf HINTS ${_imf_hints} NO_DEFAULT_PATH)
        find_library(SHRINK_LIBINTLC NAMES intlc HINTS ${_imf_hints} NO_DEFAULT_PATH)
    endif()
    if(SHRINK_LIBIMF AND SHRINK_LIBINTLC)
        get_filename_component(_imf_dir "${SHRINK_LIBIMF}" DIRECTORY)
        message(STATUS "Intel libm: ${SHRINK_LIBIMF} -- linked ahead of libm (SHRINK_INTEL_LIBM=OFF to disable)")
        # link_libraries: every target created from here on links them first,
        # ahead of the -lm the examples add and the driver's own. libintlc is
        # kept as a DIRECT dependency even under --as-needed: libimf needs it,
        # and the executable's RUNPATH only serves its direct dependencies.
        link_libraries("${SHRINK_LIBIMF}" "-Wl,--push-state,--no-as-needed" "${SHRINK_LIBINTLC}" "-Wl,--pop-state")
        add_link_options("LINKER:-rpath,${_imf_dir}")
        # And its -L, as a raw flag. CMake (3.28) writes libintlc as a bare
        # -lintlc, without the directory, so the link only worked where
        # LIBRARY_PATH held it -- a shell that had sourced oneAPI's setvars.sh,
        # as compile_all.sh does -- and failed elsewhere with "library not
        # found: intlc". link_directories() would be dropped too when the
        # directory is among the compiler's implicit ones; a raw -L is not.
        add_link_options("-L${_imf_dir}")
    else()
        message(STATUS "Intel libm: not found -- glibc libm (set INTEL_LIBM_DIR, or load the oneAPI environment)")
    endif()
endif()

# ── ArmPL's libamath for GNU / Clang builds on aarch64 ──────────────────────
# Arm's optimised libm (armpl module, free), linked ahead of glibc's: the
# benches' compute phases live in cos/sin/atan/tan (the x86-64 counterpart,
# libimf, took shrink_bench_chunk2a_256 from 15.70 s to 12.41 s on AMD Milan,
# GNU). NVHPC is left alone: it links its own optimised math library.
# SHRINK_ARM_LIBM is OFF by default: measured on Grace (clang 22, 64 threads,
# same seed, RHEL 9.6 glibc vs armpl 25.07), chunk2a_dyn 8.30 -> 8.12 s,
# chunk2a_256 6.20 -> 6.23 s, chunk2_256 5.76 -> 5.75 s -- noise; RHEL 9's
# aarch64 libm is on par with ArmPL's for these calls. Kept for re-measuring:
# -DSHRINK_ARM_LIBM=ON -DARMPL_LIBM_DIR=<dir of libamath.so>; the executables
# then carry an rpath and run without the module.
option(SHRINK_ARM_LIBM "aarch64, GNU/Clang: link the executables with ArmPL's libamath when found (default OFF: no measured gain on Grace)" OFF)
set(ARMPL_LIBM_DIR "" CACHE PATH "Directory holding libamath.so (default: from the armpl module, $ARMPL_DIR/lib)")
if(SHRINK_ARM_LIBM AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(_amath_hints "")
    if(ARMPL_LIBM_DIR)
        list(APPEND _amath_hints "${ARMPL_LIBM_DIR}")
    endif()
    if(DEFINED ENV{ARMPL_DIR})
        list(APPEND _amath_hints "$ENV{ARMPL_DIR}/lib")
    endif()
    foreach(_v ARMPL_LIBRARIES ARMPL_LIBRARY_DIR ARMPL_LIB)
        if(DEFINED ENV{${_v}})
            list(APPEND _amath_hints "$ENV{${_v}}")
        endif()
    endforeach()
    if(_amath_hints)
        find_library(SHRINK_LIBAMATH NAMES amath HINTS ${_amath_hints} NO_DEFAULT_PATH)
    endif()
    if(SHRINK_LIBAMATH)
        get_filename_component(_amath_dir "${SHRINK_LIBAMATH}" DIRECTORY)
        message(STATUS "ArmPL libm: ${SHRINK_LIBAMATH} -- linked ahead of libm (SHRINK_ARM_LIBM=ON)")
        # link_libraries: every target created from here on links it first,
        # ahead of the -lm the examples add and the driver's own.
        link_libraries("${SHRINK_LIBAMATH}")
        add_link_options("LINKER:-rpath,${_amath_dir}")
    else()
        message(STATUS "ArmPL libm: libamath not found -- glibc libm (load the armpl module before cmake, or set ARMPL_LIBM_DIR)")
    endif()
endif()

# ── Documentation ─────────────────────────────────────────────────────────────
# `cmake --build <build> --target doc` -> docs/html/index.html. The Doxyfile's
# paths are relative to docs/, hence the working directory.
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if (DOXYGEN_FOUND)
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
            COMMENT "Generating the API documentation in docs/html"
            VERBATIM)
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
# Validation harnesses (examples/validation): built where they check something
# an optimised build does not -- Debug, CHECK_OPTIM and the sanitizer builds.
# Some 100 executables of 14 s each (omp gnu debug), compiled in every one of
# the 30-odd configurations of compile_all*.sh, were most of the build time;
# the perf configurations (release, stats, aggressive compression,
# MM_PROTECT, std_alloc) skip them. -DCHUNK_BUILD_VALIDATION=ON / OFF forces
# the choice. Deliberately not an option(): the default follows the build type
# and options of THIS configure, so a cached value cannot go stale when they
# change. Decided here, before the library: see CHUNK_TEST_HOOKS below.
if(DEFINED CHUNK_BUILD_VALIDATION)
    set(_build_validation ${CHUNK_BUILD_VALIDATION})
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CHECK_OPTIM OR NOT SANITIZER STREQUAL "none")
    set(_build_validation ON)
else()
    set(_build_validation OFF)
endif()

# The harnesses' test hooks (chunk/internal/test_hooks.h: forced deferrals,
# read and pass delays) are compiled into the whole tree wherever the
# harnesses are built, OFF at run time until a harness switches them on --
# so the harnesses that use them share one executable with the others (one
# definition of the header-only engine for all) instead of one each. Recorded
# in chunk/build_options.h (CHUNK_OPT_TEST_HOOKS): the library archives and
# every executable see the same engine. Perf configurations never have it.
set(CHUNK_TEST_HOOKS ${_build_validation})

add_subdirectory(chunk)

if(NOT CHUNK_STD_ALLOC)
    set(ALLOCATORS_OMP ${OMP})
    add_subdirectory(fast_allocators)
endif()

# Examples, benchmarks and validation harnesses: top level by default
# (SHRINK_ARRAY_BUILD_EXAMPLES), never imposed on a consuming project.
if(SHRINK_ARRAY_BUILD_EXAMPLES)
    # shrink-build-stamp: [build] line in the validation binaries, `check` target
    include(${PROJECT_SOURCE_DIR}/cmake/shrink_build_stamp.cmake)
    add_subdirectory(examples)
    shrink_add_check_target()
endif()

# The reduced models of the lock-free handshakes (formal/, docs/formal_models.md):
# run natively by ctest, and under GenMC when it is installed. Tests only.
if(SHRINK_ARRAY_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(formal)
endif()

# Every raw read of a chunk's buffer in the library states why it is safe
# (scripts/check_raw_reads.py): a source check, run with the validation tests.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(SHRINK_ARRAY_BUILD_TESTS AND BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(NAME check_raw_reads
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_raw_reads.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_raw_reads PROPERTY LABELS validation)
    # Every #include says why it is there; system and project headers are not
    # mixed in one block (scripts/check_includes.py).
    add_test(NAME check_includes
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_includes.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_includes PROPERTY LABELS validation)
    # Constants are snake_case; UPPER_CASE is for macros, and no k-prefix
    # (scripts/check_naming.py).
    add_test(NAME check_naming
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_naming.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_naming PROPERTY LABELS validation)
    # Every source file carries its license: the Apache block for C++, SPDX
    # lines for scripts and build files, the upstream SPDX line for the vendored
    # headers (scripts/check_headers.py; --fix repairs).
    add_test(NAME check_headers
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_headers.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_headers PROPERTY LABELS validation)
    # Every file the sources name -- an #include, a relative Python import, a
    # CMake source, a harness, a Markdown link -- is tracked by git: a file
    # never `git add`ed builds only where it was written (scripts/check_tracked.py).
    add_test(NAME check_tracked
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_tracked.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_tracked PROPERTY LABELS validation)
    # The documentation matches the code, and the English and French versions
    # each other (scripts/check_docs.py).
    add_test(NAME check_docs
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_docs.py ${PROJECT_SOURCE_DIR})
    set_property(TEST check_docs PROPERTY LABELS "validation;docs")
    # The Doxygen documentation builds without a warning (scripts/check_doxygen.py),
    # when Doxygen is installed: find_package(Doxygen) above, for the doc target.
    if (DOXYGEN_FOUND)
        add_test(NAME check_doxygen
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check_doxygen.py
                         ${PROJECT_SOURCE_DIR} ${DOXYGEN_EXECUTABLE})
        set_property(TEST check_doxygen PROPERTY LABELS docs)
        # 77: a Doxygen older than 1.9, which misreads the sources (the script says so).
        set_property(TEST check_doxygen PROPERTY SKIP_RETURN_CODE 77)
    endif()
endif()
