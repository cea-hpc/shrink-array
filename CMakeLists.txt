cmake_minimum_required(VERSION 3.12)
project(chunk LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CHECK_OPTIM "Enable runtime invariant checks in release builds" OFF)

# An emptied slab would otherwise keep its resident pages: the allocator never
# unmaps a value slab (see the op/value alCHUNK_RECLAIM_EMPTY_SLABSiasing note in tc_allocator.h), so the
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
# TCTypeCAllocator::manage_full_empty, the VALUE family, whose free list lives
# in the heap-allocated tab_firstFree and whose metadata live in the slab
# header object -- both outside the pages being dropped -- and which promises
# zeroed memory anyway, which is what MADV_DONTNEED returns. The op/ctx family
# threads its free list THROUGH the freed elements and would be corrupted by
# this; its on_kept is an empty lambda and it is never madvised.
#
# Turn it OFF if a profile ever shows page faults from slabs that empty and
# refill in a loop; "strace -c -e trace=madvise" gives the call count directly.
option(CHUNK_RECLAIM_EMPTY_SLABS "Release an emptied slab's pages with madvise instead of keeping them resident" ON)
if(CHUNK_RECLAIM_EMPTY_SLABS)
    add_definitions(-DCHUNK_RECLAIM_EMPTY_SLABS)
endif()

# The write/compress handshake (chunk_config::asymmetric_dekker, config.h).
# ON = asymmetric where it can be: the write announces itself with a plain
# store, the compressor / flush side pays a membarrier() for everyone. OFF =
# symmetric: a seq_cst store + fence on every element write. The default since
# the write engine stopped deferring every write of a "dangerous" rank (see
# config.h for the numbers): the rare side is now a few hundred announces per
# second, and the asymmetric shape is -7% at 6 threads on x86 where the
# symmetric one is +7%. TSan builds are symmetric regardless, and so is any
# run whose membarrier registration fails.
option(CHUNK_ASYMMETRIC_DEKKER "Asymmetric write/compress handshake (plain store on writes, membarrier on the rare side)" ON)
if(CHUNK_ASYMMETRIC_DEKKER)
    add_definitions(-DCHUNK_ASYMMETRIC_DEKKER)
endif()

# Chunk2A's per-chunk second-value table holds a 4-byte identifier instead of
# the value for types wider than 4 bytes (double, long, complex<double>):
# 4 bytes per chunk instead of 8 or 16, -3.8% RSS on the 50 M-chunk reference
# run, for one more dependent load on every write of a materialised chunk --
# +3% CPU on the perf bench. Off by default; see SecondValueTraits in
# chunk/internal/tab_cst_val2_holder.h.
option(CHUNK_SECOND_VALUE_INDEXED "Chunk2A: store second values as 4-byte identifiers (less memory, +3% write CPU)" OFF)
if(CHUNK_SECOND_VALUE_INDEXED)
    add_definitions(-DCHUNK_SECOND_VALUE_INDEXED)
endif()
#if(CHECK_OPTIM)
#    target_compile_definitions(chunk PUBLIC CHECK_OPTIM)
#endif()

# ── Sanitizer option ──────────────────────────────────────────────────────────
set(SANITIZER "none" CACHE STRING "Sanitizer to enable: none, thread, address, token")
set_property(CACHE SANITIZER PROPERTY STRINGS none thread address token)

message(STATUS "Sanitizer: ${SANITIZER}")

# ── OpenMP ────────────────────────────────────────────────────────────────────
if(OMP)
    find_package(OpenMP 4.5 REQUIRED)
    set(CMAKE_LIBRARY_PATH /usr/local/lib/x86_64-unknown-linux-gnu ${CMAKE_LIBRARY_PATH})
endif()

# ── Options ───────────────────────────────────────────────────────────────────

set(CHUNK_OMP ${OMP})

include(CTest)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# ── OMPT detection (Clang/IntelLLVM only) ────────────────────────────────────
if(OMP)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # TSan is incompatible with OMPT instrumentation
        if(NOT SANITIZER STREQUAL "thread")
            add_definitions(-DHAVE_OMPT)
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
        set(GCC_TOOLCHAIN_DIR "/ccc/products/gcc-15.2.0" CACHE PATH "Path to GCC toolchain")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ --gcc-toolchain=${GCC_TOOLCHAIN_DIR}")
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
# They used to be injected only into CMAKE_CXX_FLAGS_DEBUG below, which CMake
# reads only when CMAKE_BUILD_TYPE=Debug. A RelWithDebInfo build with
# -DSANITIZER=address therefore configured cleanly, printed "Address Sanitizer
# enabled", and produced an UNinstrumented binary -- `nm -C <exe> | grep -c
# __asan` returned 1 instead of thousands. Three "the sanitizer reports
# nothing" conclusions came from binaries that had never been instrumented.
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
    # On cherche la bibliothèque 'profiler'
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
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -D_NOT_INLINE -DDEBUG"
    CACHE STRING "Debug flags" FORCE)

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -g"
    CACHE STRING "Release flags" FORCE)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O3 -g"
    CACHE STRING "RelWithDebInfo flags" FORCE)

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

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        add_compile_options(-tp=host)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        add_compile_options(-xHost)
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

        # Confirm the driver accepts it (module toolchains can surprise).
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS "${_mold_link_opts}")
        check_cxx_source_compiles("int main(){return 0;}" _mold_link_works)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
        if(_mold_link_works)
            execute_process(COMMAND "${MOLD_EXECUTABLE}" --version
                            OUTPUT_VARIABLE _mold_ver
                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            message(STATUS "Linker: mold (${_mold_ver}) — ${MOLD_EXECUTABLE} "
                           "via ${_mold_link_opts}")
            add_link_options(${_mold_link_opts})
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

# Compile-time chunk geometry (see chunk/public/chunk/chunk_size_config.h).
# Global so every TU sees the same value (chunk_size_config.h is inline → ODR).
if(DEFINED CHUNK_SIZE)
    add_compile_definitions(CHUNK_SIZE=${CHUNK_SIZE})
    message(STATUS "chunk geometry: CHUNK_SIZE=${CHUNK_SIZE} (overriding default 64)")
else()
    message(STATUS "chunk geometry: CHUNK_SIZE=64 (default; pass -DCHUNK_SIZE=N to override)")
endif()

# ── include-what-you-use (optional) ──────────────────────────────────────────
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

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(chunk)

if(NOT CHUNK_STD_ALLOC)
    set(ALLOCATORS_OMP ${OMP})
    add_subdirectory(fast_allocators)
endif()

add_subdirectory(examples)
