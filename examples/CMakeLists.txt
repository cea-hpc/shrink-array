# ── Fast math + vector libm, for the perf targets that call cos/sin/... in
# their loops. A `simd` loop that calls into libm only vectorises when the
# call itself has a vector form the compiler is told about: the classic Intel
# compiler carried SVML and vectorised std::cos on its own, gcc and clang need
# -fno-math-errno (the errno store is a side effect) AND a -fveclib -- ArmPL on
# the AArch64 node, glibc's libmvec (_ZGVdN4v_cos and friends) on x86 with
# clang; gcc on x86 picks libmvec up by itself from glibc's math-vector.h
# once __FAST_MATH__ is defined. Checked with -Rpass=loop-vectorize on
# multi-mat: the compute loop goes from "library call cannot be vectorized"
# to "vectorized loop (width 4, interleaved 4)".
#
# Not for a check target: the vector cos differs from the scalar one in the
# last ulp, and check mode compares a std::vector run with a shrink run whose
# loops are scalar by construction.
set(ARMPL_ROOT "/ccc/products2/armpl-25.07/Rhel_9__aarch64/gcc--12.3.0__cuda--12.4/default"
    CACHE PATH "ArmPL install root (lib/libamath.so + include/)")

function(chunk_enable_fast_math target)
    # NVHPC (nvc++) : skip entirely. It's a validation compiler here, its
    # fast-math flags differ, and applying gcc/clang switches breaks the build.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        return()
    endif()

    set(_fm -fno-math-errno -fno-trapping-math -funsafe-math-optimizations -fno-signed-zeros)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64"
       AND EXISTS "${ARMPL_ROOT}/lib/libamath.so")
        list(APPEND _fm -fveclib=ArmPL)
        target_link_directories(${target} PRIVATE "${ARMPL_ROOT}/lib")
        target_link_libraries(${target}   PRIVATE amath)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH   "${ARMPL_ROOT}/lib"
            INSTALL_RPATH "${ARMPL_ROOT}/lib")
        message(STATUS "fast-math[${target}]: ArmPL vector libm")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
           AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # glibc's libmvec (2.22+): the vector variants live in libm itself.
        list(APPEND _fm -fveclib=libmvec)
        message(STATUS "fast-math[${target}]: glibc libmvec")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        # gcc-16 x86_64 -O3 hits an ICE in the AVX2 expand pass on
        # shrink_bench.cpp's driver TU (see examples/shrink_bench/CMakeLists.txt);
        # -ffast-math would define __FAST_MATH__ and let glibc's libmvec in,
        # but that combination is untested on this compiler -- unsafe-math only.
    endif()

    # Why a loop did NOT vectorise, per loop, from the compiler rather than
    # from guesswork. -Wpass-failed (on by default) only reports that a pragma's
    # transformation failed; this reports the analysis behind it -- the
    # dependence, the call, the type. Applied here because this function marks
    # the targets where a `simd` pragma is expected to succeed: they are the
    # ones that get -fveclib, without which any loop calling sin/cos/tan stays
    # scalar whatever else is true.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND _fm -Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND _fm -fopt-info-vec-missed)
    endif()

    target_compile_options(${target} PRIVATE ${_fm})
    target_link_libraries(${target} PRIVATE m)
endfunction()

# Simple example
add_subdirectory(simple_array)

# Simple example chunk : double value compression
add_subdirectory(shrink_bench)

# Validation harnesses (ctest): re-chunk content invariance, dyn geometry
add_subdirectory(validation)

# Multi-dimensional array example
add_subdirectory(multi-mat)

# study vectorization whith OpenMP and chunk
add_subdirectory(test_vecto_chunk)

# The chunk-size advisor's step-0 check (dyn chunk2A only).
add_subdirectory(advisor_demo)
