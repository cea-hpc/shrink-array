# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# mdspan_tiles: one 2D field written through the mdspan view of
# chunk/mdspan.h, row-major and in tiles of 16 x 16, over a chunk2A array of
# chunk size 256; prints the chunks left materialised in each layout. Needs
# <mdspan> (libstdc++ 16, libc++ 17) or kokkos/mdspan on the include path;
# without either it exits 77, which ctest reports as skipped.
if (TARGET chunk_zero_any_256)
    add_executable(mdspan_tiles mdspan_tiles.cpp)
    target_link_libraries(mdspan_tiles PRIVATE chunk_zero_any_256)
    target_compile_definitions(mdspan_tiles PRIVATE CHUNK_VARIANT_ZERO_ANY)
    if (OMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(mdspan_tiles PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    chunk_require_cxx_std(mdspan_tiles PRIVATE)

    add_test(NAME mdspan_tiles COMMAND mdspan_tiles 512)
    set_property(TEST mdspan_tiles PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST mdspan_tiles PROPERTY LABELS validation)
    set_property(TEST mdspan_tiles PROPERTY SKIP_RETURN_CODE 77)
endif ()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
