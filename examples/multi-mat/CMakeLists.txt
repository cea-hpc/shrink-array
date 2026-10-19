# Copyright 2024-2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# Two executables from the same sources (config.h):
#   multi-mat           the manual version, every compiler: the program makes
#                       the phase calls of chunk/manage_ompt.h itself (four
#                       one-line wrappers) and compresses its field arrays by
#                       explicit compress() calls every freq_compress
#                       iterations. No OMPT tool in this binary.
#   multi-mat-embedded  the embedded version: no call at all, the OMPT tool of
#                       manage_ompt.cpp (this directory) drives the
#                       compression from the runtime's callbacks. Clang and
#                       IntelLLVM only: gcc's libgomp has no OMPT, and the
#                       sources refuse to compile without the OMPT tool. On such a
#                       compiler the target still exists, out of `all` and
#                       without its test, so building it by name gives that
#                       error rather than a binary that loses its deferred
#                       writes.
# Both link chunk_v1v2_<size>, a variant built WITHOUT the library's own
# tool, so that ompt_start_tool and g_ompt_suspended come from here.
# The sources both executables share; each adds the file that defines the
# OMPT switch of the library: manage_ompt.cpp (the tool) or ompt_off.cpp.
set(_mm_sources main.cpp
        ComputeDistribution.cpp ComputeInterval.cpp InputData.cpp
        MultiMatAlgo.h Loops.h PMP.h Containers.h Dispatch.h ComputeLoops.h Simu.h
        config.h)

if (OMP)
    find_package(OpenMP)
endif ()

function(multi_mat_setup_target tgt)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
    endif ()

    chunk_require_cxx_std(${tgt} PRIVATE)

    # A library carrying BOTH variant defines: the three field arrays are
    # ArrayChunk<double> (v1), the presence fraction ArrayChunk2<double>.
    # Not the bare `chunk` target: that one carries no variant define (the
    # std::vector shape of the library) while this example instantiates the
    # arrays itself, so its per-rank allocators and candidate lists were
    # never set up -- every write from a rank other than 0 died in
    # get_chunk_allocator().
    target_link_libraries(${tgt} PRIVATE chunk_v1v2_${CHUNK_DEFAULT_SIZE})

    # The compute loop is std::cos/sin/tan under `omp simd` (LoopsSimd, on the
    # std::vector path): without a vector libm it does not vectorise at all --
    # see chunk_enable_fast_math in examples/CMakeLists.txt. multi-mat compares
    # nothing bit for bit (it prints its result), so fast math is fine here.
    chunk_enable_fast_math(${tgt})
endfunction()

add_executable(multi-mat ${_mm_sources} ompt_off.cpp)
multi_mat_setup_target(multi-mat)

add_test(NAME multi_mat COMMAND multi-mat)
set_property(TEST multi_mat PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)

if (CHUNK_HAVE_OMPT)
    add_executable(multi-mat-embedded ${_mm_sources} manage_ompt.cpp)
    add_test(NAME multi_mat_embedded COMMAND multi-mat-embedded)
    set_property(TEST multi_mat_embedded PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
else ()
    add_executable(multi-mat-embedded EXCLUDE_FROM_ALL ${_mm_sources} manage_ompt.cpp)
    message(STATUS "multi-mat-embedded: not in `all` (no OMPT tool with ${CMAKE_CXX_COMPILER_ID}; Clang or IntelLLVM, OMP=ON)")
endif ()
multi_mat_setup_target(multi-mat-embedded)
target_compile_definitions(multi-mat-embedded PRIVATE MULTIMAT_EMBEDDED_COMPRESSION)

# multi-mat-md: the same mock on a 2D mesh, every array through an mdspan
# view (chunk/mdspan.h): a material occupies a box per iteration, the chunk
# materials are laid out in tiles of one chunk and walked tile by tile
# (chunk/mdspan_tile_ops.h). Needs <mdspan> or kokkos/mdspan on the include
# path, else it exits 77 (skipped). Its tests are small and replay the run
# serially (--verify), once per way of walking the chunk materials.
set(_mm_md_sources main_md.cpp
        SimuMd.h LoopsMd.h ComputeLoopsMd.h
        MultiMatAlgo.h Loops.h PMP.h Containers.h Dispatch.h config.h)
add_executable(multi-mat-md ${_mm_md_sources} ompt_off.cpp)
multi_mat_setup_target(multi-mat-md)
# A 200 x 328 mesh (328: the last column of tiles padded), 6 materials half
# chunk half vector, 24 iterations with resets and compression passes.
function(multi_mat_md_test name exe)
    add_test(NAME ${name}
             COMMAND ${exe} --nx 200 --ny 328 --mat 6 --iter 24 --option 3 --pc-chunk 8 --pc-vector 8
                     --freq-rtz 5 --freq-compress 4 --seed 7 --verify ${ARGN})
    set_property(TEST ${name} PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST ${name} PROPERTY SKIP_RETURN_CODE 77)
endfunction()
multi_mat_md_test(multi_mat_md_tiles multi-mat-md --chunk-mode tiles)
multi_mat_md_test(multi_mat_md_elements multi-mat-md --chunk-mode elements)
multi_mat_md_test(multi_mat_md_rows multi-mat-md --chunk-mode rows)

if (CHUNK_HAVE_OMPT)
    add_executable(multi-mat-md-embedded ${_mm_md_sources} manage_ompt.cpp)
    multi_mat_md_test(multi_mat_md_embedded multi-mat-md-embedded)
else ()
    add_executable(multi-mat-md-embedded EXCLUDE_FROM_ALL ${_mm_md_sources} manage_ompt.cpp)
endif ()
multi_mat_setup_target(multi-mat-md-embedded)
target_compile_definitions(multi-mat-md-embedded PRIVATE MULTIMAT_EMBEDDED_COMPRESSION)
