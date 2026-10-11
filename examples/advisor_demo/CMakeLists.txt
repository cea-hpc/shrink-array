# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# advisor_demo: the ChunkSizeAdvisor's step-0 check -- the same data built
# at fifteen REAL chunk sizes (powers of two and multiples of 64), then the
# advisor's per-size predictions (made from S=256, then S=1024) against that
# ground truth. Returns the number of sizes
# whose prediction is off, so it doubles as a test. Needs the runtime
# geometry (one array per size), hence the dyn chunk2A library only.
if (CHUNK_BUILD_DYN)
    add_executable(advisor_demo advisor_demo.cpp)
    target_link_libraries(advisor_demo PRIVATE chunk_v2a_dyn)
    target_compile_definitions(advisor_demo PRIVATE
        CHUNK_VARIANT_V2A CHUNK_DYNAMIC DYN_CHUNK_SIZE=${SHRINK_BENCH_DYN_SIZE})
    if (OMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(advisor_demo PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    chunk_require_cxx_std(advisor_demo PRIVATE)

    add_test(NAME advisor_demo COMMAND advisor_demo)
    set_property(TEST advisor_demo PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
    set_property(TEST advisor_demo PROPERTY LABELS validation)
endif ()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
