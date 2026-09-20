add_executable(multi-mat main.cpp
        ComputeDistribution.cpp ComputeInterval.cpp
        MultiMatAlgo.h Loops.h PMP.h Containers.h Dispatch.h ComputeLoops.h Simu.h
        InputData.cpp ComputeDistribution.cpp)

if (OMP)
    find_package(OpenMP)
endif ()

if (OpenMP_CXX_FOUND)
    target_link_libraries(multi-mat PUBLIC OpenMP::OpenMP_CXX)
endif ()

chunk_require_cxx20(multi-mat PRIVATE)

# The v1 (Chunk<T>) variant at the default static size, not the bare `chunk`
# target: that one carries no variant define (the std::vector shape of the
# library) while this example instantiates ArrayChunk<double> itself, so its
# per-rank allocators and candidate lists were never set up -- every write
# from a rank other than 0 died in get_chunk_allocator().
target_link_libraries(multi-mat PRIVATE chunk_v1_${CHUNK_DEFAULT_SIZE})

# The compute loop is std::cos/sin/tan under `omp simd` (LoopsSimd, on the
# std::vector path): without a vector libm it does not vectorise at all --
# see chunk_enable_fast_math in examples/CMakeLists.txt. multi-mat compares
# nothing bit for bit (it prints its result), so fast math is fine here.
chunk_enable_fast_math(multi-mat)

add_test(NAME multi_mat COMMAND multi-mat)

set_property(TEST multi_mat
        PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
