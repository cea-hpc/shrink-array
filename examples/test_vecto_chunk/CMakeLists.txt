add_executable(test_vecto_chunk test_vecto_chunk.cpp)

if (OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_vecto_chunk PUBLIC OpenMP::OpenMP_CXX)
endif ()
if (CMAKE_CXX_COMPILER MATCHES "icx" OR CMAKE_CXX_COMPILER MATCHES "icpx")
    target_compile_options(test_vecto_chunk PRIVATE -O3 -mavx2)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_options(test_vecto_chunk PRIVATE -O3 -mavx2 -qopt-report=5 -qopt-report-file=vectorization_report.txt)
endif ()
chunk_require_cxx20(test_vecto_chunk PRIVATE)

# test_vecto_chunk doesn't use CHUNK/CHUNK2/CHUNK2A → chunk_vector.
# The v1 (Chunk<T>) variant at the default static size, not the bare
# `chunk_vector` target: that one carries no variant define (the std::vector
# shape of the library) while this example instantiates ArrayChunk<double>
# itself -- the same mismatch multi-mat had (its allocators and lists never
# set up: SIGSEGV in create_value from the first parallel write).
target_link_libraries(test_vecto_chunk PRIVATE chunk_v1_${CHUNK_DEFAULT_SIZE})

add_test(NAME test_vecto_chunk COMMAND test_vecto_chunk)

set_property(TEST test_vecto_chunk
        PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)

set_property(TEST test_vecto_chunk
        PROPERTY ENVIRONMENT OMP_NUM_THREADS=8)
