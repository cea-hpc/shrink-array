// MULTI-TAILLES + WRITES EN REGION PARALLELE : le vrai stress de la TLS-par-thread.
#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;

template <typename T>
static int par_build_check(unsigned int CS, int N, const char* tag)
{
    set_thread_chunk_size(CS);
    ArrayChunk2A<T> tab(1);
    tab.set_chunk_size(CS);
    tab.resize(N);
    std::vector<T> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = (i % 300 == 0) ? static_cast<T>(1 + (i % 5)) : static_cast<T>(0);

    // WRITES IN A PARALLEL REGION — worker threads must route to size CS.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        tab[i] = ref[i];

    ArrayChunk2A<T>::make_block_compression();
    int mism = 0;
    for (int i = 0; i < N; ++i) if (tab[i] != ref[i]) ++mism;
    std::printf("  [%-10s] CS=%4u chunk_size=%4u  %s (%d/%d)\n",
                tag, CS, tab.chunk_size(), mism==0?"PASS":"FAIL", mism, N);
    return mism;
}

int main()
{
    std::printf("=== MULTI-SIZE + PARALLEL WRITES (Chunk2A), __chunk_size=%u ===\n",(unsigned)__chunk_size);
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);
    int fails = 0;
    fails += par_build_check<double>(256, 4000, "size=256");
    fails += par_build_check<double>(512, 4000, "size=512");
    fails += par_build_check<double>(1024, 6000, "size=1024");
    fails += par_build_check<double>(256, 2000, "size=256#2");
    std::printf("=== %s ===\n", fails==0?"ALL PASS":"SOME FAIL");
    return fails==0?0:1;
}
