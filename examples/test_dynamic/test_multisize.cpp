// Preuve MULTI-TAILLES : deux arrays de tailles de chunk differentes
// coexistant dans le meme run.
#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;

template <typename T>
static int build_check(unsigned int CS, int N, const char* tag)
{
    set_thread_chunk_size(CS);               // this thread operates at size CS
    ArrayChunk2A<T> tab(1);
    tab.set_chunk_size(CS);                   // registers CS + sets TLS
    tab.resize(N);
    std::vector<T> ref(N);
    for (int i = 0; i < N; ++i) {
        T v = (i % 300 == 0) ? static_cast<T>(1 + (i % 5)) : static_cast<T>(0);
        ref[i] = v; tab[i] = v;
    }
    ArrayChunk2A<T>::make_block_compression();
    int mism = 0;
    for (int i = 0; i < N; ++i) if (tab[i] != ref[i]) ++mism;
    std::printf("  [%-10s] CS=%4u chunk_size=%4u  %s (%d/%d)\n",
                tag, CS, tab.chunk_size(), mism==0?"PASS":"FAIL", mism, N);
    return mism;
}

int main()
{
    std::printf("=== MULTI-SIZE proof (Chunk2A), __chunk_size=%u ===\n",(unsigned)__chunk_size);
    // Register both sizes up front (sequential) via init_indexes seed = first size.
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);

    int fails = 0;
    // Two arrays, two different chunk sizes, same run:
    fails += build_check<double>(256, 3000, "A size=256");
    fails += build_check<double>(512, 3000, "B size=512");
    // Interleave a third to stress routing:
    fails += build_check<double>(256, 1500, "C size=256");
    fails += build_check<double>(1024, 5000, "D size=1024");

    std::printf("=== %s ===\n", fails==0?"ALL PASS":"SOME FAIL");
    return fails==0?0:1;
}
