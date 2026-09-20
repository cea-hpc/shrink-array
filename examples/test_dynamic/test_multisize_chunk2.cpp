// Multi-tailles Chunk2 : arrays de tailles differentes coexistant.
#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;

static int build_check(unsigned int CS, int N, const char* tag)
{
    set_thread_chunk_size(CS);
    ArrayChunk2<double> tab(1);
    tab.set_chunk_size(CS);
    tab.set_second_value(9.0, false);   // Chunk2 dual-value
    tab.resize(N);
    std::vector<double> ref(N);
    for (int i = 0; i < N; ++i) { double v = (i%500==0)?1.0:0.0; ref[i]=v; tab[i]=v; }
    tab.make_block_compression();
    int mism=0; for (int i=0;i<N;++i) if (tab[i]!=ref[i]) ++mism;
    std::printf("  [%-10s] CS=%4u chunk_size=%4u  %s (%d/%d)\n",
                tag, CS, tab.chunk_size(), mism==0?"PASS":"FAIL", mism, N);
    return mism;
}

int main()
{
    std::printf("=== MULTI-SIZE Chunk2, __chunk_size=%u ===\n",(unsigned)__chunk_size);
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);
    int fails=0;
    fails += build_check(256, 3000, "A size=256");
    fails += build_check(512, 3000, "B size=512");
    fails += build_check(256, 1500, "C size=256");
    fails += build_check(1024, 5000, "D size=1024");
    std::printf("=== %s ===\n", fails==0?"ALL PASS":"SOME FAIL");
    return fails==0?0:1;
}
