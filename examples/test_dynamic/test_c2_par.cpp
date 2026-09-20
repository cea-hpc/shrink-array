// Chunk2 multi-tailles + writes en region parallele.
#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;
static int par_check(unsigned int CS, int N, const char* tag){
    set_thread_chunk_size(CS);
    ArrayChunk2<double> tab(1); tab.set_chunk_size(CS); tab.set_second_value(9.0,false); tab.resize(N);
    std::vector<double> ref(N);
    for(int i=0;i<N;++i) ref[i]=(i%300==0)?(1+(i%5)):0.0;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) tab[i]=ref[i];
    tab.make_block_compression();
    int m=0; for(int i=0;i<N;++i) if(tab[i]!=ref[i])++m;
    std::printf("  [%-9s] CS=%4u %s (%d/%d)\n", tag, CS, m==0?"PASS":"FAIL", m, N);
    return m;
}
int main(){
    std::printf("=== Chunk2 MULTI-SIZE + PARALLEL WRITES ===\n");
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);
    int f=0;
    f+=par_check(256,4000,"size=256");
    f+=par_check(512,4000,"size=512");
    f+=par_check(1024,6000,"size=1024");
    f+=par_check(256,2000,"size=256#2");
    std::printf("=== %s ===\n", f==0?"ALL PASS":"SOME FAIL");
    return f==0?0:1;
}
