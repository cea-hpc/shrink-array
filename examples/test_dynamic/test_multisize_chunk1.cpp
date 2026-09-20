// Multi-tailles Chunk v1 : arrays de tailles differentes coexistant + writes paralleles + hostile.
#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;

static int seq_check(unsigned CS,int N,const char* tag){
    set_thread_chunk_size(CS);
    ArrayChunk<double> tab(1); tab.set_chunk_size(CS); tab.resize(N);
    std::vector<double> ref(N);
    for(int i=0;i<N;++i){ ref[i]=(i%300==0)?(1+(i%5)):0.0; tab[i]=ref[i]; }
    tab.make_block_compression();
    int m=0; for(int i=0;i<N;++i) if(tab[i]!=ref[i])++m;
    std::printf("  [seq %-8s] CS=%4u %s (%d)\n",tag,CS,m==0?"PASS":"FAIL",m); return m;
}
static int par_check(unsigned CS,int N,const char* tag){
    set_thread_chunk_size(CS);
    ArrayChunk<double> tab(1); tab.set_chunk_size(CS); tab.resize(N);
    std::vector<double> ref(N);
    for(int i=0;i<N;++i) ref[i]=(i%300==0)?(1+(i%5)):0.0;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) tab[i]=ref[i];
    tab.make_block_compression();
    int m=0; for(int i=0;i<N;++i) if(tab[i]!=ref[i])++m;
    std::printf("  [par %-8s] CS=%4u %s (%d)\n",tag,CS,m==0?"PASS":"FAIL",m); return m;
}
int main(){
    std::printf("=== MULTI-SIZE Chunk v1 ===\n");
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);
    int f=0;
    f+=seq_check(256,3000,"256"); f+=seq_check(512,3000,"512");
    f+=seq_check(256,1500,"256#2"); f+=seq_check(1024,5000,"1024");
    f+=par_check(256,4000,"256"); f+=par_check(512,4000,"512"); f+=par_check(1024,6000,"1024");
    // two live arrays interleaved
    set_thread_chunk_size(256); ArrayChunk<double> a(1); a.set_chunk_size(256); a.resize(4000);
    set_thread_chunk_size(512); ArrayChunk<double> b(1); b.set_chunk_size(512); b.resize(4000);
    std::vector<double> ra(4000),rb(4000);
    for(int i=0;i<4000;++i){ ra[i]=(i%300==0)?7.0:0.0; rb[i]=(i%250==0)?3.0:0.0; }
    for(int i=0;i<4000;++i){ a[i]=ra[i]; b[i]=rb[i]; }
    a.make_block_compression(); b.make_block_compression();
    int ma=0,mb=0; for(int i=0;i<4000;++i){ if(a[i]!=ra[i])++ma; if(b[i]!=rb[i])++mb; }
    std::printf("  [live] a256:%s(%d) b512:%s(%d)\n",ma==0?"PASS":"FAIL",ma,mb==0?"PASS":"FAIL",mb); f+=ma+mb;
    std::printf("=== %s ===\n", f==0?"ALL PASS":"SOME FAIL");
    return f?1:0;
}
