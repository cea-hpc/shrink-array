#include <cstdio>
#include <vector>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"
using namespace shrink_array;
int __mem_page_size = 0;
int main(){
    std::printf("=== TWO LIVE Chunk2 ARRAYS, DIFFERENT SIZES, INTERLEAVED ===\n");
    set_active_chunk_size(256);
    ParamChunk::init_indexes(true);
    const int N=4000;
    set_thread_chunk_size(256);
    ArrayChunk2<double> a(1); a.set_chunk_size(256); a.set_second_value(9.0,false); a.resize(N);
    set_thread_chunk_size(512);
    ArrayChunk2<double> b(1); b.set_chunk_size(512); b.set_second_value(9.0,false); b.resize(N);
    std::vector<double> ra(N), rb(N);
    for(int i=0;i<N;++i){ ra[i]=(i%300==0)?7.0:0.0; rb[i]=(i%250==0)?3.0:0.0; }
    for(int i=0;i<N;++i){ a[i]=ra[i]; b[i]=rb[i]; }   // interleaved, per-op TLS routing
    a.make_block_compression();
    b.make_block_compression();
    int ma=0,mb=0; for(int i=0;i<N;++i){ if(a[i]!=ra[i])++ma; if(b[i]!=rb[i])++mb; }
    std::printf("  a(256): %s (%d)   b(512): %s (%d)\n", ma==0?"PASS":"FAIL",ma, mb==0?"PASS":"FAIL",mb);
    std::printf("=== %s ===\n",(ma==0&&mb==0)?"ALL PASS":"SOME FAIL");
    return (ma==0&&mb==0)?0:1;
}
