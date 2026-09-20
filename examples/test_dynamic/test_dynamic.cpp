// =====================================================================
// Preuve DynamicChunkSize sur Chunk2A.
//
// Cree un ArrayChunk2A dont la taille de chunk est fixee au RUNTIME
// (CS != __chunk_size compile-time), ecrit un motif melant chunks
// constants (compresses) et chunks reels, force la compression, relit
// tout et verifie l'exactitude.
//
// Prerequis (build) : -DCHUNK2A=1 -DCHUNK_DYNAMIC
//   Sans -DCHUNK_DYNAMIC, set_chunk_size() est un no-op (mode statique)
//   et la taille reste __chunk_size ; le test passe aussi (CS ignoree).
//
// LIMITE CONNUE : une seule taille active par process (g_active_chunk_size
// est global, l'allocateur est initialise une fois par init_indexes).
// Pour tester plusieurs tailles, lancer un process par taille.
// =====================================================================
#include <cstdio>
#include "chunk/AllTypeChunk.h"
#include "chunk/param_chunk.h"

using namespace shrink_array;

// Globale attendue par le TU allocateur (comme le driver shrink_bench).
int __mem_page_size = 0;

template <typename T>
static int run_one(unsigned int CS, const char* label)
{
    const int N = 2000;

    ArrayChunk2A<T> tab(1);
    tab.set_chunk_size(CS);   // runtime en dynamique ; no-op en statique
    tab.resize(N);

    for (int i = 0; i < N; ++i)
        tab[i] = (i % 500 == 0) ? static_cast<T>(1) : static_cast<T>(0);

    ArrayChunk2A<T>::make_block_compression();

    int mism = 0, first = -1;
    for (int i = 0; i < N; ++i)
    {
        T exp = (i % 500 == 0) ? static_cast<T>(1) : static_cast<T>(0);
        if (tab[i] != exp) { if (first < 0) first = i; ++mism; }
    }
    std::printf("  [%-6s] chunk_size=%4u  %s  (mismatches=%d, first=%d)\n",
                label, tab.chunk_size(), mism == 0 ? "PASS" : "FAIL", mism, first);
    return mism;
}

int main()
{
    unsigned int CS = 512;   // <-- taille de chunk runtime a prouver

    // La taille active doit etre fixee AVANT init_indexes (l'allocateur
    // se dimensionne dessus).
    set_active_chunk_size(CS);
    ParamChunk::init_indexes(true);

    std::printf("=== DynamicChunkSize (Chunk2A) — compile __chunk_size=%u, runtime CS=%u ===\n",
                (unsigned)__chunk_size, CS);

    int fails = 0;
    fails += run_one<double>(CS, "double");
    fails += run_one<int>(CS, "int");

    std::printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "SOME FAIL");
    return fails == 0 ? 0 : 1;
}
