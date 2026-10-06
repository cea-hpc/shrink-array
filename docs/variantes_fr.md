# Les trois variantes : Chunk, Chunk2, Chunk2A

> English version: [variants.md](variants.md).

> État : fin septembre 2026. Les chiffres viennent de `shrink_bench` sur Grace (64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), mémoire mesurée exactement (voir
> [mesures_performance_fr.md](mesures_performance_fr.md)). En cas de doute, c'est le code qui
> fait foi.

***

## 1. Le principe commun

Un tableau est découpé en **chunks** de `S` éléments consécutifs. Un chunk est soit
**matérialisé** (il possède un tampon de `S` éléments), soit **compressé** : il ne stocke
qu'une constante, et ne coûte alors que ses métadonnées. Une lecture d'un chunk compressé
rend la constante ; une écriture d'une autre valeur le **matérialise** d'abord. Des passes de
compression rendent ensuite leur tampon aux chunks redevenus uniformes.

Les trois variantes partagent cette mécanique : les proxys qui interceptent lectures et
écritures, les écritures différées pendant qu'une compression est en cours, les passes de
compression par bloc, les allocateurs par rang (`fast_allocators`), et la géométrie
dynamique. Elles diffèrent par **ce qu'elles savent compresser**.

## 2. Ce que chacune sait compresser

| variante | build | un chunk est compressé s'il est uniforme à… | coût d'une constante |
|---|---|---|---|
| **Chunk** (v1) | `-DCHUNK=1` | **zéro** (`T{}`) | rien : un tampon constant partagé |
| **Chunk2** | `-DCHUNK2=1` | zéro, **ou** une **seconde valeur** propre au tableau (`set_second_value`) | rien au-delà du tableau |
| **Chunk2A** | `-DCHUNK2A=1` | zéro, **ou n'importe quelle autre valeur**, choisie chunk par chunk | un tableau du registre des secondes valeurs (`tab_cst_val2_holder.h`) par valeur distincte, partagé par tout le processus |

Les tableaux de booléens des builds Chunk2 peuvent en plus utiliser `ArrayChunkBits`, qui
range 8 booléens par octet dans un chunk matérialisé.

## 3. Ce que coûte un chunk

Tout chunk coûte ses **métadonnées**, qu'il soit compressé ou non. Pour les trois variantes,
l'objet chunk lui-même fait 16 octets. S'y ajoutent, par chunk :

- un octet « déjà en liste de compression » (compression par bloc) ;
- un bit dans le *dirty bitset* du tableau, et une part des listes de suivi (l'advisor compte
  environ 3 octets pour l'ensemble) ;
- pour **Chunk2A** : un octet de type de constante et un emplacement pour la seconde valeur
  du chunk, de `sizeof(T)` octets (8 pour un `double`), ou de 4 octets avec l'option CMake
  `CHUNK_SECOND_VALUE_INDEXED`.

C'est ce coût qui rend les **petits chunks** chers en mémoire (4 fois plus de métadonnées à
64 éléments qu'à 256), et qui justifie la géométrie dynamique : partir de petits chunks, qui
compressent finement, puis grossir quand les métadonnées dominent.

## 4. Laquelle choisir

| les données sont… | variante |
|---|---|
| surtout nulles, avec des zones de données quelconques | **Chunk** : la plus simple, et aucune métadonnée de seconde valeur |
| nulles ou égales à **une** autre valeur dominante, connue du tableau | **Chunk2** |
| faites de nombreuses **zones uniformes à des valeurs différentes** | **Chunk2A** |

Ordre de grandeur sur `shrink_bench` (tableaux de 10⁹ éléments, 64 threads), mémoire en fin
de run :

| variante | statique 256 | dynamique (départ à 64, tous les tableaux confiés au décideur) |
|---|---|---|
| Chunk2A | 7,26 Go | **6,0 Go** |
| Chunk2 | 17,3 Go | **16,3 Go** |
| Chunk (v1) | 40,2 Go | 40,3 Go (et 39,7 Go en statique 1 024, la taille qu'il atteint) |

La géométrie dynamique fait gagner nettement Chunk2 et Chunk2A. Pour v1, dont la mémoire est
surtout faite de données, elle fait jeu égal avec le statique 256 : voir les limites connues
dans [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md).

## 5. Où lire la suite

| sujet | document |
|---|---|
| lecture, écriture, parallélisme, compression en vol, opérations sur plages | [algo_chunk2_fr.md](algo_chunk2_fr.md) : décrit Chunk2, mais la machinerie de ses §5 à §8 et §10 est commune aux trois variantes |
| ce que Chunk (v1) et Chunk2A font autrement | [algo_chunk_fr.md](algo_chunk_fr.md), [algo_chunk2a_fr.md](algo_chunk2a_fr.md) |
| géométrie dynamique et rechunk, pour les trois variantes | [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) |
| synchronisation sans verrou (Dekker asymétrique) | [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) |
| réglages | [variables_environnement_fr.md](variables_environnement_fr.md) |
| index de la documentation | [README.md](README.md) |
