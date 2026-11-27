# Les trois variantes : `zero`, `zero_or_value`, `zero_any`

> English version: [variants.md](variants.md).

> État : fin septembre 2026 pour l'hôte (sections 1 à 4 : `shrink_bench` sur Grace, 64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, mémoire mesurée exactement, voir
> [mesures_performance_fr.md](mesures_performance_fr.md)) ; fin novembre 2026 pour le GPU
> (section 5 : `kokkos_shrink_bench` sur un GH200 et un MI300A). En cas de doute, c'est le code
> qui fait foi.

> Noms dans le code : chaque variante porte le nom de son namespace, `zero::Array`, `zero_or_value::Array`,
> `zero_any::Array`. Les documents plus anciens les appellent Chunk (ou v1), Chunk2 et Chunk2A (les anciens
> noms de classe `ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A` ont été retirés avant la 1.0, voir CHANGELOG.md).

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

| variante | bibliothèque (macro qu'elle définit) | un chunk est compressé s'il est uniforme à… | coût d'une constante |
|---|---|---|---|
| **`zero`** | `shrink_array::zero` (`CHUNK_VARIANT_ZERO`) | **zéro** (`T{}`) | rien : un tampon constant partagé |
| **`zero_or_value`** | `shrink_array::zero_or_value` (`CHUNK_VARIANT_ZERO_OR_VALUE`) | zéro, **ou** une **seconde valeur** propre au tableau (`set_second_value`) | rien au-delà du tableau |
| **`zero_any`** | `shrink_array::zero_any` (`CHUNK_VARIANT_ZERO_ANY`) | zéro, **ou n'importe quelle autre valeur**, choisie chunk par chunk | un tableau du registre des secondes valeurs (`second_value_registry.h`) par valeur distincte, partagé par tout le processus |

### Booléens : deux conteneurs

Un build `zero_or_value` a deux conteneurs de booléens, choisis par type, tableau par tableau.
Les deux sont la machinerie de `zero_or_value` : un chunk tout à `false` est comprimé vers la
première valeur, un chunk tout à `true` vers la seconde, qui vaut `true` par construction (pas de
`set_second_value()`), et un chunk comprimé ne coûte aucune donnée dans l'un comme dans l'autre.
Ils diffèrent par ce que contient un chunk **matérialisé** :

| conteneur | en-tête | un chunk matérialisé de S éléments | une écriture | écritures voisines par deux threads |
|---|---|---|---|---|
| `zero_or_value::Array<bool>` | `chunk/zero_or_value_array.h` | S octets, un par `bool` | un simple stockage d'octet | sûres, comme pour tout élément |
| `BitArray<>` (`zero_or_value::Array<Bit>`) | `chunk/bit_array.h` | S/8 octets, 8 `bool` par octet -- le rangement de `std::vector<bool>` | une lecture-modification-écriture atomique du mot de 64 bits qui contient le bit (builds OpenMP) | sûres : le mot est mis à jour atomiquement, aucune écriture n'est perdue |
| `std::vector<bool>`, pour comparaison | `<vector>` | tout le tableau, en bits, sans compression | une lecture-modification-écriture d'un mot, non atomique | **une course** : deux écritures dans un même mot peuvent en perdre une |

`BitArray` prend 8 fois moins de mémoire par élément d'un chunk matérialisé, pour une
écriture plus chère : c'est le conteneur des grands tableaux de booléens à nombreuses zones mélangées
(masques, indicateurs sur un maillage), écrits moins souvent que lus.
`zero_or_value::Array<bool>` est celui des booléens écrits dans des boucles chaudes, ou presque
uniformes : ses chunks constants ne coûtent rien non plus. Contrairement à `std::vector<bool>`,
`BitArray` n'est pas une spécialisation qui change le comportement d'`Array<bool>` à son
insu : c'est un type à part, et `zero_or_value::Array<bool>` reste un stockage d'octets.
L'élément se lit comme un `Bit`, qui se convertit vers et depuis `bool` ; comme pour tout
conteneur à proxy ici, un élément n'a pas d'adresse. Dans un build à géométrie dynamique, un
`BitArray` naît avec des chunks 8 fois plus grands que la taille du processus
(`packed_bit_birth_factor`), pour qu'un chunk contienne autant d'octets qu'un chunk de `bool`.

```cpp
shrink_array::zero_or_value::Array<bool> flags(n); // un octet par élément dans un chunk matérialisé
shrink_array::BitArray<>            mask(n);  // un bit par élément, 8 par octet
flags[i] = true;
mask[i]  = true;              // deux threads peuvent écrire des bits voisins en même temps
const bool b = std::as_const(mask)[i];
```

Le programme [examples/bool_arrays](../examples/bool_arrays/bool_arrays.cpp) écrit les deux depuis
tous les threads, éléments voisins compris, vérifie chaque valeur et affiche les données que garde
chacun.

Les builds `zero_any` fournissent aussi **`pattern::Array`**, pour les entiers : un membre de la
famille `zero_any` qui compresse, en plus de ses constantes, un chunk égal à un **motif partagé
plus un offset** -- la forme d'une connectivité de maillage, compressée à environ 1 % d'un vecteur
sur un maillage structuré. Voir [algo_pattern_fr.md](algo_pattern_fr.md).

## 3. Ce que coûte un chunk

Tout chunk coûte ses **métadonnées**, qu'il soit compressé ou non. Pour les trois variantes,
l'objet chunk lui-même fait 16 octets dans un build OpenMP (le pointeur, puis les portes de
lecture et d'écriture par chunk). S'y ajoutent, par chunk :

- un octet « déjà en liste de compression » (compression par bloc) ;
- deux octets pour le *dirty bitset* du tableau (un bit par chunk, dans des mots de 64 bits
  complétés chacun à 128 octets contre le faux partage), et une part des listes de suivi
  (l'advisor compte environ 3 octets pour l'ensemble) ;
- pour **`zero_any`** : un octet de type de constante et un emplacement pour la seconde valeur
  du chunk, de `sizeof(T)` octets (8 pour un `double`), ou de 4 octets avec l'option CMake
  `CHUNK_SECOND_VALUE_INDEXED`.

C'est ce coût qui rend les **petits chunks** chers en mémoire (4 fois plus de métadonnées à
64 éléments qu'à 256), et qui justifie la géométrie dynamique : partir de petits chunks, qui
compressent finement, puis grossir quand les métadonnées dominent.

## 4. Laquelle choisir

| les données sont… | variante |
|---|---|
| surtout nulles, avec des zones de données quelconques | **`zero`** : la plus simple, et aucune métadonnée de seconde valeur |
| nulles ou égales à **une** autre valeur dominante, connue du tableau | **`zero_or_value`** |
| faites de nombreuses **zones uniformes à des valeurs différentes** | **`zero_any`** |

Ordre de grandeur sur `shrink_bench` (tableaux de 10⁹ éléments, 64 threads), mémoire en fin
de run :

| variante | statique 256 | dynamique (départ à 64, tous les tableaux confiés au décideur) |
|---|---|---|
| `zero_any` | 7,26 Go | **6,0 Go** |
| `zero_or_value` | 17,3 Go | **16,3 Go** |
| `zero` | 40,2 Go | 40,3 Go (et 39,7 Go en statique 1 024, la taille qu'il atteint) |
| `std::vector` (`shrink_bench_omp_vector_perf`) | 125,9 Go | — |

La géométrie dynamique fait gagner nettement `zero_or_value` et `zero_any`. Pour `zero`, dont la mémoire est
surtout faite de données, elle fait jeu égal avec le statique 256 : voir les limites connues
dans [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md).

## 5. Sur GPU

Les trois mêmes variantes tournent sur GPU (CUDA ou HIP, en-têtes seuls, la cible CMake
`shrink_array::device`) : `zero::DeviceArray<T, S = 1024>`, `zero_or_value::DeviceArray`,
`zero_any::DeviceArray` (`chunk/device/zero_array.h`, ...), et leurs façades Kokkos
`zero::KokkosArray`, ... ([api_fr.md](api_fr.md) section 11). Un chunk se replie selon la même
règle que sur l'hôte ; c'est la machinerie qui change : un descripteur par chunk qui porte sa
constante, des écritures qui matérialisent un chunk une fois, et la passe de compression à la fin
d'une phase d'écriture (`finish()`, `compress()`) au lieu de pendant ([conception_gpu_fr.md](conception_gpu_fr.md)).
S = 1024 est la géométrie GPU.

`kokkos_shrink_bench` (`examples/kokkos`) : le motif d'accès de shrink_bench sur cinq tableaux de
2^30 `double`, écrits par des noyaux Kokkos à travers la façade, contre cinq `Kokkos::View<double*>` ;
2000 itérations de fenêtres d'au plus 107374 éléments (petites fenêtres) ou 200 d'au plus 10^7
(grandes fenêtres), `compress()` toutes les 10. GH200 (H100, nvcc 13.3) et MI300A (ROCm 10.2),
novembre 2026 (675d0d701) ; les sommes de contrôle égalent partout celles du run plat.

| | `zero_any` | `zero_or_value` | `zero` |
|---|---|---|---|
| mémoire à la fin, petites fenêtres (plat : 42,95 Go) | **1,51 Go (3,5 %)** | 4,97 Go (11,6 %) | 19,17 Go (44,6 %) |
| mémoire à la fin, grandes fenêtres | **8,08 Go (18,8 %)** | 11,35 Go (26,4 %) | 25,89 Go (60,3 %) |
| boucle / plat, petites fenêtres, GH200 / MI300A | 1,49x / 1,50x | 1,47x / 1,49x | **1,36x / 1,38x** |
| boucle / plat, grandes fenêtres, GH200 / MI300A | 1,42x / 1,57x | 1,30x / 1,49x | **1,24x / 1,40x** |
| allocation + remplissage / plat, GH200 / MI300A | 0,65x / 0,05x | 0,74x / 0,13x | 1,06x / 0,52x |

La mémoire tenue est la même à l'octet sur les deux devices : elle dépend de ce qu'écrit le banc,
pas du device (les arènes gardent en plus 0,3 à 2,3 Go de tampons libres pour les phases
suivantes). Le rapport de la boucle est pris contre des tableaux plats déjà alloués et remplis ;
l'allocation et le remplissage des tableaux compressés coûtent moins que ceux des vues plates (sur
le MI300A, 20 fois moins pour `zero_any`, les 43 Go des vues étant mappés à leur allocation). Les
20 passes de lecture de chaque itération vont à la vitesse du plat (0,93x à 1,02x) ; ce que paie la
boucle, ce sont les écritures dans la fenêtre et autour, et `compress()` (environ 0,6 ms par passe
pour les cinq tableaux).

Le même travail sur les 72 cœurs de Grace (`host_twin_bench`, les tableaux de l'hôte, OpenMP),
`zero_or_value` : la boucle du GH200 est 3,1x plus rapide sur les petites fenêtres et 25x sur les
grandes (le GPU y est limité par la bande passante, et les lectures y sont celles du plat),
[conception_gpu_fr.md](conception_gpu_fr.md) section 13.

Laquelle sur GPU : la même règle que sur l'hôte (section 4). `zero_any` tient le moins de mémoire
pour 9 à 15 % de temps de boucle en plus que `zero` ; `zero` est la plus rapide et tient le plus ;
`zero_or_value` est entre les deux quand une valeur domine en plus de zéro. Écrivez les boucles
régulières par le chemin aligné sur les chunks (`parallel_for_chunks`) quand c'est possible ;
`Writer::assign` (le `w(i) = v` de Kokkos) est le chemin général ([gpu_assign_fr.md](gpu_assign_fr.md)).

## 6. Où lire la suite

| sujet | document |
|---|---|
| les tableaux GPU : conception, chemin d'écriture, mesures | [conception_gpu_fr.md](conception_gpu_fr.md), [gpu_assign_fr.md](gpu_assign_fr.md) |
| lecture, écriture, parallélisme, compression en vol, opérations sur plages | [algo_zero_or_value_fr.md](algo_zero_or_value_fr.md) : décrit `zero_or_value`, mais la machinerie de ses §5 à §8 et §10 est commune aux trois variantes |
| ce que `zero` et `zero_any` font autrement | [algo_zero_fr.md](algo_zero_fr.md), [algo_zero_any_fr.md](algo_zero_any_fr.md) |
| géométrie dynamique et rechunk, pour les trois variantes | [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) |
| synchronisation sans verrou (Dekker asymétrique) | [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) |
| réglages | [variables_environnement_fr.md](variables_environnement_fr.md) |
| index de la documentation | [README.md](README.md) |
