# Les trois variantes : `zero`, `zero_one_value`, `zero_any`

> English version: [variants.md](variants.md).

> État : fin septembre 2026. Les chiffres viennent de `shrink_bench` sur Grace (64 threads,
> `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), mémoire mesurée exactement (voir
> [mesures_performance_fr.md](mesures_performance_fr.md)). En cas de doute, c'est le code qui
> fait foi.

> Noms dans le code : chaque variante porte le nom de son namespace, `zero::Array`, `zero_one_value::Array`,
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
| **`zero`** | `chunk::chunk_zero` (`CHUNK_VARIANT_ZERO`) | **zéro** (`T{}`) | rien : un tampon constant partagé |
| **`zero_one_value`** | `chunk::chunk_zero_one_value` (`CHUNK_VARIANT_ZERO_ONE_VALUE`) | zéro, **ou** une **seconde valeur** propre au tableau (`set_second_value`) | rien au-delà du tableau |
| **`zero_any`** | `chunk::chunk_zero_any` (`CHUNK_VARIANT_ZERO_ANY`) | zéro, **ou n'importe quelle autre valeur**, choisie chunk par chunk | un tableau du registre des secondes valeurs (`tab_cst_val2_holder.h`) par valeur distincte, partagé par tout le processus |

### Booléens : deux conteneurs

Un build `zero_one_value` a deux conteneurs de booléens, choisis par type, tableau par tableau.
Les deux sont la machinerie de `zero_one_value` : un chunk tout à `false` est comprimé vers la
première valeur, un chunk tout à `true` vers la seconde, qui vaut `true` par construction (pas de
`set_second_value()`), et un chunk comprimé ne coûte aucune donnée dans l'un comme dans l'autre.
Ils diffèrent par ce que contient un chunk **matérialisé** :

| conteneur | en-tête | un chunk matérialisé de S éléments | une écriture | écritures voisines par deux threads |
|---|---|---|---|---|
| `zero_one_value::Array<bool>` | `chunk/zero_one_value_array.h` | S octets, un par `bool` | un simple stockage d'octet | sûres, comme pour tout élément |
| `ArrayChunkBits<>` (`zero_one_value::Array<Bit>`) | `chunk/array_chunk_bits.h` | S/8 octets, 8 `bool` par octet -- le rangement de `std::vector<bool>` | une lecture-modification-écriture atomique du mot de 64 bits qui contient le bit (builds OpenMP) | sûres : le mot est mis à jour atomiquement, aucune écriture n'est perdue |
| `std::vector<bool>`, pour comparaison | `<vector>` | tout le tableau, en bits, sans compression | une lecture-modification-écriture d'un mot, non atomique | **une course** : deux écritures dans un même mot peuvent en perdre une |

`ArrayChunkBits` prend 8 fois moins de mémoire par élément d'un chunk matérialisé, pour une
écriture plus chère : c'est le conteneur des grands tableaux de booléens à nombreuses zones mélangées
(masques, indicateurs sur un maillage), écrits moins souvent que lus.
`zero_one_value::Array<bool>` est celui des booléens écrits dans des boucles chaudes, ou presque
uniformes : ses chunks constants ne coûtent rien non plus. Contrairement à `std::vector<bool>`,
`ArrayChunkBits` n'est pas une spécialisation qui change le comportement d'`Array<bool>` à son
insu : c'est un type à part, et `zero_one_value::Array<bool>` reste un stockage d'octets.
L'élément se lit comme un `Bit`, qui se convertit vers et depuis `bool` ; comme pour tout
conteneur à proxy ici, un élément n'a pas d'adresse. Dans un build à géométrie dynamique, un
`ArrayChunkBits` naît avec des chunks 8 fois plus grands que la taille du processus
(`packed_bit_birth_factor`), pour qu'un chunk contienne autant d'octets qu'un chunk de `bool`.

```cpp
shrink_array::zero_one_value::Array<bool> flags(n); // un octet par élément dans un chunk matérialisé
shrink_array::ArrayChunkBits<>            mask(n);  // un bit par élément, 8 par octet
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
| nulles ou égales à **une** autre valeur dominante, connue du tableau | **`zero_one_value`** |
| faites de nombreuses **zones uniformes à des valeurs différentes** | **`zero_any`** |

Ordre de grandeur sur `shrink_bench` (tableaux de 10⁹ éléments, 64 threads), mémoire en fin
de run :

| variante | statique 256 | dynamique (départ à 64, tous les tableaux confiés au décideur) |
|---|---|---|
| `zero_any` | 7,26 Go | **6,0 Go** |
| `zero_one_value` | 17,3 Go | **16,3 Go** |
| `zero` | 40,2 Go | 40,3 Go (et 39,7 Go en statique 1 024, la taille qu'il atteint) |
| `std::vector` (`shrink_bench_omp_vector_perf`) | 125,9 Go | — |

La géométrie dynamique fait gagner nettement `zero_one_value` et `zero_any`. Pour `zero`, dont la mémoire est
surtout faite de données, elle fait jeu égal avec le statique 256 : voir les limites connues
dans [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md).

## 5. Où lire la suite

| sujet | document |
|---|---|
| lecture, écriture, parallélisme, compression en vol, opérations sur plages | [algo_zero_one_value_fr.md](algo_zero_one_value_fr.md) : décrit `zero_one_value`, mais la machinerie de ses §5 à §8 et §10 est commune aux trois variantes |
| ce que `zero` et `zero_any` font autrement | [algo_zero_fr.md](algo_zero_fr.md), [algo_zero_any_fr.md](algo_zero_any_fr.md) |
| géométrie dynamique et rechunk, pour les trois variantes | [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) |
| synchronisation sans verrou (Dekker asymétrique) | [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) |
| réglages | [variables_environnement_fr.md](variables_environnement_fr.md) |
| index de la documentation | [README.md](README.md) |
