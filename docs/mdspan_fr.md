# Une vue multidimensionnelle : mdspan

> English version: [mdspan.md](mdspan.md).

> Octobre 2026. `chunk/mdspan.h` pose un `mdspan` C++23 sur un tableau de la bibliothèque. Le
> tableau garde son rôle, le remplaçant de `std::vector` ; la vue ajoute `f[i, j]`.

***

## 1. Ce que c'est

Le tableau reste le propriétaire : il porte les éléments, les compresse, les re-chunke et garde
toute son interface. La vue est un `mdspan` dont l'accesseur passe par l'`operator[]` du
tableau : `f[i, j] = v` est `a[k] = v`, avec `k` donné par le layout -- le même proxy
d'écriture, la même compression, les mêmes règles.

```cpp
#include "chunk/array_chunk2A.h"
#include "chunk/mdspan.h"

namespace sa = shrink_array;

sa::ArrayChunk2A<double> a(nx * ny);
auto f = sa::md::view(a, nx, ny);         // ligne par ligne : f[i, j] est a[i * ny + j]
f[i, j] = 1.0;
f[i, j] += 2.0;                           // opérateurs composés, par le proxy
double x = f[i, j];                       // une valeur, pas auto (voir section 5)

const auto& ca = a;
auto cf = sa::md::view(ca, nx, ny);       // lecture seule : f[i, j] est un double
```

| appel | donne |
|---|---|
| `md::view(a, e0, e1, ...)` | un `mdspan` en `layout_right` (ligne par ligne) d'étendues `e0, e1, ...` ; leur produit ne doit pas dépasser `a.size()` |
| `md::tiled_view<T0, T1, ...>(a, e0, e1, ...)` | un `mdspan` en `md::layout_tiled<T0, T1, ...>` : des tuiles de `T0 x T1 x ...` éléments rangées l'une après l'autre |
| `md::tiled_size<T0, T1, ...>(e0, e1, ...)` | la taille de tableau qu'il faut à cette vue en tuiles, bourrage des dernières tuiles compris |
| `md::view_type<Array, Layout, Rank>` | le type de ces vues ; `view_type<const Array, ...>` est celui en lecture seule, et se convertit depuis l'autre |
| `md::accessor<Array>`, `md::layout_tiled<...>` | l'accesseur et le layout, pour construire un `mdspan` à la main |

## 2. Les tuiles : pourquoi le layout compte pour la compression

Un chunk est une suite d'éléments consécutifs (256 dans une bibliothèque `_256`). En
`layout_right` c'est un morceau de ligne : une région 2D uniforme ne donne des chunks uniformes
que là où elle fait au moins 256 cellules de large. Avec `layout_tiled<16, 16>`, la tuile `k`
est le chunk `k` : un carré de 16 x 16 dans un seul matériau est un chunk uniforme, quelle que
soit la largeur du matériau. Choisir la tuile pour que son nombre d'éléments soit la taille de
chunk : 16 x 16 pour 256, 32 x 32 pour 1024, 8 x 8 x 4 pour 256 en 3D.

`examples/mdspan_tiles` écrit le même champ (un fond uniforme, des disques et des bandes étroites
d'un matériau chacun, et un coin bruité qui ne se compresse dans aucun des deux layouts) par les
deux vues d'un tableau `chunk::chunk_v2a_256`, puis compte les chunks restés matérialisés après
la passe de compression :

| champ de 2048 x 2048 doubles (32 Mio) | chunks | matérialisés | valeurs gardées |
|---|---|---|---|
| ligne par ligne (`layout_right`) | 16385 | 7449 (45,5 %) | 14,5 Mio |
| tuiles 16 x 16 (`layout_tiled<16, 16>`) | 16385 | 2464 (15,0 %) | 4,8 Mio |

1521 des 2464 tuiles restantes sont le coin bruité ; les autres chevauchent une frontière entre
matériaux. Le chiffre dépend du champ : un champ fait de couches plus larges qu'un chunk ne gagne
rien aux tuiles, un champ de petites taches y gagne le plus.

## 3. Tuile par tuile : `for_each_tile`, `update_tiles`, `fill_box`

`f[i, j]` paie un `operator[]` par élément. Dans une vue en tuiles une tuile est une suite
d'éléments consécutifs -- un chunk quand elle a `chunk_size()` éléments -- : `chunk/mdspan_tile_ops.h`
la lit d'un coup (une copie de bloc, ou la constante d'un chunk compressé) et la réécrit d'un coup.

```cpp
#include "chunk/mdspan_tile_ops.h"

auto f = sa::md::tiled_view<16, 16>(a, nx, ny);
double s = 0;
sa::md::for_each_tile(std::as_const(f),
    [&](const auto& t, auto b) {                  // une tuile de plusieurs valeurs
        for (long i = 0; i < t.extent[0]; ++i)
            for (long j = 0; j < t.extent[1]; ++j) s += b[i, j];
    },
    [&](const auto& t, double v) {                // une tuile uniforme : un appel
        s += v * double(t.extent[0] * t.extent[1]);
    });

sa::md::update_tiles(f,                           // f = 0.5 f + 1
    [](const auto&, auto b) {
        for (long i = 0; i < 16; ++i)
            for (long j = 0; j < 16; ++j) b[i, j] = 0.5 * b[i, j] + 1.0;
    },
    [](const auto&, double& v) { v = 0.5 * v + 1.0; return true; });
```

| appel | fait |
|---|---|
| `md::for_each_tile(f, on_block, on_uniform)` | `on_block(t, b)` pour une tuile de plusieurs valeurs, `b` un `mdspan<const T, extents<16, 16>>` sur une copie de la tuile ; `on_uniform(t, v)` pour une tuile uniforme. Sans `on_uniform`, `on_block` reçoit toutes les tuiles |
| `md::update_tiles(f, on_block, on_uniform)` | la même chose, `b` modifiable ; la tuile n'est réécrite que si elle a changé, en une constante si elle en sort uniforme. `on_uniform(t, v)` change `v` et rend true, ou rend false pour recevoir la tuile en bloc rempli de `v` |
| `md::for_each_tile(f, first, last, ...)`, `md::update_tiles(f, first, last, ...)` | les tuiles `[first, last)` dans l'ordre de stockage, pour une boucle `#pragma omp for` sur `[0, md::tile_count(f))` |
| `md::fill_box(f, lo, hi, v)` | met la boîte `[lo, hi)`, rognée aux étendues, à `v`. Vue en tuiles : une tuile que la boîte couvre est remplie entière, bourrage compris, une range op par suite de tuiles couvertes consécutives ; une tuile qu'elle coupe est mise à jour comme par `update_tiles` (laissée telle quelle si elle est déjà uniforme à `v`). Vue ligne par ligne (`md::view`) : une range op par suite contiguë, plusieurs lignes d'un coup quand la boîte couvre des lignes entières |
| `md::fill_box(f, lo, hi, v, first, last)` | vue en tuiles, tuiles `[first, last)` seulement : chaque thread sa tranche de tuiles, dans une phase |
| `t.number`, `t.origin`, `t.extent` | le rang de la tuile dans l'ordre de stockage, le multi-indice de son premier élément, sa partie dans la vue (plus petite que la tuile sur les bords bourrés) |

Le bloc couvre toute la tuile, bourrage compris : boucler jusqu'à `t.extent` là où le bourrage ne
doit pas compter. Une tuile uniforme que `update_tiles` laisse inchangée n'est pas écrite : un noyau
qui ne touche pas au fond ne le matérialise pas. Une tuile modifiée est écrite par `fill_range()`
si elle est uniforme, sinon par `copy_range()` (les range ops séquentielles hors région
parallèle, les `_mt` dedans). Un chunk compressé qui change est matérialisé pour l'écriture et mis
en file : la fin de la phase, ou la passe de compression suivante, le recompresse. Le gain est le
nombre d'appels, pas la mémoire transitoire.

Dans une région parallèle chaque thread travaille sur ses propres tuiles, dans une phase pour
`update_tiles`, et une tuile doit être faite de chunks entiers : un nombre d'éléments multiple de
`chunk_size()`, vérifié (une tuile qui partage un chunk avec celle d'un autre thread arrête le
programme). Hors région, toute taille de chunk convient.

`examples/mdspan_tiles`, même champ de 2048 x 2048 en tuiles de 16 x 16 (85 % uniformes), build
Release, GCC 14, sur un conteneur à 2 cœurs (des ordres de grandeur, pas un banc) :

| noyau | threads | élément par élément, `f[i, j]` | tuile par tuile | rapport |
|---|---|---|---|---|
| somme du champ | 1 | 23,5 ms | 1,6 ms | x 14,6 |
| somme du champ | 2 | 11,4 ms | 1,35 ms | x 8,5 |
| `f = 0.5 f + 1` | 1 | 125,8 ms | 6,3 ms | x 19,9 |
| `f = 0.5 f + 1` | 2 | 64,1 ms | 4,4 ms | x 14,7 |
| 64 rectangles peints, `md::fill_box` | 1 | 103,3 ms | 8,8 ms | x 11,7 |
| 64 rectangles peints, `md::fill_box` | 2 | 72,0 ms | 7,0 ms | x 10,4 |

Chaque paire laisse le même champ et les mêmes chunks matérialisés : 2464 après la somme et la
mise à jour, 3109 après la peinture.

Le gain vient de ce qu'on évite l'accès élément par élément, pas du multi-indice, qui coûte peu :
le même champ sommé élément par élément, 1 thread, se lit à 2,8 ns par élément par `a[k]`,
3,3 ns par `f[i, j]` ligne par ligne et 3,5 ns par `f[i, j]` en tuiles (le mapping en tuiles
divise par ses étendues de tuile constantes en arithmétique non signée : décalages et masques).

## 4. D'où vient `mdspan`

| disponible | utilisé |
|---|---|
| `<mdspan>` avec `__cpp_lib_mdspan` (libstdc++ 15, libc++ 17 et suivantes) | `std::mdspan` |
| sinon, `<mdspan/mdspan.hpp>` dans le chemin d'inclusion (l'implémentation de référence, kokkos/mdspan) | son `mdspan`, dans l'espace de noms qu'il déclare |
| ni l'un ni l'autre | rien : l'en-tête définit seulement `SHRINK_ARRAY_HAS_MDSPAN` à 0 |

`f[i, j]` demande l'indexation multidimensionnelle de C++23 (GCC 12, Clang 15 et suivants) ; la
bibliothèque exige déjà C++23. Le harness et la démo sortent avec 77 sans `mdspan`, ce que ctest
compte comme sauté.

## 5. Règles et limites

- La vue ne possède pas le tableau : elle ne doit pas lui survivre. Après un `resize()`, refaire
  la vue ; un re-chunk la laisse correcte, mais un layout en tuiles n'est aligné sur les chunks
  que tant que la taille de chunk est le nombre d'éléments de la tuile.
- Les règles du tableau s'appliquent telles quelles : écritures en parallèle dans une phase
  OpenMP (`parallel_phase()`, voir [getting_started_fr.md](getting_started_fr.md)), aucun proxy
  d'élément gardé au-delà d'un point séquentiel. En build de contrôle `auto x = f[i, j];` ne
  compile pas, comme pour `a[i]` : écrire `double x = f[i, j];`.
- Ce n'est pas un pointeur : `data_handle()` est l'adresse du tableau, pas un `T*`, et
  l'accesseur n'a pas d'`offset()` -- pas de `submdspan`, pas de passage des données à du code
  qui attend une mémoire contiguë.
- `layout_tiled` n'est pas à pas constant (`is_strided()` est faux) et n'est exhaustif que si
  chaque étendue est un multiple de sa tuile ; le code qui exige `layout_right` ou
  un layout à pas prend la vue ligne par ligne.

## 6. Fichiers

| fichier | contenu |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::layout_tiled`, `md::view`, `md::tiled_view`, `md::tiled_size` |
| `examples/validation/mdspan_validation.cpp` | écritures par la vue et relecture linéaire, opérateurs composés, rang 3, vues en lecture seule, le mapping en tuiles (bijection, bourrage), une tuile uniforme est une suite uniforme ; toutes les variantes (un ctest chacune, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile`, `md::update_tiles`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocs, étendues des tuiles sur les bords bourrés, tuiles uniformes, la mise à jour contre la formule élément par élément, les tuiles non touchées restent compressées, la mise à jour en parallèle ; toutes les variantes à 1 et 8 threads, et `mdspan_tile_ops_misaligned` (une tuile plus petite qu'un chunk refusée dans une région parallèle) |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | les comparaisons des sections 2 et 3, et le coût du multi-indice ; `mdspan_tiles 4096` pour un champ plus grand |
