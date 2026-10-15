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
double x = f[i, j];                       // une valeur, pas auto (voir section 4)

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

## 3. D'où vient `mdspan`

| disponible | utilisé |
|---|---|
| `<mdspan>` avec `__cpp_lib_mdspan` (libstdc++ 15, libc++ 17 et suivantes) | `std::mdspan` |
| sinon, `<mdspan/mdspan.hpp>` dans le chemin d'inclusion (l'implémentation de référence, kokkos/mdspan) | son `mdspan`, dans l'espace de noms qu'il déclare |
| ni l'un ni l'autre | rien : l'en-tête définit seulement `SHRINK_ARRAY_HAS_MDSPAN` à 0 |

`f[i, j]` demande l'indexation multidimensionnelle de C++23 (GCC 12, Clang 15 et suivants) ; la
bibliothèque exige déjà C++23. Le harness et la démo sortent avec 77 sans `mdspan`, ce que ctest
compte comme sauté.

## 4. Règles et limites

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

## 5. Fichiers

| fichier | contenu |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::layout_tiled`, `md::view`, `md::tiled_view`, `md::tiled_size` |
| `examples/validation/mdspan_validation.cpp` | écritures par la vue et relecture linéaire, opérateurs composés, rang 3, vues en lecture seule, le mapping en tuiles (bijection, bourrage), une tuile uniforme est une suite uniforme ; toutes les variantes (un ctest chacune, `mdspan_validation_<variant>`) |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | la comparaison de la section 2 ; `mdspan_tiles 4096` pour un champ plus grand |
