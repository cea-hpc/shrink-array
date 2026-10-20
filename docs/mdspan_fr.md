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
parallèle, les `_mt` dedans). Sur Chunk2A, un chunk compressé dont la part de la tuile en sort
uniforme change de constante sur place, sans tampon, y compris dans une région parallèle
(`fill_range_mt`, `copy_range_mt`, dans la session d'écriture). Hors région, un chunk matérialisé
qui en sort uniforme est lui aussi compressé aussitôt ; dans une région, il est écrit et mis en
file -- son tampon est libéré par la fin de la phase ou la passe suivante, qui attendent les
lecteurs. Sinon un chunk compressé qui change est matérialisé pour l'écriture et mis en file, et
la même passe le recompresse.

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

### Sous-vues : `md::submdspan`

`md::submdspan(f, tranches...)` est le `submdspan` du `mdspan` utilisé (`Kokkos::submdspan`, ou
`std::submdspan` là où la bibliothèque standard l'a), avec `md::full_extent`. La poignée de
données de la vue est un `md::handle` : le tableau et l'indice dans celui-ci du premier élément de
la sous-vue (`f.data_handle().base`) ; `*f.data_handle()` est toujours le tableau.

```cpp
auto f   = md::view(a, nx, ny);
auto blk = md::submdspan(f, std::pair{3, 20}, std::pair{5, 40}); // blk[0, 0] est f[3, 5]
auto row = md::submdspan(f, 7, md::full_extent);                 // rang 1

auto g   = md::tiled_view<16, 16>(b, nx, ny);
auto sub = md::submdspan(g, std::pair{16, 64}, std::pair{32, ny}); // de nouveau en tuiles
md::update_tiles(sub, on_block, on_uniform);                       // les seules tuiles de sub
```

- Une vue ligne par ligne accepte toutes les tranches (le `submdspan_mapping` des layouts
  standard : le résultat est `layout_right` ou `layout_stride`). `fill_box` accepte toute vue à
  pas : une range op par ligne de la boîte quand le dernier pas vaut 1, une par élément sinon.
- Une vue en tuiles accepte des tranches qui gardent la dimension, de pas 1 (`full_extent`,
  `{first, last}`), alignées sur ses tuiles : chaque plage commence sur une tuile et finit sur
  une tuile ou à l'étendue de la vue ; tout le reste arrête le programme. La sous-vue est de
  nouveau un `layout_tiled`, dans la grille de tuiles du parent : ses tuiles sont celles du
  parent, chacune toujours un chunk, et le bourrage de ses dernières tuiles est celui du parent.
  `for_each_tile`, `update_tiles` et `fill_box` s'y appliquent comme sur une vue entière -- ils
  lisent et écrivent des tuiles entières et ne touchent à rien hors de la sous-vue --, en phase
  parallèle aussi (`tile_count(sub)` tuiles).

`examples/validation/mdspan_submdspan_validation.cpp` ; vérifié avec kokkos/mdspan avant et après
ses tranches canoniques (mai 2025, octobre 2026).

### La tuile à l'exécution : `layout_tiled_dyn`

Une bibliothèque `_dyn` ne connaît sa taille de chunk qu'à l'exécution (`DYN_CHUNK_SIZE`, un
rechunk) : une tuile d'un chunk ne peut pas être un argument template.
`md::layout_tiled_dyn<Rank>` prend la tuile à l'exécution et range le tableau exactement comme
`layout_tiled` pour la même tuile (les deux vues d'un même tableau concordent) ; les opérations sur
tuiles, `submdspan` et `fill_box` la prennent comme elles prennent `layout_tiled`.

```cpp
ArrayChunk2A<double> a(md::tiled_size_dyn(md::tile_shape<2, long>(chunk_size), std::array<long, 2>{nx, ny}));
auto f = md::tiled_view_dyn(a, nx, ny);                          // une tuile de a.chunk_size() éléments
auto g = md::tiled_view_dyn(a, std::array<long, 2>{16, 24}, nx, ny); // ou toute autre tuile
```

- `md::tile_shape<Rank>(n)` donne une tuile d'exactement `n` éléments, aussi carrée que les
  diviseurs de `n` le permettent, les étendues les plus longues en dernier : 256 -> 16 x 16,
  384 -> 16 x 24, 512 -> 16 x 32, 1024 -> 32 x 32, 256 en 3D -> 4 x 8 x 8. Toute taille de chunk de
  la bibliothèque (une puissance de deux ou un multiple de 64) a une telle tuile ;
  `tiled_view_dyn(a, ext...)` la prend avec `a.chunk_size()`.
- Coût, 2048 x 2048, 1 thread : une lecture par `f[i, j]` comme avec `layout_tiled` (2,75 ns par
  élément : la division à l'exécution se cache derrière l'accès à l'élément), `update_tiles`
  environ 10 % plus lent (les étendues du bloc sont elles aussi des valeurs d'exécution).
- Après un rechunk rien ne bouge : la vue lit et écrit toujours les bons éléments, et les
  opérations sur tuiles marchent toujours en série. Mais une tuile n'est plus un chunk -- à une
  taille de chunk double, un chunk tient deux tuiles consécutives --, donc dans une région
  parallèle les opérations sur tuiles s'arrêtent (deux threads partageraient un chunk) tant que
  le champ n'a pas été copié dans une vue de la nouvelle tuile.

### Un programme complet : `multi-mat-md`

`examples/multi-mat` existe en 2D : `multi-mat-md` est la même maquette (chaque matériau porte
trois champs et une fraction de présence sur tout le maillage, en occupe une partie tirée au
hasard à chaque itération, y est écrit puis relu par le même calcul gadget), sur un maillage
`nx x ny` où un matériau occupe un rectangle. Chaque tableau est vu par un `mdspan` 2D :
`md::view` ou `md::tiled_view` pour les tableaux chunk (v1 pour les champs, Chunk2 pour la
fraction), un `mdspan` standard sur `data()` pour les `std::vector`. Le corps de boucle est écrit
une fois contre les vues ; la politique du stockage du matériau (`LoopsMd.h`) dit comment un
rectangle est écrit, remis à zéro et lu, comme `Loops.h` en 1D -- et la liste d'indirection de
la version 1D disparaît : le rectangle est la boucle.

| `--chunk-mode` | tableaux chunk | boucles sur le rectangle |
|---|---|---|
| `tiles` (défaut) | `layout_tiled`, une tuile par chunk | tuile par tuile : `update_tiles` (une tuile de l'intérieur déjà à 1 n'est pas touchée), `fill_box` pour la remise à zéro, quatre tuiles lues d'un bloc chacune pour la réduction |
| `elements` | `layout_tiled` | cellule par cellule, `f[i, j]` |
| `rows` | `md::view` ligne par ligne | cellule par cellule, `f[i, j]` |

L'équipe se partage d'un coup les cellules de tous les rectangles, au poids, tuile par tuile
(ligne par ligne d'un rectangle en `rows`, où un chunk est un morceau de ligne) ; chaque thread
calcule le même partage, sans communication, et une tuile va à un seul thread. Partagé rectangle
par rectangle, un `omp for` par matériau, un rectangle de 113 lignes fait 8 rangées de tuiles :
à 64 threads seuls 15 threads écrivaient et le plus chargé faisait 7,8 fois la moyenne (35 fois
à 288 threads). Partagé d'un coup : 1,01 à 64 threads, 1,07 à 288 (3200 x 3200, 50 matériaux,
compté sur 2 cœurs). Le calcul affiche ce rapport ; `--max-imbalance` en fait un contrôle (test
`multi_mat_md_balance`).

Maillage 1600 x 1600, 50 matériaux tous en tableaux chunk, 100 itérations, rectangles jusqu'à
113 x 113 (`--pc-chunk 0.5`), les mêmes rectangles pour chaque ligne (`--seed 3`) ; Release,
GCC 14, kokkos/mdspan, 2 threads sur un conteneur à 2 cœurs (ordres de grandeur : ± 20 % d'un lancement à l'autre là-dessus) :

| | chunks écrits, champs / fraction | mémoire max | temps | dont écritures / lectures |
|---|---|---|---|---|
| `--chunk-mode tiles` | 16,7 % / 12,5 % | 663 Mio | 2,29 s | 0,35 s / 1,82 s |
| `--chunk-mode elements` | 16,7 % / 12,5 % | 659 Mio | 4,36 s | 0,68 s / 2,36 s |
| `--chunk-mode rows` | 46,1 % / 46,1 % | 1847 Mio | 4,14 s | 1,01 s / 1,96 s |
| tout en `std::vector` (`--option 2`) | -- | 3912 Mio | 1,90 s | 0,12 s / 1,55 s |

Les quatre affichent le même résultat. Ligne par ligne, un rectangle plus étroit qu'un chunk
laisse écrit chaque chunk qu'il traverse, et la fraction ne se comprime jamais à l'intérieur ;
en tuiles, seules restent les tuiles le long des bords. Le temps qui reste après les écritures
et les lectures est l'initialisation, qui remet à zéro chaque tableau sur tout le maillage comme
la version 1D : en tuiles un `fill_box` de chunks déjà à 0, qui n'écrit rien (0,01 s) ; cellule
par cellule (`elements`, `rows`), une écriture par cellule (environ 1,2 s). `--verify` rejoue le
calcul en série sur des vecteurs simples et compare chaque tableau cellule par cellule ainsi que
le résultat (sortie 1 sur une différence) ; les tests `multi_mat_md_tiles`, `multi_mat_md_elements` et `multi_mat_md_rows` le lancent en petit.
Les options sont décrites en tête de `main_md.cpp`.

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
- Ce n'est pas un pointeur : `data_handle()` est un `md::handle` (le tableau et un indice), pas
  un `T*` -- pas de passage des données à du code qui attend une mémoire contiguë.
- `layout_tiled` n'est pas à pas constant (`is_strided()` est faux) et n'est exhaustif que si
  chaque étendue est un multiple de sa tuile ; le code qui exige `layout_right` ou
  un layout à pas prend la vue ligne par ligne.

## 6. Fichiers

| fichier | contenu |
|---|---|
| `chunk/public/chunk/mdspan.h` | `md::accessor`, `md::handle`, `md::layout_tiled` (et son `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | écritures par la vue et relecture linéaire, opérateurs composés, rang 3, vues en lecture seule, le mapping en tuiles (bijection, bourrage), une tuile uniforme est une suite uniforme ; toutes les variantes (un ctest chacune, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile`, `md::update_tiles`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocs, étendues des tuiles sur les bords bourrés, tuiles uniformes, la mise à jour contre la formule élément par élément, les tuiles non touchées restent compressées, la mise à jour en parallèle ; toutes les variantes à 1 et 8 threads, et `mdspan_tile_ops_misaligned` (une tuile plus petite qu'un chunk refusée dans une région parallèle) |
| `examples/validation/mdspan_submdspan_validation.cpp` | sous-vues d'une vue ligne par ligne (bloc, ligne, colonne, lecture seule, `fill_box`) et d'une vue en tuiles (alignée, finissant à l'étendue, sous-vue d'une sous-vue ; les opérations sur tuiles dessus, en série et en phase parallèle) ; toutes les variantes à 1 et 8 threads, et `mdspan_submdspan_misaligned` (une tranche en tuiles hors des tuiles arrête le programme) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, tous les multiples de 64 jusqu'à 4096), `layout_tiled_dyn` contre `layout_tiled` indice par indice et par leurs sous-vues, la tuile par chunk de `tiled_view_dyn`, les opérations sur tuiles sur une tuile par chunk, une tuile 3 x 5 et une au-delà du tampon local ; toutes les variantes à la taille par défaut et en dyn (un rechunk sous la vue), 1 et 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | les comparaisons des sections 2 et 3, et le coût du multi-indice ; `mdspan_tiles 4096` pour un champ plus grand |
| `examples/multi-mat/main_md.cpp`, `LoopsMd.h`, `ComputeLoopsMd.h`, `SimuMd.h` | `multi-mat-md`, la maquette multi-matériaux en 2D (section 3) ; tests `multi_mat_md_tiles`, `multi_mat_md_elements`, `multi_mat_md_rows` (`--verify`), `multi_mat_md_balance` |
