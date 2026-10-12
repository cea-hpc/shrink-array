# shrink-array : des tableaux compressés pour C++

> English version: [README.md](README.md).

shrink-array fournit des tableaux qui s'utilisent comme un `std::vector` et prennent moins de
mémoire quand de grandes parties d'entre eux contiennent la même valeur -- zéro, ou quelques
valeurs répétées. Elle est écrite pour les codes de calcul : C++20, OpenMP, tableaux de 10⁹
éléments, jusqu'à quelques centaines de threads.

Un tableau est découpé en **chunks** de `S` éléments consécutifs (`S` une puissance de deux ou un multiple de 64).
Un chunk dont tous les éléments valent la même constante ne garde pas de tampon à lui : il
pointe vers un tampon partagé de cette constante et ne coûte que quelques octets de
métadonnées. Y écrire une autre valeur lui redonne un tampon, et pendant que le programme
tourne, les chunks redevenus uniformes sont de nouveau compressés et leurs tampons retournent
à l'allocateur. Lectures et écritures gardent la syntaxe `[]`.

Elle est rentable sur des données surtout nulles, ou faites surtout de plages de quelques
valeurs : champs creux, masques de matériaux, grandeurs multi-matériaux, drapeaux. Elle ne
l'est pas sur des données où chaque chunk contient des valeurs variées : elle y coûte un peu
plus de mémoire qu'un `std::vector`, et chaque accès est plus lent.

## Trois variantes

| variante | classe | un chunk est compressé quand il est uniforme à… |
|---|---|---|
| Chunk (v1) | `ArrayChunk<T>` | `T{}` (zéro) seulement |
| Chunk2 | `ArrayChunk2<T>` | `T{}`, ou **une** seconde valeur par tableau, déclarée par `set_second_value()` |
| Chunk2A | `ArrayChunk2A<T>` | `T{}`, ou **n'importe quelle** valeur, chunk par chunk, sans rien déclarer |

La variante et la taille de chunk sont choisies par la bibliothèque liée (`chunk::chunk_v2a_256`,
`chunk::chunk_v2a_dyn`, ...) ; les en-têtes sont les mêmes pour toutes. Écrire la classe de
cette variante, ou un alias dessus ([docs/getting_started_fr.md](docs/getting_started_fr.md)
§3) : un tableau d'une autre variante que celle de la bibliothèque n'est pas pris en charge.
Chunk2A est celle qui compresse le plus, et celle à essayer d'abord.

Ce que les tableaux économisent dépend des données. Pour un ordre de grandeur, le banc
`shrink_bench` ([examples/shrink_bench](examples/shrink_bench), `--mode perf`) contient des
tableaux d'environ 10⁹ éléments d'une douzaine de types (`double`, `float`, `int`, `long`,
`std::complex`, structures, `bool`...). Chacun est rempli pour être surtout nul ou égal à
quelques valeurs répétées, le reste en données variées, puis réécrit par morceaux pendant les
2 000 itérations d'une boucle de calcul. Mémoire de tout le processus en fin d'exécution,
mesurée exactement ([docs/mesures_performance_fr.md](docs/mesures_performance_fr.md)), sur
Grace avec 64 threads :

| variante | chunks statiques de 256 éléments (`_256`) | chunks dynamiques (`_dyn`) : à partir de 64, rechunk adaptatif |
|---|---|---|
| Chunk2A | 7,26 Go | 6,0 Go |
| Chunk2 | 17,3 Go | 16,3 Go |
| Chunk (v1) | 40,2 Go | 40,3 Go |
| `std::vector` (`shrink_bench_vector_perf`) | 125,9 Go | — |

Pour comparaison, le même banc en `std::vector` utilise 125,9 Go, mesurés de la même façon ;
leurs éléments seuls font environ 100 Go. L'écart entre les variantes est la
part des données que chacune sait compresser : v1 les plages nulles seulement, Chunk2 une
valeur de plus par tableau, Chunk2A toute plage d'une valeur répétée. Sur des données où
chaque chunk contient des valeurs variées, il n'y a rien à compresser et les tableaux coûtent
un peu plus qu'un `std::vector`.
[docs/variantes_fr.md](docs/variantes_fr.md) explique ce que coûte chacune et laquelle choisir.
`ArrayChunkBits` stocke des `bool` à un bit par élément (builds Chunk2).

## Prérequis

- **Linux**, x86-64 ou aarch64. La bibliothèque s'appuie sur des appels système Linux
  (`membarrier`, `madvise`, `mincore`) et sur `/proc`.
- Un compilateur **C++20** : GCC 13 ou 14 et Clang 18 sont testés ; la compilation couvre
  aussi oneAPI (`icpx`), AOCC et NVHPC. Avec Clang, le build se lie à libc++ par défaut
  (`-DUSE_LIBCXX=OFF` pour libstdc++).
- **CMake 3.20** ou plus récent.
- **OpenMP**, facultatif (`-DOMP=ON`). Un programme qui utilise les tableaux dans des régions
  OpenMP a besoin d'une bibliothèque compilée avec. L'identité d'un thread est son rang
  OpenMP : utiliser les tableaux depuis des threads OpenMP, pas depuis des `std::thread`, et
  fixer `OMP_NUM_THREADS` (ou `CHUNK_THREAD_CAPACITY`) à la plus grande équipe avant le premier
  tableau.
- **OMPT** : avec Clang ou oneAPI, le runtime OpenMP a l'interface OMPT, et la bibliothèque
  repère d'elle-même le début et la fin des régions parallèles. Le `libgomp` de GCC n'en a pas :
  le programme marque alors ses régions -- `begin_parallel_phase()` par chaque thread en haut
  d'une région, `finalize_parallel_phase()` une fois après.
- Python 3, pour les vérifications des sources lancées par `ctest` (facultatif).

## Compiler et installer

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOMP=ON -DCMAKE_INSTALL_PREFIX=$HOME/opt/shrink-array
cmake --build build
ctest --test-dir build
cmake --install build
```

| option | défaut | effet |
|---|---|---|
| `OMP` | `OFF` | compiler avec OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | les tailles de chunk fixées à la compilation, une bibliothèque par taille et par variante |
| `CHUNK_BUILD_DYN` | `ON` | compiler aussi les bibliothèques dont la taille de chunk est choisie, et peut changer, à l'exécution (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | allouer les tampons des chunks avec l'allocateur standard au lieu des `fast_allocators` fournis (utile sous AddressSanitizer) |
| `SHRINK_ARRAY_BUILD_EXAMPLES` | `ON` au premier niveau | les exemples, le banc et les harnais de validation (les harnais dans les builds Debug, `CHECK_OPTIM` ou avec sanitizer, ou avec `-DCHUNK_BUILD_VALIDATION=ON`) |
| `SHRINK_ARRAY_BUILD_TESTS` | `ON` au premier niveau | enregistrer les tests auprès de `ctest` |

Les réglages utilisateur : [docs/variables_environnement_fr.md](docs/variables_environnement_fr.md).
Les interrupteurs des développeurs (réglages fins, traces, banc, options de mise au point, scripts
de compilation) : [docs/reglages_developpeur_fr.md](docs/reglages_developpeur_fr.md).

### Avec Spack

La recette est dans le dépôt ([spack/spack_repo/shrink_array](spack/spack_repo/shrink_array),
Spack 1.x) :

```shell
spack repo add <clone>/spack/spack_repo/shrink_array
spack install shrink-array +openmp sizes=256,1024 %gcc@15
```

Ses variantes sont les options ci-dessus : `+openmp` (`OMP`), `sizes=` (`CHUNK_STATIC_SIZES`),
`+dyn` (`CHUNK_BUILD_DYN`), `~fast_alloc` (`CHUNK_STD_ALLOC`), `+stats`, `+checks`
(`CHECK_OPTIM`), `+second_value_indexed`, `+examples` ; `spack info shrink-array` les liste. Les
versions sont les branches (`main`, `cpp23`, `develop`), récupérées depuis
GitHub avec vos propres identifiants git ; `spack develop` construit plutôt un clone local.

## L'utiliser depuis CMake

Après une installation :

```cmake
find_package(chunk 0.1 REQUIRED)            # -DCMAKE_PREFIX_PATH=<préfixe d'installation>
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

Ou depuis les sources, dans votre propre build. Seules les bibliothèques sont alors compilées
-- pas d'exemples, pas de tests -- et rien n'est écrit dans votre cache :

```cmake
add_subdirectory(shrink-array)              # ou FetchContent_Declare(shrink_array GIT_REPOSITORY <url> GIT_TAG <tag>)
                                            #    FetchContent_MakeAvailable(shrink_array)
target_link_libraries(my_app PRIVATE chunk::chunk_v2a)
```

`chunk::chunk_v2a` est Chunk2A à la première taille statique ; `chunk::chunk_v2a_1024` et
`chunk::chunk_v2a_dyn` choisissent une autre taille, `chunk::chunk_v1` et `chunk::chunk_v2`
une autre variante.

## Un premier programme

```cpp
#include <cstdio>
#include "chunk/array_chunk2A.h"   // shrink_array::ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // sa::build_config::has_not_ompt
#include "chunk/manage_ompt.h"     // begin_parallel_phase, finalize_parallel_phase
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

namespace sa = shrink_array;

int main()
{
    sa::ParamChunk::init_indexes(true);       // facultatif : le premier tableau le fait sinon
    {
        const int                n = 1 << 24;
        sa::ArrayChunk2A<double> a(n);        // tous les éléments à 0.0, aucun tampon encore

#pragma omp parallel
        {
            if constexpr (sa::build_config::has_not_ompt) sa::begin_parallel_phase();
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
                a[i] = (i < n / 8) ? 1.0 + i % 7 : 3.5;   // la plage de 3.5 se compresse
        }
        if constexpr (sa::build_config::has_not_ompt) sa::finalize_parallel_phase();

        a.shrink_to_fit();                    // replier les chunks devenus constants
        std::printf("chunks holding a buffer: %.1f %%\n", a.stats());
    }                                         // le tableau rend ses chunks ici
    sa::release_chunk_pools();                // facultatif : rendre les pools avant la fin
    sa::destroy_block_chunk();
}
```

La version complète, qui vérifie aussi ce qu'elle relit, est
[examples/getting_started](examples/getting_started/getting_started.cpp), et
[docs/getting_started_fr.md](docs/getting_started_fr.md) la commente pas à pas.

## Utiliser les tableaux

### Types d'éléments

Un tableau contient tout type **trivialement copiable**, **comparable par égalité** et
**constructible par défaut** : tous les types arithmétiques, `bool`, `std::complex`, les
énumérations, les `std::array` de ceux-ci et tout agrégat de ceux-ci. Rien n'est à déclarer à
l'avance ni à appeler d'abord : chaque tableau enregistre son type d'élément à la construction
du premier, tableaux globaux et `static` compris.

```cpp
struct Point { double x, y, z; auto operator<=>(const Point&) const = default; };
shrink_array::ArrayChunk2A<Point> pts(n);   // compresse les plages de {0,0,0} et de tout point répété
```

Un type qui possède une ressource (`std::string`, tout ce dont le destructeur compte) est
refusé à la compilation, avec le contrat dans le message d'erreur (`ArrayChunkElement`). Les
chunks sont de la mémoire brute, copiée, mise de côté et remplacée par un tampon constant
partagé sans jamais exécuter de constructeur ni de destructeur. Chunk2 ordonne ses secondes
valeurs et demande à un agrégat de flottants un `operator<` (un `operator<=>` par défaut le
fournit).

### Ce qu'il y a de `std::vector`

- **Présent :** `operator[]`, `at`, `front`, `back` ; `size`, `empty`, `capacity`, `reserve` ;
  `resize`, `clear`, `assign`, `swap`, `shrink_to_fit` ; `push_back`, `emplace_back`,
  `pop_back` ; des itérateurs const à accès aléatoire (`begin`/`end`, `rbegin`/`rend`, boucle
  `for` sur intervalle, `std::lower_bound`) ; les constructeurs `(first, last)` et
  `{a, b, c}` ; copie profonde et déplacement. `shrink_to_fit()` replie aussi tout chunk
  devenu constant, et `compress()` lance une passe de compression complète à la demande.
- **Absent, par construction :** `data()` (un chunk compressé n'a pas de tampon), les
  itérateurs modifiables et les `T&` sur un élément, `insert` / `erase` au milieu, et donc un
  `std::sort` en place (trier une copie, puis la remettre par `assign`).
- Le type d'indice est `int` par défaut : `ArrayChunk2A<T, long>` au-delà de 2³¹ éléments.

[docs/api_fr.md](docs/api_fr.md) liste tous les appels, par tâche, sur une page.

### Le proxy d'écriture

L'`operator[]` non const renvoie un *proxy*, pas un `T&` : une poignée sur l'élément, qui le
lit quand on le convertit et fait passer toute écriture par la mécanique de compression.

- `T v = a[i];` prend la valeur ; `auto v = a[i];` garde le proxy, qui suit alors l'élément.
  Un code porté depuis `std::vector` veut `T`, pas `auto` : un usage d'un tel `v` ne compile
  pas (`internal/element_ref.h`).
- `std::printf("%f", a[i])`, `std::format("{}", a[i])`, `std::max(a[i], 2.0)` recevraient le
  proxy, pas la valeur, et ne compilent pas : écrire `double(a[i])`.
- `a[i] += x;` et les autres opérateurs composés renvoient un proxy ; l'ignorer, comme ici,
  permet à la bibliothèque de différer l'opération sans coût. `a[i]++` renvoie l'ancienne
  valeur sous forme de `T`.
- Un proxy tient des références vers les chunks du tableau : tout ce qui les reconstruit --
  `resize`, `reserve`, `shrink_to_fit`, `clear`, `assign`, `swap`, une affectation au tableau,
  un rechunk -- l'invalide. Garder les proxys de courte durée.

### Threads

Sous OpenMP, le contrat est celui de `std::vector`, élément par élément :

- **Des éléments différents** n'ont besoin d'aucune synchronisation, même quand ils partagent
  un chunk.
- **Le même élément** accédé depuis plusieurs threads doit être ordonné par le programme (une
  barrière, une section `critical`, un drapeau atomique en release/acquire). Ordonné, il se
  comporte comme sur un `std::vector` : une écriture que la bibliothèque a différée est
  appliquée avant tout accès que le programme a ordonné après elle.
- **Tout ce qui reconstruit le stockage** -- les opérations qui invalident les proxys, rechunks
  compris -- est un point séquentiel : l'appeler hors de toute région parallèle.

L'écriture ne prend aucun verrou : écrivains et compresseurs s'excluent par une poignée de
main de Dekker asymétrique (un simple store pour l'écrivain, un appel `membarrier()` pour le
compresseur, rare), et une écriture qui rencontre un compresseur sur son chunk est différée
et rejouée, jamais perdue ([docs/synchronisation_dekker_fr.md](docs/synchronisation_dekker_fr.md)).

### Remplir des plages

`chunk/array_chunk_range_ops.h` décide une fois par chunk au lieu d'une fois par élément :
`fill_range(a, first, last, v)`, `copy_range(a, first, last, src_first)`, et `fill_range_mt`,
`copy_range_mt`, `transform_range_mt` pour plusieurs threads qui écrivent en même temps des plages
disjointes.

### Taille de chunk adaptative

Dans les bibliothèques `_dyn`, la taille de chunk d'un tableau est choisie, et peut changer, à
l'exécution : `a.rechunk_to(n)` rechunke un tableau, et un `RechunkDecider`
(`chunk/rechunk_decider.h`), guidé par les estimations de `ChunkSizeAdvisor`, décide passage
après passage si un rechunk en vaut la peine. Voir
[docs/rechunk_adaptatif_fr.md](docs/rechunk_adaptatif_fr.md).

## Exemples et outils

- [examples/getting_started](examples/getting_started) : le premier programme ci-dessus,
  complet.
- [examples/simple_array](examples/simple_array) : le même calcul sur un `std::vector` et sur
  un tableau compressé, pour comparer temps et mémoire.
- [examples/multi-mat](examples/multi-mat) : des champs multi-matériaux, chaque matériau en
  tableaux compressés ou en `std::vector`, avec des boucles écrites une fois pour les deux
  ([Loops.h](examples/multi-mat/Loops.h)).
- [examples/shrink_bench](examples/shrink_bench) : le banc de référence -- dix-huit tableaux de
  types d'éléments différents sous une boucle de calcul, `--mode check` contre des fichiers de
  référence `std::vector`, `--mode perf` pour les mesures de temps.
- [examples/validation](examples/validation) : un harnais par propriété (aller-retour de
  compression, écritures différées, rechunk, lecture/récupération, comptabilité des pools,
  régions imbriquées...), lancés par `ctest`.
- [examples/advisor_demo](examples/advisor_demo) et
  [examples/variant_compare](examples/variant_compare) : l'advisor de taille de chunk face à de
  vraies tailles, et Chunk2 face à Chunk2A sur une même charge.
- [python/shrink_gui](python/shrink_gui) : une fenêtre sur le portable pour soumettre la
  compilation, les tests et le banc sur le calculateur et les suivre
  ([docs/shrink_gui_fr.md](docs/shrink_gui_fr.md)).

`simple_array` et `multi-mat` travaillent sur des tableaux de 10⁹ éléments : leur `ctest`
demande un nœud de calcul (24 Go pour la version `std::vector` de `simple_array`) et est tué
sur un portable.

## Documentation

| je veux… | lire |
|---|---|
| installer, lier depuis CMake, écrire un premier programme, suivre les règles OpenMP | [docs/getting_started_fr.md](docs/getting_started_fr.md) |
| trouver un appel | [docs/api_fr.md](docs/api_fr.md) |
| choisir une variante | [docs/variantes_fr.md](docs/variantes_fr.md) |
| comprendre lecture, écriture, compression en vol, écritures différées | [docs/algo_chunk2_fr.md](docs/algo_chunk2_fr.md) |
| utiliser les tailles de chunk dynamiques | [docs/rechunk_adaptatif_fr.md](docs/rechunk_adaptatif_fr.md) |
| régler la bibliothèque ou la compilation | [docs/variables_environnement_fr.md](docs/variables_environnement_fr.md) |
| mesurer mémoire et temps sans se tromper | [docs/mesures_performance_fr.md](docs/mesures_performance_fr.md) |
| soumettre et suivre des runs sur le calculateur depuis une fenêtre | [docs/shrink_gui_fr.md](docs/shrink_gui_fr.md) |
| contribuer | [docs/contributing_fr.md](docs/contributing_fr.md) |

Chaque document existe en anglais et en français ; l'index est [docs/README.md](docs/README.md).
Ce qui a changé d'une version à l'autre, et la politique de version :
[CHANGELOG.md](CHANGELOG.md) (en anglais).

## Licence

Apache License 2.0, voir [LICENSE](LICENSE).

Copyright CEA
