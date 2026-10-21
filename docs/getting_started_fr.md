# Premiers pas avec shrink-array

> English version: [getting_started.md](getting_started.md).

> Octobre 2026. Ce guide explique comment **utiliser** la bibliothèque. Son fonctionnement
> interne est décrit dans les documents de conception, indexés dans [README.md](README.md). La
> version anglaise de ce guide est [getting_started.md](getting_started.md).

***

## 1. Ce que c'est

shrink-array fournit des tableaux qui se comportent comme un `std::vector` et prennent moins
de mémoire quand de grandes parties contiennent la même valeur. Un tableau est découpé en
**chunks** de `S` éléments consécutifs (`S` une puissance de deux ou un multiple de 64, 256 par défaut). Un chunk
dont tous les éléments valent une même constante ne garde aucun tampon à lui. Il pointe vers
un tampon partagé de cette constante et ne coûte que quelques octets de métadonnées. Y écrire
une autre valeur lui redonne un tampon. Pendant l'exécution, les chunks redevenus uniformes
sont compressés à nouveau, et leurs tampons retournent à l'allocateur.

C'est rentable sur des données surtout nulles, ou faites de plages de quelques valeurs :
champs creux, masques de matériaux, grandeurs multi-matériaux, indicateurs. Ça ne l'est pas sur
des données où chaque chunk contient des valeurs variées. La mémoire est alors un peu plus
grande qu'avec un `std::vector`, et chaque accès plus lent.

Trois variantes, choisies au moment de l'édition de liens :

| variante | classe | un chunk est compressé quand il est uniforme à… |
|---|---|---|
| Chunk (v1) | `ArrayChunk<T>` | `T{}` (zéro) seulement |
| Chunk2 | `ArrayChunk2<T>` | `T{}`, ou **une** seconde valeur par tableau, déclarée par `set_second_value()` |
| Chunk2A | `ArrayChunk2A<T>` | `T{}`, ou **n'importe quelle** valeur, chunk par chunk, sans rien déclarer |

Chunk2A compresse le plus et c'est celle à essayer d'abord. Chunk2 a des métadonnées plus
légères quand les données ont une seule valeur non nulle dominante, et Chunk (v1) est la plus
simple quand seuls les zéros comptent. [variantes_fr.md](variantes_fr.md) les compare sur un
cas réel.

## 2. Installer

Avec Spack, la recette est dans le dépôt : voir [le README](../README_fr.md#avec-spack)
(`spack install shrink-array`). Depuis les sources, avec un compilateur C++20 (essayé avec
GCC 13 et 14 et Clang 18 ; les scripts de compilation couvrent aussi oneAPI, AOCC et NVHPC ;
la branche `cpp23` demande le C++23, GCC 14 ou plus récent) et CMake ≥ 3.20 :

```sh
cmake -S shrink-array -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOMP=ON \
      -DCMAKE_INSTALL_PREFIX=$HOME/opt/shrink-array
cmake --build build
cmake --install build
```

Les options qui comptent pour un utilisateur :

| option | défaut | effet |
|---|---|---|
| `OMP` | `OFF` | construire avec OpenMP. Un programme qui utilise les tableaux dans des régions OpenMP a besoin d'une bibliothèque construite avec |
| `CHUNK_STATIC_SIZES` | `auto` | les tailles de chunk fixées à la compilation, une bibliothèque par taille et par variante ; `auto` : `256;1024`, mais `256` seule dans un build Debug ou sanitizer de ce projet (pas d'un projet qui l'inclut) |
| `CHUNK_BUILD_DYN` | `ON` | construire aussi les bibliothèques dont la taille de chunk est choisie à l'exécution, et peut changer (`_dyn`) |
| `CHUNK_STD_ALLOC` | `OFF` | allouer les tampons des chunks avec l'allocateur standard plutôt qu'avec `fast_allocators`, fourni avec la bibliothèque (utile sous AddressSanitizer) |

La première compilation construit aussi les exemples et le banc, ce qui prend un moment. Pour
ne construire que les bibliothèques, configurer avec `-DSHRINK_ARRAY_BUILD_EXAMPLES=OFF`, ou
donner leurs cibles à `cmake --build` (`--target chunk_v2a_256 ...`) avant `cmake --install`.
Intégré à un autre projet par `add_subdirectory` ou `FetchContent`, seules les bibliothèques
sont construites.

Avec Clang ou oneAPI, un build OpenMP utilise aussi l'interface **OMPT** du runtime, ce qui
dispense le programme des appels de phase du §5.

## 3. L'utiliser depuis CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(mon_appli LANGUAGES CXX)

find_package(chunk 0.1 REQUIRED)        # -DCMAKE_PREFIX_PATH=<préfixe d'installation>

add_executable(mon_appli main.cpp)
target_link_libraries(mon_appli PRIVATE chunk::chunk_v2a)
```

La bibliothèque liée **choisit la variante et la taille de chunk**. Les en-têtes sont les mêmes
pour toutes, et la bibliothèque apporte les définitions de compilation qui sélectionnent la
sienne (`CHUNK_VARIANT_V2A=1`, `CHUNK_SIZE=256` ou `CHUNK_DYNAMIC`), ainsi que le C++20,
`-fopenmp` si elle a été construite avec OpenMP, et ses dépendances. Les autres options de sa
construction sont dans `chunk/build_options.h`, installé, que lisent les en-têtes : un
programme les compile comme la bibliothèque.

| cible | variante | taille de chunk |
|---|---|---|
| `chunk::chunk_v1`, `chunk::chunk_v2`, `chunk::chunk_v2a` | v1, Chunk2, Chunk2A | la première de `CHUNK_STATIC_SIZES` (256) |
| `chunk::chunk_v2a_1024`, … | | cette taille, fixée à la compilation |
| `chunk::chunk_v2a_dyn`, … | | choisie à l'exécution, par tableau (modifiable par `rechunk_to()` ou par le décideur adaptatif) |
| `chunk::chunk_vector` | aucune : les tableaux sont des `std::vector` | — |

Lier **une seule** bibliothèque de variante par programme, et écrire **la classe de cette
variante** : `ArrayChunk2A<T>` avec `chunk::chunk_v2a`. La classe écrite est celle que l'on
obtient -- un programme qui nomme `ArrayChunk2A` la garde quelle que soit la bibliothèque --,
et un tableau d'une autre variante que celle de la bibliothèque n'est pas pris en charge :
selon les appels, il ne compile pas ou il plante. Pour que l'édition de liens choisisse,
passer par un alias sur la variante que la bibliothèque définit (`examples/shrink_bench` et
`examples/simple_array` le font) :

```cpp
#include <type_traits>
#include <vector>
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/config.h"

namespace sa = shrink_array;
template <typename T>
using Array = std::conditional_t<sa::chunk_config::use_chunk2a, sa::ArrayChunk2A<T>,
              std::conditional_t<sa::chunk_config::use_chunk2, sa::ArrayChunk2<T>,
              std::conditional_t<sa::chunk_config::use_chunk, sa::ArrayChunk<T>, std::vector<T>>>>;
```

Avec lui, `chunk::chunk_vector` construit le même programme sur de simples `std::vector`, ce
qui est la façon la plus simple de comparer mémoire et temps.

`find_package(chunk)` définit aussi `chunk_STATIC_SIZES`, `chunk_DEFAULT_SIZE`,
`chunk_HAS_DYN`, `chunk_OPENMP`, `chunk_OMPT` et `chunk_STD_ALLOC`, pour un projet qui veut
vérifier ce que contient l'installation.

## 4. Un premier programme

C'est [`examples/getting_started/getting_started.cpp`](../examples/getting_started/getting_started.cpp),
raccourci. Le répertoire contient aussi le `CMakeLists.txt` du §3 : il se construit contre une
installation comme dans l'arbre des sources.

```cpp
#include <cstdio>
#include <utility>                 // std::as_const

#include "chunk/array_chunk2A.h"   // shrink_array::ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

namespace sa = shrink_array;

int main()
{
    sa::ParamChunk::init_indexes(true);          // (1) une fois, avant le premier tableau

    {
        const int n = 1 << 24;                   // 16 M doubles : 128 Mio en std::vector
        sa::ArrayChunk2A<double> a(n);           // (2) tout à 0.0, rien d'alloué

        sa::parallel_phase([&] {                 // (3) une région parallèle, phases marquées
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
                a[i] = i < n / 8 ? 1.0 + i % 7   //     valeurs variées : ces chunks gardent un tampon
                                 : 3.5;          //     une constante : ces chunks se compressent
        });

        const auto& ca = std::as_const(a);       // (4) lire par une référence constante
        double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < n; ++i)
            sum += ca[i];

        a.shrink_to_fit();                       // (5) replier tous les chunks devenus constants
        std::printf("sum = %.1f, chunks with a buffer: %.1f %%\n", sum, a.stats());
    }                                            // (6) le tableau rend ses chunks

    sa::release_chunk_pools();                   // (7) arrêter la bibliothèque
    sa::destroy_block_chunk();
}
```

1. **`init_indexes(true)`**, une fois, avant le premier tableau : prépare les blocs
   d'allocation (`true` : gros blocs, le choix habituel). Facultatif : sans lui, le premier
   tableau du programme lance lui-même `init_indexes(true)`. Les types d'éléments ne se
   déclarent pas : chacun est enregistré à la construction du premier tableau de ce type.
2. **Un tableau neuf** lit `T{}` partout et n'alloue aucun tampon.
3. **`parallel_phase()`** ouvre une région parallèle et exécute la lambda sur chaque thread.
   Sans OMPT (§5), il marque aussi la phase pour la bibliothèque : `begin_parallel_phase()` sur
   chaque thread avant la lambda, `finalize_parallel_phase()` après la région. Avec OMPT, et
   dans un build séquentiel, ces deux appels ne compilent rien.
4. **La lecture** : sur un tableau `const`, `operator[]` rend un `T`. Sur un tableau non
   constant, il rend un *proxy* (§6).
5. **`shrink_to_fit()`** lance une passe de compression complète, hors région parallèle. Il
   n'est pas nécessaire de l'appeler : les chunks sont compressés pendant l'exécution. Il sert
   à récupérer la mémoire à un moment précis.
6. **La destruction du tableau** rend ses tampons. La bibliothèque affiche alors une ligne par
   tableau, avec son taux de compression.
7. **La paire d'arrêt**, en fin de programme, après le dernier tableau. Facultative elle aussi :
   le processus peut simplement se terminer. Si un tableau vit encore (global ou `static`,
   détruit après le retour de `main`), la paire garde ce dont il aura besoin et ne fait que
   vider les listes libres.

Résultat attendu : environ 12 % des chunks ont un tampon. Un `std::vector` les aurait tous.

## 5. OpenMP : les règles

**Ce qui se fait sans synchronisation.** Comme avec un `std::vector`, des threads différents
peuvent lire et écrire des **éléments différents** en même temps, même dans un même chunk :
compression et matérialisation sont l'affaire de la bibliothèque. Deux accès au **même
élément** depuis des threads différents doivent être ordonnés par le programme (une barrière,
une section `critical`, un verrou, des atomiques release/acquire), exactement comme sur un
`std::vector`.

**Avec OMPT, la boucle du `std::vector` telle quelle.** Avec une bibliothèque construite avec
OMPT (Clang, oneAPI), la même boucle OpenMP remplit un `std::vector` et un tableau compressé :
ni `parallel_phase()`, ni appel de phase. Lié comme au §3, avec `OpenMP::OpenMP_CXX` en plus :

```cpp
#include <cstdio>
#include <utility>                 // std::as_const
#include <vector>

#include "chunk/array_chunk2A.h"   // shrink_array::ArrayChunk2A

namespace sa = shrink_array;

template <typename Array>
void fill(Array& t, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n / 2; ++i)
        t[i] = (i < n / 8) ? 1.0 + i % 7 : 3.5;
}

int main()
{
    const int n = 1 << 24;

    std::vector<double>      v(n);
    sa::ArrayChunk2A<double> a(n);
    fill(v, n);                          // la référence
    fill(a, n);                          // la même boucle : ni parallel_phase(), ni appels de phase

    // Lire par une vue const : sur un tableau non const, a[i] est un proxy d'écriture.
    const auto& ca  = std::as_const(a);
    long        bad = 0;
    for (int i = 0; i < n; ++i)
        bad += (ca[i] != v[i]);

    std::printf("%s: %ld element(s) differ out of %d\n", bad == 0 ? "OK" : "FAIL", bad, n);
    return bad == 0 ? 0 : 1;
}
```

**Les phases, sans OMPT.** Quand un écrivain rencontre une compression en cours sur son chunk,
la bibliothèque **diffère** son écriture et la rejoue plus tard. Elle compresse aussi entre les
régions parallèles. Avec OMPT (builds Clang, oneAPI), le runtime OpenMP lui signale le début et
la fin des régions. Sans OMPT (libgomp de GCC), c'est le programme qui le fait :

```cpp
#pragma omp parallel
{
    shrink_array::begin_parallel_phase();     // chaque thread, avant son premier accès
    // ... boucles omp for sur les tableaux ...
}
shrink_array::finalize_parallel_phase();      // une fois, après la région
```

Si on les oublie, rien n'échoue tout de suite. Avec la même équipe d'une région à l'autre, les
valeurs restent justes, mais les passes de compression entre régions n'ont pas lieu : la
mémoire reste plus haute qu'elle ne devrait. Quand la taille de l'équipe change d'une région à
l'autre, un `finalize_parallel_phase()` oublié rejoue une écriture différée déjà libérée, et
si les deux appels sont oubliés, des valeurs sont fausses, sans message. Le programme
s'arrête, dans tous les builds, sur un `begin_parallel_phase()` sans
`finalize_parallel_phase()` depuis la région précédente ; les builds de contrôle (Debug,
`CHECK_OPTIM`) s'arrêtent aussi sur une écriture différée par un thread qui n'a pas appelé
`begin_parallel_phase()`. Les écrire derrière
`if constexpr (sa::build_config::has_not_ompt)`, comme ci-dessus, garde un seul source pour les deux
runtimes.

`shrink_array::parallel_phase()` écrit la région et les deux appels à votre place, et les omet
sous OMPT :

```cpp
shrink_array::parallel_phase([&] {            // ou parallel_phase(n_threads, [&] { ... })
#pragma omp for
    for (int i = 0; i < n; ++i)
        a[i] = f(i);
});
```

Avec OMPT, si le runtime n'a pas chargé l'outil de la bibliothèque (`OMP_TOOL=disabled`, un
autre outil trouvé d'abord), les phases n'ont jamais lieu. La bibliothèque l'écrit sur stderr ;
`CHUNK_REQUIRE_OMPT=1` arrête alors le programme.

Avec OMPT, on peut aussi n'écrire aucune phase : de simples boucles `#pragma omp parallel for`,
le même code que sur un `std::vector`.
[`examples/getting_started/getting_started_ompt.cpp`](../examples/getting_started/getting_started_ompt.cpp)
est le programme du §4 écrit ainsi : une même fonction remplit un `std::vector<double>` et un
`ArrayChunk2A<double>`, et le programme vérifie qu'ils se relisent à l'identique. Il refuse de
compiler contre une bibliothèque sans OMPT ; son test tourne avec `CHUNK_REQUIRE_OMPT=1`.

**Les points séquentiels.** Tout ce qui reconstruit le stockage d'un tableau s'appelle hors de
toute région parallèle : `resize()`, `reserve()`, `push_back()` au-delà de la capacité,
`clear()`, `assign()`, `swap()`, l'affectation, `shrink_to_fit()`, `compress()`,
`rechunk_to()`, et, pour Chunk2, `set_second_value()`.

**Les threads.** Les structures par thread sont dimensionnées une fois pour
`max(omp_get_max_threads(), nombre de cœurs, 256)` threads (environ 1 Kio par type d'élément et
par thread). Un programme qui lance plus de threads que cela fixe `CHUNK_THREAD_CAPACITY=<n>` ;
un thread au-delà de la capacité arrête le programme avec un message.

**À côté d'autres composants.** La bibliothèque garde son état par processus, pas par tableau :
les registres de types d'éléments, les allocateurs, le registre des secondes valeurs de Chunk2A,
les structures par thread et leur capacité, les réglages du log, et la phase. Deux parties d'un
programme qui l'utilisent sans se connaître -- votre code et une bibliothèque construite sur
shrink-array, par exemple -- partagent tout cela. Ce que cela leur demande :

- **Une phase à la fois.** Les files par thread sont indexées par le rang dans l'équipe, et la
  fin d'une phase vide la file de chaque rang. Deux équipes OpenMP de premier niveau qui écrivent
  des tableaux en même temps -- ouvertes par deux threads hôtes, deux composants chacun sur son
  `std::thread` -- partagent les rangs 0 à n-1, et la fin de phase de l'une rejoue les écritures
  de l'autre au milieu de sa région : le programme s'arrête, ou les valeurs sont fausses. Faire
  tourner ces parties l'une après l'autre, ou dans une même équipe.
- **Les régions imbriquées ne posent pas de problème.** Un solveur ou un appel BLAS qui ouvre sa
  propre région dans une phase, un composant qui appelle `parallel_phase()` dans la vôtre : seule
  la fin de la région la plus externe est une frontière de phase, et `finalize_parallel_phase()`
  ne fait rien dans une région active. Les tableaux peuvent être écrits dans une région imbriquée
  tant qu'elle est sérialisée (`OMP_MAX_ACTIVE_LEVELS=1`, le défaut) ; avec deux niveaux actifs,
  une écriture arrête le programme.
- **Le premier tableau fixe la capacité en threads** (ci-dessus) : au moins 256 threads, ou les
  cœurs, ou `OMP_NUM_THREADS` ; un composant qui ouvre des équipes plus grandes demande
  `CHUNK_THREAD_CAPACITY` fixé avant le premier tableau.
- **Des appels à l'échelle du processus.** `ParamChunk::init_indexes()` fixe la taille des blocs
  d'allocation de tout le processus : un appel, au début, par le programme (un appel ultérieur
  laisse intacts les tableaux déjà construits). `release_chunk_pools()` et `destroy_block_chunk()`
  vident les listes libres de tout le monde (un tableau encore vivant garde ce qu'il lui faut) :
  une fois, à la fin du programme, ou pas du tout. `reclaim_second_values()` couvre tous les
  tableaux d'un type ; `CHUNK_LOG` et `logging::set_sink()` sont un seul réglage pour tous.

## 6. Pièges

- **`auto` garde un proxy.** Sur un tableau non constant, `tab[i]` rend un proxy d'écriture.
  `T v = tab[i];` prend la valeur ; `auto v = tab[i];` garde le proxy, qui suit l'élément s'il
  change ensuite. Dans du code venu d'un `std::vector`, remplacer `auto` par le type de
  l'élément. Tout build refuse de compiler un usage d'un tel `v`, et le dit.
- **Convertir un élément avant de le transmettre.** `std::printf("%f", tab[i])` passerait le
  proxy, pas la valeur : refusé (GCC par `-Werror=conditionally-supported`, que pose la
  bibliothèque ; clang toujours). `std::format("{}", tab[i])`, `std::max(tab[i], 2.0)` et une
  fonction générique qui prend `tab[i]` par valeur sont refusés aussi. Écrire `double(tab[i])`,
  ou `T(tab[i])`.
- **Pas de `T&` sur un élément.** `tab[i].x = 1` ne compile pas : lire l'élément, le modifier,
  le réécrire. `data()`, les itérateurs modifiables et un `std::sort` en place n'existent pas
  non plus : trier une copie en `std::vector`, puis la remettre par `assign()`.
- **Un proxy ne doit pas survivre au stockage.** Chacune des opérations séquentielles du §5
  invalide les proxys de ce tableau. Garder les proxys le temps d'une expression.
- **Les tableaux globaux et `static` fonctionnent** : construits avant `main`, détruits après
  son retour. L'état de la bibliothèque est construit à sa première utilisation et jamais
  détruit, l'ordre des initialisations statiques n'importe donc pas.
- **Les indices sont des `int` par défaut.** Au-delà de 2³¹ éléments, utiliser le second
  paramètre du modèle : `ArrayChunk2A<double, long>`.
- **Les types d'éléments** : tout type trivialement copiable, comparable par `==` et
  constructible par défaut (types arithmétiques, `bool`, `std::complex`, énumérations,
  `std::array` et agrégats de ceux-ci). Un type qui possède une ressource (`std::string`) est
  refusé à la compilation. Chunk2 demande en plus un ordre sur la seconde valeur : une
  structure de doubles déclare `auto operator<=>(const T&) const = default;`.
- **Chunk2 ne compresse qu'à sa seconde valeur déclarée.** Sans `set_second_value()`, un
  tableau Chunk2 ne compresse qu'à zéro.
- **Mesurer la mémoire avec le bon outil.** `VmRSS` est en retard à plusieurs threads : voir
  [mesures_performance_fr.md](mesures_performance_fr.md).

## 7. Pour aller plus loin

| pour… | lire |
|---|---|
| toute l'interface du conteneur | le [README.md](../README.md) principal, et la documentation Doxygen (`cmake --build build --target doc`) |
| choisir une variante pour ses données | [variantes_fr.md](variantes_fr.md), et `examples/variant_compare` |
| laisser la taille des chunks s'adapter à l'exécution | [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) (bibliothèques `_dyn`) |
| toutes les variables de réglage et options de compilation | [variables_environnement_fr.md](variables_environnement_fr.md) |
| le fonctionnement interne | [algo_chunk2_fr.md](algo_chunk2_fr.md), [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) |
| un exemple plus complet : champs multi-matériaux, boucles écrites une fois pour les deux conteneurs | `examples/multi-mat` |
