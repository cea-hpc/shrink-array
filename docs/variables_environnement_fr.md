# Variables d'environnement et options de compilation

> English version: [environment_variables.md](environment_variables.md).

> État : octobre 2026. Cette page liste ce que règle un **utilisateur** de la bibliothèque : les
> réglages que couvre la politique de versionnage de [CHANGELOG.md](../CHANGELOG.md). Tout le
> reste -- le réglage du décideur et des allocateurs, les traces, le banc, les crochets de test,
> les options de mise au point, les scripts de compilation -- est dans
> [reglages_developpeur_fr.md](reglages_developpeur_fr.md) : des outils de développement, qui
> peuvent changer ou disparaître à chaque version.
>
> Une variable est lue une seule fois ; `0` ou une valeur vide équivaut à son absence. Une
> variable qui n'est pas lue est ignorée sans message : en cas de doute, vérifier avec
> `grep -rn '"NOM"' chunk fast_allocators`.

***

## 1. Variables d'environnement

| variable | défaut | effet |
|---|---|---|
| `CHUNK_THREAD_CAPACITY` | max(`omp_get_max_threads()`, cœurs) | nombre de rangs pour lequel les structures par thread sont dimensionnées ; ne peut qu'augmenter le défaut (sursouscription). Un rang qui l'atteint ou la dépasse arrête le programme avec un message qui nomme cette variable |
| `CHUNK_REQUIRE_OMPT` | — | builds OMPT : `1` arrête le programme si l'outil OMPT n'a pas été chargé par le runtime (sinon, un message sur stderr), puisque les fins de phase n'auraient alors jamais lieu |
| `CHUNK_RECHUNK_DRY_RUN` | — | `1` : le décideur du rechunk adaptatif analyse et journalise, mais n'applique aucun rechunk (voir [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md)) |

## 2. Options CMake

| option | défaut | effet |
|---|---|---|
| `OMP` | `OFF` | construire avec OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | tailles de chunk statiques construites (cibles `_<taille>`) |
| `CHUNK_BUILD_DYN` | `ON` | construire aussi une bibliothèque à géométrie dynamique (`_dyn`) par variante |
| `CHUNK_STD_ALLOC` | `OFF` | allocateur standard pour les tampons de chunks au lieu de `fast_allocators` (utile sous ASan : chaque tampon est un vrai `malloc`) |
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | Chunk2A : l'emplacement de seconde valeur contient un identifiant de 4 octets au lieu de la valeur (moins de mémoire, +3 % de temps d'écriture) |
| `CHUNK_STATS` | `OFF` | compteurs d'instrumentation et rapport `print_chunk_stats()` |
| `CHECK_OPTIM` | `OFF` | vérifications d'invariants en build optimisé |
| `SANITIZER` | `none` | `thread`, `address` ou `token` (AllocToken, Clang ≥ 22), pour tous les types de build |

## 3. Variante et géométrie

Le programme ne les règle pas : elles viennent avec la cible de bibliothèque qu'il lie
(`chunk::chunk_v2a_256`, `chunk::chunk_v1_dyn`, ...), qui porte les macros
`CHUNK_VARIANT_V1` / `CHUNK_VARIANT_V2` / `CHUNK_VARIANT_V2A`, et `CHUNK_DYNAMIC` pour une
géométrie dynamique (taille choisie à l'exécution, tableau par tableau ; `DYN_CHUNK_SIZE`
est une macro du bench, pas de la bibliothèque). Voir [getting_started_fr.md](getting_started_fr.md).

Les options communes à toutes les bibliothèques d'un build (§2) sont consignées par CMake dans les
en-têtes générés `chunk/build_options.h` et `fast_allocators/build_options.h`, installés avec les
bibliothèques ; les en-têtes les lisent là, quoi que définisse le programme. Un fichier compilé
avec OpenMP contre une bibliothèque construite sans, ou l'inverse, ne compile pas. Un programme qui
écrit la variante ou la géométrie autrement que la bibliothèque qu'il lie ne se lie pas : le
symbole indéfini (`shrink_array::link_check::variant_...`) nomme les réglages avec lesquels il a
été compilé.
