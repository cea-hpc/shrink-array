# L'API en un coup d'œil

> English version: [api.md](api.md).

> Octobre 2026. Une page pour ce qu'un programme appelle, regroupé par tâche. Les détails (paramètres, préconditions, complexité) sont dans la référence Doxygen
> (`cmake --build build --target doc`), où l'on peut chercher ces noms. L'usage en contexte :
> [getting_started_fr.md](getting_started_fr.md).

***

Tous les noms sont dans l'espace de noms `shrink_array`, sauf `build_config` et `chunk_config` (globaux).

## 1. Démarrer et arrêter la bibliothèque

| appel | en-tête | quand |
|---|---|---|
| `ParamChunk::init_indexes(true)` | `chunk/param_chunk.h` | une fois, avant le premier tableau (`true` : gros blocs d'allocation) |
| `release_chunk_pools()` | `chunk/chunk_std.h` | une fois, à la fin, après le dernier tableau |
| `destroy_block_chunk()` | `chunk/chunk_std.h` | juste après `release_chunk_pools()` |
| `ParamChunk::force_compression()` | `chunk/param_chunk.h` | pour compresser maintenant tous les candidats en attente, pour une mesure ou un point de reprise |
| `ParamChunk::ChunkStats()` | `chunk/param_chunk.h` | affiche les compteurs `CHUNK_STATS` (sans effet dans les autres builds) |

## 2. Les conteneurs

| classe | en-tête | compresse les chunks uniformes à… |
|---|---|---|
| `ArrayChunk<T, ind_type = int>` | `chunk/array_chunk.h` | `T{}` |
| `ArrayChunk2<T, ind_type = int>` | `chunk/array_chunk2.h` | `T{}` ou la seconde valeur du tableau |
| `ArrayChunk2A<T, ind_type = int>` | `chunk/array_chunk2A.h` | `T{}` ou n'importe quelle valeur, chunk par chunk |
| `ArrayChunkBits` | `chunk/array_chunk_bits.h` | `bool`, stocké sur un bit par élément (builds Chunk2) |

`ind_type` est le type d'indice : `long` pour plus de 2³¹ éléments. La variante compilée est
celle de la bibliothèque liée (`chunk::chunk_v2a`, ...) ; `chunk_config::use_chunk`,
`chunk_config::use_chunk2` et `chunk_config::use_chunk2a` indiquent laquelle, à la compilation.

## 3. Opérations du conteneur

Ce qu'a un `std::vector`, avec les différences qui comptent. **(S)** marque un point séquentiel :
hors de toute région parallèle, et il invalide les proxys du tableau.

| opération | membres | remarques |
|---|---|---|
| construction | `(n)`, `(first, last)`, `{a, b, c}`, copie (profonde), déplacement | un nouveau tableau lit `T{}` et n'alloue rien |
| taille | `size()`, `empty()`, `capacity()` | |
| lecture | `operator[](i) const`, `at(i) const`, `front()`, `back()`, `operator()(i) const` | renvoient un `T` |
| écriture | `operator[](i)`, `at(i)` | renvoient un proxy d'écriture (§4) |
| référence brute | `operator()(i)` (non const) | un `T&` dans le tampon : **seulement sur un chunk matérialisé**, contourne la bibliothèque |
| agrandir, réduire **(S)** | `resize(n)`, `reserve(n)`, `clear()`, `shrink_to_fit()` | `shrink_to_fit()` replie aussi chaque chunk devenu constant |
| ajout | `push_back(v)`, `emplace_back(...)`, `pop_back()` | **(S)** seulement quand il faut un nouveau chunk |
| remplacement **(S)** | `assign(n, v)`, `assign(first, last)`, `swap(other)`, `operator=` | |
| itération | `begin()`, `end()`, `cbegin()`, `rbegin()`, `rend()` | itérateurs const à accès aléatoire, par valeur |
| compression **(S)** | `compress()` | une passe de compression complète, immédiate (`ArrayChunk`, `ArrayChunk2`, `ArrayChunk2A`) |
| statistiques | `stats()`, `get_nb_chunk_written()`, `print_compression_stats()` | pourcentage et nombre de chunks qui détiennent un tampon ; une ligne par tableau |
| géométrie | `chunk_size()`, `rechunk_to(n)` **(S)** | `rechunk_to()` seulement dans les bibliothèques `_dyn` (sans effet dans les statiques) |

Chunk2 seulement : `set_second_value(v)` **(S)**, avant les écritures, et `get_second_value()`.

Absents : `data()`, les itérateurs mutables, `insert` / `erase` au milieu, un tri en place.

## 4. Le proxy d'écriture

`operator[](i)` sur un tableau non const renvoie un proxy, pas un `T&` :

| usage | effet |
|---|---|
| `T v = a[i];` | lit la valeur |
| `a[i] = v;` | l'écrit (peut matérialiser le chunk, ou être différée tant qu'un compresseur le détient) |
| `a[i] += v;` `-=` `*=` `/=` `%=` `&=` `|=` `^=` `<<=` `>>=` | opérations composées, pour les types qui les ont ; renvoient un proxy |
| `++a[i]`, `--a[i]` | renvoient un proxy ; `a[i]++` renvoie l'ancienne valeur sous forme de `T` |
| `a[i] == v`, `a[i] != v` | comparent la valeur courante ; les autres comparaisons passent par la conversion en `T` |

`auto x = a[i];` garde le proxy, qui suit l'élément. Un proxy ne doit pas survivre au
stockage du tableau : toute opération **(S)** l'invalide.

## 5. Opérations sur des plages

`chunk/array_chunk_range_ops.h`. Une décision par chunk au lieu d'une par élément.

| appel | contexte |
|---|---|
| `fill_range(a, first, last, v)` | séquentiel : `[first, last)` mis à `v` |
| `copy_range(a, first, last, src_first)` | séquentiel : `[first, last)` copié depuis un itérateur |
| `fill_range_mt(a, first, last, v)` | appelable par plusieurs threads à la fois, sur des plages disjointes (un chunk peut être partagé) |

## 6. Phases OpenMP

`chunk/manage_ompt.h`. Nécessaires seulement quand `build_config::has_not_ompt` (libgomp de GCC) ; sans
effet sinon, et dans les builds séquentiels.

| appel | où |
|---|---|
| `begin_parallel_phase()` | dans une région parallèle, par chaque thread, avant son premier accès |
| `finalize_parallel_phase()` | après la région, une fois |
| `finalize_parallel_phase_only_flush()` + `finalize_parallel_phase_after_flush()` | la même chose, en deux : la première dans la région (elle émet une barrière), la seconde après |
| `finalize_barrier_phase()` | dans une région, après un `omp barrier` explicite : rend les écritures d'avant visibles après |

## 7. Taille de chunk adaptative (bibliothèques `_dyn`)

| appel | en-tête | ce qu'il fait |
|---|---|---|
| `ChunkSizeAdvisor<T>::analyze(a)` | `chunk/chunk_size_advisor.h` | estime la mémoire du tableau à chaque taille candidate ; en lecture seule, à un point de repos |
| `RechunkDecider` avec `RechunkPolicy` | `chunk/rechunk_decider.h` | décide, tick après tick, si un rechunk en vaut la peine |
| `decider.consider<T>(a, tick, iter, "name")` | | analyse le tableau et applique le rechunk si la politique est d'accord **(S)** |
| `decider.print_rechunk_log()` | | la table des rechunks effectués |
| `a.rechunk_to(n)` **(S)** | | rechunke un tableau à `n` éléments par chunk |

La politique, ses valeurs par défaut et les résultats mesurés : [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md).

## 8. Configuration à la compilation

| nom | signification |
|---|---|
| `build_config::use_openmp` | construit avec OpenMP |
| `build_config::has_ompt`, `build_config::has_not_ompt` | construit avec l'outil OMPT, ou non |
| `build_config::chunk_dynamic` | géométrie dynamique (bibliothèque `_dyn`) |
| `chunk_config::use_chunk`, `chunk_config::use_chunk2`, `chunk_config::use_chunk2a` | la variante |
| `chunk_config::stats` | build `CHUNK_STATS` |

Variables d'environnement et options de build : [variables_environnement_fr.md](variables_environnement_fr.md).
