# Contribuer à shrink-array

> English version: [contributing.md](contributing.md).

> Octobre 2026. Comment construire, tester, mesurer et documenter une modification. Lire d'abord
> [algo_zero_one_value_fr.md](algo_zero_one_value_fr.md) : l'essentiel de la bibliothèque est la mécanique qu'il décrit.

***

## 1. Build

Un build de développement d'une seule configuration :

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DOMP=ON
cmake --build build
```

Un build Debug construit une seule taille statique, 256, au lieu de deux (`CHUNK_STATIC_SIZES=auto`), ce qui
économise une bonne part du temps de build ; `-DCHUNK_VARIANTS=zero_any` construit une seule variante au lieu de trois, avec ses seuls exemples et
harnais. Un build Debug, `-DCHECK_OPTIM=ON` ou un sanitizer construit aussi les harnais de validation
(`examples/validation`).

La matrice complète (GNU, LLVM, oneAPI, AOCC, NVHPC, sanitizers, statistiques, std-alloc...) est
`scripts/compile_all.sh`. Ses options sont dans [reglages_developpeur_fr.md](reglages_developpeur_fr.md)
§7. Celles qu'on utilise tous les jours :

| option | usage |
|---|---|
| `--target=<glob>` | ne construire que certaines configurations (un motif du shell, entre apostrophes) |
| `--exec=<glob>` | ne construire que certains exécutables dans chaque configuration |
| `--fresh-cache` | après une modification d'un `CMakeLists.txt` ou d'une option, pour qu'aucune valeur périmée du cache ne soit utilisée |
| `--jobs=<n>` | moins de configurations à la fois quand le nœud manque de mémoire (`CHUNK_MEM_PER_JOB_MB` fixe la mémoire par compilateur) |
| `--with-ledger` | ajoute le build Debug avec le registre des blocs (ledger) (`CHUNK_FORCE_LEDGER=ON`), pour le travail sur les allocateurs |

## 2. Tests

`ctest` dans un répertoire de build lance les harnais et les vérifications des sources ; `ctest -L validation`
seulement les premiers. `python/bench_and_test.py` les lance sur un ensemble de builds, avec le banc en
mode check et le balayage de performance (voir `docs/example_bench_test.txt`).

Les défauts de cette bibliothèque sont surtout des **courses rares**. Une modification d'un chemin concurrent n'est pas
testée tant qu'elle n'a pas tourné sous ThreadSanitizer, à plusieurs nombres de threads, et de nombreuses fois : utiliser
`python/multi_launch_stress.py` (voir `docs/example_multi_launch_stress.txt`).

Que lancer, selon ce que touche la modification :

| modification | lancer au moins |
|---|---|
| moteur d'écriture, protocole de Dekker, gestionnaire de compression, écritures différées | `deferred_path_validation`, `read_reclaim_validation` (aussi avec `CHUNK_TEST_READ_DELAY_EVERY`, et sous ASan dans un build std-alloc), les builds TSan, `multi_launch_stress.py` |
| marquage pour la compression, les passes | `mark_path_validation`, `compress_roundtrip`, le banc en mode check |
| allocateurs, listes de réutilisation, slabs | `pool_accounting`, `slab_geometry_validation`, `blk_allocator_validation`, le build avec registre des blocs (ledger) (`--with-ledger`), ASan dans un build std-alloc |
| rechunk, géométrie dynamique | `rechunk_validation`, `dyn_geometry_validation`, le banc avec `--force_rechunk <taille> --force_rechunk_toggle` |
| opérations sur des plages | `fill_range_mt_validation`, `fill_range_mt_zero_and_zero_one_value_validation`, `range_counters_validation` |
| rangs des threads, régions | `thread_growth_validation`, `nested_region_validation` |
| interface du conteneur | `iterator_validation`, `swap_move_validation`, `stale_proxy_validation`, `open_types_validation`, `index64_validation` |
| documentation | `check_docs` |

Les vérifications des sources tournent avec les autres (`ctest -R check_`) :

| test | règle |
|---|---|
| `check_includes` | chaque `#include` dit pourquoi il est là ; les en-têtes système et ceux du projet ne sont pas mélangés dans un même bloc |
| `check_naming` | la convention de nommage de la section 3, pour ce qu'une ligne en montre : constantes, énumérateurs, fonctions et variables en snake_case, types, alias et paramètres de template en PascalCase, aucun nom réservé en `__` |
| `check_inline_bodies` | aucun corps de méthode de plus de 5 lignes dans une classe de `chunk/` ou `fast_allocators/` : le corps est dans le compagnon `<en-tête>_inline.h` de l'en-tête |
| `check_headers` | chaque fichier C++ commence par l'en-tête de licence Apache (années à jour) et ne porte pas de tampon `Created by` de l'IDE ; les scripts et fichiers de build (`.py`, `.sh`, `.cmake`, `CMakeLists.txt`) portent `# Copyright ... CEA` et `# SPDX-License-Identifier: Apache-2.0` ; les en-têtes vendorés, la ligne SPDX de leur licence d'origine ; `scripts/check_headers.py --fix` corrige |
| `check_raw_reads` | chaque lecture brute d'un tampon de chunk dans la bibliothèque dit pourquoi elle est sûre |
| `check_tracked` | chaque fichier cité par les sources (un `#include` entre guillemets, un import Python relatif, une source ou un harnais CMake, un lien Markdown) est suivi par git, pas seulement présent sur le disque |
| `check_docs` | liens, noms de code cités dans les documents, paires anglais/français, pas de marqueurs d'historique dans les commentaires, les fichiers cités dans les commentaires existent |

## 3. Règles de code

- **C++20.** La branche `cpp23` porte le même code avec C++23 obligatoire. La branche `cpp26`
  essaie le C++26 (réflexion, contrats, peut-être modules) et n'utilise que ce que GCC 16
  implémente : elle n'est construite et testée qu'avec GCC 16.
- **Anglais** pour le code, les commentaires et les messages de commit.
- **Aucun commit de fixup n'est poussé.** Un commit `fixup!`, `squash!` ou `amend!` est fusionné
  dans sa cible avant le push (`git rebase -i --autosquash origin/<branche>`) : une fois publié,
  il ne peut plus que rester. La CI refuse un push qui en apporte un.
- **clang-format** avec le `.clang-format` du dépôt, sur les lignes que vous touchez.
- **Les noms.** Une convention : celle de la bibliothèque standard pour ce qu'elle nomme, et le
  PascalCase pour les types :

  | quoi | casse | exemple |
  |---|---|---|
  | fonctions, méthodes, variables, paramètres, namespaces | snake_case | `get_chunk_list`, `num_tab` |
  | données membres privées et protégées | snake_case terminé par `_` | `chunk_list_`, `num_` |
  | constantes (`constexpr`) et énumérateurs | snake_case | `cst_buffer_size`, `ev_flush_begin` |
  | classes, structs, concepts, alias de type, paramètres de template | PascalCase | `zero_any::Array`, `AllocT`, `ChunkT` |
  | macros | MAJUSCULES | `CHUNK_SIZE` |

  Les MAJUSCULES sont réservées aux macros : une constante ou un énumérateur écrit comme une macro
  se lit comme une macro, et peut en rencontrer une. Les formes de la bibliothèque standard
  gardent leur graphie : les types membres de ses protocoles (`value_type`, `size_type`, ... ;
  nos propres types membres sont en PascalCase, `ChunkType`), les traits et les tags (`is_cplx`,
  `rebind`). De même pour les types publics `md::` de `chunk/mdspan.h` et
  `chunk/mdspan_tile_ops.h`, écrits comme `std::mdspan` écrit les siens (`layout_tiled`,
  `accessor`, `constant_boundary`) : une vue écrite pour `std::mdspan` se lit de même avec eux
  (`STD_STYLE_API` dans `scripts/check_naming.py`) ; leurs aides `detail::` sont en PascalCase.
  Dans un nom en snake_case, la
  variante s'écrit `zero_any`, comme dans les noms de fichiers (`zero_any_array.h`). Les
  globales prennent le préfixe `g_` ; aucun nom ne commence par `__` (réservé). Un nom public qui
  change garde son ancienne graphie en alias `[[deprecated]]` jusqu'à la 1.0. `check_naming`
  vérifie ce qu'une ligne montre ; le `readability-identifier-naming` de clang-tidy vérifie le
  reste (le suffixe des membres, les paramètres) avec `scripts/clang_tidy_naming.yaml`.
- **Les corps longs vont dans le compagnon `<en-tête>_inline.h`.** Une méthode dont le corps dépasse
  5 lignes est déclarée dans la classe et définie dans `internal/<en-tête>_inline.h`, inclus en
  fin d'en-tête ; le commentaire Doxygen reste sur la déclaration. Les accesseurs de quelques
  lignes restent dans la classe. `check_inline_bodies` le vérifie ;
  `scripts/check_inline_bodies.py --apply` déplace ce qu'il liste (puis clang-format sur les fichiers).
- **Pas de préprocesseur là où `if constexpr` suffit.** Les règles qui décident quels `#ifdef` peuvent disparaître,
  et ceux qui doivent rester, sont écrites en tête de `examples/shrink_bench/bench_config.h`.
- **Les commentaires énoncent la règle, pas l'histoire.** Écrire pourquoi le code est tel qu'il est, et ce qui
  casse s'il change. Ne pas écrire comment le bug a été trouvé, quelle version l'avait, les nombres d'exécutions,
  les dates ou les noms de tickets : cela va dans le message de commit. `check_docs` rejette les
  marqueurs habituels (`fwNN`, `SEGV hunt`, `BUG FIX`, `FIX (`, une date comme `Sept '26` ou
  `2026-09-14`), dans les sources C++, les scripts et les fichiers CMake.
- **Doxygen** (`@brief`, `@param`, `@return`, `@warning`) sur tout ce qu'un utilisateur appelle : les
  classes et fonctions de [api_fr.md](api_fr.md). `cmake --build build --target doc` doit rester
  sans avertissement (`ctest -R check_doxygen` quand Doxygen est installé).
- **Code sans verrou** : suivre les règles de [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md)
  §6, et écrire l'argument d'ordonnancement à côté de chaque atomique qui en a besoin.
  Quand la modification touche un protocole, lancer aussi `formal/run_genmc.sh`
  ([formal_models_fr.md](formal_models_fr.md)).

## 4. Affirmations de performance

Une modification faite pour la vitesse ou la mémoire s'accompagne d'une mesure prise comme le décrit
[mesures_performance_fr.md](mesures_performance_fr.md) : la mémoire avec le compteur
exact, le temps sur le même binaire, les modes alternés, au moins trois tours, et seul un écart
bien au-dessus du bruit (10 % sur les bancs dynamiques à 64 threads) constitue un résultat.

## 5. Documentation et changelog

- Une modification visible par l'utilisateur met à jour les documents qu'elle touche, **dans les deux langues**. L'anglais est
  la référence ; la version française suit. `check_docs` échoue quand les deux divergent en
  structure.
- Ajouter une entrée sous _Unreleased_ dans `CHANGELOG.md` (Added, Changed, Deprecated, Removed,
  Fixed). La politique de versionnage en tête de ce fichier dit quel numéro la prochaine version
  incrémente.
- Une nouvelle variable d'environnement, macro ou option CMake va dans
  [variables_environnement_fr.md](variables_environnement_fr.md) si un utilisateur de la
  bibliothèque doit la régler, dans [reglages_developpeur_fr.md](reglages_developpeur_fr.md)
  sinon. Un interrupteur qui garde un ancien comportement pour une comparaison y reçoit sa date de
  retrait dès son ajout.
