# Réglages de développement

> English version: [developer_settings.md](developer_settings.md).

> État : octobre 2026. Les interrupteurs des développeurs de la bibliothèque : réglage du décideur
> et des allocateurs, traces, rechunks forcés, banc, macros de compilation, options de mise au
> point et de compilation, scripts de compilation. Ce sont des **outils de développement** : hors
> de la politique de versionnage de [CHANGELOG.md](../CHANGELOG.md), ils peuvent changer ou
> disparaître à chaque version. Ce que règle un utilisateur de la bibliothèque est dans
> [variables_environnement_fr.md](variables_environnement_fr.md).
>
> Sauf mention contraire, une variable est lue une seule fois, et `0` ou une valeur vide équivaut à
> son absence -- sauf pour les traces du §2 marquées *définie*, actives dès qu'elles sont définies
> (même à `0`), et pour les variables dont la ligne dit ce que fait `0`. Une variable qui n'est pas
> lue est ignorée sans message : en cas de doute, vérifier avec
> `grep -rn '"NOM"' chunk fast_allocators examples`.
>
> **Dates de retrait.** Un interrupteur qui garde un ancien comportement, ou un choix qu'aucun build
> ne teste, porte une date de retrait dans sa ligne ; à cette date il part, avec son chemin de
> code, sauf si une mesure consignée ici justifie de le garder. Un nouvel interrupteur de ce genre
> reçoit sa date à son ajout.

***

## 1. Réglage du décideur, de la reconstruction et des allocateurs

| variable | défaut | effet |
|---|---|---|
| `CHUNK_ALLOC_AFFINITY_EAGER` | selon la variante | `1` : chaque gros tableau a son jeu d'allocateurs dès sa création (défaut de Chunk, v1) ; `0` : seulement au premier rechunk (défaut de Chunk2 et Chunk2A). Voir [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) §5 |
| `CHUNK_RECHUNK_MAX_COARSEN` | 4 | facteur maximal d'un grossissement (1 = sans limite) |
| `CHUNK_ADVISOR_MIN_GAIN_MB` | 8 | seuil absolu de gain côté données, en Mo (0 le supprime) |
| `CHUNK_ADVISOR_GRID_BITS` | 2 | les tailles candidates de l'advisor entre les puissances de deux : les multiples de 64 dont C / 64 a au plus ce nombre de bits significatifs (2 : 192, 384, 768... ; 3 : aussi 320, 448, 640 ; 1 : puissances de deux seules). Voir [rechunk_adaptatif_fr.md](rechunk_adaptatif_fr.md) §2 |
| `CHUNK_COARSEN_MIN_META_MB` | 1 | gain minimal de métadonnées d'un grossissement, en Mo |
| `CHUNK_COARSEN_MIN_META_RATE` | 250 | gain de métadonnées minimal par seconde de reconstruction estimée, en Mo/s |
| `CHUNK_RECHUNK_SCHEDULE` | `guided,16` | ordonnancement OpenMP de la reconstruction (`kind[,chunk]`) |
| `CHUNK_RECHUNK_SLICES` | 16 | nombre de tranches de la reconstruction de v1 |
| `CHUNK_RECLAIM_MIN_SLABS` | 0 | un balayage ne rend les slabs vides au système que s'il y en a au moins autant |
| `CHUNK_RECLAIM_THREADS` | 1 | threads qui émettent les `madvise` d'un balayage ; `0` : `omp_get_max_threads()` |
| `CHUNK_RECHUNK_RECYCLE` | actif | reconstruction de v1 : les slabs qu'une tranche vient de vider vont à la tranche suivante au lieu d'être rendus au système ; `0` le désactive. *Retrait prévu : 31/12/2026.* |
| `CHUNK_RECHUNK_KEEP_RECYCLED_MB` | 0 | reconstruction de v1 : à sa fin, jusqu'à ce nombre de Mio par arène des slabs recyclés qu'aucune tranche n'a repris restent résidents (répartis à parts égales entre les rangs) pour les allocations suivantes, au lieu d'être rendus au système ; `0` : tous sont rendus |
| `CHUNK_RECHUNK_KEEP_RECYCLED` | inactif | `1` : idem, sans plafond (le RSS garde chaque reliquat jusqu'à sa réutilisation) |
| `CHUNK_RECHUNK_REUSE_LIST` | actif | un rechunk réutilise le tampon de la liste de chunks du rechunk précédent (pages déjà résidentes) ; `0` : un nouveau à chaque fois. *Retrait prévu : 31/12/2026.* |
| `CHUNK_THP` | 0 | non défini : le mode THP du noyau décide. `never` : pas de huge pages transparentes sur les arènes de slabs (`MADV_NOHUGEPAGE`) -- en mode `always`, un slab devient sinon résident en entier dès qu'il est touché ; comparer les deux avec le RSS et AnonHugePages de `smaps_rollup`. Expérimental : `1` les demande (`MADV_HUGEPAGE`) sur les arènes de slabs et la région cst2. Ignoré, avec un message, quand la huge page dépasse 2 Mio (noyau à pages de 64 Kio : 512 Mio). Coût : un slab entamé devient résident en entier. *Retrait prévu : 31/12/2026.* |
| `CHUNK_REUSE_SHARE_DIV` | 50 | allocateur rapide : un rang garde de côté jusqu'à chunks / (ce nombre × threads) tampons libérés par type d'élément (par tableau pour un tableau qui a ses propres allocateurs) ; les chunks comptés incluent les compressés |
| `CHUNK_REUSE_FLOOR_KB` | non défini | plancher de ce budget en Kio par rang, converti en tampons selon la taille de chunk ; non défini : 500 tampons (100 pour un tableau qui a ses propres allocateurs), quelle que soit leur taille ; `0` : pas de plancher |
| `CHUNK_RECLAIM_PHASE_KEEP` | non défini | `K` : après chaque passe de compression, rendre au système (`madvise`) les slabs vides, en gardant `K` par instance d'allocateur ; non défini : seulement au rechunk, à la destruction, à `shrink_to_fit()` |
| `CHUNK_DEFERRED_POOL_KB` | 64 | taille des slabs des pools d'écritures différées, en Kio (les slabs d'opérations ne sont jamais rendus : plus gros, moins d'appels à l'allocateur mais plus de mémoire résidente) |

## 2. Traces et diagnostics

Les traces sont des messages du journal de la bibliothèque, choisis par `CHUNK_LOG` (voir
[variables_environnement_fr.md](variables_environnement_fr.md)). Les variables des anciennes
traces marchent toujours : chacune règle le niveau qui lui correspond, avec un avertissement
de dépréciation.

| réglage | ce qu'il affiche |
|---|---|
| `CHUNK_LOG=rechunk=info` | les décisions du décideur (`decide <tableau>: re-chunk S=.. -> S'=.. ... [APPLIED]`, ou `[dry-run] would re-chunk`) |
| `CHUNK_LOG=rechunk=debug` (autrefois `CHUNK_RECHUNK_TRACE`) | en plus, les verdicts d'attente, et pour chaque rechunk la durée des phases et des tranches |
| `CHUNK_LOG=rechunk=trace` (autrefois `CHUNK_RECHUNK_TRACE_SLICES`) | en plus, la mémoire à chaque tranche de la reconstruction de v1 |
| `CHUNK_LOG=memory=info` (autrefois `CHUNK_RECHUNK_MEMTRACE`) | à chaque passage de l'advisor : mémoire exacte, dont la part résidente de l'arène des tampons |
| `CHUNK_LOG=memory=debug` (autrefois `CHUNK_RECHUNK_TRACE_MEM`) | en plus, le côté mémoire de la trace des rechunks : mémoire exacte à chaque étape (ligne `rss`), occupation de la nouvelle géométrie, chunks repliés, refusés et installés directement, recensement des allocateurs. Coûteux (des secondes par run à des dizaines de Go) : un canal à part des mesures de temps |
| `CHUNK_LOG=advisor=info` | builds de vérification : à la destruction d'un tableau, la taille de chunk qui aurait pris le moins de mémoire, quand elle en économise au moins 10 % |
| `CHUNK_ADVISOR_TABLE` | `1` ou `=nom` : la table complète de l'advisor à chaque passage (tous les tableaux ou un seul) |
| `CHUNK_RECHUNK_CHURN` | `1` ou `=nom` : matérialisations par passage et estimation du coût passager qu'elles impliquent (niveau info du canal rechunk, qu'elle active) |
| `CHUNK_AFFINITY_TRACE` (*définie*) | chaque jeu d'allocateurs par tableau créé (clé, taille, budget) |
| `CHUNK_ARENA_REPORT` | le bilan de l'arène des tampons (slabs distribués, rendus au système) |
| `CHUNK_RECYCLE_TRACE` | une ligne par bloc recyclé du pool des écritures différées (au lieu d'un résumé) |
| `CHUNK_LEDGER_REPORT` (*définie*) | le rapport du registre des blocs (builds avec registre) |
| `CHUNK_OMPT_STATS` | statistiques des régions de synchronisation (builds OMPT) |
| `CHUNK_CST2_PROBE` (*définie*) | Chunk2A : rapport du registre des secondes valeurs à chaque passage (banc) |
| `CHUNK_CST2_RECLAIM` | Chunk2A, `N` : toutes les N itérations, libère les secondes valeurs qu'aucun chunk n'utilise plus (`reclaim_second_values()`) et affiche leur nombre, puis pour chaque type les identifiants en usage face à la capacité, libres, les libérations et les refus (`second_value_stats()`) (banc, tout build) |
| `ALLOC_TOKEN_REPORT` | les builds `SANITIZER=token` impriment toujours, à la sortie, les allocations par jeton (les 40 plus grosses) ; elle en fixe le nombre : `0` aucune, `N` les N plus grosses, `all` toutes |
| `FAST_ALLOC_TRACE` (*définie*) | une ligne `[MUNMAP]` par slab vidé que la réserve d'un pool ne garde pas et qui est libéré (muet par défaut) |

## 3. Rechunks forcés (tests)

| variable | effet |
|---|---|
| `CHUNK_FORCE_RECHUNK=<taille>` | force un rechunk vers cette taille, même si le décideur ne le ferait pas |
| `CHUNK_FORCE_RECHUNK_AT=<itération>` | retarde ce rechunk forcé jusqu'à cette itération |
| `CHUNK_FORCE_RECHUNK_TOGGLE=1` | rechunk à chaque passage, en alternant avec `CHUNK_FORCE_RECHUNK_ALT` (64 par défaut) |

## 4. Le banc `shrink_bench`

| variable ou option | effet |
|---|---|
| `CHUNK_ADVISOR_PERIOD` | période des passages de l'advisor, en itérations (défaut `max(10, iter_max / 20)` ; 0 le désactive) |
| `SHRINK_BENCH_ADVISE_ALL` | défaut : tous les tableaux sauf `tab_bool` sont confiés au décideur ; `0` : les cinq tableaux historiques |
| `SHRINK_BENCH_EXACT_RSS` | `1` : toutes les lignes de mémoire sont exactes (coûteux) |
| `SHRINK_BENCH_RSS_SAMPLE_MS` | période d'échantillonnage du pic exact, en ms (désactivé par défaut) |
| `CHUNK_BENCH_MEM_PERIOD` | période de la ligne de mémoire, en itérations |
| `CHUNK_BENCH_KSUM_PERIOD` | période de la ligne de checksum, en itérations |
| `SHRINK_BENCH_AUDIT_FREQ` | période de l'audit de compression (builds de vérification) |
| `CHUNK_BENCH_ULONG_BANDS` | nombre de secondes valeurs distinctes du tableau `unsigned long` (4 096 ; 0 = sans limite) |
| `CHUNK_MALLOC_ARENAS` | nombre d'arènes `malloc` de la glibc (0 = inchangé) |
| `SHRINK_BENCH_MALLOC_TRIM` | `1` : `malloc_trim` à chaque ligne de mémoire, et pas seulement aux points de contrôle (« after init », « end ») ; déforme la courbe et coûte dans la boucle chronométrée |
| `SHRINK_BENCH_KEEP_POOLS` | `1` : pas de `trim_reuse_pools` aux points de contrôle (les tampons parqués et, avec l'allocateur rapide, les slabs vidés restent résidents) |
| `BENCH_SCHED_BLOCK` | taille des blocs d'ordonnancement des phases du banc (64 ; 0 = tranches contiguës) |
| `NO_IO`, `COMPRESS` | équivalents de `--no_io` et `--compress` |
| `--no_mask`, `--mask_scale <x>` | sans `tab_mask` ; ou sa taille, `x × sz` (défaut 1/32) |
| `--no_struct` | sans les tableaux de structures (`Point3`, `Cell`) |

## 5. Macros de compilation

Valeurs de réglage auxquelles les en-têtes donnent un défaut, et crochets de test. Les macros de
variante et de géométrie, et la façon dont les options communes parviennent aux en-têtes, sont
dans [variables_environnement_fr.md](variables_environnement_fr.md) §3.

| macro | effet |
|---|---|
| `CHUNK_DYN_MAX_SIZE=<n>` | géométrie dynamique : plus grande taille de chunk qu'un rechunk peut atteindre, puissance de deux (4096) ; dimensionne les tampons partagés des valeurs constantes, `cst_buffer_size × sizeof(T)` par seconde valeur distincte |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<octets>` | seuil de métadonnées au-delà duquel un tableau peut avoir ses propres allocateurs (16 Mio ; 0 = jamais) |
| `CHUNK_COMPRESS_MAX_YIELDS=<n>` | attente d'un compresseur garé avant de renoncer à sa passe (10 000 *yields* ; 0 : renoncer au premier écrivain vu). Voir [synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) |
| `CHUNK_WRITE_WAIT_MAX_YIELDS=<n>` | attente d'un écrivain devant un compresseur avant de différer son écriture (64 *yields*) |
| `CHUNK_READ_RECLAIM_SLOTS=<n>` | nombre de threads lecteurs suivis individuellement par la récupération des tampons (1 024) |
| `CHUNK_STATS_MAX_THR=<n>` | nombre de rangs des compteurs `CHUNK_STATS` (1 024) |
| `CHUNK_TEST_READ_DELAY_EVERY=<n>`, `CHUNK_TEST_READ_DELAY_US=<µs>` | crochet de test : une lecture sur N d'un thread qui l'a demandé attend (50 µs), pour élargir la fenêtre lecteur / récupération (`read_reclaim_validation`) |
| `CHUNK_TEST_PASS_DELAY_EVERY=<n>`, `CHUNK_TEST_PASS_DELAY_US=<µs>` | crochet de test : délai (200 µs) dans une passe de compression sur N |

Et une macro propre à Chunk2A : `CHUNK2A_COMPRESS_LEVEL=<0|1|2|3>` (0 par défaut) choisit le
jeu de seuils de déclenchement des passes de compression (`chunk2A_def.h`). Seul le niveau 0 est
construit et testé. *Retrait des niveaux 1 à 3 prévu : 31/12/2026.*

## 6. Options CMake

Les options de [variables_environnement_fr.md](variables_environnement_fr.md) §2 ne sont pas reprises.

| option | défaut | effet |
|---|---|---|
| `SHRINK_BENCH_DYN_SIZE` | 64 | taille de départ des cibles `_dyn` |
| `SHRINK_BENCH_INT64` | `OFF` | cibles à indice 64 bits (`_int64`) |
| `SHRINK_INTEL_LIBM` | `ON` | x86-64 : lier la libm d'Intel si elle est trouvée |
| `SHRINK_ARM_LIBM` | `OFF` | aarch64 : lier `libamath` d'ArmPL (aucun gain mesuré sur Grace) |
| `CHUNK_ICPX_O0_WORKAROUND` | `OFF` | contournement d'une erreur interne d'icpx sur trois fichiers |
| `CHUNK_MM_PROTECT` | `OFF` | Chunk, Chunk2 : tampons partagés des valeurs constantes en lecture seule (`mprotect`) ; une écriture parasite dans un chunk compressé fait une erreur de segmentation au lieu de corrompre la valeur (build de mise au point ; `compile_all.sh` le construit pour que ses branches continuent de compiler) |
| `CHUNK_LIB_UNITY` | `ON` | chaque archive de variante (`chunk_v2a_256`, ...) compilée en une seule unité (unity build, 4 à 6 fois moins de compilation) ; un fichier qui a ses propres options de compilation reste à part |
| `SHRINK_BENCH_UNITY` | `ON` | chaque exécutable `shrink_bench` compilé en une seule unité (unity build, 3,5 à 4 fois moins de compilation) ; `OFF` pour comparer des temps avec des campagnes antérieures, l'inlining entre fichiers du banc pouvant les changer ; `SHRINK_BENCH_UNITY_BATCH` fichiers par unité (0 = tous) |
| `CHUNK_VALIDATION_UNITY` | `ON` | les harnesses de validation regroupés en un exécutable par variante et géométrie (unity build) ; `OFF` : un exécutable par harness |
| `CHUNK_BUILD_VALIDATION` | selon le build | compiler les harnesses de `examples/validation`, et avec eux les crochets de test (`CHUNK_TEST_HOOKS`) ; par défaut `ON` en Debug, avec `CHECK_OPTIM` ou un sanitizer, `OFF` ailleurs (non mis en cache : suit la configuration courante) |
| `CHUNK_RECLAIM_EMPTY_SLABS` | `ON` | rendre au système (`madvise`) les slabs vidés aux balayages explicites (rechunk, destruction) ; `OFF` les garde résidents. *Retrait prévu : 31/12/2026.* |
| `FAST_ALLOC_CHECK_LISTS` | `OFF` | niveau de debug renforcé des listes de slabs : en Debug, vérification complète de la liste à chaque insertion et retrait (lent) ; un Debug ordinaire garde les vérifications légères |
| `HAVE_AGGRESSIVE_COMPRESSION` | `OFF` | compression « agressive » : la passe entre deux phases est sautée, et toute la compression se fait pendant les régions parallèles, à des instants imprévisibles. Plus lent, mais sollicite les chemins concurrents |
| `CHUNK_FORCE_LEDGER` | `OFF` | garde le registre des blocs (`pool_ledger.h`) actif sous OpenMP, où il est éteint par défaut : son mutex global sérialise chaque allocation et libération. Détecte double libération et remise illégale en réserve |
| `CHUNK_DEFERRED_DRAIN_THRESHOLD` | vide (4 096) | seuil de vidage de sa propre file par un rang |
| `CHUNK_FORCE_DEFER_EVERY` | 0 | une écriture sur N par le chemin différé, dans tout l'arbre (validation ; 32 threads au plus) |
| `CHUNK_DEBUG` | selon le build | les vérifications de mise au point de la bibliothèque ; `ON` avec `CMAKE_BUILD_TYPE=Debug` (non mis en cache : suit la configuration courante) |
| `CHUNK_POOL_NO_REUSE` | `OFF` | pools d'opérations différées sans liste libre : chaque bloc un vrai `new` / `delete` (ASan, TSan) |
| `CHUNK_DO_CHECK` | `OFF` | chaque valeur écrite vérifiée contre NaN |
| `CHUNK_RECHUNK_LEDGERTRACE` | `OFF` | tracer chaque transition du registre des blocs |
| `CHUNK_RING_LOG_DEBUG`, `CHUNK_RING_LOG_OP` | `OFF` | journal circulaire de mise au point de tous les événements (lourd) ; des seuls événements du cycle de vie des opérations (léger) |
| `CHUNK_GLIBCXX_ASSERTIONS` | `OFF` | assertions de la libstdc++ (`_GLIBCXX_ASSERTIONS`) dans la bibliothèque et ses consommateurs |
| `CHUNK_UBSAN` | `OFF` | UndefinedBehaviorSanitizer (à associer à ASan, pas à TSan) |
| `CHUNK_ALIGN_CHECK` | `OFF` | arrêt si un bloc de slab est mal aligné pour son type |
| `CHUNK_NO_IPA_ICF` | `OFF` | désactive `-fipa-icf` (GCC fusionne sinon des fonctions identiques, ce qui brouille les piles d'appels) |
| `CHUNK_WARNINGS`, `CHUNK_WERROR` | `ON`, `OFF` | `-Wall -Wextra` sur tous les builds ; avertissements en erreurs en Debug |
| `CHUNK_HEAVY_JOB_POOL` | vide | pool de jobs Ninja pour les exécutables compilés en une unité (limite la mémoire de compilation) |
| `USE_LIBCXX` | `ON` | builds Clang : libc++ au lieu de la libstdc++ |
| `USE_GPERF`, `ENABLE_IWYU` | `OFF` | profilage gperftools ; include-what-you-use pendant la compilation |
| `ONEAPI_XHOST`, `ONEAPI_NO_INTEL_LIBS`, `ONEAPI_FAST_FP` | `OFF` | icpx : binaire réservé à Intel (`-xHost`) ; aucune bibliothèque d'exécution Intel ; garder `-fp-model=fast` |

Pour les builds clang sur x86-64, `CHUNK_LIBMVEC_LINKS` (dans le cache, calculée à la
configuration) indique si la bibliothèque vectorielle de la glibc fournit les fonctions
mathématiques des bancs ; sinon, leurs boucles restent scalaires (glibc < 2.35).

## 7. Les scripts de compilation

`scripts/compile_all.sh` configure et construit toute la matrice (GNU, LLVM, oneAPI, AOCC,
NVHPC, sanitizers) ; il délègue les builds GNU à `scripts/compile_all_gnu.sh`, en lui
transmettant les options communes. `--help` donne la liste complète.

| option | effet |
|---|---|
| `--dryrun` | afficher les commandes sans les exécuter |
| `--build` | sans reconfigurer (reconstruction seule) |
| `--fresh-cache` | supprimer le `CMakeCache.txt` de chaque build sélectionné avant de configurer |
| `--debug-only` | seulement les builds Debug |
| `--only=<c>`, `--no-<c>` | un seul compilateur, ou sans lui (`gnu`, `oneapi`, `nvhpc`, `llvm`, `aocc`) |
| `--with-nvhpc` | ajouter NVHPC, désactivé par défaut (environ 1 h 30 par build sur x86, sans gain) |
| `--no-sanitizers` | sans les builds LLVM sous sanitizer |
| `--with-ledger` | ajouter `cmake-build-omp-gnu-debug-ledger` (Debug OpenMP avec `CHUNK_FORCE_LEDGER=ON`), désactivé par défaut ; un `--target` qui le sélectionne le construit aussi |
| `--with-token-sanitizer` | ajouter `cmake-build-omp-llvm-debug-std-alloc-token-sanitizer` (`SANITIZER=token`), désactivé par défaut ; un `--target` qui le sélectionne le construit aussi ; `--no-sanitizers` l'en retire |
| `--target=<glob>` | seulement les configurations dont le nom correspond (`omp-llvm`, ou un motif du shell entre apostrophes) ; un motif qui ne correspond à rien est une erreur |
| `--exec=<glob>` | seulement les cibles Ninja correspondantes dans chaque build (`shrink_bench_chunk_64`, ou un motif du shell entre apostrophes) |
| `--jobs=<n>` | nombre de configurations compilées en même temps (4 par défaut) ; le budget de cœurs est partagé entre elles |
| `--no-hardening` | sans `CHUNK_GLIBCXX_ASSERTIONS` ni `CHUNK_UBSAN` sur les builds Debug et sanitizer |
| `--llvm-cs=<n>` | ajouter un build LLVM avec `CHUNK_SIZE=n` |

`compile_all_gnu.sh` accepte en plus `--no-ipa-icf` (`CHUNK_NO_IPA_ICF=ON`). La mémoire de
compilation par job se règle par `CHUNK_MEM_PER_JOB_MB` (3 072 Mo par défaut, le double pour
les builds sous sanitizer).
