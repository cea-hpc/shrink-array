# Variables d'environnement, macros et options de compilation

*État : début octobre 2026. La liste a été relevée dans le code ; une variable qui n'est pas
lue est ignorée sans message, donc en cas de doute, vérifier avec
`grep -rn '"NOM"' chunk fast_allocators examples`. Sauf mention contraire, une variable est
lue une seule fois, et `0` ou une valeur vide équivaut à son absence.*

---

## 1. Réglages de la bibliothèque

| variable | défaut | effet |
|---|---|---|
| `CHUNK_ALLOC_AFFINITY_EAGER` | selon la variante | `1` : chaque gros tableau a son jeu d'allocateurs dès sa création (défaut de Chunk, v1) ; `0` : seulement au premier rechunk (défaut de Chunk2 et Chunk2A). Voir [`rechunk_adaptatif_fr.md`](rechunk_adaptatif_fr.md) §5 |
| `CHUNK_RECHUNK_MAX_COARSEN` | 4 | facteur maximal d'un grossissement (1 = sans limite) |
| `CHUNK_ADVISOR_MIN_GAIN_MB` | 8 | seuil absolu de gain côté données, en Mo (0 le supprime) |
| `CHUNK_COARSEN_MIN_META_MB` | 1 | gain minimal de métadonnées d'un grossissement, en Mo |
| `CHUNK_COARSEN_MIN_META_RATE` | 250 | gain de métadonnées minimal par seconde de reconstruction estimée, en Mo/s |
| `CHUNK_RECHUNK_DRY_RUN` | — | `1` : le décideur analyse et journalise, mais n'applique aucun rechunk |
| `CHUNK_RECHUNK_SCHEDULE` | `guided,16` | ordonnancement OpenMP de la reconstruction (`kind[,chunk]`) |
| `CHUNK_RECHUNK_SLICES` | 16 | nombre de tranches de la reconstruction de v1 |
| `CHUNK_RECLAIM_MIN_SLABS` | 0 | un balayage ne rend les slabs vides au système que s'il y en a au moins autant |
| `CHUNK_THP` | 0 | expérimental : `1` demande des huge pages transparentes (`MADV_HUGEPAGE`) sur les arènes de slabs et la région cst2. Ignoré, avec un message, quand la huge page dépasse 2 Mio (noyau à pages de 64 Kio : 512 Mio). Coût : un slab entamé devient résident en entier |
| `CHUNK_THREAD_CAPACITY` | nombre de cœurs | nombre de rangs pour lequel les structures par thread sont dimensionnées (à augmenter en cas de sursouscription) |
| `CHUNK_DEFERRED_POOL_KB` | 64 | taille des slabs des pools d'écritures différées, en Kio (les slabs d'opérations ne sont jamais rendus : plus gros, moins d'appels à l'allocateur mais plus de mémoire résidente) |
| `CHUNK_REQUIRE_OMPT` | — | builds OMPT : `1` arrête le programme si l'outil OMPT n'a pas été chargé par le runtime (sinon, un message sur stderr), puisque les fins de phase n'auraient alors jamais lieu |

## 2. Interrupteurs de comparaison

Il n'en reste plus : ceux qui permettaient de rétablir un ancien comportement ont été
retirés une fois le nouveau validé. Les réglages utiles pour comparer deux exécutions sont
dans la section 1 et dans [`mesures_performance_fr.md`](mesures_performance_fr.md) §2.

## 3. Traces et diagnostics

| variable | ce qu'elle affiche |
|---|---|
| `CHUNK_RECHUNK_TRACE` | pour chaque rechunk : durée des étapes, repliements et installations directes, mémoire à chaque étape (`[RECHUNK-RSS]`) |
| `CHUNK_RECHUNK_TRACE_SLICES` | mémoire à chaque tranche de la reconstruction de v1 |
| `CHUNK_RECHUNK_MEMTRACE` | à chaque passage de l'advisor : mémoire exacte, dont la part résidente de l'arène des tampons |
| `CHUNK_ADVISOR_TABLE` | `1` ou `=nom` : la table complète de l'advisor à chaque passage (tous les tableaux ou un seul) |
| `CHUNK_RECHUNK_CHURN` | `1` ou `=nom` : matérialisations par passage et estimation du coût passager qu'elles impliquent |
| `CHUNK_AFFINITY_TRACE` | chaque jeu d'allocateurs par tableau créé (clé, taille, budget) |
| `CHUNK_ARENA_REPORT` | le bilan de l'arène des tampons (slabs distribués, rendus au système) |
| `CHUNK_RECYCLE_TRACE` | une ligne par bloc recyclé du pool des écritures différées (au lieu d'un résumé) |
| `CHUNK_LEDGER_REPORT` | le rapport du registre des blocs (builds avec registre) |
| `CHUNK_OMPT_STATS` | statistiques des régions de synchronisation (builds OMPT) |
| `CHUNK_CST2_PROBE` | Chunk2A : rapport du registre des secondes valeurs à chaque passage (banc) |
| `ALLOC_TOKEN_REPORT` | à la sortie, le recensement des allocations par jeton |
| `FAST_ALLOC_TRACE` | une ligne `[MUNMAP]` par slab vidé que la réserve d'un pool ne garde pas et qui est libéré (muet par défaut) |

## 4. Rechunks forcés (tests)

| variable | effet |
|---|---|
| `CHUNK_FORCE_RECHUNK=<taille>` | force un rechunk vers cette taille, même si le décideur ne le ferait pas |
| `CHUNK_FORCE_RECHUNK_AT=<itération>` | retarde ce rechunk forcé jusqu'à cette itération |
| `CHUNK_FORCE_RECHUNK_TOGGLE=1` | rechunk à chaque passage, en alternant avec `CHUNK_FORCE_RECHUNK_ALT` (64 par défaut) |

## 5. Le banc `shrink_bench`

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
| `BENCH_SCHED_BLOCK` | taille des blocs d'ordonnancement des phases du banc (64 ; 0 = tranches contiguës) |
| `NO_IO`, `COMPRESS` | équivalents de `--no_io` et `--compress` |
| `--no_mask`, `--mask_scale <x>` | sans `tab_mask` ; ou sa taille, `x × sz` (défaut 1/32) |
| `--no_struct` | sans les tableaux de structures (`Point3`, `Cell`) |

## 6. Macros de compilation

| macro | effet |
|---|---|
| `CHUNK=1`, `CHUNK2=1`, `CHUNK2A=1` | la variante |
| `CHUNK_DYNAMIC`, `DYN_CHUNK_SIZE=<n>` | géométrie dynamique, et taille de départ |
| `CHUNK_DYN_MAX_SIZE=<n>` | géométrie dynamique : plus grande taille de chunk qu'un rechunk peut atteindre, puissance de deux (4096) ; dimensionne les tampons partagés des valeurs constantes, `cst_buffer_size × sizeof(T)` par seconde valeur distincte |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<octets>` | seuil de métadonnées au-delà duquel un tableau peut avoir ses propres allocateurs (16 Mio ; 0 = jamais) |
| `FAST_ALLOC_CHECK_LISTS` | build Debug : chaque insertion ou retrait dans une liste de slabs vérifie la liste entière (coût linéaire par opération ; défini par l'option CMake du même nom) |
| `CHUNK_NO_RECLAIM_EMPTY_SLABS` | garder résidents les slabs vidés, au lieu de les rendre au système aux balayages explicites (défini par l'option CMake `CHUNK_RECLAIM_EMPTY_SLABS=OFF`) |
| `CHUNK_STATS` | compteurs d'instrumentation (build stats) |
| `CHECK_OPTIM` | vérifications d'invariants en build optimisé |
| `MM_PROTECT` | Chunk, Chunk2 : tampons partagés des valeurs constantes en lecture seule (`mprotect`) ; une écriture parasite dans un chunk compressé fait une erreur de segmentation au lieu de corrompre la valeur (option CMake `CHUNK_MM_PROTECT`) |
| `CHUNK_FORCE_DEFER_EVERY=<N>` | une écriture sur N passe par le chemin différé, dans tout l'arbre (validation ; 32 threads au plus) |
| `HAVE_AGGRESSIVE_COMPRESSION` | compression « agressive » : la passe entre deux phases est sautée, et toute la compression se fait pendant les régions parallèles, à des instants imprévisibles. Plus lent, mais sollicite les chemins concurrents (option CMake du même nom ; l'ancienne orthographe `HAVE_AGRESSIVE_COMPRESSION` est encore acceptée par CMake, avec un avertissement) |
| `CHUNK_FORCE_LEDGER` | garde le registre des blocs (`pool_ledger.h`) actif sous OpenMP, où il est éteint par défaut : son mutex global sérialise chaque allocation et libération. Détecte double libération et remise illégale en réserve (option CMake du même nom) |
| `CHUNK_COMPRESS_MAX_YIELDS=<n>` | attente d'un compresseur garé avant de renoncer à sa passe (10 000 *yields* ; 0 : renoncer au premier écrivain vu). Voir [`synchronisation_dekker_fr.md`](synchronisation_dekker_fr.md) |
| `CHUNK_WRITE_WAIT_MAX_YIELDS=<n>` | attente d'un écrivain devant un compresseur avant de différer son écriture (64 *yields*) |
| `CHUNK_DEFERRED_DRAIN_THRESHOLD=<n>` | nombre d'écritures différées d'un rang au-delà duquel il vide sa propre file (4 096 ; option CMake du même nom) |
| `CHUNK_READ_RECLAIM_SLOTS=<n>` | nombre de threads lecteurs suivis individuellement par la récupération des tampons (1 024) |
| `CHUNK_STATS_MAX_THR=<n>` | nombre de rangs des compteurs `CHUNK_STATS` (1 024) |
| `CHUNK_STD_ALLOC` | tampons de chunks alloués par l'allocateur standard au lieu de `fast_allocators` (option CMake du même nom) |
| `CHUNK_TEST_READ_DELAY_EVERY=<n>`, `CHUNK_TEST_READ_DELAY_US=<µs>` | crochet de test : une lecture sur N d'un thread qui l'a demandé attend (50 µs), pour élargir la fenêtre lecteur / récupération (`read_reclaim_validation`) |
| `CHUNK_TEST_PASS_DELAY_EVERY=<n>`, `CHUNK_TEST_PASS_DELAY_US=<µs>` | crochet de test : délai (200 µs) dans une passe de compression sur N |
| `CHUNK_TEST_HOOKS` | compile les crochets de test des harnesses (`chunk/internal/test_hooks.h` : écritures différées forcées, délais de lecture et de passe), éteints à l'exécution tant qu'un harness ne les allume pas ; défini pour tout l'arbre là où les harnesses sont compilés (`CHUNK_BUILD_VALIDATION`), jamais dans les builds de performance |

## 7. Options CMake

| option | défaut | effet |
|---|---|---|
| `OMP` | `OFF` | construire avec OpenMP |
| `CHUNK_STATIC_SIZES` | `256;1024` | tailles statiques construites (cibles `_<taille>`) |
| `SHRINK_BENCH_DYN_SIZE` | 64 | taille de départ des cibles `_dyn` |
| `SHRINK_BENCH_INT64` | `OFF` | cibles à indice 64 bits (`_int64`) |
| `SHRINK_INTEL_LIBM` | `ON` | x86-64 : lier la libm d'Intel si elle est trouvée |
| `SHRINK_ARM_LIBM` | `OFF` | aarch64 : lier `libamath` d'ArmPL (aucun gain mesuré sur Grace) |
| `CHUNK_ICPX_O0_WORKAROUND` | `OFF` | contournement d'une erreur interne d'icpx sur trois fichiers |
| `CHUNK_MM_PROTECT` | `OFF` | définit `MM_PROTECT` (build de mise au point ; `compile_all.sh` le construit pour que ses branches continuent de compiler) |
| `CHUNK_LIB_UNITY` | `ON` | chaque archive de variante (`chunk_v2a_256`, ...) compilée en une seule unité (unity build, 4 à 6 fois moins de compilation) ; un fichier qui a ses propres options de compilation reste à part |
| `SHRINK_BENCH_UNITY` | `ON` | chaque exécutable `shrink_bench` compilé en une seule unité (unity build, 3,5 à 4 fois moins de compilation) ; `OFF` pour comparer des temps avec des campagnes antérieures, l'inlining entre fichiers du banc pouvant les changer ; `SHRINK_BENCH_UNITY_BATCH` fichiers par unité (0 = tous) |
| `CHUNK_VALIDATION_UNITY` | `ON` | les harnesses de validation regroupés en un exécutable par variante et géométrie (unity build) ; `OFF` : un exécutable par harness |
| `CHUNK_BUILD_VALIDATION` | selon le build | compiler les harnesses de `examples/validation`, et avec eux les crochets de test (`CHUNK_TEST_HOOKS`) ; par défaut `ON` en Debug, avec `CHECK_OPTIM` ou un sanitizer, `OFF` ailleurs (non mis en cache : suit la configuration courante) |
| `CHUNK_RECLAIM_EMPTY_SLABS` | `ON` | rendre au système (`madvise`) les slabs vidés aux balayages explicites (rechunk, destruction) ; `OFF` définit `CHUNK_NO_RECLAIM_EMPTY_SLABS` |
| `FAST_ALLOC_CHECK_LISTS` | `OFF` | niveau de debug renforcé des listes de slabs : en Debug, vérification complète de la liste à chaque insertion et retrait (lent) ; un Debug ordinaire garde les vérifications légères |
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | Chunk2A : l'emplacement de seconde valeur contient un identifiant de 4 octets au lieu de la valeur (moins de mémoire, +3 % de temps d'écriture) |
| `CHUNK_STD_ALLOC` | `OFF` | allocateur standard pour les tampons de chunks au lieu de `fast_allocators` (utile sous ASan : chaque tampon est un vrai `malloc`) |
| `CHUNK_BUILD_DYN` | `ON` | construire aussi une bibliothèque à géométrie dynamique (`_dyn`) par variante |
| `CHUNK_STATS` | `OFF` | compteurs d'instrumentation et rapport `print_chunk_stats()` |
| `CHECK_OPTIM` | `OFF` | vérifications d'invariants en build optimisé |
| `HAVE_AGGRESSIVE_COMPRESSION` | `OFF` | compression agressive (voir §6) |
| `CHUNK_FORCE_LEDGER` | `OFF` | registre des blocs actif sous OpenMP (voir §6) |
| `CHUNK_DEFERRED_DRAIN_THRESHOLD` | vide (4 096) | seuil de vidage de sa propre file par un rang |
| `CHUNK_FORCE_DEFER_EVERY` | 0 | une écriture sur N par le chemin différé (validation) |
| `SANITIZER` | `none` | `thread`, `address` ou `token` (AllocToken, Clang ≥ 22), pour tous les types de build |
| `CHUNK_GLIBCXX_ASSERTIONS` | `OFF` | assertions de la libstdc++ (`_GLIBCXX_ASSERTIONS`) dans la bibliothèque et ses consommateurs |
| `CHUNK_UBSAN` | `OFF` | UndefinedBehaviorSanitizer (à associer à ASan, pas à TSan) |
| `CHUNK_ALIGN_CHECK` | `OFF` | arrêt si un bloc de slab est mal aligné pour son type |
| `CHUNK_NO_IPA_ICF` | `OFF` | désactive `-fipa-icf` (GCC fusionne sinon des fonctions identiques, ce qui brouille les piles d'appels) |
| `CHUNK_WARNINGS`, `CHUNK_WERROR` | `ON`, `OFF` | `-Wall -Wextra` sur tous les builds ; avertissements en erreurs en Debug |
| `CHUNK_HEAVY_JOB_POOL` | vide | pool de jobs Ninja pour les exécutables compilés en une unité (limite la mémoire de compilation) |
| `USE_LIBCXX` | `ON` | builds Clang : libc++ au lieu de la libstdc++ |
| `USE_GPERF`, `ENABLE_IWYU` | `OFF` | profilage gperftools ; include-what-you-use pendant la compilation |
| `ONEAPI_XHOST`, `ONEAPI_NO_INTEL_LIBS`, `ONEAPI_FAST_FP` | `OFF` | icpx : binaire réservé à Intel (`-xHost`) ; aucune bibliothèque d'exécution Intel ; garder `-fp-model=fast` |

Et une macro propre à Chunk2A : `CHUNK2A_COMPRESS_LEVEL=<0|1|2|3>` (0 par défaut) choisit le
jeu de seuils de déclenchement des passes de compression (`chunk2A_def.h`).

Pour les builds clang sur x86-64, `CHUNK_LIBMVEC_LINKS` (dans le cache, calculée à la
configuration) indique si la bibliothèque vectorielle de la glibc fournit les fonctions
mathématiques des bancs ; sinon, leurs boucles restent scalaires (glibc < 2.35).

## 8. Les scripts de compilation

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
| `--target=<glob>` | seulement les configurations dont le nom correspond (`omp-llvm`, `'*sanitizer*'`…) ; un motif qui ne correspond à rien est une erreur |
| `--exec=<glob>` | seulement les cibles Ninja correspondantes dans chaque build (`shrink_bench_chunk_64`, `'*_dyn'`…) |
| `--jobs=<n>` | nombre de configurations compilées en même temps (4 par défaut) ; le budget de cœurs est partagé entre elles |
| `--no-hardening` | sans `CHUNK_GLIBCXX_ASSERTIONS` ni `CHUNK_UBSAN` sur les builds Debug et sanitizer |
| `--llvm-cs=<n>` | ajouter un build LLVM avec `CHUNK_SIZE=n` |

`compile_all_gnu.sh` accepte en plus `--no-ipa-icf` (`CHUNK_NO_IPA_ICF=ON`). La mémoire de
compilation par job se règle par `CHUNK_MEM_PER_JOB_MB` (3 072 Mo par défaut, le double pour
les builds sous sanitizer).
