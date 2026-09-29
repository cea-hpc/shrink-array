# Variables d'environnement, macros et options de compilation

*État : fin septembre 2026. La liste a été relevée dans le code ; une variable qui n'est pas
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
| `CHUNK_THREAD_CAPACITY` | nombre de cœurs | nombre de rangs pour lequel les structures par thread sont dimensionnées (à augmenter en cas de sursouscription) |

## 2. Interrupteurs de comparaison

Chacun vaut `1` pour activer le comportement indiqué, afin de comparer sur un même binaire
(voir [`mesures_performance_fr.md`](mesures_performance_fr.md) §2).

| variable | comportement activé |
|---|---|
| `CHUNK_PROXY2_ACQ_FENCE` | rétablit les barrières d'acquisition après la lecture du registre (Chunk2) |
| `CHUNK_PROXY2A_META_ARMED` | Chunk2A : les proxys utilisent les adresses de lignes de métadonnées armées (désactivé par défaut) |

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
| `BENCH_SCHED_BLOCK` | taille des blocs d'ordonnancement des phases du banc (64 ; 0 = tranches contiguës) |
| `NO_IO`, `COMPRESS` | équivalents de `--no_io` et `--compress` |
| `--no_mask`, `--mask_scale <x>` | sans `tab_mask` ; ou sa taille, `x × sz` (défaut 1/32) |
| `--no_struct` | sans les tableaux de structures (`Point3`, `Cell`) |

## 6. Macros de compilation

| macro | effet |
|---|---|
| `CHUNK=1`, `CHUNK2=1`, `CHUNK2A=1` | la variante |
| `CHUNK_DYNAMIC`, `DYN_CHUNK_SIZE=<n>` | géométrie dynamique, et taille de départ |
| `CHUNK_ALLOC_AFFINITY_MIN_BYTES=<octets>` | seuil de métadonnées au-delà duquel un tableau peut avoir ses propres allocateurs (16 Mio ; 0 = jamais) |
| `CHUNK_RECLAIM_EMPTY_SLABS` | rendre au système les slabs vidés (balayages explicites) |
| `CHUNK_STATS` | compteurs d'instrumentation (build stats) |
| `CHECK_OPTIM` | vérifications d'invariants en build optimisé |
| `CHUNK_FORCE_DEFER_EVERY=<N>` | une écriture sur N passe par le chemin différé (validation ; 32 threads au plus) |

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
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | Chunk2A : l'emplacement de seconde valeur contient un identifiant de 4 octets au lieu de la valeur (moins de mémoire, +3 % de temps d'écriture) |

Et une macro propre à Chunk2A : `CHUNK2A_COMPRESS_LEVEL=<0|1|2|3>` (0 par défaut) choisit le
jeu de seuils de déclenchement des passes de compression (`chunk2A_def.h`).

Pour les builds clang sur x86-64, `CHUNK_LIBMVEC_LINKS` (dans le cache, calculée à la
configuration) indique si la bibliothèque vectorielle de la glibc fournit les fonctions
mathématiques des bancs ; sinon, leurs boucles restent scalaires (glibc < 2.35).
