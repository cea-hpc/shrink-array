# Géométrie dynamique et rechunk adaptatif

> English version: [adaptive_rechunk.md](adaptive_rechunk.md).

> État : fin septembre 2026, pour les trois variantes (`zero`, `zero_one_value`, `zero_any`). Les chiffres
> viennent de `shrink_bench` sur Grace (64 threads, `OMP_PROC_BIND=close`,
> `OMP_PLACES=cores`), avec la mémoire mesurée exactement et les temps comparés sur un même
> binaire : voir [mesures_performance_fr.md](mesures_performance_fr.md). En cas de doute,
> c'est le code qui fait foi.

***

## 1. Le principe

En **géométrie statique** (`-DCHUNK_SIZE=…`, ou les cibles `_64`, `_256`, `_1024` du banc),
la taille des chunks est une constante de compilation. En **géométrie dynamique**
(`-DCHUNK_DYNAMIC`, cibles `_dyn`), chaque tableau a sa propre taille, qu'il peut changer en
cours de run par un **rechunk** : la reconstruction complète de son stockage à une autre
taille.

L'intérêt : partir de **petits chunks** (64 éléments dans le banc, `SHRINK_BENCH_DYN_SIZE`),
qui compressent finement, puis **grossir** quand les métadonnées l'emportent sur ce que la
finesse fait gagner. Les objectifs, atteints pour `zero_one_value` et `zero_any` :

- **au départ**, consommer comme le statique à la même taille ;
- **à la fin**, consommer moins que le meilleur statique.

Trois acteurs : l'**advisor** estime la mémoire du tableau à chaque taille, le **décideur**
choisit s'il faut changer et vers quelle taille, et `rechunk_to()` fait la reconstruction.

## 2. L'advisor : estimer la mémoire à chaque taille

<tt>ChunkSizeAdvisor\<T\>\::analyze(arr)</tt> (`chunk_size_advisor.h`) produit une table : pour chaque
taille candidate, le nombre de chunks, le nombre de chunks matérialisés, et la mémoire
estimée (métadonnées + données matérialisées).

Les candidates : les rapports en puissances de deux (S×2 à S×16, S÷2 et S÷4), et entre ces
bornes tout multiple de 64 dont C / 64 a au plus `Options::grid_bits` bits significatifs
(`RechunkPolicy::advisor_grid_bits`, `--rechunk_grid_bits` dans le banc ; 2 par défaut : 192, 384, 768, 1536... : des pas de 50 % au plus ;
3 ajoute 320, 448, 640, 896... : des pas de 25 %). Chaque ligne découpe le tableau en fenêtres de C éléments, lues sur les chunks actuels :

- une fenêtre **dans une suite de chunks compressés à la même constante** est compressée :
  comptée exactement, à partir des seuls états des chunks ;
- une fenêtre qui **couvre un chunk matérialisé entier** est matérialisée ;
- une fenêtre qui **chevauche en partie un chunk matérialisé** (taille plus fine ou mal
  alignée) est tranchée en lisant les données, sur un **échantillon** des chunks matérialisés
  (65 536 au plus) : elle est compressée si elle ne contient qu'une valeur, et une valeur
  représentable. Pour `zero` et `zero_one_value`, c'est la première ou la seconde valeur ; pour
  **`zero_any`**, toute valeur, puisque chaque chunk enregistre sa propre seconde valeur.

Quand S divise C (S×2, S×3, S×4...), le dernier cas ne se présente jamais : la ligne est un
**repliement exact** de l'état réel des chunks. `advisor_demo` vérifie chaque ligne contre des
tableaux construits à ces tailles, depuis S = 256 et S = 1024.

Pourquoi 2 et pas 3 (shrink_bench zero_any_dyn, `--divisor 4`, 16 threads, mémoire en fin de
run) : 1 bit (puissances de deux seules) 2320 Mo, 2 bits 2317 Mo, 3 bits 2396 Mo. Les
prédictions sont aussi justes à 3 bits, mais la taille la moins chère est alors souvent un
petit pas (256 -> 384) dont le gain en métadonnées est refusé par le seuil de débit du
décideur (`coarsen_min_meta_rate`) ; le décideur ne se rabat pas sur la taille suivante, et deux
tableaux sont restés à 256.

La classification des chunks se fait à la volée, sans tableau de travail : jusqu'en
septembre 2026, chaque analyse allouait un octet et un pointeur par chunk (140 Mo pour un
tableau de 15,6 millions de chunks), et les analyses coûtaient 1 à 2 s sur un run de 6,5 s.
Elles coûtent maintenant ~0,7 s sur un run de `zero_one_value` de 8,4 s.

`CHUNK_ADVISOR_TABLE=1` (ou `=nom_du_tableau`) affiche la table complète à chaque passage.

## 3. Le décideur : quand et vers où

`RechunkDecider::consider()` (`rechunk_decider.h`, `internal/rechunk_decider_inline.h`) est
appelé à chaque **passage** : toutes les `--advisor_period` itérations dans le banc
(par défaut `max(10, iter_max / 20)`, soit 100 pour 2 000 itérations ; 0 désactive). Il
prend la meilleure taille de la table, puis applique la politique (`RechunkPolicy`). La
bibliothèque ne lit aucun de ces réglages dans l'environnement : un programme renseigne les
champs ; le banc les expose en options (entre parenthèses) :

| réglage | défaut | rôle |
|---|---|---|
| `stability_ticks` | 3 | la meilleure taille doit rester la même pendant 3 passages |
| `cooldown_ticks` | 5 | au moins 5 passages entre deux rechunks d'un même tableau |
| `anti_reversal_ticks` | 30 | pas de retour vers une taille quittée depuis moins de 30 passages |
| `max_coarsen_factor` | **4** | un grossissement multiplie la taille par 4 au plus (`--rechunk_max_coarsen`) |
| `min_gain_bytes`, `min_gain_frac` | 8 Mo, 15 % | seuils de gain côté données (`--rechunk_min_gain_mb`) |
| `coarsen_gain_multiple` | 3 | un grossissement jugé sur les données doit gagner 3 fois le seuil |
| `coarsen_min_meta_gain_bytes` | 1 Mo | gain minimal côté métadonnées (`--rechunk_coarsen_min_meta_mb`) |
| `coarsen_min_meta_rate` | 250 Mo/s | gain de métadonnées par seconde de reconstruction estimée (`--rechunk_coarsen_min_meta_rate`) |
| `coarsen_trust_ratio` | 0,5 | apprentissage : un grossissement qui a rendu moins de la moitié de sa prévision interdit les suivants |
| `dry_run` | `true` | ne rien appliquer (le banc le met à `false` ; `--rechunk_dry_run` le garde) |
| `max_backoff_ticks` | 8 | un tableau dont le verdict est un « pas de mouvement » établi (déjà optimal, gain sous le seuil) n'est réanalysé qu'après 2, 4, puis jusqu'à 8 passages ; tout autre verdict le ramène à chaque passage (`--rechunk_backoff_max`, 0 ou 1 désactive) |
| `advisor_grid_bits` | 2 | les tailles candidates de l'advisor entre les puissances de deux, voir §2 (`--rechunk_grid_bits`) |
| `force_size`, `force_at`, `force_toggle`, `force_alt` | 0, 0, non, 64 | tests : rechunk de chaque tableau vers `force_size` à partir de l'itération `force_at`, quoi que dise l'advisor ; `force_toggle` alterne avec `force_alt` à chaque passage (`--force_rechunk`, `--force_rechunk_at`, `--force_rechunk_toggle`, `--force_rechunk_alt`) |

Un grossissement est accepté si **l'une** des deux branches passe : la branche
**métadonnées** (gain de métadonnées au-dessus de son plancher, et débit suffisant au regard
du coût estimé par `rechunk_cost_ms()`), ou la branche **données** (gain total au-dessus du
seuil multiplié). Depuis ce critère « gain par coût », `min_gain_bytes` ne suffit
plus à bloquer les grossissements : pour mesurer le coût des seules analyses, utiliser
`dry_run` (`--rechunk_dry_run`).

L'apprentissage mesure le gain d'un grossissement **à données constantes** : au passage
suivant, la ligne de l'ancienne taille dans la table (lignes plus fines) donne la mémoire
qu'aurait eue le tableau sans le rechunk, sur ses données actuelles. Comparer simplement
avant et après mélangeait le rechunk avec l'évolution des données pendant une période.

**Pourquoi un facteur 4.** Mesuré sur `zero_any` avec tous les tableaux confiés au décideur :

| facteur | rechunks | mémoire de fin | gain à données constantes / prévu |
|---|---|---|---|
| 2 | 44 | 6 729-6 737 Mo | 3 655 / 3 693 Mo (99 %) |
| **4** | **33** | **6 539-6 542 Mo** | 3 885 / 3 934 Mo (98,8 %) |
| 8 | 20 | 6 721-6 724 Mo | (non mesurable : l'ancienne taille sort de la table) |

La limite à 2 datait de deux mauvais grands sauts mesurés avant les corrections de
septembre 2026 (mesure `VmRSS` faussée, fuite des lignes de `zero_any`, repliements par copie).
À 4, l'ancienne taille reste dans les deux niveaux plus fins de la table, ce qui garde la
mesure à données constantes et la possibilité de revenir en arrière.

## 4. La reconstruction : `rechunk_to()`

`internal/array_chunk_rechunk_inline.h`. Les étapes, avec leur nom dans la trace
`CHUNK_LOG=rechunk=debug` :

1. **prologue** : à l'ancienne géométrie, vider les écritures différées du tableau, puis
   compresser ses candidats en attente avant de les retirer des listes ;
2. **tmp create** : construire le nouveau stockage, à la nouvelle taille ;
3. **refill** : le remplir en parallèle (`CHUNK_RECHUNK_SCHEDULE`, `guided,16` par défaut).
   Un nouveau chunk dont toutes les sources sont à la première valeur est sauté. Un nouveau
   chunk dont toutes les sources étaient compressées est **replié** tout de suite. Pour
   **`zero_any`**, si toutes ses sources sont à la **même** seconde constante, elle est
   **installée directement**, sans tampon ni copie (la copie suivie d'un repliement laissait
   jusqu'à 640 Mo de pages touchées par tableau) ;
4. **flush+swap** : échanger les identités entre l'ancien et le nouveau stockage ;
5. **tmp teardown** : détruire l'ancien stockage ;
6. **old-pool drain** : vider les réserves de l'ancienne taille et rendre au système les
   slabs devenus vides.

**`zero`** reconstruit **par tranches** (16 par défaut, `CHUNK_RECHUNK_SLICES`) et libère
chaque ancien chunk dès qu'il est recopié. Entre deux tranches, les réserves de l'ancienne
taille sont vidées et les slabs vidés rendus au système, y compris dans la **réserve
partagée** quand l'ancien stockage s'y trouve : sans ça, la mémoire de l'ancien stockage
restait acquise jusqu'à la fin de la reconstruction (sur `rechunk_validation`, pic de la
reconstruction 78 → 22 Mo). Mesuré sur `zero` à 64 threads : pic de +6,9 Go en une passe,
+1,7 Go avec 16 tranches.

## 5. Les allocateurs par tableau (affinité)

Un tableau dont les métadonnées dépassent 16 Mio (`CHUNK_ALLOC_AFFINITY_MIN_BYTES`) peut
avoir son **propre jeu d'allocateurs** (une clé d'affinité), par rang et par taille. Ses
tampons ne partagent alors aucun slab avec un autre tableau : quand il est reconstruit ou
détruit, ses slabs se vident entièrement et sont rendus au système.

Le prix : un slab entamé par couple (rang, tableau, taille). Avec les slabs de 1 Mio de l'ancien
plancher de seize pages (pages de 64 Kio), c'était ~780 Mio sur les 16 tableaux du banc à 64 threads ;
les slabs y font maintenant 256 Kio, le quart (`CHUNK_SLAB_MIN_PAGES=16` rétablit le plancher, voir
[reglages_developpeur_fr.md](reglages_developpeur_fr.md)).

D'où deux modes, dont le choix par défaut dépend de la variante :

| mode | quand un tableau reçoit sa clé | défaut pour | pourquoi |
|---|---|---|---|
| **paresseux** | à son premier rechunk | `zero_one_value`, `zero_any` | ils sont surtout faits de métadonnées : un départ sans jeux par tableau consomme exactement comme le statique (`zero_any` : 11 590 Mo après l'initialisation, contre 12 435 Mo en mode immédiat) |
| **immédiat** | à sa création | `zero` | `zero` est surtout fait de données, et sa reconstruction par tranches ne rend la mémoire au fil de l'eau que si l'ancien stockage a ses propres slabs. En mode paresseux, le premier rechunk part de slabs partagés, qui mêlent les blocs de plusieurs tableaux et ne se vident pas : +6,0 Go sur 39,3 Go. Pic de `zero` : 41,2 Go en immédiat, 45,6 Go en paresseux, pour la même mémoire de fin |

`CHUNK_ALLOC_AFFINITY_EAGER=0` ou `=1` impose l'un des modes.

## 6. Ce qui doit disparaître avec l'ancien stockage

Un rechunk construit le nouveau stockage sous un **nouvel identifiant de tableau**
(`num_tab`) et détruit l'ancien. Tout ce qui était indexé par l'ancien identifiant doit être
**libéré**, pas seulement vidé. Deux fuites de ce type ont été corrigées en septembre 2026 :

- **`zero_any`** : les lignes de métadonnées par chunk (type de constante, 1 octet ; seconde
  valeur, `sizeof(T)` octets par défaut) étaient vidées par `clear()`, qui garde la capacité. Sur le
  banc avec les 15 tableaux rechunkés : 10 276 → 6 730 Mo en fin de run ;
- **les trois variantes** : l'indicateur « en liste de compression » (1 octet par chunk)
  n'était jamais libéré. `zero_any` −0,52 Go, `zero` −0,75 Go.

Par ailleurs, le **seuil de déclenchement des passes de compression** et le **budget des
réserves**, calculés à partir du nombre total de chunks, sont maintenant recalculés aussi à la
destruction d'un stockage, et plus seulement à sa création. Sans effet mesurable sur le temps, mais les seuils
correspondent de nouveau à la géométrie réelle.

## 7. Résultats

Mémoire (Go) sur `shrink_bench`, 64 threads, tous les tableaux confiés au décideur (le
défaut du banc ; `--advise_five` revient aux cinq tableaux historiques) :

| variante | statique 64 (fin) | dynamique sans advisor (fin) | **dynamique avec advisor (fin)** | statique 256 (fin) |
|---|---|---|---|---|
| `zero_any` | 12,7 | 12,7 | **6,0** | 7,3 |
| `zero_one_value` | 21,4 | 21,4 | **16,3** | 17,3 |
| `zero` | 43,6 | 44,6 | **40,3** | 40,2 |

Coût en temps de l'advisor, mesuré sur **un même binaire** (moyennes sur 3 tours) :

| `zero_one_value` dynamique | temps | coût |
|---|---|---|
| sans advisor (`--advisor_period 0`) | 8,38 s | — |
| analyses seules (`--rechunk_dry_run`) | 9,07 s | +0,69 s |
| analyses et rechunks (30) | 11,43 s | +2,36 s |

Pour `zero_any`, le coût des rechunks (33) va de 1 à 3,5 s selon les séries de mesures. Pour
`zero_one_value`, les intervalles qui suivent les rechunks sont aussi rapides que ceux d'avant : le coût
se limite aux reconstructions.

## 8. Limites connues

- **`zero` ralentit après son rechunk.** Après le passage à 1 024, `zero` dynamique tourne 1,5 à
  2 fois moins vite par itération que le statique 1 024 (environ 40 cœurs occupés à un IPC de
  3,65, contre 52 cœurs à 4,24). Un build dynamique **démarré** à 1 024 va aussi vite que le
  statique, et `zero_one_value` ne ralentit pas après ses rechunks : c'est l'état laissé par la
  reconstruction de `zero` qui est en cause. Ni l'ordonnancement de la reconstruction, ni les
  seuils de compression n'expliquent l'écart. Cause encore inconnue.
  Remesuré en octobre 2026 (inti, aarch64, LLVM, 64 threads, `--mode perf`,
  2 000 itérations, `diag_zero_rechunk_slowdown.py`) : +17 % par itération après le
  passage à 1 024. Inchangé avec une seule tranche, une reconstruction en
  ordonnancement statique, sans recyclage des slabs, sans réutilisation de
  `chunk_list`, en affinité lazy ; le placement NUMA est équilibré comme celui du
  statique. L'essentiel venait de l'advisor : ses analyses continuaient à chaque
  passage une fois la géométrie stabilisée, et leurs parties séquentielles
  faisaient attendre les autres threads (__kmp_fork_barrier). Avec
  `max_backoff_ticks` (même nœud, `OMP_PROC_BIND=close`, 4 000 itérations, dernier
  rechunk à l'itération 1 700) : +22 % -> **+7 %**. Le reste est un va-et-vient de
  compression que le statique n'a pas (~15 000 buffers libérés par itération contre
  aucun), encore inexpliqué.
- **Le modèle est statique.** L'advisor juge sur l'état présent des chunks, pas sur les
  écritures à venir. Sur ce banc, les matérialisations passagères mesurées ne pèsent pas
  (`CHUNK_RECHUNK_CHURN=1`), mais une charge qui écrit beaucoup dans des zones constantes
  pourrait faire regretter un grossissement : c'est le rôle de l'apprentissage
  (`coarsen_trust_ratio`) et de l'anti-retour.
- **Le rechunk des tableaux de booléens** n'est validé que pour `ArrayChunkBits`
  (`bits_container_validation`, `zero_one_value` en dyn) ; `zero_one_value::Array<bool>` et `zero_any::Array<bool>`
  n'ont pas de harnais, et le banc ne confie pas `tab_bool` au décideur.
- **Le pic reste celui du départ** : un run dynamique qui part de 64 atteint d'abord la
  mémoire du statique 64, puis descend.
