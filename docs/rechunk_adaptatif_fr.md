# Géométrie dynamique et rechunk adaptatif

*État : fin septembre 2026, pour les trois variantes (Chunk, Chunk2, Chunk2A). Les chiffres
viennent de `shrink_bench` sur Grace (64 threads, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`), avec la mémoire mesurée exactement et les temps comparés sur un même
binaire : voir [`mesures_performance_fr.md`](mesures_performance_fr.md). En cas de doute,
c'est le code qui fait foi.*

---

## 1. Le principe

En **géométrie statique** (`-DCHUNK_SIZE=…`, ou les cibles `_64`, `_256`, `_1024` du banc),
la taille des chunks est une constante de compilation. En **géométrie dynamique**
(`-DCHUNK_DYNAMIC`, cibles `_dyn`), chaque tableau a sa propre taille, qu'il peut changer en
cours de run par un **rechunk** : la reconstruction complète de son stockage à une autre
taille.

L'intérêt : partir de **petits chunks** (64 éléments dans le banc, `SHRINK_BENCH_DYN_SIZE`),
qui compressent finement, puis **grossir** quand les métadonnées l'emportent sur ce que la
finesse fait gagner. Les objectifs, atteints pour Chunk2 et Chunk2A :

- **au départ**, consommer comme le statique à la même taille ;
- **à la fin**, consommer moins que le meilleur statique.

Trois acteurs : l'**advisor** estime la mémoire du tableau à chaque taille, le **décideur**
choisit s'il faut changer et vers quelle taille, et `rechunk_to()` fait la reconstruction.

## 2. L'advisor : estimer la mémoire à chaque taille

<tt>ChunkSizeAdvisor\<T\>\::analyze(arr)</tt> (`chunk_size_advisor.h`) produit une table : pour chaque
taille candidate, le nombre de chunks, le nombre de chunks matérialisés, et la mémoire
estimée (métadonnées + données matérialisées).

- **Tailles plus grosses** (4 niveaux, jusqu'à ×16) : un **repliement exact** de l'état réel
  des chunks. Un chunk de taille 2S est compressible si ses deux sous-chunks sont compressés
  à la même constante ; sinon il est matérialisé.
- **Tailles plus fines** (2 niveaux, jusqu'à ÷4) : un **échantillonnage** des données des
  chunks matérialisés (65 536 au plus), qui compte les sous-blocs uniformes. Pour Chunk et
  Chunk2, un sous-bloc n'est compté compressible que s'il est à une valeur représentable
  (première ou seconde valeur). Pour **Chunk2A**, toute valeur uniforme l'est, puisque chaque
  chunk enregistre sa propre seconde valeur.

La classification des chunks se fait à la volée, sans tableau de travail : jusqu'en
septembre 2026, chaque analyse allouait un octet et un pointeur par chunk (140 Mo pour un
tableau de 15,6 millions de chunks), et les analyses coûtaient 1 à 2 s sur un run de 6,5 s.
Elles coûtent maintenant ~0,7 s sur un run de Chunk2 de 8,4 s.

`CHUNK_ADVISOR_TABLE=1` (ou `=nom_du_tableau`) affiche la table complète à chaque passage.

## 3. Le décideur : quand et vers où

`RechunkDecider::consider()` (`rechunk_decider.h`, `internal/rechunk_decider_inline.h`) est
appelé à chaque **passage** : toutes les `CHUNK_ADVISOR_PERIOD` itérations dans le banc
(par défaut `max(10, iter_max / 20)`, soit 100 pour 2 000 itérations ; 0 désactive). Il
prend la meilleure taille de la table, puis applique la politique (`RechunkPolicy`) :

| réglage | défaut | rôle |
|---|---|---|
| `stability_ticks` | 3 | la meilleure taille doit rester la même pendant 3 passages |
| `cooldown_ticks` | 5 | au moins 5 passages entre deux rechunks d'un même tableau |
| `anti_reversal_ticks` | 30 | pas de retour vers une taille quittée depuis moins de 30 passages |
| `max_coarsen_factor` | **4** | un grossissement multiplie la taille par 4 au plus (`CHUNK_RECHUNK_MAX_COARSEN`) |
| `min_gain_bytes`, `min_gain_frac` | 8 Mo, 15 % | seuils de gain côté données (`CHUNK_ADVISOR_MIN_GAIN_MB`) |
| `coarsen_gain_multiple` | 3 | un grossissement jugé sur les données doit gagner 3 fois le seuil |
| `coarsen_min_meta_gain_bytes` | 1 Mo | gain minimal côté métadonnées (`CHUNK_COARSEN_MIN_META_MB`) |
| `coarsen_min_meta_rate` | 250 Mo/s | gain de métadonnées par seconde de reconstruction estimée (`CHUNK_COARSEN_MIN_META_RATE`) |
| `coarsen_trust_ratio` | 0,5 | apprentissage : un grossissement qui a rendu moins de la moitié de sa prévision interdit les suivants |
| `dry_run` | `true` | ne rien appliquer (le banc le met à `false` ; `CHUNK_RECHUNK_DRY_RUN=1` le force) |

Un grossissement est accepté si **l'une** des deux branches passe : la branche
**métadonnées** (gain de métadonnées au-dessus de son plancher, et débit suffisant au regard
du coût estimé par `rechunk_cost_ms()`), ou la branche **données** (gain total au-dessus du
seuil multiplié). Depuis ce critère « gain par coût », `CHUNK_ADVISOR_MIN_GAIN_MB` ne suffit
plus à bloquer les grossissements : pour mesurer le coût des seules analyses, utiliser
`CHUNK_RECHUNK_DRY_RUN=1`.

L'apprentissage mesure le gain d'un grossissement **à données constantes** : au passage
suivant, la ligne de l'ancienne taille dans la table (lignes plus fines) donne la mémoire
qu'aurait eue le tableau sans le rechunk, sur ses données actuelles. Comparer simplement
avant et après mélangeait le rechunk avec l'évolution des données pendant une période.

**Pourquoi un facteur 4.** Mesuré sur Chunk2A avec tous les tableaux confiés au décideur :

| facteur | rechunks | mémoire de fin | gain à données constantes / prévu |
|---|---|---|---|
| 2 | 44 | 6 729-6 737 Mo | 3 655 / 3 693 Mo (99 %) |
| **4** | **33** | **6 539-6 542 Mo** | 3 885 / 3 934 Mo (98,8 %) |
| 8 | 20 | 6 721-6 724 Mo | (non mesurable : l'ancienne taille sort de la table) |

La limite à 2 datait de deux mauvais grands sauts mesurés avant les corrections de
septembre 2026 (mesure `VmRSS` faussée, fuite des lignes de Chunk2A, repliements par copie).
À 4, l'ancienne taille reste dans les deux niveaux plus fins de la table, ce qui garde la
mesure à données constantes et la possibilité de revenir en arrière.

## 4. La reconstruction : `rechunk_to()`

`internal/array_chunk_rechunk_inline.h`. Les étapes, avec leur nom dans la trace
`CHUNK_RECHUNK_TRACE=1` :

1. **prologue** : à l'ancienne géométrie, vider les écritures différées du tableau, puis
   compresser ses candidats en attente avant de les retirer des listes ;
2. **tmp create** : construire le nouveau stockage, à la nouvelle taille ;
3. **refill** : le remplir en parallèle (`CHUNK_RECHUNK_SCHEDULE`, `guided,16` par défaut).
   Un nouveau chunk dont toutes les sources sont à la première valeur est sauté. Un nouveau
   chunk dont toutes les sources étaient compressées est **replié** tout de suite. Pour
   **Chunk2A**, si toutes ses sources sont à la **même** seconde constante, elle est
   **installée directement**, sans tampon ni copie (la copie suivie d'un repliement laissait
   jusqu'à 640 Mo de pages touchées par tableau) ;
4. **flush+swap** : échanger les identités entre l'ancien et le nouveau stockage ;
5. **tmp teardown** : détruire l'ancien stockage ;
6. **old-pool drain** : vider les réserves de l'ancienne taille et rendre au système les
   slabs devenus vides.

**Chunk (v1)** reconstruit **par tranches** (16 par défaut, `CHUNK_RECHUNK_SLICES`) et libère
chaque ancien chunk dès qu'il est recopié. Entre deux tranches, les réserves de l'ancienne
taille sont vidées et les slabs vidés rendus au système, y compris dans la **réserve
partagée** quand l'ancien stockage s'y trouve : sans ça, la mémoire de l'ancien stockage
restait acquise jusqu'à la fin de la reconstruction (sur `rechunk_validation`, pic de la
reconstruction 78 → 22 Mo). Mesuré sur v1 à 64 threads : pic de +6,9 Go en une passe,
+1,7 Go avec 16 tranches.

## 5. Les allocateurs par tableau (affinité)

Un tableau dont les métadonnées dépassent 16 Mio (`CHUNK_ALLOC_AFFINITY_MIN_BYTES`) peut
avoir son **propre jeu d'allocateurs** (une clé d'affinité), par rang et par taille. Ses
tampons ne partagent alors aucun slab avec un autre tableau : quand il est reconstruit ou
détruit, ses slabs se vident entièrement et sont rendus au système.

Le prix : un slab entamé par couple (rang, tableau, taille). Avec des slabs de 1 Mio (16 pages
de 64 Kio, pour ne retenir qu'une page par slab rendu), c'est ~780 Mio sur les 16 tableaux du
banc à 64 threads.

D'où deux modes, dont le choix par défaut dépend de la variante :

| mode | quand un tableau reçoit sa clé | défaut pour | pourquoi |
|---|---|---|---|
| **paresseux** | à son premier rechunk | Chunk2, Chunk2A | ils sont surtout faits de métadonnées : un départ sans jeux par tableau consomme exactement comme le statique (Chunk2A : 11 590 Mo après l'initialisation, contre 12 435 Mo en mode immédiat) |
| **immédiat** | à sa création | Chunk (v1) | v1 est surtout fait de données, et sa reconstruction par tranches ne rend la mémoire au fil de l'eau que si l'ancien stockage a ses propres slabs. En mode paresseux, le premier rechunk part de slabs partagés, qui mêlent les blocs de plusieurs tableaux et ne se vident pas : +6,0 Go sur 39,3 Go. Pic de v1 : 41,2 Go en immédiat, 45,6 Go en paresseux, pour la même mémoire de fin |

`CHUNK_ALLOC_AFFINITY_EAGER=0` ou `=1` impose l'un des modes.

## 6. Ce qui doit disparaître avec l'ancien stockage

Un rechunk construit le nouveau stockage sous un **nouvel identifiant de tableau**
(`num_tab`) et détruit l'ancien. Tout ce qui était indexé par l'ancien identifiant doit être
**libéré**, pas seulement vidé. Deux fuites de ce type ont été corrigées en septembre 2026 :

- **Chunk2A** : les lignes de métadonnées par chunk (type de constante, 1 octet ; seconde
  valeur, `sizeof(T)` octets par défaut) étaient vidées par `clear()`, qui garde la capacité. Sur le
  banc avec les 15 tableaux rechunkés : 10 276 → 6 730 Mo en fin de run ;
- **les trois variantes** : l'indicateur « en liste de compression » (1 octet par chunk)
  n'était jamais libéré. Chunk2A −0,52 Go, v1 −0,75 Go.

Par ailleurs, le **seuil de déclenchement des passes de compression** et le **budget des
réserves**, calculés à partir du nombre total de chunks, sont maintenant recalculés aussi à la
destruction d'un stockage, et plus seulement à sa création. Sans effet mesurable sur le temps, mais les seuils
correspondent de nouveau à la géométrie réelle.

## 7. Résultats

Mémoire (Go) sur `shrink_bench`, 64 threads, tous les tableaux confiés au décideur (le
défaut du banc ; `SHRINK_BENCH_ADVISE_ALL=0` revient aux cinq tableaux historiques) :

| variante | statique 64 (fin) | dynamique sans advisor (fin) | **dynamique avec advisor (fin)** | statique 256 (fin) |
|---|---|---|---|---|
| Chunk2A | 12,7 | 12,7 | **6,0** | 7,3 |
| Chunk2 | 21,4 | 21,4 | **16,3** | 17,3 |
| Chunk (v1) | 43,6 | 44,6 | **40,3** | 40,2 |

Coût en temps de l'advisor, mesuré sur **un même binaire** (moyennes sur 3 tours) :

| Chunk2 dynamique | temps | coût |
|---|---|---|
| sans advisor (`CHUNK_ADVISOR_PERIOD=0`) | 8,38 s | — |
| analyses seules (`CHUNK_RECHUNK_DRY_RUN=1`) | 9,07 s | +0,69 s |
| analyses et rechunks (30) | 11,43 s | +2,36 s |

Pour Chunk2A, le coût des rechunks (33) va de 1 à 3,5 s selon les séries de mesures. Pour
Chunk2, les intervalles qui suivent les rechunks sont aussi rapides que ceux d'avant : le coût
se limite aux reconstructions.

## 8. Limites connues

- **v1 ralentit après son rechunk.** Après le passage à 1 024, v1 dynamique tourne 1,5 à
  2 fois moins vite par itération que le statique 1 024 (environ 40 cœurs occupés à un IPC de
  3,65, contre 52 cœurs à 4,24). Un build dynamique **démarré** à 1 024 va aussi vite que le
  statique, et Chunk2 ne ralentit pas après ses rechunks : c'est l'état laissé par la
  reconstruction de v1 qui est en cause. Ni l'ordonnancement de la reconstruction, ni les
  seuils de compression n'expliquent l'écart. Cause encore inconnue.
- **Le modèle est statique.** L'advisor juge sur l'état présent des chunks, pas sur les
  écritures à venir. Sur ce banc, les matérialisations passagères mesurées ne pèsent pas
  (`CHUNK_RECHUNK_CHURN=1`), mais une charge qui écrit beaucoup dans des zones constantes
  pourrait faire regretter un grossissement : c'est le rôle de l'apprentissage
  (`coarsen_trust_ratio`) et de l'anti-retour.
- **Le rechunk des tableaux de booléens** n'a pas de harnais de validation : le banc ne confie
  pas `tab_bool` au décideur.
- **Le pic reste celui du départ** : un run dynamique qui part de 64 atteint d'abord la
  mémoire du statique 64, puis descend.
