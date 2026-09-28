# L'algorithme de `ArrayChunk` (Chunk, v1)

*État : fin septembre 2026. Ce document décrit ce que v1 fait **autrement** que Chunk2 ; la
machinerie commune (proxys, écritures différées, barrière de lecture, compression en vol,
opérations sur plages) est décrite dans [`algo_chunk2_fr.md`](algo_chunk2_fr.md), et la
géométrie dynamique dans [`rechunk_adaptatif_fr.md`](rechunk_adaptatif_fr.md). En cas de
doute, c'est le code qui fait foi.*

---

## 1. Ce que v1 compresse

v1 est la plus simple des trois variantes : un chunk est **compressé à zéro** (`T{}`) ou
**matérialisé**. Il n'y a pas de seconde valeur, ni par tableau (Chunk2), ni par chunk
(Chunk2A).

| état | `values` pointe vers | test |
|---|---|---|
| compressé | le tampon constant partagé `cst_val1`, rempli de zéros | `is_cst_val1()` |
| matérialisé | un tampon propre au chunk | `written()`, c'est-à-dire `values != cst_val1` |

Le pointeur `values` sert donc à la fois de données et d'état. Un chunk ne connaît pas son
propre nombre d'éléments : la géométrie vit dans la politique de taille du tableau
(`size_policy_`), et c'est pourquoi les listes de candidats à la compression transportent le
nombre d'éléments de chaque chunk (voir `TChunk.h`).

Le build : `-DCHUNK=1`. Code : `chunk/chunk_std.h`, `chunk/array_chunk.h`,
`internal/chunk_std_inline.h`, `internal/array_chunk_inline.h`, `internal/proxy.h`.

## 2. Ce que coûte un chunk

L'objet chunk fait 16 octets (dans les builds OpenMP, où le compteur de zéros du mode
séquentiel est un type vide, voir §4), plus un octet « déjà en liste de compression » et un
bit du *dirty bitset*. Sans seconde valeur, v1 n'a **aucune métadonnée de type ou de
valeur** par chunk : c'est la variante la plus économe en métadonnées, et celle dont la
mémoire est le plus faite de **données matérialisées**. Sur `shrink_bench` à 64 threads,
environ 38 Go de ses 40 Go sont des données.

## 3. Lecture et écriture

La lecture est celle de Chunk2 (§5 de [`algo_chunk2_fr.md`](algo_chunk2_fr.md)) : une valeur,
lue dans le tampon du chunk ou dans `cst_val1`, après la barrière de lecture.

L'écriture passe par le proxy `Proxy<T>` :

- écrire **zéro** dans un chunk compressé : rien à faire ;
- écrire une **autre valeur** dans un chunk compressé : le chunk est **matérialisé** (un
  tampon pris dans la liste de réutilisation du thread ou alloué, pré-rempli de zéros, publié
  par CAS), puis la valeur est écrite ;
- écrire dans un chunk matérialisé : simple écriture.

## 4. Quand un chunk devient candidat à la compression

**Builds séquentiels** : chaque chunk tient un **compteur de zéros**
(`ChunkCounterPolicy`, `nb_val_eq_zero`). Quand il atteint le nombre d'éléments du chunk, le
chunk est compressé tout de suite (`manage_full_zero`). Ce compteur doit être exactement
synchronisé avec le contenu : un compteur désynchronisé compresserait un chunk qui ne l'est
pas, d'où le soin mis à le propager dans les copies.

**Builds OpenMP** : pas de compteur. Une écriture qui fait passer un élément **d'une valeur
non nulle à zéro** (`should_mark_for_compression`) met le chunk dans la **liste de candidats**
du thread, sauf s'il y est déjà (l'octet « déjà en liste »). Le même chemin marque aussi le
chunk quand il détecte qu'un autre thread a écrit au même endroit en même temps (la valeur en
mémoire n'est plus celle attendue).

## 5. La passe de compression par bloc

Quand la liste d'un thread dépasse le seuil de déclenchement (§6), une passe
`make_block_compression` s'exécute :

1. si un vidage des écritures différées est en cours, la passe **renonce avant de toucher aux
   candidats** (`block_pass_yields_to_flush`, `block_pass_precheck.h`). À 128 threads sur
   Grace, marquer puis démarquer les candidats à chaque tentative refusée coûtait 10 millions
   de tours et doublait le temps du run ; `CHUNK_BLOCK_PRECHECK_LEGACY=1` rétablit l'ancien
   ordre ;
2. chaque candidat est **réservé** (`steal_mark`), avant l'ouverture de la fenêtre de
   compression (`CompressionManager::start()`), dont la barrière publie les réservations ;
3. `fill_list_to_compress` relit chaque candidat et ne garde que ceux qui sont **entièrement
   à zéro** ;
4. chacun de ceux-là est compressé (`clear_int_st` : `values` repasse à `cst_val1` par CAS, et
   le tampon rejoint la liste de réutilisation) ;
5. les réservations sont levées et les octets « déjà en liste » remis à zéro.

Une liste de candidats mélange les chunks de **tous les tableaux** d'un même type, qui
peuvent avoir des tailles différentes après un rechunk : chaque chunk est compressé avec
**son** nombre d'éléments, porté par l'entrée de la liste.

## 6. Le seuil de déclenchement

Il vaut `somme_chunk_create / (100 × nombre de threads)`, avec un plancher de 100 candidats
(`ChunkDef` : `factor_div_nb_chunk_for_compress`, `size_min_list_chunk_to_check_compress`).
`somme_chunk_create` est le nombre de chunks existants de ce type. Deux corrections de
septembre 2026 le tiennent juste :

- il est **décrémenté** à la destruction d'un tableau (sans quoi il ne faisait que croître :
  seuil de 390 au début d'un run, ~42 000 à la fin, et presque plus aucune compression) ;
- le seuil est **recalculé** à la destruction comme à la création
  (`CHUNK_THRESHOLD_TEARDOWN_LEGACY=1` pour l'ancien comportement).

À 64 threads, le calcul tombe presque toujours sur le plancher : c'est lui qui fixe la
fréquence des passes.

## 7. Ce que v1 fait autrement pendant un rechunk

Le déroulé général est celui de [`rechunk_adaptatif_fr.md`](rechunk_adaptatif_fr.md) §4. Deux
particularités viennent du volume de données de v1 :

- **la reconstruction par tranches** (16 par défaut, `CHUNK_RECHUNK_SLICES`) : chaque tranche
  libère les anciens chunks qu'elle a recopiés, et entre deux tranches, les réserves de
  l'ancienne taille sont vidées et les slabs vidés rendus au système. Pic de la
  reconstruction à 64 threads : +6,9 Go en une passe, +1,7 Go en 16 tranches ;
- **l'affinité immédiate par défaut** : chaque gros tableau a ses propres slabs dès sa
  création, pour que les tranches puissent vraiment les vider. En affinité paresseuse, le
  premier rechunk part de slabs partagés entre tableaux, qui ne se vident pas : pic de 45,6 Go
  au lieu de 41,2 Go, pour la même mémoire de fin.

## 8. Résultats et limites

Sur `shrink_bench` (64 threads), v1 dynamique finit à **40,3 Go**, comme le statique 256
(40,2 Go), et 0,6 Go au-dessus du statique 1 024 (39,7 Go), la taille qu'il atteint. Deux
raisons à ce gain faible :

- ses métadonnées sont déjà faibles, et c'est surtout sur elles que le grossissement fait
  gagner ;
- après son rechunk, v1 dynamique **ralentit** (1,5 à 2 fois par itération par rapport au
  statique 1 024), pour une raison encore inconnue : un build dynamique démarré directement à
  1 024 va aussi vite que le statique, et Chunk2 ne ralentit pas après ses rechunks.

Pour une charge comme celle du banc, un v1 **statique** à la bonne taille reste aujourd'hui le
meilleur choix.
