# L'algorithme de pattern::Array

> English version: [algo_pattern.md](algo_pattern.md).

> État : fin octobre 2026. `pattern::Array` est un membre de la famille `zero_any` : ce document
> décrit ce qu'il fait **autrement** que `zero_any`, dont il reprend toute la mécanique (registre
> des secondes valeurs, moteur d'écriture, passe de compression, sections de lecture), décrite
> dans [algo_zero_any_fr.md](algo_zero_any_fr.md) et
> [algo_zero_one_value_fr.md](algo_zero_one_value_fr.md). Son protocole sans verrou propre est
> vérifié par modèle : [formal_models_fr.md](formal_models_fr.md), section 7. En cas de doute, le
> code fait foi.

***

## 1. Ce que `pattern` compresse

`zero_any` compresse un chunk uniforme, à n'importe quelle valeur. Une **connectivité** de
maillage -- les numéros des nœuds de chaque maille, maille après maille -- n'est jamais uniforme,
mais très redondante : sur un maillage structuré, la maille `c + 1` a les numéros de la maille
`c` plus un, et un chunk de mailles entières est le chunk précédent plus une constante.
`pattern::Array<T>` compresse un tel chunk en **un motif plus un offset** : l'élément `j` du chunk
vaut `motif[j] + offset`, `motif` étant un tableau de la taille d'un chunk **partagé** par tous
les chunks (de tous les tableaux du type) qui ont la même forme, et `offset` la valeur propre au
chunk.

| état | `values` pointe vers | l'élément `j` |
|---|---|---|
| compressé à zéro | le tampon constant partagé de `zero_any` | 0 |
| compressé à une seconde valeur | le tableau du registre de `zero_any` pour cette valeur | la valeur |
| compressé à un motif | une entrée de la **région des motifs** (§3) | `motif[j] + offset` |
| matérialisé | un tampon propre au chunk | celui du tampon |

Un motif est **normalisé** : `motif[0] = 0`, donc l'offset est l'élément 0 du chunk.
L'arithmétique est modulaire sur le type non signé de même largeur : une lecture rend les bits
exacts, y compris pour les types signés et les offsets négatifs. Entiers seulement
(`static_assert`) : une connectivité est entière, et l'arithmétique modulaire n'a pas
d'équivalent flottant.

La classe est `pattern::Array<T, IntType>`, construite dans la bibliothèque `zero_any`
(`chunk::chunk_zero_any`, `CHUNK_VARIANT_ZERO_ANY`), en géométrie statique ou dynamique. Code :
`chunk/pattern_array.h`, `chunk/pattern_chunk.h`, `internal/pattern_proxy.h`,
`internal/motif_region.h` et leurs compagnons _inline.h ; le code partagé de la famille
bifurque sur le trait `has_motif_state_v` (`chunk/chunk_traits.h`), toujours à la compilation :
le code machine de `zero_any` est le même avec ou sans `pattern` (vérifié fonction par fonction
sur ses bibliothèques, son benchmark et ses programmes de validation).

## 2. Métadonnées par chunk : l'offset dans la case de seconde valeur

`pattern` n'ajoute aucun octet par chunk. Un chunk motif n'a pas de seconde valeur : sa case dans
la ligne des secondes valeurs de `zero_any` (`sizeof(T)`, déjà présente pour chaque chunk)
reçoit son offset. L'autre option, l'offset dans l'objet chunk, coûterait 8 octets par chunk et
épargnerait une ligne de cache par lecture de motif (la ligne est un autre tableau) ; c'est un
point ouvert (§10). Conséquences :

- une lecture d'un chunk motif a besoin de l'**indice** du chunk pour trouver sa case : les
  lectures qui le connaissent passent la case (`ArrayChunkBase::operator[]`, l'itérateur,
  `read_range`, `for_each_value`, les lectures du proxy) ; un chunk motif lu sans sa case est une
  erreur d'invariant, sauf dans les balayages relâchés (le filtre de marquage) ;
- `CHUNK_SECOND_VALUE_INDEXED` est refusé pour les types de plus de 4 octets (leur case contient
  un identifiant du registre, pas un `T`) ;
- un chunk matérialisé depuis son motif garde l'offset dans sa case -- un lecteur qui a chargé le
  pointeur du motif peut encore la lire (§6) -- et les heuristiques de `zero_any` le prennent pour
  la seconde valeur du chunk jusqu'à la compression suivante : un raccourci manqué au pire,
  jamais une valeur fausse.

Les constantes sont celles de `zero_any`, dans le registre des secondes valeurs qu'il partage
avec `pattern` pour le même `T` : chaque membre enregistre une fonction de marquage, et
`reclaim_second_values()` libère ce qu'aucun tableau de l'un ni de l'autre ne nomme plus.

## 3. Le registre des motifs

`internal/motif_region.h`, un par `T` : **une plage virtuelle réservée** (1 Gio par défaut,
`CHUNK_MOTIF_REGION_BYTES`, `MAP_NORESERVE` : seules les pages des entrées écrites sont touchées),
pour que « ce chunk est un motif » soit un test de plage sur son pointeur `values`, sur le chemin
de lecture de chaque élément.

- **Segments.** La plage est découpée en segments (64 Mio, `CHUNK_MOTIF_SEGMENT_BYTES`), chacun
  appartenant à **une** taille d'entrée, attribués à la demande : un build statique peut tous les
  utiliser, les tailles d'un build dynamique se les partagent. La taille d'une entrée est celle de
  son segment. Plus de segment pour une taille : le motif n'est pas enregistré
  (`MotifStats::full`) et le chunk reste matérialisé.
- **Recherche.** Un hash de 64 bits du motif normalisé (la taille mélangée dedans), **confirmé**
  en comparant les tableaux, dans un index à adressage ouvert lu sans verrou ; un enregistrement
  prend le mutex. Un index agrandi garde ses anciennes tables jusqu'au prochain reclaim : un
  lecteur ne tient jamais une table libérée (la technique du registre de `zero_any`).
- **Admission.** Enregistrer chaque forme distincte ferait coûter plus cher les données
  aléatoires. Un motif est enregistré à sa **deuxième** rencontre, **par un autre chunk** : la
  première ne fait qu'inscrire, dans une table de probation de 64 Ki entrées de 16 bits, 8 bits de
  plus de son hash et une étiquette de 8 bits du chunk. Le même chunk revu avec le même contenu
  (des données aléatoires réécrites à l'identique) reste en probation : un motif qu'un seul chunk
  utilise n'économise rien. Une collision coûte une entrée manquée ou inutile, jamais une valeur
  fausse. Avec 8 bits au lieu de 16, la validation dynamique a vu des chunks aléatoires
  enregistrer leur propre motif par collision, une première rencontre sur quelques centaines.
- **Reclaim.** `pattern::Array<T>::reclaim_second_values()`, à un point séquentiel : les
  constantes comme `zero_any`, puis les motifs qu'aucun chunk ne désigne plus ; leurs
  identifiants sont réutilisés, l'index reconstruit et la table de probation vidée.

## 4. Quand un chunk devient candidat à la compression

Le seul signal, ce sont les écritures, et une connectivité est écrite une fois, souvent dans
l'ordre. Aux critères de `zero_any`, `pattern` ajoute la **règle de complétion** : une écriture du
**premier ou du dernier élément** d'un chunk le met en file
(`pattern::Proxy::is_completion_write()`, dans le pré-filtre rapide et dans le prédicat). Le
balayage dit si le chunk est vraiment un motif ; un chunk mis en file trop tôt est réexaminé à la
prochaine écriture de ces positions. La règle rattrape aussi les chunks uniformes que le marquage
de `zero_any` manque (une dernière écriture sur un élément dont l'ancienne valeur n'était aucune
des deux constantes). Limite connue : un chunk rempli dans un ordre aléatoire, son premier et son
dernier élément avant le reste, peut rester matérialisé.

## 5. La passe de compression

La passe de `zero_any`, avec une troisième classification. Un chunk réservé qui n'est aucune des
deux constantes de `zero_any` est lu d'un bloc, normalisé, haché et soumis à la règle
d'admission (`Chunk::classify_motif()`) ; une correspondance garde la réservation de la passe.
Après les constantes de la même passe, `compress_motif_candidates()` compresse chacun, à sa
propre taille de chunk, dans cet ordre :

1. l'offset écrit dans la case du chunk (release) ;
2. la case de type mise à `CstSlot::motif` ;
3. le pointeur des valeurs basculé par CAS du tampon vers le motif (acq_rel) ;
4. le tampon rendu au pool après la période de grâce, comme pour une compression vers une
   constante.

## 6. La lecture, et les réécritures de la case

Un lecteur lit le pointeur, puis l'offset : deux mots. La case est réécrite par une passe de
compression, **avant** son CAS : vers un nouvel offset (compression vers un motif), vers une
seconde valeur ou vers 0 (compression vers une constante). Un lecteur qui tient un ancien
pointeur de motif pourrait alors ajouter à son motif la case d'un autre état -- une valeur que le
chunk n'a jamais tenue. La période de grâce des sections de lecture n'y peut rien : elle protège
le tampon que la passe libère, pas la case. Le modèle GenMC réduit l'a trouvé en deux exécutions
(F5, [formal_models_fr.md](formal_models_fr.md) section 8.6) ; la règle telle qu'implémentée :

- le lecteur lit la case en **acquire**, puis relit le pointeur (`Chunk::motif_offset()`) :
  inchangé, l'offset est celui de l'état que les deux lectures encadrent ; changé, il recommence ;
- chaque réécriture de la case d'un chunk d'un membre à état motif est une écriture atomique
  **release** (`SecondValueRef::reset_release()`, `set_from_array_release()`), faite après que la
  passe a vu le tampon matérialisé : un lecteur qui voit la nouvelle case ne peut plus revoir
  l'ancien pointeur ;
- la case d'abord, le pointeur ensuite ; un ABA (recompression vers le même motif) est sans
  danger.

Vérifié pour toutes les cibles, jusqu'à 3 lecteurs ou 3 lectures, sous RC11 et IMM ; chaque partie
de la règle est porteuse (les mutations du modèle). Coût : une lecture acquire et une seconde
lecture du pointeur, dans la même ligne de cache, sur la lecture d'un chunk motif seulement.

## 7. L'écriture

Le chemin d'écriture de `zero_any`. Une écriture dans un chunk motif le **matérialise** : un
tampon rempli de `motif[j] + offset`, publié par CAS contre le pointeur du motif (un CAS perdu
recycle le tampon). `wait_before_materialise()` tient la passe à l'écart du chunk (protocole 1) :
la case y est stable, lue une fois. L'offset reste dans la case (§2).

## 8. Géométrie dynamique : le rechunk et le conseiller

Un motif a le nombre d'éléments de son chunk ; la région tient un jeu de segments par taille (§3),
et la passe compresse chaque candidat à sa propre taille.

- **Rechunk.** `rechunk_to()` reconstruit le tableau depuis les valeurs des anciens chunks : un
  chunk motif est lu comme son motif plus son offset, jamais comme le raccourci « même seconde
  constante » de `zero_any` (`is_cst_val2()` est vrai pour un chunk motif, compressé pour le
  moteur d'écriture). Les chunks reconstruits, écrits par les proxies, sont mis en file par la
  règle de complétion et compressés en motifs de la nouvelle taille par la passe suivante.
- **Conseiller.** `ChunkSizeAdvisor` lit un chunk motif comme une donnée (motif plus offset),
  jamais comme une constante : ses prévisions de constantes sont celles que `zero_any` fait sur
  les mêmes valeurs. Il ne prévoit aucune compression en motif (§10) : jamais optimiste.

## 9. Résultats

`shrink_bench` porte une connectivité de maillage, `tab_conn`
(`examples/shrink_bench/bench_phase_conn.cpp`, `--no_conn` pour l'omettre) : environ `sz` entiers,
des rangées entières de 256 mailles hexaédriques ; chaque itération lit une bande de rangées sur 16
et la renumérote (l'ordre des nœuds tourné, ou rétabli), si bien que la passe recompresse une bande
vers un autre motif partagé à chaque itération. C'est un `pattern::Array<int>` dans les builds
`zero_one_value` et `zero_any` (sa mécanique est celle de `zero_any`, en en-têtes : il vit dans
l'une ou l'autre bibliothèque), un `zero::Array<int>` dans un build `zero`, un vecteur dans la
référence ; ses checksums sont comparés à ceux de la référence à chaque point de contrôle, comme
pour tous les tableaux. Sur un run de contrôle (`--divisor 50 --iter_max 40`, 4 threads), 384 de
ses 385 chunks sont compressés dans les deux builds (1 536 sur 1 537 à la taille dynamique 64) ;
le build `zero` en compresse un.

`examples/access_cost/access_cost_pattern.cpp` (`access_cost_pattern_<taille>`) chronomètre un
`std::vector<long>`, un `zero_any::Array<long>` et un `pattern::Array<long>` dans le même binaire,
et affiche la mémoire de la connectivité d'un maillage hexaédrique structuré (256 mailles de
large, 8 nœuds par maille) dans chacun. Indicatif, sur un conteneur partagé de 8 cœurs, 2^24
éléments, chunk 256 (les mesures de référence sont à faire sur la machine cible) :

| | vecteur | `zero_any` | `pattern` |
|---|---|---|---|
| mémoire de la connectivité | 134 Mo | 136 Mo (101 %) | 1,6 Mo (1,2 %) : 1 chunk matérialisé, 65 535 chunks motifs, 1 motif |
| read-varied, 8 threads (ns par élément) | 0,46 | 2,34 | 2,36 |
| read-conn, 8 threads | 0,46 | 2,35 | 1,60 |
| read-conn-random, 8 threads | 5,66 | 17,7 | 4,15 |
| write-conn, 8 threads | 4,79 | 8,61 | 9,51 |

Le test de plage des motifs ne coûte presque rien à une lecture matérialisée (read-varied) ; une
lecture de motif est plus rapide que la lecture matérialisée de `zero_any`, et une lecture
aléatoire plus rapide que celle du vecteur, le motif restant en cache ; l'écriture paie la
classification des chunks.

## 10. Limites et points ouverts

- **L'emplacement de l'offset** (§2) : la case, contre l'objet chunk (+8 octets par chunk, une
  ligne de cache de moins par lecture de motif) ; à mesurer avec `access_cost_pattern` sur des
  lectures dominées par les motifs.
- **La prévision de motifs du conseiller** (§8) : à une taille candidate, une fenêtre serait
  compressée en motif si son contenu normalisé est partagé avec une autre fenêtre. Une estimation
  sur les chunks échantillonnés coûte peu, mais change `ChunkSizeEstimate` (une struct publique)
  et le décideur.
- **Les motifs plus courts qu'un chunk** (un pas qui ne divise pas la taille du chunk, un offset
  qui croît dans le chunk) ne sont pas couverts : `chunk[j] = P[j mod s] + (j div s) * pas` serait
  une seconde sorte de motif. Mesurer d'abord sur de vraies connectivités.
- **`CHUNK_SECOND_VALUE_INDEXED`** est refusé pour les types de plus de 4 octets (§2).
- **L'ordre de remplissage aléatoire** (§4) et **une taille d'entrée par segment** (§3).
