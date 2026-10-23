# Modèles formels des protocoles sans verrou

> English version: [formal_models.md](formal_models.md).

[synchronisation_dekker_fr.md](synchronisation_dekker_fr.md) explique les trois protocoles de la
bibliothèque et argumente leur correction sur le papier. `formal/` contient les mêmes protocoles
sous forme de **modèles réduits** : de petits programmes C11, un par protocole, qui ne gardent
que les atomiques du source avec leurs ordres mémoire exacts, et qu'un vérificateur de modèles --
[GenMC](https://github.com/MPI-SWS/genmc) -- explore exhaustivement sous le modèle mémoire C11
(RC11, et IMM en second avis). Ce document dit ce que sont les modèles, comment les lancer, ce
qu'ils ont prouvé et ce qu'ils ont trouvé.

## 1. Ce qu'un modèle prouve, et ce qu'il ne prouve pas

Un modèle réduit est un programme de quelques threads (jusqu'à 5 écrivains et une passe) et d'un chunk. Le vérificateur énumère
**toutes** les exécutions que le modèle mémoire autorise -- chaque entrelacement, chaque choix
de lecture qu'un processeur faible pourrait faire -- et signale une assertion violée, un accès à
une mémoire libérée ou (avec `--check-liveness`) une boucle d'attente qui ne finit jamais. Là où
les séries de `multi_launch_stress.py` trouvent un défaut une fois sur des milliers de runs, le
vérificateur le trouve à la première exécution qui le montre, en une fraction de seconde, et dit
« aucun » sinon.

Ce que « aucun » veut dire a des limites, listées en section 8 : le modèle est une réduction (la
fidélité est le risque), il est borné (jusqu'à 5 écrivains), et C11 n'a pas de `membarrier()`. La
façon de faire confiance à un modèle est de le **calibrer** : lui donner les bugs que la
bibliothèque a vraiment eus, sous forme de mutations du modèle, et vérifier que le vérificateur
les refuse tous. La section 4 est cette calibration.

## 2. Lancer les modèles

GenMC a besoin d'une installation de développement de LLVM 19 ou plus et d'un compilateur C++20.
Sous Ubuntu 24.04 :

```shell
sudo apt install llvm-19-dev clang-19 libclang-19-dev libffi-dev g++-14
git clone https://github.com/MPI-SWS/genmc.git && cd genmc
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
      -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm -DClang_DIR=/usr/lib/llvm-19/lib/cmake/clang -B build
cmake --build build -j && export PATH=$PWD/build/bin:/usr/lib/llvm-19/bin:$PATH
```

Sans root, Spack fournit la même chose (`spack install llvm@19 +clang`, puis les lignes `cmake`
ci-dessus contre son préfixe). Ensuite, depuis le dépôt :

```shell
formal/run_genmc.sh            # toutes les configurations et mutations, RC11 ; sortie 1 sur un verdict inattendu
formal/run_genmc.sh --imm      # la même chose sous IMM
genmc --check-liveness -- -DSHAPE_SYMMETRIC -DINIT_CONSTANT formal/writer_compressor.c   # un cas à la main
```

Avec CMake, `ctest -L formal` lance `formal_genmc` (quand `genmc` est sur le PATH ou que
`-DGENMC_EXECUTABLE=` le nomme) et `formal_native_*` : chaque modèle compilé comme un programme
ordinaire et lancé 300 fois, ce qui prouve que le modèle est un vrai programme et sert d'oracle
faible pour le protocole. Rien de `formal/` n'entre dans la bibliothèque ni n'est installé.

## 3. Le modèle du protocole 1 : écrivain contre compresseur

`formal/writer_compressor.c` est les sections 5.1 à 5.3 et 5.6 du document de synchronisation :
NWRITERS écrivains (2 par défaut) écrivent chacun une fois un élément d'un chunk, pendant qu'une
passe de compression marque le chunk, ouvre sa fenêtre, le parcourt et vole son tampon s'il est
tout à zéro. Chaque nom du modèle tient lieu d'un endroit du source, avec le même ordre mémoire :

| modèle | source (`chunk/public/chunk/internal/`) |
|---|---|
| `flag[r]`, `announce_write()`, `end_write()` | `write_compress_sync_policy.h` : `WritingFlag::v`, `announce_write()`, `end_write()` |
| `any_writing_on_marked()`, `any_writing()` | `write_compress_sync_policy.h` : `any_thread_writing_on_marked()`, `any_thread_writing()` |
| `hard_inflight`, `stealing_inflight`, `must_defer_write()`, `is_stealing()` | `compression_manager.h` : `hard_inflight_`, `stealing_inflight_`, `must_defer_write()`, `is_hard_compression_inflight()` |
| `compressor_start()`, `compressor_finish()`, `heavy_barrier()` | `compression_manager_inline.h` : `start()`, `finish()` ; `dekker::heavy_barrier()` |
| `mark`, `steal_mark()`, `steal_marked()`, `steal_unmark()` | `chunk_base.h` : `steal_mark_` et ses trois verbes |
| `values`, `values_load()`, `cas_install()`, `steal()` | `chunk_base.h` : `values`, `values_load()`, `compare_and_swap_gen()` ; `zero_any_chunk.h` : `try_mark_cst_and_get_old()` |
| `materialise()`, le parcours, l'écriture d'élément | `zero_any_chunk_inline.h` : `create_value()`, `wait_before_materialise()`, `fill_list_to_compress()` ; `relaxed_element.h` |
| `writer()` | `proxy_base_inline.h` : `write_with_optional_lock_impl()` |

Trois propriétés sont affirmées : **P1**, un écrivain n'écrit jamais dans un tampon que la passe a
rendu (le modèle le libère, et GenMC signale tout accès ultérieur) ; **P2**, une écriture qui a
pris le chemin direct est dans le chunk quand tous les threads ont fini, et une écriture différée
est comptée -- une écriture n'est jamais perdue en silence ; **P3**, avec `--check-liveness`,
chaque boucle d'attente se termine.

Le modèle a les deux **formes** de la section 4 du document de synchronisation. `SHAPE_SYMMETRIC`
(le défaut) est ce que le vérificateur peut vérifier : C11 n'a pas de `membarrier()`.
`SHAPE_ASYMMETRIC_NAKED` est la forme asymétrique avec l'écriture simple de l'écrivain et *sans*
`membarrier()` en face : elle doit échouer, et son échec est exactement ce que l'appel système
répare. La réduction de l'asymétrique au symétrique est l'argument de la section 4.2 et du contrat
de `membarrier(2)`, pas un résultat du vérificateur. Les **configurations** sont `INIT_CONSTANT` (le chunk part
comprimé, les écrivains le matérialisent ; le défaut est un tampon tout à zéro, candidat pour la
passe), `BOUNDED_COMPRESSOR_WAIT` (le budget `CHUNK_COMPRESS_MAX_YIELDS` réduit à son effet : un
regard, puis abandon) et `NWRITERS`.

## 4. La calibration

Chaque ligne ci-dessous est un cas de `run_genmc.sh` ; les verdicts sont ceux de GenMC 0.19 sous
RC11 et sous IMM (identiques). « Exécutions » est le nombre d'exécutions complètes explorées.

| cas | attendu | verdict | exécutions |
|---|---|---|---|
| le protocole du source, marques, départ sur tampon | passe | passe | 21 |
| source, marques, départ sur constante (les écrivains matérialisent) | passe | passe | 26 |
| source, marques, 3 écrivains ; départ sur constante, 3 écrivains | passe | passe | 120 ; 191 |
| source, vivacité, attente **non bornée** du compresseur, départ sur tampon ou constante | passe | passe | 21 ; 26 |
| source, vivacité, départ sur constante, 3 écrivains ; attente bornée (la bibliothèque) | passe | passe | 191 ; 82 |
| source, vivacité, 5 écrivains, départ sur tampon ; sur constante | passe | passe | 3 938 ; 9 637 |
| `MUT_NO_CHECK2` : pas de regard sur la porte après l'annonce | échoue | écriture perdue | -- |
| `SHAPE_ASYMMETRIC_NAKED` : forme asymétrique sans `membarrier()` | échoue | accès à une mémoire libérée | -- |
| `MUT_NO_FENCE_COMPRESSOR` : le compresseur annonce sans sa barrière | échoue | accès à une mémoire libérée | -- |
| `MUT_WAIT_ON_HARD` : la matérialisation attend sur `hard_inflight_` (le triangle du 5.3) | échoue | violation de vivacité | 5 |
| `MUT_RELAXED_MARK` : la marque relaxed au lieu de release/acquire (5.3) | échoue | accès à une mémoire libérée | 8 |
| `MUT_MATERIALISE_WAIT_UNDER_FLAG` : l'attente de matérialisation d'avant **F2**, attente non bornée du compresseur | échoue | **violation de vivacité** | 4 |
| la même avec l'attente bornée de la bibliothèque : sûre, un arrêt seulement | passe | passe | 116 |

Toutes les mutations sont refusées, et le protocole du source passe dans toutes ses
configurations. Les deux dernières lignes sont le comportement que ce travail a changé dans le
source (section 7, F2) : la bibliothèque d'avant est refusée, la bibliothèque d'aujourd'hui passe. Le verdict sur `MUT_RELAXED_MARK` mérite un mot : la section 5.3 dit que
la marque « a besoin de sa propre paire release/acquire » et qu'une lecture relaxed a été
« confirmée par TSan » -- TSan ne modélise pas les barrières, cette confirmation seule aurait pu
être un artefact du build symétrique. Le vérificateur tranche : l'acquire *est* porteur, et la
trace montre comment. Un écrivain lit la marque à zéro **après** `steal_unmark()` ; avec une
lecture relaxed, rien ne l'ordonne après le CAS de la passe, donc son `values_load()` peut rendre
l'ancien tampon -- celui que la passe vient de rendre -- et l'écriture tombe dans une mémoire
libérée. L'acquire s'apparie avec le release de `steal_unmark()` et emporte le CAS avec lui.

## 5. Le modèle du protocole 2 : compresseur contre flush

`formal/compressor_flush.c` ajoute la section 5.4 par-dessus le protocole 1. NWRITERS écrivains (1
par défaut, jusqu'à 3 vérifiés) écrivent chacun une fois un élément -- **différé** quand le regard
n° 2 voit la marque de la passe, direct sinon -- et un écrivain qui a différé vide ensuite son
propre rang, comme la bibliothèque le fait au-delà du seuil d'arriéré : `announce_flush()`
(incrément seq_cst, barrière), l'attente des voleurs sur `stealing_inflight_`, puis l'application
exactement comme `lbd_equal_asynch` le fait : une écriture **simple** (« les flushes excluent les
voleurs »), la confirmation après écriture, la boucle de reprise. La passe a ses deux regards sur
les flushs, la sortie anticipée après `hard_inflight_++` et le second regard après
`stealing_inflight_++`. Une quatrième propriété rejoint les trois du protocole 1 : **P4**,
l'écriture simple de l'application n'est jamais en course avec le parcours de la passe (la détection de
courses de GenMC).

| cas | attendu | verdict | exécutions |
|---|---|---|---|
| source, 1 écrivain ; 2 écrivains ; 3 écrivains (vivacité vérifiée) | passe | passe | 12 ; 403 ; 25 810 |
| `MUT_NO_SECOND_LOOK` : le bug du 5.4 (1 et 2 écrivains) | échoue | **course**, et détection de courses coupée l'**application perdue** elle-même | 1 |
| `MUT_NO_SECOND_LOOK` + `MUT_NO_POSTCHECK` | échoue | application perdue | 1 |
| `MUT_FLUSH_NO_DRAIN` : le flush applique sans attendre la fin des voleurs | échoue | application perdue | 1 |
| `MUT_NO_FIRST_LOOK` : la sortie anticipée retirée (1 et 2 écrivains) | passe | passe | 10 ; 357 |
| `MUT_NO_POSTCHECK` : la confirmation après écriture retirée (1 et 2 écrivains) | passe | passe | 12 ; 403 |

Trois choses que la table dit. Le **second regard** est la paire : sans lui le premier
contre-exemple du vérificateur est le bug de l'historique, ligne pour ligne -- l'écrivain diffère
sur la marque ; le flush annonce, trouve `stealing_inflight_` à zéro (la passe est parquée),
applique l'écriture, sa confirmation après écriture passe ; la passe ouvre alors sa fenêtre, entre en vol,
parcourt les zéros d'avant l'application, remet le chunk sur la constante par CAS et rend le tampon ;
l'écriture appliquée a disparu. Le **premier regard** « ne s'apparie avec rien », comme le dit la
section 5.4 : le retirer ne change aucun verdict, c'est une optimisation. La **confirmation après
écriture** est une seconde ligne de défense, pas l'exclusion : avec le second regard en place elle
ne sert jamais, et sans le second regard elle ne sauve pas l'écriture (le CAS de la passe peut
tomber après la lecture de la confirmation).

## 6. Le modèle du protocole 3 : lecteurs contre réemploi des tampons

`formal/reader_reclaim.c` est la section 5.5 (`internal/read_reclaim.h`). Un chunk tenant un
tampon tout à zéro ; NREADERS lecteurs (2 par défaut, 3 vérifiés) faisant chacun NREADS lectures
pures (1 par défaut ; 2 et 3 exercent la règle « attendre que la séquence bouge, pas qu'elle soit
paire »), à travers `ReadSection` comme `element_at()` : l'écriture impaire d'entrée et la
barrière légère, `values_load()`, la lecture de l'élément, l'écriture paire de sortie (release).
Une passe lancée en région vole le tampon (CAS vers la constante), attend la période de grâce
comme `PassScope` -- barrière lourde, puis chaque slot, les impairs jusqu'à ce qu'ils bougent,
puis le compteur de débordement -- et recycle le bloc, modélisé par une écriture de poison (la
valeur qu'un autre chunk y mettrait) suivie du `free`. `OVERFLOW_READER` envoie un lecteur sur le
chemin du compteur partagé d'un programme qui a plus de threads lecteurs que
`CHUNK_READ_RECLAIM_SLOTS`. Les propriétés : un lecteur ne déréférence jamais un tampon libéré, ne
lit jamais une valeur que le chunk n'a jamais tenue, et la période de grâce se termine.

| cas | attendu | verdict | exécutions |
|---|---|---|---|
| source, 2 lecteurs ; 3 lecteurs ; une lecture chacun, vivacité vérifiée | passe | passe | 25 ; 125 |
| source, 2 lectures par lecteur ; 3 lectures ; 3 lecteurs × 2 lectures | passe | passe | 225 ; 1 089 ; 3 375 |
| source, lecteur en débordement ; avec 2 lectures | passe | passe | 15 ; 90 |
| `MUT_ENTER_RELAXED`, 2 lectures : l'écriture impaire d'entrée telle qu'elle était -- **F3** | échoue | **accès à une mémoire libérée** | 19 |
| `MUT_ENTER_RELAXED`, 1 lecture : le trou demande des lectures enchaînées | passe | passe | 25 |
| `MUT_WAIT_FOR_EVEN`, 2 lectures (l'issue de F3 côté passe, non retenue) | passe | passe | 169 |
| `MUT_NO_GRACE` : la passe libère sans attendre | échoue | accès à une mémoire libérée | -- |
| `MUT_NO_HEAVY_BARRIER` : la passe lit les slots sans sa barrière | échoue | accès à une mémoire libérée | -- |
| `MUT_LEAVE_RELAXED` : l'écriture paire de sortie relaxed au lieu de release | échoue | accès à une mémoire libérée | -- |
| `SHAPE_ASYMMETRIC_NAKED` : le lecteur sans sa barrière (pas de `membarrier()`) | échoue | accès à une mémoire libérée | -- |
| `MUT_OVERFLOW_NO_FENCE` : l'annonce de débordement telle qu'elle était -- **F4** | échoue | accès à une mémoire libérée | -- |

Le protocole tient, toutes les mutations sont refusées, et les deux lignes marquées F3 et F4 sont
la bibliothèque telle qu'elle était avant ce travail (section 7) : refusée telle qu'elle était,
passante telle qu'elle est.

## 7. Constats

### 7.1 F1 : la porte du build sans marques lisait `hard_inflight_` en relaxed (clos : le build a été retiré)

Le build `SHRINK_SMALL_CHUNK` -- chunks de 8 octets, pas de marque par chunk, une porte globale --
avait `CompressionManager::must_defer_write()` égal à `hard_inflight_.load(relaxed) > 0`. Le
vérificateur, sur la configuration de ce build dans le modèle, a trouvé cette exécution : la passe va jusqu'au bout -- marque, `start()`, parcours,
CAS de `values` vers la constante, tampon rendu, `finish()` -- et c'est seulement ensuite que le
regard n° 2 de l'écrivain lit `hard_inflight_`, revenu à zéro. L'écrivain continue ; son
`values_load()` (acquire) n'est pas ordonné après le CAS de la passe, puisque rien ne s'est
synchronisé avec lui, et rend l'**ancien** pointeur : l'écriture part dans le tampon rendu. Les
barrières symétriques ne sont pas en cause : elles promettent seulement que l'écrivain voit
l'incrément *ou une valeur plus tardive* du compteur, et le décrément est une valeur plus tardive.
La marque par chunk n'a pas ce trou parce que `steal_unmark()` est une écriture release et
`steal_marked()` une lecture acquire. Sur AArch64 le `ldr` relaxed du compteur peut être servi
après un `ldar` ultérieur de `values` ; sur x86 les deux lectures restent dans l'ordre et le trou
est fermé par le matériel.

Le correctif tenait en un mot, `memory_order_acquire` sur cette lecture, et le vérificateur
l'acceptait à 2 et 3 écrivains. Le build a ensuite été **retiré** (CHANGELOG, *Removed*) : un
define `-D` qu'aucun test ni script de build n'exerçait, qui échouait ses propres harnais, et un
second protocole d'écriture à garder correct. Tout build parallèle a désormais les marques
(`chunk_config::use_chunk_steal_mark`), et la configuration du build retiré a quitté le modèle
avec lui : cette section est le compte rendu du constat, et la section 5.3 de
`synchronisation_dekker_fr.md` en garde une ligne.

### 7.2 F2 : un écrivain qui matérialise et un compresseur parqué s'attendaient (corrigé)

Le commentaire de `wait_before_materialise()` disait qu'elle « est un no-op en pratique : un
écrivain du chemin rapide n'arrive jamais ici avec le chunk marqué (la porte l'a arrêté) ». Le
modèle montre la fenêtre que la phrase sautait : la marque peut arriver **entre** le regard n° 2 et
`wait_before_materialise()`. Un écrivain passe le regard n° 2 sur un chunk comprimé, trouve
`values` à la constante et entre dans `create_value()` ; la passe marque le chunk et se parque dans
`start()`, à attendre les drapeaux qui pointent vers un chunk marqué -- celui de cet écrivain.
L'écrivain tourne sur la marque, le compresseur sur le drapeau. Dans la bibliothèque l'attente du
compresseur est bornée (`CHUNK_COMPRESS_MAX_YIELDS`, 10 000 yields de 1 000 pauses) : pas un
interblocage, mais un arrêt des deux threads pendant tout ce budget, après quoi la passe est
abandonnée (le compteur de porte des abandons sur attente trop longue, dans le rapport
`CHUNK_STATS`, les compte). Il faut un chunk à une constante tout en étant dans une liste de
candidats -- le cas dont `fill_list_to_compress()` se garde avec `is_not_cst_val1()` -- donc c'est
rare, pas impossible.

Le correctif découle du protocole lui-même : un écrivain du chemin rapide tient son drapeau et a
passé le regard n° 2, donc une marque qu'il voit maintenant appartient à un compresseur
**parqué** sur ce drapeau ; rien ne peut voler le chunk avant `end_write()`, et l'attente ne
protège rien. `wait_before_materialise()` prend désormais le rang de l'appelant et rend la main
aussitôt quand le drapeau de ce rang est sur le chunk (`WriteCompressSyncPolicy::is_thread_writing_on()`,
une lecture relaxed du propre drapeau du thread). Les appelants sans drapeau -- `alloc_chunk()`,
le point d'entrée de préallocation utilisable depuis une région parallèle, et l'application d'un flush --
gardent l'attente, que le modèle ne couvre pas (protocole 2). Avec le correctif, le vérificateur
prouve sûreté et vivacité même avec une attente **non bornée** du compresseur (26 et 191
exécutions) ; `MUT_MATERIALISE_WAIT_UNDER_FLAG` est l'attente telle qu'elle était.

### 7.3 Ce que le vérificateur a confirmé

Les règles de la section 6 du document de synchronisation tiennent dans le modèle : les deux
compteurs doivent rester deux (`MUT_WAIT_ON_HARD` est le triangle, trouvé comme violation de
vivacité en 5 exécutions) ; la barrière du compresseur et le regard de l'écrivain après son
annonce sont chacun porteurs ; la forme asymétrique n'est pas sûre sans `membarrier()` ; et le
protocole tel que livré est correct, avec les marques par chunk, à 2 et 3 écrivains, que le chunk
parte comprimé ou matérialisé.

### 7.4 F3 : lectures enchaînées et la release sequence de C++20 (corrigé)

Ayant vu un slot impair, la passe attend qu'il **bouge**, à dessein : attendre qu'il soit pair
bloquerait derrière un lecteur dont les lectures se suivent. La valeur qu'elle lit alors peut être
l'**entrée impaire de la lecture suivante**, une écriture relaxed. En C++20, une écriture release
suivie d'une écriture relaxed du même thread ne forme pas de release sequence (P0982R1 a retiré ce
cas), donc lire l'entrée impaire ne synchronisait avec rien : dans le modèle, la lecture d'élément
précédente et le `free` de la passe n'étaient pas ordonnés, et GenMC signale l'accès
(`MUT_ENTER_RELAXED` à 2 lectures ; à 1 lecture il n'y a pas d'entrée suivante et le trou n'existe
pas). Le raisonnement de `read_reclaim.h` -- le release de sortie « permet au récupérateur de se
fier à une valeur paire » -- était juste pour une valeur paire et muet sur une impaire.

Sur les mappings où la bibliothèque tourne, le trou était refermé de toute façon : sur AArch64
l'écriture de sortie est un `stlr`, celle d'entrée un `str` ultérieur à la même adresse, et qui
observe la seconde observe la première et tout ce que le release a ordonné avant elle ; sur x86
toute écriture est un release. Le code était donc juste sur toutes les cibles de shrink-array, et
faux au regard du standard. Deux façons de le rendre juste au regard du standard, toutes deux
acceptées par le vérificateur : l'écriture impaire d'entrée en release au lieu de relaxed -- un
`stlr` au lieu d'un `str` sur le chemin de lecture d'AArch64 -- ou `MUT_WAIT_FOR_EVEN`, la passe
attendant une valeur paire, gratuit sur le chemin de lecture et le stall que la conception a
refusé. La première a été mesurée sur Grace avec `examples/access_cost`, deux builds du même arbre
alternés sur 5 tours × 5 répétitions (les temps du vector, même code des deux côtés, à 0,6 % près) :
à 1 thread `read-varied` +1,9 % pour un spread de 2-3 %, `read-constant` +0,6 %, `read-random`
+0,0 %, les écritures dans leur spread ; à 64 threads rien de mesurable. **Appliqué** : l'écriture
d'entrée est release, la bibliothèque est juste au regard du standard pour au plus deux pour cent
sur le pire cas de lecture, et `MUT_ENTER_RELAXED` est l'écriture telle qu'elle était.

### 7.5 F4 : le lecteur en débordement annonçait sans sa barrière (corrigé)

Un lecteur au-delà de `CHUNK_READ_RECLAIM_SLOTS` annonce par `g_overflow_readers.fetch_add(seq_cst)`
puis charge `values` ; la passe, après sa barrière, charge le compteur en acquire. Un RMW seq_cst
suivi d'une lecture acquire n'est pas une annonce Dekker en C11 : rien ne met le RMW et la barrière
de la passe dans un ordre unique avec les deux lectures, et le vérificateur trouve l'exécution de
store buffering -- le lecteur charge l'ancien tampon pendant que la passe lit le compteur à zéro et
le libère. Sur AArch64 un `ldaddal` retient un `ldar` ultérieur derrière lui et sur x86 un RMW
verrouillé est une barrière complète, donc, comme F3, le trou est celui du standard ; contrairement
à F3 le correctif est gratuit : `light_fence()` après l'incrément, comme sur le chemin des slots
(une barrière de compilateur sous `membarrier()`, une barrière sinon, sur un chemin qu'un programme
n'emprunte qu'avec plus d'un millier de threads lecteurs). Appliqué ; `MUT_OVERFLOW_NO_FENCE` est la
ligne telle qu'elle était.

## 8. Limites

- **Fidélité.** Le modèle est une réduction écrite à la main depuis le source : une ligne que le
  modèle simplifie est une ligne que le vérificateur ne voit pas. La table de correspondance de la
  section 3 est ce qu'il faut relire quand le source change, et les mutations sont la garde contre
  un modèle qui ne prouve rien.
- **Bornes.** Jusqu'à 5 écrivains (protocole 1 ; 3 pour le protocole 2, chacun avec son flush ;
  3 lecteurs et 3 lectures chacun pour le protocole 3), un chunk, une passe, une écriture chacun
  (5 écrivains avec matérialisation : 9 637 exécutions, 7 s). Un bug qui demande deux
  passes ou un chunk recyclé deux fois est hors champ.
- **`membarrier()`.** Aucun modèle mémoire de langage ne l'a. Le vérificateur vérifie la forme
  symétrique ; l'asymétrique repose sur la section 4.2 et sur le contrat de `membarrier(2)`. GenMC
  traite aussi `atomic_signal_fence()` comme une barrière de thread, donc la barrière de
  compilateur de la forme asymétrique est laissée hors du modèle (l'IR que GenMC explore garde de
  toute façon l'ordre du programme).
- **Modèle, pas matériel.** RC11 et IMM sont le modèle C11 et un raffinement de celui-ci, pas le
  modèle propre d'AArch64. Une exécution qu'ils interdisent est interdite sur tout compilateur et
  tout processeur conformes ; une exécution qu'ils autorisent peut être observable ou non sur
  Grace.

## 9. Prochaines étapes

- Dartagnan sur les trois mêmes fichiers, sous le modèle AArch64 officiel : la dernière limite
  ci-dessus tombe pour la machine que vise la bibliothèque.
