# `assign()` sur le device sans acquire ni release à l'échelle du device

> English version: [gpu_assign.md](gpu_assign.md).

> État : note de conception, novembre 2026, implémentée sur la branche `gpu`
> (`Writer::assign`, `Writer::load`, `Array::finish()`, `device_assign_validation`). Elle
> remplace, pour l'étape 1, la matérialisation de [conception_gpu_fr.md](conception_gpu_fr.md)
> 6.1 et 6.2 (pré-remplissage, publication release, réservation avec loads acquire) par le
> remplissage différé qu'utilise déjà le chemin aligné sur les chunks (conception_gpu_fr.md,
> section 13, « publication relaxed »), avec une réservation qui garde un buffer par chunk (3.3 :
> la première version sans elle ne le faisait pas).

## 1. Le problème

La bibliothèque n'écrit un tableau device que par le chemin aligné : un bloc par appel, une partie
d'un chunk (`Writer::write`). Un noyau de l'application qui écrit `a(i) = v` à des indices
quelconques, un thread par élément, n'a encore rien. Le `assign()` du prototype le faisait avec le
protocole de conception_gpu_fr.md 6.2 :

- un load **acquire** à l'échelle du device du descripteur à chaque écriture ;
- sur un chunk compressé, un buffer **pré-rempli** avec la constante par le warp, puis publié par
  un CAS **release** ; avec la réservation, les autres warps **attendent** la publication.

Sur le GH200, cela coûtait 14x à 20x une écriture plate sur une première écriture ; sur le
MI300A, 300x à 600x, parce qu'un acquire à l'échelle de l'agent invalide le L2 d'un XCD et qu'un
release le réécrit (conception_gpu_fr.md, 13). Le load relaxed seul a ramené l'écriture dans des
chunks matérialisés à 1,34x sur le MI300A, mais la matérialisation est restée à 300x et plus.

## 2. Ce que le chemin aligné a déjà montré

La publication relaxed (conception_gpu_fr.md, 13) a retiré tout acquire et tout release de
l'écriture alignée, et sa première écriture est passée de 15x à 1,6x une écriture plate sur le
MI300A (bibliothèque, S = 1024). Son idée se transpose : **pas de pré-remplissage avant la
publication**. Un buffer est publié vide, marqué frais ; chaque écrivain écrit ses propres
éléments et met leurs bits dans le bitmap `written` du chunk ; après le noyau, `settle_fresh`
écrit la constante là où aucun bit n'est mis. Personne ne lit jamais un élément qu'un autre
thread n'a pas encore écrit : rien n'a à ordonner les stores d'un thread avant ceux d'un autre.
Rien n'attend non plus.

## 3. La proposition

### 3.1 L'écriture

```cpp
__device__ void Writer::assign(std::int64_t i, T v) const
{
    const std::int64_t c = i >> shift;
    const unsigned     e = static_cast<unsigned>(i & mask);
    ChunkDesc<T>&      d = desc[c];
    T* buf = load_ptr(&d.buf);                   // relaxed
    if (buf == nullptr || buf == busy_mark<T>())
    {
        if (same_value(v, d.constant)) return;   // 1. la constante sur son chunk : rien
        buf = materialise(c, d, buf);            // 2. réserver, ou attendre la réservation (3.3)
        if (buf == nullptr) { append(i, v); return; }
    }
    if (is_fresh(buf))                           // 3. frais : l'élément, puis son bit
    {
        untag(buf)[e] = v;
        fetch_or(&written[c * words + e / 32], 1U << (e % 32));
        return;
    }
    buf[e] = v;                                  // 4. un buffer d'un noyau précédent
}
```

`materialise` réserve le chunk par un CAS **relaxed** de `nullptr` vers la marque occupée ; seul
son gagnant prend un buffer (`DeviceArena::take`) et le publie vide, marqué frais, par un échange
relaxed -- ou remet `nullptr` quand l'arène est épuisée. Les autres threads relisent le
descripteur (loads relaxed, une courte pause) jusqu'à la publication (3.3). Ni pré-remplissage,
ni fence. Un buffer pris dans l'arène n'est jamais lu avant que ses éléments soient écrits : son
ancien contenu est sans importance.

Le cas 4 est le chemin chaud : un load relaxed et un store, mesuré à 1,34x (MI300A) et 1,76x
(GH200) une écriture plate dans le prototype, 1,1x à 1,2x avec les loads de descripteurs émis
en avance.

### 3.2 Les lectures dans un noyau d'écriture

`a(i) += v` lit d'abord. `Writer::load(i)` suit les mêmes cas : la constante d'un chunk compressé ;
dans un buffer frais l'élément si son bit est mis, sinon la constante ; dans un buffer plus ancien
l'élément. Un thread lit toujours ce qu'il a écrit lui-même (même thread, même adresse), et ce
qu'un thread de son bloc a écrit avant un `__syncthreads()` (portée du bloc). Y compris une
écriture partie dans l'append-buffer (3.4) : une fois l'arène épuisée dans le noyau, un élément
qui lirait la constante est d'abord cherché dans l'append-buffer, depuis sa fin -- lent, mais
seulement jusqu'à `finish()` ; sans cela, `a(i) += 1` deux fois par un même thread perdait une
mise à jour. Une communication
entre blocs à l'intérieur d'un noyau demande des fences à l'échelle du device dans l'application,
comme sur un tableau plat ; elles ordonnent le descripteur, le bitmap et les éléments de même.

`View` (le chemin de lecture) ne sert qu'aux noyaux de lecture et ne voit jamais de marque
fraîche : `finish()` complète les buffers.

### 3.3 La réservation avant le buffer

La première version prenait le buffer **avant** son CAS, comme la course du prototype, en
comptant qu'une publication perdue coûte peu sans pré-remplissage. Ce n'est pas le coût qui
comptait, mais le nombre : avec des écritures dispersées, quelque 32 threads de warps différents
tombent en même temps sur chaque chunk compressé, chacun prenait un buffer, et une vague prenait
environ 31 buffers de trop par chunk. `device_assign_validation` l'a mesuré : une arène d'un
buffer par chunk (4 096) est passée à 26 817 buffers en 19 arènes (242 000 publications
perdues), la plupart des écritures passaient par l'append-buffer, et la limite de 32 arènes était
proche.

La réservation vient donc d'abord : un CAS de `nullptr` vers la marque occupée, puis seul le
gagnant prend un buffer. Les autres attendent, mais l'attente est courte -- entre son CAS et sa
publication, le gagnant fait une ou deux atomiques sur son shard, sans pré-remplissage -- et
bornée (`claim_spin_max` relectures, puis l'append-buffer) : la borne sert pour une réservation
tenue par une lane du même wavefront AMD sur une branche divergente, qui ne peut pas avancer
pendant que cette lane relit. Mesuré sur la T1000 : les arènes larges ne grandissent pas, aucune
attente n'a atteint la borne sur des millions.

Un chunk réservé lit toujours sa constante (le cas 1 tient sous la marque) ; `load()` le lit comme
compressé. Le chemin aligné attend de même une réservation d'`assign()` du même noyau, et note sa
plage au-delà de la borne.

L'agrégation par warp (une réservation par groupe de lanes sur le même chunk,
conception_gpu_fr.md 6.2 étape 1) est une optimisation : la première écriture en grid-stride est
son cas (toutes les lanes d'un warp attendent la même réservation), mesuré section 5.

**HIP seulement.** Sur CUDA, la réservation se fait par lane (`Writer::materialise_grouped`).
Sous l'ordonnancement indépendant des threads (Volta et suivants), les lanes qui arrivent dans
`assign()` ne sont pas forcément convergentes : celles qui tombent sur un chunk matérialisé
écrivent et passent à leur itération grid-stride suivante pendant que leurs voisines sont encore
dans la réservation groupée. Deux sous-ensembles d'un même warp calculaient alors des masques
`__activemask()` qui se désignaient l'un l'autre, et s'attendaient sans fin, l'un dans un
`__syncwarp`, l'autre dans un shuffle. Sur la RTX A1000 (sm_86, nvcc 13.3), c'est arrivé à chaque
exécution d'une dispersion grid-stride de n/4 indices mélangés dans un tableau de 2^20 éléments
aux chunks en partie matérialisés (`device_assign_validation`, étape 1b), quelle que soit
l'arène. Les validations précédentes ne dispersaient que dans un tableau compressé. En lockstep
(AMD), le masque actif est exactement l'ensemble des lanes de l'appel, et le regroupement reste.
`write_marked` prend le masque complet du warp pour la même raison : tout le bloc l'appelle, et
une divergence de `f` pourrait réduire `active_lanes()`.

Coût sur CUDA (RTX A1000, 0,5 Gio, `device_bench`) : la première écriture grid-stride par
`assign()` passe de 5,0x-5,9x à 7,0x-9,9x l'écriture plate (attentes de réservation 9 M ->
334 M ; ce n'est pas leur compteur : sans lui, les mêmes temps) ; les écritures dispersées ne
changent pas (4,1x-4,5x). Le chemin aligné sur les chunks n'est pas concerné.

GH200, 8 Gio, dans une même session (`fc2801768` contre `adda2ea5e`, `scripts/device_check.sh
cuda`, les six validations device et dix exécutions de `device_assign_validation` passent) : la
première écriture grid-stride passe de 21,0x / 20,3x à 27,4x / 26,8x à S = 256 (`zero_any` /
`zero`) et de 14,1x / 13,8x à 19,9x / 19,3x à S = 1024, soit +30 % / +40 % ; les attentes de
réservation de 0,27 G à 10,3 G (S = 256) et de 0,17 G à 7,7 G (S = 1024), une par élément écrit.
Les écritures dispersées ne changent pas (2,1x-2,3x en première écriture, 1,0x-1,3x dans des chunks
matérialisés). MI300A (`adda2ea5e`, ROCm 10.2, le chemin HIP inchangé hormis `write_marked`) :
validations vertes, aucune attente abandonnée ; écritures dispersées à 1,45x-1,70x et 1,03x-1,07x
comme à 491a05a87 ; la première écriture grid-stride à 74,6x / 48,5x (S = 256 / 1024) contre
57x / 42x alors. Contre `fc2801768` sur le même nœud et le même build (ROCm 10.2, 8 Gio) :
75,1x / 74,8x et 44,0x / 42,8x (`zero_any` / `zero`, S = 256 / 1024) avant, 74,8x / 74,5x et
46,6x / 45,6x à `ece8e73c1`, les mêmes au bruit près. L'écart à 57x à S = 256 précède la
réservation par lane : un commit entre 491a05a87 et fc2801768, à chercher par bissection.

**Le gain du regroupement retrouvé, sans opération collective.** Ce que le regroupement
économisait, c'étaient les CAS de 32 lanes sur un même descripteur (une première écriture
grid-stride : environ un CAS par élément et par lane, contre un par warp), pas les attentes entre
warps, qui restent. Le chemin CUDA laisse maintenant le premier lane actif réserver d'abord ; les
autres lanes, dans un bloc qui vient après, relisent le descripteur et ne réservent qu'un chunk
encore compressé. `active_lanes()` n'y est qu'une indication : aucune synchronisation n'en dépend,
et une indication fausse coûte un CAS, pas une attente. RTX A1000, 0,5 Gio, même session, première
écriture grid-stride à S = 256 (`zero_any` / `zero`) et 1024 : regroupé 22,2 / 20,1 et
23,2 / 22,8 ms ; par lane 31,0 / 29,9 et 30,2 / 27,9 ms ; avec l'indication 22,8 / 21,2 et
23,4 / 23,4 ms. Les écritures dispersées ne changent pas (147-160 ms), le test de stress qui
bloquait le regroupement passe 30 fois sur 30, les validations device et dix exécutions de
`device_assign_validation` passent. Les attentes de réservation restent comptées par lane (environ
une par élément, comme par lane). À mesurer sur le GH200.

### 3.4 Arène épuisée

`DeviceArena::take` renvoie `nullptr` et rien d'autre dans le noyau ne peut faire de place :
l'écriture part dans l'append-buffer (conception_gpu_fr.md 6.4), et `finish()` agrandit l'arène
et le rejoue. Le chemin aligné note une plage et la réécrit avec sa fonction ; `assign()` n'a pas
de fonction à rappeler, il garde donc la valeur.

Un cas est nouveau. Un thread peut ajouter `(i, v1)` pendant que l'arène est épuisée, puis écrire
directement le même élément plus tard dans le noyau (un autre thread a publié un buffer pour ce
chunk entre-temps). Le rejeu ne doit pas écraser la valeur la plus récente par la plus ancienne.
Deux faits rendent la règle simple :

- un thread qui a vu un buffer pour un chunk ne voit plus jamais `nullptr` pour ce chunk dans le
  noyau (cohérence d'une adresse, et les buffers ne sont retirés qu'entre les noyaux) : **ses
  ajouts pour un chunk précèdent tous ses écritures directes dans ce chunk** ;
- le compteur de l'append-buffer donne les ajouts d'un thread dans l'ordre de son programme.

Le rejeu garde donc, pour chaque élément, le dernier ajout (l'indice le plus élevé), et l'omet
quand le bit de l'élément est mis : une écriture directe est venue après. Le rejeu passe avant
`settle_fresh`, qui efface les bits. Deux threads qui écrivent un même élément dans un noyau
forment une course de données de l'application, comme sur un `std::vector` ; la bibliothèque ne
les ordonne pas.

Le même raisonnement couvre l'écriture de la constante du chunk après un ajout de l'élément
(une écriture directe qui ne fait rien) : elle est ajoutée aussi, pour être le dernier ajout. Et
la lecture d'un élément qui lit la constante peut devoir trouver un ajout d'abord. Les deux ne
concernent que les chunks qui ont ajouté dans la phase : `append()` met le bit du chunk dans
`Writer::appended` avant d'écrire son entrée, et l'ajout de la constante comme la recherche dans
l'append-buffer n'ont lieu que sous ce bit (le rejeu efface les bits avec le compteur). Les
conditionner au compteur d'ajouts de la phase, comme d'abord, faisait qu'une seule attente
abandonnée n'importe où transformait chaque écriture d'une constante en ajout et chaque telle
lecture en parcours de tout le buffer : sur des fenêtres de 10^7 éléments (GH200),
l'append-buffer débordait et la réécriture prenait 591x le plat.

### 3.5 La fin de la phase

`finish()` sans fonction (le chemin aligné garde `finish(f)` pour ses plages) :

1. rejouer l'append-buffer s'il n'est pas vide : dédoublonnage sur l'hôte (il est petit :
   l'arène a manqué), agrandissement de l'arène, un noyau de rejeu par `assign()` qui omet les
   éléments dont le bit est mis ; recommencer si le rejeu a de nouveau ajouté ;
2. `settle_fresh` : la constante là où aucun bit n'est mis, bitmaps effacés, marques retirées ;
3. la liste des perdus et les emplacements de réserve rendus à l'arène.

## 4. Pourquoi c'est correct

- **Aucun élément n'est lu avant d'être écrit, sauf comme constante.** L'élément d'un buffer frais
  n'est lu (par `load`, par `settle_fresh`) que là où son bit est mis ; les autres lisent la
  constante, qui ne change pas dans un noyau d'écriture. L'ancien contenu d'un buffer réutilisé
  n'est jamais vu.
- **Aucun store ne peut être écrasé par un remplissage ultérieur.** Le remplissage est
  `settle_fresh`, après le noyau, et seulement là où aucun bit n'est mis. Dans le noyau, personne
  n'écrit un élément sinon ses écrivains.
- **Chaque écriture arrive ou est rejouée.** Le cas 1 ne change rien ; les cas 3 et 4 écrivent ; le
  cas 2 publie, ou trouve le buffer du gagnant, ou ajoute à l'append-buffer ; le rejeu applique
  les ajouts qu'aucune écriture directe ultérieure n'a remplacés (3.4).
- **Toute attente finit.** Le détenteur d'une réservation n'attend rien entre son CAS et sa
  publication ; les relectures sont bornées, et une écriture qui abandonne part dans
  l'append-buffer : deux lanes d'un même wavefront AMD sur des branches divergentes coûtent un
  rejeu, pas un interblocage.
- **Le descripteur et le bitmap ne sont touchés que par des atomiques** (relaxed, à l'échelle du
  device) : ils sont cohérents ; les éléments sont des stores simples de threads distincts sur des
  éléments distincts, ordonnés pour les noyaux suivants par la frontière de noyau.

## 5. Ce que ça devrait coûter, et ce qu'il faut mesurer

| | prototype (GH200 / MI300A, S = 1024) | attendu |
|---|---|---|
| écriture dans un chunk matérialisé | 1,76x / 1,34x (load relaxed) | la même chose |
| première écriture, grid-stride, tous les chunks compressés | 14x / 433x à 541x | proche du chemin aligné (1,6x), plus les publications perdues |
| écriture dans un chunk frais | -- | un or atomique relaxed par élément (par 32 éléments si agrégé) |
| réécritures de plages aléatoires, puis `finish()` + `compress()` | 2,5x / 64x (réservation, load relaxed) | entre les deux ci-dessus |

T1000, 0,5 Gio de doubles, S = 256 / 1024 (`device_bench`, `finish()` compris) : `assign()` dans
des chunks matérialisés 0,91x / 0,72x, `a(i) = load(i) + 1` 1,08x / 1,18x, dispersé dans des
chunks matérialisés 2,24x / 2,32x et en première écriture 4,59x / 4,36x (contre l'écriture
dispersée plate), première écriture en grid-stride 7,31x / 6,22x : toutes les lanes d'un warp y
attendent une même réservation, le cas de l'agrégation par warp.

GH200, 8 Gio, S = 256 / 1024, avant l'agrégation : dans des chunks matérialisés 1,79x / 1,64x,
`+=` 1,18x / 1,13x, dispersé dans des chunks matérialisés 1,33x / 1,12x et en première écriture
2,30x / 2,14x, mais la première écriture en grid-stride 23,7x / 18,9x (la réservation du
prototype : 14x à S = 1024), avec environ une attente par élément écrit. Avec l'agrégation par
warp (les lanes actives sur un même chunk élisent une meneuse, qui réserve ou attend ;
`group_by_key`, `warp.h`), les attentes baissent d'un facteur 35 sur la T1000 et sa première
écriture en grid-stride passe de 7,31x / 6,22x à 4,80x / 5,04x ; les écritures dispersées ne
changent pas.

MI300A, 8 Gio, S = 256 / 1024, avec l'agrégation et la réservation en trois phases (491a05a87,
deux runs à 3 % près) : dans des chunks matérialisés 1,78x / 1,60x, `+=` 0,97x / 0,94x, dispersé
dans des chunks matérialisés 1,08x / 1,05x et en première écriture 1,69x / 1,47x (contre
l'écriture dispersée plate), la première écriture en grid-stride 57x / 42x (la réservation du
prototype : 433x à 541x). Aucune attente abandonnée (0,3 % avant les trois phases) et aucune
croissance d'arène, dans le banc comme dans `device_assign_validation`. La première écriture en
grid-stride reste le cas difficile : environ 3 attentes par chunk à S = 256 et 9 à S = 1024, les
autres wavefronts du chunk attendant la publication de la gagnante, que les atomiques entre XCD
rendent longue ; l'agrégation par warp ne regroupe pas des wavefronts distincts. Les boucles
régulières prennent le chemin aligné (1,75x à S = 1024).

Ajouts à `device_bench` : première écriture en grid-stride par `assign()`, écritures dispersées
(une permutation aléatoire des indices), `+=` sur un tableau matérialisé, réécritures de plages
par `assign()`. Chacun avec et sans agrégation par warp, et avec un or atomique par élément contre
un par 32 lanes quand elles écrivent des éléments consécutifs (un ballot).

## 6. Les alternatives écartées

- **Le protocole du prototype** (load acquire, pré-remplissage, CAS release, réservation) : 300x
  à 600x sur le MI300A.
- **Pas de réservation** (le buffer pris avant le CAS, les perdants mis en réserve) : la première
  version ; elle prenait environ 31 buffers de trop par chunk sous des écritures dispersées
  (3.3).
- **De la mémoire non cachée ou à grain fin pour les buffers**, cohérente entre les XCD sans
  opérations de cache : elle ralentirait chaque lecture et chaque écriture d'un chunk matérialisé,
  le chemin chaud.
- **Une valeur sentinelle au lieu du bitmap** (un élément non écrit contient une valeur réservée) :
  il n'y a pas de valeur réservée pour un `T` quelconque.

## 7. Risques et points ouverts

- **Les atomiques du bitmap sous des écritures dispersées** : un atomique relaxed par élément écrit
  dans un chunk frais. Les GPU exécutent les atomiques sur des adresses distinctes au débit du
  L2 ; à mesurer, contre l'écriture dispersée plate.
- **Les publications perdues de la première écriture** : S/32 warps par chunk prennent chacun un
  buffer et tous sauf un le mettent en réserve. Sans pré-remplissage, le coût est de deux
  atomiques par perdant ; l'arène doit en plus loger les emplacements de réserve (environ 60 Mo de
  buffers de 8 Kio sur un MI300A à pleine occupation).
- **Un noyau sans `finish()`** : une marque fraîche atteindrait un noyau de lecture. Une
  construction de vérification peut l'attraper dans `View` (un pointeur marqué est mal aligné).
- **La taille de l'append-buffer** : un débordement est une erreur (conception_gpu_fr.md 6.4).
  Valeur par défaut à discuter : une fraction de l'arène, ou de n.
- **Décisions prises** (24/11/2026) : la réservation du prototype (pré-remplissage, loads
  acquire) et la course restent hors de la bibliothèque, comme références dans `gpu/proto` ; la
  réservation de 3.3 les remplace. L'agrégation par warp après les mesures. L'append-buffer
  contient par défaut 1 % des éléments de l'arène (4 096 au moins), réglable à la construction.

## 8. Plan

1. `Writer::assign`, `Writer::load`, `materialise` sans pré-remplissage, les emplacements de
   réserve, l'append-buffer et `finish()` avec son rejeu (3.4, 3.5), dans `chunk/device/`.
2. `device_assign_validation` contre un miroir hôte : écritures dispersées depuis de nombreux
   threads, premières écritures en grid-stride, `+=` par le thread propriétaire, un thread qui
   écrit deux fois un élément, une petite arène (ajouts puis écritures directes des mêmes
   éléments, le rejeu), deux noyaux avant un seul `finish()`, `assign()` et écritures alignées dans
   des noyaux distincts, deux `assign()` sur des branches divergentes d'un même wavefront.
3. Les cas de `device_bench` de la section 5, sur la T1000, le GH200 et le MI300A.
4. Ensuite l'agrégation par warp et les loads de descripteurs groupés, si les mesures le
   demandent.
