# `assign()` sur le device sans acquire ni release à l'échelle du device

> English version: [gpu_assign.md](gpu_assign.md).

> État : note de conception, novembre 2026, soumise à relecture avant tout code. Elle remplace,
> pour l'étape 1, la matérialisation de [conception_gpu_fr.md](conception_gpu_fr.md) 6.1 et 6.2
> (pré-remplissage, publication release, réservation) par le remplissage différé qu'utilise déjà
> le chemin aligné sur les chunks (conception_gpu_fr.md, section 13, « publication relaxed »).

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
    if (buf == nullptr)
    {
        if (same_value(v, d.constant)) return;   // 1. la constante sur son chunk : rien
        buf = materialise(c, d);                 // 2. un buffer frais, ou nullptr (arène épuisée)
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

`materialise` relit le descripteur (un chunk publié entre-temps ne coûte rien), prend un buffer
(l'emplacement de réserve du warp, sinon l'arène : `DeviceArena::take`) et le publie marqué frais
par un CAS **relaxed** depuis `nullptr`. Le perdant garde son buffer dans son emplacement de
réserve et écrit dans celui du gagnant. Ni pré-remplissage, ni réservation, ni attente, ni
fence. Un buffer pris dans l'arène n'est jamais lu avant que ses éléments soient écrits : son
ancien contenu est sans importance.

Le cas 4 est le chemin chaud : un load relaxed et un store, mesuré à 1,34x (MI300A) et 1,76x
(GH200) une écriture plate dans le prototype, 1,1x à 1,2x avec les loads de descripteurs émis
en avance.

### 3.2 Les lectures dans un noyau d'écriture

`a(i) += v` lit d'abord. `Writer::load(i)` suit les mêmes cas : la constante d'un chunk compressé ;
dans un buffer frais l'élément si son bit est mis, sinon la constante ; dans un buffer plus ancien
l'élément. Un thread lit toujours ce qu'il a écrit lui-même (même thread, même adresse), et ce
qu'un thread de son bloc a écrit avant un `__syncthreads()` (portée du bloc). Une communication
entre blocs à l'intérieur d'un noyau demande des fences à l'échelle du device dans l'application,
comme sur un tableau plat ; elles ordonnent le descripteur, le bitmap et les éléments de même.

`View` (le chemin de lecture) ne sert qu'aux noyaux de lecture et ne voit jamais de marque
fraîche : `finish()` complète les buffers.

### 3.3 Les publications perdues

Sur une première écriture en grid-stride, les S/32 warps (S/64 wavefronts) qui partagent un chunk
le trouvent tous compressé en même temps. Sans pré-remplissage, une publication perdue coûte une
prise et un échange dans l'emplacement de réserve du warp, que sa matérialisation suivante
réutilise (le mécanisme du prototype). La pression sur l'arène est bornée : un buffer par chunk
matérialisé, plus un par warp résident dans les emplacements de réserve, plus la liste des
perdus quand un emplacement est déjà occupé (rendue à `finish()`).

L'agrégation par warp (une matérialisation par groupe de lanes sur le même chunk,
conception_gpu_fr.md 6.2 étape 1) est une optimisation, pas une nécessité : elle est mesurée, pas
supposée (section 5).

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
- **Rien n'attend.** Pas de boucle d'attente, donc pas d'interblocage entre lanes d'un même
  wavefront AMD sur des branches divergentes (le cas pour lequel la borne de la réservation
  existait).
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

Ajouts à `device_bench` : première écriture en grid-stride par `assign()`, écritures dispersées
(une permutation aléatoire des indices), `+=` sur un tableau matérialisé, réécritures de plages
par `assign()`. Chacun avec et sans agrégation par warp, et avec un or atomique par élément contre
un par 32 lanes quand elles écrivent des éléments consécutifs (un ballot).

## 6. Les alternatives écartées

- **Le protocole du prototype** (load acquire, pré-remplissage, CAS release, réservation) : 300x
  à 600x sur le MI300A.
- **La réservation sans pré-remplissage** (une marque occupée, les autres attendent le buffer) :
  l'attente demeure, et avec elle le cas du wavefront divergent et sa borne ; le remplissage
  différé rend la réservation inutile.
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
- **Décisions demandées** : retirer de la bibliothèque les variantes réservation et course (le
  prototype les garde comme références) ; l'agrégation par warp dans la première version ou après
  les mesures ; la taille par défaut de l'append-buffer.

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
