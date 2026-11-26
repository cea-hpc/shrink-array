# Portage GPU : premier document de conception

> English version: [gpu_design.md](gpu_design.md).

> État : écrit en octobre 2026 avant tout prototype ; révisé en novembre 2026 après le
> prototype de l'étape 0 ([gpu/proto](../gpu/proto/README.md) : RTX A1000 du portable, puis
> GH200). Ce document fixe ce qu'un backend GPU de shrink-array devrait être, ce qu'il reprend
> de la version CPU et ce qu'il abandonne, et le plan pour trancher par la mesure. La section 13
> donne les chiffres de l'étape 0, portable, GH200 et MI300A. Les décisions
> marquées **(tranché par le prototype)** étaient ouvertes dans la première version. L'étape 1
> a commencé (fin de la section 13).

***

## 1. Objectif et périmètre

**Objectif.** Offrir sur GPU (NVIDIA et AMD) un tableau qui se lit et s'écrit dans un kernel
comme un tableau plat, `a(i)` / `a(i) = v`, et qui ne stocke pas ses chunks uniformes --
le même format de données et les mêmes garanties (sans perte, écriture en place, toute valeur
répétée compressée par `zero_any`) que la version CPU. Le gain visé est d'abord la **mémoire**
: 80 à 192 Go de HBM, c'est là que « 20 fois moins » change ce qu'un code peut traiter.

**Ce qui est dans le périmètre de cette première conception :**

- les trois variantes `zero`, `zero_or_value`, `zero_any`, et `pattern` (section 8) ;
- géométrie fixe (S constante de compilation) d'abord, dynamique ensuite (section 9.3) ;
- un modèle d'exécution **par phases**, aligné sur les kernels (section 3) ;
- l'interopérabilité avec la version CPU : même format de chunk, transferts dans les deux
  sens, mode hybride sur les machines à mémoire cohérente (section 11).

**Ce qui n'est pas dans le périmètre :**

- faire tourner le code CPU existant sur GPU : la machinerie d'exécution (section 2) n'est
  pas portable et n'a pas à l'être ;
- les écritures concurrentes à une compression **à l'intérieur** d'un kernel ;
- `bool` compacté bit à bit (`BitArray`) ;
- un backend SYCL/Intel (possible via la couche de portabilité, non visé).

## 2. Ce qui se porte, ce qui ne se porte pas

La bibliothèque CPU a deux couches de natures très différentes :

| couche | contenu | sur GPU |
|---|---|---|
| **format** | table de chunks ; un chunk = tampon privé ou constante ; S puissance de 2 ou multiple de 64 ; indice `i >> shift`, `i & modulo` ; métadonnées par chunk | **se porte tel quel**, et s'y trouve même mieux qu'en CPU (section 4) |
| **sémantique d'accès** | lecture d'un chunk compressé = la constante ; écrire la constante d'un chunk compressé ne fait rien ; écrire autre chose matérialise, pré-rempli de la constante quittée ; compression des chunks redevenus uniformes ; rechunk par `copy_range` | **se porte** : c'est la définition du conteneur |
| **machinerie d'exécution** | Dekker asymétrique et `membarrier`, écritures différées par thread, barrière de lecture et dirty bitset, hazard pointers, OMPT, `thread_local`, `fast_allocators` (slabs par rang, arènes `mmap`), listes de réutilisation par thread, adaptateur TBB, registre des secondes valeurs sans verrou | **ne se porte pas**, et est remplacée par le modèle de la section 3 |

Les raisons, précises, pour lesquelles la troisième couche ne se porte pas :

- **pas de garantie de progression entre blocs** : un thread GPU qui attend un autre warp peut
  attendre un warp qui ne sera jamais ordonnancé tant qu'il occupe la place. Toute attente
  bornée inter-threads (`wait_out_compressor`, l'attente du compresseur sur les drapeaux)
  est interdite dans un kernel ;
- **pas d'équivalent de `membarrier`** (pas d'interruption inter-processeurs côté device) ;
  la forme symétrique à `seq_cst` existerait, mais sans garantie de progression elle ne sert à rien ;
- `std::atomic`, `thread_local`, `mmap`, les appels système n'existent pas en code device
  (nvcc) ; les atomiques device ont des **portées** (bloc, device, système) et, sous Kokkos,
  sont *relaxed* par définition ;
- les modèles GenMC de `formal/` vérifient le modèle C11 ; les modèles mémoire à portées
  (PTX, HSA) n'ont pas de vérificateur pratique. Un protocole fin à acquire/release dans un
  kernel ne pourrait pas être « vérifié plutôt que testé », ce qui est contraire à la règle de
  la bibliothèque.

## 3. Modèle d'exécution : un rôle par kernel

Les trois protocoles d'échange de la version CPU existent parce qu'une région OpenMP fait
tourner **en même temps** des écrivains, une passe de compression et des flushs, plus des
lecteurs purs contre le réemploi. Sur GPU l'exclusion est fournie par la **frontière de
kernel** : un kernel se termine, `fence`, le suivant commence. La conception repose sur un
seul invariant :

> **Invariant.** Pendant un kernel, les chunks d'un tableau ne sont touchés que par **un
> rôle** : lecteurs, ou écrivains, ou la passe de compression, ou la reconstruction du
> rechunk. Aucun tampon n'est repris pendant un kernel d'écriture ou de lecture.

| phase | rôle | conflit possible | règlement |
|---|---|---|---|
| kernel utilisateur en lecture | lecteurs | aucun | -- |
| kernel utilisateur en écriture | écrivains | plusieurs warps matérialisent le même chunk | une réservation du descripteur (CAS vers une marque « occupé »), un seul pré-remplissage, la publication ; les autres l'attendent (6.2) |
| `compress()` (fin de phase) | la passe | aucun : ni écrivain ni lecteur | vote par bloc + compaction (7) |
| `rechunk_to()` | hôte + kernels | aucun | séquence de phases (9.3) |

Conséquences :

- **personne n'attend en tenant une réservation** : la première version disait « personne
  n'attend jamais personne », avec une course au CAS où chaque warp pré-remplit son propre
  tampon et où le perdant lit celui du gagnant. Le prototype a mesuré cette course à 5,5× et
  25× une écriture plate lors d'une première écriture (6.2), et l'a remplacée par une
  réservation : les warps qui trouvent un chunk réservé attendent sa publication. L'attente
  se termine parce que le détenteur est résident et n'attend rien entre sa réservation et sa
  publication ; un warp publie ses propres réservations avant qu'aucun de ses lanes n'attende
  (6.2) ;
- la compression n'est plus « en vol » mais **en fin de phase** : la crête mémoire d'une
  phase est bornée par ce que cette phase matérialise (section 7.3 pour les atténuations) ;
- l'API de phases existe déjà côté CPU (`parallel_phase`, `begin_parallel_phase`,
  `finalize_parallel_phase`) ; côté GPU, la phase est le kernel, et `finalize` est
  `compress()` après le `fence` (section 12).

Le mot « phase » recouvre un kernel ou une suite de kernels de même rôle : plusieurs kernels
de lecture, ou plusieurs kernels d'écriture, peuvent s'enchaîner sans compression entre eux.

## 4. Le format device

### 4.1 Le descripteur de chunk

Côté CPU, un chunk est un pointeur atomique `values` (plus les portes de lecture/écriture
en OpenMP), qui pointe vers un tampon privé ou vers un tampon constant partagé. Sur GPU la
**constante est rangée dans le descripteur** : il n'y a plus de tampon constant, ni de
registre de secondes valeurs sur le chemin chaud.

```cpp
template <typename T>
struct ChunkDesc                 // 16 octets pour T de 8 octets au plus
{
    T*       buf;                // nullptr : compressé ; sinon tampon de S éléments
    T        constant;                // la constante du chunk compressé (ou quittée : 6.2)
    // zero : constant est toujours T{} ; zero_or_value : T{} ou la seconde valeur du tableau ;
    // zero_any : n'importe quelle valeur.  pattern : voir 8.4.
};
```

Pour `T` de 16 octets (`std::complex<double>`) le descripteur fait 24 octets, alignés à 32
**(à trancher : 32 octets alignés, ou pointeur tagué + table latérale ; mesurer)**. Le CAS de
matérialisation porte sur `buf` seul (8 octets) : `constant` n'est écrit que par la passe de
compression (section 7), jamais par un écrivain, donc jamais en concurrence avec un lecteur.

Le descripteur est un objet **trivialement copiable**, sans atomique dans son type : les
accès atomiques sont faits par `atomic_ref` de portée device sur `buf` aux seuls endroits qui
le changent. Il est donc partageable avec l'hôte (section 11).

### 4.2 Géométrie et alignement

S est une puissance de 2, **multiple de 64** (la taille d'un wavefront AMD ; 32 chez NVIDIA).
Les threads consécutifs d'un warp lisent des indices consécutifs : avec S ≥ 64 et des
tampons alignés, un warp tombe toujours **dans un seul chunk**. Trois propriétés en
découlent, qui rendent le format mieux adapté au GPU qu'au CPU :

- le descripteur est lu **une fois par warp** (même adresse pour tous les lanes : broadcast
  L1), pas une fois par thread ;
- le test `buf == nullptr` est **uniforme par warp** : aucune divergence ;
- les accès au tampon sont **coalescés** comme sur un tableau plat.

Taille de départ : 256 ou 1024 **(à trancher)**. Le GPU voudrait des chunks plus gros pour
amortir le descripteur et les atomiques (une matérialisation par chunk), la compression fine
les voudrait petits ; c'est le même arbitrage que la géométrie dynamique CPU (9.3).

### 4.3 Métadonnées par tableau

| donnée | taille | où |
|---|---|---|
| `ChunkDesc[n_chunks]` | 16 ou 24 o par chunk | device (ou mémoire cohérente, 11) |
| compteur de bump de l'arène, tête de pile des tampons libres | 2 mots | device |
| append-buffer des écritures non appliquées (6.4) | borné, réglé à la création | device |
| statistiques (`stats()`, chunks matérialisés, échecs d'arène) | quelques mots | device, relus par l'hôte en fin de phase |

Il n'y a **pas** de dirty bitset, pas de liste de candidats, pas de ligne « déjà en liste » :
la passe de compression regarde tous les chunks matérialisés (7.1), ce qui sur GPU coûte
moins qu'entretenir des listes.

## 5. La lecture

```cpp
__device__ T operator()(IndexT i) const noexcept
{
    const auto& d = desc_[i >> shift_];
    return d.buf ? d.buf[i & mask_] : d.constant;     // branchement uniforme par warp
}
```

Aucune barrière de lecture : il n'y a pas d'écriture différée à voir, puisque dans une phase
de lecture il n'y a pas d'écrivain, et que `compress()` et l'application de l'append-buffer
(6.4) sont terminés avant la phase.

Attente : 1,1 à 1,5 fois le temps d'un tableau plat sur un kernel limité par la bande
passante (une lecture de descripteur par warp, latence masquée par l'occupation), contre 1,9
mesuré côté CPU (`examples/access_cost`). **À mesurer en premier** (section 13) : c'est le
chiffre qui décide de tout.

## 6. L'écriture

### 6.1 Le proxy device

Comme en CPU, `a(i) = v` passe par un proxy qui connaît le tableau, le chunk et la position,
et qui **voit la valeur avant de décider**. Le proxy device est réduit à trois cas :

```cpp
__device__ void assign(T v)
{
    ChunkDesc<T>& d = desc_[nc];
    T* buf = atomic_load_acquire(d.buf);          // portée device ; marque « occupé » : 6.2
    if (buf == nullptr)
    {
        if (v == d.constant) return;                   // 1. la constante sur sa constante : rien
        buf = materialise(d);                     // 2. matérialiser (6.2), puis écrire
    }
    buf[pos] = v;                                 // 3. chunk matérialisé : écriture directe
}
```

Les cas 1 et 3 sont le chemin chaud, uniformes par warp, sans atomique de lecture-écriture.
La lecture de `buf` est un **acquire** (la première esquisse avait une lecture relaxed) :
sans lui, un écrivain peut écrire avant que le pré-remplissage du tampon qu'il voit soit
arrivé, et être écrasé par lui. Son coût ne se mesure pas (écriture dans un chunk
matérialisé : 0,82× et 1,08× une écriture plate). Les règles de
`zero_or_value` (« la seconde valeur sur un élément qui la contient déjà ») et de `zero_any`
(« découverte », « départ », `may_become_uniform`) servaient à alimenter les listes de candidats
: elles **disparaissent**, la passe de fin de phase n'a pas besoin de candidats. Il reste à
prendre en compte `+=` et les autres opérateurs composés, qui lisent avant d'écrire : même
cas, avec `v` calculée à partir de `d.constant` ou `buf[pos]`.

### 6.2 La matérialisation

1. **Agrégation par warp.** Les lanes du warp dont le chunk est le même (en pratique tous :
   4.2) élisent un lane (`__match_any_sync` / `__ballot`, `__activemask`) ; les autres
   attendent le résultat de l'élu par un shuffle. Une matérialisation par warp et par chunk,
   pas par thread.
2. **Allocation** : le lane élu prend un tampon dans la **pile des tampons libres** (rendus
   par les compressions précédentes, 7.2), sinon par **bump** sur l'arène du tableau
   (`atomicAdd` sur le compteur, un seul par tampon). Si l'arène est épuisée : 6.4.
3. **Pré-remplissage** du tampon avec `d.constant` : par **tout le warp** (S/32 éléments par lane,
   coalescé), pas par le seul élu.
4. **Publication par CAS** de `nullptr` → `buf` sur `d.buf`, portée device, `release` après
   le pré-remplissage (une `__threadfence()` avant le CAS, ou `atomic_ref::compare_exchange`
   avec `memory_order_release`). Le perdant **ne réessaie pas** : il lit le `buf` gagnant
   (`acquire`) et garde son tampon pour plus tard (6.3). Avec la réservation décrite plus
   bas, la publication est un store release, puisque le CAS de la réservation a déjà élu
   l'écrivain.
5. Tous les lanes écrivent leur valeur dans `buf`.

Pourquoi le pré-remplissage avant le CAS est sûr : tant que le CAS n'a pas eu lieu, le
tampon n'est visible de personne ; après, tout écrivain qui le voit a lu `buf` en `acquire`
et voit le pré-remplissage complet. Deux warps qui matérialisent le même chunk pré-remplissent
chacun leur tampon ; un seul est publié. Un lecteur n'existe pas pendant cette phase.

Ce sont exactement les mécanismes des tables de hachage GPU (allocation par bump, insertion
par CAS, agrégation par warp) : ils sont connus et portables sur CUDA et HIP.

**Ce que le prototype a trouvé : un chunk s'étend sur S/32 warps** (S/64 wavefronts).
L'agrégation par warp donne une matérialisation par warp, pas par chunk. Sur une première
écriture en grid-stride, 5,5 (S = 256) et 22,8 (S = 1024) warps par chunk perdent le CAS, chacun
après avoir pré-rempli un tampon entier pour rien : la première écriture coûte 5,5× et 25× une
écriture plate. Leurs tampons épuisaient en plus une arène dimensionnée à un tampon par chunk.
D'où une coordination par chunk, entre warps **(tranché par le prototype)** :

- **Réservation** (le mode par défaut du proxy) : le lane élu installe une **marque
  « occupé »** dans `d.buf` (une adresse mal alignée, jamais un tampon) par CAS depuis
  `nullptr`. Le gagnant prend un tampon, son groupe le pré-remplit, puis il le publie (store
  release). Un lane qui trouve la marque attend la publication, avec des pauses. On obtient un
  pré-remplissage par chunk, aucun tampon perdu, et une arène d'un tampon par chunk
  matérialisé. Première écriture : 2,2× (S = 256) et 3,4× (S = 1024). Le cas 1 de 6.1 reste
  valable sous la marque : la constante ne change pas avant la publication, et le
  pré-remplissage l'écrit.
- **Publier avant d'attendre.** Un warp à cheval sur deux chunks (une plage non alignée) a deux
  groupes. Dans la première version, un groupe tenait une réservation pendant que l'autre
  attendait. Avec la reconvergence SIMT, le détenteur pouvait rester bloqué derrière son voisin
  en attente, et deux warps dans cet état s'attendaient l'un l'autre. Mesuré : des attentes que
  seule la borne terminait, et une réécriture de plages 80× plus lente. Le warp passe
  maintenant par trois phases. Chaque groupe fait d'abord une tentative sans attendre. Les
  groupes qui ont réservé pré-remplissent et publient. Ce n'est qu'ensuite que les autres
  attendent, sans jamais réserver : une réservation relâchée parce que l'arène est épuisée
  envoie l'écriture dans l'append-buffer.
- **Une attente bornée.** Au-delà d'un nombre de pauses, l'écriture part dans l'append-buffer
  (6.4). La borne couvre un cas : deux appels à `assign()` sur des branches divergentes d'un
  même wavefront AMD, qui n'a pas d'ordonnancement indépendant des threads. Ce cas coûte un
  rejeu, pas un interblocage. Sur le MI300A, aucune attente n'a été abandonnée dans
  `proto_test` ni dans le banc ; un test de ce cas précis reste à écrire.
- **La course** (le schéma des étapes 1 à 4) reste dans le prototype comme référence. Ses
  perdants gardent leur tampon dans un **emplacement de réserve** par warp (un échange
  atomique), pour leur matérialisation suivante ; `compress()` reprend les réserves.
- **Les écritures alignées sur les chunks** (7.3, atténuation 2) évitent la question pour les
  boucles de la bibliothèque : un bloc par chunk.

### 6.3 Pile des tampons libres

Une pile LIFO de pointeurs, tête atomique, **sans ABA** parce qu'aucun pop ne se fait pendant
qu'un push est possible : les push n'ont lieu qu'en `compress()` (7.2). **(Tranché par le
prototype)** : aucun push pendant un kernel d'écriture. Avec la réservation (6.2), aucun
tampon n'est perdu : le bump ou le pop que fait une réservation sert toujours. Un tampon pris
pour rien arrive encore avec la course de référence, et quand deux blocs alignés partagent un
chunk (7.3). Il attend dans l'emplacement de réserve de son warp, que la matérialisation
suivante de ce warp réutilise. Un emplacement déjà occupé envoie son tampon dans une liste de
tampons perdus. Les emplacements et la liste sont rendus par `compress()`.

### 6.4 Épuisement de l'arène : l'append-buffer

Quand le bump dépasse l'arène, l'écriture n'est pas perdue : le lane écrit `(i, v)` dans un
**append-buffer** par tableau (`atomicAdd` sur son compteur, taille fixée à la création) et
lève un drapeau d'échec. C'est l'analogue GPU des écritures différées de la version CPU, à
une différence près : il n'y a **rien** qui les applique pendant le kernel. À la fin de la
phase, l'hôte lit le drapeau, fait `compress()`, **agrandit l'arène** (9.1), puis lance le
kernel d'application de l'append-buffer : une écriture par entrée, par le même proxy, avec
une arène qui suffit désormais.

Si l'append-buffer lui-même déborde, l'écriture est perdue et le tableau le sait : c'est une
**erreur** signalée par `finalize()`, pas un comportement silencieux. La taille de
l'append-buffer est un réglage du tableau (défaut proposé : 1 % de la taille de l'arène).

Deux écritures du même élément par deux threads dans un kernel sont une course de données
dans le programme utilisateur, sur GPU comme sur un `std::vector` : la bibliothèque ne les
ordonne pas, y compris entre une écriture directe et une entrée de l'append-buffer.

## 7. La compression de fin de phase

### 7.1 La passe

`compress()` est lancé par l'hôte après la phase d'écriture (après le `fence`). Un **bloc par
chunk matérialisé** (ou un warp par chunk pour S ≤ 1024, **à trancher par la mesure** ; le bloc
par chunk du prototype atteint 73 % (S = 256) et 99 % (S = 1024) de la bande passante de
lecture en flux quand tous les chunks se replient) :

1. chaque lane lit ses S/32 éléments et compare à `buf[0]` ;
2. vote `__all_sync` (réduction `&&` dans le bloc) ;
3. si uniforme à une valeur que la variante sait compresser (`zero` : `T{}` ; `zero_or_value`
   : `T{}` ou la seconde valeur ; `zero_any` : toute valeur) : `d.constant = buf[0]`, `d.buf =
   nullptr`, et le tampon est rendu (7.2).

La passe est bornée par la bande passante : relire tous les tampons matérialisés d'un tableau
de 10 Go prend quelques millisecondes à 3 To/s. Elle **remplace** à la fois la compression en
vol, les listes de candidats, les compteurs par chunk des builds série et `shrink_to_fit()`.

Pour ne pas relire ce qui n'a pas changé, un **bit « touché » par chunk** (levé en `relaxed`
par le premier écrivain du chunk dans la phase, remis à zéro par la passe) peut restreindre la
passe aux chunks écrits pendant la phase. Ce bit n'est pas un dirty bitset au sens CPU (il
ne conditionne aucune lecture) ; **à trancher** par la mesure sur un banc où peu de chunks
changent par phase.

### 7.2 Rendre les tampons

Les tampons des chunks compressés sont poussés sur la pile libre par un `atomicExch` de tête
par tampon, ou mieux, **compactés** d'abord (`cub::DeviceSelect` / `hipcub`) puis ajoutés en un
bloc. Un tampon rendu est réutilisé par la matérialisation suivante (6.2, étape 2) ; l'arène
ne grandit donc que si le nombre de chunks matérialisés **simultanément** grandit.

### 7.3 La crête mémoire d'une phase, et ses atténuations

Sans compression en vol, une phase qui écrit `second_value` sur tout un tableau à `T{}`
matérialise **tout** avant que `compress()` ne replie tout. Trois atténuations, par ordre
de préférence :

1. **Les kernels sont courts.** Un code GPU découpe déjà son travail en kernels ; la
   granularité de phase est le kernel, pas une région OpenMP de 2 000 itérations. C'est à
   l'utilisateur de mettre `compress()` là où il mettrait un `fence` de toute façon.
2. **Vote avant matérialisation.** Quand un bloc couvre un chunk entier (le cas des boucles
   régulières : S = 256, blocs de 256 threads alignés sur les chunks, ce que `parallel_for`
   de la section 12 garantit quand il peut), le bloc vote « toutes mes valeurs sont égales à
   `v` » **avant** de matérialiser : si oui, il écrit `d.constant = v` et rien d'autre. C'est la
   compression en vol, sans concurrence, parce que le bloc est le seul écrivain du chunk.
   Cette optimisation ne vaut que pour les kernels dont le découpage est fait par la
   bibliothèque ; un kernel utilisateur à indices arbitraires ne l'a pas.

   Le prototype (`block_write`, `parallel_for_chunks`) fait plus que le vote. Le bloc calcule
   ses valeurs une seule fois, vote avec `__syncthreads_or` si l'une d'elles diffère de la
   constante, et prend un seul tampon. Il écrit ensuite les valeurs, et la constante autour,
   en une passe : les éléments écrits ne sont jamais pré-remplis. Le CAS reste, pour deux plages
   qui partagent un chunk. Une première écriture coûte 1,9× (S = 256) et 1,2× (S = 1024) une
   écriture plate, et une réécriture de plages aléatoires suivie de `finalize()` coûte 1,9× à
   S = 1024. À S = 256, un bloc de 256 threads par chunk de 2 Kio paie ses barrières, et la
   réservation fait aussi bien. **Pour la façade** : le `parallel_for` Kokkos ou Arcane sur un
   tableau prend ce chemin, et les kernels quelconques passent par le proxy avec la
   réservation.
3. **La pression mémoire force la phase.** Quand l'arène est épuisée, l'append-buffer (6.4)
   termine la phase de fait : l'hôte compresse, agrandit, rejoue. L'arène est donc aussi un
   **budget** : la fixer petite force des compressions fréquentes, au prix de rejeux.

## 8. Les variantes

### 8.1 `zero`

`constant` vaut toujours `T{}` ; la passe ne replie qu'à `T{}`. Rien d'autre.

Implémenté (étape 1) comme `device::Array<T, S, Variant::zero>` : le descripteur est le seul
pointeur (8 octets, la taille du chunk de la variante zero sur l'hôte ; 16 pour un double
`zero_any`), si bien qu'un chunk compressé lit `T{}` sans charger de constante. Un état hôte
converti vers elle (`assign()`) donne un buffer à un chunk compressé vers une autre valeur.

### 8.2 `zero_or_value`

`constant` vaut `T{}` ou la seconde valeur du tableau (un champ du tableau, copié dans le
descripteur par la passe). `set_second_value()` est une opération d'hôte entre deux phases.
Sur GPU, `zero_or_value` n'a plus d'intérêt propre par rapport à `zero_any` (le registre
qui le justifiait n'existe plus) : il est conservé pour l'interopérabilité et la sémantique
(ne compresser qu'à une valeur déclarée), **(à trancher)** s'il mérite un code séparé ou un
`if constexpr` dans `zero_any`.

Tranché : un `if constexpr`, `Variant::zero_or_value`. Son descripteur est celui de
`zero_any`, `{buf, constant}` ; la seconde valeur est un champ de l'`Array`
(`set_second_value()`, `second_value()`, opérations d'hôte entre deux phases ; T{} tant
qu'aucune n'est déclarée ; une constante de construction autre que T{} est aussi la seconde
valeur), copiée dans le `Writer` (`Writer::second`) et passée à la passe.
`folds_to<V>(value, second)` (`chunk_desc.h`) est la seule différence : la passe et l'écriture
d'un chunk entier (`Writer::fold_whole`) replient vers T{} ou la seconde valeur, jamais vers une
autre. Écrire sa propre constante dans un chunk compressé ne coûte toujours rien, quelle que
soit cette constante. Un chunk compressé à une ancienne seconde valeur continue de la lire,
comme sur l'hôte (son bloc du registre ne bouge jamais). `assign(state)` prend la seconde
valeur de `state` s'il en déclare une et matérialise un chunk compressé à une valeur qui n'est
ni T{} ni la seconde, comme `import_array()` le remplit sur un `zero_or_value` hôte ;
`image()` déclare la seconde valeur sauf si c'est T{}. `device_write_validation` passe les
rondes de `zero` en `zero_or_value` (seconde valeur 2,5, puis 1,5 à partir de la ronde 4 ; les
chunks à 3,5 de la ronde 3 ne doivent pas se replier), et `device_convert_validation` son
aller-retour (2,5 déclaré après `assign()`). `kokkos_shrink_bench ... zero_or_value` déclare
les secondes valeurs du banc CPU (1, 1, 1, 2, 100).

### 8.3 `zero_any`

Le descripteur porte la constante : c'est la forme naturelle de `zero_any`, et la variante à
prototyper en premier. Ce qui disparaît : le registre des secondes valeurs
(`second_value_registry.h`), ses segments, son index haché, `reclaim_second_values()`, les lignes
`array_type_const_val` et `array_second_value`, `CHUNK_SECOND_VALUE_INDEXED`. Le coût par chunk
redevient uniforme : le descripteur.

### 8.4 `pattern`

Un chunk motif = un motif partagé (S entiers) + un offset. Le descripteur gagne une forme :
`buf` pointe vers le motif, le **bit de poids faible de `buf`** (ou un champ `kind`) dit
« motif », et `constant` sert d'offset :

```cpp
return is_motif(d) ? motif(d)[i & mask_] + d.offset : (d.buf ? d.buf[i & mask_] : d.constant);
```

La lecture reste uniforme par warp. **Écrire** dans un chunk motif le matérialise, pré-rempli
de `motif[j] + offset` (le pré-remplissage de 6.2 avec un calcul au lieu d'une copie). La
**détection** d'un motif se fait dans `compress()` : un chunk matérialisé non uniforme est
comparé à chaque motif connu à un offset près (`buf[j] - motif[j]` constant) ; le registre
des motifs est entretenu par l'**hôte** entre les phases (un motif est admis à la deuxième
occurrence, comme en CPU), avec une recherche device sur une petite table de motifs
candidats. **(À trancher)** après le prototype `zero_any` : le coût de la détection par
rapport au gain sur une connectivité.

## 9. Mémoire

### 9.1 Arènes

Un tableau a **une arène** de tampons de S éléments (allocation device en un bloc, `cudaMalloc`
/ `hipMalloc`, ou via le `MemorySpace` de Kokkos), un compteur de bump et une pile libre.
L'agrandissement se fait par l'hôte entre deux phases : une nouvelle arène est ajoutée à une
**liste d'arènes** (les tampons existants ne bougent pas, les pointeurs des descripteurs
restent valides), jamais par réallocation-copie. La taille initiale est un réglage du tableau
(défaut proposé : le nombre de chunks que la variante CPU matérialise sur le même banc,
sinon 10 % des chunks).

Aucun `fast_allocators` : pas de slab par rang, pas d'arène par type. Un allocateur device
dans le kernel (`malloc` device) est exclu : trop lent, non borné.

### 9.2 Libération

Un tableau rend ses arènes à sa destruction, sur l'hôte. Il n'y a pas de pool global de
tampons entre tableaux dans cette première conception **(à trancher plus tard)** : les
arènes d'un tableau détruit ne servent pas à un autre, sauf à ajouter une couche de pool
d'hôte, triviale puisque hors kernel.

### 9.3 Géométrie dynamique et rechunk

Le rechunk CPU (advisor, decider, reconstruction par `copy_range` en tranches, affinité de
dalle) se transpose en une séquence de phases hôte : `compress()` ; statistiques par chunk
(compte des chunks matérialisés et des sous-blocs uniformes à la taille cible : un kernel de
réduction, le même que la passe) ; décision sur l'hôte ; reconstruction par un kernel de
`copy_range` vers un nouveau tableau (un bloc par chunk cible, qui vote l'uniformité de ses
sources et n'alloue qu'en cas contraire : l'« installation directe des constantes » de
`zero_any`) ; échange ; libération des anciennes arènes. Rien de concurrent : c'est la partie
de la version CPU qui se porte le mieux, parce qu'elle était déjà faite de phases.

Elle vient **après** la géométrie fixe dans le plan (section 13).

## 10. Modèle mémoire et règles de code device

- **Aucune attente** inter-threads dans un kernel : ni spin, ni attente bornée. Un CAS perdu
  se résout par lecture du gagnant, jamais par réessai.
- Les seuls atomiques : `atomicAdd` (bump, append-buffer, compteurs), `atomicCAS` sur
  `d.buf`, `atomicExch` de tête de pile, un `atomicOr` du bit « touché ». Tous de **portée
  device** (`cuda::atomic_ref<…, cuda::thread_scope_device>` ; `__hip_atomic_*` avec
  `__HIP_MEMORY_SCOPE_AGENT`).
- Un seul ordonnancement non relaxed : `release` sur la publication du tampon pré-rempli,
  `acquire` sur sa lecture par le perdant. Tout le reste est `relaxed`, et les frontières de
  phase sont des `fence` d'hôte (`cudaDeviceSynchronize` / stream sync / `Kokkos::fence()`).
- Sous Kokkos, où les atomiques sont *relaxed*, la paire release/acquire est remplacée par
  `Kokkos::memory_fence()` avant le CAS et après la lecture du gagnant ; c'est le seul endroit
  où le backend Kokkos et le backend natif diffèrent.
- Pas de `std::atomic`, pas de `thread_local`, pas d'exception, pas d'allocation dans le
  code device. Le code device est en-tête seulement (`__host__ __device__` /
  `KOKKOS_INLINE_FUNCTION`), C++20, sans module.
- Pas de modèle GenMC pour le device : les protocoles sont réduits à « un rôle par kernel »
  plus un CAS de publication, dont l'argument tient en un paragraphe (6.2). Si un protocole
  plus fin apparaît nécessaire, c'est le signe que la conception a dévié.

## 11. Interopérabilité avec la version CPU

- **Format commun** : le descripteur device `{buf, constant}` est aussi un format CPU valide, à
  deux différences près : la version CPU pointe sur un tampon constant partagé là où le
  device met `nullptr`, et porte les portes OpenMP dans l'objet chunk. Un **convertisseur**
  dans les deux sens est simple : par chunk, `nullptr` ↔ `first_value_buffer_` / tampon du registre, et
  les tampons matérialisés sont copiés tels quels (même S, même disposition). Il tient en un
  kernel plus une boucle d'hôte, et fait le transfert hôte ↔ device **sans décompresser** :
  on ne transfère que les tampons matérialisés et la table des descripteurs.
- **Machines à mémoire cohérente** (Grace-Hopper par NVLink-C2C, MI300A) : la table des
  descripteurs et les arènes peuvent vivre en mémoire **partagée** (`cudaMallocManaged`,
  mémoire système avec ATS, ou HBM unifiée du MI300A). L'hôte fait alors l'orchestration
  (compression de fin de phase sur CPU, si le GPU est occupé ; rechunk ; registres de
  `pattern`), et le GPU les lectures et écritures en masse. La version CPU, en **mode
  phase** (sans compression en vol : la variante série, ou une région dont aucune passe ne se
  déclenche), peut lire et écrire le même tableau entre deux kernels. C'est un objectif de
  deuxième étape, mais il justifie dès maintenant que le descripteur soit trivialement
  copiable et sans atomique dans son type.
- Le cluster Grace actuel (sans GPU) sert aux mesures CPU et au convertisseur ; les mesures
  device demandent un H100 ou GH200 côté NVIDIA, un MI250X ou MI300A côté AMD.

## 12. Couche de portabilité et API

### 12.1 Langages

- **CUDA** : NVIDIA seulement. **HIP** : AMD, et NVIDIA par délégation à nvcc ; API quasi
  1:1 avec CUDA. **Kokkos** : bibliothèque C++ de portabilité (pas un langage), backends CUDA,
  HIP, SYCL, OpenMP, Serial ; atomiques *relaxed* ; n'expose pas les intrinsèques de warp
  (`__match_any_sync`, `__ballot`) de façon portable, mais les laisse accessibles dans des
  îlots `#if defined(KOKKOS_ENABLE_CUDA)` / `KOKKOS_ENABLE_HIP`.
- **Décision proposée** : le cœur (sections 5 à 7) est écrit une fois en **C++ device
  commun** -- un petit en-tête d'abstraction `chunk/device/intrinsics.h` qui mappe une
  dizaine de primitives (atomiques à portée, ballot/match/shuffle, fence, `all_sync`) sur
  CUDA et sur HIP, à la manière de ce que fait `fast_allocators` pour les plateformes CPU.
  Le **prototype est en HIP** (compile pour AMD et NVIDIA) ou en CUDA, au choix de la machine
  disponible. **Kokkos est une façade** livrée en plus (`chunk/device/kokkos_adaptor.h`,
  comme `tbb_adaptor.h`) : un équivalent de `Kokkos::View` dont `operator()` est le proxy de la
  section 6, et un `parallel_for` qui marque les phases. Cette façade est ce qu'utilisent
  les codes Kokkos ; les codes CUDA/HIP natifs utilisent les en-têtes natifs.
- Compilateurs : nvcc ≥ 12 (C++20 ; pas C++23, pas de modules ; lent sur le CRTP lourd) ;
  hipcc/clang (C++20 et 23). La branche `cpp26` n'est pas une base GPU ; la cible C++20 des
  applications est la bonne. **(À trancher)** : clang comme compilateur CUDA unique, qui
  évite nvcc et ses limites, au prix d'une prise en charge de CUDA en retard d'une version.
- CMake : une option `CHUNK_DEVICE=none|cuda|hip` (et `CHUNK_DEVICE_KOKKOS=ON` pour la
  façade), des cibles `shrink_array::zero_any_device_256`, … ; Spack : une variante `device`.

### 12.2 Esquisse d'API

```cpp
namespace sa = shrink_array;

sa::zero_any::DeviceArray<double> a(n, {.chunk = 256, .arena_chunks = n / 256 / 10});

// phase d'écriture : un kernel utilisateur, ou le parallel_for de la bibliothèque (qui
// aligne ses blocs sur les chunks et obtient le vote avant matérialisation, 7.3)
sa::device::parallel_for(0, n, [=] __device__ (long i) { a(i) = field(i); });

a.finalize();        // fence ; drapeau d'échec d'arène ? → compress, grow, rejouer l'append-buffer
                     // puis compress() : vote par bloc, compaction, tampons rendus

// phase de lecture : n'importe quel kernel, aucune barrière
reduce_kernel<<<…>>>(a.view(), out);

a.stats();           // % de chunks matérialisés, arène, échecs
auto h = a.to_host();                 // zero_any::Array<double> CPU, sans décompression
sa::zero_any::DeviceArray<double> b = sa::device::from_host(h);
```

`a.view()` est un objet trivialement copiable (descripteurs, décalage, masque, compteurs)
passé par valeur aux kernels ; le proxy d'écriture ne contient que `view`, `nc`, `pos`.
`finalize()` est l'équivalent de `finalize_parallel_phase()` et de `shrink_to_fit()` réunis.
Un kernel d'écriture lancé sans `finalize()` avant un kernel de lecture viole l'invariant de
la section 3 : c'est le seul piège d'usage, et il est documenté comme la règle des threads
l'est côté CPU (`docs/getting_started_fr.md` §5).

## 13. Plan par étapes et critères

| étape | contenu | critère de passage |
|---|---|---|
| **0. Prototype** (1 à 2 semaines) | hors arbre, HIP ou CUDA, `zero_any` seul, S = 256 et 1024 fixes, trois kernels (lecture ; écriture avec bump + agrégation par warp + CAS ; `compress()` par vote + compaction CUB/hipCUB) ; banc : lecture streaming, réécriture par plages aléatoires comme `shrink_bench`, contre un tableau plat | lecture ≤ 1,5× le tableau plat ; écriture dans un chunk matérialisé ≤ 2× ; `compress()` de 10 Go matérialisés ≤ 10 ms. **Si la lecture dépasse 2×, la conception est à revoir avant d'aller plus loin** |
| 1. Cœur dans l'arbre | `chunk/public/chunk/device/` : descripteur, view, proxy, arène, append-buffer, `compress()`, `intrinsics.h` CUDA + HIP ; `zero_any` et `zero` ; convertisseur hôte ↔ device ; tests de validation contre la version CPU sur les mêmes séquences d'écritures ; CI sur un runner GPU si disponible, sinon compilation seule | les validations passent à l'identique CPU/GPU ; même mémoire finale que la variante CPU à géométrie égale |
| 2. Phases et banc | `parallel_for` aligné et vote avant matérialisation ; `shrink_bench` device (phases du banc contre `DeviceArray`) ; mesures H100 / MI250X ou MI300A ; documentation (ce document et sa version anglaise mis à jour avec les chiffres) | chiffres publiés dans `variantes_fr.md` |
| 3. Façade Kokkos | `kokkos_adaptor.h`, `parallel_for` Kokkos marquant les phases, `memory_fence` à la place de release/acquire | un exemple Kokkos ; mêmes validations |
| 4. `zero_or_value`, `pattern` | 8.2, 8.4 | validations ; gain mesuré de `pattern` sur une connectivité |
| 5. Géométrie dynamique | 9.3 | `rechunk_validation` device |
| 6. Hybride | mémoire cohérente (GH200, MI300A), mode phase de la version CPU | un tableau lu et écrit alternativement par CPU et GPU sans copie |

**Étape 0, faite sur le GPU du portable (RTX A1000, sm_86, novembre 2026)**, 1 Gio de
doubles, rapports au tableau plat (détails et bench dans [gpu/proto](../gpu/proto/README.md)) :

| | S = 256 | S = 1024 | critère |
|---|---|---|---|
| lecture, tous les chunks matérialisés | 1,04× | 1,23× | ≤ 1,5× : tenu |
| écriture dans des chunks matérialisés | 0,82× | 1,08× | ≤ 2× : tenu |
| `compress()` de 10 Go, transposé à 3,35 To/s | 4,1 ms | 3,0 ms | ≤ 10 ms : tenu |
| première écriture, réservation / alignée (course) | 2,2× / 1,9× (5,5×) | 3,4× / 1,2× (25×) | -- |
| réécriture de plages + `finalize()`, réservation / alignée (course) | 3,7× / 4,2× (4,4×) | 2,1× / 1,9× (4,1×) | -- |

Les critères tiennent sur le portable, sur le GH200 et sur le MI300A (ci-dessous ; sur le
MI300A, l'écriture avec le load relaxed seulement). Le prototype n'utilise ni
CUB ni hipCUB : `compress()` lance un bloc par chunk avec un vote `__syncthreads_and`, et les
tampons retournent à la pile libre sans passe de compaction.

**GH200 (H100, 132 SMs, HBM3 à 3,7 To/s mesurés), 8 Gio de doubles**, premier passage (le
prototype de 91d7af328) :

| | S = 256 | S = 1024 | critère |
|---|---|---|---|
| lecture, tous les chunks matérialisés | 1,04× | 1,04× | ≤ 1,5× : tenu |
| écriture dans des chunks matérialisés (lecture acquire) | 1,46× | 1,75× | ≤ 2× : tenu, mais de peu |
| `compress()`, tous les chunks se replient : 10 Go extrapolés depuis 8 Gio | 7,7 ms | 4,6 ms | ≤ 10 ms : tenu |
| première écriture, course / réservation / alignée | 15,6× / 17,7× / 5,1× | 21,8× / 14,4× / 1,8× | -- |
| réécriture de plages + `finalize()`, course / réservation / alignée | 7,5× / 6,9× / 6,1× | 5,4× / 3,4× / 2,7× | -- |

Ce que la grosse carte montre, et que le portable cachait derrière ses 170 Go/s :

- **Le chemin d'écriture est limité par une seule lecture de descripteur en vol par warp.**
  Un store dans un tampon peut toucher les descripteurs (aliasing) : le compilateur ne peut
  donc pas lancer la lecture du descripteur de l'écriture suivante avant le store courant, et
  chaque warp attend environ 1 µs par écriture. Un second passage a mesuré les lectures
  acquire, relaxed et simple (`WriteLoad`) à égalité (1,48× à 1,52×) : la barrière n'était pas
  en cause. Le chemin aligné, qui lit le descripteur une fois par bloc, coûte 1,12× à
  S = 1024 ; le bench mesure aussi quatre descripteurs lus d'avance par thread (ce que peut
  faire une boucle de façade).
- **Une matérialisation est une chaîne d'allers-retours, sur deux compteurs globaux** (le
  sommet de la pile libre et le bump), payée à chaque pas du grid-stride : la réservation ne
  fait pas mieux que la course à S = 256. Deux raccourcis essayés entre-temps ont aggravé les
  choses et ont été annulés. Le premier jetait un coup d'œil à la pile libre avant son
  atomique ; or, dans le bench, la première écriture suit un repli, donc la pile est pleine et
  le coup d'œil ajoute un aller-retour. Le second faisait une seule barrière device par le
  leader après la barrière de warp, au lieu d'une par lane, si bien que les barrières
  s'exécutaient l'une après l'autre. Première écriture : 5,1× → 9,0× (S = 256), 1,8× → 2,5×
  (S = 1024). Les compteurs sont maintenant **partitionnés** : chaque arène est découpée en
  128 shards, chacun avec son bump et son segment de pile libre. Un chunk alloue dans son shard
  d'origine (son indice modulo 128), puis dans les suivants, et un tampon revient au shard de
  son adresse.
- **`compress()`** : un warp par chunk est maintenant le défaut (GH200 : 83 % de la bande
  passante en flux à S = 1024, contre 66 % avec un bloc par chunk). À S = 256, les deux
  n'atteignent que 36 à 39 % quand tous les chunks se replient : le coût est l'empilement sur
  la pile libre de chaque repli, sur un seul compteur (un scan sans repli prend 0,96 ms contre
  6,4 ms), que les shards répartissent.
- **S = 1024 est la géométrie GPU** : égale ou meilleure sur toutes les lignes sauf deux (la
  première écriture en course, et `assign()` dans des chunks matérialisés).
- **Un bug du chemin de course, trouvé en route** : chaque lane quittait la boucle de
  regroupement dès que son groupe était traité, si bien que les shuffles des tours suivants
  s'exécutaient avec des masques changeants pendant que d'autres lanes étaient encore dans le
  travail de leur groupe. Avec clang-CUDA et l'allocation partitionnée, plus lente, certaines
  lanes pré-remplissaient alors le mauvais tampon, et des tampons publiés gardaient des
  éléments jamais écrits (un passage de test sur deux). Le regroupement est maintenant
  uniforme sur le warp, comme l'était déjà celui de la réservation (`group_by_chunk`), et
  16 passages sur 16 réussissent.

**Troisième passage GH200** (eb29b07df : 128 shards, shard d'origine = indice du chunk modulo
128), 8 Gio :

| | S = 256 | S = 1024 |
|---|---|---|
| écriture dans des chunks matérialisés, acquire / 4 lectures d'avance | 1,85× / 1,24× | 1,79× / 1,21× |
| écriture dans des chunks matérialisés, alignée | 1,71× | 1,43× |
| première écriture, course / réservation / alignée | 15,1× / 14,8× / 5,0× | 22,1× / 14,3× / 2,3× |
| réécriture de plages + `finalize()`, course / réservation / alignée | 5,9× / 4,5× / 4,5× | 5,1× / 2,7× / 2,4× |
| `compress()` par warp, tous les chunks se replient | 62 % | 67 % |

- Quatre lectures de descripteur d'avance ramènent l'écriture à 1,2× : l'explication par une
  seule lecture en vol tient, et la boucle de la façade lira ses descripteurs par lots.
- Les shards ont aidé là où les compteurs étaient disputés (replis à S = 256 : 36 % → 62 % ;
  réécritures de plages en réservation). En revanche, l'écriture dans des chunks matérialisés
  (1,48× → 1,85×), les chemins alignés et les replis à S = 1024 (83 % → 67 %) ont ralenti. Ce
  n'est pas la place des tampons : un shard d'origine par série de 1, 64 ou 1024 chunks a donné
  les mêmes mesures. Les séries de 64 concentraient de plus les allocations sur un shard de
  16 tampons : 9 fois plus de CAS perdus, et une arène pourtant large épuisée. Option
  abandonnée. Ce n'est pas non plus le nombre de registres. Le chemin lent, compilé en ligne,
  faisait bien passer le kernel d'écriture de 32 à 38 registres (48 warps résidents par SM au
  lieu de 64). Mais imposer 32 registres avec `__launch_bounds__(256, 8)` a donné 1,92×, et le
  kernel à quatre lectures d'avance, qui débordait alors ses pointeurs en mémoire, est passé
  de 1,24× à 2,46× ; annulé. Le 1,5× à 1,9× d'`assign()` dans un chunk matérialisé bouge avec
  des changements de code sans rapport (1,75× à S = 1024 au premier passage). Plutôt que
  d'autres suppositions, l'étape suivante est un profil Nsight Compute du kernel d'écriture
  (occupation, raisons d'attente). Ce qui est solide : quatre lectures de descripteur d'avance
  par thread donnent 1,2×, et les boucles de la façade seront écrites ainsi.
- La première écriture via `assign()` reste entre 15× et 22× : son coût est la chaîne de
  latence d'une matérialisation plus que les compteurs ; le chemin aligné reste la réponse de
  la façade.

**Profil Nsight Compute sur le GH200** (S = 256, 1 Gio, nvcc 12.4) :

| kernel | registres, occupation (théorique / atteinte) | débit HBM | attente principale |
|---|---|---|---|
| `write_ramp<Flat>` | 32, 100 % / 82 % | 90 % | file mémoire pleine (limité par la bande passante) |
| `write_ramp<View>`, chunks matérialisés | 38, 75 % / 66 % | 46 % | long scoreboard 69 % (attente de la lecture du descripteur) |
| `write_ramp<View>`, première écriture (matérialise) | 38, 75 % / 74 % | 7 % | barrière mémoire 46 %, puis long scoreboard 31 % ; 26 threads actifs par warp |

- L'écriture dans un chunk matérialisé est limitée par la latence de la lecture de son
  descripteur (une seule en vol par warp), l'occupation limitée par les registres venant en
  second. Les quatre lectures d'avance (1,2×) traitent la première cause. Les boucles de la
  façade liront leurs descripteurs par lots, et `assign()` lui-même ne sera pas réglé
  davantage.
- La première écriture attend les barrières du pré-remplissage (un `__threadfence` par lane).
  La publication passe maintenant par un seul CAS (échange) release du lane qui publie, après
  la barrière de warp ou de bloc (`PROTO_PUBLISH=1`, le défaut ; `0` garde une barrière par
  lane, pour comparer). L'essai précédent, avec une barrière par le leader, avait été mesuré
  avec le coup d'œil à la pile libre et annulé avec lui : c'est sa première mesure seule.

**Publication, même session** (8 Gio) : un seul CAS release au lieu d'une barrière par lane
fait passer la première écriture alignée de 4,95× à 4,59× (S = 256) et de 2,32× à 2,17×
(S = 1024), et ses réécritures de plages de 4,51× à 4,28× et de 2,40× à 2,35×. C'est neutre sur
`assign()` (course et réservation à 2 % près). Gardé par défaut.

**Étape 0 côté NVIDIA : terminée.** Les trois critères tiennent sur le GH200 (lecture à
1,04×, écriture dans un chunk matérialisé sous 2×, `compress()` de 10 Go en 4,4 à 4,8 ms). Les
conclusions qui passent à l'étape 1 :

- S = 1024 est la géométrie GPU.
- Les boucles de la façade prennent le chemin aligné (première écriture à 2,2×, réécritures de
  plages avec `finalize()` à 2,4× à S = 1024) et lisent leurs descripteurs par lots (1,2× pour
  l'écriture dans des chunks matérialisés).
- `assign()` dans les kernels quelconques utilise la réservation, avec un regroupement
  uniforme sur le warp. C'est un repli correct (première écriture à 14×, réécritures de plages
  à 2,7× à S = 1024), pas un chemin rapide.
- Compteurs d'arène partitionnés par arène (replis à S = 256 : 36 % → 62 %) ; `compress()`
  avec un warp par chunk.

**MI300A** (24/11/2026, gfx942, 228 CU en six XCD, HBM3, 8 Gio de doubles, 57b489d70).
`proto_test` passe tous les cas : wavefronts de 64, claim (aucune attente abandonnée), loads
relaxed et plain, chemin aligné sur les chunks, rejeux.

| | S = 256 | S = 1024 | critère |
|---|---|---|---|
| lecture, tous les chunks matérialisés | 1,47x | 1,45x | <= 1,5x : tenu, de justesse |
| écriture dans des chunks matérialisés, load acquire / relaxed | 66x / 1,40x | 64x / 1,34x | <= 2x : tenu avec le load relaxed seulement |
| écriture dans des chunks matérialisés, 4 loads relaxed d'avance / alignée | 1,11x / 16,7x | 1,09x / 4,2x | -- |
| `compress()` par warp, tous les chunks se replient : 10 Go à 3,35 To/s | 3,9 ms | 2,8 ms | <= 10 ms : tenu |
| première écriture, race / claim, load acquire | 366x / 504x | 393x / 603x | -- |
| première écriture, race / claim, load relaxed | 306x / 436x | 323x / 538x | -- |
| première écriture, alignée | 48x | 15x | -- |
| réécritures de plages + `finalize()`, race / claim, load acquire | 113x / 136x | 103x / 136x | -- |
| réécritures de plages + `finalize()`, race / claim, load relaxed | 64x / 84x | 39x / 64x | -- |
| réécritures de plages + `finalize()`, alignée | 37x | 14x | -- |

- **Un acquire ou un release à l'échelle du device est une opération de cache, pas
  l'ordonnancement d'une adresse.** Le L2 du gfx942 appartient à un XCD et n'est pas cohérent
  avec les cinq autres : un acquire l'invalide (`buffer_inv sc1`), un release réécrit ses
  lignes modifiées (`buffer_wbl2 sc1`). L'assembleur des deux programmes en contient (88 et
  196 occurrences, `-save-temps`). Leur coût est par opération, quelle que soit l'adresse : un
  acquire par bloc et par chunk, sans release, rend déjà l'écriture alignée dans des chunks
  matérialisés 16,7x (S = 256, quatre fois plus de chunks) et 4,2x (S = 1024) ; la première
  écriture ajoute le release de chaque publication (48x et 15x, contre 5,0x et 2,3x sur le
  GH200).
- **Le load relaxed est le chemin d'écriture sur AMD** (1,34x à 1,40x, 1,1x avec quatre loads
  d'avance), et `proto_test` le valide sur le MI300A. Il ne sauve pas la première écriture par
  `assign()` (-15 %) : ce coût est dans la matérialisation, ses acquires, ses attentes et son
  release.
- **Race sur AMD** : ~12 CAS perdus par chunk à S = 1024 (5 sur le GH200) ont épuisé une arène
  large ; le premier passage a perdu des écritures sur un append-buffer plein (le test, qui
  s'arrêtait à la première erreur, fait maintenant tourner chaque cas et donne n entrées aux
  cas larges).

Étape 0 sur le MI300A : les trois critères tiennent, l'écriture avec le load relaxed ; la
matérialisation non, 15x au mieux. Ouvert pour l'étape 1, à mesurer avant d'écrire l'arène
device : pas d'acquire à l'échelle du device sur le chemin d'écriture (une frontière de noyau
ordonne déjà ce que les noyaux précédents ont publié ; seuls les chunks publiés dans le noyau
en cours demandent plus), et pas de release par matérialisation -- par exemple un chemin aligné
qui publie par un CAS relaxed et qui, au lieu d'écrire dans un buffer qu'un autre bloc du même
noyau a publié, passe par l'append-buffer (rare sur des plages alignées), ou des publications
groupées par bloc.

**Publication relaxed sur le MI300A** (e1b224191, `set_relaxed_publish(true)` : voir
[gpu/proto](../gpu/proto/README.md), « Coordination per chunk »). `proto_test` passe, ses trois
cas relaxed compris. Le chemin aligné, avant -> après :

| | S = 256 | S = 1024 | GH200, acquire/release (S = 256 / 1024) |
|---|---|---|---|
| première écriture | 47x -> 13,9x | 15x -> 3,4x | 5,0x / 2,3x |
| écriture dans des chunks matérialisés | 16,6x -> 1,53x | 4,2x -> 0,79x | 1,71x / 1,43x |
| réécritures de plages + `finalize()` | 36x -> 7,6x | 13,6x -> 3,0x | 4,5x / 2,4x |

Sans acquire ni release dans le noyau, le MI300A à S = 1024 arrive à 1,5x du GH200 avec.

**Et sur le GH200** (même commit ; `proto_test` passe), avant -> après : première écriture
4,72x -> 4,20x (S = 256) et 2,17x -> 1,83x (S = 1024) ; écriture dans des chunks matérialisés
1,83x -> 1,81x et 1,12x -> 1,16x ; réécritures de plages + `finalize()` 5,23x -> 5,05x et
2,09x -> 2,21x. Elle ne coûte rien sur NVIDIA et gagne sur la première écriture : **la
publication relaxed est le chemin aligné de l'étape 1, sur les deux constructeurs (tranché par
le prototype).**

**Étape 1, commencée (novembre 2026).** Les fonctions que partagent les versions hôte et device
sont dans la bibliothèque, compilées pour les deux :

- `CHUNK_HD` (`chunk/internal/host_device.h`) vaut `__host__ __device__` dans une unité de
  traduction CUDA ou HIP, et rien sinon. Il marque la géométrie (`StaticChunkSize`,
  `ChunkElems`, `log2int`), l'identité des valeurs (`same_value`, `identity_less`,
  `ValueIdentity::same`, parties de `std::complex` et de `std::array` comprises) et le test
  d'uniformité d'un chunk, `internal::walk_constant` (`chunk/internal/constant_walk.h`), qu'appellent
  les parcours de compression du CPU et qu'appellera le `compress()` device.
- `device_core_validation` (`-DCHUNK_DEVICE_CORE_CHECK=ON`) les exécute dans un noyau CUDA et
  les compare à l'hôte, zéros signés et charges utiles de NaN compris. Il compile avec
  `-Werror=all-warnings` sous nvcc, qui ne fait qu'avertir quand une telle fonction en appelle
  une de l'hôte. Il passe avec nvcc 13.1 (T1000, sm_75), sur le GH200 (sm_90) et avec clang 23.

Dans `chunk/public/chunk/device/`, à partir du prototype (`gpu/proto/zero_any_device.h`) :

- `chunk_desc.h`, `view.h` : le descripteur de chunk et le chemin de lecture
  (`device_core_validation`).
- `backend.h`, `atomics.h` : CUDA ou HIP, choisi par le compilateur ; les appels hôte (allouer,
  copier, synchroniser, avec une exception en cas d'échec) et les atomiques device, toutes
  relaxed (13, MI300A).
- `arena.h` : `DeviceArena`, les arènes de buffers à 128 shards chacune (un bump et un segment
  de pile libre par shard), où puisent les noyaux d'écriture, que la passe de compression
  réalimente, et qui grandissent entre les phases. `device_arena_validation` vérifie les prises
  concurrentes (chaque buffer une fois, épuisement, aucun recouvrement), les restitutions, la
  croissance et les shards d'origine ; il se compile en CUDA ou en HIP
  (`-DCHUNK_DEVICE_LANGUAGE=HIP`) et passe sur la T1000, le GH200 et le MI300A.

- `warp.h`, `block_write.h`, `array.h` : l'écriture alignée sur les chunks avec la publication
  relaxed (13). `Writer::write` (un bloc par appel, une partie d'un chunk) : descripteur lu en
  relaxed, vote, un buffer pris dans l'arène et publié par un CAS relaxed, marqué frais quand
  le bloc n'écrit qu'une partie du chunk (`is_fresh`, `chunk_desc.h`) ; chaque bloc note ses
  éléments dans le bitmap `written` du chunk, et `settle_fresh` met la constante dans les
  autres après le noyau. `device::Array` possède les descripteurs, l'arène et les listes :
  `parallel_for_chunks(lo, hi, f)`, ou `writer()` dans un noyau de l'appelant et `finish(f)`
  après. Quand l'arène est épuisée, un bloc n'écrit rien dans ce chunk et note sa plage ;
  `finish` agrandit l'arène et réécrit la plage avec le même f (pas d'append-buffer : f doit
  être une fonction de i). Un CAS perdu range son buffer dans une liste rendue à `finish`.
  `device_write_validation` le compare à un miroir hôte (plages partagées par deux blocs, deux
  noyaux avant un seul `finish`, écritures de la constante seule, petites arènes, chunks de fin,
  constante non nulle), et échoue si le remplissage ou la réécriture des plages ratées est cassé.
  Il passe sur la T1000, le GH200 et le MI300A.

- `compress.h` : la passe de compression, `Array::compress()`. Un warp par chunk, en
  grid-stride ; un chunk matérialisé dont les éléments sous n sont une seule valeur
  (`same_value`) la prend pour constante et rend son buffer à l'arène (après `normalise()`, sans
  prise en même temps) ; la fin du dernier chunk n'est pas comparée. `device_write_validation`
  la lance après chaque tour contre un modèle hôte (replis comptés, éléments, chunks
  matérialisés et buffers en usage vérifiés à nouveau), et échoue si la fin du dernier chunk
  est comparée ou si un buffer n'est pas rendu. Elle passe sur la T1000, le GH200 et le MI300A.

- La conversion hôte <-> device : `Array::assign(state)` charge un `ArrayState` (l'`export_array()`
  d'un tableau hôte, `chunk_exchange.h`), `Array::image()` en rend un pour `import_array()`.
  Seuls les chunks matérialisés voyagent (un motif est déplié), par lots de 64 Mio ; un chunk
  compressé se réduit à son descripteur. nvcc ne compile pas les en-têtes des tableaux hôte (il
  rejette leurs membres statiques inline de templates, section 14) : un programme convertit donc
  en deux unités de compilation, `export_array()` / `import_array()` dans une unité C++,
  `assign()` / `image()` dans l'unité device, l'état passant de l'une à l'autre. Les types de
  l'état sont passés dans `chunk/internal/chunk_state.h`, que les deux incluent (l'interface de
  `chunk_exchange.h` ne change pas). `device_convert_validation` est construit ainsi : un
  `zero_any::Array` hôte avec des chunks à la première valeur, à une seconde valeur et
  matérialisés part sur le device, y est écrit et compressé (dix chunks se replient sur 2,5,
  de nouveau une seconde valeur côté hôte), et revient égal à un miroir hôte. Il passe sur la
  T1000, le GH200 et le MI300A. En HIP, les deux unités doivent utiliser le même libstdc++ : le clang de
  ROCm choisit son propre GCC (gcc-toolset-12 sur le nœud du MI300A, contre GCC 14.1 pour
  l'unité C++, et l'édition de liens échouait) ; la construction passe donc
  `-DCMAKE_HIP_FLAGS=--gcc-install-dir=<celui du compilateur C++>`, sinon la configuration
  saute le test en le disant.

- `examples/device_bench` : les mesures du banc du prototype sur `device::Array` (lecture,
  écriture alignée, réécritures de plages, `compress()`, conversion), contre un tableau plat,
  pour vérifier le portage. T1000 (0,5 Gio, S = 256 / 1024) : lecture 1,02x / 0,96x,
  première écriture 1,59x / 0,86x, écriture dans des chunks matérialisés 0,92x / 0,80x,
  réécritures de plages + `compress()` 4,8x / 2,0x, `compress()` de 10 Go 2,5 / 2,4 ms quand
  tout se replie ; `assign()` et `image()` 3 Go/s (mémoire hôte paginée, une boucle de copie
  élément par élément : pas optimisé).
  Sur les gros GPU (8 Gio de doubles), bibliothèque contre prototype (publication relaxed) :

  | S = 1024 | GH200 | GH200 proto | MI300A | MI300A proto |
  |---|---|---|---|---|
  | lecture, tous les chunks matérialisés | 1,04x | 1,03x | 1,32x | 1,45x |
  | première écriture alignée (+ `finish()`) | 1,56x | 1,83x | 1,60x | 3,4x |
  | écriture dans des chunks matérialisés | 1,17x | 1,16x | 1,21x | 0,79x |
  | réécritures de plages + `compress()` | 2,41x | 2,21x | 3,31x | 2,99x |
  | `compress()` de 10 Go, tout se replie | 3,5 ms | 3,5 ms | 3,7 ms | 2,9 ms |

  À S = 256 : lecture 1,04x / 1,33x, première écriture 3,50x / 4,33x, écriture dans des chunks
  matérialisés 1,67x / 2,02x, `compress()` 4,0 / 4,7 ms (GH200 / MI300A). Le portage n'a rien
  perdu sur le GH200 ; sur le MI300A la première écriture est deux fois plus rapide que celle du
  prototype (qui comptait chaque matérialisation sur un compteur global) et l'écriture dans des
  chunks matérialisés plus lente (1,21x contre 0,79x à S = 1024, 2,02x à S = 256 : au
  critère), à examiner. La conversion : 12 à 18 Go/s sur le GH200 (pas optimisée).

- `Writer::assign` / `load`, `Array::finish()` : les écritures à indices quelconques sans acquire
  ni release à l'échelle du device, conçues et mesurées dans [gpu_assign_fr.md](gpu_assign_fr.md) ;
  `device_assign_validation` passe sur la T1000 et le MI300A. L'agrégation par warp des
  réservations a suivi (gpu_assign_fr.md, section 5 : sur le MI300A aucune attente abandonnée,
  écritures dispersées 1,05x à 1,69x l'écriture dispersée plate, première écriture en
  grid-stride 57x / 42x à S = 256 / 1024).

  Nouveau passage du banc complet sur le MI300A (491a05a87, 8 Gio) : par rapport au tableau du
  24/11/2026, la première écriture alignée 5,07x / 1,75x (contre 4,33x / 1,60x) et l'écriture
  dans des chunks matérialisés 2,15x / 1,27x (contre 2,02x / 1,21x ; au-delà du critère à
  S = 256), la lecture 1,33x / 1,31x, `compress()` de 10 Go 4,7 / 3,6 ms. La perte de 6 % à
  17 % du chemin aligné, mesurée contre 407674ad5 (avant `assign()`) sur un même créneau, est en
  partie l'écart du nœud d'un jour à l'autre (407674ad5 lui-même : 4,46x / 1,64x, et 2,08x pour
  l'écriture dans des chunks matérialisés à S = 256, déjà au-delà du critère) ; le reste était
  réel sur les écritures qui matérialisent, +13 % / +7 % sur la première écriture, une atomique
  device de plus par prise (l'emplacement de réserve essayé avant l'arène). Avec l'arène d'abord
  (fb14a408d) : 9,64 / 3,47 ms contre 9,41 / 3,44 ms avant `assign()`, les réécritures de plages
  égales ; l'écriture dans des chunks matérialisés garde +2 % à +3,5 % (le contrôle d'une
  réservation `assign()`). La conversion : 3,5 Go/s pour `assign()`, 1,5 Go/s pour `image()`
  (0,25 Gio ; à 1 Gio le job a manqué de mémoire hôte), pas optimisée.

- La variante `zero` (8.1) : `Variant` (`chunk_desc.h`), un paramètre template de `ChunkDesc`,
  `View`, `Writer` et `Array`, `zero_any` par défaut. `ChunkDesc<T, Variant::zero>` ne contient
  que le pointeur et `constant_of(d)` y lit `T{}` (`zero_value<T>()` : à partir d'octets nuls
  pour `std::complex` et `std::array` dans le code device, dont les constructeurs sont des
  fonctions hôte pour nvcc) ; la passe ne replie qu'à `T{}` ; le constructeur refuse une autre
  constante ; `assign(state)` matérialise un chunk compressé vers une autre valeur que `T{}`.
  Les validations font tourner les deux variantes : `device_core_validation` (la vue, la taille
  du descripteur), `device_write_validation` et `device_assign_validation` (repli vers +0,0
  seulement, -0,0 est une autre valeur), et `device_convert_validation` (un tableau hôte
  `zero_any` vers un tableau device `zero` : ses dix chunks de seconde valeur reçoivent un
  buffer et reviennent matérialisés). Elles passent sur la T1000 et le MI300A (là non plus,
  aucune attente de réservation abandonnée). `device_bench` mesure les deux variantes (un
  quatrième argument en choisit une). T1000, 0,5 Gio : la même chose au bruit près partout sauf
  pour `assign()` dispersé dans des chunks matérialisés, 1,67x contre 1,97x à S = 1024 (2,14x
  contre 2,22x à S = 256) : un accès aléatoire charge son descripteur, deux fois plus petit. La
  lecture d'un tableau compressé ne change pas (0,45x) : à cette taille les descripteurs restent
  dans le cache. MI300A, 8 Gio (bb4207da2), zero contre zero_any à S = 256 / 1024 : la lecture
  d'un tableau compressé 0,38x / 0,35x la lecture plate contre 0,65x / 0,62x (-42 % / -44 % : la
  lecture est un chargement de descripteur, 32 Mo de descripteurs contre 64 Mo à S = 256), le
  balayage de `compress()` -24 % / -16 %, l'écriture alignée dans des chunks matérialisés 2,09x /
  1,09x contre 2,15x / 1,27x, sa première écriture 4,70x / 1,65x contre 5,07x / 1,75x, les
  réécritures de plages avec `compress()` -8 % ; la lecture d'un tableau matérialisé, `+=` et
  les écritures dispersées dans des chunks matérialisés ne bougent pas (leurs buffers dominent).
  À S = 256 l'écriture dans des chunks matérialisés reste au-delà du critère de 2x dans les deux
  variantes.

- Le critère « même mémoire finale que la variante CPU à géométrie égale » :
  `device_memory_validation_zero_any` et `_zero` appliquent les mêmes six manches d'écritures à
  un tableau hôte et à un tableau device de la même variante (par `Writer::write` par plage et
  par `Writer::assign` par élément : 0,0 et -0,0, une constante par chunk, NaN avec payload, le
  dernier chunk partiel, des zéros partout) et comparent les dispositions chunk par chunk après
  chaque `shrink_to_fit()` / `compress()` : genres, constantes bit à bit, éléments. Elles sont
  égales, à une différence connue près : en `zero_any`, un dernier chunk partiel uniforme sur
  ses éléments valides se replie sur le device et garde son buffer sur l'hôte. L'hôte teste un
  chunk sur ses `chunk_size()` éléments, queue au-delà de `size()` comprise (elle contient la
  constante de la matérialisation du chunk) ; le device compare les éléments sous n. Un buffer
  de plus sur l'hôte, au plus un par tableau. **Défaut CPU, à corriger sur develop si on le
  veut** : faire passer le nombre d'éléments valides du dernier chunk par `ChunkElems` (les
  trois mises en file du proxy d'écriture, les balayages des tableaux, le repli au rechunk),
  avec un compte qu'un `resize()` entre la mise en file et la passe ne peut pas rendre périmé,
  et le rejeu des écritures différées. Pas en `zero` (sa queue vaut 0, sa seule valeur de
  repli). Les deux passent sur la T1000 et le MI300A, avec les mêmes dispositions.

- CI (`.github/workflows/device.yml`, sur la branche `gpu`) : compilation seule, les runners
  hébergés n'ont pas de GPU. nvcc 13.1 pour sm_75 et sm_90, et ROCm 7.2.4 pour gfx942,
  construisent toutes les validations device et `device_bench` (premier run vert, 25/11/2026).

**Étape 3, commencée (novembre 2026) : la façade Kokkos.** Les bancs visent les codes Kokkos et
Arcane, Kokkos d'abord. `chunk/device/kokkos_adaptor.h` : `device::kokkos::Array<T, S, V>`
possède un `device::Array` ; `write()` donne un accès en écriture (`w(i) = v`, `w(i) += v`, `-=`,
`*=`, `/=`, `T x = w(i)` : `Writer::assign` / `load` depuis une `KOKKOS_LAMBDA`), `end_write()`
termine la phase (fence, `finish()`), `read()` donne la `View` d'une phase de lecture,
`parallel_for_chunks(lo, hi, f)` est la boucle alignée de la bibliothèque (ses propres noyaux :
les équipes CUDA de Kokkos sont sur `threadIdx.y`), `compress()`, `bytes()` / `arena_bytes()`.
`-DCHUNK_DEVICE_KOKKOS=ON -DCHUNK_KOKKOS_ARCH=TURING75 | HOPPER90 | AMD_GFX942_APU` récupère
Kokkos 5.3.0 (SHA-256 épinglé ; `CHUNK_KOKKOS_URL` accepte une archive locale là où le curl de
CMake n'atteint pas GitHub), compilé pour CUDA ou HIP comme langage CMake, sans rocThrust. Sur le
MI300A, le compilateur HIP ne doit pas viser GCC 14.1, dont il refuse le `<format>` (`++this` dans
`bits/unicode.h` ; Kokkos inclut `<chrono>`) : GCC 15 ou 16. `kokkos_validation` passe sur la
T1000 et le MI300A.

Brancher Kokkos a trouvé quatre défauts de la bibliothèque, corrigés : `Writer::assign` / `load`
n'étaient que `__device__` (désormais `CHUNK_HD`, un appel hôte avorte ; une branche
`KOKKOS_IF_ON_DEVICE` dans la lambda de l'appelant ne se lançait pas) ; `lane_id()` valait
`threadIdx.x % warp_size`, 0 pour toutes les lanes d'un bloc de `RangePolicy` Kokkos (ses threads
sur y), et l'agrégation par warp ne finissait jamais (désormais la lane matérielle) ; l'arène
grandissait par petits pas jusqu'à `max_arenas` (désormais d'un huitième au moins) ; et l'écriture de
la constante du chunk après un ajout du même élément par le même thread était perdue, le rejeu
appliquait la valeur antérieure (MI300A, sommes de contrôle fausses et variables d'un run à
l'autre ; l'étape 0 de `device_assign_validation` le reproduit sur la T1000).

`examples/kokkos/kokkos_shrink_bench` : le motif d'accès de shrink_bench sur cinq
`Kokkos::View<double*>` et sur cinq tableaux de la façade, les mêmes noyaux : les bandes
d'initialisation (`bench_phase_fill.cpp`, par `parallel_for_chunks`), puis à chaque itération une
fenêtre aléatoire (`bench_phase_compute.cpp` : les écritures avant et après, la fenêtre réécrite,
20 passes de lecture, 2 passes de lecture-modification-écriture), une remise à zéro toutes les
iters/10, `compress()` toutes les N. MI300A, 5 x 2^30 doubles, 2000 itérations, fenêtres de 10^5
(871d5a370 -> a777dfc35, le settle par liste plus bas) :

| | zero_any | zero |
|---|---|---|
| mémoire tenue à la fin / les 42,95 Go des vues | 1,51 Go (3,5 %) | 19,2 Go (44,6 %) |
| avec les buffers libres des arènes | 1,98 Go (4,6 %) | 22,3 Go (52 %) |
| initialisation (alignée) | 0,55x -> 0,54x | 47,5x -> 46,6x |
| allocation + initialisation (c4dad808f) | -- | 0,64x |
| boucle | 2,84x -> 2,09x | 2,26x -> 1,86x |
| 20 passes de lecture / 2 passes RMW | 1,00x / 4,44x -> 1,01x / 4,15x | 1,01x / 4,14x -> 1,01x / 4,04x |
| écritures autour de la fenêtre / fenêtre réécrite | 36x / 39x -> 15,2x / 21,7x | 24x / 19x -> 11,6x / 10,6x |
| attentes de réservation (a777dfc35) | 3,6 millions | 2,0 millions |

Sommes de contrôle égales à celles des vues dans les deux runs, aucune attente abandonnée. L'initialisation coûtait 23x et les arènes 20 Go
avant qu'un chunk compressé écrit en entier avec une seule valeur ne la prenne comme constante
(`Writer::fold_whole`, 7.3). Les petites phases (une fenêtre fait environ 100 chunks par tableau,
15 us à plat) paient `end_write()` : une lecture des compteurs, le settle, des fences. Depuis ce
run, `finish()` décide de chaque étape d'après une seule lecture des compteurs, le settle ne visite
que les chunks publiés frais par la phase (une liste) sans synchronisation, et `write()` ne fait
plus de fence : T1000, 5 x 2^24, la boucle d'environ 2,0x à 1,24x-1,46x (zero_any) et 1,07x-1,26x
(zero), sommes égales ; le MI300A de 2,84x à 2,09x et de 2,26x à 1,86x (le tableau). Ce que paie
la boucle maintenant, zero_any : les lectures (1,21 s sur 2,88, à la vitesse des vues), les
écritures autour et dans la fenêtre (environ 1 s contre 55 ms à plat, 3,6 millions d'attentes de
réservation : des threads qui attendent qu'un autre matérialise leur chunk), les passes RMW
(0,5 s), compress() (0,17 s). L'initialisation de `zero` (47x), c'est son allocation : celle des
vues (mises à zéro, 1,84 s) n'était pas chronométrée, la croissance des arènes tombe dans le
remplissage (c4dad808f chronomètre les deux). Allocation et initialisation ensemble : 1,29 s,
0,64x celles des vues. Une arène dimensionnée dès le départ (1/2 des chunks) déplace le coût :
0,92 s d'allocation, un remplissage de 0,24 s (le premier accès à 18 Go de buffers), 1,16 s en
tout ; le premier passage raté et le second coûtent environ 50 ms. Rien à changer dans la
bibliothèque.

Les attentes de réservation n'étaient pas le coût de ces écritures : ce qu'elles attendaient
l'était. Un take passé un shard d'origine vide regardait les shards un par un, deux lectures
dépendantes chacun, 29 en moyenne (T1000, petites arènes) ; chaque arène a maintenant un masque
de ses shards à essayer (`ArenaDesc::avail`, effacé par un take qui trouve un shard vide, remis
par un give back), le suivant trouvé par ffs. Et les cinq tableaux d'une phase attendaient
chacun dans `end_write()`, une synchronisation et une copie bloquante des compteurs :
`kokkos::end_write(a1, ..., a5)` (`device::finish_all`) met les cinq copies en file et attend une
fois. T1000, 5 x 2^24, 400 itérations, runs alternés : zero_any écritures autour de la fenêtre
76 -> 52 -> 43 ms, fenêtre réécrite 58 -> 49 -> 39 ms, boucle 1,50x -> 1,39x -> 1,20x ; zero
38 -> 27 ms et 27 -> 17 ms, boucle 1,21x -> 1,13x (le masque laisse zero tel quel). MI300A,
5 x 2^30, 2000 itérations (7ee357b5e contre a707bbcf3), sommes égales, aucune attente
abandonnée, la mémoire inchangée :

| | zero_any | zero |
|---|---|---|
| boucle | 2,08x -> 1,77x | 1,87x -> 1,54x |
| écritures autour de la fenêtre | 440 -> 354 ms (12,6x) | 338 -> 255 ms (9,1x) |
| fenêtre réécrite | 529 -> 450 ms (18,2x) | 271 -> 171 ms (6,9x) |
| 2 passes RMW | 499 -> 256 ms (2,1x) | 486 -> 240 ms (2,0x) |

Le profil (rocprofv3, 5 x 2^26) : une phase d'écriture de fenêtre prenait environ 260 us, dont
cinq `hipMemset` du compte des buffers frais après chaque settle (13 us de temps hôte chacun),
cinq copies asynchrones des compteurs (5,5 us et un noyau de copie chacune), et les deux noyaux
du banc à 27 us chacun contre 3,4 us à plat. `settle_listed` vide maintenant la liste lui-même
et `finish_all` rassemble les compteurs de 16 tableaux au plus par un seul noyau (7381cc5ba) :
boucle 1,77x -> 1,64x (zero_any) et 1,54x -> 1,45x (zero) ; écritures autour de la fenêtre
354 -> 258 ms et 255 -> 193 ms, fenêtre réécrite 450 -> 380 ms et 171 -> 135 ms, RMW 256 -> 220 ms
et 240 -> 204 ms ; sommes égales.

Les buffers libres des arènes à la fin ne sont pas des arènes entières : les anciennes sont
pleines, les buffers libres sont la queue jamais utilisée de la plus récente, le pas de
croissance (T1000, par arène, utilisés sur ses buffers : `320/320 4734/4734 101/1263` pour un
tableau de la variante zero). `release_free` rend les arènes sans buffer utilisé après une
compression (aucune dans ce banc) ; le pas de croissance passe d'un quart à un huitième de
l'arène (`max_arenas` 32 -> 64), et la marge des spans ratés et des besoins du replay de même.
T1000, 5 x 2^24 : buffers libres à la fin 0,009 -> 0,006 Go (zero_any) et 0,055 -> 0,021 Go
(zero), les temps inchangés. Mapper les pages physiques à la demande (mémoire virtuelle)
ramènerait le coût de la queue à rien : pas fait.

MI300A, 5 x 2^30, 2000 itérations (d48588038) : buffers libres à la fin 0,465 -> 0,318 Go
(zero_any) et 3,087 -> 1,233 Go (zero), les chunks tenus inchangés (1,514 et 19,166 Go, 3,5 % et
44,6 % du plat), donc l'empreinte device à la fin 1,98 -> 1,83 Go et 22,25 -> 20,40 Go. Sommes
égales, 7 tests sur 7. La boucle passe de 1,64x à 1,73x (zero_any) et de 1,45x à 1,47x (zero) :
écritures autour de la fenêtre 258 -> 257 ms et 193 -> 195 ms, RMW 220 -> 213 ms et
204 -> 203 ms, mais fenêtre réécrite 380 -> 494 ms (19,8x) et 135 -> 147 ms (5,9x). La part de
la croissance plus fréquente et celle du nœud d'un jour à l'autre ne sont pas séparées :
7381cc5ba et d48588038 sont à passer dans le même créneau.

GH200, le même run (d48588038, nvcc 13.3, 8 tests sur 8, sommes égales), face au MI300A :

| | GH200 zero_any | MI300A zero_any | GH200 zero | MI300A zero |
|---|---|---|---|---|
| boucle | 1,68x | 1,73x | 1,47x | 1,47x |
| écritures autour de la fenêtre | 148 ms (6,1x) | 257 ms (9,1x) | 106 ms (4,4x) | 195 ms (6,9x) |
| fenêtre réécrite | 223 ms (11,2x) | 494 ms (19,8x) | 79 ms (4,0x) | 147 ms (5,9x) |
| 20 passes de lecture | 0,98x | 1,01x | 0,98x | 1,01x |
| 2 passes RMW | 1,56x | 1,79x | 1,54x | 1,71x |
| `compress()`, 200 passes | 176 ms | 197 ms | 204 ms | 249 ms |
| allocation + remplissage | 0,66x | 0,05x | 1,07x | 0,52x |

La mémoire est la même à l'octet près (1,514 + 0,318 Go et 19,166 + 1,233 Go) : elle dépend des
écritures du banc, pas du device. Le GH200 compte deux fois plus d'attentes de claim (8,0 M et
4,2 M contre 3,7 M et 2,0 M) et écrit pourtant plus vite. Le rapport de la boucle est le même
sur les deux devices ; `compress()` fait 13 % (zero_any) et 18 % (zero) de la boucle du GH200,
environ 0,9 ms par passe, le coût suivant après les écritures.

Suite : l'A/B sur le MI300A ; les noyaux d'écriture eux-mêmes (27 us contre 3,4) ; `compress()`
dans la boucle ; puis Arcane.

## 14. Risques et questions ouvertes

- **La lecture à plus de 1,5×** : si la latence de l'indirection n'est pas masquée sur un
  kernel déjà limité par la bande passante, le format n'a pas d'avenir GPU en l'état ; c'est
  pour cela que l'étape 0 mesure la lecture en premier.
- **Kernels à indices arbitraires** (gather/scatter irréguliers) : un warp touche alors
  plusieurs chunks, l'agrégation par warp se dégrade en plusieurs allocations, et les
  accès ne sont plus coalescés -- mais ils ne le seraient pas non plus sur un tableau plat.
  À mesurer séparément.
- **Les CAS perdus** : pas rares, finalement : S/32 warps par chunk sur une boucle régulière
  (6.2). La réservation les supprime. Il reste son attente (aucune abandonnée sur le MI300A).
- **Acquire et release à l'échelle du device sur AMD** (13, MI300A) : chacun invalide ou
  réécrit le L2, ce qui y rend chaque matérialisation 15x à 600x une écriture plate. L'étape 1
  doit les retirer du chemin d'écriture, pas les régler.
- **nvcc** sur le code template de la bibliothèque : temps de compilation et bugs de
  front-end sur les concepts ; clang-CUDA est la sortie de secours.
- **Le registre de motifs** de `pattern` entretenu par l'hôte : détection coûteuse si le
  nombre de motifs candidats est grand ; à borner comme en CPU (un motif par forme).
- **Quels utilisateurs** : codes Kokkos, Arcane, CUDA/HIP natifs ? La réponse
  décide si la façade Kokkos est l'étape 3 ou l'étape 1, et si la couche accélérateur
  d'Arcane doit être une seconde façade.
- **Les machines de mesure** : un GH200 et un MI300A sont disponibles sur le calculateur pour
  les chiffres absolus ; la RTX A1000 du portable a donné les rapports de l'étape 0.
