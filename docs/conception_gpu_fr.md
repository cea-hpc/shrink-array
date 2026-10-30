# Portage GPU : premier document de conception

> English version: à écrire quand la conception sera stabilisée (`gpu_design.md`).

> État : octobre 2026, **avant tout prototype**. Ce document fixe ce qu'un backend GPU de
> shrink-array devrait être, ce qu'il reprend de la version CPU et ce qu'il abandonne, et le
> plan pour trancher par la mesure. Les chiffres annoncés sont des attentes, pas des résultats.
> Les décisions marquées **(à trancher)** attendent le prototype de la section 13.

***

## 1. Objectif et périmètre

**Objectif.** Offrir sur GPU (NVIDIA et AMD) un tableau qui se lit et s'écrit dans un kernel
comme un tableau plat, `a(i)` / `a(i) = v`, et qui ne stocke pas ses chunks uniformes --
le même format de données et les mêmes garanties (sans perte, écriture en place, toute valeur
répétée compressée par `zero_any`) que la version CPU. Le gain visé est d'abord la **mémoire**
: 80 à 192 Go de HBM, c'est là que « 20 fois moins » change ce qu'un code peut traiter.

**Ce qui est dans le périmètre de cette première conception :**

- les trois variantes `zero`, `zero_one_value`, `zero_any`, et `pattern` (section 8) ;
- géométrie fixe (S constante de compilation) d'abord, dynamique ensuite (section 9.3) ;
- un modèle d'exécution **par phases**, aligné sur les kernels (section 3) ;
- l'interopérabilité avec la version CPU : même format de chunk, transferts dans les deux
  sens, mode hybride sur les machines à mémoire cohérente (section 11).

**Ce qui n'est pas dans le périmètre :**

- faire tourner le code CPU existant sur GPU : la machinerie d'exécution (section 2) n'est
  pas portable et n'a pas à l'être ;
- les écritures concurrentes à une compression **à l'intérieur** d'un kernel ;
- `bool` compacté bit à bit (`ArrayChunkBits`) ;
- un backend SYCL/Intel (possible via la couche de portabilité, non visé).

## 2. Ce qui se porte, ce qui ne se porte pas

La bibliothèque CPU a deux couches de natures très différentes :

| couche | contenu | sur GPU |
|---|---|---|
| **format** | table de chunks ; un chunk = tampon privé ou constante ; S puissance de 2 ou multiple de 64 ; indice `i >> decalage`, `i & modulo` ; métadonnées par chunk | **se porte tel quel**, et s'y trouve même mieux qu'en CPU (section 4) |
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
| kernel utilisateur en écriture | écrivains | deux threads matérialisent le même chunk | CAS sur le descripteur ; le perdant abandonne son emplacement d'arène, repris en fin de phase (6.3) |
| `compress()` (fin de phase) | la passe | aucun : ni écrivain ni lecteur | vote par bloc + compaction (7) |
| `rechunk_to()` | hôte + kernels | aucun | séquence de phases (9.3) |

Conséquences :

- **personne n'attend jamais personne** dans un kernel : le seul atomique du chemin
  d'écriture est un CAS sans boucle d'attente (le perdant ne réessaie pas, il lit le
  gagnant) ;
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
    T        cst;                // la constante du chunk compressé (ou quittée : 6.2)
    // zero : cst est toujours T{} ; zero_one_value : T{} ou la seconde valeur du tableau ;
    // zero_any : n'importe quelle valeur.  pattern : voir 8.4.
};
```

Pour `T` de 16 octets (`std::complex<double>`) le descripteur fait 24 octets, alignés à 32
**(à trancher : 32 octets alignés, ou pointeur tagué + table latérale ; mesurer)**. Le CAS de
matérialisation porte sur `buf` seul (8 octets) : `cst` n'est écrit que par la passe de
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
| `ChunkDesc[nb_chunks]` | 16 ou 24 o par chunk | device (ou mémoire cohérente, 11) |
| compteur de bump de l'arène, tête de pile des tampons libres | 2 mots | device |
| append-buffer des écritures non appliquées (6.4) | borné, réglé à la création | device |
| statistiques (`stats()`, chunks matérialisés, échecs d'arène) | quelques mots | device, relus par l'hôte en fin de phase |

Il n'y a **pas** de dirty bitset, pas de liste de candidats, pas de ligne « déjà en liste » :
la passe de compression regarde tous les chunks matérialisés (7.1), ce qui sur GPU coûte
moins qu'entretenir des listes.

## 5. La lecture

```cpp
__device__ T operator()(IndT i) const noexcept
{
    const auto& d = desc_[i >> shift_];
    return d.buf ? d.buf[i & mask_] : d.cst;     // branchement uniforme par warp
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
    T* buf = atomic_load_relaxed(d.buf);          // portée device
    if (buf == nullptr)
    {
        if (v == d.cst) return;                   // 1. la constante sur sa constante : rien
        buf = materialise(d);                     // 2. matérialiser (6.2), puis écrire
    }
    buf[pos] = v;                                 // 3. chunk matérialisé : écriture directe
}
```

Les cas 1 et 3 sont le chemin chaud, uniformes par warp, sans atomique. Les règles de
`zero_one_value` (« la seconde valeur sur un élément qui la contient déjà ») et de `zero_any`
(« découverte », « départ », `may_become_uniform`) servaient à alimenter les listes de candidats
: elles **disparaissent**, la passe de fin de phase n'a pas besoin de candidats. Il reste à
prendre en compte `+=` et les autres opérateurs composés, qui lisent avant d'écrire : même
cas, avec `v` calculée à partir de `d.cst` ou `buf[pos]`.

### 6.2 La matérialisation

1. **Agrégation par warp.** Les lanes du warp dont le chunk est le même (en pratique tous :
   4.2) élisent un lane (`__match_any_sync` / `__ballot`, `__activemask`) ; les autres
   attendent le résultat de l'élu par un shuffle. Une matérialisation par warp et par chunk,
   pas par thread.
2. **Allocation** : le lane élu prend un tampon dans la **pile des tampons libres** (rendus
   par les compressions précédentes, 7.2), sinon par **bump** sur l'arène du tableau
   (`atomicAdd` sur le compteur, un seul par tampon). Si l'arène est épuisée : 6.4.
3. **Pré-remplissage** du tampon avec `d.cst` : par **tout le warp** (S/32 éléments par lane,
   coalescé), pas par le seul élu.
4. **Publication par CAS** de `nullptr` → `buf` sur `d.buf`, portée device, `release` après
   le pré-remplissage (une `__threadfence()` avant le CAS, ou `atomic_ref::compare_exchange`
   avec `memory_order_release`). Le perdant **ne réessaie pas** : il lit le `buf` gagnant
   (`acquire`) et rend son tampon à la pile libre (ou l'abandonne jusqu'à la fin de phase :
   **à trancher**, la pile libre coûte un atomique de plus, l'abandon coûte de la mémoire
   jusqu'au `compress()`).
5. Tous les lanes écrivent leur valeur dans `buf`.

Pourquoi le pré-remplissage avant le CAS est sûr : tant que le CAS n'a pas eu lieu, le
tampon n'est visible de personne ; après, tout écrivain qui le voit a lu `buf` en `acquire`
et voit le pré-remplissage complet. Deux warps qui matérialisent le même chunk pré-remplissent
chacun leur tampon ; un seul est publié. Un lecteur n'existe pas pendant cette phase.

Ce sont exactement les mécanismes des tables de hachage GPU (allocation par bump, insertion
par CAS, agrégation par warp) : ils sont connus, portables sur CUDA et HIP, et sans attente.

### 6.3 Pile des tampons libres

Une pile LIFO de pointeurs, tête atomique, **sans ABA** parce qu'aucun pop ne se fait pendant
qu'un push est possible : les push n'ont lieu qu'en `compress()` (7.2) et dans le cas 4 de
6.2 (le perdant d'un CAS, pendant un kernel d'écriture où les pop sont possibles). **(À
trancher)** : soit le perdant abandonne son tampon jusqu'à la fin de phase (pas de push
pendant les pop, la pile est triviale), soit la pile prend un compteur de génération. La
première forme est préférée pour sa simplicité ; le coût est une mémoire transitoire bornée
par le nombre de CAS perdus, rare avec l'agrégation par warp.

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
chunk matérialisé** (ou un warp par chunk pour S ≤ 1024, **à trancher par la mesure**) :

1. chaque lane lit ses S/32 éléments et compare à `buf[0]` ;
2. vote `__all_sync` (réduction `&&` dans le bloc) ;
3. si uniforme à une valeur que la variante sait compresser (`zero` : `T{}` ; `zero_one_value`
   : `T{}` ou la seconde valeur ; `zero_any` : toute valeur) : `d.cst = buf[0]`, `d.buf =
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
   `v` » **avant** de matérialiser : si oui, il écrit `d.cst = v` et rien d'autre. C'est la
   compression en vol, sans concurrence, parce que le bloc est le seul écrivain du chunk.
   Cette optimisation ne vaut que pour les kernels dont le découpage est fait par la
   bibliothèque ; un kernel utilisateur à indices arbitraires ne l'a pas.
3. **La pression mémoire force la phase.** Quand l'arène est épuisée, l'append-buffer (6.4)
   termine la phase de fait : l'hôte compresse, agrandit, rejoue. L'arène est donc aussi un
   **budget** : la fixer petite force des compressions fréquentes, au prix de rejeux.

## 8. Les variantes

### 8.1 `zero`

`cst` vaut toujours `T{}` ; la passe ne replie qu'à `T{}`. Rien d'autre.

### 8.2 `zero_one_value`

`cst` vaut `T{}` ou la seconde valeur du tableau (un champ du tableau, copié dans le
descripteur par la passe). `set_second_value()` est une opération d'hôte entre deux phases.
Sur GPU, `zero_one_value` n'a plus d'intérêt propre par rapport à `zero_any` (le registre
qui le justifiait n'existe plus) : il est conservé pour l'interopérabilité et la sémantique
(ne compresser qu'à une valeur déclarée), **(à trancher)** s'il mérite un code séparé ou un
`if constexpr` dans `zero_any`.

### 8.3 `zero_any`

Le descripteur porte la constante : c'est la forme naturelle de `zero_any`, et la variante à
prototyper en premier. Ce qui disparaît : le registre des secondes valeurs
(`tab_cst_val2_holder.h`), ses segments, son index haché, `reclaim_second_values()`, les lignes
`tab_type_cst_val` et `tab_second_value`, `CHUNK_SECOND_VALUE_INDEXED`. Le coût par chunk
redevient uniforme : le descripteur.

### 8.4 `pattern`

Un chunk motif = un motif partagé (S entiers) + un offset. Le descripteur gagne une forme :
`buf` pointe vers le motif, le **bit de poids faible de `buf`** (ou un champ `kind`) dit
« motif », et `cst` sert d'offset :

```cpp
return is_motif(d) ? motif(d)[i & mask_] + d.offset : (d.buf ? d.buf[i & mask_] : d.cst);
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

- **Format commun** : le descripteur device `{buf, cst}` est aussi un format CPU valide, à
  deux différences près : la version CPU pointe sur un tampon constant partagé là où le
  device met `nullptr`, et porte les portes OpenMP dans l'objet chunk. Un **convertisseur**
  dans les deux sens est simple : par chunk, `nullptr` ↔ `cst_val1` / tampon du registre, et
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
  façade), des cibles `chunk::chunk_zero_any_device_256`, … ; Spack : une variante `device`.

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
| 2. Phases et banc | `parallel_for` aligné et vote avant matérialisation ; `shrink_bench` device (phases du banc contre `DeviceArray`) ; mesures H100 / MI250X ou MI300A ; documentation (ce document mis à jour avec les chiffres, version anglaise) | chiffres publiés dans `variantes_fr.md` |
| 3. Façade Kokkos | `kokkos_adaptor.h`, `parallel_for` Kokkos marquant les phases, `memory_fence` à la place de release/acquire | un exemple Kokkos ; mêmes validations |
| 4. `zero_one_value`, `pattern` | 8.2, 8.4 | validations ; gain mesuré de `pattern` sur une connectivité |
| 5. Géométrie dynamique | 9.3 | `rechunk_validation` device |
| 6. Hybride | mémoire cohérente (GH200, MI300A), mode phase de la version CPU | un tableau lu et écrit alternativement par CPU et GPU sans copie |

## 14. Risques et questions ouvertes

- **La lecture à plus de 1,5×** : si la latence de l'indirection n'est pas masquée sur un
  kernel déjà limité par la bande passante, le format n'a pas d'avenir GPU en l'état ; c'est
  pour cela que l'étape 0 mesure la lecture en premier.
- **Kernels à indices arbitraires** (gather/scatter irréguliers) : un warp touche alors
  plusieurs chunks, l'agrégation par warp se dégrade en plusieurs allocations, et les
  accès ne sont plus coalescés -- mais ils ne le seraient pas non plus sur un tableau plat.
  À mesurer séparément.
- **Les CAS perdus** : rares avec l'agrégation par warp sur des boucles régulières, mais un
  kernel irrégulier qui matérialise le même chunk depuis des milliers de warps gaspille
  autant de tampons jusqu'à la fin de phase (6.3). Un compteur `cas_lost` dans `stats()`
  dira si la pile libre avec génération s'impose.
- **nvcc** sur le code template de la bibliothèque : temps de compilation et bugs de
  front-end sur les concepts ; clang-CUDA est la sortie de secours.
- **Le registre de motifs** de `pattern` entretenu par l'hôte : détection coûteuse si le
  nombre de motifs candidats est grand ; à borner comme en CPU (un motif par forme).
- **Quels utilisateurs** : codes Kokkos, Arcane, CUDA/HIP natifs ? La réponse
  décide si la façade Kokkos est l'étape 3 ou l'étape 1, et si la couche accélérateur
  d'Arcane doit être une seconde façade.
- **Les machines de mesure** : le cluster Grace n'a pas de GPU ; l'étape 0 demande un accès
  H100/GH200 ou MI250X/MI300A, même modeste (le prototype est petit).
