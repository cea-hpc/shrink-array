# CHUNK_SYNC_ON_READ — état des lieux & reprise (post-fw13)

**Date de mise à jour : 20 août 2026.**
**Statut : ON prouvé sain à 10 000 cycles sur le pire cas de correctness historique
(gnu / have-agressive / ARM Grace / `_check_256`, mode ON). Défaut de production
toujours OFF. Anomalie perf x86 diagnostiquée (§4 : plancher NUMA du gate ON, pas
un bug). Reste : 2ᵉ batterie, MINIMAL clean, autres tailles, chiffrage perf iso-socket.**

Ce fichier prolonge `CHUNK_SYNC_ON_READ_investigation.md` (fw2→fw12). Il n'en
remplace rien : il ajoute fw13, corrige le « verdict de fond » du §2 de l'ancien
journal, et liste ce qui reste à faire pour une reprise propre.

---

## POINT DE RESTAURATION

> **Tag git : `gen-dev-31-fw13-green`** (annoté).
> Dernier état prouvé vert : ON 10 000/10 000 cycles clean, gnu/have-agressive/
> ARM Grace/`check_256` (oracle armé). Contenu : fw6+fw7+fw9 (base fw9-flushkeep)
> + A1/fw10 (`501914c2` + fix déclaration non-OMP) + fw13 (gatefix). fw11/fw12
> **hors** arbre. Tag de build : `SHRINK_BUILD CHUNK_SYNC_ON_READ=ON fw13-gatefix-A1seed`.
>
> **Repartir de ce tag** pour tout nouveau chantier (interface STL, réorg,
> auto-tune). Vérifier après checkout : `strings <exe> | grep SHRINK_BUILD` →
> `fw13-gatefix-A1seed`, et absence des doublons `(1)` dans l'arbre.

---

## 0. TL;DR pour reprise

- **La classe principale de corruption ON était un bug de gate de lecture, pas la
  course lecteur↔recyclage-de-buffer du §2.** fw13 la ferme.
- **fw13 (gatefix)** : les deux gates de lecture (`reader_sync`,
  `read_gate_should_wait`) comparaient le **mot pending brut** (`> 0` signé) au lieu
  de le **décoder**. Dès que deux rangs publient sur le même chunk, le champ
  `owner` devient `kOwnerMulti = 0xFFF`, bit 31 du mot → mot **négatif** → gate
  s'ouvre alors que des ops sont pendantes → lecture de la valeur pré-op. Fix : les
  deux gates délèguent à `pending_any()` (l'unique prédicat qui décode). OFF
  codegen-identique (là `pending_any()` EST `raw > 0` signé).
- **Tag de build courant** : `SHRINK_BUILD CHUNK_SYNC_ON_READ=ON fw13-gatefix-A1seed`
  (vérifier avec `strings <exe> | grep SHRINK_BUILD` **avant tout comptage**).
- **Stack conservé** : fw6+fw7+fw9 (base du zip fw9-flushkeep) + **fw10/A1**
  (seeded materialisation, cherry-pick `501914c2`) + **fw13** (gatefix).
  fw11/fw12 **volontairement dehors** (voir §5).
- **Reproducteur autonome du bug fw13** : `gate_bug_repro.cpp` (mot `0xFFF00002`
  = 2 ops pendantes, 2 owners → gate brut PASS, gate décodé WAIT).

---

## 1. fw13 — le gate de lecture ne décodait pas le mot encodé

### Symptôme
- ON plantait aléatoirement en gnu, **jamais** en llvm.
- Résiduel `-DCHUNK_SYNC_ON_READ_MINIMAL` (fw8b, read hooks seuls) inexpliqué à
  ~1/58-147 : le trou était **dans le hook de lecture lui-même**.

### Cause racine (prouvée)
Dans `DeferredReadGate<Precise>` (chunk_base.h), le mot pending est encodé
`[owner:12 | count:20]` sous sync_on_read. `pending_publish` met
`owner := kOwnerMulti (0xFFF)` dès qu'un **second rang** publie sur un chunk qui a
déjà `count > 0`. Or les deux gates de lecture faisaient :

```cpp
pending_ops_.load(acquire) > 0   // comparaison SIGNÉE sur le mot ENTIER
```

`0xFFF00002` (2 ops, 2 owners) est **négatif** en int32 → `> 0` répond faux → le
lecteur saute `wait_for_availability` alors que 2 ops sont pendantes → il lit la
valeur **pré-op** (buffer non encore rejoué). Les verbes `pending_any()` /
`foreign_pending()` décodaient déjà correctement (sign-extension du count 20 bits) ;
seuls les deux gates de lecture avaient été oubliés.

### Pourquoi gnu et pas llvm
libgomp n'a **pas** OMPT → les pendings survivent des phases entières → la fenêtre
« deux rangs pending simultanément sur le même chunk » est fréquente. Sous llvm, les
drains OMPT par-barrière vident les pendings en permanence → la fenêtre multi-owner
est quasi inexistante. C'est exactement l'asymétrie observée depuis fw6.

### Pourquoi ça colле à toutes les signatures du §2 de l'ancien journal
Valeur « en retard » (`0 != -1`), valeur voisine, constante ressuscitée
(sin(1), cos(1)) : toutes compatibles avec la lecture d'un état pré-replay. Le §2
attribuait ça au recyclage de buffer ; c'était (au moins majoritairement) le gate
ouvert. fw11 avait « prouvé » que l'ordonnancement par élément ne fermait pas la
course — vrai, mais en présence d'un gate percé en amont.

### Fix
Les deux gates délèguent à `pending_any()` :

```cpp
[[nodiscard]] bool read_gate_should_wait(int) const noexcept { return pending_any(); }
void reader_sync(int i) const noexcept {
    if (pending_any()) [[unlikely]]
        static_cast<const Derived&>(*this).wait_for_availability(i);
}
```

- OFF : `pending_any()` = `raw > 0` signé → **codegen byte-identique** au code
  d'avant. Aucun risque de régresser le baseline OFF (prouvé 10 000).
- ON : décodage correct (count signé 20 bits) → le gate tient sous multi-owner.

### Fichiers touchés par fw13
- `chunk_base.h` : les deux gates → `pending_any()`.
- `config.h` : tag bumpé `fw13-gatefix-A1seed`.

---

## 2. fw10 / A1 — piège de build rencontré à la reprise (à connaître)

Le cherry-pick de `501914c2` (seeded materialisation) était **incomplet** : la
déclaration de `create_value_seeded` n'existait que dans la branche `#ifdef _OPENMP`
de `Chunk2A` (chunk2A_std.h). Or `chunk2A_std_inline.h` fournit **deux** définitions
(OMP et non-OMP), et les TU de bibliothèque (`ManagerChunk.cpp`) sont compilés
**sans `-fopenmp`** → branche `#else` active → la définition non-OMP était orpheline :

```
error: no declaration matches 'void Chunk2A<T>::create_value_seeded(...)'
```

**Fix** : déclarer `create_value_seeded` **dans les deux branches** du `#ifdef`
(jamais les deux visibles à la fois, sinon double-déclaration sous OMP). C'est la
même asymétrie OMP/non-OMP que les bugs historiques `manage_full_second_value` /
`arithmetic_assign`. Corrigé dans `chunk2A_std.h`. **Ce fix appartient au
cherry-pick fw10, pas à fw13.**

> ⚠️ À vérifier à la reprise : Chunk2 (pas seulement Chunk2A) a-t-il un chemin
> seedé ? Si `501914c2` n'a touché que `chunk2A_std*`, Chunk2 n'a pas A1 du tout.

---

## 3. Résultat de correctness (le juge)

**Batterie : gnu / have-agressive / ARM Grace / `shrink_bench_chunk2a_256 --mode check`
(CHECK_OPTIM armé, oracle) / mode ON / tag `fw13-gatefix-A1seed`.**

| jalon                | résultat                          |
|----------------------|-----------------------------------|
| 1000 cycles          | propre                            |
| 2729 cycles          | propre                            |
| 5180 cycles          | propre                            |
| **10 000 / 10 000**  | **propre, zéro échec** ✓          |

Contexte : le stack ON complet corrompait à ~1/40-75 sur cette **même** config →
~130-250 échecs attendus sur 10 000 → **zéro observé**. Ce sont des runs
**vérifiés** (oracle), pas des « n'a pas planté ».

**TSan** (cmake-build-omp-llvm-debug-thread-sanitizer, `OMP_NUM_THREADS=64`,
1 run) : **propre**. Élimine la classe fw11 (course plain load/store élément).
NB : TSan ne peut PAS innocenter la course §2 (recyclage de slab = course sur
*quel buffer* `values` pointe, pas sur une adresse fixe traquée par TSan) — seul le
compteur have-agressive en est juge, et il est vert.

### Portée de la preuve (honnête)
- Prouvé : gnu / have-agressive / ARM / taille statique 256 / ON.
- **Non** prouvé : autres tailles (64/1024/dyn), x86, autres variantes (Chunk,
  Chunk2), régime perf établi. 10 000 propres sur UNE config = très forte baisse du
  taux, **pas** preuve d'absence universelle.

---

## 4. Anomalie perf x86 — DIAGNOSTIC RÉSOLU : plancher NUMA du mode ON

**Ce n'est PAS un bug perf, c'est un plancher de scalabilité NUMA inhérent au
design ON, amplifié sur un nœud multi-socket. À documenter comme donnée de la
piste 4, pas à « corriger » dans le code. Séparé de la correctness (matérialiser
ne corrompt pas, ça consomme).**

### Mesures (ARM = Grace 288 c ; x86 = Milan bi-socket 2×64 = 128 c, NUMA)

| config                              | init zero | mém. après init | init-to-zero (externe) |
|-------------------------------------|-----------|-----------------|------------------------|
| ARM Grace, 96 threads (96/288, 33%) | 0.55 s    | 6444 Mo         | 0.547 s                |
| x86, **64 threads (mono-socket)**   | **0.68 s**| **6209 Mo**     | **0.68 s**             |
| x86, 96 threads (déborde 2ᵉ socket) | 24.5 s    | 8566 Mo         | 35.4 s                 |
| x86, 96 threads + PROC_BIND=close   | 24.5 s    | 8566 Mo         | 35.4 s (pinning inutile ici) |

**La falaise est à la frontière de socket, pas graduelle** : 64T mono-socket se
comporte comme ARM (ON sain, 0.68 s, 6.2 Go) ; dès qu'on franchit 64 (donc qu'on
occupe le 2ᵉ socket), init ×36 et mémoire ↑. `OMP_PROC_BIND=close` ne peut pas
aider à 96T : il ne crée pas de cœurs, il remplit socket 1 puis déborde compact
sur socket 2 → le cross-socket est inévitable dès 96 threads demandés.

### Les deux mécanismes (reliés aux symboles perf)

**#1 — Hotspot `write_with_optional_lock_imp` (x86 55 % vs ARM 42 % réparti) =
ping-pong de cache-line cross-socket sur le mot `pending_ops_`.** Sous ON
have-agressive, chaque write consulte `chunk.pending_any()` (load acquire sur le
mot pending). Ce mot est écrit en permanence par les compresseurs
(`pending_publish`/`pending_retire`) et lu par chaque writer. Single-socket
(Grace) : ligne locale. Bi-socket (Milan) : aller-retour interconnexion à chaque
write. C'est la même famille que le désastre documenté dans `chunk_base_inline.h`
(HOTSPOT NOTE x86 Aug '26 : « 89 % des cycles, init 93 s @128T vs 1.4 s ARM
single-socket »). **Inhérent au gate ON** : le gate lit le mot pending par write.

**#2 — Mémoire ×2 + `make_block_compressed` (x86 5 %, absent du top ARM) =
famine de compresseurs → matérialisation eager non re-compressée.** Le seuil P3
(`deferred_drain_threshold`, absolu par rang, défaut 4096) : quand le backlog d'un
rang le franchit, `flush_deferred` rejoue les writes en direct
(`in_flush_replay`, gate bypassé) → matérialise le bloc. ARM sous-souscrit : les
compresseurs ont des cœurs libres → re-compressent aussitôt en cst → mémoire
basse. x86 96/128 2-socket : compresseurs affamés (compétition cœurs + contention
atomics cross-socket sur les compteurs `CompressionManager`) → blocs matérialisés
restent matérialisés → 8.6 Go. Le ×2 = « x86 ne re-compresse pas ce qu'il a
matérialisé », pas « x86 écrit plus ».

### Reco d'usage x86 bi-socket (vaut pour ON surtout, OFF aussi)
- **Rester ≤ 64 threads mono-socket** : `numactl --cpunodebind=0 --membind=0`,
  `OMP_NUM_THREADS=64`. C'est le run prouvé sain (0.68 s, 6.2 Go).
- **Ou 1 processus par socket** (ex. 1 rang MPI de 64T par NUMA node) plutôt qu'un
  seul processus OpenMP à cheval sur les 2 sockets — supprime le trafic cohérence
  sur les structures partagées.

### Test restant (confirme #2, pas décisif)
Monter `CHUNK_DEFERRED_DRAIN_THRESHOLD` (cache CMake, ex. 65536) et relancer le
96T x86 : si `make_block_compressed` sort du top et la mémoire redescend vers
6 Go, le mécanisme #2 est bien gouverné par le seuil (domptable même cross-socket).
N'affecte PAS #1 (inhérent au gate). Contrepartie : backlog tenu plus longtemps →
pic mémoire des ops différées plus haut.

### Conséquence pour la décision de défaut (piste 4)
ON gagne ~56 s d'init sur ARM détendu ; sur Milan bi-socket saturé, le
`pending_any()` par-write injecte du trafic cohérence qui peut manger le gain.
**ON est fait pour le régime many-cœurs-cohérent (Grace), pas pour le NUMA
saturé.** Comparatif honnête = iso-thread ET iso-occupation (64T mono-socket des
deux côtés → ON excellent partout). Le « x86 100× plus lent » n'existe qu'en
config bi-socket saturée.

## 5. Décisions de stack (ce qui est dans / hors du build benché)

**Dans** (arbre `fw13-gatefix-A1seed`) :
- fw6 (dual-word) + fw7 (signed-word) : réparent OFF, count auto-guérissant.
- fw9 (flushkeep) : le flush ne jette plus les ré-enqueues concurrents.
- **fw10 / A1** (`501914c2` + fix déclaration non-OMP du §2) : seeded write, ferme
  la fenêtre matérialisation cst→values publish-before-write.
- **fw13** (gatefix) : les deux gates de lecture décodent le mot.

**Hors, volontairement** :
- **fw11 (rel/acq élément)** : la course plain load/store sur élément établi existe
  formellement mais ne peut pas produire la signature observée (valeurs voisines /
  constantes ressuscitées). TSan propre le confirme. **En réserve** : si un résidu
  ARM/TSan apparaît un jour, le réactiver. Coût perf à chiffrer avant inclusion.
- **fw12 (initphase / g_reads_started)** : ré-active le drain d'écriture après la
  1ʳᵉ lecture (mode OFF-sûr) → **masquerait** le régime établi ON qu'on veut
  valider. À exclure de toute batterie de validation ON. Idée valable seulement
  pour codes à init strictement write-only ; insuffisante seule.

---

## 6. Reste à faire pour une reprise (ordre de priorité)

1. **2ᵉ batterie** (seeds/ordonnancement différents) sur la même config ARM
   have-agressive : si de nouveau 10 000 propres, la présomption « classe fermée »
   se renforce fortement pour un coût faible.
2. **MINIMAL clean** : rebuild `-DCHUNK_SYNC_ON_READ_MINIMAL` + fw13, batterie. S'il
   passe propre (il fuyait à ~1/58-147), c'est la **confirmation définitive** que le
   résiduel MINIMAL était ce gate et rien d'autre. Test le plus discriminant restant,
   peu coûteux.
3. **Autres tailles** (64 / 1024 / dyn) et **Chunk / Chunk2** en ON, batteries.
4. **Anomalie perf x86** (§4) : diagnostic matérialisation + test seuil P3.
5. **Chiffrage perf ON vs OFF en régime établi** (pas seulement init) — péage
   piste-1 (`init 2nd values` ×2), coût des gardes, convoi x86 — **avant** toute
   décision de basculer le défaut.
6. **Audit du sur-retire** (dette §4 de l'ancien journal) : compteur CHECK
   `min(count)` pour localiser le déséquilibre publish/retire, à froid.

**Le défaut reste OFF** tant que 1→5 ne sont pas faits. ON = expérimental, mais
désormais avec une correctness verte sur le pire cas historique.

---

## 7. Rappels méthodo (inchangés, toujours valables)

- **Tag `strings`** : `strings <exe> | grep SHRINK_BUILD` → doit dire
  `fw13-gatefix-A1seed`. Le `-D CHUNK_SYNC_ON_READ` persiste dans le cache du dir
  de build (même dir réutilisé par `--sync-on-read=on|off`), donc `strings` est le
  **seul** arbitre fiable du mode réellement compilé.
- **Runtime** : `ldd <exe> | grep gomp` (gnu, juge) vs `libomp` (llvm, ne
  discrimine rien).
- **Toggle sans wipe** : `compile_all[_gnu].sh --sync-on-read=on|off` passe un `-D`
  explicite qui écrase le cache en place (pas de `rm -rf`, dir inchangé). Repasser
  le flag à **chaque** changement de mode (le `-D` reste caché sinon).
- **Forensique** : `-DRING_LOG_OP -DSHRINK_RING_DUMP_ON_MISMATCH` +
  `dump_ring_for_chunk(idx)` au site du mismatch. Signature qui discrimine la classe
  résiduelle si un échec tardif apparaît : constante ressuscitée / valeur voisine →
  recyclage §2 (→ hazard-pointer) ; valeur en retard sur chunk quasi-cst →
  matérialisation (→ A1/fw10).
- **Reproducteur** : garder le stack ON complet comme meilleur reproducteur du
  mécanisme si on veut re-stresser fw13 ; `gate_bug_repro.cpp` reproduit le bug de
  gate isolément.

---

## 8. Pistes non épuisées (reprises de l'ancien §6, ré-évaluées)

1. **Hazard pointer sur `values`** (le vrai fix général du §2 SI un résidu subsiste
   après fw13) : le lecteur publie le pointeur lu, le recycleur vérifie l'absence de
   hazard avant de rendre le slab. O(1) lecture, pas de drain. **À ne tenter que si**
   la 2ᵉ batterie / MINIMAL / autres tailles exhibent un résidu — sinon inutile.
2. **fw11 en réserve** pour la course élément (voir §5).
3. **Audit sur-retire** (§6.6) : pourrait être une course latente amplifiée ;
   l'éliminer à la source (équilibrage publish/retire prouvé) pourrait faire
   disparaître des classes même sans hazard-pointer.

---

## 9. Chantiers ouverts (décision du 20 août 2026)

État : version opérationnelle et rapide, correctness ON verte (tag
`gen-dev-31-fw13-green`). Trois chantiers en cours, ordre décidé :

**0. Figer l'état vert (fait).** Tag annoté `gen-dev-31-fw13-green` + ce md à jour.
Point de restauration pour tout nouveau travail.

**1. Se rapprocher de l'interface `std::vector` STL** — *prioritaire*. C'est de
l'AJOUT de surface (accesseurs, itérateurs, `size`, mutateurs), pas une
modification du cœur concurrent → risque de course quasi nul. Débloque les deux
autres chantiers et uniformise le code appelant (cf. le mismatch `tab2(i)` chunk
vs `tab2[i]` vector). Ordre interne : accesseurs + itérateurs + `value_type`
d'abord (lecture), mutateurs ensuite.

**2. Auto-tune de la taille de chunk dynamique** — *autonome, après #1*. S'appuie
sur l'infra CHUNK_DYNAMIC existante (pools size-keyed, `register_runtime_chunk_size`,
`set_active_chunk_size`) → faible risque de course. MAIS prérequis : un **critère
de coût robuste**. Le §4 montre que le coût dépend brutalement de la topologie
(×36 NUMA) — une auto-tune qui optimise sur une machine et se trompe sur l'autre
serait pire que rien. À faire après avoir un modèle de coût fiable (rejoint le
chiffrage perf iso-socket).

**3. Simplification / réorganisation de la lib** — *EN DERNIER, avec garde-fou*.
Seul chantier qui touche le code déjà durci (proxies, policies CRTP, write path,
slab). Précédent frais : le bug OFF latent `4505ad27` était une simplification
d'encodage qui a transformé un sur-retire auto-guérissant en poison permanent,
invisible des centaines de cycles. Règle : batterie 10 000 ARM/have-agressive en
garde-fou APRÈS CHAQUE étape de réorg (pas à la fin), pas de pas > ~50 lignes,
repartir de `gen-dev-31-fw13-green` en cas de doute.
