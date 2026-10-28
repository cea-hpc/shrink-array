# Mesurer la mémoire et le temps sans se tromper

> English version: [performance_measurement.md](performance_measurement.md).

> État : fin septembre 2026. Ce document rassemble ce que les campagnes de septembre 2026 sur
> Grace (aarch64, pages de 64 Kio, RHEL 9.6, 64 threads) ont appris, souvent à nos dépens :
> plusieurs conclusions ont dû être retirées parce que l'instrument de mesure mentait.

***

## 1. La mémoire

### 1.1 `VmRSS` est faux à plusieurs threads

`VmRSS` (`/proc/self/status`), `/proc/self/statm` et `ru_maxrss` (`getrusage`) lisent les
compteurs de mémoire résidente du noyau. Ces compteurs sont tenus **par thread**, et reportés
dans le total du processus seulement de temps en temps. Avec 64 threads qui font des défauts
de page, le total prend du retard. Mesuré sur `shrink_bench_zero_any_dyn` :

| moment | `VmRSS` | exact (`smaps_rollup`) | retard |
|---|---|---|---|
| initialisation | 9 576 Mio | 11 675 Mio | 2 099 Mio |
| début du régime | 11 820 Mio | 12 436 Mio | 616 Mio |
| fin du run | 11 870 Mio | 13 442 Mio | 1 571 Mio |

`VmRSS` était resté à ~11,8 Gio pendant tout le run, alors que le processus grossissait d'un
gigaoctet : on en avait conclu à tort que les rechunks ne faisaient rien gagner.

### 1.2 La mesure exacte : `smaps_rollup`

`/proc/self/smaps_rollup` parcourt les tables de pages au moment de la lecture : c'est exact.
Mais le coût croît avec la mémoire résidente (~10 à 20 ms pour un processus de 13 Go) : à
réserver aux points de contrôle, jamais à un chemin périodique chaud (les 200 lectures que
faisait la boucle chronométrée du banc lui coûtaient 1 à 2 s).

Dans la bibliothèque : `shrink_array::exact_rss_kib()` (`internal/exact_rss.h`). Dans le banc :

- les lignes « Memory consumption **after init** » et « **end** » sont exactes ;
- les lignes périodiques relisent `statm` (en retard, mais gratuit), sauf avec
  `SHRINK_BENCH_EXACT_RSS=1` ;
- le « peak » est le maximum de `ru_maxrss` et des lectures exactes ;
  `SHRINK_BENCH_RSS_SAMPLE_MS=<ms>` ajoute un échantillonnage de fond ;
- les traces mémoire (`CHUNK_LOG=memory=info` / `memory=debug`) lisent la valeur exacte.

### 1.3 Localiser la mémoire

| question | outil |
|---|---|
| combien dans les tampons de chunks, combien ailleurs ? | `CHUNK_LOG=memory=info` : à chaque passage de l'advisor, la mémoire exacte, dont la part résidente de l'arène des tampons (`mincore`) |
| que font les slabs ? | `CHUNK_ARENA_REPORT=1` (slabs distribués, rendus au système) |
| quelle zone de l'espace d'adressage grossit ? | deux `pmap -x <pid>` à deux instants, comparés par zone |
| que rend chaque rechunk ? | `CHUNK_LOG=memory=debug` : ligne `rss` (mémoire à chaque étape, `end-entry`) |

Un piège : le compteur « materialised » du build stats est une différence de compteurs
(alloué − libéré). Il n'est juste que si chaque libération est comptée, ce qui n'était pas le
cas des libérations sans compression (rechunk, destruction) jusqu'à fin septembre 2026 : on
avait vu 12 982 Mo « matérialisés » dans un processus de 12 012 Mo.

**Les mesures de mémoire, elles, sont reproductibles** : d'un run à l'autre et d'un build à
l'autre, au mégaoctet près. Un seul run suffit.

## 2. Le temps

### 2.1 Trois sources de bruit, bien plus grandes que les effets cherchés

- **La bimodalité, fixée au démarrage.** Sur dix runs identiques sans advisor, sept à
  6,4-6,8 s et trois à 7,7-8,0 s (+18 %). `perf stat` : même fréquence (3,26 GHz), même nombre
  d'instructions, mêmes défauts du dernier niveau de cache, mais un IPC de 3,45 au lieu de
  3,95 et +10 points de cycles bloqués sur la mémoire. Chaque accès mémoire coûte plus cher,
  sans doute selon la disposition physique des pages reçues au démarrage. `zero_one_value` a un mode
  lent encore plus marqué (~+40 %, des runs à 16 s au lieu de 11).
- **La dérive entre fenêtres de temps.** Un même binaire, sans aucune recompilation : 10,75 à
  11,53 s dans une fenêtre, 9,49 à 10,08 s une heure plus tard.
- **L'écart entre deux builds du même code.** Même commit, mêmes options, aucune modification
  locale : un build à 10,5 s, un autre à 9,7 s, de façon stable. Vraisemblablement la
  disposition du code en mémoire (alignement des boucles chaudes).

Un **test A/A** (le même code, deux fois, en alternance) donne la mesure du bruit : sur les
bancs dynamiques à 64 threads, ~10 % d'écart entre extrêmes.

### 2.2 La règle

**Ne comparer des temps que sur un même binaire**, en changeant le comportement par une
variable d'environnement, en **alternant** les modes dans la même fenêtre de temps, sur **au
moins trois tours**, et en comparant les **moyennes**. Ne retenir qu'un écart nettement
au-dessus de 10 %.

Comparer deux builds (même le même jour, avec les mêmes options) n'est pas fiable en dessous
de ±10-15 %. Quand c'est inévitable (mesurer une modification du code), construire les deux
dans des répertoires neufs avec la même configuration, alterner, et se méfier.

### 2.3 Les interrupteurs faits pour ça

| comparer… | interrupteur |
|---|---|
| sans advisor / analyses seules / avec rechunks | `--advisor_period 0` / `--rechunk_dry_run` / rien |
| affinité paresseuse / immédiate | `CHUNK_ALLOC_AFFINITY_EAGER=0` / `=1` |
| taille maximale d'un pas de grossissement | `--rechunk_max_coarsen <n>` |
| nombre de tranches de la reconstruction de `zero` | `CHUNK_RECHUNK_SLICES=<n>` |

La liste complète est dans [reglages_developpeur_fr.md](reglages_developpeur_fr.md).
Les réglages du décideur sont des options de `shrink_bench` (la bibliothèque ne les lit dans
aucune variable). Attention : `--rechunk_min_gain_mb 1000000` ne bloque plus les grossissements
depuis le critère « gain par coût » ; utiliser `--rechunk_dry_run`.

## 3. Recettes

**Le coût de l'advisor sur un même binaire** (trois tours, moyennes) :

```
rm -f /tmp/modes.txt
for r in 1 2 3; do
  for mode in no_advisor dry_run normal; do
    case $mode in
      no_advisor) opt="--advisor_period 0" ;;
      dry_run)    opt="--rechunk_dry_run" ;;
      normal)     opt="" ;;
    esac
    t=$(OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores $BIN --mode perf --no_io $opt 2>/dev/null \
        | grep -oE "Elapsed time computation : [0-9.]+" | awk '{print $NF}')
    echo "$mode $t" >> /tmp/modes.txt
  done
done
awk '{ s[$1] += $2; n[$1]++ } END { for (k in s) printf "%-26s %.2f s\n", k, s[k] / n[k] }' /tmp/modes.txt
```

**La durée de chaque intervalle entre deux passages de l'advisor**, pour voir si un
ralentissement vient des rechunks ou de ce qui les suit :

```
tick_times() {
  env "$@" CHUNK_LOG=warn,memory=info,rechunk=info OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores \
    $BIN --mode perf --no_io 2>&1 >/dev/null \
  | while read -r l; do
      case "$l" in
        *"[memory][I]"*)  echo "T $(date +%s.%N) ${l#*tick }" ;;
        *APPLIED*)        echo "R" ;;
      esac
    done \
  | awk '$1 == "R" { nr++; next }
         { t = $2; k = $3; if (p != "") printf "  %2d->%-2d %5.2f s%s\n", pk, k, t - p, nr ? sprintf("   (%d rechunks)", nr) : ""; p = t; pk = k; nr = 0 }'
}
```

**Le comportement seconde par seconde** (cœurs occupés, IPC, cycles bloqués sur la mémoire) :

```
perf stat -I 1000 -x, -e task-clock,cycles:u,instructions:u,stall_backend_mem:u -o /tmp/pi.csv -- $BIN --mode perf --no_io
```

Le nombre de cœurs occupés est `task-clock / 1000` pour chaque seconde. C'est ce qui a montré
que `zero`, après son rechunk, n'occupait plus que ~40 cœurs au lieu de ~52.

**Temps utilisateur, temps système, défauts de page** :
`/usr/bin/time -f "%e s réel, %U s utilisateur, %S s système, %R défauts mineurs" $BIN …`.

**La fragmentation de la mémoire du nœud** : `cat /proc/buddyinfo`. Sur un noyau à pages de
64 Kio, la dernière colonne compte les blocs libres de 512 Mio.

**Ce qui distingue un run lent d'un run rapide** (la bimodalité du 2.1) :
`python/bimodal_probe.py` lance N fois le même binaire et relève, pour chaque run, les pages par
nœud NUMA, AnonHugePages, la mémoire libre et les blocs libres d'ordre maximal de chaque nœud
avant le run, et les compteurs perf pour mille instructions (parcours de TLB, recharges L1D / L2,
accès distants). Il coupe les runs au plus grand écart de temps et dit quelles grandeurs séparent
les deux groupes. `--mode` alterne des environnements (`--mode "" --mode "CHUNK_RECHUNK_SLICES=64"`,
`--mode "numactl --membind=0"`).

```
python3 python/bimodal_probe.py --runs 12 -- $BIN --mode perf --no_io
```

## 4. Pièges rencontrés

- **Un binaire qui n'a pas été reconstruit**, ou un script qui n'a pas été appliqué : une
  variable d'environnement qu'aucun code ne lit est ignorée sans message. Vérifier avec
  `strings $BIN | grep NOM_DE_LA_VARIABLE`, ou par l'effet attendu.
- **Un cache CMake hérité** : des options de configurations anciennes restent dans
  `CMakeCache.txt`. En cas de doute, reconfigurer dans un répertoire neuf. Pour comparer deux
  builds, la commande `ninja -C <build> -t commands <cible>` montre les options réellement
  utilisées.
- **Un build neuf n'a pas les options voulues** : `OMP` vaut `OFF` par défaut, et un banc sans
  OpenMP tourne sur un seul cœur. Vérifier `ldd $BIN | grep gomp`.
- **L'assembleur trop ancien** pour `-mcpu=native` avec GCC 16 (« unknown architecture ») :
  charger des `binutils` récents.
- **Une référence `std::vector` initialisée sur un seul thread** : `std::vector<T> v(n)` écrit
  ses zéros depuis le thread qui le construit, donc la création est séquentielle et toutes les
  pages sont sur le nœud NUMA de ce thread ; les boucles parallèles qui suivent lisent de la
  mémoire distante, et les tableaux par chunks, qui n'allouent rien à la création, paraissent
  plus rapides qu'ils ne sont. La référence de `shrink_bench` met à zéro en parallèle
  (`examples/shrink_bench/baseline_vector.h`) : la garder en ajoutant une phase ou un type, et
  ne pas comparer ses temps de création avec des runs antérieurs à ce changement.
- **La taille d'un tableau calculée en `int`** : 8 × 10⁹ déborde et devient négatif. Les
  cibles à indice 64 bits (`_int64`) n'ont pas ce problème.
