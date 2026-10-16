# shrink_gui : lancer et suivre les runs sur le calculateur

> English version: [shrink_gui.md](shrink_gui.md).

> Octobre 2026, `shrink_gui` 0.13 (prototype). Une fenêtre sur le portable pour soumettre
> `compile_all.sh`, `bench_and_test.py`, `multi_launch_stress.py` et `omp_speedup.py` sur le calculateur, ou
> pour les lancer sur le portable lui-même (§3, « Cette machine »), suivre
> leur log, lire leur résumé et rapatrier leurs résultats. Le détail du code est dans la
> docstring de `python/shrink_gui/__init__.py`.

***

## 1. Principe

Tout passe par `ssh <calculateur>`, donc par la connexion maître (ControlMaster) que vous
ouvrez vous-même : l'interface ne manipule jamais de mot de passe. Rien n'est installé sur le
calculateur. À chaque appel, un petit programme Python (`remote_helper.py`, compatible
Python 3.6) est envoyé sur l'entrée de ssh et exécuté par le `python3` du nœud de login.

Un **run** est un dossier sur le calculateur, `<dossier des runs>/<id>/` :

| fichier | contenu |
|---|---|
| `meta.json` | ce qui a été demandé : programme, partition, commande, mode, commit |
| `run.sh` | ce qui tourne sur le nœud de calcul : dépôt, `site_env.sh`, commande |
| `job.sh` | le script batch donné à `ccc_msub` (directives `#MSUB`) |
| `launch.sh` | mode détaché : `ccc_mprun <options> bash run.sh` |
| `out.log` | tout ce que le job a écrit |
| `jobid` | (`ccc_msub`) l'identifiant batch, pour `squeue` et `ccc_mdel` |
| `start_time` | l'heure de démarrage sur le nœud (secondes depuis l'époque), pour la colonne Durée |
| `exit_code` | écrit quand la commande se termine |
| `events.jsonl` | (`bench_and_test.py`) l'avancement structuré : phases, k/N |

## 2. Prérequis et lancement

- Sur le portable : Python 3.9 ou plus récent et **PySide6** (`pip install PySide6`).
- Sur le calculateur : Python 3.6 ou plus récent, le dépôt shrink-array cloné, et la commande de
  soumission du site pour le mode batch (`ccc_msub` sur inti ; voir §6).
- Une connexion maître ssh ouverte, une fois par session :

```sh
ssh -fNM inti          # le nom d'hôte de ~/.ssh/config, avec ControlMaster/ControlPath
```

Puis, sur le portable :

```sh
cd shrink-array/python && python3 -m shrink_gui     # ou, depuis la racine : python3 python/shrink_gui
```

Au démarrage, l'interface vérifie la connexion. La ligne du haut indique le nœud, sa version
de Python, et le commit du dépôt du calculateur, comparé à celui de votre clone local si vous en
avez donné un (« Dépôt local »). Sans connexion maître, elle affiche la commande `ssh -fNM` à
lancer.

## 3. La fenêtre

### Où et comment

| champ | rôle |
|---|---|
| Partition | `gh200-bxi` (ARM, 288 cœurs) ou `milan-bxi` (x86, 64 cœurs) ; fixe aussi l'architecture des valeurs par défaut |
| Soumission | `ccc_msub` (batch, le défaut) ou `ccc_mprun` détaché depuis le nœud de login |
| Durée max (-T) | accepte `2 j`, `1 h 30`, `90 min`, `1-00:00:00` |
| QoS (-Q) | « auto » choisit la QoS d'après la durée, selon les limites lues par `ccc_mqinfo` |
| Démarrage | après un run `compile_all.sh` pas encore fini (dépendance Slurm `afterok`/`afterany`), et/ou pas avant une heure donnée |
| Builds à jour | avant la commande, `ninja -n` sur les builds utilisés : refuser (code 96), avertir, ou ne pas vérifier |
| Directives, Lanceur (job) | les lignes `#MSUB` et le lanceur sur le nœud ; propres à la partition et mémorisés |
| Dépôt (inti), Runs (inti) | le dépôt shrink-array et le dossier des runs, sur le calculateur |
| Git (inti) | branche, commit, modifications locales, avance ou retard sur la branche suivie ; boutons `git pull` et `git pull --rebase` (§5) |
| Résultats (local), Dépôt local | où « Récupérer » copie sur le portable ; votre clone local, pour comparer les commits |

### Quoi

Le programme à lancer, et ses options **lues dans le script lui-même** sur le calculateur
(« Lire les options ») : l'`argparse` des scripts Python, l'`usage()` de `compile_all.sh`. Une
option ajoutée au script apparaît donc dans le formulaire sans toucher à l'interface. Les options
modifiées sont en gras ; une option laissée à sa valeur par défaut n'est pas mise sur la ligne de
commande. « Enregistrer comme défauts » remplace les valeurs par défaut de ce programme sur
cette architecture. « Commande » montre exactement ce qui sera exécuté.

`compile_all.sh` ne se lance qu'en batch, jamais sur le nœud de login. Ses champs `--target`
et `--exec` ont un bouton « Choisir… » qui liste les configurations et les exécutables
trouvés sur le calculateur.

`omp_speedup.py` (courbes de speedup OpenMP d'exécutables shrink_bench sur une cible :
`--target`, `--exec` ou `--all-exec`, `--min-threads`, `--max-threads`, `--multiply`) se lance
de même en batch seulement, avec les mêmes boutons « Choisir… ». Son avancement et ses erreurs
s'affichent comme ceux de `bench_and_test.py` ; le Résumé donne son `summary.csv`, et le
Graphique ses courbes (speedup avec la droite idéale, efficacité ou temps ; couleur = la
variante, forme = la taille). « Comparer avec » superpose un autre run (un autre compilateur,
un autre commit) et donne le rapport B / A. Ses sorties vont dans
`$CCCWORKDIR/omp_speedup/` (sinon `~/omp_speedup/`).

### Les runs

Le tableau liste les runs du dossier des runs, rafraîchi toutes les 15 s : programme,
partition, commit, job, statut, avancement (k/N et temps restant quand le programme les
donne), erreurs (en direct pour `bench_and_test.py`, runs en échec pour
`multi_launch_stress.py`), durée (depuis le démarrage sur le nœud, à la seconde tant que le
run tourne, puis la durée totale une fois fini ; l'infobulle donne le début, la fin et la
durée max demandée), dernière ligne du log. Une notification signale un run qui se
termine, et la première erreur d'un run en cours.

| onglet | contenu |
|---|---|
| Log | le log du run sélectionné, suivi en direct ; recherche avec Ctrl+F |
| Résumé | `multi_launch_stress.py` : par build (runs, échecs, temps) ; `bench_and_test.py` : ses phases (durée, erreurs) ou son `bench_summary.csv` |
| Graphique | `multi_launch_stress.py` : le temps de chaque run selon le nombre de threads, filtrable par build et par variante ; « Comparer avec » superpose un autre run. `bench_and_test.py` : son `bench_summary.csv`, une courbe par variante selon le nombre de threads pour une config (temps, mémoire pic ou speedup) ; « Comparer avec » superpose un autre run de la même architecture en pointillés et donne le rapport B / A (moyenne géométrique sur les points communs, et le pire point) |
| Erreurs | le log d'erreurs lu dans le dossier de sortie : `bench_errors.log` pour `bench_and_test.py` (une entrée par échec, dès qu'il survient), les `error_*.txt` des runs en échec pour `multi_launch_stress.py` ; l'onglet passe en rouge s'il y a des erreurs. Pour `bench_and_test.py`, la liste à gauche donne chaque échec (`failed_cases.jsonl`) ; un clic affiche sa sortie complète (`run_logs/`). « Relancer les échecs… » soumet un run `--rerun-failed --rerun-repeat N` du run affiché, même partition et mêmes options, ses logs à part dans `run_logs/rerun-<date>` |

| bouton | effet |
|---|---|
| Arrêter | `ccc_mdel` du job, ou arrêt de `ccc_mprun` en mode détaché |
| Récupérer les résultats | copie dans « Résultats (local) » le dossier du run et les fichiers de résumé du programme (CSV, log, erreurs ; jamais les `output_*`) |
| Reprendre ses options | remet dans le formulaire le programme, la partition et les options du run |
| Supprimer | supprime le dossier du run (run terminé seulement) ; au choix, aussi les sorties propres à ce run (dans un dossier de sortie partagé, `bench_results/<arch>` : les fichiers qu'il a écrits, jamais ceux des runs plus anciens ni `oracle/`) et la copie locale |

### Cette machine

La case « cette machine », à côté du champ de l'hôte, envoie les runs sur la machine où tourne
la fenêtre au lieu du calculateur : ni ssh ni gestionnaire de jobs. Le helper passe par
`bash -c` ici (depuis Windows, par `wsl.exe` : dans WSL), et le run tourne en arrière-plan.
Il ne dépend pas de la fenêtre : la fermer laisse le run continuer, comme un job batch.

- **Partition** : une seule, « cette machine », avec l'architecture, le nombre de cœurs et la
  mémoire trouvés à la vérification de la connexion. Ses défauts sont les siens
  (`ARCH_DEFAULTS[(programme, "local")]`). La QoS et les directives sont celles du
  gestionnaire de jobs : masquées.
- **Soumission** : « en arrière-plan sur cette machine ». `launch.sh` lance `run.sh`, sous
  `timeout` quand une durée max est donnée (code 124 au-delà).
- **Lanceur** : un préfixe de `run.sh`, mémorisé. Pour plafonner la mémoire d'un run, et
  qu'un run trop gourmand ne fasse pas tomber WSL :
  `systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0`.
- **Dépôt, Runs** : distincts de ceux du calculateur (par défaut `~/shrink-array` et
  `~/shrink_runs` de cette machine) ; passer de l'un à l'autre garde les deux.
- **Démarrage, Builds à jour, Arrêter, Récupérer, Supprimer** : comme sur le calculateur.
  « Démarrage : après » attend un run `compile_all.sh` de cette machine ; « Arrêter » termine
  le groupe de processus du run.
- `compile_all.sh` tourne aussi ici : il n'est refusé que depuis le nœud de login du
  calculateur.

## 4. Exemples

**Compiler, puis tester une fois la compilation réussie.** Programme `compile_all.sh`, mode
`ccc_msub`, « Soumettre ». Puis programme `bench_and_test.py`, « Démarrage : après » ce run
de compilation, « si elle réussit », « Soumettre ». Le second job attend le premier dans la
file Slurm ; s'il échoue, le second reste en attente et l'interface le signale.

**Une campagne de stress sur x86.** Partition `milan-bxi`, programme
`multi_launch_stress.py`, « Défauts de l'architecture » (8 à 64 threads, builds de
`build_x86_64.txt`), « Soumettre ». L'onglet Graphique se remplit au fil des runs.

**Compiler et tester sur le portable.** Cocher « cette machine », programme `compile_all.sh`
avec son `--target`, « Soumettre » ; puis `bench_and_test.py`, « Démarrage : après » ce run,
« Soumettre ». Avec le lanceur `systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0`,
un run qui demande plus de mémoire est tué seul.

Les lignes de commande équivalentes, sans interface, sont dans
[example_bench_test.txt](example_bench_test.txt) et
[example_multi_launch_stress.txt](example_multi_launch_stress.txt).

## 5. Mettre à jour le dépôt du calculateur

La ligne « Git (inti) » est remplie à chaque vérification de la connexion. Son avance et son
retard datent du dernier `fetch` ; un `git pull` les remet à jour.

- `git pull` et `git pull --rebase` tournent sans invite (pas de mot de passe demandé, qui
  bloquerait l'appel ssh), avec une limite de 5 minutes.
- L'interface demande confirmation, et prévient quand des runs sont encore actifs : un build
  ou un bench en cours pourrait lire des sources modifiées en cours de route.
- Un conflit **annule** l'opération (`git merge --abort` ou `git rebase --abort`) : le dépôt
  reste tel qu'il était, et une fenêtre montre la sortie de git.
- Si le commit a changé, la connexion est revérifiée et les options du script sont relues.

## 6. Adapter l'interface au site

Ce qui dépend du site est dans un fichier TOML, le premier trouvé de :

1. le fichier nommé par la variable d'environnement `SHRINK_GUI_SITE` ;
2. `~/.config/shrink_gui/site.toml` (`$XDG_CONFIG_HOME/shrink_gui/site.toml`) ;
3. `python/shrink_gui/sites/cea_inti.toml`, inti du CEA, le défaut.

Pour en écrire un, copier un exemple dans `~/.config/shrink_gui/site.toml` et l'éditer :
`sites/cea_inti.toml` (`ccc_msub`, `#MSUB`, `ccc_mprun` ; chaque clé commentée) ou
`sites/slurm_example.toml` (Slurm standard : `sbatch`, `#SBATCH`, `scancel`, sans mode détaché).

| table | contenu |
|---|---|
| `[site]` | le nom affiché dans l'interface, l'hôte ssh, le dépôt, le dossier des runs, la racine des builds, la partition au démarrage |
| `[batch]` | l'ordonnanceur : libellé du mode, commandes de soumission, d'annulation et des QoS, motif du numéro de job, préfixe des directives et gabarits de leurs lignes (nom, durée, sortie, erreur, QoS, conditions de démarrage Slurm) |
| `[detached]` | la commande du mode détaché et ses options de durée et de QoS ; absente si le site n'en a pas |
| `[qos]` | les limites des QoS en attendant que la vérification de la connexion les lise |
| `[partitions.<nom>]` | libellé, architecture, directives, lanceur, options du mode détaché |
| `[defaults."<programme>".<arch>]` | les valeurs par défaut des options par programme et architecture (`{build_root}` est celle du site) |

Un fichier inutilisable (syntaxe, clé inconnue ou manquante, champ de gabarit inconnu) arrête
la fenêtre avec un message qui nomme le fichier et la clé. Le fichier est lu par `tomllib`
(Python 3.11+), `tomli` s'il est installé, sinon par un petit lecteur intégré du sous-ensemble
qu'utilisent les exemples.

Les jobs batch sont des jobs Slurm : l'interface les suit par `squeue`, et les conditions de
démarrage sont des options Slurm (`--dependency`, `--begin`).

Ce qui dépend du dépôt reste dans `python/shrink_gui/config.py` :

| nom | contenu |
|---|---|
| `PROGRAMS` | les programmes : script, répertoire de travail, lecture des options, motifs d'avancement et de sortie |
| `PRESETS["local"]`, `LOCAL_DEFAULTS` | cette machine : sa « partition », son dépôt et son dossier des runs |
| `LOCAL_SHELL` | comment le helper tourne sur cette machine (`bash -c`, ou `wsl.exe -e bash -c` sous Windows) |
| `SHRINK_GUI_SSH` (variable d'environnement) | le binaire ssh à utiliser (défaut `ssh`) |

Les valeurs saisies dans la fenêtre sont mémorisées entre deux sessions (QSettings). La langue
se change en haut à droite (français ou anglais).

## 7. Tests

```sh
cd python && python3 -m pytest shrink_gui/tests -q
```

Les tests tournent sans écran (`QT_QPA_PLATFORM=offscreen`) contre un faux calculateur local :
de faux `ssh`, `ccc_msub`, `squeue`, `ccc_mdel`, `ccc_mprun` et `ccc_mqinfo` (`tests/fakes`), avec
le fichier de site d'inti quel que soit le contenu de `~/.config` ; `test_site.py` fait aussi
tourner un site Slurm standard (faux `sbatch` et `scancel`).
Ils sont sautés si PySide6 n'est pas installé.

## 8. Limites

- C'est un prototype, écrit pour les machines du TGCC ; un autre site écrit son fichier de
  site (§6). L'ordonnanceur doit être Slurm en dessous (`squeue`, `--dependency`, `--begin`).
- Une seule connexion à la fois. L'interface n'ouvre pas la connexion maître elle-même.
- « Cette machine » depuis Windows (`wsl.exe`) n'est pas couverte par les tests (ils tournent
  sous Linux). WSL peut arrêter une distribution quand plus aucune de ses fenêtres n'est
  ouverte : en garder une ouverte pendant un run.
