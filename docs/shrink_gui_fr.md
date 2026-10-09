# shrink_gui : lancer et suivre les runs sur le calculateur

> English version: [shrink_gui.md](shrink_gui.md).

> Octobre 2026, `shrink_gui` 0.11 (prototype). Une fenêtre sur le portable pour soumettre
> `compile_all.sh`, `bench_and_test.py` et `multi_launch_stress.py` sur le calculateur, suivre
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
| `exit_code` | écrit quand la commande se termine |
| `events.jsonl` | (`bench_and_test.py`) l'avancement structuré : phases, k/N |

## 2. Prérequis et lancement

- Sur le portable : Python 3.9 ou plus récent et **PySide6** (`pip install PySide6`).
- Sur le calculateur : Python 3.6 ou plus récent, le dépôt shrink-array cloné, et `ccc_msub`
  pour le mode batch.
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

### Les runs

Le tableau liste les runs du dossier des runs, rafraîchi toutes les 15 s : programme,
partition, commit, job, statut, avancement (k/N et temps restant quand le programme les
donne), dernière ligne du log. Une notification signale un run qui se termine.

| onglet | contenu |
|---|---|
| Log | le log du run sélectionné, suivi en direct ; recherche avec Ctrl+F |
| Résumé | `multi_launch_stress.py` : par build (runs, échecs, temps) ; `bench_and_test.py` : ses phases (durée, erreurs) ou son `bench_summary.csv` |
| Graphique | `multi_launch_stress.py` : le temps de chaque run selon le nombre de threads, filtrable par build et par variante ; « Comparer avec » superpose un autre run |

| bouton | effet |
|---|---|
| Arrêter | `ccc_mdel` du job, ou arrêt de `ccc_mprun` en mode détaché |
| Récupérer les résultats | copie dans « Résultats (local) » le dossier du run et les fichiers de résumé du programme (CSV, log, erreurs ; jamais les `output_*`) |
| Reprendre ses options | remet dans le formulaire le programme, la partition et les options du run |
| Supprimer | supprime le dossier du run (run terminé seulement) ; au choix, aussi les sorties propres à ce run et la copie locale |

## 4. Exemples

**Compiler, puis tester une fois la compilation réussie.** Programme `compile_all.sh`, mode
`ccc_msub`, « Soumettre ». Puis programme `bench_and_test.py`, « Démarrage : après » ce run
de compilation, « si elle réussit », « Soumettre ». Le second job attend le premier dans la
file Slurm ; s'il échoue, le second reste en attente et l'interface le signale.

**Une campagne de stress sur x86.** Partition `milan-bxi`, programme
`multi_launch_stress.py`, « Défauts de l'architecture » (8 à 64 threads, builds de
`build_x86_64.txt`), « Soumettre ». L'onglet Graphique se remplit au fil des runs.

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

Tout ce qui dépend du site est dans `python/shrink_gui/config.py` :

| nom | contenu |
|---|---|
| `PRESETS` | les partitions : directives `#MSUB`, lanceur, options `ccc_mprun` |
| `PROGRAMS` | les programmes : script, répertoire de travail, lecture des options, motifs d'avancement et de sortie |
| `ARCH_DEFAULTS` | les valeurs par défaut par (programme, architecture) |
| `DEFAULTS` | hôte, dépôt, dossier des runs, dossier local, partition et programme au démarrage |
| `SHRINK_GUI_SSH` (variable d'environnement) | le binaire ssh à utiliser (défaut `ssh`) |

Les valeurs saisies dans la fenêtre sont mémorisées entre deux sessions (QSettings). La langue
se change en haut à droite (français ou anglais).

## 7. Tests

```sh
cd python && python3 -m pytest shrink_gui/tests -q
```

Les tests tournent sans écran (`QT_QPA_PLATFORM=offscreen`) contre un faux calculateur local :
de faux `ssh`, `ccc_msub`, `squeue`, `ccc_mdel`, `ccc_mprun` et `ccc_mqinfo` (`tests/fakes`).
Ils sont sautés si PySide6 n'est pas installé.

## 8. Limites

- C'est un prototype, écrit pour les machines du TGCC et leurs commandes `ccc_*`
  (`ccc_msub`, `ccc_mprun`, `ccc_mqinfo`). Un autre site adapte `config.py` et les commandes de soumission de
  `runscripts.py`.
- Une seule connexion à la fois. L'interface n'ouvre pas la connexion maître elle-même.
