# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""French / English: T() and its tables. The interface is written in French."""
import html
import re

# --------------------------------------------------------------------------
# Language. The interface is written in French; T() gives the English text
# when LANG == "en": an exact entry of EN, else the EN_RE substitutions (the
# messages built on the fly, those sent back by inti included). The run
# statuses stay French internally -- the code compares them -- and are only
# translated where they are shown.
# --------------------------------------------------------------------------
LANG = "fr"

EN = {
    # window, groups, form rows
    "shrink-array — runs sur inti (prototype)": "shrink-array — runs on inti (prototype)",
    "Calculateur :": "Machine:", "Vérifier la connexion": "Check connection",
    "Où et comment": "Where and how", "Quoi": "What", "Partition": "Partition",
    "Soumission": "Submission", "Durée max (-T)": "Max time (-T)", "Directives": "Directives",
    "Lanceur (job)": "Launcher (job)", "Dépôt (inti)": "Repository (inti)", "Git (inti)": "Git (inti)",
    "pas de dépôt git (ou pas encore de connexion)": "no git repository (or not connected yet)",
    "pas de dépôt git sur inti : vérifiez la connexion": "no git repository on inti: check the connection",
    "Branche (inti)": "Branch (inti)", "Changer de branche": "Switch branch",
    "Relire les branches": "Reload branches",
    "Extrait dans le dépôt d'inti la branche choisie dans la liste. Une branche distante (origin/x) sans "
    "branche locale donne une branche locale x qui la suit. Refusé si le dépôt a des modifications non "
    "commitées. git pull agit ensuite sur cette branche.":
        "Checks out, in inti's repository, the branch chosen in the list. A remote branch (origin/x) with "
        "no local branch gives a local branch x tracking it. Refused when the repository has uncommitted "
        "changes. git pull then acts on that branch.",
    "git fetch dans le dépôt d'inti, puis relit la liste des branches.":
        "git fetch in inti's repository, then reloads the branch list.",
    " (actuelle)": " (current)",
    "git fetch a échoué sur inti : la liste est celle du dernier fetch":
        "git fetch failed on inti: the list is the one of the last fetch",
    "changement de branche refusé ou en échec": "branch switch refused or failed",
    "git pull dans le dépôt d'inti (fusion si les branches ont divergé). "
    "Un conflit annule le pull : le dépôt reste tel qu'il était.":
        "git pull in inti's repository (a merge if the branches have diverged). "
        "A conflict cancels the pull: the repository stays as it was.",
    "git pull --rebase dans le dépôt d'inti : vos commits locaux "
    "d'inti sont rejoués au-dessus de la branche distante. Un conflit "
    "annule le rebase : le dépôt reste tel qu'il était.":
        "git pull --rebase in inti's repository: your local commits on inti are replayed on top "
        "of the remote branch. A conflict cancels the rebase: the repository stays as it was.",
    "Runs (inti)": "Runs (inti)", "Résultats (local)": "Results (local)",
    "Langue": "Language",
    "ccc_msub (batch)": "ccc_msub (batch)", "ccc_mprun détaché": "ccc_mprun, detached",
    "vide : la commande tourne directement dans le job": "empty: the command runs directly in the job",
    "Lire les options": "Read options", "filtrer les options (nom ou aide)…": "filter the options (name or help)…",
    "Défauts de l'architecture": "Architecture defaults", "Enregistrer comme défauts": "Save as defaults",
    "arguments en plus, tels quels (facultatif)": "extra arguments, as they are (optional)",
    "Commande :": "Command:", "Soumettre": "Submit",
    # runs
    "Run": "Run", "Programme": "Program", "Job": "Job", "Statut": "Status",
    "Avancement": "Progress", "Dernière ligne": "Last line", "Durée": "Duration",
    "Pas encore démarré (ou début inconnu).": "Not started yet (or start unknown).",
    "démarré : ": "started: ", "fini : ": "ended: ", "durée : ": "duration: ",
    "en cours depuis ": "running for ", "durée max demandée : ": "max time asked: ",
    "Rafraîchir": "Refresh", "Arrêter": "Stop", "Récupérer les résultats": "Fetch results",
    "Reprendre ses options": "Reuse its options", "Supprimer": "Delete",
    "Log": "Log", "Résumé": "Summary", "Graphique": "Chart",
    "Erreurs": "Errors", "Rafraîchir les erreurs": "Refresh errors",
    "rechercher dans les erreurs (Entrée : suivant)": "search the errors (Enter: next)",
    "(vide)": "(empty)",
    "Pas de décompte d'erreurs pour ce run.": "No error count for this run.",
    "temps": "time", "mémoire": "memory", "speedup": "speedup", "Config": "Config", "Mesure": "Measure",
    "Échelle log": "Log scale",
    "efficacité": "efficiency", "idéal": "ideal",
    "speedup vs le premier nombre de threads": "speedup vs the first thread count",
    "efficacité (speedup × min / threads)": "efficiency (speedup × min / threads)",
    "meilleur temps (s)": "best time (s)",
    "Superpose un autre run de omp_speedup.py de la même architecture (en pointillés) -- un autre "
    "compilateur, un autre commit -- et donne le rapport B / A sur leurs points communs":
        "Overlays another omp_speedup.py run of the same architecture (dashed) -- another compiler, another "
        "commit -- and gives the B / A ratio over their common points",
    "tailles ou modes différents : comparer les speedups, pas les temps":
        "different sizes or modes: compare the speedups, not the times",
    "couleur : la variante (vector en gris) · forme : la taille (rond dyn, carré 1024, triangle 256, losange 64)":
        "colour: the variant (vector in grey) · shape: the size (dot dyn, square 1024, triangle 256, diamond 64)",
    "Superpose un autre run de bench_and_test.py de la même architecture (en pointillés) et donne le "
    "rapport B / A sur leurs points communs":
        "Overlays another bench_and_test.py run of the same architecture (dashed) and gives the "
        "B / A ratio over their common points",
    "temps moyen (s)": "mean time (s)", "mémoire pic (Mo)": "peak memory (MB)",
    "speedup vs la variante de référence": "speedup vs the baseline variant",
    "speedup : la variante de référence n'a pas de point (1 par définition)":
        "speedup: the baseline variant has no point (1 by definition)",
    "Relancer les échecs…": "Rerun failures…", "Relancer les échecs": "Rerun failures",
    "Répétitions": "Repetitions", "log d'erreurs": "error log",
    "pas de sortie complète enregistrée pour cet échec (une comparaison ou un garde, un bench_and_test.py "
    "plus ancien, ou --no-full-log)":
        "no complete output recorded for this failure (a comparison or a guard, an older bench_and_test.py, "
        "or --no-full-log)",
    "pas de sortie complète enregistrée pour cet échec": "no complete output recorded for this failure",
    "non rejouable par --rerun-failed : il faut un run complet": "not replayable by --rerun-failed: it takes a full run",
    "sortie complète d'une exécution en échec, trouvée dans le dossier des logs (écrite pendant ce run)":
        "complete output of a failed execution, found in the logs directory (written during this run)",
    "Chaque échec est rejoué jusqu'à N fois, arrêt au premier échec : pour un échec intermittent "
    "(--rerun-repeat).":
        "Each failure is replayed up to N times, stopping at the first failure: for an intermittent "
        "failure (--rerun-repeat).",
    "Le log d'erreurs du run, lu dans son dossier de sortie : bench_errors.log pour "
    "bench_and_test.py (une entrée par échec, écrite dès qu'il survient, avec la fin "
    "de sa sortie d'erreur ; la sortie complète de chaque exécution est dans run_logs/), "
    "les error_*.txt des runs en échec pour multi_launch_stress.py. Le plus récent "
    "d'abord, la fin de chaque fichier seulement.\n\nL'onglet passe en rouge "
    "quand il y a des erreurs.":
        "The run's error log, read in its output directory: bench_errors.log for bench_and_test.py "
        "(one entry per failure, written as soon as it happens, with the end of its error output; "
        "the full output of every execution is in run_logs/), the error_*.txt of the failed runs for "
        "multi_launch_stress.py. Newest first, the end of each file only.\n\nThe tab turns red "
        "when there are errors.",
    "rechercher dans le log (Entrée : suivant)": "search the log (Enter: next)",
    "Suivre la fin": "Follow the end", "Rafraîchir le résumé": "Refresh the summary",
    # chart controls
    "Build": "Build", "Variante": "Variant", "Taille": "Size", "Aucune": "None", "Couleur": "Colour",
    "Comparer avec": "Compare with", "(aucun)": "(none)", "tous": "all", "toutes": "all",
    "Temps en échelle log": "Log time scale", "Médiane par tranche de threads": "Median per thread bin",
    "Masquer les échecs": "Hide failures",
    "threads": "threads", "temps par run (s)": "time per run (s)", "médiane": "median",
    "tous les builds": "all builds", "autres": "others",
    # statuses (internal values stay French)
    "en cours": "running", "en attente": "pending", "démarrage": "starting", "suspendu": "suspended",
    "fini": "done", "arrêté": "stopped", "interrompu": "interrupted", "inconnu": "unknown",
    "inconnu (pas de squeue)": "unknown (no squeue)",
    "terminé sans code (limite de temps ? voir le log)": "ended without a code (time limit? see the log)",
    # summary columns
    "échecs": "failures", "moyenne s": "mean s", "médiane s": "median s",
    # misc
    "vérification…": "checking…", "?": "?",
    "Options non lues : « Lire les options » (connexion requise).":
        "Options not read: \"Read options\" (needs the connection).",
    "(pas d'aide dans le script)": "(no help in the script)",
    "erreur inconnue (voir la barre d'état)": "unknown error (see the status bar)",
    "arrêtez d'abord ce run (il est encore actif)": "stop this run first (it is still active)",
    "pas de données": "no data",
    "Options non lues": "Options not read", "Run déjà actif": "Run already active",
    # tooltips: controls
    "Nom d'hôte ssh du calculateur (tel qu'il figure dans ~/.ssh/config).\n\n"
    "Tout passe par la connexion maître que vous ouvrez vous-même : ssh -fNM inti.":
        "ssh host name of the machine (as in ~/.ssh/config).\n\n"
        "Everything goes through the master connection you open yourself: ssh -fNM inti.",
    "La partition où le run est soumis. Elle fixe aussi l'architecture (ARM ou x86), "
    "donc les valeurs par défaut des options.":
        "The partition the run is submitted to. It also sets the architecture (ARM or x86), "
        "hence the options' default values.",
    "ccc_msub : job batch, indépendant de cette fenêtre et du nœud de login "
    "(recommandé).\n\nccc_mprun détaché : lancé depuis le nœud de login ; il dépend de ce "
    "nœud tant qu'il tourne.\n\nEn arrière-plan sur cette machine : sans gestionnaire de "
    "jobs, indépendant de cette fenêtre ; la durée max est appliquée par timeout (code 124).":
        "ccc_msub: batch job, independent of this window and of the login node "
        "(recommended).\n\nccc_mprun, detached: started from the login node; it depends on that "
        "node while it runs.\n\nIn the background on this machine: no job scheduler, independent "
        "of this window; the maximum duration is applied by timeout (code 124).",
    "En arrière-plan sur cette machine : sans gestionnaire de jobs, indépendant de cette fenêtre ; "
    "la durée max est appliquée par timeout (code 124).":
        "In the background on this machine: no job scheduler, independent of this window; the maximum "
        "duration is applied by timeout (code 124).",
    "Lancer les runs sur cette machine plutôt que sur le calculateur : pas de ssh ni de "
    "gestionnaire de jobs, le run tourne en arrière-plan ici (sous Windows, dans WSL). "
    "Dépôt et dossier des runs sont mémorisés à part.":
        "Run on this machine instead of the cluster: no ssh and no job scheduler, the run goes in the "
        "background here (in WSL on Windows). The repository and the runs directory are kept apart.",
    "en arrière-plan sur cette machine": "in the background on this machine",
    "vide : run.sh directement ; pour plafonner la mémoire : systemd-run --user --scope -q "
    "-p MemoryMax=16G -p MemorySwapMax=0":
        "empty: run.sh directly; to cap the memory: systemd-run --user --scope -q "
        "-p MemoryMax=16G -p MemorySwapMax=0",
    "ccc_msub : job batch, indépendant de cette fenêtre et du nœud de login (recommandé).":
        "ccc_msub: batch job, independent of this window and of the login node (recommended).",
    "ccc_mprun détaché : lancé depuis le nœud de login ; il dépend de ce nœud tant qu'il tourne.":
        "ccc_mprun, detached: started from the login node; it depends on that node while it runs.",
    "Durée maximale du job (-T). Au-delà, le job est tué par le gestionnaire et finit "
    "« terminé sans code ». Se tape « 1 h 30 », « 90 min », « 2 j » ou « 1-00:00:00 ».\n\n"
    "Mémorisée par programme : compile_all.sh court, bench_and_test.py quelques heures, "
    "multi_launch_stress.py long.":
        "Maximum job time (-T). Beyond it the batch system kills the job, which ends \"ended "
        "without a code\". Type \"1 h 30\", \"90 min\", \"2 d\" or \"1-00:00:00\".\n\n"
        "Remembered per program: compile_all.sh short, bench_and_test.py a few hours, "
        "multi_launch_stress.py long.",
    "La QoS du job, dont dépend la durée maximale : la liste a les QoS de la partition, "
    "avec leurs limites (ccc_mqinfo).\n\n« auto » prend la plus prioritaire qui accepte la "
    "durée, puis la plus courte. Une QoS écrite dans les directives l'emporte.\n\nLes limites "
    "sont relues sur inti à chaque vérification de la connexion.":
        "The job's QoS, which sets the maximum time: the list holds the partition's QoS, with "
        "their limits (ccc_mqinfo).\n\n\"auto\" takes the highest priority one that accepts the "
        "duration, then the shortest. A QoS written in the directives wins.\n\nThe limits are "
        "read again on inti at each connection check.",
    "Choisir…": "Choose…", "filtrer…": "filter…", "En plus :": "Also:", "Tout décocher": "Clear all",
    "OK": "OK", "Annuler": "Cancel",
    "Choisir dans la liste de ce que compile_all.sh sait construire (lue sur inti, sans rien compiler).":
        "Choose from what compile_all.sh can build (read on inti, nothing is compiled).",
    "noms ou motifs tapés à la main, séparés par des virgules": "names or patterns typed by hand, comma-separated",
    "un nom ou un motif tapé à la main": "one name or pattern typed by hand",
    "Ce compile_all.sh n'accepte qu'un motif à la fois : appliquez compile_lists.py pour en cocher plusieurs.":
        "This compile_all.sh takes one pattern at a time: apply compile_lists.py to tick several.",
    "--target : configurations": "--target: configurations", "--exec : exécutables": "--exec: executables",
    "dossier sans build.ninja": "directory without build.ninja",
    "auto (selon la durée)": "auto (from the duration)", "QoS (-Q)": "QoS (-Q)",
    "prioritaire": "higher priority", " (fixée dans les directives)": " (set in the directives)",
    "Lignes #MSUB du script batch (ou options ccc_mprun en mode détaché). Propres à "
    "la partition, mémorisées.\n\n-T, -r, -o et -e sont ajoutés d'après le formulaire.":
        "#MSUB lines of the batch script (or ccc_mprun options in detached mode). Per "
        "partition, remembered.\n\n-T, -r, -o and -e are added from the form.",
    "Le dépôt shrink-array SUR INTI. Les options sont lues dans ces scripts-là, et c'est "
    "là que le job s'exécute.":
        "The shrink-array repository ON INTI. The options are read from those scripts, and the "
        "job runs there.",
    "Où l'interface range ses runs sur inti : un dossier par run (log, scripts, "
    "code de retour, numéro de job).":
        "Where the interface keeps its runs on inti: one directory per run (log, scripts, "
        "exit code, job id).",
    "Où « Récupérer les résultats » copie les fichiers sur ce portable.":
        "Where \"Fetch results\" copies the files on this laptop.",
    "Le script à lancer. Ses options sont lues dans le script lui-même sur inti.":
        "The script to run. Its options are read from the script itself on inti.",
    "Ne montre que les options dont le nom ou l'aide contient ce texte.":
        "Shows only the options whose name or help contains this text.",
    "Arguments ajoutés tels quels à la fin de la commande : pour une option que le "
    "formulaire ne connaît pas (--no-gnu de compile_all.sh, par exemple).":
        "Arguments appended as they are to the command: for an option the form does not "
        "know (compile_all.sh's --no-gnu, for instance).",
    "La commande exacte qui sera exécutée sur le nœud. Seules les options modifiées "
    "(en gras) y figurent : les autres gardent le défaut du script.":
        "The exact command run on the node. Only the changed options (in bold) are in it: "
        "the others keep the script's default.",
    "Les runs du dossier « Runs (inti) », du plus récent au plus ancien. Cliquez sur un "
    "run pour voir son log, son résumé et son graphique.\n\nRafraîchi toutes les 15 s.":
        "The runs of the \"Runs (inti)\" directory, newest first. Click a run to see its log, "
        "summary and chart.\n\nRefreshed every 15 s.",
    "Recherche dans le log (Ctrl+F). Entrée : occurrence suivante, avec retour au début.":
        "Search the log (Ctrl+F). Enter: next match, wrapping to the start.",
    "Fait défiler le log jusqu'à la fin à chaque nouvelle ligne. Décochée, le log "
    "reste où vous êtes.":
        "Scrolls the log to the end on every new line. Unchecked, the log stays where you are.",
    "Ne montre que les runs de ce build.": "Shows only the runs of this build.",
    "Ne montre que les runs de cette variante (chunk, chunk2, chunk2a).":
        "Shows only the runs of this variant (chunk, chunk2, chunk2a).",
    "Ce que la couleur distingue : la variante, la taille de chunk, ou rien. Trois "
    "couleurs au plus ; le reste est groupé en gris « autres ».":
        "What the colour tells apart: the variant, the chunk size, or nothing. Three colours "
        "at most; the rest is grouped in grey as \"others\".",
    "Axe du temps en échelle logarithmique : utile quand les temps s'étalent sur "
    "plusieurs ordres de grandeur.":
        "Logarithmic time axis: useful when the times span several orders of magnitude.",
    "Médiane du temps par tranche de threads (au moins 3 runs par tranche) : la "
    "tendance malgré la dispersion.":
        "Median time per thread bin (at least 3 runs per bin): the trend despite the spread.",
    "Retire les runs en échec (triangles) du graphique.": "Removes the failed runs (triangles) from the chart.",
    # tooltips: buttons
    "Vérifie la connexion maître ssh et le python3 d'inti, puis rafraîchit la liste des runs.":
        "Checks the ssh master connection and inti's python3, then refreshes the run list.",
    "Relit les options du script sur inti : à faire après avoir modifié ou mis à jour le script.":
        "Reads the script's options again on inti: do it after changing or updating the script.",
    "Relit les options du script sur inti (après une modification du script)":
        "Reads the script's options again on inti (after a change to the script)",
    "Remet les options aux valeurs par défaut de l'architecture de la partition (ARM ou x86).":
        "Resets the options to the defaults of the partition's architecture (ARM or x86).",
    "Enregistre les options actuelles comme valeurs par défaut de ce programme sur cette architecture.":
        "Saves the current options as this program's defaults on this architecture.",
    "Crée le dossier du run sur inti et soumet le job.": "Creates the run directory on inti and submits the job.",
    "Relit maintenant (sinon automatique).": "Reads again now (otherwise automatic).",
    "Annule le job du run sélectionné (ccc_mdel), ou arrête ccc_mprun en mode détaché.":
        "Cancels the selected run's job (ccc_mdel), or stops ccc_mprun in detached mode.",
    "Copie sur ce portable le dossier du run et les fichiers de résultats du programme "
    "(CSV, log, erreurs ; jamais les output_*).":
        "Copies the run directory and the program's result files (CSV, log, errors; never the "
        "output_*) to this laptop.",
    "Remet dans le formulaire le programme, la partition et les options du run sélectionné, "
    "pour le relancer ou le modifier.":
        "Puts the selected run's program, partition and options back in the form, to run it "
        "again or change it.",
    "Remet dans le formulaire le programme, la partition et les options de ce run":
        "Puts this run's program, partition and options back in the form",
    "Supprime le dossier du run sur inti ; au choix, aussi les sorties du "
    "programme propres à ce run et la copie locale. Seulement pour un run terminé.":
        "Deletes the run directory on inti; optionally also the program outputs that are "
        "this run's own, and the local copy. Finished runs only.",
    "Supprime le dossier du run sur inti (au choix : ses sorties, la copie locale)":
        "Deletes the run directory on inti (optionally: its outputs, the local copy)",
    "aussi le dossier de sortie du programme, propre à ce run":
        "also the program's output directory, this run's own",
    "pas de sorties propres à ce programme": "no outputs of its own for this program",
    "pas de copie locale": "no local copy",
    "Relit le fichier résumé du run sélectionné.": "Reads the selected run's summary file again.",
    "ccc_msub : ce qui lance run.sh sur le nœud réservé (le lanceur de la "
    "partition dans le fichier de site ; vide : directement dans le job).\n"
    "Mémorisé par partition.":
        "ccc_msub: what starts run.sh on the reserved node (the partition's launcher in the "
        "site file; empty: directly in the job).\n"
        "Remembered per partition.",
    "Superpose un autre run de multi_launch_stress.py : même build et même variante de "
    "part et d'autre":
        "Overlays another multi_launch_stress.py run: same build and same variant on both sides",
    # tooltips: lists, tabs
    "Stress : les binaires de check, en boucle, à un nombre de threads tiré au hasard. "
    "Résumé et graphique disponibles.":
        "Stress: the check binaries, over and over, at a random thread count. Summary and "
        "chart available.",
    "Bench et tests : validations ctest, checks, sanitizers, comparaison seq/omp et "
    "mesures de performance.":
        "Bench and tests: ctest validations, checks, sanitizers, seq/omp compare and "
        "performance measurements.",
    "Compilation de toutes les configurations (options lues dans son --help).":
        "Builds every configuration (options read from its --help).",
    "Compilation de toutes les configurations, en job batch (ccc_msub) ou sur cette machine, jamais "
    "depuis le nœud de login. Options lues dans le texte de son usage(), sans l'exécuter.":
        "Builds every configuration, as a batch job (ccc_msub) or on this machine, never from the login "
        "node. Options read from the text of its usage(), without running it.",
    "Le log du run sélectionné, suivi en direct.": "The selected run's log, followed live.",
    "Le résumé du run : par build pour multi_launch_stress.py (runs, échecs, temps), "
    "le CSV tel quel pour bench_and_test.py.":
        "The run's summary: per build for multi_launch_stress.py (runs, failures, times), "
        "the CSV as it is for bench_and_test.py.",
    "Temps de chaque run selon le nombre de threads (multi_launch_stress.py). Survolez "
    "un point pour le détail ; « Comparer avec » superpose un autre run.":
        "Time of each run against the thread count (multi_launch_stress.py). Hover a point "
        "for details; \"Compare with\" overlays another run.",
    "Commit": "Commit", "Dépôt local": "Local repository", "Phases": "Phases",
    "phase": "phase", "statut": "status", "durée s": "duration s", "erreurs": "errors", "début": "start",
    "ok": "ok", "exception": "exception", "interrompue": "interrupted", "fin": "end",
    "facultatif : pour comparer son commit à celui d'inti": "optional: to compare its commit with inti's",
    "pas de commit enregistré (pas de git, ou run plus ancien)": "no commit recorded (no git, or an older run)",
    "Votre clone du dépôt sur ce portable (facultatif). La ligne de connexion compare "
    "alors son commit à celui du dépôt d'inti : un écart veut dire qu'un push ou un "
    "git pull manque.":
        "Your clone of the repository on this laptop (optional). The connection line then compares "
        "its commit with the one of inti's repository: a difference means a push or a git pull is missing.",
    "bench_and_test.py : ses phases (durée, erreurs, d'après events.jsonl) ou son "
    "bench_summary.csv.":
        "bench_and_test.py: its phases (duration, errors, from events.jsonl) or its bench_summary.csv.",
    "Choix de la langue de l'interface (changement immédiat).":
        "Interface language (applied at once).",
    # start conditions and build check (Démarrage / Builds à jour)
    "Démarrage": "Start", "Builds à jour": "Builds up to date", "pas avant": "not before",
    "dès la soumission": "on submission", "si elle réussit": "if it succeeds",
    "quelle que soit son issue": "whatever its result",
    "refuser de démarrer si un build n'est pas à jour": "refuse to start if a build is not up to date",
    "avertir seulement": "warn only", "ne pas vérifier": "do not check",
    "Démarrer après la fin d'une compilation (run compile_all.sh pas encore fini).\n\n"
    "ccc_msub : dépendance Slurm (le job attend dans la file sans prendre de nœud). "
    "ccc_mprun détaché : le lancement attend le code de retour de ce run.\n\n"
    "« si elle réussit » : une compilation en échec, et le run ne démarre jamais "
    "(en batch il reste en attente : à arrêter).":
        "Start after the end of a compilation (a compile_all.sh run not finished yet).\n\n"
        "ccc_msub: a Slurm dependency (the job waits in the queue without holding a node). "
        "ccc_mprun, detached: the launch waits for that run's exit code.\n\n"
        "\"if it succeeds\": a failed compilation and the run never starts "
        "(in batch it stays pending: stop it).",
    "Ne pas démarrer avant ce délai, compté depuis la soumission "
    "(ccc_msub : --begin de Slurm ; ccc_mprun détaché : une attente avant le lancement).":
        "Do not start before this delay, counted from the submission "
        "(ccc_msub: Slurm's --begin; ccc_mprun, detached: a wait before the launch).",
    "Avant la commande, sur le nœud : ninja -n sur les exécutables déjà construits "
    "de chaque build que le programme utilisera (--configs de bench_and_test.py, "
    "--builds de multi_launch_stress.py, sous --build-root). Un build dont une "
    "source, un en-tête ou la configuration a changé depuis sa compilation n'est "
    "pas à jour.\n\n« refuser » : le run s'arrête (code 96) en listant ces builds.":
        "Before the command, on the node: ninja -n on the executables already built in "
        "each build the program will use (--configs of bench_and_test.py, --builds of "
        "multi_launch_stress.py, under --build-root). A build whose sources, headers or "
        "configuration changed since it was compiled is not up to date.\n\n\"refuse\": "
        "the run stops (code 96), listing those builds.",
}

# Messages built on the fly (and those inti sends back): regex on the French,
# English replacement. Applied in order, on the whole text.
EN_RE = [(re.compile(a), b) for a, b in (
    (r"cette machine", "this machine"), (r"(\d+) cœurs\b", r"\1 cores"), (r"(\d) Gio\b", r"\1 GiB"),
    (r"^shrink-array — runs sur (.+) \(prototype\)$", r"shrink-array — runs on \1 (prototype)"),
    (r"^Dépôt \((.+)\)$", r"Repository (\1)"),
    (r"^pas de dépôt git sur (.+) : vérifiez la connexion$", r"no git repository on \1: check the connection"),
    (r"python3 en échec sur this machine \(voir la barre d'état\)",
     "python3 failed on this machine (see the status bar)"),
    (r"\bson groupe de processus \(SIGTERM\)", "its process group (SIGTERM)"),
    (r"réponse illisible de (\S+) :", r"unreadable answer from \1:"),
    (r"<b>Options non lues\.</b>", "<b>Options not read.</b>"),
    (r"Corrigez puis cliquez sur « Lire les options »\. En attendant, le champ « arguments en plus » "
     r"passe les options telles quelles\.",
     "Fix it, then click \"Read options\". Meanwhile the \"extra arguments\" field passes options as they are."),
    (r"\(pas d'aide dans le script\)", "(no help in the script)"),
    (r"^la QoS (\S+) n'est pas ouverte sur (\S+)$", r"QoS \1 is not open on \2"),
    (r"^après (\S+)$", r"after \1"),
    (r"^en attente sans fin \((\S+) a échoué : à arrêter\)$", r"pending for ever (\1 failed: stop it)"),
    (r"^soumission refusée : ", "submission refused: "),
    (r"^le run (\S+) a échoué \(code (\S+)\) : rien soumis$", r"run \1 failed (code \2): nothing submitted"),
    (r"^le run (\S+) n'existe pas$", r"run \1 does not exist"),
    (r"^le run (\S+) n'a pas de job batch à attendre$", r"run \1 has no batch job to wait for"),
    (r"^(.+) dépasse le maximum de la QoS (\S+) sur (\S+) \((.+)\)$", r"\1 exceeds the maximum of QoS \2 on \3 (\4)"),
    (r"^(\d+) cœurs dépassent le maximum de la QoS (\S+) sur (\S+) \((\d+)\)$",
     r"\1 cores exceed the maximum of QoS \2 on \3 (\4)"),
    (r"^aucune QoS n'accepte (.+?) sur (\S+)", r"no QoS accepts \1 on \2"), (r" \(au plus (.+)\)$", r" (\1 at most)"),
    (r"^(\d+) job\(s\) soumis au plus$", r"at most \1 submitted job(s)"),
    (r"(\d+) j (\d\d) h", r"\1 d \2 h"),
    (r"^configurée, (\d+) exécutable\(s\), ", r"configured, \1 executable(s), "),
    (r"^motif : (\d+) configurations$", r"pattern: \1 configurations"),
    (r"^motif : (\d+) exécutables$", r"pattern: \1 executables"),
    (r"^dans (\d+)/(\d+) build\(s\)$", r"in \1/\2 build(s)"),
    (r"^lecture des builds de (\S+) sur (.+)…$", r"reading the \1 builds on \2…"),
    (r"^(\d+) configurations, (\d+) build\(s\) (\S+) dans (\S+)$", r"\1 configurations, \2 \3 build(s) in \4"),
    (r"^Configurations de compile_all\.sh \(cmake-build-<nom>\)\. « configurée » : son build existe déjà "
     r"pour (\S+) dans (\S+)\.$",
     r'compile_all.sh configurations (cmake-build-<name>). "configured": its build already exists for \1 in \2.'),
    (r"^Exécutables des (\d+) build\(s\) (\S+) configurés( retenus par --target=\S+)?\. Ninja construit aussi "
     r"leurs dépendances \(bibliothèques chunk\), pas les autres exécutables\.$",
     lambda m: f"Executables of the {m.group(1)} configured {m.group(2)} build(s)"
               + (m.group(3).replace(" retenus par", " kept by") if m.group(3) else "")
               + ". Ninja also builds their dependencies (the chunk libraries), not the other executables."),
    (r"^Aucun build (\S+) configuré( pour --target=\S+)? dans (\S+) : rien à lister\. Lancez d'abord "
     r"compile_all\.sh sans --exec \(ou tapez le nom à la main\)\.$",
     lambda m: f"No configured {m.group(1)} build" + (m.group(2).replace(" pour", " for") if m.group(2) else "")
               + f" in {m.group(3)}: nothing to list. Run compile_all.sh without --exec first (or type the name)."),
    (r"(\d+) cœurs", r"\1 cores"),
    (r"\noption : ", "\noption: "), (r"\ndéfaut du script : ", "\nscript default: "),
    (r"^défaut : ", "default: "), (r"\nlanceur : ", "\nlauncher: "),
    (r"^lecture des options de (.*)…$", r"reading the options of \1…"),
    (r"^(\S+) : (\d+) options lues \(python (\S+) sur (.+)\)$", r"\1: \2 options read (python \3 on \4)"),
    (r"^(\S+) : (\d+) options lues$", r"\1: \2 options read"),
    (r"^défauts enregistrés pour (\S+) sur (\S+)$", r"defaults saved for \1 on \2"),
    (r"pas de connexion maître : lancez (<b>ssh -fNM \S+</b>) dans un terminal",
     r"no master connection: run \1 in a terminal"),
    (r"connexion maître OK, mais python3 distant en échec \(voir la barre d'état\)",
     "master connection OK, but the remote python3 failed (see the status bar)"),
    (r"connecté — ", "connected — "), (r"<b>ccc_msub introuvable</b>", "<b>ccc_msub not found</b>"),
    (r"Les options du script n'ont pas été lues : seuls les arguments en plus seront passés\.\n\n"
     r"Soumettre quand même \?",
     "The script's options were not read: only the extra arguments will be passed.\n\nSubmit anyway?"),
    (r"^(\S+) a déjà un run actif sur (\S+) \((.*)\)\.\nDeux runs simultanés de même architecture "
     r"peuvent écrire le même répertoire \(oracles, résumé\)\.\n\nSoumettre quand même \?$",
     r"\1 already has an active run on \2 (\3).\nTwo simultaneous runs of one architecture may write "
     r"the same directory (oracles, summary).\n\nSubmit anyway?"),
    (r"^soumis : job (\S+)$", r"submitted: job \1"), (r"^lancé : (.*)$", r"started: \1"),
    (r"^options de (\S+) reprises$", r"options of \1 reused"),
    (r"\ble job (\S+) \(ccc_mdel\)", r"job \1 (ccc_mdel)"),
    (r"^Arrêter le run (\S+) \?\n", r"Stop run \1?\n"),
    (r"^(\S+) : arrêt demandé$", r"\1: stop requested"),
    (r" · reste ~(\d+) h (\d+)", r" · ~\1 h \2 left"), (r" · reste ~(\d+) min", r" · ~\1 min left"),
    (r" · reste < 1 min", " · < 1 min left"),
    (r"^« (.*) » absent du log$", r'"\1" not in the log'),
    (r"^Comparaison — temps par run selon le nombre de threads — ",
     "Comparison — time per run against the thread count — "),
    (r"^temps par run selon le nombre de threads — ", "time per run against the thread count — "),
    (r"tous les builds", "all builds"),
    (r"^médiane (\S+)$", r"median \1"), (r"^échec \((\d+)\)$", r"failure (\1)"),
    (r"^autres \((\d+)\)$", r"others (\1)"),
    (r"(\d+) run\(s\) affiché\(s\)", r"\1 run(s) shown"),
    (r" \(les 30000 derniers runs de chaque fichier\)", " (the last 30000 runs of each file)"),
    (r"\(([\d.]+) % plus lent\), moyenne géométrique des médianes sur (\d+) tranche\(s\) de (\d+) "
     r"thread\(s\) communes", r"(\1 % slower), geometric mean of the medians over \2 common bin(s) of \3 thread(s)"),
    (r"\(([\d.]+) % plus rapide\), moyenne géométrique des médianes sur (\d+) tranche\(s\) de (\d+) "
     r"thread\(s\) communes", r"(\1 % faster), geometric mean of the medians over \2 common bin(s) of \3 thread(s)"),
    (r"aucune tranche de threads commune aux deux runs", "no thread bin common to both runs"),
    (r"^Supprimer le dossier du run (\S+) sur inti \?\n\(log, scripts, code de retour ; les sorties du "
     r"programme restent en place\)$",
     r"Delete the directory of run \1 on inti?\n(log, scripts, exit code; the program's outputs stay)"),
    (r"^(\S+) supprimé$", r"\1 deleted"),
    (r"^\[gui\] … les (\S+) premiers Mo du log ne sont pas affichés \(« Récupérer les résultats » copie le log "
     r"complet\)$", r"[gui] … the first \1 MB of the log are not shown (\"Fetch results\" copies the whole log)"),
    (r"^(\S+) supprimé( \(\+ (\d+) sortie\(s\) sur inti\))?( \(\+ copie locale\))?$",
     lambda m: m.group(1) + " deleted" + (f" (+ {m.group(3)} output(s) on inti)" if m.group(2) else "")
     + (" (+ local copy)" if m.group(4) else "")),
    (r"^Supprimer le run (\S+) \?\n\nSur inti, son dossier \(log, scripts, code de retour\)\.$",
     r"Delete run \1?\n\nOn inti, its directory (log, scripts, exit code)."),
    (r"^aussi ce qu'il a écrit dans le dossier de sortie partagé \((.+) ; les fichiers des autres runs et "
     r"oracle/ restent\)$",
     r"also what it wrote in the shared output directory (\1; the other runs' files and oracle/ stay)"),
    (r"date de soumission de (\S+) illisible : rien supprimé", r"submission date of \1 unreadable: nothing deleted"),
    (r"^aussi la copie locale (.+)$", r"also the local copy \1"),
    (r"^pas de dossier de sortie annoncé dans le log de (\S+)$", r"no output directory announced in the log of \1"),
    (r"^dossier de sortie refusé : (.+)$", r"output directory refused: \1"),
    (r"^(\S+) utilise aussi (.+) : rien supprimé$", r"\1 uses \2 too: nothing deleted"),
    (r"^(\S+), plus récent ou actif, écrit aussi dans (.+) : rien supprimé$",
     r"\1, newer or active, writes to \2 too: nothing deleted"),
    (r"^récupération de (\S+)…$", r"fetching \1…"), (r"^récupération : ", "fetch: "),
    (r"^(\d+) fichier\(s\) → ", r"\1 file(s) → "),
    (r" sur (\S+) : ", r" on \1: "),
    (r"^(\S+) : (fini|arrêté|interrompu)", r"\1: \2"),
    (r"\bfini \(code (-?\d+)\)", r"done (code \1)"),
    (r"en cours \? \(lancé depuis (\S+)\)", r"running? (started from \1)"),
    # sent back by inti
    (r"ne lit pas ses options avec argparse", "does not read its options with argparse"),
    (r"\(à ([^)]+)\), avec python", r"(at \1), with python"),
    (r"aucune option reconnue dans (\S+) --help", r"no option recognised in \1 --help"),
    (r"^pas encore de (\S+)$", r"no \1 yet"), (r"(\S+) est vide", r"\1 is empty"),
    (r"pas de résumé pour ce programme", "no summary for this program"),
    (r"pas de log d'erreurs pour ce programme", "no error log for this program"),
    (r"aucun échec dans le log du run", "no failure in the run's log"),
    (r"pas encore de dossier de sortie annoncé dans le log", "no output directory announced in the log yet"),
    (r"^aucun fichier (.+) dans (\S+)$", r"no \1 file in \2"),
    (r"^(\d+) erreur\(s\) — ", r"\1 error(s) — "), (r"^aucune erreur — ", "no error — "),
    (r"^attention : (\S+), plus récent ou actif, écrit aussi dans ce dossier : ces fichiers sont peut-être "
     r"les siens$", r"warning: \1, newer or active, also writes in this directory: these files may be its own"),
    (r"^les (\d+) plus récents sur (\d+) fichiers$", r"the \1 newest of \2 files"),
    (r"^══ (.+) : introuvable ══$", r"══ \1: not found ══"),
    (r"^Comparaison — ", "Comparison — "),
    (r"^(?:Comparison — )?(speedup vs le premier nombre de threads|efficacité \(speedup × min / threads\)|"
     r"meilleur temps \(s\)) — ",
     lambda m: m.group(0).replace("speedup vs le premier nombre de threads", "speedup vs the first thread count")
     .replace("efficacité (speedup × min / threads)", "efficiency (speedup × min / threads)")
     .replace("meilleur temps (s)", "best time (s)")),
    (r"^A : (.+), traits pleins — B : (.+), pointillés$", r"A: \1, solid — B: \2, dashed"),
    (r"  —  B / A : ×([\d.]+), moyenne géométrique sur (\d+) point\(s\) communs$",
     r"  —  B / A: ×\1, geometric mean over \2 common point(s)"),
    (r"^pas encore de summary\.csv$", "no summary.csv yet"),
    (r"^A : (\S+) \(traits pleins, ronds\) — B : (\S+) \(pointillés, triangles\)$",
     r"A: \1 (solid, dots) — B: \2 (dashed, triangles)"),
    (r"^(?:Comparison — )?(temps moyen \(s\)|mémoire pic \(Mo\)|speedup vs la variante de référence) selon "
     r"le nombre de threads — ",
     lambda m: m.group(0).replace("temps moyen (s)", "mean time (s)").replace("mémoire pic (Mo)", "peak memory (MB)")
     .replace("speedup vs la variante de référence", "speedup vs the baseline variant")
     .replace(" selon le nombre de threads — ", " against the thread count — ")),
    (r"^(\d+) point\(s\)", r"\1 point(s)"),
    (r"  —  (\S+) / (\S+) : ×", r"  —  \1 / \2: ×"),
    (r" % plus lent\)", " % slower)"), (r" % plus rapide\)", " % faster)"),
    (r" % de mémoire en plus\)", " % more memory)"), (r" % de mémoire en moins\)", " % less memory)"),
    (r" % de speedup en plus\)", " % more speedup)"), (r" % de speedup en moins\)", " % less speedup)"),
    (r", moyenne géométrique sur (\d+) point\(s\) communs \(toutes configs\)",
     r", geometric mean over \1 common point(s) (all configs)"),
    (r" ; le pire : ×", "; the worst: ×"),
    (r"(×[\d.]+ \([\d.]+ % [a-z ]+\)) sur (\S+)", r"\1 on \2"),
    (r"(\(all configs\)) ; ×", r"\1; ×"),
    (r"  —  aucun point commun aux deux runs", "  —  no point common to both runs"),
    (r"^les deux runs lisent le même bench_summary\.csv partagé \(lancés avant que chaque run en garde une "
     r"copie\) : la comparaison n'a pas de sens$",
     "both runs read the same shared bench_summary.csv (submitted before each run kept a copy): "
     "the comparison means nothing"),
    (r"^(\S+) : bench_summary\.csv partagé de l'architecture, peut-être réécrit par un run plus récent$",
     r"\1: the architecture's shared bench_summary.csv, maybe rewritten by a newer run"),
    (r"^(\d+) run\(s\) en échec \(timings_summary\.csv\)\. Le détail : onglet Erreurs\.$",
     r"\1 failed run(s) (timings_summary.csv). The details: Errors tab."),
    (r"^(\d+) erreur\(s\) \(events\.jsonl\)\. Le détail : onglet Erreurs\.", r"\1 error(s) (events.jsonl). "
     r"The details: Errors tab."),
    (r"\n\nDernière : ", "\n\nLast: "),
    (r" ?: première erreur, le run continue$", ": first error, the run goes on"),
    (r"^(\S+) : première erreur$", r"\1: first error"),
    (r"^relance des échecs de (\S+) : la liste ne garde que ce qui échoue encore, bench_errors\.log est "
     r"celui du run rejoué$",
     r"rerun of the failures of \1: the list keeps only what still fails, bench_errors.log is the replayed run's"),
    (r"^Soumet un nouveau run qui rejoue les (\d+) échec\(s\) rejouable\(s\) de ce run \(--rerun-failed\), "
     r"mêmes partition et options\. Désactivé tant que le run tourne\.$",
     r"Submits a new run replaying the \1 replayable failure(s) of this run (--rerun-failed), same partition "
     r"and options. Disabled while the run is running."),
    (r"^Rejouer les (\d+) échec\(s\) rejouable\(s\) de (\S+) \(bench_and_test\.py --rerun-failed\), sur la "
     r"même partition, avec les mêmes options\.", r"Replay the \1 replayable failure(s) of \2 "
     r"(bench_and_test.py --rerun-failed), on the same partition, with the same options."),
    (r"\n(\d+) autre\(s\) \(une comparaison, un garde\) demandent un run complet\.",
     r"\n\1 other(s) (a comparison, a guard) need a full run."),
    (r"\n\nAttention : (\S+), plus récent ou actif, écrit aussi dans ce dossier : failed_cases\.jsonl est "
     r"peut-être le sien\.", r"\n\nWarning: \1, newer or active, also writes in this directory: "
     r"failed_cases.jsonl may be its own."),
    (r"^\[gui\] … les (.+) du début ne sont pas affichés \(« Récupérer les résultats » copie le fichier "
     r"complet\)$", r'[gui] … the first \1 are not shown ("Fetch results" copies the whole file)'),
    (r"« (.*) » absent des erreurs", r'"\1" not found in the errors'),
    (r"graphique disponible pour multi_launch_stress\.py seulement", "chart available for multi_launch_stress.py only"),
    (r"graphique disponible pour multi_launch_stress\.py et bench_and_test\.py seulement",
     "chart available for multi_launch_stress.py and bench_and_test.py only"),
    (r"graphique disponible pour multi_launch_stress\.py, bench_and_test\.py et omp_speedup\.py seulement",
     "chart available for multi_launch_stress.py, bench_and_test.py and omp_speedup.py only"),
    (r"^pas encore de (\S+) \(écrit à la fin du run\)$", r"no \1 yet (written at the end of the run)"),
    (r"— (\d+) ligne\(s\), les 1000 dernières", r"— \1 row(s), the last 1000"),
    (r"— (\d+) ligne\(s\)", r"— \1 row(s)"),
    (r"nom de run invalide", "invalid run name"), (r"n'est pas un run", "is not a run"),
    (r"ni ccc_mdel ni scancel n'ont pu annuler le job", "neither ccc_mdel nor scancel could cancel job"),
    (r"(\S+(?: / \S+)*) : le job (\S+) n'a pas pu être annulé", r"\1: job \2 could not be cancelled"),
    (r"lancé depuis (\S+), ce master arrive sur (\S+)", r"started from \1, this master lands on \2"),
    (r"arrêt impossible", "cannot stop"),
    (r"^(\S+) se soumet par ccc_msub seulement \(il ne doit pas tourner depuis le nœud de login\) : "
     r"choisissez « (.+) »$",
     r'\1 is submitted with ccc_msub only (it must not run from the login node): choose "\2"'),
    (r"pas de bloc usage\(\) \{ cat <<EOF \.\.\. EOF \} où lire les options \(le script n'est pas exécuté ici\)",
     "no usage() { cat <<EOF ... EOF } block to read the options from (the script is not run here)"),
    (r" — ([^<:]+) : <b>", r" — \1: <b>"), (r" · local : identique", " · local: same"),
    (r"^branche <b>", "branch <b>"), (r" · modifications locales", " · local changes"),
    (r"modifications locales</span>", "local changes</span>"),
    (r" · pas de branche suivie", " · no upstream branch"), (r" · suit ", " · tracks "),
    (r"</b> en retard</span>", "</b> behind</span>"), (r"</b> en avance", "</b> ahead"),
    (r" · à jour au dernier fetch", " · up to date as of the last fetch"),
    (r"^(git pull(?: --rebase)?) dans (\S+) \(branche (\S+)\) \?$", r"\1 in \2 (branch \3)?"),
    (r"^(\d+) run\(s\) encore actif\(s\) : un build ou un bench en cours peut lire des sources "
     r"modifiées en cours de route\.$",
     r"\1 run(s) still active: a build or a bench in progress may read sources changed under it."),
    (r"^(git pull(?: --rebase)?) sur (.+)…$", r"\1 on \2…"),
    (r"^(git pull(?: --rebase)?) : déjà à jour$", r"\1: already up to date"),
    (r"^(git pull(?: --rebase)?) : (\S+) → (\S+)$", r"\1: \2 → \3"),
    (r"^(git pull(?: --rebase)?) a échoué$", r"\1 failed"),
    (r"^ ; (rebase|merge) annulé, le dépôt est revenu à son état d'avant$",
     r"; \1 cancelled, the repository is back as it was"),
    (r"^git_pull : pas de dépôt git dans (\S+)$", r"git_pull: no git repository in \1"),
    (r"local : (<b>\w+</b>) \(([^)]*)\) ≠ ", r"local: \1 (\2) ≠ "),
    (r"^branche : ", "branch: "),
    (r"^Changer de branche dans (\S+) : (\S+) → (\S+) \?$", r"Switch branch in \1: \2 → \3?"),
    (r"^Une branche locale (\S+) suivant (\S+) est utilisée, ou créée si elle n'existe pas\.$",
     r"A local branch \1 tracking \2 is used, or created if it does not exist."),
    (r"^changement de branche sur inti : (\S+)…$", r"switching branch on inti: \1…"),
    (r"^modifications non commitées dans (\S+) : commitez-les, ou mettez-les de côté \(git stash\), avant "
     r"de changer de branche$", r"uncommitted changes in \1: commit them, or set them aside (git stash), "
     r"before switching branch"),
    (r"^branche inconnue : ", "unknown branch: "),
    (r"\* modifications non commitées dans le dépôt d'inti à la soumission",
     "* uncommitted changes in inti's repository at submission"),
    (r"≠ le job a démarré sur (\w+) : le dépôt a changé entre la soumission et le démarrage",
     r"≠ the job started on \1: the repository changed between submission and start"),
    (r" — même commit (\S+) des deux côtés", r" — same commit \1 on both sides"),
    (r" — commits (\S+) → (\S+)", r" — commits \1 → \2"),
    (r"^(\d+) phase\(s\)", r"\1 phase(s)"),
    (r" — fin : (\d+) erreur\(s\)(, interrompu)?, ([\d.]+) min",
     lambda m: f" — end: {m.group(1)} error(s){', interrupted' if m.group(2) else ''}, {m.group(3)} min"),
    (r"^fin · ", "end · "),
)]

# The site's words: the texts are written for CEA's inti (inti, ccc_msub,
# #MSUB, ccc_mprun, ccc_mdel, ccc_mqinfo); T() puts the site's own in their
# place, in both languages (sitecfg.Site.words, set by config).
SITE_WORDS = {}
_SITE_RE = None


def set_site_words(words):
    """{CEA word: the site's}: identical pairs are dropped."""
    global SITE_WORDS, _SITE_RE
    SITE_WORDS = {k: v for k, v in words.items() if k != v}
    _SITE_RE = (re.compile("|".join(r"(?<![\w#])" + re.escape(k) + r"(?!\w)"
                                    for k in sorted(SITE_WORDS, key=len, reverse=True)))
                if SITE_WORDS else None)


def site_words(text):
    """@p text with the site's words in place of CEA's."""
    return _SITE_RE.sub(lambda m: SITE_WORDS[m.group()], text) if _SITE_RE and text else text


def T(text):
    """The text in the interface language (see EN, EN_RE), with the site's
    words (set_site_words)."""
    if LANG != "en" or not text:
        return site_words(text)
    if text in EN:
        return site_words(EN[text])
    for rx, en in EN_RE:
        text = rx.sub(en, text)
    return site_words(text)

def status_text(run):
    """A run's status as shown: translated, with the exit code when done."""
    st = T(run["status"])
    return st + (f" (code {run['code']})" if run["status"] == "fini" else "")

def tip(text):
    """A tooltip that wraps: Qt only wraps rich text, plain text stays one line
    as wide as the screen (the long argparse helps did)."""
    paras = [html.escape(p).replace("\n", "<br>") for p in T(str(text)).split("\n\n")]
    return "<qt>" + "".join(f"<p>{p}</p>" for p in paras) + "</qt>"



def set_lang(code):
    """Switch the interface language ("fr" or "en")."""
    global LANG
    LANG = code if code in ("fr", "en") else "fr"
