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
    "Lanceur (job)": "Launcher (job)", "Dépôt (inti)": "Repository (inti)",
    "Runs (inti)": "Runs (inti)", "Résultats (local)": "Results (local)",
    "Langue": "Language",
    "gh200-bxi — ARM, 288 cœurs": "gh200-bxi — ARM, 288 cores",
    "milan-bxi — x86, 64 cœurs": "milan-bxi — x86, 64 cores",
    "ccc_msub (batch)": "ccc_msub (batch)", "ccc_mprun détaché": "ccc_mprun, detached",
    "vide : la commande tourne directement dans le job": "empty: the command runs directly in the job",
    "Lire les options": "Read options", "filtrer les options (nom ou aide)…": "filter the options (name or help)…",
    "Défauts de l'architecture": "Architecture defaults", "Enregistrer comme défauts": "Save as defaults",
    "arguments en plus, tels quels (facultatif)": "extra arguments, as they are (optional)",
    "Commande :": "Command:", "Soumettre": "Submit",
    # runs
    "Run": "Run", "Programme": "Program", "Job": "Job", "Statut": "Status",
    "Avancement": "Progress", "Dernière ligne": "Last line",
    "Rafraîchir": "Refresh", "Arrêter": "Stop", "Récupérer les résultats": "Fetch results",
    "Reprendre ses options": "Reuse its options", "Supprimer": "Delete",
    "Log": "Log", "Résumé": "Summary", "Graphique": "Chart",
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
    "nœud tant qu'il tourne.":
        "ccc_msub: batch job, independent of this window and of the login node "
        "(recommended).\n\nccc_mprun, detached: started from the login node; it depends on that "
        "node while it runs.",
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
    "La QoS du job (-Q), dont dépend la durée maximale (ccc_mqinfo) : test (prioritaire, "
    "court, 2 jobs soumis au plus), normal (1 jour), long (3 jours, pas sur milan-bxi), "
    "xlong (pas sur gh200-bxi).\n\n« auto » prend la plus prioritaire qui accepte la durée, "
    "puis la plus courte. Un -Q écrit dans les directives l'emporte.\n\nLes limites sont "
    "relues sur inti à chaque vérification de la connexion.":
        "The job's QoS (-Q), which sets the maximum time (ccc_mqinfo): test (higher priority, "
        "short, 2 submitted jobs at most), normal (1 day), long (3 days, not on milan-bxi), "
        "xlong (not on gh200-bxi).\n\n\"auto\" takes the highest priority one that accepts the "
        "duration, then the shortest. A -Q written in the directives wins.\n\nThe limits are "
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
    "Supprime le dossier du run sur inti (pas les sorties du programme). "
    "Seulement pour un run terminé.":
        "Deletes the run directory on inti (not the program's outputs). Finished runs only.",
    "Supprime le dossier du run sur inti (pas les sorties du programme)":
        "Deletes the run directory on inti (not the program's outputs)",
    "Relit le fichier résumé du run sélectionné.": "Reads the selected run's summary file again.",
    "ccc_msub : ce qui lance run.sh sur le nœud réservé, par exemple "
    "« ccc_mprun -C ccc-rhel9 -n 1 -c 288 » (environnement RHEL 9 du gh200).\n"
    "Mémorisé par partition.":
        "ccc_msub: what starts run.sh on the reserved node, for instance "
        "\"ccc_mprun -C ccc-rhel9 -n 1 -c 288\" (the gh200's RHEL 9 environment).\n"
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
    "Compilation de toutes les configurations, en job batch (ccc_msub) seulement. Options lues dans "
    "le texte de son usage(), sans l'exécuter sur le nœud de login.":
        "Builds every configuration, as a batch job (ccc_msub) only. Options read from the text of its "
        "usage(), without running it on the login node.",
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
}

# Messages built on the fly (and those inti sends back): regex on the French,
# English replacement. Applied in order, on the whole text.
EN_RE = [(re.compile(a), b) for a, b in (
    (r"réponse illisible de (\S+) :", r"unreadable answer from \1:"),
    (r"<b>Options non lues\.</b>", "<b>Options not read.</b>"),
    (r"Corrigez puis cliquez sur « Lire les options »\. En attendant, le champ « arguments en plus » "
     r"passe les options telles quelles\.",
     "Fix it, then click \"Read options\". Meanwhile the \"extra arguments\" field passes options as they are."),
    (r"\(pas d'aide dans le script\)", "(no help in the script)"),
    (r"^la QoS (\S+) n'est pas ouverte sur (\S+)$", r"QoS \1 is not open on \2"),
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
    (r"^lecture des builds de (\S+) sur inti…$", r"reading the \1 builds on inti…"),
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
    (r"^(\S+) : (\d+) options lues \(python (\S+) sur inti\)$", r"\1: \2 options read (python \3 on inti)"),
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
    (r"graphique disponible pour multi_launch_stress\.py seulement", "chart available for multi_launch_stress.py only"),
    (r"— (\d+) ligne\(s\), les 1000 dernières", r"— \1 row(s), the last 1000"),
    (r"— (\d+) ligne\(s\)", r"— \1 row(s)"),
    (r"nom de run invalide", "invalid run name"), (r"n'est pas un run", "is not a run"),
    (r"ni ccc_mdel ni scancel n'ont pu annuler le job", "neither ccc_mdel nor scancel could cancel job"),
    (r"lancé depuis (\S+), ce master arrive sur (\S+)", r"started from \1, this master lands on \2"),
    (r"arrêt impossible", "cannot stop"),
    (r"^(\S+) se soumet par ccc_msub seulement \(il ne doit pas tourner depuis le nœud de login\) : "
     r"choisissez « ccc_msub \(batch\) »$",
     r'\1 is submitted with ccc_msub only (it must not run from the login node): choose "ccc_msub (batch)"'),
    (r"pas de bloc usage\(\) \{ cat <<EOF \.\.\. EOF \} où lire les options \(le script n'est pas exécuté ici\)",
     "no usage() { cat <<EOF ... EOF } block to read the options from (the script is not run here)"),
    (r" — inti : ", " — inti: "), (r" · local : identique", " · local: same"),
    (r"local : (<b>\w+</b>) \(([^)]*)\) ≠ inti", r"local: \1 (\2) ≠ inti"),
    (r"^branche : ", "branch: "),
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

def T(text):
    """The text in the interface language (see EN, EN_RE)."""
    if LANG != "en" or not text:
        return text
    if text in EN:
        return EN[text]
    for rx, en in EN_RE:
        text = rx.sub(en, text)
    return text

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
