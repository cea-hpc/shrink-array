# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The window's tooltips: the texts (French, translated by T) and where they go.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from shrink_gui.config import PRESETS
from shrink_gui.i18n import tip


class Tooltips:
    # ------------------------------------------------------------- tooltips
    TIPS = {
        "host": "Nom d'hôte ssh du calculateur (tel qu'il figure dans ~/.ssh/config).\n\n"
                "Tout passe par la connexion maître que vous ouvrez vous-même : ssh -fNM inti.",
        "preset": "La partition où le run est soumis. Elle fixe aussi l'architecture (ARM ou x86), "
                  "donc les valeurs par défaut des options.",
        "mode": "ccc_msub : job batch, indépendant de cette fenêtre et du nœud de login "
                "(recommandé).\n\nccc_mprun détaché : lancé depuis le nœud de login ; il dépend de ce "
                "nœud tant qu'il tourne.\n\nEn arrière-plan sur cette machine : sans gestionnaire de "
                "jobs, indépendant de cette fenêtre ; la durée max est appliquée par timeout (code 124).",
        "local_cb": "Lancer les runs sur cette machine plutôt que sur le calculateur : pas de ssh ni de "
                    "gestionnaire de jobs, le run tourne en arrière-plan ici (sous Windows, dans WSL). "
                    "Dépôt et dossier des runs sont mémorisés à part.",
        "time_s": "Durée maximale du job (-T). Au-delà, le job est tué par le gestionnaire et finit "
                  "« terminé sans code ». Se tape « 1 h 30 », « 90 min », « 2 j » ou « 1-00:00:00 ».\n\n"
                  "Mémorisée par programme : compile_all.sh court, bench_and_test.py quelques heures, "
                  "multi_launch_stress.py long.",
        "qos": "La QoS du job, dont dépend la durée maximale : la liste a les QoS de la partition, "
               "avec leurs limites (ccc_mqinfo).\n\n« auto » prend la plus prioritaire qui accepte la "
               "durée, puis la plus courte. Une QoS écrite dans les directives l'emporte.\n\nLes limites "
               "sont relues sur inti à chaque vérification de la connexion.",
        "after": "Démarrer après la fin d'une compilation (run compile_all.sh pas encore fini).\n\n"
                 "ccc_msub : dépendance Slurm (le job attend dans la file sans prendre de nœud). "
                 "ccc_mprun détaché : le lancement attend le code de retour de ce run.\n\n"
                 "« si elle réussit » : une compilation en échec, et le run ne démarre jamais "
                 "(en batch il reste en attente : à arrêter).",
        "delay_on": "Ne pas démarrer avant ce délai, compté depuis la soumission "
                    "(ccc_msub : --begin de Slurm ; ccc_mprun détaché : une attente avant le lancement).",
        "check_builds": "Avant la commande, sur le nœud : ninja -n sur les exécutables déjà construits "
                        "de chaque build que le programme utilisera (--configs de bench_and_test.py, "
                        "--builds de multi_launch_stress.py, sous --build-root). Un build dont une "
                        "source, un en-tête ou la configuration a changé depuis sa compilation n'est "
                        "pas à jour.\n\n« refuser » : le run s'arrête (code 96) en listant ces builds.",
        "directives": "Lignes #MSUB du script batch (ou options ccc_mprun en mode détaché). Propres à "
                      "la partition, mémorisées.\n\n-T, -r, -o et -e sont ajoutés d'après le formulaire.",
        "repo": "Le dépôt shrink-array SUR INTI. Les options sont lues dans ces scripts-là, et c'est "
                "là que le job s'exécute.",
        "runs_dir": "Où l'interface range ses runs sur inti : un dossier par run (log, scripts, "
                    "code de retour, numéro de job).",
        "local_dir": "Où « Récupérer les résultats » copie les fichiers sur ce portable.",
        "local_repo": "Votre clone du dépôt sur ce portable (facultatif). La ligne de connexion compare "
                      "alors son commit à celui du dépôt d'inti : un écart veut dire qu'un push ou un "
                      "git pull manque.",
        "program": "Le script à lancer. Ses options sont lues dans le script lui-même sur inti.",
        "opt_filter": "Ne montre que les options dont le nom ou l'aide contient ce texte.",
        "extra": "Arguments ajoutés tels quels à la fin de la commande : pour une option que le "
                 "formulaire ne connaît pas (--no-gnu de compile_all.sh, par exemple).",
        "preview": "La commande exacte qui sera exécutée sur le nœud. Seules les options modifiées "
                   "(en gras) y figurent : les autres gardent le défaut du script.",
        "table": "Les runs du dossier « Runs (inti) », du plus récent au plus ancien. Cliquez sur un "
                 "run pour voir son log, son résumé et son graphique.\n\nRafraîchi toutes les 15 s.",
        "find": "Recherche dans le log (Ctrl+F). Entrée : occurrence suivante, avec retour au début.",
        "follow": "Fait défiler le log jusqu'à la fin à chaque nouvelle ligne. Décochée, le log "
                  "reste où vous êtes.",
        "pt_build": "Ne montre que les runs de ce build.",
        "pt_variant": "Ne montre que les runs de cette variante (zero, zero_one_value, zero_any).",
        "pt_color": "Ce que la couleur distingue : la variante, la taille de chunk, ou rien. Trois "
                    "couleurs au plus ; le reste est groupé en gris « autres ».",
        "pt_log": "Axe du temps en échelle logarithmique : utile quand les temps s'étalent sur "
                  "plusieurs ordres de grandeur.",
        "pt_median": "Médiane du temps par tranche de threads (au moins 3 runs par tranche) : la "
                     "tendance malgré la dispersion.",
        "pt_ok_only": "Retire les runs en échec (triangles) du graphique.",
        "sum_view": "bench_and_test.py : ses phases (durée, erreurs, d'après events.jsonl) ou son "
                    "bench_summary.csv.",
    }

    BUTTON_TIPS = {
        "Vérifier la connexion": "Vérifie la connexion maître ssh et le python3 d'inti, puis "
                                 "rafraîchit la liste des runs.",
        "Lire les options": "Relit les options du script sur inti : à faire après avoir modifié ou "
                            "mis à jour le script.",
        "Défauts de l'architecture": "Remet les options aux valeurs par défaut de l'architecture "
                                     "de la partition (ARM ou x86).",
        "Enregistrer comme défauts": "Enregistre les options actuelles comme valeurs par défaut "
                                     "de ce programme sur cette architecture.",
        "Soumettre": "Crée le dossier du run sur inti et soumet le job.",
        "Rafraîchir": "Relit maintenant (sinon automatique).",
        "Arrêter": "Annule le job du run sélectionné (ccc_mdel), ou arrête ccc_mprun en mode détaché. "
                   "Plusieurs runs sélectionnés : tous ceux qui sont actifs.",
        "Récupérer les résultats": "Copie sur ce portable le dossier du run et les fichiers de "
                                   "résultats du programme (CSV, log, erreurs ; jamais les output_*). "
                                   "Plusieurs runs sélectionnés : chacun, l'un après l'autre.",
        "Reprendre ses options": "Remet dans le formulaire le programme, la partition et les options "
                                 "du run sélectionné, pour le relancer ou le modifier.",
        "Supprimer": "Supprime le dossier du run sur inti ; au choix, aussi les sorties du "
                     "programme propres à ce run et la copie locale. Seulement pour un run terminé. "
                     "Plusieurs runs sélectionnés : une seule confirmation pour tous (les actifs "
                     "sont laissés).",
        "Rafraîchir le résumé": "Relit le fichier résumé du run sélectionné.",
        "git pull": "git pull dans le dépôt d'inti (fusion si les branches ont divergé). "
                    "Un conflit annule le pull : le dépôt reste tel qu'il était.",
        "git pull --rebase": "git pull --rebase dans le dépôt d'inti : vos commits locaux "
                             "d'inti sont rejoués au-dessus de la branche distante. Un conflit "
                             "annule le rebase : le dépôt reste tel qu'il était.",
        "Changer de branche": "Extrait dans le dépôt d'inti la branche choisie dans la liste. Une "
                              "branche distante (origin/x) sans branche locale donne une branche "
                              "locale x qui la suit. Refusé si le dépôt a des modifications non "
                              "commitées. git pull agit ensuite sur cette branche.",
        "Relire les branches": "git fetch dans le dépôt d'inti, puis relit la liste des branches.",
    }

    def set_tooltips(self):
        for name, text in self.TIPS.items():
            w = getattr(self, name, None)
            if w is not None:
                w.setToolTip(tip(text))
        for b in self.findChildren(QPushButton):
            if b.text() in self.BUTTON_TIPS and not b.toolTip():
                b.setToolTip(tip(self.BUTTON_TIPS[b.text()]))
            elif b.toolTip() and not b.toolTip().startswith("<qt>"):
                b.setToolTip(tip(b.toolTip()))
        if self.launcher.toolTip() and not self.launcher.toolTip().startswith("<qt>"):
            self.launcher.setToolTip(tip(self.launcher.toolTip()))
        if self.pt_cmp.toolTip() and not self.pt_cmp.toolTip().startswith("<qt>"):
            self.pt_cmp.setToolTip(tip(self.pt_cmp.toolTip()))
        # the entries of the drop-down lists
        for i in range(self.preset.count()):
            p = PRESETS[self.preset.itemData(i)]
            self.preset.setItemData(i, tip(f"{p['label']}\n\n" + p["msub"].strip()
                                           + (f"\nlanceur : {p['launcher']}" if p.get("launcher") else "")),
                                    Qt.ToolTipRole)
        prog_tips = {
            "multi_launch_stress.py": "Stress : les binaires de check, en boucle, à un nombre de "
                                      "threads tiré au hasard. Résumé et graphique disponibles.",
            "bench_and_test.py": "Bench et tests : validations ctest, checks, sanitizers, comparaison "
                                 "seq/omp et mesures de performance.",
            "compile_all.sh": "Compilation de toutes les configurations, en job batch (ccc_msub) ou sur cette "
                              "machine, jamais depuis le nœud de login. Options lues dans le texte de son "
                              "usage(), sans l'exécuter.",
        }
        for i in range(self.program.count()):
            t = prog_tips.get(self.program.itemText(i))
            if t:
                self.program.setItemData(i, tip(t), Qt.ToolTipRole)
        mode_tips = {"msub": self.TIPS["mode"].split("\n\n")[0],
                     "mprun": self.TIPS["mode"].split("\n\n")[1],
                     "local": self.TIPS["mode"].split("\n\n")[2]}
        for i in range(self.mode.count()):
            self.mode.setItemData(i, tip(mode_tips[self.mode.itemData(i)]), Qt.ToolTipRole)
        tab_tips = ["Le log du run sélectionné, suivi en direct.",
                    "Le résumé du run : par build pour multi_launch_stress.py (runs, échecs, temps), "
                    "le CSV tel quel pour bench_and_test.py.",
                    "Temps de chaque run selon le nombre de threads (multi_launch_stress.py). "
                    "Survolez un point pour le détail ; « Comparer avec » superpose un autre run.",
                    "Le log d'erreurs du run, lu dans son dossier de sortie : bench_errors.log pour "
                    "bench_and_test.py (une entrée par échec, écrite dès qu'il survient, avec la fin "
                    "de sa sortie d'erreur ; la sortie complète de chaque exécution est dans run_logs/), "
                    "les error_*.txt des runs en échec pour multi_launch_stress.py. Le plus récent "
                    "d'abord, la fin de chaque fichier seulement.\n\nL'onglet passe en rouge "
                    "quand il y a des erreurs."]
        for i, t in enumerate(tab_tips):
            self.tabs.setTabToolTip(i, tip(t))
