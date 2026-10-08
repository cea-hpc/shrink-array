"""The main window."""
import ast
import base64
import html
import io
import json
import os
import re
import sys
import tarfile
import time

from PySide6.QtCharts import (QChart, QLegend, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis)
from PySide6.QtCore import QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence, QPen, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSplitter, QStyle, QSystemTrayIcon,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from shrink_gui import i18n, qos
from shrink_gui.config import (APP, ARCH_DEFAULTS, BUILD_REP, COLOR_BY, DEFAULTS, PRESETS, PROGRAMS, VIZ,
                               exe_parts)
from shrink_gui.i18n import T, status_text, tip
from shrink_gui.remote import Remote
from shrink_gui.runscripts import make_launch_sh, make_run_sh
from shrink_gui.widgets import (DurationBox, LogView, NumItem, OptionsForm, PickerDialog, ScatterView, TLabel,
                                ask, families, split_list, unique_globs)


class Main(QMainWindow):
    COLS = ["Run", "Programme", "Partition", "Commit", "Job", "Statut", "Avancement", "Dernière ligne"]
    COL_STATUS = 5
    ACTIVE = ("en cours", "en attente", "démarrage")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("shrink-array — runs sur inti (prototype)")
        self.settings = QSettings(APP, APP)
        self.remote = Remote(self, lambda: self.host.text().strip(), self.error)
        self.runs = []
        self.current = None      # id of the run shown in the log
        self.offset = 0
        self.tail_busy = False
        self.list_busy = False
        self.prev_status = None  # id -> status at the previous refresh (notifications)
        self.build_info = {}     # arch suffix -> act_build_targets (the "Choisir…" lists)

        # --- connection
        self.host = QLineEdit(self._get("host"))
        self.conn = TLabel("?")
        bcheck = QPushButton("Vérifier la connexion")
        bcheck.clicked.connect(self.check)
        top = QHBoxLayout()
        top.addWidget(QLabel("Calculateur :"))
        top.addWidget(self.host)
        top.addWidget(bcheck)
        top.addWidget(self.conn, 1)
        self.lang = QComboBox()
        self.lang.addItem("Français", "fr")
        self.lang.addItem("English", "en")
        self.lang.setCurrentIndex(max(0, self.lang.findData(i18n.LANG)))
        self.lang.setToolTip(tip("Choix de la langue de l'interface (changement immédiat)."))
        self.lang.currentIndexChanged.connect(self.change_language)
        top.addWidget(self.lang)

        # --- where / how
        self.preset = QComboBox()
        for key, p in PRESETS.items():
            self.preset.addItem(p["label"], key)
        self.preset.setCurrentIndex(max(0, self.preset.findData(self._get("preset"))))
        self.mode = QComboBox()
        self.mode.addItem("ccc_msub (batch)", "msub")
        self.mode.addItem("ccc_mprun détaché", "mprun")
        self.mode.setCurrentIndex(max(0, self.mode.findData(self._get("mode"))))
        self.time_s = DurationBox()
        self.time_s.setValue(int(self._get("time_s")))
        # QoS: the limits of ccc_mqinfo (read by the connection check, else
        # those of qos.FALLBACK), "auto" picks one from the duration
        self.qos_rows = qos.parse_mqinfo(self.settings.value("mqinfo", "")) or qos.parse_mqinfo(qos.FALLBACK)
        self.qos = QComboBox()
        self.fill_qos(self.settings.value("qos", "auto"))
        self.qos_note = TLabel("")
        self.qos_note.setWordWrap(True)
        qrow = QHBoxLayout()
        qrow.addWidget(self.qos)
        qrow.addWidget(self.qos_note, 1)
        # Start: after a compile run (Slurm dependency, or a wait in the
        # detached mode), not before a delay; the builds checked on the node.
        self.after = QComboBox()
        self.after_cond = QComboBox()
        self.after_cond.addItem("si elle réussit", "ok")
        self.after_cond.addItem("quelle que soit son issue", "any")
        self.after_cond.setCurrentIndex(max(0, self.after_cond.findData(self.settings.value("after_cond", "ok"))))
        self.after.currentIndexChanged.connect(lambda *_: self.after_cond.setEnabled(bool(self.after.currentData())))
        self.fill_after()
        self.delay_on = QCheckBox("pas avant")
        self.delay = DurationBox()
        self.delay.setValue(int(self.settings.value("delay_s", 3600)))
        self.delay.setEnabled(False)
        self.delay_on.toggled.connect(self.delay.setEnabled)
        arow = QHBoxLayout()
        arow.addWidget(self.after, 2)
        arow.addWidget(self.after_cond, 1)
        arow.addWidget(self.delay_on)
        arow.addWidget(self.delay, 1)
        self.check_builds = QComboBox()
        for label, key in (("refuser de démarrer si un build n'est pas à jour", "refuse"),
                           ("avertir seulement", "warn"), ("ne pas vérifier", "off")):
            self.check_builds.addItem(label, key)
        self.check_builds.setCurrentIndex(
            max(0, self.check_builds.findData(self.settings.value("check_builds", "refuse"))))
        self.directives = QPlainTextEdit()
        self.directives.setFont(QFont("Monospace", 9))
        self.directives.setFixedHeight(92)
        self.directives.setToolTip("Lignes #MSUB du script batch (ccc_msub) ou options ccc_mprun "
                                   "(mode détaché), propres à la partition et mémorisées.\n"
                                   "-T, -r, -o et -e sont ajoutés d'après le formulaire.")
        self.repo = QLineEdit(self._get("repo"))
        self.runs_dir = QLineEdit(self._get("runs_dir"))
        self.local_dir = QLineEdit(self._get("local_dir"))
        where = QFormLayout()
        where.addRow("Partition", self.preset)
        where.addRow("Soumission", self.mode)
        where.addRow("Durée max (-T)", self.time_s)
        where.addRow("QoS (-Q)", qrow)
        where.addRow("Démarrage", arow)
        where.addRow("Builds à jour", self.check_builds)
        where.addRow("Directives", self.directives)
        self.launcher = QLineEdit()
        self.launcher.setPlaceholderText("vide : la commande tourne directement dans le job")
        self.launcher.setToolTip("ccc_msub : ce qui lance run.sh sur le nœud réservé, par exemple "
                                 "« ccc_mprun -C ccc-rhel9 -n 1 -c 288 » (environnement RHEL 9 du gh200).\n"
                                 "Mémorisé par partition.")
        self.launcher_row_label = QLabel("Lanceur (job)")
        where.addRow(self.launcher_row_label, self.launcher)
        where.addRow("Dépôt (inti)", self.repo)
        where.addRow("Runs (inti)", self.runs_dir)
        where.addRow("Résultats (local)", self.local_dir)
        self.local_repo = QLineEdit(self._get("local_repo"))
        self.local_repo.setPlaceholderText("facultatif : pour comparer son commit à celui d'inti")
        self.local_repo.editingFinished.connect(self.check)
        where.addRow("Dépôt local", self.local_repo)
        gw = QGroupBox("Où et comment")
        gw.setLayout(where)

        # --- what
        self.program = QComboBox()
        self.program.addItems(PROGRAMS)
        self.program.setCurrentText(self._get("program"))
        bscan = QPushButton("Lire les options")
        bscan.setToolTip("Relit les options du script sur inti (après une modification du script)")
        bscan.clicked.connect(self.scan)
        prow = QHBoxLayout()
        prow.addWidget(self.program, 1)
        prow.addWidget(bscan)
        self.form = OptionsForm()
        self.form.changed = self.update_preview
        self.opt_filter = QLineEdit()
        self.opt_filter.setPlaceholderText("filtrer les options (nom ou aide)…")
        self.opt_filter.setClearButtonEnabled(True)
        self.opt_filter.textChanged.connect(self.form.filter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        bdef_reset = QPushButton("Défauts de l'architecture")
        bdef_reset.clicked.connect(self.apply_arch_defaults)
        bdef_save = QPushButton("Enregistrer comme défauts")
        bdef_save.clicked.connect(self.save_arch_defaults)
        drow = QHBoxLayout()
        drow.addWidget(bdef_reset)
        drow.addWidget(bdef_save)
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("arguments en plus, tels quels (facultatif)")
        self.extra.textChanged.connect(self.update_preview)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Monospace", 9))
        self.preview.setFixedHeight(64)
        blaunch = QPushButton("Soumettre")
        blaunch.setStyleSheet("font-weight: bold; padding: 6px;")
        blaunch.clicked.connect(self.launch)
        what = QVBoxLayout()
        what.addLayout(prow)
        what.addWidget(self.opt_filter)
        what.addWidget(scroll, 1)
        what.addLayout(drow)
        what.addWidget(self.extra)
        what.addWidget(QLabel("Commande :"))
        what.addWidget(self.preview)
        what.addWidget(blaunch)
        gp = QGroupBox("Quoi")
        gp.setLayout(what)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(gw)
        ll.addWidget(gp, 1)

        # --- runs table + actions
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([T(c) for c in self.COLS])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        for c in range(len(self.COLS) - 1):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.select)
        brefresh = QPushButton("Rafraîchir")
        brefresh.clicked.connect(self.refresh)
        bstop = QPushButton("Arrêter")
        bstop.clicked.connect(self.stop)
        bfetch = QPushButton("Récupérer les résultats")
        bfetch.clicked.connect(self.fetch)
        bagain = QPushButton("Reprendre ses options")
        bagain.setToolTip("Remet dans le formulaire le programme, la partition et les options de ce run")
        bagain.clicked.connect(self.reuse)
        bdelete = QPushButton("Supprimer")
        bdelete.setToolTip("Supprime le dossier du run sur inti (pas les sorties du programme)")
        bdelete.clicked.connect(self.delete)
        acts = QHBoxLayout()
        for b in (brefresh, bstop, bfetch, bagain, bdelete):
            acts.addWidget(b)
        acts.addStretch(1)
        self.log = LogView()

        # log tab: search + follow
        self.find = QLineEdit()
        self.find.setPlaceholderText("rechercher dans le log (Entrée : suivant)")
        self.find.setClearButtonEnabled(True)
        self.find.returnPressed.connect(self.find_next)
        self.follow = QCheckBox("Suivre la fin")
        self.follow.setChecked(True)
        self.follow.toggled.connect(lambda on: setattr(self.log, "follow", on))
        frow = QHBoxLayout()
        frow.addWidget(self.find, 1)
        frow.addWidget(self.follow)
        logw = QWidget()
        lgl = QVBoxLayout(logw)
        lgl.setContentsMargins(0, 0, 0, 0)
        lgl.addLayout(frow)
        lgl.addWidget(self.log, 1)
        find_act = QAction(self)
        find_act.setShortcut(QKeySequence.Find)
        find_act.triggered.connect(lambda: (self.tabs.setCurrentIndex(0), self.find.setFocus(),
                                            self.find.selectAll()))
        self.addAction(find_act)

        # summary tab
        self.sum_note = TLabel("")
        self.sum_table = QTableWidget(0, 0)
        self.sum_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sum_table.verticalHeader().setVisible(False)
        self.sum_table.setSortingEnabled(True)
        bsum = QPushButton("Rafraîchir le résumé")
        bsum.clicked.connect(self.load_summary)
        self.sum_view = QComboBox()
        self.sum_view.setVisible(False)
        self.sum_view.currentIndexChanged.connect(self.show_section)
        srow = QHBoxLayout()
        srow.addWidget(self.sum_view)
        srow.addWidget(self.sum_note, 1)
        srow.addWidget(bsum)
        sumw = QWidget()
        sl = QVBoxLayout(sumw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addLayout(srow)
        sl.addWidget(self.sum_table, 1)

        # chart tab: seconds per run against threads (multi_launch_stress.py)
        self.pt_data = None
        self.pt_data_b = None
        self.pt_build = QComboBox()
        self.pt_build.currentIndexChanged.connect(self.draw_points)
        self.pt_variant = QComboBox()
        for v in ("toutes", "chunk", "chunk2", "chunk2a"):
            self.pt_variant.addItem(T(v), v)
        self.pt_variant.currentIndexChanged.connect(self.draw_points)
        self.pt_cmp = QComboBox()
        self.pt_cmp.setMinimumContentsLength(24)
        self.pt_cmp.setToolTip("Superpose un autre run de multi_launch_stress.py : "
                               "même build et même variante de part et d'autre")
        self.pt_cmp.currentIndexChanged.connect(self.load_points)
        self.pt_color = QComboBox()
        for k in COLOR_BY:
            self.pt_color.addItem(T(k), k)
        self.pt_color.currentIndexChanged.connect(self.draw_points)
        self.pt_log = QCheckBox("Temps en échelle log")
        self.pt_log.toggled.connect(self.draw_points)
        self.pt_median = QCheckBox("Médiane par tranche de threads")
        self.pt_median.setChecked(True)
        self.pt_median.toggled.connect(self.draw_points)
        self.pt_ok_only = QCheckBox("Masquer les échecs")
        self.pt_ok_only.toggled.connect(self.draw_points)
        bpts = QPushButton("Rafraîchir")
        bpts.clicked.connect(self.load_points)
        prow2 = QHBoxLayout()
        prow2.addWidget(QLabel("Build"))
        prow2.addWidget(self.pt_build)
        prow2.addWidget(QLabel("Variante"))
        prow2.addWidget(self.pt_variant)
        prow2.addWidget(QLabel("Couleur"))
        prow2.addWidget(self.pt_color)
        prow2.addWidget(QLabel("Comparer avec"))
        prow2.addWidget(self.pt_cmp, 1)
        prow2.addWidget(bpts)
        prow3 = QHBoxLayout()
        prow3.addWidget(self.pt_log)
        prow3.addWidget(self.pt_median)
        prow3.addWidget(self.pt_ok_only)
        prow3.addStretch(1)
        self.pt_note = TLabel("")
        self.pt_view = ScatterView()
        ptw = QWidget()
        ptl = QVBoxLayout(ptw)
        ptl.setContentsMargins(0, 0, 0, 0)
        ptl.addLayout(prow2)
        ptl.addLayout(prow3)
        ptl.addWidget(self.pt_note)
        ptl.addWidget(self.pt_view, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(logw, "Log")
        self.tabs.addTab(sumw, "Résumé")
        self.tabs.addTab(ptw, "Graphique")
        self.tabs.currentChanged.connect(self.tab_changed)

        # end-of-run notification: the tray when there is one
        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.SP_ComputerIcon), self)
            self.tray.setToolTip("shrink-array — runs sur inti")
            self.tray.show()

        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.table)
        tl.addLayout(acts)
        split_v = QSplitter(Qt.Vertical)
        split_v.addWidget(tw)
        split_v.addWidget(self.tabs)
        split_v.setSizes([260, 480])

        split_h = QSplitter(Qt.Horizontal)
        split_h.addWidget(left)
        split_h.addWidget(split_v)
        split_h.setSizes([480, 900])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addLayout(top)
        cl.addWidget(split_h, 1)
        self.setCentralWidget(central)
        self.resize(1400, 860)

        quit_act = QAction("Quitter", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)

        self._preset_key = None
        self._launcher_key = None
        self.preset.currentIndexChanged.connect(self.preset_changed)
        self.mode.currentIndexChanged.connect(self.preset_changed)
        self.program.currentTextChanged.connect(self.program_changed)
        self.preset_changed()
        self.program_changed()
        self.time_s.valueChanged.connect(self.time_changed)
        self.qos.currentIndexChanged.connect(self.update_qos)
        self.directives.textChanged.connect(self.update_qos)
        self.update_qos()

        # --- polling: run list every 15 s, the shown log every 2 s
        self.t_list = QTimer(self)
        self.t_list.timeout.connect(self.refresh)
        self.t_list.start(15000)
        self.t_tail = QTimer(self)
        self.t_tail.timeout.connect(self.poll_tail)
        self.t_tail.start(2000)
        self.set_tooltips()
        self.translate_widgets()
        QTimer.singleShot(0, self.check)

    def change_language(self, *_):
        """Rebuild the window in the other language; the runs on inti do not
        care, and every field is read back from the settings."""
        self.save()
        i18n.set_lang(self.lang.currentData())
        self.settings.setValue("lang", i18n.LANG)
        w = Main()
        w.setGeometry(self.geometry())
        w.show()
        QApplication.instance().main_window = w
        self.close()

    def translate_widgets(self):
        """Static texts of the window, once built (labels, buttons, boxes,
        placeholders, tabs). Lists whose text is a key carry it as data."""
        if i18n.LANG == "fr":
            return
        self.setWindowTitle(T(self.windowTitle()))
        for w in self.findChildren(QLabel):
            if not isinstance(w, TLabel) and w.text():
                w.setText(T(w.text()))
        for w in self.findChildren(QPushButton) + self.findChildren(QCheckBox):
            w.setText(T(w.text()))
        for w in self.findChildren(QGroupBox):
            w.setTitle(T(w.title()))
        for w in self.findChildren(QLineEdit):
            if w.placeholderText():
                w.setPlaceholderText(T(w.placeholderText()))
        for i in range(self.tabs.count()):
            self.tabs.setTabText(i, T(self.tabs.tabText(i)))
        for c in (self.preset, self.mode, self.after_cond, self.check_builds):
            for i in range(c.count()):
                c.setItemText(i, T(c.itemText(i)))

    # ------------------------------------------------------------- tooltips
    TIPS = {
        "host": "Nom d'hôte ssh du calculateur (tel qu'il figure dans ~/.ssh/config).\n\n"
                "Tout passe par la connexion maître que vous ouvrez vous-même : ssh -fNM inti.",
        "preset": "La partition où le run est soumis. Elle fixe aussi l'architecture (ARM ou x86), "
                  "donc les valeurs par défaut des options.",
        "mode": "ccc_msub : job batch, indépendant de cette fenêtre et du nœud de login "
                "(recommandé).\n\nccc_mprun détaché : lancé depuis le nœud de login ; il dépend de ce "
                "nœud tant qu'il tourne.",
        "time_s": "Durée maximale du job (-T). Au-delà, le job est tué par le gestionnaire et finit "
                  "« terminé sans code ». Se tape « 1 h 30 », « 90 min », « 2 j » ou « 1-00:00:00 ».\n\n"
                  "Mémorisée par programme : compile_all.sh court, bench_and_test.py quelques heures, "
                  "multi_launch_stress.py long.",
        "qos": "La QoS du job (-Q), dont dépend la durée maximale (ccc_mqinfo) : test (prioritaire, "
               "court, 2 jobs soumis au plus), normal (1 jour), long (3 jours, pas sur milan-bxi), "
               "xlong (pas sur gh200-bxi).\n\n« auto » prend la plus prioritaire qui accepte la durée, "
               "puis la plus courte. Un -Q écrit dans les directives l'emporte.\n\nLes limites sont "
               "relues sur inti à chaque vérification de la connexion.",
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
        "pt_variant": "Ne montre que les runs de cette variante (chunk, chunk2, chunk2a).",
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
        "Arrêter": "Annule le job du run sélectionné (ccc_mdel), ou arrête ccc_mprun en mode détaché.",
        "Récupérer les résultats": "Copie sur ce portable le dossier du run et les fichiers de "
                                   "résultats du programme (CSV, log, erreurs ; jamais les output_*).",
        "Reprendre ses options": "Remet dans le formulaire le programme, la partition et les options "
                                 "du run sélectionné, pour le relancer ou le modifier.",
        "Supprimer": "Supprime le dossier du run sur inti (pas les sorties du programme). "
                     "Seulement pour un run terminé.",
        "Rafraîchir le résumé": "Relit le fichier résumé du run sélectionné.",
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
            "compile_all.sh": "Compilation de toutes les configurations, en job batch (ccc_msub) seulement. "
                              "Options lues dans le texte de son usage(), sans l'exécuter sur le nœud de login.",
        }
        for i in range(self.program.count()):
            t = prog_tips.get(self.program.itemText(i))
            if t:
                self.program.setItemData(i, tip(t), Qt.ToolTipRole)
        mode_tips = {"msub": self.TIPS["mode"].split("\n\n")[0],
                     "mprun": self.TIPS["mode"].split("\n\n")[1]}
        for i in range(self.mode.count()):
            self.mode.setItemData(i, tip(mode_tips[self.mode.itemData(i)]), Qt.ToolTipRole)
        tab_tips = ["Le log du run sélectionné, suivi en direct.",
                    "Le résumé du run : par build pour multi_launch_stress.py (runs, échecs, temps), "
                    "le CSV tel quel pour bench_and_test.py.",
                    "Temps de chaque run selon le nombre de threads (multi_launch_stress.py). "
                    "Survolez un point pour le détail ; « Comparer avec » superpose un autre run."]
        for i, t in enumerate(tab_tips):
            self.tabs.setTabToolTip(i, tip(t))

    # ---------------------------------------------------------------- utils
    def _get(self, key):
        return self.settings.value(key, DEFAULTS[key])

    def arch(self):
        return PRESETS[self.preset.currentData()]["arch"]

    def _directives_key(self, preset=None, mode=None):
        return f"directives2/{preset or self.preset.currentData()}/{mode or self.mode.currentData()}"

    def save(self):
        for key, w in (("host", self.host), ("repo", self.repo), ("runs_dir", self.runs_dir),
                       ("local_dir", self.local_dir), ("local_repo", self.local_repo)):
            self.settings.setValue(key, w.text())
        self.settings.setValue(f"time_s/{self.program.currentText()}", self.time_s.value())
        self.settings.setValue("qos", self.qos.currentData())
        self.settings.setValue("after_cond", self.after_cond.currentData())
        self.settings.setValue("delay_s", self.delay.value())
        self.settings.setValue("check_builds", self.check_builds.currentData())
        self.settings.setValue("preset", self.preset.currentData())
        self.settings.setValue("mode", self.mode.currentData())
        self.settings.setValue("program", self.program.currentText())
        if self._preset_key:
            self.settings.setValue(self._preset_key, self.directives.toPlainText())
            self.settings.setValue(self._launcher_key, self.launcher.text())

    def closeEvent(self, ev):
        self.save()
        self.t_list.stop()
        self.t_tail.stop()
        # The runs live on inti: only the pending ssh calls are dropped here.
        for p in list(self.remote.procs):
            p.finished.disconnect()
            p.kill()
            p.waitForFinished(2000)
        self.remote.procs.clear()
        if self.tray:
            self.tray.hide()
        super().closeEvent(ev)

    def error(self, msg):
        msg = T(msg)
        self.last_error = msg
        self.statusBar().showMessage(msg.splitlines()[0] if msg else "", 20000)
        print(msg, file=sys.stderr)

    def info(self, msg):
        self.statusBar().showMessage(T(msg), 10000)

    # ------------------------------------------------- partition / program
    def preset_changed(self, *_):
        if self._preset_key:   # keep what was typed for the previous one
            self.settings.setValue(self._preset_key, self.directives.toPlainText())
            self.settings.setValue(self._launcher_key, self.launcher.text())
        p = PRESETS[self.preset.currentData()]
        builtin = p["msub"] if self.mode.currentData() == "msub" else p["mprun"]
        self._preset_key = self._directives_key()
        self.directives.setPlainText(self.settings.value(self._preset_key, builtin))
        self._launcher_key = f"launcher/{self.preset.currentData()}"
        self.launcher.setText(self.settings.value(self._launcher_key, p.get("launcher", "")))
        is_msub = self.mode.currentData() == "msub"
        self.launcher.setVisible(is_msub)
        self.launcher_row_label.setVisible(is_msub)
        self.apply_arch_defaults()
        if hasattr(self, "qos_note"):
            self.update_qos()

    def program_changed(self, *_):
        prog = self.program.currentText()
        self.time_s.setValue(int(self.settings.value(f"time_s/{prog}",
                                                     PROGRAMS[prog].get("time_s", DEFAULTS["time_s"]))))
        cached = self.settings.value(f"schema/{self.program.currentText()}")
        self.form.pickers = {d: self.pick for d in PROGRAMS[prog].get("pickers", {})}
        self.check_builds.setEnabled("builds_opt" in PROGRAMS[prog])
        self.form.build(json.loads(cached) if cached else None)
        self.form.filter(self.opt_filter.text())
        self.apply_arch_defaults()
        if not cached:
            QTimer.singleShot(0, self.scan)

    def scan(self):
        prog = self.program.currentText()
        self.build_info.clear()      # re-read too, with the options
        self.info(f"lecture des options de {prog}…")
        self.last_error = None

        def done(r):
            if r is None:
                if prog == self.program.currentText() and not self.form.schema:
                    self.form.build(None, self.last_error or "erreur inconnue (voir la barre d'état)")
                return
            keep = self.form.values() if self.form.schema else None
            self.settings.setValue(f"schema/{prog}", json.dumps(r))
            if prog == self.program.currentText():
                self.form.build(r)
                if keep is not None:
                    self.form.set_values(keep)
                else:
                    self.apply_arch_defaults()
            self.form.filter(self.opt_filter.text())
            via = f" (python {r['python']} sur inti)" if r.get("python") else ""
            self.info(f"{prog} : {len(r['actions'])} options lues{via}")

        spec = PROGRAMS[prog]
        self.remote.helper("options", [self.repo.text().strip(), spec["script"], spec["scan"],
                                       spec["interp"]], done)

    def arch_defaults(self):
        prog = self.program.currentText()
        saved = self.settings.value(f"defaults/{prog}/{self.arch()}")
        return json.loads(saved) if saved else dict(ARCH_DEFAULTS.get((prog, self.arch()), {}))

    def apply_arch_defaults(self):
        if hasattr(self, "form"):
            self.form.set_values(self.arch_defaults())

    def save_arch_defaults(self):
        prog = self.program.currentText()
        self.settings.setValue(f"defaults/{prog}/{self.arch()}", json.dumps(self.form.values()))
        self.info(f"défauts enregistrés pour {prog} sur {self.arch()}")

    # ------------------------------------------------- start conditions
    def fill_after(self):
        """What a submission can wait for: the compile_all.sh runs not
        finished yet (a finished one needs no waiting)."""
        keep = self.after.currentData() if self.after.count() else ""
        self.after.blockSignals(True)
        self.after.clear()
        self.after.addItem(T("dès la soumission"), "")
        for r in self.runs:
            if (r["meta"].get("program") == "compile_all.sh"
                    and (r["status"] in self.ACTIVE or r["status"].startswith("en attente"))):
                self.after.addItem(T(f"après {r['id']}"), r["id"])
        self.after.setCurrentIndex(max(0, self.after.findData(keep)))
        self.after.blockSignals(False)
        self.after_cond.setEnabled(bool(self.after.currentData()))

    def form_value(self, dest):
        """The value option @p dest will have: the form's, else the script's
        default (a list default arrives as its repr)."""
        v = self.form.values().get(dest)
        if v not in (None, ""):
            return v
        for ac in (self.form.schema or {}).get("actions", []):
            if ac["dest"] == dest:
                d = ac["default"]
                if isinstance(d, str) and d[:1] in "[(":
                    try:
                        return list(ast.literal_eval(d))
                    except (ValueError, SyntaxError):
                        return d
                return d
        return None

    def build_check(self):
        """The build check run.sh makes before the command
        (runscripts.check_builds_call), or None: the builds the program will
        use, read from its own options."""
        opt = PROGRAMS[self.program.currentText()].get("builds_opt")
        mode = self.check_builds.currentData()
        if not opt or mode == "off" or not self.form.schema:
            return None
        builds = self.form_value(opt["builds"])
        if not builds:
            return None
        root = self.form_value(opt["root"])
        check = {"mode": mode, "root": root if isinstance(root, str) and root else "$CHUNK_BUILD_ROOT"}
        if isinstance(builds, str):
            parts = [x for x in re.split(r"[,\s]+", builds) if x]
            if len(parts) == 1:          # a file of names (as multi_launch_stress --builds), or one name
                check["builds_file"] = parts[0]
                return check
            builds = parts
        check["builds"] = [str(b) for b in builds]
        return check

    def command(self):
        prog = self.program.currentText()
        parts = [PROGRAMS[prog]["interp"], os.path.basename(PROGRAMS[prog]["script"])] + self.form.args()
        if self.extra.text().strip():
            parts.append(self.extra.text().strip())
        return " ".join(parts)

    def update_preview(self, *_):
        if hasattr(self, "preview"):
            self.preview.setPlainText(self.command())

    # ---------------------------------------------- --target / --exec pickers
    def arch_suffix(self):
        """The <arch> component of compile_all.sh's build dirs ends with uname -m."""
        return "aarch64" if self.arch() == "arm" else "x86_64"

    def pick(self, dest):
        """"Choisir…" of --target / --exec: what compile_all.sh can build on
        this partition's architecture, read on inti (cached until "Lire les
        options")."""
        sfx = self.arch_suffix()
        info = self.build_info.get(sfx)
        if info is None:
            self.info(f"lecture des builds de {sfx} sur inti…")

            def done(r):
                if r is not None:
                    self.build_info[sfx] = r
                    self.info(f"{len(r['configs'])} configurations, {len(r['dirs'])} build(s) {sfx} "
                              f"dans {r['build_rep']}")
                    self.pick(dest)
            self.remote.helper("build_targets", [self.repo.text().strip(), BUILD_REP, sfx], done)
            return
        ac, w, _lab = self.form.widgets[dest]
        multi = "comma" in (ac.get("help") or "").lower()
        kind = PROGRAMS[self.program.currentText()]["pickers"][dest]
        dlg = (self.target_dialog if kind == "configs" else self.exec_dialog)(info, w.text(), multi, sfx)
        if dlg.exec():
            w.setText(dlg.value())

    def target_dialog(self, info, value, multi, sfx):
        dirs = {d["name"]: d for d in info["dirs"]}
        names = sorted(set(info["configs"]) | set(dirs))
        items = []
        for n in names:
            d = dirs.get(n)
            if d and d["configured"]:
                when = time.strftime("%Y-%m-%d", time.localtime(d["mtime"]))
                note = f"configurée, {len(d['execs'])} exécutable(s), {when}"
            elif d:
                note = "dossier sans build.ninja"
            else:
                note = ""
            items.append((n, note))
        pats = {g: c for g, c in families(names, "-").items()}
        for kw in ("sanitizer", "debug", "check", "stats", "ledger", "cs"):
            hits = [n for n in names if kw in n]
            if 2 <= len(hits) < len(names):
                pats[f"*{kw}*"] = len(hits)
        order = sorted(pats.items(), key=lambda kv: (not (kv[0].startswith("*") and kv[0].endswith("*")),
                                                     -kv[1], kv[0]))
        patterns = [(g, f"motif : {c} configurations") for g, c in unique_globs(order, names)[:24]]
        intro = (f"Configurations de compile_all.sh (cmake-build-<nom>). « configurée » : son build "
                 f"existe déjà pour {sfx} dans {info['build_rep']}.")
        return PickerDialog(self, "--target : configurations", intro, items, patterns, value, multi)

    def exec_dialog(self, info, value, multi, sfx):
        target = self.form.widgets["target"][1].text() if "target" in self.form.widgets else ""
        import fnmatch
        dirs = [d for d in info["dirs"] if d["configured"]
                and (not split_list(target) or any(fnmatch.fnmatchcase(d["name"], g) for g in split_list(target)))]
        count = {}
        for d in dirs:
            for e in d["execs"]:
                count[e] = count.get(e, 0) + 1
        items = [(e, f"dans {c}/{len(dirs)} build(s)") for e, c in sorted(count.items())]
        patterns = [(g, f"motif : {c} exécutables")
                    for g, c in unique_globs(families(sorted(count), "_").items(), sorted(count))[:24]]
        if dirs:
            scope = f"des {len(dirs)} build(s) {sfx} configurés" + (f" retenus par --target={target}" if target else "")
            intro = (f"Exécutables {scope}. Ninja construit aussi leurs dépendances (bibliothèques chunk), "
                     f"pas les autres exécutables.")
        else:
            intro = (f"Aucun build {sfx} configuré" + (f" pour --target={target}" if target else "")
                     + f" dans {info['build_rep']} : rien à lister. Lancez d'abord compile_all.sh sans "
                       "--exec (ou tapez le nom à la main).")
        return PickerDialog(self, "--exec : exécutables", intro, items, patterns, value, multi)

    # ------------------------------------------------------------------ QoS
    def fill_qos(self, keep):
        self.qos.blockSignals(True)
        self.qos.clear()
        self.qos.addItem(T("auto (selon la durée)"), "auto")
        for name in qos.names(self.qos_rows):
            self.qos.addItem(name, name)
        self.qos.setCurrentIndex(max(0, self.qos.findData(keep)))
        self.qos.blockSignals(False)

    def time_changed(self, v):
        """The duration is remembered per program (compile_all.sh is short,
        multi_launch_stress.py long)."""
        self.settings.setValue(f"time_s/{self.program.currentText()}", v)
        self.update_qos()

    def qos_choice(self):
        """(qos to add with -Q or None, its row, error or None). A -Q already in
        the directives wins over the list: it is only checked."""
        directives = self.directives.toPlainText()
        fixed = qos.explicit_qos(directives)
        wanted = fixed or self.qos.currentData() or "auto"
        name, row, err = qos.choose(self.qos_rows, self.preset.currentData(), qos.job_cpus(directives),
                                    self.time_s.value(), wanted)
        return (None if fixed else name), row, err

    def update_qos(self, *_):
        name, row, err = self.qos_choice()
        if err:
            self.qos_note.setText(f"<span style='color:#c0392b'>{html.escape(T(err), quote=False)}</span>")
            return
        name = name or qos.explicit_qos(self.directives.toPlainText())
        bits = [T(f"max {qos.fmt(row['max_time'], i18n.LANG)}")]
        if row["priority"] > min(r["priority"] for r in self.qos_rows):
            bits.append(T("prioritaire"))
        if row["max_sub"] and row["max_sub"] <= 10:     # test: 2; normal/long: 100, not worth a word
            bits.append(T(f"{row['max_sub']} job(s) soumis au plus"))
        fixed = T(" (fixée dans les directives)") if qos.explicit_qos(self.directives.toPlainText()) else ""
        self.qos_note.setText(f"→ <b>{html.escape(name)}</b>{fixed} : " + ", ".join(bits)
                              if i18n.LANG == "fr" else
                              f"→ <b>{html.escape(name)}</b>{fixed}: " + ", ".join(bits))

    # ----------------------------------------------------------- connection
    def check(self):
        host = self.host.text().strip()
        self.conn.setText("vérification…")

        def master(ok):
            if not ok:
                self.conn.setText(f"<span style='color:#c0392b'>pas de connexion maître : "
                                  f"lancez <b>ssh -fNM {host}</b> dans un terminal</span>")
                return
            self.remote.helper("check", [self.repo.text().strip()], helper_ok)

        def helper_ok(r):
            if r is None:
                self.conn.setText("<span style='color:#c0392b'>connexion maître OK, mais python3 "
                                  "distant en échec (voir la barre d'état)</span>")
                return
            warn = "" if r.get("msub") else " — <b>ccc_msub introuvable</b>"
            rows = qos.parse_mqinfo(r.get("mqinfo") or "")
            if rows:
                self.settings.setValue("mqinfo", r["mqinfo"])
                self.qos_rows = rows
                self.fill_qos(self.qos.currentData())
                self.update_qos()
            self.inti_git = r.get("git")
            self.conn.setText(f"<span style='color:#1e8449'>connecté — {r['host']}, "
                              f"python {r['python']}</span>{warn}{self.git_line()}")
            self.refresh()
            if not self.form.schema:
                self.scan()

        self.remote.check_master(master)

    # --------------------------------------------------------------- launch
    def launch(self):
        self.save()
        prog = self.program.currentText()
        if not self.form.schema and ask(
                self, "Options non lues",
                "Les options du script n'ont pas été lues : seuls les arguments en plus seront "
                "passés.\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        pkey = self.preset.currentData()
        mode = self.mode.currentData()
        if PROGRAMS[prog].get("msub_only") and mode != "msub":
            self.error(f"{prog} se soumet par ccc_msub seulement (il ne doit pas tourner depuis le nœud "
                       "de login) : choisissez « ccc_msub (batch) »")
            return
        qname, _row, qerr = self.qos_choice()
        if qerr:
            self.error(qerr)
            return
        busy = [r["id"] for r in self.runs
                if r["status"] in ("en cours", "en attente", "démarrage")
                and r["meta"].get("partition") == pkey and r["meta"].get("program") == prog]
        if busy and ask(
                self, "Run déjà actif",
                f"{prog} a déjà un run actif sur {pkey} ({', '.join(busy)}).\n"
                "Deux runs simultanés de même architecture peuvent écrire le même répertoire "
                "(oracles, résumé).\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        cmd = self.command()
        after_id = self.after.currentData() or ""
        after = {"run": after_id, "cond": self.after_cond.currentData()} if after_id else None
        delay_s = self.delay.value() if self.delay_on.isChecked() else 0
        check = self.build_check()
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + prog.split(".")[0].split("_")[0] + "-" + pkey
        spec = {
            "mode": mode,
            "run.sh": make_run_sh(self.repo.text().strip(), PROGRAMS[prog]["cwd"], cmd, check),
            "msub": self.directives.toPlainText() if mode == "msub" else "",
            "launcher": self.launcher.text().strip() if mode == "msub" else "",
            "launch.sh": make_launch_sh(f"{' '.join(self.directives.toPlainText().split())} "
                                        f"-T{self.time_s.value()}" + (f" -Q {qname}" if qname else ""),
                                        after, delay_s)
                         if mode == "mprun" else "",
            "qos": qname or "",
            "after": after, "delay_s": delay_s,
            "name": re.sub(r"[^A-Za-z0-9_.-]", "_", f"{prog.split('.')[0]}_{pkey}")[:40],
            "time_s": self.time_s.value(),
            "meta": {"program": prog, "partition": pkey, "arch": self.arch(), "mode": mode,
                     "launcher": self.launcher.text().strip() if mode == "msub" else "",
                     "command": cmd, "values": self.form.values(), "extra": self.extra.text(),
                     "repo": self.repo.text().strip(), "time_s": self.time_s.value(),
                     "qos": qname or qos.explicit_qos(self.directives.toPlainText()) or "",
                     "qos_wanted": self.qos.currentData(),
                     "after": after, "delay_s": delay_s,
                     "check_builds": check["mode"] if check else "off",
                     "created": time.strftime("%F %T"),
                     **{k: PROGRAMS[prog][k] for k in ("out_re", "out_files", "progress_re",
                                                       "eta_re", "summary", "summary_file")}},
        }
        blob = base64.b64encode(json.dumps(spec).encode()).decode()

        def done(r):
            if r is None:
                return
            self.info(f"soumis : job {r['jobid']}" if "jobid" in r else f"lancé : {r['run_dir']}")
            self.current, self.offset = rid, 0
            self.log.clear()
            self.refresh(select=rid)

        self.remote.helper("launch", [self.runs_dir.text().strip(), rid, blob], done)

    # ----------------------------------------------------------------- runs
    def refresh(self, select=None):
        if self.list_busy:
            return
        self.list_busy = True

        def done(r):
            self.list_busy = False
            if r is None:
                return
            self.runs = r["runs"]
            self.fill_after()
            self.notify_changes()
            self.fill_compare()
            cur = next((x for x in self.runs if x["id"] == self.current), None)
            if cur and cur["status"] in self.ACTIVE:
                if self.tabs.currentIndex() == 1:
                    self.load_summary()
                elif self.tabs.currentIndex() == 2:
                    self.load_points()
            keep = select or self.current
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.runs))
            for i, run in enumerate(self.runs):
                commit, commit_tip = self.commit_text(run)
                vals = [run["id"], run["meta"].get("program", ""), run["meta"].get("partition", ""),
                        commit, run.get("jobid", ""), status_text(run), T(self.progress_text(run)),
                        run["last"]]
                for j, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if j == 3:
                        f = QFont("Monospace")
                        it.setFont(f)
                    if j == self.COL_STATUS:
                        s = run["status"]
                        color = ("#1e8449" if s == "fini" and run["code"] == 0 else
                                 "#c0392b" if s == "fini" or s.startswith("terminé") else
                                 "#2471a3" if s in ("en cours", "démarrage") else
                                 "#7d3c98" if s == "en attente" else "#b9770e")
                        it.setForeground(QColor(color))
                    it.setToolTip(commit_tip if j == 3 else
                                  run["meta"].get("command", "") if j < 2 else str(v))
                    self.table.setItem(i, j, it)
                if run["id"] == keep:
                    self.table.selectRow(i)
            self.table.blockSignals(False)
            if select:
                self.select()

        self.remote.helper("list", [self.runs_dir.text().strip()], done)

    def selected_run(self):
        rows = self.table.selectionModel().selectedRows()
        return self.runs[rows[0].row()] if rows and rows[0].row() < len(self.runs) else None

    def selected(self):
        r = self.selected_run()
        return r["id"] if r else None

    def select(self):
        rid = self.selected()
        if rid and rid != self.current:
            self.current, self.offset = rid, 0
            self.log.clear()
            self.poll_tail()
            self.pt_data = None
            self.fill_compare()
            self.tab_changed(self.tabs.currentIndex())

    def poll_tail(self):
        if not self.current or self.tail_busy:
            return
        self.tail_busy = True
        rid = self.current

        def done(r):
            self.tail_busy = False
            if r is None or rid != self.current:
                return
            if r["offset"] < self.offset:      # file restarted
                self.log.clear()
            if r["data"]:
                self.log.append_raw(r["data"])
            self.offset = r["offset"]
            if len(r["data"]) >= 512 * 1024:   # more to read
                QTimer.singleShot(0, self.poll_tail)

        self.remote.helper("tail", [self.runs_dir.text().strip(), rid, self.offset], done)

    def reuse(self):
        run = self.selected_run()
        if not run:
            return
        m = run["meta"]
        if m.get("program") in PROGRAMS:
            self.program.setCurrentText(m["program"])
        i = self.preset.findData(m.get("partition"))
        if i >= 0:
            self.preset.setCurrentIndex(i)
        i = self.mode.findData(m.get("mode"))
        if i >= 0:
            self.mode.setCurrentIndex(i)
        if m.get("time_s"):
            self.time_s.setValue(int(m["time_s"]))
        if m.get("qos_wanted"):
            self.qos.setCurrentIndex(max(0, self.qos.findData(m["qos_wanted"])))
        self.form.set_values(m.get("values", {}))
        self.extra.setText(m.get("extra", ""))
        if m.get("check_builds") and self.check_builds.isEnabled():
            self.check_builds.setCurrentIndex(max(0, self.check_builds.findData(m["check_builds"])))
        self.info(f"options de {run['id']} reprises")

    def stop(self):
        run = self.selected_run()
        if not run:
            return
        what = f"le job {run['jobid']} (ccc_mdel)" if run.get("jobid") else "ccc_mprun (SIGTERM)"
        if ask(self, "Arrêter", f"Arrêter le run {run['id']} ?\n{what}") != QMessageBox.Yes:
            return

        def done(r):
            if r is not None:
                self.info(f"{run['id']} : arrêt demandé")
            QTimer.singleShot(2000, self.refresh)

        self.remote.helper("stop", [self.runs_dir.text().strip(), run["id"]], done)

    @staticmethod
    def commit_text(run):
        """("1a2b3c4d*", tooltip) of the run's code; * = uncommitted changes,
        ≠ = the job started on another commit than it was submitted on."""
        g = run["meta"].get("git")
        if not g:
            return "", T("pas de commit enregistré (pas de git, ou run plus ancien)")
        txt = g["commit"][:8] + ("*" if g.get("dirty") else "")
        lines = [f"{g['commit']}", f"branche : {g.get('branch', '')}"]
        if g.get("dirty"):
            lines.append("* modifications non commitées dans le dépôt d'inti à la soumission")
        start = run.get("git_start")
        if start and start != g["commit"]:
            txt += " ≠"
            lines.append(f"≠ le job a démarré sur {start[:8]} : le dépôt a changé entre la "
                         "soumission et le démarrage")
        return txt, tip("\n".join(lines))

    @staticmethod
    def progress_text(run):
        p = run.get("progress")
        if not p:
            return ""
        done, total, eta = p[:3]
        phase = p[3] if len(p) > 3 else None
        txt = f"{done}/{total} ({100.0 * done / total:.0f} %)" if total else (f"{done}" if done else "")
        if phase:
            txt = (f"{phase} · " + txt) if txt else phase
        if eta and run["status"] in Main.ACTIVE:
            h, m = divmod(int(eta) // 60, 60)
            txt += (f" · reste ~{h} h {m:02d}" if h else
                    f" · reste ~{m} min" if m else " · reste < 1 min")
        return txt

    def local_git(self):
        """{commit, branch} of the local clone ("Dépôt local"), or None."""
        path = os.path.expanduser(self.local_repo.text().strip())
        if not path:
            return None
        import subprocess
        try:
            c = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            b = subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return {"commit": c, "branch": b} if c else None

    def git_line(self):
        """ — inti : 1a2b3c4d (branch) · local : … ≠, for the connection line."""
        g = getattr(self, "inti_git", None)
        if not g:
            return ""
        txt = f" — inti : <b>{g['commit'][:8]}</b>{'*' if g.get('dirty') else ''} ({html.escape(g['branch'])})"
        loc = self.local_git()
        if loc:
            if loc["commit"] == g["commit"]:
                txt += " · local : identique"
            else:
                txt += (f" · <span style='color:#b9770e'>local : <b>{loc['commit'][:8]}</b> "
                        f"({html.escape(loc['branch'])}) ≠ inti</span>")
        return txt

    def notify_changes(self):
        """A run that leaves the active states since the previous refresh."""
        now = {r["id"]: r["status"] for r in self.runs}
        if self.prev_status is not None:
            for r in self.runs:
                before = self.prev_status.get(r["id"])
                if before in self.ACTIVE and r["status"] not in self.ACTIVE:
                    st = status_text(r)
                    msg = T(f"{r['meta'].get('program', '')} sur {r['meta'].get('partition', '')} : ") + st
                    self.statusBar().showMessage(f"{r['id']} : {st}" if i18n.LANG == "fr" else f"{r['id']}: {st}", 10000)
                    if self.tray:
                        icon = (QSystemTrayIcon.Information if r["status"] == "fini" and r["code"] == 0
                                else QSystemTrayIcon.Warning)
                        self.tray.showMessage(f"Run {r['id']}", msg, icon, 15000)
                    QApplication.alert(self)
        self.prev_status = now

    def find_next(self):
        text = self.find.text()
        if not text:
            return
        if not self.log.find(text):
            self.log.moveCursor(QTextCursor.Start)       # wrap around
            if not self.log.find(text):
                self.info(f"« {text} » absent du log")
                return
        self.follow.setChecked(False)                    # stay on the match

    def load_summary(self):
        rid = self.current
        if not rid:
            return
        self.sum_note.setText("chargement…")

        def done(r):
            if r is None or rid != self.current:
                return
            secs = r.get("sections") or [r]
            self.sum_sections = secs
            keep = self.sum_view.currentText()
            self.sum_view.blockSignals(True)
            self.sum_view.clear()
            for s_ in secs:
                self.sum_view.addItem(T(s_.get("title", "")) or "—")
            if keep and self.sum_view.findText(keep) >= 0:
                self.sum_view.setCurrentIndex(self.sum_view.findText(keep))
            self.sum_view.blockSignals(False)
            self.sum_view.setVisible(len(secs) > 1)
            self.show_section()

        self.remote.helper("summary", [self.runs_dir.text().strip(), rid], done)

    def show_section(self, *_):
        secs = getattr(self, "sum_sections", None)
        if not secs:
            return
        r = secs[max(0, min(self.sum_view.currentIndex(), len(secs) - 1))]
        if True:
            self.sum_note.setText(r.get("note", ""))
            cols, rows = r.get("columns", []), r.get("rows", [])
            self.sum_table.setSortingEnabled(False)
            self.sum_table.clear()
            self.sum_table.setColumnCount(len(cols))
            self.sum_table.setHorizontalHeaderLabels([T(c) for c in cols])
            self.sum_table.setRowCount(len(rows))
            fails = set(r.get("fail_rows", []))
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    it = NumItem(T(v) if r.get("title") == "Phases" and j == 1 else v)
                    if i in fails:
                        it.setForeground(QColor("#c0392b"))
                    if row and row[0] == "TOTAL":
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                    self.sum_table.setItem(i, j, it)
            self.sum_table.resizeColumnsToContents()
            self.sum_table.setSortingEnabled(True)

    def tab_changed(self, i):
        if i == 1:
            self.load_summary()
        elif i == 2:
            self.load_points()

    def fill_compare(self):
        """The runs the current one can be compared with: the other
        multi_launch_stress.py runs, newest first."""
        if not hasattr(self, "pt_cmp"):
            return
        keep = self.pt_cmp.currentData()
        self.pt_cmp.blockSignals(True)
        self.pt_cmp.clear()
        self.pt_cmp.addItem(T("(aucun)"), None)
        for r in self.runs:
            if r["id"] != self.current and r["meta"].get("summary") == "multi_launch":
                self.pt_cmp.addItem(f"{r['id']}  [{T(r['status'])}]", r["id"])
        i = self.pt_cmp.findData(keep) if keep else 0
        self.pt_cmp.setCurrentIndex(max(0, i))
        self.pt_cmp.blockSignals(False)

    def load_points(self, *_):
        rid = self.current
        if not rid:
            return
        rid_b = self.pt_cmp.currentData()
        pending = {"a": None, "b": None}
        need = {"a", "b"} if rid_b else {"a"}

        def finish():
            if rid != self.current or rid_b != self.pt_cmp.currentData():
                return
            if any(pending[k] is None for k in need):
                return
            r = pending["a"]
            if "rows" not in r:
                self.pt_data = None
                self.pt_note.setText(r.get("note", ""))
                self.pt_view.setChart(QChart())
                self.pt_view.points = []
                return
            rb = pending["b"] if rid_b and "rows" in pending["b"] else None
            if rid_b and rb is None:
                self.error(f"{rid_b} : {pending['b'].get('note', 'pas de données')}")
            r["id"] = rid
            if rb:
                rb["id"] = rid_b
            first = self.pt_data is None
            self.pt_data, self.pt_data_b = r, rb
            builds = list(dict.fromkeys(r["builds"] + (rb["builds"] if rb else [])))
            keep = self.pt_build.currentData()
            self.pt_build.blockSignals(True)
            self.pt_build.clear()
            self.pt_build.addItem(T("tous"), "tous")
            for b in builds:
                self.pt_build.addItem(b, b)
            if not first and keep in builds:
                self.pt_build.setCurrentIndex(self.pt_build.findData(keep))
            self.pt_build.blockSignals(False)
            self.draw_points()

        def got(key):
            def done(r):
                pending[key] = r if r is not None else {"note": "erreur de lecture"}
                finish()
            return done

        self.remote.helper("points", [self.runs_dir.text().strip(), rid], got("a"))
        if rid_b:
            self.remote.helper("points", [self.runs_dir.text().strip(), rid_b], got("b"))

    def commits_note(self, rid_a, rid_b):
        """ — commits A → B, or a warning when they are the same code."""
        def c(rid):
            run = next((x for x in self.runs if x["id"] == rid), None)
            g = run and run["meta"].get("git")
            return (g["commit"][:8] + ("*" if g.get("dirty") else "")) if g else "?"
        ca, cb = c(rid_a), c(rid_b)
        if "?" in (ca, cb):
            return ""
        if ca == cb and "*" not in ca:
            return f" — même commit {ca} des deux côtés"
        return f" — commits {ca} → {cb}"

    def _select(self, r, log_y):
        """(ok, failed) points of one dataset after the Build / Variante
        filters; a point is (threads, seconds, tooltip, exe)."""
        build_sel = self.pt_build.currentData()
        var_sel = self.pt_variant.currentData()
        ok, fails = [], []
        label = f"<br><i>{html.escape(r['id'])}</i>" if self.pt_data_b else ""
        for run, b, e, th, sec, rc in r["rows"]:
            build, exe = r["builds"][b], r["exes"][e]
            if build_sel != "tous" and build != build_sel:
                continue
            if var_sel != "toutes" and exe_parts(exe)[0] != var_sel:
                continue
            if log_y and sec <= 0:
                continue
            tip = (f"<b>{sec:.3f} s</b> · {th} threads<br>"
                   f"{html.escape(build)} · {html.escape(exe)}<br>"
                   f"run {run}" + (f" · <b>code {rc}</b>" if rc else "") + label)
            (fails if rc else ok).append((th, sec, tip, exe))
        return ok, fails

    @staticmethod
    def _bins(pts, lo, width):
        """{bin: median seconds} of the bins holding at least 3 runs."""
        import statistics
        bins = {}
        for p in pts:
            bins.setdefault(int((p[0] - lo) // width), []).append(p[1])
        return {k: statistics.median(v) for k, v in bins.items() if len(v) >= 3}

    def draw_points(self, *_):
        r = self.pt_data
        if not r:
            return
        rb = self.pt_data_b
        dark = self.palette().window().color().lightness() < 128
        V = VIZ["dark" if dark else "light"]
        log_y = self.pt_log.isChecked()
        show_fail = not self.pt_ok_only.isChecked()

        chart = QChart()
        chart.setBackgroundBrush(QBrush(QColor(V["surface"])))
        chart.setBackgroundRoundness(0)
        what = self.pt_build.currentData()
        what = "tous les builds" if what in ("tous", None) else what
        if self.pt_variant.currentData() != "toutes":
            what += " · " + self.pt_variant.currentData()
        chart.setTitle(("Comparaison — " if rb else "") + "temps par run selon le nombre de threads — " + what)
        chart.setTitleBrush(QBrush(QColor(V["ink"])))
        chart.legend().setLabelColor(QColor(V["ink2"]))
        chart.legend().setAlignment(Qt.AlignTop)
        chart.legend().setMarkerShape(QLegend.MarkerShapeFromSeries)   # dot, triangle, line

        def scatter(name, pts, color, size, shape, alpha=170, ring=True):
            sr = QScatterSeries()
            sr.setName(name)
            sr.setMarkerShape(shape)
            sr.setMarkerSize(size)
            c = QColor(color)
            c.setAlpha(alpha)
            sr.setColor(c)
            sr.setBorderColor(QColor(V["surface"]) if ring else c)
            sr.setPen(QPen(QColor(V["surface"]) if ring else c, 1))
            sr.append([QPointF(p[0], p[1]) for p in pts])
            chart.addSeries(sr)

        def median_line(name, med, lo, width, color):
            pts = [QPointF(lo + (k + 0.5) * width, med[k]) for k in sorted(med)]
            if len(pts) < 2:
                return
            # A surface-coloured halo under the line: it stays readable on top
            # of a dense cloud of its own colour. The halo has no legend entry.
            halo = QLineSeries()
            halo.setPen(QPen(QColor(V["surface"]), 6))
            halo.append(pts)
            chart.addSeries(halo)
            for m in chart.legend().markers(halo):
                m.setVisible(False)
            ln = QLineSeries()
            ln.setName(name)
            ln.setPen(QPen(QColor(color), 2.5))
            ln.append(pts)
            chart.addSeries(ln)

        ok_a, fail_a = self._select(r, log_y)
        drawn, note_extra = [], ""
        if rb is None:
            key_of, slots = COLOR_BY[self.pt_color.currentData()]
            groups = {}
            for p in ok_a:
                cat = key_of(p[3])
                groups.setdefault(cat if cat in slots else "autres", []).append(p)
            for i, cat in enumerate(slots):
                if groups.get(cat):
                    scatter(f"{cat} ({len(groups[cat])})", groups[cat], V["series"][i], 7,
                            QScatterSeries.MarkerShapeCircle)
                    drawn += groups[cat]
            if groups.get("autres"):
                scatter(f"autres ({len(groups['autres'])})", groups["autres"], V["muted"], 7,
                        QScatterSeries.MarkerShapeCircle)
                drawn += groups["autres"]
            fail_pts = fail_a if show_fail else []
            if self.pt_median.isChecked() and len(ok_a) >= 5:
                lo = min(p[0] for p in ok_a)
                hi = max(p[0] for p in ok_a)
                width = max(1, round((hi - lo) / 15)) if hi > lo else 1
                median_line("médiane", self._bins(ok_a, lo, width), lo, width, V["ink2"])
        else:
            # Two runs: colour = the run (slots 1 and 2), each with its median.
            ok_b, fail_b = self._select(rb, log_y)
            short = lambda rid: rid[:15]
            for (ok, fails, rid), col in zip(((ok_a, fail_a, r["id"]), (ok_b, fail_b, rb["id"])),
                                             V["series"][:2]):
                if ok:
                    scatter(f"{short(rid)} ({len(ok)})", ok, col, 7, QScatterSeries.MarkerShapeCircle)
                    drawn += ok
            fail_pts = (fail_a + fail_b) if show_fail else []
            if len(ok_a) >= 5 and len(ok_b) >= 5:
                both = ok_a + ok_b
                lo = min(p[0] for p in both)
                hi = max(p[0] for p in both)
                width = max(1, round((hi - lo) / 15)) if hi > lo else 1
                med_a, med_b = self._bins(ok_a, lo, width), self._bins(ok_b, lo, width)
                if self.pt_median.isChecked():
                    median_line(f"médiane {short(r['id'])}", med_a, lo, width, V["series"][0])
                    median_line(f"médiane {short(rb['id'])}", med_b, lo, width, V["series"][1])
                common = [k for k in med_a if k in med_b and med_a[k] > 0]
                if common:
                    import math
                    g = math.exp(sum(math.log(med_b[k] / med_a[k]) for k in common) / len(common))
                    pct = (g - 1) * 100
                    sens = "plus lent" if pct > 0 else "plus rapide"
                    note_extra = (f"  —  {short(rb['id'])} / {short(r['id'])} : ×{g:.3f} "
                                  f"({abs(pct):.1f} % {sens}), moyenne géométrique des médianes "
                                  f"sur {len(common)} tranche(s) de {width} thread(s) communes"
                                  + self.commits_note(r["id"], rb["id"]))
                else:
                    note_extra = "  —  aucune tranche de threads commune aux deux runs"

        if fail_pts:
            # Failures last, on top of everything: a shape in ink, named in the
            # legend -- not a status red, which the colour-vision checks cannot
            # tell from the orange slot.
            scatter(f"échec ({len(fail_pts)})", fail_pts, V["ink"], 12,
                    QScatterSeries.MarkerShapeTriangle, alpha=255, ring=False)
            drawn += fail_pts

        ax = QValueAxis()
        ax.setTitleText("threads")
        ax.setLabelFormat("%d")
        ay = QLogValueAxis() if log_y else QValueAxis()
        ay.setTitleText("temps par run (s)")
        ay.setLabelFormat("%.3g" if log_y else "%.1f")
        if log_y:
            ay.setMinorTickCount(-1)
        for axis in (ax, ay):
            axis.setLabelsColor(QColor(V["muted"]))
            axis.setTitleBrush(QBrush(QColor(V["ink2"])))
            axis.setGridLineColor(QColor(V["grid"]))
            axis.setLinePenColor(QColor(V["axis"]))
            axis.setMinorGridLineVisible(False)
        chart.addAxis(ax, Qt.AlignBottom)
        chart.addAxis(ay, Qt.AlignLeft)
        for sr in chart.series():
            sr.attachAxis(ax)
            sr.attachAxis(ay)
        if drawn:
            xs = [p[0] for p in drawn]
            ys = [p[1] for p in drawn]
            pad = max(1, (max(xs) - min(xs)) * 0.03)
            ax.setRange(min(xs) - pad, max(xs) + pad)
            if log_y:
                lo_y, hi_y = min(ys) / 1.15, max(ys) * 1.15
                # Qt labels only the powers of the base: base 2 when the data
                # spans less than a decade, or the axis has no label at all.
                ay.setBase(10.0 if hi_y / lo_y >= 20 else 2.0)
                ay.setRange(lo_y, hi_y)
            else:
                ay.setRange(0, max(ys) * 1.08)
                ay.applyNiceNumbers()
            ax.applyNiceNumbers()
        chart.setTitle(T(chart.title()))
        for sr in chart.series():
            if sr.name():
                sr.setName(T(sr.name()))
        for a_ in chart.axes():
            a_.setTitleText(T(a_.titleText()))
        self.pt_view.setChart(chart)
        self.pt_view.points = [p[:3] for p in drawn]
        self.pt_view.log_y = log_y
        more = ""
        for d in (r, rb):
            if d and d["total"] > len(d["rows"]):
                more = " (les 30000 derniers runs de chaque fichier)"
        self.pt_note.setText(f"{len(drawn)} run(s) affiché(s){more}{note_extra}")

    def delete(self):
        run = self.selected_run()
        if not run:
            return
        if run["status"] in self.ACTIVE or run["status"].startswith("en attente"):
            self.error("arrêtez d'abord ce run (il est encore actif)")
            return
        if ask(
                self, "Supprimer",
                f"Supprimer le dossier du run {run['id']} sur inti ?\n"
                "(log, scripts, code de retour ; les sorties du programme restent en place)"
        ) != QMessageBox.Yes:
            return

        def done(r):
            if r is None:
                return
            if self.current == run["id"]:
                self.current = None
                self.log.clear()
            self.info(f"{run['id']} supprimé")
            self.refresh()

        self.remote.helper("delete", [self.runs_dir.text().strip(), run["id"]], done)

    def fetch(self):
        rid = self.selected()
        if not rid:
            return
        dest = os.path.join(os.path.expanduser(self.local_dir.text().strip()), rid)

        def done(blob):
            if blob is None:
                return
            try:
                data = base64.b64decode(blob)
                os.makedirs(dest, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                    members = [m for m in t.getmembers()
                               if m.isfile() and not os.path.isabs(m.name) and ".." not in m.name.split("/")]
                    if sys.version_info >= (3, 12):
                        t.extractall(dest, members=members, filter="data")
                    else:
                        t.extractall(dest, members=members)
            except (ValueError, tarfile.TarError, OSError) as e:
                self.error(f"récupération : {e}")
                return
            self.info(f"{len(members)} fichier(s) → {dest}")

        self.info(f"récupération de {rid}…")
        self.remote.helper("pack", [self.runs_dir.text().strip(), rid], done, binary=True)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    from PySide6.QtCore import QLocale
    i18n.set_lang(QSettings(APP, APP).value("lang", "fr" if QLocale.system().name().startswith("fr") else "en"))
    w = Main()
    app.main_window = w
    w.show()
    sys.exit(app.exec())

