# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The main window."""
import sys
import time

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton,
                               QScrollArea, QSplitter, QStyle, QSystemTrayIcon, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from shrink_gui import i18n, qos
from shrink_gui.config import APP, DEFAULTS, LOCAL, PROGRAMS, QOS_FALLBACK
from shrink_gui.git_panel import GitPanel
from shrink_gui.i18n import T, status_text, tip
from shrink_gui.launch_form import LaunchForm
from shrink_gui.machine_panel import MachinePanel
from shrink_gui.remote import Remote
from shrink_gui.run_actions import RunActions
from shrink_gui.tab_bench_chart import BenchChart
from shrink_gui.tab_chart import ChartTab
from shrink_gui.tab_errors import ErrorsTab
from shrink_gui.tab_speedup_chart import SpeedupChart
from shrink_gui.tab_summary import SummaryTab
from shrink_gui.tooltips import Tooltips
from shrink_gui.widgets import DurationBox, LogView, OptionsForm, TLabel


class Main(MachinePanel, LaunchForm, RunActions, GitPanel, SummaryTab, ChartTab, BenchChart, ErrorsTab,
           SpeedupChart, Tooltips, QMainWindow):
    COLS = ["Run", "Programme", "Partition", "Commit", "Job", "Statut", "Avancement", "Erreurs",
            "Durée", "Dernière ligne"]
    COL_STATUS = 5
    COL_ERRORS = 7
    COL_DURATION = 8
    # the tabs under the runs table, in this order
    TAB_LOG, TAB_SUMMARY, TAB_CHART, TAB_ERRORS = range(4)
    ACTIVE = ("en cours", "en attente", "démarrage")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("shrink-array — runs sur inti (prototype)")
        self.settings = QSettings(APP, APP)
        self.remote = Remote(self, self.machine, self.error)
        self.runs = []
        self.clock_skew = 0.0      # laptop clock - cluster clock, from the last list
        self.current = None      # id of the run shown in the log
        self.offset = 0
        self.tail_busy = False
        self.list_busy = False
        self.prev_status = None  # id -> status at the previous refresh (notifications)
        self.prev_errors = {}    # id -> errors at the previous refresh (first-error notice)
        self.build_info = {}     # arch suffix -> act_build_targets (the "Choisir…" lists)
        self.local_info = None   # this machine, from the check: machine, cpus, mem_gb
        self.rescan = False      # read the options again at the next check (machine changed)

        # --- connection: the cluster through ssh, or this machine
        self.host = QLineEdit(self._get("host"))
        self.local_cb = QCheckBox("cette machine")
        self.local_cb.setChecked(self.settings.value("machine", DEFAULTS["machine"]) == LOCAL)
        self.host.setEnabled(not self.local_cb.isChecked())
        self.conn = TLabel("?")
        bcheck = QPushButton("Vérifier la connexion")
        bcheck.clicked.connect(self.check)
        top = QHBoxLayout()
        top.addWidget(QLabel("Calculateur :"))
        top.addWidget(self.host)
        top.addWidget(self.local_cb)
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
        self.mode = QComboBox()
        self.fill_machine_lists()
        self.time_s = DurationBox()
        self.time_s.setValue(int(self._get("time_s")))
        # QoS: the limits of the site's qos_info (read by the connection check,
        # kept per site, else the site's [qos] fallback), "auto" picks one from
        # the duration
        self.qos_rows = (qos.parse_mqinfo(self.settings.value(self.mqinfo_key(), "")) or
                         qos.parse_mqinfo(QOS_FALLBACK))
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
        self.repo = QLineEdit(self._mget("repo"))
        self.runs_dir = QLineEdit(self._mget("runs_dir"))
        self.local_dir = QLineEdit(self._get("local_dir"))
        where = QFormLayout()
        self.where = where
        where.addRow("Partition", self.preset)
        where.addRow("Soumission", self.mode)
        where.addRow("Durée max (-T)", self.time_s)
        self.qos_row = qrow
        where.addRow("QoS (-Q)", qrow)
        where.addRow("Démarrage", arow)
        where.addRow("Builds à jour", self.check_builds)
        where.addRow("Directives", self.directives)
        self.launcher = QLineEdit()
        self.launcher.setPlaceholderText("vide : la commande tourne directement dans le job")
        self.launcher.setToolTip("ccc_msub : ce qui lance run.sh sur le nœud réservé (le lanceur de la "
                                 "partition dans le fichier de site ; vide : directement dans le job).\n"
                                 "Mémorisé par partition.")
        self.launcher_row_label = QLabel("Lanceur (job)")
        where.addRow(self.launcher_row_label, self.launcher)
        self.repo_label = QLabel("Dépôt (inti)")
        where.addRow(self.repo_label, self.repo)
        # Branch and state of inti's repository, and the pulls.
        self.git_state = TLabel("?")
        self.git_state.setTextFormat(Qt.RichText)
        self.bpull = QPushButton("git pull")
        self.bpull.clicked.connect(lambda: self.git_pull(rebase=False))
        self.bpull_rebase = QPushButton("git pull --rebase")
        self.bpull_rebase.clicked.connect(lambda: self.git_pull(rebase=True))
        grow = QHBoxLayout()
        grow.addWidget(self.git_state, 1)
        grow.addWidget(self.bpull)
        grow.addWidget(self.bpull_rebase)
        self.git_label = QLabel("Git (inti)")
        where.addRow(self.git_label, grow)
        # The repository's branches (load_branches: after a git fetch) and the
        # switch to the one chosen (git_switch). Created disabled: show_git_state
        # enables them once the repository is known.
        self.branch = QComboBox()
        self.branch.currentIndexChanged.connect(lambda *_: self.show_git_state())
        self.bswitch = QPushButton("Changer de branche")
        self.bswitch.clicked.connect(self.git_switch)
        self.bbranches = QPushButton("Relire les branches")
        self.bbranches.clicked.connect(self.load_branches)
        for b in (self.bswitch, self.bbranches):
            b.setEnabled(False)
        brow = QHBoxLayout()
        brow.addWidget(self.branch, 1)
        brow.addWidget(self.bswitch)
        brow.addWidget(self.bbranches)
        self.branch_label = QLabel("Branche (inti)")
        where.addRow(self.branch_label, brow)
        self.runs_label = QLabel("Runs (inti)")
        where.addRow(self.runs_label, self.runs_dir)
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
        bdelete.setToolTip("Supprime le dossier du run sur inti (au choix : ses sorties, la copie locale)")
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
        find_act.triggered.connect(self.focus_find)
        self.addAction(find_act)

        sumw = self.build_summary_tab()
        ptw = self.build_chart_tab()
        errw = self.build_errors_tab()

        self.tabs = QTabWidget()
        self.tabs.addTab(logw, "Log")
        self.tabs.addTab(sumw, "Résumé")
        self.tabs.addTab(ptw, "Graphique")
        self.tabs.addTab(errw, "Erreurs")
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
        self.update_machine_labels()
        self.local_cb.toggled.connect(self.machine_changed)
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
        # the Durée column of the running runs, every second between two lists
        self.t_clock = QTimer(self)
        self.t_clock.timeout.connect(self.tick_durations)
        self.t_clock.start(1000)
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

    def _get(self, key):
        return self.settings.value(key, DEFAULTS[key])

    def save(self):
        for key, w in (("host", self.host), ("local_dir", self.local_dir), ("local_repo", self.local_repo)):
            self.settings.setValue(key, w.text())
        for key, w in (("repo", self.repo), ("runs_dir", self.runs_dir)):
            self.settings.setValue(self._mkey(key), w.text())
        self.settings.setValue("machine", LOCAL if self.is_local() else "")
        self.settings.setValue(f"time_s/{self.program.currentText()}", self.time_s.value())
        self.settings.setValue("qos", self.qos.currentData())
        self.settings.setValue("after_cond", self.after_cond.currentData())
        self.settings.setValue("delay_s", self.delay.value())
        self.settings.setValue("check_builds", self.check_builds.currentData())
        if not self.is_local():      # this machine has a single partition and mode
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
        self.t_clock.stop()
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

    # ----------------------------------------------------------------- runs
    def refresh(self, select=None):
        if self.list_busy:
            return
        self.list_busy = True
        source = (self.is_local(), self.runs_dir.text().strip())

        def done(r):
            self.list_busy = False
            if source != (self.is_local(), self.runs_dir.text().strip()):
                # The list of a machine or runs directory the window has left
                # meanwhile (a refresh asked for the new one found this one
                # still running, and returned): list the current one.
                self.refresh(select)
                return
            if r is None:
                return
            self.runs = r["runs"]
            if r.get("now"):
                self.clock_skew = time.time() - r["now"]
            self.fill_after()
            self.notify_changes()
            self.fill_compare()
            cur = next((x for x in self.runs if x["id"] == self.current), None)
            if cur and cur["status"] in self.ACTIVE:
                self.tab_changed(self.tabs.currentIndex())
            keep = select or self.current
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.runs))
            for i, run in enumerate(self.runs):
                commit, commit_tip = self.commit_text(run)
                vals = [run["id"], run["meta"].get("program", ""), run["meta"].get("partition", ""),
                        commit, run.get("jobid", ""), status_text(run), T(self.progress_text(run)),
                        self.errors_text(run), self.duration_text(run), run["last"]]
                for j, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if j == 3:
                        f = QFont("Monospace")
                        it.setFont(f)
                    if j == self.COL_STATUS:
                        s = run["status"]
                        color = ("#1e8449" if s == "fini" and run["code"] == 0 else
                                 "#c0392b" if s == "fini" or s.startswith(("terminé", "soumission refusée")) else
                                 "#2471a3" if s in ("en cours", "démarrage") else
                                 "#7d3c98" if s == "en attente" else "#b9770e")
                        it.setForeground(QColor(color))
                    if j == self.COL_ERRORS and (run.get("errors") or [0])[0]:
                        it.setForeground(QColor("#c0392b"))
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                    if j == self.COL_DURATION:
                        self.style_duration(it, run)
                        self.table.setItem(i, j, it)
                        continue
                    if j == self.COL_ERRORS:
                        it.setTextAlignment(Qt.AlignCenter)
                        it.setToolTip(tip(self.errors_tip(run)))
                        self.table.setItem(i, j, it)
                        continue
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
            self.current, self.offset = rid, -1     # -1: the end of the log first (act_tail)
            self.log.clear()
            self.clear_errors()
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
            if r.get("skipped"):
                self.log.append_raw(T(f"[gui] … les {r['skipped'] / 1e6:.1f} premiers Mo du log ne sont pas "
                                      "affichés (« Récupérer les résultats » copie le log complet)") + "\n")
            if r["data"]:
                self.log.append_raw(r["data"])
            self.offset = r["offset"]
            if r.get("more"):                 # appended faster than one block: read on
                QTimer.singleShot(0, self.poll_tail)

        self.remote.helper("tail", [self.runs_dir.text().strip(), rid, self.offset], done)

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

    # ------------------------------------------------------------ durations
    @staticmethod
    def fmt_elapsed(seconds):
        """Seconds -> "3:59:50", "1 j 02:03:04" ("1 d ..." in English)."""
        v = max(0, int(seconds))
        d, v = divmod(v, 86400)
        h, v = divmod(v, 3600)
        m, sec = divmod(v, 60)
        hms = f"{h}:{m:02d}:{sec:02d}"
        return (f"{d} {'d' if i18n.LANG == 'en' else 'j'} {h:02d}:{m:02d}:{sec:02d}") if d else hms

    def elapsed(self, run):
        """(seconds, over) of @p run: since its start up to its end, or up to
        now (cluster clock) while it runs; (None, False) when it has not
        started (or the start is unknown)."""
        started, ended = (run.get("times") or [None, None])[:2]
        if started is None:
            return None, False
        if ended is not None:
            return ended - started, True
        if run["status"] not in self.ACTIVE and not run["status"].startswith("en cours"):
            return None, False            # over, end unknown: no growing counter
        return time.time() - self.clock_skew - started, False

    def duration_text(self, run):
        s, _over = self.elapsed(run)
        return "" if s is None else self.fmt_elapsed(s)

    def duration_tip(self, run):
        started, ended = (run.get("times") or [None, None])[:2]
        if started is None:
            return T("Pas encore démarré (ou début inconnu).")
        at = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t + self.clock_skew))
        lines = [T("démarré : ") + at(started)]
        s, over = self.elapsed(run)
        if over:
            lines.append(T("fini : ") + at(ended))
            lines.append(T("durée : ") + self.fmt_elapsed(s))
        elif s is not None:
            lines.append(T("en cours depuis ") + self.fmt_elapsed(s))
        limit = run["meta"].get("time_s")
        if limit and run["meta"].get("mode") in ("msub", "local"):    # the detached mode has no -T
            lines.append(T("durée max demandée : ") + self.fmt_elapsed(limit))
        return "\n".join(lines)

    def style_duration(self, it, run):
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        it.setToolTip(self.duration_tip(run))
        s, over = self.elapsed(run)
        f = QFont("Monospace")
        f.setItalic(s is not None and not over)       # still counting
        it.setFont(f)

    def tick_durations(self):
        """Advances the Durée of the running runs between two lists."""
        for i, run in enumerate(self.runs[:self.table.rowCount()]):
            s, over = self.elapsed(run)
            if s is None or over:
                continue
            it = self.table.item(i, self.COL_DURATION)
            if it is not None:
                it.setText(self.fmt_elapsed(s))
                it.setToolTip(self.duration_tip(run))

    @staticmethod
    def errors_text(run):
        """The Erreurs column: how many so far (bench_and_test.py's events,
        multi_launch_stress.py's failed runs), empty when unknown."""
        e = run.get("errors")
        return "" if e is None else str(e[0])

    @staticmethod
    def errors_tip(run):
        e = run.get("errors")
        if e is None:
            return "Pas de décompte d'erreurs pour ce run."
        if run["meta"].get("summary") == "multi_launch":
            return f"{e[0]} run(s) en échec (timings_summary.csv). Le détail : onglet Erreurs."
        last = f"\n\nDernière : {e[1]}" if e[1] else ""
        return f"{e[0]} erreur(s) (events.jsonl). Le détail : onglet Erreurs.{last}"

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

    def notify_changes(self):
        """A run that leaves the active states since the previous refresh, and
        the first error of an active one (a long run is worth stopping or
        looking at before it ends)."""
        now = {r["id"]: r["status"] for r in self.runs}
        errs = {r["id"]: (r.get("errors") or [0])[0] for r in self.runs}
        if self.prev_status is not None:
            for r in self.runs:
                if (r["status"] in self.ACTIVE and errs[r["id"]] > 0
                        and self.prev_errors.get(r["id"], 0) == 0 and r["id"] in self.prev_status):
                    last = (r.get("errors") or [0, None])[1]
                    msg = T(f"{r['meta'].get('program', '')} sur {r['meta'].get('partition', '')} : "
                            f"première erreur, le run continue") + (f"\n{last}" if last else "")
                    self.statusBar().showMessage(T(f"{r['id']} : première erreur") + (f" — {last}" if last else ""),
                                                 15000)
                    if self.tray:
                        self.tray.showMessage(f"Run {r['id']}", msg, QSystemTrayIcon.Warning, 15000)
                    QApplication.alert(self)
        self.prev_errors = errs
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

    def tab_changed(self, i):
        """Load the tab shown (the log has its own 2 s polling)."""
        if i == self.TAB_SUMMARY:
            self.load_summary()
        elif i == self.TAB_CHART:
            self.load_points()
        elif i == self.TAB_ERRORS:
            self.load_errors()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    from PySide6.QtCore import QLocale
    i18n.set_lang(QSettings(APP, APP).value("lang", "fr" if QLocale.system().name().startswith("fr") else "en"))
    w = Main()
    app.main_window = w
    w.show()
    sys.exit(app.exec())

