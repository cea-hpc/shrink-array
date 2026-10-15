# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Erreurs tab: the run's error files, each failure's complete output, and the replay of the failures.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import base64
import json
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget

from shrink_gui import i18n
from shrink_gui.config import PROGRAMS
from shrink_gui.i18n import T, tip
from shrink_gui.runscripts import rerun_command
from shrink_gui.widgets import DurationBox, LogView, TLabel


class ErrorsTab:
    def build_errors_tab(self):
        """The Erreurs tab's widgets."""
        # errors tab: the error files the program writes in its output
        # directory (PROGRAMS[...]["errors_files"]: bench_errors.log...)
        self.err_note = TLabel("")
        self.err_note.setWordWrap(True)
        self.err_find = QLineEdit()
        self.err_find.setPlaceholderText("rechercher dans les erreurs (Entrée : suivant)")
        self.err_find.setClearButtonEnabled(True)
        self.err_find.returnPressed.connect(self.err_find_next)
        berr = QPushButton("Rafraîchir les erreurs")
        berr.clicked.connect(self.load_errors)
        self.err_rerun = QPushButton("Relancer les échecs…")
        self.err_rerun.clicked.connect(self.rerun_failed)
        self.err_rerun.setVisible(False)
        erow = QHBoxLayout()
        erow.addWidget(self.err_note, 1)
        erow.addWidget(self.err_find)
        erow.addWidget(self.err_rerun)
        erow.addWidget(berr)
        self.err_view = LogView()
        self.err_view.follow = False
        self.err_view.setMaximumBlockCount(0)       # the helper already bounds what it sends
        # the failures one by one (bench_and_test.py: failed_cases.jsonl), each
        # opening its complete output; the first item is the errors log itself
        self.err_list = QListWidget()
        self.err_list.currentItemChanged.connect(self.err_item_changed)
        self.err_list.setVisible(False)
        esplit = QSplitter(Qt.Horizontal)
        esplit.addWidget(self.err_list)
        esplit.addWidget(self.err_view)
        esplit.setSizes([320, 700])
        errw = QWidget()
        el = QVBoxLayout(errw)
        el.setContentsMargins(0, 0, 0, 0)
        el.addLayout(erow)
        el.addWidget(esplit, 1)
        self.err_busy = False
        self._err_shown = None
        self._err_result = None
        return errw

    def focus_find(self):
        """Ctrl+F: the search field of the errors tab when it is shown, else
        the log's."""
        if self.tabs.currentIndex() == self.TAB_ERRORS:
            self.err_find.setFocus()
            self.err_find.selectAll()
            return
        self.tabs.setCurrentIndex(self.TAB_LOG)
        self.find.setFocus()
        self.find.selectAll()

    def err_find_next(self):
        text = self.err_find.text()
        if not text:
            return
        if not self.err_view.find(text):
            self.err_view.moveCursor(QTextCursor.Start)       # wrap around
            if not self.err_view.find(text):
                self.info(f"« {text} » absent des erreurs")

    def clear_errors(self):
        self.err_view.clear()
        self.err_note.setText("")
        self.err_list.blockSignals(True)
        self.err_list.clear()
        self.err_list.blockSignals(False)
        self.err_list.setVisible(False)
        self.err_rerun.setVisible(False)
        self.tabs.tabBar().setTabTextColor(self.TAB_ERRORS, QColor())
        self._err_shown = None
        self._err_result = None

    @staticmethod
    def size_text(n):
        mb, kb, b = ("MB", "kB", "B") if i18n.LANG == "en" else ("Mo", "Ko", "o")
        return f"{n / 1e6:.1f} {mb}" if n >= 1e6 else f"{n / 1e3:.1f} {kb}" if n >= 1e3 else f"{n} {b}"

    def error_files_text(self, files):
        """The files act_errors / act_errfile send, one after the other, each
        under a header line."""
        parts = []
        for f in files:
            if f.get("missing"):
                parts.append(T(f"══ {f['name']} : introuvable ══") + "\n\n")
                continue
            when = time.strftime("%F %T", time.localtime(f["mtime"])) if f.get("mtime") else ""
            parts.append(T(f"══ {f['name']} · {self.size_text(f['size'])} · {when} ══") + "\n")
            if f.get("skipped"):
                parts.append(T(f"[gui] … les {self.size_text(f['skipped'])} du début ne sont pas affichés "
                               "(« Récupérer les résultats » copie le fichier complet)") + "\n")
            parts.append(f["data"] if f["data"].strip() else T("(vide)") + "\n")
            if not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append("\n")
        return "".join(parts)

    def show_error_text(self, text):
        if text == self._err_shown:      # unchanged: keep the scroll position and the selection
            return
        same_item = self._err_shown is not None and getattr(self, "_err_shown_key", None) == self._err_key()
        self._err_shown = text
        self._err_shown_key = self._err_key()
        bar = self.err_view.verticalScrollBar()
        pos = bar.value() if same_item else 0
        self.err_view.clear()
        self.err_view.append_raw(text)
        bar.setValue(pos)

    def _err_key(self):
        it = self.err_list.currentItem()
        return json.dumps(it.data(Qt.UserRole)) if it else None

    def fill_error_list(self, r):
        """Item 0: the errors log; then one per failure (failed_cases.jsonl)
        with its complete output, or the failed executions' logs found in
        run_logs/ when no failure points to one (an older bench_and_test.py)."""
        cases, logs = r.get("cases"), r.get("logs") or []
        keep = self._err_key()
        self.err_list.blockSignals(True)
        self.err_list.clear()
        head = QListWidgetItem(" + ".join(f["name"] for f in r.get("files") or []) or T("log d'erreurs"))
        head.setData(Qt.UserRole, {"kind": "errors"})
        self.err_list.addItem(head)
        with_logs = False
        for c in cases or []:
            it = QListWidgetItem(("✗ " if c["logs"] else "· ") + (c["summary"] or c["kind"]))
            it.setData(Qt.UserRole, {"kind": "case", "paths": c["logs"], "summary": c["summary"]})
            tipt = c["summary"]
            if c["logs"]:
                with_logs = True
                tipt += "\n\n" + "\n".join(c["logs"])
            else:
                it.setForeground(QColor("#898781"))
                tipt += "\n\n" + T("pas de sortie complète enregistrée pour cet échec (une comparaison ou "
                                     "un garde, un bench_and_test.py plus ancien, ou --no-full-log)")
            if not c["replayable"]:
                tipt += "\n" + T("non rejouable par --rerun-failed : il faut un run complet")
            it.setToolTip(tip(tipt))
            self.err_list.addItem(it)
        if not with_logs:
            for name in logs:
                it = QListWidgetItem("✗ " + name)
                it.setData(Qt.UserRole, {"kind": "log", "paths": [name]})
                it.setToolTip(tip(T("sortie complète d'une exécution en échec, trouvée dans le dossier "
                                    "des logs (écrite pendant ce run)")))
                self.err_list.addItem(it)
        row = 0
        for i in range(self.err_list.count()):
            if json.dumps(self.err_list.item(i).data(Qt.UserRole)) == keep:
                row = i
        self.err_list.setCurrentRow(row)
        self.err_list.blockSignals(False)
        self.err_list.setVisible(self.err_list.count() > 1)

    def load_errors(self, *_):
        """The run's error files (remote_helper act_errors), newest first, and
        its failures one by one."""
        run = next((x for x in self.runs if x["id"] == self.current), None)
        if not run or self.err_busy:
            return
        rid = run["id"]
        prog = PROGRAMS.get(run["meta"].get("program"), {})
        opts = {"files": prog.get("errors_files") or [], "cases": prog.get("failed_cases"),
                "full_logs": prog.get("full_logs")}
        self.err_busy = True
        if not self.err_view.toPlainText():
            self.err_note.setText("chargement…")

        def done(r):
            self.err_busy = False
            if r is None or rid != self.current:
                return
            self._err_result = r
            notes = []
            if r.get("note"):
                notes.append(r["note"])
            if run["meta"].get("rerun_of"):
                notes.append(f"relance des échecs de {run['meta']['rerun_of']} : la liste ne garde que ce "
                             "qui échoue encore, bench_errors.log est celui du run rejoué")
            if r.get("newer"):
                notes.append(f"attention : {r['newer']}, plus récent ou actif, écrit aussi dans ce dossier : "
                             "ces fichiers sont peut-être les siens")
            n, shown = r.get("count", 0), r.get("files") or []
            if shown:
                head = f"{n} erreur(s)" if n else "aucune erreur"
                notes.insert(0, f"{head} — {r['dir']}")
                if r.get("total_files", 0) > len(shown):
                    notes.append(f"les {len(shown)} plus récents sur {r['total_files']} fichiers")
            self.err_note.setText("\n".join(T(x) for x in notes))
            self.tabs.tabBar().setTabTextColor(self.TAB_ERRORS, QColor("#c0392b") if n else QColor())
            replayable = sum(1 for c in r.get("cases") or [] if c["replayable"])
            self.err_rerun.setVisible(bool(prog.get("rerun")))
            self.err_rerun.setEnabled(replayable > 0 and run["status"] not in self.ACTIVE)
            self.err_rerun.setToolTip(tip(
                f"Soumet un nouveau run qui rejoue les {replayable} échec(s) rejouable(s) de ce run "
                "(--rerun-failed), mêmes partition et options. Désactivé tant que le run tourne."))
            self.fill_error_list(r)
            self.err_item_changed()

        blob = base64.b64encode(json.dumps(opts).encode()).decode()
        self.remote.helper("errors", [self.runs_dir.text().strip(), rid, blob], done)

    def err_item_changed(self, *_):
        """Show the item chosen in the list: the errors log (already read), or
        the complete output of a failure (act_errfile; a finished file, read
        once per selection)."""
        it = self.err_list.currentItem()
        data = it.data(Qt.UserRole) if it else {"kind": "errors"}
        r = self._err_result or {}
        if data["kind"] == "errors":
            self.show_error_text(self.error_files_text(r.get("files") or []))
            return
        if not data.get("paths"):
            self.show_error_text(T("pas de sortie complète enregistrée pour cet échec") + "\n\n"
                                 + data.get("summary", "") + "\n")
            return
        if self._err_shown_key_is_current():
            return
        rid, key = self.current, self._err_key()

        def done(res):
            if res is None or rid != self.current or key != self._err_key():
                return
            self.show_error_text(self.error_files_text(res.get("files") or []))

        blob = base64.b64encode(json.dumps({"paths": data["paths"]}).encode()).decode()
        self.remote.helper("errfile", [self.runs_dir.text().strip(), rid, blob], done)

    def _err_shown_key_is_current(self):
        return self._err_shown is not None and getattr(self, "_err_shown_key", None) == self._err_key()

    # ------------------------------------------------------ rerun failures
    def rerun_dialog(self, run, replayable, other, newer):
        """(repetitions, -T) for the replay, or None (cancelled)."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(T("Relancer les échecs"))
        lay = QVBoxLayout(dlg)
        text = (f"Rejouer les {replayable} échec(s) rejouable(s) de {run['id']} "
                f"(bench_and_test.py --rerun-failed), sur la même partition, avec les mêmes options.")
        if other:
            text += f"\n{other} autre(s) (une comparaison, un garde) demandent un run complet."
        if newer:
            text += (f"\n\nAttention : {newer}, plus récent ou actif, écrit aussi dans ce dossier : "
                     "failed_cases.jsonl est peut-être le sien.")
        lab = QLabel(T(text))
        lab.setWordWrap(True)
        lay.addWidget(lab)
        form = QFormLayout()
        reps = QSpinBox()
        reps.setRange(1, 100)
        reps.setToolTip(tip("Chaque échec est rejoué jusqu'à N fois, arrêt au premier échec : "
                            "pour un échec intermittent (--rerun-repeat)."))
        dur = DurationBox()
        dur.setValue(3600)
        form.addRow(T("Répétitions"), reps)
        form.addRow(T("Durée max (-T)"), dur)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        return reps.value(), dur.value()

    def rerun_failed(self):
        """Submit a run replaying the failures of the run shown: its command
        with --rerun-failed --rerun-repeat N, its full logs apart
        (run_logs/rerun-<date>), same partition, mode and options."""
        run = next((x for x in self.runs if x["id"] == self.current), None)
        r = self._err_result
        spec = PROGRAMS.get(run["meta"].get("program"), {}).get("rerun") if run else None
        if not run or not r or not spec:
            return
        cases = r.get("cases") or []
        replayable = sum(1 for c in cases if c["replayable"])
        choice = self.rerun_dialog(run, replayable, len(cases) - replayable, r.get("newer"))
        if not choice:
            return
        repeat, time_s = choice
        logs = PROGRAMS[run["meta"]["program"]].get("full_logs") or "run_logs"
        log_dir = f"{logs}/rerun-{time.strftime('%Y%m%d-%H%M%S')}"
        cmd = rerun_command(run["meta"].get("command", ""), spec, repeat, log_dir)
        extra = rerun_command(run["meta"].get("extra", ""), spec, repeat, log_dir)
        self.reuse_run(run)
        old = self.time_s.value()
        self.time_s.blockSignals(True)          # not remembered as the program's duration
        self.time_s.setValue(time_s)
        try:
            self.launch(command=cmd, extra=extra, rerun_of=run["id"])
        finally:
            self.time_s.setValue(old)
            self.time_s.blockSignals(False)
