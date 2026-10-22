# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The "Où et comment" part of the window: this machine or the cluster, the
connection check, the partition, the QoS.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import base64
import html
import json

from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from shrink_gui import i18n, qos
from shrink_gui.config import BATCH, DETACHED, LOCAL, LOCAL_DEFAULTS, PRESETS, SITE
from shrink_gui.i18n import T


class MachinePanel:
    # ---------------------------------------------------------------- utils
    @staticmethod
    def mqinfo_key():
        """The settings key of the QoS table last read: the CEA site keeps the
        historical key, another site its own."""
        return "mqinfo" if SITE.name == "inti" else f"mqinfo/{SITE.name}"

    @staticmethod
    def batch_arg():
        """The site's batch table, as the helper's check and stop take it."""
        return base64.b64encode(json.dumps(BATCH).encode()).decode()

    # ------------------------------------------------------------- machine
    def is_local(self):
        """The runs go on this machine ("cette machine"), not on the cluster."""
        return self.local_cb.isChecked()

    def machine(self):
        """The host the helper runs on: LOCAL, or the ssh host name."""
        return LOCAL if self.is_local() else self.host.text().strip()

    def machine_name(self):
        """The machine, as the labels name it."""
        return "cette machine" if self.is_local() else (self.host.text().strip() or "?")

    def _mkey(self, key, local=None):
        """The settings key of a field kept per machine (repository, runs
        directory): the cluster keeps the historical keys."""
        local = self.is_local() if local is None else local
        return f"local/{key}" if local else key

    def _mget(self, key):
        if self.is_local():
            return self.settings.value(self._mkey(key), LOCAL_DEFAULTS[key])
        return self._get(key)

    def fill_machine_lists(self):
        """The partitions and submission modes of the machine: the cluster's,
        or this machine's single one (no scheduler: in the background)."""
        local = self.is_local()
        for c in (self.preset, self.mode):
            c.blockSignals(True)
            c.clear()
        for key, p in PRESETS.items():
            if bool(p.get("local")) == local:
                self.preset.addItem(T(self.local_label()) if local else T(p["label"]), key)
        if local:
            self.mode.addItem(T("en arrière-plan sur cette machine"), "local")
        else:
            self.mode.addItem(T(BATCH["label"]), "msub")
            if DETACHED:
                self.mode.addItem(T(DETACHED["label"]), "mprun")
            self.preset.setCurrentIndex(max(0, self.preset.findData(self._get("preset"))))
            self.mode.setCurrentIndex(max(0, self.mode.findData(self._get("mode"))))
        for c in (self.preset, self.mode):
            c.blockSignals(False)

    def local_label(self):
        """This machine's "partition": what the check found there."""
        r = self.local_info or {}
        bits = [b for b in (r.get("machine"), f"{r['cpus']} cœurs" if r.get("cpus") else "",
                            f"{r['mem_gb']:g} Gio" if r.get("mem_gb") else "") if b]
        return "cette machine" + (" — " + ", ".join(bits) if bits else "")

    def update_machine_labels(self):
        name = self.machine_name()
        self.setWindowTitle(T(f"shrink-array — runs sur {name} (prototype)"))
        self.repo_label.setText(T(f"Dépôt ({name})"))
        self.git_label.setText(T(f"Git ({name})"))
        self.branch_label.setText(T(f"Branche ({name})"))
        self.runs_label.setText(T(f"Runs ({name})"))
        local = self.is_local()
        self.host.setEnabled(not local)
        for w in (self.qos_row, self.directives):    # the scheduler's QoS and directive lines
            self.where.setRowVisible(w, not local and (w is not self.qos_row or bool(BATCH["qos"])))

    def machine_changed(self, *_):
        """Cluster <-> this machine: each keeps its repository, runs directory,
        partition and mode; the run list and the options are read again."""
        was_local = not self.is_local()
        for key, w in (("repo", self.repo), ("runs_dir", self.runs_dir)):
            self.settings.setValue(self._mkey(key, was_local), w.text())
        if not was_local:
            self.settings.setValue("preset", self.preset.currentData())
            self.settings.setValue("mode", self.mode.currentData())
        self.settings.setValue("machine", LOCAL if self.is_local() else "")
        self.repo.setText(self._mget("repo"))
        self.runs_dir.setText(self._mget("runs_dir"))
        self.fill_machine_lists()
        self.preset_changed()
        self.update_machine_labels()
        self.runs, self.prev_status, self.current = [], None, None
        self.table.setRowCount(0)
        self.log.clear()
        self.clear_errors()
        self.inti_git = None
        self.show_git_state()
        self.build_info.clear()
        self.fill_after()
        self.set_tooltips()
        self.rescan = True
        self.check()

    def arch(self):
        return PRESETS[self.preset.currentData()]["arch"]

    def _directives_key(self, preset=None, mode=None):
        return f"directives2/{preset or self.preset.currentData()}/{mode or self.mode.currentData()}"

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
        # the launcher: of run.sh in the batch job, or on this machine
        has_launcher = self.mode.currentData() in ("msub", "local")
        self.launcher.setVisible(has_launcher)
        self.launcher_row_label.setVisible(has_launcher)
        self.launcher.setPlaceholderText(T(
            "vide : run.sh directement ; pour plafonner la mémoire : systemd-run --user --scope -q "
            "-p MemoryMax=16G -p MemorySwapMax=0" if self.mode.currentData() == "local"
            else "vide : la commande tourne directement dans le job"))
        self.apply_arch_defaults()
        if hasattr(self, "qos_note"):
            self.update_qos()

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
        if not BATCH["qos"] or not self.qos_rows:    # no QoS known: the scheduler's default
            return None, None, None
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
        if row is None:
            self.qos_note.setText("")
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
            self.conn_ok = ok
            if not ok:
                self.conn.setText(f"<span style='color:#c0392b'>pas de connexion maître : "
                                  f"lancez <b>ssh -fNM {host}</b> dans un terminal</span>")
                return
            self.remote.helper("check", [self.repo.text().strip(), self.batch_arg()], helper_ok)

        def helper_ok(r):
            if r is None:
                self.conn.setText("<span style='color:#c0392b'>python3 en échec sur cette machine "
                                  "(voir la barre d'état)</span>" if self.is_local() else
                                  "<span style='color:#c0392b'>connexion maître OK, mais python3 "
                                  "distant en échec (voir la barre d'état)</span>")
                return
            if self.is_local():
                self.local_info = r
                self.preset.setItemText(0, T(self.local_label()))
                self.inti_git = r.get("git")
                self.show_git_state()
                self.conn.setText(f"<span style='color:#1e8449'>cette machine — {r['host']}, "
                                  f"python {r['python']}</span>{self.git_line()}")
                self.refresh()
                if self.rescan or not self.form.schema:
                    self.rescan = False
                    self.scan()
                return
            warn = "" if r.get("msub") else " — <b>ccc_msub introuvable</b>"
            rows = qos.parse_mqinfo(r.get("mqinfo") or "")
            if rows:
                self.settings.setValue(self.mqinfo_key(), r["mqinfo"])
                self.qos_rows = rows
                self.fill_qos(self.qos.currentData())
                self.update_qos()
            self.inti_git = r.get("git")
            self.show_git_state()
            if self.inti_git:
                self.load_branches()
            self.conn.setText(f"<span style='color:#1e8449'>connecté — {r['host']}, "
                              f"python {r['python']}</span>{warn}{self.git_line()}")
            self.refresh()
            if self.rescan or not self.form.schema:
                self.rescan = False
                self.scan()

        self.remote.check_master(master)

    def watch_connection(self):
        """Every WATCH_MS (app.Main): is the cluster still answering? Losing
        it pops a window (once, until it comes back) and a tray message;
        when it comes back, the full check runs again. Before the first
        successful check there is nothing to lose: no popup."""
        if self.watch_busy or self.is_local():
            return
        self.watch_busy = True
        host = self.host.text().strip()

        def done(ok):
            self.watch_busy = False
            if host != self.host.text().strip() or self.is_local():
                return                    # the machine changed meanwhile
            if ok and not self.conn_ok:
                self.info(f"connexion à {host} rétablie")
                if self.tray:
                    self.tray.showMessage("shrink-array", T(f"connexion à {host} rétablie"),
                                          QSystemTrayIcon.Information, 10000)
                if self.conn_box:
                    self.conn_box.close()
                self.check()
            elif not ok and self.conn_ok:
                self.conn_ok = False
                self.connection_lost(host)

        self.remote.ping(done)

    def connection_lost(self, host):
        msg = T(f"La connexion à {host} ne répond plus.\n\n"
                f"Relancez ssh -fNM {host} dans un terminal : l'interface s'en apercevra "
                f"d'elle-même. Les runs déjà soumis continuent sur {host}.")
        self.conn.setText(f"<span style='color:#c0392b'>connexion perdue : "
                          f"lancez <b>ssh -fNM {html.escape(host)}</b> dans un terminal</span>")
        if self.tray:
            self.tray.showMessage("shrink-array", T(f"connexion à {host} perdue"),
                                  QSystemTrayIcon.Critical, 15000)
        QApplication.alert(self)
        if not self.conn_box:             # non-modal: the timers keep running behind it
            self.conn_box = QMessageBox(QMessageBox.Warning, T("Connexion perdue"), msg,
                                        QMessageBox.Ok, self)
            self.conn_box.setModal(False)
            self.conn_box.finished.connect(lambda _r: setattr(self, "conn_box", None))
            self.conn_box.show()
