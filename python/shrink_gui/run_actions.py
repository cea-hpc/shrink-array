# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The buttons under the runs table: stop, reuse the options, delete, fetch
the results.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import base64
import io
import json
import os
import sys
import tarfile

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QLabel, QMessageBox, QVBoxLayout

from shrink_gui.config import PROGRAMS
from shrink_gui.i18n import T
from shrink_gui.widgets import ask


class RunActions:
    def reuse(self):
        run = self.selected_run()
        if run:
            self.reuse_run(run)

    def reuse_run(self, run):
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
        what = (f"le job {run['jobid']} (ccc_mdel)" if run.get("jobid") else
                "son groupe de processus (SIGTERM)" if run["meta"].get("mode") == "local" else
                "ccc_mprun (SIGTERM)")
        if ask(self, "Arrêter", f"Arrêter le run {run['id']} ?\n{what}") != QMessageBox.Yes:
            return

        def done(r):
            if r is not None:
                self.info(f"{run['id']} : arrêt demandé")
            QTimer.singleShot(2000, self.refresh)

        self.remote.helper("stop", [self.runs_dir.text().strip(), run["id"], self.batch_arg()], done)

    def delete_dialog(self, run, own, local):
        """Confirmation of "Supprimer": (ok, also the program outputs on inti,
        also the local copy). The two boxes are remembered."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(T("Supprimer"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(T(f"Supprimer le run {run['id']} ?\n\nSur inti, son dossier (log, scripts, "
                               "code de retour).")))
        c_out = QCheckBox(T("aussi le dossier de sortie du programme, propre à ce run" if own == "dir" else
                            "aussi ce qu'il a écrit dans le dossier de sortie partagé (" + ", ".join(own or []) +
                            " ; les fichiers des autres runs et oracle/ restent)" if own else
                            "pas de sorties propres à ce programme"))
        c_out.setEnabled(bool(own))
        c_out.setChecked(bool(own) and self.settings.value("delete_outputs", "false") == "true")
        c_loc = QCheckBox(T(f"aussi la copie locale {local}") if local else T("pas de copie locale"))
        c_loc.setEnabled(bool(local))
        c_loc.setChecked(bool(local) and self.settings.value("delete_local", "false") == "true")
        lay.addWidget(c_out)
        lay.addWidget(c_loc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return False, False, False
        if own:
            self.settings.setValue("delete_outputs", "true" if c_out.isChecked() else "false")
        if local:
            self.settings.setValue("delete_local", "true" if c_loc.isChecked() else "false")
        return True, c_out.isChecked() and bool(own), c_loc.isChecked() and bool(local)

    def delete(self):
        run = self.selected_run()
        if not run:
            return
        if run["status"] in self.ACTIVE or run["status"].startswith("en attente"):
            self.error("arrêtez d'abord ce run (il est encore actif)")
            return
        own = PROGRAMS.get(run["meta"].get("program"), {}).get("own_outputs")
        local = os.path.join(os.path.expanduser(self.local_dir.text().strip()), run["id"])
        local = local if os.path.isdir(local) else None
        ok, with_out, with_local = self.delete_dialog(run, own, local)
        if not ok:
            return

        def done(r):
            if r is None:
                return
            if self.current == run["id"]:
                self.current = None
                self.log.clear()
            if with_local:
                import shutil
                shutil.rmtree(local, ignore_errors=True)
            n = len(r.get("removed", []))
            self.info(f"{run['id']} supprimé" + (f" (+ {n} sortie(s) sur inti)" if n else "")
                      + (" (+ copie locale)" if with_local else ""))
            self.refresh()

        opts = base64.b64encode(json.dumps({"own": own} if with_out else {}).encode()).decode()
        self.remote.helper("delete", [self.runs_dir.text().strip(), run["id"], opts], done)

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
