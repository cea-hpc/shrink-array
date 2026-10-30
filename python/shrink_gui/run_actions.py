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

from shrink_gui.config import PROGRAMS, program_of
from shrink_gui.i18n import T
from shrink_gui.widgets import ask


class RunActions:
    def reuse(self):
        run = self.selected_run()
        if run:
            self.reuse_run(run)

    def reuse_run(self, run):
        m = run["meta"]
        if program_of(m) in PROGRAMS:
            self.program.setCurrentText(program_of(m))
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
        if self.pool_opt():          # the number of nodes of its submission (1: a single job), as it was made
            if m.get("nodes"):
                self.nodes_layout.setCurrentIndex(
                    max(0, self.nodes_layout.findData("job" if m.get("layout") == "job" else "jobs")))
            self.set_nodes(int(m.get("nodes", 1)))   # the run's duration is already per node
        self.rescale_armed = True
        self.form.set_values(m.get("values", {}))
        self.extra.setText(m.get("extra", ""))
        if m.get("check_builds") and self.check_builds.isEnabled():
            self.check_builds.setCurrentIndex(max(0, self.check_builds.findData(m["check_builds"])))
        self.info(f"options de {run['id']} reprises")

    def stop(self):
        runs = self.selected_runs()
        if len(runs) > 1:
            return self.stop_many(runs)
        run = runs[0] if runs else None
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

    def stop_many(self, runs):
        active = [r for r in runs if r["status"] in self.ACTIVE or r["status"].startswith("en attente")
                  or r["status"].startswith("en cours")]
        if not active:
            self.info("aucun des runs sélectionnés n'est actif")
            return
        if ask(self, "Arrêter", f"Arrêter {len(active)} run(s) ?\n" + "\n".join(r["id"] for r in active[:15])
               + ("\n…" if len(active) > 15 else "")) != QMessageBox.Yes:
            return
        todo, failed = list(active), []

        def run_next():
            if not todo:
                self.info(f"arrêt demandé pour {len(active) - len(failed)} run(s)"
                          + (f", en échec : {', '.join(failed)}" if failed else ""))
                QTimer.singleShot(2000, self.refresh)
                return
            run = todo.pop(0)

            def done(r):
                if r is None:
                    failed.append(run["id"])
                run_next()

            self.remote.helper("stop", [self.runs_dir.text().strip(), run["id"], self.batch_arg()], done)

        run_next()

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

    def delete_many_dialog(self, runs, n_own, n_local):
        """Confirmation of "Supprimer" for several runs: (ok, also their own
        outputs on inti, also their local copies); the boxes are the ones of
        the single-run dialog, remembered alike."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(T("Supprimer"))
        lay = QVBoxLayout(dlg)
        names = "\n".join(r["id"] for r in runs[:15]) + ("\n…" if len(runs) > 15 else "")
        lay.addWidget(QLabel(T(f"Supprimer {len(runs)} runs ?\n\n{names}\n\nSur inti, leurs dossiers (log, "
                               "scripts, code de retour).")))
        c_out = QCheckBox(T(f"aussi les sorties propres à ces runs ({n_own} run(s) en ont)") if n_own else
                          T("pas de sorties propres à ces programmes"))
        c_out.setEnabled(bool(n_own))
        c_out.setChecked(bool(n_own) and self.settings.value("delete_outputs", "false") == "true")
        c_loc = QCheckBox(T(f"aussi les copies locales ({n_local})") if n_local else T("pas de copie locale"))
        c_loc.setEnabled(bool(n_local))
        c_loc.setChecked(bool(n_local) and self.settings.value("delete_local", "false") == "true")
        lay.addWidget(c_out)
        lay.addWidget(c_loc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return False, False, False
        if n_own:
            self.settings.setValue("delete_outputs", "true" if c_out.isChecked() else "false")
        if n_local:
            self.settings.setValue("delete_local", "true" if c_loc.isChecked() else "false")
        return True, c_out.isChecked() and bool(n_own), c_loc.isChecked() and bool(n_local)

    def local_copy(self, run):
        p = os.path.join(os.path.expanduser(self.local_dir.text().strip()), run["id"])
        return p if os.path.isdir(p) else None

    def delete_many(self, runs):
        busy = [r for r in runs if r["status"] in self.ACTIVE or r["status"].startswith("en attente")
                or r["status"].startswith("en cours")]
        runs = [r for r in runs if r not in busy]
        if not runs:
            self.error("les runs sélectionnés sont encore actifs : arrêtez-les d'abord")
            return
        own = {r["id"]: PROGRAMS.get(program_of(r["meta"]), {}).get("own_outputs") for r in runs}
        local = {r["id"]: self.local_copy(r) for r in runs}
        ok, with_out, with_local = self.delete_many_dialog(runs, sum(1 for v in own.values() if v),
                                                           sum(1 for v in local.values() if v))
        if not ok:
            return
        items = [dict({"id": r["id"]}, **({"own": own[r["id"]]} if with_out and own[r["id"]] else {}))
                 for r in runs]

        def done(r):
            if r is None:
                return
            res = r.get("results", {})
            gone = [rid for rid, x in res.items() if x.get("ok")]
            bad = {rid: x.get("error", "?") for rid, x in res.items() if not x.get("ok")}
            if self.current in gone:
                self.current = None
                self.log.clear()
            if with_local:
                import shutil
                for rid in gone:
                    if local.get(rid):
                        shutil.rmtree(local[rid], ignore_errors=True)
            kept = {rid: res[rid]["kept"] for rid in gone if res[rid].get("kept")}
            n_out = sum(len(res[rid].get("removed", [])) for rid in gone)
            msg = (f"{len(gone)} run(s) supprimé(s)" + (f" (+ {n_out} sortie(s) sur inti)" if n_out else "")
                   + (" (+ copies locales)" if with_local else ""))
            if kept:
                msg += f" ; sorties gardées pour {len(kept)} run(s) (pas les leurs : voir l'infobulle)"
            if busy:
                msg += f" ; {len(busy)} encore actif(s) laissé(s)"
            if bad:
                msg += f" ; {len(bad)} en échec (voir l'infobulle)"
            details = ([T("sorties gardées :")] + [f"  {k} : {T(v)}" for k, v in sorted(kept.items())]
                       if kept else []) + \
                      ([T("en échec :")] + [f"  {k} : {T(v)}" for k, v in sorted(bad.items())] if bad else [])
            (self.error if bad else self.info)(msg)
            self.statusBar().setToolTip("\n".join(details))
            if details:
                print("\n".join(details), file=sys.stderr)
            self.refresh()

        blob = base64.b64encode(json.dumps(items).encode()).decode()
        self.remote.helper("delete_many", [self.runs_dir.text().strip(), blob], done)

    def delete(self):
        runs = self.selected_runs()
        if len(runs) > 1:
            return self.delete_many(runs)
        run = runs[0] if runs else None
        if not run:
            return
        if run["status"] in self.ACTIVE or run["status"].startswith("en attente"):
            self.error("arrêtez d'abord ce run (il est encore actif)")
            return
        own = PROGRAMS.get(program_of(run["meta"]), {}).get("own_outputs")
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
                      + (" (+ copie locale)" if with_local else "")
                      + (" ; sorties gardées, pas les siennes (voir l'infobulle)" if r.get("kept") else ""))
            if r.get("kept"):
                self.statusBar().setToolTip(T(r["kept"]))
            self.refresh()

        # outputs that are not the run's own (none announced, a shared
        # directory a newer run rewrote) are kept; the run goes all the same
        opts = base64.b64encode(json.dumps({"own": own, "keep_others": True} if with_out else {})
                                .encode()).decode()
        self.remote.helper("delete", [self.runs_dir.text().strip(), run["id"], opts], done)

    def fetch(self):
        ids = self.selected_ids()
        if len(ids) > 1:
            todo = list(ids)

            def next_one():
                if todo:
                    self.fetch_one(todo.pop(0), then=next_one)
                else:
                    self.info(f"{len(ids)} run(s) récupéré(s) dans {self.local_dir.text().strip()}")
            return next_one()
        if ids:
            self.fetch_one(ids[0])

    def fetch_one(self, rid, then=None):
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
                if then:
                    then()
                return
            self.info(f"{len(members)} fichier(s) → {dest}")
            if then:
                then()

        self.info(f"récupération de {rid}…")
        self.remote.helper("pack", [self.runs_dir.text().strip(), rid], done, binary=True)
