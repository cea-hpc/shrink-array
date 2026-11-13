# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
""""Nettoyer les builds…": what the build directories hold that the next build
re-creates (object files, precompiled headers, static libraries), and its
removal (remote_helper.act_clean_builds).

A mixin of the main window (app.Main): its methods use the window's
attributes. The dialog lists every configured build directory of the build
root, every architecture: the space each one would free, its total, the last
clean. The executables, the shared libraries and the build files stay: the next
compile_all.sh recompiles everything and relinks, nothing has to be configured
again. The build check of run.sh ("Builds à jour") knows a cleaned build
(runscripts.CHECK_BUILDS_SH) and judges it by its sources until the next build."""
import base64
import datetime
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel, QMessageBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from shrink_gui.config import BUILD_REP, program_of
from shrink_gui.i18n import T
from shrink_gui.widgets import ask


def fmt_bytes(n):
    """1536 -> "1.5 Kio", in binary units, one decimal from Kio up."""
    n = float(n or 0)
    for unit in ("o", "Kio", "Mio", "Gio", "Tio"):
        if n < 1024 or unit == "Tio":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} Tio"


class SizeItem(QTableWidgetItem):
    """A size cell: shown as fmt_bytes, sorted by bytes."""

    def __init__(self, n):
        super().__init__(fmt_bytes(n))
        self.n = n or 0
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        return self.n < other.n if isinstance(other, SizeItem) else super().__lt__(other)


INTRO = ("Supprime de chaque build coché ce que la compilation suivante refait : les fichiers objets "
         "(.o, .o.d), les en-têtes précompilés et les bibliothèques statiques (.a). Restent les exécutables, "
         "les bibliothèques partagées et les fichiers de build : le prochain compile_all.sh recompile tout, "
         "sans reconfigurer.")


class CleanBuildsDialog(QDialog):
    """The build directories of a scan ({"build_rep", "dirs"}), checkable; the
    ones with something to free are checked."""

    COLS = ("Build", "Architecture", "Libérable", "Total", "Dernier nettoyage")

    def __init__(self, parent, scan):
        super().__init__(parent)
        self.setWindowTitle(T("Nettoyer les builds"))
        self.resize(820, 520)
        lay = QVBoxLayout(self)
        intro = QLabel(T(INTRO) + "\n" + T(f"Racine des builds : {scan['build_rep']}"))
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.PlainText)
        lay.addWidget(intro)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([T(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(self.COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        for row, d in enumerate(scan["dirs"]):
            self.table.insertRow(row)
            name = QTableWidgetItem(d["name"])
            name.setData(Qt.UserRole, d["path"])
            name.setToolTip(d["path"])
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked if d["removable"] else Qt.Unchecked)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(d["arch"]))
            self.table.setItem(row, 2, SizeItem(d["removable"]))
            self.table.setItem(row, 3, SizeItem(d["total"]))
            when = (datetime.datetime.fromtimestamp(d["cleaned"]).strftime("%Y-%m-%d %H:%M")
                    if d.get("cleaned") else "")
            self.table.setItem(row, 4, QTableWidgetItem(when))
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.DescendingOrder)
        self.table.itemChanged.connect(self._update_total)
        lay.addWidget(self.table, 1)
        self.total = QLabel()
        lay.addWidget(self.total)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setText(T("Nettoyer la sélection"))
        bb.button(QDialogButtonBox.Cancel).setText(T("Fermer"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._update_total()

    def _rows(self):
        return range(self.table.rowCount())

    def selected(self):
        """[(path, removable bytes)] of the checked rows."""
        out = []
        for r in self._rows():
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append((it.data(Qt.UserRole), self.table.item(r, 2).n))
        return out

    def _update_total(self, *_):
        sel = self.selected()
        free = sum(n for _, n in sel)
        self.total.setText(T(f"{len(sel)} build(s) coché(s), {fmt_bytes(free)} à libérer"))
        self.ok.setEnabled(bool(sel) and free > 0)


class BuildsPanel:
    def build_root(self):
        """The build root the helper scans: the site's (CHUNK_BUILD_REP) on the
        cluster, the repository itself on this machine (as pick() does)."""
        return self.repo.text().strip() if self.is_local() else BUILD_REP

    def clean_builds(self):
        """"Nettoyer les builds…": scan, dialog, confirmation, clean."""
        root = self.build_root()
        self.info(f"lecture des builds de {root} sur {self.machine_name()}…")

        def scanned(r):
            if r is None:
                return
            if not r["dirs"]:
                self.info(f"aucun build configuré dans {r['build_rep']}")
                return
            self.info(f"{len(r['dirs'])} build(s) dans {r['build_rep']}")
            dlg = CleanBuildsDialog(self, r)
            if dlg.exec() != QDialog.Accepted:
                return
            sel = dlg.selected()
            free = sum(n for _, n in sel)
            text = (f"Nettoyer {len(sel)} build(s) et libérer {fmt_bytes(free)} ? Le prochain compile_all.sh "
                    f"de ces builds recompilera tout.")
            compiling = [x for x in self.runs if x["status"] in self.ACTIVE
                         and program_of(x.get("meta", {})) == "compile_all.sh"]
            if compiling:
                text += "\n\n" + (f"{len(compiling)} compile_all.sh encore actif(s) : nettoyer un build qu'il "
                                    f"compile le fera échouer.")
            if ask(self, "Nettoyer les builds", text) != QMessageBox.Yes:
                return
            blob = base64.b64encode(json.dumps([p for p, _ in sel]).encode()).decode()
            self.info(f"nettoyage de {len(sel)} build(s)…")
            self.remote.helper("clean_builds", [root, "clean", blob], cleaned)

        def cleaned(r):
            if r is None:
                return
            freed = sum(c["freed"] for c in r["cleaned"])
            unknown = [c["path"] for c in r["cleaned"] if c["up_to_date"] is None]
            stale = [c["path"] for c in r["cleaned"] if c["up_to_date"] is False]
            msg = f"{len(r['cleaned'])} build(s) nettoyé(s), {fmt_bytes(freed)} libérés"
            if stale:
                msg += (f" ; {len(stale)} pas à jour avant le nettoyage : « Builds à jour » les tiendra pour "
                        f"pas à jour jusqu'au prochain compile_all.sh")
            elif unknown:
                msg += (f" ; {len(unknown)} non vérifié(s) (pas de ninja ici, ou rien de construit) : "
                        f"« Builds à jour » les tiendra pour pas à jour jusqu'au prochain compile_all.sh")
            self.info(msg)

        self.remote.helper("clean_builds", [root, "scan"], scanned)
