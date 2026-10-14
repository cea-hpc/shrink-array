# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Résumé tab.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QTableWidget, QVBoxLayout, QWidget

from shrink_gui.i18n import T
from shrink_gui.widgets import NumItem, TLabel


class SummaryTab:
    def build_summary_tab(self):
        """The Résumé tab's widgets."""
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
        return sumw

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
