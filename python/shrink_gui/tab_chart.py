# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Graphique tab: multi_launch_stress.py's seconds per run against threads.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import html

from PySide6.QtCharts import QChart, QLegend, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from shrink_gui.config import COLOR_BY, VIZ, exe_parts
from shrink_gui.i18n import T
from shrink_gui.tab_bench_chart import is_bench_chart
from shrink_gui.tab_speedup_chart import is_speedup_chart
from shrink_gui.widgets import ScatterView, TLabel


class ChartTab:
    def build_chart_tab(self):
        """The Graphique tab's widgets."""
        # chart tab: seconds per run against threads (multi_launch_stress.py)
        self.pt_data = None
        self.pt_data_b = None
        self.pt_build = QComboBox()
        self.pt_build.currentIndexChanged.connect(self.draw_points)
        self.pt_variant = QComboBox()
        for v in ("toutes", "zero", "zero_one_value", "zero_any"):
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
        # page 0: multi_launch_stress.py, page 1: bench_and_test.py (tab_bench_chart),
        # page 2: omp_speedup.py (tab_speedup_chart)
        self.pt_stack = QStackedWidget()
        self.pt_stack.addWidget(ptw)
        self.pt_stack.addWidget(self.build_bench_page())
        self.pt_stack.addWidget(self.build_speedup_page())      # page 2: omp_speedup.py
        return self.pt_stack

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
        self.fill_bench_compare()
        self.fill_speedup_compare()

    def load_points(self, *_):
        rid = self.current
        if not rid:
            return
        run = next((x for x in self.runs if x["id"] == rid), None)
        if run and is_speedup_chart(run["meta"]):
            self.pt_stack.setCurrentIndex(2)
            self.load_speedup_points()
            return
        if run and is_bench_chart(run["meta"]):
            self.pt_stack.setCurrentIndex(1)
            self.load_bench_points()
            return
        self.pt_stack.setCurrentIndex(0)
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
