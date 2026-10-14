# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Graphique tab of bench_and_test.py: bench_summary.csv as one line per
variant (chunk, chunk2, chunk2a) against threads, for one config; "Comparer
avec" overlays another run of the same architecture, dashed, and gives the
ratio B / A over their common points.

A mixin of the main window (app.Main): its methods use the window's
attributes. The data come from remote_helper bench_points."""
import html
import math

from PySide6.QtCharts import QChart, QLegend, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from shrink_gui.config import VIZ
from shrink_gui.i18n import T, tip
from shrink_gui.widgets import ScatterView, TLabel

# Colour follows the variant, never its rank (config.COLOR_BY "Variante").
VARIANTS = ["chunk", "chunk2", "chunk2a"]

# What the y axis shows: (row index in bench_points, axis title, unit, lower is better).
METRICS = {
    "temps": (3, "temps moyen (s)", "s", True),
    "mémoire": (4, "mémoire pic (Mo)", "Mo", True),
    "speedup": (5, "speedup vs la variante de référence", "×", False),
}


def is_bench_chart(meta):
    """Drawn as bench_and_test.py's (meta "chart", set from PROGRAMS; the
    runs submitted before it by the program's name)."""
    return meta.get("chart") == "bench" or meta.get("program") == "bench_and_test.py"


def geomean(ratios):
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None


class BenchChart:
    def build_bench_page(self):
        """The bench_and_test.py page of the Graphique tab."""
        self.bc_data = None
        self.bc_data_b = None
        self.bc_config = QComboBox()
        self.bc_config.setMinimumContentsLength(16)
        self.bc_config.currentIndexChanged.connect(self.draw_bench)
        self.bc_metric = QComboBox()
        for k in METRICS:
            self.bc_metric.addItem(T(k), k)
        self.bc_metric.currentIndexChanged.connect(self.draw_bench)
        self.bc_cmp = QComboBox()
        self.bc_cmp.setMinimumContentsLength(24)
        self.bc_cmp.setToolTip(tip("Superpose un autre run de bench_and_test.py de la même architecture "
                               "(en pointillés) et donne le rapport B / A sur leurs points communs"))
        self.bc_cmp.currentIndexChanged.connect(self.load_bench_points)
        self.bc_log = QCheckBox("Échelle log")
        self.bc_log.toggled.connect(self.draw_bench)
        bref = QPushButton("Rafraîchir")
        bref.clicked.connect(self.load_bench_points)
        row = QHBoxLayout()
        for lab, w in (("Config", self.bc_config), ("Mesure", self.bc_metric)):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addWidget(self.bc_log)
        row.addWidget(QLabel("Comparer avec"))
        row.addWidget(self.bc_cmp, 1)
        row.addWidget(bref)
        self.bc_note = TLabel("")
        self.bc_note.setWordWrap(True)
        self.bc_view = ScatterView()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self.bc_note)
        lay.addWidget(self.bc_view, 1)
        return w

    def fill_bench_compare(self):
        """The runs the current one can be compared with: the other
        bench_and_test.py runs of its architecture, newest first."""
        if not hasattr(self, "bc_cmp"):
            return
        cur = next((x for x in self.runs if x["id"] == self.current), None)
        arch = cur["meta"].get("arch") if cur else None
        keep = self.bc_cmp.currentData()
        self.bc_cmp.blockSignals(True)
        self.bc_cmp.clear()
        self.bc_cmp.addItem(T("(aucun)"), None)
        for r in self.runs:
            if r["id"] != self.current and is_bench_chart(r["meta"]) and r["meta"].get("arch") == arch \
                    and not r["meta"].get("rerun_of"):
                self.bc_cmp.addItem(f"{r['id']}  [{T(r['status'])}]", r["id"])
        i = self.bc_cmp.findData(keep) if keep else 0
        self.bc_cmp.setCurrentIndex(max(0, i))
        self.bc_cmp.blockSignals(False)

    def load_bench_points(self, *_):
        rid = self.current
        if not rid:
            return
        rid_b = self.bc_cmp.currentData()
        pending = {"a": None, "b": None}
        need = {"a", "b"} if rid_b else {"a"}

        def finish():
            if rid != self.current or rid_b != self.bc_cmp.currentData():
                return
            if any(pending[k] is None for k in need):
                return
            r = pending["a"]
            if "rows" not in r:
                self.bc_data = None
                self.bc_note.setText(T(r.get("note", "")))
                self.bc_view.setChart(QChart())
                self.bc_view.points = []
                return
            rb = pending["b"] if rid_b and "rows" in pending["b"] else None
            if rid_b and rb is None:
                self.error(f"{rid_b} : {pending['b'].get('note', 'pas de données')}")
            r["id"] = rid
            if rb:
                rb["id"] = rid_b
            self.bc_data, self.bc_data_b = r, rb
            configs = list(dict.fromkeys(r["configs"] + (rb["configs"] if rb else [])))
            keep = self.bc_config.currentData()
            self.bc_config.blockSignals(True)
            self.bc_config.clear()
            for c in configs:
                self.bc_config.addItem(c, c)
            if keep in configs:
                self.bc_config.setCurrentIndex(self.bc_config.findData(keep))
            self.bc_config.blockSignals(False)
            self.draw_bench()

        def got(key):
            def done(r):
                pending[key] = r if r is not None else {"note": "erreur de lecture"}
                finish()
            return done

        self.remote.helper("points", [self.runs_dir.text().strip(), rid], got("a"))
        if rid_b:
            self.remote.helper("points", [self.runs_dir.text().strip(), rid_b], got("b"))

    @staticmethod
    def bench_values(d, col, config=None):
        """{(config, variant, threads): value} of one dataset, positive values only."""
        out = {}
        for row in d["rows"]:
            v = row[col]
            if v is not None and v > 0 and (config is None or row[0] == config):
                out[(row[0], row[1], row[2])] = v
        return out

    def bench_ratio_text(self, a, b, metric, config):
        """B / A over the common points: everything, then the config shown."""
        col, _title, _unit, lower_better = METRICS[metric]
        va, vb = self.bench_values(a, col), self.bench_values(b, col)
        common = [k for k in va if k in vb]
        if not common:
            return "  —  aucun point commun aux deux runs"

        def word(g):
            pct = abs(g - 1) * 100
            if metric == "temps":
                sens = "plus lent" if g > 1 else "plus rapide"
            elif metric == "mémoire":
                sens = "de mémoire en plus" if g > 1 else "de mémoire en moins"
            else:
                sens = "de speedup en plus" if g > 1 else "de speedup en moins"
            return f"×{g:.3f} ({pct:.1f} % {sens})"

        g_all = geomean([vb[k] / va[k] for k in common])
        here = [k for k in common if k[0] == config]
        txt = (f"  —  B / A : {word(g_all)}, moyenne géométrique sur "
               f"{len(common)} point(s) communs (toutes configs)")
        if here and len(here) != len(common):
            txt += f" ; {word(geomean([vb[k] / va[k] for k in here]))} sur {config}"
        worse = sorted(common, key=lambda k: vb[k] / va[k], reverse=lower_better)[0]
        r = vb[worse] / va[worse]
        if (r > 1.05) if lower_better else (r < 0.95):
            txt += f" ; le pire : ×{r:.3f} ({worse[0]} · {worse[1]} · {worse[2]} threads)"
        return txt + self.commits_note(a["id"], b["id"])

    def draw_bench(self, *_):
        r = self.bc_data
        if not r:
            return
        rb = self.bc_data_b
        metric = self.bc_metric.currentData()
        col, title, unit, _lower = METRICS[metric]
        config = self.bc_config.currentData()
        log_y = self.bc_log.isChecked()
        dark = self.palette().window().color().lightness() < 128
        V = VIZ["dark" if dark else "light"]

        chart = QChart()
        chart.setBackgroundBrush(QBrush(QColor(V["surface"])))
        chart.setBackgroundRoundness(0)
        chart.setTitle(("Comparaison — " if rb else "") + title + " selon le nombre de threads — " + (config or ""))
        chart.setTitleBrush(QBrush(QColor(V["ink"])))
        chart.legend().setLabelColor(QColor(V["ink2"]))
        chart.legend().setAlignment(Qt.AlignTop)
        chart.legend().setMarkerShape(QLegend.MarkerShapeFromSeries)

        drawn = []
        va = self.bench_values(r, col, config)
        vb = self.bench_values(rb, col, config) if rb else {}
        for dset, vals, dashed in ((r, va, False), (rb, vb, True)):
            if not dset:
                continue
            variants = list(dict.fromkeys(k[1] for k in vals))
            for var in sorted(variants, key=lambda v: (v not in VARIANTS, VARIANTS.index(v)
                                                       if v in VARIANTS else 0, v)):
                color = QColor(V["series"][VARIANTS.index(var)] if var in VARIANTS else V["muted"])
                pts = sorted((k[2], v) for k, v in vals.items() if k[1] == var)
                if not pts:
                    continue
                name = var + ((" · B" if dashed else " · A") if rb else "")
                ln = QLineSeries()
                ln.setName(name)
                pen = QPen(color, 2)
                if dashed:
                    pen.setStyle(Qt.DashLine)
                ln.setPen(pen)
                ln.append([QPointF(x, y) for x, y in pts])
                chart.addSeries(ln)
                dots = QScatterSeries()
                dots.setMarkerShape(QScatterSeries.MarkerShapeTriangle if dashed
                                    else QScatterSeries.MarkerShapeCircle)
                dots.setMarkerSize(9)
                dots.setColor(color)
                dots.setBorderColor(QColor(V["surface"]))
                dots.setPen(QPen(QColor(V["surface"]), 1.5))   # a ring: overlapping marks stay apart
                dots.append([QPointF(x, y) for x, y in pts])
                chart.addSeries(dots)
                for m in chart.legend().markers(dots):
                    m.setVisible(False)
                other = va if dashed else vb
                for th, v in pts:
                    k = (config, var, th)
                    cmp_ = (f"<br>×{v / other[k]:.3f} vs {'A' if dashed else 'B'}" if k in other else "")
                    tipt = (f"<b>{v:.4g} {unit}</b> · {th} threads<br>{html.escape(config or '')} · "
                            f"{html.escape(var)}" + (f"<br><i>{'B' if dashed else 'A'} : "
                                                     f"{html.escape(dset['id'])}</i>" if rb else "")
                            + cmp_)
                    drawn.append((th, v, tipt))

        ax = QLogValueAxis()
        ax.setBase(2.0)
        ax.setLabelFormat("%d")
        ax.setTitleText("threads")
        ay = QLogValueAxis() if log_y else QValueAxis()
        ay.setTitleText(title)
        ay.setLabelFormat("%.3g" if log_y else "%.2f")
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
            ax.setRange(min(xs) / 1.2, max(xs) * 1.2)
            if log_y:
                lo, hi = min(ys) / 1.15, max(ys) * 1.15
                ay.setBase(10.0 if hi / lo >= 20 else 2.0)
                ay.setRange(lo, hi)
            else:
                ay.setRange(0, max(ys) * 1.08)
                ay.applyNiceNumbers()
        chart.setTitle(T(chart.title()))
        for a_ in chart.axes():
            a_.setTitleText(T(a_.titleText()))
        self.bc_view.setChart(chart)
        self.bc_view.points = drawn
        self.bc_view.log_x = True
        self.bc_view.log_y = log_y

        head = f"{len(drawn)} point(s)"
        notes = []
        if rb:
            notes.append(f"A : {r['id']} (traits pleins, ronds) — B : {rb['id']} (pointillés, triangles)")
            if not r.get("own") and not rb.get("own") and r.get("mtime") == rb.get("mtime"):
                notes.append("les deux runs lisent le même bench_summary.csv partagé (lancés avant que "
                             "chaque run en garde une copie) : la comparaison n'a pas de sens")
            else:
                head += self.bench_ratio_text(r, rb, metric, config)
        notes.insert(0, head)
        for dset in (r, rb):
            if dset and not dset.get("own"):
                notes.append(f"{dset['id']} : bench_summary.csv partagé de l'architecture, peut-être "
                             "réécrit par un run plus récent")
        if metric == "speedup":
            notes.append("speedup : la variante de référence n'a pas de point (1 par définition)")
        self.bc_note.setText("\n".join(T(n) for n in notes))
