# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Graphique tab of omp_speedup.py: its summary.csv as one line per
executable against threads -- speedup (with the ideal line), efficiency or
time. Colour = the variant (vector, zero, zero_one_value, zero_any), marker =
the size (dyn, 1024, 256, 64): with --all-exec a run has a dozen curves, and
colour alone cannot tell them apart. "Comparer avec" overlays another run of the same
architecture, dashed (another compiler, another commit), and gives the ratio
B / A over the common points.

A mixin of the main window (app.Main): its methods use the window's
attributes. The data come from remote_helper speedup_points."""
import html
import math
import re

from PySide6.QtCharts import QChart, QLegend, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from shrink_gui.config import VIZ
from shrink_gui.remote_helper import normalize_variant
from shrink_gui.i18n import T, tip
from shrink_gui.widgets import ScatterView, TLabel

# Colour follows the variant, never its rank; vector is the reference, in ink.
VARIANT_SLOT = {"zero": 0, "zero_one_value": 1, "zero_any": 2}
SIZE_SHAPES = ["dyn", "1024", "256", "64"]
# The old names too (shrink_bench_chunk2a_dyn), read as the new ones; longest first.
EXE_RE = re.compile(r"^shrink_bench_(vector|std_alloc|zero_any|zero_one_value|zero|chunk2a|chunk2|chunk)"
                    r"(?:_int64)?(?:_(\w+))?$")

# What the y axis shows: (row index in speedup_points, axis title, log scale by default).
METRICS = {
    "speedup": (3, "speedup vs le premier nombre de threads", True),
    "efficacité": (4, "efficacité (speedup × min / threads)", False),
    "temps": (2, "meilleur temps (s)", False),
}


def is_speedup_chart(meta):
    """Drawn as omp_speedup.py's (meta "chart", set from PROGRAMS)."""
    return meta.get("chart") == "speedup" or meta.get("program") == "omp_speedup.py"


def exe_key(exe):
    """shrink_bench_zero_any_int64_dyn -> ("zero_any", "dyn") (shrink_bench_chunk2a_int64_dyn
    too); vector_perf -> ("vector", "")."""
    m = EXE_RE.match(exe)
    if not m:
        return exe, ""
    size = m.group(2) or ""
    return normalize_variant(m.group(1)), "" if size in ("perf", "check") else size


def _shape(size):
    names = ["MarkerShapeCircle", "MarkerShapeRectangle", "MarkerShapeTriangle", "MarkerShapeRotatedRectangle"]
    name = names[SIZE_SHAPES.index(size)] if size in SIZE_SHAPES else "MarkerShapeStar"
    return getattr(QScatterSeries, name, QScatterSeries.MarkerShapeCircle)   # older Qt: circles


class SpeedupChart:
    def build_speedup_page(self):
        """The omp_speedup.py page of the Graphique tab."""
        self.sc_data = None
        self.sc_data_b = None
        self.sc_metric = QComboBox()
        for k in METRICS:
            self.sc_metric.addItem(T(k), k)
        self.sc_metric.currentIndexChanged.connect(self.draw_speedup)
        self.sc_variant = QComboBox()
        for v in ("toutes", "vector", "zero", "zero_one_value", "zero_any"):
            self.sc_variant.addItem(T(v), v)
        self.sc_variant.currentIndexChanged.connect(self.draw_speedup)
        self.sc_size = QComboBox()
        for v in ["toutes"] + SIZE_SHAPES:
            self.sc_size.addItem(T(v), v)
        self.sc_size.currentIndexChanged.connect(self.draw_speedup)
        self.sc_cmp = QComboBox()
        self.sc_cmp.setMinimumContentsLength(24)
        self.sc_cmp.setToolTip(tip("Superpose un autre run de omp_speedup.py de la même architecture "
                                   "(en pointillés) -- un autre compilateur, un autre commit -- et donne le "
                                   "rapport B / A sur leurs points communs"))
        self.sc_cmp.currentIndexChanged.connect(self.load_speedup_points)
        self.sc_log = QCheckBox("Échelle log")
        self.sc_log.setChecked(True)
        self.sc_log.toggled.connect(self.draw_speedup)
        bref = QPushButton("Rafraîchir")
        bref.clicked.connect(self.load_speedup_points)
        row = QHBoxLayout()
        for lab, w in (("Mesure", self.sc_metric), ("Variante", self.sc_variant), ("Taille", self.sc_size)):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addWidget(self.sc_log)
        row.addWidget(QLabel("Comparer avec"))
        row.addWidget(self.sc_cmp, 1)
        row.addWidget(bref)
        self.sc_note = TLabel("")
        self.sc_note.setWordWrap(True)
        self.sc_view = ScatterView()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self.sc_note)
        lay.addWidget(self.sc_view, 1)
        return w

    def fill_speedup_compare(self):
        """The other omp_speedup.py runs of the current one's architecture."""
        if not hasattr(self, "sc_cmp"):
            return
        cur = next((x for x in self.runs if x["id"] == self.current), None)
        arch = cur["meta"].get("arch") if cur else None
        keep = self.sc_cmp.currentData()
        self.sc_cmp.blockSignals(True)
        self.sc_cmp.clear()
        self.sc_cmp.addItem(T("(aucun)"), None)
        for r in self.runs:
            if r["id"] != self.current and is_speedup_chart(r["meta"]) and r["meta"].get("arch") == arch:
                self.sc_cmp.addItem(f"{r['id']}  [{T(r['status'])}]", r["id"])
        i = self.sc_cmp.findData(keep) if keep else 0
        self.sc_cmp.setCurrentIndex(max(0, i))
        self.sc_cmp.blockSignals(False)

    def load_speedup_points(self, *_):
        rid = self.current
        if not rid:
            return
        rid_b = self.sc_cmp.currentData()
        pending = {"a": None, "b": None}
        need = {"a", "b"} if rid_b else {"a"}

        def finish():
            if rid != self.current or rid_b != self.sc_cmp.currentData():
                return
            if any(pending[k] is None for k in need):
                return
            r = pending["a"]
            if "rows" not in r:
                self.sc_data = None
                self.sc_note.setText(T(r.get("note", "")))
                self.sc_view.setChart(QChart())
                self.sc_view.points = []
                return
            rb = pending["b"] if rid_b and "rows" in pending["b"] else None
            if rid_b and rb is None:
                self.error(f"{rid_b} : {pending['b'].get('note', 'pas de données')}")
            r["id"] = rid
            if rb:
                rb["id"] = rid_b
            self.sc_data, self.sc_data_b = r, rb
            self.draw_speedup()

        def got(key):
            def done(r):
                pending[key] = r if r is not None else {"note": "erreur de lecture"}
                finish()
            return done

        self.remote.helper("points", [self.runs_dir.text().strip(), rid], got("a"))
        if rid_b:
            self.remote.helper("points", [self.runs_dir.text().strip(), rid_b], got("b"))

    def _speedup_keep(self, exe):
        var, size = exe_key(exe)
        v, s = self.sc_variant.currentData(), self.sc_size.currentData()
        return (v == "toutes" or var == v) and (s == "toutes" or size == s)

    @staticmethod
    def speedup_values(d, col):
        return {(row[0], row[1]): row[col] for row in d["rows"] if row[col] is not None and row[col] > 0}

    @staticmethod
    def _run_text(d):
        i = d.get("info") or {}
        mult = "%g" % i["multiply"] if isinstance(i.get("multiply"), (int, float)) else "?"
        return f"{d['id']} ({i.get('target') or '?'}, --multiply {mult}, {i.get('mode') or '?'})"

    def draw_speedup(self, *_):
        r = self.sc_data
        if not r:
            return
        rb = self.sc_data_b
        metric = self.sc_metric.currentData()
        col, title, _log_default = METRICS[metric]
        log_y = self.sc_log.isChecked() and metric != "efficacité"
        dark = self.palette().window().color().lightness() < 128
        V = VIZ["dark" if dark else "light"]
        info = r.get("info") or {}

        chart = QChart()
        chart.setBackgroundBrush(QBrush(QColor(V["surface"])))
        chart.setBackgroundRoundness(0)
        chart.setTitle(("Comparaison — " if rb else "") + title + " — " + (info.get("target") or ""))
        chart.setTitleBrush(QBrush(QColor(V["ink"])))
        chart.legend().setLabelColor(QColor(V["ink2"]))
        chart.legend().setAlignment(Qt.AlignTop)
        chart.legend().setMarkerShape(QLegend.MarkerShapeFromSeries)

        va = {k: v for k, v in self.speedup_values(r, col).items() if self._speedup_keep(k[0])}
        vb = {k: v for k, v in self.speedup_values(rb, col).items() if self._speedup_keep(k[0])} if rb else {}
        drawn, xs = [], sorted({k[1] for k in list(va) + list(vb)})
        tmin = min((row[1] for row in r["rows"]), default=1)

        if xs and metric in ("speedup", "efficacité"):
            ideal = QLineSeries()
            ideal.setName("idéal")
            pen = QPen(QColor(V["muted"]), 1.5)
            pen.setStyle(Qt.DotLine)
            ideal.setPen(pen)
            ideal.append([QPointF(x, x / tmin if metric == "speedup" else 1.0) for x in xs])
            chart.addSeries(ideal)

        for dset, vals, dashed in ((r, va, False), (rb, vb, True)):
            if not dset:
                continue
            execs = list(dict.fromkeys(k[0] for k in vals))
            for exe in sorted(execs, key=lambda e: (exe_key(e)[0] != "vector", exe_key(e))):
                var, size = exe_key(exe)
                color = QColor(V["ink2"] if var == "vector" else
                               V["series"][VARIANT_SLOT[var]] if var in VARIANT_SLOT else V["muted"])
                pts = sorted((k[1], v) for k, v in vals.items() if k[0] == exe)
                short = exe.replace("shrink_bench_", "")
                ln = QLineSeries()
                ln.setName(short + ((" · B" if dashed else " · A") if rb else ""))
                pen = QPen(color, 2)
                if dashed:
                    pen.setStyle(Qt.DashLine)
                ln.setPen(pen)
                ln.append([QPointF(x, y) for x, y in pts])
                chart.addSeries(ln)
                dots = QScatterSeries()
                dots.setMarkerShape(_shape(size))
                dots.setMarkerSize(9)
                dots.setColor(QColor(V["surface"]) if dashed else color)    # B: hollow markers
                dots.setBorderColor(color)
                dots.setPen(QPen(color, 1.5))
                dots.append([QPointF(x, y) for x, y in pts])
                chart.addSeries(dots)
                for m in chart.legend().markers(dots):
                    m.setVisible(False)
                other = va if dashed else vb
                for th, v in pts:
                    k = (exe, th)
                    cmp_ = f"<br>×{v / other[k]:.3f} vs {'A' if dashed else 'B'}" if k in other else ""
                    tipt = (f"<b>{v:.4g}</b> · {th} threads<br>{html.escape(short)}"
                            + (f"<br><i>{'B' if dashed else 'A'} : {html.escape(dset['id'])}</i>" if rb else "")
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
            ay.setBase(2.0)
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
            ys = [p[1] for p in drawn] + ([x / tmin for x in xs] if metric == "speedup" else [])
            ax.setRange(min(xs) / 1.2, max(xs) * 1.2)
            if log_y:
                ay.setRange(min(ys) / 1.2, max(ys) * 1.2)
            elif metric == "efficacité":
                ay.setRange(0, max(1.1, max(ys) * 1.05))
            else:
                ay.setRange(0, max(ys) * 1.08)
                ay.applyNiceNumbers()
        chart.setTitle(T(chart.title()))
        for a_ in chart.axes():
            a_.setTitleText(T(a_.titleText()))
        for sr in chart.series():
            if sr.name() == "idéal":
                sr.setName(T("idéal"))
        self.sc_view.setChart(chart)
        self.sc_view.points = drawn
        self.sc_view.log_x = True
        self.sc_view.log_y = log_y

        head = f"{len(drawn)} point(s)"
        notes = []
        if rb:
            notes.append(f"A : {self._run_text(r)}, traits pleins — B : {self._run_text(rb)}, pointillés")
            ib = rb.get("info") or {}
            if (info.get("multiply"), info.get("mode")) != (ib.get("multiply"), ib.get("mode")):
                notes.append("tailles ou modes différents : comparer les speedups, pas les temps")
            common = [k for k in va if k in vb]
            if common:
                g = math.exp(sum(math.log(vb[k] / va[k]) for k in common) / len(common))
                head += f"  —  B / A : ×{g:.3f}, moyenne géométrique sur {len(common)} point(s) communs"
            else:
                head += "  —  aucun point commun aux deux runs"
        notes.insert(0, head)
        notes.append("couleur : la variante (vector en gris) · forme : la taille (rond dyn, carré 1024, "
                     "triangle 256, losange 64)")
        self.sc_note.setText("\n".join(T(n) for n in notes))
