"""Widgets: the log, the options form, the chart, table cells."""
import html
import re
import shlex

from PySide6.QtCharts import QChartView
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QTextCursor, QValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidgetItem, QToolTip,
                               QVBoxLayout, QWidget)

from shrink_gui import i18n, qos
from shrink_gui.i18n import T, tip
from shrink_gui.runscripts import shq

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

BOOL_KINDS = {"_StoreTrueAction", "_StoreFalseAction", "_StoreConstAction",
              "_AppendConstAction", "BooleanOptionalAction"}

def ask(parent, title, text):
    """QMessageBox.question in the interface language."""
    return QMessageBox.question(parent, T(title), T(text))

DUR_RE = re.compile(r"(\d+)\s*(j|d|h|min|mn|m|s)?", re.I)
DUR_UNIT = {"j": 86400, "d": 86400, "h": 3600, "min": 60, "mn": 60, "m": 60, "s": 1}

def parse_duration(text):
    """"1 h 30", "90 min", "2j", "2 d 12 h", "1-00:00:00", "5400" -> seconds
    (None if it is none of these). A bare number after hours counts as
    minutes ("1 h 30"), after days as hours ("2 j 12"); alone, as seconds."""
    text = text.strip()
    if not text:
        return None
    if ":" in text:
        return qos.parse_time(text)
    pos, total, prev = 0, 0, None
    for m in DUR_RE.finditer(text):
        if text[pos:m.start()].strip():
            return None
        unit = (m.group(2) or "").lower()
        if not unit:
            unit = {"h": "min", "j": "h", "d": "h"}.get(prev, "s")
        total += int(m.group(1)) * DUR_UNIT[unit]
        prev, pos = unit, m.end()
    return total if pos and not text[pos:].strip() else None

class DurationBox(QSpinBox):
    """A duration in seconds, shown and typed as "1 h 30", "2 j 00 h",
    "90 min" or "1-00:00:00"; the arrows step by half an hour."""

    def __init__(self):
        super().__init__()
        self.setRange(60, 365 * 86400)
        self.setSingleStep(1800)
        self.setAccelerated(True)

    def textFromValue(self, v):
        return qos.fmt(v, i18n.LANG)

    def valueFromText(self, text):
        v = parse_duration(text)
        return self.value() if v is None else v

    def validate(self, text, pos):
        v = parse_duration(text)
        ok = v is not None and self.minimum() <= v <= self.maximum()
        return (QValidator.Acceptable if ok else QValidator.Intermediate), text, pos

class LogView(QPlainTextEdit):
    """Appends ssh output and honours \\r (a progress line rewritten in place)."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Monospace", 9))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setMaximumBlockCount(20000)
        self.follow = True

    @staticmethod
    def _collapse(line):
        """A line holding \\r (a progress bar rewritten in place): what a
        terminal ends up showing, the last non-empty rewrite."""
        parts = [p for p in line.split("\r") if p]
        return parts[-1] if parts else ""

    def append_raw(self, text):
        """Appends @p text in ONE insertion: \\r rewrites are resolved here,
        in Python, and only the last maximumBlockCount() lines are kept (the
        view would drop the others anyway). Inserting line by line cost one
        Qt call per line -- minutes for a long log."""
        text = ANSI_RE.sub("", text).replace("\r\n", "\n")
        if not text:
            return
        bar = self.verticalScrollBar()
        at_end = self.follow
        cur = self.textCursor()
        cur.beginEditBlock()
        cur.movePosition(QTextCursor.End)
        if "\r" in text:
            lines = text.split("\n")
            if "\r" in lines[0]:        # rewrites the line in progress
                cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
                lines[0] = self._collapse(cur.selectedText() + lines[0])
                cur.removeSelectedText()
            lines = lines[:1] + [self._collapse(ln) if "\r" in ln else ln for ln in lines[1:]]
            text = "\n".join(lines)
        keep = self.maximumBlockCount()
        if keep > 0 and text.count("\n") > keep:
            cut = len(text)
            for _ in range(keep):
                cut = text.rfind("\n", 0, cut)
                if cut < 0:
                    break
            if cut > 0:
                text = text[cut + 1:]
        cur.insertText(text)
        cur.endEditBlock()
        if at_end:
            bar.setValue(bar.maximum())

class TLabel(QLabel):
    """A QLabel whose setText() translates (status lines set at run time)."""

    def setText(self, text):
        super().setText(T(text))

class OptionsForm(QWidget):
    """The form of one script's argparse options. An empty field / a flag in
    its default state is not put on the command line: the script's own
    default applies."""

    changed = None  # callback, set by the owner
    pickers = {}    # dest -> callback(dest): a "Choisir…" button beside that field

    def __init__(self):
        super().__init__()
        self.lay = QFormLayout(self)
        self.lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.widgets = {}   # dest -> (action, widget, label)
        self.schema = None

    def build(self, schema, error=None):
        while self.lay.rowCount():
            self.lay.removeRow(0)
        self.widgets.clear()
        self.schema = schema
        if not schema:
            lab = TLabel()
            lab.setText(f"<b>Options non lues.</b><br>{html.escape(error)}<br><br>"
                         "Corrigez puis cliquez sur « Lire les options ». En attendant, le champ "
                         "« arguments en plus » passe les options telles quelles."
                         if error else "Options non lues : « Lire les options » (connexion requise).")
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if error:
                lab.setStyleSheet("color: #c0392b;")
            self.lay.addRow(lab)
            return
        for ac in schema["actions"]:
            flag = (max(ac["flags"], key=len) if ac["flags"] else ac["dest"])
            help_txt = ((ac["help"] or "(pas d'aide dans le script)") + f"\n\noption : {flag}"
                        + (f"\ndéfaut du script : {ac['default']}" if ac["default"] not in (None, False) else ""))
            if ac["kind"] in BOOL_KINDS:
                w = QCheckBox()
                w.setChecked(self._bool_default(ac))
                w.toggled.connect(self._emit)
            elif ac["choices"]:
                w = QComboBox()
                w.addItem("")
                w.addItems(ac["choices"])
                w.currentTextChanged.connect(self._emit)
            else:
                w = QLineEdit()
                w.setPlaceholderText(T(f"défaut : {ac['default']}") if ac["default"] is not None else
                                     (ac["metavar"] or ""))
                w.textChanged.connect(self._emit)
            w.setToolTip(tip(help_txt))
            lab = QLabel(flag)
            lab.setToolTip(tip(help_txt))
            field = w
            if ac["dest"] in self.pickers and isinstance(w, QLineEdit):
                field = QWidget()
                row = QHBoxLayout(field)
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(w, 1)
                b = QPushButton(T("Choisir…"))
                b.setObjectName(f"pick_{ac['dest']}")
                b.setToolTip(tip("Choisir dans la liste de ce que compile_all.sh sait construire "
                                 "(lue sur inti, sans rien compiler)."))
                b.clicked.connect(lambda _=False, d=ac["dest"]: self.pickers[d](d))
                row.addWidget(b)
            self.lay.addRow(lab, field)
            self.widgets[ac["dest"]] = (ac, w, lab)

    def filter(self, text):
        """Show only the options whose flag or help contains @p text."""
        t = text.strip().lower()
        for ac, w, lab in self.widgets.values():
            hay = (" ".join(ac["flags"]) + " " + ac["dest"] + " " + ac["help"]).lower()
            self.lay.setRowVisible(lab, not t or t in hay)

    def _mark(self):
        """The options that differ from the script default, in bold."""
        vals = self.values()
        for dest, (_ac, _w, lab) in self.widgets.items():
            f = lab.font()
            if f.bold() != (dest in vals):
                f.setBold(dest in vals)
                lab.setFont(f)

    @staticmethod
    def _bool_default(ac):
        if ac["kind"] == "_StoreFalseAction":
            return True          # checked = the option's default (True)
        if ac["kind"] == "BooleanOptionalAction":
            return bool(ac["default"])
        return False

    def _emit(self, *_):
        self._mark()
        if self.changed:
            self.changed()

    def set_values(self, values):
        """Reset every field to the script default, then apply @p values."""
        for dest, (ac, w, _lab) in self.widgets.items():
            w.blockSignals(True)
            v = values.get(dest)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v) if v is not None else self._bool_default(ac))
            elif isinstance(w, QComboBox):
                w.setCurrentText("" if v is None else str(v))
            else:
                w.setText("" if v is None or v is True or v is False else str(v))
            w.blockSignals(False)
        self._emit()

    def values(self):
        """{dest: value} of the fields that differ from the script default."""
        out = {}
        for dest, (ac, w, _lab) in self.widgets.items():
            if isinstance(w, QCheckBox):
                if w.isChecked() != self._bool_default(ac):
                    out[dest] = w.isChecked()
            else:
                t = w.currentText() if isinstance(w, QComboBox) else w.text()
                if t.strip():
                    out[dest] = t.strip()
        return out

    def args(self):
        """The command-line arguments of values()."""
        opts, pos = [], []
        vals = self.values()
        for dest, (ac, _w, _lab) in self.widgets.items():
            if dest not in vals:
                continue
            v = vals[dest]
            if not ac["flags"]:
                pos += [shq(x) for x in shlex.split(v)]
                continue
            long_flag = max(ac["flags"], key=len)
            if ac["kind"] == "BooleanOptionalAction":
                no = [f for f in ac["flags"] if f.startswith("--no-")]
                yes = [f for f in ac["flags"] if not f.startswith("--no-")]
                opts.append(yes[0] if v else (no[0] if no else long_flag))
            elif ac["kind"] in BOOL_KINDS:
                opts.append(long_flag)
            elif ac["kind"] == "_CountAction":
                opts += [long_flag] * max(0, int(v) if v.isdigit() else 0)
            elif ac.get("eq"):            # shell script: --flag=value
                opts.append(f"{long_flag}={shq(v)}")
            elif ac["nargs"] in ("+", "*") or isinstance(ac["nargs"], int):
                opts.append(long_flag)
                opts += [shq(x) for x in shlex.split(v)]
            else:
                opts += [long_flag, shq(v)]
        return opts + pos

def split_list(value):
    """"a, b,,c" -> ["a", "b", "c"]."""
    return [x.strip() for x in (value or "").split(",") if x.strip()]

def families(names, sep):
    """Globs naming at least two of @p names, from their last and first
    @p sep-separated component: shrink_bench_chunk_64, shrink_bench_chunk2_64
    -> "*_64"; omp-llvm, omp-llvm-debug -> "omp-llvm*". {glob: count}."""
    out = {}
    for n in names:
        if sep in n:
            head, tail = n.rsplit(sep, 1)
            out.setdefault(f"*{sep}{tail}", set()).add(n)
            out.setdefault(f"{head}{sep}*", set()).add(n)
        out.setdefault(f"{n}*", set()).add(n)
    import fnmatch
    res = {}
    for g in out:
        hits = [n for n in names if fnmatch.fnmatchcase(n, g)]
        if 2 <= len(hits) < len(names):
            res[g] = len(hits)
    return dict(sorted(res.items(), key=lambda kv: (-kv[1], kv[0])))

def unique_globs(pairs, names):
    """[(glob, count)] without the globs that select the same names as one
    kept before them ("*-sanitizer" after "*sanitizer*")."""
    import fnmatch
    seen, out = set(), []
    for g, c in pairs:
        key = frozenset(n for n in names if fnmatch.fnmatchcase(n, g))
        if key not in seen:
            seen.add(key)
            out.append((g, c))
    return out

class PickerDialog(QDialog):
    """Pick names (and family globs) for a comma-separated option value:
    --target / --exec of compile_all.sh. @p items: [(name, note)], @p
    patterns: [(glob, note)]. @p multi False (the script takes one glob):
    checking one unchecks the others. What the value holds that is in
    neither list stays in "en plus", untouched."""

    def __init__(self, parent, title, intro, items, patterns, value, multi=True):
        super().__init__(parent)
        self.setWindowTitle(T(title))
        self.multi = multi
        self.resize(560, 620)
        lay = QVBoxLayout(self)
        lab = QLabel(T(intro))
        lab.setTextFormat(Qt.PlainText)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        self.filt = QLineEdit()
        self.filt.setPlaceholderText(T("filtrer…"))
        self.filt.setClearButtonEnabled(True)
        self.filt.textChanged.connect(self._filter)
        lay.addWidget(self.filt)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        chosen = split_list(value)
        known = {n for n, _ in items} | {g for g, _ in patterns}
        italic = QFont()
        italic.setItalic(True)
        for name, note, is_glob in ([(g, n, True) for g, n in patterns] + [(n, t, False) for n, t in items]):
            it = QListWidgetItem(f"{name}    — {T(note)}" if note else name)
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
            if is_glob:
                it.setFont(italic)
            self.list.addItem(it)
        self.list.itemChanged.connect(self._changed)
        row = QHBoxLayout()
        row.addWidget(QLabel(T("En plus :")))
        self.extra = QLineEdit(", ".join(c for c in chosen if c not in known))
        self.extra.setPlaceholderText(T("noms ou motifs tapés à la main, séparés par des virgules")
                                      if multi else T("un nom ou un motif tapé à la main"))
        row.addWidget(self.extra, 1)
        lay.addLayout(row)
        if not multi:
            note = QLabel(T("Ce compile_all.sh n'accepte qu'un motif à la fois : appliquez compile_lists.py "
                            "pour en cocher plusieurs."))
            note.setWordWrap(True)
            note.setStyleSheet("color: #8a6d00;")
            lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        bb.button(QDialogButtonBox.Reset).setText(T("Tout décocher"))
        bb.button(QDialogButtonBox.Reset).clicked.connect(self._clear)
        bb.button(QDialogButtonBox.Ok).setText(T("OK"))
        bb.button(QDialogButtonBox.Cancel).setText(T("Annuler"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _filter(self, text):
        t = text.strip().lower()
        for it in self._items():
            it.setHidden(bool(t) and t not in it.text().lower())

    def _changed(self, item):
        if not self.multi and item.checkState() == Qt.Checked:
            self.list.blockSignals(True)
            for it in self._items():
                if it is not item:
                    it.setCheckState(Qt.Unchecked)
            self.list.blockSignals(False)
            self.extra.clear()

    def _clear(self):
        self.list.blockSignals(True)
        for it in self._items():
            it.setCheckState(Qt.Unchecked)
        self.list.blockSignals(False)
        self.extra.clear()

    def value(self):
        picked = [it.data(Qt.UserRole) for it in self._items() if it.checkState() == Qt.Checked]
        picked += [x for x in split_list(self.extra.text()) if x not in picked]
        return ",".join(picked if self.multi else picked[:1])

class ScatterView(QChartView):
    """Seconds per run against threads. The nearest point within 24 px gets
    the tooltip -- a 7 px dot is too small to aim at."""

    HIT_PX = 24

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.points = []      # (threads, seconds, tooltip html) of what is drawn
        self.log_y = False
        self._last = None

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        pos = ev.position()
        if self._last and abs(pos.x() - self._last.x()) + abs(pos.y() - self._last.y()) < 2:
            return
        self._last = pos
        chart = self.chart()
        if not self.points or not chart.axes():
            return
        area = chart.plotArea()
        ax = chart.axes(Qt.Horizontal)[0]
        ay = chart.axes(Qt.Vertical)[0]
        x0, x1, y0, y1 = ax.min(), ax.max(), ay.min(), ay.max()
        if x1 <= x0 or y1 <= y0:
            return
        sx = area.width() / (x1 - x0)
        if self.log_y:
            import math
            ly0, ly1 = math.log10(y0), math.log10(y1)
            ty = lambda y: area.bottom() - (math.log10(y) - ly0) / (ly1 - ly0) * area.height()
        else:
            sy = area.height() / (y1 - y0)
            ty = lambda y: area.bottom() - (y - y0) * sy
        sp = self.mapToScene(pos.toPoint())
        cp = chart.mapFromScene(sp)
        best, bd = None, self.HIT_PX ** 2
        for p in self.points:
            if p[1] <= 0 and self.log_y:
                continue
            dx = area.left() + (p[0] - x0) * sx - cp.x()
            if dx * dx > bd:
                continue
            dy = ty(p[1]) - cp.y()
            d = dx * dx + dy * dy
            if d < bd:
                best, bd = p, d
        if best:
            QToolTip.showText(ev.globalPosition().toPoint(), best[2], self)
        else:
            QToolTip.hideText()

class NumItem(QTableWidgetItem):
    """A cell shown as its text ("2.00", "40–200") but sorted by its number."""

    def __init__(self, text):
        super().__init__(text)
        m = re.match(r"^-?\d+(?:\.\d+)?", text)
        self.num = float(m.group(0)) if m else None

    def __lt__(self, other):
        if isinstance(other, NumItem) and self.num is not None and other.num is not None:
            return self.num < other.num
        return super().__lt__(other)

