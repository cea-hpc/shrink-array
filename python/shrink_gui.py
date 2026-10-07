#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""shrink_gui.py -- prototype: submit and follow the shrink-array runs on inti
from the laptop.

Everything goes through `ssh <host>`, so through the ControlMaster you opened
yourself (`ssh -fNM inti`): this program never handles a password. Nothing is
installed on inti: each call sends a small Python helper (REMOTE_HELPER below)
on the ssh stdin and runs it with the login node's python3.

Programs (PROGRAMS): multi_launch_stress.py, bench_and_test.py and
scripts/compile_all.sh. Their options are READ FROM THE SCRIPTS THEMSELVES on
inti ("Lire les options"): a Python script is run with argparse's
parse_known_args() replaced by a dump of the parser; a shell script's --help is
parsed ("  --flag[=<v>]   description" lines). The form follows the script
when an option is added; the options you changed show in bold. Each
(program, architecture) has its own default values: built in (ARCH_DEFAULTS),
replaced by yours with "Enregistrer comme défauts".

Submission: ccc_msub (batch, the default) or ccc_mprun detached on the login
node (the first prototype's way). A run is a directory on inti
(<runs dir>/<id>/):
    meta.json   what was asked (program, partition, command, mode, ...)
    run.sh      what runs on the compute node: repo, site_env.sh, command
    job.sh      the batch script given to ccc_msub (#MSUB directives)
    launch.sh   (ccc_mprun mode) ccc_mprun <options> bash run.sh
    out.log     everything the job printed
    jobid       (ccc_msub) the batch id, for squeue / ccc_mdel
    exit_code   written when the command returns
    stopped     written by "Arrêter"

"Récupérer" fetches the run directory, plus the summary files of the
program's own output directory, read from the log ("📁 Outputs, CSV and log
in ..." / "📁 Run outputs → ...") -- never the per-run output_* files.

Requires PySide6 on the laptop (pip install PySide6). Python >= 3.6 on inti.
"""
import base64
import html
import io
import json
import os
import re
import shlex
import sys
import tarfile
import time

from PySide6.QtCharts import (QChart, QChartView, QLegend, QLineSeries, QLogValueAxis,
                              QScatterSeries, QValueAxis)
from PySide6.QtCore import QPointF, QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QKeySequence, QPainter, QPen,
                           QTextCursor, QTextDocument)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QStyle, QSystemTrayIcon, QTableWidget, QTableWidgetItem,
                               QTabWidget, QToolTip, QVBoxLayout, QWidget)

APP = "shrink_gui"

# --------------------------------------------------------------------------
# Site knowledge -- edit here.
# --------------------------------------------------------------------------
# Partitions. "msub": the #MSUB lines of the batch script (-T, -r, -o and -e
# are added from the form); editable in the window and remembered per
# partition. "mprun": the ccc_mprun options of the detached mode (no -s, no -T).
PRESETS = {
    "gh200-bxi": {
        "label": "gh200-bxi — ARM, 288 cœurs",
        "arch": "arm",
        "msub": "#MSUB -q gh200-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 288\n"
                "#MSUB -E \"--constraint=ccc-rhel9\"\n",
        "mprun": "-N1 -c 288 -p gh200-bxi -C ccc-rhel9 -Xfirst",
    },
    "milan-bxi": {
        "label": "milan-bxi — x86, 64 cœurs",
        "arch": "x86",
        "msub": "#MSUB -q milan-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 64\n",
        "mprun": "-N1 -c 64 -p milan-bxi -Xfirst",
    },
}

# Programs: script and working directory relative to the repository, how the
# options are read ("argparse" or "help"), what the log says about progress,
# and what "Récupérer" / the Résumé tab take from the output directory the
# program announces in its log (out_re).
PROGRAMS = {
    "multi_launch_stress.py": {
        "script": "python/multi_launch_stress.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Outputs, CSV and log in (\S+)",
        "out_files": ["timings_summary.csv", "multi_launch_output.log", "error_*.txt"],
        "progress_re": r"Run (\d+)/(\d+) on build",
        "eta_re": r"Estimated Time Left: ([\d.]+)s",
        "summary": "multi_launch", "summary_file": "timings_summary.csv",
    },
    "bench_and_test.py": {
        "script": "python/bench_and_test.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Run outputs \S+ (\S+)",
        "out_files": ["bench_summary.csv", "bench_errors.log", "failed_cases.jsonl"],
        "progress_re": None, "eta_re": None,
        "summary": "csv", "summary_file": "bench_summary.csv",
    },
    "compile_all.sh": {
        "script": "scripts/compile_all.sh", "cwd": "scripts",
        "interp": "bash", "scan": "help",
        "out_re": None, "out_files": [],
        "progress_re": None, "eta_re": None,
        "summary": None, "summary_file": None,
    },
}

# Built-in defaults per (program, arch), by argparse dest. A string is the
# option's value (shell variables such as $USER are expanded on inti), True/
# False the state of a flag. "Enregistrer comme défauts" replaces them.
_BUILD_ROOT = "$CCCWORKDIR/shrink-array"
ARCH_DEFAULTS = {
    ("multi_launch_stress.py", "arm"): {
        "min_threads": "40", "max_threads": "200", "regen_ref": True, "all_exec": True,
        "build_root": _BUILD_ROOT, "builds": "build_arm.txt",
    },
    ("multi_launch_stress.py", "x86"): {
        "min_threads": "8", "max_threads": "64", "regen_ref": True, "all_exec": True,
        "build_root": _BUILD_ROOT, "builds": "build_x86_64.txt",
    },
    ("bench_and_test.py", "arm"): {"build_root": _BUILD_ROOT},
    ("bench_and_test.py", "x86"): {"build_root": _BUILD_ROOT},
}

DEFAULTS = {
    "host": "inti",
    "repo": "~/shrink-array",
    "runs_dir": "~/shrink_runs",
    "local_dir": os.path.join(os.path.expanduser("~"), "shrink_results"),
    "time_s": 18000,
    "preset": "gh200-bxi",
    "program": "multi_launch_stress.py",
    "mode": "msub",
}

# The ssh binary; SHRINK_GUI_SSH overrides it (tests, or a wrapper).
SSH = os.environ.get("SHRINK_GUI_SSH", "ssh")

# --------------------------------------------------------------------------
# Remote helper: sent on ssh's stdin, run as `python3 - <action> <args...>`.
# Python 3.6 compatible (the login node's system python).
# --------------------------------------------------------------------------
REMOTE_HELPER = r'''
import base64, glob, io, json, os, re, shlex, signal, socket, subprocess, sys, tarfile

def rdir(base, rid=None):
    d = os.path.expanduser(base)
    return os.path.join(d, rid) if rid else d

def read(p, default=None):
    try:
        with open(p) as f:
            return f.read()
    except (IOError, OSError):
        return default

def write(p, text):
    with open(p, "w") as f:
        f.write(text)

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def cmd_out(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT, universal_newlines=True)
    except (OSError, subprocess.CalledProcessError) as e:
        return None

def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 8192))
            tail = f.read().decode("utf-8", "replace")
    except (IOError, OSError):
        return ""
    tail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", tail)
    for ln in reversed(re.split(r"[\r\n]", tail)):
        if ln.strip():
            return ln.strip()[:300]
    return ""

def queue_states():
    """{jobid: STATE} of the user's batch jobs, or None without squeue."""
    out = cmd_out(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i %T"])
    if out is None:
        return None
    q = {}
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            q[parts[0]] = parts[1]
    return q

STATES = {"PENDING": "en attente", "CONFIGURING": "démarrage", "RUNNING": "en cours",
          "COMPLETING": "en cours", "SUSPENDED": "suspendu"}

def status(d, q):
    stopped = os.path.exists(os.path.join(d, "stopped"))
    code = (read(os.path.join(d, "exit_code")) or "").strip()
    if code:
        if stopped:
            return "arrêté", None
        return "fini", int(code) if code.lstrip("-").isdigit() else code
    jobid = (read(os.path.join(d, "jobid")) or "").strip()
    if jobid:
        if q is None:
            return "inconnu (pas de squeue)", None
        if jobid in q:
            return STATES.get(q[jobid], q[jobid].lower()), None
        if stopped:
            return "arrêté", None
        return "terminé sans code (limite de temps ? voir le log)", None
    pid = read(os.path.join(d, "pid"))
    if not pid:
        return "inconnu", None
    host = (read(os.path.join(d, "host")) or "").strip()
    if host != socket.gethostname():
        return "en cours ? (lancé depuis %s)" % host, None
    if alive(int(pid)):
        return "en cours", None
    return ("arrêté" if stopped else "interrompu"), None

def act_check(_):
    print(json.dumps({"host": socket.gethostname(), "python": sys.version.split()[0],
                      "msub": cmd_out(["which", "ccc_msub"]) is not None}))

HELP_RE = re.compile(r"^\s{1,8}(?:-\w,\s*)?(--[A-Za-z][\w-]*)(=\S+)?(?=\s|$)\s*(.*)$")

def scan_help(path, interp):
    """The options of a shell script, from its --help: lines
    "  --flag[=<value>]   description", continued by deeper-indented lines."""
    try:
        p = subprocess.Popen([interp, path, "--help"], cwd=os.path.dirname(path),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, universal_newlines=True)
        text = p.communicate(timeout=60)[0]
    except (OSError, subprocess.SubprocessError) as e:
        print(json.dumps({"error": "%s --help : %s" % (path, e)}))
        return
    acts, cur, seen = [], None, set()
    for ln in text.splitlines():
        m = HELP_RE.match(ln)
        if m and m.group(1) not in ("--help",):
            flag, val, desc = m.group(1), m.group(2), m.group(3)
            if flag in seen:
                cur = None
                continue
            seen.add(flag)
            cur = {"dest": flag.lstrip("-").replace("-", "_"), "flags": [flag],
                   "kind": "_StoreAction" if val else "_StoreTrueAction",
                   "default": None if val else False, "choices": None, "nargs": None,
                   "help": desc.strip(), "required": False, "eq": bool(val),
                   "metavar": val.lstrip("=").strip("<>") if val else None}
            acts.append(cur)
        elif cur is not None and ln.startswith("   ") and ln.strip():
            cur["help"] += " " + ln.strip()
        else:
            cur = None
    if not acts:
        print(json.dumps({"error": "aucune option reconnue dans %s --help" % os.path.basename(path)}))
        return
    first = text.strip().splitlines()[0] if text.strip() else ""
    print(json.dumps({"description": first, "actions": acts}))

def act_options(a):
    """The options of a script, without running it. argparse: parse_known_args
    is replaced by a dump of the parser, then the script is executed up to its
    parse_args() call. help: its --help is parsed (scan_help)."""
    import argparse, runpy
    path = os.path.join(os.path.expanduser(a[0]), a[1])
    if len(a) > 2 and a[2] == "help":
        return scan_help(path, a[3] if len(a) > 3 else "bash")
    real_stdout = sys.stdout

    class Done(Exception):
        pass

    def plain(v):
        return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)

    def dump(self, args=None, namespace=None):
        acts = []
        for ac in self._actions:
            if isinstance(ac, (argparse._HelpAction, argparse._VersionAction)):
                continue
            h = ac.help or ""
            try:
                h = h % dict(vars(ac), prog=self.prog)
            except Exception:
                pass
            acts.append({"dest": ac.dest, "flags": list(ac.option_strings),
                         "kind": type(ac).__name__, "default": plain(ac.default),
                         "choices": [str(c) for c in ac.choices] if ac.choices else None,
                         "nargs": plain(ac.nargs), "help": h, "required": bool(ac.required),
                         "metavar": ac.metavar if isinstance(ac.metavar, str) else None})
        sys.stdout = real_stdout
        print(json.dumps({"description": self.description or "", "actions": acts}))
        raise Done()

    argparse.ArgumentParser.parse_known_args = dump
    sys.stdout = sys.stderr      # what the script prints before parsing stays off the JSON
    os.chdir(os.path.dirname(path))
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    except Done:
        return
    sys.stdout = real_stdout
    print(json.dumps({"error": "%s ne lit pas ses options avec argparse" % a[1]}))

def act_launch(a):
    base, rid, spec = a[0], a[1], json.loads(base64.b64decode(a[2]).decode())
    d = rdir(base, rid)
    os.makedirs(d)
    write(os.path.join(d, "run.sh"), spec["run.sh"])
    write(os.path.join(d, "meta.json"), json.dumps(spec["meta"], indent=1))
    if spec["mode"] == "msub":
        log = os.path.join(d, "out.log")
        job = ("#!/bin/bash\n" + spec["msub"].rstrip("\n") + "\n"
               + "#MSUB -r %s\n#MSUB -T %d\n#MSUB -o %s\n#MSUB -e %s\n"
               % (spec["name"], spec["time_s"], log, log)
               + 'D=%s\n' % shlex.quote(d)
               + 'bash "$D/run.sh"\ncode=$?\necho "$code" > "$D/exit_code"\n'
               + 'echo "[gui] end $(date \'+%F %T\')  exit code $code"\n')
        write(os.path.join(d, "job.sh"), job)
        try:
            p = subprocess.Popen(["ccc_msub", "job.sh"], cwd=d, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            out = p.communicate()[0]
        except OSError as e:
            print(json.dumps({"error": "ccc_msub : %s" % e}))
            return
        write(os.path.join(d, "submit.log"), out)
        ids = re.findall(r"\b(\d{2,})\b", out)
        if p.returncode != 0 or not ids:
            print(json.dumps({"error": "ccc_msub (code %s) : %s" % (p.returncode, out.strip()[-1500:])}))
            return
        write(os.path.join(d, "jobid"), ids[-1])
        print(json.dumps({"run_dir": d, "jobid": ids[-1]}))
    else:
        write(os.path.join(d, "launch.sh"), spec["launch.sh"])
        out = open(os.path.join(d, "out.log"), "wb")
        p = subprocess.Popen(["bash", os.path.join(d, "launch.sh")], cwd=d, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        write(os.path.join(d, "pid"), str(p.pid))
        write(os.path.join(d, "host"), socket.gethostname())
        print(json.dumps({"run_dir": d, "pid": p.pid}))

def tail_text(path, nbytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - nbytes))
            return f.read().decode("utf-8", "replace")
    except (IOError, OSError):
        return ""

def progress_of(path, meta):
    """(done, total, eta_s) from the log, with the program's own regexes."""
    if not meta.get("progress_re"):
        return None
    t = tail_text(path)
    m = None
    for m in re.finditer(meta["progress_re"], t):
        pass
    if m is None:
        return None
    eta = None
    if meta.get("eta_re"):
        for e in re.finditer(meta["eta_re"], t):
            eta = float(e.group(1))
    return [int(m.group(1)), int(m.group(2)), eta]

def act_list(a):
    base = rdir(a[0])
    runs = []
    q = queue_states()
    if os.path.isdir(base):
        for rid in sorted(os.listdir(base), reverse=True)[:200]:
            d = os.path.join(base, rid)
            meta = read(os.path.join(d, "meta.json"))
            if meta is None:
                continue
            st, code = status(d, q)
            meta = json.loads(meta)
            runs.append({"id": rid, "meta": meta, "status": st, "code": code,
                         "jobid": (read(os.path.join(d, "jobid")) or "").strip(),
                         "progress": progress_of(os.path.join(d, "out.log"), meta),
                         "last": last_line(os.path.join(d, "out.log"))})
    print(json.dumps({"host": socket.gethostname(), "runs": runs}))

def act_tail(a):
    d, off = rdir(a[0], a[1]), int(a[2])
    p = os.path.join(d, "out.log")
    data = b""
    if os.path.exists(p):          # a pending batch job has no log yet
        size = os.path.getsize(p)
        if off > size:
            off = 0
        with open(p, "rb") as f:
            f.seek(off)
            data = f.read(512 * 1024)
    else:
        off = 0
    print(json.dumps({"offset": off + len(data), "data": data.decode("utf-8", "replace")}))

def act_stop(a):
    d = rdir(a[0], a[1])
    jobid = (read(os.path.join(d, "jobid")) or "").strip()
    if jobid:
        write(os.path.join(d, "stopped"), "")
        out = cmd_out(["ccc_mdel", jobid])
        if out is None:
            out = cmd_out(["scancel", jobid])
        if out is None:
            print(json.dumps({"error": "ni ccc_mdel ni scancel n'ont pu annuler le job %s" % jobid}))
        else:
            print(json.dumps({"ok": True}))
        return
    host = (read(os.path.join(d, "host")) or "").strip()
    if host != socket.gethostname():
        print(json.dumps({"error": "lancé depuis %s, ce master arrive sur %s" % (host, socket.gethostname())}))
        return
    try:
        os.killpg(int(read(os.path.join(d, "pid"))), signal.SIGTERM)
        print(json.dumps({"ok": True}))
    except (OSError, ValueError, TypeError) as e:
        print(json.dumps({"error": str(e)}))

def act_pack(a):
    d = rdir(a[0], a[1])
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in sorted(os.listdir(d)):
            t.add(os.path.join(d, name), arcname="run/" + name)
        m = re.search(meta.get("out_re") or r"$^", read(os.path.join(d, "out.log"), "")[:400000])
        if m:
            od = m.group(1)
            for pat in meta.get("out_files", []):
                for p in sorted(glob.glob(os.path.join(od, pat)))[:200]:
                    if os.path.isfile(p):
                        t.add(p, arcname="outputs/" + os.path.basename(p))
    sys.stdout.write(base64.b64encode(buf.getvalue()).decode())

def out_dir(d, meta):
    if not meta.get("out_re"):
        return None
    m = re.search(meta["out_re"], read(os.path.join(d, "out.log"), "")[:400000])
    return m.group(1) if m else None

def act_summary(a):
    """The Résumé tab: multi_launch_stress's timings_summary.csv aggregated per
    build, or the last rows of another program's summary CSV as they are."""
    import csv, statistics
    d = rdir(a[0], a[1])
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}"))
    kind, name = meta.get("summary"), meta.get("summary_file")
    if not kind or not name:
        print(json.dumps({"note": "pas de résumé pour ce programme"}))
        return
    od = out_dir(d, meta)
    path = os.path.join(od, name) if od else None
    if not path or not os.path.isfile(path):
        print(json.dumps({"note": "pas encore de %s" % name}))
        return
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        print(json.dumps({"note": "%s est vide" % name}))
        return
    head, body = rows[0], rows[1:]
    if kind != "multi_launch":
        print(json.dumps({"columns": head, "rows": body[-1000:],
                          "note": "%s — %d ligne(s)%s" % (name, len(body),
                                  ", les 1000 dernières" if len(body) > 1000 else "")}))
        return
    col = {h: i for i, h in enumerate(head)}
    per = {}
    for r in body:
        try:
            b, wall, rc, th = r[col["build"]], float(r[col["wall_seconds"]]), int(r[col["returncode"]]), int(r[col["num_threads"]])
        except (KeyError, ValueError, IndexError):
            continue
        g = per.setdefault(b, {"n": 0, "fail": 0, "walls": [], "th": []})
        g["n"] += 1
        g["fail"] += rc != 0
        g["walls"].append(wall)
        g["th"].append(th)
    out = []
    tot = {"n": 0, "fail": 0, "walls": [], "th": []}
    for b in sorted(per):
        g = per[b]
        for k in tot:
            tot[k] = tot[k] + g[k]
        out.append([b] + summary_row(g, statistics))
    if per:
        out.append(["TOTAL"] + summary_row(tot, statistics))
    print(json.dumps({"columns": ["build", "runs", "échecs", "moyenne s", "médiane s", "max s", "threads"],
                      "rows": out, "fail_rows": [i for i, r in enumerate(out) if r[2] != "0"],
                      "note": "%s — %d run(s)" % (name, len(body))}))

def summary_row(g, statistics):
    w = g["walls"]
    return [str(g["n"]), str(g["fail"]), "%.2f" % statistics.mean(w), "%.2f" % statistics.median(w),
            "%.2f" % max(w), "%d–%d" % (min(g["th"]), max(g["th"]))]

def act_points(a):
    """The Graphique tab: one point per run of timings_summary.csv (the last
    30000), names factored out: {"builds", "exes", "rows": [run, b, e, threads,
    seconds, returncode]}."""
    import csv
    d = rdir(a[0], a[1])
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}"))
    if meta.get("summary") != "multi_launch":
        print(json.dumps({"note": "graphique disponible pour multi_launch_stress.py seulement"}))
        return
    od = out_dir(d, meta)
    path = os.path.join(od, meta["summary_file"]) if od else None
    if not path or not os.path.isfile(path):
        print(json.dumps({"note": "pas encore de %s" % meta["summary_file"]}))
        return
    builds, exes, rows = {}, {}, []
    with open(path) as f:
        rd = csv.reader(f)
        head = next(rd, [])
        col = {h: i for i, h in enumerate(head)}
        for r in rd:
            try:
                b, e = r[col["build"]], r[col["executable"]]
                rows.append([int(r[col["run_index"]]), builds.setdefault(b, len(builds)),
                             exes.setdefault(e, len(exes)), int(r[col["num_threads"]]),
                             float(r[col["wall_seconds"]]), int(r[col["returncode"]])])
            except (KeyError, ValueError, IndexError):
                continue
    n = len(rows)
    print(json.dumps({"builds": sorted(builds, key=builds.get), "exes": sorted(exes, key=exes.get),
                      "rows": rows[-30000:], "total": n}))

def act_delete(a):
    """Remove one run directory -- never the program's own output directory."""
    import shutil
    base, rid = rdir(a[0]), a[1]
    if not re.match(r"^[\w.-]+$", rid) or rid in (".", ".."):
        print(json.dumps({"error": "nom de run invalide : %s" % rid}))
        return
    d = os.path.join(base, rid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        print(json.dumps({"error": "%s n'est pas un run" % d}))
        return
    shutil.rmtree(d)
    print(json.dumps({"ok": True}))

ACTIONS = {"check": act_check, "options": act_options, "launch": act_launch, "list": act_list,
           "tail": act_tail, "stop": act_stop, "pack": act_pack, "summary": act_summary,
           "points": act_points, "delete": act_delete}
ACTIONS[sys.argv[1]](sys.argv[2:])
'''


def shq(v):
    """Quote for bash when needed, keeping $VAR expansion (a default such as
    /ccc/work/.../$USER/... is meant to expand on inti)."""
    if re.fullmatch(r"[A-Za-z0-9_./:,+@%=-]+", str(v)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


def make_run_sh(repo, cwd, command):
    """What runs on the compute node."""
    return f"""#!/bin/bash
echo "[gui] node $(hostname)  start $(date '+%F %T')  job ${{SLURM_JOB_ID:-}}"
R={shlex.quote(repo)}
R="${{R/#\\~/$HOME}}"
cd "$R" || {{ echo "[gui] repository $R not found"; exit 97; }}
# A batch job's shell may not have Environment Modules set up (compile_all.sh
# needs `module`): take the login profile then.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
[ -f scripts/site_env.sh ] && source scripts/site_env.sh
export CHUNK_BUILD_ROOT="${{CHUNK_BUILD_ROOT:-${{CHUNK_BUILD_REP:-.}}}}"
# Python writes to a file here, not a terminal: unbuffered, or the log only
# shows up by 8 KiB blocks.
export PYTHONUNBUFFERED=1
cd {shlex.quote(cwd)} || exit 97
echo "[gui] cwd $PWD"
echo "[gui] command: "{shlex.quote(command)}
{command}
"""


def make_launch_sh(mprun_opts):
    """ccc_mprun mode: runs on the login node, detached."""
    return f"""#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
trap 'touch "$D/stopped"' TERM
echo "[gui] ccc_mprun {mprun_opts}"
ccc_mprun {mprun_opts} bash "$D/run.sh" &
m=$!
wait $m
code=$?
while kill -0 $m 2>/dev/null; do wait $m; code=$?; done
echo "$code" > "$D/exit_code"
echo "[gui] end $(date '+%F %T')  exit code $code"
"""


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BOOL_KINDS = {"_StoreTrueAction", "_StoreFalseAction", "_StoreConstAction",
              "_AppendConstAction", "BooleanOptionalAction"}


class Remote:
    """One ssh call = one QProcess (asynchronous, never blocks the window)."""

    def __init__(self, parent, host_fn, on_error):
        self.parent = parent
        self.host_fn = host_fn
        self.on_error = on_error
        self.procs = set()

    def _run(self, args, stdin_text, done, binary=False):
        p = QProcess(self.parent)
        self.procs.add(p)
        out = bytearray()
        p.readyReadStandardOutput.connect(lambda: out.extend(bytes(p.readAllStandardOutput())))

        def finished(code, _status):
            out.extend(bytes(p.readAllStandardOutput()))
            err = bytes(p.readAllStandardError()).decode("utf-8", "replace").strip()
            self.procs.discard(p)
            p.deleteLater()
            if code != 0:
                self.on_error(f"ssh {self.host_fn()} : code {code}\n{err[-2000:]}")
                done(None)
                return
            done(bytes(out) if binary else bytes(out).decode("utf-8", "replace"))

        p.finished.connect(finished)
        p.errorOccurred.connect(
            lambda e: self.on_error(f"{SSH} : {e}") if e == QProcess.FailedToStart else None)
        p.start(SSH, args)
        if stdin_text is not None:
            p.write(stdin_text.encode())
            p.closeWriteChannel()

    def check_master(self, done):
        self._run(["-O", "check", self.host_fn()], None, lambda o: done(o is not None))

    def helper(self, action, args, done, binary=False):
        remote_cmd = "python3 - " + " ".join(shlex.quote(str(a)) for a in [action, *args])

        def parse(out):
            if out is None:
                done(None)
            elif binary:
                done(out)
            else:
                try:
                    r = json.loads(out)
                except ValueError:
                    self.on_error(f"réponse illisible de {action} :\n{out[-2000:]}")
                    done(None)
                    return
                if isinstance(r, dict) and "error" in r:
                    self.on_error(f"{action} : {r['error']}")
                    done(None)
                    return
                done(r)

        self._run(["-o", "BatchMode=yes", self.host_fn(), remote_cmd], REMOTE_HELPER, parse,
                  binary=binary)


class LogView(QPlainTextEdit):
    """Appends ssh output and honours \\r (a progress line rewritten in place)."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Monospace", 9))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setMaximumBlockCount(20000)
        self.follow = True

    def append_raw(self, text):
        text = ANSI_RE.sub("", text)
        bar = self.verticalScrollBar()
        at_end = self.follow
        cur = self.textCursor()
        cur.movePosition(QTextCursor.End)
        for piece in re.split(r"(\r\n|\r|\n)", text):
            if piece in ("\n", "\r\n"):
                cur.insertText("\n")
            elif piece == "\r":
                cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
                cur.removeSelectedText()
            elif piece:
                cur.insertText(piece)
        if at_end:
            bar.setValue(bar.maximum())


class OptionsForm(QWidget):
    """The form of one script's argparse options. An empty field / a flag in
    its default state is not put on the command line: the script's own
    default applies."""

    changed = None  # callback, set by the owner

    def __init__(self):
        super().__init__()
        self.lay = QFormLayout(self)
        self.lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.widgets = {}   # dest -> (action, widget, label)
        self.schema = None

    def build(self, schema):
        while self.lay.rowCount():
            self.lay.removeRow(0)
        self.widgets.clear()
        self.schema = schema
        if not schema:
            self.lay.addRow(QLabel("Options non lues : « Lire les options » (connexion requise)."))
            return
        for ac in schema["actions"]:
            flag = (max(ac["flags"], key=len) if ac["flags"] else ac["dest"])
            tip = ac["help"] + (f"\n\ndéfaut du script : {ac['default']}" if ac["default"] not in (None, False) else "")
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
                w.setPlaceholderText(f"défaut : {ac['default']}" if ac["default"] is not None else
                                     (ac["metavar"] or ""))
                w.textChanged.connect(self._emit)
            w.setToolTip(tip)
            lab = QLabel(flag)
            lab.setToolTip(tip)
            self.lay.addRow(lab, w)
            self.widgets[ac["dest"]] = (ac, w, lab)

    def filter(self, text):
        """Show only the options whose flag or help contains @p text."""
        t = text.strip().lower()
        for ac, w, lab in self.widgets.values():
            hay = (" ".join(ac["flags"]) + " " + ac["dest"] + " " + ac["help"]).lower()
            self.lay.setRowVisible(w, not t or t in hay)

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


# Chart colours (data-viz reference palette). A scatter keeps three hues at
# most: they pass the all-pairs colour-vision checks in both modes, a fourth
# does not -- further categories fold into "autres", in muted ink.
VIZ = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}
# Colour follows the entity, never its rank: fixed slots per category.
COLOR_BY = {
    "Variante": (lambda exe: exe_parts(exe)[0], ["chunk", "chunk2", "chunk2a"]),
    "Taille": (lambda exe: exe_parts(exe)[1], ["256", "1024", "dyn"]),
    "Aucune": (lambda exe: "runs", ["runs"]),
}
EXE_RE = re.compile(r"shrink_bench_(chunk2a|chunk2|chunk)(?:_(\w+))?$")


def exe_parts(exe):
    """shrink_bench_chunk2a_dyn -> ("chunk2a", "dyn"); no size -> "default"."""
    m = EXE_RE.search(exe)
    return (m.group(1), m.group(2) or "default") if m else (exe, "?")


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


class Main(QMainWindow):
    COLS = ["Run", "Programme", "Partition", "Job", "Statut", "Avancement", "Dernière ligne"]
    ACTIVE = ("en cours", "en attente", "démarrage")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("shrink-array — runs sur inti (prototype)")
        self.settings = QSettings(APP, APP)
        self.remote = Remote(self, lambda: self.host.text().strip(), self.error)
        self.runs = []
        self.current = None      # id of the run shown in the log
        self.offset = 0
        self.tail_busy = False
        self.list_busy = False
        self.prev_status = None  # id -> status at the previous refresh (notifications)

        # --- connection
        self.host = QLineEdit(self._get("host"))
        self.conn = QLabel("?")
        bcheck = QPushButton("Vérifier la connexion")
        bcheck.clicked.connect(self.check)
        top = QHBoxLayout()
        top.addWidget(QLabel("Calculateur :"))
        top.addWidget(self.host)
        top.addWidget(bcheck)
        top.addWidget(self.conn, 1)

        # --- where / how
        self.preset = QComboBox()
        for key, p in PRESETS.items():
            self.preset.addItem(p["label"], key)
        self.preset.setCurrentIndex(max(0, self.preset.findData(self._get("preset"))))
        self.mode = QComboBox()
        self.mode.addItem("ccc_msub (batch)", "msub")
        self.mode.addItem("ccc_mprun détaché", "mprun")
        self.mode.setCurrentIndex(max(0, self.mode.findData(self._get("mode"))))
        self.time_s = QSpinBox()
        self.time_s.setRange(60, 86400 * 3)
        self.time_s.setSingleStep(600)
        self.time_s.setSuffix(" s")
        self.time_s.setValue(int(self._get("time_s")))
        self.directives = QPlainTextEdit()
        self.directives.setFont(QFont("Monospace", 9))
        self.directives.setFixedHeight(92)
        self.directives.setToolTip("Lignes #MSUB du script batch (ccc_msub) ou options ccc_mprun "
                                   "(mode détaché), propres à la partition et mémorisées.\n"
                                   "-T, -r, -o et -e sont ajoutés d'après le formulaire.")
        self.repo = QLineEdit(self._get("repo"))
        self.runs_dir = QLineEdit(self._get("runs_dir"))
        self.local_dir = QLineEdit(self._get("local_dir"))
        where = QFormLayout()
        where.addRow("Partition", self.preset)
        where.addRow("Soumission", self.mode)
        where.addRow("Durée max (-T)", self.time_s)
        where.addRow("Directives", self.directives)
        where.addRow("Dépôt (inti)", self.repo)
        where.addRow("Runs (inti)", self.runs_dir)
        where.addRow("Résultats (local)", self.local_dir)
        gw = QGroupBox("Où et comment")
        gw.setLayout(where)

        # --- what
        self.program = QComboBox()
        self.program.addItems(PROGRAMS)
        self.program.setCurrentText(self._get("program"))
        bscan = QPushButton("Lire les options")
        bscan.setToolTip("Relit les options du script sur inti (après une modification du script)")
        bscan.clicked.connect(self.scan)
        prow = QHBoxLayout()
        prow.addWidget(self.program, 1)
        prow.addWidget(bscan)
        self.form = OptionsForm()
        self.form.changed = self.update_preview
        self.opt_filter = QLineEdit()
        self.opt_filter.setPlaceholderText("filtrer les options (nom ou aide)…")
        self.opt_filter.setClearButtonEnabled(True)
        self.opt_filter.textChanged.connect(self.form.filter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        bdef_reset = QPushButton("Défauts de l'architecture")
        bdef_reset.clicked.connect(self.apply_arch_defaults)
        bdef_save = QPushButton("Enregistrer comme défauts")
        bdef_save.clicked.connect(self.save_arch_defaults)
        drow = QHBoxLayout()
        drow.addWidget(bdef_reset)
        drow.addWidget(bdef_save)
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("arguments en plus, tels quels (facultatif)")
        self.extra.textChanged.connect(self.update_preview)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Monospace", 9))
        self.preview.setFixedHeight(64)
        blaunch = QPushButton("Soumettre")
        blaunch.setStyleSheet("font-weight: bold; padding: 6px;")
        blaunch.clicked.connect(self.launch)
        what = QVBoxLayout()
        what.addLayout(prow)
        what.addWidget(self.opt_filter)
        what.addWidget(scroll, 1)
        what.addLayout(drow)
        what.addWidget(self.extra)
        what.addWidget(QLabel("Commande :"))
        what.addWidget(self.preview)
        what.addWidget(blaunch)
        gp = QGroupBox("Quoi")
        gp.setLayout(what)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(gw)
        ll.addWidget(gp, 1)

        # --- runs table + actions
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        for c in range(len(self.COLS) - 1):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.select)
        brefresh = QPushButton("Rafraîchir")
        brefresh.clicked.connect(self.refresh)
        bstop = QPushButton("Arrêter")
        bstop.clicked.connect(self.stop)
        bfetch = QPushButton("Récupérer les résultats")
        bfetch.clicked.connect(self.fetch)
        bagain = QPushButton("Reprendre ses options")
        bagain.setToolTip("Remet dans le formulaire le programme, la partition et les options de ce run")
        bagain.clicked.connect(self.reuse)
        bdelete = QPushButton("Supprimer")
        bdelete.setToolTip("Supprime le dossier du run sur inti (pas les sorties du programme)")
        bdelete.clicked.connect(self.delete)
        acts = QHBoxLayout()
        for b in (brefresh, bstop, bfetch, bagain, bdelete):
            acts.addWidget(b)
        acts.addStretch(1)
        self.log = LogView()

        # log tab: search + follow
        self.find = QLineEdit()
        self.find.setPlaceholderText("rechercher dans le log (Entrée : suivant)")
        self.find.setClearButtonEnabled(True)
        self.find.returnPressed.connect(self.find_next)
        self.follow = QCheckBox("Suivre la fin")
        self.follow.setChecked(True)
        self.follow.toggled.connect(lambda on: setattr(self.log, "follow", on))
        frow = QHBoxLayout()
        frow.addWidget(self.find, 1)
        frow.addWidget(self.follow)
        logw = QWidget()
        lgl = QVBoxLayout(logw)
        lgl.setContentsMargins(0, 0, 0, 0)
        lgl.addLayout(frow)
        lgl.addWidget(self.log, 1)
        find_act = QAction(self)
        find_act.setShortcut(QKeySequence.Find)
        find_act.triggered.connect(lambda: (self.tabs.setCurrentIndex(0), self.find.setFocus(),
                                            self.find.selectAll()))
        self.addAction(find_act)

        # summary tab
        self.sum_note = QLabel("")
        self.sum_table = QTableWidget(0, 0)
        self.sum_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sum_table.verticalHeader().setVisible(False)
        self.sum_table.setSortingEnabled(True)
        bsum = QPushButton("Rafraîchir le résumé")
        bsum.clicked.connect(self.load_summary)
        srow = QHBoxLayout()
        srow.addWidget(self.sum_note, 1)
        srow.addWidget(bsum)
        sumw = QWidget()
        sl = QVBoxLayout(sumw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addLayout(srow)
        sl.addWidget(self.sum_table, 1)

        # chart tab: seconds per run against threads (multi_launch_stress.py)
        self.pt_data = None
        self.pt_build = QComboBox()
        self.pt_build.currentIndexChanged.connect(self.draw_points)
        self.pt_color = QComboBox()
        self.pt_color.addItems(COLOR_BY)
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
        prow2.addWidget(QLabel("Couleur"))
        prow2.addWidget(self.pt_color)
        prow2.addWidget(self.pt_log)
        prow2.addWidget(self.pt_median)
        prow2.addWidget(self.pt_ok_only)
        prow2.addStretch(1)
        prow2.addWidget(bpts)
        self.pt_note = QLabel("")
        self.pt_view = ScatterView()
        ptw = QWidget()
        ptl = QVBoxLayout(ptw)
        ptl.setContentsMargins(0, 0, 0, 0)
        ptl.addLayout(prow2)
        ptl.addWidget(self.pt_note)
        ptl.addWidget(self.pt_view, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(logw, "Log")
        self.tabs.addTab(sumw, "Résumé")
        self.tabs.addTab(ptw, "Graphique")
        self.tabs.currentChanged.connect(self.tab_changed)

        # end-of-run notification: the tray when there is one
        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.SP_ComputerIcon), self)
            self.tray.setToolTip("shrink-array — runs sur inti")
            self.tray.show()

        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.table)
        tl.addLayout(acts)
        split_v = QSplitter(Qt.Vertical)
        split_v.addWidget(tw)
        split_v.addWidget(self.tabs)
        split_v.setSizes([260, 480])

        split_h = QSplitter(Qt.Horizontal)
        split_h.addWidget(left)
        split_h.addWidget(split_v)
        split_h.setSizes([480, 900])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addLayout(top)
        cl.addWidget(split_h, 1)
        self.setCentralWidget(central)
        self.resize(1400, 860)

        quit_act = QAction("Quitter", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)

        self._preset_key = None
        self.preset.currentIndexChanged.connect(self.preset_changed)
        self.mode.currentIndexChanged.connect(self.preset_changed)
        self.program.currentTextChanged.connect(self.program_changed)
        self.preset_changed()
        self.program_changed()

        # --- polling: run list every 15 s, the shown log every 2 s
        self.t_list = QTimer(self)
        self.t_list.timeout.connect(self.refresh)
        self.t_list.start(15000)
        self.t_tail = QTimer(self)
        self.t_tail.timeout.connect(self.poll_tail)
        self.t_tail.start(2000)
        QTimer.singleShot(0, self.check)

    # ---------------------------------------------------------------- utils
    def _get(self, key):
        return self.settings.value(key, DEFAULTS[key])

    def arch(self):
        return PRESETS[self.preset.currentData()]["arch"]

    def _directives_key(self, preset=None, mode=None):
        return f"directives/{preset or self.preset.currentData()}/{mode or self.mode.currentData()}"

    def save(self):
        for key, w in (("host", self.host), ("repo", self.repo), ("runs_dir", self.runs_dir),
                       ("local_dir", self.local_dir)):
            self.settings.setValue(key, w.text())
        self.settings.setValue("time_s", self.time_s.value())
        self.settings.setValue("preset", self.preset.currentData())
        self.settings.setValue("mode", self.mode.currentData())
        self.settings.setValue("program", self.program.currentText())
        if self._preset_key:
            self.settings.setValue(self._preset_key, self.directives.toPlainText())

    def closeEvent(self, ev):
        self.save()
        self.t_list.stop()
        self.t_tail.stop()
        # The runs live on inti: only the pending ssh calls are dropped here.
        for p in list(self.remote.procs):
            p.finished.disconnect()
            p.kill()
            p.waitForFinished(2000)
        self.remote.procs.clear()
        if self.tray:
            self.tray.hide()
        super().closeEvent(ev)

    def error(self, msg):
        self.statusBar().showMessage(msg.splitlines()[0] if msg else "", 20000)
        print(msg, file=sys.stderr)

    def info(self, msg):
        self.statusBar().showMessage(msg, 10000)

    # ------------------------------------------------- partition / program
    def preset_changed(self, *_):
        if self._preset_key:   # keep what was typed for the previous one
            self.settings.setValue(self._preset_key, self.directives.toPlainText())
        p = PRESETS[self.preset.currentData()]
        builtin = p["msub"] if self.mode.currentData() == "msub" else p["mprun"]
        self._preset_key = self._directives_key()
        self.directives.setPlainText(self.settings.value(self._preset_key, builtin))
        self.apply_arch_defaults()

    def program_changed(self, *_):
        cached = self.settings.value(f"schema/{self.program.currentText()}")
        self.form.build(json.loads(cached) if cached else None)
        self.form.filter(self.opt_filter.text())
        self.apply_arch_defaults()
        if not cached:
            QTimer.singleShot(0, self.scan)

    def scan(self):
        prog = self.program.currentText()
        self.info(f"lecture des options de {prog}…")

        def done(r):
            if r is None:
                return
            keep = self.form.values() if self.form.schema else None
            self.settings.setValue(f"schema/{prog}", json.dumps(r))
            if prog == self.program.currentText():
                self.form.build(r)
                if keep is not None:
                    self.form.set_values(keep)
                else:
                    self.apply_arch_defaults()
            self.form.filter(self.opt_filter.text())
            self.info(f"{prog} : {len(r['actions'])} options lues")

        spec = PROGRAMS[prog]
        self.remote.helper("options", [self.repo.text().strip(), spec["script"], spec["scan"],
                                       spec["interp"]], done)

    def arch_defaults(self):
        prog = self.program.currentText()
        saved = self.settings.value(f"defaults/{prog}/{self.arch()}")
        return json.loads(saved) if saved else dict(ARCH_DEFAULTS.get((prog, self.arch()), {}))

    def apply_arch_defaults(self):
        if hasattr(self, "form"):
            self.form.set_values(self.arch_defaults())

    def save_arch_defaults(self):
        prog = self.program.currentText()
        self.settings.setValue(f"defaults/{prog}/{self.arch()}", json.dumps(self.form.values()))
        self.info(f"défauts enregistrés pour {prog} sur {self.arch()}")

    def command(self):
        prog = self.program.currentText()
        parts = [PROGRAMS[prog]["interp"], os.path.basename(PROGRAMS[prog]["script"])] + self.form.args()
        if self.extra.text().strip():
            parts.append(self.extra.text().strip())
        return " ".join(parts)

    def update_preview(self, *_):
        if hasattr(self, "preview"):
            self.preview.setPlainText(self.command())

    # ----------------------------------------------------------- connection
    def check(self):
        host = self.host.text().strip()
        self.conn.setText("vérification…")

        def master(ok):
            if not ok:
                self.conn.setText(f"<span style='color:#c0392b'>pas de connexion maître : "
                                  f"lancez <b>ssh -fNM {host}</b> dans un terminal</span>")
                return
            self.remote.helper("check", [], helper_ok)

        def helper_ok(r):
            if r is None:
                self.conn.setText("<span style='color:#c0392b'>connexion maître OK, mais python3 "
                                  "distant en échec (voir la barre d'état)</span>")
                return
            warn = "" if r.get("msub") else " — <b>ccc_msub introuvable</b>"
            self.conn.setText(f"<span style='color:#1e8449'>connecté — {r['host']}, "
                              f"python {r['python']}</span>{warn}")
            self.refresh()
            if not self.form.schema:
                self.scan()

        self.remote.check_master(master)

    # --------------------------------------------------------------- launch
    def launch(self):
        self.save()
        prog = self.program.currentText()
        if not self.form.schema and QMessageBox.question(
                self, "Options non lues",
                "Les options du script n'ont pas été lues : seuls les arguments en plus seront "
                "passés.\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        pkey = self.preset.currentData()
        mode = self.mode.currentData()
        busy = [r["id"] for r in self.runs
                if r["status"] in ("en cours", "en attente", "démarrage")
                and r["meta"].get("partition") == pkey and r["meta"].get("program") == prog]
        if busy and QMessageBox.question(
                self, "Run déjà actif",
                f"{prog} a déjà un run actif sur {pkey} ({', '.join(busy)}).\n"
                "Deux runs simultanés de même architecture peuvent écrire le même répertoire "
                "(oracles, résumé).\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        cmd = self.command()
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + prog.split(".")[0].split("_")[0] + "-" + pkey
        spec = {
            "mode": mode,
            "run.sh": make_run_sh(self.repo.text().strip(), PROGRAMS[prog]["cwd"], cmd),
            "msub": self.directives.toPlainText() if mode == "msub" else "",
            "launch.sh": make_launch_sh(f"{' '.join(self.directives.toPlainText().split())} "
                                        f"-T{self.time_s.value()}") if mode == "mprun" else "",
            "name": re.sub(r"[^A-Za-z0-9_.-]", "_", f"{prog.split('.')[0]}_{pkey}")[:40],
            "time_s": self.time_s.value(),
            "meta": {"program": prog, "partition": pkey, "arch": self.arch(), "mode": mode,
                     "command": cmd, "values": self.form.values(), "extra": self.extra.text(),
                     "repo": self.repo.text().strip(), "time_s": self.time_s.value(),
                     "created": time.strftime("%F %T"),
                     **{k: PROGRAMS[prog][k] for k in ("out_re", "out_files", "progress_re",
                                                       "eta_re", "summary", "summary_file")}},
        }
        blob = base64.b64encode(json.dumps(spec).encode()).decode()

        def done(r):
            if r is None:
                return
            self.info(f"soumis : job {r['jobid']}" if "jobid" in r else f"lancé : {r['run_dir']}")
            self.current, self.offset = rid, 0
            self.log.clear()
            self.refresh(select=rid)

        self.remote.helper("launch", [self.runs_dir.text().strip(), rid, blob], done)

    # ----------------------------------------------------------------- runs
    def refresh(self, select=None):
        if self.list_busy:
            return
        self.list_busy = True

        def done(r):
            self.list_busy = False
            if r is None:
                return
            self.runs = r["runs"]
            self.notify_changes()
            cur = next((x for x in self.runs if x["id"] == self.current), None)
            if cur and cur["status"] in self.ACTIVE:
                if self.tabs.currentIndex() == 1:
                    self.load_summary()
                elif self.tabs.currentIndex() == 2:
                    self.load_points()
            keep = select or self.current
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.runs))
            for i, run in enumerate(self.runs):
                st = run["status"] + (f" (code {run['code']})" if run["status"] == "fini" else "")
                vals = [run["id"], run["meta"].get("program", ""), run["meta"].get("partition", ""),
                        run.get("jobid", ""), st, self.progress_text(run), run["last"]]
                for j, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if j == 4:
                        s = run["status"]
                        color = ("#1e8449" if s == "fini" and run["code"] == 0 else
                                 "#c0392b" if s == "fini" or s.startswith("terminé") else
                                 "#2471a3" if s in ("en cours", "démarrage") else
                                 "#7d3c98" if s == "en attente" else "#b9770e")
                        it.setForeground(QColor(color))
                    it.setToolTip(run["meta"].get("command", "") if j < 2 else str(v))
                    self.table.setItem(i, j, it)
                if run["id"] == keep:
                    self.table.selectRow(i)
            self.table.blockSignals(False)
            if select:
                self.select()

        self.remote.helper("list", [self.runs_dir.text().strip()], done)

    def selected_run(self):
        rows = self.table.selectionModel().selectedRows()
        return self.runs[rows[0].row()] if rows and rows[0].row() < len(self.runs) else None

    def selected(self):
        r = self.selected_run()
        return r["id"] if r else None

    def select(self):
        rid = self.selected()
        if rid and rid != self.current:
            self.current, self.offset = rid, 0
            self.log.clear()
            self.poll_tail()
            self.pt_data = None
            self.tab_changed(self.tabs.currentIndex())

    def poll_tail(self):
        if not self.current or self.tail_busy:
            return
        self.tail_busy = True
        rid = self.current

        def done(r):
            self.tail_busy = False
            if r is None or rid != self.current:
                return
            if r["offset"] < self.offset:      # file restarted
                self.log.clear()
            if r["data"]:
                self.log.append_raw(r["data"])
            self.offset = r["offset"]
            if len(r["data"]) >= 512 * 1024:   # more to read
                QTimer.singleShot(0, self.poll_tail)

        self.remote.helper("tail", [self.runs_dir.text().strip(), rid, self.offset], done)

    def reuse(self):
        run = self.selected_run()
        if not run:
            return
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
        self.form.set_values(m.get("values", {}))
        self.extra.setText(m.get("extra", ""))
        self.info(f"options de {run['id']} reprises")

    def stop(self):
        run = self.selected_run()
        if not run:
            return
        what = f"le job {run['jobid']} (ccc_mdel)" if run.get("jobid") else "ccc_mprun (SIGTERM)"
        if QMessageBox.question(self, "Arrêter", f"Arrêter le run {run['id']} ?\n{what}") != QMessageBox.Yes:
            return

        def done(r):
            if r is not None:
                self.info(f"{run['id']} : arrêt demandé")
            QTimer.singleShot(2000, self.refresh)

        self.remote.helper("stop", [self.runs_dir.text().strip(), run["id"]], done)

    @staticmethod
    def progress_text(run):
        p = run.get("progress")
        if not p:
            return ""
        done, total, eta = p
        txt = f"{done}/{total} ({100.0 * done / total:.0f} %)" if total else f"{done}"
        if eta and run["status"] in Main.ACTIVE:
            h, m = divmod(int(eta) // 60, 60)
            txt += (f" · reste ~{h} h {m:02d}" if h else
                    f" · reste ~{m} min" if m else " · reste < 1 min")
        return txt

    def notify_changes(self):
        """A run that leaves the active states since the previous refresh."""
        now = {r["id"]: r["status"] for r in self.runs}
        if self.prev_status is not None:
            for r in self.runs:
                before = self.prev_status.get(r["id"])
                if before in self.ACTIVE and r["status"] not in self.ACTIVE:
                    st = r["status"] + (f" (code {r['code']})" if r["status"] == "fini" else "")
                    msg = f"{r['meta'].get('program', '')} sur {r['meta'].get('partition', '')} : {st}"
                    self.info(f"{r['id']} : {st}")
                    if self.tray:
                        icon = (QSystemTrayIcon.Information if r["status"] == "fini" and r["code"] == 0
                                else QSystemTrayIcon.Warning)
                        self.tray.showMessage(f"Run {r['id']}", msg, icon, 15000)
                    QApplication.alert(self)
        self.prev_status = now

    def find_next(self):
        text = self.find.text()
        if not text:
            return
        if not self.log.find(text):
            self.log.moveCursor(QTextCursor.Start)       # wrap around
            if not self.log.find(text):
                self.info(f"« {text} » absent du log")
                return
        self.follow.setChecked(False)                    # stay on the match

    def load_summary(self):
        rid = self.current
        if not rid:
            return
        self.sum_note.setText("chargement…")

        def done(r):
            if r is None or rid != self.current:
                return
            self.sum_note.setText(r.get("note", ""))
            cols, rows = r.get("columns", []), r.get("rows", [])
            self.sum_table.setSortingEnabled(False)
            self.sum_table.clear()
            self.sum_table.setColumnCount(len(cols))
            self.sum_table.setHorizontalHeaderLabels(cols)
            self.sum_table.setRowCount(len(rows))
            fails = set(r.get("fail_rows", []))
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    it = NumItem(v)
                    if i in fails:
                        it.setForeground(QColor("#c0392b"))
                    if row and row[0] == "TOTAL":
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                    self.sum_table.setItem(i, j, it)
            self.sum_table.resizeColumnsToContents()
            self.sum_table.setSortingEnabled(True)

        self.remote.helper("summary", [self.runs_dir.text().strip(), rid], done)

    def tab_changed(self, i):
        if i == 1:
            self.load_summary()
        elif i == 2:
            self.load_points()

    def load_points(self):
        rid = self.current
        if not rid:
            return

        def done(r):
            if r is None or rid != self.current:
                return
            if "rows" not in r:
                self.pt_data = None
                self.pt_note.setText(r.get("note", ""))
                self.pt_view.setChart(QChart())
                self.pt_view.points = []
                return
            first = self.pt_data is None
            self.pt_data = r
            keep = self.pt_build.currentText()
            self.pt_build.blockSignals(True)
            self.pt_build.clear()
            self.pt_build.addItem("tous")
            self.pt_build.addItems(r["builds"])
            if not first and keep in r["builds"]:
                self.pt_build.setCurrentText(keep)
            self.pt_build.blockSignals(False)
            self.draw_points()

        self.remote.helper("points", [self.runs_dir.text().strip(), rid], done)

    def draw_points(self, *_):
        r = self.pt_data
        if not r:
            return
        dark = self.palette().window().color().lightness() < 128
        V = VIZ["dark" if dark else "light"]
        key_of, slots = COLOR_BY[self.pt_color.currentText()]
        build_sel = self.pt_build.currentText()
        bi = r["builds"].index(build_sel) if build_sel in r["builds"] else None
        log_y = self.pt_log.isChecked()

        groups = {}            # category -> [(threads, seconds, tip)]
        fails, ok_pts = [], []
        for run, b, e, th, sec, rc in r["rows"]:
            if bi is not None and b != bi:
                continue
            if log_y and sec <= 0:
                continue
            exe, build = r["exes"][e], r["builds"][b]
            tip = (f"<b>{sec:.3f} s</b> · {th} threads<br>"
                   f"{html.escape(build)} · {html.escape(exe)}<br>"
                   f"run {run}" + (f" · <b>code {rc}</b>" if rc else ""))
            p = (th, sec, tip)
            if rc != 0:
                fails.append(p)
                continue
            ok_pts.append(p)
            cat = key_of(exe)
            groups.setdefault(cat if cat in slots else "autres", []).append(p)

        chart = QChart()
        chart.setBackgroundBrush(QBrush(QColor(V["surface"])))
        chart.setBackgroundRoundness(0)
        chart.setMargins(chart.margins())
        title = "Temps par run selon le nombre de threads — " + (build_sel if bi is not None else "tous les builds")
        chart.setTitle(title)
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
            return sr

        drawn = []
        for i, cat in enumerate(slots):
            if groups.get(cat):
                scatter(f"{cat} ({len(groups[cat])})", groups[cat],
                        V["series"][i], 7, QScatterSeries.MarkerShapeCircle)
                drawn += groups[cat]
        if groups.get("autres"):
            scatter(f"autres ({len(groups['autres'])})", groups["autres"],
                    V["muted"], 7, QScatterSeries.MarkerShapeCircle)
            drawn += groups["autres"]
        if fails and not self.pt_ok_only.isChecked():
            # Failures: a shape in ink, named in the legend -- not a status red,
            # which the colour-vision checks cannot tell from the orange slot.
            scatter(f"échec ({len(fails)})", fails, V["ink"], 12,
                    QScatterSeries.MarkerShapeTriangle, alpha=255, ring=False)
            drawn += fails

        if self.pt_median.isChecked() and len(ok_pts) >= 5:
            import statistics
            lo = min(p[0] for p in ok_pts)
            hi = max(p[0] for p in ok_pts)
            width = max(1, round((hi - lo) / 15)) if hi > lo else 1
            bins = {}
            for th, sec, _t in ok_pts:
                bins.setdefault((th - lo) // width, []).append(sec)
            med = QLineSeries()
            med.setName("médiane")
            med.setPen(QPen(QColor(V["ink2"]), 2))
            for k in sorted(bins):
                if len(bins[k]) >= 3:
                    med.append(lo + (k + 0.5) * width, statistics.median(bins[k]))
            if med.count() >= 2:
                chart.addSeries(med)

        ax = QValueAxis()
        ax.setTitleText("threads")
        ax.setLabelFormat("%d")
        ay = QLogValueAxis() if log_y else QValueAxis()
        ay.setTitleText("temps par run (s)")
        ay.setLabelFormat("%.3g" if log_y else "%.1f")
        if log_y:
            ay.setMinorTickCount(-1)
        for a in (ax, ay):
            a.setLabelsColor(QColor(V["muted"]))
            a.setTitleBrush(QBrush(QColor(V["ink2"])))
            a.setGridLineColor(QColor(V["grid"]))
            a.setLinePenColor(QColor(V["axis"]))
            a.setMinorGridLineVisible(False)
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
        self.pt_view.setChart(chart)
        self.pt_view.points = drawn
        self.pt_view.log_y = log_y
        shown = len(drawn)
        more = (f" (les {len(r['rows'])} derniers sur {r['total']})" if r["total"] > len(r["rows"]) else "")
        self.pt_note.setText(f"{shown} run(s) affiché(s){more}. Survolez un point pour le détail ; "
                             "le tableau est dans l'onglet Résumé.")

    def delete(self):
        run = self.selected_run()
        if not run:
            return
        if run["status"] in self.ACTIVE:
            self.error("arrêtez d'abord ce run (il est encore actif)")
            return
        if QMessageBox.question(
                self, "Supprimer",
                f"Supprimer le dossier du run {run['id']} sur inti ?\n"
                "(log, scripts, code de retour ; les sorties du programme restent en place)"
        ) != QMessageBox.Yes:
            return

        def done(r):
            if r is None:
                return
            if self.current == run["id"]:
                self.current = None
                self.log.clear()
            self.info(f"{run['id']} supprimé")
            self.refresh()

        self.remote.helper("delete", [self.runs_dir.text().strip(), run["id"]], done)

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


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
