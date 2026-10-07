"""Run on inti by shrink_gui: `python3 - <action> <args...>`, this file on stdin.

Nothing is installed on inti: every ssh call sends this file again, so it is
always the laptop's version. It must stay Python 3.6 compatible (the system
python3 of a RHEL 8 login node) and import nothing outside the standard
library. Imported by the tests, it runs nothing (see the end).
"""
import base64, glob, io, json, os, re, shlex, signal, socket, subprocess, sys, tarfile

def rdir(base, rid=None):
    d = os.path.expanduser(base)
    return os.path.join(d, rid) if rid else d

def read(p, default=None):
    """The file as text, UTF-8 whatever the locale: a login shell in the C
    locale gives Python 3.6 an ASCII default, and the scripts are UTF-8."""
    try:
        with open(p, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except (IOError, OSError):
        return default

def write(p, text):
    with open(p, "wb") as f:
        f.write(text.encode("utf-8"))

def text_open(p):
    """open() for reading text, UTF-8 whatever the locale (see read)."""
    return io.open(p, encoding="utf-8", errors="replace", newline="")

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def cmd_out(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT, universal_newlines=True)
    except (OSError, subprocess.CalledProcessError):
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

def git_info(repo):
    """{commit, branch, dirty} of the repository, or None (no git, no repo)."""
    r = os.path.expanduser(repo)

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", r] + list(args), stderr=subprocess.DEVNULL,
                                           universal_newlines=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    if not commit:
        return None
    return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "",
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}

def act_check(a):
    r = {"host": socket.gethostname(), "python": sys.version.split()[0],
         "msub": cmd_out(["which", "ccc_msub"]) is not None,
         "mqinfo": cmd_out(["ccc_mqinfo"])}       # the QoS limits (parsed by the GUI)
    if a:
        r["git"] = git_info(a[0])
    print(json.dumps(r))

HELP_RE = re.compile(r"^\s{1,8}(?:-\w,\s*)?(--[A-Za-z][\w-]*)(=\S+)?(?=\s|$)\s*(.*)$")

USAGE_RE = re.compile(r"^usage\(\)\s*\{[^\n]*\n\s*cat\b[^\n]*<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1\s*$",
                      re.M | re.S)

def scan_help(path, interp):
    """The options of a shell script, from its help text: lines
    "  --flag[=<value>]   description", continued by deeper-indented lines.

    The text is READ from the script's usage() here-document, never obtained
    by running it: this runs on the login node, and compile_all.sh loads its
    modules before it looks at --help. No usage() block: an error, not a run."""
    source = read(path)
    if source is None:
        print(json.dumps({"error": "%s : illisible" % path}))
        return
    m = USAGE_RE.search(source)
    if not m:
        print(json.dumps({"error": "%s : pas de bloc usage() { cat <<EOF ... EOF } où lire "
                                   "les options (le script n'est pas exécuté ici)" % os.path.basename(path)}))
        return
    text = m.group(2)
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
        print(json.dumps({"description": self.description or "", "actions": acts,
                          "python": sys.version.split()[0]}))
        raise Done()

    argparse.ArgumentParser.parse_known_args = dump
    sys.stdout = sys.stderr      # what the script prints before parsing stays off the JSON
    os.chdir(os.path.dirname(path))
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path]
    pyv = "python %s (%s)" % (sys.version.split()[0], sys.executable)
    try:
        runpy.run_path(path, run_name="__main__")
    except Done:
        return
    except SystemExit:
        pass
    except BaseException as e:
        import traceback
        tb = traceback.extract_tb(sys.exc_info()[2])
        where = "%s:%d" % (os.path.basename(tb[-1].filename), tb[-1].lineno) if tb else "?"
        sys.stdout = real_stdout
        print(json.dumps({"error": "%s : %s: %s (à %s), avec %s"
                          % (a[1], type(e).__name__, e, where, pyv)}))
        return
    sys.stdout = real_stdout
    print(json.dumps({"error": "%s ne lit pas ses options avec argparse (%s)" % (a[1], pyv)}))

def act_launch(a):
    base, rid, spec = a[0], a[1], json.loads(base64.b64decode(a[2]).decode())
    d = rdir(base, rid)
    os.makedirs(d)
    write(os.path.join(d, "run.sh"), spec["run.sh"])
    # The code the run is submitted on; run.sh records the one it STARTS on
    # (git_start): a pending job can start after a git pull.
    spec["meta"]["git"] = git_info(spec["meta"].get("repo", "~"))
    write(os.path.join(d, "meta.json"), json.dumps(spec["meta"], indent=1))
    if spec["mode"] == "msub":
        log = os.path.join(d, "out.log")
        job = ("#!/bin/bash\n" + spec["msub"].rstrip("\n") + "\n"
               + "#MSUB -r %s\n#MSUB -T %d\n#MSUB -o %s\n#MSUB -e %s\n"
               % (spec["name"], spec["time_s"], log, log)
               + ("#MSUB -Q %s\n" % spec["qos"] if spec.get("qos") else "")
               + 'D=%s\n' % shlex.quote(d)
               + ('echo "[gui] launcher: %s"\n' % spec["launcher"] if spec.get("launcher") else "")
               + '%sbash "$D/run.sh"\ncode=$?\necho "$code" > "$D/exit_code"\n'
               % (spec["launcher"] + " " if spec.get("launcher") else "")
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

def read_events(d):
    """The records of the run's events.jsonl (bench_runner/events.py), or []."""
    p = os.path.join(d, "events.jsonl")
    out = []
    try:
        with text_open(p) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass      # a line being written
    except (IOError, OSError):
        pass
    return out

def progress_from_events(ev):
    """[done, total, eta_s, phase] from the events, or None."""
    if not ev:
        return None
    start = next((e for e in ev if e.get("kind") == "start"), None)
    phase = None
    for e in ev:
        if e.get("kind") == "phase_start":
            phase = e.get("phase")
        elif e.get("kind") == "phase_end" and e.get("phase") == phase:
            phase = None
        elif e.get("kind") == "end":
            phase = "fin"
    prog = None
    for e in ev:
        if e.get("kind") == "progress":
            prog = e
    done = prog["done"] if prog else 0
    total = prog["total"] if prog else (start or {}).get("total_runs", 0)
    eta = None
    if prog and start and done and total and phase != "fin":
        eta = (prog["t"] - start["t"]) / done * (total - done)
    if prog and prog.get("phase") and prog["phase"] != phase and phase != "fin":
        phase = prog["phase"]
    return [done, total, eta, phase]

def progress_of(path, meta):
    """(done, total, eta_s[, phase]): bench_and_test.py's events.jsonl when
    there is one, else the log with the program's own regexes."""
    ev = read_events(os.path.dirname(path))
    if ev:
        return progress_from_events(ev)
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
                         "git_start": (read(os.path.join(d, "git_start")) or "").strip(),
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
    phases = phases_section(read_events(d))
    if not kind or not name:
        if phases:
            print(json.dumps({"sections": [phases]}))
        else:
            print(json.dumps({"note": "pas de résumé pour ce programme"}))
        return
    od = out_dir(d, meta)
    path = os.path.join(od, name) if od else None
    if not path or not os.path.isfile(path):
        if phases:
            print(json.dumps({"sections": [phases, {"title": name, "note": "pas encore de %s" % name,
                                                    "columns": [], "rows": []}]}))
        else:
            print(json.dumps({"note": "pas encore de %s" % name}))
        return
    with text_open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        print(json.dumps({"note": "%s est vide" % name}))
        return
    head, body = rows[0], rows[1:]
    if kind != "multi_launch":
        table = {"title": name, "columns": head, "rows": body[-1000:],
                 "note": "%s — %d ligne(s)%s" % (name, len(body),
                         ", les 1000 dernières" if len(body) > 1000 else "")}
        print(json.dumps({"sections": [phases, table]} if phases else table))
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

def phases_section(ev):
    """The phases of a bench_and_test.py run, from its events, as a table."""
    if not ev:
        return None
    import time as _time
    rows, fails, open_ = [], [], {}
    for e in ev:
        k = e.get("kind")
        if k == "phase_start":
            open_[e["phase"]] = e
        elif k == "phase_end":
            s0 = open_.pop(e["phase"], None)
            st = {"ok": "ok", "errors": "erreurs", "exception": "exception",
                  "interrupted": "interrompue"}.get(e.get("status"), e.get("status", "?"))
            if st != "ok":
                fails.append(len(rows))
            rows.append([e["phase"], st, "%.1f" % e.get("seconds", 0), str(e.get("errors", 0)),
                         _time.strftime("%H:%M:%S", _time.localtime(s0["t"])) if s0 else ""])
    now = _time.time()
    for name, s0 in open_.items():
        rows.append([name, "en cours", "%.1f" % (now - s0["t"]), "", _time.strftime("%H:%M:%S", _time.localtime(s0["t"]))])
    end = next((e for e in ev if e.get("kind") == "end"), None)
    note = "%d phase(s)" % len(rows)
    if end:
        note += " — fin : %d erreur(s)%s, %.1f min" % (end.get("errors", 0),
                                                     ", interrompu" if end.get("interrupted") else "",
                                                     end.get("minutes", 0))
    return {"title": "Phases", "columns": ["phase", "statut", "durée s", "erreurs", "début"],
            "rows": rows, "fail_rows": fails, "note": note}

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
    with text_open(path) as f:
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

BUILD_NAME_RE = re.compile(r'cmake-build-([^/"\s]*)')
EXE_RULE_RE = re.compile(rb"CXX_EXECUTABLE_LINKER__(\w+)")
CONFIG_SUFFIX_RE = re.compile(r"_(Debug|Release|RelWithDebInfo|MinSizeRel)?$")

def act_build_targets(a):
    """What compile_all.sh can build, for the --target / --exec pickers --
    read, never built nor configured here:
      configs  the cmake-build-<name> names written in compile_all.sh and
               compile_all_gnu.sh (as known_build_names does);
      dirs     the configured build dirs <build_rep>/cmake-build-<name>/<arch>
               whose <arch> ends with the one asked (aarch64, x86_64), with
               the executables of their build.ninja (the rules
               CXX_EXECUTABLE_LINKER__<target>_<Config>, as
               __ninja_target_names reads them through ninja)."""
    repo = os.path.expanduser(a[0])
    build_rep = os.path.expanduser(os.path.expandvars(a[1])) if len(a) > 1 and a[1] else repo
    arch = a[2] if len(a) > 2 else ""
    configs = set()
    for f in ("scripts/compile_all.sh", "scripts/compile_all_gnu.sh"):
        text = read(os.path.join(repo, f), "")
        for name in BUILD_NAME_RE.findall(text):
            name = name.replace("${ICF_SUFFIX}", "")
            if "$" not in name and name:
                configs.add(name)
    dirs = []
    for d in sorted(glob.glob(os.path.join(build_rep, "cmake-build-*", "*"))):
        sub = os.path.basename(d)
        if not os.path.isdir(d) or (arch and not sub.endswith(arch)):
            continue
        ninja = os.path.join(d, "build.ninja")
        execs = set()
        try:
            with open(ninja, "rb") as f:
                for m in EXE_RULE_RE.finditer(f.read()):
                    execs.add(CONFIG_SUFFIX_RE.sub("", m.group(1).decode()))
            mtime = os.path.getmtime(ninja)
        except (IOError, OSError):
            mtime = None
        dirs.append({"name": os.path.basename(os.path.dirname(d))[len("cmake-build-"):], "arch": sub,
                     "configured": mtime is not None, "mtime": mtime, "execs": sorted(execs)})
    print(json.dumps({"configs": sorted(configs), "dirs": dirs, "build_rep": build_rep}))

ACTIONS = {"build_targets": act_build_targets, "check": act_check, "options": act_options, "launch": act_launch, "list": act_list,
           "tail": act_tail, "stop": act_stop, "pack": act_pack, "summary": act_summary,
           "points": act_points, "delete": act_delete}
if __name__ == "__main__":
    ACTIONS[sys.argv[1]](sys.argv[2:])
