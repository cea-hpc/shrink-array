# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Run on the cluster by shrink_gui: `python3 - <action> <args...>`, this file on stdin.

Nothing about the site is written here: the batch commands (submit, cancel,
QoS table, directive lines) come with each call, from the site's TOML file
(sitecfg.py) -- only Slurm's squeue and its start conditions are assumed.

Nothing is installed on the cluster: every ssh call sends this file again, so it is
always the laptop's version. It must stay Python 3.6 compatible (the system
python3 of a RHEL 8 login node) and import nothing outside the standard
library. Imported by the tests, it runs nothing (see the end).
"""
import base64, glob, io, json, os, re, shlex, signal, socket, subprocess, sys, tarfile

# The variants were named chunk / chunk2 / chunk2a (tbb_chunk...) until they took
# the name of their namespace: the summaries, logs and executables of before read
# as the new names until 1.0. The table of bench_runner/settings.py, copied: this
# file imports nothing of the repository.
LEGACY_VARIANTS = {
    "chunk": "zero", "chunk2": "zero_one_value", "chunk2a": "zero_any",
    "tbb_chunk": "tbb_zero", "tbb_chunk2": "tbb_zero_one_value", "tbb_chunk2a": "tbb_zero_any",
}

def normalize_variant(label):
    """chunk2a -> zero_any, chunk2a@256 -> zero_any@256, chunk_int64@dyn ->
    zero_int64@dyn; a label that is not an old one comes back as it is."""
    base, at, size = label.partition("@")
    suffix = "_int64" if base.endswith("_int64") else ""
    new = LEGACY_VARIANTS.get(base[:len(base) - len(suffix)])
    return new + suffix + at + size if new else label

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

def after_runs(after):
    """The runs a start condition waits for: after["runs"] (the jobs of a
    submission over several nodes), else after["run"]."""
    return after.get("runs") or [after["run"]]

def pending_detail(d, st):
    """A waiting run whose start condition can no longer be met: a run it
    waits for failed (Slurm keeps such a job pending, DependencyNeverSatisfied)."""
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}") or "{}")
    after = meta.get("after")
    if after and after.get("cond", "ok") == "ok":
        for rid in after_runs(after):
            code = (read(os.path.join(os.path.dirname(d), rid, "exit_code")) or "").strip()
            if code and code != "0":
                return "en attente sans fin (%s a échoué : à arrêter)" % rid
    return st

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
            st = STATES.get(q[jobid], q[jobid].lower())
            if q[jobid] == "PENDING":
                return pending_detail(d, st), None
            return st, None
        if stopped:
            return "arrêté", None
        return "terminé sans code (limite de temps ? voir le log)", None
    pid = read(os.path.join(d, "pid"))
    if not pid:
        # submit.log without a jobid: the batch system refused the job
        if os.path.exists(os.path.join(d, "submit.log")):
            return "soumission refusée : %s" % (last_line(os.path.join(d, "submit.log")) or "?"), None
        return "inconnu", None
    host = (read(os.path.join(d, "host")) or "").strip()
    if host != socket.gethostname():
        return "en cours ? (lancé depuis %s)" % host, None
    if alive(int(pid)):
        if os.path.exists(os.path.join(d, "waiting")):
            return pending_detail(d, "en attente"), None
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
    info = {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "",
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "upstream": git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") or ""}
    # Ahead / behind the upstream, as of the last fetch (a pull fetches).
    counts = git("rev-list", "--left-right", "--count", "HEAD...@{u}") if info["upstream"] else None
    if counts and len(counts.split()) == 2:
        info["ahead"], info["behind"] = (int(x) for x in counts.split())
    return info

def git_run(repo, args, timeout):
    """(code, output) of `git -C repo args`: no prompt (a password asked on a
    closed terminal would hang the ssh call), stdin closed, a time limit."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND="ssh -o BatchMode=yes")
    try:
        p = subprocess.Popen(["git", "-C", repo] + list(args), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except OSError as e:
        return 127, str(e)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return 124, out.decode("utf-8", "replace") + "\n(time limit reached: %d s)" % timeout
    return p.returncode, out.decode("utf-8", "replace")

def act_git_pull(a):
    """git pull (a[1] == "merge") or git pull --rebase (a[1] == "rebase") in
    the repository a[0]. A pull that stops half-way (conflict) is aborted:
    the repository is never left mid-merge or mid-rebase on the cluster."""
    repo = os.path.expanduser(a[0])
    before = git_info(repo)
    if not before:
        print(json.dumps({"error": "pas de dépôt git dans %s" % a[0]}))
        return
    rebase = len(a) > 1 and a[1] == "rebase"
    code, out = git_run(repo, ["pull", "--rebase" if rebase else "--no-rebase"], 300)
    aborted = None
    if code != 0:
        gitdir = cmd_out(["git", "-C", repo, "rev-parse", "--git-dir"])
        gitdir = os.path.join(repo, gitdir.strip()) if gitdir else os.path.join(repo, ".git")
        if os.path.isdir(os.path.join(gitdir, "rebase-merge")) or os.path.isdir(os.path.join(gitdir, "rebase-apply")):
            git_run(repo, ["rebase", "--abort"], 60)
            aborted = "rebase"
        elif os.path.exists(os.path.join(gitdir, "MERGE_HEAD")):
            git_run(repo, ["merge", "--abort"], 60)
            aborted = "merge"
    print(json.dumps({"ok": code == 0, "output": out[-8000:], "aborted": aborted,
                      "before": before["commit"], "git": git_info(repo)}))

def mem_gb():
    """The machine's memory in GiB (MemTotal of /proc/meminfo), or None."""
    m = re.search(r"^MemTotal:\s+(\d+) kB", read("/proc/meminfo", "") or "", re.M)
    return round(int(m.group(1)) / 1048576.0, 1) if m else None

def git_branches(repo):
    """(local, remote) branch names of the repository: refs/heads, and the
    remote-tracking branches (origin/x) without their symbolic HEAD."""
    def refs(prefix):
        code, out = git_run(repo, ["for-each-ref", "--format=%(refname)", prefix], 60)
        if code != 0:
            return []
        names = [l.strip()[len(prefix) + 1:] for l in out.splitlines() if l.strip().startswith(prefix + "/")]
        return [n for n in names if n and not n.endswith("/HEAD")]
    return refs("refs/heads"), refs("refs/remotes")

def act_git_branches(a):
    """The branches of the repository a[0], after a `git fetch --prune`: a
    failed fetch (no network, no credential) still lists what the last fetch
    knew, and says so. remote: the remote-tracking branches with no local
    branch of the same name (origin/x) -- a switch creates one for them."""
    repo = os.path.expanduser(a[0])
    if not git_info(repo):
        print(json.dumps({"error": "pas de dépôt git dans %s" % a[0]}))
        return
    code, out = git_run(repo, ["fetch", "--prune", "--quiet"], 120)
    local, remotes = git_branches(repo)
    remote_only = [r for r in remotes if r.split("/", 1)[-1] not in local]
    print(json.dumps({"local": local, "remote": remote_only, "git": git_info(repo),
                      "fetch_error": None if code == 0 else (out.strip()[-2000:] or "code %d" % code)}))

def act_git_switch(a):
    """Check out the branch a[1] in the repository a[0]: a local branch, or a
    remote-tracking one (origin/x), for which the local branch x is taken if
    it exists, created tracking origin/x otherwise. Refused when the working
    tree has changes: a checkout would carry them over to the other branch,
    or stop half-way."""
    repo = os.path.expanduser(a[0])
    before = git_info(repo)
    if not before:
        print(json.dumps({"error": "pas de dépôt git dans %s" % a[0]}))
        return
    target = a[1] if len(a) > 1 else ""
    local, remotes = git_branches(repo)
    if target.startswith("-") or (target not in local and target not in remotes):
        print(json.dumps({"ok": False, "output": "branche inconnue : %s" % target, "before": before["branch"],
                          "git": before}))
        return
    if before["dirty"]:
        print(json.dumps({"ok": False, "before": before["branch"], "git": before,
                          "output": "modifications non commitées dans %s : commitez-les, ou mettez-les de côté "
                                    "(git stash), avant de changer de branche" % a[0]}))
        return
    if target in local:
        args = ["checkout", target]
    else:
        name = target.split("/", 1)[-1]
        args = ["checkout", name] if name in local else ["checkout", "--track", target]
    code, out = git_run(repo, args, 120)
    print(json.dumps({"ok": code == 0, "output": out[-8000:], "before": before["branch"], "git": git_info(repo)}))

def site_arg(a, i):
    """The site's batch table, base64 JSON argument @p i (sitecfg.Site.batch),
    or {} when not given."""
    return json.loads(base64.b64decode(a[i]).decode()) if len(a) > i and a[i] else {}

def act_check(a):
    batch = site_arg(a, 1)
    submit = shlex.split(batch.get("submit", ""))[:1]
    r = {"host": socket.gethostname(), "python": sys.version.split()[0],
         "msub": bool(submit) and cmd_out(["which", submit[0]]) is not None,
         # the QoS limits (parsed by the GUI)
         "mqinfo": cmd_out(shlex.split(batch["qos_info"])) if batch.get("qos_info") else None,
         # what this machine is, for the local mode's "partition"
         "machine": os.uname()[4], "cpus": os.cpu_count(), "mem_gb": mem_gb()}
    if a and a[0]:
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

def act_validate(a):
    """The command line a[2] (base64 JSON list of arguments) checked by the
    script's own parsing, before a submission: {"ok": true}, {"refused":
    message} (what the job would have stopped on: an argparse error or a
    check of the script's), or {"skipped": why} (the check could not run: the
    submission goes on). A script with a module-level parse_args(argv)
    function (thread_speedup.py) is checked by it -- its own checks included --
    without running main; otherwise the script runs up to its parse_args()
    call, as for "options"."""
    import argparse, inspect, runpy
    path = os.path.abspath(os.path.join(os.path.expanduser(a[0]), a[1]))
    argv = [os.path.expandvars(x) for x in json.loads(base64.b64decode(a[2]).decode())]
    real_out, real_err = sys.stdout, sys.stderr
    captured = io.StringIO()

    class Done(Exception):
        pass

    orig_pa = argparse.ArgumentParser.parse_args
    orig_pka = argparse.ArgumentParser.parse_known_args
    state = {"depth": 0, "stop": True}

    def hooked(orig):
        def f(self, args=None, namespace=None):
            if state["depth"]:          # parse_args -> parse_known_args, subparsers
                return orig(self, args, namespace)
            state["depth"] += 1
            try:
                r = orig(self, args, namespace)
            finally:
                state["depth"] -= 1
            if state["stop"]:
                raise Done()
            return r
        return f

    def answer(**kw):
        sys.stdout, sys.stderr = real_out, real_err
        print(json.dumps(kw))

    def refusal(e):
        if isinstance(e.code, str):     # sys.exit("message")
            return e.code
        lines = [ln.strip() for ln in captured.getvalue().splitlines() if ln.strip()]
        err = [ln for ln in lines if ": error: " in ln]
        return (err or lines or ["code de sortie %s" % e.code])[-1]

    argparse.ArgumentParser.parse_args = hooked(orig_pa)
    argparse.ArgumentParser.parse_known_args = hooked(orig_pka)
    sys.stdout = sys.stderr = captured
    try:
        os.chdir(os.path.dirname(path))
        sys.path.insert(0, os.path.dirname(path))
        sys.argv = [path] + argv
        g = runpy.run_path(path, run_name="__shrink_gui_validate__")
        fn = g.get("parse_args")
        if inspect.isfunction(fn) and fn.__code__.co_argcount >= 1:
            state["stop"] = False
            fn(list(argv))
            return answer(ok=True)
        runpy.run_path(path, run_name="__main__")
        return answer(skipped="%s n'a pas appelé parse_args" % a[1])
    except Done:
        return answer(ok=True)
    except SystemExit as e:
        if e.code in (0, None):
            return answer(ok=True)
        return answer(refused=refusal(e))
    except BaseException as e:
        return answer(skipped="%s: %s" % (type(e).__name__, e))
    finally:
        argparse.ArgumentParser.parse_args = orig_pa
        argparse.ArgumentParser.parse_known_args = orig_pka

def after_extra(base, spec):
    """(Slurm options, error): the start conditions of a batch
    run -- after the runs of spec["after"] = {"run", "runs", "cond": "ok"|"any"}
    (--dependency=afterok|afterany on their jobs, after_runs), not before
    spec["delay_s"] seconds (--begin=now+N). A run already finished is decided
    here: Slurm forgets a finished job after a few minutes, a dependency on it
    would be refused."""
    extra = []
    after = spec.get("after")
    if after:
        jobids = []
        for rid in after_runs(after):
            ad = rdir(base, rid)
            if not os.path.isfile(os.path.join(ad, "meta.json")):
                return None, "le run %s n'existe pas" % rid
            code = (read(os.path.join(ad, "exit_code")) or "").strip()
            jobid = (read(os.path.join(ad, "jobid")) or "").strip()
            if code:
                if after.get("cond", "ok") == "ok" and code != "0":
                    return None, "le run %s a échoué (code %s) : rien soumis" % (rid, code)
            elif jobid:
                jobids.append(jobid)
            else:
                return None, "le run %s n'a pas de job batch à attendre" % rid
        if jobids:
            extra.append("--dependency=%s:%s" % ("afterok" if after.get("cond", "ok") == "ok" else "afterany",
                                                 ":".join(jobids)))
    if spec.get("delay_s"):
        extra.append("--begin=now+%d" % int(spec["delay_s"]))
    return extra, None

def hms(seconds):
    """Slurm's [D-]HH:MM:SS."""
    days, r = divmod(int(seconds), 86400)
    return ("%d-" % days if days else "") + "%02d:%02d:%02d" % (r // 3600, r % 3600 // 60, r % 60)

def job_header(batch, name, seconds, log, qos, slurm):
    """The directive lines of the batch script from the site's templates
    (sitecfg: [batch] name, time, output, error, qos, slurm); an empty
    template, or no value, leaves its line out."""
    def line(key, **fields):
        tpl = batch.get(key) or ""
        return "%s %s\n" % (batch["directive"], tpl.format(**fields)) if tpl else ""
    out = (line("name", name=name)
           + line("time", seconds=int(seconds), minutes=(int(seconds) + 59) // 60, hms=hms(seconds))
           + line("output", log=log) + line("error", log=log)
           + (line("qos", qos=qos) if qos else ""))
    if slurm:
        tpl = batch.get("slurm") or ""
        if "{option}" in tpl:
            out += "".join(line("slurm", option=o) for o in slurm)
        else:
            out += line("slurm", options=" ".join(slurm))
    return out

def act_launch(a):
    base, rid, spec = a[0], a[1], json.loads(base64.b64decode(a[2]).decode())
    d = rdir(base, rid)
    extra = []
    if spec["mode"] == "msub":
        extra, err = after_extra(base, spec)
        if err:
            print(json.dumps({"error": err}))
            return
    os.makedirs(d)
    write(os.path.join(d, "run.sh"), spec["run.sh"])
    # The code the run is submitted on; run.sh records the one it STARTS on
    # (git_start): a pending job can start after a git pull.
    spec["meta"]["git"] = git_info(spec["meta"].get("repo", "~"))
    write(os.path.join(d, "meta.json"), json.dumps(spec["meta"], indent=1))
    if spec["mode"] == "msub":
        batch = spec["batch"]
        log = os.path.join(d, "out.log")
        job = ("#!/bin/bash\n" + spec["msub"].rstrip("\n") + "\n"
               + job_header(batch, spec["name"], spec["time_s"], log, spec.get("qos"), extra)
               + 'D=%s\n' % shlex.quote(d)
               + ('echo "[gui] launcher: %s"\n' % spec["launcher"] if spec.get("launcher") else "")
               + '%sbash "$D/run.sh"\ncode=$?\necho "$code" > "$D/exit_code"\n'
               % (spec["launcher"] + " " if spec.get("launcher") else "")
               + 'echo "[gui] end $(date \'+%F %T\')  exit code $code"\n')
        write(os.path.join(d, "job.sh"), job)
        submit = shlex.split(batch["submit"])
        # The command as run, for the window (shown as soon as the job is
        # submitted, while run.sh's "[gui] command:" line waits for the node)
        # and as the first line of submit.log.
        launched = " ".join(shlex.quote(x) for x in submit + ["job.sh"])
        try:
            p = subprocess.Popen(submit + ["job.sh"], cwd=d, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            out = p.communicate()[0]
        except OSError as e:
            print(json.dumps({"error": "%s : %s" % (submit[0], e)}))
            return
        write(os.path.join(d, "submit.log"), "$ %s\n%s" % (launched, out))
        ids = re.findall(batch.get("jobid_re") or r"\b(\d{2,})\b", out)
        if p.returncode != 0 or not ids:
            print(json.dumps({"error": "%s (code %s) : %s" % (submit[0], p.returncode, out.strip()[-1500:])}))
            return
        write(os.path.join(d, "jobid"), ids[-1])
        print(json.dumps({"run_dir": d, "jobid": ids[-1],
                          "launched": "cd %s && %s" % (shlex.quote(d), launched),
                          "submit_out": out.strip()}))
    else:
        write(os.path.join(d, "launch.sh"), spec["launch.sh"])
        out = open(os.path.join(d, "out.log"), "wb")
        p = subprocess.Popen(["bash", os.path.join(d, "launch.sh")], cwd=d, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        write(os.path.join(d, "pid"), str(p.pid))
        write(os.path.join(d, "host"), socket.gethostname())
        # launch.sh itself prints the detached command ("[gui] ccc_mprun ...")
        # as its first line of out.log.
        print(json.dumps({"run_dir": d, "pid": p.pid,
                          "launched": "cd %s && bash launch.sh" % shlex.quote(d)}))

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

def progress_of(path, meta, ev=None):
    """(done, total, eta_s[, phase]): bench_and_test.py's events.jsonl when
    there is one (@p ev, already read), else the log with the program's own
    regexes."""
    if ev is None:
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

def errors_from_events(ev):
    """[count, last error's first line] of a bench_and_test.py run: its end
    record once there, else the errors logged so far (an "error" record each,
    or phase_end counts before bench_runner wrote them), or None."""
    if not ev:
        return None
    last = None
    n_ev = n_phase = 0
    for e in ev:
        k = e.get("kind")
        if k == "error":
            n_ev += 1
            last = e.get("summary") or e.get("kind")
        elif k == "phase_end":
            n_phase += e.get("errors", 0) or 0
        elif k == "end":
            return [e.get("errors", 0) or 0, last]
    return [max(n_ev, n_phase), last]

def errors_from_csv(d, meta):
    """[failed runs, None] of multi_launch_stress.py (returncode != 0 in its
    summary CSV), or None."""
    import csv
    od = out_dir(d, meta)
    p = os.path.join(od, meta.get("summary_file") or "") if od else None
    if not p or not os.path.isfile(p):
        return None
    n = 0
    try:
        with text_open(p) as f:
            rows = csv.reader(f)
            head = next(rows, [])
            if "returncode" not in head:
                return None
            i = head.index("returncode")
            for r in rows:
                n += len(r) > i and r[i].strip() not in ("0", "")
    except (IOError, OSError):
        return None
    return [n, None]

def errors_of(d, meta, st, ev):
    """The Erreurs column: errors_from_events, else (multi_launch_stress.py)
    errors_from_csv -- kept in the run directory once the run is over, the
    CSV no longer changes."""
    r = errors_from_events(ev)
    if r is not None or meta.get("summary") != "multi_launch":
        return r
    cache = os.path.join(d, "errors_count.json")
    final = st not in ("en cours", "en attente", "démarrage") and not st.startswith("en attente")
    if final:
        c = read(cache)
        if c:
            try:
                return json.loads(c)
            except ValueError:
                pass
    r = errors_from_csv(d, meta)
    if final and r is not None:
        try:
            write(cache, json.dumps(r))
        except (IOError, OSError):
            pass
    return r

_START_RE = re.compile(r"\[gui\] node \S+\s+start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")

def run_times(d, final):
    """[started, ended] of run dir @p d, in seconds since the epoch by the
    cluster's clock (None when unknown). Started: the start_time file run.sh
    writes, else (a run older than that file) the "[gui] node ... start" line
    at the top of out.log. Ended, once the run has an exit code: when its
    exit_code was written; a run over without one (@p final: a job that left
    the queue at its time limit, a stopped run): the last write to out.log."""
    import time as _time
    started = None
    s = (read(os.path.join(d, "start_time")) or "").strip()
    if s.isdigit():
        started = int(s)
    else:
        try:
            with open(os.path.join(d, "out.log"), "rb") as f:
                head = f.read(16384).decode("utf-8", "replace")
            m = _START_RE.search(head)
            if m:
                started = int(_time.mktime(_time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")))
        except (IOError, OSError, ValueError, OverflowError):
            pass
    if started is None:
        return [None, None]
    ended = None
    for name in ("exit_code", "out.log"):
        p = os.path.join(d, name)
        if os.path.isfile(p) and (name == "exit_code" or final):
            ended = os.path.getmtime(p)
            break
    return [started, max(started, int(ended)) if ended is not None else None]

def time_now():
    import time as _time
    return _time.time()

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
            ev = read_events(d)
            runs.append({"id": rid, "meta": meta, "status": st, "code": code,
                         "errors": errors_of(d, meta, st, ev),
                         "jobid": (read(os.path.join(d, "jobid")) or "").strip(),
                         "progress": progress_of(os.path.join(d, "out.log"), meta, ev),
                         "git_start": (read(os.path.join(d, "git_start")) or "").strip(),
                         "times": run_times(d, st in ("fini", "arrêté", "interrompu")
                                            or st.startswith("terminé")),
                         "last": last_line(os.path.join(d, "out.log"))})
    # "now" by the cluster's clock: the window counts a running run's time
    # from it, whatever the offset of the laptop's clock.
    print(json.dumps({"host": socket.gethostname(), "now": int(time_now()), "runs": runs}))

TAIL_START = 2 * 1024 * 1024   # a first display: the end of the log only
TAIL_CHUNK = 4 * 1024 * 1024   # then what was appended, by blocks of this size

def act_tail(a):
    """The log from byte a[2] on. a[2] < 0: a first display -- the last
    TAIL_START bytes from a line start, "skipped" telling how many came
    before (the view keeps 20000 lines: reading a long log from its start
    cost one ssh round trip per block for lines it then dropped)."""
    d, off = rdir(a[0], a[1]), int(a[2])
    p = os.path.join(d, "out.log")
    data, skipped = b"", 0
    if os.path.exists(p):          # a pending batch job has no log yet
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            if off < 0:
                off = max(0, size - TAIL_START)
                if off:
                    f.seek(off)
                    head = f.read(64 * 1024)
                    nl = head.find(b"\n")
                    off += nl + 1 if nl >= 0 else 0
                skipped = off
            elif off > size:
                off = 0
            f.seek(off)
            data = f.read(TAIL_CHUNK)
        more = off + len(data) < size
    else:
        off, more = 0, False
    print(json.dumps({"offset": off + len(data), "data": data.decode("utf-8", "replace"),
                      "skipped": skipped, "more": more}))

def act_stop(a):
    d = rdir(a[0], a[1])
    jobid = (read(os.path.join(d, "jobid")) or "").strip()
    if jobid:
        write(os.path.join(d, "stopped"), "")
        cancel = site_arg(a, 2).get("cancel") or ["scancel"]
        out = None
        for c in cancel:
            out = cmd_out(shlex.split(c) + [jobid])
            if out is not None:
                break
        if out is None:
            print(json.dumps({"error": "%s : le job %s n'a pas pu être annulé" % (" / ".join(cancel), jobid)}))
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

def own_copy(d, name):
    """The run's own copy of output file @p name, kept in its directory when
    it ended (bench_and_test.py: $CHUNK_BENCH_KEEP_DIR, report.py), or None.
    It wins over the shared output directory, which later runs rewrite."""
    if not name or os.sep in name or name in (".", ".."):
        return None
    p = os.path.join(d, name)
    return p if os.path.isfile(p) else None

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
    path = own_copy(d, name) or (os.path.join(od, name) if od else None)
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

def is_speedup_chart(meta):
    """The run is drawn as thread_speedup.py's (meta "chart", from PROGRAMS; an old
    run under its former name, omp_speedup.py, too)."""
    return meta.get("chart") == "speedup" or meta.get("program") in ("thread_speedup.py", "omp_speedup.py")

def speedup_points(d, meta):
    """The Graphique tab of thread_speedup.py: its summary.csv, in the output
    directory of its own the log announces. {"execs", "rows": [exe, threads,
    best s, speedup or None, efficiency or None, peak Mo or None, ok runs],
    "info": its meta.json (target, multiply, mode, metric...)}."""
    import csv
    od = out_dir(d, meta)
    if not od:
        return {"note": "pas encore de dossier de sortie annoncé dans le log"}
    od = os.path.expanduser(od)
    path = os.path.join(od, meta.get("summary_file") or "summary.csv")
    if not os.path.isfile(path):
        return {"note": "pas encore de summary.csv"}
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    execs, rows = [], []
    with text_open(path) as f:
        rd = csv.reader(f)
        col = {h: i for i, h in enumerate(next(rd, []))}
        for r in rd:
            try:
                e, th = r[col["exe"]], int(r[col["threads"]])
            except (KeyError, ValueError, IndexError):
                continue
            get = lambda k: num(r[col[k]]) if k in col and len(r) > col[k] else None
            if e not in execs:
                execs.append(e)
            rows.append([e, th, get("best_s"), get("speedup"), get("efficiency"), get("peak_mem_mo"),
                         int(get("ok_runs") or 0)])
    info = {}
    try:
        info = json.loads(read(os.path.join(od, "meta.json"), "{}") or "{}")
    except ValueError:
        pass
    return {"kind": "speedup", "execs": execs, "rows": rows,
            "info": {k: info.get(k) for k in ("target", "multiply", "mode", "metric", "host", "cpu",
                                               "threads", "reps")}}

def is_bench_chart(meta):
    """The run is drawn as bench_and_test.py's (meta "chart", set from
    PROGRAMS; the runs submitted before it by the program's name)."""
    return meta.get("chart") == "bench" or meta.get("program") == "bench_and_test.py"

def bench_points(d, meta):
    """The Graphique tab of bench_and_test.py: bench_summary.csv, the run's
    own copy (kept when it ended) or, before that, the shared one -- "own"
    says which. {"configs", "variants", "rows": [config, variant, threads,
    avg s, peak Mo or None, speedup or None]}."""
    import csv
    name = meta.get("summary_file") or "bench_summary.csv"
    path = own_copy(d, name)
    own = path is not None
    if not own:
        od = out_dir(d, meta)
        path = os.path.join(os.path.expanduser(od), name) if od else None
    if not path or not os.path.isfile(path):
        return {"note": "pas encore de %s (écrit à la fin du run)" % name}
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    configs, variants, rows = [], [], []
    with text_open(path) as f:
        rd = csv.reader(f)
        head = next(rd, [])
        col = {h: i for i, h in enumerate(head)}
        for r in rd:
            try:
                c, v, th, avg = r[col["Config"]], r[col["Variant"]], int(r[col["Threads"]]), float(r[col["AvgTime_s"]])
                v = normalize_variant(v)        # a summary written before the rename
            except (KeyError, ValueError, IndexError):
                continue
            mem = num(r[col["PeakMem_Mo"]]) if "PeakMem_Mo" in col and len(r) > col["PeakMem_Mo"] else None
            sp = (num(r[col["Speedup_vs_baseline"]])
                  if "Speedup_vs_baseline" in col and len(r) > col["Speedup_vs_baseline"] else None)
            if c not in configs:
                configs.append(c)
            if v not in variants:
                variants.append(v)
            rows.append([c, v, th, avg, mem, sp])
    return {"kind": "bench", "configs": configs, "variants": variants, "rows": rows, "own": own,
            "mtime": os.path.getmtime(path)}

def act_points(a):
    """The Graphique tab: one point per run of timings_summary.csv (the last
    30000), names factored out: {"builds", "exes", "rows": [run, b, e, threads,
    seconds, returncode]}; bench_and_test.py: bench_points."""
    import csv
    d = rdir(a[0], a[1])
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}"))
    if is_speedup_chart(meta):
        print(json.dumps(speedup_points(d, meta)))
        return
    if is_bench_chart(meta):
        print(json.dumps(bench_points(d, meta)))
        return
    if meta.get("summary") != "multi_launch":
        print(json.dumps({"note": "graphique disponible pour multi_launch_stress.py, bench_and_test.py et thread_speedup.py seulement"}))
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

def _same_dir(a, b):
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (TypeError, ValueError):
        return False

def runs_sharing(base, rid, od):
    """(id, dir, meta) of every other run of @p base whose log announces the
    output directory @p od."""
    for other in sorted(os.listdir(base)):
        od2 = os.path.join(base, other)
        m2 = read(os.path.join(od2, "meta.json"))
        if other == rid or m2 is None:
            continue
        m2 = json.loads(m2)
        if _same_dir(out_dir(od2, m2) or "", od):
            yield other, od2, m2

def newer_writer(base, rid, meta, od):
    """The first other run announcing @p od that is still active or newer
    than run @p rid (the shared files are then its own, rewritten after
    ours), or None."""
    q = None
    for other, od2, m2 in runs_sharing(base, rid, od):
        if q is None:
            q = queue_states()
        st = status(od2, q)[0]
        if st in ("en cours", "en attente", "démarrage") or st.startswith("en attente") \
                or m2.get("created", "") > meta.get("created", ""):
            return other
    return None

def outputs_to_delete(base, rid, d, meta, own):
    """(paths to remove, error): the program outputs of run @p rid, as
    @p own says ("dir": the output directory its log announces; a list: those
    entries of that directory). Refused when another run announces the same
    directory -- any other one for "dir", for a list one still active or
    newer (the files are then its own, rewritten after ours)."""
    od = out_dir(d, meta)
    if not od:
        return None, "pas de dossier de sortie annoncé dans le log de %s" % rid
    od = os.path.expanduser(od)
    real = os.path.realpath(od)
    home = os.path.realpath(os.path.expanduser("~"))
    if (not os.path.isabs(od) or real in ("/", home) or real.count(os.sep) < 3
            or os.path.realpath(os.path.join(home, "shrink-array")).startswith(real)):
        return None, "dossier de sortie refusé : %s" % od
    if not os.path.isdir(od):
        return [], None                        # already gone: nothing to do
    if own == "dir":
        for other, _, _ in runs_sharing(base, rid, od):
            return None, "%s utilise aussi %s : rien supprimé" % (other, od)
    else:
        other = newer_writer(base, rid, meta, od)
        if other:
            return None, "%s, plus récent ou actif, écrit aussi dans %s : rien supprimé" % (other, od)
    if own == "dir":
        return [od], None
    # A shared directory: what this run wrote, i.e. after it was submitted
    # (meta.json) -- run_logs/ also holds the older runs' logs, kept.
    try:
        t0 = os.path.getmtime(os.path.join(d, "meta.json"))
    except OSError:
        return None, "date de soumission de %s illisible : rien supprimé" % rid
    out = []
    for name in own:
        if os.sep in name or name in ("", ".", ".."):
            continue
        p = os.path.join(od, name)
        if os.path.isdir(p) and not os.path.islink(p):
            for top, _dirs, files in os.walk(p):
                for f in files:
                    fp = os.path.join(top, f)
                    if written_since(fp, t0):
                        out.append(fp)
        elif os.path.lexists(p) and written_since(p, t0):
            out.append(p)
    return out, None

def written_since(p, t0):
    try:
        return os.lstat(p).st_mtime >= t0
    except OSError:
        return False

def prune_empty(root):
    """Remove the empty directories under @p root, @p root included."""
    for top, _dirs, _files in os.walk(root, topdown=False):
        try:
            os.rmdir(top)
        except OSError:
            pass

def act_delete(a):
    """Remove one run directory and, when a[2] asks for it ({"own": ...},
    base64 JSON), the program outputs that are the run's own
    (outputs_to_delete). Everything is checked before anything is removed."""
    opts = json.loads(base64.b64decode(a[2]).decode()) if len(a) > 2 and a[2] else {}
    print(json.dumps(delete_run(rdir(a[0]), a[1], opts, keep_others=bool(opts.get("keep_others")))))

def act_delete_many(a):
    """delete_run for each {"id", "own"?} of a[1] (base64 JSON list), in one
    call, oldest first: {"results": {id: {"ok", "removed", "kept"?} or
    {"error"}}}. Outputs that are not the run's own (none announced, a shared
    directory a newer run rewrote) are kept -- "kept" says why -- and the run
    directory goes all the same: one such run must not keep a whole selection.
    A run that cannot be deleted leaves the others alone."""
    base = rdir(a[0])
    out = {}
    items = sorted(json.loads(base64.b64decode(a[1]).decode()), key=lambda it: str(it.get("id", "")))
    for item in items:
        rid = str(item.get("id", ""))
        try:
            out[rid] = delete_run(base, rid, {"own": item["own"]} if item.get("own") else {},
                                  keep_others=True)
        except Exception as e:          # one bad run must not stop the others
            out[rid] = {"error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps({"results": out}))

def delete_run(base, rid, opts, keep_others=False):
    """Remove run @p rid (and, opts["own"], its own outputs). An output that
    is not its own refuses everything, unless @p keep_others: then it is kept
    ("kept": why) and the run directory goes."""
    import shutil
    if not re.match(r"^[\w.-]+$", rid) or rid in (".", ".."):
        return {"error": "nom de run invalide : %s" % rid}
    d = os.path.join(base, rid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        return {"error": "%s n'est pas un run" % d}
    paths = []
    meta = {}
    kept = None
    if opts.get("own"):
        meta = json.loads(read(os.path.join(d, "meta.json"), "{}") or "{}")
        paths, err = outputs_to_delete(base, rid, d, meta, opts["own"])
        if err and not keep_others:
            return {"error": err}
        if err:
            paths, kept, opts = [], err, {}
    removed = []
    for p in paths:
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                continue
        removed.append(p)
    if isinstance(opts.get("own"), list):
        od = out_dir(d, meta)
        for name in opts["own"]:
            sub = os.path.join(os.path.expanduser(od), name) if od and os.sep not in name else None
            if sub and os.path.isdir(sub) and not os.path.islink(sub):
                prune_empty(sub)      # emptied by this run's files only
    shutil.rmtree(d)
    return dict({"ok": True, "removed": removed}, **({"kept": kept} if kept else {}))

ERR_FILE_MAX = 1024 * 1024          # per file: its end only
ERR_TOTAL_MAX = 4 * 1024 * 1024     # per call, every file together
ERR_FILES_MAX = 100                 # the newest ones (multi_launch_stress: one error_*.txt per failure)
# An entry of bench_errors.log: "CHECK_FAIL: ...", "BASELINE_REF_FAIL: ...",
# its details indented below.
ERR_ENTRY_RE = re.compile(r"^[A-Z][A-Z0-9_]*:", re.M)

def act_errors(a):
    """The Erreurs tab: the error files the program writes in the output
    directory its log announces -- a[2], base64 JSON {"files": [names or
    glob patterns]}, comes from the window's PROGRAMS (an older run gets
    them too). Newest first, each one's end only. "newer": another run,
    active or newer, announcing the same directory (bench_results/<arch> is
    shared and bench_errors.log rewritten by each run): the files may be its."""
    base, rid = rdir(a[0]), a[1]
    d = os.path.join(base, rid)
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}") or "{}")
    opts = json.loads(base64.b64decode(a[2]).decode()) if len(a) > 2 and a[2] else {}
    pats = [p for p in opts.get("files") or [] if p and os.sep not in p and p not in (".", "..")]
    if opts.get("fail_re"):
        # No error file: the log of each failure, named by the run's own
        # output (compile_all.sh: "[FAIL] <build> ... → <its log>").
        logs = []
        for m in re.finditer(opts["fail_re"], read(os.path.join(d, "out.log"), ""), re.M):
            p = os.path.expanduser(m.group(1))
            if os.path.isfile(p) and p not in logs:
                logs.append(p)
        if not logs:
            print(json.dumps({"note": "aucun échec dans le log du run", "files": [], "count": 0}))
            return
        found, od, pats = {p: True for p in logs}, os.path.dirname(logs[0]), []
    elif not pats:
        print(json.dumps({"note": "pas de log d'erreurs pour ce programme", "files": []}))
        return
    else:
        od = out_dir(d, meta)
        if not od:
            print(json.dumps({"note": "pas encore de dossier de sortie annoncé dans le log", "files": []}))
            return
        od = os.path.expanduser(od)
        found = {}
    for pat in pats:
        mine = own_copy(d, pat)
        if mine:                 # the copy the run kept when it ended
            found[mine] = False
            continue
        for p in glob.glob(os.path.join(od, pat)):
            if os.path.isfile(p):
                found[p] = bool(re.search(r"[*?[]", pat))
    def mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0
    paths = sorted(found, key=lambda p: (-mtime(p), p))
    files, budget, count = [], ERR_TOTAL_MAX, 0
    for p in paths[:ERR_FILES_MAX]:
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        n = min(ERR_FILE_MAX, budget)
        if n <= 0:
            break
        data = tail_text(p, n)
        skipped = max(0, size - n)
        if skipped:
            nl = data.find("\n")
            data = data[nl + 1:] if nl >= 0 else data
        budget -= len(data.encode("utf-8", "replace"))
        if data.strip():
            # A file per failure (a pattern): one each; a log: its entries.
            count += 1 if found[p] else max(1, len(ERR_ENTRY_RE.findall(data)))
        files.append({"name": os.path.basename(p), "size": size, "mtime": mtime(p),
                      "data": data, "skipped": skipped})
    shared = [p for p in paths if os.path.dirname(p) != d]
    if opts.get("cases") and not own_copy(d, opts["cases"]):
        shared.append(opts["cases"])
    out = {"dir": od if shared or not paths else d, "files": files, "count": count,
           "total_files": len(paths),
           "newer": newer_writer(base, rid, meta, od) if shared and not opts.get("fail_re") else None}
    if not paths:
        out["note"] = "aucun fichier %s dans %s" % (", ".join(pats), od)
    if opts.get("cases"):
        out["cases"] = failed_cases(od, opts["cases"], d)
    if opts.get("full_logs"):
        out["logs"] = failure_logs(base, rid, d, meta, od, opts["full_logs"])
    print(json.dumps(out))

def inside(od, rel):
    """os.path.join(od, rel) when it stays inside the output directory @p od
    (a name read in a file, or sent by the window), else None."""
    if not rel:
        return None
    p = os.path.realpath(os.path.join(od, rel))
    root = os.path.realpath(od)
    return p if p.startswith(root + os.sep) else None

def failed_cases(od, name, d=None):
    """The failures failed_cases.jsonl records (bench_runner/runs.py), one per
    entry of bench_errors.log: its first line, whether --rerun-failed can
    replay it, and the complete output of its failed runs (relative to od)."""
    out = []
    p = own_copy(d, name) or inside(od, name)
    if not p or not os.path.isfile(p):
        return out
    with text_open(p) as f:
        for ln in f:
            try:
                c = json.loads(ln)
            except ValueError:
                continue
            runs = c.get("runs") or []
            logs = []
            for r in runs:
                lp = inside(od, r.get("log"))        # absolute as bench_runner writes it
                if lp:
                    logs.append(os.path.relpath(lp, os.path.realpath(od)))
            out.append({"kind": c.get("kind", ""), "summary": c.get("summary", ""),
                        "replayable": bool(runs), "logs": logs})
    return out

def run_started(d):
    """When run dir @p d started, by the cluster's clock: its "start" event,
    else None."""
    for e in read_events(d):
        if e.get("kind") == "start":
            return e.get("t")
    return None

def failure_logs(base, rid, d, meta, od, sub):
    """The full logs of FAILED executions (bench_runner names them
    <binary>__omp<N>__<idx>__<status>.log, status ok / rc<code> / timeout) in
    od/@p sub written during this run: after its launch, before a newer run
    sharing od started. For the runs whose failed_cases.jsonl does not point
    to them (bench_runner before that), newest first."""
    ld = inside(od, sub)
    if not ld or not os.path.isdir(ld):
        return []
    try:
        t0 = os.path.getmtime(os.path.join(d, "meta.json"))
    except OSError:
        return []
    t1 = None
    for _other, od2, _m2 in runs_sharing(base, rid, od):
        t = run_started(od2)
        if t is not None and t > t0 and (t1 is None or t < t1):
            t1 = t
    out = []
    for p in glob.glob(os.path.join(ld, "*.log")):
        if p.endswith("__ok.log"):
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_mtime >= t0 and (t1 is None or st.st_mtime < t1):
            out.append((st.st_mtime, os.path.relpath(p, od)))
    return [name for _t, name in sorted(out, reverse=True)[:ERR_FILES_MAX]]

def act_errfile(a):
    """Some files of the run's output directory, as act_errors sends them:
    a[2], base64 JSON {"paths": [relative to the directory]} -- the complete
    output of a failure. Nothing outside the directory is read."""
    base, rid = rdir(a[0]), a[1]
    d = os.path.join(base, rid)
    meta = json.loads(read(os.path.join(d, "meta.json"), "{}") or "{}")
    opts = json.loads(base64.b64decode(a[2]).decode()) if len(a) > 2 and a[2] else {}
    od = out_dir(d, meta)
    if not od:
        print(json.dumps({"error": "pas de dossier de sortie annoncé dans le log de %s" % rid}))
        return
    od = os.path.expanduser(od)
    files, budget = [], ERR_TOTAL_MAX
    for rel in opts.get("paths") or []:
        p = inside(od, rel)
        if not p or not os.path.isfile(p):
            files.append({"name": rel, "size": 0, "mtime": 0, "data": "", "skipped": 0, "missing": True})
            continue
        size, n = os.path.getsize(p), min(ERR_FILE_MAX * 2, budget)
        data = tail_text(p, n)
        skipped = max(0, size - n)
        if skipped:
            nl = data.find("\n")
            data = data[nl + 1:] if nl >= 0 else data
        budget -= len(data.encode("utf-8", "replace"))
        files.append({"name": rel, "size": size, "mtime": os.path.getmtime(p), "data": data,
                      "skipped": skipped})
    print(json.dumps({"dir": od, "files": files}))

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

ACTIONS = {"build_targets": act_build_targets, "check": act_check, "git_pull": act_git_pull,
           "git_branches": act_git_branches, "git_switch": act_git_switch, "options": act_options,
           "validate": act_validate, "launch": act_launch, "list": act_list,
           "tail": act_tail, "stop": act_stop, "pack": act_pack, "summary": act_summary,
           "points": act_points, "delete": act_delete,
           "delete_many": act_delete_many, "errors": act_errors,
           "errfile": act_errfile}
if __name__ == "__main__":
    ACTIONS[sys.argv[1]](sys.argv[2:])
