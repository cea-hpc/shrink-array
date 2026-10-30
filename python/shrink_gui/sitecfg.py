# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The site: where and how the runs are submitted, read from a TOML file.

The file, first found of:
  - $SHRINK_GUI_SITE (a path);
  - $XDG_CONFIG_HOME/shrink_gui/site.toml (~/.config/shrink_gui/site.toml);
  - the bundled sites/cea_inti.toml (CEA, inti: ccc_msub, #MSUB, ccc_mprun).
sites/ holds the examples, commented key by key: cea_inti.toml and
slurm_example.toml (plain Slurm: sbatch, #SBATCH, no detached mode). Copy one
to ~/.config/shrink_gui/site.toml and edit it.

TOML is read with tomllib (Python >= 3.11), else tomli if installed, else the
small reader below (the subset the examples use: tables, strings, integers,
booleans, arrays). No Qt here.
"""
import os
import pathlib
import re
import shlex

SITES = pathlib.Path(__file__).resolve().with_name("sites")
BUNDLED = SITES / "cea_inti.toml"


class SiteError(Exception):
    """A site file that cannot be used: the message names the file and the key."""


# --------------------------------------------------------------------------
# TOML
# --------------------------------------------------------------------------
_BARE = re.compile(r"[A-Za-z0-9_-]+")
_ESC = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}


class _Reader:
    """The TOML subset of the site files: [tables] (dotted, quoted keys),
    key = value with basic, literal and multi-line strings, integers, floats,
    booleans and arrays (over several lines). Anything else is an error."""

    def __init__(self, text, path):
        self.s, self.i, self.path = text, 0, path

    def fail(self, what):
        line = self.s.count("\n", 0, self.i) + 1
        raise SiteError("%s, ligne %d : %s" % (self.path, line, what))

    def ws(self, newlines=False):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t\r" or (newlines and c == "\n"):
                self.i += 1
            elif c == "#":
                while self.i < len(self.s) and self.s[self.i] != "\n":
                    self.i += 1
            else:
                break

    def key(self):
        if self.s.startswith('"', self.i) or self.s.startswith("'", self.i):
            return self.string()
        m = _BARE.match(self.s, self.i)
        if not m:
            self.fail("clé attendue")
        self.i = m.end()
        return m.group()

    def dotted(self):
        parts = [self.key()]
        self.ws()
        while self.s.startswith(".", self.i):
            self.i += 1
            self.ws()
            parts.append(self.key())
            self.ws()
        return parts

    def string(self):
        s, i = self.s, self.i
        for q in ('"""', "'''"):
            if s.startswith(q, i):
                end = s.find(q, i + 3)
                if end < 0:
                    self.fail("chaîne non terminée")
                while s.startswith(q[0], end + 3):   # up to two quotes just before the closing ones
                    end += 1
                body = s[i + 3:end]
                self.i = end + 3
                if body.startswith("\r\n"):
                    body = body[2:]
                elif body.startswith("\n"):
                    body = body[1:]
                return self.unescape(body, multiline=True) if q == '"""' else body
        q = s[i]
        j = i + 1
        out = []
        while True:
            if j >= len(s) or s[j] == "\n":
                self.fail("chaîne non terminée")
            if s[j] == q:
                break
            if q == '"' and s[j] == "\\":
                out.append(s[j:j + 2])
                j += 2
                continue
            out.append(s[j])
            j += 1
        self.i = j + 1
        body = "".join(out)
        return self.unescape(body) if q == '"' else body

    def unescape(self, body, multiline=False):
        if multiline:   # a backslash at the end of a line joins it to the next non-blank text
            body = re.sub(r"\\[ \t]*\r?\n\s*", "", body)

        def esc(m):
            c = m.group(1)
            if m.group(2) or m.group(3):
                return chr(int(m.group(2) or m.group(3), 16))
            if c in _ESC:
                return _ESC[c]
            self.fail("échappement inconnu \\%s" % c)
        return re.sub(r"\\(u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|.)", esc, body)

    def value(self):
        s = self.s
        c = s[self.i:self.i + 1]
        if c in ('"', "'"):
            return self.string()
        if c == "[":
            self.i += 1
            out = []
            while True:
                self.ws(newlines=True)
                if s.startswith("]", self.i):
                    self.i += 1
                    return out
                out.append(self.value())
                self.ws(newlines=True)
                if s.startswith(",", self.i):
                    self.i += 1
                elif not s.startswith("]", self.i):
                    self.fail("',' ou ']' attendu dans le tableau")
        for word, v in (("true", True), ("false", False)):
            if s.startswith(word, self.i):
                self.i += len(word)
                return v
        m = re.compile(r"[+-]?\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?").match(s, self.i)
        if m:
            self.i = m.end()
            t = m.group().replace("_", "")
            return float(t) if (m.group(1) or m.group(2)) else int(t)
        self.fail("valeur attendue (chaîne, nombre, true/false ou tableau)")

    def parse(self):
        root = {}
        cur = root
        while True:
            self.ws(newlines=True)
            if self.i >= len(self.s):
                return root
            if self.s.startswith("[", self.i):
                if self.s.startswith("[[", self.i):
                    self.fail("les tableaux de tables [[...]] ne sont pas lus")
                self.i += 1
                self.ws()
                path = self.dotted()
                if not self.s.startswith("]", self.i):
                    self.fail("']' attendu")
                self.i += 1
                cur = root
                for p in path:
                    cur = cur.setdefault(p, {})
                    if not isinstance(cur, dict):
                        self.fail("%s n'est pas une table" % ".".join(path))
            else:
                path = self.dotted()
                if not self.s.startswith("=", self.i):
                    self.fail("'=' attendu après la clé %s" % ".".join(path))
                self.i += 1
                self.ws()
                t = cur
                for p in path[:-1]:
                    t = t.setdefault(p, {})
                if path[-1] in t:
                    self.fail("clé %s en double" % ".".join(path))
                t[path[-1]] = self.value()
            self.ws()
            if self.i < len(self.s) and self.s[self.i] != "\n":
                self.fail("fin de ligne attendue")


def parse_toml(text, path="<site>"):
    """The TOML @p text as a dict: tomllib, tomli, else the subset reader."""
    try:
        import tomllib as toml
    except ImportError:
        try:
            import tomli as toml
        except ImportError:
            toml = None
    if toml is None:
        return _Reader(text, path).parse()
    try:
        return toml.loads(text)
    except toml.TOMLDecodeError as e:
        raise SiteError("%s : %s" % (path, e))


# --------------------------------------------------------------------------
# The site
# --------------------------------------------------------------------------
class Site:
    """A checked site file. Attributes: path, name, host, repo, runs_dir,
    build_root, partition (the default one), batch (dict), detached (dict
    or None), qos_fallback (str), partitions {key: {label, arch, directives,
    launcher, detached}}, discovered (dict, or None: the templates of the
    partitions batch["partition_info"] lists, see partitions.py), defaults
    {(program, arch): {dest: value}}."""

    def __init__(self, data, path):
        self.path = str(path)
        self.data = data
        chk = _Checker(data, self.path)
        s = chk.table("site")
        self.name = chk.text(s, "site", "name")
        self.host = chk.text(s, "site", "host", self.name)
        self.repo = chk.text(s, "site", "repo", "~/shrink-array")
        self.runs_dir = chk.text(s, "site", "runs_dir", "~/shrink_runs")
        self.build_root = chk.text(s, "site", "build_root", "")
        b = chk.table("batch")
        self.batch = {
            "label": chk.text(b, "batch", "label", "batch"),
            "submit": chk.command(b, "batch", "submit"),
            "jobid_re": chk.regex(b, "batch", "jobid_re", r"\b(\d{2,})\b"),
            "cancel": chk.commands(b, "batch", "cancel", ["scancel"]),
            "qos_info": chk.text(b, "batch", "qos_info", ""),
            # Prints the machine's partitions (partitions.parse); "" when there is none.
            "partition_info": chk.text(b, "batch", "partition_info", ""),
            "directive": chk.text(b, "batch", "directive"),
            # Inside a batch job of several nodes: what starts one task per
            # node (runscripts.node_launch), when the partition has no launcher.
            "step": chk.command(b, "batch", "step") if "step" in b else "srun",
        }
        for k, default, fields in (("name", "", ("name",)), ("time", None, ("seconds", "minutes", "hms")),
                                   ("output", None, ("log",)), ("error", "", ("log",)),
                                   ("qos", "", ("qos",)), ("slurm", "", ("options", "option"))):
            self.batch[k] = chk.template(b, "batch", k, default, fields)
        if self.batch["slurm"] and "{options}" in self.batch["slurm"] and "{option}" in self.batch["slurm"]:
            chk.fail("batch.slurm", "{options} (toutes sur une ligne) ou {option} (une ligne chacune), pas les deux")
        d = chk.table("detached", required=False)
        self.detached = None
        if d is not None:
            self.detached = {"label": chk.text(d, "detached", "label", "détaché"),
                             "command": chk.command(d, "detached", "command"),
                             "time": chk.template(d, "detached", "time", "", ("seconds", "minutes", "hms")),
                             "qos": chk.template(d, "detached", "qos", "", ("qos",))}
        q = chk.table("qos", required=False) or {}
        self.qos_fallback = chk.text(q, "qos", "fallback", "")
        parts = chk.table("partitions")
        if not parts:
            chk.fail("partitions", "au moins une partition [partitions.<nom>]")
        self.partitions = {}
        for key, p in parts.items():
            where = "partitions." + key
            if not isinstance(p, dict):
                chk.fail(where, "une table attendue")
            if key == "local":
                chk.fail(where, "« local » est réservé à cette machine")
            self.partitions[key] = {
                "label": chk.text(p, where, "label", key),
                "arch": chk.text(p, where, "arch"),
                "directives": chk.text(p, where, "directives", ""),
                "launcher": chk.text(p, where, "launcher", ""),
                "detached": chk.text(p, where, "detached", ""),
            }
            chk.unknown(p, where, ("label", "arch", "directives", "launcher", "detached"))
        # The partitions the machine lists, beside these: the templates of their entries.
        disc = chk.table("discovered", required=False)
        self.discovered = None
        if disc is not None:
            fields = ("name", "cores")
            self.discovered = {
                "directives": chk.template(disc, "discovered", "directives", None, fields),
                "launcher": chk.template(disc, "discovered", "launcher", "", fields),
                "detached": chk.template(disc, "discovered", "detached", "", fields),
                "arm": chk.pattern(disc, "discovered", "arm", ""),
                "skip": chk.pattern(disc, "discovered", "skip", ""),
            }
            chk.unknown(disc, "discovered", tuple(self.discovered))
        if self.batch["partition_info"] and self.discovered is None:
            chk.fail("discovered", "table manquante : batch.partition_info lit les partitions de la machine, "
                                   "[discovered] dit comment les soumettre")
        self.partition = chk.text(s, "site", "partition", next(iter(self.partitions)))
        if self.partition not in self.partitions:
            chk.fail("site.partition", "« %s » n'est pas dans [partitions]" % self.partition)
        self.defaults = {}
        for prog, by_arch in (chk.table("defaults", required=False) or {}).items():
            if not isinstance(by_arch, dict):
                chk.fail("defaults." + prog, "une table par architecture attendue")
            for arch, values in by_arch.items():
                where = "defaults.%s.%s" % (prog, arch)
                if not isinstance(values, dict):
                    chk.fail(where, "une table attendue")
                out = {}
                for dest, v in values.items():
                    if isinstance(v, str):
                        v = v.replace("{build_root}", self.build_root)
                    elif not isinstance(v, (bool, int, float)):
                        chk.fail(where + "." + dest, "une chaîne, un nombre ou true/false")
                    out[dest] = v
                self.defaults[(prog, arch)] = out
        chk.unknown(s, "site", ("name", "host", "repo", "runs_dir", "build_root", "partition"))
        chk.unknown(b, "batch", tuple(self.batch))
        if d is not None:
            chk.unknown(d, "detached", tuple(self.detached))
        chk.unknown(q, "qos", ("fallback",))
        chk.unknown(data, "", ("site", "batch", "detached", "qos", "partitions", "discovered", "defaults"))

    def detached_options(self, options, seconds, qos):
        """The detached command's options: the partition's @p options, then
        the time and the QoS from the [detached] templates."""
        d = self.detached or {}
        out = " ".join(options.split())
        if d.get("time"):
            out += " " + d["time"].format(seconds=int(seconds), minutes=(int(seconds) + 59) // 60,
                                          hms=hms(seconds))
        if qos and d.get("qos"):
            out += " " + d["qos"].format(qos=qos)
        return out.strip()

    def words(self):
        """The CEA words of the interface texts -> this site's (i18n)."""
        w = {"inti": self.name,
             "ccc_msub": shlex.split(self.batch["submit"])[0],
             "ccc_mdel": self.batch["cancel"][0].split()[0],
             "#MSUB": self.batch["directive"]}
        if self.batch["qos_info"]:
            w["ccc_mqinfo"] = shlex.split(self.batch["qos_info"])[0]
        if self.batch["partition_info"]:
            w["ccc_mpinfo"] = shlex.split(self.batch["partition_info"])[0]
        if self.detached:
            w["ccc_mprun"] = shlex.split(self.detached["command"])[0]
        return w


class _Checker:
    def __init__(self, data, path):
        self.data, self.path = data, path

    def fail(self, key, what):
        raise SiteError("%s : %s : %s" % (self.path, key, what))

    def table(self, name, required=True):
        t = self.data.get(name)
        if t is None:
            if required:
                self.fail(name, "table [%s] manquante" % name)
            return None
        if not isinstance(t, dict):
            self.fail(name, "une table [%s] attendue" % name)
        return t

    def text(self, t, where, key, default=None):
        v = t.get(key, default)
        if v is None:
            self.fail("%s.%s" % (where, key), "clé manquante")
        if not isinstance(v, str):
            self.fail("%s.%s" % (where, key), "une chaîne attendue")
        return v

    def command(self, t, where, key):
        v = self.text(t, where, key)
        try:
            argv = shlex.split(v)
        except ValueError as e:
            self.fail("%s.%s" % (where, key), str(e))
        if not argv:
            self.fail("%s.%s" % (where, key), "commande vide")
        return v

    def commands(self, t, where, key, default):
        v = t.get(key, default)
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not v or not all(isinstance(c, str) and c.strip() for c in v):
            self.fail("%s.%s" % (where, key), "une commande ou une liste de commandes attendue")
        return v

    def regex(self, t, where, key, default):
        v = self.text(t, where, key, default)
        try:
            if re.compile(v).groups < 1:
                self.fail("%s.%s" % (where, key), "un groupe (...) autour du numéro de job")
        except re.error as e:
            self.fail("%s.%s" % (where, key), "expression invalide : %s" % e)
        return v

    def pattern(self, t, where, key, default):
        v = self.text(t, where, key, default)
        try:
            re.compile(v)
        except re.error as e:
            self.fail("%s.%s" % (where, key), "expression invalide : %s" % e)
        return v

    def template(self, t, where, key, default, fields):
        v = self.text(t, where, key, default)
        for f in re.findall(r"\{(\w*)\}", v):
            if f not in fields:
                self.fail("%s.%s" % (where, key), "champ {%s} inconnu (permis : %s)"
                          % (f, ", ".join("{%s}" % x for x in fields)))
        return v

    def unknown(self, t, where, known):
        for k in t:
            if k not in known:
                self.fail(("%s.%s" % (where, k)) if where else k,
                          "clé inconnue (permises : %s)" % ", ".join(known))


def site_path():
    """The site file to read (see the module doc)."""
    env = os.environ.get("SHRINK_GUI_SITE")
    if env:
        return pathlib.Path(os.path.expanduser(env))
    cfg = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"))
    user = cfg / "shrink_gui" / "site.toml"
    return user if user.is_file() else BUNDLED


def load(path=None):
    """The Site of @p path (default: site_path()); SiteError if unusable."""
    path = pathlib.Path(path) if path else site_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SiteError("%s : illisible (%s)" % (path, e.strerror or e))
    return Site(parse_toml(text, str(path)), path)


def hms(seconds):
    """Slurm's [D-]HH:MM:SS (remote_helper.hms, which cannot import this)."""
    days, r = divmod(int(seconds), 86400)
    return ("%d-" % days if days else "") + "%02d:%02d:%02d" % (r // 3600, r % 3600 // 60, r % 60)
