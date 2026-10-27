# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The "Quoi" part of the window: the program, its options form, the start
conditions, the build check, the --target / --exec pickers, and the
submission.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import ast
import base64
import json
import os
import platform
import re
import shlex
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from shrink_gui import qos
from shrink_gui.config import ARCH_DEFAULTS, BATCH, BUILD_REP, DEFAULTS, DETACHED, PROGRAMS, SITE
from shrink_gui.i18n import T
from shrink_gui.runscripts import make_launch_sh, make_local_launch_sh, make_run_sh
from shrink_gui.widgets import PickerDialog, ask, families, split_list, unique_globs


class LaunchForm:
    def program_changed(self, *_):
        prog = self.program.currentText()
        self.time_s.setValue(int(self.settings.value(f"time_s/{prog}",
                                                     PROGRAMS[prog].get("time_s", DEFAULTS["time_s"]))))
        cached = self.settings.value(f"schema/{self.program.currentText()}")
        self.form.pickers = {d: self.pick for d in PROGRAMS[prog].get("pickers", {})}
        self.check_builds.setEnabled("builds_opt" in PROGRAMS[prog])
        self.update_nodes_row()
        self.form.build(json.loads(cached) if cached else None)
        self.form.filter(self.opt_filter.text())
        self.apply_arch_defaults()
        if not cached:
            QTimer.singleShot(0, self.scan)

    def scan(self):
        prog = self.program.currentText()
        self.build_info.clear()      # re-read too, with the options
        self.info(f"lecture des options de {prog}…")
        self.last_error = None

        def done(r):
            if r is None:
                if prog == self.program.currentText() and not self.form.schema:
                    self.form.build(None, self.last_error or "erreur inconnue (voir la barre d'état)")
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
            via = f" (python {r['python']} sur {self.machine_name()})" if r.get("python") else ""
            self.info(f"{prog} : {len(r['actions'])} options lues{via}")

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

    # ------------------------------------------------- start conditions
    def fill_after(self):
        """What a submission can wait for: the compile_all.sh runs not
        finished yet (a finished one needs no waiting). The jobs of a
        submission over several nodes (meta "pool") are one entry, the
        submission's: its data is the pool, and after_spec() waits for them all."""
        keep = self.after.currentData() if self.after.count() else ""
        self.after.blockSignals(True)
        self.after.clear()
        self.after.addItem(T("dès la soumission"), "")
        pools = set()
        for r in self.runs:
            if (r["meta"].get("program") == "compile_all.sh"
                    and (r["status"] in self.ACTIVE or r["status"].startswith("en attente"))):
                pool = r["meta"].get("pool")
                if not pool:
                    self.after.addItem(T(f"après {r['id']}"), r["id"])
                elif pool not in pools:
                    pools.add(pool)
                    self.after.addItem(T(f"après {pool} ({len(self.pool_runs(pool))} nœuds)"), pool)
        self.after.setCurrentIndex(max(0, self.after.findData(keep)))
        self.after.blockSignals(False)
        self.after_cond.setEnabled(bool(self.after.currentData()))

    def pool_runs(self, pool):
        """The run ids of the jobs of submission @p pool, first node first."""
        return sorted((r["id"] for r in self.runs if r["meta"].get("pool") == pool),
                      key=lambda rid: next(r["meta"].get("node", 0) for r in self.runs if r["id"] == rid))

    def after_spec(self):
        """The start condition chosen: {"run", "runs", "cond"} -- "runs" all
        the runs to wait for (the jobs of a submission over several nodes),
        "run" the first of them -- or None."""
        chosen = self.after.currentData() or ""
        if not chosen:
            return None
        runs = self.pool_runs(chosen) or [chosen]
        return {"run": runs[0], "runs": runs, "cond": self.after_cond.currentData()}

    # ------------------------------------------------------ several nodes
    def pool_opt(self):
        """The option that shares a program's work between jobs (compile_all.sh:
        --pool), or None when it cannot be split, or not in this mode."""
        if not hasattr(self, "mode") or self.mode.currentData() != "msub":
            return None
        return PROGRAMS.get(self.program.currentText(), {}).get("pool_opt")

    def nodes_wanted(self):
        """How many batch jobs a submission makes: the "Nœuds" field, or 1."""
        return self.nodes.value() if hasattr(self, "nodes") and self.pool_opt() else 1

    def update_nodes_row(self):
        if hasattr(self, "nodes") and hasattr(self, "where"):
            self.where.setRowVisible(self.nodes, bool(self.pool_opt()))

    def form_value(self, dest):
        """The value option @p dest will have: the form's, else the script's
        default (a list default arrives as its repr)."""
        v = self.form.values().get(dest)
        if v not in (None, ""):
            return v
        for ac in (self.form.schema or {}).get("actions", []):
            if ac["dest"] == dest:
                d = ac["default"]
                if isinstance(d, str) and d[:1] in "[(":
                    try:
                        return list(ast.literal_eval(d))
                    except (ValueError, SyntaxError):
                        return d
                return d
        return None

    def build_check(self):
        """The build check run.sh makes before the command
        (runscripts.check_builds_call), or None: the builds the program will
        use, read from its own options."""
        opt = PROGRAMS[self.program.currentText()].get("builds_opt")
        mode = self.check_builds.currentData()
        if not opt or mode == "off" or not self.form.schema:
            return None
        builds = self.form_value(opt["builds"])
        if not builds:
            return None
        root = self.form_value(opt["root"])
        check = {"mode": mode, "root": root if isinstance(root, str) and root else "$CHUNK_BUILD_ROOT"}
        if isinstance(builds, str):
            parts = [x for x in re.split(r"[,\s]+", builds) if x]
            if len(parts) == 1:          # a file of names (as multi_launch_stress --builds), or one name
                check["builds_file"] = parts[0]
                return check
            builds = parts
        check["builds"] = [str(b) for b in builds]
        return check

    def missing_options(self, prog):
        """The refusal of PROGRAMS[prog]["one_of"] for the form as it is (none
        of a group's options set, in the form or the "arguments en plus"), or
        None."""
        vals = self.form.values()
        extra = self.extra.text().split()
        widgets = self.form.widgets

        def flags(dest):
            return widgets[dest][0]["flags"] if dest in widgets else ["--" + dest.replace("_", "-")]

        def given(dest):
            return dest in vals or any(a == f or a.startswith(f + "=") for a in extra for f in flags(dest))

        for rule in PROGRAMS[prog].get("one_of", []):
            if any(given(d) for d in rule["dests"]):
                continue
            names = " ou ".join(max(flags(d), key=len) for d in rule["dests"])
            msg = f"{prog} : il faut {names} (sinon le job s'arrête aussitôt sur une erreur d'usage)"
            useless = [max(flags(d), key=len) for d in rule.get("with", []) if given(d)]
            if useless and rule.get("one_with"):
                msg += (f" -- {', '.join(useless)} ne servent qu'avec {max(flags(rule['one_with']), key=len)} : "
                        "est-ce lui qu'il fallait cocher ?")
            return msg
        return None

    def command(self):
        prog = self.program.currentText()
        parts = [PROGRAMS[prog]["interp"], os.path.basename(PROGRAMS[prog]["script"])] + self.form.args()
        if self.extra.text().strip():
            parts.append(self.extra.text().strip())
        return " ".join(parts)

    def update_preview(self, *_):
        self.apply_time_by_mode()
        if hasattr(self, "preview"):
            n = self.nodes_wanted()
            self.preview.setPlainText(self.command() + (
                f" {self.pool_opt()}=<{T('la soumission')}>\n# {T(f'{n} jobs, un nœud chacun')}" if n > 1 else ""))

    def time_by_mode(self):
        """The -T the program's "time_by_mode" gives for the chosen mode and
        this partition's architecture, or None (no such entry, this machine)."""
        spec = PROGRAMS[self.program.currentText()].get("time_by_mode")
        if not spec or not hasattr(self, "form") or self.is_local():
            return None
        mode = self.form.values().get(spec["dest"]) or spec["default"]
        secs = spec["seconds"].get(mode)
        if secs is None:
            return None
        return int(round(secs * spec.get("arch_factor", {}).get(self.arch(), 1.0)))

    def apply_time_by_mode(self):
        """Set -T from time_by_mode() when the (program, mode, partition)
        changed since the last time: a duration typed by hand stays until then."""
        if not hasattr(self, "time_s") or not hasattr(self, "preset"):
            return
        t = self.time_by_mode()
        if t is None:
            self._time_mode_key = None
            return
        prog = self.program.currentText()
        spec = PROGRAMS[prog]["time_by_mode"]
        key = (prog, self.form.values().get(spec["dest"]) or spec["default"], self.preset.currentData())
        if key != getattr(self, "_time_mode_key", None):
            self._time_mode_key = key
            if self.time_s.value() != t:
                self.time_s.setValue(t)

    # ---------------------------------------------- --target / --exec pickers
    def arch_suffix(self):
        """The <arch> component of compile_all.sh's build dirs ends with uname -m."""
        if self.is_local():
            return (self.local_info or {}).get("machine") or platform.machine()
        return "aarch64" if self.arch() == "arm" else "x86_64"

    def pick(self, dest):
        """"Choisir…" of --target / --exec: what compile_all.sh can build on
        this partition's architecture, read on inti (cached until "Lire les
        options")."""
        sfx = self.arch_suffix()
        info = self.build_info.get(sfx)
        if info is None:
            self.info(f"lecture des builds de {sfx} sur {self.machine_name()}…")

            def done(r):
                if r is not None:
                    self.build_info[sfx] = r
                    self.info(f"{len(r['configs'])} configurations, {len(r['dirs'])} build(s) {sfx} "
                              f"dans {r['build_rep']}")
                    self.pick(dest)
            # this machine: the builds are where site_env.sh puts them (the repository)
            build_rep = "" if self.is_local() else BUILD_REP
            self.remote.helper("build_targets", [self.repo.text().strip(), build_rep, sfx], done)
            return
        ac, w, _lab = self.form.widgets[dest]
        multi = "comma" in (ac.get("help") or "").lower()
        kind = PROGRAMS[self.program.currentText()]["pickers"][dest]
        dlg = (self.target_dialog if kind == "configs" else self.exec_dialog)(info, w.text(), multi, sfx)
        if dlg.exec():
            w.setText(dlg.value())

    def target_dialog(self, info, value, multi, sfx):
        dirs = {d["name"]: d for d in info["dirs"]}
        names = sorted(set(info["configs"]) | set(dirs))
        items = []
        for n in names:
            d = dirs.get(n)
            if d and d["configured"]:
                when = time.strftime("%Y-%m-%d", time.localtime(d["mtime"]))
                note = f"configurée, {len(d['execs'])} exécutable(s), {when}"
            elif d:
                note = "dossier sans build.ninja"
            else:
                note = ""
            items.append((n, note))
        pats = {g: c for g, c in families(names, "-").items()}
        for kw in ("sanitizer", "debug", "check", "stats", "ledger", "cs"):
            hits = [n for n in names if kw in n]
            if 2 <= len(hits) < len(names):
                pats[f"*{kw}*"] = len(hits)
        order = sorted(pats.items(), key=lambda kv: (not (kv[0].startswith("*") and kv[0].endswith("*")),
                                                     -kv[1], kv[0]))
        patterns = [(g, f"motif : {c} configurations") for g, c in unique_globs(order, names)[:24]]
        intro = (f"Configurations de compile_all.sh (cmake-build-<nom>). « configurée » : son build "
                 f"existe déjà pour {sfx} dans {info['build_rep']}.")
        return PickerDialog(self, "--target : configurations", intro, items, patterns, value, multi)

    def exec_dialog(self, info, value, multi, sfx):
        target = self.form.widgets["target"][1].text() if "target" in self.form.widgets else ""
        import fnmatch
        dirs = [d for d in info["dirs"] if d["configured"]
                and (not split_list(target) or any(fnmatch.fnmatchcase(d["name"], g) for g in split_list(target)))]
        count = {}
        for d in dirs:
            for e in d["execs"]:
                count[e] = count.get(e, 0) + 1
        items = [(e, f"dans {c}/{len(dirs)} build(s)") for e, c in sorted(count.items())]
        patterns = [(g, f"motif : {c} exécutables")
                    for g, c in unique_globs(families(sorted(count), "_").items(), sorted(count))[:24]]
        if dirs:
            scope = f"des {len(dirs)} build(s) {sfx} configurés" + (f" retenus par --target={target}" if target else "")
            intro = (f"Exécutables {scope}. Ninja construit aussi leurs dépendances (bibliothèques chunk), "
                     f"pas les autres exécutables.")
        else:
            intro = (f"Aucun build {sfx} configuré" + (f" pour --target={target}" if target else "")
                     + f" dans {info['build_rep']} : rien à lister. Lancez d'abord compile_all.sh sans "
                       "--exec (ou tapez le nom à la main).")
        return PickerDialog(self, "--exec : exécutables", intro, items, patterns, value, multi)

    # --------------------------------------------------------------- launch
    def launch(self, *_, command=None, extra=None, rerun_of=None):
        """Submit the form -- or @p command as it is (a replay of failures:
        rerun_failed), with the form's partition, mode, directives and -T."""
        self.save()
        prog = self.program.currentText()
        if command is None and not self.form.schema and ask(
                self, "Options non lues",
                "Les options du script n'ont pas été lues : seuls les arguments en plus seront "
                "passés.\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        if command is None and self.form.schema:
            err = self.missing_options(prog)
            if err:
                self.error(err)
                return
        pkey = self.preset.currentData()
        mode = self.mode.currentData()
        if PROGRAMS[prog].get("msub_only") and mode == "mprun":
            self.error(f"{prog} se soumet par ccc_msub seulement (il ne doit pas tourner depuis le nœud "
                       f"de login) : choisissez « {BATCH['label']} »")
            return
        qname, qerr = None, None
        if mode != "local":          # no scheduler, no QoS on this machine
            qname, _row, qerr = self.qos_choice()
        if qerr:
            self.error(qerr)
            return
        busy = [r["id"] for r in self.runs
                if r["status"] in ("en cours", "en attente", "démarrage")
                and r["meta"].get("partition") == pkey and r["meta"].get("program") == prog]
        if busy and ask(
                self, "Run déjà actif",
                f"{prog} a déjà un run actif sur {pkey} ({', '.join(busy)}).\n"
                "Deux runs simultanés de même architecture peuvent écrire le même répertoire "
                "(oracles, résumé).\n\nSoumettre quand même ?") != QMessageBox.Yes:
            return
        cmd = command if command is not None else self.command()
        spec = PROGRAMS[prog]
        if command is None and spec.get("scan") == "argparse" and spec.get("validate", True):
            # The script's own parsing on the cluster first: a bad option
            # (or one of its checks) refuses here, not in the job's log.
            try:
                argv = shlex.split(cmd)[2:]
            except ValueError as e:
                self.error(f"commande illisible : {e}")
                return

            def checked(r):
                if r is None:
                    return
                if r.get("refused"):
                    self.error(f"{prog} refuse ces options : {r['refused']}")
                    return
                if r.get("skipped"):
                    self.info(f"options non vérifiées ({r['skipped']}) : soumis quand même")
                self.submit(prog, pkey, mode, qname, cmd, extra, rerun_of)

            blob = base64.b64encode(json.dumps(argv).encode()).decode()
            self.info(f"vérification des options de {prog}…")
            self.remote.helper("validate", [self.repo.text().strip(), spec["script"], blob], checked)
            return
        self.submit(prog, pkey, mode, qname, cmd, extra, rerun_of)

    def submit(self, prog, pkey, mode, qname, cmd, extra, rerun_of):
        """The second half of launch(): the run directory, its scripts, the job."""
        after = self.after_spec()
        delay_s = self.delay.value() if self.delay_on.isChecked() else 0
        check = self.build_check()
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + prog.split(".")[0].split("_")[0] + "-" + pkey
        nodes = self.nodes_wanted()
        if nodes > 1 and re.search(r"(^|\s)%s(=|\s|$)" % re.escape(self.pool_opt()), cmd):
            self.error(f"{self.pool_opt()} est ajouté par le champ « Nœuds » : retirez-le de la commande")
            return
        if nodes > 1:
            # One batch job per node, the same command and --pool=<rid>: the
            # program shares its work between them (compile_all.sh: each
            # configuration built by the first job that claims it). The form's
            # values stay without it: "Reprendre ses options" makes a new pool.
            pool = rid
            jobs = [(f"{rid}-n{k}", f"{cmd} {self.pool_opt()}={pool}",
                     {"pool": pool, "node": k, "nodes": nodes}) for k in range(1, nodes + 1)]
        else:
            jobs = [(rid, cmd, {})]
        specs = [(job_rid, self.job_spec(prog, pkey, mode, qname, job_cmd, extra, rerun_of, after, delay_s,
                                         check, job_meta))
                 for job_rid, job_cmd, job_meta in jobs]
        submitted, answers = [], []

        def next_job(r=False):
            if r is None:          # refused: the status bar says why
                if len(submitted) > 1:
                    self.refresh(select=submitted[0])
                return
            if r:
                answers.append(r)
            if specs:
                job_rid, spec = specs.pop(0)
                submitted.append(job_rid)
                blob = base64.b64encode(json.dumps(spec).encode()).decode()
                self.remote.helper("launch", [self.runs_dir.text().strip(), job_rid, blob], next_job)
                return
            self.info(f"soumis : {len(submitted)} jobs, un nœud chacun" if len(submitted) > 1 else
                      f"soumis : job {r['jobid']}" if "jobid" in r else f"lancé : {r['run_dir']}")
            self.current, self.offset = submitted[0], -1     # -1: the end of the log first (act_tail)
            self.log.clear()
            # What was just run, at once (the first job's, the one shown): the
            # job's own log (run.sh's "[gui] command:" line) only starts with
            # the job, on the node.
            first = answers[0]
            head = [f"[gui] run {first['run_dir']}", f"[gui] commande : {jobs[0][1]}"]
            if first.get("launched"):
                head.append(f"[gui] soumission : {first['launched']}")
            if first.get("submit_out"):
                head.append(first["submit_out"])
            self.log.append_raw("".join(T(line) + "\n" for line in head))
            self.refresh(select=submitted[0])

        next_job()

    def job_spec(self, prog, pkey, mode, qname, cmd, extra, rerun_of, after, delay_s, check, job_meta):
        """What remote_helper's launch needs for one job; @p job_meta: what the
        job of a submission over several nodes adds to its meta.json."""
        name = f"{prog.split('.')[0]}_{pkey}" + (f"_{job_meta['node']}" if job_meta else "")
        return {
            "mode": mode,
            "run.sh": make_run_sh(self.repo.text().strip(), PROGRAMS[prog]["cwd"], cmd, check),
            "msub": self.directives.toPlainText() if mode == "msub" else "",
            "launcher": self.launcher.text().strip() if mode == "msub" else "",
            "launch.sh": make_launch_sh(SITE.detached_options(self.directives.toPlainText(),
                                                              self.time_s.value(), qname),
                                        after, delay_s, command=DETACHED["command"])
                         if mode == "mprun" else
                         make_local_launch_sh(self.launcher.text().strip(), self.time_s.value(), after, delay_s)
                         if mode == "local" else "",
            "qos": qname or "",
            "batch": BATCH,
            "after": after, "delay_s": delay_s,
            "name": re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:40],
            "time_s": self.time_s.value(),
            "meta": {"program": prog, "partition": pkey, "arch": self.arch(), "mode": mode,
                     "launcher": self.launcher.text().strip() if mode in ("msub", "local") else "",
                     "command": cmd, "values": self.form.values(),
                     "extra": extra if extra is not None else self.extra.text(),
                     "repo": self.repo.text().strip(), "time_s": self.time_s.value(),
                     "qos": "" if mode == "local" else qname or qos.explicit_qos(self.directives.toPlainText()) or "",
                     "qos_wanted": self.qos.currentData(),
                     "after": after, "delay_s": delay_s,
                     "check_builds": check["mode"] if check else "off",
                     "created": time.strftime("%F %T"),
                     **({"rerun_of": rerun_of} if rerun_of else {}),
                     **job_meta,
                     **{k: PROGRAMS[prog][k] for k in ("out_re", "out_files", "progress_re",
                                                       "eta_re", "summary", "summary_file")},
                     **({"chart": PROGRAMS[prog]["chart"]} if PROGRAMS[prog].get("chart") else {})},
        }
