# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Git (inti) rows: the repository's state, git pull, branches.

A mixin of the main window (app.Main): its methods use the window's
attributes."""
import html
import os

from PySide6.QtWidgets import QMessageBox

from shrink_gui.i18n import T
from shrink_gui.widgets import ask


class GitPanel:
    def local_git(self):
        """{commit, branch} of the local clone ("Dépôt local"), or None."""
        path = os.path.expanduser(self.local_repo.text().strip())
        if not path:
            return None
        import subprocess
        try:
            c = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            b = subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return {"commit": c, "branch": b} if c else None

    def show_git_state(self):
        """The "Git (inti)" row: branch, commit, changes, upstream, ahead/behind."""
        g = getattr(self, "inti_git", None)
        if not g:
            self.git_state.setText("pas de dépôt git (ou pas encore de connexion)")
            for b in (self.bpull, self.bpull_rebase, self.bswitch, self.bbranches):
                b.setEnabled(False)
            return
        self.bbranches.setEnabled(True)
        chosen = self.branch.currentData()
        self.bswitch.setEnabled(bool(chosen) and chosen != g["branch"])
        txt = f"branche <b>{html.escape(g['branch'])}</b> @ {g['commit'][:8]}"
        if g.get("dirty"):
            txt += " · <span style='color:#b9770e'>modifications locales</span>"
        up = g.get("upstream")
        if not up:
            txt += " · pas de branche suivie"
        else:
            txt += f" · suit {html.escape(up)}"
            ahead, behind = g.get("ahead", 0), g.get("behind", 0)
            if behind:
                txt += f" · <span style='color:#b9770e'><b>{behind}</b> en retard</span>"
            if ahead:
                txt += f" · <b>{ahead}</b> en avance"
            if not ahead and not behind:
                txt += " · à jour au dernier fetch"
        self.git_state.setText(txt)
        for b in (self.bpull, self.bpull_rebase):
            b.setEnabled(bool(up))
        if getattr(self, "git_busy", 0):
            # One git command at a time in the repository: a pull or a switch
            # started while load_branches' fetch runs would fail on git's locks.
            for b in (self.bpull, self.bpull_rebase, self.bswitch, self.bbranches):
                b.setEnabled(False)

    def git_pull(self, rebase):
        """git pull [--rebase] in inti's repository, after a confirmation."""
        g = getattr(self, "inti_git", None)
        if not g:
            self.error(f"pas de dépôt git sur {self.machine_name()} : vérifiez la connexion")
            return
        what = "git pull --rebase" if rebase else "git pull"
        active = [r for r in self.runs if r["status"] in self.ACTIVE]
        text = T(f"{what} dans {self.repo.text().strip()} (branche {g['branch']}) ?")
        if active:
            text += "\n\n" + T(f"{len(active)} run(s) encore actif(s) : un build ou un bench en cours "
                                 f"peut lire des sources modifiées en cours de route.")
        if ask(self, "Git", text) != QMessageBox.Yes:
            return
        if getattr(self, "git_busy", 0):
            return
        self.git_busy = getattr(self, "git_busy", 0) + 1
        self.show_git_state()
        self.info(f"{what} sur {self.machine_name()}…")

        def done(r):
            self.git_busy -= 1
            if r is None:  # the error is already in the status bar
                self.show_git_state()
                return
            self.inti_git = r.get("git") or self.inti_git
            self.show_git_state()
            if r["ok"]:
                moved = r.get("git") and r["git"]["commit"] != r.get("before")
                self.info(f"{what} : " + (f"{r['before'][:8]} → {r['git']['commit'][:8]}" if moved
                                          else "déjà à jour"))
                if moved:
                    self.check()   # the connection line, then the script options again
                    self.scan()
            else:
                head = T(f"{what} a échoué")
                if r.get("aborted"):
                    head += T(f" ; {r['aborted']} annulé, le dépôt est revenu à son état d'avant")
                QMessageBox.warning(self, "Git", head + "\n\n" + r.get("output", "")[-3000:])

        self.remote.helper("git_pull", [self.repo.text().strip(), "rebase" if rebase else "merge"], done)

    def load_branches(self):
        """git fetch on inti, then the "Branche (inti)" list: the local branches,
        then the remote ones without a local branch; the current one selected."""
        if not getattr(self, "inti_git", None) or getattr(self, "git_busy", 0):
            return
        self.git_busy = getattr(self, "git_busy", 0) + 1
        self.show_git_state()

        def done(r):
            self.git_busy -= 1
            if r is None:  # the error is already in the status bar
                self.show_git_state()
                return
            if r.get("git"):
                self.inti_git = r["git"]
            cur = self.inti_git["branch"]
            self.branch.blockSignals(True)
            self.branch.clear()
            for b in r.get("local", []):
                self.branch.addItem(b + (T(" (actuelle)") if b == cur else ""), b)
            if r.get("remote"):
                self.branch.insertSeparator(self.branch.count())
                for b in r["remote"]:
                    self.branch.addItem(b, b)
            i = self.branch.findData(cur)
            if i >= 0:
                self.branch.setCurrentIndex(i)
            self.branch.blockSignals(False)
            self.show_git_state()
            if r.get("fetch_error"):
                self.info(T("git fetch a échoué sur inti : la liste est celle du dernier fetch"))

        self.remote.helper("git_branches", [self.repo.text().strip()], done)

    def git_switch(self):
        """Check out, in inti's repository, the branch chosen in the list."""
        g = getattr(self, "inti_git", None)
        target = self.branch.currentData()
        if not g or not target or target == g["branch"]:
            return
        active = [r for r in self.runs if r["status"] in self.ACTIVE]
        text = T(f"Changer de branche dans {self.repo.text().strip()} : {g['branch']} → {target} ?")
        if "/" in target:
            text += "\n\n" + T(f"Une branche locale {target.split('/', 1)[1]} suivant {target} est utilisée, "
                                 f"ou créée si elle n'existe pas.")
        if active:
            text += "\n\n" + T(f"{len(active)} run(s) encore actif(s) : un build ou un bench en cours "
                                 f"peut lire des sources modifiées en cours de route.")
        if ask(self, "Git", text) != QMessageBox.Yes:
            return
        if getattr(self, "git_busy", 0):
            return
        self.git_busy = getattr(self, "git_busy", 0) + 1
        self.show_git_state()
        self.info(T(f"changement de branche sur inti : {target}…"))

        def done(r):
            self.git_busy -= 1
            if r is None:
                self.show_git_state()
                return
            self.inti_git = r.get("git") or self.inti_git
            self.show_git_state()
            if r["ok"]:
                self.info(T(f"branche : {r.get('before', '')} → {self.inti_git['branch']}"))
                self.check()   # the connection line and the branch list, then the script options
                self.scan()
            else:
                QMessageBox.warning(self, "Git", T("changement de branche refusé ou en échec") + "\n\n"
                                    + T(r.get("output", ""))[-3000:])
                self.load_branches()

        self.remote.helper("git_switch", [self.repo.text().strip(), target], done)

    def git_line(self):
        """ — inti : 1a2b3c4d (branch) · local : … ≠, for the connection line."""
        g = getattr(self, "inti_git", None)
        if not g:
            return ""
        name = self.machine_name()
        txt = f" — {name} : <b>{g['commit'][:8]}</b>{'*' if g.get('dirty') else ''} ({html.escape(g['branch'])})"
        loc = self.local_git()
        if loc:
            if loc["commit"] == g["commit"]:
                txt += " · local : identique"
            else:
                txt += (f" · <span style='color:#b9770e'>local : <b>{loc['commit'][:8]}</b> "
                        f"({html.escape(loc['branch'])}) ≠ {name}</span>")
        return txt
