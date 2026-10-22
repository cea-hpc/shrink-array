# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Helper calls: one QProcess each, the helper sent on stdin -- through ssh,
or on this machine (host LOCAL) through LOCAL_SHELL."""
import json
import pathlib
import shlex

from PySide6.QtCore import QProcess, QTimer

from shrink_gui.config import LOCAL, LOCAL_SHELL, NEWEST_PY, SSH

# The helper run on inti, sent on ssh's stdin (see remote_helper.py).
REMOTE_HELPER = pathlib.Path(__file__).with_name("remote_helper.py").read_text()


class Remote:
    """One call = one QProcess (asynchronous, never blocks the window). The
    host LOCAL is this machine: the same command line, run by LOCAL_SHELL
    instead of ssh's remote shell."""

    def __init__(self, parent, host_fn, on_error):
        self.parent = parent
        self.host_fn = host_fn
        self.on_error = on_error
        self.procs = set()

    def is_local(self):
        return self.host_fn() == LOCAL

    def _run(self, args, stdin_text, done, binary=False, program=None, timeout_ms=None, quiet=False):
        """quiet: a failure only gives done(None), not on_error. timeout_ms:
        the process is killed after that long (a dead network leaves ssh
        waiting), which counts as a failure."""
        program = program or SSH
        p = QProcess(self.parent)
        self.procs.add(p)
        out = bytearray()
        p.readyReadStandardOutput.connect(lambda: out.extend(bytes(p.readAllStandardOutput())))
        if timeout_ms:
            t = QTimer(p)            # dies with p
            t.setSingleShot(True)
            t.timeout.connect(p.kill)
            t.start(timeout_ms)

        def finished(code, status):
            out.extend(bytes(p.readAllStandardOutput()))
            err = bytes(p.readAllStandardError()).decode("utf-8", "replace").strip()
            self.procs.discard(p)
            p.deleteLater()
            if code != 0 or status != QProcess.NormalExit:
                if quiet:
                    done(None)
                    return
                where = "local" if self.is_local() else f"ssh {self.host_fn()}"
                self.on_error(f"{where} : code {code}\n{err[-2000:]}")
                done(None)
                return
            done(bytes(out) if binary else bytes(out).decode("utf-8", "replace"))

        p.finished.connect(finished)
        p.errorOccurred.connect(
            lambda e: self.on_error(f"{program} : {e}") if e == QProcess.FailedToStart and not quiet else None)
        p.start(program, args)
        if stdin_text is not None:
            p.write(stdin_text.encode())
            p.closeWriteChannel()

    def check_master(self, done):
        if self.is_local():          # no ssh, no master connection
            done(True)
            return
        self._run(["-O", "check", self.host_fn()], None, lambda o: done(o is not None))

    def ping(self, done, timeout_ms=20000):
        """done(True) when the host answers through the master connection
        within timeout_ms; no message on failure (the caller reports it).
        `ssh -O check` alone only sees the local master process, which
        outlives a dead network until ssh's own keepalive gives up."""
        if self.is_local():
            done(True)
            return

        def master(o):
            if o is None:
                done(False)
                return
            self._run(["-o", "BatchMode=yes", self.host_fn(), "true"], None,
                      lambda o2: done(o2 is not None), timeout_ms=timeout_ms, quiet=True)

        self._run(["-O", "check", self.host_fn()], None, master, timeout_ms=timeout_ms, quiet=True)

    def helper(self, action, args, done, binary=False):
        py = f'"{NEWEST_PY}"' if action in ("options", "validate") else "python3"
        remote_cmd = f"{py} - " + " ".join(shlex.quote(str(a)) for a in [action, *args])

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

        if self.is_local():
            shell = LOCAL_SHELL
            self._run(shell[1:] + [remote_cmd], REMOTE_HELPER, parse, binary=binary, program=shell[0])
        else:
            self._run(["-o", "BatchMode=yes", self.host_fn(), remote_cmd], REMOTE_HELPER, parse,
                      binary=binary)

