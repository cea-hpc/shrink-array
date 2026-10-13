# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures: a fake inti on this machine.

`site` builds, in a temporary directory:
  - bin/  the fakes of ssh, ccc_msub, squeue, ccc_mdel, ccc_mprun, ccc_mqinfo,
          sbatch, scancel (tests/fakes);
  - home/ the home directory "on inti", whose shrink-array is a link to THIS
    repository: the option reading runs the real scripts;
  - settings/ where QSettings writes, so the tests never touch yours.
The site is CEA's inti (sites/cea_inti.toml), whatever ~/.config holds:
BATCH is its [batch] table, BATCH_ARG the argument check and stop take.
The GUI tests run offscreen (no display needed); they are skipped without
PySide6.

Run from python/:  python3 -m pytest shrink_gui/tests -q
"""
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TESTS = pathlib.Path(__file__).resolve().parent
PKG = TESTS.parent
PYDIR = PKG.parent
REPO = PYDIR.parent
sys.path.insert(0, str(PYDIR))
# The tests are written for inti: never the developer's own site file.
os.environ["SHRINK_GUI_SITE"] = str(PKG / "sites" / "cea_inti.toml")

from shrink_gui import sitecfg  # noqa: E402

BATCH = sitecfg.load(PKG / "sites" / "cea_inti.toml").batch
BATCH_ARG = base64.b64encode(json.dumps(BATCH).encode()).decode()

FAKE_PROGRAM = {
    "script": "python/shrink_gui/tests/data/fake_stress.py", "cwd": "python/shrink_gui/tests/data",
    "interp": "python3", "scan": "argparse",
    "out_re": r"Outputs, CSV and log in (\S+)",
    "out_files": ["timings_summary.csv", "multi_launch_output.log", "error_*.txt"],
    "progress_re": r"Run (\d+)/(\d+) on build", "eta_re": r"Estimated Time Left: ([\d.]+)s",
    "summary": "multi_launch", "summary_file": "timings_summary.csv",
    "errors_files": ["error_*.txt"],
}


class Site:
    """The fake inti: paths, environment, and the helper run "remotely"."""

    def __init__(self, root):
        self.root = root
        self.bin = root / "bin"
        self.home = root / "home"
        self.queue = root / "queue"
        self.settings = root / "settings"
        self.local = root / "local"
        for d in (self.bin, self.home, self.queue, self.settings, self.local):
            d.mkdir()
        for f in (TESTS / "fakes").iterdir():
            shutil.copy(f, self.bin / f.name)
            (self.bin / f.name).chmod(0o755)
        (self.home / "shrink-array").symlink_to(REPO, target_is_directory=True)
        self.runs = self.home / "shrink_runs"
        self.env = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                        FAKE_HOME=str(self.home), FAKE_Q=str(self.queue),
                        SHRINK_GUI_PYDIR=str(PYDIR))

    def helper(self, action, *args, binary=False):
        """Run remote_helper.py as the GUI does (through the fake ssh)."""
        src = (PKG / "remote_helper.py").read_text()
        cmd = [str(self.bin / "ssh"), "inti",
               " ".join(["python3", "-", action] + [_q(a) for a in args])]
        res = subprocess.run(cmd, input=src.encode(), env=self.env, capture_output=True, timeout=120)
        assert res.returncode == 0, res.stderr.decode()
        return res.stdout if binary else json.loads(res.stdout)

    def wait(self, cond, timeout=30.0, step=0.1):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return True
            time.sleep(step)
        return False


def _q(a):
    import shlex
    return shlex.quote(str(a))


@pytest.fixture
def site(tmp_path, monkeypatch):
    s = Site(tmp_path)
    for k in ("PATH", "FAKE_HOME", "FAKE_Q", "SHRINK_GUI_PYDIR"):
        monkeypatch.setenv(k, s.env[k])
    yield s
    # leave no fake job behind
    for f in s.queue.iterdir():
        try:
            os.killpg(int(f.name), 9)
        except (OSError, ValueError):
            pass


@pytest.fixture(scope="session")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture
def gui(qapp, site, monkeypatch):
    """A main window talking to the fake inti, in French, with the fake
    stress program in the program list. Yields (window, pump)."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from shrink_gui import app as app_mod
    from shrink_gui import config, i18n, remote

    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(site.settings))
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(site.settings))
    QSettings(config.APP, config.APP).clear()
    monkeypatch.setattr(remote, "SSH", str(site.bin / "ssh"))
    monkeypatch.setitem(config.PROGRAMS, "fake_stress.py", FAKE_PROGRAM)
    monkeypatch.setitem(config.DEFAULTS, "local_dir", str(site.local))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    i18n.set_lang("fr")

    def pump(cond, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            qapp.processEvents()
            if cond():
                return True
            time.sleep(0.03)
        qapp.processEvents()
        return cond()

    w = app_mod.Main()
    w.local_dir.setText(str(site.local))
    qapp.main_window = w
    w.show()
    assert pump(lambda: "connecté" in w.conn.text()), w.conn.text()
    yield w, pump
    for x in (qapp.main_window, w):
        try:
            x.close()
        except RuntimeError:
            pass
    i18n.set_lang("fr")
