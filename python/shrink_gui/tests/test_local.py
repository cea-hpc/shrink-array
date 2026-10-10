# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The local mode ("cette machine"): launch.sh runs run.sh in the background,
without a scheduler -- the helper side, without Qt."""
import base64
import json
import os

from conftest import FAKE_PROGRAM
from shrink_gui.runscripts import make_local_launch_sh, make_run_sh


def _spec(command, time_s=0, launcher="", after=None, delay_s=0):
    return {"mode": "local",
            "run.sh": make_run_sh("~/shrink-array", FAKE_PROGRAM["cwd"], command),
            "msub": "", "launcher": "", "name": "t", "time_s": time_s, "after": after, "delay_s": delay_s,
            "launch.sh": make_local_launch_sh(launcher, time_s, after, delay_s),
            "meta": dict({k: FAKE_PROGRAM[k] for k in ("out_re", "out_files", "progress_re", "eta_re",
                                                       "summary", "summary_file")},
                         program="fake_stress.py", partition="local", mode="local", repo="~/shrink-array",
                         command=command, launcher=launcher, after=after, delay_s=delay_s)}


def _launch(site, rid, spec):
    blob = base64.b64encode(json.dumps(spec).encode()).decode()
    return site.helper("launch", "~/shrink_runs", rid, blob)


def _run(site, rid):
    return {r["id"]: r for r in site.helper("list", "~/shrink_runs")["runs"]}.get(rid)


def _log(site, rid):
    return site.helper("tail", "~/shrink_runs", rid, 0)["data"]


def test_check_describes_the_machine(site):
    r = site.helper("check", "~/shrink-array")
    assert r["machine"] == os.uname()[4] and r["cpus"] == os.cpu_count()
    assert r["mem_gb"] is None or r["mem_gb"] > 0


def test_local_lifecycle(site):
    r = _launch(site, "l1", _spec("python3 fake_stress.py --runs 5 --step 0.01", time_s=600))
    assert r["pid"] and "jobid" not in r
    assert site.wait(lambda: _run(site, "l1")["status"] == "fini")
    run = _run(site, "l1")
    assert run["code"] == 0 and run["jobid"] == "" and run["progress"][:2] == [5, 5]
    log = _log(site, "l1")
    assert "[gui] on this machine (at most 600 s)" in log and "exit code 0" in log
    assert site.helper("summary", "~/shrink_runs", "l1")["rows"][-1][0] == "TOTAL"


def test_local_failure_code(site):
    _launch(site, "l2", _spec("python3 fake_stress.py --runs 2 --rc 3"))
    assert site.wait(lambda: _run(site, "l2")["status"] == "fini")
    assert _run(site, "l2")["code"] == 3


def test_local_time_limit(site):
    _launch(site, "l3", _spec("sleep 30", time_s=1))
    assert site.wait(lambda: _run(site, "l3")["status"] == "fini", 20)
    assert _run(site, "l3")["code"] == 124 and "time limit reached (1 s)" in _log(site, "l3")


def test_local_stop_reaches_the_command(site):
    # timeout puts the command in a process group of its own: "Arrêter" must
    # still end it, and what it started.
    pidfile = site.root / "child.pid"
    _launch(site, "l4", _spec(f"sleep 300 & echo $! > {pidfile}; wait", time_s=600))
    assert site.wait(lambda: pidfile.exists() and pidfile.read_text().strip(), 10)
    assert site.wait(lambda: _run(site, "l4")["status"] == "en cours", 10)
    assert site.helper("stop", "~/shrink_runs", "l4")["ok"]
    assert site.wait(lambda: _run(site, "l4")["status"] == "arrêté", 20)
    child = int(pidfile.read_text())

    def gone():
        try:
            os.kill(child, 0)
            return False
        except OSError:
            return True
    assert site.wait(gone, 10)


def test_local_launcher_prefix(site):
    _launch(site, "l5", _spec('echo "mark=$SHRINK_MARK"', launcher="env SHRINK_MARK=42"))
    assert site.wait(lambda: _run(site, "l5")["status"] == "fini")
    log = _log(site, "l5")
    assert "mark=42" in log and "env SHRINK_MARK=42 bash run.sh" in log


def test_local_waits_for_a_local_compile(site):
    site.env["SHRINK_GUI_POLL"] = "0.2"
    _launch(site, "c1", _spec("sleep 2; exit 0"))
    _launch(site, "l6", _spec("echo STARTED", after={"run": "c1", "cond": "ok"}))
    assert site.wait(lambda: _run(site, "l6")["status"] == "en attente", 5)
    assert site.wait(lambda: _run(site, "l6")["status"] == "fini", 30)
    assert _run(site, "l6")["code"] == 0 and "STARTED" in _log(site, "l6")
