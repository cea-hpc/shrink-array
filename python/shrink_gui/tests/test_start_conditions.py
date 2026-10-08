# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Start conditions of a run -- after another run (Slurm dependency, or the
detached mode's wait), not before a delay -- and the up-to-date check of the
builds, against the fake inti (no Qt)."""
import base64
import json
import os
import platform

from conftest import FAKE_PROGRAM
from shrink_gui.runscripts import make_launch_sh, make_run_sh


def _spec(command, mode="msub", program="fake_stress.py", after=None, delay_s=0, check=None):
    return {"mode": mode,
            "run.sh": make_run_sh("~/shrink-array", FAKE_PROGRAM["cwd"], command, check),
            "msub": "#MSUB -q fake\n", "launcher": "",
            "launch.sh": make_launch_sh("-n 1", after, delay_s) if mode == "mprun" else "",
            "name": "t", "time_s": 600, "after": after, "delay_s": delay_s,
            "meta": {"program": program, "partition": "gh200-bxi", "mode": mode, "repo": "~/shrink-array",
                     "command": command, "after": after, "delay_s": delay_s}}


def _launch(site, rid, spec):
    blob = base64.b64encode(json.dumps(spec).encode()).decode()
    return site.helper("launch", "~/shrink_runs", rid, blob)


def _run(site, rid):
    return {r["id"]: r for r in site.helper("list", "~/shrink_runs")["runs"]}.get(rid)


def _log(site, rid):
    return site.helper("tail", "~/shrink_runs", rid, 0)["data"]


def _compile(site, rid, seconds, code):
    """A stand-in for a compile_all.sh run: sleeps, then exits with @p code."""
    r = _launch(site, rid, _spec(f"sleep {seconds}; exit {code}", program="compile_all.sh"))
    assert r.get("jobid"), r
    return r["jobid"]


def test_after_ok_waits_then_starts(site):
    jc = _compile(site, "c1", 3, 0)
    r = _launch(site, "b1", _spec("true", after={"run": "c1", "cond": "ok"}))
    assert r.get("jobid"), r
    assert f'#MSUB -E "--dependency=afterok:{jc}"' in (site.runs / "b1" / "job.sh").read_text()
    assert site.wait(lambda: _run(site, "b1")["status"] == "en attente", 5)
    assert site.wait(lambda: _run(site, "c1")["status"] == "fini", 20)
    assert site.wait(lambda: _run(site, "b1")["status"] == "fini", 20)
    assert _run(site, "b1")["code"] == 0


def test_after_ok_failed_compile_never_starts(site):
    _compile(site, "c2", 2, 3)
    _launch(site, "b2", _spec("true", after={"run": "c2", "cond": "ok"}))
    assert site.wait(lambda: _run(site, "c2")["status"] == "fini", 20)
    assert site.wait(lambda: _run(site, "b2")["status"].startswith("en attente sans fin"), 10)
    assert "c2 a échoué" in _run(site, "b2")["status"]
    site.helper("stop", "~/shrink_runs", "b2")


def test_after_already_finished(site):
    _compile(site, "c3", 0, 4)
    assert site.wait(lambda: _run(site, "c3")["status"] == "fini", 20)
    r = _launch(site, "b3", _spec("true", after={"run": "c3", "cond": "ok"}))
    assert "a échoué (code 4)" in r["error"] and not (site.runs / "b3").exists()
    r = _launch(site, "b4", _spec("true", after={"run": "c3", "cond": "any"}))
    assert r.get("jobid") and "-E" not in (site.runs / "b4" / "job.sh").read_text()
    assert site.wait(lambda: _run(site, "b4")["status"] == "fini", 20)
    r = _launch(site, "b5", _spec("true", after={"run": "nope", "cond": "any"}))
    assert "n'existe pas" in r["error"]


def test_delay(site):
    _launch(site, "d1", _spec("true", delay_s=2))
    assert '#MSUB -E "--begin=now+2"' in (site.runs / "d1" / "job.sh").read_text()
    assert _run(site, "d1")["status"] == "en attente"
    assert site.wait(lambda: _run(site, "d1")["status"] == "fini", 20)


def test_delay_and_dependency_together(site):
    jc = _compile(site, "c6", 1, 0)
    _launch(site, "d2", _spec("true", after={"run": "c6", "cond": "any"}, delay_s=1))
    assert f'#MSUB -E "--dependency=afterany:{jc} --begin=now+1"' in (site.runs / "d2" / "job.sh").read_text()
    assert site.wait(lambda: _run(site, "d2")["status"] == "fini", 20)


def test_mprun_waits_for_the_run(site, monkeypatch):
    monkeypatch.setenv("SHRINK_GUI_POLL", "0.2")
    site.env["SHRINK_GUI_POLL"] = "0.2"
    _compile(site, "c4", 3, 0)
    _launch(site, "m1", _spec("true", mode="mprun", after={"run": "c4", "cond": "ok"}))
    assert site.wait(lambda: _run(site, "m1")["status"] == "en attente", 5)
    assert site.wait(lambda: _run(site, "m1")["status"] == "fini", 30)
    assert _run(site, "m1")["code"] == 0 and "waiting for the end of run c4" in _log(site, "m1")


def test_mprun_failed_run_not_started(site, monkeypatch):
    site.env["SHRINK_GUI_POLL"] = "0.2"
    _compile(site, "c5", 1, 2)
    _launch(site, "m2", _spec("echo STARTED", mode="mprun", after={"run": "c5", "cond": "ok"}))
    assert site.wait(lambda: _run(site, "m2")["status"] == "fini", 30)
    assert _run(site, "m2")["code"] == 95
    log = _log(site, "m2")
    assert "c5 failed: not started" in log and "STARTED" not in log


def test_mprun_stopped_while_waiting(site):
    site.env["SHRINK_GUI_POLL"] = "0.2"
    _launch(site, "m3", _spec("echo STARTED", mode="mprun", delay_s=60))
    assert site.wait(lambda: _run(site, "m3")["status"] == "en attente", 5)
    site.helper("stop", "~/shrink_runs", "m3")
    assert site.wait(lambda: _run(site, "m3")["status"] == "arrêté", 10)


# ------------------------------------------------------------ build check
def _build(site, name, stale):
    d = site.home / "br" / f"cmake-build-{name}" / f"RHEL_9.4__{platform.machine()}"
    (d / "examples" / "shrink_bench").mkdir(parents=True)
    (d / "build.ninja").write_text("")
    exe = d / "examples" / "shrink_bench" / "shrink_bench_chunk_dyn"
    exe.write_text("")
    exe.chmod(0o755)
    if stale:
        (d / "STALE").write_text("")
    return d


def _checked(site, rid, check):
    _launch(site, rid, _spec("echo STARTED", check=check))
    assert site.wait(lambda: _run(site, rid)["status"] == "fini", 20)
    return _run(site, rid)["code"], _log(site, rid)


def test_build_check(site):
    _build(site, "omp-gnu", stale=False)
    stale = _build(site, "omp-llvm", stale=True)
    root = str(site.home / "br")
    code, log = _checked(site, "k1", {"mode": "refuse", "root": root, "builds": ["omp-gnu"]})
    assert code == 0 and "1 build(s) up to date" in log and "STARTED" in log
    code, log = _checked(site, "k2", {"mode": "refuse", "root": root, "builds": ["cmake-build-omp-gnu", "omp-llvm"]})
    assert code == 96 and "1 of 2 build(s) NOT up to date" in log and str(stale) in log and "STARTED" not in log
    code, log = _checked(site, "k3", {"mode": "warn", "root": root, "builds": ["omp-llvm"]})
    assert code == 0 and "warning only" in log and "STARTED" in log
    lst = site.home / "builds.txt"
    lst.write_text("# a comment\nomp-gnu\n\nomp-llvm   # trailing\n")
    code, log = _checked(site, "k4", {"mode": "refuse", "root": root, "builds_file": str(lst)})
    assert code == 96 and "1 of 2" in log
    code, log = _checked(site, "k5", {"mode": "refuse", "root": root, "builds_file": "omp-gnu"})
    assert code == 0 and "1 build(s) up to date" in log
