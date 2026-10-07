"""remote_helper.py, run as on inti (through the fake ssh), without Qt."""
import ast
import base64
import json
import subprocess
import tarfile
import io

import pytest

from conftest import FAKE_PROGRAM, PKG, REPO
from shrink_gui.runscripts import make_run_sh

HELPER = (PKG / "remote_helper.py").read_text()


def test_helper_is_python36():
    # The system python3 of a RHEL 8 login node.
    ast.parse(HELPER, feature_version=(3, 6))


def test_check(site):
    r = site.helper("check", "~/shrink-array")
    assert r["host"] and r["python"]
    assert r["msub"] is True                      # the fake ccc_msub is on the PATH
    assert "git" in r


def test_options_multi_launch_stress(site):
    r = site.helper("options", "~/shrink-array", "python/multi_launch_stress.py", "argparse", "python3")
    dests = {a["dest"] for a in r["actions"]}
    assert {"regen_ref", "builds", "min_threads", "max_threads", "all_exec"} <= dests


def test_options_bench_and_test(site):
    r = site.helper("options", "~/shrink-array", "python/bench_and_test.py", "argparse", "python3")
    dests = {a["dest"] for a in r["actions"]}
    assert len(dests) > 40 and "output_dir" in dests


def test_options_compile_all_from_help(site):
    if not (REPO / "scripts/compile_all.sh").exists():
        pytest.skip("no scripts/compile_all.sh")
    r = site.helper("options", "~/shrink-array", "scripts/compile_all.sh", "help", "bash")
    by_flag = {a["flags"][0]: a for a in r["actions"]}
    assert "--dryrun" in by_flag and by_flag["--dryrun"]["kind"] == "_StoreTrueAction"
    assert by_flag["--jobs"]["eq"] is True       # --jobs=N


def test_shell_script_is_read_not_run(site):
    # The login node must not run the script: its usage() text is parsed.
    sh = site.home / "build.sh"
    sh.write_text('#!/bin/bash\ntouch "$HOME/EXECUTED"\nusage() {\n    cat >&2 <<EOF\n'
                  'Usage: $0 [--fast]\n\n  --fast        Go fast.\n  --jobs=N      N at once,\n'
                  '                continued here.\nEOF\n}\nusage\n')
    r = site.helper("options", "~", "build.sh", "help", "bash")
    flags = {a["flags"][0]: a for a in r["actions"]}
    assert set(flags) == {"--fast", "--jobs"} and flags["--jobs"]["eq"]
    assert "continued here" in flags["--jobs"]["help"]
    assert not (site.home / "EXECUTED").exists()
    (site.home / "nousage.sh").write_text("#!/bin/bash\necho hi\n")
    out = subprocess.run([str(site.bin / "ssh"), "inti", "python3 - options ~ nousage.sh help bash"],
                         input=HELPER.encode(), env=site.env, capture_output=True, timeout=60)
    assert "usage()" in json.loads(out.stdout)["error"]


def test_options_error_is_reported(site, tmp_path):
    bad = site.home / "bad.py"
    bad.write_text("import argparse\nraise RuntimeError('broken before parse_args')\n")
    out = subprocess.run([str(site.bin / "ssh"), "inti", "python3 - options ~ bad.py argparse python3"],
                         input=HELPER.encode(), env=site.env, capture_output=True, timeout=60)
    r = json.loads(out.stdout)
    assert "RuntimeError" in r["error"] and "bad.py" in r["error"]


def test_git_info(site, tmp_path):
    repo = site.home / "g"
    repo.mkdir()
    env = dict(site.env, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    for c in (["git", "init", "-q"], ["git", "checkout", "-q", "-b", "dev"]):
        subprocess.run(c, cwd=repo, env=env, check=True)
    (repo / "f").write_text("1")
    subprocess.run(["git", "add", "f"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=repo, env=env, check=True)
    g = site.helper("check", "~/g")["git"]
    assert len(g["commit"]) == 40 and g["branch"] == "dev" and g["dirty"] is False
    (repo / "f").write_text("2")
    assert site.helper("check", "~/g")["git"]["dirty"] is True


def _spec(command, mode="msub", repo="~/shrink-array"):
    return {"mode": mode, "run.sh": make_run_sh(repo, FAKE_PROGRAM["cwd"], command),
            "msub": "#MSUB -q fake\n", "launcher": "", "launch.sh": "", "name": "t", "time_s": 600,
            "meta": dict({k: FAKE_PROGRAM[k] for k in ("out_re", "out_files", "progress_re", "eta_re",
                                                       "summary", "summary_file")},
                         program="fake_stress.py", partition="gh200-bxi", mode=mode, repo=repo,
                         command=command)}


def _launch(site, rid, command):
    blob = base64.b64encode(json.dumps(_spec(command)).encode()).decode()
    return site.helper("launch", "~/shrink_runs", rid, blob)


def _status(site, rid):
    runs = {r["id"]: r for r in site.helper("list", "~/shrink_runs")["runs"]}
    return runs.get(rid)


def test_msub_lifecycle(site):
    r = _launch(site, "r1", "python3 fake_stress.py --runs 12 --step 0.01")
    assert r["jobid"]
    assert site.wait(lambda: _status(site, "r1")["status"] == "fini")
    run = _status(site, "r1")
    assert run["code"] == 0
    assert run["progress"][:2] == [12, 12]
    log = site.helper("tail", "~/shrink_runs", "r1", 0)["data"]
    assert "Run 12/12" in log and "exit code 0" in log
    s = site.helper("summary", "~/shrink_runs", "r1")
    assert s["columns"][0] == "build" and s["rows"][-1][0] == "TOTAL" and s["rows"][-1][1] == "12"
    pts = site.helper("points", "~/shrink_runs", "r1")
    assert pts["total"] == 12 and len(pts["rows"]) == 12
    tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(site.helper("pack", "~/shrink_runs", "r1",
                                                                      binary=True))))
    names = set(tar.getnames())
    assert "run/out.log" in names and "outputs/timings_summary.csv" in names
    assert site.helper("delete", "~/shrink_runs", "r1")["ok"]
    assert _status(site, "r1") is None


def test_failure_code_and_stop(site):
    _launch(site, "bad", "python3 fake_stress.py --runs 2 --rc 3")
    assert site.wait(lambda: _status(site, "bad")["status"] == "fini")
    assert _status(site, "bad")["code"] == 3
    _launch(site, "long", "python3 fake_stress.py --runs 500 --step 0.5")
    assert site.wait(lambda: _status(site, "long")["status"] == "en cours")
    assert site.helper("stop", "~/shrink_runs", "long")["ok"]
    assert site.wait(lambda: _status(site, "long")["status"] == "arrêté")


def test_events_progress_and_phases(site):
    d = site.runs / "b1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"program": "bench_and_test.py", "summary": "csv",
                                             "summary_file": "bench_summary.csv",
                                             "out_re": r"Run outputs \S+ (\S+)"}))
    (d / "exit_code").write_text("0")
    t = 1000.0
    ev = [{"t": t, "kind": "start", "total_runs": 40},
          {"t": t + 1, "kind": "phase_start", "phase": "ctest"},
          {"t": t + 5, "kind": "phase_end", "phase": "ctest", "status": "errors", "errors": 2, "seconds": 4},
          {"t": t + 5, "kind": "phase_start", "phase": "check"},
          {"t": t + 15, "kind": "progress", "phase": "check", "done": 10, "total": 40}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n{partial")
    run = _status(site, "b1")
    done, total, eta, phase = run["progress"]
    assert (done, total, phase) == (10, 40, "check") and round(eta) == 45
    s = site.helper("summary", "~/shrink_runs", "b1")
    phases = s["sections"][0]
    assert phases["title"] == "Phases"
    assert phases["rows"][0][:2] == ["ctest", "erreurs"] and phases["fail_rows"] == [0]
    assert phases["rows"][1][:2] == ["check", "en cours"]


def test_run_sh_is_valid_bash():
    sh = make_run_sh("~/shrink-array", "python", "python3 x.py --a 1")
    assert "CHUNK_BENCH_EVENTS" in sh and "git_start" in sh
    subprocess.run(["bash", "-n"], input=sh.encode(), check=True)
