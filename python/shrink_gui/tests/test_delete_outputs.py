# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""'Supprimer' with the program outputs: the run's own output directory
(multi_launch_stress.py), or the logs a run rewrites in a SHARED one
(bench_and_test.py, never oracle/), refused when another run owns them."""
import base64
import json

from shrink_gui import config
from test_remote_helper import _spec
from test_start_conditions import _launch, _run

MULTI = config.PROGRAMS["multi_launch_stress.py"]["own_outputs"]
BENCH = config.PROGRAMS["bench_and_test.py"]["own_outputs"]


def _delete(site, rid, own=None):
    opts = base64.b64encode(json.dumps({"own": own} if own else {}).encode()).decode()
    return site.helper("delete", "~/shrink_runs", rid, opts)


def _fake_run(site, rid, out_line, created, program="multi_launch_stress.py"):
    """A finished run whose log announces an output directory."""
    d = site.runs / rid
    d.mkdir(parents=True)
    meta = {"program": program, "out_re": config.PROGRAMS[program]["out_re"], "created": created}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "out.log").write_text(out_line + "\n")
    (d / "exit_code").write_text("0\n")
    return d


def test_own_output_dir(site):
    _launch(site, "r1", _spec("python3 fake_stress.py --runs 3"))
    assert site.wait(lambda: _run(site, "r1")["status"] == "fini")
    od = site.helper("tail", "~/shrink_runs", "r1", 0)["data"].split("Outputs, CSV and log in ")[1].split()[0]
    assert (site.home / "mlr").exists() and od.startswith(str(site.home))
    r = _delete(site, "r1", MULTI)
    assert r["ok"] and r["removed"] == [od]
    assert not (site.runs / "r1").exists()
    import os
    assert not os.path.exists(od)


def test_own_output_dir_shared_is_refused(site):
    od = site.home / "shared_out"
    od.mkdir()
    (od / "timings_summary.csv").write_text("x")
    _fake_run(site, "a1", f"📁 Outputs, CSV and log in {od}", "2026-10-08 10:00:00")
    _fake_run(site, "a2", f"📁 Outputs, CSV and log in {od}", "2026-10-08 09:00:00")
    r = _delete(site, "a1", MULTI)
    assert "a2 utilise aussi" in r["error"]
    assert (site.runs / "a1").exists() and od.exists()          # nothing removed
    assert _delete(site, "a1")["ok"] and od.exists()            # without the option: the run only


def test_own_output_dir_with_older_files_is_refused(site):
    # A log announcing a directory the run did not make (a mistyped option):
    # its files predate the submission, nothing of it goes.
    import os
    od = site.home / "Documents"
    (od / "notes").mkdir(parents=True)
    (od / "notes" / "keep.txt").write_text("precious")
    old = os.path.getmtime(od / "notes" / "keep.txt") - 3600
    os.utime(od / "notes" / "keep.txt", (old, old))
    _fake_run(site, "m1", f"📁 Outputs, CSV and log in {od}", "2026-10-08 10:00:00")
    r = _delete(site, "m1", MULTI)
    assert "antérieurs à m1" in r["error"] and "notes/keep.txt" in r["error"]
    assert (od / "notes" / "keep.txt").read_text() == "precious" and (site.runs / "m1").exists()


def test_own_output_dir_with_a_repository_is_refused(site):
    od = site.home / "work"
    (od / ".git").mkdir(parents=True)
    _fake_run(site, "g1", f"📁 Outputs, CSV and log in {od}", "2026-10-08 10:00:00")
    (od / "fresh.csv").write_text("x")
    assert "dépôt git" in _delete(site, "g1", MULTI)["error"] and (od / ".git").exists()


def _bench_dir(site):
    od = site.home / "bench_results" / "aarch64"
    for f in ("bench_output.log", "bench_errors.log", "failed_cases.jsonl", "bench_summary.csv"):
        (od).mkdir(parents=True, exist_ok=True)
        (od / f).write_text("x")
    for sub in ("run_logs", "runs/a/b", "oracle/cmake-build-omp-gnu"):
        (od / sub).mkdir(parents=True)
        (od / sub / "f").write_text("x")
    return od


def test_shared_bench_dir_keeps_the_oracle(site):
    import os
    od = _bench_dir(site)
    old = os.path.getmtime(od / "run_logs" / "f") - 3600
    for p in od.rglob("*"):                 # all written before b1 was submitted: the older runs'
        os.utime(p, (old, old))
    b1 = _fake_run(site, "b1", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 10:00:00", "bench_and_test.py")
    _fake_run(site, "b0", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 09:00:00", "bench_and_test.py")   # older
    os.utime(b1 / "meta.json", (old + 60, old + 60))
    # what b1 wrote: the files it rewrites, its logs, its replay's logs
    for f in ("bench_output.log", "bench_errors.log", "failed_cases.jsonl", "bench_summary.csv",
              "run_logs/x__omp4__00001__rc1.log", "run_logs/rerun-1/y.log", "runs/a/b/g"):
        (od / f).parent.mkdir(parents=True, exist_ok=True)
        (od / f).write_text("mine")
    r = _delete(site, "b1", BENCH)
    assert r["ok"] and len(r["removed"]) == 7
    assert sorted(p.name for p in od.iterdir()) == ["oracle", "run_logs", "runs"]
    assert sorted(str(p.relative_to(od)) for p in od.rglob("*") if p.is_file()) == [
        "oracle/cmake-build-omp-gnu/f", "run_logs/f", "runs/a/b/f"]     # the older runs' logs stay
    assert not (od / "run_logs" / "rerun-1").exists()                   # emptied: pruned


def test_shared_bench_dir_newer_run_is_refused(site):
    od = _bench_dir(site)
    _fake_run(site, "b1", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 10:00:00", "bench_and_test.py")
    _fake_run(site, "b2", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 11:00:00", "bench_and_test.py")
    r = _delete(site, "b1", BENCH)
    assert "b2, plus récent ou actif" in r["error"]
    assert (od / "bench_output.log").exists() and (site.runs / "b1").exists()


def test_no_output_dir_announced(site):
    _fake_run(site, "n1", "nothing here", "2026-10-08 10:00:00")
    assert "pas de dossier de sortie" in _delete(site, "n1", MULTI)["error"]


def test_delete_many_in_one_call(site):
    out_a = site.home / "out_a"
    out_a.mkdir()
    _fake_run(site, "m1", f"Outputs, CSV and log in {out_a}", "2026-10-16 10:00:00")
    (out_a / "timings_summary.csv").write_text("x\n")       # written by the run, after its submission
    _fake_run(site, "m2", "nothing announced", "2026-10-16 10:01:00")
    items = [{"id": "m1", "own": MULTI}, {"id": "m2"}, {"id": "nope"}, {"id": "../x"}]
    blob = base64.b64encode(json.dumps(items).encode()).decode()
    res = site.helper("delete_many", "~/shrink_runs", blob)["results"]
    assert res["m1"]["ok"] and str(out_a) in res["m1"]["removed"] and not out_a.exists()
    assert res["m2"] == {"ok": True, "removed": []}
    assert "n'est pas un run" in res["nope"]["error"] and "invalide" in res["../x"]["error"]
    assert not (site.runs / "m1").exists() and not (site.runs / "m2").exists()


def test_delete_many_keeps_outputs_that_are_not_the_runs_own(site):
    """Bulk delete with the outputs: a run with no output directory announced,
    or whose shared files a newer run rewrote, keeps them (with the reason)
    and is deleted all the same -- the newer one's files go with it."""
    od = site.home / "bench_results" / "aarch64"
    od.mkdir(parents=True)
    line = f"📁 Run outputs → {od}  (arch: aarch64)"
    _fake_run(site, "s1", line, "2026-10-07 10:00:00", program="bench_and_test.py")
    _fake_run(site, "s2", line, "2026-10-08 10:00:00", program="bench_and_test.py")
    import time
    time.sleep(0.05)
    for f in ("bench_output.log", "bench_summary.csv"):     # written by s2, the newer one
        (od / f).write_text("x\n")
    _fake_run(site, "s3", "no output line", "2026-10-08 11:00:00", program="bench_and_test.py")
    items = [{"id": r, "own": BENCH} for r in ("s2", "s3", "s1")]
    blob = base64.b64encode(json.dumps(items).encode()).decode()
    res = site.helper("delete_many", "~/shrink_runs", blob)["results"]
    assert all(res[r]["ok"] for r in ("s1", "s2", "s3")), res
    assert "s2" in res["s1"]["kept"] and "pas de dossier de sortie" in res["s3"]["kept"]
    assert "kept" not in res["s2"] and not (od / "bench_output.log").exists()
    assert not any((site.runs / r).exists() for r in ("s1", "s2", "s3"))
    # one run: strict without keep_others; with it (what the window asks), deleted, outputs kept
    _fake_run(site, "s4", "no output line", "2026-10-09 10:00:00", program="bench_and_test.py")
    assert "error" in _delete(site, "s4", BENCH) and (site.runs / "s4").exists()
    opts = base64.b64encode(json.dumps({"own": BENCH, "keep_others": True}).encode()).decode()
    r = site.helper("delete", "~/shrink_runs", "s4", opts)
    assert r["ok"] and "pas de dossier de sortie" in r["kept"] and not (site.runs / "s4").exists()


def test_own_output_dir_holding_another_runs_dir_is_refused(site):
    # a1 was given a base directory as its own; a2, later, writes below it
    # (thread_speedup.py --out <the default base>): a1's "whole directory" is
    # not a1's alone, however recent its files.
    od = site.home / "work" / "thread_speedup"
    (od / "sub").mkdir(parents=True)
    _fake_run(site, "a1", f"📁 Outputs, CSV and log in {od}", "2026-10-08 09:00:00")
    (od / "timings_summary.csv").write_text("x")
    _fake_run(site, "a2", f"📁 Outputs, CSV and log in {od / 'sub'}", "2026-10-08 10:00:00")
    (od / "sub" / "timings_summary.csv").write_text("y")
    r = _delete(site, "a1", MULTI)
    assert "a2 utilise aussi" in r["error"]
    assert (od / "sub" / "timings_summary.csv").exists() and (site.runs / "a1").exists()
    # the other way round, a2's own directory goes: a1 announces its parent, not it
    r = _delete(site, "a2", MULTI)
    assert r["ok"] and not (od / "sub").exists() and (od / "timings_summary.csv").exists()


def test_own_output_dir_with_files_written_after_the_run_is_refused(site):
    # ... and a run the GUI does not know (launched by hand) wrote there since.
    import os
    od = site.home / "work" / "base"
    od.mkdir(parents=True)
    d = _fake_run(site, "e1", f"📁 Outputs, CSV and log in {od}", "2026-10-08 09:00:00")
    (od / "timings_summary.csv").write_text("x")
    (od / "later").mkdir()
    (od / "later" / "result.csv").write_text("someone else's")
    end = os.path.getmtime(d / "exit_code") - 3600
    for p in (d / "meta.json", d / "exit_code", od / "timings_summary.csv"):
        os.utime(p, (end, end))                      # the run ended an hour ago
    r = _delete(site, "e1", MULTI)
    assert "écrits après la fin de e1" in r["error"] and "later/result.csv" in r["error"]
    assert (od / "later" / "result.csv").exists() and (site.runs / "e1").exists()


def test_shared_bench_dir_keeps_what_was_written_after_the_run(site):
    # b2, newer, was deleted WITHOUT its outputs: no run directory says they
    # are b2's any more. Deleting b1 with its outputs must not take them.
    import os
    od = _bench_dir(site)
    b1 = _fake_run(site, "b1", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 10:00:00", "bench_and_test.py")
    now = os.path.getmtime(b1 / "exit_code")
    for p in od.rglob("*"):
        os.utime(p, (now - 9000, now - 9000))        # the older runs' files
    os.utime(b1 / "meta.json", (now - 7200, now - 7200))     # b1 submitted two hours ago
    (od / "run_logs" / "b1.log").write_text("b1's")
    for p in (od / "run_logs" / "b1.log", b1 / "exit_code"):
        os.utime(p, (now - 3600, now - 3600))        # b1 ended an hour ago
    _fake_run(site, "b2", f"📁 Run outputs → {od}  (arch: aarch64)", "2026-10-08 11:00:00", "bench_and_test.py")
    (od / "run_logs" / "b2.log").write_text("b2's")
    (od / "bench_summary.csv").write_text("b2's")
    assert _delete(site, "b2")["ok"]                 # the run only
    r = _delete(site, "b1", BENCH)
    assert r["ok"] and r["removed"] == [str(od / "run_logs" / "b1.log")]
    assert (od / "run_logs" / "b2.log").read_text() == "b2's" and (od / "bench_summary.csv").read_text() == "b2's"


def test_a_run_that_may_still_run_is_not_deleted(site):
    # The helper checks by itself, whatever the window thought of the status.
    import os
    import subprocess
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        for rid, host in (("here", os.uname()[1]), ("elsewhere", "another-login-node")):
            d = site.runs / rid
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({"mode": "local", "created": "2026-10-08 09:00:00"}))
            (d / "pid").write_text(str(p.pid))
            (d / "host").write_text(host)
        force = base64.b64encode(json.dumps({"force": True}).encode()).decode()
        assert "encore actif" in _delete(site, "here")["error"]
        assert "encore actif" in site.helper("delete", "~/shrink_runs", "here", force)["error"]
        assert "tourne peut-être encore" in _delete(site, "elsewhere")["error"]
        blob = base64.b64encode(json.dumps([{"id": "here"}, {"id": "elsewhere"}]).encode()).decode()
        res = site.helper("delete_many", "~/shrink_runs", blob)["results"]
        assert "error" in res["here"] and "error" in res["elsewhere"]
        assert (site.runs / "here").exists() and (site.runs / "elsewhere").exists()
        assert site.helper("delete", "~/shrink_runs", "elsewhere", force)["ok"]      # asked twice: goes
        assert p.poll() is None
    finally:
        p.kill()
