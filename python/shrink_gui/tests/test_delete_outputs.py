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
