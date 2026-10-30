# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""multi_launch_stress.py --pool ("Nœuds" N > 1): the jobs share the --runs,
each claiming the next free run index in the pool directory, and the first
failure stops them all. The helper's Résumé / Graphique / Erreurs gather the
jobs' outputs; the script itself is run twice at once on a fake build tree."""
import base64
import csv
import json
import os
import subprocess
import sys

from conftest import FAKE_PROGRAM, PYDIR

HEAD = ["run_index", "build", "executable", "num_threads", "wall_seconds", "returncode",
        "stdout_file", "stderr_file", "node"]


def _pool_job(runs, out, pool, k, nodes, rows, errors=()):
    """Job <pool>-n<k> of a submission over @p nodes nodes: its run dir (meta,
    log announcing its output directory), its timings_summary.csv and error files."""
    d = runs / f"{pool}-n{k}"
    d.mkdir(parents=True)
    meta = dict(FAKE_PROGRAM, program="multi_launch_stress.py", pool=pool, node=k, nodes=nodes)
    (d / "meta.json").write_text(json.dumps(meta))
    od = out / f"x86_64_n{k}"
    od.mkdir(parents=True)
    (d / "out.log").write_text(f"📁 Outputs, CSV and log in {od}\n")
    with open(od / "timings_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for i, b, rc in rows:
            w.writerow([i, b, "shrink_bench_omp_zero_one_value_dyn", 4, "1.5", rc, "", "", f"n{k}"])
    for name in errors:
        (od / name).write_text(f"boom in {name}\n")
    return d


def _call(rh, capsys, act, *args):
    getattr(rh, "act_" + act)(list(args))
    return json.loads(capsys.readouterr().out)


def test_pooled_summary_points_errors(tmp_path, capsys):
    from shrink_gui import remote_helper as rh
    runs, out = tmp_path / "runs", tmp_path / "out"
    # Two jobs, the run indices interleaved as the claims fell; job 2 failed run 6.
    _pool_job(runs, out, "P", 1, 2, [(1, "a", 0), (3, "a", 0), (4, "b", 0)])
    _pool_job(runs, out, "P", 2, 2, [(2, "b", 0), (5, "a", 0), (6, "b", 3)],
              errors=["error_6_b_shrink_bench_omp_zero_one_value_dyn.txt"])
    s = _call(rh, capsys, "summary", str(runs), "P-n1")
    assert s["rows"][-1][:3] == ["TOTAL", "6", "1"]
    assert {r[0]: r[1] for r in s["rows"]} == {"a": "3", "b": "3", "TOTAL": "6"}
    assert s["note"] == "timings_summary.csv — 6 run(s) — 2 nœuds sur 2"
    p = _call(rh, capsys, "points", str(runs), "P-n2")     # from either job
    assert [r[0] for r in p["rows"]] == [1, 2, 3, 4, 5, 6] and p["total"] == 6
    assert sorted(p["builds"]) == ["a", "b"] and p["rows"][-1][5] == 3
    blob = base64.b64encode(json.dumps({"files": ["error_*.txt"]}).encode()).decode()
    e = _call(rh, capsys, "errors", str(runs), "P-n1", blob)    # job 1's own directory has none
    assert [f["name"] for f in e["files"]] == ["error_6_b_shrink_bench_omp_zero_one_value_dyn.txt"]
    assert e["count"] == 1
    d1 = runs / "P-n1"
    meta = json.loads((d1 / "meta.json").read_text())
    assert rh.errors_from_csv(str(d1), meta) == [1, None]
    # Job 1 over, job 2 still running: the pool's count is not final, never cached.
    (d1 / "exit_code").write_text("1")
    assert rh.errors_of(str(d1), meta, "terminé (code 1)", []) == [1, None]
    assert not (d1 / "errors_count.json").exists()
    (runs / "P-n2" / "exit_code").write_text("3")
    assert rh.errors_of(str(d1), meta, "terminé (code 1)", []) == [1, None]
    assert json.loads((d1 / "errors_count.json").read_text()) == [1, None]
    # Outside a pool, the job's own CSV only, as before.
    alone = dict(meta)
    alone.pop("pool")
    assert rh.errors_from_csv(str(d1), alone) == [0, None]


def _fake_tree(root, calls, fail_at=0):
    """Two flat builds whose zero_one_value@dyn binary exits 0 at once --
    except the @p fail_at-th call over both builds and both jobs (counted by
    an atomic mkdir), which exits 3 -- and whose vector binary writes a data_tab_x."""
    body = ("#!/bin/bash\n"
            f"k=1; until mkdir \"{calls}/c_$k\" 2>/dev/null; do k=$((k+1)); done\n"
            "sleep 0.02\n"
            f"if [ $k -eq {fail_at} ]; then echo \"boom at call $k\" >&2; exit 3; fi\n")
    calls.mkdir()
    for b in ("omp-a", "omp-b"):
        d = root / f"cmake-build-{b}"
        d.mkdir(parents=True)
        (d / "shrink_bench_omp_zero_one_value_dyn").write_text(body)
        (d / "shrink_bench_omp_vector_check").write_text("#!/bin/bash\nsleep 0.3\necho x > data_tab_x\n")
        for f in d.iterdir():
            f.chmod(0o755)


def _run_pool(root, pool, runs, extra=()):
    """multi_launch_stress.py twice at once, the same --pool, as two
    shrink_gui jobs <pool>-n1 / -n2 (CHUNK_BENCH_KEEP_DIR names the job)."""
    procs = []
    for k in (1, 2):
        env = dict(os.environ, CHUNK_BENCH_KEEP_DIR=str(root / "gui" / f"{pool}-n{k}"))
        env.pop("CHUNK_BUILD_ROOT", None)
        procs.append(subprocess.Popen(
            [sys.executable, str(PYDIR / "multi_launch_stress.py"), "--builds", "omp-a,omp-b",
             "--build-root", str(root), "--runs", str(runs), "--min_threads", "1", "--max_threads", "2",
             "--numa-interleave", "off", "--pool", pool] + list(extra),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(root)))
    outs = [p.communicate(timeout=120)[0].decode() for p in procs]
    return [p.returncode for p in procs], outs


def _csv_rows(root):
    rows = []
    for k in (1, 2):
        (d,) = (root / "multi_launch_runs").glob(f"*_n{k}")
        with open(d / "timings_summary.csv") as f:
            r = list(csv.DictReader(f))
        assert all(x["node"] == f"n{k}" for x in r)
        rows += r
    return rows


def test_two_jobs_share_the_runs(tmp_path):
    root = tmp_path / "root"
    _fake_tree(root, tmp_path / "calls")
    codes, outs = _run_pool(root, "P1", 24, ["--regen-ref"])
    assert codes == [0, 0], outs
    idx = sorted(int(r["run_index"]) for r in _csv_rows(root))
    assert idx == list(range(1, 25))                     # all of them, none twice
    for k, o in zip((1, 2), outs):
        assert f"this job is n{k} (" in o and "[pool] P1: no run index left" in o
    # Each build's reference regenerated once, by the job that claimed it.
    assert sum(o.count("🧬 Regenerating the reference") for o in outs) == 2
    pool = root / "multi_launch_runs" / ".pools" / "P1"
    assert (pool / "ref_cmake-build-omp-a" / "done").exists() and not (pool / "stop").exists()
    # Output files named by the global index: no clash between the two directories.
    names = [p.name for p in (root / "multi_launch_runs").glob("*_n*/output_*.txt")]
    assert len(names) == len(set(names)) == 24


def test_a_failure_stops_both_jobs(tmp_path):
    root = tmp_path / "root"
    _fake_tree(root, tmp_path / "calls", fail_at=5)
    codes, outs = _run_pool(root, "P2", 200)
    assert sorted(codes) == [1, 3], outs        # the failing one: its code; the other stopped
    stop = root / "multi_launch_runs" / ".pools" / "P2" / "stop"
    assert stop.exists() and "return code 3" in stop.read_text()
    rows = _csv_rows(root)
    idx = [int(r["run_index"]) for r in rows]
    assert len(idx) == len(set(idx)) and len(idx) < 20     # stopped long before --runs 200
    assert [r["returncode"] for r in rows].count("3") >= 1
    assert "🛑 [pool] P2: stopped by " in outs[codes.index(1)]
    assert "stopped for every job" in outs[codes.index(3)]
