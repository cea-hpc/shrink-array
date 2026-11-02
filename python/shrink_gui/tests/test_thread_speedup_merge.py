# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""thread_speedup.py --merge: the output directories of the jobs of a pool
(each the curves of the executables it claimed) merged into one, as one run
over them all would have written it. The directories are written here by hand,
as thread_speedup.py writes them; the merge runs under the interpreter of the
tests and, when it is another one, under the system's python3 (3.6 on the
cluster)."""
import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import PYDIR

SCRIPT = PYDIR / "thread_speedup.py"
SUMMARY_HEADER = ["target", "exe", "threads", "ok_runs", "best_s", "median_s", "best_wall_s",
                  "best_compute_s", "peak_mem_mo", "speedup", "efficiency"]
RESULTS_HEADER = ["target", "exe", "threads", "rep", "wall_s", "compute_s", "mem_mo", "rc"]
PLAN = ["shrink_bench_omp_vector_perf", "shrink_bench_omp_zero_dyn", "shrink_bench_omp_zero_one_value_dyn",
        "shrink_bench_omp_zero_any_dyn"]


def fake_run(root, host, execs, base, cpu="Neoverse-V2", multiply=2.0):
    """The output directory of one job of a pool on @p host: @p execs measured
    at 1, 2, 4 threads, a time of base / threads (a perfect speedup)."""
    d = root / ("speedup_omp-llvm_%s_20261102_120000" % host)
    d.mkdir()
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        for e in execs:
            for t in (1, 2, 4):
                v = "%.4f" % (base / t)
                w.writerow(["omp-llvm", e, t, 1, v, v, v, v, "10.0", "%.3f" % t, "1.000"])
    with open(d / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESULTS_HEADER)
        for e in execs:
            for t in (1, 2, 4):
                w.writerow(["omp-llvm", e, t, 1, "%.4f" % (base / t), "%.4f" % (base / t), "10.0", 0])
    meta = {"command": "python3 thread_speedup.py --target omp-llvm --all-exec --pool=p1", "host": host + ".inti",
            "cpu": cpu, "arch": "aarch64", "target": "omp-llvm", "execs": {e: "/br/" + e for e in PLAN},
            "threads": [1, 2, 4], "reps": 1, "multiply": multiply, "mode": "perf", "metric": "compute_s",
            "numa": "off", "bench_args": ["--mode", "perf", "--no_io", "--multiply", "%g" % multiply],
            "pool": "p1"}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def interpreters():
    """This one, and the system's python3 when it is another (3.6 on the cluster)."""
    out = [sys.executable]
    other = shutil.which("python3")
    if other and os.path.realpath(other) != os.path.realpath(sys.executable):
        out.append(other)
    return out


@pytest.mark.parametrize("python", interpreters())
def test_merge_of_three_pool_jobs(tmp_path, python):
    # zero_any_dyn measured twice (a claim given twice): kept from the first
    # directory given, n3's left out.
    a = fake_run(tmp_path, "inti5012", ["shrink_bench_omp_zero_dyn", "shrink_bench_omp_zero_any_dyn"], 8.0)
    b = fake_run(tmp_path, "inti5013", ["shrink_bench_omp_vector_perf"], 4.0)
    c = fake_run(tmp_path, "inti5014", ["shrink_bench_omp_zero_one_value_dyn", "shrink_bench_omp_zero_any_dyn"],
                 6.0, cpu="AMD EPYC 7763", multiply=1.0)
    out = tmp_path / "merged"
    p = subprocess.run([python, str(SCRIPT), "--merge", str(a), str(b), str(c), "--out", str(out)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert p.returncode == 0, p.stderr
    err = p.stderr
    assert "shrink_bench_omp_zero_any_dyn is in 2 directories, kept from the first: %s (left out: %s)" % (a, c) in err
    assert "different CPU models" in err and "AMD EPYC 7763" in err
    assert "different workloads (--multiply)" in err and "different executable arguments" in err
    assert "different thread lists" not in err and "different build targets" not in err

    rows = list(csv.reader(open(out / "summary.csv")))
    assert rows[0] == SUMMARY_HEADER
    # summarize()'s order: executable, then threads ascending
    assert [(r[1], r[2]) for r in rows[1:]] == [(e, t) for e in sorted(PLAN) for t in ("1", "2", "4")]
    zero_any = [r[4] for r in rows[1:] if r[1] == "shrink_bench_omp_zero_any_dyn"]
    assert zero_any == ["8.0000", "4.0000", "2.0000"]                # a's, not c's 6.0 / t
    res = list(csv.reader(open(out / "results.csv")))
    assert res[0] == RESULTS_HEADER and len(res) == 1 + 4 * 3

    meta = json.loads((out / "meta.json").read_text())
    assert meta["host"] == "inti5012+inti5013+inti5014" and meta["threads"] == [1, 2, 4]
    assert [m["dir"] for m in meta["merge"]] == [str(a), str(b), str(c)]
    assert meta["merge"][2]["execs"] == ["shrink_bench_omp_zero_one_value_dyn"]
    gp = (out / "plot.gp").read_text()
    assert "omp-llvm -- inti5012+inti5013+inti5014 (perf, --multiply 2, compute time)" in gp
    # the gnuplot curves in the runs' plan order: vector first
    assert [e for e in PLAN if '"%s"' % e in gp] == PLAN
    assert gp.index("omp_vector_perf") < gp.index("omp_zero_dyn") < gp.index("omp_zero_any_dyn")
    has_mpl = subprocess.run([python, "-c", "import matplotlib"], stderr=subprocess.DEVNULL).returncode == 0
    if has_mpl:
        assert (out / "speedup.png").stat().st_size > 1000
        assert "speedup.png" in p.stdout
    else:
        assert not (out / "speedup.png").exists() and "gnuplot plot.gp -> speedup.png" in p.stdout

    # a second merge into the same directory is refused, nothing overwritten
    p = subprocess.run([python, str(SCRIPT), "--merge", str(a), "--out", str(out)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert p.returncode != 0 and "exists and is not empty" in p.stderr


def test_merge_refuses_a_directory_without_summary(tmp_path):
    a = fake_run(tmp_path, "inti5012", ["shrink_bench_omp_zero_dyn"], 8.0)
    empty = tmp_path / "empty"
    empty.mkdir()
    p = subprocess.run([sys.executable, str(SCRIPT), "--merge", str(a), str(empty), "--out", str(tmp_path / "m")],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert p.returncode != 0
    assert "%s: no summary.csv" % empty in p.stderr and "(given: " in p.stderr
    assert not (tmp_path / "m").exists()


def test_host_label():
    sys.path.insert(0, str(PYDIR))
    import thread_speedup
    assert thread_speedup.host_label(["inti5012.inti", "inti5013", "inti5013.x"]) == "inti5012+inti5013"
    assert thread_speedup.host_label(["n%d" % k for k in range(6)]) == "6 nodes"
    assert thread_speedup.host_label(["", None]) == "merged runs"
