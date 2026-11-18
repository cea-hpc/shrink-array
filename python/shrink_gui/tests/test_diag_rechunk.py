# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""diag_zero_rechunk_slowdown.py on a fake bench: its memory and per-array churn
tables, read from the iter and churn lines."""
import subprocess
import sys

from conftest import PYDIR

ITERS = 'for k in 100 200 300 400 500; do echo "iter $k : time : 1"; sleep 0.02; done\n'
DYN = """#!/bin/bash
echo "iter 100 : time : 1"
echo "[shrink_array][rechunk][D][0] 64 -> 1024 phases"
for k in 200 300 400 500; do
    echo "iter $k : time : 1"
    echo "[shrink_array][rechunk][D][0] churn v1 S=1024: $((k * 10)) materialisations over 10 phases since the last \
tick (0/phase), resident materialised 7 -> 5: transient ~1 chunks/phase = 1.0 MB at S=1024; at S=64: 1.0 MB if \
scattered, 1.0 MB if grouped"
    echo "[shrink_array][rechunk][D][0] churn v2 S=1024: 0 materialisations over 10 phases since the last tick \
(0/phase), resident materialised 3 -> 3: transient ~0 chunks/phase = 0.0 MB at S=1024; at S=64: 0.0 MB if \
scattered, 0.0 MB if grouped"
    sleep 0.02
done
"""


def _exe(d, name, body):
    p = d / name
    p.write_text(body)
    p.chmod(0o755)


def test_memory_and_churn_tables(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir, "shrink_bench_omp_zero_1024", "#!/bin/bash\n" + ITERS)
    _exe(bindir, "shrink_bench_omp_zero_dyn", DYN)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(PYDIR / "diag_zero_rechunk_slowdown.py"), "--bindir", str(bindir),
                        "--out", str(out), "--configs", "static,dyn"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    text = (out / "summary.txt").read_text()
    assert "resident memory after the last re-chunk" in text
    rows = [ln.split() for ln in text.splitlines() if ln.startswith(("v1 ", "v2 "))]
    # v1 first, the churning one. Its line at k=200 spans the re-chunk: left out.
    assert [row[0] for row in rows] == ["v1", "v2"]
    assert rows[0][1:4] == ["1024", "40", "400"] and rows[0][4:7] == ["5", "->", "5"]
    assert (out / "dyn_r0" / "churn.csv").read_text().count("\n") == 9
    assert "rss_mb" in (out / "dyn_r0" / "iterations.csv").read_text()
