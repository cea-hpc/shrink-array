# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The site file (sitecfg.py, sites/): reading, checks, and a plain Slurm
site (sbatch, #SBATCH, scancel) from the helper to the window."""
import base64
import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE_PROGRAM, PKG, PYDIR
from shrink_gui import i18n, qos, sitecfg
from shrink_gui.runscripts import make_run_sh

SITES = PKG / "sites"
SLURM = SITES / "slurm_example.toml"


@pytest.mark.parametrize("name", ["cea_inti.toml", "slurm_example.toml"])
def test_subset_reader_matches_tomllib(name):
    tomllib = pytest.importorskip("tomllib")
    text = (SITES / name).read_text()
    assert sitecfg._Reader(text, name).parse() == tomllib.loads(text)


def test_cea_site_is_the_former_config():
    s = sitecfg.load(SITES / "cea_inti.toml")
    assert (s.name, s.host, s.partition) == ("inti", "inti", "gh200-bxi")
    gh = s.partitions["gh200-bxi"]
    assert gh["directives"] == "#MSUB -q gh200-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 288\n"
    assert gh["launcher"] == "ccc_mprun -C ccc-rhel9 -n 1 -c 288"
    assert s.defaults[("bench_and_test.py", "arm")]["output_dir"] == \
        "$CCCWORKDIR/shrink-array/bench_results"
    assert s.detached_options(gh["detached"], 600, "long") == \
        "-N1 -c 288 -p gh200-bxi -C ccc-rhel9 -Xfirst -T600 -Q long"
    rows = qos.parse_mqinfo(s.qos_fallback)
    assert qos.limits(rows, "long", "gh200-bxi")["max_time"] == 3 * 86400


def test_site_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SHRINK_GUI_SITE")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert sitecfg.site_path() == sitecfg.BUNDLED           # no user file
    (tmp_path / "shrink_gui").mkdir()
    (tmp_path / "shrink_gui" / "site.toml").write_text(SLURM.read_text())
    assert sitecfg.load().name == "cluster"
    monkeypatch.setenv("SHRINK_GUI_SITE", str(SITES / "cea_inti.toml"))
    assert sitecfg.load().name == "inti"


@pytest.mark.parametrize("change, key", [
    (("[site]\n", "[site]\ncolour = 1\n"), "site.colour"),                   # unknown key
    (('submit = "sbatch"\n', ""), "batch.submit"),                          # missing key
    (('time = "--time={minutes}"', 'time = "--time={hours}"'), "batch.time"),  # unknown field
    (("jobid_re = 'Submitted batch job (\\d+)'", "jobid_re = 'job \\d+'"), "batch.jobid_re"),  # no group
    (('partition = "cpu"', 'partition = "gpu"'), "site.partition"),         # not a partition
    (('arch = "x86"\n', ""), "partitions.cpu.arch"),
    (('cancel = "scancel"', "cancel = []"), "batch.cancel"),
    (('slurm = "{option}"', "slurm = '{option} {options}'"), "batch.slurm"),
])
def test_invalid_site_names_the_key(tmp_path, change, key):
    text = SLURM.read_text()
    assert change[0] in text
    f = tmp_path / "site.toml"
    f.write_text(text.replace(change[0], change[1], 1))
    with pytest.raises(sitecfg.SiteError) as e:
        sitecfg.load(f)
    assert str(f) in str(e.value) and key in str(e.value)


@pytest.mark.parametrize("bad", ['name = "unterminated\n', "name = \n", "[site\n", "a = 1 2\n"])
def test_toml_syntax_errors(tmp_path, bad):
    f = tmp_path / "site.toml"
    f.write_text(bad)
    with pytest.raises(sitecfg.SiteError) as e:
        sitecfg.load(f)
    assert str(f) in str(e.value)
    with pytest.raises(sitecfg.SiteError):                  # the subset reader too
        sitecfg._Reader(bad, "x").parse()


def test_missing_site_file(tmp_path):
    with pytest.raises(sitecfg.SiteError) as e:
        sitecfg.load(tmp_path / "none.toml")
    assert "illisible" in str(e.value)


def test_qos_reads_sbatch_directives():
    d = "#SBATCH --partition=cpu\n#SBATCH --ntasks=2\n#SBATCH --cpus-per-task=32\n#SBATCH --qos=long\n"
    assert qos.job_cpus(d) == 64 and qos.explicit_qos(d) == "long"
    assert qos.job_cpus("#MSUB -n 1\n#MSUB -c 288\n") == 288 and qos.explicit_qos("#MSUB -Q test") == "test"
    sacct = ("      Name   Priority     MaxWall MaxSubmitPU \n---------- ---------- ----------- ----------- \n"
             "    normal         20  1-00:00:00         100 \n      test         40    02:00:00           2 \n")
    rows = qos.parse_mqinfo(sacct)
    assert qos.limits(rows, "test", "cpu")["max_sub"] == 2
    assert qos.choose(rows, "cpu", 64, 3 * 3600)[0] == "normal"


def test_qos_star_row_with_max_sub_0_is_closed():
    # inti's ccc_mqinfo: long is open on gh200-bxi only, xlong on milan-bxi
    # ("long" on milan-bxi: "Invalid qos specification").
    mq = ("Name     Partition  Priority  MaxCPUs  SumCPUs  MaxNodes  MaxRun  MaxSub        MaxTime\n"
          "-------  ---------  --------  -------  -------  --------  ------  ------  -------------\n"
          "long             *        20        0        0                         0     3-00:00:00\n"
          "long     gh200-bxi        20     2880     2880                       100     3-00:00:00\n"
          "normal           *        20                                         100     1-00:00:00\n"
          "xlong            *        20        0        0                         0   365-00:00:00\n"
          "xlong    milan-bxi        20      128      128                       100   365-00:00:00\n")
    rows = qos.parse_mqinfo(mq)
    assert qos.limits(rows, "long", "milan-bxi") is None
    assert qos.limits(rows, "normal", "milan-bxi")["max_sub"] == 100
    assert qos.choose(rows, "milan-bxi", 64, 2 * 86400)[0] == "xlong"
    assert qos.choose(rows, "gh200-bxi", 288, 2 * 86400)[0] == "long"


def test_site_words():
    i18n.set_site_words(sitecfg.load(SLURM).words())
    try:
        assert i18n.T("Lignes #MSUB du script batch (ccc_msub) sur inti") == \
            "Lignes #SBATCH du script batch (sbatch) sur cluster"
        assert i18n.T("pas intitulé ccc_msubx") == "pas intitulé ccc_msubx"      # whole words only
    finally:
        i18n.set_site_words({})


# ------------------------------------------------------- a plain Slurm site
def _slurm_spec(command, rid_after=None, delay_s=0):
    batch = sitecfg.load(SLURM).batch
    return {"mode": "msub", "run.sh": make_run_sh("~/shrink-array", FAKE_PROGRAM["cwd"], command),
            "msub": "#SBATCH --partition=cpu\n", "launcher": "", "launch.sh": "", "name": "t", "time_s": 90,
            "qos": "normal", "batch": batch,
            "after": {"run": rid_after, "cond": "ok"} if rid_after else None, "delay_s": delay_s,
            "meta": {"program": "fake_stress.py", "partition": "cpu", "mode": "msub", "repo": "~/shrink-array",
                     "command": command}}


def _run(site, rid):
    return {r["id"]: r for r in site.helper("list", "~/shrink_runs")["runs"]}.get(rid)


def test_slurm_site_helper(site):
    arg = base64.b64encode(json.dumps(sitecfg.load(SLURM).batch).encode()).decode()
    r = site.helper("check", "~/shrink-array", arg)
    assert r["msub"] is True                         # the fake sbatch is on the PATH
    blob = base64.b64encode(json.dumps(_slurm_spec("python3 fake_stress.py --runs 3 --step 0.01")).encode())
    jobid = site.helper("launch", "~/shrink_runs", "s1", blob.decode())["jobid"]
    job = (site.runs / "s1" / "job.sh").read_text()
    assert "#SBATCH --partition=cpu\n#SBATCH --job-name=t\n#SBATCH --time=2\n#SBATCH --output=" in job
    assert "#SBATCH --qos=normal\n" in job and "#MSUB" not in job and "--error" not in job
    assert site.wait(lambda: _run(site, "s1")["status"] == "fini")
    assert _run(site, "s1")["code"] == 0 and (site.runs / "s1" / "jobid").read_text() == jobid
    # Start conditions: one directive line each.
    blob = base64.b64encode(json.dumps(_slurm_spec("sleep 30", rid_after="s1", delay_s=60)).encode())
    site.helper("launch", "~/shrink_runs", "s2", blob.decode())
    job = (site.runs / "s2" / "job.sh").read_text()
    assert "#SBATCH --begin=now+60\n" in job                # s1 is done: no dependency, only the delay
    assert site.wait(lambda: _run(site, "s2")["status"].startswith("en attente"), 10)
    assert site.helper("stop", "~/shrink_runs", "s2", arg)["ok"]      # scancel
    assert site.wait(lambda: _run(site, "s2")["status"] == "arrêté", 20)


def test_slurm_site_window(site, tmp_path):
    """The window on the Slurm site: its names, one submission mode, no
    detached one; then a run submitted with sbatch."""
    code = f"""
import os, sys, time
sys.path.insert(0, {str(PYDIR)!r})
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
app = QApplication([])
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, {str(tmp_path / 'settings')!r})
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, {str(tmp_path / 'settings')!r})
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
from shrink_gui import app as app_mod, config, i18n, remote
i18n.set_lang("fr")
remote.SSH = {str(site.bin / 'ssh')!r}
w = app_mod.Main()
app.main_window = w
w.show()
t0 = time.time()
while "connecté" not in w.conn.text() and time.time() - t0 < 30:
    app.processEvents(); time.sleep(0.03)
modes = [w.mode.itemText(i) for i in range(w.mode.count())]
print("RESULT", repr((w.windowTitle(), w.host.text(), modes, w.preset.currentData(),
                      w.directives.toPlainText(), w.conn.text(), w.repo_label.text())))
"""
    env = dict(os.environ, SHRINK_GUI_SITE=str(SLURM), QT_QPA_PLATFORM="offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, res.stdout + res.stderr
    title, host, modes, preset, directives, conn, repo_label = eval(line[0][7:])
    assert "cluster" in title and host == "cluster" and repo_label == "Dépôt (cluster)"
    assert modes == ["sbatch (batch)"] and preset == "cpu"
    assert directives.startswith("#SBATCH --partition=cpu")
    assert "connecté" in conn and "introuvable" not in conn   # sbatch found
