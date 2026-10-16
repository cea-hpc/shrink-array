# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The window, end to end, offscreen, against the fake inti."""
import base64
import json
import re
import time

import pytest

pytest.importorskip("PySide6")


def _submit(w, pump, values, n_before):
    w.form.set_values(values)
    time.sleep(1.05)               # run ids are per second
    w.launch()
    assert pump(lambda: len(w.runs) == n_before + 1, 20)
    return w.runs[0]["id"]


def _wait_status(w, pump, rid, status, timeout=40):
    def ok():
        w.refresh()
        run = next((r for r in w.runs if r["id"] == rid), None)
        return run is not None and run["status"] == status
    return pump(ok, timeout)


def _choose_fake(w, pump):
    w.program.setCurrentText("fake_stress.py")
    assert pump(lambda: w.form.schema is not None and w.form.schema["description"] == "fake")


def test_options_and_defaults(gui):
    w, pump = gui
    w.program.setCurrentText("multi_launch_stress.py")
    assert pump(lambda: w.form.schema is not None and "regen_ref" in w.form.widgets)
    cmd = w.preview.toPlainText()
    assert "--builds build_arm.txt" in cmd and "--regen-ref" in cmd
    w.preset.setCurrentIndex(w.preset.findData("milan-bxi"))
    assert "--builds build_x86_64.txt" in w.preview.toPlainText()
    w.opt_filter.setText("regen")
    visible = [d for d, (_a, wd, _l) in w.form.widgets.items() if w.form.lay.isRowVisible(wd)]
    assert visible == ["regen_ref"]


def test_submit_follow_summary_fetch(gui, site):
    w, pump = gui
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "30", "step": "0.01", "fail_every": "11"}, 0)
    assert _wait_status(w, pump, rid, "fini")
    w.table.selectRow(0)
    assert pump(lambda: "exit code 0" in w.log.toPlainText(), 10)
    w.tabs.setCurrentIndex(1)
    assert pump(lambda: w.sum_table.rowCount() == 4, 10)        # 3 builds + TOTAL
    w.tabs.setCurrentIndex(2)
    assert pump(lambda: len(w.pt_view.points) == 30, 10)
    w.tabs.setCurrentIndex(w.TAB_ERRORS)                         # runs 11 and 22 failed
    assert pump(lambda: w.err_note.text().startswith("2 erreur(s)"), 10)
    assert "run 22 failed: code 1" in w.err_view.toPlainText()
    assert w.tabs.tabBar().tabTextColor(w.TAB_ERRORS).name() == "#c0392b"
    w.fetch()
    assert pump(lambda: (site.local / rid / "outputs" / "timings_summary.csv").exists(), 10)


def test_compare_two_runs(gui):
    w, pump = gui
    _choose_fake(w, pump)
    a = _submit(w, pump, {"runs": "300", "step": "0.001"}, 0)
    assert _wait_status(w, pump, a, "fini")
    b = _submit(w, pump, {"runs": "300", "step": "0.001", "speed": "0.87"}, 1)
    assert _wait_status(w, pump, b, "fini")
    w.table.selectRow(1)                                        # a (older) as the base
    w.tabs.setCurrentIndex(2)
    assert pump(lambda: w.pt_view.points, 10)
    w.pt_cmp.setCurrentIndex(w.pt_cmp.findData(b))
    assert pump(lambda: "×0.870" in w.pt_note.text(), 10), w.pt_note.text()


def test_stop(gui):
    w, pump = gui
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "500", "step": "0.5"}, 0)
    assert _wait_status(w, pump, rid, "en cours")
    w.table.selectRow(0)
    w.stop()
    assert _wait_status(w, pump, rid, "arrêté")


def _french_left(w):
    """The visible texts, entries and tooltips of the window still in French."""
    from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton
    fr = re.compile(r"[éèêàùçôîœ«»]|\b(les|des|du|une|pas|aucun|tous|toutes|lire|relit|rafraîchir|"
                    r"arrêter|supprimer|récupérer|résumé|commande|défaut|reste|sur inti)\b", re.I)
    left = []
    for cls, get in ((QLabel, "text"), (QPushButton, "text"), (QCheckBox, "text"), (QGroupBox, "title")):
        for x in w.findChildren(cls):
            t = re.sub("<[^>]+>", "", getattr(x, get)())
            if x.isVisibleTo(w) and fr.search(t) and not t.startswith("--"):
                left.append(t)
    for x in w.findChildren(QLineEdit):
        if x.isVisibleTo(w) and fr.search(x.placeholderText()):
            left.append(x.placeholderText())
    for x in w.findChildren(QComboBox):
        left += [x.itemText(i) for i in range(x.count())
                 if fr.search(x.itemText(i)) and x.itemText(i) != "Français"]
    for x in w.findChildren(QPushButton) + w.findChildren(QComboBox):
        t = re.sub("<[^>]+>", "", x.toolTip())
        if fr.search(t):
            left.append("tooltip: " + t[:60])
    return left


def test_english_has_no_french_left(gui, qapp):
    w, pump = gui
    w.lang.setCurrentIndex(w.lang.findData("en"))
    w = qapp.main_window
    assert pump(lambda: "connected" in w.conn.text())
    assert _french_left(w) == []


def test_commit_column(gui, site):
    import subprocess
    w, pump = gui
    repo = site.home / "gitrepo"
    (repo / "python").mkdir(parents=True)
    env = dict(site.env, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True)
    (repo / "README").write_text("x")
    subprocess.run(["git", "add", "README"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=repo, env=env, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    # the fake program must be reachable from that repository too
    (repo / "python" / "shrink_gui").symlink_to(site.home / "shrink-array" / "python" / "shrink_gui")
    w.repo.setText("~/gitrepo")
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "3"}, 0)
    assert _wait_status(w, pump, rid, "fini")
    assert pump(lambda: w.table.item(0, 3) is not None and w.table.item(0, 3).text() == head[:8], 10)
    w.local_repo.setText(str(repo))
    w.check()
    assert pump(lambda: "identique" in w.conn.text(), 10), w.conn.text()


def test_compile_all_is_batch_only(gui, site):
    w, pump = gui
    w.program.setCurrentText("compile_all.sh")
    assert pump(lambda: w.form.schema is not None and "dryrun" in w.form.widgets)
    w.mode.setCurrentIndex(w.mode.findData("mprun"))
    n = len(w.runs)
    w.launch()
    pump(lambda: False, 1)
    assert len(w.runs) == n and "ccc_msub" in w.statusBar().currentMessage()
    assert not site.runs.exists() or not any(site.runs.iterdir())


def test_qos_follows_the_duration(gui, site):
    w, pump = gui
    assert pump(lambda: qos_max(w, "test", "gh200-bxi") == 3 * 3600, 10)   # read from the fake ccc_mqinfo
    w.program.setCurrentText("compile_all.sh")
    assert w.time_s.value() == 5400 and "test" in w.qos_note.text()
    w.program.setCurrentText("multi_launch_stress.py")
    assert w.time_s.value() == 2 * 86400 and "<b>long</b>" in w.qos_note.text()
    w.time_s.setValue(4 * 86400)
    assert "aucune QoS" in w.qos_note.text()
    w.preset.setCurrentIndex(w.preset.findData("milan-bxi"))
    assert "<b>xlong</b>" in w.qos_note.text()
    w.time_s.setValue(2 * 86400)
    w.program.setCurrentText("compile_all.sh")
    w.program.setCurrentText("multi_launch_stress.py")
    assert w.time_s.value() == 2 * 86400                                     # remembered per program
    w.qos.setCurrentIndex(w.qos.findData("long"))
    assert "n'est pas ouverte" in w.qos_note.text()


def test_submit_adds_qos(gui, site):
    w, pump = gui
    _choose_fake(w, pump)
    w.time_s.setValue(2 * 86400)
    rid = _submit(w, pump, {"runs": "2"}, 0)
    assert "#MSUB -Q long" in (site.runs / rid / "job.sh").read_text()
    w.time_s.setValue(5 * 86400)
    n = len(w.runs)
    w.launch()
    pump(lambda: False, 1)
    assert len(w.runs) == n and "aucune QoS" in w.statusBar().currentMessage()


def qos_max(w, name, part):
    from shrink_gui import qos
    row = qos.limits(w.qos_rows, name, part)
    return row and row["max_time"]


def test_target_and_exec_pickers(gui, site, monkeypatch):
    from PySide6.QtCore import Qt
    from shrink_gui import app as app_mod
    from shrink_gui import widgets as widgets_mod
    from test_remote_helper import _fake_build_rep
    _fake_build_rep(site)
    from shrink_gui import launch_form
    monkeypatch.setattr(launch_form, "BUILD_REP", "~/br")       # read where the pickers are
    w, pump = gui
    w.program.setCurrentText("compile_all.sh")
    assert pump(lambda: w.form.schema is not None and "exec" in w.form.widgets)
    assert w.findChild(type(w.findChildren(app_mod.QPushButton)[0]), "pick_exec") is not None
    seen = {}

    def fake_exec(dlg, ticks):
        items = {dlg.list.item(i).data(Qt.UserRole): dlg.list.item(i) for i in range(dlg.list.count())}
        seen[dlg.windowTitle()] = items
        for t in ticks:
            items[t].setCheckState(Qt.Checked)
        return 1

    monkeypatch.setattr(widgets_mod.PickerDialog, "exec", lambda dlg: fake_exec(dlg, ["omp-llvm", "omp-gnu-debug"]))
    w.pick("target")
    assert pump(lambda: w.form.widgets["target"][1].text() != "", 20)
    multi = "comma" in w.form.widgets["target"][0]["help"].lower()
    got = w.form.widgets["target"][1].text().split(",")
    assert sorted(got) == (["omp-gnu-debug", "omp-llvm"] if multi else ["omp-gnu-debug"])   # one glob: the last ticked
    w.form.widgets["target"][1].setText("omp-llvm")
    monkeypatch.setattr(widgets_mod.PickerDialog, "exec",
                        lambda dlg: fake_exec(dlg, ["shrink_bench_chunk_64", "shrink_bench_chunk_dyn"]))
    w.pick("exec")
    exes = seen["--exec : exécutables"]
    assert set(exes) >= {"shrink_bench_chunk_64", "shrink_bench_chunk_dyn"}
    assert "shrink_bench_chunk_256" not in exes                     # x86_64 build: not on gh200
    assert ("--exec=" + w.form.widgets["exec"][1].text()) in w.preview.toPlainText().replace("'", "")


def test_start_after_a_compile_run(gui, site):
    import base64
    import json
    from test_start_conditions import _spec
    w, pump = gui
    blob = base64.b64encode(json.dumps(_spec("sleep 4", program="compile_all.sh")).encode()).decode()
    jc = site.helper("launch", "~/shrink_runs", "0-compile", blob)["jobid"]   # sorts after the new run
    w.refresh()
    assert pump(lambda: w.after.findData("0-compile") >= 0, 10)          # an active compile run is offered
    assert not w.after_cond.isEnabled()
    w.after.setCurrentIndex(w.after.findData("0-compile"))
    assert w.after_cond.isEnabled()
    _choose_fake(w, pump)
    w.delay_on.setChecked(True)
    w.delay.setValue(120)
    rid = _submit(w, pump, {"runs": "2"}, len(w.runs))
    job = (site.runs / rid / "job.sh").read_text()
    assert f'#MSUB -E "--dependency=afterok:{jc} --begin=now+120"' in job
    assert pump(lambda: (w.refresh() or True) and next(r for r in w.runs if r["id"] == rid)["status"] == "en attente", 10)
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == rid))
    w.delete()                                                    # still queued: refused
    assert (site.runs / rid).exists()


def test_build_check_follows_the_options(gui):
    w, pump = gui
    w.program.setCurrentText("multi_launch_stress.py")
    assert pump(lambda: w.form.schema is not None and "builds" in w.form.widgets)
    assert w.check_builds.isEnabled()
    c = w.build_check()
    assert c["mode"] == "refuse" and c["builds_file"] == "build_arm.txt" and "shrink-array" in c["root"]
    w.form.set_values({"builds": "omp-gnu,omp-llvm"})
    assert w.build_check()["builds"] == ["omp-gnu", "omp-llvm"]
    w.check_builds.setCurrentIndex(w.check_builds.findData("off"))
    assert w.build_check() is None
    w.check_builds.setCurrentIndex(w.check_builds.findData("warn"))
    w.program.setCurrentText("bench_and_test.py")
    assert pump(lambda: w.form.schema is not None and "configs" in w.form.widgets, 60)
    c = w.build_check()
    assert c["mode"] == "warn" and "cmake-build-omp-gnu" in c["builds"]
    w.program.setCurrentText("compile_all.sh")
    assert not w.check_builds.isEnabled()


def test_delete_with_outputs_and_local_copy(gui, site, monkeypatch):
    import os
    w, pump = gui
    _choose_fake(w, pump)
    w.program.setCurrentText("fake_stress.py")
    monkeypatch.setitem(__import__("shrink_gui.config", fromlist=["x"]).PROGRAMS["fake_stress.py"],
                        "own_outputs", "dir")
    rid = _submit(w, pump, {"runs": "2"}, len(w.runs))
    assert _wait_status(w, pump, rid, "fini")
    log = site.helper("tail", "~/shrink_runs", rid, 0)["data"]
    od = log.split("Outputs, CSV and log in ")[1].split()[0]
    local = site.local / rid
    local.mkdir()
    seen = {}

    def fake_dialog(run, own, loc):
        seen.update(own=own, loc=loc)
        return True, True, True
    monkeypatch.setattr(w, "delete_dialog", fake_dialog)
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == rid))
    w.delete()
    assert pump(lambda: not (site.runs / rid).exists() and not local.exists(), 10)
    assert seen == {"own": "dir", "loc": str(local)} and not os.path.exists(od)


def test_log_view_rewrites_and_trims(qapp):
    from shrink_gui.widgets import LogView
    v = LogView()
    for c in ["start\n", "prog 1\r", "prog 2\rprog 3\n", "done\r\n", "a\rbb\rc\n", "x"]:
        v.append_raw(c)
    assert v.toPlainText().splitlines() == ["start", "prog 3", "done", "c", "x"]
    v = LogView()
    v.setMaximumBlockCount(50)
    v.append_raw("".join(f"{i}\n" for i in range(100000)))
    lines = v.toPlainText().splitlines()
    assert lines[-1] == "99999" and len(lines) < 50


def test_git_row_and_pull(gui, site, tmp_path):
    """The "Git (inti)" row shows the branch; "git pull" brings the commit."""
    import subprocess
    w, pump = gui
    env = dict(site.env, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")

    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()

    origin, up = tmp_path / "o.git", tmp_path / "up"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(up))
    git(up, "checkout", "-q", "-b", "dev")
    (up / "f").write_text("1\n")
    git(up, "add", "f")
    git(up, "commit", "-qm", "one")
    git(up, "push", "-q", "origin", "dev")
    git(site.home, "clone", "-q", "-b", "dev", str(origin), "c")
    (up / "f").write_text("2\n")
    git(up, "commit", "-qam", "two")
    git(up, "push", "-q", "origin", "dev")

    w.repo.setText("~/c")
    w.check()
    assert pump(lambda: "branche <b>dev</b>" in w.git_state.text()), w.git_state.text()
    # enabled once the branch list's fetch is over: one git command at a time
    assert pump(lambda: w.bpull.isEnabled() and w.bpull_rebase.isEnabled())
    w.bpull.click()
    want = git(up, "rev-parse", "HEAD")
    assert pump(lambda: getattr(w, "inti_git", {}).get("commit") == want)
    assert pump(lambda: want[:8] in w.git_state.text())


def test_branch_list_and_switch(gui, site, tmp_path):
    """"Branche (inti)" lists the local and remote branches after a fetch;
    "Changer de branche" checks the chosen one out, then git pull acts on it."""
    import subprocess
    w, pump = gui
    env = dict(site.env, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")

    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()

    origin, up = tmp_path / "o.git", tmp_path / "up"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(up))
    git(up, "checkout", "-q", "-b", "dev")
    (up / "f").write_text("1\n")
    git(up, "add", "f")
    git(up, "commit", "-qm", "one")
    git(up, "push", "-q", "origin", "dev")
    git(site.home, "clone", "-q", "-b", "dev", str(origin), "c")
    git(up, "checkout", "-q", "-b", "feature")
    (up / "f").write_text("feature\n")
    git(up, "commit", "-qam", "feature")
    git(up, "push", "-q", "origin", "feature")

    w.repo.setText("~/c")
    w.check()
    assert pump(lambda: w.branch.findData("origin/feature") >= 0), [w.branch.itemText(i) for i in range(w.branch.count())]
    assert w.branch.currentData() == "dev" and not w.bswitch.isEnabled()
    w.branch.setCurrentIndex(w.branch.findData("origin/feature"))
    assert w.bswitch.isEnabled()
    w.bswitch.click()
    want = git(up, "rev-parse", "feature")
    assert pump(lambda: getattr(w, "inti_git", {}).get("branch") == "feature")
    assert w.inti_git["commit"] == want
    assert pump(lambda: "branche <b>feature</b>" in w.git_state.text())
    assert pump(lambda: w.branch.currentData() == "feature" and w.branch.findData("origin/feature") < 0)
    assert w.bpull.isEnabled() and not w.bswitch.isEnabled()


# ------------------------------------------------------------ this machine
def _to_local(w, pump, site):
    """Switch the window to this machine, its repository and runs in the test's directory."""
    w.settings.setValue("local/repo", str(site.home / "shrink-array"))
    w.settings.setValue("local/runs_dir", str(site.root / "local_runs"))
    w.local_cb.setChecked(True)
    assert pump(lambda: "cette machine —" in w.conn.text(), 20), w.conn.text()


def test_local_machine(gui, site):
    from shrink_gui.config import LOCAL
    w, pump = gui
    _to_local(w, pump, site)
    assert not w.host.isEnabled() and w.repo_label.text() == "Dépôt (cette machine)"
    assert [w.preset.itemData(i) for i in range(w.preset.count())] == [LOCAL]
    assert [w.mode.itemData(i) for i in range(w.mode.count())] == ["local"]
    assert "cœurs" in w.preset.itemText(0)
    assert not w.where.isRowVisible(w.qos_row) and not w.where.isRowVisible(w.directives)
    assert w.launcher.isVisibleTo(w)
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "4", "step": "0.01"}, 0)
    assert rid.endswith("-" + LOCAL) and _wait_status(w, pump, rid, "fini")
    assert (site.root / "local_runs" / rid / "launch.sh").exists() and not (site.runs / rid).exists()
    w.table.selectRow(0)
    assert pump(lambda: "exit code 0" in w.log.toPlainText(), 10)
    w.fetch()
    assert pump(lambda: (site.local / rid / "run" / "out.log").exists(), 10)
    # back to the cluster: its own repository, partition, mode and runs
    w.local_cb.setChecked(False)
    assert pump(lambda: "connecté" in w.conn.text(), 20), w.conn.text()
    assert w.repo.text() == "~/shrink-array" and w.preset.currentData() == "gh200-bxi"
    assert w.mode.currentData() == "msub" and w.where.isRowVisible(w.qos_row)
    assert all(r["id"] != rid for r in w.runs)
    # and this machine's fields are remembered
    w.local_cb.setChecked(True)
    assert pump(lambda: "cette machine —" in w.conn.text(), 20)
    assert w.repo.text() == str(site.home / "shrink-array")
    assert pump(lambda: any(r["id"] == rid for r in w.runs), 10)


def test_compile_all_on_this_machine(gui, site, monkeypatch):
    w, pump = gui
    _to_local(w, pump, site)
    w.program.setCurrentText("compile_all.sh")
    assert pump(lambda: w.form.schema is not None and "dryrun" in w.form.widgets)
    calls = []
    monkeypatch.setattr(w.remote, "helper", lambda action, args, done, binary=False: calls.append((action, args)))
    w.launch()
    launches = [a for act, a in calls if act == "launch"]
    assert launches, w.statusBar().currentMessage()
    spec = json.loads(base64.b64decode(launches[-1][2]))
    assert spec["mode"] == "local" and "timeout -k 60" in spec["launch.sh"] and spec["meta"]["qos"] == ""


def test_local_machine_in_english(gui, site, qapp):
    w, pump = gui
    _to_local(w, pump, site)
    w.lang.setCurrentIndex(w.lang.findData("en"))
    w = qapp.main_window
    assert pump(lambda: "this machine —" in w.conn.text(), 20), w.conn.text()
    assert w.repo_label.text() == "Repository (this machine)" and w.preset.itemText(0).startswith("this machine")
    assert _french_left(w) == []


def test_errors_tab_failures_and_rerun(gui, site, monkeypatch):
    w, pump = gui
    w.program.setCurrentText("fake_bench.py")
    assert pump(lambda: w.form.schema is not None and w.form.schema["description"] == "fake bench")
    rid = _submit(w, pump, {}, len(w.runs))
    assert _wait_status(w, pump, rid, "fini")
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == rid))
    w.tabs.setCurrentIndex(w.TAB_ERRORS)
    assert pump(lambda: w.err_list.count() == 3, 10)        # the log + 2 failures
    assert "Segmentation fault" in w.err_view.toPlainText()
    assert w.err_rerun.isVisible() and w.err_rerun.isEnabled()
    w.err_list.setCurrentRow(1)                            # its complete output
    assert pump(lambda: "Segmentation fault at chunk 17" in w.err_view.toPlainText(), 10)
    assert "__rc-11.log" in w.err_view.toPlainText()
    seen = {}

    def dialog(run, replayable, other, newer):
        seen.update(replayable=replayable, other=other)
        return 2, 1800
    monkeypatch.setattr(w, "rerun_dialog", dialog)
    n = len(w.runs)
    time.sleep(1.05)                                       # run ids are per second
    w.rerun_failed()
    assert seen == {"replayable": 1, "other": 1}
    assert pump(lambda: len(w.runs) == n + 1, 20)
    again = w.runs[0]
    assert again["meta"]["rerun_of"] == rid and again["meta"]["time_s"] == 1800
    assert re.search(r"--rerun-failed --rerun-repeat 2 --full-log-dir run_logs/rerun-\d{8}-\d{6}$",
                     again["meta"]["command"])
    assert w.extra.text() == "" and w.time_s.value() != 1800      # the form is left as it was
    assert _wait_status(w, pump, again["id"], "fini")
    w.table.selectRow(0)
    assert pump(lambda: "rerun 2" in w.log.toPlainText(), 10)
    w.tabs.setCurrentIndex(0)
    w.tabs.setCurrentIndex(w.TAB_ERRORS)
    assert pump(lambda: "relance des échecs de " + rid in w.err_note.text(), 10)
    assert pump(lambda: w.err_list.count() == 2, 10)
    w.err_list.setCurrentRow(1)
    assert pump(lambda: "still crashing" in w.err_view.toPlainText(), 10)


def test_errors_column_and_first_error_notice(gui, site):
    w, pump = gui
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "6", "fail_every": "3", "step": "0.01"}, len(w.runs))
    assert _wait_status(w, pump, rid, "fini")
    row = next(i for i, r in enumerate(w.runs) if r["id"] == rid)
    assert pump(lambda: w.table.item(row, w.COL_ERRORS) is not None
                and w.table.item(row, w.COL_ERRORS).text() == "2", 10)
    assert w.table.item(row, w.COL_ERRORS).font().bold()
    # the first error of an active run: once, not again for the next ones
    run = {"id": "r9", "status": "en cours", "code": None, "meta": {"program": "bench_and_test.py",
                                                                    "partition": "gh200-bxi"}}
    w.runs = [dict(run, errors=[0, None])]
    w.prev_status = None
    w.notify_changes()
    w.notify_changes()
    w.statusBar().clearMessage()
    w.runs = [dict(run, errors=[1, "CHECK_FAIL: chunk2a t=4"])]
    w.notify_changes()
    assert w.statusBar().currentMessage() == "r9 : première erreur — CHECK_FAIL: chunk2a t=4"
    w.statusBar().clearMessage()
    w.runs = [dict(run, errors=[3, "CHECK_FAIL: x"])]
    w.notify_changes()
    assert w.statusBar().currentMessage() == ""


def test_bench_chart_and_comparison(gui, site):
    w, pump = gui
    w.program.setCurrentText("fake_bench.py")
    assert pump(lambda: w.form.schema is not None and w.form.schema["description"] == "fake bench")
    a = _submit(w, pump, {}, len(w.runs))
    b = _submit(w, pump, {"speed": "1.1"}, len(w.runs))
    assert _wait_status(w, pump, a, "fini") and _wait_status(w, pump, b, "fini")
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == a))
    w.tabs.setCurrentIndex(w.TAB_CHART)
    assert pump(lambda: len(w.bc_view.points) == 12, 10)              # 3 variants x 4 thread counts
    assert w.pt_stack.currentIndex() == 1 and w.bc_config.count() == 2
    assert w.bc_note.text().startswith("12 point(s)")
    w.bc_cmp.setCurrentIndex(w.bc_cmp.findData(b))
    assert pump(lambda: "×1.100 (10.0 % plus lent)" in w.bc_note.text(), 10)
    assert "24 point(s) communs (toutes configs)" in w.bc_note.text()
    assert len(w.bc_view.points) == 24
    w.bc_metric.setCurrentIndex(w.bc_metric.findData("speedup"))
    assert len(w.bc_view.points) == 16 and "×1.000" in w.bc_note.text()
    # a multi_launch_stress.py run: back to its own page
    _choose_fake(w, pump)
    m = _submit(w, pump, {"runs": "6", "step": "0.01"}, len(w.runs))
    assert _wait_status(w, pump, m, "fini")
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == m))
    assert pump(lambda: w.pt_stack.currentIndex() == 0 and len(w.pt_view.points) == 6, 10)


def _fake_speedup_build(site):
    """A build root with fake shrink_bench binaries: vector and chunk2a@dyn
    scale, chunk@dyn always fails (an error line, rc 1)."""
    import platform
    d = site.home / "br" / "cmake-build-omp-llvm" / f"Linux__{platform.machine()}" / "examples" / "shrink_bench"
    d.mkdir(parents=True)
    body = ('#!/bin/bash\nm=1; while [ $# -gt 0 ]; do [ "$1" = --multiply ] && m=$2; shift; done\n'
            'python3 -c "print(\'Memory consumption (peak): 10.0 Mo\'); '
            'print(\'Elapsed time computation : %%.4fs\' %% (%s * $m / $OMP_NUM_THREADS))"\n')
    for name, base in (("shrink_bench_vector_perf", 2.0), ("shrink_bench_chunk2a_dyn", 1.5)):
        (d / name).write_text(body % base)
        (d / name).chmod(0o755)
    (d / "shrink_bench_chunk_dyn").write_text("#!/bin/bash\necho 'error: no chunk today' >&2\nexit 1\n")
    (d / "shrink_bench_chunk_dyn").chmod(0o755)
    return site.home / "br"


def test_omp_speedup_submitted_from_the_window(gui, site):
    w, pump = gui
    br = _fake_speedup_build(site)
    w.program.setCurrentText("omp_speedup.py")
    assert pump(lambda: w.form.schema is not None and "all_exec" in w.form.widgets)
    from PySide6.QtWidgets import QPushButton
    assert w.findChild(QPushButton, "pick_execs") is not None and w.findChild(QPushButton, "pick_target")
    vals = {"target": "omp-llvm", "execs": "vector,chunk2a@dyn", "min_threads": "1", "max_threads": "2",
            "reps": "2", "build_root": str(br)}
    a = _submit(w, pump, vals, len(w.runs))
    assert _wait_status(w, pump, a, "fini")
    run = next(r for r in w.runs if r["id"] == a)
    assert run["code"] == 0 and run["errors"] == [0, None] and run["progress"][:2] == [8, 8]
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == a))
    assert pump(lambda: "Outputs in " in w.log.toPlainText(), 10)
    w.tabs.setCurrentIndex(w.TAB_SUMMARY)
    assert pump(lambda: w.sum_view.count() == 2, 10)              # Phases (events.jsonl), summary.csv
    assert w.sum_table.item(0, 0).text() == "speedup" and w.sum_table.item(0, 1).text() == "ok"
    w.sum_view.setCurrentIndex(1)
    assert w.sum_table.rowCount() == 4                             # 2 execs x 2 thread counts
    w.tabs.setCurrentIndex(w.TAB_CHART)
    assert pump(lambda: w.pt_stack.currentIndex() == 2 and len(w.sc_view.points) == 4, 10)
    assert w.sc_note.text().startswith("4 point(s)")
    # B: the same at --multiply 2, plus an executable that fails
    b = _submit(w, pump, dict(vals, execs="vector,chunk2a@dyn,chunk@dyn", multiply="2"), len(w.runs))
    assert _wait_status(w, pump, b, "fini")
    run = next(r for r in w.runs if r["id"] == b)
    assert run["code"] == 1 and run["errors"][0] == 4 and run["errors"][1].startswith("RUN_FAIL: shrink_bench_chunk_dyn")
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == a))
    w.sc_cmp.setCurrentIndex(w.sc_cmp.findData(b))
    assert pump(lambda: "B / A : ×1.000" in w.sc_note.text(), 10)  # the speedups match
    assert "tailles ou modes différents" in w.sc_note.text()
    w.sc_metric.setCurrentIndex(w.sc_metric.findData("temps"))
    assert "B / A : ×2.000" in w.sc_note.text()                     # twice the work
    w.table.selectRow(next(i for i, r in enumerate(w.runs) if r["id"] == b))
    w.tabs.setCurrentIndex(w.TAB_ERRORS)
    assert pump(lambda: "error: no chunk today" in w.err_view.toPlainText(), 10)
    assert w.err_note.text().startswith("4 erreur(s)")


def test_duration_column(gui, site):
    w, pump = gui
    assert w.fmt_elapsed(14390) == "3:59:50" and w.fmt_elapsed(90061) == "1 j 01:01:01"
    _choose_fake(w, pump)
    rid = _submit(w, pump, {"runs": "3", "step": "0.01"}, len(w.runs))
    assert _wait_status(w, pump, rid, "fini")
    row = next(i for i, r in enumerate(w.runs) if r["id"] == rid)
    assert pump(lambda: w.table.item(row, w.COL_DURATION) is not None
                and re.fullmatch(r"\d+:\d\d:\d\d", w.table.item(row, w.COL_DURATION).text()), 10)
    it = w.table.item(row, w.COL_DURATION)
    assert "démarré : " in it.toolTip() and "fini : " in it.toolTip() and not it.font().italic()
    # a running run counts on between two lists, from the cluster's clock
    import time
    w.clock_skew = 0.0
    run = {"id": "r8", "status": "en cours", "code": None, "times": [int(time.time()) - 125, None],
           "meta": {"program": "omp_speedup.py", "mode": "msub", "time_s": 14400}}
    assert w.duration_text(run) in ("0:02:05", "0:02:06")
    assert "en cours depuis" in w.duration_tip(run) and "4:00:00" in w.duration_tip(run)
    assert w.duration_text(dict(run, status="en attente", times=[None, None])) == ""


def test_omp_speedup_refused_without_exec(gui, site):
    w, pump = gui
    w.program.setCurrentText("omp_speedup.py")
    assert pump(lambda: w.form.schema is not None and "all_exec" in w.form.widgets)
    w.form.set_values({"target": "omp-llvm", "min_threads": "1", "max_threads": "8",
                       "size": "dyn,1024", "int64": True})
    n = len(w.runs)
    w.launch()
    assert "--exec ou --all-exec" in w.last_error and "--size, --int64" in w.last_error
    assert len(w.runs) == n
    assert w.missing_options("omp_speedup.py")
    w.form.set_values({"target": "omp-llvm", "min_threads": "1", "max_threads": "8", "all_exec": True})
    assert w.missing_options("omp_speedup.py") is None
    w.form.set_values({"target": "omp-llvm", "min_threads": "1", "max_threads": "8"})
    w.extra.setText("--exec chunk2a@dyn")
    assert w.missing_options("omp_speedup.py") is None
    w.extra.setText("")
