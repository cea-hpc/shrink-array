"""The window, end to end, offscreen, against the fake inti."""
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


def test_english_has_no_french_left(gui, qapp):
    from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton
    w, pump = gui
    w.lang.setCurrentIndex(w.lang.findData("en"))
    w = qapp.main_window
    assert pump(lambda: "connected" in w.conn.text())
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
    assert left == []


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
