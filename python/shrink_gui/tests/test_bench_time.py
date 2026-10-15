# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The -T of bench_and_test.py follows its --mode and the partition's architecture."""
import pytest

pytest.importorskip("PySide6")


def test_bench_time_follows_mode(gui):
    w, pump = gui
    w.program.setCurrentText("bench_and_test.py")
    assert pump(lambda: w.form.schema is not None and "mode" in w.form.widgets, 60)
    w.preset.setCurrentIndex(w.preset.findData("gh200-bxi"))
    w.form.set_values({"mode": "fast"})
    assert w.time_s.value() == 4500                 # 1 h + 25%, ARM
    w.form.set_values({"mode": "full"})
    assert w.time_s.value() == 108000
    w.preset.setCurrentIndex(w.preset.findData("milan-bxi"))
    assert w.time_s.value() == 67500                # the partition reset the form: normal, x86 x2.5
    w.form.set_values({"mode": "fast"})
    assert w.time_s.value() == 11250
    # A duration typed by hand stays while neither the mode nor the partition changes.
    w.time_s.setValue(20000)
    w.form.set_values({"mode": "fast", "threads": "64"})
    assert w.time_s.value() == 20000
    w.form.set_values({"mode": "normal"})
    assert w.time_s.value() == 67500
