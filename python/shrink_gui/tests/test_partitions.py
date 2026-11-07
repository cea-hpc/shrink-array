# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The partitions the machine lists (partitions.py): the table read, the
entries built from the site's [discovered], the check, the window."""
import pathlib

import pytest

from shrink_gui import partitions, sitecfg

from conftest import BATCH_ARG, TESTS

SITES = TESTS.parent / "sites"
INTI = sitecfg.load(SITES / "cea_inti.toml")


def test_parse_ccc_mpinfo():
    rows = partitions.parse((TESTS / "data" / "ccc_mpinfo_inti.txt").read_text())
    assert [(r["name"], r["status"], r["cores"], r["gpus"], r["mem_per_core"]) for r in rows] == [
        ("gh200-bxi", "up", 288, 4, 1700), ("milan-bxi", "up", 64, 0, 3900),
        ("rome-bxi", "up", 128, 0, 1900), ("a64fx", "down", 48, 0, 666)]


def test_parse_sinfo():
    text = "PARTITION CPUS NODES AVAIL\ncpu*      64   10    up\ngpu       32+  4     up\ncpu       128  2     up\n"
    rows = partitions.parse(text)
    assert [(r["name"], r["cores"], r["status"]) for r in rows] == [("cpu", 64, "up"), ("gpu", 32, "up")]


@pytest.mark.parametrize("text", ["", "no table here\n", "Name Partition Priority MaxTime\nx y 1 1:00\n"])
def test_parse_nothing(text):
    assert partitions.parse(text) == []


def test_presets_from_the_templates():
    rows = partitions.parse((TESTS / "data" / "ccc_mpinfo_inti.txt").read_text())
    found = partitions.presets(rows, INTI.discovered, set(INTI.partitions))
    assert list(found) == ["rome-bxi", "a64fx"]          # the site file's two win
    rome, a64 = found["rome-bxi"], found["a64fx"]
    assert rome["label"] == "rome-bxi — x86, 128 cœurs" and rome["arch"] == "x86"
    assert rome["msub"] == "#MSUB -q rome-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 128\n"
    assert rome["mprun"] == "-N1 -c 128 -p rome-bxi -Xfirst" and rome["launcher"] == ""
    assert a64["arch"] == "arm" and a64["label"] == "a64fx — ARM, 48 cœurs, down"
    skip = dict(INTI.discovered, skip="^a64")
    assert list(partitions.presets(rows, skip, set(INTI.partitions))) == ["rome-bxi"]
    gpu = partitions.presets([dict(rows[0], name="gpu-x")], INTI.discovered, set())
    assert gpu["gpu-x"]["label"] == "gpu-x — x86, 288 cœurs, 4 GPU"


def _site_text(change):
    text = (SITES / "slurm_example.toml").read_text()
    old, new = change
    assert old in text
    return text.replace(old, new)


@pytest.mark.parametrize("change, key", [
    (("[discovered]\n", "[unused]\n"), "discovered"),                    # partition_info without templates
    (('arm = "grace|gh200|a64fx|arm"', 'arm = "("'), "discovered.arm"),  # not a pattern
    (("--cpus-per-task={cores}", "--cpus-per-task={cpus}"), "discovered.directives"),  # unknown field
])
def test_invalid_discovered(tmp_path, change, key):
    p = tmp_path / "site.toml"
    p.write_text(_site_text(change))
    with pytest.raises(sitecfg.SiteError) as e:
        sitecfg.load(p)
    assert key in str(e.value)


def test_check_reads_the_partitions(site):
    r = site.helper("check", "~/shrink-array", BATCH_ARG)
    assert [p["name"] for p in partitions.parse(r["mpinfo"])] == ["gh200-bxi", "milan-bxi", "rome-bxi", "a64fx"]


def test_window_offers_the_machine_partitions(gui, site):
    w, pump = gui
    from shrink_gui import config
    assert pump(lambda: w.preset.findData("rome-bxi") >= 0), [w.preset.itemData(i) for i in range(w.preset.count())]
    keys = [w.preset.itemData(i) for i in range(w.preset.count())]
    assert keys[:2] == ["gh200-bxi", "milan-bxi"] and "a64fx" in keys and "local" not in keys
    assert keys.count("gh200-bxi") == 1                      # the site file's entry, not the table's
    w.preset.setCurrentIndex(w.preset.findData("rome-bxi"))
    assert w.directives.toPlainText().startswith("#MSUB -q rome-bxi\n")
    assert "#MSUB -c 128" in w.directives.toPlainText() and w.arch() == "x86"
    assert "ccc_mpinfo" in w.preset.itemData(w.preset.currentIndex(), 3)   # Qt.ToolTipRole
    # remembered: the next start lists them before any check
    assert "rome-bxi" in w.settings.value(w.mpinfo_key(), "")
    assert config.set_discovered("") == []                   # no table: nothing changes
    assert "rome-bxi" in config.PRESETS
