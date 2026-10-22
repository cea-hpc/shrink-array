# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""QoS: ccc_mqinfo's table, and the QoS a job gets."""
import pathlib

import pytest

from shrink_gui import qos
from shrink_gui.widgets import parse_duration

# inti's real ccc_mqinfo output (the "*" rows, MaxSub 0 closing a QoS); the
# site file's [qos] fallback is only a short stand-in for it.
ROWS = qos.parse_mqinfo((pathlib.Path(__file__).parent / "data" / "ccc_mqinfo_inti.txt").read_text())
H, D = 3600, 86400


def test_parse():
    assert len(ROWS) == 32
    test_gh = qos.limits(ROWS, "test", "gh200-bxi")
    assert (test_gh["priority"], test_gh["max_cpus"], test_gh["max_sub"], test_gh["max_time"]) == (40, 5184, 2, 2 * H)
    normal = qos.limits(ROWS, "normal", "gh200-bxi")            # the "*" row, no CPU limit
    assert normal["partition"] == "*" and normal["max_cpus"] is None and normal["max_time"] == D
    assert qos.limits(ROWS, "long", "milan-bxi") is None        # "*" row with MaxCPUs 0
    assert qos.limits(ROWS, "xlong", "gh200-bxi") is None


@pytest.mark.parametrize("part, cpus, t, want", [
    ("gh200-bxi", 288, 1.5 * H, "test"),       # compile_all.sh
    ("gh200-bxi", 288, 6 * H, "normal"),       # bench_and_test.py
    ("gh200-bxi", 288, 2 * D, "long"),         # multi_launch_stress.py
    ("milan-bxi", 64, 1.5 * H, "normal"),      # test is 30 min there
    ("milan-bxi", 64, 2 * D, "xlong"),         # no long on milan-bxi
    ("gh200-bxi", 288, 4 * D, None),
])
def test_auto(part, cpus, t, want):
    name, _row, err = qos.choose(ROWS, part, cpus, t)
    assert name == want and (err is None) == (want is not None)


def test_explicit_and_cpus():
    assert "n'est pas ouverte" in qos.choose(ROWS, "milan-bxi", 64, H, "long")[2]
    assert "dépasse" in qos.choose(ROWS, "gh200-bxi", 288, 3 * H, "test")[2]
    assert "cœurs" in qos.choose(ROWS, "gh200-bxi", 6000, H, "test")[2]
    assert qos.job_cpus("#MSUB -q gh200-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 288\n") == 288
    assert qos.job_cpus("-N1 -c 64 -p milan-bxi -C x -Xfirst") == 64
    assert qos.explicit_qos("#MSUB -q x\n#MSUB -Q long\n") == "long"
    assert qos.explicit_qos("-N1 -c 64") is None


@pytest.mark.parametrize("text, s", [
    ("1 h 30", 5400), ("90 min", 5400), ("2j", 2 * D), ("2 d 12 h", 2 * D + 12 * H), ("2 j 12", 2 * D + 12 * H),
    ("1-00:00:00", D), ("00:30:00", 1800), ("5400", 5400), ("1h30m", 5400), ("abc", None), ("", None),
])
def test_parse_duration(text, s):
    assert parse_duration(text) == s


def test_fmt():
    assert [qos.fmt(x) for x in (5400, 2 * D, 45 * 60)] == ["1 h 30", "2 j 00 h", "45 min"]
    assert qos.fmt(2 * D, "en") == "2 d 00 h"
