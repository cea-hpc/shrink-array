# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""One machine name on two systems: each tool takes the build of its own."""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import thread_speedup                     # noqa: E402
from bench_runner import builds, settings  # noqa: E402

MACHINE = platform.machine()
RHEL8 = f"RHEL_8.10__{MACHINE}"
RHEL9 = f"RHEL_9.6__{MACHINE}"
EXE = "shrink_bench_omp_zero_dyn"


def _exe(root, arch):
    d = root / "cmake-build-omp-gnu" / arch / "examples" / "shrink_bench"
    d.mkdir(parents=True)
    (d / EXE).write_text("")
    (d / EXE).chmod(0o755)
    return str((d / EXE).resolve())


def _on(monkeypatch, arch):
    monkeypatch.setattr(settings, "arch_dir_name", lambda: arch)
    monkeypatch.setattr(builds, "arch_dir_name", lambda: arch)


def test_arch_dir_name_is_compile_all_arch(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('NAME="Red Hat"\nID="rhel"\nVERSION_ID="9.6"\n')
    assert settings.arch_dir_name(str(f)) == RHEL9
    assert settings.arch_dir_name(str(tmp_path / "none")) is None


def test_results_dir_is_the_systems(tmp_path, monkeypatch):
    _on(monkeypatch, RHEL9)
    assert settings.results_dir_name() == RHEL9
    monkeypatch.setenv("CCCWORKDIR", str(tmp_path))
    monkeypatch.setattr(thread_speedup, "results_dir_name", settings.results_dir_name)
    assert thread_speedup.default_oracle_dir("omp-gnu") == str(
        tmp_path / "bench_results" / RHEL9 / "oracle" / "cmake-build-omp-gnu")
    _on(monkeypatch, None)
    assert settings.results_dir_name() == MACHINE


def test_this_system_first(tmp_path, monkeypatch):
    rhel8, rhel9 = _exe(tmp_path, RHEL8), _exe(tmp_path, RHEL9)
    _on(monkeypatch, RHEL9)
    assert builds.resolve_exe(str(tmp_path), "cmake-build-omp-gnu", "zero@dyn", "perf") == rhel9
    assert thread_speedup.find_exe(str(tmp_path), "omp-gnu", EXE) == rhel9
    assert thread_speedup.glob_execs(str(tmp_path), "omp-gnu", "shrink_bench_*") == [(EXE, rhel9)]
    _on(monkeypatch, RHEL8)
    assert builds.resolve_exe(str(tmp_path), "cmake-build-omp-gnu", "zero@dyn", "perf") == rhel8


def test_other_system_is_a_noted_fallback(tmp_path, monkeypatch, capsys):
    rhel8 = _exe(tmp_path, RHEL8)
    _on(monkeypatch, RHEL9)
    assert builds.resolve_exe(str(tmp_path), "cmake-build-omp-gnu", "zero@dyn", "perf") == rhel8
    assert thread_speedup.find_exe(str(tmp_path), "omp-gnu", EXE) == rhel8
    assert f"nothing in {RHEL9}, taking {RHEL8}" in capsys.readouterr().out
