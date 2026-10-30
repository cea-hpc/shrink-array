# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""bench_runner: a failed run records its complete output (failed_cases.jsonl
"log"), a timeout keeps what it printed, --rerun-failed points to the newest."""
import json
import sys
import types

import pytest

runs = pytest.importorskip("bench_runner.runs")
state = pytest.importorskip("bench_runner.state")


@pytest.fixture
def full_logs(tmp_path, monkeypatch):
    d = tmp_path / "run_logs"
    d.mkdir()
    monkeypatch.setattr(state, "_FULL_LOG_DIR", str(d))
    monkeypatch.setattr(state, "_NUMA_MODE", "off")
    state._PENDING_FAILED_RUNS.clear()
    yield d
    state._PENDING_FAILED_RUNS.clear()


def _py(code):
    return [sys.executable, "-c", code]


def test_failed_run_points_to_its_log(full_logs):
    ok = runs.run_cmd(_py("print('fine')"), {}, 30)
    assert ok[0] and state._PENDING_FAILED_RUNS == []
    bad = runs.run_cmd(_py("import sys; print('out'); sys.exit(3)"), {}, 30)
    assert not bad[0]
    log = state._PENDING_FAILED_RUNS[-1]["log"]
    assert log.endswith("__rc3.log") and "out" in open(log).read()


def test_must_contain_failure_named_ok_still_points_to_its_log(full_logs):
    runs.run_cmd(_py("print('no marker')"), {}, 30, must_contain="[APPLIED]")
    log = state._PENDING_FAILED_RUNS[-1]["log"]
    assert log.endswith("__ok.log")          # exit 0: only the record says it failed


def test_timeout_keeps_partial_output(full_logs):
    runs.run_cmd(_py("import time; print('before hang', flush=True); time.sleep(30)"), {}, 1)
    log = state._PENDING_FAILED_RUNS[-1]["log"]
    assert log.endswith("__timeout.log") and "before hang" in open(log).read()


def test_rerun_failed_points_to_the_newest_log(full_logs, tmp_path):
    runs.run_cmd(_py("import sys; sys.exit(2)"), {}, 30)
    first = state._PENDING_FAILED_RUNS[-1]["log"]
    errors_log = tmp_path / "bench_errors.log"
    runs._append_error_now(str(errors_log), "CHECK_FAIL: x\n  stderr_tail:\n")
    args = types.SimpleNamespace(rerun_failed="", rerun_repeat=1, errors_log=str(errors_log))
    assert runs.run_rerun_failed(args) == 1
    case = json.loads((tmp_path / "failed_cases.jsonl").read_text().splitlines()[0])
    assert case["runs"][0]["log"] != first and case["runs"][0]["log"].startswith(str(full_logs))


def test_error_is_an_event(tmp_path, monkeypatch):
    events = pytest.importorskip("bench_runner.events")
    p = tmp_path / "ev.jsonl"
    monkeypatch.setenv("CHUNK_BENCH_EVENTS", str(p))
    events.open_file(None)
    try:
        with events.phase("check", []):
            runs._append_error_now(str(tmp_path / "bench_errors.log"), "CHECK_FAIL: zero_any t=4\n  tail\n")
    finally:
        events.open_file("")
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    err = next(r for r in recs if r["kind"] == "error")
    assert err["phase"] == "check" and err["summary"] == "CHECK_FAIL: zero_any t=4"


def test_keep_run_outputs(tmp_path, monkeypatch):
    report = pytest.importorskip("bench_runner.report")
    od, keep = tmp_path / "aarch64", tmp_path / "run"
    od.mkdir()
    keep.mkdir()
    for n in ("bench_summary.csv", "bench_errors.log", "failed_cases.jsonl"):
        (od / n).write_text(n)
    monkeypatch.delenv("CHUNK_BENCH_KEEP_DIR", raising=False)
    report.keep_run_outputs(str(od / "bench_summary.csv"), str(od / "bench_errors.log"))
    assert not list(keep.iterdir())
    monkeypatch.setenv("CHUNK_BENCH_KEEP_DIR", str(keep))
    report.keep_run_outputs(str(od / "bench_summary.csv"), str(od / "bench_errors.log"))
    assert sorted(p.name for p in keep.iterdir()) == ["bench_errors.log", "bench_summary.csv",
                                                     "failed_cases.jsonl"]


killinfo = pytest.importorskip("bench_runner.killinfo")


def test_signals_in_words(full_logs, monkeypatch):
    monkeypatch.setattr(killinfo, "oom_counts", lambda: {})
    r = runs.run_cmd(_py("import os, signal; print('step 3', flush=True); os.kill(os.getpid(), signal.SIGSEGV)"),
                     {}, 30)
    tail = r[3]
    assert tail.startswith("KILLED BY SIGSEGV (signal 11): invalid memory access")
    assert "(stderr empty)\n--- stdout tail ---\nstep 3" in tail
    assert "full output: " + state._PENDING_FAILED_RUNS[-1]["log"] in tail
    r = runs.run_cmd(_py("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), {}, 30)
    assert r[2] == -9 and r[3].startswith("KILLED BY SIGKILL (signal 9) from outside the bench")
    assert "no cgroup information" in r[3]
    r = runs.run_cmd(_py("import sys; sys.stderr.write('boom\\n'); sys.exit(2)"), {}, 30)
    assert r[3].startswith("boom") and "full output:" in r[3]          # an exit code: as before


def test_oom_kill_is_named(monkeypatch):
    counts = iter([{"/cg/job": 0, "/cg": 3}, {"/cg/job": 1, "/cg": 3}])
    monkeypatch.setattr(killinfo, "oom_counts", lambda: next(counts))
    monkeypatch.setattr(killinfo, "_limit", lambda d: 64 * 2**30 if d == "/cg/job" else None)
    monkeypatch.setattr(state, "_FULL_LOG_DIR", None)
    r = runs.run_cmd(_py("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), {}, 30)
    assert r[3].startswith("KILLED BY THE OUT-OF-MEMORY KILLER (SIGKILL)") and "(limit 64.0 GiB) [/cg/job]" in r[3]


def test_real_oom_kill_in_a_memory_cgroup(tmp_path):
    """The real thing: a 100 MB memory cgroup (v1, as RHEL 8), a run that
    allocates 400 MB. Skipped where no cgroup can be made."""
    import os
    import subprocess
    try:
        own = next(l.split(":", 2)[2] for l in open("/proc/self/cgroup") if ":memory:" in l)
    except (OSError, StopIteration):
        pytest.skip("no v1 memory cgroup")
    cg = f"/sys/fs/cgroup/memory{own.strip().rstrip('/')}/shrink_oom_test_{os.getpid()}"
    try:
        os.mkdir(cg)
        open(f"{cg}/memory.limit_in_bytes", "w").write("100000000")
    except OSError:
        pytest.skip("cannot make a memory cgroup here")
    try:
        code = (f"import sys; sys.path.insert(0, {str(runs.__file__.rsplit('/bench_runner', 1)[0])!r})\n"
                f"open({cg + '/cgroup.procs'!r}, 'w').write(str(__import__('os').getpid()))\n"
                "from bench_runner import runs, state\n"
                "state._NUMA_MODE = 'off'\n"
                f"r = runs.run_cmd([sys.executable, '-c', 'x = bytearray(400 * 2**20)'], {{}}, 60)\n"
                "print(r[2]); print(r[3])\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout
        assert out.startswith("-9\nKILLED BY THE OUT-OF-MEMORY KILLER (SIGKILL)"), out
        assert "(limit 0.1 GiB)" in out
    finally:
        try:
            os.rmdir(cg)
        except OSError:
            pass
