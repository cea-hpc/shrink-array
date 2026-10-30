# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""bench_and_test.py --pool ("Nœuds" N > 1): the jobs share the campaign by
(phase, configuration) units, each claimed by the first job that reaches it;
each configuration's oracle is generated once, the other jobs waiting for
it; each job writes to its own <arch>/pool_<id>/n<K>, oracle/ staying shared.
The helper's Résumé / Graphique / Erreurs gather the jobs' outputs; the real
script is run twice at once on a fake build tree."""
import base64
import csv
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from conftest import FAKE_BENCH, PYDIR

HEAD = ["Config", "Variant", "Threads", "AvgTime_s", "PeakMem_Mo", "Speedup_vs_baseline",
        "MemRatio_vs_baseline", "MemSaved_Mo_vs_baseline"]


# ── the helper's side: two fake members of a pool ───────────────────────────

def _bench_job(runs, out, pool, k, nodes, configs, errors=0, layout="jobs"):
    """Job <pool>-n<k>: its run dir (meta, log announcing its own output
    directory <arch>/pool_<pool>/n<k>, events.jsonl) and, there, the
    bench_summary.csv of @p configs, @p errors CHECK_FAIL entries each with
    its full log under run_logs/. @p layout "job": node <k> of the one job
    <pool> of @p nodes nodes instead, its directory <pool>/n<k> (no meta.json
    there: the run's)."""
    if layout == "job":
        top = runs / pool
        top.mkdir(parents=True, exist_ok=True)
        (top / "meta.json").write_text(json.dumps(dict(FAKE_BENCH, program="bench_and_test.py", pool=pool,
                                                       nodes=nodes, layout="job")))
        d = top / f"n{k}"
        d.mkdir()
    else:
        d = runs / f"{pool}-n{k}"
        d.mkdir(parents=True)
        meta = dict(FAKE_BENCH, program="bench_and_test.py", pool=pool, node=k, nodes=nodes)
        (d / "meta.json").write_text(json.dumps(meta))
    od = out / "x86_64" / f"pool_{pool}" / f"n{k}"
    (od / "run_logs").mkdir(parents=True)
    (d / "out.log").write_text(f"📁 Run outputs → {od}  (arch: x86_64)\n"
                               f"📁 Shared by the pool: {out / 'x86_64'} (oracle/, .pools/)\n")
    with open(od / "bench_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for c in configs:
            for var, t in (("vector", 2.0), ("zero@256", 1.0)):
                for th in (1, 4):
                    w.writerow([c, var, th, "%.6f" % (t / th), "100.0", "" if var == "vector" else "2.000", "", ""])
    entries, cases = [], []
    for i in range(errors):
        log = od / "run_logs" / f"shrink_bench_omp_zero_256__omp1__{i + 1:05d}__rc4.log"
        log.write_text(f"no oracle (job n{k}, failure {i})\n")
        entries.append(f"CHECK_FAIL: {c} zero@256 iter={i + 1} code=4\n  stderr_tail:\n    no oracle")
        cases.append({"kind": "CHECK_FAIL", "summary": f"CHECK_FAIL: {c} zero@256 iter={i + 1} code=4",
                      "runs": [{"cmd": ["x"], "log": str(log)}]})
    (od / "bench_errors.log").write_text("\n\n".join(entries))
    (od / "failed_cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    t0 = time.time() - 100 + k
    ev = [{"t": t0, "kind": "start", "total_runs": 10, "pool": pool, "node": f"n{k}"},
          {"t": t0 + 1, "kind": "phase_start", "phase": "check"}]
    ev += [{"t": t0 + 2 + i, "kind": "error", "phase": "check", "summary": c["summary"]}
           for i, c in enumerate(cases)]
    ev += [{"t": t0 + 9, "kind": "phase_end", "phase": "check", "status": "errors" if errors else "ok",
            "errors": errors, "seconds": 8.0},
           {"t": t0 + 10, "kind": "end", "errors": errors, "interrupted": False, "minutes": 0.2}]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    return d, od


def _call(rh, capsys, act, *args):
    getattr(rh, "act_" + act)(list(args))
    return json.loads(capsys.readouterr().out)


def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


@pytest.mark.parametrize("layout", ["jobs", "job"])
def test_pooled_bench_summary_points_errors(tmp_path, capsys, layout):
    """Two jobs of one node P-n1 / P-n2, or the two nodes of one job P (its
    nodes in P/n1, P/n2): the same results from the run's tabs."""
    from shrink_gui import remote_helper as rh
    runs, out = tmp_path / "runs", tmp_path / "out"
    # Job 1 did llvm, job 2 gnu and gnu-aggressive (one unit) with 2 failures.
    _bench_job(runs, out, "P", 1, 2, ["llvm"], layout=layout)
    d2, od2 = _bench_job(runs, out, "P", 2, 2, ["gnu", "gnu-have-aggressive-compression"], errors=2, layout=layout)
    one, two = ("P", "P") if layout == "job" else ("P-n1", "P-n2")

    s = _call(rh, capsys, "summary", str(runs), one)
    phases, table = s["sections"]
    assert phases["columns"][0] == "nœud" and [r[0] for r in phases["rows"]] == ["n1", "n2"]
    assert phases["fail_rows"] == [1] and "2 fini(s), 2 erreur(s)" in phases["note"]
    assert len(table["rows"]) == 12 and table["note"].endswith(" — 2 nœuds sur 2")
    # The rows of both jobs, in one job's order (config, variant, threads).
    assert [r[0] for r in table["rows"]][::4] == ["gnu", "gnu-have-aggressive-compression", "llvm"]

    p = _call(rh, capsys, "points", str(runs), two)
    assert p["kind"] == "bench" and sorted(p["configs"]) == ["gnu", "gnu-have-aggressive-compression", "llvm"]
    assert len(p["rows"]) == 12 and p["own"] and (p["nodes"], p["nodes_found"]) == (2, 2)

    opts = {"files": ["bench_errors.log"], "cases": "failed_cases.jsonl", "full_logs": "run_logs"}
    e = _call(rh, capsys, "errors", str(runs), one, _b64(opts))
    assert sorted(f["name"] for f in e["files"]) == ["n1/bench_errors.log", "n2/bench_errors.log"]
    assert e["count"] == 2 and e["dir"] == str(out / "x86_64" / "pool_P") and e["newer"] is None
    assert [c["summary"][:14] for c in e["cases"]] == ["n2: CHECK_FAIL"] * 2
    assert e["cases"][0]["logs"] == ["n2/run_logs/shrink_bench_omp_zero_256__omp1__00001__rc4.log"]
    # The complete output of a failure of job 2, opened from job 1's tab.
    f = _call(rh, capsys, "errfile", str(runs), one, _b64({"paths": e["cases"][1]["logs"]}))
    assert f["files"][0]["data"] == "no oracle (job n2, failure 1)\n"
    assert _call(rh, capsys, "errfile", str(runs), one, _b64({"paths": ["../x"]}))["files"][0]["missing"]

    # The Erreurs column of either job: the pool's count, the newest error.
    meta1 = json.loads((runs / one / "meta.json").read_text())
    assert rh.errors_of(str(runs / one), meta1, "fini", rh.read_events(str(runs / one))) == \
        [2, "CHECK_FAIL: gnu-have-aggressive-compression zero@256 iter=2 code=4"]
    # Outside a pool, the job's own events and files, as before.
    if layout == "job":
        return
    alone = dict(meta1)
    alone.pop("pool")
    assert rh.errors_of(str(runs / one), alone, "fini", rh.read_events(str(runs / one))) == [0, None]
    assert rh.pooled_bench(str(runs / one), alone) is None


# ── bench_runner/share.py: the oracle wait ──────────────────────────────────

@pytest.fixture
def pooled(tmp_path, monkeypatch):
    from bench_runner import pool as pool_mod, share
    p = pool_mod.Pool("W", str(tmp_path))
    os.makedirs(p.dir)
    monkeypatch.setattr(share, "pool", p)
    monkeypatch.setattr(share, "POLL_S", 0.05)
    return share, p


def _other_claims(p, name):
    """Claim @p name as another job would (its owner file names it)."""
    os.makedirs(p.path(name), exist_ok=True)
    with open(p.path(name, "owner"), "w") as f:
        f.write("otherhost.1\n")


def test_oracle_wait(pooled):
    share, p = pooled
    assert share.wait_oracle("cfg", "cfg")                       # nobody makes it: at once
    _other_claims(p, "check/cfg")                              # its check unit, oracle step not settled
    threading.Timer(0.3, lambda: p.mark("check/cfg", "oracle")).start()
    t0 = time.time()
    assert share.wait_oracle("cfg", "cfg") and time.time() - t0 >= 0.25
    _other_claims(p, "oracle_cfg")                             # its generation, not done
    share.ORACLE_WAIT_S, keep = 0.2, share.ORACLE_WAIT_S
    try:
        assert not share.wait_oracle("cfg", "cfg")             # gave up
        assert share.oracle_timeout_entry("cfg", "cfg").startswith("POOL_ORACLE_TIMEOUT: the oracle of cfg")
    finally:
        share.ORACLE_WAIT_S = keep
    p.stop("n9 interrupted")
    with pytest.raises(share.PoolStopped):
        share.wait_oracle("cfg", "cfg")
    with pytest.raises(share.PoolStopped):
        share.take("perf", "cfg")


def test_take_once_per_unit(pooled):
    share, p = pooled
    assert share.take("perf", "a") and share.take("perf", "a")      # asked twice: still this job's
    _other_claims(p, "perf/b")
    assert not share.take("perf", "b") and not share.take("perf", "b")
    assert share.units == [1, 1]


# ── the real script, twice at once ──────────────────────────────────────────

CONFIGS = ["cmake-build-omp-gnu", "cmake-build-omp-gnu-have-aggressive-compression", "cmake-build-omp-llvm"]


def _fake_tree(root, calls):
    """Three flat builds whose binaries record each call (an atomic mkdir
    of calls/c_<n>, what: name, arguments, cwd); vector_check writes the
    oracle (data_tab_x, after 1 s), a check without it in its cwd exits 4,
    the other modes exit 0 at once."""
    body = ("#!/bin/bash\n"
            f"k=1; until mkdir \"{calls}/c_$k\" 2>/dev/null; do k=$((k+1)); done\n"
            f"echo \"$(basename \"$0\") $* cwd=$PWD\" > \"{calls}/c_$k/what\"\n"
            "sleep ${FAKE_BENCH_SLEEP:-0.05}\n"
            "case \" $* \" in *\" --mode perf \"*|*\" --test_stl \"*|*\" --no_io \"*) exit 0;; esac\n"
            "case \"$(basename \"$0\")\" in *vector_check) sleep 1; echo x > data_tab_x; exit 0;; esac\n"
            "[ -f data_tab_x ] || { echo \"no oracle in $PWD\" >&2; exit 4; }\n")
    calls.mkdir()
    for c in CONFIGS:
        d = root / c
        d.mkdir(parents=True)
        for n in ("vector_check", "vector_perf", "zero_256"):
            (d / f"shrink_bench_omp_{n}").write_text(body)
            (d / f"shrink_bench_omp_{n}").chmod(0o755)


def _bench_cmd(root, out, extra=()):
    return ([sys.executable, str(PYDIR / "bench_and_test.py"), "--build-root", str(root),
             "--output-dir", str(out), "--configs"] + CONFIGS
            + ["--variants", "vector", "zero", "--sizes", "256", "--threads", "1", "2",
               "--full-reps-threads", "1", "--reps_perf", "2", "--reps_check", "2", "--check-threads", "1",
               "--max-threads", "2", "--skip-ctest", "--skip-sanitizers", "--skip-seq-omp-compare",
               "--skip-rechunk", "--skip-scaling", "--fuzz-cases", "6", "--numa-interleave", "off"]
            + list(extra))


def _run_pool(tmp_path, pool, jobs=2, extra=(), on_start=None, env_extra=None):
    """bench_and_test.py @p jobs times at once, the same --pool, as the
    shrink_gui jobs <pool>-n<k> (CHUNK_BENCH_KEEP_DIR names the job, its
    events in its run dir). @p on_start(procs): called once they are all
    started (an interrupt sent to one)."""
    procs = []
    for k in range(1, jobs + 1):
        gui = tmp_path / "gui" / f"{pool}-n{k}"
        gui.mkdir(parents=True)
        env = dict(os.environ, CHUNK_BENCH_KEEP_DIR=str(gui), CHUNK_BENCH_EVENTS=str(gui / "events.jsonl"))
        env.pop("CHUNK_BUILD_ROOT", None)
        env.update(env_extra or {})
        procs.append(subprocess.Popen(_bench_cmd(tmp_path / "root", tmp_path / "out", ["--pool", pool] + list(extra)),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(tmp_path)))
    if on_start:
        on_start(procs)
    outs = [p.communicate(timeout=300)[0].decode() for p in procs]
    return [p.returncode for p in procs], outs


def _calls(tmp_path):
    out = []
    for d in (tmp_path / "calls").iterdir():
        what = (d / "what").read_text().split()
        out.append((what[0], " ".join(w for w in what[1:] if not w.startswith("cwd=")),
                    next(w[4:] for w in what if w.startswith("cwd="))))
    return out


def test_two_jobs_share_the_campaign(tmp_path):
    _fake_tree(tmp_path / "root", tmp_path / "calls")
    codes, outs = _run_pool(tmp_path, "P1")
    assert codes == [0, 0], outs
    arch = next(p for p in (tmp_path / "out").iterdir())
    calls = _calls(tmp_path)
    # Each oracle generated once, in the shared oracle/<config>.
    gens = sorted(cwd for exe, args, cwd in calls if exe == "shrink_bench_omp_vector_check")
    assert gens == [str(arch / "oracle" / c) for c in sorted(CONFIGS)]
    # Each check unit once: --reps_check 2 per config, in its oracle; none failed.
    checks = [cwd for exe, args, cwd in calls if exe.endswith("zero_256") and args == ""]
    assert sorted(checks) == sorted(str(arch / "oracle" / c) for c in CONFIGS for _ in range(2))
    # Each perf unit once: vector 1+1 and zero 2+1 on gnu and llvm, zero 2+1 on
    # the aggressive config -- run by the job that did gnu (its baseline).
    perf = [(exe, cwd) for exe, args, cwd in calls if args == "--mode perf"]
    assert len(perf) == 13
    perf_dirs = {cwd for _e, cwd in perf}
    assert perf_dirs <= {str(arch / "pool_P1" / "n1"), str(arch / "pool_P1" / "n2")}
    # STL: one per front-end; fuzz: each case index claimed by one job at most.
    assert len([1 for _e, args, _c in calls if "--test_stl" in args]) == 2
    pool = arch / ".pools" / "P1"
    assert len(list((pool / "fuzz").iterdir())) <= 6 and not (pool / "stop").exists()
    # Every unit claimed exactly once, by one of the two jobs.
    units = sorted(str(p.relative_to(pool)) for p in pool.glob("*/*") if (p / "owner").exists())
    assert units == sorted(set(units))
    assert [u for u in units if u.startswith("perf/")] == ["perf/cmake-build-omp-gnu", "perf/cmake-build-omp-llvm"]
    # Each job its own outputs, the CSV rows of the two disjoint and complete.
    rows = []
    for k in (1, 2):
        jd = arch / "pool_P1" / f"n{k}"
        assert (jd / "bench_output.log").exists() and (jd / "run_logs").is_dir()
        with open(jd / "bench_summary.csv") as f:
            rows += [(r["Config"], r["Variant"], r["Threads"]) for r in csv.DictReader(f)]
        assert (tmp_path / "gui" / f"P1-n{k}" / "bench_summary.csv").exists()     # keep_run_outputs
    assert len(rows) == len(set(rows)) == 10
    assert not (arch / "bench_summary.csv").exists() and not (arch / "bench_errors.log").exists()
    for k, o in zip((1, 2), outs):
        assert f"this job is n{k} (" in o and f"Run outputs → {arch / 'pool_P1' / f'n{k}'}" in o
        assert f"[pool] P1: n{k} did " in o
    # The skips name the other job by its tag.
    assert any("done by n2" in o for o in outs) or any("done by n1" in o for o in outs)


def test_a_stopped_pool_stops_a_job(tmp_path):
    """Another job stopped the pool (interrupted, crashed): this one claims
    nothing, writes its (empty) summary and ends with code 1."""
    _fake_tree(tmp_path / "root", tmp_path / "calls")
    arch = tmp_path / "out" / os.uname().machine
    (arch / ".pools" / "P2").mkdir(parents=True)
    (arch / ".pools" / "P2" / "stop").write_text("n7: n7 interrupted\n")
    codes, outs = _run_pool(tmp_path, "P2", jobs=1)
    assert codes == [1], outs
    assert "🛑 [pool] P2: stopped by n7: n7 interrupted" in outs[0]
    assert (arch / "pool_P2" / "n1" / "bench_summary.csv").exists() and not _calls(tmp_path)
    end = json.loads((tmp_path / "gui" / "P2-n1" / "events.jsonl").read_text().splitlines()[-1])
    assert end["kind"] == "end" and end["stopped_by"] == "n7: n7 interrupted"


def test_an_interrupted_job_stops_the_pool(tmp_path):
    """Ctrl+C (SIGINT) in one job: its partial results written, code 130,
    and the pool stopped -- the other job claims nothing more (code 1)
    rather than leave the interrupted job's units undone unknowingly."""
    import signal
    _fake_tree(tmp_path / "root", tmp_path / "calls")

    def interrupt_first(procs):
        calls = tmp_path / "calls"
        t0 = time.time()
        while len(list(calls.iterdir())) < 4 and time.time() - t0 < 60:   # both well into the campaign
            time.sleep(0.05)
        procs[0].send_signal(signal.SIGINT)

    codes, outs = _run_pool(tmp_path, "P4", on_start=interrupt_first, env_extra={"FAKE_BENCH_SLEEP": "0.4"})
    assert codes == [130, 1], outs
    stop = next((tmp_path / "out").iterdir()) / ".pools" / "P4" / "stop"
    assert stop.read_text().strip().endswith(": n1 interrupted")
    assert "User interrupt" in outs[0] and "stopped for every job (n1 interrupted)" in outs[0]
    assert "🛑 [pool] P4: stopped by " in outs[1] and "this job (n2) claims nothing more" in outs[1]


def test_rerun_failed_replays_the_whole_pool(tmp_path):
    """--rerun-failed with --pool: the failures every job recorded, in one job."""
    _fake_tree(tmp_path / "root", tmp_path / "calls")
    arch = tmp_path / "out" / os.uname().machine
    for k, code in ((1, 0), (2, 3)):
        jd = arch / "pool_P3" / f"n{k}"
        jd.mkdir(parents=True)
        run = {"executor": "run_cmd", "cmd": [sys.executable, "-c", f"import sys; sys.exit({code})"], "env": {},
               "timeout": 30, "ld": None, "cwd": None, "must_contain": None}
        (jd / "failed_cases.jsonl").write_text(json.dumps({"kind": "CHECK_FAIL", "summary": f"CHECK_FAIL n{k}",
                                                           "runs": [run]}) + "\n")
    env = dict(os.environ)
    env.pop("CHUNK_BENCH_KEEP_DIR", None)
    p = subprocess.run(_bench_cmd(tmp_path / "root", tmp_path / "out", ["--pool", "P3", "--rerun-failed"]),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(tmp_path), timeout=120)
    o = p.stdout.decode()
    assert p.returncode == 1, o
    assert "2 case(s)" in o and "1 pass, 1 still fail" in o
    (mine,) = [d for d in (arch / "pool_P3").iterdir() if d.name not in ("n1", "n2")]   # <host>.<pid>
    assert [json.loads(l)["summary"] for l in (mine / "failed_cases.jsonl").read_text().splitlines()] == \
        ["CHECK_FAIL n2"]
    assert not (arch / ".pools").exists()          # a replay claims nothing


def test_without_pool_the_paths_are_unchanged(tmp_path):
    _fake_tree(tmp_path / "root", tmp_path / "calls")
    env = dict(os.environ)
    env.pop("CHUNK_BENCH_KEEP_DIR", None)
    p = subprocess.run(_bench_cmd(tmp_path / "root", tmp_path / "out", ["--fuzz-cases", "0"]),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(tmp_path), timeout=300)
    o = p.stdout.decode()
    assert p.returncode == 0, o
    arch = tmp_path / "out" / os.uname().machine
    assert sorted(x.name for x in arch.iterdir()) == ["bench_errors.log", "bench_output.log", "bench_summary.csv",
                                                       "failed_cases.jsonl", "oracle", "run_logs"]
    assert "[pool]" not in o
