# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
""""Nœuds" N > 1 as ONE batch job of N nodes ("un job de N nœuds"): the
directives ask N nodes, run.sh checks once and starts node.sh on every node
(a task each), node K works in <run>/n<K>. The helper gathers the nodes as
it gathers the N jobs of one node. Slurm is faked: node.sh run twice with
SLURM_PROCID 0 / 1, or the fake ccc_mprun starting -n N tasks."""
import base64
import json
import os
import subprocess

from conftest import BATCH, FAKE_PROGRAM, PYDIR, REPO
from test_multi_launch_pool import _fake_tree

from shrink_gui import qos
from shrink_gui.runscripts import directive_cores, make_node_sh, make_run_sh, node_launch, nodes_directives

MILAN = "#MSUB -q milan-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 128\n"


def test_directives_and_launch_for_three_nodes():
    d = nodes_directives(MILAN, 3, "#MSUB")
    assert d == "#MSUB -q milan-bxi\n#MSUB -N 3\n#MSUB -n 3\n#MSUB -c 128\n"
    assert qos.job_cpus(d) == 3 * 128                       # the QoS sees the job's cores
    assert nodes_directives("#MSUB -q x\n", 2, "#MSUB") == "#MSUB -q x\n#MSUB -N 2\n#MSUB -n 2\n"
    sb = "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=64\n"
    assert nodes_directives(sb, 4, "#SBATCH") == "#SBATCH --nodes=4\n#SBATCH --ntasks=4\n#SBATCH --cpus-per-task=64\n"
    assert directive_cores(MILAN) == 128 and directive_cores(sb) == 64
    # No launcher: the site's step command; gh200's launcher keeps its container.
    assert node_launch("", BATCH["step"], 3, 128) == "ccc_mprun -N 3 -n 3 -c 128"
    assert node_launch("ccc_mprun -C ccc-rhel9 -n 1 -c 288", "ccc_mprun", 3, 288) == \
        "ccc_mprun -N 3 -n 3 -C ccc-rhel9 -c 288"
    assert node_launch("", None, 2, None) == "srun -N 2 -n 2"


def test_run_sh_and_node_sh_text():
    launch = node_launch("", "ccc_mprun", 3, 128)
    run = make_run_sh("~/r", "python", "python3 x.py --pool=R", {"mode": "warn", "builds": ["a"]},
                      {"n": 3, "launch": launch})
    # The checks once, before the launch; then one task per node, the exit of each.
    assert run.index("gui_check_builds warn") < run.index('ccc_mprun -N 3 -n 3 -c 128 bash "$RUN_DIR/node.sh"')
    assert "\npython3 x.py --pool=R\n" not in run            # the command runs in node.sh only
    assert 'for k in $(seq 1 3); do' in run and run.rstrip().endswith("exit $code")
    node = make_node_sh("~/r", "python", "python3 x.py --pool=R")
    for line in ('K=$(( ${SLURM_PROCID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}} + 1 ))',
                 'ND="$RUN_DIR/n$K"', 'export CHUNK_BENCH_EVENTS="$ND/events.jsonl"',
                 'export CHUNK_BENCH_KEEP_DIR="$ND"', 'echo "$code" > "$ND/exit_code"',
                 "\npython3 x.py --pool=R\n"):
        assert line in node
    assert "gui_check_builds" not in node and "git_start" not in node
    # Without nodes, run.sh is as before.
    plain = make_run_sh("~/r", "python", "python3 x.py")
    assert "node.sh" not in plain and "\npython3 x.py\n" in plain


def _job_run(runs, rid, nodes, command, program="multi_launch_stress.py"):
    """Run dir @p rid of a job of @p nodes nodes, as remote_helper.act_launch makes it."""
    d = runs / rid
    d.mkdir(parents=True)
    meta = dict(FAKE_PROGRAM, program=program, pool=rid, nodes=nodes, layout="job", command=command)
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "node.sh").write_text(make_node_sh(str(REPO), str(PYDIR), command))
    return d, meta


def _call(rh, capsys, act, *args):
    getattr(rh, "act_" + act)(list(args))
    return json.loads(capsys.readouterr().out)


def test_two_nodes_share_the_runs_and_are_gathered(tmp_path, capsys):
    """multi_launch_stress.py --pool on two "nodes" of one job: node.sh twice
    at once (SLURM_PROCID 0, 1), on a fake build tree; one run fails. The
    scripts name their jobs n1 / n2 from $CHUNK_BENCH_KEEP_DIR = <run>/n<K>."""
    from shrink_gui import remote_helper as rh
    root, runs = tmp_path / "root", tmp_path / "runs"
    _fake_tree(root, tmp_path / "calls", fail_at=7)
    cmd = (f"python3 multi_launch_stress.py --builds omp-a,omp-b --build-root {root} --runs 200 "
           "--min_threads 1 --max_threads 2 --numa-interleave off --pool=J")
    d, meta = _job_run(runs, "J", 2, cmd)
    env = dict(os.environ)
    env.pop("CHUNK_BUILD_ROOT", None)
    procs = [subprocess.Popen(["bash", str(d / "node.sh")], env=dict(env, SLURM_PROCID=str(r)),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for r in (0, 1)]
    outs = [p.communicate(timeout=180)[0].decode() for p in procs]
    assert sorted(p.returncode for p in procs) == [1, 3], outs
    # Node n1's output on the task's output too (the run's out.log), n2's in its file only.
    assert "this job is n1 (" in outs[0] and outs[1] == ""
    for k in (1, 2):
        log = (d / f"n{k}" / "out.log").read_text()
        assert f"this job is n{k} (" in log and f"node n{k} end" in log
        assert (d / f"n{k}" / "exit_code").read_text().strip() in ("1", "3")
    members = rh.pool_members(str(d), meta)
    assert [os.path.basename(m) for m, _ in members] == ["n1", "n2"] and members[1][1]["node"] == 2
    s = _call(rh, capsys, "summary", str(runs), "J")
    total = s["rows"][-1]
    assert total[0] == "TOTAL" and int(total[2]) >= 1 and s["note"].endswith("— 2 nœuds sur 2")
    p = _call(rh, capsys, "points", str(runs), "J")
    idx = [r[0] for r in p["rows"]]
    assert idx == sorted(idx) and len(idx) == len(set(idx)) == int(total[1])   # the two nodes' runs, none twice
    blob = base64.b64encode(json.dumps({"files": ["error_*.txt"]}).encode()).decode()
    e = _call(rh, capsys, "errors", str(runs), "J", blob)
    assert e["count"] >= 1 and all(f["name"].startswith("error_") for f in e["files"])
    (d / "exit_code").write_text("3")
    assert rh.errors_of(str(d), meta, "fini", []) == [int(total[2]), None]
    assert json.loads((d / "errors_count.json").read_text()) == [int(total[2]), None]   # both nodes over: cached
    assert rh.run_progress(str(d), meta, [])[1] == 200


def _spec(command, nodes):
    launch = node_launch("", BATCH["step"], nodes, 4)
    return {"mode": "msub", "batch": BATCH, "name": "t", "time_s": 600, "launcher": "", "launch.sh": "",
            "run.sh": make_run_sh("~/shrink-array", FAKE_PROGRAM["cwd"], command, None,
                                  {"n": nodes, "launch": launch}),
            "node.sh": make_node_sh("~/shrink-array", FAKE_PROGRAM["cwd"], command),
            "msub": nodes_directives("#MSUB -q fake\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 4\n", nodes, "#MSUB"),
            "meta": dict({k: FAKE_PROGRAM[k] for k in ("out_re", "out_files", "progress_re", "eta_re",
                                                       "summary", "summary_file")},
                         program="fake_stress.py", partition="gh200-bxi", mode="msub", repo="~/shrink-array",
                         command=command, launcher=launch, pool="JN", nodes=nodes, layout="job")}


def test_one_batch_job_of_three_nodes(site):
    """Through the helper and the fake ccc_msub / ccc_mprun: one job, one
    run dir, three node directories; the run's exit code is non-zero when a
    node failed, its progress and summary gather the nodes."""
    blob = base64.b64encode(json.dumps(_spec("python3 fake_stress.py --runs 6 --step 0.01", 3)).encode()).decode()
    r = site.helper("launch", "~/shrink_runs", "JN", blob)
    d = site.runs / "JN"
    job = (d / "job.sh").read_text()
    assert "#MSUB -N 3\n#MSUB -n 3\n#MSUB -c 4\n" in job and 'bash "$D/run.sh"' in job and r["jobid"]

    def run():
        return {x["id"]: x for x in site.helper("list", "~/shrink_runs")["runs"]}["JN"]
    assert site.wait(lambda: run()["status"] == "fini", 60)
    assert run()["code"] == 0 and run()["progress"][:2] == [6, 6]
    log = (d / "out.log").read_text()
    assert "[fake ccc_mprun] options: -N 3 -n 3 -c 4" in log
    assert all(f"[gui] node n{k}: exit code 0" in log for k in (1, 2, 3))
    assert log.count("Run 6/6") == 1                         # node n1's output, the others in their files
    for k in (1, 2, 3):
        assert "Run 6/6" in (d / f"n{k}" / "out.log").read_text()
    s = site.helper("summary", "~/shrink_runs", "JN")
    assert s["rows"][-1][:2] == ["TOTAL", "18"] and s["note"].endswith("3 nœuds sur 3")
    # A node that fails fails the job.
    blob = base64.b64encode(json.dumps(_spec("python3 fake_stress.py --runs 2 --step 0.01 --rc 3", 2)).encode()).decode()
    site.helper("launch", "~/shrink_runs", "JF", blob)
    assert site.wait(lambda: {x["id"]: x for x in site.helper("list", "~/shrink_runs")["runs"]}["JF"]["status"] == "fini", 60)
    assert {x["id"]: x for x in site.helper("list", "~/shrink_runs")["runs"]}["JF"]["code"] == 3
    assert site.helper("delete", "~/shrink_runs", "JN")["ok"] and not d.exists()


# inti's ccc_mqinfo rows of the user's case (milan-bxi, 128 cores a node).
MQINFO = """Name     Partition  Priority  MaxCPUs  SumCPUs  MaxNodes  MaxRun  MaxSub        MaxTime
-------  ---------  --------  -------  -------  --------  ------  ------  -------------
long      rome        20      512      512                       100     3-00:00:00
long             *        20        0        0                         0     3-00:00:00
normal           *        20                                         100     1-00:00:00
xlong    milan-bxi        20      128      128                       100   365-00:00:00
test     milan-bxi        40     2048     2048                         2       00:30:00
"""


def test_qos_of_a_job_of_several_nodes():
    rows = qos.parse_mqinfo(MQINFO)
    H = 3600
    xl = qos.limits(rows, "xlong", "milan-bxi")
    assert xl["sum_cpus"] == 128 and qos.limits(rows, "normal", "milan-bxi")["sum_cpus"] is None
    cpus = qos.job_cpus(nodes_directives(MILAN, 4, "#MSUB"))
    assert cpus == 512
    assert qos.choose(rows, "milan-bxi", cpus, 12 * H)[0] == "normal"
    name, _row, err = qos.choose(rows, "milan-bxi", cpus, 48 * H)
    assert name is None and err == "aucune QoS n'accepte 2 j 00 h sur milan-bxi pour 512 cœurs (au plus 1 j 00 h)"
    assert qos.choose(rows, "milan-bxi", 128, 48 * H)[0] == "xlong"          # one node: xlong
    # N jobs of one node under xlong: one at a time; under normal, together.
    assert qos.serializes(xl, 128, 4) and not qos.serializes(xl, 128, 1)
    assert not qos.serializes(qos.limits(rows, "normal", "milan-bxi"), 128, 4)
    # sacctmgr's per-user TRES list.
    assert qos._int("cpu=256,node=2") == 256


def test_node_tag_from_the_keep_dir(monkeypatch):
    """bench_runner/pool.py names the job n<K> from $CHUNK_BENCH_KEEP_DIR in
    both layouts: the run <pool>-n<K> of a job of one node, the directory
    <pool>/n<K> of a node of one job; anything else: <host>.<pid>."""
    from bench_runner.pool import Pool
    for keep, tag in (("/r/P-n2", "n2"), ("/r/P/n3", "n3"), ("/r/Q/n3", None), ("/r/P/x3", None), ("", None)):
        monkeypatch.setenv("CHUNK_BENCH_KEEP_DIR", keep)
        assert Pool._tag("P", "h") == (tag or "h.%d" % os.getpid())
