# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The shell scripts a run is made of: run.sh (compute node), node.sh (each
node of a job of several nodes), launch.sh (detached mode: the site's
detached command -- ccc_mprun on inti -- from the login node, or the run
itself on this machine)."""
import re
import shlex

from shrink_gui.config import NEWEST_PY


def shq(v):
    """Quote for bash when needed, keeping $VAR expansion (a default such as
    /ccc/work/.../$USER/... is meant to expand on the cluster)."""
    if re.fullmatch(r"[A-Za-z0-9_./:,+@%=-]+", str(v)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'

# Are the executables of the builds a run uses up to date? On the node, before
# the command: `ninja -n` on the executables already built under examples/ of
# each build directory of this architecture -- the targets compile_all.sh made,
# not those it was never asked to build. "[n/m] ..." lines = work to do =
# stale. Exit 96 ("refuse") or a warning ("warn"). Without ninja, skipped.
CHECK_BUILDS_SH = r"""
gui_check_builds() {
    local mode="$1" root="$2"; shift 2
    root="${root/#\~/$HOME}"
    if ! command -v ninja >/dev/null 2>&1; then
        echo "[gui] build check: no ninja on this node, skipped"; return 0
    fi
    local arch b name d n=0 stale=() unknown=() exes out
    arch=$(uname -m)
    for b in "$@"; do
        name="$b"; [[ "$name" == cmake-build-* ]] || name="cmake-build-$name"
        for d in "$root/$name"/*__"$arch" "$root/$name"; do
            [ -f "$d/build.ninja" ] || continue
            mapfile -t exes < <(cd "$d" && find examples -type f -perm -u+x -not -path '*/CMakeFiles/*' 2>/dev/null | sort)
            [ ${#exes[@]} -gt 0 ] || break
            n=$((n + 1))
            if ! out=$(ninja -C "$d" -n "${exes[@]}" 2>&1); then
                unknown+=("$d")
            elif grep -q '^\[' <<< "$out"; then
                stale+=("$d ($(grep -c '^\[' <<< "$out") step(s))")
            fi
            break
        done
    done
    for d in "${unknown[@]}"; do echo "[gui] build check: ninja -n failed in $d (not checked)"; done
    if [ ${#stale[@]} -eq 0 ]; then
        echo "[gui] build check: $n build(s) up to date"; return 0
    fi
    echo "[gui] build check: ${#stale[@]} of $n build(s) NOT up to date (compile them first):"
    printf '[gui]   %s\n' "${stale[@]}"
    if [ "$mode" = refuse ]; then
        echo "[gui] not started (the check can be set to warn or off in the form)"; exit 96
    fi
    echo "[gui] warning only: started anyway"
}
"""


def check_builds_call(check):
    """The run.sh lines of the build check @p check = {"mode": "refuse"|"warn",
    "root": build root (may hold $USER), "builds": [names] or None,
    "builds_file": a file of names, one per line (# comments), relative to the
    program's directory -- or a single name when there is no such file};
    "" when there is none."""
    if not check or check.get("mode") not in ("refuse", "warn"):
        return ""
    root = shq(check.get("root") or "$CHUNK_BUILD_ROOT")
    if check.get("builds_file"):   # a file of names if there is one, else a single name
        f = shq(check["builds_file"])
        names = (f'if [ -f {f} ]; then mapfile -t GUI_BUILDS < <(sed -e "s/#.*//" -e "s/[[:space:]]//g" {f} '
                 f'| grep -v "^$"); else GUI_BUILDS=({f}); fi')
    else:
        names = "GUI_BUILDS=(" + " ".join(shq(b) for b in check.get("builds") or []) + ")"
    return (CHECK_BUILDS_SH + names + "\n"
            + f'gui_check_builds {check["mode"]} {root} "${{GUI_BUILDS[@]}}"\n')


def _env_sh(repo, cwd, with_git=True):
    """The part of run.sh (and node.sh) that sets the scene: the repository
    (its commit recorded in git_start when @p with_git), the modules and
    site_env.sh, the newest python3, the program's directory @p cwd."""
    git = f"""# The code the run actually starts on (meta.json has the one it was submitted on).
if git rev-parse HEAD > "$RUN_DIR/git_start" 2>/dev/null; then
    echo "[gui] commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))$(git status --porcelain --untracked-files=no 2>/dev/null | grep -q . && echo ' + local changes')"
else
    rm -f "$RUN_DIR/git_start"
fi
""" if with_git else ""
    return f"""R={shlex.quote(repo)}
R="${{R/#\\~/$HOME}}"
cd "$R" || {{ echo "[gui] repository $R not found"; exit 97; }}
{git}# A batch job's shell may not have Environment Modules set up (compile_all.sh
# needs `module`): take the login profile then.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
[ -f scripts/site_env.sh ] && source scripts/site_env.sh
export CHUNK_BUILD_ROOT="${{CHUNK_BUILD_ROOT:-${{CHUNK_BUILD_REP:-.}}}}"
# Python writes to a file here, not a terminal: unbuffered, or the log only
# shows up by 8 KiB blocks.
export PYTHONUNBUFFERED=1
# The newest python3 of the PATH, under the name the command uses: the
# system one can be too old for the scripts (bench_and_test.py wants 3.9).
PY={NEWEST_PY}
if [ -n "$PY" ] && [ "$PY" != "$(command -v python3)" ]; then
    python3() {{ "$PY" "$@"; }}
fi
echo "[gui] $(python3 --version 2>&1) ($PY)"
cd {shlex.quote(cwd)} || exit 97
echo "[gui] cwd $PWD"
"""


def make_run_sh(repo, cwd, command, check=None, nodes=None):
    """What runs on the compute node. @p check: the build check (check_builds_call).
    @p nodes: a job of several nodes, {"n": N, "launch": the command that starts
    one task per node (node_launch)}: run.sh then makes the checks once, on
    the job's first node, and starts node.sh (make_node_sh) on every node,
    which runs @p command there; run.sh's exit code is the launch's, non-zero
    when any node failed."""
    if nodes:
        tail = f"""echo "[gui] command on each of the {int(nodes['n'])} nodes: "{shlex.quote(command)}
echo "[gui] {nodes['launch']} bash node.sh  (logs in $RUN_DIR/n<K>/out.log, node n1's below)"
{nodes['launch']} bash "$RUN_DIR/node.sh"
code=$?
for k in $(seq 1 {int(nodes['n'])}); do
    echo "[gui] node n$k: exit code $(cat "$RUN_DIR/n$k/exit_code" 2>/dev/null || echo '? (not started, or killed)')"
done
exit $code
"""
    else:
        tail = f"""echo "[gui] command: "{shlex.quote(command)}
{command}
"""
    return f"""#!/bin/bash
echo "[gui] node $(hostname)  start $(date '+%F %T')  job ${{SLURM_JOB_ID:-}}"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
# When the run started, by the node's clock (the Durée column of the window).
date +%s > "$RUN_DIR/start_time" 2>/dev/null
# bench_and_test.py writes its structured progress here (bench_runner/events.py).
export CHUNK_BENCH_EVENTS="$RUN_DIR/events.jsonl"
# ... and, once over, a copy of its CSV / errors log / failure list (the output
# directory is shared by the runs of an architecture, each rewrites them).
export CHUNK_BENCH_KEEP_DIR="$RUN_DIR"
{_env_sh(repo, cwd)}{check_builds_call(check)}{tail}"""


def make_node_sh(repo, cwd, command):
    """A job of several nodes: what each of its tasks runs (one per node,
    started by run.sh). Node K = the task's rank + 1 works in $RUN_DIR/n<K>:
    its out.log (node n1's also on the task's output, i.e. the run's out.log),
    events.jsonl, the copies the program keeps ($CHUNK_BENCH_KEEP_DIR: the
    scripts name the job n<K> from that directory, bench_runner/pool.py), its
    start_time and exit_code. The scene is set again: a task can start in
    another environment than the job's shell (gh200's launcher runs it in a
    RHEL 9 container), and shell functions do not cross srun."""
    return f"""#!/bin/bash
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
# The task's rank: Slurm's (srun, ccc_mprun), else an MPI launcher's.
K=$(( ${{SLURM_PROCID:-${{PMI_RANK:-${{OMPI_COMM_WORLD_RANK:-0}}}}}} + 1 ))
ND="$RUN_DIR/n$K"
mkdir -p "$ND" || exit 97
date +%s > "$ND/start_time" 2>/dev/null
if [ "$K" = 1 ]; then
    exec > >(tee -a "$ND/out.log") 2>&1
    TEE=$!
else
    exec >> "$ND/out.log" 2>&1
fi
echo "[gui] node $(hostname) n$K  start $(date '+%F %T')  job ${{SLURM_JOB_ID:-}}"
export CHUNK_BENCH_EVENTS="$ND/events.jsonl"
export CHUNK_BENCH_KEEP_DIR="$ND"
{_env_sh(repo, cwd, with_git=False)}echo "[gui] command: "{shlex.quote(command)}
{command}
code=$?
echo "$code" > "$ND/exit_code"
echo "[gui] node n$K end $(date '+%F %T')  exit code $code"
if [ -n "${{TEE:-}}" ]; then exec >&- 2>&-; wait "$TEE" 2>/dev/null; fi
exit $code
"""


# The node and task options of a launcher (-N 1, -n1, --nodes=1, --ntasks 1),
# replaced by those of a job of several nodes.
_NODE_OPTS_RE = re.compile(r"(?<!\S)(?:-[Nn]\s*\d+|--(?:nodes|ntasks)(?:=|\s+)\d+)(?!\S)")
_CORES_RE = re.compile(r"(?<!\S)(?:-c\s*|--cpus-per-task(?:=|\s+))(\d+)(?!\S)")


def directive_cores(directives):
    """The cores per task of the directives (-c / --cpus-per-task), or None."""
    m = _CORES_RE.findall(re.sub(r"^#[A-Z]+", " ", directives or "", flags=re.M))
    return int(m[-1]) if m else None


def nodes_directives(directives, n, directive):
    """The partition's directive lines made for one job of @p n nodes, one
    task per node: the values of -N / --nodes and -n / --ntasks set to @p n
    (lines "<directive> -N <n>" / "-n <n>" added when there are none); -c,
    the cores of each task, stays the node's."""
    out, seen = [], set()
    for ln in (directives or "").splitlines():
        if ln.lstrip().startswith("#"):
            def put(m):
                tok = m.group(0)
                flag = "N" if tok.startswith(("-N", "--nodes")) else "n"
                seen.add(flag)
                if tok.startswith("--"):
                    return re.sub(r"\d+$", str(n), tok)
                return f"-{flag} {n}"
            ln = _NODE_OPTS_RE.sub(put, ln)
        out.append(ln)
    for flag in ("N", "n"):
        if flag not in seen:
            out.append(f"{directive} -{flag} {n}")
    return "\n".join(out) + "\n"


def node_launch(launcher, step, n, cores):
    """The command run.sh starts node.sh with, one task on each of @p n nodes:
    the partition's @p launcher when it has one (gh200: ccc_mprun -C
    ccc-rhel9, its container), else the site's @p step command (srun;
    ccc_mprun on inti), its node and task options replaced by -N n -n n, and
    -c @p cores (all the node's) unless it gives its own."""
    words = (launcher or "").strip() or (step or "srun").strip()
    first, _, rest = words.partition(" ")
    rest = " ".join(_NODE_OPTS_RE.sub("", rest).split())
    opts = f"-N {int(n)} -n {int(n)}"
    if cores and not _CORES_RE.search(rest):
        opts += f" -c {int(cores)}"
    return " ".join(x for x in (first, opts, rest) if x)


def wait_sh(after=None, delay_s=0):
    """The detached mode's wait before its command: @p delay_s seconds, then the
    end of the runs of @p after = {"run": id, "runs": [ids], "cond": "ok"|"any"}
    ("runs", when given, all of them; else "run"), siblings in the
    runs directory (their exit_code files; a batch run that left the queue
    without one counts as failed). Not started: exit code 95. The file
    "waiting" marks the wait (status "en attente")."""
    if not after and not delay_s:
        return ""
    # Stopped while waiting ("Arrêter" sends TERM to the group, the sleep
    # included): end there, as a stopped run.
    out = ['trap \'rm -f "$D/waiting"; touch "$D/stopped"; echo 143 > "$D/exit_code"; exit 143\' TERM',
           'touch "$D/waiting"']
    if delay_s:
        out.append(f'echo "[gui] waiting {int(delay_s)} s before starting"; sleep {int(delay_s)}')
    # Each run in turn (the jobs of a submission over several nodes: all of
    # them, after["runs"]); one that failed stops the wait at once.
    for rid in (after.get("runs") or [after["run"]]) if after else []:
        cond = after.get("cond", "ok")
        out.append(f"""W="$D/../{rid}"
echo "[gui] waiting for the end of run {rid} ({'if it succeeds' if cond == 'ok' else 'whatever its result'})"
POLL="${{SHRINK_GUI_POLL:-30}}"
while [ ! -s "$W/exit_code" ]; do
    J=$(cat "$W/jobid" 2>/dev/null)
    if [ -n "$J" ] && ! squeue -h -j "$J" -o %i 2>/dev/null | grep -q "^$J\\b"; then
        sleep "$POLL"; [ -s "$W/exit_code" ] && break
        echo "[gui] run {rid} left the queue without an exit code: not started"
        rm -f "$D/waiting"; echo 95 > "$D/exit_code"; exit 0
    fi
    sleep "$POLL"
done
c=$(cat "$W/exit_code")
echo "[gui] run {rid} ended (exit code $c)"
""")
        if cond == "ok":
            out.append(f"""if [ "$c" != 0 ]; then
    echo "[gui] run {rid} failed: not started"; rm -f "$D/waiting"; echo 95 > "$D/exit_code"; exit 0
fi""")
    out.append('rm -f "$D/waiting"')
    out.append('trap \'touch "$D/stopped"\' TERM')
    return "\n".join(out) + "\n"


def make_launch_sh(mprun_opts, after=None, delay_s=0, *, command):
    """Detached mode: `<command> <mprun_opts> bash run.sh` (the site's [detached]
    command), run on the login node, detached; waits first if asked (wait_sh)."""
    return f"""#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
trap 'touch "$D/stopped"' TERM
{wait_sh(after, delay_s)}echo "[gui] {command} {mprun_opts}"
{command} {mprun_opts} bash "$D/run.sh" &
m=$!
wait $m
code=$?
while kill -0 $m 2>/dev/null; do wait $m; code=$?; done
echo "$code" > "$D/exit_code"
echo "[gui] end $(date '+%F %T')  exit code $code"
"""



def make_local_launch_sh(launcher="", time_s=0, after=None, delay_s=0):
    """Local mode: run.sh on this machine, detached, after the wait asked
    (wait_sh). @p time_s: the maximum duration (timeout; exit code 124 past
    it), 0 for none. @p launcher: a prefix of run.sh, such as
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0."""
    limit = f"timeout -k 60 {int(time_s)} " if time_s else ""
    prefix = launcher.strip() + " " if launcher.strip() else ""
    what = f"at most {int(time_s)} s" if time_s else "no time limit"
    return f"""#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
trap 'touch "$D/stopped"' TERM
{wait_sh(after, delay_s)}echo "[gui] on this machine ({what}): {prefix}bash run.sh"
{limit}{prefix}bash "$D/run.sh" &
m=$!
# timeout runs the command in a process group of its own, which the TERM of
# "Arrêter" (sent to this script's group) does not reach: pass it on.
trap 'touch "$D/stopped"; kill -TERM $m 2>/dev/null' TERM
wait $m
code=$?
while kill -0 $m 2>/dev/null; do wait $m; code=$?; done
{f'[ "$code" = 124 ] && echo "[gui] time limit reached ({int(time_s)} s)"' if time_s else ""}
echo "$code" > "$D/exit_code"
echo "[gui] end $(date '+%F %T')  exit code $code"
"""


def rerun_command(cmd, spec, repeat, log_dir):
    """@p cmd (a bench_and_test.py command line, as the window wrote it) made
    to replay the failures its last run recorded: spec["opt"] (--rerun-failed)
    with spec["repeat"] N and its full logs in @p log_dir (relative to the
    output directory, so the first run's logs are never overwritten). The
    same options of an earlier replay are replaced; the rest is kept as
    written ($VAR expansion included)."""
    s = " " + cmd
    s = re.sub(r"\s%s(?:=\S*|\s+(?!-)\S+)?(?=\s|$)" % re.escape(spec["opt"]), "", s)
    for flag in (spec["repeat"], spec["log_dir"]):
        s = re.sub(r"\s%s(?:=|\s+)\S+" % re.escape(flag), "", s)
    for flag in spec.get("drop", []):
        s = re.sub(r"\s%s(?=\s|$)" % re.escape(flag), "", s)
    tail = f"{spec['opt']} {spec['repeat']} {int(repeat)} {spec['log_dir']} {shq(log_dir)}"
    return (s.strip() + " " + tail).strip()
