"""The shell scripts a run is made of: run.sh (compute node), launch.sh (detached mode)."""
import re
import shlex

from shrink_gui.config import NEWEST_PY


def shq(v):
    """Quote for bash when needed, keeping $VAR expansion (a default such as
    /ccc/work/.../$USER/... is meant to expand on inti)."""
    if re.fullmatch(r"[A-Za-z0-9_./:,+@%=-]+", str(v)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'

def make_run_sh(repo, cwd, command):
    """What runs on the compute node."""
    return f"""#!/bin/bash
echo "[gui] node $(hostname)  start $(date '+%F %T')  job ${{SLURM_JOB_ID:-}}"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
R={shlex.quote(repo)}
R="${{R/#\\~/$HOME}}"
cd "$R" || {{ echo "[gui] repository $R not found"; exit 97; }}
# The code the run actually starts on (meta.json has the one it was submitted on).
if git rev-parse HEAD > "$RUN_DIR/git_start" 2>/dev/null; then
    echo "[gui] commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))$(git status --porcelain --untracked-files=no 2>/dev/null | grep -q . && echo ' + local changes')"
else
    rm -f "$RUN_DIR/git_start"
fi
# bench_and_test.py writes its structured progress here (bench_runner/events.py).
export CHUNK_BENCH_EVENTS="$RUN_DIR/events.jsonl"
# A batch job's shell may not have Environment Modules set up (compile_all.sh
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
echo "[gui] command: "{shlex.quote(command)}
{command}
"""

def make_launch_sh(mprun_opts):
    """ccc_mprun mode: runs on the login node, detached."""
    return f"""#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
trap 'touch "$D/stopped"' TERM
echo "[gui] ccc_mprun {mprun_opts}"
ccc_mprun {mprun_opts} bash "$D/run.sh" &
m=$!
wait $m
code=$?
while kill -0 $m 2>/dev/null; do wait $m; code=$?; done
echo "$code" > "$D/exit_code"
echo "[gui] end $(date '+%F %T')  exit code $code"
"""

