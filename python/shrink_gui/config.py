"""Site knowledge: partitions, programs, defaults, chart palette. Edit here."""
import os
import re

APP = "shrink_gui"

# --------------------------------------------------------------------------
# Site knowledge -- edit here.
# --------------------------------------------------------------------------
# Partitions. "msub": the #MSUB lines of the batch script (-T, -r, -o and -e
# are added from the form). "launcher": how the batch job starts the command
# on its node -- empty runs it directly in the job's shell; gh200 goes through
# `ccc_mprun -C ccc-rhel9` as the interactive commands do (a Slurm
# --constraint=ccc-rhel9 is rejected: "Invalid feature specification").
# "mprun": the ccc_mprun options of the detached mode (no -s, no -T). All three
# are editable in the window and remembered per partition.
PRESETS = {
    "gh200-bxi": {
        "label": "gh200-bxi — ARM, 288 cœurs",
        "arch": "arm",
        "msub": "#MSUB -q gh200-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 288\n",
        "launcher": "ccc_mprun -C ccc-rhel9 -n 1 -c 288",
        "mprun": "-N1 -c 288 -p gh200-bxi -C ccc-rhel9 -Xfirst",
    },
    "milan-bxi": {
        "label": "milan-bxi — x86, 64 cœurs",
        "arch": "x86",
        "msub": "#MSUB -q milan-bxi\n#MSUB -N 1\n#MSUB -n 1\n#MSUB -c 64\n",
        "launcher": "",
        "mprun": "-N1 -c 64 -p milan-bxi -Xfirst",
    },
}

# Programs: script and working directory relative to the repository, how the
# options are read, "msub_only" (refused in the detached ccc_mprun mode), how the
# options are read ("argparse" or "help"), what the log says about progress,
# and what "Récupérer" / the Résumé tab take from the output directory the
# program announces in its log (out_re).
PROGRAMS = {
    "multi_launch_stress.py": {
        "script": "python/multi_launch_stress.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Outputs, CSV and log in (\S+)",
        "out_files": ["timings_summary.csv", "multi_launch_output.log", "error_*.txt"],
        "progress_re": r"Run (\d+)/(\d+) on build",
        "eta_re": r"Estimated Time Left: ([\d.]+)s",
        "summary": "multi_launch", "summary_file": "timings_summary.csv",
    },
    "bench_and_test.py": {
        "script": "python/bench_and_test.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Run outputs \S+ (\S+)",
        "out_files": ["bench_summary.csv", "bench_errors.log", "failed_cases.jsonl"],
        "progress_re": None, "eta_re": None,
        "summary": "csv", "summary_file": "bench_summary.csv",
    },
    "compile_all.sh": {
        "script": "scripts/compile_all.sh", "cwd": "scripts",
        "interp": "bash", "scan": "help",
        # Never on the login node: only as a batch job (ccc_msub).
        "msub_only": True,
        "out_re": None, "out_files": [],
        "progress_re": None, "eta_re": None,
        "summary": None, "summary_file": None,
    },
}

# Built-in defaults per (program, arch), by argparse dest. A string is the
# option's value (shell variables such as $USER are expanded on inti), True/
# False the state of a flag. "Enregistrer comme défauts" replaces them.
_BUILD_ROOT = "$CCCWORKDIR/shrink-array"

ARCH_DEFAULTS = {
    ("multi_launch_stress.py", "arm"): {
        "min_threads": "40", "max_threads": "200", "regen_ref": True, "all_exec": True,
        "build_root": _BUILD_ROOT, "builds": "build_arm.txt",
    },
    ("multi_launch_stress.py", "x86"): {
        "min_threads": "8", "max_threads": "64", "regen_ref": True, "all_exec": True,
        "build_root": _BUILD_ROOT, "builds": "build_x86_64.txt",
    },
    ("bench_and_test.py", "arm"): {"build_root": _BUILD_ROOT, "output_dir": _BUILD_ROOT + "/bench_results"},
    ("bench_and_test.py", "x86"): {"build_root": _BUILD_ROOT, "output_dir": _BUILD_ROOT + "/bench_results"},
}

DEFAULTS = {
    "host": "inti",
    "repo": "~/shrink-array",
    "runs_dir": "~/shrink_runs",
    "local_dir": os.path.join(os.path.expanduser("~"), "shrink_results"),
    "local_repo": "",
    "time_s": 18000,
    "preset": "gh200-bxi",
    "program": "multi_launch_stress.py",
    "mode": "msub",
}

# The newest python3 of the remote PATH, as a shell expression. A login node's
# system python3 can be old (3.6 on RHEL 8) while the scripts want more:
# bench_and_test.py needed 3.9 just to build its argument parser.
NEWEST_PY = ("$(command -v python3.13 || command -v python3.12 || command -v python3.11 || "
             "command -v python3.10 || command -v python3.9 || command -v python3)")

# The ssh binary; SHRINK_GUI_SSH overrides it (tests, or a wrapper).
SSH = os.environ.get("SHRINK_GUI_SSH", "ssh")

# Chart colours (data-viz reference palette). A scatter keeps three hues at
# most: they pass the all-pairs colour-vision checks in both modes, a fourth
# does not -- further categories fold into "autres", in muted ink.
VIZ = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}

# Colour follows the entity, never its rank: fixed slots per category.
COLOR_BY = {
    "Variante": (lambda exe: exe_parts(exe)[0], ["chunk", "chunk2", "chunk2a"]),
    "Taille": (lambda exe: exe_parts(exe)[1], ["256", "1024", "dyn"]),
    "Aucune": (lambda exe: "runs", ["runs"]),
}

EXE_RE = re.compile(r"shrink_bench_(chunk2a|chunk2|chunk)(?:_(\w+))?$")

def exe_parts(exe):
    """shrink_bench_chunk2a_dyn -> ("chunk2a", "dyn"); no size -> "default"."""
    m = EXE_RE.search(exe)
    return (m.group(1), m.group(2) or "default") if m else (exe, "?")

