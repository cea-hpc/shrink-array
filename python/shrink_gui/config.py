# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Partitions, programs, defaults, chart palette. What depends on the site
(host, paths, batch commands, partitions, QoS, option defaults) comes from its
TOML file (sitecfg.py, sites/); what depends on the repository is here."""
import os
import re

from shrink_gui import i18n, sitecfg
from shrink_gui.remote_helper import normalize_variant

APP = "shrink_gui"

# The "host" of this machine: the helper runs here, without ssh (LOCAL_SHELL).
LOCAL = "local"

# The site (sitecfg.load: $SHRINK_GUI_SITE, ~/.config/shrink_gui/site.toml,
# else sites/cea_inti.toml). A file that cannot be used raises SiteError here:
# app.main shows it.
SITE = sitecfg.load()
i18n.set_site_words(SITE.words())

# Partitions: the site's, as the window uses them -- "msub": the directive
# lines of the batch script (name, time, output, QoS... are added from the
# form), "launcher": how the batch job starts the command on its node,
# "mprun": the options of the detached mode. All three are editable in the
# window and remembered per partition.
PRESETS = {key: {"label": p["label"], "arch": p["arch"], "msub": p["directives"], "launcher": p["launcher"],
                 "mprun": p["detached"]}
           for key, p in SITE.partitions.items()}
PRESETS.update({
    # This machine ("cette machine" beside the host field): no scheduler, the
    # run goes in the background here (launch.sh, mode "local"). Its own
    # defaults ("arch" local), no #MSUB lines; "launcher" prefixes run.sh, for
    # a memory cap: systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0.
    LOCAL: {
        "label": "cette machine",
        "arch": LOCAL,
        "local": True,
        "msub": "",
        "launcher": "",
        "mprun": "",
    },
})

# Programs: script and working directory relative to the repository, how the
# options are read, "builds_opt" (the options naming the builds the run uses,
# for the up-to-date check), "time_s" (default job duration), "msub_only" (refused in the detached ccc_mprun mode), how the
# options are read ("argparse" or "help"), what the log says about progress,
# and what "Récupérer" / the Résumé tab take from the output directory the
# program announces in its log (out_re).
PROGRAMS = {
    "multi_launch_stress.py": {
        "time_s": 172800,  # default -T (long runs: 2 days); then remembered per program
        "script": "python/multi_launch_stress.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Outputs, CSV and log in (\S+)",
        "out_files": ["timings_summary.csv", "multi_launch_output.log", "error_*.txt"],
        "progress_re": r"Run (\d+)/(\d+) on build",
        "eta_re": r"Estimated Time Left: ([\d.]+)s",
        "summary": "multi_launch", "summary_file": "timings_summary.csv",
        # The Erreurs tab: these files of the output directory (names or glob
        # patterns), newest first. One error_*.txt per failed run.
        "errors_files": ["error_*.txt"],
        # The build check before the command ("Builds à jour"): the option
        # naming the build root, the one naming the builds (a list, or a file
        # of names one per line, relative to cwd).
        "builds_opt": {"root": "build_root", "builds": "builds"},
        # What "Supprimer" may also remove ("ses sorties"): "dir" = the output
        # directory the log announces, the run's own; a list = those entries
        # of a SHARED output directory (never the rest of it).
        "own_outputs": "dir",
    },
    "bench_and_test.py": {
        "time_s": 21600,  # default -T (a few hours); then remembered per program
        # -T from --mode (MODE_PRESETS of bench_runner/settings.py: about 1 h /
        # 6 h / 24 h on the aarch64 matrix) + 25%, times "arch_factor" on the
        # partition's architecture. Set on a change of mode or partition.
        "time_by_mode": {
            "dest": "mode", "default": "normal",
            "seconds": {"fast": 4500, "normal": 27000, "full": 108000},
            "arch_factor": {"x86": 2.5},
        },
        "script": "python/bench_and_test.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        "out_re": r"Run outputs \S+ (\S+)",
        "out_files": ["bench_summary.csv", "bench_errors.log", "failed_cases.jsonl"],
        "progress_re": None, "eta_re": None,
        "summary": "csv", "summary_file": "bench_summary.csv",
        # The Graphique tab: bench_summary.csv per config, variant and threads.
        "chart": "bench",
        # The Erreurs tab: one entry per failure, written as it happens (the
        # full output of each run is in run_logs/).
        "errors_files": ["bench_errors.log"],
        # Its failures one by one (failed_cases.jsonl: the complete output of
        # each, under run_logs/), and "Relancer les échecs" (--rerun-failed).
        "failed_cases": "failed_cases.jsonl", "full_logs": "run_logs",
        "rerun": {"opt": "--rerun-failed", "repeat": "--rerun-repeat", "log_dir": "--full-log-dir",
                  "drop": ["--no-full-log"]},
        "builds_opt": {"root": "build_root", "builds": "configs"},
        # bench_results/<arch> is shared by every run of the architecture and
        # holds the oracles (oracle/): only what each run rewrites goes.
        "own_outputs": ["bench_output.log", "bench_errors.log", "failed_cases.jsonl", "bench_summary.csv",
                        "run_logs", "runs"],
    },
    "omp_speedup.py": {
        "time_s": 14400,  # default -T (4 h); then remembered per program
        "script": "python/omp_speedup.py", "cwd": "python",
        "interp": "python3", "scan": "argparse",
        # On a compute node only (it measures it): batch, or this machine.
        "msub_only": True,
        # "Choisir…" beside these fields, as compile_all.sh's.
        "pickers": {"target": "configs", "execs": "execs"},
        # Its own output directory per run (under <work>/omp_speedup/).
        "out_re": r"Outputs in (\S+)",
        "out_files": ["summary.csv", "results.csv", "errors.log", "plot.gp", "speedup.png", "meta.json"],
        # Progress, Erreurs column, first-error notice: its events.jsonl.
        "progress_re": None, "eta_re": None,
        "summary": "csv", "summary_file": "summary.csv",
        "chart": "speedup",
        "errors_files": ["errors.log"],
        "builds_opt": {"root": "build_root", "builds": "target"},
        "own_outputs": "dir",
        # Checked before submitting (its own parse_args test, which the
        # option scan cannot see): at least one of these options, else the
        # job would only print a usage error. "with": options that do nothing
        # without that group's one_with (here --size / --int64 need --all-exec).
        "one_of": [{"dests": ["execs", "all_exec"], "one_with": "all_exec",
                    "with": ["size", "int64"]}],
    },
    "compile_all.sh": {
        "time_s": 5400,  # default -T (short); then remembered per program
        "script": "scripts/compile_all.sh", "cwd": "scripts",
        "interp": "bash", "scan": "help",
        # Never on the login node: as a batch job (ccc_msub), or on this machine.
        "msub_only": True,
        # "Choisir…" beside these fields: the configurations / executables
        # compile_all.sh can build, read on inti (see BUILD_REP).
        "pickers": {"target": "configs", "exec": "execs"},
        "out_re": None, "out_files": [],
        "progress_re": None, "eta_re": None,
        "summary": None, "summary_file": None,
        "errors_files": [],
        # Erreurs tab: the log of each failed build, named by its line
        # "[FAIL] <build> <time> rc=<n> → <log>" (functions_for_compile.sh).
        "fail_re": r"^\[FAIL\] .*→ (\S+)\s*$",
    },
}

# Built-in defaults per (program, arch), by argparse dest. A string is the
# option's value (shell variables such as $USER are expanded on inti), True/
# False the state of a flag. "Enregistrer comme défauts" replaces them.
ARCH_DEFAULTS = dict(SITE.defaults)
ARCH_DEFAULTS.update({
    # This machine: the builds are where scripts/site_env.sh puts them (the
    # repository itself on a workstation), run.sh exports it as CHUNK_BUILD_ROOT.
    ("multi_launch_stress.py", LOCAL): {
        "min_threads": "2", "max_threads": str(os.cpu_count() or 8), "regen_ref": True, "all_exec": True,
    },
})

# Where compile_all.sh puts its cmake-build-<name>/<arch> trees on inti
# (CHUNK_BUILD_REP, default of scripts/site_env.sh there): the "Choisir…" of
# --exec lists the executables of the build dirs found there.
BUILD_REP = SITE.build_root

DEFAULTS = {
    "host": SITE.host,
    "repo": SITE.repo,
    "runs_dir": SITE.runs_dir,
    "local_dir": os.path.join(os.path.expanduser("~"), "shrink_results"),
    "local_repo": "",
    "time_s": 18000,
    "preset": SITE.partition,
    "program": "multi_launch_stress.py",
    "mode": "msub",         # the batch mode ("mprun": the detached one, "local")
    "machine": "",          # LOCAL: this machine; else the host field
}

# This machine's own repository and runs directory (kept apart from the
# cluster's, under the settings keys local/repo and local/runs_dir).
LOCAL_DEFAULTS = {"repo": "~/shrink-array", "runs_dir": "~/shrink_runs"}

# The newest python3 of the remote PATH, as a shell expression. A login node's
# system python3 can be old (3.6 on RHEL 8) while the scripts want more:
# bench_and_test.py needed 3.9 just to build its argument parser.
NEWEST_PY = ("$(command -v python3.13 || command -v python3.12 || command -v python3.11 || "
             "command -v python3.10 || command -v python3.9 || command -v python3)")

# The ssh binary; SHRINK_GUI_SSH overrides it (tests, or a wrapper).
SSH = os.environ.get("SHRINK_GUI_SSH", "ssh")

# How the helper runs on this machine: the command line ssh would give the
# remote shell, run by bash here -- in WSL when the window runs on Windows.
LOCAL_SHELL = ["wsl.exe", "-e", "bash", "-c"] if os.name == "nt" else ["bash", "-c"]

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
    "Variante": (lambda exe: exe_parts(exe)[0], ["zero", "zero_one_value", "zero_any"]),
    "Taille": (lambda exe: exe_parts(exe)[1], ["256", "1024", "dyn"]),
    "Aucune": (lambda exe: "runs", ["runs"]),
}

# The old names too (shrink_bench_chunk2a_dyn: a build made before the rename),
# read as the new ones. Longest first: zero would also match zero_any.
EXE_RE = re.compile(r"shrink_bench_(zero_any|zero_one_value|zero|chunk2a|chunk2|chunk)(?:_(\w+))?$")

def exe_parts(exe):
    """shrink_bench_zero_any_dyn -> ("zero_any", "dyn"), shrink_bench_chunk2a_dyn
    too; no size -> "default"."""
    m = EXE_RE.search(exe)
    return (normalize_variant(m.group(1)), m.group(2) or "default") if m else (exe, "?")

# The batch commands and the detached mode of the site, as the helper takes
# them (remote_helper.py: act_check, act_launch, act_stop).
BATCH = SITE.batch
DETACHED = SITE.detached
QOS_FALLBACK = SITE.qos_fallback
