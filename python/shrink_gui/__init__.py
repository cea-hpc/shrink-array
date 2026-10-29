# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""shrink_gui -- prototype: submit and follow the shrink-array runs on inti
from the laptop, or run them on the laptop itself ("cette machine").

The site -- host, paths, batch commands, partitions, QoS, option defaults --
is a TOML file (sitecfg.py): $SHRINK_GUI_SITE, else
~/.config/shrink_gui/site.toml, else sites/cea_inti.toml (CEA's inti, the
names used below). sites/slurm_example.toml is a plain Slurm one (sbatch,
#SBATCH, scancel): copy one to ~/.config/shrink_gui/site.toml and edit it.

Everything goes through `ssh <host>`, so through the ControlMaster you opened
yourself (`ssh -fNM inti`): this program never handles a password. Nothing is
installed on inti: each call sends a small Python helper (remote_helper.py)
on the ssh stdin and runs it with the login node's python3.

Programs (PROGRAMS): multi_launch_stress.py, bench_and_test.py and
scripts/compile_all.sh. Their options are READ FROM THE SCRIPTS THEMSELVES on
inti ("Lire les options"): a Python script is run with argparse's
parse_known_args() replaced by a dump of the parser; a shell script's --help is
parsed ("  --flag[=<v>]   description" lines). The form follows the script
when an option is added; the options you changed show in bold. Each
(program, architecture) has its own default values: built in (ARCH_DEFAULTS),
replaced by yours with "Enregistrer comme défauts".

Submission: ccc_msub (batch, the default) or ccc_mprun detached on the login
node (the first prototype's way). A run is a directory on inti
(<runs dir>/<id>/):
    meta.json   what was asked (program, partition, command, mode, ...)
    run.sh      what runs on the compute node: repo, site_env.sh, command
    job.sh      the batch script given to ccc_msub (#MSUB directives)
    launch.sh   (ccc_mprun mode) ccc_mprun <options> bash run.sh
    out.log     everything the job printed
    jobid       (ccc_msub) the batch id, for squeue / ccc_mdel
    exit_code   written when the command returns
    stopped     written by "Arrêter"

Start conditions ("Démarrage"): after a compile_all.sh run not finished yet
(ccc_msub: a Slurm dependency, afterok or afterany, passed with -E; detached
mode: launch.sh waits for that run's exit_code), and/or not before a delay
(--begin). "Nœuds" (compile_all.sh and thread_speedup.py in batch mode,
PROGRAMS[...]["pool_opt"]): N > 1 submits N one-node jobs <id>-n1..nN, the same
command plus --pool=<id> (meta.json: pool, node, nodes); compile_all.sh builds
each configuration in the first job that claims it (functions_for_compile.sh,
pool_init), thread_speedup.py measures each executable there (its Pool class),
and the Graphique / Résumé tabs of any job gather the pool's
(remote_helper.pool_members). "Démarrage" then offers a compile_all.sh
submission as one entry and waits for all its jobs (after["runs"];
afterok:j1:...:jN).
"Builds à jour": run.sh first runs `ninja -n` on the executables
already built in each build the program will use (--configs / --builds under
--build-root) and refuses to start (exit code 96) or warns when one is stale.

"Récupérer" fetches the run directory, plus the summary files of the
program's own output directory, read from the log ("📁 Outputs, CSV and log
in ..." / "📁 Run outputs → ...") -- never the per-run output_* files.

This machine ("cette machine" beside the host field): no ssh, the helper
runs through config.LOCAL_SHELL (bash -c; wsl.exe from Windows), and a run is
launch.sh in the background -- runscripts.make_local_launch_sh: run.sh under
`timeout` (the maximum duration), behind the launcher prefix (systemd-run
--scope -p MemoryMax=... caps its memory). One partition, PRESETS["local"],
with its own defaults (arch "local"); the repository and the runs directory
are kept apart (settings local/repo, local/runs_dir). Status and "Arrêter" go
by pid and process group, as the detached ccc_mprun mode does.

Requires PySide6 on the laptop (pip install PySide6). Python >= 3.6 on inti.
Each run also records the git commit it was submitted on and the one it
started on (Commit column); bench_and_test.py writes its structured progress
to the run directory (events.jsonl, see bench_runner/events.py), shown as
the phase and k/N in the Progress column and as a Phases table.

Run (on the laptop, from python/):   python3 -m shrink_gui
                    or from the root: python3 python/shrink_gui
Tests (offscreen, fake ssh/ccc_msub): python3 -m pytest shrink_gui/tests -q

Modules:
  sitecfg.py        the site file: reading, checks (sites/: the examples)
  config.py         programs, and the site's partitions and defaults as the window uses them
  remote_helper.py  what runs on inti (sent on ssh's stdin; Python 3.6)
  runscripts.py     run.sh / launch.sh of a run
  remote.py         the ssh calls (QProcess)
  i18n.py           French / English
  widgets.py        log, options form, chart, cells
  app.py            the window (Main): its construction, the runs table, the Log tab
  Mixins of Main, one file per part of the window:
    machine_panel.py    this machine or the cluster, connection, partition, QoS
    launch_form.py      program, options form, start conditions, pickers, submission
    run_actions.py      stop, reuse the options, delete, fetch the results
    git_panel.py        the Git (inti) rows
    tab_summary.py      the Résumé tab
    tab_chart.py        the Graphique tab (multi_launch_stress.py)
    tab_bench_chart.py  ... its bench_and_test.py page
    tab_speedup_chart.py ... its thread_speedup.py page
    tab_errors.py       the Erreurs tab, the replay of failures
    tooltips.py         the tooltip texts
"""

__version__ = "0.14"
