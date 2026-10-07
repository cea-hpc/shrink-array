"""shrink_gui -- prototype: submit and follow the shrink-array runs on inti
from the laptop.

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

"Récupérer" fetches the run directory, plus the summary files of the
program's own output directory, read from the log ("📁 Outputs, CSV and log
in ..." / "📁 Run outputs → ...") -- never the per-run output_* files.

Requires PySide6 on the laptop (pip install PySide6). Python >= 3.6 on inti.
Each run also records the git commit it was submitted on and the one it
started on (Commit column); bench_and_test.py writes its structured progress
to the run directory (events.jsonl, see bench_runner/events.py), shown as
the phase and k/N in the Progress column and as a Phases table.

Run (on the laptop, from python/):   python3 -m shrink_gui
                    or from the root: python3 python/shrink_gui
Tests (offscreen, fake ssh/ccc_msub): python3 -m pytest shrink_gui/tests -q

Modules:
  config.py         partitions, programs, per-architecture defaults -- edit here
  remote_helper.py  what runs on inti (sent on ssh's stdin; Python 3.6)
  runscripts.py     run.sh / launch.sh of a run
  remote.py         the ssh calls (QProcess)
  i18n.py           French / English
  widgets.py        log, options form, chart, cells
  app.py            the window
"""

__version__ = "0.11"
