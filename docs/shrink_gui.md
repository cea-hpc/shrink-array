# shrink_gui: submit and follow the runs on the cluster

> French version: [shrink_gui_fr.md](shrink_gui_fr.md).

> October 2026, `shrink_gui` 0.12 (prototype). A window on the laptop to submit
> `compile_all.sh`, `bench_and_test.py` and `multi_launch_stress.py` on the cluster, or to
> run them on the laptop itself (§3, "This machine"), follow
> their log, read their summary and fetch their results. The code is described in the
> docstring of `python/shrink_gui/__init__.py`.

***

## 1. How it works

Everything goes through `ssh <cluster>`, so through the master connection (ControlMaster) you
open yourself: the interface never handles a password. Nothing is installed on the cluster.
Each call sends a small Python program (`remote_helper.py`, Python 3.6 compatible) on ssh's
standard input, run by the login node's `python3`.

A **run** is a directory on the cluster, `<runs directory>/<id>/`:

| file | contents |
|---|---|
| `meta.json` | what was asked: program, partition, command, mode, commit |
| `run.sh` | what runs on the compute node: repository, `site_env.sh`, command |
| `job.sh` | the batch script given to `ccc_msub` (`#MSUB` directives) |
| `launch.sh` | detached mode: `ccc_mprun <options> bash run.sh` |
| `out.log` | everything the job printed |
| `jobid` | (`ccc_msub`) the batch id, for `squeue` and `ccc_mdel` |
| `exit_code` | written when the command returns |
| `events.jsonl` | (`bench_and_test.py`) structured progress: phases, k/N |

## 2. Requirements and start

- On the laptop: Python 3.9 or later and **PySide6** (`pip install PySide6`).
- On the cluster: Python 3.6 or later, the shrink-array repository cloned, and `ccc_msub`
  for the batch mode.
- An ssh master connection, opened once per session:

```sh
ssh -fNM inti          # the host name of ~/.ssh/config, with ControlMaster/ControlPath
```

Then, on the laptop:

```sh
cd shrink-array/python && python3 -m shrink_gui     # or, from the root: python3 python/shrink_gui
```

At start the interface checks the connection. The top line shows the node, its Python
version, and the commit of the cluster's repository, compared with your local clone's if you
gave one ("Local repository"). Without a master connection it shows the `ssh -fNM` command to
run.

## 3. The window

### Where and how

| field | role |
|---|---|
| Partition | `gh200-bxi` (ARM, 288 cores) or `milan-bxi` (x86, 64 cores); also sets the architecture of the defaults |
| Submission | `ccc_msub` (batch, the default) or `ccc_mprun` detached from the login node |
| Max time (-T) | takes `2 d`, `1 h 30`, `90 min`, `1-00:00:00` |
| QoS (-Q) | "auto" picks the QoS from the duration, by the limits `ccc_mqinfo` reports |
| Start | after a `compile_all.sh` run not finished yet (Slurm dependency `afterok`/`afterany`), and/or not before a given time |
| Builds up to date | before the command, `ninja -n` on the builds used: refuse (code 96), warn, or skip |
| Directives, Launcher (job) | the `#MSUB` lines and the launcher on the node; per partition, remembered |
| Repository (inti), Runs (inti) | the shrink-array repository and the runs directory, on the cluster |
| Git (inti) | branch, commit, local changes, ahead of or behind the upstream; `git pull` and `git pull --rebase` buttons (§5) |
| Results (local), Local repository | where "Fetch" copies on the laptop; your local clone, to compare the commits |

### What

The program to run, and its options **read from the script itself** on the cluster ("Read
options"): the Python scripts' `argparse`, `compile_all.sh`'s `usage()`. An option added to a
script therefore appears in the form without touching the interface. Changed options are in
bold; an option left at its default is not put on the command line. "Save as defaults"
replaces this program's defaults on this architecture. "Command" shows exactly what will run.

`compile_all.sh` only runs as a batch job, never on the login node. Its `--target` and
`--exec` fields have a "Choose…" button listing the configurations and executables found on
the cluster.

### The runs

The table lists the runs of the runs directory, refreshed every 15 s: program, partition,
commit, job, status, progress (k/N and time left when the program reports them), last line of
the log. A notification announces a run that ends.

| tab | contents |
|---|---|
| Log | the selected run's log, followed live; search with Ctrl+F |
| Summary | `multi_launch_stress.py`: per build (runs, failures, times); `bench_and_test.py`: its phases (duration, errors) or its `bench_summary.csv` |
| Chart | `multi_launch_stress.py`: each run's time against the thread count, filtered by build and variant; "Compare with" overlays another run |

| button | effect |
|---|---|
| Stop | `ccc_mdel` of the job, or stops `ccc_mprun` in detached mode |
| Fetch results | copies to "Results (local)" the run directory and the program's summary files (CSV, log, errors; never the `output_*`) |
| Reuse its options | puts the run's program, partition and options back in the form |
| Delete | deletes the run directory (finished runs only); optionally also the outputs that are this run's own, and the local copy |

### This machine

The "this machine" box beside the host field sends the runs to the machine the window runs
on instead of the cluster: no ssh and no job scheduler. The helper runs through `bash -c`
here (from Windows, through `wsl.exe`: in WSL), and the run goes in the background. It does
not depend on the window: closing it leaves the run going, as a batch job would.

- **Partition**: a single one, "this machine", with the architecture, core count and memory
  the connection check found. Its defaults are its own (`ARCH_DEFAULTS[(program, "local")]`).
  The QoS and the directives are the scheduler's: hidden.
- **Submission**: "in the background on this machine". `launch.sh` runs `run.sh`, under
  `timeout` when a maximum duration is set (exit code 124 past it).
- **Launcher**: a prefix of `run.sh`, remembered. To cap the memory of a run, and keep an
  out-of-memory run from bringing WSL down:
  `systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0`.
- **Repository, Runs**: kept apart from the cluster's (by default `~/shrink-array` and
  `~/shrink_runs` of this machine); switching back and forth keeps both.
- **Start, Builds up to date, Stop, Fetch, Delete**: as on the cluster. "Start: after" waits
  for a `compile_all.sh` run of this machine; "Stop" ends the run's process group.
- `compile_all.sh` runs here too: it is refused only from the cluster's login node.

## 4. Examples

**Build, then test once the build succeeded.** Program `compile_all.sh`, mode `ccc_msub`,
"Submit". Then program `bench_and_test.py`, "Start: after" that build run, "if it succeeds",
"Submit". The second job waits for the first in the Slurm queue; if the first fails, the
second stays pending and the interface says so.

**A stress campaign on x86.** Partition `milan-bxi`, program `multi_launch_stress.py`,
"Architecture defaults" (8 to 64 threads, the builds of `build_x86_64.txt`), "Submit". The
Chart tab fills in as the runs come.

**Build and test on the laptop.** Tick "this machine", program `compile_all.sh` with its
`--target`, "Submit"; then `bench_and_test.py`, "Start: after" that run, "Submit". With a
launcher `systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0`, a run that
needs more memory is killed alone.

The equivalent command lines, without the interface, are in
[example_bench_test.txt](example_bench_test.txt) and
[example_multi_launch_stress.txt](example_multi_launch_stress.txt).

## 5. Updating the cluster's repository

The "Git (inti)" row is filled at every connection check. Its ahead/behind counts date from
the last `fetch`; a `git pull` brings them up to date.

- `git pull` and `git pull --rebase` run with no prompt (a password request would hang the
  ssh call), within a 5-minute limit.
- The interface asks first, and warns when runs are still active: a build or a bench in
  progress could read sources changed under it.
- A conflict **cancels** the operation (`git merge --abort` or `git rebase --abort`): the
  repository stays as it was, and a window shows git's output.
- When the commit changed, the connection is checked again and the script options read again.

## 6. Adapting it to a site

Everything site-specific is in `python/shrink_gui/config.py`:

| name | contents |
|---|---|
| `PRESETS` | the partitions: `#MSUB` directives, launcher, `ccc_mprun` options; `local` is this machine |
| `PROGRAMS` | the programs: script, working directory, how options are read, progress and output patterns |
| `ARCH_DEFAULTS` | the defaults per (program, architecture) |
| `DEFAULTS` | host, repository, runs directory, local directory, partition and program at start |
| `LOCAL_DEFAULTS` | this machine's repository and runs directory |
| `LOCAL_SHELL` | how the helper runs on this machine (`bash -c`, or `wsl.exe -e bash -c` on Windows) |
| `SHRINK_GUI_SSH` (environment variable) | the ssh binary to use (default `ssh`) |

What you type in the window is remembered from one session to the next (QSettings). The
language is chosen at the top right (French or English).

## 7. Tests

```sh
cd python && python3 -m pytest shrink_gui/tests -q
```

The tests run without a display (`QT_QPA_PLATFORM=offscreen`) against a fake local cluster:
fake `ssh`, `ccc_msub`, `squeue`, `ccc_mdel`, `ccc_mprun` and `ccc_mqinfo` (`tests/fakes`).
They are skipped when PySide6 is not installed.

## 8. Limits

- A prototype, written for the TGCC machines and their `ccc_*` commands (`ccc_msub`,
  `ccc_mprun`, `ccc_mqinfo`). Another site adapts `config.py` and the submission commands
  of `runscripts.py`.
- One connection at a time. The interface does not open the master connection itself.
- This machine from Windows (`wsl.exe`) is not covered by the tests (they run on Linux).
  WSL may stop a distribution once no window of it is left open: keep one open while a
  run goes.
