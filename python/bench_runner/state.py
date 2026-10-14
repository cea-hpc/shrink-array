# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Run-wide mutable state: set once by cli.main(), read by the run helpers."""
import os
import sys
import threading

from .settings import DEFAULT_STATS_MARK_RATIO_MIN


# --numa-interleave: numactl prefix for runs spanning several NUMA nodes.
# numa_launch.py sits in python/, next to this package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numa_launch as _numa  # noqa: E402
except ImportError:
    # Copied alone next to the binaries (cp python/<script> build_dir/): run
    # without NUMA interleave rather than not at all. Copy numa_launch.py too,
    # or run the script from python/, to get it back.
    import types as _types  # noqa: E402

    _numa = _types.SimpleNamespace(MODES=("auto", "off"), interleave_prefix=lambda env, mode="auto": [])
    print(f"[{os.path.basename(sys.argv[0])}] numa_launch.py not found next to the script: "
          "--numa-interleave has no effect", file=sys.stderr)
_NUMA_MODE = "auto"  # set from --numa-interleave in main()


_FULL_LOG_DIR = None
_FULL_LOG_LOCK = threading.Lock()
_FULL_LOG_COUNTER = 0

# The full log the last run of this thread wrote (_write_full_log), for
# _note_run to record beside a failed run: the window opens it from the
# failure, whatever the file's name says (a sanitizer hit exits 0).
_LAST_FULL_LOG = threading.local()

def _set_full_log_dir(path):
    global _FULL_LOG_DIR
    _FULL_LOG_DIR = path


# Runtime copy, overridden from --stats-mark-ratio-min / --no-stats-guard.
_STATS_MARK_RATIO_MIN = dict(DEFAULT_STATS_MARK_RATIO_MIN)


# ── --rerun-failed: recording the failed runs ────────────────────────────────
# Every FAILED run of the two executors is remembered here until the phase that
# made it logs its failure (_append_error_now), which moves it to
# failed_cases.jsonl. A later SUCCESSFUL run of the same command (the
# rebuild-and-retry path) forgets it, so a failure that healed is not replayed.
_PENDING_FAILED_RUNS = []
