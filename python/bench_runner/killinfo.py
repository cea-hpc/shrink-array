# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Why a run died: the signal that killed it, and whether it was the OOM killer.

runs.py reads oom_counts() before a run and, when the run failed,
explain(returncode, oom_before, oom_counts()) for the first line of its error
(runs._why_failed). A SIGKILL while an OOM-kill counter rose is told apart from
any other SIGKILL (a time limit, the batch system, a user).

Reconstructed on 2026-10-14 from those call sites: 0b621539 imported this
module but did not add it. Best effort throughout: an unreadable counter is
None, and explain() never raises.
"""
import os
import signal

# Signals that leave a core-sized hint worth giving.
_CRASH = {"SIGSEGV": "invalid memory access", "SIGBUS": "bus error", "SIGABRT": "abort() (an assertion, "
          "an invariant failure or std::terminate)", "SIGFPE": "arithmetic exception", "SIGILL": "illegal instruction"}


def _counter(path, key):
    """The value of `key <n>` in @p path (/proc/vmstat, memory.events), or None."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def _cgroup_events():
    """This process's cgroup v2 memory.events (a Slurm job's own cgroup), or None."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as f:
            for line in f:
                hier, _, rest = line.rstrip("\n").partition(":")
                ctrl, _, path = rest.partition(":")
                if hier == "0" and ctrl == "":
                    events = os.path.join("/sys/fs/cgroup", path.lstrip("/"), "memory.events")
                    return events if os.path.exists(events) else None
    except OSError:
        pass
    return None


def oom_counts():
    """The OOM-kill counters now: (system-wide oom_kill from /proc/vmstat,
    this cgroup's oom_kill from memory.events); None where unreadable."""
    events = _cgroup_events()
    return (_counter("/proc/vmstat", "oom_kill"), _counter(events, "oom_kill") if events else None)


def _rose(before, after):
    """Whether any counter readable at both ends went up."""
    return any(b is not None and a is not None and a > b for b, a in zip(before or (), after or ()))


def explain(returncode, oom_before, oom_after):
    """One line on why a run with @p returncode died, or "" when no signal killed it.

    @p returncode is a subprocess return code (-N: killed by signal N);
    @p oom_before / @p oom_after are oom_counts() around the run."""
    try:
        if returncode is None or returncode >= 0:
            return ""
        signum = -returncode
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        if signum == signal.SIGKILL:
            if _rose(oom_before, oom_after):
                return ("killed by SIGKILL from the OOM killer (oom_kill rose during the run): "
                        "it ran out of memory")
            return "killed by SIGKILL, no OOM kill counted (a time limit, the batch system or a user)"
        hint = _CRASH.get(name)
        return f"killed by {name}" + (f": {hint}" if hint else "")
    except Exception:  # noqa: BLE001 -- a diagnostic must not hide the failure it describes
        return ""
