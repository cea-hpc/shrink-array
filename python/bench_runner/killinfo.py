# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Why a run died, in words, for bench_errors.log.

A negative return code is a signal. SIGKILL (-9) is the hard one: the bench's
own timeout reports itself ("TIMEOUT after N s"), so a bare -9 came from
outside -- most often the out-of-memory killer. The memory cgroup the bench
runs in (Slurm's task/cgroup: the job's, or its step's) counts every OOM kill
(cgroup v1 memory.oom_control, v2 memory.events, "oom_kill N"): reading the
counters before and after a run tells an OOM kill from the rest.

Best effort, never raises: no cgroup information gives a vaguer sentence."""
import os
import signal

_NO_LIMIT = 1 << 60       # v1 writes "no limit" as a huge number

# The words explain() opens an OOM kill's sentence with: a caller that must
# act on an OOM kill (the scalability guard stops a curve there) tests for it.
OOM_KILL_MARK = "KILLED BY THE OUT-OF-MEMORY KILLER"

_HINTS = {
    signal.SIGSEGV: "invalid memory access: see its full output, or rerun it under gdb / ASan",
    signal.SIGABRT: "abort(): a failed assert, std::terminate, or a sanitizer report",
    signal.SIGBUS: "bus error: a misaligned access, or a mapped file / huge page gone",
    signal.SIGFPE: "arithmetic exception (integer division by zero?)",
    signal.SIGILL: "illegal instruction: a binary built for another CPU (-march) than this node's?",
    signal.SIGTERM: "stopped from outside: scancel / ccc_mdel, or the job's time limit",
}


def _memory_cgroup_dirs():
    """This process's memory cgroup directory and its ancestors, innermost
    first ([] when there is none to read)."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    v1 = v2 = None
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        if "memory" in parts[1].split(","):
            v1 = parts[2]
        elif parts[0] == "0" and parts[1] == "":
            v2 = parts[2]
    if v1 is not None:
        root, rel = "/sys/fs/cgroup/memory", v1
    elif v2 is not None:
        root = ("/sys/fs/cgroup" if os.path.exists("/sys/fs/cgroup/cgroup.controllers")
                else "/sys/fs/cgroup/unified")
        rel = v2
    else:
        return []
    rel = rel.strip("/")
    dirs = []
    while True:
        d = os.path.join(root, rel) if rel else root
        if os.path.isdir(d):
            dirs.append(d)
        if not rel:
            break
        rel = os.path.dirname(rel)
    return dirs


def _read_kv(path, key):
    try:
        with open(path) as f:
            for ln in f:
                k, _, v = ln.partition(" ")
                if k == key:
                    return int(v)
    except (OSError, ValueError):
        pass
    return None


def oom_counts():
    """{cgroup dir: oom_kill count} for this process's memory cgroups."""
    out = {}
    for d in _memory_cgroup_dirs():
        n = _read_kv(os.path.join(d, "memory.oom_control"), "oom_kill")
        if n is None:
            n = _read_kv(os.path.join(d, "memory.events"), "oom_kill")
        if n is not None:
            out[d] = n
    return out


def _limit(d):
    """The memory limit of cgroup @p d in bytes, or None (none / unreadable)."""
    for name in ("memory.limit_in_bytes", "memory.max"):
        try:
            with open(os.path.join(d, name)) as f:
                v = f.read().strip()
        except OSError:
            continue
        if v == "max":
            return None
        try:
            v = int(v)
        except ValueError:
            return None
        return None if v >= _NO_LIMIT else v
    return None


def _gib(n):
    return f"{n / 2**30:.1f} GiB"


def explain(returncode, before, after):
    """One or two lines saying why a run that ended with @p returncode died,
    @p before / @p after: oom_counts() around it. "" for a plain exit code."""
    if returncode is None or returncode >= 0:
        return ""
    sig = -returncode
    try:
        name = signal.Signals(sig).name
    except ValueError:
        name = f"signal {sig}"
    if sig != signal.SIGKILL:
        hint = _HINTS.get(sig)
        return f"KILLED BY {name} (signal {sig})" + (f": {hint}" if hint else "")
    hit = [d for d, n in after.items() if n > before.get(d, n)]
    if hit:
        lim = next((x for x in (_limit(d) for d in hit) if x), None)
        where = hit[0]
        return (f"{OOM_KILL_MARK} (SIGKILL): the memory cgroup counted an OOM kill "
                f"during this run{f' (limit {_gib(lim)})' if lim else ''} [{where}].\n"
                "Too much memory for the job: fewer threads or a smaller size, or more memory "
                "(more cores for the job, or #MSUB -M / --mem).")
    lims = [x for x in (_limit(d) for d in after) if x]
    known = (f"no OOM kill counted by its memory cgroup (limit {_gib(min(lims))})" if lims else
             "no OOM kill counted by its memory cgroup" if after else
             "no cgroup information to tell an OOM kill")
    return ("KILLED BY SIGKILL (signal 9) from outside the bench -- its own timeout says TIMEOUT: "
            f"{known}.\nLikely causes: the node's out-of-memory killer, the job's time limit, "
            "a scancel / ccc_mdel. Look for 'oom' in the job's log (Log tab).")
