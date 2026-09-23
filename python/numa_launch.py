#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""numa_launch.py -- a `numactl --interleave=<nodes>` prefix for the runs
that span more than one NUMA node, shared by bench_and_test.py and
multi_launch_stress.py.

Why: on Grace (4 sockets of 72 cores, one CPU node each, plus 32 CPU-less
nodes -- the GPUs' memory), shrink_bench_chunk2_int64_1024 --multiply 4 at 144
threads (close, cores) ran 14.18 s by default and 13.20 s under
`numactl --interleave=0,1` (-6.9%, 3 interleaved rounds each). The process
memory was already split evenly between the two nodes (numastat: 34.6 / 33.5
GB); what interleaving evens out is WHICH pages each thread reaches.
`--interleave=all` was SLOWER (15.66 s): it spreads pages over the CPU-less GPU
nodes too. Hence: only the CPU nodes of the cores the run actually uses.

Which cores: the calling process's allowed CPUs (sched_getaffinity), one per
physical core (OMP_PLACES=cores), in id order; with OMP_PROC_BIND close /
master / primary / true, the first OMP_NUM_THREADS of them; with spread or no
binding, all of them (the threads may land anywhere in the set). No prefix
when that is a single node, when numactl is missing, or when the thread count
is unknown.
"""
import functools
import glob
import os
import shutil

MODES = ("auto", "off")


def _parse_list(s):
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


@functools.lru_cache(maxsize=None)
def cpu_to_node():
    """CPU id -> NUMA node, for the nodes that have CPUs."""
    m = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            node = int(d.rsplit("node", 1)[1])
            with open(os.path.join(d, "cpulist")) as f:
                cpus = _parse_list(f.read())
        except (OSError, ValueError):
            continue
        for c in cpus:
            m[c] = node
    return m


@functools.lru_cache(maxsize=None)
def allowed_cores():
    """Allowed CPUs, one per physical core (the lowest sibling), in id order."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return ()
    seen, cores = set(), []
    for c in cpus:
        base = f"/sys/devices/system/cpu/cpu{c}/topology"
        try:
            with open(os.path.join(base, "physical_package_id")) as f:
                pkg = f.read().strip()
            with open(os.path.join(base, "core_id")) as f:
                core = f.read().strip()
            key = (pkg, core)
        except OSError:
            key = ("cpu", c)
        if key in seen:
            continue
        seen.add(key)
        cores.append(c)
    return tuple(cores)


@functools.lru_cache(maxsize=None)
def _numactl():
    return shutil.which("numactl")


def interleave_nodes(env):
    """The CPU nodes a run with this environment uses (sorted), or []."""
    try:
        tc = int(str(env.get("OMP_NUM_THREADS", "")).split(",")[0])
    except ValueError:
        return []
    if tc <= 0:
        return []
    c2n = cpu_to_node()
    cores = allowed_cores()
    if not c2n or not cores:
        return []
    bind = str(env.get("OMP_PROC_BIND", "")).split(",")[0].strip().lower()
    used = cores[:tc] if bind in ("close", "master", "primary", "true") else cores
    return sorted({c2n[c] for c in used if c in c2n})


def interleave_prefix(env, mode="auto"):
    """`numactl --interleave=<nodes>` when the run spans 2+ CPU nodes, else []."""
    if mode == "off" or not _numactl():
        return []
    nodes = interleave_nodes(env)
    if len(nodes) < 2:
        return []
    return [_numactl(), "--interleave=" + ",".join(str(n) for n in nodes)]
