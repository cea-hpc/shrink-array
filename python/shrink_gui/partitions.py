# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The partitions the machine lists (ccc_mpinfo, sinfo), offered beside the
site file's [partitions].

The connection check (remote_helper.py "check") runs the site's [batch]
partition_info; parse() reads what it prints, and presets() turns each
partition the site file does not describe into an entry of the partition
list, from the site's [discovered] templates. No Qt here.

That command prints a table: a header line naming the columns, usually a
line of dashes under it, then one row per partition. The columns read:

    PARTITION (or PartitionName)       the partition; a trailing "*" (sinfo's
                                       default partition) is dropped
    STATUS / AVAIL / STATE             up, down, drain... (blank: up)
    CpN (ccc_mpinfo) / CPUS (sinfo %c) the cores of one node
    GpN                                the GPUs of one node (0 or blank: none)
    MpC                                the memory per core, in MB

Other columns are ignored: ccc_mpinfo's grouped TOTAL / DOWN / USED / FREE
(cores, then nodes), SpN, CpS, TpC, ... As in qos.py, a row is cut at the
column extents of the dash line -- or, when its runs do not match the header
names one for one, of the header -- not at whitespace. A partition listed
twice (sinfo prints one row per node type) keeps its first row.

From the repository's python/ directory, on the cluster, this checks what the
window will read:

    ccc_mpinfo | python3 -m shrink_gui.partitions
"""
import re
import sys

from shrink_gui.qos import _spans

# Header name (lower case, letters only) -> row key.
_COLS = {"partition": "name", "partitionname": "name",
         "status": "status", "avail": "status", "state": "status",
         "cpn": "cores", "cpus": "cores", "corespernode": "cores", "cpuspernode": "cores",
         "gpn": "gpus", "gpuspernode": "gpus",
         "mpc": "mem_per_core"}


def _key(header):
    return _COLS.get(re.sub(r"[^a-z]", "", header.lower()))


def _lead_int(text):
    """The integer that @p text starts with ("128", "64+", "4(S:0-1)"), or None."""
    m = re.match(r"\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def parse(text):
    """The partition table -> rows {name, status, cores, gpus, mem_per_core}
    (cores, gpus: per node; mem_per_core: MB, or None), in the order printed.
    [] when nothing usable is found (empty, unknown format): the caller keeps
    what it had."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    hdr = next((i for i, ln in enumerate(lines)
                if any(_key(t) == "name" for t in ln.split()) and any(_key(t) == "cores" for t in ln.split())),
               None)
    if hdr is None:
        return []
    body = lines[hdr + 1:]
    dashes = body[0] if body and re.fullmatch(r"[-\s+|]+", body[0]) and "-" in body[0] else None
    if dashes is not None:
        body = body[1:]
    names = lines[hdr].split()
    spans = _spans(dashes) if dashes is not None and len(_spans(dashes)) == len(names) else _spans(lines[hdr])
    rows, seen = [], set()
    for ln in body:
        if not ln.strip() or re.fullmatch(r"[-\s+|=]+", ln):
            continue
        # Each value goes to the column it overlaps most (nearest if none),
        # the first value of a column only: a "GPU Type" header spans two
        # names, a blank cell leaves no value at all.
        cells = {}
        for a, b in _spans(ln):
            over = [max(0, min(b, e) - max(a, s)) for s, e in spans]
            if max(over) > 0:
                i = over.index(max(over))
            else:
                i = min(range(len(spans)), key=lambda j: abs((spans[j][0] + spans[j][1]) - (a + b)))
            k = _key(names[i])
            if k and k not in cells:
                cells[k] = ln[a:b]
        name = (cells.get("name") or "").rstrip("*")
        cores = _lead_int(cells.get("cores"))
        if not name or not cores or name in seen:
            continue
        seen.add(name)
        rows.append({"name": name, "status": (cells.get("status") or "up").lower(), "cores": cores,
                     "gpus": _lead_int(cells.get("gpus")) or 0, "mem_per_core": _lead_int(cells.get("mem_per_core"))})
    return rows


def presets(rows, discovered, known):
    """The partition-list entries of @p rows that are not in @p known (the
    site file's keys), from the site's [discovered] table @p discovered
    (directives, launcher, detached: templates with {name} and {cores};
    arm, skip: patterns on the name): {key: preset}, in the order of @p rows,
    each preset as config.PRESETS holds them, marked "discovered"."""
    arm = re.compile(discovered["arm"]) if discovered.get("arm") else None
    skip = re.compile(discovered["skip"]) if discovered.get("skip") else None
    out = {}
    for r in rows:
        name = r["name"]
        if name in known or name == "local" or (skip and skip.search(name)):
            continue
        is_arm = bool(arm and arm.search(name))
        bits = ["ARM" if is_arm else "x86", f"{r['cores']} cœurs"]
        if r["gpus"]:
            bits.append(f"{r['gpus']} GPU")
        if r["status"] not in ("up", "open"):
            bits.append(r["status"])
        fields = {"name": name, "cores": r["cores"]}
        out[name] = {"label": f"{name} — " + ", ".join(bits), "arch": "arm" if is_arm else "x86",
                     "msub": discovered["directives"].format(**fields),
                     "launcher": discovered["launcher"].format(**fields),
                     "mprun": discovered["detached"].format(**fields), "discovered": True}
    return out


if __name__ == "__main__":
    found = parse(sys.stdin.read())
    if not found:
        print("no partition table found (a header naming PARTITION and CpN, or CPUS)")
        sys.exit(1)
    for row in found:
        print("{name:<20} {status:<6} {cores:>4} cores/node  {gpus} GPU  {mpc}".format(
            mpc=f"{row['mem_per_core']} MB/core" if row["mem_per_core"] else "", **row))
