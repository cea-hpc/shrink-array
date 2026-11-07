# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""QoS of the batch jobs (ccc_msub -Q, sbatch --qos): the limits the site's
QoS command prints (ccc_mqinfo, sacctmgr show qos), and the choice of a QoS
for a partition, a core count and a duration.

The limits are read on the cluster by the connection check (remote_helper.py
"check" runs the site's [batch] qos_info); the site's [qos] fallback stands in
until then (config.QOS_FALLBACK). No Qt here.

That command prints a table: a header line (Name, Partition, Priority, MaxCPUs,
MaxSub, MaxTime, ...), usually a line of dashes under it, then one row per
(QoS, partition). Columns may be blank (no limit), so a row is cut at the
column extents of the dash line -- or, without one, of the header -- not at
whitespace. A Partition of "*" or blank, or no Partition column at all, means
every partition. Times are Slurm's: [D-]HH:MM:SS, MM:SS, or minutes.
"""
import re

# Header name (lower case, no separators) -> row key.
_COLS = {"name": "name", "qos": "name", "partition": "partition", "priority": "priority",
         "maxcpus": "max_cpus", "maxcpusperuser": "max_cpus", "maxcores": "max_cpus",
         "maxsub": "max_sub", "maxsubmit": "max_sub", "maxsubmitjobs": "max_sub",
         "maxsubmitpu": "max_sub", "maxsubmitjobsperuser": "max_sub", "maxcpuspu": "max_cpus",
         "maxrun": "max_run", "maxjobs": "max_run", "maxjobspu": "max_run", "maxtime": "max_time", "maxwall": "max_time",
         "maxwalldurationperjob": "max_time",
         # The cores of all of a user's jobs together (ccc_mqinfo's SumCPUs,
         # sacctmgr's MaxTRESPerUser "cpu=N"): beyond, the jobs wait their turn.
         "sumcpus": "sum_cpus", "maxtrespu": "sum_cpus", "maxtresperuser": "sum_cpus"}

_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$")


def parse_time(text):
    """Slurm time -> seconds: "D-HH:MM:SS", "D-HH:MM", "D-HH", "HH:MM:SS",
    "MM:SS", "MM" (minutes). None if it is none of these."""
    m = _TIME_RE.match((text or "").strip())
    if not m:
        return None
    d, a, b, c = m.groups()
    days = int(d) if d else 0
    if d is not None:                       # D-HH[:MM[:SS]]
        h, mi, s = int(a), int(b or 0), int(c or 0)
    elif c is not None:                     # HH:MM:SS
        h, mi, s = int(a), int(b), int(c)
    elif b is not None:                     # MM:SS
        h, mi, s = 0, int(a), int(b)
    else:                                   # MM
        h, mi, s = 0, int(a), 0
    return ((days * 24 + h) * 60 + mi) * 60 + s


def fmt(seconds, lang="fr"):
    """Seconds -> "2 j 00 h", "1 h 30", "45 min", "30 s" ("d" in English):
    what widgets.parse_duration reads back."""
    v = int(seconds or 0)
    d, rest = divmod(v, 86400)
    h, rest = divmod(rest, 3600)
    m, s = divmod(rest, 60)
    tail = f" {s} s" if s else ""
    if d:
        return f"{d} {'d' if lang == 'en' else 'j'} {h:02d} h" + (f" {m:02d}" if m else "") + tail
    if h:
        return f"{h} h {m:02d}" + tail
    if m:
        return f"{m} min" + tail
    return f"{s} s"


def _key(header):
    return _COLS.get(re.sub(r"[^a-z]", "", header.lower()))


def _spans(line):
    """(start, end) of each run of non-blank characters of @p line."""
    return [m.span() for m in re.finditer(r"\S+", line)]


def _int(text):
    text = text.strip()
    m = re.search(r"(?:^|,)cpu=(\d+)", text)      # a TRES list: its cpu count
    if m:
        return int(m.group(1))
    return int(text) if text.isdigit() else None


def parse_mqinfo(text):
    """ccc_mqinfo's output -> rows {name, partition ("*" = all), priority,
    max_cpus, sum_cpus (all the user's jobs), max_sub, max_run (None = no
    limit), max_time (seconds)}.
    [] when nothing usable is found (empty, unknown format): the caller keeps
    what it had."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    hdr = next((i for i, ln in enumerate(lines)
                if any(_key(t) == "name" for t in ln.split()) and any(_key(t) == "max_time" for t in ln.split())),
               None)
    if hdr is None:
        return []
    body = lines[hdr + 1:]
    dashes = body[0] if body and re.fullmatch(r"[-\s+|]+", body[0]) and "-" in body[0] else None
    if dashes is not None:
        body = body[1:]
    names = lines[hdr].split()
    spans = _spans(dashes) if dashes is not None and len(_spans(dashes)) == len(names) else _spans(lines[hdr])
    rows = []
    for ln in body:
        if not ln.strip() or re.fullmatch(r"[-\s+|=]+", ln):
            continue
        # Each value goes to the column it overlaps most (nearest if none):
        # values are left- or right-aligned under their header, and blank
        # cells leave no token at all.
        cells = {}
        for a, b in _spans(ln):
            over = [max(0, min(b, e) - max(a, s)) for s, e in spans]
            if max(over) > 0:
                i = over.index(max(over))
            else:
                i = min(range(len(spans)), key=lambda j: abs((spans[j][0] + spans[j][1]) - (a + b)))
            k = _key(names[i])
            if k:
                cells[k] = (cells.get(k, "") + " " + ln[a:b]).strip()
        t = parse_time(cells.get("max_time", ""))
        if not cells.get("name") or t is None:
            continue
        part = cells.get("partition") or "*"
        rows.append({"name": cells["name"], "partition": part,
                     "priority": _int(cells.get("priority", "")) or 0,
                     "max_cpus": _int(cells.get("max_cpus", "")),
                     "sum_cpus": _int(cells.get("sum_cpus", "")),
                     "max_sub": _int(cells.get("max_sub", "")),
                     "max_run": _int(cells.get("max_run", "")),
                     "max_time": t})
    return rows


def names(rows):
    """The QoS names, in the order ccc_mqinfo lists them."""
    seen = []
    for r in rows:
        if r["name"] not in seen:
            seen.append(r["name"])
    return seen


def limits(rows, name, partition):
    """The row of QoS @p name on @p partition (its own row first, then a
    "*" one), or None: the QoS is not open there. A MaxSub of 0 (not blank)
    closes it: ccc_mqinfo's "long  *  ... 0" row means long is open only on
    the partitions it lists, and ccc_msub rejects it elsewhere ("Invalid qos
    specification")."""
    own = [r for r in rows if r["name"] == name and r["partition"] == partition]
    anyp = [r for r in rows if r["name"] == name and r["partition"] in ("*", "")]
    row = (own or anyp or [None])[0]
    return None if row is None or row.get("max_sub") == 0 else row


# A directive prefix at the start of a line (#MSUB, #SBATCH, #PBS...).
_DIRECTIVE_RE = re.compile(r"^#[A-Z]+", re.M)
_QOS_RE = re.compile(r"(?:^|\s)(?:-Q\s*|--qos[=\s]+)(\S+)")


def explicit_qos(directives):
    """The QoS the directives set themselves ("#MSUB -Q long", "-Q long",
    "--qos=long"), or None."""
    for ln in (directives or "").splitlines():
        m = _QOS_RE.search(_DIRECTIVE_RE.sub(" ", ln))
        if m:
            return m.group(1)
    return None


_LONG = {"n": "ntasks", "c": "cpus-per-task"}


def _opt(directives, flag):
    """The integer value of -n / -c (--ntasks / --cpus-per-task) in the
    directives (#MSUB or #SBATCH lines, or detached-mode options), or None."""
    m = re.findall(rf"(?:^|\s)(?:-{flag}\s*|--{_LONG[flag]}[=\s]+)(\d+)\b",
                   _DIRECTIVE_RE.sub(" ", directives or ""))
    return int(m[-1]) if m else None


def job_cpus(directives):
    """Cores the job asks for: -n (tasks) x -c (cores per task), 1 each by default."""
    return (_opt(directives, "n") or 1) * (_opt(directives, "c") or 1)


def serializes(row, cpus, jobs):
    """True when @p jobs jobs of @p cpus cores each, under QoS @p row, cannot
    all run at once: together beyond its per-user total (sum_cpus) -- they
    then run one after another (inti's xlong on milan-bxi: 128 cores, one node)."""
    return bool(row and row.get("sum_cpus") and cpus and jobs > 1 and cpus * jobs > row["sum_cpus"])


def choose(rows, partition, cpus, seconds, wanted="auto"):
    """(name, row, error). @p wanted names a QoS (checked: open on the
    partition, duration and cores within its limits) or is "auto": among the
    QoS open on the partition that accept the job, the highest priority, then
    the shortest maximum time. The error is a French sentence (i18n.EN_RE
    translates it); name and row are those of the QoS concerned, or None."""
    from shrink_gui import i18n   # the language of the durations in the messages
    lang = i18n.LANG

    def check(row, name):
        if seconds > row["max_time"]:
            return (f"{fmt(seconds, lang)} dépasse le maximum de la QoS {name} sur {partition} "
                    f"({fmt(row['max_time'], lang)})")
        if row["max_cpus"] and cpus and cpus > row["max_cpus"]:
            return f"{cpus} cœurs dépassent le maximum de la QoS {name} sur {partition} ({row['max_cpus']})"
        if row.get("sum_cpus") and cpus and cpus > row["sum_cpus"]:      # a job that never starts
            return f"{cpus} cœurs dépassent le maximum de la QoS {name} sur {partition} ({row['sum_cpus']})"
        return None

    if wanted and wanted != "auto":
        row = limits(rows, wanted, partition)
        if row is None:
            return wanted, None, f"la QoS {wanted} n'est pas ouverte sur {partition}"
        return wanted, row, check(row, wanted)

    open_rows = [r for r in (limits(rows, n, partition) for n in names(rows)) if r is not None]
    fits = [r for r in open_rows if check(r, r["name"]) is None]
    if not fits:
        # The longest duration a QoS gives these cores (a job of several
        # nodes: 48 h on milan-bxi fits xlong's time, not its 128 cores).
        sized = [r for r in open_rows if not ((r["max_cpus"] and cpus and cpus > r["max_cpus"])
                                              or (r.get("sum_cpus") and cpus and cpus > r["sum_cpus"]))]
        most = max((r["max_time"] for r in sized), default=None)
        for_cpus = f" pour {cpus} cœurs" if len(sized) < len(open_rows) else ""
        return None, None, (f"aucune QoS n'accepte {fmt(seconds, lang)} sur {partition}{for_cpus}"
                            + (f" (au plus {fmt(most, lang)})" if most else ""))
    best = sorted(fits, key=lambda r: (-r["priority"], r["max_time"]))[0]
    return best["name"], best, None
