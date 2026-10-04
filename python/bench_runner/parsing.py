"""Reading a run's output: memory, checksums, CHUNK_STATS counters, seeds."""
import os
import re as _re

from . import state


# Matches the binary's line: "Memory consumption <msg> : <value> Mo"
_MEM_RE = _re.compile(r"Memory consumption[^:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*Mo")


def parse_peak_memory_mo(stdout_text):
    """Return the peak 'Memory consumption ... : X Mo' value from stdout, or None.
    The binary prints this several times (after init, per iteration, at end);
    ru_maxrss is monotonic, so we keep the maximum."""
    if not stdout_text:
        return None
    vals = [float(m) for m in _MEM_RE.findall(stdout_text)]
    return max(vals) if vals else None


# Final-checksum line, e.g.:
#   "k : 1641 val_sum = nan long_val_sum = 115... cv_val_sum = (-5.7e+09,-1.3e+13)"
# \b before val_sum avoids matching inside long_val_sum / cv_val_sum (the char
# before is '_', a word char -> no boundary); cv_val_sum has its own regex.
_VALSUM_RE = _re.compile(r"\bval_sum\s*=\s*([^\s]+)")
_CVSUM_RE  = _re.compile(r"\bcv_val_sum\s*=\s*\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)")


def _is_nonfinite(tok):
    """True if the numeric token is nan or +/-inf."""
    t = tok.strip().lower()
    if "nan" in t or "inf" in t:
        return True
    try:
        v = float(t)
    except ValueError:
        return False  # unrecognised token -> do not flag
    return v != v or v == float("inf") or v == float("-inf")


def detect_nonfinite_checksum(stdout_text):
    """Return a short reason if the LAST final checksum in the output is non-finite
    (nan/inf) -> corrupted result; else None. Only the final value matters (the
    binary prints one per iteration). No val_sum in output -> None, so this never
    fires on runs that do not print a checksum."""
    if not stdout_text:
        return None
    bad = []
    vs = _VALSUM_RE.findall(stdout_text)
    if vs and _is_nonfinite(vs[-1]):
        bad.append(f"val_sum={vs[-1]}")
    cv = _CVSUM_RE.findall(stdout_text)
    if cv:
        re_, im_ = cv[-1]
        if _is_nonfinite(re_) or _is_nonfinite(im_):
            bad.append(f"cv_val_sum=({re_},{im_})")
    return ", ".join(bad) if bad else None


# --- Compression-health guard (CHUNK_STATS) --------------------------------
# `still compressed (at exit)` (checked by the seq-vs-OMP phase) cannot see a
# collapse of IN-FLIGHT compression: the final sweep compresses everything
# anyway, so that counter stays identical while memory blows up. The counters
# that DO move when in-flight marking breaks are `mark` (writes that newly
# flagged a chunk) against `create` (chunks materialised): a healthy chunk2 run
# sits around mark/create ~0.65; the Sept-2026 chunk2a regression (isv
# pre-check rejecting the chunk2a-only predicate clause) dropped it to ~0.07
# while `still compressed (at exit)` did not budge. We flag a run whose ratio
# falls under a per-variant floor. Only variants listed in the floor map are
# checked; a run without a CHUNK_STATS block (non-stats build) is never flagged.
#
# Floors are deliberately loose (roughly half the healthy value) so they catch
# a collapse, not normal noise. Tune with --stats-mark-ratio-min.
_STATS_MARK_RE   = _re.compile(r"^\s*mark\s*=\s*(\d+)", _re.MULTILINE)
_STATS_CREATE_RE = _re.compile(r"^\s*create\s*=\s*(\d+)", _re.MULTILINE)
_STATS_CEXEC_RE  = _re.compile(r"^\s*compress \(during exec\)\s*=\s*(\d+)", _re.MULTILINE)


# Below this many created chunks the ratio is too noisy to judge (tiny runs).
_STATS_MIN_CREATE_FOR_GUARD = 10_000


def parse_chunk_stats_counters(stdout_text):
    """Return {stat_tag: {"mark": int, "create": int, "compress_exec": int|None}}
    for every CHUNK_STATS block in the output (tag = 'chunk', 'chunk2',
    'chunk2a'...). Missing counters are omitted from the inner dict."""
    out = {}
    if not stdout_text:
        return out
    for m in _CHUNK_STATS_BLOCK_RE.finditer(stdout_text):
        tag, body = m.group(1), m.group(2)
        d = {}
        mk = _STATS_MARK_RE.search(body)
        cr = _STATS_CREATE_RE.search(body)
        ce = _STATS_CEXEC_RE.search(body)
        if mk:
            d["mark"] = int(mk.group(1))
        if cr:
            d["create"] = int(cr.group(1))
        if ce:
            d["compress_exec"] = int(ce.group(1))
        out[tag] = d
    return out


def detect_compression_collapse(stdout_text):
    """Return a short reason if any checked CHUNK_STATS block shows in-flight
    marking collapsed (mark/create under its floor); else None. Never fires
    when the stats block, the counters, or the floor for that tag are absent."""
    reasons = []
    for tag, d in parse_chunk_stats_counters(stdout_text).items():
        floor = state._STATS_MARK_RATIO_MIN.get(tag)
        if floor is None or "mark" not in d or "create" not in d:
            continue
        create = d["create"]
        if create < _STATS_MIN_CREATE_FOR_GUARD:
            continue
        ratio = d["mark"] / create
        if ratio < floor:
            extra = ""
            if d.get("compress_exec") is not None:
                extra = f", compress(during exec)={d['compress_exec']}"
            reasons.append(
                f"[{tag}] mark/create={ratio:.3f} < {floor} "
                f"(mark={d['mark']}, create={create}{extra})"
            )
    return "; ".join(reasons) if reasons else None


_CHUNK_STATS_BLOCK_RE = _re.compile(
    r"----- CHUNK_STATS \[(\w+)\] -----\n(.*?)\n-{5,}", _re.DOTALL
)
_STILL_AT_EXIT_RE = _re.compile(r"still compressed \(at exit\)\s*=\s*(\d+)")
_CHECKSUM_RE = _re.compile(
    r"k\s*:\s*(\d+)\s+val_sum\s*=\s*(\S+)\s+long_val_sum\s*=\s*(\S+)\s+cv_val_sum\s*=\s*(\([^)]*\))"
)
_SEED_FALLBACK_RE = _re.compile(r"Save seed in file")


def _parse_still_at_exit(stdout_text, variant_base):
    """`still compressed (at exit)` from `variant_base`'s CHUNK_STATS block in
    this run's stdout, or None if that block never appeared (e.g. the binary
    reports a different stat_variant tag than expected, or zero activity)."""
    for m in _CHUNK_STATS_BLOCK_RE.finditer(stdout_text):
        if m.group(1) == variant_base:
            sm = _STILL_AT_EXIT_RE.search(m.group(2))
            return int(sm.group(1)) if sm else None
    return None


def _parse_last_long_val_sum(stdout_text):
    """The LAST long_val_sum printed (the final iteration's exact integer
    checksum), or None if the bench never printed one (e.g. a build without
    the checksum line, or a run that crashed before finishing)."""
    matches = list(_CHECKSUM_RE.finditer(stdout_text))
    return matches[-1].group(3) if matches else None


def _seed_was_pinned(stdout_text):
    """False if the run fell back to a random seed (our pinned seed failed to
    load from seed<iter_max>.txt) -- see initializeRNG in bench_utils.cpp."""
    return _SEED_FALLBACK_RE.search(stdout_text) is None


def _write_pinned_seed(output_dir, iter_max, seed):
    """(Re)write the fixed seed into seed<iter_max>.txt in output_dir -- the
    SAME cwd every run in this phase uses, so all of them read the identical
    value regardless of which binary/config is launched. Rewritten before
    EVERY run: initializeRNG only regenerates when the file is absent or
    holds 0, so a pinned non-zero value is never overwritten by the bench
    itself, but writing it fresh each time removes any doubt (and self-heals
    if a previous run's crash left the file missing or truncated)."""
    seed_path = os.path.join(output_dir, f"seed{iter_max}.txt")
    with open(seed_path, "w") as f:
        f.write(str(seed))
    return seed_path
