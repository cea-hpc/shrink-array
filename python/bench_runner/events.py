# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Structured progress of a bench_and_test.py run: one JSON object per line.

Written to --events-file, else $CHUNK_BENCH_EVENTS, else nowhere. Kinds:
start, phase_start, progress, phase_end, end (see emit's callers). Every
record carries "t" (epoch seconds). Writing is best effort: a full disk or a
missing directory never reaches the bench.
"""
import contextlib
import json
import os
import threading
import time

_path = None
_phase = None
_lock = threading.Lock()
_last_progress = 0.0


def open_file(path=None):
    """Start writing to @p path (or $CHUNK_BENCH_EVENTS); None/empty: off."""
    global _path
    _path = path or os.environ.get("CHUNK_BENCH_EVENTS") or None
    if _path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(_path)), exist_ok=True)
        except OSError:
            _path = None


def enabled():
    return _path is not None


def emit(kind, **fields):
    """Append one record. Never raises."""
    if not _path:
        return
    rec = {"t": round(time.time(), 3), "kind": kind}
    rec.update(fields)
    try:
        line = json.dumps(rec, default=str)
        with _lock, open(_path, "a") as f:
            f.write(line + "\n")
    except (OSError, TypeError, ValueError):
        pass


def current_phase():
    return _phase


@contextlib.contextmanager
def phase(name, errors):
    """Wrap one phase: phase_start, then phase_end with its status, its
    duration and the errors it added to @p errors (the run's error list)."""
    global _phase
    n0, t0 = len(errors), time.time()
    _phase = name
    emit("phase_start", phase=name)
    status = "ok"
    try:
        yield
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "exception"
        raise
    finally:
        added = len(errors) - n0
        if status == "ok" and added:
            status = "errors"
        emit("phase_end", phase=name, status=status, errors=added, seconds=round(time.time() - t0, 1))
        _phase = None


def progress(done, total, min_interval=2.0, **fields):
    """done/total of the current phase, at most one record per @p min_interval
    seconds (and always the last one)."""
    global _last_progress
    if not _path:
        return
    now = time.time()
    if now - _last_progress < min_interval and done != total:
        return
    _last_progress = now
    emit("progress", phase=fields.pop("phase", _phase), done=done, total=total, **fields)
