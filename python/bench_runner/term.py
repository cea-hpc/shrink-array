"""Terminal output: colours (only on a tty, never with NO_COLOR), the log tee, the clock."""
import os
import sys
import time
import re as _re


# --- Colour / display helpers --------------------------------------------
# Colour is emitted only to a real terminal and only when NO_COLOR is unset,
# so redirected logs / CSV pipelines stay clean (no escape sequences on disk).
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else str(s)


def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)
def blue(s):   return _c("34", s)
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)


_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")

class _Tee:
    """Mirror everything written to the real stdout into a log file as well.

    The screen keeps the live experience (colours + \r-animated progress bars).
    The file gets a cleaned copy: ANSI colour codes stripped, and carriage-return
    progress redraws collapsed so the log holds only finalised lines (the same
    text you'd see if you scrolled back), not hundreds of half-drawn bars.
    """
    def __init__(self, real, log_file):
        self._real = real
        self._log = log_file
        self._pending = ""   # buffer for the current (possibly \r-rewritten) line

    def write(self, s):
        # 1) screen: unchanged, immediately
        self._real.write(s)
        # 2) file: strip colours, resolve \r rewrites into single final lines
        try:
            clean = _ANSI_RE.sub("", s)
            for ch in clean:
                if ch == "\r":
                    # progress redraw: drop whatever was buffered for this line
                    self._pending = ""
                elif ch == "\n":
                    self._log.write(self._pending + "\n")
                    self._pending = ""
                else:
                    self._pending += ch
        except Exception:
            pass  # logging must never break the run

    def flush(self):
        self._real.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        # keep colour/tty detection keyed to the REAL terminal
        return getattr(self._real, "isatty", lambda: False)()

    def close_log(self):
        try:
            if self._pending:
                self._log.write(self._pending + "\n")
                self._pending = ""
            self._log.flush()
            self._log.close()
        except Exception:
            pass


def _hms():
    """Wall-clock HH:MM:SS, stamped at the head of every per-test progress line
    in every phase.

    A full run is hours long and each line is rewritten in place over its own
    predecessor, so without a stamp the log says what was attempted but never
    WHEN: a run that took forty minutes on a loaded node and one that hung look
    identical after the fact, and a phase that turned out to have nothing to do
    (see the -stats pairs in run_seq_omp_compare_phase) is indistinguishable
    from one that did real work. The stamp is taken when the line is composed,
    i.e. just before the run it describes starts, so consecutive lines read as a
    timeline and the gap between two of them IS that run's duration.
    """
    return time.strftime("%H:%M:%S")
