"""The live progress line and bar shared by every phase."""
import time

from .builds import short_config_label
from .term import _hms, dim, green


# Fixed field widths so every progress line aligns regardless of config/variant
# name length. CONFIG_W fits the longest short label
# ("llvm-have-aggressive-compression" = 31) with one space of slack, so nothing
# is truncated (the old [:30] chopped the trailing 'n' off "...compression").
CONFIG_W = 32
VARIANT_W = 14  # longest label is "std_alloc@1024" (14); avoids truncating "chunk2a@256"

# Fixed-width column widths for the live progress line, shared by every phase
# (CHECK / SAN / PERF / STL) so their columns line up vertically. Any phase that
# doesn't use a column passes None and gets a blank of the right width.
TH_W  = 5   # "  8th"
RUN_W = 11  # "Run:  1/16 "


def _prog(phase, color_fn, cfg, var="", *, th=None, run=None, reps=None, extra=""):
    """Format one live progress line with fixed columns:

        HH:MM:SS PHASE  config......... | variant....... |   Nth | Run:  r/reps | extra

    The phase label is padded to width BEFORE colouring, so ANSI escape codes
    never enter the width computation (which would break alignment). Columns the
    phase doesn't use (th / run) are blanked to their fixed width.
    """
    th_s  = f"{th:>3}th" if th is not None else " " * TH_W
    run_s = (f"Run:{run:3}/{reps:<3}" if (run is not None and reps is not None)
             else " " * RUN_W)
    return (f"{dim(_hms())} "
            f"{color_fn(f'{phase:<5}')} "
            f"{short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
            f"{var[:VARIANT_W]:<{VARIANT_W}} | "
            f"{th_s} | {run_s} | {extra}")


def get_progress_bar(current, total, width=20):
    if total <= 0:
        return f"|{'-' * width}| n/a"
    percent = float(current) / total
    done = int(width * percent)
    bar = green("█" * done) + dim("-" * (width - done))
    return f"|{bar}| {percent:5.1%}"


class RunCounter:
    """Runs done so far out of the campaign's total: the progress bar and the
    ETA of the check, STL, sanitizer and perf phases."""

    def __init__(self, total):
        self.idx = 0
        self.total = total
        self.start = time.time()
