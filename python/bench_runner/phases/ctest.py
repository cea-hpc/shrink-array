"""Phase 0b: the validation harnesses (ctest -L validation) of every build dir."""
import subprocess
import sys
import time
import re as _re

from ..builds import resolve_build_dir, short_config_label, validation_built
from ..progress import CONFIG_W
from ..runs import _append_error_now, effective_timeout, run_capture_stdout_streaming
from ..settings import SANITIZER_CONFIGS, is_sanitizer_config, sanitizer_env_for
from ..term import _hms, bold, cyan, dim, green, red, yellow


_CTEST_SUMMARY_RE = _re.compile(
    r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")

# ctest's own progress lines: `    Start  12: name` when a test is launched,
# ` 3/184 Test  #12: name ......   Passed    4.21 sec` when it ends (also
# `***Failed`, `***Timeout`, `***Exception: ...`, always closed by `N sec`).
_CTEST_START_RE = _re.compile(r"^\s*Start\s+\d+:\s+(\S+)")
_CTEST_DONE_RE  = _re.compile(r"^\s*(\d+)/(\d+)\s+Test\s+#\d+:\s+(\S+)\s.*?(\d+(?:\.\d+)?)\s+sec\s*$")
_CTEST_TOTAL_RE = _re.compile(r"Total Tests:\s*(\d+)")

# Below this, a config's run is not worth a "slowest tests" line.
_SLOWEST_SHOWN_ABOVE_S = 60.0


def _fmt_dur(s):
    """12.3s / 4m05s / 1h02m."""
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def _ctest_count(cmd, cwd):
    """Number of tests `cmd` will run (`ctest -N` with the same filters), or
    None. Costs a second at most; only for the `k/N` of the progress line."""
    try:
        res = subprocess.run([cmd[0], "-N"] + [a for a in cmd[1:] if a != "--output-on-failure"],
                             cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=60)
        m = _CTEST_TOTAL_RE.search(res.stdout or "")
        return int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class _CtestProgress:
    """The live line of one `ctest` run: `k/N`, elapsed, failures so far, and
    the test that has been running the longest with its age.

    Under a sanitizer one test can run for many minutes with nothing printed
    by ctest; naming the oldest test still running is what tells a slow test
    from a hung one while it happens, not after the phase."""

    def __init__(self, lead, total=None):
        self.lead = lead
        self.t0 = time.perf_counter()
        self.done = 0
        self.total = total
        self.n_fail = 0
        self.running = {}    # name -> perf_counter() at its Start line
        self.durations = []  # (seconds, name), ctest's own figure
        self._w = 0          # visible width of the previous line, to blank it

    def on_line(self, line):
        m = _CTEST_START_RE.match(line)
        if m:
            self.running[m.group(1)] = time.perf_counter()
            return
        m = _CTEST_DONE_RE.match(line)
        if m:
            self.done, self.total = int(m.group(1)), int(m.group(2))
            name = m.group(3)
            self.running.pop(name, None)
            self.durations.append((float(m.group(4)), name))
            if "Passed" not in line:
                self.n_fail += 1
            self.render()

    def oldest(self):
        if not self.running:
            return None
        name, t = min(self.running.items(), key=lambda kv: kv[1])
        return name, time.perf_counter() - t

    def still_running(self):
        now = time.perf_counter()
        return sorted(((now - t, n) for n, t in self.running.items()), reverse=True)

    def slowest(self, n=3):
        return sorted(self.durations, reverse=True)[:n]

    def render(self):
        plain = [f"running… {self.done}/{self.total if self.total else '?'}",
                 _fmt_dur(time.perf_counter() - self.t0)]
        shown = list(plain)
        if self.n_fail:
            plain.append(f"{self.n_fail} failed")
            shown.append(red(f"{self.n_fail} failed"))
        o = self.oldest()
        if o:
            txt = f"longest running: {o[0][:48]} ({_fmt_dur(o[1])})"
            plain.append(txt)
            shown.append(yellow(txt) if o[1] >= _SLOWEST_SHOWN_ABOVE_S else txt)
        width = len(" | ".join(plain))
        pad = " " * max(0, self._w - width)
        self._w = width
        sys.stdout.write("\r" + self.lead + " | ".join(shown) + pad)
        sys.stdout.flush()

    def clear_pad(self):
        """Blanks to append to the final line so no tail of the live one shows."""
        return " " * self._w


def run_ctest_phase(args, extra_ldpath, errors):
    """Validation harnesses, through ctest.

    examples/validation registers exit-0/1 programs under the `validation`
    label (rechunk_validation, dyn_geometry_validation, compress_roundtrip,
    pool_accounting, deferred_path_validation, open_types_validation -- one
    executable per variant, the last four tested at 1 and 8 threads). They
    are quick and they run FIRST: a library that cannot round-trip a
    compression pass or tear itself down is not worth benchmarking, and the
    failure is far easier to read from a 150-line harness than from the bench.

    Runs `ctest -L <label> --output-on-failure` in each config's build
    directory (the one holding CTestTestfile.cmake, with or without the $arch
    level), then in the sanitizer configs' (SANITIZER_CONFIGS, unless
    --skip-sanitizers): the default configs are perf builds, which do not
    compile the harnesses. ctest's own per-test timeout applies; the phase
    timeout is the bench one at 8 threads, x4 (x20 under a sanitizer). A
    config with no build directory is
    noted once and skipped, and so is one whose configuration does not build
    the harnesses: only Debug, CHECK_OPTIM and sanitizer builds compile them
    (examples/CMakeLists.txt, CHUNK_BUILD_VALIDATION).
    """
    print(cyan(bold(f"\n🧪 Validation harnesses — ctest -L {args.ctest_label}")))

    # The sanitizer configs too: they are Debug builds, the ones that compile
    # the harnesses, while every default config is a perf (Release) build
    # that does not. Without them the phase found nothing to run.
    configs = list(args.configs)
    if not args.skip_sanitizers:
        configs += [c for c in SANITIZER_CONFIGS if c not in configs]

    n_pass = n_total = 0
    n_not_built = 0  # configs skipped: CHUNK_VALIDATION_BUILT=OFF
    for cfg in configs:
        bdir = resolve_build_dir(args.build_root, cfg)
        lead = f"{dim(_hms())} {short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
        if not bdir:
            print(lead + dim("no CTestTestfile.cmake (not a CMake build dir) — skipped"))
            continue
        if not validation_built(bdir):
            print(lead + dim("validation not built here (Debug, CHECK_OPTIM and sanitizer builds only) — skipped"))
            n_not_built += 1
            continue

        cmd = ["ctest", "-L", args.ctest_label, "--output-on-failure",
               "-j", str(args.ctest_jobs)]
        if args.ctest_regex:
            cmd += ["-R", args.ctest_regex]
        # The harnesses set OMP_NUM_THREADS themselves (test property); only
        # the sanitizer options need to reach them.
        env = sanitizer_env_for(cfg) if is_sanitizer_config(cfg) else {}
        # A sanitizer runs the harnesses 5-20x slower (settings.py).
        to  = effective_timeout(args.timeout, args.timeout_per_thread, 8) * (
            20 if is_sanitizer_config(cfg) else 4)

        prog = _CtestProgress(lead, _ctest_count(cmd, bdir))
        prog.render()
        ok, dur, code, out, err_tail = run_capture_stdout_streaming(
            cmd, env, to, on_line=prog.on_line, on_tick=prog.render,
            extra_ld_library_path=extra_ldpath, cwd=bdir)
        if code == -9 and prog.running:
            err_tail = (err_tail + "\n" if err_tail else "") + "still running at the timeout: " + ", ".join(
                f"{n} ({_fmt_dur(a)})" for a, n in prog.still_running())

        m = _CTEST_SUMMARY_RE.search(out)
        if m:
            failed, total = int(m.group(2)), int(m.group(3))
            passed = total - failed
        else:
            failed, total, passed = (0 if ok else 1), (1 if not ok else 0), 0
        n_pass  += passed
        n_total += total

        if ok and total > 0:
            sys.stdout.write("\r" + lead + green(f"{passed}/{total} passed") +
                             f" ({dur:.1f}s)" + prog.clear_pad() + "\n")
        elif total == 0 and ok:
            sys.stdout.write("\r" + lead + yellow("no test carries the label") + prog.clear_pad() + "\n")
        else:
            sys.stdout.write("\r" + lead + red(f"{passed}/{total} passed") +
                             f" (ctest code {code}, {dur:.1f}s)" + prog.clear_pad() + "\n")
            # Keep what ctest prints for the failing tests (--output-on-failure):
            # the harness names the first mismatching element, chunk and offset.
            failing = [l for l in out.splitlines()
                       if "***" in l or "Failed" in l or "FAIL" in l]
            _entry = (
                f"CTEST_FAIL: {' '.join(cmd)} in {bdir} code={code}\n"
                f"  {failed} of {total} test(s) failed:\n    "
                + "\n    ".join(failing[:40])
                + ("\n  stderr_tail:\n    " + err_tail.replace("\n", "\n    ") if err_tail else "")
            )
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        if code == -9 and prog.running:
            print(" " * (len(_hms()) + 1 + CONFIG_W) + yellow("   still running at the timeout: " + ", ".join(
                f"{n} ({_fmt_dur(a)})" for a, n in prog.still_running())))
        if dur >= _SLOWEST_SHOWN_ABOVE_S and prog.durations:
            print(" " * (len(_hms()) + 1 + CONFIG_W) + dim("   slowest: " + ", ".join(
                f"{n} {_fmt_dur(s)}" for s, n in prog.slowest(3))))
        sys.stdout.flush()

    if n_total == 0 and n_not_built > 0:
        # Perf configurations only: nothing went wrong, the harnesses are
        # not compiled there (-DCHUNK_BUILD_VALIDATION=ON forces them).
        tag = dim("not built in any of these configurations (Debug, CHECK_OPTIM and "
                  "sanitizer builds only; -DCHUNK_BUILD_VALIDATION=ON forces them)")
    elif n_total == 0:
        tag = yellow("no validation test found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Validation harnesses: {tag}")
    return n_pass, n_total
