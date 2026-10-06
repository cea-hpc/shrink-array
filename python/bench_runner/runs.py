"""Launching one run: timeout, environment, NUMA prefix, logs, failure records."""
import subprocess
import os
import queue
import signal
import sys
import threading
import time
import json

from . import state
from .builds import resolve_exe
from .parsing import detect_compression_collapse, detect_nonfinite_checksum, parse_peak_memory_mo
from .settings import SANITIZER_HIT_MARKERS
from .state import _numa
from .term import bold, cyan, dim, green, red


def _write_full_log(cmd, child_env, returncode, stdout_text, stderr_text):
    """When --full-log-dir is set, dump one child run's COMPLETE stdout+stderr to
    its own file. The name encodes binary, OMP thread count, a global run counter
    and the status, so each execution is identifiable (chunk2 vs chunk2a, which run
    re-chunked / crashed). No-op when off. Never raises: logging must not break the
    bench."""
    if not state._FULL_LOG_DIR:
        return
    try:
        binary = os.path.basename(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else "run"
        omp = str(child_env.get("OMP_NUM_THREADS", "?"))
        with state._FULL_LOG_LOCK:
            state._FULL_LOG_COUNTER += 1
            idx = state._FULL_LOG_COUNTER
        status = "ok" if returncode == 0 else f"rc{returncode}"
        path = os.path.join(state._FULL_LOG_DIR, f"{binary}__omp{omp}__{idx:05d}__{status}.log")
        with open(path, "w") as fh:
            fh.write(f"# cmd: {' '.join(map(str, cmd))}\n")
            fh.write(f"# returncode: {returncode}\n")
            fh.write("# ===== STDOUT =====\n")
            fh.write(stdout_text or "")
            if not (stdout_text or "").endswith("\n"):
                fh.write("\n")
            fh.write("# ===== STDERR =====\n")
            fh.write(stderr_text or "")
    except Exception:
        pass  # logging must never break the run


def _run_cmd_impl(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, must_contain=None):
    """Run cmd. Returns (success, duration_s, returncode, stderr_tail, mem_mo).

    extra_ld_library_path: path(s) to PREPEND to LD_LIBRARY_PATH for this run.
    Needed for llvm binaries linked with -stdlib=libc++ that look for
    libc++.so.1 in the llvm module's lib/ at runtime (or the gcc lib64 for
    libstdc++). Not needed once the binaries carry a proper RPATH.

    cwd: working directory for the child. The binaries read/write data_tab_*
    and memory_mo.txt relative to their cwd, so pointing this at the shared
    output directory keeps every run's data together.

    mem_mo: peak memory in MB parsed from the binary's stdout, or None.
    """
    start = time.perf_counter()
    # NUMA binding: pin OpenMP threads to cores (local-socket memory).
    # Unbound threads on a multi-socket node migrate and hit remote RAM,
    # which dominates and hides real scaling. Caller/outer env can override.
    _bind = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    child_env = {**_bind, **os.environ, **env_vars}
    if extra_ld_library_path:
        prev = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            extra_ld_library_path + (":" + prev if prev else "")
        )
    cmd = _numa.interleave_prefix(child_env, state._NUMA_MODE) + list(cmd)  # --numa-interleave
    try:
        res = subprocess.run(
            cmd, env=child_env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout,
        )
        err_tail = "\n".join(res.stderr.splitlines()[-30:]) if res.stderr else ""
        mem_mo = parse_peak_memory_mo(res.stdout)
        _write_full_log(cmd, child_env, res.returncode, res.stdout, res.stderr)
        # Sanitizer hit detection: ASan/TSan write their report to stderr (and
        # sometimes stdout). A report means FAIL even if the process exited 0
        # (ASan can continue past an error depending on options). We surface it
        # by making success False and prefixing err_tail, so existing callers
        # treat it exactly like any other failure with no signature change.
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        san_hit = any(m in combined for m in SANITIZER_HIT_MARKERS)
        success = (res.returncode == 0) and not san_hit
        if san_hit:
            # Keep the sanitizer report lines (the most relevant tail), not just
            # the last 30 stderr lines which may be past the SUMMARY.
            report_lines = [ln for ln in combined.splitlines()
                            if any(m in ln for m in SANITIZER_HIT_MARKERS)]
            err_tail = "SANITIZER HIT:\n" + "\n".join(report_lines[-30:]) + \
                       ("\n--- stderr tail ---\n" + err_tail if err_tail else "")
        # Optional required-output assertion: a run that exits cleanly but did NOT
        # print `must_contain` is a FAIL (used by the adaptive re-chunk feature
        # test to require an [APPLIED] line). Non-invasive: default None.
        if success and must_contain is not None and must_contain not in combined:
            success = False
            err_tail = (f"MISSING EXPECTED OUTPUT: {must_contain!r} not found in run output\n"
                        "--- output tail ---\n"
                        + "\n".join(combined.splitlines()[-30:]))
        # Final-checksum sanity: a run can exit 0 yet produce a corrupted result
        # (val_sum = nan / inf from a bad re-chunk, a race, an overflow). The perf
        # runs otherwise never check correctness, so this passed silently. Flag a
        # non-finite final checksum as a FAIL, like any other failure.
        if success:
            ck_bad = detect_nonfinite_checksum(res.stdout)
            if ck_bad is not None:
                success = False
                err_tail = ("CORRUPTED CHECKSUM: " + ck_bad + "\n"
                            "--- output tail ---\n"
                            + "\n".join(combined.splitlines()[-30:]))
        # Compression-health guard: a run can exit 0 with a perfect checksum and
        # still have stopped compressing in flight (memory then grows several-
        # fold). `still compressed (at exit)` cannot see it; mark/create can.
        # Same FAIL plumbing as the checks above, no signature change.
        if success:
            cc_bad = detect_compression_collapse(res.stdout)
            if cc_bad is not None:
                success = False
                err_tail = ("COMPRESSION COLLAPSE: " + cc_bad + "\n"
                            "--- output tail ---\n"
                            + "\n".join(combined.splitlines()[-30:]))
        return success, time.perf_counter() - start, res.returncode, err_tail, mem_mo
    except subprocess.TimeoutExpired:
        return False, float(timeout), -9, f"TIMEOUT after {timeout}s", None
    except FileNotFoundError as e:
        # The binary vanished between resolution and launch — almost always a
        # concurrent rebuild replacing the file. Signalled with -2 so the caller
        # can wait and re-resolve instead of spamming the same error.
        return False, 0.0, -2, f"FileNotFoundError: {e}", None
    except OSError as e:
        # ETXTBSY (Errno 26, "Text file busy"): the binary is open for writing
        # by another process — a concurrent link/copy or `cmake --build` that
        # just regenerated it (the non-OMP configs are built last, so the bench
        # can reach them mid-write). Transient, exactly like a vanished binary,
        # so signal -2 to trigger the wait-and-retry path instead of a hard
        # FAIL. Other OSErrors keep the generic -1.
        if getattr(e, "errno", None) == 26:  # errno.ETXTBSY
            return False, 0.0, -2, f"ETXTBSY (text file busy): {e}", None
        return False, 0.0, -1, f"{type(e).__name__}: {e}", None
    except Exception as e:
        return False, 0.0, -1, f"{type(e).__name__}: {e}", None


_CTEST_FAILED_NAME_RE = __import__("re").compile(r"^\s*\d+\s+-\s+(\S+)\s+\(", __import__("re").M)


def _run_key(r):
    return (tuple(r["cmd"]), r.get("cwd"), tuple(sorted((r.get("env") or {}).items())))


def _note_run(executor, ok, cmd, env_vars, timeout, extra_ld_library_path, cwd, must_contain, stdout_text=None):
    rec = {"executor": executor, "cmd": [str(c) for c in cmd], "env": dict(env_vars or {}),
           "timeout": timeout, "ld": extra_ld_library_path, "cwd": cwd, "must_contain": must_contain}
    if ok:
        k = _run_key(rec)
        state._PENDING_FAILED_RUNS[:] = [r for r in state._PENDING_FAILED_RUNS if _run_key(r) != k]
        return
    # ctest: replay only the tests that failed, not the whole label.
    if os.path.basename(rec["cmd"][0]) == "ctest" and stdout_text:
        tail = stdout_text.split("The following tests FAILED", 1)
        names = _CTEST_FAILED_NAME_RE.findall(tail[1]) if len(tail) == 2 else []
        if names:
            import re as _re2
            rec["cmd"] = [rec["cmd"][0], "-R", "^(" + "|".join(_re2.escape(n) for n in names) + ")$",
                          "--output-on-failure"]
            rec["ctest_tests"] = names
    state._PENDING_FAILED_RUNS.append(rec)


def run_cmd(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, must_contain=None):
    """_run_cmd_impl(), recording a failed run for --rerun-failed."""
    res = _run_cmd_impl(cmd, env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
                        cwd=cwd, must_contain=must_contain)
    _note_run("run_cmd", res[0], cmd, env_vars, timeout, extra_ld_library_path, cwd, must_contain)
    return res


def run_capture_stdout(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, merge_stderr=False):
    """_run_capture_stdout_impl(), recording a failed run for --rerun-failed."""
    res = _run_capture_stdout_impl(cmd, env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
                                   cwd=cwd, merge_stderr=merge_stderr)
    _note_run("capture", res[0], cmd, env_vars, timeout, extra_ld_library_path, cwd, None,
              stdout_text=res[3])
    return res


def run_capture_stdout_streaming(cmd, env_vars, timeout, on_line=None, on_tick=None, tick_s=1.0,
                                 extra_ld_library_path=None, cwd=None):
    """run_capture_stdout() for a long child whose progress is in its stdout.

    Each stdout line goes to on_line(line) as it arrives, and on_tick() runs
    every tick_s seconds even when nothing arrives -- a test under a sanitizer
    can run for many minutes before ctest prints its line. Same return shape
    and same --rerun-failed record as run_capture_stdout() (the replay itself
    goes through the non-streaming capture). The callbacks only display:
    an exception in one is swallowed, never raised into the run.
    """
    res = _run_capture_stdout_streaming_impl(cmd, env_vars, timeout, on_line, on_tick, tick_s,
                                             extra_ld_library_path, cwd)
    _note_run("capture", res[0], cmd, env_vars, timeout, extra_ld_library_path, cwd, None,
              stdout_text=res[3])
    return res


def _run_capture_stdout_streaming_impl(cmd, env_vars, timeout, on_line, on_tick, tick_s,
                                       extra_ld_library_path, cwd):
    """_run_capture_stdout_impl() with stdout read line by line (see
    run_capture_stdout_streaming). stderr is drained by its own thread so a
    child that fills the stderr pipe cannot block on it."""
    start = time.perf_counter()
    _bind = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    child_env = {**_bind, **os.environ, **env_vars}
    if extra_ld_library_path:
        prev = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            extra_ld_library_path + (":" + prev if prev else "")
        )
    cmd = _numa.interleave_prefix(child_env, state._NUMA_MODE) + list(cmd)  # --numa-interleave

    def _safe(fn, *a):
        if fn is None:
            return
        try:
            fn(*a)
        except Exception:
            pass

    try:
        # Own process group: on a timeout the whole tree goes (ctest AND the
        # tests it started). Killing ctest alone leaves the tests running and
        # holding the stdout pipe open.
        proc = subprocess.Popen(cmd, env=child_env, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1,
                                start_new_session=True)
    except FileNotFoundError as e:
        return False, 0.0, -2, "", f"FileNotFoundError: {e}"
    except OSError as e:
        if getattr(e, "errno", None) == 26:  # ETXTBSY
            return False, 0.0, -2, "", f"ETXTBSY (text file busy): {e}"
        return False, 0.0, -1, "", f"{type(e).__name__}: {e}"

    lines_q = queue.Queue()
    err_parts = []

    def _pump_out():
        try:
            for ln in proc.stdout:
                lines_q.put(ln)
        finally:
            lines_q.put(None)  # EOF

    def _pump_err():
        try:
            err_parts.append(proc.stderr.read())
        except Exception:
            pass

    def _kill_tree():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()

    t_out = threading.Thread(target=_pump_out, daemon=True)
    t_err = threading.Thread(target=_pump_err, daemon=True)
    t_out.start()
    t_err.start()

    out_lines = []
    timed_out = False
    next_tick = time.perf_counter() + tick_s
    try:
        while True:
            now = time.perf_counter()
            if timeout is not None and now - start > timeout:
                timed_out = True
                break
            try:
                ln = lines_q.get(timeout=max(0.05, min(tick_s, next_tick - now)))
            except queue.Empty:
                ln = ""
            if ln is None:
                break
            if ln:
                out_lines.append(ln)
                _safe(on_line, ln.rstrip("\n"))
            if time.perf_counter() >= next_tick:
                _safe(on_tick)
                next_tick = time.perf_counter() + tick_s

        if not timed_out:
            try:
                left = None if timeout is None else max(1.0, timeout - (time.perf_counter() - start))
                proc.wait(timeout=left)
            except subprocess.TimeoutExpired:
                timed_out = True
    except KeyboardInterrupt:
        # The group no longer gets the terminal's SIGINT: take it down here.
        _kill_tree()
        raise

    if timed_out:
        _kill_tree()
        t_out.join(timeout=5)
        while True:  # what the pump read before the kill
            try:
                ln = lines_q.get_nowait()
            except queue.Empty:
                break
            if ln:
                out_lines.append(ln)
        return False, float(timeout), -9, "".join(out_lines), f"TIMEOUT after {timeout}s"

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    out_text = "".join(out_lines)
    err_text = "".join(err_parts)
    err_tail = "\n".join(err_text.splitlines()[-30:]) if err_text else ""
    _write_full_log(cmd, child_env, proc.returncode, out_text, err_text)
    return proc.returncode == 0, time.perf_counter() - start, proc.returncode, out_text, err_tail


def _failed_cases_path(errors_log):
    return os.path.join(os.path.dirname(os.path.abspath(errors_log)), "failed_cases.jsonl")


def _append_error_now(errors_log, entry):
    """Append a single error entry to the log immediately, so failures are on
    disk the instant they happen — not only at the final flush_summary(). This
    is what lets you `tail -f` the log during a run and see the real stderr of a
    FAIL right away instead of waiting for the run to finish (or guessing when it
    is interrupted). Best-effort: never let logging raise into the run loop.

    Also appends the failed runs behind this entry to failed_cases.jsonl, the
    list --rerun-failed replays (none: the entry is recorded as not
    replayable -- a comparison across runs, a guard)."""
    try:
        with open(errors_log, "a") as f:
            f.write(entry.rstrip("\n") + "\n")
    except OSError:
        pass
    try:
        first = entry.strip().splitlines()[0] if entry.strip() else ""
        case = {"kind": first.split(":", 1)[0], "summary": first[:300],
                "runs": list(state._PENDING_FAILED_RUNS)}
        state._PENDING_FAILED_RUNS.clear()
        with open(_failed_cases_path(errors_log), "a") as f:
            f.write(json.dumps(case) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def run_rerun_failed(args):
    """--rerun-failed: replay the recorded failed runs; returns the exit code."""
    src = args.rerun_failed or _failed_cases_path(args.errors_log)
    if not os.path.isfile(src):
        print(red(f"--rerun-failed: {src} not found (no failed run recorded there)"))
        return 2
    cases = []
    with open(src) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    cases.append(json.loads(ln))
                except ValueError:
                    pass
    reps = max(1, args.rerun_repeat)
    print(cyan(bold(f"\n🔁 Re-running the failed cases of {src}  ({len(cases)} case(s), x{reps})")))
    still, n_ok, n_fail, n_skip = [], 0, 0, 0
    for case in cases:
        runs = case.get("runs") or []
        head = case.get("summary", "")[:150]
        if not runs:
            n_skip += 1
            print(dim(f"  ⏭  not replayable (no single failed run behind it): {head}"))
            continue
        case_fail = None
        for rep in range(reps):
            for r in runs:
                if r.get("executor") == "capture":
                    ok, _dur, code, _out, tail = _run_capture_stdout_impl(
                        r["cmd"], r.get("env") or {}, r.get("timeout"),
                        extra_ld_library_path=r.get("ld"), cwd=r.get("cwd"))
                    if not tail:  # ctest reports on stdout
                        tail = "\n".join((_out or "").splitlines()[-12:])
                else:
                    ok, _dur, code, tail, _mem = _run_cmd_impl(
                        r["cmd"], r.get("env") or {}, r.get("timeout"),
                        extra_ld_library_path=r.get("ld"), cwd=r.get("cwd"),
                        must_contain=r.get("must_contain"))
                if not ok:
                    case_fail = (rep + 1, code, tail)
                    break
            if case_fail:
                break
        if case_fail:
            n_fail += 1
            still.append(case)
            rep, code, tail = case_fail
            print(red(f"  ✗ still fails (repeat {rep}/{reps}, code {code}): {head}"))
            for tl in (tail or "").splitlines()[-8:]:
                print(dim("      " + tl))
        else:
            n_ok += 1
            print(green(f"  ✓ passes x{reps}: {head}"))
    # Keep the previous list, then keep only what still fails (the not
    # replayable entries are kept too: they still need a full run to clear).
    dst = _failed_cases_path(args.errors_log)
    try:
        if os.path.abspath(src) == os.path.abspath(dst):
            os.replace(dst, dst.replace(".jsonl", ".prev.jsonl"))
        with open(dst, "w") as f:
            for case in still + [c for c in cases if not (c.get("runs") or [])]:
                f.write(json.dumps(case) + "\n")
    except OSError:
        pass
    print(bold(f"\n🔁 {n_ok} pass, {n_fail} still fail, {n_skip} not replayable  -> {dst}"))
    return 1 if n_fail else 0


def effective_timeout(base_timeout, per_thread, tc):
    """Timeout for a run at `tc` threads: base + per_thread * tc."""
    return int(base_timeout + per_thread * tc)


def run_with_rebuild_retry(build_root, config, var, stage, exe, env_vars,
                           timeout, extra_ld_library_path,
                           rebuild_retries, rebuild_wait, cwd=None):
    """Run `exe`; if it vanished (concurrent rebuild → code -2), wait, re-resolve
    and retry up to rebuild_retries times. Returns
    (ok, dur, code, err_tail, mem_mo, exe_used)."""
    for attempt in range(rebuild_retries + 1):
        # --mode selects the stage at run time (one executable per variant).
        # Harmless on the vector targets, which still come in two flavours and
        # simply get told the mode they were built for.
        ok, dur, code, err_tail, mem_mo = run_cmd(
            [exe, "--mode", stage], env_vars, timeout, extra_ld_library_path=extra_ld_library_path,
            cwd=cwd
        )
        if code != -2:
            return ok, dur, code, err_tail, mem_mo, exe
        # Binary disappeared — likely a rebuild in progress.
        if attempt < rebuild_retries:
            sys.stdout.write(
                f"\n   ⏳ binary busy/missing (rebuild in progress?), waiting "
                f"{rebuild_wait:.0f}s and re-resolving "
                f"[{attempt + 1}/{rebuild_retries}]...\n"
            )
            sys.stdout.flush()
            time.sleep(rebuild_wait)
            new_exe = resolve_exe(build_root, config, var, stage)
            if new_exe:
                exe = new_exe
    return ok, dur, code, err_tail, mem_mo, exe


def _run_capture_stdout_impl(cmd, env_vars, timeout, extra_ld_library_path=None, cwd=None, merge_stderr=False):
    """Sibling of run_cmd() that returns the FULL stdout instead of parsing
    peak memory / sanitizer hits / must_contain. This phase needs the raw
    CHUNK_STATS text and the checksum line, neither of which run_cmd()
    surfaces to its caller. Kept as a separate function rather than changing
    run_cmd()'s established return shape, which every other phase depends on.

    Returns (success, duration_s, returncode, stdout_text, stderr_tail).
    With merge_stderr, stdout_text is followed by the whole stderr: what the
    library traces there (the re-chunk decider's [RECHUNK-DECIDE] lines) is
    then readable by the caller too.
    """
    start = time.perf_counter()
    _bind = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    child_env = {**_bind, **os.environ, **env_vars}
    if extra_ld_library_path:
        prev = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            extra_ld_library_path + (":" + prev if prev else "")
        )
    cmd = _numa.interleave_prefix(child_env, state._NUMA_MODE) + list(cmd)  # --numa-interleave
    try:
        res = subprocess.run(
            cmd, env=child_env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout,
        )
        err_tail = "\n".join(res.stderr.splitlines()[-30:]) if res.stderr else ""
        _write_full_log(cmd, child_env, res.returncode, res.stdout, res.stderr)
        text = (res.stdout or "") + ("\n" + res.stderr if merge_stderr and res.stderr else "")
        return (res.returncode == 0, time.perf_counter() - start, res.returncode, text, err_tail)
    except subprocess.TimeoutExpired:
        return False, float(timeout), -9, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return False, 0.0, -2, "", f"FileNotFoundError: {e}"
    except OSError as e:
        if getattr(e, "errno", None) == 26:  # ETXTBSY
            return False, 0.0, -2, "", f"ETXTBSY (text file busy): {e}"
        return False, 0.0, -1, "", f"{type(e).__name__}: {e}"
    except Exception as e:
        return False, 0.0, -1, "", f"{type(e).__name__}: {e}"
