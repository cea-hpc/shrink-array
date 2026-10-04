"""Phase 0b: the validation harnesses (ctest -L validation) of every build dir."""
import sys
import re as _re

from ..builds import resolve_build_dir, short_config_label, validation_built
from ..progress import CONFIG_W
from ..runs import _append_error_now, effective_timeout, run_capture_stdout
from ..settings import is_sanitizer_config, sanitizer_env_for
from ..term import _hms, bold, cyan, dim, green, red, yellow


_CTEST_SUMMARY_RE = _re.compile(
    r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")


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
    level). ctest's own per-test timeout applies; the phase timeout is the
    bench one scaled by the test count. A config with no build directory is
    noted once and skipped, and so is one whose configuration does not build
    the harnesses: only Debug, CHECK_OPTIM and sanitizer builds compile them
    (examples/CMakeLists.txt, CHUNK_BUILD_VALIDATION).
    """
    print(cyan(bold(f"\n🧪 Validation harnesses — ctest -L {args.ctest_label}")))

    n_pass = n_total = 0
    for cfg in args.configs:
        bdir = resolve_build_dir(args.build_root, cfg)
        lead = f"{dim(_hms())} {short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
        if not bdir:
            print(lead + dim("no CTestTestfile.cmake (not a CMake build dir) — skipped"))
            continue
        if not validation_built(bdir):
            print(lead + dim("validation not built here (Debug, CHECK_OPTIM and sanitizer builds only) — skipped"))
            continue

        cmd = ["ctest", "-L", args.ctest_label, "--output-on-failure",
               "-j", str(args.ctest_jobs)]
        if args.ctest_regex:
            cmd += ["-R", args.ctest_regex]
        # The harnesses set OMP_NUM_THREADS themselves (test property); only
        # the sanitizer options need to reach them.
        env = sanitizer_env_for(cfg) if is_sanitizer_config(cfg) else {}
        to  = effective_timeout(args.timeout, args.timeout_per_thread, 8) * 4

        sys.stdout.write("\r" + lead + "running…       ")
        sys.stdout.flush()
        ok, dur, code, out, err_tail = run_capture_stdout(
            cmd, env, to, extra_ld_library_path=extra_ldpath, cwd=bdir)

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
                             f" ({dur:.1f}s)\n")
        elif total == 0 and ok:
            sys.stdout.write("\r" + lead + yellow("no test carries the label") + "\n")
        else:
            sys.stdout.write("\r" + lead + red(f"{passed}/{total} passed") +
                             f" (ctest code {code}, {dur:.1f}s)\n")
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
        sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no validation test found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Validation harnesses: {tag}")
    return n_pass, n_total
