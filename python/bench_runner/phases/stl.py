# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 (2c): the STL interface self-check, once per compiler front-end."""
import sys

from ..progress import _prog, get_progress_bar
from ..runs import _append_error_now, effective_timeout, run_cmd
from ..term import cyan, green, red


# STL self-check: one run per gnu/llvm front-end present (phase 2c).
# One STL self-check per compiler front-end (gnu/llvm). The check is
# functional and config-agnostic, so prefer the plain config name but fall
# back to ANY config of that front-end that has a chunk2a/chunk check exe —
# otherwise a build set with only suffixed configs (e.g. -std-alloc) reports
# 0 STL runs and silently skips the check.
def stl_exe_for(check_exes, config):
    """A chunk2a (else chunk) check binary of `config`, whatever its size.

    check_exes is keyed by the EXPANDED variant -- "chunk2a@dyn", "chunk@256"
    -- so a lookup by BARE variant name never matches (and the phase would
    silently report "0 STL"). The check is functional and size-agnostic, so any
    declination will do; chunk2a first because it exercises the widest surface.
    """
    for base in ("chunk2a", "chunk"):
        for (c, var), exe in check_exes.items():
            if c == config and (var == base or var.startswith(base + "@")):
                return exe
    return None

def stl_representative_exes(args, check_exes):
    reps = {}
    for c in args.configs:
        if "omp-gnu" in c:
            fe = "gnu"
        elif "omp-llvm" in c:
            fe = "llvm"
        else:
            continue
        exe = stl_exe_for(check_exes, c)
        if not exe:
            continue
        is_plain = c in ("cmake-build-omp-gnu", "cmake-build-omp-llvm")
        if fe not in reps or is_plain:
            reps[fe] = (c, exe)
    # Fallback: if no gnu/llvm-named config is built (e.g. fast mode on an arch
    # where only fast_alloc configs are present, whose dir names don't carry
    # the omp-gnu/omp-llvm marker), the STL self-check would silently be
    # skipped (0 STL). It is a functional, config-agnostic test, so run it once
    # on the first present config that has a chunk2a/chunk check exe.
    if not reps:
        for c in args.configs:
            exe = stl_exe_for(check_exes, c)
            if exe:
                reps["any"] = (c, exe)
                break
    return list(reps.values())


def run_stl_phase(args, plan, counter, errors):
    """--test_stl --no_io on one representative check binary per front-end."""
    # 2c. STL interface self-checks: functional, thread-independent, so
    #     run ONCE per (compiler) front-end, not once per matrix cell.
    #     Uses --test_stl --no_io on the check binary; pass/fail read
    #     from the run's exit code (the section prints ALL PASSED / N
    #     FAILURE(S) and exits non-zero on failure).
    for cfg, exe in stl_representative_exes(args, plan.check_exes):
        counter.idx += 1
        sys.stdout.write(
            f"\r{get_progress_bar(counter.idx, counter.total)} "
            + _prog("STL", cyan, cfg,
                    th=args.sanitizer_threads,
                    extra="--test_stl --no_io ")
        )
        sys.stdout.flush()
        ok, _, code, err_tail, _ = run_cmd(
            [exe, "--test_stl", "--no_io"],
            {"OMP_NUM_THREADS": str(args.sanitizer_threads)},
            effective_timeout(args.timeout, args.timeout_per_thread, args.sanitizer_threads),
            extra_ld_library_path=plan.extra_ldpath, cwd=args.output_dir
        )
        if ok:
            sys.stdout.write(" -> " + green("PASS") + "\n")
        else:
            sys.stdout.write(" -> " + red(f"FAIL (code {code})") + "\n")
            _entry = (f"STL_TEST_FAIL: {exe} --test_stl --no_io code={code}\n"
                      f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        sys.stdout.flush()
