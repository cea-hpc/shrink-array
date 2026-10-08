# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Phase 3b: 64-bit index checks against the 32-bit oracles."""
import sys

from ..builds import int64_variant, oracle_complete, oracle_dir, resolve_exe, short_config_label
from ..progress import CONFIG_W, VARIANT_W
from ..runs import _append_error_now, effective_timeout, repro_line, run_cmd
from ..settings import INT64_CHECK_VARIANTS, sanitizer_env_for
from ..term import _hms, bold, cyan, dim, green, red, yellow


def run_int64_check_phase(args, extra_ldpath, errors):
    """64-bit index checks: a few --mode check runs of the int64 binaries.

    INT64_CHECK_VARIANTS, once each, at --check-threads, in every config that
    has both int64 binaries (the release OpenMP ones) and a complete oracle.
    They run in that oracle, read-only, as the 32-bit checks do: the reference
    files are the 32-bit vector_check's, element values that do not depend on
    the index type, so an int64 binary must reproduce them. A failure is a
    32/64-bit divergence -- in the library's index path, or an index
    expression of the bench that overflows at 32 bits.
    """
    print(cyan(bold("\n🔢 64-bit index checks — int64 binaries against the 32-bit oracle")))

    env = {"OMP_NUM_THREADS": str(args.check_threads)}
    to  = effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads)

    n_pass = n_total = 0
    for cfg in args.configs:
        label = short_config_label(cfg)
        exes  = [(var, resolve_exe(args.build_root, cfg, int64_variant(var), "check"))
                 for var in INT64_CHECK_VARIANTS]
        exes  = [(var, exe) for var, exe in exes if exe]
        if not exes:
            continue  # not an int64 config: nothing to say
        if not oracle_complete(args.output_dir, cfg):
            print(yellow(f"   {label}: no complete oracle in {oracle_dir(args.output_dir, cfg)} -- skipped"))
            continue
        for var, exe in exes:
            n_total += 1
            shown = int64_variant(var)
            lead  = (f"{dim(_hms())} {label[:CONFIG_W]:<{CONFIG_W}} | "
                     f"{shown[:VARIANT_W + 6]:<{VARIANT_W + 6}} | ")
            sys.stdout.write("\r" + lead + "running…       ")
            sys.stdout.flush()
            run_env = {**env, **sanitizer_env_for(cfg)}
            run_cwd = oracle_dir(args.output_dir, cfg)
            ok, dur, code, err_tail, _ = run_cmd(
                [exe, "--mode", "check"], run_env, to,
                extra_ld_library_path=extra_ldpath, cwd=run_cwd)
            if ok:
                n_pass += 1
                sys.stdout.write("\r" + lead + green("check OK ✓") + f" ({dur:.1f}s)\n")
            else:
                sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                _entry = (f"INT64_CHECK_FAIL: {exe} --mode check threads={args.check_threads} code={code}\n"
                          f"  (int64 build against the 32-bit oracle of {label})\n"
                          f"  reproduce: {repro_line([exe, '--mode', 'check'], run_env, run_cwd)}\n"
                          f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                errors.append(_entry)
                _append_error_now(args.errors_log, _entry)
            sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no int64 binaries found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   64-bit index checks: {tag}")
    return n_pass, n_total
