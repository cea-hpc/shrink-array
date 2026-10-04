"""Phase 1 (2d): the sanitizer checks (ASan with std-alloc, TSan)."""
import sys

from ..builds import _variant_base, config_dir_present, resolve_exe, skip_variant_on_config
from ..progress import _prog, get_progress_bar
from ..runs import _append_error_now, effective_timeout, run_cmd
from ..settings import SANITIZER_CONFIGS, SANITIZER_REPS, sanitizer_env_for
from ..term import cyan, green, red


def run_sanitizer_phase(args, plan, counter, errors):
    """Every chunk-family check binary of the sanitizer configs, at the
    sanitizer thread count, SANITIZER_REPS run(s) each."""
    # 2d. Sanitizer CHECK: ASan (std-alloc) + TSan, at a fixed low thread
    #     count, SANITIZER_REPS run(s). Diagnostic, not timed. A run FAILS on a
    #     non-zero exit OR any sanitizer report in the output (run_cmd
    #     detects the latter and flips success to False).
    for cfg in SANITIZER_CONFIGS:
        if not config_dir_present(args.build_root, cfg):
            continue
        san_env = sanitizer_env_for(cfg)
        for var in args.variants:
            if skip_variant_on_config(var, cfg):
                continue
            # The baseline (vector) holds no chunk code at all, so a
            # sanitizer run on it can only ever exercise libstdc++.
            # At 5-20x slowdown x SANITIZER_REPS that is pure cost
            # for zero signal -- skip it.
            if _variant_base(var) == args.baseline_variant:
                continue
            exe = resolve_exe(args.build_root, cfg, var, "check")
            if not exe:
                continue
            ok_in_row = 0
            for r in range(SANITIZER_REPS):
                counter.idx += 1
                sys.stdout.write(
                    f"\r{get_progress_bar(counter.idx, counter.total)} "
                    + _prog("SAN", cyan, cfg, var,
                            th=args.sanitizer_threads, run=r + 1,
                            reps=SANITIZER_REPS,
                            extra=f"OK:{ok_in_row:3} ")
                )
                sys.stdout.flush()
                ok, _, code, err_tail, _ = run_cmd(
                    [exe, "--no_io"],
                    {"OMP_NUM_THREADS": str(args.sanitizer_threads), **san_env},
                    effective_timeout(args.timeout, args.timeout_per_thread,
                                      args.sanitizer_threads),
                    extra_ld_library_path=plan.extra_ldpath, cwd=args.output_dir
                )
                if ok:
                    ok_in_row += 1
                else:
                    _entry = (
                        f"SANITIZER_FAIL: {exe} --no_io "
                        f"({'TSan' if 'thread' in cfg else 'ASan'}) "
                        f"iter={r+1} code={code} threads={args.sanitizer_threads}\n"
                        f"  report:\n    " + err_tail.replace('\n', '\n    ')
                    )
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
            verdict = f"{ok_in_row}/{SANITIZER_REPS}"
            tag = green(f"{verdict} clean") if ok_in_row == SANITIZER_REPS \
                else red(f"{verdict} HIT")
            sys.stdout.write(f" -> {tag}\n")
            sys.stdout.flush()
