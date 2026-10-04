"""Phase 5: randomised parameter sweep (opt-in, --fuzz-cases)."""
import sys

from ..builds import _variant_base, resolve_exe, short_config_label, skip_variant_on_config
from ..parsing import _parse_last_long_val_sum, _write_pinned_seed
from ..progress import CONFIG_W, VARIANT_W
from ..runs import _append_error_now, effective_timeout, run_cmd
from ..term import _hms, bold, cyan, dim, green, red


# --- Randomised parameter sweep (fuzz) ------------------------------------
# The pools a case is drawn from. Constants, not flags: the point of the phase
# is to reach parameter combinations nobody thought to ask for, and a flag that
# narrows the pool works against that. Widen a pool by editing it here.
FUZZ_SIZES     = ["256", "1024", "dyn"]
FUZZ_DIVISORS  = [5, 10, 20, 50, 100, 200]
FUZZ_ITER_MAX  = [200, 500, 1000, 2000]
FUZZ_THREADS   = [2, 4, 8, 16, 64]


def run_fuzz_phase(args, extra_ldpath, errors):
    """Run each chunk-family variant on RANDOM workload parameters instead of
    the one trajectory every other phase repeats.

    Rationale: the whole suite exercises a single geometry/iteration profile,
    over and over. A defect that only shows up at some other divisor, iteration
    count or thread count is invisible to it no matter how many repetitions are
    run -- which is exactly how a re-chunk teardown crash can sit unnoticed
    while 690 check runs come back green.

    The oracle is the one the bench already trusts: the run must exit 0, must
    not print a sanitizer report, and its integer checksum (long_val_sum) must
    match the SEQUENTIAL run of the same parameters -- with the seed pinned, so
    a difference means behaviour, not scheduling.

    NOT the data_tab_* oracle, hence --no_io. In --mode check a chunk build
    reads the dumps the vector baseline wrote and compares element by element;
    that oracle was produced with the suite's default divisor / iter_max, and
    a case drawn here has other ones, so the arrays cannot match -- every case
    aborted in read_check_binary_file with the same stale checksum on its
    "read" side. Regenerating the dumps per case would cost a baseline run
    each and overwrite the oracle the other phases share; --no_io keeps the
    check-mode workload and skips only the file comparison.

    --fuzz-cases is the only knob. The pools are the FUZZ_* constants above,
    and the master seed comes from the clock and is echoed, so the phase has
    nothing to configure and nothing to get wrong.

    Replay does not need a flag: every case prints its variant, parameters and
    seed, and a failure is logged as the literal command line that produced it.
    """
    import random as _random
    import time as _time

    if args.fuzz_cases <= 0:
        return 0, 0

    # Drawn from the clock, deliberately not settable: a sweep you can pin is a
    # sweep that stops finding new combinations. Replay is per case, below.
    rng = _random.Random(int(_time.time()))
    print(cyan(bold(f"\n🎲 Randomised parameter sweep — {args.fuzz_cases} case(s)")))
    print(dim("   replay a case with the parameters and seed printed on its line"))

    bases = [b for b in ("chunk", "chunk2", "chunk2a")
             if any(_variant_base(v) == b for v in args.variants)]
    if not bases:
        print(dim("   (no chunk-family variant selected -- skipped)"))
        return 0, 0

    n_pass = n_total = 0
    for case in range(1, args.fuzz_cases + 1):
        base    = rng.choice(bases)
        size    = rng.choice(FUZZ_SIZES)
        divisor = rng.choice(FUZZ_DIVISORS)
        itmax   = rng.choice(FUZZ_ITER_MAX)
        threads = rng.choice(args.fuzz_threads)
        seed    = rng.randrange(1, 2 ** 31 - 1)
        var     = f"{base}@{size}"

        cfg = next((c for c in args.configs
                    if not skip_variant_on_config(var, c)
                    and resolve_exe(args.build_root, c, var, "check")), None)
        if cfg is None:
            continue

        exe = resolve_exe(args.build_root, cfg, var, "check")
        to  = effective_timeout(args.timeout, args.timeout_per_thread, threads)
        bench_args = ["--mode", "check", "--no_io",
                      "--divisor", str(divisor), "--iter_max", str(itmax)]

        lead = (f"{dim(_hms())} [{case:3}/{args.fuzz_cases}] "
                f"{short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                f"{var[:VARIANT_W]:<{VARIANT_W}} | div={divisor:<4} it={itmax:<5} "
                f"{threads:3}th seed={seed:<11} | ")
        n_total += 1

        # Sequential reference first, then the threaded run, same pinned seed.
        _write_pinned_seed(args.output_dir, itmax, seed)
        sys.stdout.write("\r" + lead + "seq…      ")
        sys.stdout.flush()
        ok_seq, _, code_seq, tail_seq, _ = run_cmd(
            [exe] + bench_args, {"OMP_NUM_THREADS": "1"}, to,
            extra_ld_library_path=extra_ldpath, cwd=args.output_dir)

        ref = _parse_last_long_val_sum(tail_seq) if ok_seq else None

        _write_pinned_seed(args.output_dir, itmax, seed)
        sys.stdout.write("\r" + lead + f"{threads}th…     ")
        sys.stdout.flush()
        ok_omp, dur, code_omp, tail_omp, _ = run_cmd(
            [exe] + bench_args, {"OMP_NUM_THREADS": str(threads)}, to,
            extra_ld_library_path=extra_ldpath, cwd=args.output_dir)

        got = _parse_last_long_val_sum(tail_omp) if ok_omp else None

        if not ok_seq or not ok_omp:
            which = "sequential" if not ok_seq else f"{threads}-thread"
            code  = code_seq if not ok_seq else code_omp
            tail  = tail_seq if not ok_seq else tail_omp
            sys.stdout.write("\r" + lead + red(f"FAIL ({which}, code {code})") + "\n")
            _entry = (f"FUZZ_FAIL: {exe} {' '.join(bench_args)} "
                      f"OMP_NUM_THREADS={threads} seed={seed} ({which}, code={code})\n"
                      f"  stderr_tail:\n    " + tail.replace('\n', '\n    '))
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        elif ref is not None and got is not None and ref != got:
            sys.stdout.write("\r" + lead + red("CHECKSUM MISMATCH") + "\n")
            _entry = (f"FUZZ_CHECKSUM: {exe} {' '.join(bench_args)} seed={seed}\n"
                      f"  sequential long_val_sum = {ref}\n"
                      f"  {threads}-thread  long_val_sum = {got}")
            errors.append(_entry)
            _append_error_now(args.errors_log, _entry)
        else:
            n_pass += 1
            sys.stdout.write("\r" + lead + green("ok") + f" ({dur:.1f}s)\n")
        sys.stdout.flush()

    print(f"   {n_pass}/{n_total} case(s) passed")
    return n_pass, n_total
