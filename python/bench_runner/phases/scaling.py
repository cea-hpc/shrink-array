"""Phase 4: scalability guard (omp-llvm, omp-gnu)."""
import sys

from ..builds import (
    _run_dir, int64_variant, resolve_exe, short_config_label, skip_variant_on_config,
)
from ..progress import CONFIG_W, VARIANT_W
from ..runs import _append_error_now, effective_timeout, run_cmd
from ..settings import (
    SCALING_CONFIGS, SCALING_FALLBACK_DIVISOR, SCALING_FALLBACK_ITER_MAX, SCALING_ITER_MAX,
    SCALING_MAX_SLOWDOWN, SCALING_MULTIPLY, SCALING_REFERENCE_VARIANT, SCALING_THREADS,
    SCALING_TIMEOUT,
)
from ..term import _hms, bold, cyan, dim, green, red, yellow


def run_scaling_phase(args, extra_ldpath, errors):
    """Scalability guard: one perf run per thread count and per variant, on
    omp-llvm and omp-gnu (SCALING_CONFIGS).

    The averaged perf phase answers "how fast", this one answers "does it
    still scale". It runs EVERY perf binary of the sweep (--variants x --sizes,
    as the perf phase does, plus vector as the reference curve even when
    --variants leaves it out) over SCALING_THREADS (4..128) with a shrunk
    workload, prints each curve with the speedup against its first point, and
    FAILS when a point is slower than the BEST point of the same variant so far
    by more than SCALING_MAX_SLOWDOWN (1.25: a single run has spread, the top
    of the curve legitimately plateaus, and on a hybrid machine the points past
    the performance cores are genuinely slower -- a serialised pass shows up as
    1.5x and beyond). Each variant is its own curve: a regression confined to
    one container (a lock only chunk2 takes, a pass only v1 runs) is exactly
    what a single-variant guard would miss.

    Everything it needs is a constant at the top of this file, deliberately:
    seven knobs for a guard that either says "still scales" or names the point
    where it stopped would be seven ways to make its verdict unreadable.
    --skip-scaling turns it off, and it runs in EVERY mode -- most of all in
    --mode fast, which cuts the perf sweep down to 64/128 and would otherwise
    see no scalability regression at all.

    Why these two configs: LLVM is the fastest of the matrix, hence the one
    whose curves are worth believing; GNU is the one without OMPT, whose
    phase-end path differs (see SCALING_CONFIGS). The allocator, aggressive-
    compression and stats configs would add runs, not information.
    The binaries are the int64 ones at --multiply 2 (past INT_MAX elements) and
    --iter_max 500 (see SCALING_MULTIPLY): a REGRESSION signal on the 64-bit
    path at full scale, not a measurement campaign. A config built without
    the int64 binaries falls back to its 32-bit ones at --divisor 50
    --iter_max 200 (SCALING_FALLBACK_*). The thread counts
    the machine cannot honour (more than CHUNK_PMP_MAX_THREADS / the CPUs
    present) are dropped with a note rather than measured against an
    oversubscribed run.
    """
    print(cyan(bold("\n📈 Scalability guard — "
                    + ", ".join(short_config_label(c) for c in SCALING_CONFIGS))))

    # Oversubscription measures the scheduler, not the library: drop the
    # points the machine cannot run one thread per CPU.
    ncpu = args.max_threads
    threads = [t for t in SCALING_THREADS if t <= ncpu]
    dropped = [t for t in SCALING_THREADS if t > ncpu]
    if dropped:
        print(dim(f"   {ncpu} CPUs: thread count(s) {dropped} dropped (would oversubscribe)"))
    if len(threads) < 2:
        print(dim("   (fewer than two usable thread counts -- skipped)"))
        return 0, 0

    args_int64 = ["--mode", "perf", "--no_io",
                  "--multiply", str(SCALING_MULTIPLY),
                  "--iter_max", str(SCALING_ITER_MAX)]
    args_32bit = ["--mode", "perf", "--no_io",
                  "--divisor", str(SCALING_FALLBACK_DIVISOR),
                  "--iter_max", str(SCALING_FALLBACK_ITER_MAX)]

    # vector first: the reference every other curve is read against.
    scaling_vars = [SCALING_REFERENCE_VARIANT] + [v for v in args.variants
                                                  if v != SCALING_REFERENCE_VARIANT]

    n_pass = n_total = 0
    n_curves = 0
    failed_vars = []
    for config in SCALING_CONFIGS:
        label = short_config_label(config)
        if config not in args.configs:
            print(dim(f"   ({label} not in --configs -- skipped)"))
            continue

        candidates = [v for v in scaling_vars if not skip_variant_on_config(v, config)]
        variants   = []
        missing    = []
        for var in candidates:
            exe = resolve_exe(args.build_root, config, int64_variant(var), "perf")
            if exe:
                variants.append((var, exe))
            else:
                missing.append(var)

        if variants:
            bench_args = args_int64
            if missing:
                print(dim(f"   (no int64 binary in {label} for {', '.join(missing)} -- skipped)"))
        else:
            # No int64 binary at all in this config: fall back to its 32-bit
            # ones, on the shrunk workload (SCALING_FALLBACK_*).
            bench_args = args_32bit
            for var in candidates:
                exe = resolve_exe(args.build_root, config, var, "perf")
                if exe:
                    variants.append((var, exe))
                else:
                    print(dim(f"   (no {var} perf binary in {label} -- skipped)"))
            if not variants:
                print(dim(f"   (no perf binary in {label} -- skipped)"))
                continue
            print(yellow(f"   {label}: no int64 binary (built without -DSHRINK_BENCH_INT64=ON) -- "
                         f"32-bit fallback, {' '.join(bench_args[3:])}"))
        is_int64 = bench_args is args_int64
        n_curves += len(variants)

        for var, exe in variants:
            rdir = _run_dir(args.output_dir, "scaling", label, var)
            v_pass = v_total = 0
            best_t = best_dur = None
            first_dur = None
            for tc in threads:
                v_total += 1
                lead = (f"{dim(_hms())} {label[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | {tc:3}th | ")
                sys.stdout.write("\r" + lead + "running…       ")
                sys.stdout.flush()

                ok, dur, code, err_tail, _ = run_cmd(
                    [exe] + bench_args, {"OMP_NUM_THREADS": str(tc)},
                    (max(SCALING_TIMEOUT, effective_timeout(args.timeout, args.timeout_per_thread, tc))
                     if is_int64 else effective_timeout(args.timeout, args.timeout_per_thread, tc)),
                    extra_ld_library_path=extra_ldpath, cwd=rdir)

                if not ok:
                    sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                    _entry = (f"SCALING_RUN_FAIL: {exe} {' '.join(bench_args)} threads={tc} code={code}\n"
                              f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    sys.stdout.flush()
                    continue

                if first_dur is None:
                    first_dur = dur
                speedup = first_dur / dur if dur > 0 else 0.0
                ratio   = (dur / best_dur) if best_dur else 1.0
                regress = best_dur is not None and ratio > SCALING_MAX_SLOWDOWN

                note = f" x{speedup:5.2f} vs {threads[0]}th"
                if best_dur is not None:
                    note += f", {ratio:4.2f}x the best ({best_t}th)"
                if regress:
                    sys.stdout.write("\r" + lead + red(f"{dur:7.2f}s") + red(note) + "\n")
                    _entry = (f"SCALING_REGRESSION: {label} "
                              f"{var} best {best_t}th {best_dur:.2f}s -> {tc}th {dur:.2f}s "
                              f"({ratio:.2f}x, limit {SCALING_MAX_SLOWDOWN})\n"
                              f"  ({' '.join(bench_args)})\n")
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                else:
                    v_pass += 1
                    sys.stdout.write("\r" + lead + bold(f"{dur:7.2f}s") + dim(note) + "\n")
                sys.stdout.flush()
                if best_dur is None or dur < best_dur:
                    best_t, best_dur = tc, dur
            n_pass  += v_pass
            n_total += v_total
            if v_pass != v_total:
                failed_vars.append(f"{label}/{var}")

    if n_total == 0:
        tag = yellow("nothing to run")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} points scale ({n_curves} curves)")
    else:
        tag = red(f"{n_pass}/{n_total} points scale ({n_curves} curves; "
                  f"failing: {', '.join(failed_vars)})")
    print(f"   Scalability: {tag}")
    return n_pass, n_total
