"""Phase 2: the timed sweep -- every perf binary over its thread counts."""
import sys
import time

from ..builds import is_compression_only_config, skip_variant_on_config
from ..progress import _prog, get_progress_bar
from ..runs import _append_error_now, effective_timeout, run_with_rebuild_retry
from ..term import bold, cyan, dim, green, red, yellow


def run_perf_phase(args, plan, counter, errors, results):
    """Time every (config, variant, thread count), --reps_perf runs where the
    policy says so, and record (average, peak memory) in results."""
    print(cyan(bold(f"\n🚀 Performance — threads {args.threads}")))
    baseline_times = {}
    baseline_mems  = {}
    for cfg in args.configs:
        # Baseline per (variant, tc): time of the baseline_variant for
        # this tc, on this config. A compression-only config (e.g.
        # -have-aggressive-compression) does NOT re-run vector — its
        # binary is identical to the base config — so it inherits the
        # baseline captured on the preceding base config instead of
        # resetting it.
        if not is_compression_only_config(cfg):
            baseline_times = {}
            baseline_mems  = {}
        for var in args.variants:
            if skip_variant_on_config(var, cfg):
                continue
            exe = plan.perf_exes.get((cfg, var))
            if not exe:
                continue

            for tc in plan.threads_for(var, cfg):
                run_times = []
                run_mems  = []
                _reps = plan.reps_perf_for(var, tc, cfg)
                _timeout = effective_timeout(
                    args.timeout, args.timeout_per_thread, tc)
                for r in range(_reps):
                    counter.idx += 1

                    elapsed = time.time() - counter.start
                    eta_min = ((elapsed / counter.idx) *
                               (counter.total - counter.idx) / 60)
                    sys.stdout.write(
                        f"\r{get_progress_bar(counter.idx, counter.total)} "
                        + _prog("PERF", yellow, cfg, var,
                                th=tc, run=r + 1, reps=_reps,
                                extra=dim(f"ETA:{eta_min:5.1f}min") + "  ")
                    )
                    sys.stdout.flush()

                    ok, dur, code, err_tail, mem_mo, exe = run_with_rebuild_retry(
                        args.build_root, cfg, var, "perf", exe,
                        {"OMP_NUM_THREADS": str(tc)}, _timeout,
                        plan.extra_ldpath,
                        args.rebuild_retries, args.rebuild_wait,
                        cwd=args.output_dir
                    )
                    if ok:
                        run_times.append(dur)
                        if mem_mo is not None:
                            run_mems.append(mem_mo)
                    else:
                        _entry = (
                            f"PERF_ERR: {exe} @ {tc}th run={r+1} code={code}\n"
                            f"  stderr_tail:\n    "
                            + err_tail.replace('\n', '\n    ')
                        )
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)

                if run_times:
                    avg = sum(run_times) / len(run_times)
                    peak_mem = max(run_mems) if run_mems else None
                    results[(cfg, var, tc)] = (avg, peak_mem)
                    mem_str = (f" | {peak_mem:7.0f} Mo"
                               if peak_mem is not None else " |          ")

                    if var == args.baseline_variant:
                        baseline_times[tc] = avg
                        if peak_mem is not None:
                            baseline_mems[tc] = peak_mem
                        sys.stdout.write(
                            f" -> Avg: {avg:8.3f}s{mem_str} (baseline)\n")
                    else:
                        # Time speedup vs vector (>1 = faster).
                        if tc in baseline_times:
                            speedup = baseline_times[tc] / avg
                            _sp = f"speedup: {speedup:.2f}x"
                            speed_str = green(_sp) if speedup >= 1.0 else red(_sp)
                        else:
                            speed_str = None
                        # Memory ratio vs vector (>1 = uses LESS memory =
                        # compression gain). Also show the absolute saving.
                        if peak_mem is not None and tc in baseline_mems \
                                and peak_mem > 0:
                            base_mem = baseline_mems[tc]
                            mem_ratio = base_mem / peak_mem
                            mem_saved = base_mem - peak_mem
                            # Wording matters here: mem_ratio is
                            # base/peak, so 2.89x means the variant uses
                            # 2.89x LESS memory. The old label read
                            # "mem: 2.89x (+56772 Mo vs vector)", whose
                            # "x" and leading "+" both suggest the exact
                            # opposite. Say less/more explicitly and sign
                            # the delta by what actually happened.
                            if mem_ratio >= 1.0:
                                _mem = (
                                    f"mem: {mem_ratio:.2f}x less "
                                    f"(-{mem_saved:.0f} Mo vs {args.baseline_variant})"
                                )
                            else:
                                _mem = (
                                    f"mem: {1.0 / mem_ratio:.2f}x more "
                                    f"(+{-mem_saved:.0f} Mo vs {args.baseline_variant})"
                                )
                            mem_gain_str = green(_mem) if mem_ratio >= 1.0 else yellow(_mem)
                        else:
                            mem_gain_str = None

                        parts = [p for p in (speed_str, mem_gain_str) if p]
                        suffix = f" ({' | '.join(parts)})" if parts else ""
                        sys.stdout.write(
                            f" -> Avg: {bold(f'{avg:8.3f}s')}{mem_str}{suffix}\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f" -> {red(f'⚠️  no valid run out of {_reps}')}\n")
                    sys.stdout.flush()
