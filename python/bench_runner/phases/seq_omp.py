# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Phase 1b: seed-pinned sequential-vs-OpenMP compression-classification compare."""
import sys

from ..builds import (
    find_llvm_seq_omp_pairs, resolve_exe, short_config_label, skip_variant_on_config,
)
from ..parsing import (
    _parse_applied_rechunks, _parse_last_long_val_sum, _parse_still_at_exit, _seed_was_pinned,
    _write_pinned_seed,
)
from ..progress import CONFIG_W, VARIANT_W
from ..runs import _append_error_now, effective_timeout, run_capture_stdout
from ..settings import is_stats_config
from ..term import _hms, bold, cyan, dim, green, red, yellow


def run_seq_omp_compare_phase(args, extra_ldpath, errors):
    """For each LLVM seq/OMP config pair, each requested chunk-family variant
    and size, and each requested OMP thread count: run the SEQUENTIAL binary
    once and the OMP binary once per thread count, all with the SAME PINNED
    SEED, and compare `still compressed (at exit)`. With the seed pinned,
    this number is a function of the array's final CONTENT alone (verified
    via long_val_sum) -- it must be identical across every mode and thread
    count. A mismatch is a real, actionable signal, not scheduling noise.

    Except where the geometry itself differs: a dyn run whose re-chunk
    decider applied other re-chunks than the sequential run ends with other
    chunk counts on the same data. Those runs are reported "not comparable"
    (the first differing decision is printed) and are not errors; a dyn run
    with the same decisions is compared like a static size.
    """
    print(cyan(bold("\n🌱 Seq vs OMP compression-classification compare "
                    f"(seed={args.seq_omp_seed}, threads={args.seq_omp_threads})")))

    if args.seq_omp_seed == 0:
        print(red("   --seq-omp-seed=0 is invalid: the bench treats a 0 seed "
                  "as \"generate a random one\", which defeats this phase's "
                  "whole point. Skipped."))
        return 0, 0

    pairs = find_llvm_seq_omp_pairs(args.build_root, args.configs)
    if not pairs:
        print(dim("   (no seq/omp config pair present -- skipped)"))
        return 0, 0

    # The compare reads `still compressed (at exit)` out of the CHUNK_STATS
    # block, which ONLY a -DCHUNK_STATS=ON build prints. Whether a pair is a
    # stats build is knowable from its name (is_stats_config), so drop the
    # others here rather than run a full sequential bench per pair whose only
    # possible outcome is "no CHUNK_STATS block in the sequential output -- not
    # a stats build" (one pointless multi-minute run per non-stats pair).
    non_stats = [(o, q) for o, q in pairs
                 if not (is_stats_config(o) and is_stats_config(q))]
    pairs = [(o, q) for o, q in pairs
             if is_stats_config(o) and is_stats_config(q)]
    if non_stats:
        print(dim("   seq/omp compare: not a CHUNK_STATS build, nothing to compare for "
                  + ", ".join(short_config_label(o) for o, _ in non_stats)
                  + " -- skipped"))
    if not pairs:
        print("   Seq vs OMP compare: "
              + yellow(f"nothing compared: {len(non_stats)} pair(s) without CHUNK_STATS "
                       "-- build the -stats pair (-DCHUNK_STATS=ON, OMP on and off)"))
        return 0, 0

    bench_args = ["--mode", "perf",
                  "--divisor", str(args.seq_omp_divisor),
                  "--iter_max", str(args.seq_omp_iter_max)]
    to = effective_timeout(args.timeout, args.timeout_per_thread,
                           max(args.seq_omp_threads))

    n_pass = n_total = 0
    n_pairs_skipped = 0
    n_diverged = 0  # dyn runs whose re-chunk decisions differed: not comparable
    for omp_cfg, seq_cfg in pairs:
        # The compare reads CHUNK_STATS. A pair whose binaries are not stats
        # builds has nothing to compare: found out on its FIRST sequential run
        # and skipped as a whole, rather than running every variant, size and
        # thread count to print "no still@exit data" each time.
        pair_skipped = False
        for base in args.seq_omp_variants:
            if pair_skipped:
                break
            for size in args.seq_omp_sizes:
                if pair_skipped:
                    break
                var = f"{base}@{size}"
                if skip_variant_on_config(var, omp_cfg) or \
                   skip_variant_on_config(var, seq_cfg):
                    continue
                seq_exe = resolve_exe(args.build_root, seq_cfg, var, "perf")
                omp_exe = resolve_exe(args.build_root, omp_cfg, var, "perf")
                if not seq_exe or not omp_exe:
                    continue

                lead = (f"{dim(_hms())} {short_config_label(omp_cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | ")

                # --- sequential reference run -----------------------------
                _write_pinned_seed(args.output_dir, args.seq_omp_iter_max,
                                   args.seq_omp_seed)
                sys.stdout.write("\r" + lead + "sequential…          ")
                sys.stdout.flush()
                ok, _dur, code, out, err_tail = run_capture_stdout(
                    [seq_exe] + bench_args, {}, to,
                    extra_ld_library_path=extra_ldpath, cwd=args.output_dir, merge_stderr=True,
                )
                if not ok:
                    sys.stdout.write("\r" + lead + red("sequential FAIL") +
                                     f" (code {code})\n")
                    _entry = (f"SEQ_OMP_COMPARE_RUN_FAIL: {seq_exe} "
                             f"{' '.join(bench_args)} code={code}\n"
                             f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    continue
                if not _seed_was_pinned(out):
                    _entry = (f"SEQ_OMP_COMPARE_SEED_FALLBACK: {seq_exe} did not use "
                             f"the pinned seed (seed{args.seq_omp_iter_max}.txt failed "
                             f"to load) -- comparison skipped for {var} on "
                             f"{short_config_label(omp_cfg)}")
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                    continue
                seq_still = _parse_still_at_exit(out, base)
                seq_checksum = _parse_last_long_val_sum(out)
                seq_rechunks = _parse_applied_rechunks(out)
                if seq_still is None:
                    sys.stdout.write("\r" + lead +
                                     yellow("no CHUNK_STATS block in the sequential output "
                                            "-- not a stats build, pair skipped") + "\n")
                    pair_skipped = True
                    n_pairs_skipped += 1
                    break

                # --- one OMP run per requested thread count ----------------
                for tc in args.seq_omp_threads:
                    _write_pinned_seed(args.output_dir, args.seq_omp_iter_max,
                                       args.seq_omp_seed)
                    sys.stdout.write("\r" + lead + f"omp {tc:>3}th…            ")
                    sys.stdout.flush()
                    ok, _dur, code, out, err_tail = run_capture_stdout(
                        [omp_exe] + bench_args, {"OMP_NUM_THREADS": str(tc)}, to,
                        extra_ld_library_path=extra_ldpath, cwd=args.output_dir, merge_stderr=True,
                    )
                    n_total += 1
                    if not ok:
                        sys.stdout.write("\r" + lead + red(f"omp {tc}th FAIL") +
                                         f" (code {code})\n")
                        _entry = (f"SEQ_OMP_COMPARE_RUN_FAIL: {omp_exe} "
                                 f"{' '.join(bench_args)} threads={tc} code={code}\n"
                                 f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    '))
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue
                    if not _seed_was_pinned(out):
                        _entry = (f"SEQ_OMP_COMPARE_SEED_FALLBACK: {omp_exe} @ {tc}th "
                                 f"did not use the pinned seed -- comparison skipped")
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue

                    omp_checksum = _parse_last_long_val_sum(out)
                    if seq_checksum is not None and omp_checksum is not None \
                            and seq_checksum != omp_checksum:
                        sys.stdout.write("\r" + lead +
                                         red(f"omp {tc}th CONTENT MISMATCH") + "\n")
                        _entry = (
                            f"SEQ_OMP_COMPARE_CONTENT_MISMATCH: {var} on "
                            f"{short_config_label(omp_cfg)} @ {tc}th -- long_val_sum "
                            f"differs from sequential DESPITE the pinned seed "
                            f"(seq={seq_checksum} omp={omp_checksum}). The array's own "
                            f"final content differs by thread count; this is a "
                            f"correctness question upstream of chunk-compression "
                            f"bookkeeping."
                        )
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                        continue

                    # A dyn run's geometry is the re-chunk decider's to choose, and
                    # its marginal calls (a gain a hair above or below the
                    # deadband) read the live compression state -- which differs
                    # by construction between the builds: zero_one_value/zero_any compress
                    # by deferred block passes in OpenMP only. Two runs that
                    # re-chunked differently end with different chunk counts on
                    # the same data; their still@exit says nothing about either.
                    # Not comparable, said so, not an error. Same decisions: the
                    # counts must agree, and are compared as for a static size.
                    omp_rechunks = _parse_applied_rechunks(out)
                    if omp_rechunks != seq_rechunks:
                        n_diverged += 1
                        first = next((k for k, (a, b) in enumerate(zip(seq_rechunks, omp_rechunks)) if a != b),
                                     min(len(seq_rechunks), len(omp_rechunks)))
                        def _fmt(lst):
                            return f"{lst[first][0]} {lst[first][1]}->{lst[first][2]}" if first < len(lst) else "none"
                        sys.stdout.write(
                            "\r" + lead +
                            yellow(f"omp {tc:>3}th re-chunk geometry diverged ({len(seq_rechunks)} vs "
                                   f"{len(omp_rechunks)} applied; #{first + 1}: seq {_fmt(seq_rechunks)}, "
                                   f"omp {_fmt(omp_rechunks)}) -- not comparable") + "\n")
                        sys.stdout.flush()
                        continue

                    omp_still = _parse_still_at_exit(out, base)
                    if seq_still is None or omp_still is None:
                        sys.stdout.write("\r" + lead +
                                         yellow(f"omp {tc}th no still@exit data") + "\n")
                        continue

                    if seq_still == omp_still:
                        n_pass += 1
                        sys.stdout.write(
                            "\r" + lead +
                            green(f"omp {tc:>3}th still@exit={omp_still} == seq") + "\n")
                    else:
                        sys.stdout.write(
                            "\r" + lead +
                            red(f"omp {tc:>3}th still@exit={omp_still} != "
                                f"seq({seq_still})") + "\n")
                        _entry = (
                            f"SEQ_OMP_COMPARE_MISMATCH: {var} on "
                            f"{short_config_label(omp_cfg)} @ {tc}th -- still compressed "
                            f"(at exit): seq={seq_still} omp={omp_still} "
                            f"(seed={args.seq_omp_seed} pinned, long_val_sum matched -- "
                            f"same data, different classification)"
                        )
                        errors.append(_entry)
                        _append_error_now(args.errors_log, _entry)
                    sys.stdout.flush()

    if n_total == 0 and n_pairs_skipped:
        tag = yellow(f"nothing compared: {n_pairs_skipped} pair(s) without CHUNK_STATS "
                     "-- build the -stats pair (-DCHUNK_STATS=ON, OMP on and off)")
    elif n_total == 0:
        tag = yellow("no runs (missing binaries?)")
    elif n_pass + n_diverged == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    if n_diverged:
        tag += yellow(f", {n_diverged} not comparable (re-chunk geometry diverged)")
    print(f"   Seq vs OMP compare: {tag}")
    return n_pass, n_total
