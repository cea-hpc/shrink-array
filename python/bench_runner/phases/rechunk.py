# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Phase 3: adaptive re-chunk feature test (dyn)."""
import sys

from ..builds import (
    _run_dir, _variant_base, oracle_complete, oracle_dir, resolve_exe, short_config_label,
    skip_variant_on_config,
)
from ..progress import CONFIG_W, VARIANT_W
from ..runs import _append_error_now, effective_timeout, repro_line, run_cmd
from ..settings import sanitizer_env_for
from ..term import _hms, bold, cyan, dim, green, red, yellow


def run_rechunk_phase(args, extra_ldpath, errors):
    """Feature test for the runtime adaptive re-chunk (dynamic builds).

    For every chunk-family variant it runs the _dyn binary of BOTH stages
    (perf and check) with the advisor enabled and asserts that a re-chunk is
    actually applied ([APPLIED]) with no crash / sanitizer hit — i.e. the
    advisor -> decider -> rechunk_to pipeline works end to end in the live loop.

      * no --compress            : quiescent between-phase path (no in-loop compr).
      * CHUNK_ADVISOR_PERIOD=4    : tick often so a verdict lands within the run.
      * CHUNK_COARSEN_MIN_META_MB : the floor of a coarsening's metadata gain;
                                    0 here so the move fires on the shrunk
                                    arrays this test uses.
      * CHUNK_COARSEN_MIN_META_RATE : its other gate, MB per second of
                                    estimated rebuild; scale-free, so it would
                                    pass anyway, but 0 keeps the test about
                                    the pipeline, not the policy.
      * CHUNK_ADVISOR_MIN_GAIN_MB : the absolute deadband (8 MB by policy); 0
                                    here for the same reason -- at --divisor 200
                                    the 64 -> 128 metadata gain is ~0.7 MB, and
                                    the deadband alone refused every tick.
      * check stage              : --mode check reads the oracle (data_<tab>_<k>
                                   .bin, written once by the baseline run of
                                   the correctness phase in output_dir), so it
                                   runs THERE, read-only, and is skipped with a
                                   notice when the oracle is absent (--skip-check,
                                   or a failed baseline). Not a directory of
                                   its own: there the binary would abort on the
                                   first missing file.
      * perf stage               : its own cwd (nothing read, nothing kept);
                                   --divisor/--iter_max shrink the huge perf data
                                   so the feature test stays quick (it validates
                                   the feature under -DPERF, not perf numbers).
    """
    print(cyan(bold("\n🔁 Adaptive re-chunk feature test — dyn binaries (perf + check)")))

    bases = []
    for v in args.variants:
        b = _variant_base(v)
        if b in ("chunk", "chunk2", "chunk2a") and b not in bases:
            bases.append(b)
    if not bases:
        print(dim("   (no chunk-family variants selected — skipped)"))
        return 0, 0

    stage_extra_args = {
        "check": ["--mode", "check"],
        "perf":  ["--mode", "perf", "--divisor", "200", "--iter_max", "40"],
    }
    env_common = {
        "OMP_NUM_THREADS":      str(args.check_threads),
        "CHUNK_ADVISOR_PERIOD": "4",
        # The gates of a coarsening, off: the metadata floor and rate, and the
        # absolute deadband of the data branch (the ~0.7 MB gain of the
        # arrays --divisor leaves must fire).
        "CHUNK_COARSEN_MIN_META_MB": "0",
        "CHUNK_COARSEN_MIN_META_RATE": "0",
        "CHUNK_ADVISOR_MIN_GAIN_MB": "0",
    }

    def _oracle_ok(cfg):
        return oracle_complete(args.output_dir, cfg)

    n_pass = n_total = 0
    for cfg in args.configs:
        san = sanitizer_env_for(cfg)
        to  = effective_timeout(args.timeout, args.timeout_per_thread, args.check_threads)
        if not _oracle_ok(cfg):
            print(yellow(f"   {short_config_label(cfg)}: no complete oracle in "
                         f"{oracle_dir(args.output_dir, cfg)}: check stage skipped, perf only"))
        for base in bases:
            var = f"{base}@dyn"
            if skip_variant_on_config(var, cfg):
                continue
            for stage in ("check", "perf"):
                if stage == "check" and not _oracle_ok(cfg):
                    continue
                exe = resolve_exe(args.build_root, cfg, var, stage)
                if not exe:
                    continue
                n_total += 1
                lead = (f"{dim(_hms())} {short_config_label(cfg)[:CONFIG_W]:<{CONFIG_W}} | "
                        f"{var[:VARIANT_W]:<{VARIANT_W}} | {stage:5} | ")
                sys.stdout.write("\r" + lead + "running…       ")
                sys.stdout.flush()
                # check reads the config's oracle where the correctness phase
                # wrote it (a shrink binary never writes reference files:
                # io_res is 2, read-only, in every chunk build); perf touches
                # no file and gets a directory of its own.
                rdir = (oracle_dir(args.output_dir, cfg) if stage == "check"
                        else _run_dir(args.output_dir, "rechunk", short_config_label(cfg), var, stage))
                run_env = {**env_common, **san}
                ok, dur, code, err_tail, _ = run_cmd(
                    [exe] + stage_extra_args[stage],
                    run_env, to,
                    extra_ld_library_path=extra_ldpath, cwd=rdir,
                    # "[APPLIED]", not "[RECHUNK-APPLY]": the library prints
                    # "[RECHUNK-DECIDE] <array>: re-chunk S=.. -> S'=.. (...) [APPLIED]".
                    # The old marker existed nowhere in the tree, so this test
                    # reported FAIL on a process that had exited 0.
                    must_contain="[APPLIED]",
                )
                if ok:
                    n_pass += 1
                    sys.stdout.write("\r" + lead + green("re-chunk applied ✓") +
                                     f" ({dur:.1f}s)\n")
                else:
                    sys.stdout.write("\r" + lead + red("FAIL") + f" (code {code})\n")
                    _entry = (
                        f"RECHUNK_FEATURE_FAIL: {exe} "
                        f"{' '.join(stage_extra_args[stage])} stage={stage} code={code}\n"
                        f"  (expected '[APPLIED]'; env CHUNK_ADVISOR_PERIOD=4 "
                        f"CHUNK_COARSEN_MIN_META_MB=0 CHUNK_COARSEN_MIN_META_RATE=0 CHUNK_ADVISOR_MIN_GAIN_MB=0)\n"
                        f"  reproduce: {repro_line([exe] + stage_extra_args[stage], run_env, rdir)}\n"
                        f"  stderr_tail:\n    " + err_tail.replace('\n', '\n    ')
                    )
                    errors.append(_entry)
                    _append_error_now(args.errors_log, _entry)
                sys.stdout.flush()

    if n_total == 0:
        tag = yellow("no _dyn binaries found")
    elif n_pass == n_total:
        tag = green(f"{n_pass}/{n_total} passed")
    else:
        tag = red(f"{n_pass}/{n_total} passed")
    print(f"   Adaptive re-chunk feature: {tag}")
    return n_pass, n_total
