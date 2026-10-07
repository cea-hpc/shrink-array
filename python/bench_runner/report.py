# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""End-of-run summary: CSV, error log, memory guards."""
import csv

from .builds import _variant_base, short_config_label
from .settings import DEFAULT_BASELINE
from .term import blue, bold, green, red, yellow


def _memory_guard_errors(norm, baseline_variant,
                         chunk2a_vs_chunk2_max, vs_baseline_max):
    """Turn the memory figures flush_summary already computes into verdicts.

    norm maps (cfg, var, tc) -> (avg_time_s, peak_mem_mo_or_None).

    Two guards, both on the same (cfg, tc) and same size declination so the
    runs processed identical data:

    1. chunk2a vs chunk2: by design Chunk2A must never cost MORE memory than
       Chunk2 (it is Chunk2 plus a second constant). A ratio above
       `chunk2a_vs_chunk2_max` is the exact signature of the Sept-2026
       regression (x2.7 at 64 threads), reported as an error.
    2. any chunk variant vs the baseline (default `vector`): a chunk build
       that uses more than `vs_baseline_max` x the plain-vector memory has lost
       its reason to exist. Off when `vs_baseline_max` is None.

    Returns a list of error strings (possibly empty)."""
    errs = []
    if not norm:
        return errs

    def _size_of(var):
        return var.split("@", 1)[1] if "@" in var else ""

    # (cfg, tc, size) -> mem, per bare variant
    by_key = {}
    for (cfg, var, tc), (_avg, mem) in norm.items():
        if mem is None or mem <= 0:
            continue
        by_key[(cfg, tc, _size_of(var), _variant_base(var))] = mem

    if chunk2a_vs_chunk2_max is not None:
        for (cfg, tc, sz, base), mem in by_key.items():
            if base != "chunk2a":
                continue
            ref = by_key.get((cfg, tc, sz, "chunk2"))
            if ref is None:
                continue
            ratio = mem / ref
            if ratio > chunk2a_vs_chunk2_max:
                errs.append(
                    f"MEMORY REGRESSION [chunk2a vs chunk2] "
                    f"cfg={short_config_label(cfg)} threads={tc} size={sz or '-'}: "
                    f"chunk2a={mem:.0f} Mo, chunk2={ref:.0f} Mo, "
                    f"ratio={ratio:.2f} > {chunk2a_vs_chunk2_max}"
                )

    if vs_baseline_max is not None:
        for (cfg, tc, sz, base), mem in by_key.items():
            if base == baseline_variant or not base.startswith("chunk"):
                continue
            # The baseline (vector) has no size declination: match on (cfg, tc)
            # with any size.
            refs = [m for (c, t, _s, b), m in by_key.items()
                    if c == cfg and t == tc and b == baseline_variant]
            if not refs:
                continue
            ref = refs[0]
            ratio = mem / ref
            if ratio > vs_baseline_max:
                errs.append(
                    f"MEMORY REGRESSION [{base} vs {baseline_variant}] "
                    f"cfg={short_config_label(cfg)} threads={tc} size={sz or '-'}: "
                    f"{base}={mem:.0f} Mo, {baseline_variant}={ref:.0f} Mo, "
                    f"ratio={ratio:.2f} > {vs_baseline_max}"
                )
    return errs


def flush_summary(results, errors, summary_csv, errors_log, elapsed_min,
                  interrupted, baseline_variant=DEFAULT_BASELINE,
                  mem_chunk2a_vs_chunk2_max=None, mem_vs_baseline_max=None):
    """Write CSV + log — called even when interrupted.

    results maps (cfg, var, tc) -> (avg_time_s, peak_mem_mo_or_None).
    The CSV also carries, for non-baseline variants, the time speedup and the
    memory ratio vs the baseline variant (same cfg + thread count).

    Memory guards (see _memory_guard_errors) are evaluated here, once every
    run is in, and appended to `errors` so they count as real failures.
    """
    label = "interrupted" if interrupted else "finished"
    done_dot = blue("🔵") if not interrupted else yellow("🟡")
    print(bold(f"\n{done_dot} Test {label} in {elapsed_min:.2f} minutes."))

    # Normalise every entry to (avg, mem) and index baselines by (cfg, tc).
    norm = {}
    base_time = {}   # (cfg, tc) -> avg
    base_mem  = {}   # (cfg, tc) -> mem
    for (cfg, var, tc), val in results.items():
        avg, mem = val if isinstance(val, tuple) else (val, None)
        norm[(cfg, var, tc)] = (avg, mem)
        if var == baseline_variant:
            base_time[(cfg, tc)] = avg
            if mem is not None:
                base_mem[(cfg, tc)] = mem

    # Memory verdicts: computed from the same figures the CSV reports, but a
    # bad ratio is a FAIL, not a column in a spreadsheet.
    mem_errs = _memory_guard_errors(norm, baseline_variant,
                                    mem_chunk2a_vs_chunk2_max, mem_vs_baseline_max)
    if mem_errs:
        for e in mem_errs:
            print(red("⚠️  " + e))
        errors.extend(mem_errs)

    if errors:
        print(red(f"⚠️  {len(errors)} error(s) detected — see {errors_log}"))
        with open(errors_log, "w") as f:
            f.write("\n\n".join(errors))
    else:
        print(green("✅ No crash or error detected."))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Variant", "Threads", "AvgTime_s", "PeakMem_Mo",
                    "Speedup_vs_baseline", "MemRatio_vs_baseline",
                    "MemSaved_Mo_vs_baseline"])
        for (cfg, var, tc), (avg, mem) in sorted(norm.items()):
            mem_str = f"{mem:.1f}" if mem is not None else ""
            # Time speedup (>1 = faster than baseline).
            speedup_str = ""
            if var != baseline_variant and (cfg, tc) in base_time and avg > 0:
                speedup_str = f"{base_time[(cfg, tc)] / avg:.3f}"
            # Memory ratio (>1 = uses less memory than baseline) + absolute saving.
            mem_ratio_str = ""
            mem_saved_str = ""
            if var != baseline_variant and mem is not None \
                    and (cfg, tc) in base_mem and mem > 0:
                bm = base_mem[(cfg, tc)]
                mem_ratio_str = f"{bm / mem:.3f}"
                mem_saved_str = f"{bm - mem:.1f}"
            w.writerow([short_config_label(cfg), var, tc, f"{avg:.6f}", mem_str,
                        speedup_str, mem_ratio_str, mem_saved_str])
    print(f"📄 Summary written to {summary_csv}")
