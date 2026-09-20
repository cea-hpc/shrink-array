#!/usr/bin/env python3
"""
Compare `still compressed (at exit)` between a SEQUENTIAL run and OpenMP
runs at several thread counts, for the chunk / chunk2 / chunk2a variants,
all in an OPTIMISED + CHUNK_STATS build, with a FIXED RNG seed.

Why the fixed seed matters: the benchmark's main loop draws its
tab_size/tab_start/tab_nba1/tab_nba2 values from an std::mt19937 whose seed
is read from (and saved to) seed<iter_max>.txt -- see initializeRNG in
bench_utils.cpp. If that file changes between two runs, the two runs are
operating on different data and comparing their final chunk state is
meaningless. This script pins the seed before EVERY run so all of them are
guaranteed to see the exact same generated values.

What it does:
  1. writes a FIXED seed into seed<iter_max>.txt before every run;
  2. runs the sequential binary once, and the OpenMP binary once per entry
     in OMP_THREAD_COUNTS (same executable, different OMP_NUM_THREADS);
  3. extracts, for each variant (chunk/chunk2/chunk2a) actually present in
     the run's output, `compress (during exec)` and
     `still compressed (at exit)` from the matching CHUNK_STATS block;
  4. prints one comparison table per variant and flags any mismatch on
     `still compressed (at exit)` -- which, with a fixed seed, is expected
     to be IDENTICAL across every mode and thread count tested (it is a
     final state, not an event count sensitive to thread scheduling).

Only the `_256` (static chunk size) perf binaries are exercised for now;
switch EXEC_NAME per variant to *_dyn once this is confirmed working.
Meant as a first pass on a local laptop; once validated, port the same
logic into bench_and_tests.py.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# ADJUST HERE if your layout differs. Both build dirs must be OPTIMISED
# (Release/RelWithDebInfo) + CHUNK_STATS; only OMP:BOOL differs between them.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path("~/CLionProjects/library-handling-zeros-fast-compression-of-arrays").expanduser()

BUILD_DIRS = {
    "sequential": PROJECT_ROOT / "cmake-build-release-stats",
    "openmp": PROJECT_ROOT / "cmake-build-release-omp-stats",
}

# Variant name -> executable name (relative to
# <build_dir>/examples/simple_array2/). _256 only for now (static chunk
# size, fastest to validate); switch to *_dyn once this is confirmed working.
VARIANT_EXECS = {
    "chunk": "simple_array2_chunk_perf_256",
    "chunk2": "simple_array2_chunk2_perf_256",
    "chunk2a": "simple_array2_chunk2a_perf_256",
}
EXEC_SUBDIR = Path("examples/simple_array2")

# Thread counts to try for the "openmp" build. The sequential build ignores
# OMP_NUM_THREADS entirely (a non-OpenMP binary does not read it), so it is
# only ever run once regardless of this list.
OMP_THREAD_COUNTS = [8, 16]

# Must match the bench's iter_max (2000 by default -- see
# examples/simple_array2/simple_array2.cpp) so we target the right
# seed<iter_max>.txt.
ITER_MAX = 2000

# Fixed seed: any non-zero integer works (0 makes the bench fall back to a
# random seed -- see initializeRNG).
FIXED_SEED = 42

# Common CLI args for every run (tune --divisor for local run time).
EXTRA_ARGS = ["--divisor", "100"]

# Every run executes with this as its working directory, so all of them
# read/write the SAME seed<iter_max>.txt regardless of where each binary
# itself lives.
RUN_DIR = Path.cwd()
SEED_FILE = RUN_DIR / f"seed{ITER_MAX}.txt"

# Block: "----- CHUNK_STATS [variant] ----- ... -------------------------------------"
BLOCK_RE = re.compile(
    r"----- CHUNK_STATS \[(\w+)\] -----\n(.*?)\n-{5,}", re.DOTALL
)
COMPRESS_RE = re.compile(r"compress \(during exec\)\s*=\s*(\d+)")
STILL_RE = re.compile(r"still compressed \(at exit\)\s*=\s*(\d+)")
# Full funnel, so a low `compress` can be diagnosed as "few chunks were ever
# QUEUED" (marking problem) vs "many were queued but few panned out by drain
# time" (benign timing -- the chunk was rewritten before the batched drain
# ran). Any of these can be absent from a given block (e.g. `mark` only
# appears once at least one write hit the queueing gate).
MARK_RE = re.compile(r"^\s*mark\s*=\s*(\d+)", re.MULTILINE)
QUEUED_RE = re.compile(r"queued\s*=\s*(\d+)")
REFUSED_RE = re.compile(r"refused\s*=\s*(\d+)")
ATTEMPTS_RE = re.compile(r"attempts\s*=\s*(\d+)")
CREATE_RE = re.compile(r"^\s*create\s*=\s*(\d+)", re.MULTILINE)
TEARDOWN_RE = re.compile(r"teardown_clear\s*=\s*(\d+)")

# Final data-content checksums, printed by the bench each k (we want the LAST
# occurrence = the final iteration's values). long_val_sum is an INTEGER sum:
# associative/exact regardless of reduction order, so it is the trustworthy
# "same final content" check. val_sum / cv_val_sum are FLOATING-POINT sums:
# not associative in finite precision, so a parallel reduction can legitimately
# differ in the last few digits from a sequential one even on IDENTICAL data --
# they are reported for reference but are not proof of a real discrepancy by
# themselves.
CHECKSUM_RE = re.compile(
    r"k\s*:\s*(\d+)\s+val_sum\s*=\s*(\S+)\s+long_val_sum\s*=\s*(\S+)\s+cv_val_sum\s*=\s*(\([^)]*\))"
)
# Printed by initializeRNG (bench_utils.cpp) ONLY when it failed to load our
# pinned seed and had to fall back to std::random_device -- i.e. the run did
# NOT use FIXED_SEED. Any occurrence invalidates the whole comparison.
SEED_FALLBACK_RE = re.compile(r"Save seed in file")


def write_fixed_seed() -> None:
    SEED_FILE.write_text(str(FIXED_SEED))
    print(f"  seed file {SEED_FILE} <- {FIXED_SEED}")


def parse_final_checksum(output: str) -> "dict | None":
    """Return the LAST k/val_sum/long_val_sum/cv_val_sum line found, or None."""
    matches = list(CHECKSUM_RE.finditer(output))
    if not matches:
        return None
    m = matches[-1]
    return {
        "k": int(m.group(1)),
        "val_sum": m.group(2),
        "long_val_sum": m.group(3),
        "cv_val_sum": m.group(4),
    }


def seed_was_pinned(output: str) -> bool:
    """False if the run had to fall back to a random seed (our pinned seed
    failed to load) -- see initializeRNG in bench_utils.cpp."""
    return SEED_FALLBACK_RE.search(output) is None


def parse_chunk_stats(output: str) -> dict:
    """Extract per-variant funnel counters from the run's output.

    Returns {variant: {"compress": int|None, "still": int|None,
                        "mark": int|None, "queued": int|None,
                        "refused": int|None, "attempts": int|None}}.
    """
    result = {}
    for m in BLOCK_RE.finditer(output):
        variant, body = m.group(1), m.group(2)

        def grab(pattern: re.Pattern) -> "int | None":
            mm = pattern.search(body)
            return int(mm.group(1)) if mm else None

        result[variant] = {
            "compress": grab(COMPRESS_RE),
            "still": grab(STILL_RE),
            "mark": grab(MARK_RE),
            "queued": grab(QUEUED_RE),
            "refused": grab(REFUSED_RE),
            "attempts": grab(ATTEMPTS_RE),
            "create": grab(CREATE_RE),
            "teardown": grab(TEARDOWN_RE),
        }
    return result


def run_one(label: str, binary: Path, omp_threads: "int | None") -> tuple:
    """Returns (chunk_stats_dict, final_checksum_dict_or_None, seed_was_pinned_bool)."""
    if not binary.exists():
        print(f"!! {label}: binary not found ({binary}) -- check BUILD_DIRS / VARIANT_EXECS above")
        return {}, None, True

    write_fixed_seed()

    env = {**os.environ}
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
        thread_note = f"(OMP_NUM_THREADS={omp_threads})"
    else:
        thread_note = "(no OpenMP)"

    print(f"\n=== running {label}: {binary} {' '.join(EXTRA_ARGS)} {thread_note} ===")

    proc = subprocess.run(
        [str(binary), *EXTRA_ARGS],
        capture_output=True, text=True, env=env, cwd=RUN_DIR,
    )
    output = proc.stdout + proc.stderr

    if proc.returncode != 0:
        print(f"!! {label}: process exited with code {proc.returncode}")

    seed_ok = seed_was_pinned(output)
    if not seed_ok:
        print(f"!! {label}: the pinned seed was NOT used -- the run fell back to a "
              f"random seed (found \"Save seed in file\" in the output). This run's "
              f"data is NOT comparable to the others; fix the seed file/permissions "
              f"before trusting any delta below.")

    checksum = parse_final_checksum(output)
    if checksum is None:
        print(f"!! {label}: no final checksum line found (no \"k : N val_sum = ...\" "
              f"in the output)")

    stats = parse_chunk_stats(output)
    if not stats:
        print(f"!! {label}: no CHUNK_STATS block found in the output "
              f"(is the binary built with -DCHUNK_STATS?)")
    return stats, checksum, seed_ok


def print_table(variant: str, results: dict) -> bool:
    """results: {run_label: {metric: int|None, ...}}. Returns True if a still@exit mismatch was found.

    Prints the full funnel (mark -> queued/refused -> attempts -> compress ->
    still@exit) so a low `compress` in one run can be diagnosed as a marking
    problem (few `queued`) vs a benign drain-timing effect (`queued` high,
    `compress` low because chunks were rewritten again before the batched
    drain ran).
    """
    labels = list(results.keys())
    print("\n" + "=" * 90)
    print(f"variant: {variant}")
    print(f"{'metric':<12}", end="")
    for label in labels:
        print(f"{label:>18}", end="")
    print(f"{'delta':>12}")
    print("-" * 90)

    any_mismatch = False
    metrics = (
        ("mark", "mark"),
        ("queued", "queued"),
        ("refused", "refused"),
        ("attempts", "attempts"),
        ("compress", "compress"),
        ("create", "create"),
        ("teardown_clear", "teardown"),
        ("still@exit", "still"),
    )
    for metric, key in metrics:
        vals = [results[label].get(key) for label in labels]
        if all(v is None for v in vals):
            continue  # this metric never appeared in any run for this variant -- skip the row
        print(f"{metric:<12}", end="")
        for v in vals:
            print(f"{(v if v is not None else '--'):>18}", end="")

        numeric = [v for v in vals if v is not None]
        if metric == "still@exit" and len(numeric) == len(labels) and len(labels) > 1:
            lo, hi = min(numeric), max(numeric)
            delta = hi - lo
            flag = "  <-- MISMATCH" if delta != 0 else ""
            if delta != 0:
                any_mismatch = True
            print(f"{delta:>12}{flag}")
        else:
            print()

    # Derived row: of the chunks still compressed at exit, how many were ever
    # materialised at all (`create`)? still@exit MINUS create is a lower bound
    # on chunks that were NEVER written -- they were born at the constant and
    # stayed there the whole run, which is why `compress` (an event count) can
    # legitimately be far smaller than `still@exit` (a final-state count).
    # Negative values (create > still@exit) mean some created chunks were
    # re-materialised more than once and ended up written at exit, or were
    # recompressed several times -- not an error, just means the simple
    # subtraction underestimates "never touched" for that run.
    still_vals = [results[label].get("still") for label in labels]
    create_vals = [results[label].get("create") for label in labels]
    if any(s is not None and c is not None for s, c in zip(still_vals, create_vals)):
        print(f"{'never touched (lower bound)':<28}", end="")
        for s, c in zip(still_vals, create_vals):
            if s is not None and c is not None:
                print(f"{max(s - c, 0):>18}", end="")
            else:
                print(f"{'--':>18}", end="")
        print()

    print("=" * 90)
    return any_mismatch


def print_checksum_table(variant: str, checksums: dict, seed_flags: dict) -> bool:
    """checksums: {run_label: dict|None}. Returns True if long_val_sum disagrees
    across runs (the exact-arithmetic, trustworthy signal -- val_sum/cv_val_sum
    are floating-point sums and may legitimately differ by a hair due to
    non-associative parallel reduction order, so they are NOT used to flag a
    mismatch here, only reported for reference)."""
    labels = list(checksums.keys())
    if any(not seed_flags.get(l, True) for l in labels):
        print(f"\n!! variant {variant}: at least one run did NOT use the pinned seed "
              f"-- skipping checksum comparison, it would be meaningless.")
        return False

    if any(checksums.get(l) is None for l in labels):
        print(f"\n!! variant {variant}: missing a final checksum for at least one run "
              f"-- cannot compare.")
        return False

    print(f"\nfinal data checksum (variant: {variant}), same fixed seed:")
    print(f"{'':<14}{'k':>8}{'long_val_sum (exact)':>24}{'val_sum (fp, informative)':>28}")
    long_sums = set()
    for label in labels:
        c = checksums[label]
        print(f"{label:<14}{c['k']:>8}{c['long_val_sum']:>24}{c['val_sum']:>28}")
        long_sums.add(c["long_val_sum"])

    if len(long_sums) > 1:
        print(f"!! long_val_sum DISAGREES across runs ({sorted(long_sums)}) despite the "
              f"same pinned seed -- the array's actual final CONTENT differs between "
              f"modes. This is an exact integer sum (associative, no floating-point "
              f"excuse), so this is real: either a genuine race in the benchmark's "
              f"parallel write ranges, or a correctness issue in the library itself.")
        return True

    print("OK: long_val_sum matches exactly across every mode -- the array's final "
          "content is identical regardless of thread count. Any remaining mismatch "
          "in the CHUNK_STATS tables above is therefore about WHEN chunks got "
          "compressed, not about WHAT is in them.")
    return False


def main() -> int:
    any_mismatch = False
    any_checksum_mismatch = False

    for variant, exec_name in VARIANT_EXECS.items():
        print(f"\n\n########## VARIANT: {variant} ##########")

        run_results = {}
        checksums = {}
        seed_flags = {}

        empty = {"compress": None, "still": None, "mark": None,
                 "queued": None, "refused": None, "attempts": None,
                 "create": None, "teardown": None}

        seq_binary = BUILD_DIRS["sequential"] / EXEC_SUBDIR / exec_name
        seq_stats, seq_checksum, seq_seed_ok = run_one("sequential", seq_binary, omp_threads=None)
        run_results["sequential"] = seq_stats.get(variant, dict(empty))
        checksums["sequential"] = seq_checksum
        seed_flags["sequential"] = seq_seed_ok

        omp_binary = BUILD_DIRS["openmp"] / EXEC_SUBDIR / exec_name
        for nthreads in OMP_THREAD_COUNTS:
            label = f"openmp-{nthreads}t"
            omp_stats, omp_checksum, omp_seed_ok = run_one(label, omp_binary, omp_threads=nthreads)
            run_results[label] = omp_stats.get(variant, dict(empty))
            checksums[label] = omp_checksum
            seed_flags[label] = omp_seed_ok

        if print_checksum_table(variant, checksums, seed_flags):
            any_checksum_mismatch = True

        if all(all(x is None for x in v.values()) for v in run_results.values()):
            print(f"!! {variant}: no data collected for this variant -- check the executable path")
            continue

        if print_table(variant, run_results):
            any_mismatch = True

    print()
    if any_checksum_mismatch:
        print(
            "long_val_sum disagreed between modes for at least one variant (see "
            "above) -- STOP here and resolve that first. If the array's own final "
            "content differs by thread count, every CHUNK_STATS delta downstream is "
            "a symptom of that, not an independent bug in the compression "
            "bookkeeping.\n"
        )
    if any_mismatch:
        print(
            "With a FIXED seed, `still compressed (at exit)` should be "
            "identical across every mode and thread count tested (it is a "
            "final state, not an event count sensitive to thread scheduling). "
            "A mismatch here, seed pinned, points to:\n"
            "  - a genuine data race in the benchmark (two OMP threads writing\n"
            "    overlapping ranges without synchronisation), or\n"
            "  - a bug in the counting/audit itself (chunks skipped or double-\n"
            "    counted depending on the execution path).\n"
            "Worth checking the benchmark's parallel write ranges for overlap\n"
            "before re-suspecting the CHUNK_STATS counters."
        )
    else:
        print("OK: `still compressed (at exit)` matches across every mode and "
              "thread count tested, for every variant.")

    return 1 if (any_mismatch or any_checksum_mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
