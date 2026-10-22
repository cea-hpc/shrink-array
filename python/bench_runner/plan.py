# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""What a campaign runs: thread counts, configs, binaries, oracles, repetitions."""
import subprocess
import os
import sys
from pathlib import Path

from .builds import (
    CURRENT_MACHINE, ORACLE_MARKER, config_dir_present, exe_fingerprint, is_compression_only_config,
    is_threaded_config, oracle_dir, oracle_exe_fingerprint, resolve_exe, short_config_label,
    size_not_built, skip_variant_on_config,
)
from .phases.fuzz import FUZZ_THREADS
from .phases.stl import stl_representative_exes
from .settings import (
    DEFAULT_CONFIGS, DEFAULT_SEQ_OMP_THREADS, DEFAULT_VARIANTS, DEFAULT_THREADS_CHECK, MODE_PRESETS,
    PERF_EXCLUDED_CONFIGS, REFERENCE_THREADS, SANITIZER_CONFIGS, SANITIZER_REPS, sanitizer_skip_reason,
    SANITIZER_THREADS, fit_threads, is_stats_config, is_tbb_variant, machine_cpus,
)
from .term import bold, cyan, dim, red, yellow


def resolve_run_parameters(args):
    """Fit the thread counts to this machine, apply the --mode preset to every
    option left unset, expand the variants over --sizes."""
    # --- Fit the thread counts to this machine (see machine_cpus) -----------
    usable, physical, sockets = machine_cpus()
    if args.max_threads is None:
        args.max_threads = usable
    cap  = args.max_threads
    phys = min(physical, cap)
    ref  = min(REFERENCE_THREADS, phys)
    machine_points = (ref, phys, cap)
    print(f"🖥️  Machine: {usable} usable CPUs, {physical} physical cores, {sockets} socket(s)"
          f" -> up to {cap} threads, reference {ref}")

    # --- Apply the --mode preset for any arg the user did NOT set explicitly ---
    preset = MODE_PRESETS[args.mode]
    if args.threads is None:
        args.threads = fit_threads(preset["threads"], cap, machine_points)
    if args.reps_perf is None:
        args.reps_perf = preset["reps_perf"]
    if args.reps_check is None:
        args.reps_check = preset["reps_check"]
    if args.full_reps_threads is None:
        args.full_reps_threads = fit_threads(preset["full_reps_threads"], cap, (ref,))
        # --threads given without --full-reps-threads: the full repetitions go
        # to the count of that list nearest the reference, rather than to a
        # count the sweep never runs.
        if not set(args.full_reps_threads) & set(args.threads):
            args.full_reps_threads = [min(args.threads, key=lambda t: (abs(t - ref), t))]
    if args.light_variants is None:
        args.light_variants = list(preset.get("light_variants", []))
    if args.light_threads is None:
        light = preset.get("light_threads", REFERENCE_THREADS)
        args.light_threads = light if light <= cap else ref
    if args.check_threads is None:
        args.check_threads = min(DEFAULT_THREADS_CHECK, cap)
    if args.seq_omp_threads is None:
        args.seq_omp_threads = fit_threads(preset["seq_omp"]["threads"], cap, machine_points)
    if args.seq_omp_sizes is None:
        args.seq_omp_sizes = list(preset["seq_omp"]["sizes"])
    # The phases the mode leaves out (an explicit --skip-* stays as given).
    if "sanitizers" in preset["skip"] and not args.skip_sanitizers:
        args.skip_sanitizers = True
        print(dim(f"   {args.mode}: no sanitizer phase (--mode normal or full runs it)"))
    # The scalability guard of the mode (see run_scaling_phase).
    args.scaling_configs = list(preset["scaling"]["configs"])
    args.scaling_threads = list(preset["scaling"]["threads"])
    args.scaling_sizes   = preset["scaling"]["sizes"]
    args.fuzz_threads      = fit_threads(FUZZ_THREADS, cap, machine_points)
    args.sanitizer_threads = min(SANITIZER_THREADS, cap)
    over = sorted({t for t in [*args.threads, args.check_threads, *args.seq_omp_threads] if t > cap})
    if over:
        print(yellow(f"⚠️  Thread count(s) {over} passed explicitly exceed the {cap} usable CPUs: "
                     "those runs are oversubscribed"))
    # --- Expand variants across the requested size declinations -----------
    # Shrink variants (chunk/chunk2/chunk2a/std_alloc) gain a "@<size>" suffix
    # so each (variant, size) pair is handled independently by the existing
    # machinery (resolve_exe decodes "base@size"). "vector" has no chunk
    # geometry and is left as-is. light_variants get the same treatment so a
    # light variant like std_alloc still resolves to real per-size binaries.
    SIZE_AGNOSTIC = {"vector"}
    # The TBB bench is part of the default variants but only exists where
    # oneTBB was found at compile time: a missing one is a note, not a warning,
    # unless the variants were named on the command line.
    args.tbb_implicit = args.variants == DEFAULT_VARIANTS

    def _expand_with_sizes(variant_list):
        out = []
        for v in variant_list:
            if v in SIZE_AGNOSTIC or "@" in v:
                out.append(v)
            else:
                out.extend(f"{v}@{s}" for s in args.sizes)
        return out

    args.variants        = _expand_with_sizes(args.variants)
    args.light_variants  = _expand_with_sizes(args.light_variants)
    # The baseline variant may also need a size to resolve its check binary.
    # vector is size-agnostic, so the common baseline (vector) is unaffected; if
    # a shrink variant is used as baseline, pin it to the first requested size.
    if args.baseline_variant not in SIZE_AGNOSTIC and "@" not in args.baseline_variant:
        args.baseline_variant = f"{args.baseline_variant}@{args.sizes[0]}"

    light_note = (f" light={args.light_variants}@{args.light_threads}t"
                  if args.light_variants else "")
    print(f"🎚️  Mode '{args.mode}': threads={args.threads} reps_perf={args.reps_perf} "
          f"full_reps_threads={args.full_reps_threads} reps_check={args.reps_check}{light_note}")


def reduce_matrix(args):
    """--mode fast: drop what the mode does not need. Returns the configs kept
    out of the PERF phase (they stay in CHECK)."""
    # --- Fast-mode matrix reduction --------------------------------------
    # Only trims work the user did not pin explicitly on the command line.
    perf_excluded = set(PERF_EXCLUDED_CONFIGS)
    if args.mode == "fast":
        # 1. Drop -have-aggressive-compression configs: benchmarks confirm they
        #    are within noise of the release configs on this workload, so running
        #    them doubles the matrix for no signal. (Skipped only if the user did
        #    not name configs explicitly.)
        if args.configs == DEFAULT_CONFIGS:
            kept = [c for c in args.configs if not is_compression_only_config(c)]
            dropped = [short_config_label(c) for c in args.configs if c not in kept]
            if dropped:
                print(dim(f"   fast: skipping aggressive-compression configs "
                          f"({', '.join(dropped)}) — ≈ release on this workload"))
            args.configs = kept
        args.light_variants = sorted(set(args.light_variants) | {"std_alloc"})
        # No rule demoting chunk/chunk2/chunk2a on a -std-alloc config to a
        # light smoke run: the allocator moves the numbers well outside
        # smoke-test territory -- on aarch64, chunk@1024 at 64 threads is
        # 14.53 s under gnu and 12.60 s under gnu-std-alloc (+15%), and
        # chunk@dyn 56.4 s vs 40.3 s (+40%). Those are results, not duplicates.
        # (Any such test must compare _variant_base(var): `var` carries its
        # @<size> declination.)
        #
        # 2. Drop the PERF phase of every serial (non-OpenMP) config. They stay
        #    in CHECK, in the validation harnesses and in the seq/omp compare --
        #    only the timing sweep goes. It was the single largest item in the
        #    campaign: on aarch64, gnu-noomp 89 min and llvm-noomp about 2 h, so
        #    3h29 of a 5 h run, to time a single-threaded build whose numbers do
        #    not move between campaigns. `--mode normal` and `--mode full` still
        #    measure them.
        _serial = [c for c in args.configs if not is_threaded_config(c)]
        if _serial:
            perf_excluded |= set(_serial)
            print(dim(f"   fast: skipping the PERF phase of the serial configs "
                      f"({', '.join(short_config_label(c) for c in _serial)}) "
                      f"— kept in CHECK and validation"))
    # 3. The configs the preset leaves out of the timing (perf_drop markers).
    _drop = [c for c in args.configs
             if any(m in c for m in MODE_PRESETS[args.mode].get("perf_drop", []))
             and c not in perf_excluded]
    if _drop:
        perf_excluded |= set(_drop)
        print(dim(f"   {args.mode}: skipping the PERF phase of "
                  f"{', '.join(short_config_label(c) for c in _drop)} — kept in CHECK"))
    return perf_excluded


def resolve_ldpath(args):
    """The extra LD_LIBRARY_PATH of every run (--ld-library-path, --auto-libpath)."""
    # --- Resolve the extra LD_LIBRARY_PATH -------------------------------
    extra_ldpath = args.ld_library_path
    if args.auto_libpath and not extra_ldpath:
        import shutil
        cxx = shutil.which("clang++")
        if cxx:
            # /.../llvm-20.1.8/.../bin/clang++ -> /.../llvm-20.1.8/.../lib
            lib = str(Path(cxx).resolve().parent.parent / "lib")
            if Path(lib).is_dir():
                extra_ldpath = lib
                print(f"🔧 --auto-libpath: LD_LIBRARY_PATH += {lib}")
            else:
                print(f"⚠️  --auto-libpath: {lib} does not exist, ignored.")
        else:
            print("⚠️  --auto-libpath: clang++ not found in PATH "
                  "(load the llvm module first).")
    if extra_ldpath:
        print(f"🔧 Extra LD_LIBRARY_PATH for every run: {extra_ldpath}")
    return extra_ldpath


def drop_absent_configs(args):
    """Keep the configs built for this architecture; exit when none is."""
    # --- Drop configs whose build directory is absent (one note each) ----
    present, absent = [], []
    for cfg in args.configs:
        (present if config_dir_present(args.build_root, cfg) else absent).append(cfg)
    if absent:
        print(dim("ℹ️  Skipping unbuilt configs (no binary for this arch): "
                  + ", ".join(short_config_label(c) for c in absent)))
    args.configs = present
    if not args.configs:
        print(red(f"\n❌ None of the requested configs are built under "
                  f"{args.build_root!r}."))
        sys.exit(2)
    # The two stats checks read the CHUNK_STATS block; without a stats build in
    # the run they pass without having looked at anything. Say so up front
    # rather than letting a green run suggest the compression health was
    # verified.
    if not any(is_stats_config(c) for c in args.configs):
        print(yellow("⚠️  No CHUNK_STATS config in this run: the mark/create "
                     "compression-health guard and the seq-vs-OMP still@exit "
                     "compare have nothing to read and will pass vacuously. "
                     "Build cmake-build-omp-gnu-stats / cmake-build-gnu-stats "
                     "(-DCHUNK_STATS=ON) to arm them."))


def discover_binaries(args, perf_excluded):
    """The perf and check binary of every (config, variant), and each config's
    vector check binary (its oracle generator). Exits when there is none."""
    # --- Phase 0: binary discovery ---------------------------------------
    print(f"🔎 Resolving executables under build_root={args.build_root!r}")
    print(f"   Current machine: {CURRENT_MACHINE} → only *__{CURRENT_MACHINE} "
          f"builds will be run")
    perf_exes  = {}   # (config, var) -> path
    check_exes = {}   # (config, var) -> path   (non-baseline variants)
    # config -> path of ITS vector check binary: every config generates its
    # own oracle (see oracle_dir), whatever skip_variant_on_config says about
    # running the vector variant on it.
    baseline_check_exe_of = {}
    missing_perf, missing_check = [], []
    no_tbb_cfgs = []      # std-backend configs without the TBB bench (default variants only)
    sizes_not_built = {}  # config -> sizes it does not build (CHUNK_STATIC_SIZES)

    for cfg in args.configs:
        if not args.skip_check:
            _bp = resolve_exe(args.build_root, cfg, args.baseline_variant, "check")
            if _bp:
                baseline_check_exe_of[cfg] = _bp
        for var in args.variants:
            if skip_variant_on_config(var, cfg):
                # e.g. vector on -have-aggressive-compression: identical binary
                # to the base config, not re-run. Don't resolve or flag missing.
                continue
            if size_not_built(args.build_root, cfg, var):
                # e.g. @1024 on a Debug or sanitizer build: 256 alone by default.
                sizes_not_built.setdefault(cfg, set()).add(var.split("@", 1)[1])
                continue
            if is_tbb_variant(var) and getattr(args, "tbb_implicit", False):
                # Asked for by default, not by name: a std build made without
                # oneTBB has no TBB bench, which is not a defect. One note per
                # config; any other config of the run is unaffected.
                if not resolve_exe(args.build_root, cfg, var, "check"):
                    if cfg not in no_tbb_cfgs:
                        no_tbb_cfgs.append(cfg)
                    continue
            if cfg in perf_excluded:
                pass  # kept in CHECK, dropped from PERF (redundant with gnu-std-alloc)
            else:
                p = resolve_exe(args.build_root, cfg, var, "perf")
                if p:
                    perf_exes[(cfg, var)] = p
                else:
                    missing_perf.append((cfg, var))

            if var == args.baseline_variant:
                # The baseline check (vector, uncompressed) only generates the
                # oracle: resolved per config above (baseline_check_exe_of),
                # never a CHECK run of its own.
                continue
            p = resolve_exe(args.build_root, cfg, var, "check")
            if p:
                check_exes[(cfg, var)] = p
            else:
                missing_check.append((cfg, var))

    def _fmt(cfg, var):
        return f"[{short_config_label(cfg):40} / {var:10}]"

    for cfg in no_tbb_cfgs:
        print(dim(f"   ℹ️  {short_config_label(cfg)}: no TBB bench (shrink_bench_tbb_*) built there "
                  f"-- oneTBB was not found when it was compiled (TBBROOT, CHUNK_WITH_TBB=1); "
                  f"the tbb_* variants are skipped"))
    for cfg, sizes in sorted(sizes_not_built.items()):
        print(dim(f"   ℹ️  {short_config_label(cfg)}: size(s) {', '.join(sorted(sizes))} "
                  f"not built there (CHUNK_STATIC_SIZES), skipped"))
    for (cfg, var), p in sorted(perf_exes.items()):
        print(f"   ✅ perf  {_fmt(cfg, var)} → {p}")
    for (cfg, var), p in sorted(check_exes.items()):
        print(f"   ✅ check {_fmt(cfg, var)} → {p}")
    for cfg, var in missing_perf:
        print(f"   ⚠️  MISSING perf  {_fmt(cfg, var)} — skipped")
    for cfg, var in missing_check:
        print(f"   ⚠️  MISSING check {_fmt(cfg, var)} — skipped")

    if not perf_exes and not check_exes:
        print(f"\n❌ No executable found under {args.build_root!r}.")
        print("   Check --build-root (default = cwd), or pass --configs "
              "explicitly, or --variants.")
        sys.exit(2)
    return perf_exes, check_exes, baseline_check_exe_of


def check_linkage(perf_exes, check_exes, extra_ldpath):
    """ldd one binary per config, before thousands of runs fail with code 127."""
    # --- Pre-flight: check dynamic linkage -------------------------------
    # Run ldd on one binary per config (with extra_ldpath applied) and warn if
    # any .so is missing, BEFORE launching thousands of runs that would all fail
    # with code 127 (e.g. libc++.so.1 / libomp.so for the llvm builds).
    import shutil as _shutil
    if _shutil.which("ldd"):
        seen_configs = set()
        libpb = False
        for (cfg, var), exe in sorted({**perf_exes, **check_exes}.items()):
            if cfg in seen_configs:
                continue
            seen_configs.add(cfg)
            env = {**os.environ}
            if extra_ldpath:
                prev = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = extra_ldpath + (":" + prev if prev else "")
            try:
                r = subprocess.run(["ldd", exe], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, timeout=30)
                miss = [ln.strip() for ln in r.stdout.splitlines() if "not found" in ln]
                if miss:
                    libpb = True
                    print(f"   ❌ LINKAGE [{short_config_label(cfg)}]: "
                          f"{len(miss)} missing lib(s):")
                    for m in miss:
                        print(f"        {m}")
            except Exception:
                pass
        if libpb:
            print("\n⚠️  Some shared libraries are missing (above).")
            print("   → load the matching module BEFORE running, or pass")
            print("     --ld-library-path <dir containing the .so>, or --auto-libpath")
            print("     (after loading the llvm module). The clean fix is to build")
            print("     with a proper RPATH so binaries are self-contained.")
            print("   The affected runs will fail with code 127 if you continue.\n")


def prepare_oracles(args, check_exes, baseline_check_exe_of):
    """Which configs must (re)generate their oracle. Returns the configs with
    check binaries and, per config, whether its baseline must run."""
    # The baseline check (vector, uncompressed) ONLY serves to generate the
    # reference files data_tab* (an expensive output). One oracle PER CONFIG
    # (see oracle_dir: the reference is not portable across compilers), each
    # generated once by the config's own vector binary and kept across runs:
    #   - only if the config has check binaries to compare (no oracle for a
    #     config that would not read it),
    #   - and only if its oracle is absent or --regen-ref says so.
    import glob as _glob

    def _oracle_files(d):
        return [f for f in _glob.glob(os.path.join(d, "data_tab*")) if not f.endswith(".marker")]

    def _purge_oracle(d):
        n = 0
        for f in _glob.glob(os.path.join(d, "data_tab*")):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
        return n

    # A legacy single oracle (data_tab*) in output_dir itself is read by
    # nothing; say so once rather than silently keeping it around.
    if not args.skip_check and os.path.isfile(os.path.join(args.output_dir, ORACLE_MARKER)):
        print(f"ℹ️  A legacy single oracle sits in {args.output_dir} (data_tab*): it is no "
              f"longer used -- each config now has its own under oracle/<config>/. "
              f"Delete it when convenient.")

    # config -> True when its baseline must run to (re)generate the oracle.
    oracle_needed = {}
    configs_checked = sorted({cfg for (cfg, _v) in check_exes})
    for cfg in configs_checked if not args.skip_check else []:
        d = oracle_dir(args.output_dir, cfg)
        marker = os.path.join(d, ORACLE_MARKER)
        files = _oracle_files(d)
        # The oracle counts as present ONLY if the completeness marker exists
        # (written by the baseline AFTER a successful full generation): files
        # without it are an interrupted generation, purged and regenerated.
        present = os.path.isfile(marker) and bool(files)
        if files and not present:
            _purge_oracle(d)
            print(f"⚠️  Partial oracle for {short_config_label(cfg)} in {d} "
                  f"({len(files)} file(s), no completeness marker) → purged, regenerating.")
        if args.regen_ref and (present or files):
            n = _purge_oracle(d)
            print(f"ℹ️  --regen-ref: removed {n} file(s) in {d} → regenerating.")
            present = False
        # Stale oracle: the config's vector binary is not the one that wrote
        # it (rebuilt with other flags, another compiler...). Without this the
        # check compared every variant against a reference the current
        # baseline no longer produces. A marker with no fingerprint predates
        # the check: regenerated once.
        if present and cfg in baseline_check_exe_of:
            recorded = oracle_exe_fingerprint(marker)
            current  = exe_fingerprint(baseline_check_exe_of[cfg])
            if recorded != current:
                n = _purge_oracle(d)
                why = ("no binary fingerprint in its marker (older run)" if recorded is None
                       else "the vector binary changed since it was generated")
                print(f"ℹ️  Oracle for {short_config_label(cfg)}: {why} → "
                      f"removed {n} file(s), regenerating.")
                present = False
        if present:
            print(f"ℹ️  Oracle for {short_config_label(cfg)} present in {d} → "
                  f"'{args.baseline_variant}' run SKIPPED.")
            oracle_needed[cfg] = False
        elif cfg in baseline_check_exe_of:
            print(f"ℹ️  No oracle for {short_config_label(cfg)} → "
                  f"'{args.baseline_variant}' of that config run once to generate it in {d}.")
            oracle_needed[cfg] = True
        else:
            print(f"⚠️  No oracle for {short_config_label(cfg)} AND no "
                  f"'{args.baseline_variant}' check binary in that config → its CHECK "
                  f"runs will abort on missing references. Build the vector variant there.")
            oracle_needed[cfg] = False
    return configs_checked, oracle_needed


class Plan:
    """The campaign's binaries, oracles and repetition policy, shared by the
    check, STL, sanitizer and perf phases."""

    def __init__(self, args, perf_excluded, extra_ldpath):
        self.args = args
        self.extra_ldpath = extra_ldpath
        self.perf_exes, self.check_exes, self.baseline_check_exe_of = discover_binaries(args, perf_excluded)
        check_linkage(self.perf_exes, self.check_exes, extra_ldpath)

        # --- Perf repetition policy ------------------------------------------
        # 1. The baseline (vector): a single run (no need to average).
        # 2. --full-reps-threads: if given, only those thread counts get --reps_perf
        #    repetitions (averaged measurement); all other thread counts get a single
        #    run (scalability curve). If not given, every thread count gets
        #    --reps_perf.
        self.full_reps_tc = set(args.full_reps_threads) if args.full_reps_threads else None

        self.light_set = set(args.light_variants or [])

        if self.full_reps_tc is not None:
            missing = self.full_reps_tc - set(args.threads)
            if missing:
                print(f"⚠️  --full-reps-threads {sorted(missing)} not in --threads "
                      f"{args.threads} → those values will never be run.")
            print(f"ℹ️  {args.reps_perf} runs on thread count(s) {sorted(self.full_reps_tc)}; "
                  f"1 run on the others (scalability).")

        self.configs_checked, self.oracle_needed = prepare_oracles(args, self.check_exes,
                                                                   self.baseline_check_exe_of)

    def is_light(self, var, cfg):
        """A (var, cfg) pair runs as a single smoke test rather than the full
        sweep when the variant is globally light (--light-variants)."""
        return var in self.light_set

    def threads_for(self, var, cfg):
        """Thread counts to sweep for (var, cfg). Light pairs run only on the
        single light_threads count (a smoke test), never the full sweep."""
        args = self.args
        if self.is_light(var, cfg):
            return [args.light_threads]
        # A serial build (neither OpenMP nor the std backend) ignores
        # OMP_NUM_THREADS, so sweeping it just runs the SAME measurement N times. It showed: in the aarch64 fast
        # campaign, gnu-noomp vector came out at 109.154 / 108.772 / 109.094 s
        # for 16 / 64 / 128 threads, and the two noomp configs alone took 3h29
        # of a 5h run -- two thirds of it for nothing. Keep one count, preferring
        # the reference one so the noomp rows line up with the OMP rows.
        if not is_threaded_config(cfg):
            if self.full_reps_tc:
                for tc in args.threads:
                    if tc in self.full_reps_tc:
                        return [tc]
            return [args.threads[0]]
        return args.threads

    def reps_perf_for(self, var, tc, cfg):
        args = self.args
        if self.is_light(var, cfg):
            return 1                       # light: one smoke run only
        # Non-OMP builds are single-threaded and deterministic: repeating them
        # yields identical timings, so one run is enough (and saves a lot of
        # wall time — a noomp chunk run can be minutes). OMP configs keep the
        # full repetition count (scheduling / NUMA / contention add real noise).
        if not is_threaded_config(cfg):
            return 1
        if var == args.baseline_variant:
            return 1
        if self.full_reps_tc is None:
            return args.reps_perf
        return args.reps_perf if tc in self.full_reps_tc else 1

    def reps_check_for(self, cfg):
        # Non-OMP builds are single-threaded and deterministic: one CHECK run is
        # enough (repeating gives identical results). OMP configs keep the full
        # reps_check count, where scheduling/contention make repetition useful.
        if not is_threaded_config(cfg):
            return 1
        return self.args.reps_check

    def count_runs(self):
        """Print the run totals the progress bar counts against; return the total."""
        args = self.args
        # --- Totals (accounting for the policies above) ----------------------
        total_perf_runs = 0
        if not args.skip_perf:
            for (cfg, var) in self.perf_exes:
                if skip_variant_on_config(var, cfg):
                    continue
                for tc in self.threads_for(var, cfg):
                    total_perf_runs += self.reps_perf_for(var, tc, cfg)

        total_check_runs = 0
        if not args.skip_check:
            total_check_runs = sum(
                self.reps_check_for(cfg) for (cfg, _var) in self.check_exes.keys()
            )
            total_check_runs += sum(1 for v in self.oracle_needed.values() if v)

        total_stl_runs = 0
        if not args.skip_check and not args.skip_stl_test:
            total_stl_runs = len(stl_representative_exes(args, self.check_exes))

        # Sanitizer CHECK runs (phase 2d): SANITIZER_REPS per present (config,variant).
        total_san_runs = 0
        if not args.skip_check and not args.skip_sanitizers:
            for cfg in SANITIZER_CONFIGS:
                if not config_dir_present(args.build_root, cfg):
                    continue
                if sanitizer_skip_reason(cfg):
                    continue
                for var in args.variants:
                    if skip_variant_on_config(var, cfg):
                        continue
                    if resolve_exe(args.build_root, cfg, var, "check"):
                        total_san_runs += SANITIZER_REPS

        overall_runs = (total_perf_runs + total_check_runs
                        + total_stl_runs + total_san_runs)

        print(bold(f"\n🚀 Starting: {cyan(total_perf_runs)} perf runs + "
                   f"{cyan(total_check_runs)} check runs + "
                   f"{cyan(total_stl_runs)} STL + "
                   f"{cyan(total_san_runs)} sanitizer = {cyan(overall_runs)} total."))
        return overall_runs
