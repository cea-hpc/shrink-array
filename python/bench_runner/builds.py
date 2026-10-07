# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Build directories, binaries and oracle locations: what to run, and where."""
import hashlib
import os
import platform as _platform
from pathlib import Path

from .settings import COMPRESSION_INSENSITIVE_VARIANTS, COMPRESSION_ONLY_CONFIG_MARKERS
from .term import dim


EXE_SUBPATH = "examples/shrink_bench"

# $arch sub-directory of the CURRENT machine. build_rep may hold builds for
# several architectures (RHEL_9.6__aarch64 AND RHEL_8.10__x86_64); we must only
# run the ones matching the machine the bench runs on, otherwise we get an
# "Exec format error" (x86 binary launched on ARM or vice-versa).
CURRENT_MACHINE = _platform.machine()  # 'aarch64' or 'x86_64'


def _arch_matches_current(arch_dirname):
    """True if the $arch directory name (e.g. 'RHEL_9.6__aarch64') matches the
    current machine. We match on the suffix after the last '__'."""
    # 'RHEL_9.6__aarch64' -> 'aarch64' ; robust if there is no '__'.
    tail = arch_dirname.rsplit("__", 1)[-1]
    return tail == CURRENT_MACHINE


def _variant_base(var):
    """Strip a "@<size>" declination suffix, returning the bare variant name
    (e.g. "chunk2a@256" -> "chunk2a", "vector" -> "vector"). Used wherever the
    logic keys on the variant identity rather than its size."""
    return var.split("@", 1)[0]


def int64_variant(var):
    """The int64 declination of a --variants entry, spelled for resolve_exe:
    "chunk2a@1024" -> "chunk2a_int64@1024" (shrink_bench_chunk2a_int64_1024),
    "vector" -> "vector_int64" (with stage perf: shrink_bench_vector_int64_perf).
    Built on the release OpenMP configs only, for 1024 and dyn (see
    examples/shrink_bench/CMakeLists.txt, SHRINK_BENCH_INT64)."""
    if "@" in var:
        base, size = var.split("@", 1)
        return f"{base}_int64@{size}"
    return f"{var}_int64"


# Written by a config's vector baseline AFTER a complete oracle generation:
# data_tab* files without it are an interrupted generation.
ORACLE_MARKER = "data_tab_ORACLE_COMPLETE.marker"


# The marker also records which vector binary wrote the oracle, as a line
# "<ORACLE_EXE_KEY><sha256>". A config rebuilt with different codegen (other
# flags, other compiler, a fix in the vector path) must not keep comparing
# against the old reference: prepare_oracles regenerates it when the
# fingerprint differs or is missing.
ORACLE_EXE_KEY = "exe-sha256: "


def exe_fingerprint(path):
    """SHA-256 of the executable's content: a relink that changes nothing
    keeps the oracle, a codegen change does not."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def oracle_exe_fingerprint(marker_path):
    """The fingerprint recorded in an oracle marker, or None (no marker, or
    one written before fingerprints were recorded)."""
    try:
        with open(marker_path) as f:
            for line in f:
                if line.startswith(ORACLE_EXE_KEY):
                    return line[len(ORACLE_EXE_KEY):].strip()
    except OSError:
        pass
    return None


def oracle_complete(output_dir, config):
    """Whether `config`'s oracle (oracle_dir) was written to the end."""
    return os.path.exists(os.path.join(oracle_dir(output_dir, config), ORACLE_MARKER))


def resolve_exe(build_root, config, var, stage):
    """
    Look up the executable for (config, var, stage) under several layouts:
      1. $build_root/$config/<arch>/examples/shrink_bench/shrink_bench_{var}_{stage}
         (CEA layout with $arch sub-directory — filtered on the current arch)
      2. $build_root/$config/examples/shrink_bench/shrink_bench_{var}_{stage}
         (build without the $arch level)
      3. $build_root/$config/shrink_bench_{var}_{stage}
         (config = flat directory)
      4. $build_root/shrink_bench_{var}_{stage}_{config}
         (old flat convention)
    Returns the absolute path of the first hit, or None.
    """
    # A variant may carry a size declination as "base@size" (e.g. "chunk2a@dyn",
    # "chunk2@256"). The executable name is then shrink_bench_{base}_{size}.
    #
    # No {stage} in that name any more: check and perf are ONE binary, the mode
    # being a run-time flag (--mode, added by run_with_rebuild_retry and by the
    # phases that build their own argument list). `stage` is still a parameter
    # because it still selects the MODE -- and because "vector" is the exception:
    # its two targets differ by compiler flags (fast-math on perf only), so they
    # stay separate executables and keep the staged name.
    if "@" in var:
        base, size = var.split("@", 1)
        exe_name = f"shrink_bench_{base}_{size}"
    else:
        exe_name = f"shrink_bench_{var}_{stage}"
    root = Path(build_root)

    # Layout 1: glob over the $arch level, but ONLY the current arch.
    for exe in sorted((root / config).glob(f"*/{EXE_SUBPATH}/{exe_name}")):
        # exe = .../{config}/{arch_dir}/examples/shrink_bench/{exe_name}
        # go up 3 levels to get the arch directory name.
        arch_dir = exe.parents[2].name
        if not _arch_matches_current(arch_dir):
            continue
        if exe.is_file() and os.access(exe, os.X_OK):
            return str(exe.resolve())

    # Layout 2: no $arch level
    candidate = root / config / EXE_SUBPATH / exe_name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    # Layout 3: flat executable inside config
    candidate = root / config / exe_name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    # Layout 4: old convention
    candidate = root / f"{exe_name}_{config}"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    return None


def config_dir_present(build_root, config):
    """True if `config` has at least one built executable for the CURRENT arch.
    Lets us skip an entirely-unbuilt config (e.g. a non-OMP dir the user did not
    build) with one note instead of a MISSING warning per variant."""
    # Probe both legacy names (vector, or a build without size declinations)
    # and the multi-size names (base@size) so a config built either way is
    # detected. We only need ONE hit to consider the config present.
    probe_vars = ["vector", "chunk", "chunk2", "chunk2a", "std_alloc"]
    probe_vars += [f"{v}@{s}" for v in ("chunk", "chunk2", "chunk2a", "std_alloc")
                   for s in ("256", "1024", "dyn")]
    for var in probe_vars:
        for stage in ("perf", "check"):
            if resolve_exe(build_root, config, var, stage):
                return True
    return False


def short_config_label(config):
    """Compact label for display / CSV."""
    if config.startswith("cmake-build-omp-"):
        return config[len("cmake-build-omp-"):]  # not removeprefix: Python >= 3.9 only
    if config.startswith("cmake-build-"):
        # non-OMP build dirs (cmake-build-gnu, cmake-build-llvm) would otherwise
        # collapse onto the OMP label ("gnu"/"llvm") and look like a duplicate,
        # single-threaded run in the perf output. Tag them so they stay distinct
        # in the progress bar and the CSV.
        return config[len("cmake-build-"):] + "-noomp"
    return config


def is_compression_only_config(config):
    """True if this config only differs from its base in chunk* compression
    behaviour (e.g. the aggressive-compression variant)."""
    return any(m in config for m in COMPRESSION_ONLY_CONFIG_MARKERS)


def is_omp_config(config):
    """True if this build directory was configured WITH OpenMP.

    The only marker is the directory name: cmake-build-omp-* is an OpenMP build,
    everything else (cmake-build-gnu, cmake-build-llvm-std-alloc, ...) is serial
    and ignores OMP_NUM_THREADS entirely."""
    return config.startswith("cmake-build-omp-")


def is_std_alloc_config(config):
    """True if this is a std-allocator build (its dir name ends in -std-alloc).
    The std_alloc *variant* binary is only built there (CHUNK_STD_ALLOC=ON)."""
    return "std-alloc" in config


def skip_variant_on_config(var, config):
    """Whether to skip (var, config) entirely.

    - Compression-insensitive variants (vector) on a compression-only config
      (-have-aggressive-compression): identical binary to the base config, so
      re-timing it is wasted work.
    - The std_alloc variant anywhere except a -std-alloc config: it is only
      built there now (gated on CHUNK_STD_ALLOC in CMake), so looking for it
      elsewhere would just produce spurious MISSING warnings.
    """
    base = _variant_base(var)
    if base in COMPRESSION_INSENSITIVE_VARIANTS and is_compression_only_config(config):
        return True
    if base == "std_alloc" and not is_std_alloc_config(config):
        return True
    return False


def oracle_dir(output_dir, config):
    """The directory holding `config`'s reference files (data_tab*), under
    output_dir/oracle/<config>: the check binaries of that config run there.

    One oracle PER CONFIG, generated by that config's own vector binary. The
    reference is not portable across compilers: the bench's branches that
    depend on a floating-point value (tab_int[i] = tab_bool[i] ? (tab_float[i]
    > 0 ? 1 : 2) : 0, ...) turn a last-bit difference into 0/1 differences on
    the integer and bool arrays, which the read-back's tolerance cannot
    absorb -- a clang chunk binary against a gcc oracle failed at iteration 1
    with `tab1[29581] 0 != 1` on x86 at -O0. The single oracle this script
    shared across every config (output_dir/data_tab*) is no longer read."""
    return os.path.join(output_dir, "oracle", config)


def _run_dir(output_dir, *parts):
    """A working directory of this run's own, under output_dir.

    The binaries read and write data_<tab>_<k>.bin relative to their cwd, and
    that name carries no variant, no geometry and no compiler -- so two
    binaries sharing a directory silently share reference files. A static
    chunk_256 writing data_tab_bool_1.bin and a chunk2a_dyn reading it back is
    a mismatch at element 450159 on iteration 1, which looks like a library bug
    and is not one. One directory per run, and the question cannot arise.

    Cheap: mkdir plus, where the caller asks for it, a copy of the pinned seed.
    """
    safe = [str(p).replace("/", "_").replace("@", "-").replace(" ", "_") for p in parts if p]
    d = os.path.join(output_dir, "runs", *safe) if safe else output_dir
    os.makedirs(d, exist_ok=True)
    return d


def find_llvm_seq_omp_pairs(build_root, configs):
    """Among `configs`, find every OpenMP config that has a matching non-OMP
    (sequential) counterpart actually present on disk, e.g.
      "cmake-build-omp-llvm"           <-> "cmake-build-llvm"
      "cmake-build-omp-gnu-stats"      <-> "cmake-build-gnu-stats"
      "cmake-build-omp-llvm-std-alloc" <-> "cmake-build-llvm-std-alloc"
    Returns a list of (omp_config, seq_config) pairs. An OMP config without a
    present sequential counterpart is skipped with a single note, not an
    error -- most build trees only have the plain pair, not a sequential
    build of every OMP variant (-std-alloc, -have-aggressive-compression).
    (Used to be LLVM-only, from when only the LLVM pair existed; the compare
    reads `still compressed (at exit)`, which any CHUNK_STATS pair prints --
    the name is kept for its callers.)"""
    pairs, skipped = [], []
    for cfg in configs:
        if not cfg.startswith("cmake-build-omp-"):
            continue
        seq_cfg = cfg.replace("cmake-build-omp-", "cmake-build-", 1)
        if config_dir_present(build_root, seq_cfg):
            pairs.append((cfg, seq_cfg))
        else:
            skipped.append(cfg)
    if skipped:
        print(dim("   seq/omp compare: no sequential counterpart built for "
                  + ", ".join(short_config_label(c) for c in skipped)
                  + " -- skipped"))
    return pairs


def resolve_build_dir(build_root, config):
    """The CMake build directory of `config` for the CURRENT arch -- where
    ctest is run from. Same layouts as resolve_exe():
      1. $build_root/$config/<arch>/   (CEA layout, filtered on the arch)
      2. $build_root/$config/
    A build directory is one holding a CTestTestfile.cmake. Returns the
    absolute path, or None."""
    root = Path(build_root)
    for d in sorted((root / config).glob("*/")):
        if d.is_dir() and _arch_matches_current(d.name) and (d / "CTestTestfile.cmake").is_file():
            return str(d.resolve())
    d = root / config
    if (d / "CTestTestfile.cmake").is_file():
        return str(d.resolve())
    return None


def validation_built(bdir):
    """False when examples/CMakeLists.txt skipped the validation harnesses in
    this build dir (CHUNK_VALIDATION_BUILT=OFF in its CMakeCache.txt: a perf
    configuration). True when the cache says ON, or predates the switch."""
    try:
        with open(os.path.join(bdir, "CMakeCache.txt")) as f:
            for line in f:
                if line.startswith("CHUNK_VALIDATION_BUILT:"):
                    return line.strip().split("=", 1)[1].upper() not in ("OFF", "0", "FALSE", "NO")
    except OSError:
        pass
    return True
