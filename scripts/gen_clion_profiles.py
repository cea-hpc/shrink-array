#!/usr/bin/env python3
"""
gen_clion_profiles.py -- give CLion the build configurations of the scripts,
under the scripts' names.

The build scripts (compile_all.sh, compile_all_gnu.sh) are the one source of
truth: this runs `compile_all.sh --dryrun --with-nvhpc`, reads every
`CXX=<compiler> cmake -S <src> -B <build_rep>/cmake-build-<name>/<arch> <options>`
it prints, and writes one CLion CMake profile per configuration into
.idea/workspace.xml (component CMakeSettings):

  PROFILE_NAME    <name>                         -- as on the calculators
  GENERATION_DIR  cmake-build-<name>/<arch>      -- as on the calculators
  TOOLCHAIN_NAME  from the compiler: g++ -> Default, clang++ -> clang,
                  icpx -> oneapi, nvc++ -> nvhpc (the CLion toolchains)
  CONFIG_NAME     the options' CMAKE_BUILD_TYPE (Release when absent: the
                  project's default)
  GENERATION_OPTIONS  the other options (the compiler is the toolchain's)
  BUILD_OPTIONS   -j 7 (the laptop's compile cap)

Every configuration gets a profile; only the everyday ones are ENABLED (CLion
configures every enabled profile when the project opens) -- the others are one
click away. --enable replaces that list.

CLion rewrites workspace.xml when it closes: run this with CLion CLOSED
(refused otherwise, unless --force). A copy of the previous file is kept next
to it (workspace.xml.bak-<date>).

  python3 scripts/gen_clion_profiles.py            # write the profiles
  python3 scripts/gen_clion_profiles.py --list     # show them, write nothing
"""
import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from xml.sax.saxutils import quoteattr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The everyday profiles, enabled; every other configuration is generated
# disabled. (The former hand-made profiles map onto these names.)
DEFAULT_ENABLED = [
    "omp-gnu-debug", "gnu-debug", "omp-gnu", "gnu", "omp-gnu-stats", "gnu-stats", "omp-gnu-check",
    "omp-gnu-debug-std-alloc", "gnu-std-alloc-debug", "omp-gnu-debug-thread-sanitizer",
    "omp-gnu-debug-std-alloc-address-sanitizer", "omp-llvm-debug", "omp-llvm", "omp-llvm-stats",
]

TOOLCHAINS = [("nvc++", "nvhpc"), ("icpx", "oneapi"), ("clang++", "clang"), ("g++", None)]

LINE = re.compile(r"^CXX=(\S+) cmake -S (\S+) -B (\S+) (.*)$")


def toolchain_of(compiler):
    base = os.path.basename(compiler.strip("'\""))
    for key, name in TOOLCHAINS:
        if base.startswith(key):
            return name
    sys.exit(f"error: no CLion toolchain for compiler {compiler}")


def configurations():
    """(name, generation_dir, toolchain, build_type, options) per configuration."""
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "compile_all.sh"), "--dryrun", "--with-nvhpc"],
                         capture_output=True, text=True, cwd="/")
    confs, seen = [], set()
    for line in out.stdout.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        compiler, _src, bdir, rest = m.groups()
        bdir = bdir.strip("'\"")
        parts = bdir.rstrip("/").split("/")
        name = next((p[len("cmake-build-"):] for p in parts if p.startswith("cmake-build-")), None)
        if name is None or name in seen:
            continue
        seen.add(name)
        gen_dir = os.path.relpath(bdir, ROOT) if bdir.startswith(ROOT + "/") else bdir
        opts, build_type, toks = [], "Release", shlex.split(rest)
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("-DCMAKE_BUILD_TYPE="):
                build_type = t.split("=", 1)[1]
            elif t.startswith("-DCMAKE_CXX_COMPILER="):
                pass  # the toolchain's compiler
            elif t == "-G":
                opts += ["-G", toks[i + 1]]
                i += 1
            else:
                opts.append(t)
            i += 1
        confs.append((name, gen_dir, toolchain_of(compiler), build_type, " ".join(shlex.quote(o) for o in opts)))
    if not confs:
        sys.exit("error: compile_all.sh --dryrun printed no configuration:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return confs


def xml_block(confs, enabled):
    lines = ['  <component name="CMakeSettings">', "    <configurations>"]
    for name, gen_dir, toolchain, build_type, opts in confs:
        attrs = [("PROFILE_NAME", name), ("ENABLED", "true" if name in enabled else "false"),
                 ("GENERATION_DIR", gen_dir), ("CONFIG_NAME", build_type)]
        if toolchain:
            attrs.append(("TOOLCHAIN_NAME", toolchain))
        attrs += [("GENERATION_OPTIONS", opts), ("BUILD_OPTIONS", "-j 7")]
        lines.append("      <configuration " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs) + " />")
    lines += ["    </configurations>", "  </component>"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the profiles, write nothing")
    ap.add_argument("--enable", help="comma-separated names to enable (replaces the default list)")
    ap.add_argument("--workspace", default=os.path.join(ROOT, ".idea", "workspace.xml"))
    ap.add_argument("--force", action="store_true", help="write even if CLion seems to be running")
    a = ap.parse_args()

    enabled = set(a.enable.split(",")) if a.enable else set(DEFAULT_ENABLED)
    confs = configurations()
    names = {c[0] for c in confs}
    for n in sorted(enabled - names):
        print(f"warning: --enable name '{n}' is not a configuration of the scripts", file=sys.stderr)

    if a.list:
        for name, gen_dir, toolchain, build_type, _ in confs:
            print(f"{'*' if name in enabled else ' '} {name:62s} {toolchain or 'Default':8s} {build_type:15s} {gen_dir}")
        print(f"{len(confs)} profiles, {len(enabled & names)} enabled (*)")
        return

    if not a.force and subprocess.run(["pgrep", "-f", "clion"], capture_output=True).returncode == 0:
        sys.exit("error: CLion is running -- it rewrites workspace.xml when it closes. Close it, or pass --force.")

    ws = open(a.workspace, encoding="utf-8").read()
    block = re.compile(r'  <component name="CMakeSettings">.*?\n  </component>', re.S)
    if len(block.findall(ws)) != 1:
        sys.exit(f"error: no single CMakeSettings component in {a.workspace}")
    backup = f"{a.workspace}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(a.workspace, backup)
    open(a.workspace, "w", encoding="utf-8").write(block.sub(lambda _: xml_block(confs, enabled), ws))
    print(f"{a.workspace}: {len(confs)} profiles ({len(enabled & names)} enabled); previous file in {backup}")


if __name__ == "__main__":
    main()
