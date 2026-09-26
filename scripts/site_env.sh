#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# site_env.sh -- the site layer of the build scripts, sourced by
# compile_all.sh and compile_all_gnu.sh before anything else.
#
# The scripts were written for the CEA machines: `module load` for every tool,
# the sources in ~/shrink-array, the build trees under /ccc/work, the GCC
# toolchain for clang / icpx / nvc++ under /ccc/products. Everything
# site-specific is decided HERE, so the same scripts run on a workstation:
#
#   CHUNK_SITE              cea when /ccc/products exists, else local
#   CHUNK_SRC_DIR           the repository (from this script's location)
#   CHUNK_BUILD_REP         where the cmake-build-<name>/<arch> trees go:
#                             cea   $CCCWORKDIR/shrink-array
#                             local the repository itself (as CLion does)
#   CHUNK_GCC_TOOLCHAIN_DIR the libstdc++ that clang / icpx / nvc++ use:
#                             cea   /ccc/products/gcc-15.2.0/system/default
#                             local the newest ~/opt/gcc-X.Y.Z or /opt/gcc-X.Y.Z
#                                   built from source (CHUNK_LOCAL_GCC_DIR,
#                                   also the gnu compiler), else empty (the
#                                   compilers' own default)
#   CHUNK_HAVE_MODULES      true when Environment Modules is available
#
# Each can be set in the environment first to override. The build names
# (cmake-build-<name>/<arch>, arch = ID_VERSION__machine from os-release) are
# the same on every site: scripts/gen_clion_profiles.py derives the CLion
# profiles from them.
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHUNK_SRC_DIR:=$(cd "$CHUNK_SCRIPT_DIR/.." && pwd)}"

if [ -z "${CHUNK_SITE:-}" ]; then
    if [ -d /ccc/products ]; then CHUNK_SITE=cea; else CHUNK_SITE=local; fi
fi

# chunk_local_gcc: the newest GCC built from source under ~/opt or /opt
# (<root>/gcc-X.Y.Z/bin/g++, i.e. configure --prefix=$HOME/opt/gcc-16.2.0),
# the workstation's counterpart of /ccc/products/gcc-X.Y.Z. Empty if none.
chunk_local_gcc() {
    ls -d "$HOME"/opt/gcc-[0-9]*.[0-9]*.[0-9]* /opt/gcc-[0-9]*.[0-9]*.[0-9]* 2>/dev/null \
        | while read -r d; do [ -x "$d/bin/g++" ] && echo "$d"; done \
        | sort -t- -k2 -V | tail -1
}

if [ "$CHUNK_SITE" = cea ]; then
    : "${CHUNK_BUILD_REP:=$CCCWORKDIR/shrink-array}"
    : "${CHUNK_GCC_TOOLCHAIN_DIR:=/ccc/products/gcc-15.2.0/system/default}"
else
    : "${CHUNK_BUILD_REP:=$CHUNK_SRC_DIR}"
    # A GCC built from source is both the gnu compiler (CHUNK_GXX below) and
    # the libstdc++ of clang / icpx, as gcc-15.2.0 is at CEA. None: the
    # drivers' own default.
    : "${CHUNK_LOCAL_GCC_DIR:=$(chunk_local_gcc)}"
    : "${CHUNK_GCC_TOOLCHAIN_DIR:=$CHUNK_LOCAL_GCC_DIR}"
fi

# Environment Modules: absent on a workstation. A stand-in that fails quietly
# keeps every `module load x || ...` and `if run_cmd "module load x"` of the
# scripts on their "not available" path; the tools are then taken from PATH.
if type module >/dev/null 2>&1; then
    CHUNK_HAVE_MODULES=true
else
    CHUNK_HAVE_MODULES=false
    module() { return 1; }
fi

export CHUNK_SITE CHUNK_SRC_DIR CHUNK_BUILD_REP CHUNK_GCC_TOOLCHAIN_DIR CHUNK_LOCAL_GCC_DIR CHUNK_HAVE_MODULES CHUNK_SCRIPT_DIR

# chunk_newest <tool>: the highest versioned <tool>-N on PATH (g++-16,
# clang++-22), else <tool> itself -- what a workstation without modules has.
# A version N.0.0 (no point release yet: clang 23.0.0 in Sept. '26) is
# skipped by default: the calculators do not have it yet, and the CLion
# `clang` toolchain points at the newest N.x.y (clang-22). So is a GCC
# snapshot, "(experimental)" (the toolchain PPA's g++-16 16.0.1 of March '26).
# CHUNK_CLANGXX / CHUNK_GXX pick another one explicitly.
chunk_newest() {
    local tool=$1 cand
    for cand in $(compgen -c "$tool-" 2>/dev/null | grep -E "^${tool//+/\\+}-[0-9]+$" | sort -t- -k2 -n -r -u); do
        if ! "$cand" --version 2>/dev/null | head -1 | grep -qE 'version [0-9]+\.0\.0([^.0-9]|$)|\(trunk\)|\(experimental\)'; then
            echo "$cand"
            return
        fi
    done
    echo "$tool"
}

# Local compilers (the modules decide them at CEA): override with these.
if [ "$CHUNK_HAVE_MODULES" = false ]; then
    if [ -n "${CHUNK_LOCAL_GCC_DIR:-}" ]; then
        : "${CHUNK_GXX:=$CHUNK_LOCAL_GCC_DIR/bin/g++}"
    fi
    : "${CHUNK_GXX:=$(chunk_newest g++)}"
    : "${CHUNK_CLANGXX:=$(chunk_newest clang++)}"
    # oneAPI: its setvars.sh puts icpx on PATH (sourced by compile_all.sh).
    : "${CHUNK_ONEAPI_SETVARS:=/opt/intel/oneapi/setvars.sh}"
    # nvhpc: the newest x.y release under the usual SDK root.
    if [ -z "${CHUNK_NVCXX:-}" ]; then
        CHUNK_NVCXX=$(ls -d /opt/nvidia/hpc_sdk/Linux_"$(uname -m)"/[0-9]*.[0-9]*/compilers/bin/nvc++ 2>/dev/null | sort -V | tail -1)
    fi
    export CHUNK_GXX CHUNK_CLANGXX CHUNK_ONEAPI_SETVARS CHUNK_NVCXX
fi

# chunk_gcc_toolchain_opt <form>: the libstdc++ option for clang-like drivers,
# or nothing when no toolchain dir is set (local: the driver's default).
#   flags -> -DCMAKE_CXX_FLAGS=--gcc-toolchain=<dir>   (icpx, nvc++)
#   cmake -> -DGCC_TOOLCHAIN_DIR=<dir>                  (clang, aocc)
chunk_gcc_toolchain_opt() {
    [ -z "$CHUNK_GCC_TOOLCHAIN_DIR" ] && return 0
    case $1 in
        flags) echo "-DCMAKE_CXX_FLAGS=--gcc-toolchain=$CHUNK_GCC_TOOLCHAIN_DIR" ;;
        cmake) echo "-DGCC_TOOLCHAIN_DIR=$CHUNK_GCC_TOOLCHAIN_DIR" ;;
    esac
}

echo "[site] $CHUNK_SITE: sources $CHUNK_SRC_DIR, builds under $CHUNK_BUILD_REP, modules $CHUNK_HAVE_MODULES${CHUNK_GCC_TOOLCHAIN_DIR:+, libstdc++ from $CHUNK_GCC_TOOLCHAIN_DIR}"
