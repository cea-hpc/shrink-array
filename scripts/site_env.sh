#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
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
#                             cea   $CCCWORKDIR/shrink-array (the TGCC work area)
#                             local the repository itself (as CLion does)
#   CHUNK_GCC_TOOLCHAIN_DIR the libstdc++ that clang / icpx / nvc++ use:
#                             cea   /ccc/products/gcc-15.2.0/system/default
#                             local the newest ~/opt/gcc-X.Y.Z or /opt/gcc-X.Y.Z
#                                   built from source (CHUNK_LOCAL_GCC_DIR,
#                                   also the gnu compiler), else empty (the
#                                   compilers' own default)
#   CHUNK_HAVE_MODULES      true when Environment Modules is available
#   TBBROOT                 a oneTBB for the std builds' TBB bench, when one is
#                           found (kept if already set): CHUNK_TBB_DIR, else
#                             cea   $CCCWORKDIR/tbb-<arch>, $CCCWORKDIR/tbb/<arch>,
#                                   $CCCWORKDIR/tbb (<arch> = uname -m)
#                             local ~/opt/tbb-<arch>, ~/opt/tbb*, ~/opt/oneTBB*, /opt/tbb*
#                           and only a libtbb built for this architecture
#                           exported with its CMAKE_PREFIX_PATH and
#                           LD_LIBRARY_PATH entries; CHUNK_WITH_TBB=0: no search
#
# Each can be set in the environment first to override. The build names
# (cmake-build-<name>/<arch>, arch = ID_VERSION__machine from os-release) are
# the same on every site: scripts/gen_clion_profiles.py derives the CLion
# profiles from them.
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHUNK_SRC_DIR:=$(cd "$CHUNK_SCRIPT_DIR/.." && pwd)}"

if [ -z "${CHUNK_SITE:-}" ]; then
    # The GCC toolchain, not the bare /ccc/products: a workstation can carry an
    # empty /ccc/products (and Environment Modules without any compiler).
    if [ -d /ccc/products/gcc-15.2.0 ]; then CHUNK_SITE=cea; else CHUNK_SITE=local; fi
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
    # Not under the sources in ~ (quota): stop rather than guess when the TGCC
    # environment did not set CCCWORKDIR.
    : "${CHUNK_BUILD_REP:=${CCCWORKDIR:?site_env.sh: CCCWORKDIR unset, set CHUNK_BUILD_REP}/shrink-array}"
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
# A local site never takes its compilers from modules, even where the command
# exists (AlmaLinux ships Environment Modules with no compiler module).
if [ "$CHUNK_SITE" = cea ] && type module >/dev/null 2>&1; then
    CHUNK_HAVE_MODULES=true
else
    CHUNK_HAVE_MODULES=false
    module() { return 1; }
fi

export CHUNK_SITE CHUNK_SRC_DIR CHUNK_BUILD_REP CHUNK_GCC_TOOLCHAIN_DIR CHUNK_LOCAL_GCC_DIR CHUNK_HAVE_MODULES CHUNK_SCRIPT_DIR

# chunk_module_load_first <module>...: load the first module of the list that
# exists; the last one is loaded unconditionally (the fallback, e.g. the bare
# default `cmake`). A missing module is never TRIED: `module load cmake/4.0.3`
# on a machine without it (the ARM calculators) printed "Unable to locate a
# modulefile" even with 2>/dev/null, the module command reporting through the
# shell it evaluates into. `module is-avail` answers silently (Environment
# Modules >= 4.1, Lmod); `module -t avail` is the fallback for older ones.
# Without modules (workstation) the stand-in above fails every load quietly.
chunk_module_load_first() {
    local m
    for m in "${@:1:$#-1}"; do
        if module is-avail "$m" >/dev/null 2>&1 ||
            module -t avail "$m" 2>&1 | grep -qE "^${m//+/\\+}(\(.*\))?$"; then
            module load "$m"
            return
        fi
    done
    module load "${@: -1}"
}

# chunk_newest <tool>: the highest versioned <tool>-N on PATH (g++-16,
# clang++-22), else <tool> itself -- what a workstation without modules has.
# A version N.0.0 (no point release yet, e.g. clang 23.0.0) is
# skipped by default: the calculators do not have it yet, and the CLion
# `clang` toolchain points at the newest N.x.y (clang-22). So is a GCC
# snapshot, "(experimental)" (e.g. the toolchain PPA's g++-16 16.0.1).
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
    # RHEL-likes: the newest gcc-toolset before the system g++.
    if [ -z "${CHUNK_GXX:-}" ]; then
        CHUNK_GXX=$(ls -d /opt/rh/gcc-toolset-*/root/usr/bin/g++ 2>/dev/null | sort -V | tail -1)
        [ -n "$CHUNK_GXX" ] || CHUNK_GXX=$(chunk_newest g++)
    fi
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

# ── oneTBB ───────────────────────────────────────────────────────────────────
# The TBB bench of the std builds needs a oneTBB that CMake finds (TBBConfig.cmake)
# and that the programs load at run time. A user-installed one, under a path
# that names its user, is found here from the environment rather than written
# in the repository. TBBROOT set beforehand (vars.sh, spack load) wins.
# chunk_tbb_arch_ok <libdir>: its libtbb.so is an ELF of this machine.
# $CCCWORKDIR is shared by Grace (aarch64) and Milan (x86_64): a oneTBB built
# on one is visible on the other, and ld refuses it ("file in wrong format").
chunk_tbb_arch_ok() {
    local lib want got
    lib=$(ls "$1"/libtbb.so* 2>/dev/null | head -1)
    [ -n "$lib" ] || return 1
    case "$(uname -m)" in
        x86_64)  want=3e ;;
        aarch64) want=b7 ;;
        *)       return 0 ;;   # no check elsewhere
    esac
    got=$(od -An -tx1 -j18 -N1 "$lib" 2>/dev/null | tr -d ' \n')
    [ "$got" = "$want" ]
}

chunk_find_tbb() {
    local d l m
    m=$(uname -m)
    for d in "${CHUNK_TBB_DIR:-}" \
             ${CCCWORKDIR:+"$CCCWORKDIR/tbb-$m" "$CCCWORKDIR/tbb/$m" "$CCCWORKDIR/tbb"} \
             "$HOME/opt/tbb-$m" "$HOME"/opt/tbb* "$HOME"/opt/oneTBB* /opt/tbb*; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        for l in lib64 lib; do
            [ -f "$d/$l/cmake/TBB/TBBConfig.cmake" ] || continue
            ls "$d/$l"/libtbb.so* >/dev/null 2>&1 || continue
            if chunk_tbb_arch_ok "$d/$l"; then
                echo "$d"
                return 0
            fi
            echo "[site] oneTBB $d: not built for $m, skipped" >&2
        done
    done
    return 1
}

# chunk_path_prepend <var> <dir>: <dir> in front of the colon list <var>, once.
chunk_path_prepend() {
    local var=$1 dir=$2
    case ":${!var:-}:" in
        *":$dir:"*) ;;
        *) export "$var=$dir${!var:+:${!var}}" ;;
    esac
}

if [ "${CHUNK_WITH_TBB:-}" != 0 ]; then
    if [ -z "${TBBROOT:-}" ]; then
        TBBROOT=$(chunk_find_tbb) || TBBROOT=""
    fi
    if [ -n "$TBBROOT" ]; then
        export TBBROOT
        chunk_path_prepend CMAKE_PREFIX_PATH "$TBBROOT"
        for _chunk_tbb_lib in lib64 lib; do
            [ -d "$TBBROOT/$_chunk_tbb_lib" ] && chunk_path_prepend LD_LIBRARY_PATH "$TBBROOT/$_chunk_tbb_lib"
        done
        unset _chunk_tbb_lib
    else
        unset TBBROOT
    fi
fi

echo "[site] $CHUNK_SITE: sources $CHUNK_SRC_DIR, builds under $CHUNK_BUILD_REP, modules $CHUNK_HAVE_MODULES${CHUNK_GCC_TOOLCHAIN_DIR:+, libstdc++ from $CHUNK_GCC_TOOLCHAIN_DIR}${TBBROOT:+, oneTBB $TBBROOT}"
