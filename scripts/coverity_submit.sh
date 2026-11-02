#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# Coverity Scan: capture a clean build with cov-build, then submit it to
# scan.coverity.com (project cea-hpc-shrink-array) in the three steps of the
# current upload flow -- builds/init, the PUT of the archive to the signed
# Google Storage URL, builds/<id>/enqueue. Without the enqueue the upload
# succeeds and nothing is analysed.
#
# Usage: scripts/coverity_submit.sh [--capture-only | --submit-only]
#   --capture-only  build cov-int/ and cov-int.tgz, submit nothing
#   --submit-only   submit the cov-int.tgz already in COV_BUILD_DIR
#
# Environment:
#   COVERITY_TOKEN  the project token (scan.coverity.com, Project Settings);
#                   TOKEN is read if COVERITY_TOKEN is unset. Never written.
#   COVERITY_EMAIL  where the result mail goes (default: git config user.email)
#   COV_BIN         cov-analysis bin/ directory (default: the 2024.12.1 tree
#                   under /tmp/ballereaup/coverity, unpacked from
#                   ~/Downloads/cov-analysis-linux64-2024.12.1.tar.gz if absent).
#                   Coverity Scan only accepts cov-build 2024.12.1 or 2024.6.1.
#   COV_BUILD_DIR   the build tree (default: /tmp/$USER/build-cov; emptied)
#   GCC_MODULE      compiler module, where modules exist (default: gcc/gcc-14.2.0-x86_64)
#   CXX             the C++ compiler (default: the g++ on PATH after the module)
#   CMAKE_ARGS      extra CMake arguments (default: -DOMP=ON, size 256)
#   JOBS            build parallelism (default: 2)
#
# The proxy is the shell's (https_proxy): scan.coverity.com and
# storage.googleapis.com go through it. .github/workflows/coverity.yml runs
# this script weekly (COV_BIN from the download, CXX=g++-14).
set -euo pipefail

PROJECT=cea-hpc-shrink-array
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODE=all
case "${1:-}" in
    --capture-only) MODE=capture ;;
    --submit-only) MODE=submit ;;
    "") ;;
    *) sed -n '4,30p' "$0"; exit 2 ;;
esac

COV_BUILD_DIR=${COV_BUILD_DIR:-/tmp/$USER/build-cov}
GCC_MODULE=${GCC_MODULE:-gcc/gcc-14.2.0-x86_64}
CMAKE_ARGS=${CMAKE_ARGS:--DOMP=ON -DCHUNK_STATIC_SIZES=256 -DCHUNK_BUILD_DYN=OFF}
JOBS=${JOBS:-2}

die() { echo "coverity_submit: $*" >&2; exit 1; }

capture() {
    local cov_root=/tmp/$USER/coverity/cov-analysis-linux64-2024.12.1
    local tarball=$HOME/Downloads/cov-analysis-linux64-2024.12.1.tar.gz
    if [ -z "${COV_BIN:-}" ]; then
        if [ ! -x "$cov_root/bin/cov-build" ]; then
            [ -f "$tarball" ] || die "no cov-analysis: set COV_BIN, or download $tarball from scan.coverity.com"
            echo "unpacking $tarball"
            mkdir -p "$(dirname "$cov_root")"
            tar xzf "$tarball" -C "$(dirname "$cov_root")"
        fi
        COV_BIN=$cov_root/bin
    fi
    export PATH=$COV_BIN:$PATH
    cov-build --ident | head -3

    # A clean tree: cov-build only sees what it watches being compiled.
    # (\rm: rm may be aliased to rm -i in the login shell.)
    \rm -rf "$COV_BUILD_DIR"
    # shellcheck disable=SC1091
    if command -v module > /dev/null 2>&1 || [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh 2> /dev/null || true
        module load "$GCC_MODULE"
    fi
    # shellcheck disable=SC2086
    cmake -S "$REPO" -B "$COV_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${CXX:-$(command -v g++)}" $CMAKE_ARGS
    cd "$COV_BUILD_DIR"
    nice -n 19 cov-build --dir cov-int cmake --build . -j"$JOBS"
    tail -3 cov-int/build-log.txt
    grep -q 'compilation units (100%) are ready for analysis' cov-int/build-log.txt ||
        die "not every compilation unit was captured: see $COV_BUILD_DIR/cov-int/build-log.txt"
    tar czf cov-int.tgz cov-int
    ls -l cov-int.tgz
}

submit() {
    local token=${COVERITY_TOKEN:-${TOKEN:-}}
    local email=${COVERITY_EMAIL:-$(git -C "$REPO" config user.email || true)}
    local version desc resp code
    [ -n "$token" ] || die "COVERITY_TOKEN (or TOKEN) is empty"
    [ -n "$email" ] || die "COVERITY_EMAIL is empty"
    command -v jq > /dev/null || die "jq is needed"
    cd "$COV_BUILD_DIR"
    [ -f cov-int.tgz ] || die "no $COV_BUILD_DIR/cov-int.tgz: run the capture first"
    version=$(sed -n 's/^project(shrink_array VERSION \([0-9.]*\).*/\1/p' "$REPO/CMakeLists.txt")
    desc="shrink-array $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"

    # 1. A new build. Refused while an earlier one is still queued.
    resp=$(curl -sS -X POST --data-urlencode "token=$token" --data-urlencode "email=$email" \
        -d file_name=cov-int.tgz --data-urlencode "version=$version" --data-urlencode "description=$desc" \
        "https://scan.coverity.com/projects/$PROJECT/builds/init")
    echo "$resp" | jq -e .build_id > /dev/null 2>&1 || die "builds/init refused: $resp"
    echo "$resp" > resp.json # in the build tree, never in the repository
    echo "build $(jq -r .build_id resp.json): $(jq -r .info resp.json)"

    # 2. The archive to the signed URL (jq turns the & back into &).
    # curl alone exits 0 on an HTTP error: the code is checked.
    code=$(curl -sS -X PUT --header 'Content-Type: application/json' --upload-file cov-int.tgz \
        -o /dev/null -w '%{http_code}' "$(jq -r .url resp.json)")
    [ "$code" = 200 ] || die "upload: HTTP $code"
    echo "upload: HTTP 200"

    # 3. Into the analysis queue.
    resp=$(curl -sS -X PUT --data-urlencode "token=$token" \
        "https://scan.coverity.com/projects/$PROJECT/builds/$(jq -r .build_id resp.json)/enqueue")
    echo "$resp" | jq -e .id > /dev/null 2>&1 || die "enqueue refused: $resp"
    echo "enqueued: $resp -- the result mail goes to $email"
}

case $MODE in
    capture) capture ;;
    submit) submit ;;
    all) capture; submit ;;
esac
