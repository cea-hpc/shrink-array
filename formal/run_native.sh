#!/usr/bin/env bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# formal/run_native.sh <executable> -- run a natively compiled model SHRINK_FORMAL_RUNS times
# (default 300). A model aborts on a violated property; exit 1 at the first abort.
set -u
exe=$1
runs=${SHRINK_FORMAL_RUNS:-300}
for ((i = 1; i <= runs; i++)); do
    if ! "$exe"; then
        echo "formal: $exe failed at run $i of $runs"
        exit 1
    fi
done
echo "formal: $exe, $runs runs, no violation"
