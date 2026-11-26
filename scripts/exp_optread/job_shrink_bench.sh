#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# EXPERIMENT exp/optimistic-read: shrink_bench in modes 0 and 7 on Grace.
# Submit from the repository's scripts/exp_optread: ccc_msub job_shrink_bench.sh
# Defaults (shrink_bench.sh): gcc/llvm x mode 0/7, 3 variants, 8 and 72 threads,
# 3 rounds = 72 runs after 4 builds. Narrow with CONFIGS / VARIANTS / THREADS /
# ROUNDS exported before ccc_msub if the time limit is short.
#MSUB -r optread_sb
#MSUB -q gh200-bxi
#MSUB -N 1
#MSUB -n 1
#MSUB -c 288
#MSUB -T 10800
#MSUB -o optread_sb_%I.log
#MSUB -e optread_sb_%I.log

HERE=${BRIDGE_MSUB_PWD:-$PWD}
ccc_mprun -C ccc-rhel9 -n 1 -c 288 bash "$HERE/shrink_bench.sh"
