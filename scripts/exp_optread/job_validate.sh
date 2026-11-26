#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# EXPERIMENT exp/optimistic-read: mode 7's validations on Grace (validate.sh).
# Submit from the repository's scripts/exp_optread: ccc_msub job_validate.sh
# Defaults: A = every ctest validation, check build, mode 7, gcc and llvm;
# B = read_reclaim_validation plain / bulk / compound under the read delay,
# modes 7 and 3 (control), 3 variants, 8 and 72 threads, 20 s per run.
# Narrow with PARTS / COMPS / THREADS / SECONDS_PER_RUN exported before ccc_msub.
#MSUB -r optread_val
#MSUB -q gh200-bxi
#MSUB -N 1
#MSUB -n 1
#MSUB -c 288
#MSUB -T 10800
#MSUB -o optread_val_%I.log
#MSUB -e optread_val_%I.log

HERE=${BRIDGE_MSUB_PWD:-$PWD}
ccc_mprun -C ccc-rhel9 -n 1 -c 288 bash "$HERE/validate.sh"
