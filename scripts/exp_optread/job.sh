#!/bin/bash
# EXPERIMENT exp/optimistic-read: the read protocol's cost on Grace.
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository's scripts/exp_optread: ccc_msub job.sh
#MSUB -r optread
#MSUB -q gh200-bxi
#MSUB -N 1
#MSUB -n 1
#MSUB -c 288
#MSUB -T 10800
#MSUB -o optread_%I.log
#MSUB -e optread_%I.log

HERE=${BRIDGE_MSUB_PWD:-$PWD}
ccc_mprun -C ccc-rhel9 -n 1 -c 288 bash "$HERE/run.sh"
