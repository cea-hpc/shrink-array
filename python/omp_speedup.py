#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""omp_speedup.py -- the former name of thread_speedup.py, kept for the commands
and job scripts that call it: the curves are not OpenMP's only (std::thread
backend, TBB bench). Runs thread_speedup.py with the same arguments; to be
removed in a later release."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread_speedup  # noqa: E402

if __name__ == "__main__":
    print("omp_speedup.py is now thread_speedup.py: running it", file=sys.stderr)
    sys.exit(thread_speedup.main())
