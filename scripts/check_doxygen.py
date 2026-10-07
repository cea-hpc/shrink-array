#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""The Doxygen documentation builds without a warning.

Runs Doxygen on docs/Doxyfile, as `cmake --build <build> --target doc` does,
but writes the output to a temporary directory instead of docs/html, and fails
if the warning log is not empty: an unknown command, a @param that names no
parameter, a broken reference or a Markdown construct Doxygen misreads.

Usage: scripts/check_doxygen.py [repo root] [doxygen executable]   (exit 1 on a warning)
"""

import pathlib
import subprocess
import sys
import tempfile


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    doxygen = sys.argv[2] if len(sys.argv) > 2 else "doxygen"
    docs = root / "docs"
    with tempfile.TemporaryDirectory() as out:
        log = pathlib.Path(out) / "warnings.log"
        config = (docs / "Doxyfile").read_text(encoding="utf-8")
        config += "\nOUTPUT_DIRECTORY = %s\nWARN_LOGFILE = %s\nQUIET = YES\n" % (out, log)
        run = subprocess.run([doxygen, "-"], input=config, text=True, cwd=docs, capture_output=True)
        if run.returncode != 0:
            print(run.stdout + run.stderr)
            print("check_doxygen: doxygen failed (exit %d)" % run.returncode)
            return 1
        warnings = log.read_text(encoding="utf-8", errors="replace").strip() if log.exists() else ""
    if warnings:
        print(warnings.replace(str(root) + "/", ""))
        print("check_doxygen: %d warning line(s)" % len(warnings.splitlines()))
        return 1
    print("check_doxygen: OK, no warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
