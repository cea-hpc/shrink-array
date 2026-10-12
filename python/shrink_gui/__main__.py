# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""python3 -m shrink_gui (from python/), or python3 python/shrink_gui."""
import os
import sys

if __package__ in (None, ""):          # run as a directory: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shrink_gui.sitecfg import SiteError  # noqa: E402

try:
    from shrink_gui.app import main  # noqa: E402  (reads the site file)
except SiteError as e:
    msg = "shrink_gui : fichier de site inutilisable\n%s" % e
    print(msg, file=sys.stderr)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        _app = QApplication(sys.argv)
        QMessageBox.critical(None, "shrink_gui", msg)
    except ImportError:
        pass
    sys.exit(2)

main()
