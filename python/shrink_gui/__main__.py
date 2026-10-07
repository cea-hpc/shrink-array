"""python3 -m shrink_gui (from python/), or python3 python/shrink_gui."""
import os
import sys

if __package__ in (None, ""):          # run as a directory: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shrink_gui.app import main  # noqa: E402

main()
