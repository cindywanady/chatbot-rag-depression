#!/usr/bin/env python
"""Run the chunking pipeline from a source checkout (no install required).

Prepends ``src/`` to sys.path then delegates to the package CLI, so this works
whether or not ``pip install -e .`` has been run. Run from the project root:

    ./.venv/bin/python scripts/run_chunking.py
    ./.venv/bin/python scripts/run_chunking.py --scope MI.4 --only structure-512-0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from depression_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
