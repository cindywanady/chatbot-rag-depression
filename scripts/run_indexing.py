#!/usr/bin/env python
"""Run the embedding + indexing pipeline from a source checkout (no install).

    ./.venv/bin/python scripts/run_indexing.py
    ./.venv/bin/python scripts/run_indexing.py --models minilm --configs structure-512-0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from depression_rag.cli_index import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
