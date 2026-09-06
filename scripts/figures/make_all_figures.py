#!/usr/bin/env python3
"""Regenerate every manuscript figure that is produced from data.

    python scripts/figures/make_all_figures.py

Figures 1 and 2 are hand-built diagrams and live in
documents/figures/figure1_workflow_20260831/ and figure2_allocation_20260903/;
they are not regenerated here. Figures 3 to 6 are plotted from the analysis
artifacts and are written to documents/figures/manuscript/ as PNG (300 dpi, for
submission) and PDF (vector, for the thesis).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [
    "fig3_retrieval_by_type.py",
    "fig4_grounding_agreement.py",
    "fig5_psychologist_domains.py",
    "fig6_paired_responses.py",
]

if __name__ == "__main__":
    failed = []
    for name in SCRIPTS:
        print(f"\n=== {name}")
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAILED: {exc}")
    if failed:
        print(f"\n{len(failed)} figure(s) failed")
        sys.exit(1)
    print("\nall figures regenerated")
