#!/usr/bin/env python
"""Relevance-denominator sensitivity check.

Confirms the retrieval SELECTION does not depend on the overlap-denominator choice
by scoring every index under two relevance rules and comparing the rankings:

  * primary     configs/relevance.yaml       denom=shorter  (size-robust; the pick)
  * sensitivity configs/relevance_gold.yaml  denom=gold     (answer-containment / DPR-style)

It runs the eval under the `gold` config into outputs/analysis/sensitivity_gold/
(reusing scripts/run_retrieval_eval.py so scoring is identical), then diffs that
against the primary outputs/analysis/retrieval_metrics.csv on the primary metric
and writes outputs/analysis/sensitivity_gold/sensitivity_report.md.

    python scripts/retrieval_sensitivity.py --device cuda           # re-score + compare
    python scripts/retrieval_sensitivity.py --skip-run              # compare existing csvs

Deterministic given the built indexes (see run_retrieval_eval.py).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pin the model cache like the build stages do (cli_index.py, cli.py). Without
# this the scoring stage resolves models against $HOME/.cache/huggingface and
# re-downloads ~12.7 GB into a second location.
os.environ.setdefault("HF_HOME", str((ROOT / ".hf_cache").resolve()))

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from depression_rag.evaluation.relevance import load_relevance_config

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"


def _load(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["index_id"]: r for r in csv.DictReader(fh)}


def _spearman(order_a: list[str], order_b: list[str]) -> float:
    """Spearman rank correlation between two rankings of the same items.

    Hand-rolled on purpose: scipy is an OPTIONAL dependency in this project
    (`retrieval_significance.py` guards its import; it is not in pyproject), and
    this script should not be the one that makes it mandatory. The closed form is
    exact here because both arguments are permutations — positions never tie.
    Note that ties in the underlying *metric values* still make the order within
    a tie arbitrary-but-deterministic (`sorted` is stable), which is one more
    reason the report leans on the tie-breakers rather than on rank 1 alone.
    """
    ra = {idx: i for i, idx in enumerate(order_a)}
    rb = {idx: i for i, idx in enumerate(order_b)}
    n = len(order_a)
    if n < 2:
        return float("nan")
    d2 = sum((ra[i] - rb[i]) ** 2 for i in order_a)
    return 1 - 6 * d2 / (n * (n * n - 1))


@dataclass
class View:
    """One comparison over a subset of indexes (the grid, or everything)."""

    label: str
    ids: list[str]
    order_p: list[str]
    order_g: list[str]
    rho: float
    mean_shift: float

    @property
    def winner_stable(self) -> bool:
        return self.order_p[0] == self.order_g[0]

    @property
    def top3_same(self) -> bool:
        return set(self.order_p[:3]) == set(self.order_g[:3])


def _view(label: str, ids: list[str], primary: dict, gold: dict, metric: str) -> View:
    val = lambda tbl, i: float(tbl[i][metric])  # noqa: E731
    order_p = sorted(ids, key=lambda i: val(primary, i), reverse=True)
    order_g = sorted(ids, key=lambda i: val(gold, i), reverse=True)
    shift = sum(abs(val(primary, i) - val(gold, i)) for i in ids) / len(ids)
    return View(label, ids, order_p, order_g, _spearman(order_p, order_g), shift)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retrieval-sensitivity", description=__doc__)
    ap.add_argument("--primary", default=str(ANALYSIS / "retrieval_metrics.csv"))
    ap.add_argument("--relevance-config", default=str(ROOT / "configs" / "relevance.yaml"),
                    help="source of selection_grid_models (the primary rule)")
    ap.add_argument("--gold-config", default=str(ROOT / "configs" / "relevance_gold.yaml"))
    ap.add_argument("--gold-out", default=str(ANALYSIS / "sensitivity_gold"))
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (for the re-score)")
    ap.add_argument("--top", type=int, default=6, help="rows to show in the side-by-side")
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse an existing sensitivity_gold/retrieval_metrics.csv")
    args = ap.parse_args(argv)

    gold_out = Path(args.gold_out)
    gold_csv = gold_out / "retrieval_metrics.csv"

    if not args.skip_run:
        gold_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_retrieval_eval.py"),
               "--relevance-config", args.gold_config, "--out-dir", str(gold_out)]
        if args.device:
            cmd += ["--device", args.device]
        print(f"[sensitivity] scoring under denom=gold: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    rel_primary = load_relevance_config(args.relevance_config)
    metric = load_relevance_config(args.gold_config).primary_metric
    primary = _load(Path(args.primary))
    gold = _load(gold_csv)
    common = [i for i in primary if i in gold]
    val = lambda tbl, i: float(tbl[i][metric])  # noqa: E731

    # Split the indexes the way the selection was actually made. Comparing all of
    # them lumps in models added AFTER the pick, and a post-selection challenger
    # taking nominal rank 1 is not the same claim as "the selection changed".
    grid_models = set(rel_primary.selection_grid_models)
    in_grid = [i for i in common if primary[i].get("model") in grid_models] if grid_models else []
    views = []
    if in_grid and len(in_grid) < len(common):
        views.append(_view(f"selection grid ({', '.join(sorted(grid_models))})",
                           in_grid, primary, gold, metric))
    views.append(_view("all scored indexes (incl. post-selection challengers)",
                       common, primary, gold, metric))
    headline = views[0]          # the selection claim is made on the first view

    lines = [f"# Relevance-denominator sensitivity — {metric}\n",
             "Does the selection depend on the overlap denominator? Two rules, same "
             "indexes / questions / metric:\n",
             "- **primary**: `denom=shorter` (size-robust; the selected pick)",
             "- **sensitivity**: `denom=gold` (answer-containment / DPR-style, ≥50% of gold span)\n"]

    if len(views) > 1:
        lines += ["\n> Reported on two index sets. The **selection grid** is the set of "
                  f"models present when the configuration was picked "
                  f"(`selection_grid_models` in `{Path(rel_primary.source_path).name}`); "
                  "**all scored indexes** additionally includes models added afterwards "
                  "as robustness challengers. A claim about *the selection* belongs to "
                  "the first; the second is disclosure.\n"]

    lines.append("\n## Headline\n")
    lines += ["| index set | n | winner unchanged | top-3 identical | Spearman ρ | mean \\|Δ\\| |",
              "|---|--:|:--:|:--:|--:|--:|"]
    for v in views:
        lines.append(f"| {v.label} | {len(v.ids)} | **{'yes' if v.winner_stable else 'NO'}** | "
                     f"{'yes' if v.top3_same else 'NO'} | {v.rho:.4f} | {v.mean_shift:.4f} |")
    lines.append("")
    for v in views:
        lines.append(f"- {v.label}: rank 1 = `{v.order_p[0]}` (shorter) vs "
                     f"`{v.order_g[0]}` (gold)")

    lines += [f"\n## Top {args.top} under each rule — {headline.label}\n",
              f"| rank | denom=shorter (primary) | {metric} | denom=gold | {metric} |",
              "|--:|---|--:|---|--:|"]
    for i in range(min(args.top, len(headline.ids))):
        a, b = headline.order_p[i], headline.order_g[i]
        lines.append(f"| {i+1} | {a} | {val(primary,a):.4f} | {b} | {val(gold,b):.4f} |")

    lines.append("\n## Conclusion\n")
    if headline.winner_stable:
        lines.append("The selected configuration is **invariant to the relevance-denominator "
                     "choice** on the grid it was selected from: the denominator moves absolute "
                     "scores and mid-table ordering, not the pick.")
    else:
        gap = val(gold, headline.order_g[0]) - val(gold, headline.order_p[0])
        lines.append(f"On the selection grid the nominal rank-1 differs under `denom=gold` "
                     f"(`{headline.order_g[0]}` edges `{headline.order_p[0]}` by {gap:.4f}). "
                     "Read it with the significance table: a nominal reordering inside a "
                     "statistically tied top group does not overturn a selection that rests "
                     "on the pre-stated tie-breakers.")
    for v in views[1:]:
        if not v.winner_stable:
            gap = val(gold, v.order_g[0]) - val(gold, v.order_p[0])
            lines.append(f"\n**Disclosure — {v.label}:** rank 1 changes here, "
                         f"`{v.order_g[0]}` ahead of `{v.order_p[0]}` by {gap:.4f}. That index "
                         "is not part of the selection grid, so this does not bear on the "
                         "pick; quote it alongside the sensitivity claim rather than omitting "
                         "it. See `documents/bge_m3_robustness_addendum.md`.")

    out = gold_out / "sensitivity_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
