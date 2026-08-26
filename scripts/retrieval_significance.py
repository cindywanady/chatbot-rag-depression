#!/usr/bin/env python
"""Paired significance between the top retrieval configs.

Consumes ``retrieval_scores_per_query.csv`` (written by run_retrieval_eval.py) and,
for the primary metric, tests whether the rank-1 index significantly beats each of
the next configs over the shared question set. Pairing on question_id removes
per-question difficulty as a nuisance. Two paired tests are reported side by side:

* **paired percentile bootstrap** on the per-query difference (Δ, 95% CI, p);
* **Wilcoxon signed-rank** (zero differences dropped, scipy defaults otherwise),
  with **Holm-adjusted** p-values across this table's comparisons — the
  family-wise correction — and the **rank-biserial** effect size.

    python scripts/retrieval_significance.py                 # winner vs next 5
    python scripts/retrieval_significance.py --top 8 --metric recall@5

Writes outputs/analysis/retrieval_significance.md. Does not re-run retrieval.
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
from collections import defaultdict
from pathlib import Path

from depression_rag.evaluation.bootstrap import (
    bootstrap_ci,
    holm_adjust,
    paired_bootstrap_diff,
    rank_biserial,
)
from depression_rag.evaluation.relevance import load_relevance_config

ROOT = Path(__file__).resolve().parents[1]


def _load_per_query(path: Path, metric: str) -> dict[str, dict[str, float]]:
    """index_id -> {question_id: value} for the chosen metric."""
    by_index: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_index[r["index_id"]][r["question_id"]] = float(r[metric])
    return by_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retrieval-significance", description=__doc__)
    ap.add_argument("--scores", default=str(ROOT / "outputs" / "analysis" / "retrieval_scores_per_query.csv"))
    ap.add_argument("--relevance-config", default=str(ROOT / "configs" / "relevance.yaml"))
    ap.add_argument("--metric", default=None, help="default: primary metric from relevance.yaml")
    ap.add_argument("--top", type=int, default=5, help="compare winner vs this many next configs")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "analysis" / "retrieval_significance.md"))
    args = ap.parse_args(argv)

    rel = load_relevance_config(args.relevance_config)
    metric = args.metric or rel.primary_metric
    by_index = _load_per_query(Path(args.scores), metric)

    means = {idx: sum(v.values()) / len(v) for idx, v in by_index.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    winner = ranked[0]
    w_scores = by_index[winner]
    qids = sorted(w_scores)  # fixed question order for pairing

    try:
        from scipy.stats import wilcoxon
    except ImportError:  # keep the bootstrap columns usable without scipy
        wilcoxon = None

    a = [w_scores[q] for q in qids]
    rows = []
    for idx in ranked[1: 1 + args.top]:
        b = [by_index[idx][q] for q in qids]
        t = paired_bootstrap_diff(a, b)
        diffs = [x - y for x, y in zip(a, b)]
        if wilcoxon is not None and any(d != 0 for d in diffs):
            p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        elif wilcoxon is not None:
            p_w = 1.0  # identical per-query values
        else:
            p_w = float("nan")
        rows.append((idx, t, p_w, rank_biserial(diffs)))
    p_holm = holm_adjust([r[2] for r in rows]) if wilcoxon is not None else [float("nan")] * len(rows)

    lines = [f"# Paired significance — {metric}\n",
             f"- winner (rank 1): **{winner}**  mean {metric} = {means[winner]:.4f}",
             "- **bootstrap**: paired percentile bootstrap on the per-query difference "
             "(10k resamples, seed 20260628); winner − challenger; p uncorrected.",
             "- **Wilcoxon**: signed-rank on the same pairs (zero diffs dropped, scipy "
             "defaults); **Holm** adjusts the Wilcoxon p across this table's "
             f"{len(rows)} comparisons; **r_rb** = rank-biserial effect size "
             "((W+ − W−)/(W+ + W−): +1 wins every pair, 0 symmetric).",
             "- significance claim = Holm-adjusted Wilcoxon p < 0.05 "
             "(family-wise error controlled across the comparisons).\n",
             "\n| challenger | mean | Δ (winner−chal.) | 95% CI of Δ | p boot | p Wilcoxon | p Holm | r_rb | sig? |",
             "|---|--:|--:|---|--:|--:|--:|--:|:--:|"]
    for (idx, t, p_w, rb), ph in zip(rows, p_holm):
        sig = "**yes**" if ph < 0.05 else "no"
        lines.append(f"| {idx} | {means[idx]:.4f} | {t.mean_diff:+.4f} | "
                     f"[{t.low:+.4f}, {t.high:+.4f}] | {t.p_value:.4f} | "
                     f"{p_w:.4f} | {ph:.4f} | {rb:+.3f} | {sig} |")
    if wilcoxon is None:
        lines.append("\n> scipy not installed — Wilcoxon/Holm columns are NaN; "
                     "only the (uncorrected) bootstrap columns are populated.")

    ci = bootstrap_ci(list(w_scores.values()))
    lines.append(f"\nWinner {metric}: {ci.mean:.4f}  (95% CI [{ci.low:.4f}, {ci.high:.4f}], n={ci.n}).")
    lines.append("\n> If the winner does not significantly beat the nearest challengers "
                 "(Holm-adjusted), the choice between them rests on the tie-breakers "
                 "(type robustness → index size / latency → build cost), not on the primary "
                 "metric alone. State that in the writeup.")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
