#!/usr/bin/env python
"""Is the multi-vector variant an unfair advantage handed only to the winner?

The deployed index is `structure-512-0__e5-large__mv`: the selected configuration
with oversize chunks embedded as several <=cap windows instead of being clipped.
No other model has an `__mv` index, which invites a fair question — *you applied a
representation fix to your winner and not to the models it beat*.

This probe answers it by measurement instead of argument. It:

  1. reads each model's single-vector index meta.json for the target chunk config
     and splits the models into "cap exceeded" and "no-op" (a model whose longest
     chunk fits inside max_seq_len would get a byte-identical __mv index);
  2. builds the missing __mv variants for the cap-limited models;
  3. scores every single/mv pair on the same 124-question gold set;
  4. reports, paired over queries: what mv buys each model, and whether the best
     mv challenger overtakes the selected configuration.

Everything is written to --out, NEVER to outputs/indexes/ — run_retrieval_eval.py
discovers every directory it finds, so leaving probe indexes in the live tree
would silently fold them into the published results table.

    ./.venv/bin/python scripts/mv_fairness_probe.py --out /tmp/mv_probe --device cpu

Result as of 2026-07-26 is written up in documents/multi_vector_fairness_probe.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse run_retrieval_eval as a module

from depression_rag.evaluation.bootstrap import (  # noqa: E402
    paired_bootstrap_diff,
    rank_biserial,
)


def _meta(index_dir: Path) -> dict:
    return json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))


def classify_models(index_root: Path, chunk_config: str) -> tuple[list[str], list[str]]:
    """-> (models whose chunks exceed the cap, models for which __mv is a no-op)."""
    exceeded, noop = [], []
    for d in sorted(index_root.glob(f"{chunk_config}__*")):
        if not d.is_dir() or d.name.endswith("__mv"):
            continue
        m = _meta(d)
        (exceeded if m["truncated_chunks"] else noop).append(m["model"])
    return exceeded, noop


def build_missing_mv(models: list[str], chunk_config: str, out: Path, device: str) -> None:
    """Build __mv indexes for `models` into `out`, using the project model specs."""
    from depression_rag.cli_index import main as index_main

    cfg = yaml.safe_load((ROOT / "configs" / "embedding_models.yaml").read_text(encoding="utf-8"))
    cfg["output"]["index_dir"] = str(out / "indexes")
    cfg["output"]["manifests_dir"] = str(out / "manifests")
    cfg["output"]["logs_dir"] = str(out / "logs")
    probe_cfg = out / "embedding_models.yaml"
    probe_cfg.parent.mkdir(parents=True, exist_ok=True)
    probe_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    argv = ["--config", str(probe_cfg), "--models", *models,
            "--configs", chunk_config, "--multi-vector", "--run-id", "mv-probe"]
    if device:
        argv += ["--device", device]
    index_main(argv)


def copy_baselines(index_root: Path, out: Path, chunk_config: str, models: list[str]) -> None:
    """Copy the existing single-vector indexes (and any existing __mv) in, so every
    index in the comparison is scored by the same run."""
    for model in models:
        for suffix in ("", "__mv"):
            src = index_root / f"{chunk_config}__{model}{suffix}"
            dst = out / "indexes" / src.name
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)


def score(out: Path, device: str) -> None:
    from run_retrieval_eval import main as eval_main  # noqa: PLC0415

    argv = ["--index-dir", str(out / "indexes"), "--out-dir", str(out / "analysis")]
    if device:
        argv += ["--device", device]
    eval_main(argv)


def per_query(path: Path, metric: str) -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["index_id"], {})[r["question_id"]] = float(r[metric])
    return by


def paired_line(pq: dict, a: str, b: str, label: str) -> str:
    qids = sorted(pq[a])
    x = [pq[a][q] for q in qids]
    y = [pq[b][q] for q in qids]
    t = paired_bootstrap_diff(x, y)
    diffs = [u - v for u, v in zip(x, y)]
    try:
        from scipy.stats import wilcoxon
        p = f"{float(wilcoxon(x, y).pvalue):.4g}" if any(diffs) else "1"
    except ImportError:
        p = "n/a"
    return (f"  {label:46s} Δ={t.mean_diff:+.4f}  CI [{t.low:+.4f},{t.high:+.4f}]  "
            f"p={p:>9s}  r_rb={rank_biserial(diffs):+.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="scratch directory (never outputs/indexes/)")
    ap.add_argument("--chunk-config", default="structure-512-0", help="the selected config")
    ap.add_argument("--metric", default="ndcg@5")
    ap.add_argument("--device", default="cpu", help="auto|cuda|cpu")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if out.is_relative_to(ROOT / "outputs" / "indexes"):
        raise SystemExit("--out must not point inside outputs/indexes/ (see the module docstring)")
    index_root = ROOT / "outputs" / "indexes"

    exceeded, noop = classify_models(index_root, args.chunk_config)
    print(f"[probe] chunk config: {args.chunk_config}")
    print(f"[probe] __mv is a NO-OP for (nothing exceeds the cap): {noop or '—'}")
    print(f"[probe] cap-limited, __mv is meaningful for:           {exceeded}")

    missing = [m for m in exceeded
               if not (index_root / f"{args.chunk_config}__{m}__mv").is_dir()]
    print(f"[probe] building missing __mv variants: {missing or '—'}")
    if missing:
        build_missing_mv(missing, args.chunk_config, out, args.device)
    copy_baselines(index_root, out, args.chunk_config, exceeded)
    score(out, args.device)

    metrics = {r["index_id"]: r for r in
               csv.DictReader(open(out / "analysis" / "retrieval_metrics.csv", encoding="utf-8"))}
    pq = per_query(out / "analysis" / "retrieval_scores_per_query.csv", args.metric)

    print(f"\n=== all indexes, ranked by {args.metric} ===")
    for i in sorted(metrics, key=lambda x: -float(metrics[x][args.metric])):
        r = metrics[i]
        print(f"  {i.replace(args.chunk_config + '__', ''):26s} {args.metric}={float(r[args.metric]):.4f}  "
              f"recall@5={float(r['recall@5']):.4f}  recall@10={float(r['recall@10']):.4f}")

    print(f"\n=== what mv buys each model (mv − single, paired over queries) ===")
    for m in exceeded:
        a, b = f"{args.chunk_config}__{m}__mv", f"{args.chunk_config}__{m}"
        if a in pq and b in pq:
            print(paired_line(pq, a, b, f"{m}: mv − single"))

    # does the best challenger overtake the selected configuration?
    best = max((i for i in metrics if not i.endswith("__e5-large") and not i.endswith("__e5-large__mv")),
               key=lambda i: float(metrics[i][args.metric]))
    print(f"\n=== does the best non-winner overtake the selection? (best = {best}) ===")
    for sel in (f"{args.chunk_config}__e5-large", f"{args.chunk_config}__e5-large__mv"):
        if sel in pq:
            print(paired_line(pq, sel, best, f"{sel.replace(args.chunk_config + '__', '')} − {best.replace(args.chunk_config + '__', '')}"))
    print(f"\n[probe] artifacts in {out} — delete when done; they are deliberately "
          "outside outputs/indexes/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
