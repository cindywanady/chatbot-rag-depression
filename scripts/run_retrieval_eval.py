#!/usr/bin/env python
"""Run the retrieval evaluation over every built index.

For each index (chunk_config × embedding model) this:
  1. loads the index + its chunk spans and builds the matching embedder,
  2. retrieves top-max(k) chunks for every gold question,
  3. maps relevance at the source-span level (frozen threshold, configs/relevance.yaml)
     and computes Recall@k / MRR@k / nDCG@k, overall and per question_type,
  4. writes a per-index row + a per-(index,type) breakdown.

    python scripts/run_retrieval_eval.py                      # all indexes
    python scripts/run_retrieval_eval.py --models e5-large --device cpu
    python scripts/run_retrieval_eval.py --indexes fixed-256-0__e5-large

Outputs (default outputs/analysis/):
    retrieval_metrics.csv        one row per index (overall means)
    retrieval_metrics_by_type.csv        one row per (index, question_type)
    retrieval_metrics_by_difficulty.csv  one row per (index, difficulty)
    retrieval_eval_report.md     ranked table on the primary metric + selection note

Scoring is deterministic given the built indexes; query embedding reuses the exact
embedder each index was built with, so this does not rebuild indexes.
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
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path

from depression_rag.evaluation import (
    build_gold_passages,
    load_questions,
    load_relevance_config,
    mean_scores,
    score_query,
)
from depression_rag.evaluation.bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED, bootstrap_ci
from depression_rag.observability import ManifestBuilder
from depression_rag.evaluation.relevance import gold_spans_for, relevant_chunk_ids
from depression_rag.evaluation.retrieval import (
    RetrievalRunner,
    build_embedder_for_model,
)

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def _discover_indexes(index_dir: Path) -> list[Path]:
    return sorted(p for p in index_dir.iterdir() if p.is_dir() and (p / "index.faiss").exists())


def _gold_by_id(gp_result) -> dict[str, tuple[int, int]]:
    return {p.passage_id: (p.char_start, p.char_end) for p in gp_result.passages}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run-retrieval-eval", description=__doc__)
    ap.add_argument("--embedding-config", default=str(ROOT / "configs" / "embedding_models.yaml"))
    ap.add_argument("--relevance-config", default=str(ROOT / "configs" / "relevance.yaml"))
    ap.add_argument("--questions", default=str(DERIVED / "questions_gold.jsonl"))
    ap.add_argument("--index-dir", default=str(ROOT / "outputs" / "indexes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "analysis"))
    ap.add_argument("--models", nargs="+", help="restrict to these embedding models")
    ap.add_argument("--indexes", nargs="+", help="restrict to these index ids")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu")
    ap.add_argument("--manifests-dir", default=str(ROOT / "outputs" / "manifests"))
    ap.add_argument("--run-id", default=None, help="default: eval-<timestamp>")
    args = ap.parse_args(argv)

    rel = load_relevance_config(args.relevance_config)
    k_max = max(rel.k_values)

    gold_cfg = ROOT / "configs" / "gold_passages.yaml"
    segments_path, cleaned_path = DERIVED / "segments.jsonl", DERIVED / "cleaned_text.txt"
    gp = build_gold_passages(gold_cfg, segments_path, cleaned_path)

    # Every input that can move a published number, hashed. Without this the
    # results tables were the only stage output in the project with no provenance
    # record at all: no seed, no index hashes, no package versions.
    run_id = args.run_id or _dt.datetime.now().strftime("eval-%Y%m%d-%H%M%S")
    manifest = ManifestBuilder(run_id=run_id, description="retrieval evaluation")
    for label, path in (("questions", args.questions),
                        ("relevance_config", args.relevance_config),
                        ("embedding_config", args.embedding_config),
                        ("gold_passages_config", gold_cfg),
                        ("segments", segments_path),
                        ("cleaned_text", cleaned_path)):
        manifest.add_input(label, path)
    manifest.set_seed({"bootstrap_seed": DEFAULT_SEED, "n_boot": DEFAULT_N_BOOT})
    gold_by_id = _gold_by_id(gp)
    questions = load_questions(args.questions)
    print(f"[eval] {len(questions)} questions, {len(gold_by_id)} gold passages, "
          f"threshold={rel.denom} ≥{rel.min_overlap_frac}, k={rel.k_values}")

    index_dirs = _discover_indexes(Path(args.index_dir))
    if args.indexes:
        want = set(args.indexes)
        index_dirs = [p for p in index_dirs if p.name in want]

    embedder_cache: dict[str, object] = {}
    overall_rows: list[dict] = []
    by_type_rows: list[dict] = []
    by_difficulty_rows: list[dict] = []
    per_query_rows: list[dict] = []

    for idx_path in index_dirs:
        meta = json.loads((idx_path / "meta.json").read_text(encoding="utf-8"))
        model = meta["model"]
        chunk_cfg = meta["chunk_config_id"]
        if args.models and model not in args.models:
            continue
        # the two files that decide this index's scores: the vectors, and the
        # char spans the relevance mapping projects gold passages onto
        manifest.add_input(f"index:{idx_path.name}", idx_path / "index.faiss")
        manifest.add_input(f"chunks:{idx_path.name}", idx_path / "chunks.jsonl")

        if model not in embedder_cache:
            print(f"[eval] loading embedder: {model}")
            embedder_cache[model] = build_embedder_for_model(
                model, args.embedding_config, device=args.device
            )
        runner = RetrievalRunner(idx_path, embedder_cache[model])

        results = {r.question_id: r for r in runner.search(questions, k_max)}

        scores = []
        for q in questions:
            g_all = gold_spans_for(q["passage_ids"], gold_by_id)
            qrels = relevant_chunk_ids(runner.chunk_spans, g_all, rel)
            # per-passage qrels for the coverage metric; always computed so every
            # query row has the same metric keys (single-passage coverage == recall)
            per_passage = [relevant_chunk_ids(runner.chunk_spans, [g], rel) for g in g_all]
            ranked = results[q["question_id"]].ranked_chunk_ids
            scores.append(score_query(
                q["question_id"], q["question_type"], q["difficulty"],
                ranked, qrels, rel.k_values, relevant_per_passage=per_passage,
            ))

        overall = mean_scores(scores)
        # bootstrap 95% CI on the primary metric, reported next to every mean
        ci = bootstrap_ci([s.values[rel.primary_metric] for s in scores])
        overall_rows.append({"index_id": idx_path.name, "chunk_config": chunk_cfg,
                             "model": model, "n_queries": len(scores), **overall,
                             f"{rel.primary_metric}_ci_low": ci.low,
                             f"{rel.primary_metric}_ci_high": ci.high})

        # per-query dump (long-lived: enables paired significance tests without re-running)
        for s in scores:
            per_query_rows.append({"index_id": idx_path.name, "model": model,
                                   "chunk_config": chunk_cfg, "question_id": s.question_id,
                                   "question_type": s.question_type, "difficulty": s.difficulty,
                                   **s.values})

        by_type: dict[str, list] = defaultdict(list)
        for s in scores:
            by_type[s.question_type].append(s)
        for qtype, group in sorted(by_type.items()):
            by_type_rows.append({"index_id": idx_path.name, "model": model,
                                 "chunk_config": chunk_cfg, "question_type": qtype,
                                 "n": len(group), **mean_scores(group)})

        by_diff: dict[str, list] = defaultdict(list)
        for s in scores:
            by_diff[s.difficulty].append(s)
        for diff, group in sorted(by_diff.items()):
            by_difficulty_rows.append({"index_id": idx_path.name, "model": model,
                                       "chunk_config": chunk_cfg, "difficulty": diff,
                                       "n": len(group), **mean_scores(group)})

        prim = overall.get(rel.primary_metric, float("nan"))
        print(f"[eval] {idx_path.name:40s} {rel.primary_metric}={prim:.4f} "
              f"[{ci.low:.4f}, {ci.high:.4f}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _write_csv(out_dir / "retrieval_metrics.csv", overall_rows),
        _write_csv(out_dir / "retrieval_metrics_by_type.csv", by_type_rows),
        _write_csv(out_dir / "retrieval_metrics_by_difficulty.csv", by_difficulty_rows),
        _write_csv(out_dir / "retrieval_scores_per_query.csv", per_query_rows),
        _write_report(out_dir / "retrieval_eval_report.md", overall_rows, rel, len(questions)),
    ]

    manifest.record("n_questions", len(questions))
    manifest.record("n_indexes_scored", len(overall_rows))
    manifest.record("relevance", {"denom": rel.denom, "min_overlap_frac": rel.min_overlap_frac,
                                  "k_values": list(rel.k_values), "graded": rel.graded,
                                  "primary_metric": rel.primary_metric})
    manifest.record("n_gold_passages", len(gold_by_id))
    manifest.record(rel.primary_metric,
                    {r["index_id"]: r.get(rel.primary_metric) for r in overall_rows})
    for p in written:
        manifest.add_output(p)
    manifest_path = manifest.write(Path(args.manifests_dir) / f"{run_id}.json")

    print("\n[eval] wrote:")
    for p in written:
        print(f"  {p}")
    print(f"  {manifest_path}   (provenance: inputs, index hashes, seed, versions)")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # union of keys (rows may differ, e.g. optional coverage columns), first-seen order
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def _write_report(path: Path, rows: list[dict], rel, n_queries: int) -> Path:
    prim = rel.primary_metric
    ranked = sorted(rows, key=lambda r: r.get(prim, 0.0), reverse=True)
    # show the primary metric (+ its 95% CI) first, then the standard @5 columns
    cols = [c for c in ("recall@5", "mrr@5", "ndcg@5") if c != prim]
    lo_k, hi_k = f"{prim}_ci_low", f"{prim}_ci_high"
    has_ci = ranked and lo_k in ranked[0]
    ci_hdr = f" {prim} 95% CI |" if has_ci else ""
    n_q = n_queries or 0
    lines = ["# Retrieval evaluation — results\n",
             # NOT "declared before results": the 2026-07-02 audit established that no
             # author pre-registration happened (pipeline_audit.md §4, correction 1).
             # nDCG@5 was fixed in the config before any index was scored, which is an
             # implementation-time commitment, not a pre-registration. Say the true thing.
             f"- primary metric (frozen in `{Path(rel.source_path).name}` before any index "
             f"was scored; an implementation-time choice, not a pre-registration — see "
             f"`documents/pipeline_audit.md` §4): **{prim}**",
             f"- relevance threshold: `{rel.denom}` overlap ≥ {rel.min_overlap_frac} (frozen)",
             f"- CIs: percentile bootstrap ({DEFAULT_N_BOOT:,} resamples over the {n_q} "
             f"queries, seed {DEFAULT_SEED})",
             f"- indexes scored: {len(rows)}\n",
             f"\n## Ranked by {prim}\n",
             f"| rank | index | model | chunk_config | **{prim}** |{ci_hdr} "
             + " | ".join(cols) + " |",
             "|--:|---|---|---|--:|" + ("--:|" if has_ci else "") + "--:|" * len(cols)]
    for i, r in enumerate(ranked, 1):
        ci_cell = f" [{r[lo_k]:.4f}, {r[hi_k]:.4f}] |" if has_ci else ""
        vals = " | ".join(f"{r.get(c, 0):.4f}" for c in cols)
        lines.append(f"| {i} | {r['index_id']} | {r['model']} | {r['chunk_config']} | "
                     f"{r.get(prim,0):.4f} |{ci_cell} {vals} |")
    lines += ["\n> Selection rule: pick the top row on the primary metric, then apply "
              "tie-breakers — robustness across question types (esp. risk_suicide_emergency, "
              "see `retrieval_metrics_by_type.csv`) → index size / latency → build cost. "
              "Add CIs / significance before finalising.\n"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
