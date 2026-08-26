#!/usr/bin/env python
"""Write the Stage-1 selection down as an artifact, and check it still holds.

Stage 1's deliverable is a decision: *which chunking x embedding configuration is
the retriever?* Until now that decision existed only as a string in
`configs/chatbot.yaml`, prose in `documents/phase2_6_selection_notes.md`, and
rank 1 of a generated table. Nothing stated the selection, the metric it was made
on, the value, or the tie-breaker that actually decided it — so nothing could be
checked, and nothing failed when the inputs moved underneath it.

This writes `outputs/analysis/selection.json`: the selected configuration, the
frozen rule it was selected under, its score with CI, the challengers it did not
significantly beat, the tie-breaker that resolved that tie, and the index the
chatbot is wired to. Re-run it after any change to the gold set, the indexes or
the relevance rule; `--check` re-derives the same facts and exits non-zero if the
selection no longer follows from the artifacts.

The selection RULE is not re-litigated here. It is the one frozen in
`configs/relevance.yaml`: rank on `primary_metric`, then apply the pre-stated
tie-breakers (robustness across question types, especially
risk_suicide_emergency -> index size/latency -> build cost).

    ./.venv/bin/python scripts/declare_selection.py            # write it
    ./.venv/bin/python scripts/declare_selection.py --check    # verify, write nothing
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
import sys
from pathlib import Path

import yaml

from depression_rag.evaluation.relevance import load_relevance_config

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"

# The safety-critical question type: the tie-breaker the selection notes lean on.
SAFETY_TYPE = "risk_suicide_emergency"


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def derive(metrics_csv: Path, by_type_csv: Path, rel, grid_only: bool) -> dict:
    """Re-derive the selection from the scored artifacts."""
    rows = _rows(metrics_csv)
    grid = set(rel.selection_grid_models)
    if grid_only and grid:
        rows = [r for r in rows if r["model"] in grid]
    prim = rel.primary_metric
    ranked = sorted(rows, key=lambda r: float(r[prim]), reverse=True)
    winner = ranked[0]

    # the tie-breaker only matters among indexes whose CI overlaps the winner's:
    # that is the "statistical tie" the selection notes appeal to
    lo = float(winner[f"{prim}_ci_low"])
    tied = [r for r in ranked if float(r[f"{prim}_ci_high"]) >= lo]

    by_type = {}
    for r in _rows(by_type_csv):
        by_type.setdefault(r["index_id"], {})[r["question_type"]] = float(r["recall@5"])
    floors = {r["index_id"]: min(by_type.get(r["index_id"], {1: 1}).values()) for r in tied}
    safety = {r["index_id"]: by_type.get(r["index_id"], {}).get(SAFETY_TYPE) for r in tied}

    return {
        "winner": winner["index_id"],
        "primary_metric": prim,
        "score": round(float(winner[prim]), 4),
        "ci": [round(float(winner[f"{prim}_ci_low"]), 4),
               round(float(winner[f"{prim}_ci_high"]), 4)],
        "n_queries": int(winner["n_queries"]),
        "n_indexes_considered": len(rows),
        "statistically_tied_with": [r["index_id"] for r in tied[1:]],
        "tie_breaker": {
            "rule": f"robustness across question types, especially {SAFETY_TYPE}",
            f"{SAFETY_TYPE}_recall@5": {k: v for k, v in safety.items()},
            "worst_type_floor_recall@5": {k: round(v, 4) for k, v in floors.items()},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=str(ANALYSIS / "retrieval_metrics.csv"))
    ap.add_argument("--by-type", default=str(ANALYSIS / "retrieval_metrics_by_type.csv"))
    ap.add_argument("--relevance-config", default=str(ROOT / "configs" / "relevance.yaml"))
    ap.add_argument("--chatbot-config", default=str(ROOT / "configs" / "chatbot.yaml"))
    ap.add_argument("--out", default=str(ANALYSIS / "selection.json"))
    ap.add_argument("--all-indexes", action="store_true",
                    help="rank over every scored index, not just the selection grid")
    ap.add_argument("--check", action="store_true",
                    help="verify the recorded selection still follows; write nothing")
    args = ap.parse_args(argv)

    rel = load_relevance_config(args.relevance_config)
    sel = derive(Path(args.metrics), Path(args.by_type), rel, grid_only=not args.all_indexes)

    # what the chatbot is actually wired to — the selection is only real if this agrees
    deployed = yaml.safe_load(Path(args.chatbot_config).read_text(encoding="utf-8"))["retriever"]["index_dir"]
    deployed_id = Path(deployed).name
    sel["deployed_index_id"] = deployed_id
    sel["deployed_matches_winner"] = deployed_id in (sel["winner"], sel["winner"] + "__mv")
    sel["index_scope"] = ("selection grid: " + ", ".join(sorted(rel.selection_grid_models))
                          if not args.all_indexes and rel.selection_grid_models
                          else "all scored indexes")
    sel["relevance_rule"] = {"denom": rel.denom, "min_overlap_frac": rel.min_overlap_frac,
                             "source": Path(rel.source_path).name}

    print(f"selected      : {sel['winner']}")
    print(f"  {sel['primary_metric']} = {sel['score']}  CI {sel['ci']}  n={sel['n_queries']}")
    print(f"  considered  : {sel['n_indexes_considered']} indexes ({sel['index_scope']})")
    print(f"  tied with   : {sel['statistically_tied_with'] or '(nothing — clear winner)'}")
    print(f"  tie-breaker : {SAFETY_TYPE} recall@5")
    for k, v in sel["tie_breaker"][f"{SAFETY_TYPE}_recall@5"].items():
        print(f"      {k:38s} {v}")
    print(f"  deployed    : {sel['deployed_index_id']}  "
          f"({'agrees' if sel['deployed_matches_winner'] else 'DISAGREES with the winner'})")

    problems = []
    if not sel["deployed_matches_winner"]:
        problems.append(f"configs/chatbot.yaml deploys {deployed_id!r}, which is neither the "
                        f"winner {sel['winner']!r} nor its __mv variant")
    safety = sel["tie_breaker"][f"{SAFETY_TYPE}_recall@5"]
    if safety.get(sel["winner"]) is not None:
        best = max(v for v in safety.values() if v is not None)
        if safety[sel["winner"]] < best:
            problems.append(f"the winner does not hold the best {SAFETY_TYPE} recall@5 "
                            f"({safety[sel['winner']]} vs {best}) — the pre-stated tie-breaker "
                            "no longer favours it")

    # The safety type alone rarely discriminates (it is 1.0 for most tied indexes);
    # the quantity that actually separates them is the worst-type floor. Guard it too,
    # otherwise the declared tie-breaker is checked on the one axis where every
    # candidate already passes.
    floors = sel["tie_breaker"]["worst_type_floor_recall@5"]
    if floors.get(sel["winner"]) is not None:
        best_floor = max(floors.values())
        if floors[sel["winner"]] < best_floor:
            problems.append(f"the winner does not hold the best worst-type floor "
                            f"({floors[sel['winner']]} vs {best_floor}) — robustness "
                            "across question types no longer favours it")

    if args.check:
        prev = Path(args.out)
        if prev.exists():
            old = json.loads(prev.read_text(encoding="utf-8"))
            moved = {k for k in ("winner", "score", "n_queries", "deployed_index_id")
                     if old.get(k) != sel.get(k)}
            if moved:
                problems.append(f"recorded selection is out of date on {sorted(moved)}: "
                                + ", ".join(f"{k} {old.get(k)!r} -> {sel.get(k)!r}" for k in sorted(moved)))
        else:
            problems.append(f"{prev} does not exist — run without --check to write it")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.check:
        print("\nOK — the recorded selection still follows from the artifacts.")
        return 0

    sel["derived_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    sel["derived_from"] = {"metrics": str(Path(args.metrics).relative_to(ROOT)),
                           "by_type": str(Path(args.by_type).relative_to(ROOT))}
    out = Path(args.out)
    out.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
