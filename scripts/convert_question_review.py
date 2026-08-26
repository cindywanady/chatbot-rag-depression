#!/usr/bin/env python
"""Turn the completed Step-3 review worksheet back into the gold question set.

This is the half that was missing. Before it existed, `questions_gold.jsonl` was
a byte-copy of the LLM candidate draft: the worksheet had all 124 boxes unticked,
and nothing connected a review to the file the whole Phase-1 evaluation is scored
on. A ticked worksheet with no converter is still not evidence.

What it does:

  1. parses the worksheet written by `scripts/build_question_review.py`;
  2. **refuses to proceed while any `[ ]` remains** — a partly-done review cannot
     become a "validated" gold set;
  3. applies the verdicts (`[x]` keep, `[e]` keep as edited, `[-]` drop);
  4. runs the same schema/passage checks as `scripts/validate_questions.py`;
  5. reports exactly what changed against the current gold set, because any
     change moves every Phase-1 number and the write-up has to follow;
  6. on `--write`, writes the gold set and a manifest recording the worksheet's
     own sha256 next to the resulting file — so the review is provable from the
     artifacts rather than asserted in prose.

DRY RUN BY DEFAULT. Nothing is written without `--write`.

    ./.venv/bin/python scripts/convert_question_review.py
    ./.venv/bin/python scripts/convert_question_review.py --write
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

from depression_rag.evaluation import build_gold_passages, load_questions, validate_questions
from depression_rag.observability import ManifestBuilder

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

_HEAD = re.compile(r"^- \[(.)\] \*\*([A-Za-z0-9_.-]+)\*\*\s*$")
_FIELD = re.compile(r"^\s+(TYPE|DIFFICULTY|PASSAGES|Q|A|NOTE):\s?(.*)$")
_CONT = re.compile(r"^\s{6,}(?!\*)(?!>)(\S.*)$")

VERDICTS = {"x": "accept", "e": "edit", "-": "drop", " ": "todo"}


def parse_worksheet(path: Path) -> list[dict]:
    """-> [{verdict, question_id, TYPE, DIFFICULTY, PASSAGES, Q, A, NOTE}, ...]"""
    entries: list[dict] = []
    cur: dict | None = None
    field: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _HEAD.match(raw)
        if m:
            cur = {"verdict": m.group(1).lower(), "question_id": m.group(2)}
            entries.append(cur)
            field = None
            continue
        if cur is None:
            continue
        m = _FIELD.match(raw)
        if m:
            field = m.group(1)
            cur[field] = m.group(2).strip()
            continue
        m = _CONT.match(raw)
        if m and field:                      # wrapped continuation of the last field
            cur[field] = (cur[field] + " " + m.group(1).strip()).strip()
            continue
        if not raw.strip():
            field = None
    return entries


def to_record(e: dict) -> dict:
    return {
        "question_id": e["question_id"],
        "question": e.get("Q", "").strip(),
        "question_type": e.get("TYPE", "").strip(),
        "difficulty": e.get("DIFFICULTY", "").strip(),
        "passage_ids": [p.strip() for p in e.get("PASSAGES", "").split(",") if p.strip()],
        "reference_answer": e.get("A", "").strip(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worksheet", default=str(ROOT / "outputs" / "analysis" / "question_review_worksheet.md"))
    ap.add_argument("--current", default=str(DERIVED / "questions_gold.jsonl"),
                    help="the gold set being replaced (for the change report)")
    ap.add_argument("--out", default=str(DERIVED / "questions_gold.jsonl"))
    ap.add_argument("--manifests-dir", default=str(ROOT / "outputs" / "manifests"))
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args(argv)

    ws = Path(args.worksheet)
    if not ws.exists():
        raise SystemExit(f"no worksheet at {ws} — build it first:\n"
                         "  ./.venv/bin/python scripts/build_question_review.py")

    entries = parse_worksheet(ws)
    counts = {v: 0 for v in VERDICTS.values()}
    unknown = []
    for e in entries:
        name = VERDICTS.get(e["verdict"])
        if name is None:
            unknown.append((e["question_id"], e["verdict"]))
        else:
            counts[name] += 1

    print(f"[review] {ws}")
    print(f"  parsed {len(entries)} question blocks: " +
          ", ".join(f"{k}={v}" for k, v in counts.items()))
    if unknown:
        raise SystemExit(f"unrecognised verdict marker(s): {unknown}\n"
                         "use [x] accept, [e] edited, [-] drop")
    if counts["todo"]:
        todo = [e["question_id"] for e in entries if e["verdict"] == " "][:10]
        raise SystemExit(
            f"\n{counts['todo']} question(s) still unreviewed — refusing to write a "
            f"'validated' gold set from a partial review.\nfirst few: {todo}"
        )

    kept = [to_record(e) for e in entries if e["verdict"] in ("x", "e")]

    gp = build_gold_passages(ROOT / "configs" / "gold_passages.yaml",
                             DERIVED / "segments.jsonl", DERIVED / "cleaned_text.txt")
    v = validate_questions(kept, {p.passage_id: p.concept for p in gp.passages})
    print(f"  validation: {'OK' if v.ok else str(len(v.errors)) + ' ERRORS'} "
          f"({v.n} questions, {v.passages_covered}/{v.passages_total} passages covered)")
    if not v.ok:
        for err in v.errors[:20]:
            print(f"    - {err}")
        raise SystemExit("fix the worksheet and re-run")

    # what actually changed — this decides whether the write-up has to move
    current = {q["question_id"]: q for q in load_questions(args.current)} if Path(args.current).exists() else {}
    new = {q["question_id"]: q for q in kept}
    dropped = sorted(set(current) - set(new))
    added = sorted(set(new) - set(current))
    edited = sorted(i for i in set(current) & set(new) if current[i] != new[i])

    print("\n[review] change against the current gold set")
    print(f"  unchanged : {len(set(current) & set(new)) - len(edited)}")
    print(f"  edited    : {len(edited)}  {edited[:8]}")
    print(f"  dropped   : {len(dropped)}  {dropped[:8]}")
    print(f"  added     : {len(added)}  {added[:8]}")
    moved = bool(edited or dropped or added)
    if moved:
        print("\n  ** the gold set CHANGES — every Phase-1 number moves. After --write:")
        print("       ./.venv/bin/python scripts/run_retrieval_eval.py")
        print("       ./.venv/bin/python scripts/retrieval_significance.py")
        print("       ./.venv/bin/python scripts/retrieval_sensitivity.py")
        print("     then update the figures quoted in README §7 and the thesis.")
    else:
        print("\n  the review accepted every question unchanged — no Phase-1 number moves,")
        print("  and the worksheet now evidences that as a reviewed result rather than a copy.")

    if not args.write:
        print("\n[review] DRY RUN — nothing written. Re-run with --write to apply.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for q in kept:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    run_id = _dt.datetime.now().strftime("qreview-%Y%m%d-%H%M%S")
    manifest = ManifestBuilder(run_id=run_id, description="gold question set — step-3 human review")
    manifest.add_input("review_worksheet", ws)
    manifest.add_input("gold_passages_config", ROOT / "configs" / "gold_passages.yaml")
    manifest.add_input("segments", DERIVED / "segments.jsonl")
    manifest.add_input("cleaned_text", DERIVED / "cleaned_text.txt")
    manifest.record("verdicts", {k: v for k, v in counts.items() if k != "todo"})
    manifest.record("changed_vs_previous",
                    {"edited": edited, "dropped": dropped, "added": added})
    manifest.record("n_questions", len(kept))
    manifest.record("by_type", v.by_type)
    manifest.record("by_difficulty", v.by_difficulty)
    manifest.add_output(out)
    mpath = manifest.write(Path(args.manifests_dir) / f"{run_id}.json")

    print(f"\n[review] wrote {out} ({len(kept)} questions)")
    print(f"[review] wrote {mpath}")
    print("         the worksheet's sha256 is recorded beside the gold set — the review")
    print("         is now checkable from the artifacts, not asserted in prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
