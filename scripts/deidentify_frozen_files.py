#!/usr/bin/env python
"""Rewrite the frozen study files so they hold de-identified text (one-way).

Until 2026-07-27 the two frozen files kept the raw Alodokter text and
de-identification happened only at render time, on the argument that the files
were researcher-only. That argument stops holding the moment the repository is
published, so this script moves the scrub upstream into the files themselves.

It is deliberately a script and not a manual edit: it is a destructive, one-way
rewrite of hash-pinned artifacts, so it has to be repeatable, reviewable as a
diff, and archived before it runs.

What it changes, per record:

  * ``question``        — clean_text + SCRUB, strict (post-condition: no needle
                          may survive the substitution)
  * ``question_title``  — clean_text + SCRUB (a title may not repeat the name)
  * ``doctor_answer``   — clean_text + SCRUB (the doctor may or may not greet the
                          asker by name)
  * ``url``             — replaced by ``source_sha256``, 16 hex chars. This is
                          the important one. A scrubbed question next to a link
                          to the unscrubbed original is not de-identification.
                          The full mapping is written to a gitignored sealed
                          file so provenance survives for the examiners.

What it does NOT change, and why:

  * ``answer_doctors`` — the responding GP's name. Public professional
    attribution on a public forum, not patient data. Kept so the source stays
    creditable; drop it here if the ethics committee prefers.
  * ``question_date``, ``block``, ``study_id``, ``risky`` — not identifying.
  * ``generated_answer`` — checked, never echoes a scrubbed name, and is
    regenerated from de-identified questions anyway.

RESIDUAL RISK, stated plainly: the questions remain verbatim forum text. Anyone
can paste a distinctive sentence into a search engine and find the original post
with its original name. Removing names and the URL defeats casual
re-identification, not determined re-identification. The only fix for that is
paraphrasing, which would change the very stimuli the study measures. This is a
limitation to disclose, not a bug to fix.

    .venv/bin/python scripts/deidentify_frozen_files.py            # dry run
    .venv/bin/python scripts/deidentify_frozen_files.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.evaluation.deidentify import (  # noqa: E402
    SCRUB, assert_deidentified, clean_text, deidentify)

TARGETS = [
    ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl",
    ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl",
]
STRICT_FIELD = "question"                       # the name is known to be here
LOOSE_FIELDS = ("question_title", "doctor_answer")


def _scrub_loose(qid: str, text: str) -> str:
    """clean_text + SCRUB without requiring the target to be present."""
    out = clean_text(text)
    for needle, replacement in SCRUB.get(qid, []):
        out = out.replace(needle, replacement)
    return out


def transform(rec: dict) -> tuple[dict, list[str]]:
    """Return (new record, list of field names that changed)."""
    qid, changed = rec["question_id"], []
    out = dict(rec)

    if STRICT_FIELD in out:
        new = deidentify(qid, out[STRICT_FIELD])       # raises if a needle survives
        if new != out[STRICT_FIELD]:
            out[STRICT_FIELD] = new
            changed.append(STRICT_FIELD)
    for f in LOOSE_FIELDS:
        if out.get(f):
            new = _scrub_loose(qid, out[f])
            if new != out[f]:
                out[f] = new
                changed.append(f)
    if "url" in out:
        # rebuild in place so the key keeps its position in the record
        out = {("source_sha256" if k == "url" else k):
               (hashlib.sha256(v.encode()).hexdigest()[:16] if k == "url" else v)
               for k, v in out.items()}
        changed.append("url->source_sha256")
    return out, changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="apply the rewrite (default is a dry run)")
    args = ap.parse_args(argv)

    stamp = date.today().isoformat().replace("-", "")
    archive = ROOT / "archive" / f"frozen_raw_pre_deidentification_{stamp}"
    url_map: dict[str, str] = {}
    total_changed = 0

    for path in TARGETS:
        if not path.exists():
            raise SystemExit(f"missing target: {path}")
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        new_records, per_file = [], 0
        print(f"\n=== {path.relative_to(ROOT)}  ({len(records)} records) ===")
        for rec in records:
            if rec.get("url"):
                url_map[rec["question_id"]] = rec["url"]
            new, changed = transform(rec)
            new_records.append(new)
            if changed:
                per_file += 1
                text_changes = [c for c in changed if not c.startswith("url")]
                if text_changes:
                    print(f"  {new.get('study_id','?'):5s} {new['question_id']:10s} {', '.join(text_changes)}")
        print(f"  -> {per_file} records changed "
              f"({sum(1 for r in records if r.get('url'))} urls hashed)")
        total_changed += per_file

        for r in new_records:                       # belt and braces
            for f in (STRICT_FIELD, *LOOSE_FIELDS):
                if r.get(f):
                    assert_deidentified(r[f], f"{path.name}:{r['question_id']}:{f}")

        if args.write:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / path.name)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in new_records), encoding="utf-8")
            print(f"  written; raw copy -> {(archive / path.name).relative_to(ROOT)}")

    if not args.write:
        print(f"\nDRY RUN — {total_changed} records would change. Re-run with --write.")
        return 0

    sealed = ROOT / "data" / "derived" / "SEALED_source_urls.csv"
    sealed.write_text("question_id,source_sha256,url\n" + "".join(
        f"{q},{hashlib.sha256(u.encode()).hexdigest()[:16]},{u}\n"
        for q, u in sorted(url_map.items())), encoding="utf-8")
    (archive / "MANIFEST.md").write_text(f"""# Raw frozen files, archived {date.today()}

The pre-de-identification originals of the two hash-pinned study files. Kept so
the rewrite is auditable — you can diff these against the live files and see
exactly which characters changed.

**These contain patient names and resolvable source URLs. They must never be
committed.** `archive/` is inside the repo, so confirm `.gitignore` covers it
before any push.

Replaced by the de-identified versions on {date.today()}, which required
`freeze_eval_assignments.py --force` (their sha256s are pinned in
`eval_case_assignments.json`) and a regeneration of all 50 briefings.

The url -> source_sha256 mapping lives in `data/derived/SEALED_source_urls.csv`,
also researcher-only.
""", encoding="utf-8")
    print(f"\nsealed url map -> {sealed.relative_to(ROOT)} ({len(url_map)} rows)")
    print("NEXT: freeze_eval_assignments.py --force, then regenerate the briefings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
