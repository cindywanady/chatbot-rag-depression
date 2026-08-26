"""Single source of truth for how a patient question is shown to a HUMAN.

Every participant-facing artifact renders the study questions through
`deidentify()` — the counselor workbooks, the psychologist P1 packets, the
counselor case sheets and the P2 rating sequences — so no participant group
sees a version another group does not. Before this module existed the scrub
lived only in the counselor workbook builder, and the psychologists were being
handed the un-scrubbed text.

The frozen data files ARE de-identified as of 2026-07-27 (they were not before).
`data/derived/questions_alodokter_sample50.jsonl` and
`outputs/analysis/study_answers_chatbot.jsonl` now hold the rendered text, so the
raw names exist only in the ungitignored-nowhere scrape under `data/raw/`. That
re-freeze was deliberate and costly — it invalidated the recorded sha256s in
`eval_case_assignments.json` and forced a regeneration of every briefing and
every automatic score — and it was taken because the repository is intended for
publication, where "researcher-only" stops being true.

Rendering still goes through `deidentify()`. Applying it to already-clean text is
a no-op, so the render path stays the single source of truth and keeps working if
a raw question is ever reintroduced.

Two transformations, in order:

1. `clean_text` — readability repairs that change no clinical content: strip an
   emoji that survived as raw latin-1 bytes, and restore a missing space after a
   sentence-ending full stop ("tanya.Saya" -> "tanya. Saya").
2. `SCRUB` — explicit, auditable removal of direct identifiers. A substitution
   table, never a regex guess, so every removal can be listed in the ethics
   section. `[nama]` matches Alodokter's own redaction convention, which already
   appears in other questions, so the placeholder does not stand out.

Three questions in the frozen 50 carry a self-introduced name. A scan for e-mail
addresses, phone-length digit strings, national-ID-length digits, URLs, social
handles, named facilities and non-sentence-initial capitalised tokens — over the
question, title and doctor-reply fields of all 50 — found nothing else. The other
18 names were already `[nama]` from Alodokter's own redaction.

The table itself is NOT in this file. It maps question_id to the real names it
removes, so shipping it would publish the very identifiers the module exists to
withhold — and would do so in a repository whose own README concedes the question
text stays searchable back to its source. It is read at import from a sealed file
under `data/`, which `.gitignore` excludes, alongside the sealed URL map.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_SEALED_ENV = "DEPRESSION_RAG_SCRUB_TABLE"
_SEALED_DEFAULT = (Path(__file__).resolve().parents[3]
                   / "data" / "derived" / "SEALED_scrub_table.json")


class ScrubTableUnavailable(RuntimeError):
    """The sealed substitution table could not be loaded."""


def load_scrub_table(
    path: str | Path | None = None,
) -> tuple[dict[str, list[tuple[str, str]]], str | None]:
    """Read the sealed table -> ({question_id: [(needle, replacement), ...]}, source).

    A missing file yields an empty table rather than an error, because that is the
    fresh-clone case: `data/` is withheld, nothing is rendered, and the frozen
    files have been pre-scrubbed since 2026-07-27 anyway. Running a build script
    without the table is a different matter, and `assert_deidentified` refuses it.
    """
    resolved = Path(path or os.environ.get(_SEALED_ENV) or _SEALED_DEFAULT)
    if not resolved.is_file():
        return {}, None
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    rules = raw.get("rules", {}) if "rules" in raw else raw
    return ({qid: [(n, r) for n, r in subs]
             for qid, subs in rules.items() if not qid.startswith("_")},
            str(resolved))


# question_id -> [(exact string to remove, replacement), ...]
SCRUB, SCRUB_SOURCE = load_scrub_table()

# One of the three targets is why the pre-pass regexes in
# prepare_alodokter_questions.py are not a safety net: it evaded all three at once
# — _NAME_AGE wants umur|usia|berusia and the writer used a bare "(16 th)",
# _GREETING only fires on alo|halo|hai and the writer opened with "Pagi", and
# _INTRO needs an explicit anchor. That writer is a self-reported minor. The
# regexes catch the common shapes; SCRUB is what actually guarantees a removal.

MOJIBAKE = re.compile("\u00f0[\u0080-\u009f]{0,3}")  # emoji left as raw latin-1 bytes
C1 = re.compile("[\u0080-\u009f]")                    # stray C1 controls from the same accident
_SENTENCE_RUN_ON = re.compile(r"(?<=[a-z]{2})\.(?=[A-Z][a-z])")


def clean_text(text: str) -> str:
    """Readability-only repairs: drop mojibake, restore missing sentence spaces."""
    text = MOJIBAKE.sub("", text)
    text = C1.sub("", text)
    text = _SENTENCE_RUN_ON.sub(". ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def deidentify(question_id: str, text: str, *, strict: bool = True) -> str:
    """Render-ready question text: cleaned, with direct identifiers removed.

    Idempotent, and checked as a POST-condition: after substitution no needle may
    survive. `strict` (the default) raises if one does.

    It used to be a pre-condition — a target that was not found raised, on the
    reasoning that a missing target meant a stale table. That stopped working on
    2026-07-27 when the frozen files became pre-scrubbed (the needle is now
    legitimately absent every time), and the obvious patch — "tolerate a miss if
    the replacement is already there" — is unsound: every rule replaces with the
    same `[nama]`, and 18 questions contain `[nama]` natively from Alodokter's own
    redaction. So one rule firing, or an unrelated pre-existing placeholder, would
    mask a genuine miss on a second name in the same question.

    The pre-condition is not recoverable from the text alone, so it moved to where
    it can actually be evaluated: `tests/test_deidentify.py` checks every rule
    still matches the archived raw corpus, catching a rule that has rotted. What
    remains here is the property that holds on every call, and callers pair it
    with `assert_deidentified()` — that pairing, not this function, is the gate.
    """
    out = clean_text(text)
    for needle, replacement in SCRUB.get(question_id, []):
        out = out.replace(needle, replacement)
    if strict:
        survived = [n for n, _ in SCRUB.get(question_id, []) if n in out]
        if survived:
            raise ValueError(
                f"de-identification failed for {question_id}: {survived} survived "
                "substitution")
    return out


def assert_deidentified(text: str, where: str = "") -> None:
    """Guard for build scripts: raise if any known identifier survived.

    Also raises when the sealed table was never loaded. Every caller treats a
    silent return as proof the text is safe to hand to a participant, so a guard
    that cannot see any identifier must fail loudly rather than certify clean —
    an empty table would otherwise pass every input, including raw scraped text.
    """
    if SCRUB_SOURCE is None:
        raise ScrubTableUnavailable(
            f"cannot certify {where or 'rendered text'}: no sealed scrub table "
            f"loaded (looked for {_SEALED_DEFAULT}, override with ${_SEALED_ENV}). "
            "Restore it before building any participant-facing artifact.")
    low = text.lower()
    found = [n for subs in SCRUB.values() for n, _ in subs if n.lower() in low]
    if found:
        raise ValueError(f"identifier(s) {found} still present in {where or 'rendered text'}")


def scrub_manifest() -> list[tuple[str, str, str]]:
    """(question_id, removed, replacement) — for the ethics/audit write-up."""
    return [(qid, needle, repl) for qid, subs in SCRUB.items() for needle, repl in subs]
