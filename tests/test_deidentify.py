"""Tests for the de-identification renderer.

The 2026-07-26 audit found this module had zero test coverage despite being the
single gate between scraped patient text and every participant-facing artifact.
Then on 2026-07-27 its semantics changed — `deidentify()` became idempotent so it
could keep working after the frozen files were themselves de-identified — which
is exactly the kind of change that can silently weaken a guard.

The property that matters is the strict-mode one: a scrub target that is neither
present nor already replaced must RAISE, because the alternative is shipping a
real name to a psychologist. Idempotence must not be implemented by simply
swallowing misses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from depression_rag.evaluation import deidentify as deid
from depression_rag.evaluation.deidentify import (
    SCRUB, ScrubTableUnavailable, assert_deidentified, clean_text, deidentify,
    scrub_manifest)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = [
    ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl",
    ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl",
]


@pytest.fixture
def scrub(monkeypatch):
    """Install a synthetic substitution table for the behavioural tests.

    These tests must not name a real study participant: the assertions would be
    the identifiers, in the one file guaranteed to ship. They must also not
    depend on the sealed table, which is absent on every clone — a behavioural
    guarantee that silently skips is not a guarantee. Synthetic needles give both.
    """
    table = {"q_single": [("Andi", "[nama]")],
             "q_double": [("Budi", "[nama]"), ("Cahya", "[nama]")]}
    monkeypatch.setattr(deid, "SCRUB", table)
    monkeypatch.setattr(deid, "SCRUB_SOURCE", "<synthetic table (tests)>")
    return table


# --- the substitution table itself -------------------------------------------

def test_scrub_table_is_well_formed():
    """Shape check on the real sealed table, when this machine has one."""
    if deid.SCRUB_SOURCE is None:
        pytest.skip("no sealed scrub table on this machine")
    for qid, subs in SCRUB.items():
        assert qid.startswith("alo_"), f"{qid} is not a question_id"
        for needle, replacement in subs:
            assert needle and replacement, f"empty rule in {qid}"
            assert needle != replacement
            assert needle not in replacement, (
                f"{qid}: replacing {needle!r} with {replacement!r} would leave the "
                "identifier in place")


def test_scrub_manifest_lists_every_rule(scrub):
    assert len(scrub_manifest()) == sum(len(v) for v in scrub.values())


def test_missing_sealed_table_yields_an_empty_table_not_an_error(tmp_path):
    """The fresh-clone case: `data/` is withheld and nothing needs rendering."""
    rules, source = deid.load_scrub_table(tmp_path / "absent.json")
    assert rules == {} and source is None


def test_sealed_table_round_trips(tmp_path):
    p = tmp_path / "sealed.json"
    p.write_text(json.dumps({"_comment": "ignored",
                             "rules": {"alo_0001": [["Dwi", "[nama]"]]}}),
                 encoding="utf-8")
    rules, source = deid.load_scrub_table(p)
    assert rules == {"alo_0001": [("Dwi", "[nama]")]} and source == str(p)


# --- the core guarantee -------------------------------------------------------

def test_name_is_removed(scrub):
    out = deidentify("q_single", "Halo dok, saya Andi dan saya sedih.")
    assert "Andi" not in out
    assert "[nama]" in out


def test_text_without_the_target_passes_through(scrub):
    """A pre-scrubbed question is the normal case now, not an error."""
    assert deidentify("q_single", "Halo dok, saya sedih.") == "Halo dok, saya sedih."


def test_idempotent_on_already_scrubbed_text(scrub):
    """The frozen files store rendered text, so re-rendering must be a no-op."""
    once = deidentify("q_single", "Nama saya Andi, umur 20.")
    twice = deidentify("q_single", once)
    assert once == twice
    assert "Andi" not in twice


def test_every_rule_fires_independently(scrub):
    """The bug this replaces: rule 1 firing left `[nama]` in the text, which the
    old idempotence check read as "rule 2 already applied" — so a second name in
    the same question survived. All rules share the `[nama]` placeholder, so that
    masking was guaranteed, not hypothetical.
    """
    out = deidentify("q_double", "Saya Budi dan adik saya Cahya.")
    assert "Budi" not in out and "Cahya" not in out
    assert out.count("[nama]") == 2


def test_pre_existing_placeholder_does_not_mask_a_real_name(scrub):
    """18 questions contain `[nama]` natively from Alodokter's own redaction; that
    must not be mistaken for "this question is already handled"."""
    out = deidentify("q_single", "Saya [nama], teman saya Andi juga depresi.")
    assert "Andi" not in out


# --- assert_deidentified ------------------------------------------------------

def test_assert_deidentified_catches_a_surviving_name(scrub):
    with pytest.raises(ValueError, match="still present"):
        assert_deidentified("saya Andi", "unit test")


def test_assert_deidentified_is_case_insensitive(scrub):
    with pytest.raises(ValueError):
        assert_deidentified("saya ANDI", "unit test")


def test_assert_deidentified_passes_clean_text(scrub):
    assert_deidentified("Saya [nama], umur 20 tahun.", "unit test")


def test_assert_deidentified_refuses_to_certify_without_a_table(monkeypatch):
    """An unarmed guard must not report clean: every caller reads a silent return
    as permission to hand the text to a participant."""
    monkeypatch.setattr(deid, "SCRUB", {})
    monkeypatch.setattr(deid, "SCRUB_SOURCE", None)
    with pytest.raises(ScrubTableUnavailable, match="sealed scrub table"):
        assert_deidentified("saya Andi", "unit test")


# --- clean_text ---------------------------------------------------------------

def test_clean_text_restores_missing_sentence_space():
    assert clean_text("mau tanya.Saya sedih") == "mau tanya. Saya sedih"


def test_clean_text_strips_mojibake_emoji():
    assert "ð" not in clean_text("terima kasih ð")


def test_clean_text_preserves_clinical_content():
    """Readability repairs only — dosages and abbreviations must survive."""
    for s in ("fluoxetine 20 mg/hari", "PHQ-9 skor 15", "rujuk ke IGD",
              "sejak 3 bulan lalu", "usia 16 th"):
        assert s in clean_text(f"Pasien: {s}.")


def test_clean_text_does_not_split_decimals_or_abbreviations():
    assert clean_text("dosis 2.5 mg") == "dosis 2.5 mg"


# --- the frozen data ----------------------------------------------------------

@pytest.mark.parametrize("path", FROZEN, ids=lambda p: p.name)
def test_frozen_files_carry_no_scrubbed_name(path: Path):
    """data/ and outputs/ are gitignored, so skip on a fresh clone."""
    if not path.exists():
        pytest.skip(f"{path.name} not present (regenerate locally)")
    text = path.read_text(encoding="utf-8").lower()
    survived = [n for subs in SCRUB.values() for n, _ in subs if n.lower() in text]
    assert not survived, f"{path.name} still contains {survived}"


def test_frozen_questions_carry_no_resolvable_source_url():
    """The URL is a sharper re-identification vector than any name: it resolves to
    the original, unscrubbed post. It must be hashed, not stored."""
    path = FROZEN[0]
    if not path.exists():
        pytest.skip("sample50 not present")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert records, "sample50 is empty"
    for r in records:
        assert "url" not in r, f"{r['question_id']} still carries a resolvable url"
        assert r.get("source_sha256"), f"{r['question_id']} has no source_sha256"


def test_every_scrub_rule_still_matches_the_raw_corpus():
    """Catches a rule that has rotted.

    This is the check that used to live inside `deidentify()` as a pre-condition
    ("target not found -> raise"). It cannot be evaluated per-render any more,
    because the live files are pre-scrubbed and the needle is always absent. It
    CAN be evaluated against the archived raw corpus, which is what it was really
    asking about: does every rule still describe text that actually existed?

    A rule that matches nothing is either a typo or a leftover from a question
    that has since been dropped — in both cases the name it claims to remove is
    not being removed from anywhere.
    """
    raw = (ROOT / "archive" / "frozen_raw_pre_deidentification_20260727" /
           "questions_alodokter_sample50.jsonl")
    if not raw.exists():
        pytest.skip("raw archive not present")
    records = {r["question_id"]: r
               for r in (json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip())}
    unmatched = []
    for qid, subs in SCRUB.items():
        rec = records.get(qid)
        if rec is None:
            unmatched.append(f"{qid}: no such question in the corpus")
            continue
        blob = " ".join(str(rec.get(f, "")) for f in
                        ("question", "question_title", "doctor_answer"))
        for needle, _ in subs:
            if needle not in blob:
                unmatched.append(f"{qid}: {needle!r} matches nothing")
    assert not unmatched, "stale SCRUB rules: " + "; ".join(unmatched)


def test_rendering_the_frozen_questions_is_a_no_op():
    """Since the files hold rendered text, deidentify() must neither change them
    nor raise — this is the assertion that would fail if SCRUB and the data ever
    drift apart."""
    path = FROZEN[0]
    if not path.exists():
        pytest.skip("sample50 not present")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assert deidentify(r["question_id"], r["question"]) == r["question"], \
            f"{r['question_id']} would change on render"
