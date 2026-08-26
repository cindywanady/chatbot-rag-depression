"""Tests for the human-study analysis (scripts/analyze_study.py).

This script produces the study's primary human result, and until now nothing
exercised it: no human scores exist yet, so every path in it had run zero times.
That is the wrong order — an analysis whose first execution is on real data
cannot be corrected afterwards without the correction being indistinguishable
from choosing the analysis that gave the better answer.

The fixtures here are synthetic scores over the REAL frozen assignment file, so
the joins are tested against the actual keys rather than an idealised copy.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

ROOT_MARKER = "scripts/analyze_study.py"


@pytest.fixture(scope="module")
def mod():
    """Load analyze_study.py by path — scripts/ is not an importable package."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_study.py"
    spec = importlib.util.spec_from_file_location("analyze_study", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def assignments():
    from pathlib import Path

    p = (Path(__file__).resolve().parents[1]
         / "data" / "derived" / "eval_case_assignments.json")
    if not p.exists():
        pytest.skip("eval_case_assignments.json not present (data/ is gitignored)")
    return pd.DataFrame(json.loads(p.read_text())["cases"])


# --- the counting bug ---------------------------------------------------------

def test_strata_n_counts_cases_not_rating_rows(mod):
    """The 8 shared cases are rated by BOTH psychologists and so appear twice in
    p1. Counting rows inflates n and shrinks the CI; the case is the unit."""
    asg = pd.DataFrame([
        {"study_id": "Q01", "risky": True},
        {"study_id": "Q02", "risky": True},
        {"study_id": "Q03", "risky": False},
    ])
    p1 = pd.DataFrame([
        # Q01 is a shared case: two raters, two rows, ONE case
        {"case_id": "Q01", "rater": "psychologist_1", "global": 4.0},
        {"case_id": "Q01", "rater": "psychologist_2", "global": 2.0},
        {"case_id": "Q02", "rater": "psychologist_1", "global": 3.0},
        {"case_id": "Q03", "rater": "psychologist_2", "global": 5.0},
    ])
    lines: list[str] = []
    tbl = mod.analysis_p1_strata(p1, asg, lines)

    risk = tbl[(tbl.stratum == "risk") & (tbl.dimension == "global")].iloc[0]
    assert risk["n"] == 2, f"expected 2 cases in the risk stratum, got n={risk['n']}"
    # Q01 collapses to the mean of its two raters (3.0), then mean(3.0, 3.0)
    assert risk["mean"] == pytest.approx(3.0)

    nonrisk = tbl[(tbl.stratum == "non-risk") & (tbl.dimension == "global")].iloc[0]
    assert nonrisk["n"] == 1


def test_strata_reports_the_counting_unit(mod):
    """A reader must be able to tell n counts cases without reading the source."""
    asg = pd.DataFrame([{"study_id": "Q01", "risky": True}])
    p1 = pd.DataFrame([{"case_id": "Q01", "rater": "r1", "global": 4.0}])
    lines: list[str] = []
    mod.analysis_p1_strata(p1, asg, lines)
    assert any("CASES" in ln for ln in lines)


# --- the crash path -----------------------------------------------------------

def test_irr_survives_an_empty_p2_frame(mod):
    """main() passes an empty frame with only `study_id` before P2 data exists.
    That must reach the "no shared-item data" message, not raise KeyError."""
    asg = pd.DataFrame([
        {"study_id": "Q01", "risky": True, "p1_rater": "both"},
        {"study_id": "Q02", "risky": False, "p1_rater": "psychologist_1"},
    ])
    p1 = pd.DataFrame([
        {"case_id": "Q01", "rater": "psychologist_1", "global": 4.0, "gerbang": 1},
        {"case_id": "Q01", "rater": "psychologist_2", "global": 4.0, "gerbang": 1},
    ])
    lines: list[str] = []
    mod.analysis_irr(p1, pd.DataFrame(columns=["study_id"]), asg, lines)  # must not raise
    assert lines


# --- multiplicity -------------------------------------------------------------

def _paired_p2long(n_cases: int = 10) -> pd.DataFrame:
    """Synthetic P2 long table mirroring the real crossover.

    Each case is answered by BOTH counselors — one with the chatbot, one without —
    and the assignment mirrors between blocks. That makes `condition` aliased with
    the counselor x block interaction exactly as the frozen design is, which is
    the property `_aliasing_note` has to detect.
    """
    rows = []
    for i in range(n_cases):
        sid, block = f"Q{i:02d}", 1 + (i % 2)
        for counselor in ("counselor_1", "counselor_2"):
            chatbot = (counselor == "counselor_1") == (block == 1)
            cond = "chatbot" if chatbot else "no_chatbot"
            base = 4.0 if chatbot else 3.0
            rows.append({
                "study_id": sid, "condition": cond, "block": block,
                "counselor": counselor, "rater": "psychologist_1",
                "response_id": f"R{i:02d}{counselor[-1]}", "risky": i % 3 == 0,
                "global": base + (i % 3) * 0.1,
                "empati": base, "kebermanfaatan": base, "relevansi": base,
                "K1": 1, "K2": 1, "K3": 1, "K4": 1, "K5": 1,
                # A property of the QUESTION, so identical across conditions —
                # a stratifier here, never an outcome. K6 itself left this arm
                # on 2026-08-17; see test_scope_flag_is_never_an_outcome.
                "cakupan": 1 if i % 3 == 0 else 0,
            })
    return pd.DataFrame(rows)


def test_secondary_dimensions_are_holm_adjusted(mod):
    """Ten uncorrected p-values on a two-counselor pilot invites the
    garden-of-forking-paths objection; `global` stays the unadjusted primary."""
    lines: list[str] = []
    tbl = mod.analysis_primary(_paired_p2long(), lines)
    assert "wilcoxon_p_holm" in tbl.columns

    primary = tbl[tbl.dimension == "global"].iloc[0]
    assert pd.isna(primary["wilcoxon_p_holm"]), "the primary endpoint must stay unadjusted"

    sec = tbl[(tbl.dimension != "global") & tbl["wilcoxon_p"].notna()]
    assert len(sec) >= 2
    for _, r in sec.iterrows():
        assert r["wilcoxon_p_holm"] >= r["wilcoxon_p"] - 1e-12, (
            f"{r['dimension']}: adjusted p must not be smaller than raw")


def test_scope_flag_is_never_an_outcome(mod):
    """`cakupan` classifies the patient's QUESTION, which both conditions share.

    So it can never differ by condition, and scoring it as a dimension would
    manufacture a null. It belongs in the case-mix note instead — which is also
    what makes the scope finding on the briefing arm reportable at all, since the
    non-depression subset was previously flagged in no data file anywhere.
    """
    lines: list[str] = []
    tbl = mod.analysis_primary(_paired_p2long(), lines)

    assert "cakupan" not in set(tbl.dimension), "a case flag must not be a dimension"
    assert "K6" not in set(tbl.dimension), "K6 left the counselor arm on 2026-08-17"

    body = "\n".join(lines)
    assert "Case mix" in body, "the case mix must still be stated"
    assert "not tested" in body


def test_primary_survives_a_dimension_the_workbook_never_supplied(mod):
    """A column missing from the score table must skip, not crash.

    Score columns come from mapping Indonesian workbook headings; a renamed or
    unmapped heading yields a missing column. That is not hypothetical — K6 was
    absent from the converter's maps from 2026-08-06 to 2026-08-17. One such gap
    must not take the whole pre-registered primary analysis down with it.
    """
    lines: list[str] = []
    tbl = mod.analysis_primary(_paired_p2long().drop(columns=["K5", "K4"]), lines)
    assert "global" in set(tbl.dimension), "the primary endpoint must survive"
    assert {"K5", "K4"}.isdisjoint(set(tbl.dimension))


# --- the primary model must fail loudly ---------------------------------------

def test_mixed_model_failure_is_a_banner_not_a_footnote(mod):
    """A missing column used to demote the pre-registered primary analysis to one
    italic line in a report that otherwise looked complete."""
    lines: list[str] = []
    mod._mixed_model(_paired_p2long().drop(columns=["counselor"]), lines)
    body = "\n".join(lines)
    assert "PRE-REGISTERED PRIMARY MODEL DID NOT FIT" in body
    assert "counselor" in body


def test_mixed_model_reports_the_aliasing_when_it_fits(mod):
    """condition is identically the counselor x block interaction in this design;
    the report must say so where the coefficient is printed."""
    pytest.importorskip("statsmodels")
    lines: list[str] = []
    d = _paired_p2long(12)
    # sanity-check the fixture really reproduces the aliasing before asserting on it
    cells = d.groupby(["counselor", "block"])["condition"].nunique()
    assert len(cells) > 1 and (cells <= 1).all(), "fixture does not reproduce the design"
    mod._mixed_model(d, lines)
    body = "\n".join(lines)
    assert "Effective n" in body
    if "DID NOT FIT" not in body:
        assert "aliased" in body


# --- the real keys still join -------------------------------------------------

def test_build_p2_long_joins_against_the_real_assignment_file(mod, assignments):
    sealed = pd.DataFrame([
        {"response_id": "R001", "question_id": assignments.iloc[0]["question_id"],
         "counselor": "counselor_1", "condition": "chatbot", "raters": "psychologist_1"},
    ])
    p2 = pd.DataFrame([{"response_id": "R001", "rater": "psychologist_1", "global": 4}])
    out = mod.build_p2_long(p2, sealed, assignments)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["study_id"] == assignments.iloc[0]["study_id"]
    assert pd.notna(row["block"]) and pd.notna(row["condition"])


# --- the three analyses added 2026-08-06 --------------------------------------

def _block1_frame(shared_case="Q10"):
    """Block-1 rows in the real crossover shape: counselor_1 = chatbot,
    counselor_2 = no_chatbot, with one case scored by BOTH raters."""
    rows = []
    for i in range(12):
        sid = f"Q{i:02d}"
        rows.append({"response_id": f"A{i}", "study_id": sid, "block": 1,
                     "counselor": "counselor_1", "condition": "chatbot",
                     "rater": "psychologist_1", "global": 7.0})
        rows.append({"response_id": f"B{i}", "study_id": sid, "block": 1,
                     "counselor": "counselor_2", "condition": "no_chatbot",
                     "rater": "psychologist_1", "global": 5.0})
    # the shared case: a SECOND rating of answers that already exist
    rows.append({"response_id": "A0", "study_id": shared_case, "block": 1,
                 "counselor": "counselor_1", "condition": "chatbot",
                 "rater": "psychologist_2", "global": 7.0})
    rows.append({"response_id": "B0", "study_id": shared_case, "block": 1,
                 "counselor": "counselor_2", "condition": "no_chatbot",
                 "rater": "psychologist_2", "global": 5.0})
    return pd.DataFrame(rows)


def test_first_block_counts_answers_not_rating_rows(mod):
    """Shared cases are scored twice; counting rows would make n = 13 per arm and
    treat two ratings of ONE answer as independent observations."""
    lines = []
    tbl = mod.analysis_first_block(_block1_frame(), lines)
    g = tbl[tbl.dimension == "global"].iloc[0]
    assert int(g["n_chatbot"]) == 12, f"n inflated by rater rows: {g['n_chatbot']}"
    assert int(g["n_no_chatbot"]) == 12


def test_first_block_states_the_counselor_confound(mod):
    """Carryover-free is worthless if the reader misses that it is also fully
    confounded with counselor — so it must be a banner, not an omission."""
    lines = []
    mod.analysis_first_block(_block1_frame(), lines)
    text = "\n".join(lines)
    assert "CONFOUNDED" in text
    assert "counselor" in text.lower()


def test_first_block_effect_size_sign_follows_the_data(mod):
    """positive rank_biserial must mean with-chatbot scored HIGHER."""
    lines = []
    tbl = mod.analysis_first_block(_block1_frame(), lines)
    g = tbl[tbl.dimension == "global"].iloc[0]
    assert g["median_diff"] > 0 and g["rank_biserial"] > 0


def test_direction_agreement_counts_sign_matches(mod, assignments):
    shared = sorted(assignments.loc[
        (assignments["in_counselor_sample"] == True)          # noqa: E712
        & (assignments["p1_rater"] == "both"), "study_id"])
    assert shared, "expected shared counselor cases in the frozen file"
    rows = []
    # first case: raters disagree on direction; second: they agree
    for sid, (g1, g2) in zip(shared[:2], [(2.0, -1.0), (1.5, 0.5)]):
        for rater, gap in (("psychologist_1", g1), ("psychologist_2", g2)):
            rows.append({"study_id": sid, "rater": rater, "condition": "chatbot",
                         "global": 5.0 + gap})
            rows.append({"study_id": sid, "rater": rater, "condition": "no_chatbot",
                         "global": 5.0})
    lines = []
    tbl = mod.analysis_direction_agreement(pd.DataFrame(rows), assignments, lines)
    assert len(tbl) == 2
    assert list(tbl.sort_values("study_id")["agree_direction"]) == [False, True]
    assert "1 of 2" in "\n".join(lines)


def test_k1_refuses_to_correlate_two_counselors(mod, tmp_path):
    """n = 2 makes any correlation +/-1 by construction. The report must say so
    instead of printing one."""
    csv = tmp_path / "exit.csv"
    pd.DataFrame([
        {"counselor": "counselor_1", "item_no": 1, "kind": "skala",
         "response": "4 - setuju", "section": "Manfaat alat bantu"},
        {"counselor": "counselor_1", "item_no": 2, "kind": "angka", "response": 8},
        {"counselor": "counselor_2", "item_no": 1, "kind": "skala",
         "response": "2 - tidak setuju", "section": "Manfaat alat bantu"},
        {"counselor": "counselor_2", "item_no": 2, "kind": "angka", "response": 3},
    ]).to_csv(csv, index=False)
    lines = []
    tbl = mod.analysis_k1_helpfulness(csv, pd.DataFrame(), lines)
    text = "\n".join(lines)
    # the labelled workbook strings must parse to their leading digit
    assert list(tbl.sort_values("counselor")["likert_mean_1_5"]) == [4.0, 2.0]
    assert list(tbl.sort_values("counselor")["overall_0_10"]) == [8.0, 3.0]
    assert "Not correlated on purpose" in text
    assert "rho" not in text.lower() and "ρ" not in text


def test_k1_missing_input_is_reported_not_skipped(mod, tmp_path):
    """A pre-specified analysis that silently vanishes reads, to a reader, like
    one that was never specified."""
    lines = []
    tbl = mod.analysis_k1_helpfulness(tmp_path / "absent.csv", pd.DataFrame(), lines)
    assert tbl.empty
    assert "counselor-exit" in "\n".join(lines)


# --- BAGIAN 0: validating the safety screen against clinical judgement --------

def _bagian0_frame():
    """4 cases x 2 raters with one deliberate screen MISS and one over-fire.

    system risky:  Q00 yes, Q01 yes, Q02 no,  Q03 no
    rater says  :  Q00 yes, Q01 NO,  Q02 YES, Q03 no
    -> per rater: both_risk 1, system_only 1, rater_only 1, neither 1
    """
    rows = []
    for rater in ("psychologist_1", "psychologist_2"):
        for case, says in (("Q00", 1), ("Q01", 0), ("Q02", 1), ("Q03", 0)):
            rows.append({"case_id": case, "rater": rater, "bagian0_risiko": says})
    return pd.DataFrame(rows)


def _bagian0_asg():
    return pd.DataFrame([
        {"study_id": "Q00", "risky": True}, {"study_id": "Q01", "risky": True},
        {"study_id": "Q02", "risky": False}, {"study_id": "Q03", "risky": False},
    ])


def test_bagian0_confusion_matrix_counts(mod, tmp_path):
    lines = []
    tbl = mod.analysis_bagian0(_bagian0_frame(), _bagian0_asg(),
                               tmp_path / "absent.jsonl", lines)
    per = tbl[tbl.reference == "psychologist_1"].iloc[0]
    assert (int(per["both_risk"]), int(per["system_only"]),
            int(per["rater_only"]), int(per["neither"])) == (1, 1, 1, 1)


def test_bagian0_sensitivity_uses_the_psychologist_as_reference(mod, tmp_path):
    """sensitivity = caught / (caught + missed) against clinical judgement."""
    lines = []
    tbl = mod.analysis_bagian0(_bagian0_frame(), _bagian0_asg(),
                               tmp_path / "absent.jsonl", lines)
    per = tbl[tbl.reference == "psychologist_1"].iloc[0]
    assert per["sensitivity"] == pytest.approx(0.5)   # 1 caught of 2 expert-risky
    assert per["specificity"] == pytest.approx(0.5)   # 1 correct of 2 expert-safe


def test_bagian0_names_the_dangerous_error_direction(mod, tmp_path):
    """A miss means no safety banner reached the counselor; the report must say
    which column that is rather than leaving both errors looking equivalent."""
    lines = []
    mod.analysis_bagian0(_bagian0_frame(), _bagian0_asg(),
                         tmp_path / "absent.jsonl", lines)
    text = "\n".join(lines)
    assert "rater_only" in text and "no safety banner" in text


def test_bagian0_flags_an_unstable_expert_reference(mod, tmp_path):
    """If the two psychologists cannot agree with each other, nothing downstream
    supports a strong claim about the screen — that has to be stated."""
    df = _bagian0_frame()
    # make rater 2 disagree on 3 of 4 cases -> low kappa
    df.loc[(df.rater == "psychologist_2") & (df.case_id != "Q00"),
           "bagian0_risiko"] = [1, 0, 1]
    lines = []
    mod.analysis_bagian0(df, _bagian0_asg(), tmp_path / "absent.jsonl", lines)
    text = "\n".join(lines)
    assert "Between the two psychologists" in text


def test_bagian0_missing_column_is_reported_not_skipped(mod, tmp_path):
    lines = []
    tbl = mod.analysis_bagian0(pd.DataFrame([{"case_id": "Q00", "rater": "p1"}]),
                               _bagian0_asg(), tmp_path / "absent.jsonl", lines)
    assert tbl.empty
    assert "bagian0_risiko" in "\n".join(lines)
