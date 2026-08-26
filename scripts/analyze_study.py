#!/usr/bin/env python
"""Turnkey analysis of the human evaluation study (design v2 §7).

Joins the filled psychologist score workbooks to the frozen keys and runs the
analyses below in one command.

Every analysis design v2 §7 pre-specifies is now implemented (the last three
landed 2026-08-06, before any scores were collected — see §7–9 below and
documents/evaluation/protocol_amendment_20260806.md).

Implemented:

  1. PRIMARY — counselor answers, with vs without chatbot (24 matched pairs):
     paired Wilcoxon + paired t on the holistic score and each quality/safety
     dimension, and the pre-registered mixed model
     score ~ condition + block + counsellor + rater + (1|question).
  2. Safety-gate failure rate by condition (risk cases).
  3. Inter-rater agreement on the 16 shared items: ICC(2,1) for graded items,
     quadratic-weighted Cohen's kappa for the binary gate.
  4. P1 briefing descriptives, STRATIFIED by risk (never pooled — the P1 set is
     67% risk).
  5. Triangulation: per-case P1 briefing quality vs the (with - without) gap.
  6. Human vs automatic faithfulness calibration on the 18 full-form packets.
  7. First-block sensitivity — the carryover-free half of the crossover. Reported
     as UNPAIRED and fully confounded with counselor, because in block 1 every
     chatbot answer is counselor_1's and every no-chatbot answer is counselor_2's.
  8. Pair-level direction agreement on the 4 shared counselor cases (Q10, Q22,
     Q26, Q38): do both psychologists agree on the SIGN of with-without?
  9. K1 perceived helpfulness (counselor exit questionnaire), via
     `--counselor-exit`. Reported descriptively: with two counselors the
     correlation §7 asks for is not estimable, and the script says so rather
     than printing a two-point rho.
 10. BAGIAN 0 — the deployed keyword+LLM risk screen against the psychologists'
     independent clinical judgement: confusion matrix, sensitivity/specificity,
     and a deterministic recomputation of the keyword stage alone. This is the
     study's only validation of the safety screen, and the reason the packet
     never prints the system's verdict.

Inputs default to the kit / pipeline locations; override as needed. Writes a
markdown report plus tidy CSVs for plotting.

    .venv/bin/python scripts/analyze_study.py
    .venv/bin/python scripts/analyze_study.py --scores A.xlsx:psychologist_1 B.xlsx:psychologist_2

`--counselor-exit` expects a long CSV transcribed from the «Kuesioner Akhir»
sheet of the counselor workbooks:

    counselor,item_no,kind,response[,section]
    counselor_1,1,skala,4 - setuju
    counselor_1,9,angka,8

`kind` mirrors build_counselor_workbooks.EXIT_ITEMS (`skala` = 1-5 Likert,
`angka` = 0-10 overall, `teks` = free text, ignored here). Leading digits are
parsed, so the workbook's own "4 - setuju" strings can be pasted verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from depression_rag.evaluation.bootstrap import holm_adjust

ROOT = Path(__file__).resolve().parents[1]

# `kualitas_bahasa` (ESHRO fluency) removed 2026-08-17 — least clinical item on the
# form, and it pointed the rater at the dimension most likely to leak chatbot style.
QUALITY_P2 = ["empati", "kebermanfaatan", "relevansi"]
SAFETY = ["K1", "K2", "K3", "K4", "K5"]
QUALITY_P1 = ["kebenaran", "kelengkapan", "relevansi", "kejelasan"]
# «Ketepatan rujukan [n]» (citation precision, Liu/Zhang/Liang 2023) was the third
# member until 2026-08-17. Removed from the human instrument: it fed nothing here —
# one descriptive row, no test, no calibration partner — while `dukungan_konteks`
# is the one calibrated against the automatic faithfulness metrics in §6. It is
# also the most automatable judgement on the form (does the sentence tagged [3]
# follow from passage [3]), so spending scarce clinician time on it was the worst
# trade in the block. Not redundant with `dukungan_konteks` in construct — grounded
# claims can still be mis-numbered — so nothing now checks the [n] markers the
# counselor actually sees. State that in the limitations.
FAITH_P1 = ["dukungan_konteks", "kecukupan_konteks"]

# K6 «Kesesuaian cakupan» — kept OUT of SAFETY on purpose, though it shares the
# Park-block numbering. It reached the workbooks in the 2026-08-06 amendment but
# never reached this file or the converter's maps, so until 2026-08-17 it was
# rated by both psychologists and then silently discarded.
#
# Restructured 2026-08-17: BRIEFINGS ONLY, and conditional. It is blank on a
# pure-depression case — a skip, not a zero — so every K6 row already IS the
# discriminating subset, and its n is legitimately smaller than its neighbours'.
# It was dropped from the counselor arm entirely: the amendment that added it
# already recorded that contrast as not estimable (~2 cases per counselor x
# condition cell, `condition` aliased with counselor x block), so it could never
# have been more than description there.
SCOPE = ["K6"]

# The companion classification, asked on BOTH arms about the QUESTION: 1 = the
# case raises a condition other than depression. New 2026-08-17, and the reason
# the scope finding is reportable at all — the non-depression subset had been
# counted ad hoc for the amendment and never written to any data file. Being a
# property of the question, it is identical across conditions, so on the
# counselor arm it is a case-level STRATIFIER, never an outcome.
CASE_FLAGS = ["cakupan"]


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame -> markdown table (no external 'tabulate' dependency)."""
    d = df.reset_index() if index else df.copy()

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if pd.notna(x) else ""
        return "" if pd.isna(x) else str(x)

    cols = [str(c) for c in d.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- loading -----
def load_assignment(path: Path) -> pd.DataFrame:
    cases = json.loads(path.read_text())["cases"]
    return pd.DataFrame(cases)


def load_scores(p1_path: str, p2_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two tidy score tables produced by convert_psychologist_scores.py.

    The rater workbooks are not read here on purpose: they carry Indonesian display
    headings and anchor-carrying dropdown values ("4 - lancar") for the rater's sake.
    Translating those into the internal column names is one job, done in one place,
    so the analysis never has to know what a rating sheet looks like.
    """
    frames = []
    for path, id_col in ((p1_path, "case_id"), (p2_path, "response_id")):
        if not Path(path).exists():
            sys.exit(f"missing {path}\n"
                     "run: .venv/bin/python scripts/convert_psychologist_scores.py")
        df = pd.read_csv(path)
        if id_col not in df.columns:
            sys.exit(f"{path} has no '{id_col}' column — regenerate it with "
                     "scripts/convert_psychologist_scores.py")
        frames.append(df.dropna(subset=[id_col]))
    return frames[0], frames[1]


def coerce(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# --------------------------------------------------------------- helpers ------
def icc21(matrix: np.ndarray) -> float | None:
    """ICC(2,1): two-way random effects, single rater, absolute agreement.
    matrix: n subjects x k raters (complete rows only)."""
    m = matrix[~np.isnan(matrix).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return None
    grand = m.mean()
    ms_r = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # rows/subjects
    ms_c = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # columns/raters
    ss_tot = ((m - grand) ** 2).sum()
    # residual SS = total - rows - cols
    ss_r = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ms_e = (ss_tot - ss_r - ss_c) / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom else None


def weighted_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or len(set(a.tolist() + b.tolist())) < 2:
        return None
    try:
        return float(cohen_kappa_score(a.round().astype(int), b.round().astype(int),
                                       weights="quadratic"))
    except Exception:  # noqa: BLE001
        return None


def paired_report(diffs: pd.Series) -> dict:
    d = diffs.dropna()
    n = len(d)
    out = {"n_pairs": n, "median_diff": float(d.median()) if n else None,
           "mean_diff": float(d.mean()) if n else None}
    if n >= 5 and d.abs().sum() > 0:
        try:
            w, pw = stats.wilcoxon(d)
            out["wilcoxon_p"] = float(pw)
        except ValueError:
            out["wilcoxon_p"] = None
        t, pt = stats.ttest_1samp(d, 0)
        out["t_p"] = float(pt)
        out["dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else None
    return out


# --------------------------------------------------------------- analyses -----
def build_p2_long(p2: pd.DataFrame, sealed: pd.DataFrame, asg: pd.DataFrame) -> pd.DataFrame:
    df = p2.merge(sealed, on="response_id", how="left", suffixes=("", "_key"))
    meta = asg[["question_id", "study_id", "block", "risky"]]
    df = df.merge(meta, on="question_id", how="left")
    return df


def analysis_primary(p2long: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    lines.append("## 1. Primary — counselor answers, with vs without chatbot\n")
    dims = ["global"] + QUALITY_P2 + SAFETY
    coerce(p2long, dims)
    rows = []
    for dim in dims:
        # p2long's columns come from mapping Indonesian workbook headings, so a
        # renamed or never-mapped heading yields a missing column — which is
        # exactly how K6 went unnoticed until 2026-08-17. Skip the dimension, the
        # same convention analysis_p1_strata and analysis_irr already use; without
        # this, one absent column raises KeyError inside pivot_table and takes the
        # whole primary analysis down with it.
        if dim not in p2long.columns:
            continue
        wide = p2long.pivot_table(index="study_id", columns="condition", values=dim, aggfunc="mean")
        if not {"chatbot", "no_chatbot"}.issubset(wide.columns):
            continue
        diff = wide["chatbot"] - wide["no_chatbot"]
        r = paired_report(diff); r["dimension"] = dim
        rows.append(r)
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        lines.append("_No paired data yet._\n")
        return tbl

    # `global` is the pre-specified primary and stays unadjusted; the other nine
    # dimensions are a family and get Holm. Reporting ten Wilcoxon p-values at face
    # value on a two-counselor pilot is the garden-of-forking-paths objection that
    # retrieval_significance.py already guards against for its six comparisons.

    if "wilcoxon_p" in tbl:
        sec = [i for i in tbl.index
               if tbl.loc[i, "dimension"] != "global"
               and pd.notna(tbl.loc[i, "wilcoxon_p"])]
        if sec:
            tbl["wilcoxon_p_holm"] = np.nan
            adjusted = holm_adjust([float(tbl.loc[i, "wilcoxon_p"]) for i in sec])
            for i, a in zip(sec, adjusted):
                tbl.loc[i, "wilcoxon_p_holm"] = a

    lines.append(to_md(tbl))
    lines.append("\n_`global` is the pre-specified primary endpoint and is reported "
                 "unadjusted. `wilcoxon_p_holm` is the Holm-adjusted p across the "
                 "secondary dimensions as a family; prefer it over `wilcoxon_p` for "
                 "those rows, and treat `t_p` as a sensitivity check only._\n")
    _case_mix_note(p2long, lines)
    prim = tbl[tbl.dimension == "global"]
    if not prim.empty:
        p = prim.iloc[0]
        lines.append(f"\n**Holistic (primary):** {int(p['n_pairs'])} pairs, "
                     f"median with−without = {p['median_diff']:.2f}, "
                     f"Wilcoxon p = {p.get('wilcoxon_p')}. Positive = with-chatbot rated higher.\n")
    _mixed_model(p2long, lines)
    return tbl


def _case_mix_note(p2long: pd.DataFrame, lines: list[str]) -> None:
    """How much of the counselor set is not plain depression.

    `cakupan` is a property of the QUESTION, so it is identical across
    conditions and cannot itself be an outcome here. What it buys on this arm is
    a stated case mix — a reader can see how much of the primary estimate rests
    on cases the tool's guideline does not fully cover. Reported, not tested:
    splitting 24 pairs on a design where `condition` is already aliased with
    counselor x block gives ~2 cases per cell, which is why the scale that used
    to sit here was removed rather than kept as a p-value nobody could read.
    """
    if "cakupan" not in p2long.columns:
        return
    coerce(p2long, ["cakupan"])
    per_case = p2long.groupby("study_id")["cakupan"].mean().dropna()
    if per_case.empty:
        return
    n_mixed = int((per_case > 0.5).sum())
    lines.append(
        f"\n_Case mix: {n_mixed} of {len(per_case)} counselor cases were judged to "
        "raise a condition other than depression. The estimates above pool both "
        "kinds — the split is stated, not tested (≈2 cases per counselor × "
        "condition cell). See §4 for the scope item itself, which runs on the "
        "briefings._\n")


def _aliasing_note(d: pd.DataFrame, lines: list[str]) -> None:
    """State the crossover's confound at the point the coefficient is printed.

    Every block-1 case is {counselor_1: chatbot, counselor_2: no_chatbot} and
    block 2 is the exact mirror, so `condition` IS the counselor x block
    interaction and its coefficient cannot be separated from differential
    carryover between the two counselors. The pairs are also clustered in four
    counselor-blocks rather than being independent replicates — a variance
    problem, not a bias one. Neither fact should have to be derived from the
    assignment file by whoever reads the report.
    """
    cells = d.groupby(["counselor", "block"])["condition"].nunique()
    lines.append(
        f"\n> **Effective n.** The {d['study_id'].nunique()} question-level pairs sit in "
        f"{len(cells)} counselor x block cell(s) and are not independent replicates, "
        "so the interval above is optimistic.\n")
    if len(cells) > 1 and bool((cells <= 1).all()):
        lines.append(
            "> Each cell holds exactly one condition, so `condition` is aliased with "
            "the counselor x block interaction: the coefficient cannot be separated "
            "from differential carryover between the two counselors. With n = 2 "
            "counselors this follows from the design, not the data — report it as a "
            "limitation rather than trying to resolve it statistically.\n")


def _mixed_model(p2long: pd.DataFrame, lines: list[str]) -> None:
    """Fit and report the pre-registered primary model.

    Failures are reported as a BANNER, not a footnote. This is the study's
    primary analysis: if it does not fit, the report must not read as though it
    merely omitted an optional extra. The handler is also kept narrow — a broad
    `except Exception` here would swallow a renamed column or a typo'd formula
    and demote the primary result to one italic line in an otherwise complete
    report.
    """
    import warnings

    lines.append("\n### Pre-registered mixed model (holistic)\n")

    def failed(reason: str) -> None:
        lines.append(f"> **PRE-REGISTERED PRIMARY MODEL DID NOT FIT — {reason}**\n>\n"
                     "> The with/without comparison above is descriptive only until "
                     "this is resolved.\n")

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        failed("statsmodels is not installed in this environment "
               "(`.venv/bin/pip install statsmodels`)")
        return

    required = ["global", "condition", "block", "counselor", "rater", "study_id"]
    missing = [c for c in required if c not in p2long.columns]
    if missing:
        failed(f"missing column(s) {missing} in the joined P2 table")
        return

    d = p2long.dropna(subset=required).copy()
    d["global"] = pd.to_numeric(d["global"], errors="coerce")
    d = d.dropna(subset=["global"])
    if d["condition"].nunique() < 2 or len(d) < 10:
        failed(f"not enough data ({len(d)} complete rows, "
               f"{d['condition'].nunique()} condition level(s); need >=10 and 2)")
        return

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Q("global") quotes the column: bare `global` is a Python keyword
            m = smf.mixedlm('Q("global") ~ C(condition) + C(block) + C(counselor) + C(rater)',
                            d, groups=d["study_id"]).fit(reml=False)
        # mixedlm WARNS rather than raising on a singular or non-converged fit, so
        # without this check a bad fit formats identically to a good one.
        problems = [str(w.message).strip() for w in caught
                    if "converg" in str(w.message).lower()
                    or "singular" in str(w.message).lower()]
    except (ValueError, np.linalg.LinAlgError, KeyError) as e:
        failed(f"fit raised {type(e).__name__}: {e}")
        return

    coef = [p for p in m.params.index if "condition" in p]
    if not coef:
        failed("the fit produced no `condition` coefficient "
               f"(parameters: {list(m.params.index)})")
        return

    name = coef[0]
    lines.append(f"`score ~ condition + block + counsellor + rater + (1|question)` — "
                 f"coefficient `{name}` = {m.params[name]:.3f} "
                 f"(SE {m.bse[name]:.3f}, p = {m.pvalues[name]:.3f}). The bracketed "
                 f"level is the one being contrasted against the reference condition, "
                 f"so read the sign accordingly.\n")
    if problems:
        lines.append(f"\n> **Fit warning — interpret with caution:** {'; '.join(problems)}\n")
    _aliasing_note(d, lines)


def analysis_safety(p2long: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n## 2. Safety-gate failure rate by condition (risk cases)\n")
    coerce(p2long, ["gerbang"])
    if "risky" not in p2long.columns or "gerbang" not in p2long.columns:
        # coerce() only converts a column that is already there, so an unfilled or
        # renamed «Gerbang keselamatan» heading leaves it absent — which used to
        # raise KeyError here instead of reaching the message immediately below.
        lines.append("_No safety-gate data yet (no `gerbang` column in the P2 scores)._\n")
        return
    risk = p2long[p2long["risky"] == True]  # noqa: E712
    if risk.empty or risk["gerbang"].isna().all():
        lines.append("_No safety-gate data yet._\n"); return
    # convention: gate value 1 = pass, 0 = fail (a low score = failure).
    # collapse shared cases (rated by both raters) to one value per case×condition
    g = (risk.dropna(subset=["gerbang"])
             .groupby(["study_id", "condition"])["gerbang"].mean().reset_index())
    tbl = (g.assign(fail=(g["gerbang"] < 1).astype(int))
             .groupby("condition")["fail"].agg(["sum", "count"]))
    tbl["rate"] = tbl["sum"] / tbl["count"]
    lines.append(to_md(tbl, index=True))
    lines.append("\n_Report by condition; n is small (≤12 risk cases/condition) — treat as descriptive._\n")
    # The gate gained a TIDAK BERLAKU option on 2026-08-06, which converts to
    # blank and drops out here. A rater can therefore mark "no danger signs" on a
    # case the SYSTEM flagged risky — legitimate disagreement, but it silently
    # shrinks the denominator, so the count has to be visible.
    n_na = int(risk["gerbang"].isna().sum())
    if n_na:
        lines.append(f"\n_{n_na} rating(s) on system-flagged risk cases were marked "
                     "«tidak berlaku» by the rater (they judged no danger signs "
                     "present) and are excluded from the denominator above. That "
                     "disagreement is itself a finding — see the BAGIAN 0 "
                     "comparison._\n")

    # The counselor-arm gate was widened on 2026-08-17: TIDAK AMAN now also covers an
    # answer that INTRODUCES danger (e.g. telling the asker to stop medication) on a
    # question that carried no danger signs. The table above is the risk stratum only,
    # so without this block those failures would be collected and never printed —
    # which would defeat the point of widening it. The counselor sample is 12 risk /
    # 12 non-risk, so this is half the rows.
    safe_q = p2long[p2long["risky"] != True]  # noqa: E712
    if not safe_q.empty and "gerbang" in safe_q:
        harmed = safe_q.dropna(subset=["gerbang"])
        harmed = harmed[harmed["gerbang"] < 1]
        if not harmed.empty:
            per = (harmed.groupby("condition")["gerbang"].size()
                   if "condition" in harmed else None)
            detail = (", ".join(f"{c}: {n}" for c, n in per.items()) if per is not None
                      else f"{len(harmed)} rating(s)")
            lines.append(f"\n> **TIDAK AMAN on cases whose QUESTION carried no danger "
                         f"signs — {detail}.** These are answers judged to introduce "
                         "harm rather than mishandle it, and they are NOT in the table "
                         "above (which is the risk stratum). Report them separately and "
                         "read the rater's `catatan`; the gate was widened to catch "
                         "exactly this. See `INTERNAL_catatan_item_psikolog.md` §4.12._\n")
        else:
            lines.append("\n_No TIDAK AMAN outside the risk stratum: no answer was judged "
                         "to introduce harm on a question that carried no danger signs._\n")


def analysis_irr(p1: pd.DataFrame, p2long: pd.DataFrame, asg: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n## 3. Inter-rater agreement — the 16 shared items\n")
    shared_cases = set(asg.loc[asg["p1_rater"] == "both", "study_id"])
    # P1 shared packets
    coerce(p1, ["global", "gerbang"] + QUALITY_P1)
    out = []
    sp1 = p1[p1["case_id"].isin(shared_cases)]
    for dim in ["global"] + QUALITY_P1:
        # p1's columns come from mapping Indonesian workbook headings, so a renamed
        # or unfilled heading yields a missing column. Skip the dimension rather
        # than crashing the whole report over one of them.
        if sp1.empty or dim not in sp1.columns or "rater" not in sp1.columns:
            continue
        w = sp1.pivot_table(index="case_id", columns="rater", values=dim, aggfunc="mean")
        if w.shape[1] == 2 and w.dropna().shape[0] >= 2:
            out.append({"instrument": "P1 briefing", "item": dim, "metric": "ICC(2,1)",
                        "value": icc21(w.dropna().to_numpy()), "n": w.dropna().shape[0]})
    if "gerbang" in sp1:
        w = sp1.pivot_table(index="case_id", columns="rater", values="gerbang", aggfunc="mean")
        if w.shape[1] == 2 and w.dropna().shape[0] >= 2:
            a, b = w.dropna().to_numpy().T
            out.append({"instrument": "P1 briefing", "item": "gerbang", "metric": "weighted κ",
                        "value": weighted_kappa(a, b), "n": w.dropna().shape[0]})
    # P2 shared answers (both raters scored the same response_ids on shared cases)
    coerce(p2long, ["global"])
    sp2 = p2long[p2long["study_id"].isin(shared_cases)]
    # main() passes an empty frame with only `study_id` when there is no P2 data
    # yet; without this guard pivot_table raises KeyError instead of falling
    # through to the "no shared-item data" message below.
    if not sp2.empty and {"response_id", "rater", "global"}.issubset(sp2.columns):
        w = sp2.pivot_table(index="response_id", columns="rater", values="global", aggfunc="mean")
        if w.shape[1] == 2 and w.dropna().shape[0] >= 2:
            out.append({"instrument": "P2 answer", "item": "global", "metric": "ICC(2,1)",
                        "value": icc21(w.dropna().to_numpy()), "n": w.dropna().shape[0]})
    if out:
        lines.append(to_md(pd.DataFrame(out)))
        lines.append("\n_ICC/κ ≥ 0.75 = excellent, 0.60–0.74 good, 0.40–0.59 fair (Koo & Li 2016)._\n")
    else:
        lines.append("_No shared-item data yet (need both raters on the 8 shared cases)._\n")


def analysis_p1_strata(p1: pd.DataFrame, asg: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    lines.append("\n## 4. P1 briefing descriptives, stratified by risk\n")
    coerce(p1, ["global"] + QUALITY_P1 + SAFETY + SCOPE + FAITH_P1)
    df = p1.merge(asg[["study_id", "risky"]], left_on="case_id", right_on="study_id", how="left")
    dims = ["global"] + QUALITY_P1 + SAFETY + SCOPE + FAITH_P1
    rows = []
    for risky, g in df.groupby("risky"):
        for dim in dims:
            if dim not in g:
                continue
            # The CASE is the unit, not the row. The 8 shared cases are rated by
            # both psychologists and so appear twice in p1 — `analysis_irr` pivots
            # on `rater` precisely to exploit that. Counting rows would inflate n
            # by up to 8 and shrink every interval accordingly. Averaging over
            # raters first matches what analysis_triangulation and
            # analysis_calibration already do.
            v = (pd.to_numeric(g[dim], errors="coerce")
                 .groupby(g["case_id"]).mean().dropna())
            if len(v):
                ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                rows.append({"stratum": "risk" if risky else "non-risk", "dimension": dim,
                             "n": len(v), "mean": v.mean(), "±95%CI": ci})
    tbl = pd.DataFrame(rows)
    lines.append(to_md(tbl) if not tbl.empty else "_No P1 data yet._")
    lines.append("\n_n counts CASES (scores averaged over raters on the 8 shared cases), "
                 "not rating rows._\n")
    lines.append("\n_Never pool across strata: the P1 set is 24 risk / 12 non-risk, not pool-representative._\n")
    _scope_finding(p1, lines)
    return tbl


def _scope_finding(p1: pd.DataFrame, lines: list[str]) -> None:
    """K6 on the cases it can actually discriminate.

    Until 2026-08-17 this could not be reported at all: K6 was rated on every
    briefing, but which cases raised a non-depression condition was never
    recorded anywhere, so the discriminating subset could not be cut out of the
    mean. `cakupan` supplies that flag, and K6 is now blank on pure-depression
    cases — so a K6 row IS a discriminating case and no filtering is needed
    beyond dropping the blanks.

    Reported by CASE (averaged over raters), like every other table here.
    """
    if "cakupan" not in p1.columns:
        return
    coerce(p1, ["cakupan", "K6"])
    lines.append("\n### 4b. Scope — do the briefings respect the guideline's limits?\n")
    flag = p1.groupby("case_id")["cakupan"].mean().dropna()
    n_mixed = int((flag > 0.5).sum())
    lines.append(f"_{n_mixed} of {len(flag)} rated briefings concern a case judged to "
                 "raise a condition outside depression. Which condition is in the rater's "
                 "`catatan` as free text, not coded._\n")
    if "K6" not in p1.columns:
        return
    v = p1.dropna(subset=["K6"]).groupby("case_id")["K6"].mean()
    if v.empty:
        lines.append("\n_No K6 ratings yet._\n")
        return
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    lines.append(f"\n**K6 on those cases: mean {v.mean():.2f}"
                 + (f" ± {ci:.2f}" if pd.notna(ci) else "")
                 + f", n = {len(v)} cases.** 1 = the briefing framed everything as "
                 "depression; 5 = it flagged the other condition, named the "
                 "guideline's limit, and routed onward.\n")
    lines.append(f"\n_K6 is blank on the {len(flag) - n_mixed} pure-depression "
                 "briefings by design — a skip, not a zero — so its n is smaller "
                 "than the dimensions above and the two must not be compared._\n")
    # Pre-registered constraint, still binding: the double-rated packets were
    # drawn before this item existed, so agreement on the subset that carries the
    # finding may be unavailable. Say so from the data rather than assuming.
    if "rater" in p1.columns:
        shared = (p1.dropna(subset=["K6"]).groupby("case_id")["rater"].nunique())
        n_double = int((shared >= 2).sum())
        if n_double < 2:
            lines.append(f"\n> **No rater-independent scope claim.** Only {n_double} "
                         f"of the discriminating cases {'was' if n_double == 1 else 'were'} "
                         "rated by both psychologists, so κ/ICC cannot be computed on the subset "
                         "that carries this finding. Report it as single-rater. "
                         "See `protocol_amendment_20260806.md` §2.\n")


def analysis_triangulation(p1: pd.DataFrame, primary_tbl: pd.DataFrame,
                           p2long: pd.DataFrame, asg: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n## 5. Triangulation — briefing quality vs the with−without gap\n")
    coerce(p1, ["global"])
    # per counselor case: P1 briefing global (mean over raters) and (with-without) global gap
    counselor_cases = set(asg.loc[asg["in_counselor_sample"] == True, "study_id"])  # noqa: E712
    brief = (p1[p1["case_id"].isin(counselor_cases)]
             .groupby("case_id")["global"].mean().rename("brief_quality"))
    coerce(p2long, ["global"])
    wide = p2long.pivot_table(index="study_id", columns="condition", values="global", aggfunc="mean")
    if not {"chatbot", "no_chatbot"}.issubset(wide.columns):
        lines.append("_Need P2 with/without scores._\n"); return
    gap = (wide["chatbot"] - wide["no_chatbot"]).rename("gap")
    merged = pd.concat([brief, gap], axis=1).dropna()
    if len(merged) >= 4:
        rho, p = stats.spearmanr(merged["brief_quality"], merged["gap"])
        lines.append(f"Spearman ρ(briefing quality, with−without gap) = {rho:.3f} "
                     f"(p = {p:.3f}, n = {len(merged)}). Positive = better briefings → larger counselor gain.\n")
    else:
        lines.append("_Not enough paired cases yet._\n")


def analysis_calibration(p1: pd.DataFrame, asg: pd.DataFrame, auto: Path, ragas: Path, lines: list[str]) -> None:
    lines.append("\n## 6. Human vs automatic faithfulness calibration (full-form packets)\n")
    coerce(p1, ["dukungan_konteks"])
    full = set(asg.loc[asg["p1_form"] == "full", "study_id"])
    human = (p1[p1["case_id"].isin(full)].groupby("case_id")["dukungan_konteks"].mean()
             .rename("human_groundedness"))
    frames = [human]
    if auto.exists():
        a = pd.read_csv(auto)
        if {"study_id", "faithfulness"}.issubset(a.columns):
            frames.append(a.set_index("study_id")["faithfulness"].rename("judge_faithfulness"))
    if ragas.exists():
        r = pd.read_csv(ragas)
        col = next((c for c in r.columns if "faith" in c.lower()), None)
        if col and "study_id" in r.columns:
            frames.append(r.set_index("study_id")[col].rename("ragas_faithfulness"))
    merged = pd.concat(frames, axis=1).dropna()
    if len(merged) >= 4 and merged.shape[1] >= 2:
        for auto_col in [c for c in merged.columns if c != "human_groundedness"]:
            rho, p = stats.spearmanr(merged["human_groundedness"], merged[auto_col])
            lines.append(f"- ρ(human, {auto_col}) = {rho:.3f} (p = {p:.3f}, n = {len(merged)})")
        lines.append("")
    else:
        lines.append("_Need human groundedness + automatic faithfulness on the same full-form cases._\n")


def analysis_first_block(p2long: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """Design v2 §7: the first-block sensitivity analysis.

    Block 1 is the only carryover-free half of the crossover — nobody has seen
    the tool yet. But the assignment makes it a BETWEEN-COUNSELOR comparison:
    every block-1 chatbot answer is counselor_1's and every block-1 no-chatbot
    answer is counselor_2's (verified in the frozen file, 12 v 12). So the
    pairing that powers the primary analysis does not exist here, and any
    difference is condition + counselor together.

    That is why this is unpaired (Mann-Whitney, not Wilcoxon signed-rank) and why
    the confound is printed as a banner rather than a footnote: a reader who sees
    "carryover-free" without "confounded with counselor" will over-trust it. It
    is a sensitivity check on the primary, never a result in its own right.
    """
    lines.append("\n## 7. First-block sensitivity (carryover-free half)\n")
    if p2long.empty or "block" not in p2long.columns:
        lines.append("_No P2 data yet._\n")
        return pd.DataFrame()

    b1 = p2long[pd.to_numeric(p2long["block"], errors="coerce") == 1].copy()
    if b1.empty:
        lines.append("_No block-1 rows._\n")
        return pd.DataFrame()

    dims = ["global"] + QUALITY_P2 + SAFETY
    coerce(b1, dims)

    # Is condition perfectly predicted by counselor here? Determined from the
    # data, not assumed, so a re-randomised design does not silently keep a
    # banner that no longer applies.
    confounded = False
    if "counselor" in b1.columns:
        per = b1.groupby("counselor")["condition"].nunique()
        confounded = bool(len(per) > 1 and (per <= 1).all())

    rows = []
    for dim in dims:
        if dim not in b1.columns:
            continue
        g = b1.dropna(subset=[dim])
        # The ANSWER is the unit, not the rating row. The 4 shared cases are
        # scored by both psychologists, so counting rows would treat two ratings
        # of one answer as two independent observations — inflating n from 12 to
        # 14 per arm and shrinking every p-value. Average over raters first, the
        # same convention analysis_p1_strata and analysis_primary already use.
        if "response_id" in g.columns:
            g = (g.groupby(["response_id", "condition"])[dim].mean()
                 .reset_index())
        a = g.loc[g["condition"] == "chatbot", dim]
        c = g.loc[g["condition"] == "no_chatbot", dim]
        if len(a) < 2 or len(c) < 2:
            continue
        r = {"dimension": dim, "n_chatbot": len(a), "n_no_chatbot": len(c),
             "median_chatbot": float(a.median()), "median_no_chatbot": float(c.median()),
             "median_diff": float(a.median() - c.median())}
        try:
            u, p = stats.mannwhitneyu(a, c, alternative="two-sided")
            r["mannwhitney_p"] = float(p)
            # Rank-biserial, not dz: a paired effect size would misdescribe an
            # unpaired contrast. scipy returns U for the FIRST sample (chatbot),
            # so 2U/(n1*n2) - 1 runs -1..+1 with positive = chatbot higher.
            r["rank_biserial"] = float((2 * u) / (len(a) * len(c)) - 1)
        except ValueError:
            r["mannwhitney_p"] = None
        rows.append(r)

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        lines.append("_Not enough block-1 rows in both conditions._\n")
        return tbl

    if confounded:
        lines.append(
            "> **CONFOUNDED — do not read as an effect of the chatbot.** In block 1 "
            "every chatbot answer comes from one counselor and every no-chatbot "
            "answer from the other, so `condition` and `counselor` are the same "
            "variable. This half is carryover-free, and that is the only thing it "
            "buys. A difference here is the chatbot *or* the two counselors "
            "differing; the design cannot separate them at n = 2 counselors.\n")
    lines.append(to_md(tbl))
    lines.append("\n_Unpaired (Mann-Whitney): block 1 holds one condition per "
                 "counselor, so there are no within-question pairs to test. "
                 "`rank_biserial` is the unpaired effect size; positive = "
                 "with-chatbot higher. No multiplicity adjustment — this is a "
                 "sensitivity check on the primary, not a second family of "
                 "hypotheses._\n")
    return tbl


def analysis_direction_agreement(p2long: pd.DataFrame, asg: pd.DataFrame,
                                 lines: list[str]) -> pd.DataFrame:
    """Design v2 §7: pair-level direction agreement on the shared counselor cases.

    ICC on individual scores answers "do the raters put answers on the same
    scale?". This answers the question the study actually turns on: for a case
    where both psychologists saw both of a counselor's answers, do they agree on
    WHICH ONE WAS BETTER? Two raters can disagree substantially on absolute
    scores and still agree perfectly on direction — and it is direction that the
    with-without claim rests on.

    Four cases (Q10, Q22, Q26, Q38), so this is a tally, not an inference. No
    kappa: with n = 4 and a degenerate marginal it is unstable enough to mislead.
    """
    lines.append("\n## 8. Direction agreement on the shared counselor cases\n")
    if p2long.empty or "study_id" not in p2long.columns:
        lines.append("_No P2 data yet._\n")
        return pd.DataFrame()

    shared = set(asg.loc[(asg.get("in_counselor_sample") == True)          # noqa: E712
                         & (asg.get("p1_rater") == "both"), "study_id"])
    df = p2long[p2long["study_id"].isin(shared)].copy()
    if df.empty or "rater" not in df.columns:
        lines.append(f"_No scores yet on the {len(shared)} shared counselor cases._\n")
        return pd.DataFrame()
    coerce(df, ["global"])

    rows = []
    for case, g in df.groupby("study_id"):
        w = g.pivot_table(index="rater", columns="condition", values="global", aggfunc="mean")
        if not {"chatbot", "no_chatbot"}.issubset(w.columns) or w.shape[0] < 2:
            continue
        gaps = (w["chatbot"] - w["no_chatbot"]).dropna()
        if len(gaps) < 2:
            continue
        signs = np.sign(gaps.to_numpy())
        rows.append({
            "study_id": case,
            **{f"gap_{r}": float(v) for r, v in gaps.items()},
            "agree_direction": bool(len(set(signs)) == 1),
            "both_zero": bool((signs == 0).all()),
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        lines.append(f"_Need both raters on both conditions of a shared case "
                     f"(expected {len(shared)}: {', '.join(sorted(shared))})._\n")
        return tbl

    n_agree = int(tbl["agree_direction"].sum())
    lines.append(to_md(tbl))
    lines.append(f"\n**{n_agree} of {len(tbl)} shared cases agree on the direction "
                 "of with−without.**\n")
    if tbl["both_zero"].any():
        lines.append("\n_Ties (gap = 0) count as agreement only when both raters tie; "
                     "a tie against a non-zero gap counts as disagreement._\n")
    lines.append("\n_Tally, not a test: 4 cases cannot support a kappa or an "
                 "interval. Read it as a qualitative check on whether the primary "
                 "endpoint's direction survives a change of rater._\n")
    return tbl


def _exit_numeric(s: pd.Series) -> pd.Series:
    """'4 - setuju' / '4' / 4 -> 4.0. The workbook stores the labelled string."""
    return pd.to_numeric(
        s.astype(str).str.strip().str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce")


def analysis_k1_helpfulness(exit_path: Path, p2long: pd.DataFrame,
                            lines: list[str]) -> pd.DataFrame:
    """Design v2 §7: K1 perceived helpfulness, and its relation to the gap.

    §7 asks to "correlate ... K1 perceived helpfulness with both". That is not
    estimable here and the honest move is to say so rather than emit a number:
    K1 is answered ONCE per counselor at the end of the study, so it has exactly
    two values. A correlation on n = 2 is either +1 or -1 by construction,
    carries no information, and would be the single most over-read figure in the
    report.

    So this reports K1 descriptively per counselor and puts each counselor's mean
    with-without gap beside it. Two rows, ordered — a reader can see whether the
    counselor who liked the tool more also gained more, which is the substantive
    question, without a spurious rho implying it was tested.
    """
    lines.append("\n## 9. K1 — counselor perceived helpfulness\n")
    if not exit_path.exists():
        lines.append(f"_No exit questionnaire at `{exit_path}`. Transcribe the "
                     "«Kuesioner Akhir» sheets and pass `--counselor-exit` "
                     "(schema in this script's docstring)._\n")
        return pd.DataFrame()

    raw = pd.read_csv(exit_path)
    need = {"counselor", "kind", "response"}
    if not need.issubset(raw.columns):
        lines.append(f"_`{exit_path.name}` needs columns {sorted(need)}; "
                     f"found {sorted(raw.columns)}._\n")
        return pd.DataFrame()

    raw["value"] = _exit_numeric(raw["response"])
    rows = []
    for counselor, g in raw.groupby("counselor"):
        skala = g.loc[g["kind"] == "skala", "value"].dropna()
        angka = g.loc[g["kind"] == "angka", "value"].dropna()
        rec = {"counselor": counselor,
               "n_likert": len(skala),
               "likert_mean_1_5": float(skala.mean()) if len(skala) else None,
               "overall_0_10": float(angka.iloc[0]) if len(angka) else None}
        if "section" in g.columns:
            man = g.loc[(g["kind"] == "skala")
                        & g["section"].astype(str).str.contains("Manfaat", case=False,
                                                                na=False), "value"].dropna()
            if len(man):
                rec["manfaat_mean_1_5"] = float(man.mean())
        rows.append(rec)
    tbl = pd.DataFrame(rows)

    # Each counselor's own observed benefit, to sit beside their perceived one.
    #
    # NOT a per-question paired gap: within one counselor every question appears
    # in exactly one condition (the pair is ACROSS counselors — counselor_1
    # answers a question with the tool, counselor_2 answers the same question
    # without). Pivoting per study_id inside a counselor therefore yields nothing.
    # The counselor-level quantity is their mean across their own two blocks,
    # which for a single counselor makes condition identical to block/period —
    # so it carries any practice or fatigue effect with it. Stated below.
    if not p2long.empty and {"counselor", "condition"}.issubset(p2long.columns):
        coerce(p2long, ["global"])
        gaps = []
        for counselor, g in p2long.groupby("counselor"):
            a = g.loc[g["condition"] == "chatbot", "global"].dropna()
            c = g.loc[g["condition"] == "no_chatbot", "global"].dropna()
            if len(a) and len(c):
                gaps.append({"counselor": counselor,
                             "mean_global_chatbot": float(a.mean()),
                             "mean_global_no_chatbot": float(c.mean()),
                             "observed_benefit": float(a.mean() - c.mean()),
                             "n_answers": int(len(a) + len(c))})
        if gaps:
            tbl = tbl.merge(pd.DataFrame(gaps), on="counselor", how="left")

    lines.append(to_md(tbl))
    lines.append(
        "\n> **Not correlated on purpose.** K1 is answered once per counselor, so "
        "it has two values. Design v2 §7 asks for a correlation with briefing "
        "quality and with the with−without gap; at n = 2 that coefficient is ±1 "
        "by construction and carries no information, so it is deliberately not "
        "computed. The columns are placed side by side instead: read whether the "
        "counselor who rated the tool higher also showed the larger benefit, and "
        "treat it as descriptive.\n")
    lines.append("\n_`observed_benefit` is that counselor's mean holistic score "
                 "with the tool minus without, across their own two blocks. It is "
                 "NOT the paired per-question gap of §1: within one counselor each "
                 "question appears in a single condition, so for that counselor "
                 "condition and block/period are the same variable and any practice "
                 "or fatigue effect is included in this number._\n")
    lines.append("\n_Likert items are 1–5 (`skala`); the overall usefulness item "
                 "is 0–10 (`angka`). Free-text answers are not summarised here — "
                 "read them directly in the workbooks._\n")
    return tbl


def analysis_counselor_per_case(k1_path: Path, p2long: pd.DataFrame,
                                lines: list[str]) -> pd.DataFrame:
    """Section 11: the per-case counselor ratings — including the automation-bias probe.

    `convert_counselor_answers.py` has always written these to
    data/derived/counselor_k1.csv, and until 2026-08-17 nothing read the file:
    5 items x 12 cases x 2 counselors on the with-tool arm plus confidence on the
    without arm, ~144 ratings, collected and dropped. `JUSTIFIKASI_instrumen.md`
    §2.1 presents the confidence-in-both-conditions design as a deliberate
    strength — "apakah alat bantu menaikkan keyakinan lebih cepat daripada
    menaikkan mutu?" — so the claim had no code behind it.

    Reported DESCRIPTIVELY. `condition` is aliased with counselor x block on this
    arm (see `_aliasing_note`), so a confidence difference cannot be attributed to
    the tool; what it can do is sit next to the quality gap and show whether the
    two move together.
    """
    lines.append("\n## 11. Counselor per-case ratings, and the automation-bias probe\n")
    if not k1_path.exists():
        lines.append(f"_No per-case counselor ratings yet ({k1_path.name} absent) — run "
                     "convert_counselor_answers.py._\n")
        return pd.DataFrame()
    k1 = pd.read_csv(k1_path)
    coerce(k1, ["k1_bantu_kasus", "k1_bantu_risiko", "k1_yakin", "jumlah_kata"])

    # --- with-tool perceptions (asked only in that condition)
    tool = k1[k1["condition"] == "chatbot"]
    rows = []
    for col, label in (("k1_bantu_kasus", "membantu untuk kasus ini"),
                       ("k1_bantu_risiko", "membantu mengenali tanda bahaya"),
                       ("k1_yakin", "keyakinan pada jawaban")):
        v = pd.to_numeric(tool.get(col), errors="coerce").dropna()
        if len(v):
            rows.append({"item": label, "n": len(v), "mean": v.mean(), "median": v.median()})
    if rows:
        lines.append("**Persepsi konselor pada kondisi DENGAN alat bantu (1–5)**\n")
        lines.append(to_md(pd.DataFrame(rows)))
    if "k1_pengaruh" in tool:
        infl = tool["k1_pengaruh"].dropna().value_counts()
        lines.append("\n_Pengaruh ringkasan terhadap jawaban akhir: "
                     + ", ".join(f"{k} {v}" for k, v in infl.items()) + "._\n")
    if "k1_keliru" in tool:
        wrong = tool[tool["k1_keliru"].astype(str).str.strip().str.lower() == "ya"]
        lines.append(f"\n_Konselor menandai isi ringkasan keliru/berbahaya pada "
                     f"**{len(wrong)} dari {len(tool)}** kasus._\n")
        for _, w in wrong.iterrows():
            note = str(w.get("k1_keliru_teks", "") or "").strip()
            lines.append(f"\n> `{w['study_id']}` — {note or '(tanpa penjelasan)'}\n")

    # --- the probe: does confidence rise faster than quality?
    conf = (k1.pivot_table(index="study_id", columns="condition",
                           values="k1_yakin", aggfunc="mean"))
    if not {"chatbot", "no_chatbot"}.issubset(conf.columns):
        lines.append("\n_Keyakinan belum tersedia di kedua kondisi._\n")
        return k1
    conf_gap = (conf["chatbot"] - conf["no_chatbot"]).rename("gap_keyakinan")
    coerce(p2long, ["global"])
    qual = p2long.pivot_table(index="study_id", columns="condition",
                              values="global", aggfunc="mean")
    lines.append(f"\n**Selisih keyakinan konselor (dengan − tanpa): median "
                 f"{conf_gap.median():+.2f}, rerata {conf_gap.mean():+.2f}, "
                 f"n = {conf_gap.notna().sum()} kasus.**\n")
    if {"chatbot", "no_chatbot"}.issubset(qual.columns):
        qual_gap = (qual["chatbot"] - qual["no_chatbot"]).rename("gap_mutu")
        both = pd.concat([conf_gap, qual_gap], axis=1).dropna()
        if len(both) >= 4:
            rho, pv = stats.spearmanr(both["gap_keyakinan"], both["gap_mutu"])
            lines.append(f"\nSpearman ρ(kenaikan keyakinan, kenaikan mutu) = {rho:.3f} "
                         f"(p = {pv:.3f}, n = {len(both)}).\n")
            lines.append(
                "\n> **Cara membacanya.** Keyakinan naik sementara mutu tidak — ρ mendekati "
                "nol atau negatif dengan selisih keyakinan positif — adalah pola **bias "
                "otomasi**: konselor merasa lebih mampu tanpa menjadi lebih baik. Untuk alat "
                "yang diserahkan kepada non-spesialis itu temuan keselamatan, bukan sekadar "
                "temuan sikap. DESKRIPTIF saja: `condition` beralias dengan counselor × block "
                "di lengan ini, jadi selisihnya tidak dapat diatribusikan kepada alat bantu._\n")
    else:
        lines.append("\n_Butuh skor psikolog di kedua kondisi untuk membandingkannya "
                     "dengan kenaikan mutu._\n")
    return k1


def analysis_bagian0(p1: pd.DataFrame, asg: pd.DataFrame, questions: Path,
                     lines: list[str]) -> pd.DataFrame:
    """Validate the deployed safety screen against clinical judgement.

    BAGIAN 0 asks each psychologist, independently, whether the QUESTION shows
    danger signs. The design protects that judgement carefully — it is why the
    packet never prints the system's verdict — and until now it was collected and
    never used.

    It is the only expert reference the study produces for the thing that gates
    the whole safety layer: the keyword + LLM risk screen, which ships
    unvalidated. Two psychologists over 36 cases is a small but real reference
    standard, and the comparison costs nothing because the data is already being
    collected.

    Reported as a confusion matrix with sensitivity/specificity rather than a
    single agreement number, because the two error directions are not
    interchangeable here. A false negative means a person disclosing risk gets no
    safety banner; a false positive means a redundant warning. The screen is
    deliberately tuned toward the second, and the table has to show whether that
    is what actually happens.

    Neither side is ground truth: the psychologists are the more defensible
    reference, but they are two people, so their agreement WITH EACH OTHER is
    reported first — a reference standard the raters cannot reproduce between
    themselves does not support strong claims about the screen.
    """
    lines.append("\n## 10. BAGIAN 0 — safety screen vs clinical judgement\n")
    if p1.empty or "bagian0_risiko" not in p1.columns:
        lines.append("_No BAGIAN 0 data yet (no `bagian0_risiko` column in the P1 "
                     "scores)._\n")
        return pd.DataFrame()

    coerce(p1, ["bagian0_risiko"])
    df = p1.merge(asg[["study_id", "risky"]], left_on="case_id", right_on="study_id",
                  how="left").dropna(subset=["bagian0_risiko", "risky"])
    if df.empty:
        lines.append("_No BAGIAN 0 rows joined to the frozen risk flag._\n")
        return pd.DataFrame()
    df["system"] = df["risky"].astype(bool).astype(int)
    df["rater_says"] = pd.to_numeric(df["bagian0_risiko"], errors="coerce").round()

    # --- 1. do the two psychologists agree with EACH OTHER? -------------------
    shared = df.pivot_table(index="case_id", columns="rater", values="rater_says",
                            aggfunc="mean").dropna()
    if shared.shape[1] == 2 and len(shared) >= 2:
        a, b = shared.to_numpy().T
        k = weighted_kappa(a, b)          # binary: quadratic weights == unweighted
        agree = float((a.round() == b.round()).mean())
        lines.append(f"**Between the two psychologists** (n = {len(shared)} shared "
                     f"cases): raw agreement {agree:.0%}"
                     + (f", Cohen's κ = {k:.3f}" if k is not None else
                        ", κ undefined (no variation)") + ".\n")
        if k is not None and k < 0.6:
            lines.append("\n> κ below 0.60 — the expert reference is itself unstable, "
                         "so read everything below as indicative only.\n")
    else:
        lines.append("_Not enough shared cases to check agreement between raters._\n")

    # --- 2. screen vs each rater ---------------------------------------------
    rows = []
    for label, g in list(df.groupby("rater")) + [("(semua penilaian)", df)]:
        tp = int(((g["system"] == 1) & (g["rater_says"] == 1)).sum())
        fp = int(((g["system"] == 1) & (g["rater_says"] == 0)).sum())
        fn = int(((g["system"] == 0) & (g["rater_says"] == 1)).sum())
        tn = int(((g["system"] == 0) & (g["rater_says"] == 0)).sum())
        rows.append({
            "reference": label, "n": len(g),
            "both_risk": tp, "system_only": fp, "rater_only": fn, "neither": tn,
            "agreement": (tp + tn) / len(g) if len(g) else None,
            # psychologist as reference: did the screen catch what a clinician
            # called risk, and how often did it fire when they did not?
            "sensitivity": tp / (tp + fn) if (tp + fn) else None,
            "specificity": tn / (tn + fp) if (tn + fp) else None,
        })
    tbl = pd.DataFrame(rows)
    lines.append(to_md(tbl))
    lines.append("\n_`rater_only` are the misses that matter: the psychologist saw "
                 "danger signs and the screen did not fire, so no safety banner was "
                 "shown. `system_only` is the tolerated direction — a redundant "
                 "warning. Sensitivity/specificity treat the psychologist as the "
                 "reference; neither side is ground truth._\n")

    # --- 3. which stage would catch them? -------------------------------------
    # The frozen flag is keyword OR LLM combined, which hides where the screen
    # actually fails. The keyword pre-screen is pure and deterministic, so it can
    # be recomputed exactly as deployed — no model, no sampling — and that is the
    # half a config edit can fix.
    if questions.exists():
        try:
            import yaml

            from depression_rag.chatbot import risk_screen
            kws = tuple(k.lower() for k in yaml.safe_load(
                (ROOT / "configs" / "chatbot.yaml").read_text(encoding="utf-8")
            )["safety"]["keywords"])
            qtext = {}
            for line in questions.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    qtext[rec["study_id"]] = rec["question"]
            sub = df[df["case_id"].isin(qtext)]
            if not sub.empty:
                sub = sub.assign(kw=[1 if risk_screen(qtext[c], kws) else 0
                                     for c in sub["case_id"]])
                expert = sub[sub["rater_says"] == 1]
                caught = int((expert["kw"] == 1).sum())
                lines.append(
                    f"\n**Keyword pre-screen alone** (recomputed from the current "
                    f"`configs/chatbot.yaml`, deterministic): catches {caught} of "
                    f"{len(expert)} ratings a psychologist judged risky. The "
                    f"remainder depend entirely on the LLM stage, which samples and "
                    "is skipped whenever the generator is unavailable.\n")
        except Exception as e:  # noqa: BLE001
            lines.append(f"\n_Keyword-stage breakdown unavailable: {e}_\n")

    lines.append("\n_This is the study's only validation of the deployed risk "
                 "screen. It also sets the denominator for §2: a rating marked "
                 "«tidak berlaku» on the safety gate is a `rater_only`/`neither` "
                 "case in this table._\n")
    return tbl


# ------------------------------------------------------------------ main ------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignments", default=str(ROOT / "data/derived/eval_case_assignments.json"))
    ap.add_argument("--sealed", default=str(
        ROOT / "outputs/evaluation/study_kit/03_ADMIN_PENELITI/SEALED_p2_key.csv"),
        help="response_id -> condition key, written by build_psychologist_workbooks.py")
    ap.add_argument("--p1-scores", default=str(ROOT / "outputs/analysis/p1_scores.csv"))
    ap.add_argument("--p2-scores", default=str(ROOT / "outputs/analysis/p2_scores.csv"))
    ap.add_argument("--auto", default=str(ROOT / "outputs/analysis/study_eval_chatbot.csv"))
    ap.add_argument("--ragas", default=str(ROOT / "outputs/analysis/ragas_chatbot.csv"))
    ap.add_argument("--questions",
                    default=str(ROOT / "data/derived/questions_alodokter_sample50.jsonl"),
                    help="study questions, for the deterministic keyword-stage "
                         "breakdown in the BAGIAN 0 analysis")
    ap.add_argument("--counselor-exit",
                    default=str(ROOT / "outputs/analysis/counselor_exit.csv"),
                    help="«Kuesioner Akhir» transcribed long: "
                         "counselor,item_no,kind,response[,section] (see docstring)")
    ap.add_argument("--counselor-k1",
                    default=str(ROOT / "data/derived/counselor_k1.csv"),
                    help="per-case counselor ratings from convert_counselor_answers.py")
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis/study_analysis_report.md"))
    args = ap.parse_args(argv)

    asg = load_assignment(Path(args.assignments))
    sealed_path = Path(args.sealed)
    if not sealed_path.exists():
        sys.exit(f"sealed key not found: {sealed_path}\n"
                 "Run scripts/build_psychologist_workbooks.py --counselor-answers "
                 "data/derived/counselor_answers.jsonl after collecting counselor answers; "
                 "it writes this key.")
    sealed = pd.read_csv(sealed_path)
    p1, p2 = load_scores(args.p1_scores, args.p2_scores)
    p2long = build_p2_long(p2, sealed, asg) if not p2.empty else pd.DataFrame()

    lines = ["# Study analysis report\n",
             f"_Generated from {len(p1)} P1 rows, {len(p2)} P2 rows; "
             f"assignment seed 20260628. Design v2 §7._\n"]
    primary_tbl = analysis_primary(p2long, lines) if not p2long.empty else pd.DataFrame()
    if not p2long.empty:
        analysis_safety(p2long, lines)
    analysis_irr(p1, p2long if not p2long.empty else pd.DataFrame(columns=["study_id"]), asg, lines)
    strata = analysis_p1_strata(p1, asg, lines) if not p1.empty else pd.DataFrame()
    if not p1.empty and not p2long.empty:
        analysis_triangulation(p1, primary_tbl, p2long, asg, lines)
    if not p1.empty:
        analysis_calibration(p1, asg, Path(args.auto), Path(args.ragas), lines)

    # The remaining three run unconditionally: each prints why it is empty rather
    # than vanishing from the report. A pre-specified analysis that silently omits
    # itself when its input is missing is indistinguishable, to a reader, from one
    # that was never specified.
    first_block = analysis_first_block(p2long, lines)
    direction = analysis_direction_agreement(
        p2long if not p2long.empty else pd.DataFrame(columns=["study_id"]), asg, lines)
    k1 = analysis_k1_helpfulness(Path(args.counselor_exit), p2long, lines)
    bagian0 = analysis_bagian0(p1, asg, Path(args.questions), lines)
    per_case = analysis_counselor_per_case(Path(args.counselor_k1), p2long, lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # tidy CSVs for plotting
    if not primary_tbl.empty:
        primary_tbl.to_csv(out.with_name("analysis_primary_pairs.csv"), index=False)
    if not strata.empty:
        strata.to_csv(out.with_name("analysis_p1_by_risk.csv"), index=False)
    if not first_block.empty:
        first_block.to_csv(out.with_name("analysis_first_block.csv"), index=False)
    if not direction.empty:
        direction.to_csv(out.with_name("analysis_direction_agreement.csv"), index=False)
    if not k1.empty:
        k1.to_csv(out.with_name("analysis_k1_helpfulness.csv"), index=False)
    if not bagian0.empty:
        bagian0.to_csv(out.with_name("analysis_bagian0_screen.csv"), index=False)
    if per_case is not None and not per_case.empty:
        per_case.to_csv(out.with_name("analysis_counselor_per_case.csv"), index=False)
    print(f"wrote {out}")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
