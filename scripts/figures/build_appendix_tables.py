#!/usr/bin/env python3
"""Build the supplementary tables (S1-S7) cited by the manuscript.

    python scripts/figures/build_appendix_tables.py

Writes one CSV per table into documents/thesis_clean/appendix/. Free-text
columns are deliberately excluded: participant notes need a manual
anonymisation pass before they can be published, and that is not something a
script should decide.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "documents/thesis_clean/appendix"
OUT.mkdir(parents=True, exist_ok=True)

written: list[tuple[str, int, str]] = []


def write(name: str, header: list[str], rows: list[list], note: str) -> None:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    written.append((name, len(rows), note))


def f(x, nd=3):
    return "" if x in ("", None) else f"{float(x):.{nd}f}"


# ---------------------------------------------------------------- S1
def s1_retrieval_grid():
    metrics = {r["index_id"]: r for r in
               csv.DictReader((ROOT / "outputs/analysis/retrieval_metrics.csv").open())}
    rows = []
    for iid, m in sorted(metrics.items()):
        meta_p = ROOT / "outputs/indexes" / iid / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        rows.append([
            iid, m["chunk_config"], m["model"],
            meta.get("max_seq_len", ""), meta.get("n_chunks", ""),
            meta.get("n_vectors", ""), meta.get("truncated_chunks", ""),
            f(meta.get("truncation_rate", ""), 3),
            f(m["ndcg@5"]), f(m["ndcg@5_ci_low"]), f(m["ndcg@5_ci_high"]),
            f(m["recall@5"]), f(m["mrr@5"]), f(m["recall@10"]), m["n_queries"],
        ])
    write("Table_S1a_retrieval_grid.csv",
          ["index_id", "chunk_config", "embedding_model", "encoder_window_tokens",
           "n_chunks", "n_vectors", "truncated_chunks", "truncation_rate",
           "nDCG@5", "nDCG@5_CI_low", "nDCG@5_CI_high", "Hit@5", "MRR@5",
           "Hit@10", "n_questions"],
          rows,
          "Every scored index. Hit@k is the artifact's recall@k column, renamed as in the manuscript.")


def s1b_holm():
    src = ROOT / "outputs/analysis/retrieval_significance.md"
    rows = []
    for line in src.read_text().splitlines():
        if line.startswith("| ") and "---" not in line and "challenger" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 9:
                rows.append(cells[:9])
    write("Table_S1b_holm_comparisons.csv",
          ["challenger", "mean_nDCG@5", "delta_winner_minus_challenger",
           "CI_of_delta", "p_bootstrap", "p_wilcoxon", "p_holm",
           "rank_biserial", "significant"],
          rows,
          "Declared paired-comparison family; winner = structure-512-0__e5-large.")


def s1c_by_question_type():
    """Table S1c - the deployed index by question type.

    The manuscript reports only the two strata the tie-breakers used, so the
    full profile lives here.
    """
    rows = []
    src = ROOT / "outputs/analysis/retrieval_metrics_by_type.csv"
    by_type = [r for r in csv.DictReader(src.open())
               if r["index_id"] == "structure-512-0__e5-large__mv"]
    by_type.sort(key=lambda r: float(r["ndcg@5"]), reverse=True)
    for r in by_type:
        n = int(r["n"])
        rows.append([r["question_type"], n,
                     f(r["ndcg@5"]), f(r["recall@5"]), f(r["recall@10"]),
                     round(float(r["recall@5"]) * n)])
    write("Table_S1c_deployed_by_question_type.csv",
          ["question_type", "n_questions", "nDCG@5", "Hit@5", "Hit@10",
           "questions_with_a_relevant_passage_in_top_5"],
          rows,
          "Deployed multi-vector index, all 121 questions, ordered by nDCG@5.")


# ---------------------------------------------------------------- S2
def s2_per_briefing():
    judge = {r["study_id"]: r for r in
             csv.DictReader((ROOT / "outputs/analysis/study_eval_chatbot.csv").open())}
    ragas = {r["study_id"]: r for r in
             csv.DictReader((ROOT / "outputs/analysis/ragas_chatbot.csv").open())}
    rows = []
    for q in sorted(judge):
        j, g = judge[q], ragas[q]
        in_scope = int(j["sections_present"]) == 4
        rows.append([q, "yes" if in_scope else "no (out-of-scope reply)",
                     j["sections_present"], f(j["faithfulness"]),
                     f(g["faithfulness"]), f(g["answer_relevancy"]),
                     f(g["llm_context_precision_without_reference"])])
    write("Table_S2_per_briefing_automatic.csv",
          ["study_id", "in_scope", "sections_present", "claim_support",
           "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision"],
          rows,
          "All 50 outputs. Manuscript statistics use the 48 in-scope rows.")


# ---------------------------------------------------------------- S3, S4
P1_ITEMS = [("kebenaran", "clinical_correctness"), ("kelengkapan", "completeness"),
            ("relevansi", "case_relevance"), ("kejelasan", "clarity_practicality"),
            ("K1", "harm_avoidance"), ("K2", "risk_recognition"),
            ("K3", "immediate_escalation"), ("K4", "resources_referral"),
            ("K5", "decision_space_role_boundary"), ("K6", "scope_appropriateness"),
            ("gerbang", "safety_gate"), ("kecukupan_konteks", "context_sufficient"),
            ("dukungan_konteks", "support_for_statements"),
            ("halusinasi", "unsupported_statement"), ("global", "global_0_10"),
            ("bagian0_risiko", "rater_danger_sign_judgment"),
            ("cakupan", "condition_outside_depression")]


def s3_psychologist_items():
    rows = []
    for r in csv.DictReader((ROOT / "outputs/analysis/p1_scores.csv").open()):
        rows.append([r["rater"], r["case_id"]] + [r[c] for c, _ in P1_ITEMS])
    write("Table_S3_psychologist_briefing_ratings.csv",
          ["rater", "case_id"] + [n for _, n in P1_ITEMS], rows,
          "44 ratings of 36 packets. Free-text notes excluded pending anonymisation review.")


def s4_scope_by_rater():
    rows_in = list(csv.DictReader((ROOT / "outputs/analysis/p1_scores.csv").open()))
    out = []
    for rater in ("psychologist_1", "psychologist_2"):
        sub = [r for r in rows_in if r["rater"] == rater]
        cak = [r for r in sub if r["cakupan"] == "1"]
        k6 = [float(r["K6"]) for r in sub if r["K6"] not in ("", "None")]
        out.append([rater, len(sub), len(cak),
                    f"{100 * len(cak) / len(sub):.0f}%",
                    len(k6), f"{statistics.mean(k6):.2f}" if k6 else ""])
    allr = rows_in
    cak_all = [r for r in allr if r["cakupan"] == "1"]
    out.append(["all_ratings", len(allr), len(cak_all),
                f"{100 * len(cak_all) / len(allr):.0f}%",
                len([r for r in allr if r["K6"] not in ("", "None")]),
                f"{statistics.mean(float(r['K6']) for r in allr if r['K6'] not in ('', 'None')):.2f}"])
    write("Table_S4_scope_appropriateness_by_rater.csv",
          ["rater", "ratings", "condition_outside_depression_n",
           "condition_outside_depression_pct", "scope_item_rated_n",
           "scope_item_mean"], out,
          "Reported by rater because the conditional item was applied inconsistently.")


# ---------------------------------------------------------------- S5
def s5_counselor_process():
    rows = []
    for r in csv.DictReader((ROOT / "outputs/analysis/analysis_counselor_per_case.csv").open()):
        rows.append([r["study_id"], r["counselor"], r["condition"],
                     r["waktu_mulai"], r["waktu_selesai"], r["jumlah_kata"],
                     r["k1_yakin"], r["k1_bantu_kasus"], r["k1_bantu_risiko"],
                     r["k1_pengaruh"], r["k1_keliru"]])
    write("Table_S5_counselor_per_case_process.csv",
          ["study_id", "counselor", "condition", "start_time", "end_time",
           "word_count", "self_confidence_1_5", "helpful_for_case_1_5",
           "helpful_danger_signs_1_5", "influence_on_answer", "flagged_problem"],
          rows,
          "48 cases. Free-text problem descriptions excluded pending anonymisation review.")


# ---------------------------------------------------------------- S6
def s6_pairs():
    key = {r["response_id"]: r for r in
           csv.DictReader((ROOT / "outputs/evaluation/study_kit/03_ADMIN_PENELITI/SEALED_p2_key.csv").open())}
    cases = {c["question_id"]: c for c in
             json.loads((ROOT / "data/derived/eval_case_assignments.json").read_text())["cases"]}
    scores = defaultdict(list)
    for r in csv.DictReader((ROOT / "outputs/analysis/p2_scores.csv").open()):
        if r["global"]:
            scores[r["response_id"]].append(float(r["global"]))
    pairs = defaultdict(dict)
    for rid, vals in scores.items():
        k = key[rid]
        pairs[cases[k["question_id"]]["study_id"]][k["condition"]] = (
            statistics.mean(vals), k["counselor"], rid, len(vals))
    rows = []
    for q in sorted(pairs):
        v = pairs[q]
        if len(v) != 2:
            continue
        w, wo = v["chatbot"], v["no_chatbot"]
        rows.append([q, w[2], w[1], f"{w[0]:.2f}", w[3],
                     wo[2], wo[1], f"{wo[0]:.2f}", wo[3],
                     f"{w[0] - wo[0]:.2f}",
                     "with" if w[0] > wo[0] else "without" if w[0] < wo[0] else "tied"])
    write("Table_S6_pair_level_response_ratings.csv",
          ["study_id", "response_id_with_access", "counselor_with_access",
           "global_with_access", "n_raters_with", "response_id_no_access",
           "counselor_no_access", "global_no_access", "n_raters_without",
           "difference_with_minus_without", "direction"], rows,
          "24 matched pairs; double-rated responses averaged before pairing.")


# ---------------------------------------------------------------- S7
def s7_allocation():
    d = json.loads((ROOT / "data/derived/eval_case_assignments.json").read_text())
    rows = []
    for c in d["cases"]:
        cc = c.get("counselor_condition") or {}
        rows.append([c["study_id"], c["question_id"], c["block"],
                     "yes" if c["risky"] else "no",
                     "yes" if c.get("in_counselor_sample") else "no",
                     cc.get("counselor_1", ""), cc.get("counselor_2", ""),
                     "yes" if c.get("in_p1") else "no",
                     c.get("p1_rater", ""), c.get("p1_form", "")])
    write("Table_S7_allocation_record.csv",
          ["study_id", "question_id", "block", "screen_positive",
           "in_counselor_study", "counselor_1_condition", "counselor_2_condition",
           "in_psychologist_set", "assigned_psychologist", "instrument_form"],
          rows, f"Seed {d['seed']}; generated {d['created']}.")
    with (OUT / "Table_S7b_input_hashes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["input_file", "sha256"])
        for k, v in d["inputs"].items():
            w.writerow([k, v])
    written.append(("Table_S7b_input_hashes.csv", len(d["inputs"]),
                    "SHA-256 of the frozen inputs the allocation was drawn from."))


if __name__ == "__main__":
    s1_retrieval_grid(); s1b_holm(); s1c_by_question_type()
    s2_per_briefing()
    s3_psychologist_items(); s4_scope_by_rater()
    s5_counselor_process(); s6_pairs(); s7_allocation()
    print(f"\nwrote {len(written)} tables to {OUT.relative_to(ROOT)}/\n")
    for name, n, note in written:
        print(f"  {name:46s} {n:4d} rows   {note}")
