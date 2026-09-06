"""Figure 3 — retrieval performance of the deployed index by question type.

Caption: Bars show Hit@5 and nDCG@5 for each of the eight content categories
with the number of questions per category, ordered by nDCG@5.

Source : outputs/analysis/retrieval_metrics_by_type.csv
Index  : structure-512-0__e5-large__mv  (the deployed multi-vector index)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import apply_style, tidy, save, PALETTE, ROOT

# ---------------------------------------------------------------- data
SRC = ROOT / "outputs/analysis/retrieval_metrics_by_type.csv"
INDEX_ID = "structure-512-0__e5-large__mv"
OVERALL_NDCG = 0.795          # deployed index, all 121 questions (Table 2)

LABELS = {
    "special_populations":      "Special populations",
    "referral_criteria":        "Referral criteria",
    "pharmacotherapy_dosing":   "Pharmacotherapy and dosing",
    "diagnostic_criteria":      "Diagnostic criteria",
    "risk_suicide_emergency":   "Suicide risk and emergency",
    "psychoeducation":          "Psychoeducation",
    "symptom_recognition":      "Symptom recognition",
    "differential_comorbidity": "Differential diagnosis and comorbidity",
}
# Content the deployed chatbot never presents as a counselor action. Flagged
# because it inflates the aggregate relative to the intended use case.
WITHHELD = {"pharmacotherapy_dosing"}


def load():
    rows = [r for r in csv.DictReader(SRC.open()) if r["index_id"] == INDEX_ID]
    if not rows:
        raise SystemExit(f"no rows for {INDEX_ID} in {SRC}")
    data = [{
        "key":   r["question_type"],
        "label": LABELS[r["question_type"]],
        "n":     int(r["n"]),
        "hit5":  float(r["recall@5"]),   # column name in the artifact; Hit@5 in the paper
        "ndcg5": float(r["ndcg@5"]),
    } for r in rows]
    return sorted(data, key=lambda d: d["ndcg5"])   # weakest first -> plots at bottom


# ---------------------------------------------------------------- plot
def build(data):
    apply_style()
    plt.rcParams["hatch.linewidth"] = 0.55
    fig, ax = plt.subplots(figsize=(6.7, 3.5))

    y = list(range(len(data)))
    h = 0.34
    gap = 0.19

    ax.barh([i + gap for i in y], [d["ndcg5"] for d in data], height=h,
            color=PALETTE["blue"], edgecolor="white", linewidth=0.4,
            label="nDCG@5 (ranking quality)", zorder=3)
    ax.barh([i - gap for i in y], [d["hit5"] for d in data], height=h,
            facecolor="#FDF1DC", edgecolor=PALETTE["orange"], linewidth=0.8,
            hatch="//", label="Hit@5 (any relevant passage in top 5)", zorder=3)

    for i, d in enumerate(data):
        ax.text(d["ndcg5"] + 0.010, i + gap, f"{d['ndcg5']:.3f}", va="center",
                ha="left", fontsize=7.3, color=PALETTE["blue"], zorder=5)
        ax.text(d["hit5"] + 0.010, i - gap, f"{d['hit5']:.3f}", va="center",
                ha="left", fontsize=7.3, color="#9A6B00", zorder=5)

    ax.axvline(OVERALL_NDCG, color=PALETTE["grey"], linestyle=(0, (4, 3)),
               linewidth=0.9, zorder=2)
    ax.annotate(f"overall nDCG@5 = {OVERALL_NDCG:.3f}",
                xy=(OVERALL_NDCG, len(data) - 0.42), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.3, color=PALETTE["grey"])

    ax.set_yticks(y)
    ax.set_yticklabels([f"{d['label']}  ({d['n']})" for d in data])
    for tick, d in zip(ax.get_yticklabels(), data):
        if d["key"] in WITHHELD:
            tick.set_color(PALETTE["vermil"])

    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Score (0\u20131)")
    ax.tick_params(axis="y", pad=2)
    ax.set_ylim(-0.62, len(data) - 0.18)
    tidy(ax, grid_axis="x")

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              columnspacing=1.6, handlelength=1.6, handleheight=1.1)

    fig.text(0.0, -0.045,
             "Number of benchmark questions per category in parentheses; categories ordered by nDCG@5.\n"
             "Pharmacotherapy and dosing (red) is content the deployed chatbot never presents as a counselor action.",
             fontsize=7.2, color=PALETTE["grey"], ha="left", va="top", linespacing=1.45)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    d = load()
    for p in save(build(d), "figure3_retrieval_by_question_type"):
        print("wrote", p.relative_to(ROOT))
