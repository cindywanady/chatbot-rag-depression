"""Figure 5 — psychologist ratings by instrument domain and rater.

Caption: Psychologist ratings by instrument domain and rater, showing both
raters' means on the 1-5 scale.

Source : outputs/analysis/p1_scores.csv  (44 ratings of 36 briefing packets)

The point of the figure is the pattern the Results paragraph states: the two
raters diverge on domains concerning the counselor's scope of action and agree
closely on domains concerning the clinical content of the briefing. Domains are
therefore ordered by the size of the between-rater gap.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import apply_style, tidy, save, PALETTE, ROOT

# ---------------------------------------------------------------- data
SRC = ROOT / "outputs/analysis/p1_scores.csv"

# instrument column -> label used in the manuscript (Table 4)
DOMAINS = [
    ("kebenaran",   "Clinical correctness"),
    ("kelengkapan", "Completeness for the case"),
    ("relevansi",   "Case relevance"),
    ("kejelasan",   "Clarity and practicality"),
    ("K1",          "Harm avoidance in advice"),
    ("K2",          "Risk recognition and handling"),
    ("K3",          "Immediate escalation"),
    ("K4",          "Resources and referral pathway"),
    ("K5",          "Decision space and role boundary"),
]
RATERS = ("psychologist_1", "psychologist_2")
DIVERGENT = 1.0     # gap above which the Results call the raters divergent


def load():
    rows = list(csv.DictReader(SRC.open()))
    out = []
    for col, label in DOMAINS:
        means = {}
        for r in RATERS:
            vals = [float(x[col]) for x in rows
                    if x["rater"] == r and x[col] not in ("", "None")]
            means[r] = statistics.mean(vals)
        out.append({
            "label": label,
            "p1": means["psychologist_1"],
            "p2": means["psychologist_2"],
            "gap": abs(means["psychologist_1"] - means["psychologist_2"]),
        })
    return sorted(out, key=lambda d: d["gap"])   # smallest gap first -> bottom


# ---------------------------------------------------------------- plot
def build(data):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for i, d in enumerate(data):
        wide = d["gap"] >= DIVERGENT
        ax.plot([d["p1"], d["p2"]], [i, i],
                color=PALETTE["vermil"] if wide else PALETTE["lgrey"],
                linewidth=2.2 if wide else 1.6,
                alpha=0.85 if wide else 1.0, solid_capstyle="round", zorder=2)

    # Psychologist 2 drawn larger and beneath, Psychologist 1 smaller and on top,
    # so that a domain where the two means coincide exactly (Case relevance,
    # gap 0.00) still shows both raters rather than hiding one.
    ax.scatter([d["p2"] for d in data], range(len(data)), s=62, marker="s",
               facecolor=PALETTE["orange"], edgecolor="white", linewidth=0.7,
               zorder=4, label="Psychologist 2")
    ax.scatter([d["p1"] for d in data], range(len(data)), s=28, marker="o",
               facecolor=PALETTE["blue"], edgecolor="white", linewidth=0.7,
               zorder=5, label="Psychologist 1")

    for i, d in enumerate(data):
        ax.text(5.16, i, f"{d['gap']:.2f}", va="center", ha="right",
                fontsize=7.5, zorder=4,
                color=PALETTE["vermil"] if d["gap"] >= DIVERGENT else PALETTE["grey"])

    # boundary between the concordant and divergent blocks
    split = next(i for i, d in enumerate(data) if d["gap"] >= DIVERGENT)
    ax.axhline(split - 0.5, color="#C9C9C9", linewidth=0.7,
               linestyle=(0, (3, 3)), zorder=1)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d["label"] for d in data])
    ax.set_xlim(1.0, 5.22)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.7, len(data) - 0.15)
    ax.set_xlabel("Mean rating (1–5, higher is more favourable)")
    tidy(ax, grid_axis="x")

    ax.text(5.16, len(data) - 0.35, "gap", va="center", ha="right",
            fontsize=7.5, style="italic", color=PALETTE["grey"])

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.0,
               markerfacecolor=PALETTE["blue"], markeredgecolor="white",
               label="Psychologist 1"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=7.4,
               markerfacecolor=PALETTE["orange"], markeredgecolor="white",
               label="Psychologist 2"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              ncol=2, columnspacing=1.8)

    fig.text(0.0, -0.045,
             "Domains ordered by the between-rater gap. Above the dashed rule the raters differ by more than 1 point;\n"
             "those domains all concern the counselor's scope of action rather than the clinical content of the briefing.",
             fontsize=7.2, color=PALETTE["grey"], ha="left", va="top", linespacing=1.5)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    d = load()
    for row in sorted(d, key=lambda r: -r["gap"]):
        print(f"  {row['label']:34s} P1={row['p1']:.3f}  P2={row['p2']:.3f}  gap={row['gap']:.3f}")
    for p in save(build(d), "figure5_psychologist_domains_by_rater"):
        print("wrote", p.relative_to(ROOT))
