"""Figure 4 — claim support against RAGAS faithfulness, per briefing.

Caption: Claim support against RAGAS faithfulness for the 48 in-scope
briefings, one point per briefing, with the diagonal marking equality and
dotted lines marking each measure's mean. Points above the diagonal scored
higher on claim support than on RAGAS faithfulness.

Each briefing carries one value per measure, from a single scoring run. The
manuscript reports single-run RAGAS values throughout.

Sources: outputs/analysis/study_eval_chatbot.csv (claim support)
         outputs/analysis/ragas_chatbot.csv        (RAGAS faithfulness)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from figstyle import PALETTE, ROOT, apply_style, save, tidy


JUDGE = ROOT / "outputs/analysis/study_eval_chatbot.csv"
RAGAS = ROOT / "outputs/analysis/ragas_chatbot.csv"
OUT_OF_SCOPE = {"Q03", "Q38"}


def load():
    judge = {row["study_id"]: row for row in csv.DictReader(JUDGE.open())}
    ragas = {row["study_id"]: row for row in csv.DictReader(RAGAS.open())}
    return [
        {
            "id": study_id,
            "claim": float(judge[study_id]["faithfulness"]),
            "ragas": float(ragas[study_id]["faithfulness"]),
        }
        for study_id in sorted(judge)
        if study_id not in OUT_OF_SCOPE
    ]


def build(rows):
    apply_style()
    # Means are computed from the same rows the figure plots, so the reference
    # lines cannot drift from the points.
    mean_claim = sum(row["claim"] for row in rows) / len(rows)
    mean_ragas = sum(row["ragas"] for row in rows) / len(rows)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lo, hi = 0.22, 0.90

    ax.plot(
        [lo, hi], [lo, hi], linestyle=(0, (5, 4)), linewidth=0.9,
        color=PALETTE["grey"], zorder=2,
    )
    ax.text(
        hi - 0.015, hi - 0.015, "equal scores", rotation=45, fontsize=7.2,
        color=PALETTE["grey"], ha="right", va="bottom",
        rotation_mode="anchor",
    )

    ax.axhline(
        mean_claim, color=PALETTE["blue"], linewidth=0.7,
        linestyle=(0, (1, 2.5)), zorder=2,
    )
    ax.axvline(
        mean_ragas, color=PALETTE["orange"], linewidth=0.7,
        linestyle=(0, (1, 2.5)), zorder=2,
    )
    ax.annotate(
        f"mean claim support {mean_claim:.3f}", xy=(lo, mean_claim),
        xytext=(4, -3), textcoords="offset points", fontsize=7.2,
        color=PALETTE["blue"], ha="left", va="top",
    )
    ax.annotate(
        f"mean RAGAS {mean_ragas:.3f}", xy=(mean_ragas, hi),
        xytext=(4, -4), textcoords="offset points", fontsize=7.2,
        color="#9A6B00", ha="left", va="top",
    )

    x = [row["ragas"] for row in rows]
    y = [row["claim"] for row in rows]
    ax.scatter(
        x, y, s=26, facecolor=PALETTE["blue"], edgecolor="white",
        linewidth=0.5, alpha=0.9, zorder=4,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("RAGAS faithfulness")
    ax.set_ylabel("Claim support (custom LLM judge)")
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    tidy(ax, grid_axis="both")

    rho, pvalue = spearmanr(x, y)
    # manuscript notation: rho=-0.09; P=.54 — true minus sign, no leading zero on P
    rho_txt = f"{rho:.2f}".replace("-", "\u2212")
    p_txt = f"{pvalue:.2f}".lstrip("0")
    ax.text(
        0.035, 0.968,
        f"Spearman $\\rho$={rho_txt}; $P$={p_txt}\n"
        f"{len(rows)} in-scope briefings",
        transform=ax.transAxes, fontsize=7.8, va="top", ha="left",
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "white",
              "edgecolor": "#D8D8D8", "linewidth": 0.6},
    )

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=PALETTE["blue"], markeredgecolor="white",
            label=f"Briefing (n={len(rows)})",
        )
    ]
    ax.legend(
        handles=handles, loc="lower right", bbox_to_anchor=(1.015, 0.005),
        labelspacing=0.5,
    )

    note = (
        "Points above the diagonal scored higher on claim support than on RAGAS faithfulness.\n"
        "Claim support evaluates distinct clinical claims and excludes headings and procedural sentences;\n"
        "RAGAS faithfulness evaluates all statements."
    )
    fig.text(
        0.0, -0.02, note, fontsize=7.2, color=PALETTE["grey"],
        ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rows = load()
    assert len(rows) == 48, f"expected 48 in-scope briefings, got {len(rows)}"
    for path in save(build(rows), "figure3_claim_support_vs_ragas"):
        print("wrote", path.relative_to(ROOT))
