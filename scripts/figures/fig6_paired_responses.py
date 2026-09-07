"""Figure 6 — paired counselor-response global ratings.

Caption: Paired counselor-response global ratings for the 24 matched questions.
Each line joins the response written with assigned access to the response
written without for the same question, with line style indicating which
counselor wrote the with-access response.

Sources : outputs/analysis/p2_scores.csv                                (ratings)
          outputs/evaluation/study_kit/03_ADMIN_PENELITI/SEALED_p2_key.csv
                                        (blinded response id -> condition, counselor)
          data/derived/eval_case_assignments.json                       (question ids)

Ratings for the 8 responses rated by both psychologists are averaged before the
pair is formed, matching the analysis in the manuscript.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import apply_style, tidy, save, PALETTE, ROOT

# ---------------------------------------------------------------- data
P2 = ROOT / "outputs/analysis/p2_scores.csv"
KEY = ROOT / "outputs/evaluation/study_kit/03_ADMIN_PENELITI/SEALED_p2_key.csv"
CASES = ROOT / "data/derived/eval_case_assignments.json"

# small deterministic horizontal offset so that pairs sharing identical ratings
# remain distinguishable. Vertical positions are the exact ratings; only the
# x positions are nudged.
JITTER = 0.045


def load():
    key = {r["response_id"]: r for r in csv.DictReader(KEY.open())}
    cases = {c["question_id"]: c for c in json.load(CASES.open())["cases"]}

    scores = defaultdict(list)
    for r in csv.DictReader(P2.open()):
        if r["global"]:
            scores[r["response_id"]].append(float(r["global"]))

    pairs = defaultdict(dict)
    for rid, vals in scores.items():
        k = key[rid]
        pairs[cases[k["question_id"]]["study_id"]][k["condition"]] = {
            "score": statistics.mean(vals),
            "counselor": k["counselor"],
            "n_raters": len(vals),
        }

    out = []
    for qid, v in sorted(pairs.items()):
        if len(v) != 2:
            continue
        out.append({
            "qid": qid,
            "with": v["chatbot"]["score"],
            "without": v["no_chatbot"]["score"],
            "writer": v["chatbot"]["counselor"],   # who wrote the with-access response
        })
    return out


# ---------------------------------------------------------------- plot
STYLE = {
    "counselor_1": dict(linestyle="-",           label="Counselor 1"),
    "counselor_2": dict(linestyle=(0, (4, 2.5)), label="Counselor 2"),
}


def build(pairs):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    up = sum(1 for p in pairs if p["with"] > p["without"])
    down = sum(1 for p in pairs if p["with"] < p["without"])
    tie = sum(1 for p in pairs if p["with"] == p["without"])

    for i, p in enumerate(sorted(pairs, key=lambda d: d["qid"])):
        d = p["with"] - p["without"]
        colour = (PALETTE["green"] if d > 0 else
                  PALETTE["vermil"] if d < 0 else PALETTE["lgrey"])
        off = ((i % 5) - 2) * JITTER
        ax.plot([0 + off, 1 + off], [p["with"], p["without"]],
                color=colour, linewidth=1.15, alpha=0.85, zorder=3,
                **{k: v for k, v in STYLE[p["writer"]].items() if k != "label"})
        ax.plot([0 + off, 1 + off], [p["with"], p["without"]],
                linestyle="none", marker="o", markersize=2.8, color=colour,
                alpha=0.9, zorder=4)

    m_with = statistics.mean(p["with"] for p in pairs)
    m_without = statistics.mean(p["without"] for p in pairs)
    ax.plot([0, 1], [m_with, m_without], color=PALETTE["ink"], linewidth=2.4,
            marker="D", markersize=6, markerfacecolor="white",
            markeredgewidth=1.6, zorder=6)
    ax.annotate(f"mean {m_with:.2f}", xy=(0, m_with), xytext=(-9, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=8.2, fontweight="bold")
    ax.annotate(f"mean {m_without:.2f}", xy=(1, m_without), xytext=(9, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=8.2, fontweight="bold")

    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Assigned access", "No assigned access"])
    # full outcome scale: a truncated axis makes the within-pair differences
    # look larger than they are (observed ratings run 3 to 10)
    ax.set_ylim(-0.4, 10.4)
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel("Psychologist global rating of the response (0–10)")
    tidy(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0, pad=6)

    handles = [
        Line2D([0], [0], color=PALETTE["ink"], linewidth=1.15, linestyle="-",
               label="Assigned-access response by Counselor 1"),
        Line2D([0], [0], color=PALETTE["ink"], linewidth=1.15,
               linestyle=(0, (4, 2.5)), label="Assigned-access response by Counselor 2"),
        Line2D([0], [0], color=PALETTE["green"], linewidth=2.0,
               label=f"Higher for assigned-access response ({up})"),
        Line2D([0], [0], color=PALETTE["vermil"], linewidth=2.0,
               label=f"Higher for response without assigned access ({down})"),
        Line2D([0], [0], color=PALETTE["lgrey"], linewidth=2.0,
               label=f"Tied ({tie})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, columnspacing=1.6, labelspacing=0.42, handlelength=2.2)

    ax.text(0.0, -0.33,
            "Each thin line is one of the 24 matched questions; the bold line joins the condition means.\n"
            "Lines are offset horizontally so pairs with identical ratings stay visible; ratings themselves are exact.",
            transform=ax.transAxes, fontsize=7.2, color=PALETTE["grey"],
            ha="left", va="top", linespacing=1.5)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    pairs = load()
    assert len(pairs) == 24, f"expected 24 matched pairs, got {len(pairs)}"
    up = sum(1 for p in pairs if p["with"] > p["without"])
    dn = sum(1 for p in pairs if p["with"] < p["without"])
    print(f"  pairs={len(pairs)}  higher with={up}  higher without={dn}  tied={len(pairs)-up-dn}")
    print(f"  mean with={statistics.mean(p['with'] for p in pairs):.2f}  "
          f"without={statistics.mean(p['without'] for p in pairs):.2f}")
    for p in save(build(pairs), "figure5_paired_response_ratings"):
        print("wrote", p.relative_to(ROOT))
