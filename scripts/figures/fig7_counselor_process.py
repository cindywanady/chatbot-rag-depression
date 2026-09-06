"""Figure 6 — counselor use of the chatbot and process measures (RQ3b).

Caption: Counselor use of the chatbot across the 12 assigned-access cases for
each counselor. Panel A gives the case-level helpfulness rating, panel B the
response length in each condition, and panel C the counselor's own report of
whether the briefing influenced the response written.

Sources : outputs/analysis/analysis_counselor_per_case.csv

Panels A and C cover only the assigned-access cases, because the helpfulness
and influence items were collected in that condition alone. Panel B covers all
48 responses, since word count exists for both conditions.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import apply_style, tidy, save, PALETTE, ROOT

CASES = ROOT / "outputs/analysis/analysis_counselor_per_case.csv"

COUNSELORS = ["counselor_1", "counselor_2"]
LABELS = {"counselor_1": "Counselor 1", "counselor_2": "Counselor 2"}

# The influence item was answered on a 4-point Indonesian scale. Ordered from
# strongest to none, so the stack reads left to right as decreasing influence.
INFLUENCE = [
    ("Sangat memengaruhi", "Strongly"),
    ("Cukup memengaruhi", "Moderately"),
    ("Sedikit memengaruhi", "Slightly"),
    ("Tidak memengaruhi", "Not at all"),
]
# Colour carries the ordinal level and hatch repeats it, so the stack survives
# greyscale printing.
INFLUENCE_STYLE = {
    "Strongly": (PALETTE["blue"], ""),
    "Moderately": ("#7FBBE0", "//"),
    "Slightly": ("#CBE1F0", ".."),
    "Not at all": ("#FFFFFF", "xx"),
}

JITTER = 0.055          # deterministic spread for overlapping identical ratings


def load():
    rows = list(csv.DictReader(CASES.open(encoding="utf-8")))
    assert len(rows) == 48, len(rows)
    helpfulness = defaultdict(list)
    influence = defaultdict(Counter)
    words = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["counselor"]
        words[c][r["condition"]].append(int(r["jumlah_kata"]))
        if r["condition"] != "chatbot":
            continue
        if r["k1_bantu_kasus"]:
            helpfulness[c].append(float(r["k1_bantu_kasus"]))
        influence[c][r["k1_pengaruh"]] += 1
    return helpfulness, influence, words


def spread(values):
    """Symmetric offsets for equal values, so overlapping dots stay countable."""
    order = defaultdict(list)
    for i, v in enumerate(values):
        order[v].append(i)
    offsets = [0.0] * len(values)
    for v, idx in order.items():
        n = len(idx)
        for k, i in enumerate(idx):
            offsets[i] = (k - (n - 1) / 2) * JITTER
    return offsets


def panel_helpfulness(ax, helpfulness):
    ax.set_title("A  Helpfulness for the case")
    for x, c in enumerate(COUNSELORS):
        vals = helpfulness[c]
        for dx, v in zip(spread(vals), vals):
            ax.plot(x + dx, v, marker="o", markersize=4.6, linestyle="none",
                    markerfacecolor=PALETTE["blue"], markeredgecolor="white",
                    markeredgewidth=0.5, alpha=0.9, zorder=3)
        mean = sum(vals) / len(vals)
        ax.plot([x - 0.30, x + 0.30], [mean, mean], color=PALETTE["vermil"],
                linewidth=1.6, solid_capstyle="butt", zorder=4)
        ax.annotate(f"mean {mean:.2f}", xy=(x + 0.32, mean),
                    xytext=(0, 0), textcoords="offset points",
                    fontsize=7.4, color=PALETTE["vermil"], va="center", ha="left")
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(0.6, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xticks(range(len(COUNSELORS)))
    ax.set_xticklabels([LABELS[c] for c in COUNSELORS])
    ax.set_ylabel("Rating (1–5)")
    tidy(ax, grid_axis="y")


def panel_words(ax, words):
    ax.set_title("B  Response length")
    conditions = [("chatbot", "with access", PALETTE["blue"], "o"),
                  ("no_chatbot", "without", PALETTE["orange"], "s")]
    for x, c in enumerate(COUNSELORS):
        for k, (cond, _, colour, marker) in enumerate(conditions):
            vals = words[c][cond]
            base = x + (k - 0.5) * 0.42
            for dx, v in zip(spread(vals), vals):
                ax.plot(base + dx * 0.55, v, marker=marker, markersize=4.0,
                        linestyle="none", markerfacecolor=colour,
                        markeredgecolor="white", markeredgewidth=0.4,
                        alpha=0.85, zorder=3)
            mean = sum(vals) / len(vals)
            ax.plot([base - 0.15, base + 0.15], [mean, mean], color=PALETTE["ink"],
                    linewidth=1.3, solid_capstyle="butt", zorder=4)
            side = -1 if k == 0 else 1
            ax.annotate(f"{mean:.0f}", xy=(base + side * 0.17, mean),
                        xytext=(side * 2, 0), textcoords="offset points",
                        fontsize=7.4, color=PALETTE["ink"],
                        ha="right" if side < 0 else "left", va="center")
    ax.set_xlim(-0.80, 1.78)
    ax.set_xticks(range(len(COUNSELORS)))
    ax.set_xticklabels([LABELS[c] for c in COUNSELORS])
    ax.set_ylabel("Words per response")
    tidy(ax, grid_axis="y")
    ax.legend(handles=[
        Line2D([0], [0], marker=m, linestyle="none", markersize=4.4,
               markerfacecolor=col, markeredgecolor="white", label=lab)
        for _, lab, col, m in conditions
    ], loc="lower left", bbox_to_anchor=(-0.02, -0.02), ncol=1, columnspacing=1.0)


def panel_influence(ax, influence):
    ax.set_title("C  Reported influence on the response")
    for y, c in enumerate(reversed(COUNSELORS)):
        left = 0
        for raw, label in INFLUENCE:
            n = influence[c][raw]
            if not n:
                continue
            colour, hatch = INFLUENCE_STYLE[label]
            ax.barh(y, n, left=left, height=0.52, facecolor=colour, hatch=hatch,
                    edgecolor=PALETTE["ink"], linewidth=0.6, zorder=3)
            ax.annotate(str(n), xy=(left + n / 2, y), ha="center", va="center",
                        fontsize=7.6, color=PALETTE["ink"], zorder=5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.92))
            left += n
    ax.set_yticks(range(len(COUNSELORS)))
    ax.set_yticklabels([LABELS[c] for c in reversed(COUNSELORS)])
    ax.set_ylim(-0.52, 2.05)
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("Assigned-access cases (12 per counselor)")
    tidy(ax, grid_axis="x")
    ax.legend(handles=[
        Patch(facecolor=INFLUENCE_STYLE[lab][0], hatch=INFLUENCE_STYLE[lab][1],
              edgecolor=PALETTE["ink"], linewidth=0.6, label=lab)
        for _, lab in INFLUENCE
    ], loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, columnspacing=1.4,
       handlelength=1.6)


def build(helpfulness, influence, words):
    apply_style()
    fig = plt.figure(figsize=(6.6, 4.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.95], hspace=0.46,
                            wspace=0.30)
    panel_helpfulness(fig.add_subplot(grid[0, 0]), helpfulness)
    panel_words(fig.add_subplot(grid[0, 1]), words)
    panel_influence(fig.add_subplot(grid[1, :]), influence)
    fig.text(0.0, -0.045,
             "Panels A and C cover the 12 assigned-access cases per counselor, the condition in which the "
             "helpfulness and\ninfluence items were collected. Panel B covers all 48 responses. Horizontal "
             "rules mark means.",
             fontsize=7.2, color=PALETTE["grey"], ha="left", va="top",
             linespacing=1.5)
    return fig


if __name__ == "__main__":
    helpfulness, influence, words = load()
    for c in COUNSELORS:
        assert len(helpfulness[c]) == 12, (c, len(helpfulness[c]))
        assert sum(influence[c].values()) == 12, (c, influence[c])
        assert len(words[c]["chatbot"]) == len(words[c]["no_chatbot"]) == 12
    for p in save(build(helpfulness, influence, words), "figure6_counselor_process"):
        print("wrote", p.relative_to(ROOT))
