"""Shared visual style for the manuscript figures.

One place for typography, palette, sizing and save behaviour so every figure in
the paper looks like it came from the same hand. Import this before plotting:

    from figstyle import apply_style, PALETTE, save

Design rules encoded here:
  * Serif typeface, Times-metric, to match the typeset body text. JMIR asks for
    Times New Roman inside figures where practical; Nimbus Roman is the
    metric-compatible clone available on this machine.
  * Colour is never the only channel. Every categorical distinction is carried
    by a marker shape, a line style or a hatch as well, so the figures survive
    greyscale printing (a JMIR requirement).
  * Minimal furniture: no top or right spine, one light grid axis only, no
    frames around legends, no background fill.
  * Direct labelling in preference to legends wherever the plot allows it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "documents/figures/manuscript"

# --- typography -------------------------------------------------------------
SERIF = ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"]
BASE_PT = 8.5          # tick labels, annotations
LABEL_PT = 9.5         # axis labels
TITLE_PT = 9.5         # panel titles, where used

# --- palette ----------------------------------------------------------------
# Okabe-Ito, colour-vision-deficiency safe. Keep assignments stable across
# figures: BLUE is always the primary/headline measure, ORANGE the secondary.
PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "vermil":  "#D55E00",
    "purple":  "#7B52AB",
    "grey":    "#6E6E6E",
    "lgrey":   "#BFBFBF",
    "ink":     "#1A1A1A",
}
GRID = "#DCDCDC"


def apply_style() -> None:
    """Set rcParams. Call once at the top of every figure script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "dejavuserif",
        "font.size": BASE_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "xtick.labelsize": BASE_PT,
        "ytick.labelsize": BASE_PT,
        "legend.fontsize": BASE_PT,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.35,
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.7,
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,   # embed as TrueType so text stays editable
        "ps.fonttype": 42,
    })


def tidy(ax, grid_axis: str = "x") -> None:
    """Strip chart furniture down to one light grid axis."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="-", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem: str, outdir: Path = OUTDIR) -> list[Path]:
    """Write PNG at 300 dpi for submission and PDF (vector) for the thesis."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("png", 300), ("pdf", None)):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, **({"dpi": dpi} if dpi else {}))
        paths.append(p)
    plt.close(fig)
    return paths
