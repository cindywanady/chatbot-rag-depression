#!/usr/bin/env python3
"""Build the methodology figure (Figure 1) as publication-quality SVG.

Standard library only, so it runs in any of the project's virtualenvs without
adding a dependency. Output is vector SVG sized for a full-page journal figure
(180 mm column width); rasterise with ImageMagick if a reviewer wants PNG:

    python scripts/build_methodology_figure.py
    convert -density 600 -background white documents/figures/methodology_figure.svg \
            documents/figures/methodology_figure.png

Every number in the figure is sourced from the frozen artifacts, not retyped
from prose. The mapping is recorded in documents/METHODOLOGY.md, section 7.

Design constraints, deliberately conservative for a clinical-informatics venue:
  * one accent family only (safety path); everything else neutral, so the figure
    survives greyscale printing and the common forms of colour blindness;
  * meaning is carried by position and label, never by hue alone;
  * smallest type is 7 pt at final printed size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

# --- canvas -----------------------------------------------------------------
# 1200 user units == 180 mm, i.e. 6.667 units/mm. Point sizes below are derived
# from that scale: 7 pt = 2.47 mm = 16.4 units.
W = 1200
H = 1576
UNITS_PER_MM = W / 180.0

FS_BODY = 17        # ~7.2 pt
FS_TITLE = 19       # ~8.1 pt
FS_PANEL = 24       # ~10.2 pt
FS_LETTER = 30      # ~12.7 pt
FS_NOTE = 16        # ~6.8 pt
LH = 21             # body line height

FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"

# --- palette ----------------------------------------------------------------
INK = "#14181F"          # primary text
INK_2 = "#454F5B"        # secondary text, arrows
BAND = "#F4F6F8"         # panel band fill
BAND_EDGE = "#C6CDD5"
BOX_FILL = "#FFFFFF"
BOX_EDGE = "#5A6572"
RESULT_FILL = "#E4EBF2"  # selected configuration / analysis
RESULT_EDGE = "#2A557B"
SAFE_FILL = "#F8EAE3"    # safety path
SAFE_EDGE = "#9C4A2A"
OUT_EDGE = "#8A939D"     # outside-the-system boundary

out: list[str] = []


def esc(s: str) -> str:
    return escape(s)


def text(x, y, s, size=FS_BODY, fill=INK, weight="normal", anchor="start", style=""):
    out.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}"'
        + (f' font-style="{style}"' if style else "")
        + f">{esc(s)}</text>"
    )


def box(x, y, w, h, title, lines=(), kind="plain", title_size=FS_TITLE):
    """A process box. kind: plain | result | safety | outside | input"""
    fill, edge, dash, tcol = BOX_FILL, BOX_EDGE, "", INK
    if kind == "result":
        fill, edge = RESULT_FILL, RESULT_EDGE
    elif kind == "safety":
        fill, edge = SAFE_FILL, SAFE_EDGE
    elif kind == "outside":
        fill, edge, dash, tcol = "#FFFFFF", OUT_EDGE, ' stroke-dasharray="7 5"', INK_2
    elif kind == "input":
        fill, edge = "#EDF0F3", BOX_EDGE
    rx = h / 2 if kind == "input" and not lines else 7
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1.6"{dash}/>'
    )
    cx = x + w / 2
    ty = y + 27 if lines else y + h / 2 + 6
    text(cx, ty, title, size=title_size, weight="bold", anchor="middle", fill=tcol)
    for i, ln in enumerate(lines):
        text(cx, ty + 22 + i * LH, ln, size=FS_BODY, anchor="middle", fill=INK_2)


def band(x, y, w, h, letter, title):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{BAND}" stroke="{BAND_EDGE}" stroke-width="1.4"/>'
    )
    text(x + 20, y + 38, letter, size=FS_LETTER, weight="bold", fill=INK)
    text(x + 20 + 34, y + 37, title, size=FS_PANEL, weight="bold", fill=INK)


def path(pts, head=True, color=INK_2, dash=""):
    d = " ".join(("M" if i == 0 else "L") + f" {px} {py}" for i, (px, py) in enumerate(pts))
    marker = ' marker-end="url(#ah)"' if head else ""
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.9" '
        f'stroke-linejoin="round"{dd}{marker}/>'
    )


def label(x, y, s, anchor="middle"):
    text(x, y, s, size=FS_NOTE, fill=INK_2, anchor=anchor, style="italic")


# ---------------------------------------------------------------------------
def build() -> str:
    out.clear()
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" '
        f'height="{H / UNITS_PER_MM:.1f}mm" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}">'
    )
    out.append(
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK_2}"/></marker></defs>'
    )
    out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ===== PANEL A — retriever selection ===================================
    band(24, 24, 1152, 424, "A", "Retriever selection (RQ1)")

    r1y, r1h = 96, 126
    xs = [44, 330, 616, 902]
    bw = 252
    box(xs[0], r1y, bw, r1h, "Guideline corpus", [
        "Indonesian primary-care", "mental-health guideline",
        "(Ministry of Health, 2017)", "5 of 9 units → 68 segments"])
    box(xs[1], r1y, bw, r1h, "Chunking", [
        "fixed · recursive · structure-", "aware, 128/256/512 tokens,",
        "overlap 0/64 → 14 configura-", "tions, one reference tokenizer"])
    box(xs[2], r1y, bw, r1h, "Embedding", [
        "multilingual-E5-large,", "Nomic-Indonesian, MiniLM,",
        "IndoBERT (weak baseline)", "→ 4 models"])
    box(xs[3], r1y, bw, r1h, "Index", [
        "56 exact-cosine FAISS", "indexes (14 × 4); exact",
        "search removes approxi-", "mation variance"])
    for i in range(3):
        path([(xs[i] + bw, r1y + r1h / 2), (xs[i + 1], r1y + r1h / 2)])

    r2y, r2h = 268, 140
    box(44, r2y, 520, r2h, "Gold standard", [
        "31 curated gold passages over 12 clinical concepts · 121 Indonesian",
        "questions generated from the source text, never from the chunkings",
        "under test · 8 clinical types, 3 difficulty levels · relevance frozen",
        "before scoring: span overlap ≥ 50% of the shorter span"])
    box(598, r2y, 270, r2h, "Scoring", [
        "nDCG@5 (primary,", "frozen a priori)",
        "10,000-resample", "bootstrap CIs · paired",
        "Wilcoxon, Holm-corrected"])
    box(902, r2y, 252, r2h, "Selected", [
        "structure-aware 512-token", "chunks + multilingual-E5",
        "nDCG@5 = 0.798", "(95% CI 0.747–0.846)",
        "suicide-risk recall@5 = 1.000"], kind="result")
    path([(1028, r1y + r1h), (1028, 244), (733, 244), (733, r2y)])
    path([(564, r2y + r2h / 2), (598, r2y + r2h / 2)])
    path([(868, r2y + r2h / 2), (902, r2y + r2h / 2)])
    label(1154, 433, "tie-breakers pre-stated before scoring: "
          "robustness across question types → index size → build cost", anchor="end")

    # ===== PANEL B — deployed system =======================================
    band(24, 472, 1152, 384, "B", "Deployed system")

    b1y, b1h = 544, 126
    box(44, b1y, 252, b1h, "Counselor's question", [
        "a help-seeker's message,", "put into the counselor's",
        "own words"], kind="input")
    box(330, b1y, 252, b1h, "Two-stage risk screen", [
        "58-term keyword pre-screen;", "on a miss, an LLM classifier",
        "issues a second verdict"])
    box(616, b1y, 252, b1h, "Retrieval, k = 5", [
        "selected index, multi-vector", "build (90 chunks, 96 vectors)"])
    box(902, b1y, 252, b1h, "Generation", [
        "Gemma-4-12B-it via vLLM", "temperature 0.2, top-p 0.9",
        "≤ 1200 new tokens"])
    for x0 in (296, 582, 868):
        path([(x0, b1y + b1h / 2), (x0 + 34, b1y + b1h / 2)])

    b2y, b2h = 714, 118
    box(44, b2y, 520, b2h, "If flagged: fixed escalation banner", [
        "emergency department or community health centre first, then the",
        "national crisis line · not model-generated, identical on every flagged",
        "case, and placed ahead of the briefing"], kind="safety")
    box(616, b2y, 252, b2h, "Counselor briefing", [
        "four sections with", "bracketed citations to the",
        "five retrieved passages"], kind="result")
    box(902, b2y, 252, b2h, "Patient-facing reply", [
        "written by the counselor;", "the system never drafts it"], kind="outside")
    path([(456, b1y + b1h), (456, 692), (304, 692), (304, b2y)])
    path([(1028, b1y + b1h), (1028, 692), (742, 692), (742, b2y)])
    path([(564, b2y + b2h / 2), (616, b2y + b2h / 2)])
    path([(868, b2y + b2h / 2), (902, b2y + b2h / 2)], dash="7 5")
    label(410, 686, "risk", anchor="end")

    # ===== PANEL C — evaluation ============================================
    band(24, 880, 1152, 606, "C", "Three-layer evaluation")

    c0y, c0h = 952, 100
    box(44, c0y, 1110, c0h, "Real-world question set", [
        "373 public depression threads from an Indonesian online health forum → frozen random sample of 50",
        "24 risk-flagged / 26 non-risk · two length-balanced blocks of 25 · de-identified, questions kept verbatim"])

    cly, clh = 1122, 214
    cxs = [44, 426, 808]
    cw = 348
    box(cxs[0], cly, cw, clh, "Layer 1 — automatic", [
        "all 50 briefings, no human",
        "",
        "custom judge (Qwen3, a different",
        "model family from the generator)",
        "+ RAGAS, sharing no code",
        "",
        "faithfulness · structural complete-",
        "ness · citation presence · risk surfacing"])
    box(cxs[1], cly, cw, clh, "Layer 2 — expert rating", [
        "36 briefings, 2 licensed psychologists",
        "",
        "the 24 counselor-study cases ∪ all 24",
        "risk cases, overlapping in 12",
        "",
        "clinical quality · safety items and a",
        "binary gate · groundedness against the",
        "exact five passages the model received"])
    box(cxs[2], cly, cw, clh, "Layer 3 — counselor study", [
        "2 counselors × 24 cases, answered",
        "with and without the tool in a counter-",
        "balanced crossover → 24 matched pairs",
        "(48 answers)",
        "",
        "rated blind by the same 2 psychologists;",
        "condition, author and order withheld"])
    path([(599, c0y + c0h), (599, 1092)], head=False)
    path([(219, 1092), (982, 1092)], head=False)
    for cx in (219, 599, 982):
        path([(cx, 1092), (cx, cly)])

    c4y, c4h = 1382, 84
    box(44, c4y, 1110, c4h, "Analysis", [
        "primary endpoint: score ~ condition + block + counselor + rater + (1 | question) over the 24 pairs · Holm-corrected secondary family",
        "ICC(2,1) and weighted κ on 16 double-rated items · safety gate reported as a failure rate, never averaged into a scale"], kind="result")
    for cx in (219, 599, 982):
        path([(cx, cly + clh), (cx, 1358)], head=False)
    path([(219, 1358), (982, 1358)], head=False)
    path([(599, 1358), (599, c4y)])

    # ===== footnote ========================================================
    out.append(
        f'<rect x="24" y="1502" width="1152" height="50" rx="7" fill="#FFFFFF" '
        f'stroke="{BAND_EDGE}" stroke-width="1.4"/>'
    )
    text(44, 1524, "Frozen before data collection.", size=FS_NOTE, weight="bold", fill=INK)
    text(296, 1524,
         "The case sample, the assignment of cases to conditions, the split of rating work between the two",
         size=FS_NOTE, fill=INK_2)
    text(44, 1543,
         "psychologists, the form versions and the blind response codes were all drawn under one seed (20260628) and written to a checksummed assignment file.",
         size=FS_NOTE, fill=INK_2)

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="documents/figures/methodology_figure.svg")
    a = ap.parse_args()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(), encoding="utf-8")
    print(f"[figure] wrote {p}  ({W}×{H} units = 180.0×{H / UNITS_PER_MM:.1f} mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
