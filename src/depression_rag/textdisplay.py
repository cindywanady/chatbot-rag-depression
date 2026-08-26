"""Render-time text repairs. No dependencies, on purpose.

Guideline passages are stored with one paragraph per blank line, and the pipeline
is right to keep them that way — the segmenter and the chunkers rely on that
structure. But the guideline's "paragraphs" are individual bullets and their
continuations, so anything that DISPLAYS a passage verbatim shows a wall of
double-spaced fragments with bullets split across them::

    • Depresi pada lansia 1-2 %, ... yang ada di RS atau institusi lain
    <blank>
    sampai dengan 40%

Two very different consumers need the same repair: the chatbot's citation panel
and the psychologist rating packets. Hence a shared home.

**Why this is a top-level module and not part of `extraction/`.** It lived in
`extraction/cleaning.py` for about ten minutes, which broke the chatbot outright:
`extraction/__init__` imports `pdf_loader`, which needs PyMuPDF, which the
chatbot environment (`.venv-chat`) deliberately does not install. Anything the
runtime imports has to be reachable from a minimal environment, so this module
imports nothing beyond `re`. Keep it that way.

Stored text is never rewritten by any of this.
"""

from __future__ import annotations

import re

# A line that opens its own bullet / numbered point / citation — i.e. one that is
# NOT a continuation of the line above it.
_LINE_START = re.compile(
    r"^(?:[•▪◦✓✗]|[-–—]\s|\[\d+\]|\d+[.)]\s)")
_SENTENCE_END = (".", ";", ":", "?", "!")


def reflow_for_display(text: str) -> str:
    """Rejoin lines broken mid-sentence and drop blank ones, for display only.

    A line is glued to the one above only when all three hold: it does not open a
    bullet/number/citation, the previous line did not end on sentence
    punctuation, and it starts lower-case or with a digit. Anything ambiguous is
    left alone — a wrongly split bullet is untidy, but wrongly merging two
    clinical statements changes what they say, and this text is read by
    clinicians and rated for accuracy.
    """
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if (out and not _LINE_START.match(line)
                and not out[-1].endswith(_SENTENCE_END)
                and (line[0].islower() or line[0].isdigit())):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)
