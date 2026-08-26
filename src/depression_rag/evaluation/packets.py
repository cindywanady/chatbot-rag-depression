"""What a psychologist reads when rating one chatbot briefing.

Two participant-facing artifacts render the same P1 packet: the standalone
markdown file (`scripts/build_eval_packets.py`) and, since 2026-07-27, a sheet
inside the psychologist workbook (`scripts/build_psychologist_workbooks.py`) so
the workbook is self-contained. If those two ever disagree, two raters are
scoring different material and the P1 comparison is void — so the *content* is
assembled once, here, and the scripts only choose a presentation.

Deliberately structured rather than pre-rendered: markdown wants `**bold**` and
blockquotes, Excel wants cells and fonts. Returning blocks lets each renderer do
its own thing without re-deriving what a packet contains.

What a packet does NOT contain, by design:

  * the internal `risky` flag — BAGIAN 0 is the rater's own judgement
    (evaluation design v2 §3), so printing our classifier's verdict would anchor
    them;
  * any answer key — correctness and completeness are judged against the MoH
    guideline by the psychologist, not against a pre-made answer.

Light-form packets carry the BAGIAN-3 skip notice in print, so the rater is never
the one deciding what to skip.
"""

from __future__ import annotations

from dataclasses import dataclass

from depression_rag.evaluation.deidentify import assert_deidentified, deidentify

FULL_FORM_NOTE = ("Isi baris kasus ini di sheet «Materi Chatbot» pada lembar "
                  "penilaian Anda. Kolom kesetiaan pada konteks dinilai HANYA "
                  "terhadap konteks [1]–[5] di bawah.")
LIGHT_FORM_NOTE = ("Isi baris kasus ini di sheet «Materi Chatbot» pada lembar "
                   "penilaian Anda. BAGIAN 3 (kesetiaan pada konteks): TIDAK "
                   "DINILAI untuk kasus ini — kolomnya sudah dikunci.")

# The packet carries NO note about the safety banner, and neither does anything
# else a psychologist reads.
#
# A five-sentence note shipped here briefly on 2026-08-06 and was removed the
# same day: repeated on all 22 packets, sitting beside the very material it told
# the rater not to score. It cost more attention than it bought, and printing it
# next to the rated text is exactly the confusion it was meant to prevent — the
# safety-gate item asks whether "materi" is AMAN.
#
# It then survived as background in the Petunjuk sheet until 2026-08-17, when the
# banner was dropped from the rater-facing text entirely: a psychologist has no
# reason to know how the deployed tool is assembled, and describing a component
# they never see raises questions the packet cannot answer. What remains, in the
# instructions and in k4/`gerbang` themselves, is the scope rule alone — rate the
# text printed in this packet. The banner rationale is now researcher-only; see
# the comment above `gerbang` in build_psychologist_workbooks.py and the
# reporting obligation in protocol_amendment_20260806.md §1.


# Shared with the chatbot's citation panel, which displays the same passages.
from depression_rag.textdisplay import reflow_for_display as reflow  # noqa: E402,F401


@dataclass(frozen=True)
class ContextBlock:
    """One of the k=5 guideline passages the generator actually received."""

    n: int
    heading: str
    pages: str
    text: str

    @property
    def label(self) -> str:
        return f"[{self.n}] {self.heading} (hal. {self.pages})"


@dataclass(frozen=True)
class PacketContent:
    study_id: str
    question_id: str
    full_form: bool
    form_note: str
    question: str
    briefing: str
    contexts: list[ContextBlock]

    @property
    def title(self) -> str:
        return f"Paket P1 — Kasus {self.study_id}"

    @property
    def sheet_name(self) -> str:
        """Excel caps sheet names at 31 chars; `study_id` is `Q01`-style."""
        return f"Paket {self.study_id}"[:31]

    @property
    def file_name(self) -> str:
        return f"P1_{self.study_id}.md"


def build_packet_content(case: dict, answer: dict) -> PacketContent:
    """Assemble one packet from a frozen case and its generated answer.

    The question goes through `deidentify()` and is then asserted clean — the
    same path every other participant-facing artifact uses, so no rater group
    sees a version another group does not.
    """
    question = deidentify(case["question_id"], answer["question"])
    assert_deidentified(question, f"P1 packet {case['study_id']}")
    full = case["p1_form"] == "full"
    return PacketContent(
        study_id=case["study_id"],
        question_id=case["question_id"],
        full_form=full,
        form_note=FULL_FORM_NOTE if full else LIGHT_FORM_NOTE,
        question=question,
        briefing=answer["generated_answer"].strip(),
        contexts=[ContextBlock(i, b["heading"], str(b["pages"]), b["text"].strip())
                  for i, b in enumerate(answer["retrieved_context"], 1)],
    )


def render_markdown(p: PacketContent) -> str:
    """The standalone printable packet."""
    note = p.form_note
    if not p.full_form:                       # keep the emphasis the print form had
        note = note.replace("BAGIAN 3", "**BAGIAN 3").replace("dikunci.", "dikunci.**")
    ctx = "\n\n".join(f"**{b.label}**\n\n{reflow(b.text)}" for b in p.contexts)
    return (f"# {p.title}\n\n"
            f"*{note}*\n\n"
            f"## Pertanyaan penanya\n\n> {p.question}\n\n"
            f"## Materi chatbot yang dinilai\n\n{reflow(p.briefing)}\n\n"
            f"## Konteks pedoman yang diterima chatbot\n\n{ctx}\n")
