"""Segmentation: group cleaned blocks into labelled segments.

The guideline follows one template per Materi Inti (MI) unit::

    MATERI INTI x / <TITLE>
    I. DESKRIPSI ... TUJUAN ... POKOK BAHASAN (overview) ... METODE ... LANGKAH   <- scaffolding
    URAIAN MATERI            <- clinical body STARTS
       Pokok Bahasan A..D  + caps/roman subsections (criteria, dosing, ...)
    REFERENSI                <- references (dropped)

So the clinical body is exactly the span ``URAIAN MATERI`` -> ``REFERENSI``
(or the next unit). This single anchor cleanly separates clinical content from
teaching scaffolding and resolves the fact that "Pokok Bahasan A" appears twice
(once in the overview list, once as the real content header).

Within the clinical body we start a new segment at a Pokok Bahasan header or a
caps / roman subsection title, but NOT at plain numbered or bulleted list items
- so tightly-coupled lists (the diagnostic-criteria block, dose lists) stay
intact. ``content_type`` is then assigned by keyword rules and adjacent atomic
blocks are merged, giving the structure-aware chunker whole clinical units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from depression_rag.config.loader import (
    CleaningConfig,
    SegmentationConfig,
    SourceConfig,
)
from depression_rag.domain import (
    ATOMIC_CONTENT_TYPES,
    SEGMENT_SEPARATOR,
    ContentType,
    PageText,
    SectionRole,
    Segment,
    SegmentedDocument,
)
from depression_rag.observability import get_logger
from depression_rag.ports.interfaces import TextCleaner

log = get_logger("segmentation")

_PB_RE = re.compile(r"(?i)^\s*pokok\s+bahasan\s+([A-E])\b\s*[:.\-]?\s*(.*)$")
_ROMAN_HEAD = re.compile(r"^[IVX]{1,3}\.\s+[A-Za-z]")
_MG_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*mg\b", re.IGNORECASE)


@dataclass
class _RawSeg:
    """Mutable working segment before offsets/ids are assigned."""

    text: str
    unit: str
    pokok_bahasan: str | None
    heading_path: str
    page_start: int
    page_end: int
    section_role: SectionRole
    content_type: ContentType = ContentType.CLINICAL_EXPOSITION
    atomic: bool = False
    from_figure: bool = False


def _collapse(line: str) -> str:
    return " ".join(line.split())


def _is_heading(block: str) -> bool:
    """A short, all-caps or roman-numbered-titlecase line = a subsection title.

    Deliberately conservative: plain numbered/bulleted list items (lowercase
    bodies) are NOT headings, so clinical lists are never split here.
    """
    line = _collapse(block)
    if _ROMAN_HEAD.match(line) and len(line) <= 60:
        return True
    if not (3 <= len(line) <= 80):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.75


class GuidelineSegmenter:
    """Groups this guideline's cleaned blocks into labelled :class:`Segment` objects
    and builds the canonical ``cleaned_text`` they index into."""

    def __init__(
        self,
        source: SourceConfig,
        cleaning: CleaningConfig,
        seg: SegmentationConfig,
        cleaner: TextCleaner,
    ) -> None:
        self.source = source
        self.cleaning = cleaning
        self.seg = seg
        self.cleaner = cleaner

    # -- public API ---------------------------------------------------------
    def segment(self, pages: Sequence[PageText]) -> SegmentedDocument:
        by_unit: dict[str, list[PageText]] = {}
        for page in pages:
            by_unit.setdefault(page.unit, []).append(page)

        raw: list[_RawSeg] = []
        for unit_name, upages in by_unit.items():
            raw.extend(self._segment_unit(unit_name, upages))

        raw = self._inject_figures(raw)

        for i, s in enumerate(raw):
            if not s.from_figure:
                s.content_type = self._classify(s)
            s.atomic = (
                s.section_role == SectionRole.CLINICAL
                and s.content_type in ATOMIC_CONTENT_TYPES
                and len(s.text) <= self.seg.max_atomic_chars
            )
            raw[i] = s

        retain = {SectionRole(r) for r in self.seg.retain_roles}
        retained = [s for s in raw if s.section_role in retain]
        dropped = [s for s in raw if s.section_role not in retain]

        if self.seg.merge_adjacent_atomic:
            retained = self._merge_adjacent_atomic(retained, self.seg.max_atomic_chars)
        retained = self._merge_tiny(retained, self.seg.min_segment_chars)

        cleaned_text, segments = self._finalize(retained)
        segments = self._apply_overrides(segments)
        dropped_segments = self._finalize_dropped(dropped)

        self._log_summary(segments, dropped_segments, cleaned_text)
        return SegmentedDocument(
            cleaned_text=cleaned_text,
            segments=tuple(segments),
            dropped_segments=tuple(dropped_segments),
            scope=tuple(self.source.scope),
            printed_offset=self.source.printed_offset,
        )

    # -- per-unit -----------------------------------------------------------
    def _segment_unit(self, unit_name: str, upages: list[PageText]) -> list[_RawSeg]:
        blocks_pp: list[tuple[str, int]] = []
        for page in upages:
            printed = page.printed_page if page.printed_page is not None else -1
            for block in self.cleaner.clean_blocks(page):
                blocks_pp.append((block, printed))

        clinical_start = self._find_clinical_start(blocks_pp)
        if clinical_start is None:
            log.warning(
                "segmentation.no_uraian_materi",
                extra={"unit": unit_name, "fallback": "treat whole unit as clinical"},
            )
            scaffolding, clinical, references = [], blocks_pp, []
        else:
            ref_start = self._find_reference_start(blocks_pp, clinical_start + 1)
            scaffolding = blocks_pp[:clinical_start]
            if ref_start is None:
                clinical = blocks_pp[clinical_start + 1 :]
                references = []
            else:
                clinical = blocks_pp[clinical_start + 1 : ref_start]
                references = blocks_pp[ref_start:]

        out: list[_RawSeg] = []
        if scaffolding:
            out.append(self._region_segment(unit_name, scaffolding, SectionRole.SCAFFOLDING, "(scaffolding)"))
        out.extend(self._walk_clinical(unit_name, clinical))
        if references:
            out.append(self._region_segment(unit_name, references, SectionRole.REFERENCE, "Referensi"))
        return out

    def _find_clinical_start(self, blocks_pp: list[tuple[str, int]]) -> int | None:
        for i, (block, _) in enumerate(blocks_pp):
            low = block.lower()
            if any(m in low for m in self.seg.clinical_start_markers):
                return i
        return None

    def _find_reference_start(self, blocks_pp: list[tuple[str, int]], begin: int) -> int | None:
        for i in range(begin, len(blocks_pp)):
            low = blocks_pp[i][0].lower().lstrip()
            if any(low.startswith(m) for m in self.seg.reference_markers):
                return i
        return None

    @staticmethod
    def _region_segment(unit: str, blocks_pp: list[tuple[str, int]], role: SectionRole, leaf: str) -> _RawSeg:
        text = SEGMENT_SEPARATOR.join(b for b, _ in blocks_pp).strip()
        pages = [p for _, p in blocks_pp if p >= 0] or [-1]
        return _RawSeg(
            text=text,
            unit=unit,
            pokok_bahasan=None,
            heading_path=f"{unit} > {leaf}",
            page_start=min(pages),
            page_end=max(pages),
            section_role=role,
        )

    def _marker_hit(self, block: str) -> str | None:
        """Return the curated subsection marker this block starts with, if any."""
        s = _collapse(block).lower().lstrip("✓•◦▪-–—* ").strip()
        for m in self.seg.subsection_markers:
            if s.startswith(m):
                return m
        return None

    def _walk_clinical(self, unit: str, blocks_pp: list[tuple[str, int]]) -> list[_RawSeg]:
        segs: list[_RawSeg] = []
        cur_pb: str | None = None
        cur_pb_title: str | None = None
        cur_sub: str | None = None
        pending_pb_title = False
        buf: list[str] = []
        buf_pages: list[int] = []
        buf_has_body = False  # whether real (non-heading) text is in the buffer

        def flush() -> None:
            nonlocal buf, buf_pages, buf_has_body
            text = SEGMENT_SEPARATOR.join(buf).strip()
            if text:
                nodes = [unit]
                if cur_pb:
                    node = f"Pokok Bahasan {cur_pb}"
                    if cur_pb_title:
                        node += f": {cur_pb_title}"
                    nodes.append(node)
                if cur_sub:
                    nodes.append(cur_sub)
                pages = [p for p in buf_pages if p >= 0] or [-1]
                segs.append(
                    _RawSeg(
                        text=text,
                        unit=unit,
                        pokok_bahasan=cur_pb,
                        heading_path=" > ".join(nodes),
                        page_start=min(pages),
                        page_end=max(pages),
                        section_role=SectionRole.CLINICAL,
                    )
                )
            buf, buf_pages, buf_has_body = [], [], False

        def add(block: str, page: int, *, body: bool) -> None:
            nonlocal buf_has_body
            buf.append(block)
            buf_pages.append(page)
            if body:
                buf_has_body = True

        for block, page in blocks_pp:
            pb_m = _PB_RE.match(block)
            if pb_m:
                flush()
                cur_pb = pb_m.group(1).upper()
                inline = _collapse(pb_m.group(2))
                cur_pb_title = inline or None
                cur_sub = None
                pending_pb_title = cur_pb_title is None
                add(block, page, body=False)  # keep heading text; don't drop content
                continue

            if self._marker_hit(block) is not None:
                flush()
                cur_sub = _collapse(block)[:80]
                add(block, page, body=True)  # marker line is content (a list lead-in)
                continue

            if _is_heading(block):
                if pending_pb_title and not buf_has_body:
                    cur_pb_title = (
                        f"{cur_pb_title} {_collapse(block)}".strip()
                        if cur_pb_title
                        else _collapse(block)
                    )
                    pending_pb_title = False
                    add(block, page, body=False)
                    continue
                if buf and not buf_has_body:
                    # consecutive heading lines = one title wrapped across blocks
                    cur_sub = (
                        f"{cur_sub} {_collapse(block)}".strip()
                        if cur_sub
                        else _collapse(block)
                    )
                    add(block, page, body=False)
                    continue
                flush()
                cur_sub = _collapse(block)
                add(block, page, body=False)
                continue

            pending_pb_title = False
            add(block, page, body=True)
        flush()
        return segs

    # -- transcribed figures -------------------------------------------------
    def _inject_figures(self, raw: list[_RawSeg]) -> list[_RawSeg]:
        offset = self.source.printed_offset
        for fig in self.source.figure_transcriptions:
            if not fig.text.strip():
                log.warning(
                    "figure.transcription_pending",
                    extra={
                        "unit": fig.unit,
                        "physical_page": fig.physical_page,
                        "printed_page": fig.physical_page - offset,
                        "heading": fig.heading,
                        "action": "fill source.figure_transcriptions[].text to inject the figure",
                    },
                )
                continue
            printed = fig.physical_page - offset
            nodes = [fig.unit]
            if fig.pokok_bahasan:
                nodes.append(f"Pokok Bahasan {fig.pokok_bahasan}")
            if fig.heading:
                nodes.append(fig.heading)
            seg = _RawSeg(
                text=fig.text.strip(),
                unit=fig.unit,
                pokok_bahasan=fig.pokok_bahasan,
                heading_path=" > ".join(nodes),
                page_start=printed,
                page_end=printed,
                section_role=SectionRole.CLINICAL,
                content_type=ContentType(fig.content_type),
                from_figure=True,
            )
            insert_at = len(raw)
            for i, s in enumerate(raw):
                if s.unit == fig.unit and s.section_role == SectionRole.CLINICAL:
                    insert_at = i + 1
            raw.insert(insert_at, seg)
            log.info("figure.injected", extra={"unit": fig.unit, "printed_page": printed})
        return raw

    # -- manual/judge overrides ---------------------------------------------
    def _apply_overrides(self, segments):
        """Apply configured content_type overrides to final segments by id.

        Overrides win over the rule cascade and `atomic` is recomputed so the
        structure chunker stays consistent. Applied to FINAL (post-merge)
        segments; it relabels them but does not re-run adjacent-atomic merging.
        """
        ov = self.seg.content_type_overrides
        if not ov:
            return segments
        out = []
        for s in segments:
            new = ov.get(s.segment_id)
            if new and new != s.content_type.value:
                ct = ContentType(new)
                atomic = (
                    s.section_role == SectionRole.CLINICAL
                    and ct in ATOMIC_CONTENT_TYPES
                    and len(s.text) <= self.seg.max_atomic_chars
                )
                log.info("content_type.override", extra={
                    "segment_id": s.segment_id, "from": s.content_type.value,
                    "to": new, "atomic": atomic})
                s = replace(s, content_type=ct, atomic=atomic)
            out.append(s)
        return out

    # -- content typing -----------------------------------------------------
    def _classify(self, seg: _RawSeg) -> ContentType:
        text_l = seg.text.lower()
        head_l = seg.heading_path.lower()
        # 1) case vignette: explicit marker in heading or at text start
        if any(m in head_l for m in self.seg.case_markers) or any(
            text_l.startswith(m) for m in self.seg.case_markers
        ):
            return ContentType.CASE_EXAMPLE
        # 2) suicide risk: require the section HEADING to be about suicide. A body
        #    mention alone (even several) does not relabel a segment whose topic is
        #    something else (epidemiology, prescribing, referral, emergency triage).
        #    A body-mention-count rule was tried first and retired - it over-fired
        #    on incidental mentions, and a blind two-judge label audit confirmed
        #    the heading-only rule.
        if self.seg.suicide_terms:
            if any(t in head_l for t in self.seg.suicide_terms):
                return ContentType.RISK_SUICIDE
        # 3) remaining types via configured keyword rules + priority
        matched: set[str] = set()
        for rule in self.seg.content_type_rules:
            if rule.matches(text_l) or rule.matches(head_l):
                matched.add(rule.content_type)
        if _MG_RE.search(seg.text):
            matched.add(ContentType.DOSAGE.value)
        for ct in self.seg.content_type_priority:
            if ct in matched:
                return ContentType(ct)
        return ContentType.CLINICAL_EXPOSITION

    @staticmethod
    def _merge_adjacent_atomic(retained: list[_RawSeg], max_chars: int) -> list[_RawSeg]:
        out: list[_RawSeg] = []
        for s in retained:
            prev = out[-1] if out else None
            if (
                prev is not None
                and s.atomic
                and prev.atomic
                and s.content_type == prev.content_type
                and s.unit == prev.unit
                # only merge if the union stays compact (else keep both atomic
                # but separate, so we never create a giant unsplittable chunk)
                and len(prev.text) + len(SEGMENT_SEPARATOR) + len(s.text) <= max_chars
            ):
                out[-1] = replace(
                    prev,
                    text=prev.text + SEGMENT_SEPARATOR + s.text,
                    page_end=max(prev.page_end, s.page_end),
                )
            else:
                out.append(s)
        return out

    @staticmethod
    def _merge_tiny(retained: list[_RawSeg], min_chars: int) -> list[_RawSeg]:
        """Fold stray sub-min-length fragments into a neighbour (no text lost)."""
        out: list[_RawSeg] = []
        for s in retained:
            if out and len(s.text.strip()) < min_chars:
                prev = out[-1]
                out[-1] = replace(
                    prev,
                    text=prev.text + SEGMENT_SEPARATOR + s.text,
                    page_end=max(prev.page_end, s.page_end),
                )
            else:
                out.append(s)
        if len(out) >= 2 and len(out[0].text.strip()) < min_chars:
            nxt = out[1]
            out[1] = replace(
                nxt,
                text=out[0].text + SEGMENT_SEPARATOR + nxt.text,
                page_start=min(out[0].page_start, nxt.page_start),
            )
            out = out[1:]
        return out

    # -- finalization -------------------------------------------------------
    def _finalize(self, retained: list[_RawSeg]) -> tuple[str, list[Segment]]:
        cleaned_text = SEGMENT_SEPARATOR.join(s.text for s in retained)
        segments: list[Segment] = []
        cursor = 0
        for i, s in enumerate(retained):
            start = cursor
            end = start + len(s.text)
            if cleaned_text[start:end] != s.text:  # invariant guard
                raise AssertionError(f"offset mismatch for retained segment {i}")
            segments.append(
                Segment(
                    segment_id=f"{s.unit.replace('.', '')}_{i:04d}",
                    text=s.text,
                    char_start=start,
                    char_end=end,
                    source_unit=s.unit,
                    pokok_bahasan=s.pokok_bahasan,
                    heading_path=s.heading_path,
                    page_start=s.page_start,
                    page_end=s.page_end,
                    content_type=s.content_type,
                    section_role=s.section_role,
                    atomic=s.atomic,
                )
            )
            cursor = end + len(SEGMENT_SEPARATOR)
        return cleaned_text, segments

    @staticmethod
    def _finalize_dropped(dropped: list[_RawSeg]) -> list[Segment]:
        out: list[Segment] = []
        for i, s in enumerate(dropped):
            out.append(
                Segment(
                    segment_id=f"{s.unit.replace('.', '')}_drop_{i:04d}",
                    text=s.text,
                    char_start=-1,  # dropped: no offset into cleaned_text
                    char_end=-1,
                    source_unit=s.unit,
                    pokok_bahasan=s.pokok_bahasan,
                    heading_path=s.heading_path,
                    page_start=s.page_start,
                    page_end=s.page_end,
                    content_type=s.content_type,
                    section_role=s.section_role,
                    atomic=False,
                )
            )
        return out

    def _log_summary(
        self, segments: list[Segment], dropped: list[Segment], cleaned_text: str
    ) -> None:
        by_type: dict[str, int] = {}
        by_unit: dict[str, int] = {}
        for s in segments:
            by_type[s.content_type.value] = by_type.get(s.content_type.value, 0) + 1
            by_unit[s.source_unit] = by_unit.get(s.source_unit, 0) + 1
        log.info(
            "segmentation.done",
            extra={
                "retained_segments": len(segments),
                "dropped_segments": len(dropped),
                "cleaned_chars": len(cleaned_text),
                "atomic_segments": sum(1 for s in segments if s.atomic),
                "by_content_type": by_type,
                "by_unit": by_unit,
            },
        )
