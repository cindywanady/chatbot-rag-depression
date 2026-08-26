"""Domain models for the chunking pipeline.

One offset convention is used everywhere: ``char_start`` / ``char_end`` index
the *canonical cleaned source text* (``SegmentedDocument.cleaned_text``), which
is the concatenation of all retained segments. Chunk offsets and the gold-QA
relevance mapping in the evaluation stage therefore share a single coordinate
system, and the invariant ::

    text == cleaned_text[char_start:char_end]

holds for every :class:`Segment` and every :class:`Chunk`. It is enforced by
the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

#: Separator inserted between retained segments when building ``cleaned_text``.
SEGMENT_SEPARATOR = "\n\n"


class ContentType(str, Enum):
    """Clinical content tag for a retained segment.

    Assigned by a best-effort, keyword-driven heuristic (see
    ``extraction/segmentation.py``) and deliberately *reversible*: it lets an
    experiment include or exclude e.g. case examples without re-running
    extraction. Spot-check the tags before quoting anything derived from them.
    """

    CLINICAL_EXPOSITION = "clinical_exposition"
    CRITERIA = "criteria"
    DOSAGE = "dosage"
    REFERRAL_CRITERIA = "referral_criteria"
    SOMATIC_SYMPTOMS = "somatic_symptoms"
    CASE_EXAMPLE = "case_example"
    RISK_SUICIDE = "risk_suicide"


class SectionRole(str, Enum):
    """Structural role of a segment - decides what gets indexed.

    Only :attr:`CLINICAL` segments are retained by default (keep the clinical
    exposition, drop the training-manual scaffolding). Everything else is
    persisted to an audit file so the filter is explicit and reversible.
    """

    CLINICAL = "clinical"        # Uraian Materi / Pokok Bahasan A-D content
    SCAFFOLDING = "scaffolding"  # tujuan, metode, media, langkah, PB overview, deskripsi
    REFERENCE = "reference"      # Referensi / Daftar Pustaka
    FRONT_MATTER = "front_matter"


#: Content types treated as "tightly-coupled clinical units" that the
#: structure-aware chunker must never split.
ATOMIC_CONTENT_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.CRITERIA,
        ContentType.DOSAGE,
        ContentType.REFERRAL_CRITERIA,
        ContentType.SOMATIC_SYMPTOMS,
    }
)


@dataclass(frozen=True)
class SourceUnit:
    """A Materi Inti (MI) unit and its *physical* (1-based) page range."""

    name: str          # e.g. "MI.4"
    phys_start: int    # 1-based inclusive physical page
    phys_end: int      # 1-based inclusive physical page
    title: str = ""

    def printed_start(self, printed_offset: int) -> int:
        return self.phys_start - printed_offset

    def printed_end(self, printed_offset: int) -> int:
        return self.phys_end - printed_offset


@dataclass(frozen=True)
class PageText:
    """Raw extracted text for one PDF page, before cleaning.

    Carries both numbering systems because the guideline is cited by *printed*
    page numbers (MI.4 = printed 36-57) while PyMuPDF indexes *physical* pages.
    ``blocks`` are paragraph-ish text blocks in reading order (PyMuPDF
    "blocks" mode) which the cleaner reflows. Only ``extraction/pdf_loader.py``
    imports ``fitz``; everything downstream consumes :class:`PageText`, keeping
    the rest of the pipeline backend-agnostic.
    """

    physical_index: int          # 0-based PyMuPDF index
    physical_page: int           # 1-based physical page
    printed_page: Optional[int]  # printed page number, or None if unmapped
    unit: str                    # owning MI unit, e.g. "MI.4"
    blocks: tuple[str, ...]      # raw paragraph blocks, reading order
    n_images: int = 0

    @property
    def text(self) -> str:
        return SEGMENT_SEPARATOR.join(self.blocks)


@dataclass(frozen=True)
class Segment:
    """A retained, cleaned span of clinical text with provenance metadata.

    ``char_start`` / ``char_end`` index ``SegmentedDocument.cleaned_text``.
    """

    segment_id: str
    text: str
    char_start: int
    char_end: int
    source_unit: str
    pokok_bahasan: Optional[str]   # "A".."E" or None
    heading_path: str             # "MI.4 > Pokok Bahasan B > KRITERIA DIAGNOSIS"
    page_start: int               # printed page
    page_end: int                 # printed page
    content_type: ContentType
    section_role: SectionRole
    atomic: bool = False          # tightly-coupled: never split (structure-aware)
    lang: str = "id"

    @property
    def metadata(self) -> dict[str, Any]:
        """Segment metadata in the flat dict shape carried by every chunk."""
        return {
            "source_unit": self.source_unit,
            "pokok_bahasan": self.pokok_bahasan,
            "heading_path": self.heading_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content_type": self.content_type.value,
            "lang": self.lang,
            "segment_id": self.segment_id,
            "atomic": self.atomic,
        }

    @property
    def n_chars(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True)
class Chunk:
    """A unit of text to be embedded and indexed.

    Attributes:
        chunk_id: Stable unique id, e.g. "fixed-256-0_MI4_0007".
        text: The chunk text, cleaned, no embedding prefix added yet.
        char_start: Start offset of this chunk in the cleaned source text.
        char_end: End offset (exclusive) in the cleaned source text.
        metadata: Segment metadata (heading_path, content_type, pages, ...).
        strategy: "fixed" | "recursive" | "structure".
        params: The exact params that produced this chunk (size, overlap, ...).
    """

    chunk_id: str
    text: str
    char_start: int
    char_end: int
    metadata: dict
    strategy: str
    params: dict

    @property
    def n_chars(self) -> int:
        return self.char_end - self.char_start

    def to_record(self) -> dict[str, Any]:
        """Flatten to a single row for JSONL / Parquet output.

        Metadata keys are prefixed ``meta_`` and params ``param_`` so the
        columnar output never collides with the top-level fields.
        """
        rec: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_chars": self.n_chars,
        }
        for key, value in self.metadata.items():
            rec[f"meta_{key}"] = value
        for key, value in self.params.items():
            rec[f"param_{key}"] = value
        return rec


@dataclass(frozen=True)
class ChunkConfig:
    """One chunking configuration in the sweep.

    ``config_id`` keys the persisted chunk files and, later, the indexes. The
    indexing stage crosses these configs with the embedding models to form the
    full ``(chunking x params x model)`` grid; that cross does not live here.
    """

    strategy: str                   # "fixed" | "recursive" | "structure"
    chunk_size: int                 # tokens, per the reference tokenizer
    overlap: int = 0                # tokens
    context_enriched: bool = False  # prepend heading_path at embed time
    reference_tokenizer: str = "intfloat/multilingual-e5-large"

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "recursive", "structure"):
            raise ValueError(f"unknown chunking strategy: {self.strategy!r}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0 <= self.overlap < self.chunk_size):
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    @property
    def config_id(self) -> str:
        cid = f"{self.strategy}-{self.chunk_size}-{self.overlap}"
        if self.context_enriched:
            cid += "-ctx"
        return cid

    @property
    def params(self) -> dict[str, Any]:
        """Exact params recorded on every Chunk this config produces."""
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "context_enriched": self.context_enriched,
            "reference_tokenizer": self.reference_tokenizer,
        }


@dataclass(frozen=True)
class SegmentedDocument:
    """Output of segmentation; input to every chunker.

    ``cleaned_text`` is the canonical coordinate system: the concatenation of
    all RETAINED (clinical) segments joined by :data:`SEGMENT_SEPARATOR`.
    ``segments`` index into it. ``dropped_segments`` (scaffolding / references)
    are retained for audit only and carry no offsets into ``cleaned_text``.
    """

    cleaned_text: str
    segments: tuple[Segment, ...]          # retained, with valid offsets
    dropped_segments: tuple[Segment, ...]  # audit only
    scope: tuple[str, ...]                 # MI units included
    printed_offset: int

    @property
    def n_chars(self) -> int:
        return len(self.cleaned_text)


# ---- embedding + indexing ---------------------------------------------------


class EmbeddingSide(str, Enum):
    """Which side of the retrieval pair text belongs to (drives the prefix)."""

    QUERY = "query"
    PASSAGE = "passage"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Configuration of one embedding model.

    The asymmetric ``query_prefix`` / ``doc_prefix`` are applied at encode time
    (e5 and nomic were trained with them; omitting them silently degrades those
    models). ``max_seq_len`` is the model's truncation point, used to report the
    per-model truncation rate.
    """

    name: str
    checkpoint: str
    backend: str            # "sentence-transformers" | "transformers-mean"
    pooling: str            # "mean"
    dim: int
    query_prefix: str
    doc_prefix: str
    max_seq_len: int
    trust_remote_code: bool = False

    def prefix(self, side: EmbeddingSide | str) -> str:
        """Return the prefix for the given side (query vs passage)."""
        side = EmbeddingSide(side)
        return self.query_prefix if side is EmbeddingSide.QUERY else self.doc_prefix

    def apply_prefix(self, texts: "list[str]", side: EmbeddingSide | str) -> list[str]:
        """Prepend the correct prefix to each text (critical for e5/nomic quality)."""
        pre = self.prefix(side)
        return [pre + t for t in texts] if pre else list(texts)
