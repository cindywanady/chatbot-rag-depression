"""Load and validate ``configs/pipeline.yaml`` into typed dataclasses.

The loader is the only place that knows the YAML shape; everything downstream
consumes typed config objects. It also expands the compact ``chunking.sweep``
into an explicit, de-duplicated list of :class:`ChunkConfig` objects.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from depression_rag.domain import ChunkConfig, SourceUnit
from depression_rag.observability import get_logger

log = get_logger("config")


@dataclass(frozen=True)
class FigureTranscription:
    """Manually transcribed text for a raster figure the text layer misses."""

    physical_page: int
    unit: str
    pokok_bahasan: str | None
    heading: str
    content_type: str
    text: str  # empty => pending TODO, skipped with a warning


@dataclass(frozen=True)
class ContentTypeRule:
    """A keyword rule mapping a segment to a content_type."""

    content_type: str
    any_keywords: tuple[str, ...] = ()
    all_keywords: tuple[str, ...] = ()

    def matches(self, text_lower: str) -> bool:
        any_ok = any(k in text_lower for k in self.any_keywords) if self.any_keywords else False
        all_ok = all(k in text_lower for k in self.all_keywords) if self.all_keywords else False
        return any_ok or all_ok


@dataclass(frozen=True)
class SourceConfig:
    pdf_path: str
    printed_offset: int
    units: tuple[SourceUnit, ...]
    scope: tuple[str, ...]
    figure_transcriptions: tuple[FigureTranscription, ...]

    def unit(self, name: str) -> SourceUnit:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(f"unknown MI unit: {name!r}")

    def scoped_units(self) -> list[SourceUnit]:
        return [self.unit(n) for n in self.scope]


@dataclass(frozen=True)
class CleaningConfig:
    drop_page_number_header: bool
    dehyphenate: bool
    bullet_marker: str
    glyph_map: dict[str, str]  # actual char -> replacement (hex resolved at load)


@dataclass(frozen=True)
class SegmentationConfig:
    clinical_start_markers: tuple[str, ...]
    reference_markers: tuple[str, ...]
    retain_roles: tuple[str, ...]
    merge_adjacent_atomic: bool
    content_type_priority: tuple[str, ...]
    content_type_rules: tuple[ContentTypeRule, ...]
    # extra structural sub-headings this guideline uses that are not all-caps
    # (e.g. "Kriteria Diagnosis", "Belajar Kasus", "Pengingat"). Each starts a
    # new clinical subsection. Documented in pipeline.yaml.
    subsection_markers: tuple[str, ...] = ()
    # explicit case-vignette markers -> content_type = case_example
    case_markers: tuple[str, ...] = ()
    # suicide-risk terms; risk_suicide is only assigned on heading evidence
    suicide_terms: tuple[str, ...] = ()
    # a segment is "atomic" (never split by structure-aware) only if it is also
    # compact: a tightly-coupled clinical unit is small. Large sections that
    # merely match an atomic keyword are split normally.
    max_atomic_chars: int = 6000
    # segments shorter than this are merged into their neighbour (drops stray
    # one-word heading fragments like "Penting:").
    min_segment_chars: int = 20
    # manual/judge-adjudicated content_type corrections applied AFTER the rule
    # cascade, keyed by segment_id (from configs/content_type_overrides.yaml).
    # These let labels be set by review, not rules alone; `atomic` is recomputed.
    content_type_overrides: dict[str, str] = field(default_factory=dict)
    # where those overrides came from, so the run manifest can checksum the file
    # that produced them. None = no override file was used.
    content_type_overrides_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    formats: tuple[str, ...]
    derived_dir: str
    chunks_dir: str
    manifests_dir: str
    logs_dir: str


@dataclass(frozen=True)
class RunConfig:
    seed: int
    log_level: str


@dataclass(frozen=True)
class ChunkingConfig:
    reference_tokenizer: str
    offline: bool
    configs: tuple[ChunkConfig, ...]


@dataclass(frozen=True)
class PipelineConfig:
    source: SourceConfig
    cleaning: CleaningConfig
    segmentation: SegmentationConfig
    chunking: ChunkingConfig
    output: OutputConfig
    run: RunConfig
    config_path: str
    raw: dict[str, Any]  # original YAML, recorded verbatim in the manifest


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _parse_units(raw_units: list[dict]) -> tuple[SourceUnit, ...]:
    return tuple(
        SourceUnit(
            name=u["name"],
            phys_start=int(u["phys_start"]),
            phys_end=int(u["phys_end"]),
            title=u.get("title", ""),
        )
        for u in raw_units
    )


def _parse_glyph_map(raw_map: dict[str, str]) -> dict[str, str]:
    """Resolve hex-string keys ("F0B7") to actual characters."""
    out: dict[str, str] = {}
    for hex_key, replacement in raw_map.items():
        out[chr(int(str(hex_key), 16))] = replacement
    return out


def _parse_figures(raw_figs: list[dict] | None) -> tuple[FigureTranscription, ...]:
    figs: list[FigureTranscription] = []
    for f in raw_figs or []:
        figs.append(
            FigureTranscription(
                physical_page=int(f["physical_page"]),
                unit=f["unit"],
                pokok_bahasan=f.get("pokok_bahasan"),
                heading=f.get("heading", ""),
                content_type=f.get("content_type", "clinical_exposition"),
                text=f.get("text", "") or "",
            )
        )
    return tuple(figs)


def _parse_content_rules(raw_rules: list[dict]) -> tuple[ContentTypeRule, ...]:
    rules: list[ContentTypeRule] = []
    for r in raw_rules:
        rules.append(
            ContentTypeRule(
                content_type=r["content_type"],
                any_keywords=tuple(k.lower() for k in r.get("any_keywords", [])),
                all_keywords=tuple(k.lower() for k in r.get("all_keywords", [])),
            )
        )
    return tuple(rules)


def _expand_sweep(raw_chunking: dict) -> tuple[ChunkConfig, ...]:
    """Expand ``chunking.sweep`` (or an explicit ``chunking.configs``) into ChunkConfigs."""
    tokenizer = raw_chunking["reference_tokenizer"]
    by_id: dict[str, ChunkConfig] = {}

    # explicit list takes precedence if present
    for c in raw_chunking.get("configs", []) or []:
        cc = ChunkConfig(
            strategy=c["strategy"],
            chunk_size=int(c["chunk_size"]),
            overlap=int(c.get("overlap", 0)),
            context_enriched=bool(c.get("context_enriched", False)),
            reference_tokenizer=c.get("reference_tokenizer", tokenizer),
        )
        by_id[cc.config_id] = cc

    for strategy, spec in (raw_chunking.get("sweep") or {}).items():
        sizes = spec.get("sizes", [])
        overlaps = spec.get("overlaps", [0])
        ctx_flags = spec.get("context_enriched", [False])
        for size, overlap, ctx in itertools.product(sizes, overlaps, ctx_flags):
            if overlap >= size:
                raise ValueError(
                    f"invalid sweep combo for {strategy}: overlap {overlap} >= size {size}"
                )
            cc = ChunkConfig(
                strategy=strategy,
                chunk_size=int(size),
                overlap=int(overlap),
                context_enriched=bool(ctx),
                reference_tokenizer=tokenizer,
            )
            by_id[cc.config_id] = cc

    if not by_id:
        raise ValueError("chunking config produced no configs (set sweep or configs)")
    return tuple(by_id.values())


_DEFAULT_OVERRIDES_FILE = "content_type_overrides.yaml"


def _parse_content_type_overrides(path: Path) -> dict[str, str]:
    """Parse an override file into ``{segment_id: content_type}``.

    Each entry is `segment_id: content_type` or `segment_id: {content_type: ...}`
    (extra keys like `source`/`reason` stay in the file for audit, ignored here).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("overrides", {}) or {}
    return {
        seg_id: (val["content_type"] if isinstance(val, dict) else val)
        for seg_id, val in raw.items()
    }


def _resolve_content_type_overrides(
    config_path: Path, raw_seg: dict
) -> tuple[dict[str, str], str | None]:
    """-> (overrides, the path they came from) for ``segmentation.content_type_overrides``.

    This file is a real pipeline input, not decoration: an override flips a
    segment's content_type and therefore its ``atomic`` flag, which changes what
    the structure-aware chunker emits. Losing it rewrites every chunk file, so
    the three cases are kept apart rather than collapsing to "no overrides":

      * key names a file  -> it MUST exist; absent is a hard error
      * key is null/false -> rules only, declared on purpose
      * key absent        -> legacy default beside the config, WARNING if absent

    Relative paths resolve against the config file's directory, so a config and
    its overrides travel together.
    """
    declared = "content_type_overrides" in raw_seg
    spec = raw_seg.get("content_type_overrides", _DEFAULT_OVERRIDES_FILE)

    if declared and not spec:
        log.info("config.content_type_overrides_disabled",
                 extra={"config": str(config_path)})
        return {}, None

    path = Path(spec)
    if not path.is_absolute():
        path = config_path.parent / path

    if not path.is_file():
        if declared:
            raise FileNotFoundError(
                f"segmentation.content_type_overrides names {spec!r}, resolved to "
                f"{path}, which does not exist. Point it at the real file, or set "
                "the key to null to run on the keyword rules alone — an absent "
                "override file silently changes every chunk file."
            )
        log.warning(
            "config.content_type_overrides_missing",
            extra={"expected": str(path),
                   "effect": "segmentation runs on keyword rules alone; "
                             "content_type and atomic may differ from the "
                             "adjudicated labels",
                   "fix": "set segmentation.content_type_overrides in the config"},
        )
        return {}, None

    overrides = _parse_content_type_overrides(path)
    log.info("config.content_type_overrides_loaded",
             extra={"path": str(path), "n_overrides": len(overrides)})
    return overrides, str(path)


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    """Parse the pipeline YAML into a validated :class:`PipelineConfig`."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    src = raw["source"]
    units = _parse_units(src["units"])
    unit_names = {u.name for u in units}
    scope = tuple(src["scope"])
    unknown = [s for s in scope if s not in unit_names]
    if unknown:
        raise ValueError(f"scope refers to unknown units: {unknown}")

    source = SourceConfig(
        pdf_path=src["pdf_path"],
        printed_offset=int(src["printed_offset"]),
        units=units,
        scope=scope,
        figure_transcriptions=_parse_figures(src.get("figure_transcriptions")),
    )

    clean = raw["cleaning"]
    cleaning = CleaningConfig(
        drop_page_number_header=bool(clean.get("drop_page_number_header", True)),
        dehyphenate=bool(clean.get("dehyphenate", True)),
        bullet_marker=clean.get("bullet_marker", "•"),
        glyph_map=_parse_glyph_map(clean.get("glyph_map", {})),
    )

    seg = raw["segmentation"]
    overrides, overrides_path = _resolve_content_type_overrides(path, seg)
    segmentation = SegmentationConfig(
        clinical_start_markers=tuple(m.lower() for m in seg["clinical_start_markers"]),
        reference_markers=tuple(m.lower() for m in seg["reference_markers"]),
        retain_roles=tuple(seg.get("retain_roles", ["clinical"])),
        merge_adjacent_atomic=bool(seg.get("merge_adjacent_atomic", True)),
        content_type_priority=tuple(seg["content_type_priority"]),
        content_type_rules=_parse_content_rules(seg["content_type_rules"]),
        subsection_markers=tuple(m.lower() for m in seg.get("subsection_markers", [])),
        case_markers=tuple(m.lower() for m in seg.get("case_markers", [])),
        suicide_terms=tuple(m.lower() for m in seg.get("suicide_terms", [])),
        max_atomic_chars=int(seg.get("max_atomic_chars", 6000)),
        min_segment_chars=int(seg.get("min_segment_chars", 20)),
        content_type_overrides=overrides,
        content_type_overrides_path=overrides_path,
    )

    chk = raw["chunking"]
    chunking = ChunkingConfig(
        reference_tokenizer=chk["reference_tokenizer"],
        offline=bool(chk.get("offline", False)),
        configs=_expand_sweep(chk),
    )

    out = raw["output"]
    output = OutputConfig(
        formats=tuple(out.get("formats", ["jsonl"])),
        derived_dir=out["derived_dir"],
        chunks_dir=out["chunks_dir"],
        manifests_dir=out["manifests_dir"],
        logs_dir=out.get("logs_dir", "outputs/logs"),
    )

    run = raw.get("run", {})
    run_cfg = RunConfig(
        seed=int(run.get("seed", 20260628)),
        log_level=run.get("log_level", "INFO"),
    )

    return PipelineConfig(
        source=source,
        cleaning=cleaning,
        segmentation=segmentation,
        chunking=chunking,
        output=output,
        run=run_cfg,
        config_path=str(path),
        raw=raw,
    )
