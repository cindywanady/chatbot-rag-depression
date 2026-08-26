"""The judge-adjudicated content_type overrides: loaded, applied, and hard to lose.

These labels are not cosmetic. An override flips a segment's ``content_type``,
which flips ``atomic``, which decides whether the structure-aware chunker emits
the segment whole or splits it — so silently dropping the override file rewrites
every chunk file and every index built from them.

That is exactly what used to happen: the file was resolved as a sibling of
whatever ``--config`` pointed at, and a missing file returned ``{}`` with no
warning. Running the pipeline from a copied config in another directory moved 7
content_types and 6 atomic flags without a word. These tests pin the three
behaviours that make that impossible to repeat.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from depression_rag.config import load_pipeline_config
from depression_rag.domain import ATOMIC_CONTENT_TYPES
from depression_rag.extraction import (
    GuidelineSegmenter,
    GuidelineTextCleaner,
    PyMuPdfLoader,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"

# The 2026-07-01 blind two-judge consensus (configs/content_type_overrides.yaml).
# Every one of these was labelled by the keyword rules as something else; the
# right-hand side is what the adjudication settled on. Hard-coded here so a
# silent change to the override file fails a test instead of a thesis claim.
ADJUDICATED = {
    "MI1_0001": "clinical_exposition",
    "MI1_0010": "clinical_exposition",
    "MI1_0017": "clinical_exposition",
    "MI4_0030": "clinical_exposition",
    "MI4_0035": "clinical_exposition",
    "MI4_0039": "clinical_exposition",
    "MI7_0057": "clinical_exposition",
}


def _segment_with(config):
    loader = PyMuPdfLoader(config.source)
    cleaner = GuidelineTextCleaner(config.cleaning)
    segmenter = GuidelineSegmenter(
        config.source, config.cleaning, config.segmentation, cleaner
    )
    try:
        return segmenter.segment(loader.load_pages(config.source.scope))
    finally:
        loader.close()


def _config_copy(tmp_path: Path, *, with_overrides: bool, overrides_key=...):
    """A runnable copy of pipeline.yaml in tmp_path, optionally without the
    override file beside it and/or with the key rewritten."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if overrides_key is not ...:
        if overrides_key is None:
            raw["segmentation"]["content_type_overrides"] = None
        else:
            raw["segmentation"]["content_type_overrides"] = overrides_key
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    if with_overrides:
        shutil.copy(CONFIG.parent / "content_type_overrides.yaml", tmp_path)
    return path


def test_overrides_are_loaded_from_the_project_config(config):
    """The shipped config resolves the override file and records where it came from."""
    assert config.segmentation.content_type_overrides == ADJUDICATED
    recorded = config.segmentation.content_type_overrides_path
    assert recorded is not None, "the path must be recorded so the manifest can hash it"
    assert Path(recorded).is_file()


def test_overrides_reach_the_segments_and_recompute_atomic(segmented):
    """The adjudicated label wins over the rule label, and `atomic` follows it."""
    by_id = {s.segment_id: s for s in segmented.segments}
    for seg_id, expected in ADJUDICATED.items():
        seg = by_id[seg_id]
        assert seg.content_type.value == expected, f"{seg_id} lost its adjudicated label"
        # clinical_exposition is not atomic, so every override here must have
        # cleared the flag the rule cascade would otherwise have set
        assert seg.atomic == (seg.content_type in ATOMIC_CONTENT_TYPES)
        assert not seg.atomic


def test_a_named_but_missing_override_file_is_a_hard_error(tmp_path):
    """The failure this whole module exists to prevent: config moved, file left behind."""
    cfg_path = _config_copy(tmp_path, with_overrides=False)
    with pytest.raises(FileNotFoundError, match="content_type_overrides"):
        load_pipeline_config(str(cfg_path))


def test_overrides_can_be_disabled_on_purpose(tmp_path):
    """`null` means rules-only, and rules-only really does produce different labels.

    The second half is the point: it demonstrates the overrides are load-bearing,
    so the hard error above is protecting something real rather than being noise.
    """
    cfg_path = _config_copy(tmp_path, with_overrides=False, overrides_key=None)
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg.segmentation.content_type_overrides == {}
    assert cfg.segmentation.content_type_overrides_path is None

    rules_only = {s.segment_id: s for s in _segment_with(cfg).segments}
    differing = {
        seg_id: (rules_only[seg_id].content_type.value, expected)
        for seg_id, expected in ADJUDICATED.items()
        if rules_only[seg_id].content_type.value != expected
    }
    assert differing, (
        "rules alone reproduced every adjudicated label — the override file would "
        "then be dead weight, and this test's premise is wrong"
    )
    # and the flags move with the labels, which is what reaches the chunker
    assert any(rules_only[seg_id].atomic for seg_id in differing)


def test_relative_override_path_resolves_against_the_config_not_the_cwd(tmp_path):
    """A config carries its overrides with it when both are moved together."""
    cfg_path = _config_copy(tmp_path, with_overrides=True)
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg.segmentation.content_type_overrides == ADJUDICATED
    assert Path(cfg.segmentation.content_type_overrides_path).parent == tmp_path
