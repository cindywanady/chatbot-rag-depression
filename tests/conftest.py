"""Shared pytest fixtures.

Builds the segmented document and the reference tokenizer once per session and
reuses them, so the suite runs fast. The HF cache is pinned to the in-project
``.hf_cache`` and the tokenizer is loaded offline (cache only), so tests are
hermetic and need no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.chdir(PROJECT_ROOT)  # config paths are relative to the project root

from depression_rag.config import load_pipeline_config  # noqa: E402
from depression_rag.extraction import (  # noqa: E402
    GuidelineSegmenter,
    GuidelineTextCleaner,
    PyMuPdfLoader,
)
from depression_rag.observability import setup_logging  # noqa: E402

setup_logging(level="ERROR", json_console=False)


@pytest.fixture(scope="session")
def config():
    return load_pipeline_config(str(PROJECT_ROOT / "configs" / "pipeline.yaml"))


@pytest.fixture(scope="session")
def cleaner(config):
    return GuidelineTextCleaner(config.cleaning)


@pytest.fixture(scope="session")
def segmented(config, cleaner):
    loader = PyMuPdfLoader(config.source)
    segmenter = GuidelineSegmenter(config.source, config.cleaning, config.segmentation, cleaner)
    try:
        return segmenter.segment(loader.load_pages(config.source.scope))
    finally:
        loader.close()


@pytest.fixture(scope="session")
def tokenizer(config):
    from depression_rag.chunking import ReferenceTokenizer

    return ReferenceTokenizer(config.chunking.reference_tokenizer, offline=True)


@pytest.fixture(scope="session")
def embedding_config():
    from depression_rag.config import load_embedding_config

    return load_embedding_config(str(PROJECT_ROOT / "configs" / "embedding_models.yaml"))
