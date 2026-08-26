"""The single reference tokenizer that defines chunk sizes.

One fixed tokenizer - the XLM-RoBERTa tokenizer that multilingual-e5 uses -
measures ``chunk_size`` for every config, so "256 tokens" means the same amount
of text under every strategy. Each embedding model still encodes with its own
tokenizer at indexing time; this one only *measures* sizes and provides
char-offset mapping. It is a fast tokenizer (Rust backend), so no
PyTorch/TensorFlow is required; note ``return_offsets_mapping`` needs
``is_fast``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Quieten transformers before it is imported: we use the tokenizer only, so the
# "None of PyTorch/TensorFlow/Flax found" advisory is expected and not useful.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ReferenceTokenizer:
    """Implements the ``TokenCounter`` port over a HF fast tokenizer."""

    def __init__(
        self,
        checkpoint: str = "intfloat/multilingual-e5-large",
        cache_dir: str | None = None,
        offline: bool = False,
    ) -> None:
        import transformers
        from transformers import AutoTokenizer

        transformers.logging.set_verbosity_error()  # silence > model_max_length warnings
        self._checkpoint = checkpoint
        self._tok = AutoTokenizer.from_pretrained(
            checkpoint,
            cache_dir=cache_dir or os.environ.get("HF_HOME"),
            local_files_only=offline,
        )
        if not self._tok.is_fast:
            raise ValueError(
                f"reference tokenizer {checkpoint!r} is not a fast tokenizer; "
                "char-offset mapping requires a fast tokenizer"
            )

    @property
    def name(self) -> str:
        return self._checkpoint

    @property
    def revision(self) -> str | None:
        """Resolved commit hash (pins the tokenizer in the run manifest)."""
        commit = getattr(self._tok, "_commit_hash", None)
        if commit:
            return commit
        # fall back to the HF cache's refs/main pointer
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            ref = (
                Path(hf_home)
                / "hub"
                / f"models--{self._checkpoint.replace('/', '--')}"
                / "refs"
                / "main"
            )
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip()
        return None

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Per-token ``(char_start, char_end)`` offsets into ``text``."""
        if not text:
            return []
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(a), int(b)) for a, b in enc["offset_mapping"]]
