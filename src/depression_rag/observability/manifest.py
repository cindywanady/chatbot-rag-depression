"""Run-manifest writer.

A manifest is a single JSON file capturing everything needed to reproduce and
defend a run. It records:

  * identity        - run_id, UTC timestamp, host, working dir
  * environment     - Python version, platform, key package versions, git commit
  * determinism     - the seed snapshot
  * inputs          - source PDF + config files, each with sha256 + size
  * configuration   - the fully-resolved config dict actually used
  * reference model - tokenizer checkpoint + resolved revision hash
  * outputs         - every file written, with sha256 + size
  * metrics         - free-form counters the pipeline records (segment/chunk counts...)

Nothing here imports the pipeline stages, so the manifest can wrap any run.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

# Packages whose exact versions matter for reproducing chunk output.
_TRACKED_PACKAGES = (
    "PyMuPDF",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
)


def _record_path(path: Path) -> str:
    """Path as recorded in a manifest: repo-relative when possible.

    Stages disagreed on this before — the pipelines passed relative paths from
    their configs while the eval built absolute ones from ``ROOT / ...`` — so
    chaining one manifest's outputs to the next one's inputs by string equality
    silently produced zero matches even though every hash agreed. Normalising
    here makes the chunking -> indexing -> evaluation chain checkable without
    resolving paths first. Anything outside the working directory (a scratch
    ``--out-dir``) stays absolute, which is the honest record.
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (handles the multi-MB PDF without loading it)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None  # not a git repo / git unavailable - recorded honestly as null


@dataclass
class ManifestBuilder:
    """Accumulates manifest fields during a run, then writes JSON."""

    run_id: str
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    seed: dict[str, Any] = field(default_factory=dict)
    reference_model: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    _created_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )

    # -- inputs -------------------------------------------------------------
    def add_input(self, label: str, path: Path | str) -> "ManifestBuilder":
        path = Path(path)
        info: dict[str, Any] = {"path": _record_path(path)}
        if path.is_file():
            info["sha256"] = sha256_file(path)
            info["bytes"] = path.stat().st_size
        else:
            info["missing"] = True
        self.inputs[label] = info
        return self

    # -- outputs ------------------------------------------------------------
    def add_output(self, path: Path | str, **extra: Any) -> "ManifestBuilder":
        path = Path(path)
        info: dict[str, Any] = {"path": _record_path(path), **extra}
        if path.is_file():
            info["sha256"] = sha256_file(path)
            info["bytes"] = path.stat().st_size
        self.outputs.append(info)
        return self

    # -- setters ------------------------------------------------------------
    def set_config(self, config: dict[str, Any]) -> "ManifestBuilder":
        self.config = config
        return self

    def set_seed(self, seed_state: Any) -> "ManifestBuilder":
        # accepts a SeedState dataclass or a plain dict
        self.seed = getattr(seed_state, "__dict__", None) or dict(seed_state)
        return self

    def set_reference_model(self, **fields: Any) -> "ManifestBuilder":
        self.reference_model.update(fields)
        return self

    def record(self, key: str, value: Any) -> "ManifestBuilder":
        self.metrics[key] = value
        return self

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "description": self.description,
            "created_at": self._created_at,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "cwd": str(Path.cwd()),
                "packages": _package_versions(),
                "git_commit": _git_commit(),
            },
            "seed": self.seed,
            "inputs": self.inputs,
            "config": self.config,
            "reference_model": self.reference_model,
            "metrics": self.metrics,
            "outputs": self.outputs,
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, default=str)
        return path
