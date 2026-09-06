#!/usr/bin/env python3
"""Assemble Multimedia Appendix 1 — prompts and evaluation criteria.

    python scripts/figures/build_appendix1_prompts.py

Pulls each prompt verbatim from the file that actually produced it, so the
appendix cannot drift from the code. Nothing is retyped or paraphrased.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "documents/thesis_clean/appendix/Multimedia_Appendix_1_Prompts.md"

# --- prompts held as standalone markdown specs -------------------------------
SPEC_FILES = [
    ("A1.1", "Question generation for the retrieval evaluation set",
     "configs/prompts/qa_generation.md"),
    ("A1.2", "Concept induction for the reference passages (stage 1)",
     "configs/prompts/gold_concept_induction.md"),
    ("A1.3", "Concept and split annotation for the reference passages (stage 2)",
     "configs/prompts/gold_passage_concept_audit.md"),
    ("A1.4", "Content-type audit of the source segments",
     "configs/prompts/content_type_audit.md"),
]

# --- prompts held in the deployed chatbot configuration ----------------------
YAML_KEYS = [
    ("A1.5", "Deployed system prompt, shared base", "base"),
    ("A1.6", "Deployed briefing prompt, counselor briefing", "counselor_briefing"),
]

# --- prompts held in the evaluation code -------------------------------------
CODE_CONSTS = [
    ("A1.7", "Claim-support judge", "FAITHFULNESS_PROMPT"),
    ("A1.8", "Structural compliance judge", "STRUCTURE_PROMPT"),
    ("A1.9", "Question-level risk classification (evaluation)", "QUESTION_RISK_PROMPT"),
    ("A1.10", "Briefing safety-content assessment", "RISK_PROMPT"),
]
CODE_FILE = "scripts/judge_study_answers.py"


def extract_consts(path: Path, names: list[str]) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    out = {}
    for n in names:
        m = re.search(rf'^{n}\s*=\s*"""(.*?)"""', src, re.S | re.M)
        if m:
            out[n] = m.group(1).strip("\n")
    return out


def main() -> int:
    parts: list[str] = []
    parts.append(
        "# Multimedia Appendix 1 — Prompts and evaluation criteria\n\n"
        "Every prompt below is reproduced verbatim from the file that produced it; "
        "none has been retyped or edited for presentation. Prompts are in Indonesian "
        "because the corpus, the questions and the participants are Indonesian.\n\n"
        "| Section | Prompt | Source in the repository |\n|---|---|---|\n"
    )
    index_rows = []

    body: list[str] = []
    for tag, title, rel in SPEC_FILES:
        p = ROOT / rel
        index_rows.append(f"| {tag} | {title} | `{rel}` |")
        body.append(f"\n\n---\n\n## {tag} {title}\n\n"
                    f"Source: `{rel}`\n\n" + p.read_text(encoding="utf-8").strip())

    cfg = yaml.safe_load((ROOT / "configs/chatbot.yaml").read_text(encoding="utf-8"))
    prompts = cfg.get("prompts", {})
    for tag, title, key in YAML_KEYS:
        index_rows.append(f"| {tag} | {title} | `configs/chatbot.yaml` → `prompts.{key}` |")
        text = str(prompts.get(key, "")).strip()
        body.append(f"\n\n---\n\n## {tag} {title}\n\n"
                    f"Source: `configs/chatbot.yaml`, key `prompts.{key}`\n\n"
                    "```\n" + text + "\n```")

    consts = extract_consts(ROOT / CODE_FILE, [n for _, _, n in CODE_CONSTS])
    for tag, title, name in CODE_CONSTS:
        index_rows.append(f"| {tag} | {title} | `{CODE_FILE}` → `{name}` |")
        body.append(f"\n\n---\n\n## {tag} {title}\n\n"
                    f"Source: `{CODE_FILE}`, constant `{name}`\n\n"
                    "```\n" + consts.get(name, "[NOT FOUND]") + "\n```")

    text = parts[0] + "\n".join(index_rows) + "".join(body) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(text.split())} words, "
          f"{len(SPEC_FILES) + len(YAML_KEYS) + len(CODE_CONSTS)} prompts + 1 placeholder)")
    for _, title, _ in SPEC_FILES + YAML_KEYS + CODE_CONSTS:
        print(f"   included: {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
