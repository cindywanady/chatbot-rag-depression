# Gold-concept induction prompt (Stage 1) — `gold-concept-induction-v1`

**Purpose.** Inductively **derive the concept taxonomy** for depression case
management *from the segments themselves* — no predefined concept list is given.
This provides a bottom-up justification for the gold-passage concepts (instead of
asserting them). Run on >=1 LLM (e.g. Claude + GPT); their proposed concepts are
then reconciled with each other and with `build_spec.md` line 228 to fix the final
vocabulary used in Stage 2 (`gold-concept-v1`).

**Stage in the workflow:**
```
segments_for_judge.jsonl
  ├─(Stage 1, THIS prompt, no codebook)→ Claude concepts + GPT concepts
  │        → reconcile with each other and spec line 228 → FINAL concept list
  └─(Stage 2, gold-concept-v1, with final codebook)→ assign concepts + splits
```

**Generator.** Any Claude/GPT model — record the exact model name as provenance.

**Attach exactly one file:** `data/derived/segments_for_judge.jsonl` (blind:
`segment_id`, `source_unit`, `heading_path`, `n_chars`, `text`). Do NOT attach
`gold_passages.yaml`, the spec, or any concept list — Stage 1 must be blind to the
target taxonomy.

**After generation:** save the output (e.g. `data/derived/gold_concept_induction_claude.json`
and `..._gpt.json`); these feed the reconciliation step.

---

```text
You are a clinical content analyst. You are given text segments extracted from an
Indonesian Ministry of Health mental-health guideline for general practitioners.
By reading the segments, INDUCTIVELY derive the set of clinical concepts that the
DEPRESSION case-management content covers — propose a concept taxonomy grounded in
what the text actually contains. Do NOT use any predefined list; derive the
concepts yourself.

INPUT: the attached file `segments_for_judge.jsonl`, one JSON object per line, with
segment_id, source_unit, heading_path, n_chars, text. Use only the text provided.

SCOPE: consider only content useful for managing a patient with DEPRESSION
(definition, symptoms, diagnosis, treatment, referral, suicide-risk handling,
depression in special populations). Ignore out-of-scope material: general non-
depression mental-health detection, psychiatric-interview technique, non-depression
emergencies (delirium, substance intoxication/withdrawal, psychotic agitation),
and reference lists / teaching scaffolding.

TASK:
1. Identify the distinct clinical concepts present. A concept is a coherent clinical
   topic a counselor would seek as a unit (e.g. how depression is defined, its risk
   factors, its core symptoms, how it is diagnosed, how it is treated and dosed,
   when to refer, how to handle suicide risk, etc.). Derive them at a CONSISTENT
   granularity — neither one giant "treatment" concept nor dozens of tiny ones.
2. For each concept: give a one-sentence definition and the segment_ids that
   exemplify it.
3. Flag segments that appear to contain MORE THAN ONE concept (candidates for
   splitting), naming the segment_id and the concepts involved.

RULES:
- Ground every concept in the actual text; do not invent concepts the segments do
  not contain. Name concepts in lowercase snake_case.
- Aim for a compact, non-overlapping set (roughly 8-14 concepts).

OUTPUT: return ONLY one JSON object, no prose, no code fences:
{
  "concepts": [
    {"concept": "...", "definition": "...", "example_segment_ids": ["..."]}
  ],
  "split_candidates": [
    {"segment_id": "...", "concepts": ["...", "..."]}
  ]
}
```
