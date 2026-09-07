# `content_type` / `meta_content_type` — definition note

*Working notes on extraction, cleaning and segmentation. Written 2026-06-30.
Defines how the `content_type` label is assigned to segments and how it becomes
`meta_content_type` on chunks, so the field's provenance is auditable.*

---

## 1. What it is and why it exists

`content_type` is a **descriptive label** attached to each segment (and inherited
by each chunk) that records what *kind* of clinical content the text is. It is
used for:

- segmentation logic — "atomic" tightly-coupled units (criteria, dosage, somatic,
  referral) are kept whole and only merged with adjacent atomics of the **same**
  `content_type` (so a dose table is never glued to unrelated prose);
- inspection / analysis and run-manifest reporting (per-type distribution).

It is **not** a retrieval input: the bake-off scores relevance by char-overlap
with gold passages (§2.5 Step 4), so `content_type` cannot bias any metric.

## 2. The label set (fixed enum)

`ContentType` in `src/depression_rag/domain/models.py` — 7 values:

`clinical_exposition`, `somatic_symptoms`, `risk_suicide`, `criteria`,
`case_example`, `dosage`, `referral_criteria`.

`ATOMIC_CONTENT_TYPES` marks the subset treated as non-splittable clinical units.

## 3. How a segment's `content_type` is assigned

Assigned by `GuidelineSegmenter._classify()`
(`src/depression_rag/extraction/segmentation.py`) as a **priority-ordered
cascade** (first match wins). Matching is case-insensitive substring/regex on the
segment **text** and **heading_path** — deterministic, no model, no randomness:

1. **`case_example`** — if a case marker (e.g. "Belajar Kasus", from
   `case_markers`) appears in the heading or at the start of the text (Decision 3).
2. **`risk_suicide`** — requires a `suicide_terms` term in the **heading**. A body
   mention alone (even several) does not relabel a segment whose topic is something
   else. *(Tightened 2026-07-01 after the LLM-judge audit in §8; the earlier
   `OR ≥ risk_suicide_min_mentions` clause over-fired on incidental mentions and is
   now unused.)*
3. **configured keyword rules** (`content_type_rules` in `configs/pipeline.yaml`)
   — each rule fires if ANY `any_keywords` is present OR ALL `all_keywords` are
   present. The `dosage` rule additionally fires on a numeric mg-dose regex
   (`_MG_RE`). Multiple matches are resolved by a fixed **priority order**
   (`content_type_priority`):

   ```
   criteria > dosage > somatic_symptoms > referral_criteria > clinical_exposition
   ```
4. **default** → `clinical_exposition` if nothing matched.

The rules themselves (current config):

| content_type | trigger (any/all keywords; +regex) |
|---|---|
| criteria | ALL of {"gejala utama","gejala tambahan"}; or ANY of {"kriteria diagnosis","pedoman diagnostik","gejala utama","gejala tambahan"} |
| dosage | ANY drug name {fluoksetin, sertralin, amitriptilin, imipramin, klomipramin, maprotilin, …} or "mg/hari"; **+ numeric mg-dose regex** |
| somatic_symptoms | ANY of {"gejala somatik","keluhan somatik","keluhan fisik","gejala fisik tanpa"} |
| referral_criteria | ANY of {"kriteria rujukan","indikasi rujukan","kapan merujuk","rujuk balik","rujukan kasus"} |
| risk_suicide | via step (2) above (suicide term in heading), not these rules |
| case_example | via the marker step (1) above |
| clinical_exposition | default |

## 4. How a chunk gets `meta_content_type`

A chunk does **not** recompute its type. In `src/depression_rag/chunking/base.py`,
`span_metadata()` calls `dominant_segment()` — the segment with the **largest
char-overlap** with the chunk's `[char_start, char_end]` — and copies that
segment's metadata onto the chunk. Serialized into the chunk record it is prefixed
`meta_`, giving **`meta_content_type`**. So:

> `meta_content_type` of a chunk = `content_type` of the segment it most overlaps.

(`spans_segments` records how many segments the chunk touches, for diagnostics.)

## 5. Observed distribution (68 segments, current scope)

After the 2026-07-01 `risk_suicide` tightening (§8):
`clinical_exposition` 44, `dosage` 8, `risk_suicide` 5, `referral_criteria` 4,
`somatic_symptoms` 3, `criteria` 3, `case_example` 1.
(Before tightening: clinical 42, risk_suicide 9, dosage 7, referral 3.)

## 5a. Why these categories (rationale)

The taxonomy is **functional, not a universal clinical ontology**. It exists to
serve structure-aware chunking (spec §2.3): for each piece of text it answers
*"if a chunker cut through the middle of this, would the result still be usable?"*

- Narrative/didactic prose → a partial chunk is still useful → safe to split.
- Certain self-contained clinical units → a partial chunk is wrong or unsafe →
  must stay whole ("atomic").

So at its core the scheme is **atomic vs. non-atomic**; the categories are the
refinement that says *which* kinds are atomic and let the merger combine only like
with like.

**Each atomic category is a place where partial retrieval destroys meaning or
safety:**

| category | why it must stay a unit |
|---|---|
| `criteria` | a thresholded rule (≥2 of 3 main, ≥3 of 7 additional, ≥2 weeks); split → threshold without symptoms, or symptoms without threshold — useless |
| `dosage` | drug + dose + titration bound together; split → a drug separated from its dose — meaningless/unsafe |
| `somatic_symptoms` | a coherent symptom profile; split → a fragmented list |
| `referral_criteria` | a "when to refer" decision rule meant to be retrieved whole |
| `risk_suicide` | safety-critical content; coherence matters most where errors are most dangerous |
| `case_example` | worked vignettes are structurally different from didactic text; tagging them lets us treat them distinctly (e.g. excluded from gold passages) |
| `clinical_exposition` | the default — splittable prose |

These mirror the **forms the guideline itself uses** (exposition, symptom lists,
criteria, dosing, referral rules, case studies, emergency/suicide); the boundaries
follow the document's own structure rather than an imposed scheme.

**Why this granularity (not coarser or finer):**

- *Coarser* (just atomic/not) would lose same-type adjacent merging — the merger
  only joins a dose block with another dose block, never with a criteria block, so
  no giant mixed unsplittable chunk is formed — and would lose per-type reporting.
- *Finer* (more categories) would introduce labels the document does not clearly
  signal, producing noisy/over-fit tags. The 7 chosen are exactly those with
  clear, keyword/regex-detectable signals in *this* guideline, so labeling stays
  deterministic and auditable rather than speculative. Any category that could not
  be reliably detected, or that the document did not contain, was not invented.

## 6. One-line answer (for defense)

> `meta_content_type` is a chunk's inherited content label — the `content_type` of
> the segment it most overlaps. That segment label is assigned deterministically by
> a config-driven, priority-ordered cascade (case-vignette marker → suicide-in-
> heading → keyword/dose rules by fixed priority → default `clinical_exposition`),
> defined in `configs/pipeline.yaml` and `extraction/segmentation.py`. It is
> descriptive metadata, not a retrieval input.

## 7. Where it's defined

| concern | location |
|---|---|
| enum + atomic set | `src/depression_rag/domain/models.py` (`ContentType`, `ATOMIC_CONTENT_TYPES`) |
| assignment cascade | `src/depression_rag/extraction/segmentation.py` (`_classify`) |
| rules + priority + markers | `configs/pipeline.yaml` (`content_type_rules`, `content_type_priority`, `case_markers`, `suicide_terms`; `risk_suicide_min_mentions` now unused after the §8 tightening) |
| propagation to chunks | `src/depression_rag/chunking/base.py` (`span_metadata`, `dominant_segment`) |

## 8. Auditing the labels (LLM-as-judge)

The rule-based labels are rechecked with an **independent LLM rater**, at the
**segment** level (chunk `meta_content_type` is inherited, so fixing a segment
fixes its chunks). The audit is **blind**: the judge classifies each segment from
the category definitions only — it never sees the rule label, the keyword rules,
or this note — so it is a genuine second rater, not a rubber stamp.

Workflow (prompt `content-type-audit-v1`, recorded in
`configs/prompts/content_type_audit.md`):

1. `scripts/export_segments_for_judge.py` → `data/derived/segments_for_judge.jsonl`
   (segment_id, heading_path, text only; `content_type` withheld).
2. Run the judge prompt on the LLM with **only that file attached**; save output
   (record the model name used).
3. `scripts/compare_content_type.py` (single judge) or
   `scripts/compare_content_type_multi.py` (multiple judges) reports agreement,
   Cohen's kappa, a confusion table, and every disagreement.

The rule labels are **not** ground truth: disagreements are candidates for human
adjudication, not automatic edits. A confirmed fix is made by editing the rule /
keyword in `configs/pipeline.yaml` / `extraction/segmentation.py`, never the data
by hand, so the labeling stays reproducible.

**Result (2026-07-01).** Two independent blind judges (Claude + GPT) were run
(`content_type_judge_claude.jsonl`, `content_type_judge_gpt.jsonl`). They agreed
with **each other** near-perfectly (98.5%, κ = 0.964) and with the rule ~84%
(κ ≈ 0.68), jointly flagging 10 segments. The dominant pattern — `risk_suicide`
over-firing on incidental suicide mentions (e.g. `MI4_0042` "RUJUKAN KASUS",
`MI4_0022` epidemiology) — was fixed by the heading-only tightening in §3.
Post-fix agreement rose to **rule vs Claude 89.7% (κ = 0.793)** and **rule vs GPT
88.2% (κ = 0.759)**, with 8 residual disagreements (the separate `dosage`/`somatic`/
`criteria` over-tags + the out-of-scope `MI1_0017` heading-merge artifact), none of
which affect the gold passages or retrieval scoring. Full report:
`data/derived/content_type_audit_report.md`.
