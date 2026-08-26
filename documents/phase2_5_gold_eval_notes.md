# Phase 1 §2.5 — Gold evaluation set: design notes

*Companion to `documents/build_spec_phase1.md` §2.5. Written 2026-06-30. Covers
the gold relevance passages and the depression-specific question set that the
retrieval bake-off (§2.6) is scored against. Style matches
`reference_tokenizer_notes.md` and `phase2_embedding_indexing_notes.md`.*

> **Dated record — describes the 2026-06-30 state.** The counts below are v1:
> the gold set grew to **32** passages and the question set was regenerated with
> the qa-gen-v2 prompt to **124** human-validated questions (**121** since
> 2026-07-26, when passage `MI1_0003` and its 3 questions were removed; the gold
> set is now **31** passages). The v1 authoring
> script was archived 2026-07-26 (it cites 4 gold passages that no longer exist
> and cannot run); the 100 v1 questions survive as data in
> `archive/questions_candidate_old.jsonl`. Current wiring:
> [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. What §2.5 has to produce, and why

The bake-off compares 56 indexes (chunking × model) and must pick a winner using
rank-aware IR metrics (Recall@k, MRR, nDCG@k). Those metrics need two things the
raw corpus cannot supply on its own:

1. **a question**, and
2. **a known correct answer location** to score retrieval against.

The spec's core anti-bias rule (line 226): the evaluation set must be generated
from the **source clinical text**, with relevance labelled at the **source-passage
level**, and **never from the chunks** produced by the strategies under test —
otherwise the eval favours whichever strategy produced the questions.

This gives the two-artifact design:

```
data/derived/cleaned_text.txt        raw substrate + char coordinate system
  -> data/derived/gold_passages.jsonl   curated, concept-labelled slices (relevance units)
     -> data/derived/questions_candidate.jsonl  questions, each tagged with its answer passage(s)
```

Gold passages are not a *different* text — they are exact, ID'd, concept-labelled
slices of `cleaned_text.txt` (same coordinate system). They add the
`(concept, passage_id, char_span)` scaffolding that turns "some text" into a
**measurable retrieval target**, which the raw blob cannot be.

---

## 2. Gold passages (Step 1)

### 2.1 Scope

Built only from the **case-management scope** (Decision 1): MI.1, MI.2, MI.4,
MI.7, MI.8 — the same scope indexed in the bake-off, so questions test exactly
what is retrievable. Non-depression MI.7 content (delirium, NAPZA, psychotic
agitation) is excluded from gold passages even though it is *indexed*; it remains
in the vector space as realistic distractor/noise.

### 2.2 Construction

The judgement layer is a versioned, human-auditable config,
`configs/gold_passages.yaml` (segment_id → concept, or ordered splits). The
builder resolves it deterministically:

* **whole-segment passages** take the segment's `[char_start, char_end]`;
* **split passages** (multi-concept segments) are cut at literal `anchor`
  substrings — e.g. `MI4_0025` is split at `"Gejala-gejala tambahan depresi yaitu"`
  and `"Kriteria Diagnosis Depresi"` into main symptoms / additional symptoms /
  diagnostic criteria.

Every resolved span is checked so that
`cleaned_text[char_start:char_end] == passage.text` for **all** passages. Splits
trim surrounding whitespace while keeping spans exact.

Code: `src/depression_rag/evaluation/gold_passages.py`; runner
`scripts/build_gold_passages.py`; output `data/derived/gold_passages.jsonl`.

### 2.3 Result — 29 passages, 12 concepts

Concepts align with the spec's expected-concept list (line 228) and were
independently corroborated by inductive LLM concept-derivation (§2.5). Plus
`special_populations` (required by the question taxonomy, line 232) and
`differential_comorbidity` (surfaced by both inductive runs; added §2.5).

| concept | passages | chars | source |
|---|--:|--:|---|
| definition_course | 3 | 4,131 | MI.4 A |
| risk_factors | 1 | 1,114 | MI.4 A |
| somatic_presentation | 1 | 1,494 | MI.4 B |
| main_symptoms | 1 | 330 | MI.4 B |
| additional_symptoms | 1 | 341 | MI.4 B |
| diagnostic_criteria | 2 | 843 | MI.4 B |
| differential_comorbidity | 6 | 1,894 | MI.4 B |
| pharmacotherapy_dosing | 3 | 10,291 | MI.4 C |
| psychoeducation | 1 | 3,967 | MI.4 C |
| special_populations | 3 | 2,514 | MI.4 B |
| referral_criteria | 4 | 24,803 | MI.4 D + MI.8 |
| suicide_risk_emergency | 3 | 10,582 | MI.7 |

By source unit: MI.4 (23), MI.7 (3), MI.8 (3).

### 2.4 No separate "benzodiazepine" concept

There is no standalone benzodiazepine-use/tapering concept, because the scope
contains no benzodiazepine *tapering* guidance for depression (graded-withdrawal
phrasing: 0 occurrences). The only depression-context benzodiazepine mention is a
fixed one-week diazepam course for SSRI-induced akathisia, which belongs to —
and is captured under — `pharmacotherapy_dosing` (`MI4_0040`). Other
benzodiazepine content (emergency sedation, substance withdrawal) is in MI.7
outside the depression scope. Concepts track the document; none is invented to
fill a checklist (no hallucination).

### 2.5 Taxonomy justification — inductive LLM concept derivation

The concept list is justified two ways that agree: **top-down** (spec line 228)
and **bottom-up** (an LLM, blind to the spec list, derives concepts from the
segments). Stage 1 (`configs/prompts/gold_concept_induction.md`,
`gold-concept-induction-v1`) was run independently on Claude and GPT over the
blind segments (`data/derived/segments_for_judge.jsonl`); outputs in
`data/derived/gold_concept_induction_{claude,gpt}.json`.

**Result.** Both models independently produced ~14 concepts that map cleanly onto
the spec's 11 — strong convergence (definition/burden, risk/etiology, somatic
presentation, symptoms, diagnosis, pharmacotherapy, psychoeducation, referral,
suicide, special populations). Two deviations were handled deliberately:

1. **Symptom granularity.** Both models *lumped* main + additional symptoms into a
   single "symptom" concept; they did **not** independently split `main_symptoms`
   from `additional_symptoms`. That finer split is retained because it is mandated
   by the spec (separate expected concepts) and matches the document's own headings
   ("gejala utama" / "gejala tambahan") — i.e. justified top-down + structurally,
   not by induction.
2. **A concept the spec omitted: `differential_comorbidity`.** Both models surfaced
   it (Claude "differential_diagnosis"; GPT "comorbidity_and_integrated_care"),
   covering the bipolar/psychotic recognition reminders and comorbid-illness
   treatment cautions (`MI4_0028`–`MI4_0033`). It was **added** as a 12th concept on
   that basis (6 passages, 1,894 chars).

**Split corroboration (Step-1 `split_candidates`).** The `MI4_0022` three-way split
(definition / risk_factors / impact) was proposed **independently by both models**
with exactly those three parts — strong justification. The `MI4_0025` symptoms-vs-
criteria boundary was corroborated; the main-vs-additional sub-split was not (see
deviation 1). GPT's split list was noisy (it tagged incidental
suicide/comorbidity mentions); Claude's seven were the clean signal.

**Next (Stage 2).** `gold-concept-v1` (`gold_passage_concept_audit.md`) assigns
concepts + split anchors per segment against this frozen 12-concept codebook, on
Claude + GPT; those outputs are then compared to `gold_passages.yaml` to confirm
the per-segment assignment and the anchors.

---

## 3. Question generation (Step 2)

### 3.1 Generator and reproducibility

* **Generator:** `claude-opus-4-8`, run via its chat interface (no API
  scripting).
* **Prompts:** `configs/prompts/qa_generation.md`, version `qa-gen-v1` (system +
  rule prompt verbatim, plus a concept→question_type table).
* **Temperature:** fixed by the chat interface and **not user-settable**, so
  it is not pinned to a number. Reproducibility instead rests on four fixed,
  recorded items: (1) the generator model id, (2) the versioned prompts, (3) the
  exact input `gold_passages.jsonl` (stable passage_ids + exact char spans), and
  (4) the human-validation step. Regeneration may reword questions; the *method*,
  *inputs*, and *validated output* are the reproducible artifacts.
* **Authoring artifact:** `scripts/author_questions_v1.py` holds the authored
  questions as data and re-serialises + validates them to
  `data/derived/questions_candidate.jsonl`.

### 3.2 Constraints (from the prompt)

Persona: a **non-specialist counselor** managing depression with the guideline
(method adapted from Baur et al., RAG patient-education study). Hard rules:
grounded only in the passage text, no answer leakage, depression-specific,
natural phrasing, **Bahasa Indonesia**, and no forced tags (produce fewer
questions rather than fabricate). Each question carries `passage_ids`,
`question_type` (1 of 7), `difficulty` (factoid / multi_hop / applied_case), and a
`reference_answer` used only for human review, never scored.

### 3.3 The candidate set — 100 questions

| question_type | n | | difficulty | n |
|---|--:|---|---|--:|
| pharmacotherapy_dosing | 19 | | factoid | 61 |
| psychoeducation | 17 | | multi_hop | 21 |
| risk_suicide_emergency | 16 | | applied_case | 18 |
| referral_criteria | 14 | | | |
| special_populations | 14 | | | |
| symptom_recognition | 12 | | | |
| diagnostic_criteria | 8 | | | |

All 7 taxonomy types present (no collapse on safety-critical suicide-risk), all 3
difficulties present, and all **23/23** gold passages covered. Multi-hop questions
list every required passage (≥2); factoid/applied-case list exactly one.

`psychoeducation` is the largest type because `definition_course` and
`risk_factors` passages map there (no exact taxonomy bucket exists for them); easy
to rebalance during validation.

---

## 4. Validation (Step 3) — your workflow

A human reviews `questions_candidate.jsonl` and removes items that are
unanswerable from the document, ambiguous, leak their answer, or are not about
depression; deduplicates near-duplicates; fixes tags. Save the kept/edited set as
`data/derived/questions_gold.jsonl` and re-check:

```bash
python scripts/validate_questions.py data/derived/questions_gold.jsonl
```

The validator (`src/depression_rag/evaluation/questions.py`) enforces the schema,
that every `passage_id` exists, and that difficulty↔passage-count is consistent;
it prints the taxonomy/difficulty/concept distributions. Target a fixed final size
with balanced coverage (spec: ~80–150 validated; report the actual counts).

---

## 5. Relevance mapping & metrics (Step 4 / §2.6) — next

Relevance is defined at the **source-span level** and mapped to chunks by
char-overlap, identically for every strategy (spec line 241): a retrieved chunk is
relevant to a question if its `[char_start, char_end]` overlaps the gold passage
span by at least a fixed threshold (recommended: chunk contains the answer span,
or token-overlap ≥ 50%). This uses the char offsets stored on every chunk
(§2.3) and lets Recall@k / MRR / nDCG be computed identically across all 56
indexes. Threshold is declared once and kept fixed. (Module not yet built.)

---

## 6. Artifacts & how to reproduce

| file | role |
|---|---|
| `configs/gold_passages.yaml` | segment→concept judgement (auditable) |
| `configs/prompts/qa_generation.md` | generation method (prompts, provenance) |
| `src/depression_rag/evaluation/gold_passages.py` | gold-passage builder + char-invariant check |
| `src/depression_rag/evaluation/questions.py` | question-set validator |
| `scripts/build_gold_passages.py` | build gold passages + coverage report |
| `scripts/author_questions_v1.py` | author + serialise the 100 candidates |
| `scripts/validate_questions.py` | validate any question file |
| `data/derived/gold_passages.jsonl` | 23 relevance units (exact char spans) |
| `data/derived/questions_candidate.jsonl` | 100 LLM-drafted candidates |
| `data/derived/questions_gold.jsonl` | human-validated final set (you produce) |

```bash
# reproduce the gold passages (the questions are not script-generated — see the
# banner at the top of this file; the v1 authoring script has been archived)
python scripts/build_gold_passages.py
python scripts/validate_questions.py data/derived/questions_gold.jsonl
```
