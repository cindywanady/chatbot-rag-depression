# QA generation — summary & design decisions

*Companion to `documents/phase2_5_gold_eval_notes.md`. Consolidates the
question-generation design, the methodology choices made, and the rationale behind
them (spec §2.5 Step 2). Written 2026-07-01.*

---

## 1. What this is and where it fits

Phase-1 §2.5 builds a depression-specific evaluation question set used to score the
retrieval bake-off. The pipeline:

```
data/derived/segments.jsonl            (Phase-1 segments)
  + configs/gold_passages.yaml         (curated concept mapping)
  -> data/derived/gold_passages.jsonl  (32 relevance units, 12 concepts)   [Step 1]
  -> data/derived/questions_candidate.jsonl  (>=100 LLM-drafted questions)  [Step 2]
  -> data/derived/questions_gold.jsonl       (human-validated final set)    [Step 3]
```

Questions are generated **from the gold passages**, never from the chunks under
test — so no chunking strategy is favoured (spec line 226).

## 2. Why a gold-passage layer (not raw text)

Retrieval metrics need a **ground-truth target location**, not just a question.
`cleaned_text.txt` is the raw substrate; `gold_passages.jsonl` are labelled, ID'd
slices of it with exact char spans. They give the `(concept, passage_id, char_span)`
scaffolding that lets Recall@k / MRR / nDCG be computed by char-overlap (Step 4).
Generating from gold passages (vs the whole blob) also forces balanced concept
coverage, keeps questions grounded/on-scope, and lets multi-hop questions name the
≥2 passages they need.

`concept` (gold passages) is a **different axis** from `content_type` (segments):
`content_type` drives chunking ("atomic"); `concept` is the evaluation vocabulary.

## 3. The gold passages (32, 12 concepts)

Built deterministically by `scripts/build_gold_passages.py` from
`configs/gold_passages.yaml`; the builder verifies
`cleaned_text[char_start:char_end] == passage.text` for every passage. Whole-segment
passages take the segment span; multi-concept segments are split at literal `anchor`
substrings (e.g. `MI4_0025` → main / additional / criteria).

**Taxonomy justification (both top-down and bottom-up):**
- Top-down: spec line 228 expected concepts.
- Bottom-up: Stage-1 LLM **induction** (`gold-concept-induction-v1`, Claude + GPT,
  blind) independently produced ~14 concepts that mapped onto the spec's 11 — strong
  convergence. Both runs also surfaced a concept the spec omitted,
  **`differential_comorbidity`**, which was therefore added (12th concept, 6
  passages: bipolar/psychotic reminders + comorbid-illness cautions, `MI4_0028-0033`).
- `special_populations` added because the question taxonomy (line 232) needs it.
- No `benzodiazepine` concept: the scope has no benzodiazepine-tapering content.

**Concept assignment & splits (Stage 2).** `gold-concept-v1` was run on Claude and
GPT (blind). The final gold set **follows the Claude Stage-2 annotation**. GPT was
systematically more expansive (34 in-scope vs Claude's 26); the trustworthy signal
was the intersection. Decisions taken:
- `MI4_0022` (def/risk/impact) and `MI4_0025` (main/additional/criteria) splits —
  corroborated by both judges (same anchors).
- `MI4_0034` split into special_populations + pharmacotherapy_dosing.
- Added `MI1_0003`, `MI4_0026`, `MI1_0017` (split), `MI4_0037`.
- **MI.8 referral dropped** (general/administrative, not depression-specific);
  depression referral is covered by `MI4_0042` alone.

## 4. The generation prompt (`qa-gen-v2`)

`configs/prompts/qa_generation.md` — a **single, self-contained** prompt (no
system/rule split). Paste it into a Claude/GPT web UI and **attach only**
`data/derived/gold_passages.jsonl`.

Key properties:
- **Persona:** a non-specialist counselor managing depression (FKTP framing removed
  per review; method adapted from Baur et al.).
- **Hard rules:** grounded only in the passage; no answer leakage; depression-
  specific; natural phrasing; **Bahasa Indonesia**; no hallucination / no padding.
- **Set size:** **at least 100** (target 100) — stated inside the prompt with a
  self-count step and a "don't stop at a partial set / continue" instruction, so a
  model does not stop short. With 32 passages that is ~3 each.
- **Taxonomy (8 question_types — Option B):** the spec's 7 (diagnostic_criteria,
  symptom_recognition, pharmacotherapy_dosing, referral_criteria,
  risk_suicide_emergency, psychoeducation, special_populations) **plus
  `differential_comorbidity`** as its own type, so differential questions get their
  own row in the §2.6 results instead of being folded into diagnosis.
- **Difficulty (3):** factoid (1 passage), multi_hop (≥2 passages, list all),
  applied_case (vignette).
- **Output:** JSONL, one object per line, fields `question_id, question,
  question_type, difficulty, passage_ids, reference_answer`.

**`reference_answer`** is a 1-2 sentence answer drawn strictly from the cited
passage, used **only for human sanity-checking** (Step 3) — never scored, and not a
reference for any reference-free metric (spec line 235). Retrieval metrics use
char-overlap of retrieved chunk spans vs gold spans, not the answer text.

**Multi-hop / ≥2 passage_ids.** A question with ≥2 `passage_ids` must be
`multi_hop`; factoid/applied_case must have exactly 1 (validator-enforced). In
scoring, a chunk is a hit if it overlaps **any** listed gold span; "multi-hop
coverage" additionally checks the top-k covers **all** listed passages.

## 5. Methodology decisions & rationale (the "why"s)

- **Prompt provenance — no meta-prompt.** The prompt was not machine-generated from
  a meta-instruction; it is a hand-designed instrument, drafted with AI assistance
  and refined by the author, traceable to spec §2.5 + Baur + author constraints
  (rule→source table in the prompt md). AI assistance is disclosed.
- **Reproducibility & temperature.** Web sampling temperature is not user-settable,
  so it is not pinned. Reproducibility rests on: the recorded generator model name,
  the verbatim versioned prompt, the fixed input (`gold_passages.jsonl` with stable
  ids/spans), and the human-validation step. Regeneration yields different wording;
  the *method, inputs, and validated output* are the reproducible artifacts — not
  token-level identity. The **validated `questions_gold.jsonl` is the artifact the
  study stands on**, not the raw draft.
- **RAGAS is not used here.** Phase-1 retrieval selection uses deterministic IR
  metrics over gold passages; RAGAS (reference-free) is a Phase-2 tool for the
  deployed chatbot's answers. The gold passages can later be reused as RAGAS
  reference contexts.
- **Generators are not needed in Phase 1.** No Qwen/Gemma — only embedding models +
  FAISS + char-overlap. Gemma-3-12B is the Phase-2 answer generator.
- **Differential as its own question_type (Option B).** Chosen so the safety/diagnosis
  -adjacent "is it bipolar/psychotic?" questions are visible in the results rather
  than hidden inside `diagnostic_criteria`.

## 6. Files

| file | role |
|---|---|
| `configs/prompts/qa_generation.md` | the single `qa-gen-v2` prompt + provenance/reproducibility notes |
| `configs/gold_passages.yaml` | curated concept→segment mapping + splits (Claude Stage-2 result) |
| `configs/prompts/gold_concept_induction.md` | Stage-1 induction prompt (taxonomy justification) |
| `configs/prompts/gold_passage_concept_audit.md` | Stage-2 concept+split annotation prompt |
| `src/depression_rag/evaluation/gold_passages.py` | gold-passage builder + char-invariant check |
| `src/depression_rag/evaluation/questions.py` | question-set validator (schema, passage ids, 8 question_types, difficulty↔count) |
| `scripts/build_gold_passages.py` | build gold passages + coverage report |
| `scripts/validate_questions.py` | validate a question file + print counts/distributions |
| `data/derived/gold_passages.jsonl` | the 32 relevance units (the file to attach) |
| `data/derived/questions_candidate.jsonl` | LLM-drafted candidates (>=100) |
| `data/derived/questions_gold.jsonl` | human-validated final set (you produce) |

## 7. Workflow

```bash
# 1. (if needed) rebuild gold passages after editing configs/gold_passages.yaml
python scripts/build_gold_passages.py

# 2. generate: paste configs/prompts/qa_generation.md prompt into Claude/GPT web,
#    attach data/derived/gold_passages.jsonl, save reply as questions_candidate.jsonl

# 3. validate; if count < 100, reply `continue` in the web chat and append
python scripts/validate_questions.py data/derived/questions_candidate.jsonl

# 4. human-validate -> data/derived/questions_gold.jsonl, then re-validate it
python scripts/validate_questions.py data/derived/questions_gold.jsonl
```

## 8. Status / next

- Gold passages: **32 passages, 12 concepts** (follows Claude Stage-2). Done.
- Prompt: **`qa-gen-v2`**, single block, ≥100 floor, 8 question_types. Done.
- Pending: generate the ≥100 candidates against the 32-passage set (the previous
  `questions_candidate.jsonl` was built from the older 23-passage set and is stale),
  then human-validate to `questions_gold.jsonl`. After that: §2.5 Step 4 relevance
  mapping + §2.6 metrics.
