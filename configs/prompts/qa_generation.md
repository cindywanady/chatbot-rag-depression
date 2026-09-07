# Gold-question generation prompt (spec §2.5 Step 2)

**Version:** `qa-gen-v2` (2026-07-01) — single self-contained prompt (no system/rule split).
**Generator:** any Claude/GPT model run via its web UI — **record the exact model
name shown at generation time** as provenance.
**Input:** `data/derived/gold_passages.jsonl` (32 gold passages, spec §2.5 Step 1).
Attach this file, and **only** this file.
**Output (candidates):** `data/derived/questions_candidate.jsonl`.
**Method reference:** adapted from Baur et al., *Development and Evaluation of a
Retrieval-Augmented Generation Chatbot for Orthopedic and Trauma Surgery Patient
Education* — questions synthesized from source clinical text, constrained to a
realistic non-specialist user, then human-validated.

## How to use

1. Paste **the prompt below** into Claude/GPT web (or set it as a Project's custom
   instructions) and **attach `data/derived/gold_passages.jsonl`** — only that file.
   Do NOT attach the guideline PDF, `cleaned_text.txt`, or this file: that breaks
   the "answerable only from the passage" grounding.
2. Save the model's output as `data/derived/questions_candidate.jsonl`.
3. Run `python scripts/validate_questions.py data/derived/questions_candidate.jsonl`
   — it prints the count and checks schema/passage-ids. **If the count is below
   100, reply `continue`** (or run per source unit MI.1 / MI.4 / MI.7) and append
   until it is >= 100.

## Prompt provenance & design basis

*Retrospective justification — not a record of a meta-prompt.* There is **no single
prompt that generated this prompt.** It is a **hand-designed instrument**, drafted
with AI assistance and refined by the author by operationalizing the evaluation
requirements into explicit rules, then frozen verbatim and versioned here so the
exact instruction is auditable (like a coding codebook in qualitative research).

**Design inputs (the basis, not a generating prompt):**
- **the retrieval evaluation requirements** (question types, grounding rules);
- **Baur et al.** (method reference) — synthesize from source text, constrain to a
  realistic non-specialist user, then human-validate;
- **Author constraints** — non-specialist-counselor persona, remove FKTP framing,
  reduce overfitting, *do not hallucinate*, output in Bahasa Indonesia.

**Rule → source traceability:**

| Prompt element | Traces to |
|---|---|
| GROUNDED ONLY (answerable from the passage) | spec §2.5 line 226 (generate from source; relevance at passage level), line 241 (overlap relevance) |
| NO ANSWER LEAKAGE | spec §2.5 line 237 |
| DEPRESSION-SPECIFIC | spec §2.5 lines 224/226/237 |
| NATURAL / counselor persona (FKTP removed) | Baur et al. + author constraint |
| LANGUAGE = Bahasa Indonesia | author constraint + source language |
| NO HALLUCINATION / no padding | author constraint + spec line 230 |
| TAXONOMY (8 question_types) | spec §2.5 line 232 (7) + `differential_comorbidity` from the gold concept (Option B) |
| DIFFICULTY (factoid / multi_hop / applied_case) | spec §2.5 line 233 |
| Output fields (passage_ids, question_type, difficulty, reference_answer) | spec §2.5 line 235 |
| Set size (>=100 candidates; ~80-150 validated) | spec §2.5 line 237 + generation decision |

**Process:** draft → review ("remove FKTP", "don't overfit", "ensure >=100") →
revise → freeze + version. AI assistance in drafting is disclosed; the design
decisions are traceable to the table above.

## Reproducibility & temperature note

The generator is whichever Claude/GPT web model you run the prompt on; sampling
temperature is not user-settable there, so it is not pinned. Reproducibility rests
on four fixed, recorded items:

1. the generator model name (recorded at generation time),
2. **this prompt**, verbatim, under the version tag above,
3. the exact input — `gold_passages.jsonl`, whose passages carry stable
   `passage_id`s and exact char spans into `cleaned_text.txt`,
4. the human-validation step (spec §2.5 Step 3) that produces the final gold set.

Regeneration may yield differently-worded questions; the *method*, *inputs*, and
*validated output* are the reproducible artifacts, not token-level identity.

---

## The prompt (single, self-contained — paste directly)

```text
You are a clinical content specialist building an evaluation dataset for a
depression-information retrieval system, from an Indonesian Ministry of Health
mental-health guideline for general practitioners. Write realistic
information-seeking questions that a non-specialist counselor would genuinely ask
while managing a patient with depression.

INPUT: The attached file `gold_passages.jsonl` contains the source passages, one
JSON object per line, each with: passage_id, concept, heading_path, char_start,
char_end, and text. The passage `text` is the ONLY source you may use.

HARD RULES (override everything else):
1. GROUNDED ONLY. Every question must be answerable using ONLY the cited passage
   text(s). Never use outside medical knowledge; never invent drugs, doses,
   criteria, or referral rules not in the passage.
2. NO ANSWER LEAKAGE. The question must not contain its own answer (do not embed
   the exact dose, cut-off, or verbatim criteria). Ask for the fact; don't state it.
3. DEPRESSION-SPECIFIC. Every question must be about depression as covered by the
   guideline. Discard anything generic or off-topic.
4. NATURAL. Phrase questions the way a non-specialist counselor would actually ask
   them — practical and concrete, not exam-style trivia.
5. LANGUAGE. Write every question and reference_answer in Bahasa Indonesia.
6. NO HALLUCINATION / NO PADDING. If a passage cannot honestly support more
   questions without leaking or repeating, draw on other passages instead — but do
   NOT fabricate or duplicate to hit the count.

HOW MANY: Produce AT LEAST 100 questions in total (target 100; never fewer than
100). There are 32 passages, so this is about 3 per passage. Allocate ~3 per
passage; give richer passages (pharmacotherapy, suicide, diagnosis, differential,
definition) 4-5, and short passages (main_symptoms, additional_symptoms,
risk_factors) 2-3 — but the total MUST reach 100. Cover the taxonomy and difficulty
axes broadly. Before you finish, COUNT the objects you have written: if there are
fewer than 100, keep generating from the richer passages until you have at least
100. If a single reply cannot hold all 100, continue in the next reply until all
are produced — do not stop at a partial set.

TAXONOMY (assign exactly one `question_type` per question):
- diagnostic_criteria
- symptom_recognition            # includes somatic/physical presentation
- pharmacotherapy_dosing         # drug choice, dosing, titration
- referral_criteria
- risk_suicide_emergency         # risk assessment, self-harm, emergency handling
- psychoeducation
- special_populations            # postpartum, elderly, comorbid chronic illness
- differential_comorbidity       # bipolar/psychotic differential; comorbid-illness treatment cautions
Mapping help (gold concept -> question_type): definition_course / risk_factors ->
psychoeducation or symptom_recognition; somatic_presentation / main_symptoms /
additional_symptoms -> symptom_recognition or diagnostic_criteria; everything else
maps to its same-named question_type.

DIFFICULTY (assign exactly one `difficulty` per question):
- factoid       : answerable from ONE passage.
- multi_hop     : needs >=2 passages; list ALL required passage_ids.
- applied_case  : a short realistic vignette where the counselor must map a
                  presentation to the guideline; keep the vignette free of the
                  literal answer.
Aim for a mix: majority factoid, plus a solid share of multi_hop and applied_case.

CONSTRUCTION:
- factoid/applied_case: passage_ids has exactly 1 id; multi_hop: >=2 ids.
- Only use passage_ids that actually appear in the attached file.
- reference_answer: 1-2 sentences in Bahasa Indonesia, drawn strictly from the
  cited passage(s); used only for human review, never scored.
- Avoid near-duplicate questions.

OUTPUT: Return ONLY JSON Lines (one JSON object per line), no prose, no code
fences, this exact schema:
{"question_id": "...", "question": "...", "question_type": "...",
 "difficulty": "factoid|multi_hop|applied_case", "passage_ids": ["..."],
 "reference_answer": "..."}
```

---

## Concept → question_type guidance (this corpus)

Gold-passage `concept` (12 values) maps onto 8 `question_type`s (the spec's 7 plus
`differential_comorbidity`, added as its own type — Option B). When a concept has
no exact taxonomy match, use the closest realistic type:

| gold concept | typical question_type(s) |
|---|---|
| definition_course | psychoeducation, symptom_recognition |
| risk_factors | psychoeducation, symptom_recognition |
| somatic_presentation | symptom_recognition |
| main_symptoms / additional_symptoms | symptom_recognition, diagnostic_criteria |
| diagnostic_criteria | diagnostic_criteria |
| differential_comorbidity | differential_comorbidity |
| pharmacotherapy_dosing | pharmacotherapy_dosing |
| psychoeducation | psychoeducation |
| referral_criteria | referral_criteria |
| suicide_risk_emergency | risk_suicide_emergency |
| special_populations | special_populations |
