# Gold-passage concept & split annotation prompt (Stage 2) — `gold-concept-v1`

**Purpose.** Using a fixed concept codebook, independently assign the gold-passage
**concept(s)** to each segment and **propose split points** for multi-concept
segments — so the concept mapping AND the splits in `configs/gold_passages.yaml`
are justified by external LLM annotation, not asserted by hand. Run on >=1 LLM;
agreement justifies the design, disagreements are reviewed.

**Stage in the workflow.** Stage 2 of the gold-concept process. The codebook below is the **frozen final concept list** (12) reconciled from
Stage 1 (`gold-concept-induction-v1`, Claude + GPT) and `build_spec.md` line 228:
the spec's 11 + `differential_comorbidity` (added because both inductive runs
surfaced it). See `phase2_5_gold_eval_notes.md` §2.5.

**Generator.** Any Claude/GPT model — record the exact model name as provenance.

**Attach exactly one file:** `data/derived/segments_for_judge.jsonl` (blind:
`segment_id`, `source_unit`, `heading_path`, `n_chars`, `text`). Do NOT attach
`gold_passages.yaml`, `segments.jsonl`, or this file — that would reveal the target
mapping/splits and anchor the annotation.

**After generation:** save the output (e.g. `data/derived/gold_concept_claude.jsonl`,
`..._gpt.jsonl`) and compare to `gold_passages.yaml` to confirm / refine concepts
and splits. If output truncates, reply `continue` or run per source unit.

---

```text
You are a clinical content annotator building the relevance units for evaluating a
depression-information retrieval system, from an Indonesian Ministry of Health
mental-health guideline for general practitioners. For each text segment in the
attached file, decide (a) whether it is an in-scope relevance unit for DEPRESSION
case management, (b) which clinical concept(s) it covers, and (c) if it covers more
than one concept, where it should be split.

INPUT: the attached file `segments_for_judge.jsonl`, one JSON object per line, with
segment_id, source_unit, heading_path, n_chars, text. Use ONLY the segment's text.

SCOPE: in-scope = content useful for managing a patient with DEPRESSION
(definition, symptoms, diagnosis, treatment, referral, suicide-risk handling, or
depression in special populations). Out-of-scope: general non-depression detection,
psychiatric-interview technique, non-depression emergencies (delirium, substance
intoxication/withdrawal, psychotic agitation), reference lists / scaffolding.

CONCEPTS (assign one or more; use these exact names):
- definition_course        : what depression is; course/prognosis; epidemiology /
                             burden / comorbidity; consequences/impact.
- risk_factors             : predisposing/precipitating factors (biological,
                             psychological, life events, drugs) and protective factors.
- somatic_presentation     : physical/somatic symptom presentation; suspecting
                             depression behind physical complaints.
- main_symptoms            : the core/cardinal symptoms ("gejala utama").
- additional_symptoms      : the additional/accessory symptoms ("gejala tambahan").
- diagnostic_criteria      : diagnostic criteria/thresholds and steps to establish
                             the diagnosis.
- pharmacotherapy_dosing   : antidepressant choice, dosing, titration, prescribing
                             precautions, side effects, discontinuation.
- psychoeducation          : psychosocial/psychoeducational intervention; what to
                             tell patient/family; non-pharmacological management.
- referral_criteria        : when/how to refer; referral system; back-referral;
                             referral-letter contents.
- suicide_risk_emergency   : suicide/self-harm risk, warning signs, assessment, or
                             emergency handling.
- special_populations      : depression in pregnancy/postpartum, children/
                             adolescents, elderly, or comorbid chronic illness.
- differential_comorbidity : distinguishing depression from / screening for bipolar,
                             mania, or psychosis, and treatment cautions when a
                             comorbid physical illness is present.

SPLITTING:
- One concept -> list it in `concepts`, leave `splits` empty.
- Two or more distinct concepts in sequence -> list all in `concepts` AND give
  ordered `splits`: one object per concept part, in text order. The FIRST split has
  `anchor: ""` (starts at segment start). Each later split's `anchor` MUST be a
  VERBATIM substring copied exactly from the segment text marking where that part
  begins (a heading or the first words of that part).
- Only split where the text genuinely changes concept; do not over-split.

RULES:
1. Decide from meaning, not isolated keywords. Do not invent concepts a segment
   does not contain.
2. Out-of-scope segments: in_scope "no", concepts [], splits [].
3. Use only the concept names above, spelled exactly. Anchors must be exact
   substrings of the segment text.
4. Keep the rationale to one short clause.

OUTPUT: return ONLY JSON Lines (one object per input segment), no prose, no code
fences, this exact schema:
{"segment_id": "...", "in_scope": "yes|no",
 "concepts": ["..."],
 "splits": [{"concept": "...", "anchor": "..."}],
 "rationale": "<<=1 short clause>"}
```
