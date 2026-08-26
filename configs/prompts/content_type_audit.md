# content_type audit prompt (LLM-as-judge) — `content-type-audit-v1`

**Purpose.** Independently re-classify each segment's `content_type` to audit the
rule-based labels (`extraction/segmentation.py` + `configs/pipeline.yaml`). The
judge is **blind**: it never sees the rule label or the keyword rules, so it is a
genuine second rater, not a rubber stamp.

**Granularity.** Audit at the **segment** level (68 segments). `meta_content_type`
on chunks is inherited from the dominant segment, so fixing a segment fixes every
chunk that inherits it.

**Generator.** Any Claude model on claude.ai — record the exact model name shown at
generation time as provenance.

**Attach exactly one file:** `data/derived/segments_for_judge.jsonl` (produced by
`scripts/export_segments_for_judge.py`; fields: segment_id, source_unit,
heading_path, n_chars, text — `content_type` deliberately withheld). Do **not**
attach `segments.jsonl`, `content_type_notes.md`, `pipeline.yaml`, or the
guideline PDF — they would reveal the rule label/rules and anchor the judge.

**After generation:** save the output as `data/derived/content_type_judge.jsonl`
and run:
```
python scripts/compare_content_type.py data/derived/content_type_judge.jsonl
```
which reports agreement, Cohen's kappa, a confusion table, and every disagreement
to inspect. The rule labels are not ground truth; disagreements are candidates for
human review, not automatic edits.

If output truncates, reply `continue` or run per source unit (MI.1, MI.2, MI.4,
MI.7, MI.8) and concatenate.

---

```text
You are a clinical content annotator auditing a dataset built from an Indonesian
Ministry of Health mental-health guideline for general practitioners. For each
text segment in the attached file `segments_for_judge.jsonl`, assign the SINGLE
best-fit content_type from the fixed list below, using your own judgement of what
the segment is primarily about. Judge each segment independently.

INPUT: the attached file, one JSON object per line, with: segment_id,
source_unit, heading_path, n_chars, text. Base your decision on `text` (and
`heading_path` for context).

CONTENT_TYPE CATEGORIES (choose exactly one as the PRIMARY type):
- criteria            : diagnostic criteria / diagnostic checklists or thresholds
                        for depression (main vs additional symptoms with
                        counts/duration, step-by-step diagnosis).
- dosage              : pharmacotherapy dosing detail — specific drugs, doses,
                        titration, administration schedules.
- somatic_symptoms    : the physical/somatic symptom presentation of depression
                        (bodily complaints, somatic symptom profiles/lists).
- risk_suicide        : suicide or self-harm content — risk, warning signs,
                        assessment, or emergency handling.
- referral_criteria   : when or how to refer — referral indications, the referral
                        system, back-referral, referral-letter contents.
- case_example        : a worked clinical case or vignette (e.g. "Belajar Kasus")
                        illustrating application to a patient.
- clinical_exposition : general explanatory/didactic prose that is none of the
                        above — definitions, background, principles, epidemiology,
                        course/impact, or narrative treatment description.

RULES:
1. Decide from the meaning of the segment, not from isolated keywords. Pick the
   PRIMARY type that best describes what the segment is mostly about.
2. If a segment mixes content, choose the dominant type as `content_type` and put
   the next-most-relevant type (or "") in `secondary`.
3. Use ONLY the seven labels above, spelled exactly.
4. Keep the rationale to one short clause.

OUTPUT: return ONLY JSON Lines (one object per input segment), no prose, no code
fences, this exact schema:
{"segment_id": "...", "content_type": "<one of the seven>",
 "secondary": "<one of the seven or empty>", "confidence": "high|medium|low",
 "rationale": "<<=1 short clause>"}
```
