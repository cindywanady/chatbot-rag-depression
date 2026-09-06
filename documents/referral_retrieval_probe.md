# Referral retrieval probe — does `OPSI RUJUKAN` have anything to cite?

> **Update 2026-08-01.** The 50 briefings were regenerated after the system-prompt
> fix recorded in `prompt_fix_20260801.md`. Stages 1–2 (retrieval) are unchanged —
> neither the index nor the questions moved. Stage 3 changed only in citation
> *coverage*: 48/48 referral sections now carry `[n]` markers (was 43/48), but what
> those markers point at did not improve (4 % referral chunks; 2/24 on risk cases).
> The finding below stands as written.

*Run 2026-07-31 against the deployed index `structure-512-0__e5-large__mv` at the
deployed `k = 5`. Every number below is produced by
`scripts/referral_retrieval_probe.py` (`--device cpu --out
outputs/analysis/referral_probe`) and re-derived from the three CSV artifacts,
not quoted from prose. The probe is read-only with respect to the study: it
changes no config, no index, and no generated answer.*

**The finding**: the gold evaluation's `recall@5 = 1.000` on `referral_criteria`
is true and is not transferable. It was measured on clinician-voiced questions
keyed to MI.4-D. On the patient narratives the chatbot actually receives, a
referral chunk reaches the context window in **8 %** of the 50 frozen study cases,
and MI.8 reaches the 24 risk cases **never** — not at k = 5, not at k = 10. The
briefings still write their mandatory `OPSI RUJUKAN` section and still attach
`[n]` markers to it; those markers point at passages that are not referral
guidance. This is a construct-validity gap in the retrieval evaluation, not a
broken index — §3 proves the index is fine.

---

## 1. The question

The deployed system prompt (`configs/chatbot.yaml`, `prompts.counselor_briefing`)
makes referral a **mandatory** section of every briefing:

> Sajikan HANYA empat bagian berikut: … 4) OPSI RUJUKAN sesuai pedoman.

and `prompts.base` rule 5 routes risk cases "sesuai pedoman (**MI.7 dan MI.8**)".

A mandatory section is a standing demand on the retriever: if no referral passage
is in the context window, the generator must either write the section unsupported
or decline it. Nothing else is available to it — rule 1 forbids outside knowledge.

So: does the retriever meet that demand?

## 2. What the gold evaluation already answers — and what it does not

The gold set does test referral retrieval. Six of its 121 questions are typed
`referral_criteria`, and e5-large scores **recall@5 = 1.000** on them
(`multi_vector_fairness_probe.md` §5). That number is real. It is also narrower
than it looks, in two ways that both matter here.

**Every one of the six is keyed to MI.4-D.**

| question | gold passage(s) | unit |
|---|---|---|
| dep_0092 | MI4_0042 | MI.4 |
| dep_0093 | MI4_0042 | MI.4 |
| dep_0094 | MI4_0042 | MI.4 |
| dep_0095 | MI4_0042 | MI.4 |
| dep_0113 | MI4_0042, MI4_0040 | MI.4 |
| dep_0124 | MI4_0042, MI4_0029 | MI.4 |

`MI4_0042` is *Pokok Bahasan D: RUJUKAN KASUS GANGGUAN DEPRESI* — the criteria for
*when* to refer. **MI.8 is never a gold passage anywhere in the set.** MI.8 carries
the referral *procedure* (sistem rujukan, ketentuan umum, tatacara, rujuk balik)
in **11 of the index's 90 chunks** — 12 % of the corpus, and the single largest
block of referral content. Its retrievability has never been measured.

**Every one of the six is clinician-voiced.** "Kapan saya harus merujuk pasien
depresi ke spesialis?", "Apa saja yang harus saya tulis di dalam surat rujukan
pasien depresi?" These read like a doctor querying a manual. The deployed chatbot
does not receive questions like that — see §4.

## 3. Control: the index is not the problem

Stage 1 asks 14 referral-shaped questions written in the same clinician register
as the gold set, but not copied from it (`CONTROL_QUESTIONS` in the probe).

| | k = 5 | k = 10 |
|---|--:|--:|
| ≥1 `referral_criteria` chunk in top-k | **11/14 (79 %)** | 13/14 (93 %) |
| MI.8 present | 8/14 (57 %) | 12/14 (86 %) |
| MI.4-D present | 10/14 (71 %) | 12/14 (86 %) |

Top-1 result is a `referral_criteria` chunk for 9 of 14. Referral content fills
**47 %** of all top-5 slots.

**MI.8 is retrievable.** Ask for referral procedure in the register the corpus is
written in, and MI.8 comes back more than half the time at k = 5. Whatever goes
wrong in §4 is not a missing index, a bad chunking boundary, or an embedding
failure.

## 4. Study: on the real input distribution it collapses

Stage 2 runs the **50 frozen study questions** — `eval_case_assignments.json`,
seed 20260628, the exact items participants receive.

| | k = 5 | k = 10 |
|---|--:|--:|
| ≥1 `referral_criteria` chunk | **4/50 (8 %)** | 14/50 (28 %) |
| MI.8 | 1/50 (2 %) | 3/50 (6 %) |
| MI.4-D | 3/50 (6 %) | 11/50 (22 %) |

Restricted to the **24 risk cases** — the ones where the prompt explicitly demands
MI.7 *and MI.8*:

| | k = 5 | k = 10 |
|---|--:|--:|
| ≥1 `referral_criteria` chunk | 2/24 (8 %) | 4/24 (17 %) |
| **MI.8** | **0/24** | **0/24** |
| MI.4-D | 2/24 (8 %) | 4/24 (17 %) |

**Raising k does not fix the risk cases.** MI.8 is absent from the top 10 for all
24 of them. The documented k ∈ {3, 5} tuning range
(`README_evaluation_design.md` §5c) is not the lever here.

### Why: the register gap

The gold and control questions are short clinical queries. The study questions are
first-person forum narratives.

| | median length | range |
|---|--:|--:|
| control / gold register | 11 words | 9–17 |
| frozen study questions | **87 words** | 30–263 |

> *[Q01, flagged risky: an 80-word first-person narrative describing intermittent
> insomnia lasting up to two weeks, tied to periods of stress. The verbatim text
> is not reproduced here; the study questions are de-identified but not anonymous.]*

A narrative like this is dense in symptom description and contains no referral
vocabulary at all. Under dense retrieval it lands squarely on symptom and
management passages. Top-1 content type bears this out:

| top-1 chunk is… | control | study |
|---|--:|--:|
| `referral_criteria` | 9/14 | 0/50 |
| `clinical_exposition` | 1/14 | 37/50 |
| `risk_suicide` | 4/14 | 9/50 |

On the 24 risk cases, top-5 is split evenly across MI.7 / MI.4 / MI.1 (40 slots
each) with `referral_criteria` taking **2 % of slots**. MI.7's 27 risk and
exposition chunks crowd out the referral block on exactly the queries where the
prompt asks for it most loudly.

## 5. What the generator does with an empty hand

Stage 3 parses the `OPSI RUJUKAN` section out of the 50 already-generated
briefings (`outputs/analysis/study_answers_chatbot.jsonl`) and resolves each `[n]`
back to the chunk that occupied that slot.

- **48/50** briefings write the section — the format instruction holds.
- **43/48** of those attach at least one `[n]` — the citation instruction holds.
- Of the chunks those markers point at:

| cited chunk type | count | share |
|---|--:|--:|
| `clinical_exposition` | 55 | 66 % |
| `risk_suicide` | 18 | 22 % |
| **`referral_criteria`** | **4** | **5 %** |
| `dosage` | 3 | 4 % |
| `case_example` | 2 | 2 % |
| `somatic_symptoms` | 1 | 1 % |

By unit: MI.7 39, MI.4 37, MI.1 6, **MI.8 1**.

**Referral sections supported by an actual referral chunk: 4/48 overall, 2/24 on
risk cases.**

Q01 (risky) is the representative case:

> **4) OPSI RUJUKAN sesuai pedoman**
> * **Rujukan Segera:** Karena adanya risiko bunuh diri dan perilaku melukai diri,
>   penanya harus segera dirujuk ke fasilitas kesehatan yang mampu menangani
>   kegawatdaruratan psikiatri **[1]**.

The clinical advice is defensible and the guideline would support it. But `[1]` is
an MI.7 chunk on *penilaian kegawatdaruratan* — assessment, not referral. The
marker asserts support the passage does not provide.

This is the model behaving correctly under a contradictory instruction: told to
produce a referral section, told to cite, given no referral passage. It does not
refuse and it does not invent free-floating content — it grounds the section in
the nearest thing it was handed.

## 6. Consequence for the human evaluation

Four Tahap-2 items sit directly on this behaviour
(`INTERNAL_catatan_item_psikolog.md`, briefing deck slide 19):

| group | item | what it will register |
|---|---|---|
| Kesetiaan konteks | **Ketepatan rujukan [n]** | a marker whose passage does not support the sentence — the dominant pattern here |
| Kesetiaan konteks | **Dukungan konteks** | referral claims with no basis in the five excerpts |
| Kesetiaan konteks | **Kecukupan konteks** | the honest reading: the context was insufficient for a mandatory section |
| Keselamatan | **Rujukan** | likely scored *higher* — the advice is usually clinically right |

These items are asked only on the **full form**: 14 of the 36 P1 cases, which each
rater meets as 11 of their 22 packets (the 8 double-rated cases carry the full form
and count for both raters). So the faithfulness items are evaluated on 14 distinct
briefings, 8 of them twice.

Of those 14 full-form briefings, the probe finds:

| | |
|---|--:|
| has a `referral_criteria` chunk in top-5 | **1/14** |
| referral section supported by a referral chunk | **1/14** |

**13 of the 14 briefings that will be scored on `Ketepatan rujukan [n]` carry a
referral citation that its passage does not support.** The items are not sampling
an occasional defect; they are sampling a constant.

Two predictions worth recording **before** raters start, so they are confirmations
rather than post-hoc explanations:

1. `Ketepatan rujukan [n]` and `Dukungan konteks` should score systematically
   lower than `Kebenaran klinis` on the same briefings — the split the psychologist
   briefing spends a whole slide on (deck slide 20, the red box).
2. The effect should be **structural, not case-specific**: near-uniform across
   packets rather than concentrated in a few.

If instead these items come back with high variance across packets, that is
evidence the raters are responding to something other than this mechanism, and the
scores should be re-examined before interpretation.

A consequence for inter-rater agreement: with 13/14 briefings sharing the same
defect, `Ketepatan rujukan [n]` has almost no true variance to agree about. A low
κ on that item would reflect the restricted range, not rater disagreement — it
should not be read as instrument failure.

**This does not invalidate the study.** The counselor comparison (Jalur 1) does not
depend on it, and the instrument is measuring a real property of the artifact,
which is what Tahap 2 is for. It does mean the Tahap-2 faithfulness scores should
be read as *"the briefing's referral section is under-grounded on this input
distribution"* — a finding about the deployed configuration — rather than as a
general statement about the model's citation discipline, which §3 shows is sound
when referral passages are present.

## 7. Safety path: a separate, unrelated exposure

Surfaced while probing, worth recording because it is operational rather than
analytical.

The fixed escalation template (`safety.safe_response` — IGD/puskesmas, then
Healing119 on 119 ext. 8) is prepended deterministically at `chatbot.py:451`, so
the counselor always receives the emergency route regardless of what retrieval
did. **The retrieval gap above does not touch the crisis banner.**

But of the 24 risk cases, only **11 are caught by the keyword screen**; the other
**13 are caught only by the LLM classifier** (`risk_reason` in
`study_answers_chatbot.jsonl`: 11 "kata kunci", 13 "klasifikasi LLM"). That stage
is conditional:

```python
if not matched and self.generator is not None and cfg.llm_check:   # chatbot.py:393
```

If the vLLM server is unreachable the app falls back to dry-run, `self.generator`
is `None`, and those **13 of 24 risk cases silently receive no safety banner**.
Counselors would not see anything missing. Verify the server is up before each
counselor session.

## 8. What this does and does not license

**Does.** Referral content is retrievable (§3) but is not retrieved on the study's
input distribution (§4), and the resulting citations are unsupported (§5). All
three are measured on the deployed index at the deployed k, over the frozen study
items and the already-generated briefings.

**Does not.**
- It does not evaluate whether the referral *advice* is clinically correct — only
  whether the cited passage supports it. Q01's advice is sound; its citation is not.
- The 14 control questions are the probe author's, not a validated instrument.
  They establish an existence proof, not a rate.
- It says nothing about other chunk configs or other k values beyond the k = 5 / 10
  pair reported.
- MI.8's 0/24 is on these 24 risk cases. It is not a claim that MI.8 is
  unreachable for any patient narrative.

**Open, deliberately not acted on.** Whether to change anything is a protocol
decision, not a probe finding. The options and their costs, for the record:
raising k does not help risk cases (§4); softening section 4 to conditional would
change the instrument mid-study; adding MI.8 gold questions would measure the gap
but not close it; retrieval hints or content-type-aware retrieval would change the
deployed configuration after selection. **No change has been made.**

## 9. Reproducing

```bash
./.venv/bin/python scripts/referral_retrieval_probe.py --device cpu \
    --out outputs/analysis/referral_probe
```

≈ 3 min on CPU. Stage 3 alone needs no embedder and runs in seconds:

```bash
./.venv/bin/python scripts/referral_retrieval_probe.py --stage briefings
```

Artifacts: `control_questions.csv` (14 rows), `study_questions.csv` (50 rows, with
`risky`), `briefing_citations.csv` (50 rows). Every table above is re-derivable
from these three files.
