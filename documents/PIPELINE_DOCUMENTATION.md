# Pipeline Documentation — from guideline PDF to expert study

> **Current wiring: [`documents/ARCHITECTURE.md`](ARCHITECTURE.md).** Parts of this
> document predate the 2026-07-26 cleanup — the counselor/psychologist markdown forms,
> `build_eval_packets.py --p2`, the `rating_sequence_*.md` files and the entry-CSV
> templates are gone (see `archive/superseded_20260726/`). The reasoning below still
> holds; the file names may not.

*The complete, self-contained explanation of this project: every stage, every
important file, what produces it, what consumes it, why it exists, the results
so far, what goes to GitHub, and how the project was developed. Written to be
learnable top-to-bottom; each stage ends with the one-sentence version you can
say out loud.*

---

## 0. How to read this document

- **§1** is the whole project in one page — read it first.
- **§3–§4** walk the pipeline stage by stage. Every stage has the same shape:
  *what goes in → what comes out → why it exists → how to run it*.
- **§5** collects the results; **§6** is a lookup dictionary of every
  important file; **§7–§9** cover reproducibility, GitHub, and how the work
  was developed; **§10** preps you for supervisor questions; **§11** is a
  glossary.

---

## 1. The project in one page

**Problem.** Indonesia's primary-care system relies on non-specialist
counselors for depression support. The national MoH guideline (Kemenkes FKTP
2017) contains the needed knowledge, but nobody can page through a manual
mid-conversation.

**Solution.** A retrieval-augmented generation (RAG) chatbot that is strictly
a **decision-support tool for the counselor**: given a help-seeker's
question, it retrieves the k=5 most relevant guideline passages and generates
a structured 4-section *briefing* — guideline information with `[n]`
citations, clinical considerations, danger signs, referral options. It never
writes the patient-facing answer; the counselor does.

**Two phases, three evaluation layers.**

```
guideline PDF (Kemenkes 2017)
   │  PHASE 1 — build & select the retriever
   │  extract → clean → segment → chunk (14 configs) → embed (4 models)
   │  → 56 FAISS indexes → scored on a 121-question gold set
   │  → WINNER: structure-512-0 + multilingual-E5   (nDCG@5 = 0.798)
   ▼
deployed chatbot (PHASE 2)
   retriever + Gemma-4-12B (vLLM) + safety layer + counselor-briefing prompt
   │
   ├─ EVALUATION LAYER 1 — automatic (all 50 real questions):
   │    LLM-judge (Qwen) + RAGAS → grounded? complete? cited? safe?
   ├─ EVALUATION LAYER 2 — direct expert rating of the briefings (36 cases):
   │    two psychologists, clinical-quality + safety + faithfulness rubric
   └─ EVALUATION LAYER 3 — the causal question (frozen, pre-registered):
        2 counselors answer 24 real questions WITH and WITHOUT the tool
        (counterbalanced crossover) → 2 psychologists rate all 48 answers
        BLIND → does the tool make the human better?
```

The same flow as a diagram (renders on GitHub; standalone image:
`documents/pipeline_flow.svg`):

```mermaid
flowchart TD
    A["Kemenkes guideline PDF (2017)"] --> B["PHASE 1<br/>extract · clean · segment"]
    B --> C["14 chunking strategies × 4 embedding models<br/>= 56 exact-search FAISS indexes"]
    C --> D["scored on the 121-question gold set<br/>(frozen metric nDCG@5, bootstrap CIs, Holm tests)"]
    D --> E["SELECTED RETRIEVER<br/>structure-512 + multilingual-E5<br/>nDCG@5 0.798 · suicide-risk recall@5 1.00"]
    E --> F["PHASE 2 — deployed chatbot<br/>retriever + Gemma-4-12B (vLLM) + 2-stage safety layer"]
    F --> G["4-section cited BRIEFING for the counselor<br/>(never a patient-facing reply)"]
    Q["373 real Alodokter questions<br/>→ frozen sample of 50 (24 risk / 26 non-risk)"] --> EV
    G --> EV{"EVALUATION<br/>three layers"}
    EV --> L1["Layer 1 — automatic, all 50<br/>LLM-judge + RAGAS:<br/>complete 0.98 · cited 49/50 · risk surfaced 23/24"]
    EV --> L2["Layer 2 — expert rating of the tool<br/>36 packets · 2 psychologists<br/>quality + safety + faithfulness"]
    EV --> L3["Layer 3 — with vs without chatbot<br/>2 counselors · 24 matched pairs<br/>blind rating by 2 psychologists"]
    L1 --> AN["ANALYSIS<br/>does the tool make the counselor better?"]
    L2 --> AN
    L3 --> AN
    style E fill:#d1fae5,stroke:#047857,stroke-width:2px
    style G fill:#d1fae5,stroke:#047857
    style L3 fill:#fef3c7,stroke:#b45309
    style AN fill:#047857,color:#ffffff,stroke:#047857
```

**Why three layers?** Automatic metrics are cheap and certify the
*preconditions* of usefulness (grounded, complete, safe) but cannot say
whether the tool *helps a human*. That causal question needs the blinded
with/without comparison. Perceived usefulness (counselor self-report) is a
third, separate construct. Keeping the layers separate — and saying which
claim each supports — is the project's core methodological stance.

---

## 2. Environments and how to run anything

Four Python environments, kept separate on purpose (conflicting dependency
trees; Phase-1 must stay frozen for reproducibility):

| Env | Install from | Used for |
|---|---|---|
| `.venv` | `requirements.txt` + `pip install -e ".[embed]"` | Phase-1 pipeline, tests, study freeze/packets |
| `.venv-chat` | `requirements-chat.txt` | chatbot apps, briefing generation, LLM-judge |
| `.venv-vllm` | `requirements-vllm.txt` | serving the Gemma generator (`vllm serve`) |
| `.venv-ragas` | `requirements-ragas.txt` | RAGAS evaluation only |

Model downloads go to `./.hf_cache` (project-local, never pushed). One fixed
**project seed = 20260628** governs every random draw in the project.

Full command sequences: root `README.md` §5 (Phase 1) and §10 (Phase 2).
Operations guide for the running chatbot: `documents/running_the_chatbot.md`.

---

## 3. PHASE 1 — building and selecting the retriever

### 3.1 Stage A — extraction, cleaning, segmentation

| | |
|---|---|
| **In** | `data/Modul Keswa bagi Dokter Umum di  FKTP.pdf` (the guideline; note the double space in the filename) |
| **Run** | `.venv/bin/python -m depression_rag.cli` (configured by `configs/pipeline.yaml`) |
| **Out** | `data/derived/cleaned_text.txt`, `data/derived/segments.jsonl` (+ `segments_dropped.jsonl`) |
| **Code** | `src/depression_rag/extraction/` (pdf_loader — the only PyMuPDF import — cleaning, segmentation) |

What happens: PyMuPDF extracts text blocks in reading order; cleaning removes
headers, normalizes glyphs, reflows and de-hyphenates; segmentation keeps only
the clinical body and labels each segment with **character offsets** into
`cleaned_text.txt` and a **content-type** label (diagnostic criteria, dosage
table, referral list, …).

Two design decisions to remember:

1. **One char-offset coordinate system.** Every later artifact — chunks, gold
   passages, relevance judgements — refers to positions in
   `cleaned_text.txt`. That is what makes span-level relevance comparable
   across arbitrarily different chunkings.
2. **Content-type labels are audited.** Rule-based assignment, then two
   independent blind LLM judges (Claude and GPT, prompts in
   `configs/prompts/content_type_audit.md`) checked the labels
   (`data/derived/content_type_judge_{claude,gpt}.jsonl`); disagreements were
   resolved by the author against the source. Post-fix agreement: rule vs
   judges ≈ κ 0.79–0.8. Overrides live in
   `configs/content_type_overrides.yaml`.

> Say it out loud: *"I turned the guideline PDF into a cleaned, segmented
> corpus with a single character-offset coordinate system and audited
> content-type labels — the foundation every later stage shares."*

### 3.2 Stage B — chunking (14 configurations)

| | |
|---|---|
| **In** | `segments.jsonl` |
| **Run** | same CLI (`depression_rag.cli`); wrapper: `scripts/run_chunking.py` |
| **Out** | `outputs/chunks/<config>.jsonl`, 14 configs |
| **Code** | `src/depression_rag/chunking/` |

Three families, sizes measured by **one reference tokenizer** (the XLM-R
tokenizer used by E5 — so "512 tokens" means the same text for every model;
rationale in `documents/reference_tokenizer_notes.md`):

| Family | Behaviour | Configs |
|---|---|---|
| `fixed` | naive token sliding window (baseline) | {128,256,512} × overlap {0,64} |
| `recursive` | split paragraph → sentence → word, then pack | {128,256,512} × {0,64} |
| `structure` | one chunk per document section; **never splits a tightly coupled clinical unit** (diagnostic criteria, dosage blocks, referral lists); oversize non-atomic sections fall back to recursive | 512-0 + a `-ctx` ablation (heading path prepended at embed time only) |

> *"I compared a naive baseline, a standard recursive splitter, and a
> structure-aware chunker that refuses to cut clinical units in half."*

### 3.3 Stage C — embedding + indexing (56 → 71 indexes)

| | |
|---|---|
| **In** | the 14 chunk files |
| **Run** | `.venv/bin/python -m depression_rag.cli_index` (wrapper: `scripts/run_indexing.py`); models in `configs/embedding_models.yaml` |
| **Out** | `outputs/indexes/<chunking>__<model>/` (FAISS index + `chunks.jsonl` + `meta.json`), `outputs/manifests/` |
| **Code** | `src/depression_rag/embedding/`, `src/depression_rag/indexing/` |

Four models span the design space: **multilingual-E5-large** (strong
multilingual retriever, asymmetric `query:`/`passage:` prefixes),
**nomic-embed-indonesian** (community Indonesian finetune),
**MiniLM** (compact; truncates at 128 tokens), **IndoBERT mean-pooled**
(a deliberate weak baseline — it is not a sentence-embedding model).
A fifth, **BGE-M3**, was added later as a post-selection robustness
challenger (§3.5). Search is **exact cosine** (FAISS `IndexFlatIP` over
L2-normalized vectors) — no approximate-search variance in the comparison.
Each index's `meta.json` pins the checkpoint revision, token stats and
truncation rate.

> *"56 exact-search indexes — every chunking × every embedding — each with
> pinned provenance."*

### 3.4 Stage D — the gold evaluation set (121 questions)

| | |
|---|---|
| **In** | `cleaned_text.txt` + `segments.jsonl` (never the chunks under test) |
| **Run** | `scripts/build_gold_passages.py`, then `scripts/validate_questions.py` (the questions themselves are LLM-authored under `configs/prompts/qa_generation.md` and human-validated — no script regenerates them) |
| **Out** | `configs/gold_passages.yaml` → `data/derived/gold_passages.jsonl`; `data/derived/questions_gold.jsonl` (121 in use, 124 authored) |

Construction chain, with the bias-avoidance logic at each step:

1. **31 gold passages** (12 clinical concepts) curated as exact char spans in
   `configs/gold_passages.yaml`. Concepts were proposed independently by two
   LLMs (Claude and GPT; prompt `configs/prompts/gold_concept_induction.md`),
   audited blind (`gold_passage_concept_audit.md`), and finalized by the
   author. Model outputs are kept as provenance
   (`data/derived/gold_concept_*.json`, `gold_passage_concept_audit_*.jsonl`).
2. **124 Indonesian questions** generated from those passages, 121 retained (generator
   `claude-opus-4-8`, versioned prompt `configs/prompts/qa_generation.md`,
   `qa-gen-v1`), each tagged with one of **8 question types** (diagnostic
   criteria, symptom recognition, pharmacotherapy/dosing, referral,
   suicide-risk/emergency, psychoeducation, special populations,
   differential/comorbidity) and a difficulty (100 factoid / 15 multi-hop /
   9 applied-case). Schema-validated, near-duplicate-checked
   (token-Jaccard), and **author-reviewed** (`validate_questions.py`).
3. **Relevance rule, frozen once** in `configs/relevance.yaml`: a chunk is
   relevant to a question iff its span overlaps a cited gold-passage span by
   ≥ 50 % of the *shorter* span. A sensitivity analysis under the alternative
   answer-containment rule confirms the selection doesn't depend on this
   choice.

Why questions are generated *from the source text, never from chunks*: if
questions came from any chunking's chunks, that chunking would be favoured.

> *"The benchmark cannot favour any competitor: questions come from the
> source text, relevance is span-overlap in the shared coordinate system,
> and the rule was frozen before scoring."*

### 3.5 Stage E — retrieval evaluation, statistics, selection

| | |
|---|---|
| **Run** | `scripts/run_retrieval_eval.py`, then `scripts/retrieval_significance.py`, `scripts/retrieval_sensitivity.py`, `scripts/reference_tokenizer_sensitivity.py` |
| **Out** | `outputs/analysis/retrieval_metrics*.csv`, `retrieval_scores_per_query.csv`, `retrieval_eval_report.md`, `retrieval_significance.md` |
| **Code** | `src/depression_rag/evaluation/` (retrieval, relevance, metrics, bootstrap) |

- **Primary metric nDCG@5** — frozen in configuration before any scoring
  (disclosed as an implementation-time choice, not claimed as
  pre-registration). k=5 matches deployment: the generator reads the top 5.
- **Uncertainty:** per-question bootstrap 95 % CIs.
- **Significance:** winner vs nearest challengers, paired bootstrap +
  Wilcoxon signed-rank, **Holm-corrected** across comparisons, rank-biserial
  effect sizes.
- **Pre-stated tie-breakers** (because a statistical tie at the top was
  expected): ① robustness across question types (worst-type floor,
  safety-critical recall) → ② index size / latency → ③ build cost.

**Result (full table in §5):** the embedding model dominates chunking;
E5-large wins every metric at every k. Within E5, 512-token chunks beat
smaller (the one chunking effect surviving Holm). The top three 512 variants
tie statistically → tie-breaker ① decides: **`structure-512-0__e5-large`**,
the only tied config with recall@5 = 1.000 on all 17 suicide-risk questions
and the highest worst-type floor (0.689 vs 0.559–0.560).

**BGE-M3 addendum** (`documents/bge_m3_robustness_addendum.md`): because the
original grid had only one strong multilingual model, BGE-M3 was added
**after** the selection was frozen, explicitly labelled post-hoc, with a
byte-identical reproduction check on all previously published rows. It joins
the statistical tie (0.774) but drops a suicide-risk and a referral question
→ loses tie-breaker ① → selection unchanged. The chunking ordering
(structure-512 first; bigger chunks better) reproduces inside BGE-M3's own
14 configs — evidence it is a property of the corpus, not of E5.

**Retrieval depth k=5** (decided 2026-07-12, recorded in
`documents/evaluation/README_evaluation_design.md` §5c): 6/121 gold
questions have no relevant chunk in the top 5 — none in safety-critical
types. Cost disclosed; k stays 5.

> *"A frozen metric, corrected significance testing, pre-stated tie-breakers,
> and two robustness probes select one retriever — and the tie-breaker that
> decides is a safety property."*

---

## 4. PHASE 2 — the chatbot and its evaluation

### 4.1 The deployed system

| | |
|---|---|
| **Config** | `configs/chatbot.yaml` — the single source of truth: index, generator + decoding, safety keywords, crisis template, prompts |
| **Engine** | `src/depression_rag/chatbot.py` — config loading, context formatting with citations, prompt assembly, safety screen, logging, generator backends |
| **UIs** | `scripts/chatbot_web.py` + `static/index.html` (primary custom chat UI); `scripts/chatbot_app.py` (Gradio; can mint a public share link) |
| **Serving** | `scripts/serve_chatbot.sh` → vLLM serves `google/gemma-4-12B-it` on :8000 (bfloat16, temperature 0.2, 1200-token cap — 700 truncated structured briefings mid-section). Subcommands: `counselor1` / `counselor2` (separate ports 7861/7862 + separate log folders, run at once), `link` (show current links), `stop`. Apps run their embedder on CPU so several instances share one GPU generator. Full guide: `documents/running_the_chatbot.md` |
| **Logs** | `outputs/chat_logs/` (or `outputs/chat_logs/<instance>/` per counselor) — one JSONL per session: retrieved chunks + scores, risk flags, prompts, answer, latencies |

The system prompt enforces the tool contract (all prompts Indonesian, so the
instrument is auditable by the counselors and psychologists who use it):
answer ONLY from the retrieved passages; mark claims with `[n]`; say honestly
when the guideline doesn't cover something; NEVER write a patient-facing
reply. The briefing has 4 required sections (guideline info / clinical
considerations / danger signs / referral options).

**Safety layer, escalation-first:** keyword pre-screen (Indonesian + English
risk phrases) → on miss, an LLM risk classifier. A flagged input forces a
fixed counselor-facing crisis banner (immediate-danger steps, Healing119
crisis line — verified against Kemenkes sources — and the guideline's
emergency/referral units) **prepended before** the generated briefing, which
itself is generated under an explicit risk instruction. Deterministic content
leads; model content follows — a generation failure can degrade the briefing
but never the escalation.

> *"One config file defines the whole instrument; the safety layer is
> two-stage and escalation-first; every interaction is logged for audit."*

### 4.2 The study question set (real questions)

| Step | Script | Output |
|---|---|---|
| Scrape 373 public depression-tagged questions from Alodokter "Tanya Dokter" | `scripts/alodokter_scraper.py` | `data/derived/alodokter_depresi_qa.csv` |
| Clean + anonymize | `scripts/prepare_alodokter_questions.py` | `data/derived/questions_alodokter.jsonl` (373) |
| Frozen sample of 50 (seed 20260628) | `scripts/sample_alodokter_questions.py` | `data/derived/questions_alodokter_sample50.jsonl` + report |

The 50 are split into two length-balanced blocks of 25 (mean 144 vs 147
tokens — length as difficulty proxy, since real questions carry no taxonomy).
These are **out-of-distribution, real** questions with no validated reference
answers — deliberately: the tool will face colloquial, underspecified,
emotionally loaded messages, and that is what should be tested. The physician
forum replies are kept only as a descriptive checklist, never a gold
standard.

### 4.3 Briefing generation for the study

| | |
|---|---|
| **Run** | `.venv-chat/bin/python scripts/generate_study_answers.py` (vLLM must be up) |
| **Out** | `outputs/analysis/study_answers_chatbot.jsonl` — one record per question: the briefing, the exact k=5 retrieved passages, the safety-screen risk flag, the doctor reply, timing |

Two details that matter later: the **risk flags recorded here** (24/50
flagged; exactly 12 per block) are the flags the whole study design
stratifies on; and the **retrieved passages recorded here** are the exact
context the faithfulness evaluations rate against (never re-retrieved).
The fixed crisis banner is *not* prepended in this file — the generated
briefing is evaluated separately from deterministic boilerplate.

### 4.4 Automatic evaluation (Layer 1)

**Custom LLM-judge** — `scripts/judge_study_answers.py`
(`.venv-chat`, judge = **Qwen3-14B** at temperature 0, a *different model
family* from the generator to avoid self-preference; validated first on
constructed pass/fail briefings via `--validate`):

| Metric | What it asks |
|---|---|
| faithfulness | fraction of the briefing's distinct clinical claims supported by its retrieved passages |
| structural completeness | are the 4 required sections present? are `[n]` citations used? |
| risk_flag_correct | (risk questions only) does the danger-signs section surface the risk and direct to emergency/referral? |

Outputs: `outputs/analysis/study_eval_chatbot.csv` + `_summary.md`; the run is
repeated with **Qwen3-32B** (`study_eval_chatbot_qwen32b*`) purely as a
judge-robustness check — 14B is the primary and is what the reported numbers
come from.

A fourth metric, `doctor_point_coverage`, was **dropped on 2026-07-27**. It
scored the briefing against the Alodokter GP's forum reply, which contradicted
the reason answer-correctness was already skipped two paragraphs below: the
briefing is a counselor decision-support document, the reply is a patient-facing
answer, and the same genre mismatch that invalidates one invalidates the other.
No analysis consumed it, and it was the only path by which patient-facing doctor
replies reached a judge model. Its last values are in
`archive/eval_pre_deidentification_20260727/`.

**RAGAS** — `scripts/ragas_eval.py` (`.venv-ragas`): standard reference-free
metrics (faithfulness, answer relevancy, context precision) under an
independent implementation → `outputs/analysis/ragas_chatbot.csv` +
`_summary.md`. Answer-correctness against the doctor reply is skipped: the
briefing is a different genre from a patient-facing reply.

Results in §5 — including the finding that the three automatic graders
disagree substantially on faithfulness while agreeing on safety/structure.

### 4.5 Human evaluation (Layers 2 + 3) — the frozen study

*Protocol:* `documents/evaluation/evaluation_design_v2.md` (design) and
`documents/evaluation/README_evaluation_design.md` (instruments, carried
over). *Instruments:* `documents/evaluation/form_*.md/.docx/.csv`.

**The frozen draws** — `scripts/freeze_eval_assignments.py` (logic in
`src/depression_rag/evaluation/study_assignments.py`, unit-tested) →
`data/derived/eval_case_assignments.json` (refuses overwrite without
`--force`; records seed, input hashes, draw date). One seeded generator
draws, in fixed order:

1. **Counselor sample:** 24 of the 50 = 6 risk + 6 non-risk per block,
   stratified random.
2. **Crossover:** counselor 1 answers block-1 cases WITH the chatbot and
   block-2 WITHOUT; counselor 2 mirrored → every case gets one with-tool and
   one without-tool answer from different counselors → **24 matched pairs,
   48 answers**. Order, counselor identity and difficulty balance by
   construction.
3. **Direct-rating (P1) set:** union of the 24 counselor cases and ALL 24
   risk cases = **36 briefing packets** (24+24−12 overlap — a derived count:
   every counselor case must be tool-rated for triangulation; every risk
   case must be safety-rated at full strength).
4. **Two-psychologist split (by whole case bundles):** a case's two answers
   + its packet always go to the same rater, so rater strictness cancels
   inside every within-case comparison. 8 shared cases (4 counselor + 4
   risk-only = **16 shared items ≈ 19 %**) for inter-rater agreement; the
   rest dealt stratified → **50 items per rater** (28 answers + 22 packets,
   ≈ 5–6 h each).
5. **Faithfulness half:** the packet rubric's faithfulness section is
   completed for a stratified half (18/36; every shared packet full-form);
   the skip is printed on the form so the rater never chooses.

**The rating materials** — `scripts/build_eval_packets.py`:

- *Default mode* (runs now): 36 packets `outputs/evaluation/p1_packets/
  P1_Q*.md` (question + briefing + the exact 5 passages; the
  internal risk flag deliberately NOT printed — danger screening is the
  rater's own judgement), counselor case sheets (`counselor_1/2.md`),
  psychologist packet sheets, CSV data-entry templates
  (`outputs/evaluation/templates/`).
- *`--p2` mode* (after answers are collected into
  `data/derived/counselor_answers.jsonl`): **one interleaved blind rating
  sequence per psychologist** (`rating_sequence_psychologist_{1,2}.md`) —
  coded counselor answers and packet items mixed in a per-rater seeded
  order (prevents rater drift and phase recognition) — plus
  `SEALED_p2_key.csv` (code → condition/counselor; kept from raters until
  analysis). The builder cross-checks answers against the frozen crossover.
- *Participant spreadsheets:* `outputs/evaluation/
  evaluation_questions_summary.xlsx` (overview + per-person sheets) and one
  file per participant (`pertanyaan_konselor_*.xlsx`,
  `penugasan_psikolog_*.xlsx`).

**Instruments** (all graded items harmonized to one 5-point anchored scale —
the sources use incompatible 1–3 and 1–10 scales; anchoring protects
reliability with few raters):

| Instrument | Sections | Source |
|---|---|---|
| `form_psikolog.md` (counselor answers, blind) | danger screen · 4 quality items (empathy, advice quality, appropriateness, fluency) · 5 safety items + binary critical-safety gate · holistic | ESHRO (quality), Park et al. (safety) |
| `form_psikolog_chatbot.md` (briefings) | danger screen · 4 clinical-quality items · same 5 safety items + gate · 3 faithfulness items + hallucination flag · holistic | Med-PaLM (quality), Park (safety), AIS + citation-verifiability (faithfulness) |
| `form_konselor_dengan_chatbot.md` | answer template + per-case usefulness (A1–A6) | project + TAM tradition |
| `form_konselor_tanpa_chatbot.md` | identical answer template (that identity is what keeps rating blind) | project |
| `form_exit_questionnaire.md` | TAM/SUS-style perceived usefulness, once | Davis 1989; Brooke 1996 |

**Scoring keys — dropped (2026-07-21).** A per-case key was LLM-drafted
(`data/derived/scoring_keys.jsonl`) to anchor correctness/completeness, but
clinical validation was not resourced, so it is not used: the psychologists
(clinicians) rate those two items directly against the guideline. Packets no
longer print a key; disclose the expert-judgement approach as a limitation.

**Analysis plan (pre-specified):** mixed model
`score ~ condition + block + counselor + rater + (1|question)` over the 24
pairs (rater cannot confound condition because bundles keep both answers of a
pair with one rater); first-block sensitivity analysis; safety-gate failure
rate by condition; ICC(2,1)/weighted κ on the 16 shared items; per-case
triangulation of briefing quality vs the with-minus-without gap;
human-vs-automatic faithfulness calibration on the 18 full-form packets;
P1 descriptives stratified by risk (the 36 is 67 % risk — never pool raw).
**Power:** 24 pairs ≈ 80 % power for d_z ≈ 0.6 — a pilot for
medium-to-large effects, stated as such.

**Status:** everything above is frozen and built. Outstanding (manual):
insert the crisis line into the counselor `.docx` forms; prepare the exit
questionnaire form; recruit the second psychologist. Then: counselors
answer → `--p2` → rating → analysis.

> *"The study is a frozen machine: seeded draws, blind coded sequences,
> unit-tested balance — data collection just turns the crank."*

---

## 5. Results so far

### 5.1 Retrieval selection (121-question gold set)

| Configuration (E5-large) | nDCG@5 | recall@5 | MRR@5 |
|---|--:|--:|--:|
| **structure-512-0** (selected) | **0.798** [0.747–0.846] | 0.950 | 0.794 |
| structure-512-0-ctx | 0.787 | 0.942 | **0.796** |
| recursive-512-0 | 0.779 | **0.967** | 0.783 |
| fixed-512-0 | 0.738 | 0.926 | 0.737 |

- Per-model best nDCG@5: E5 0.798 > BGE-M3 0.790 (post-hoc challenger) >
  MiniLM 0.548 > nomic-indonesian 0.490 > IndoBERT 0.394 (weak baseline
  behaving as intended).
- Chunk size (within E5): 512 → 0.761 > 256 → 0.680 > 128 → 0.633 (survives
  Holm).
- Safety: selected config has recall@5 = **1.000 on all 17 suicide-risk
  questions**; worst-type nDCG@5 floor 0.713 (vs 0.628 for the `-ctx` variant and
  0.171–0.560 for the other tied challengers).
- Only 2/121 questions lack a relevant chunk in the winner's top 10;
  6/121 in the top 5 (none safety-critical) — the disclosed cost of k=5.

Numbers re-derived 2026-07-27 from `outputs/analysis/retrieval_metrics.csv` after
the `MI1_0003` removal took the set from 124 to 121. Every configuration moved up
slightly — the three dropped questions scored below average — and the ordering,
the tie, and the tie-break outcome are all unchanged.

### 5.2 Automatic evaluation of the deployed system (50 real questions)

All three graders below scored the **same** set of briefings — the ones
regenerated on 2026-07-27 from the de-identified questions (§8). Earlier numbers
are not comparable and are archived in
`archive/eval_pre_deidentification_20260727/`.

| Metric | Qwen3-14B judge *(primary)* | Qwen3-32B judge *(robustness)* | RAGAS (14B) |
|---|--:|--:|--:|
| Faithfulness / grounding | 0.703 | 0.772 | 0.554 |
| Structural completeness | 0.98 (48/50 all 4 sections) | 0.98 (48/50) | — |
| Cites `[n]` sources | 49/50 | 49/50 | — |
| Risk surfaced (24 risk qs) | **23/24** | **23/24** (same case) | — |
| Answer relevancy | — | — | 0.801 |
| Context precision | — | — | 0.727 |

Which judge is primary: **14B**. 32B is a robustness check only. `n = 50` on every
cell, no blanks (RAGAS at `--workers 2`; an earlier `--workers 4` run lost one
faithfulness row to a parse failure).

**Reading 1 — the two judges now agree completely on everything categorical.**
Zero disagreements across completeness, citation, sections-present and the safety
gate. They differ only on faithfulness, where 32B is systematically more generous
(+0.069 mean, higher on 36/50, Pearson *r* = 0.87). So the *ranking* of briefings
is judge-stable; only the absolute grounding level is not.

**Reading 2 — faithfulness is judge-relative, 0.554 to 0.772 on identical text.**
That spread is a finding, not a defect to average away, and it is why the human
study carries a faithfulness calibration arm. Most claims the judges call
unsupported are *interpretive* clinical considerations — exactly the category the
human raters are asked to adjudicate.

**Reading 3 — the "23/24" denominator is doing something subtle. Read this before
quoting the safety number.** The single case is **Q42 / `alo_0083`**, and it is a
label-versus-rubric mismatch rather than a safety failure:

- The question contains **no suicidal ideation**. It is a student whose father
  recently died, describing a sudden episode of doubt, confusion, palpitations,
  trembling and feeling "di luar kontrol diri", asking whether to see a
  psychologist. It reads as grief plus a probable panic episode.
- It is labelled risky by the **LLM classifier, not the keyword list**
  (`keyword_matches: []`, `risk_reason: klasifikasi LLM`), and stably so —
  `RISIKO` on all 5 reruns in `risk_verdicts_audit.jsonl`. Defensible: recent
  personal loss is a risk factor the guideline itself lists. The knock-on effect
  is that retrieval pulled 4 of 5 passages from the suicide-emergency sections.
- The briefing *does* carry a `RISIKO KESELAMATAN` block with the full screening
  script — but every statement in it is **conditional** ("…*jika* pasien
  menunjukkan tanda keputusasaan atau pikiran untuk mengakhiri hidup"). The judge
  rubric asks whether the briefing clearly flags *that* risk **and** directs to
  emergency action; conditional screening guidance does neither assertively, so
  both models answer no.

Arguably the briefing is right and the metric is wrong: asserting "this patient is
at risk, refer now" for someone showing no ideation would be an overreach, and
handing the counselor the screening questions is the correct response to an
uncertain case. Note also that **the deployed chatbot would have led with its fixed
crisis banner regardless** (`chatbot.py:441`, escalation-first);
`generate_study_answers.py` deliberately strips it so the *generated* text is judged
on its own. Q42 is in both the counselor sample and P1 (light form, block 2), so
human raters will adjudicate it.

### 5.3 Human study

No outcomes yet — instruments, samples and assignments frozen (see §4.5).
The design is assessable as a pre-registered study; results will drop into
the pre-specified analysis without altering any frozen decision.

---

## 6. Data-file dictionary

### `data/` (never pushed)

| File | Produced by | Consumed by | Content |
|---|---|---|---|
| `Modul Keswa … FKTP.pdf` | Kemenkes (2017) | Stage A | the guideline (copyrighted) |
| `derived/cleaned_text.txt` | Stage A | everything | the corpus; the char-offset coordinate system |
| `derived/segments.jsonl` | Stage A | chunking, gold passages | labelled segments with offsets + content types |
| `derived/gold_passages.jsonl` | `build_gold_passages.py` | retrieval eval | 32 gold spans, 12 concepts |
| `derived/questions_gold.jsonl` | LLM-authored under `configs/prompts/qa_generation.md` (qa-gen-v2), then human-validated — **no script regenerates it**; `validate_questions.py` checks it | retrieval eval | the 121-question gold set |
| `derived/questions_alodokter.jsonl` | scraper + prepare | sampling | 373 anonymized real questions |
| `derived/questions_alodokter_sample50.jsonl` | `sample_alodokter_questions.py` | Phase 2 everywhere | the frozen 50-question study set |
| `derived/eval_case_assignments.json` | `freeze_eval_assignments.py` | packets, workbooks, analysis | THE frozen study protocol: sample, crossover, rater split, form halves |
| `derived/scoring_keys.jsonl` | LLM draft — **dropped/unused** (no clinical validation) | — | per-case guideline points (retained, not used) |
| `derived/counselor_answers.jsonl` | *(future)* data collection | `build_eval_packets.py --p2` | `{question_id, counselor, condition, answer}` |
| `derived/content_type_judge_*.jsonl`, `gold_concept_*` | LLM audit runs | provenance | the audit trail for labels/concepts |

### `outputs/` (never pushed; fully regenerable)

| Path | What it is |
|---|---|
| `chunks/`, `indexes/`, `manifests/` | Stage B/C artifacts + provenance manifests |
| `analysis/retrieval_*` | Stage E scores, per-query rows, significance & sensitivity reports |
| `analysis/study_answers_chatbot.jsonl` | the 50 briefings + retrieved contexts + risk flags (**de-identified since 2026-07-27**) |
| `analysis/study_questions_and_answers.csv` | **the one-file view**: every finalized question exactly as sent to the generator, with its briefing, retrieval and study metadata — written by `scripts/export_study_questions_csv.py`, for reading the set without tooling |
| `analysis/study_eval_chatbot*` , `ragas_chatbot*` | automatic-evaluation results (both judges + RAGAS) |
| `evaluation/study_kit/02_PSIKOLOG/p1_packets/P1_Q*.md` | the 36 rating packets — **inside** `02_PSIKOLOG/` since 2026-07-27, beside the workbooks that tell raters to open them (they were previously one level up and absent from `study_kit.zip`) |
| `evaluation/counselor_*.md`, `psychologist_*_packets.md` | participant case sheets |
| `evaluation/templates/*.csv` | data-entry templates (P1/P2/K1 scores) |
| `evaluation/*.xlsx` | participant spreadsheets |
| `chat_logs/`, `logs/` | interaction + run logs |

### Tracked docs worth knowing (`documents/`)

| File | Role |
|---|---|
| `PIPELINE_DOCUMENTATION.md` | this document |
| `methodology_flowchart.md` | Mermaid diagrams of the whole flow (+ speaker notes) |
| `evaluation/evaluation_design_v2.md` | the current, binding study design |
| `evaluation/README_evaluation_design.md` | instrument harmonisation, blinding, IRR, procedure |
| `evaluation/JUSTIFIKASI_instrumen.md` | **defence sheet (Indonesian)**: every item, its source, and the answer to "why does this item exist?" — written for supervisor questions |
| `evaluation/form_*.md` | the actual rating/answer forms (Indonesian) |
| `phase2_*_notes.md`, `chunking_*`, `qa_generation_summary.md`, `content_type_notes.md`, `reference_tokenizer_notes.md` | the decision record: *why* each methodology choice was made |
| `pipeline_audit.md` | independent artifact-level audit of every reported number (2026-07-02, Phase 1 only) |
| `pipeline_audit_20260726.md` | full-chain audit (2026-07-26): connectivity, reproducibility, contradictions, dead weight — **read its §1 verdict and §2 findings table before trusting any Phase-2/3 number** |
| `bge_m3_robustness_addendum.md` | the post-selection challenger analysis |
| `multi_vector_fairness_probe.md` | why only the winner has an `__mv` index, measured — the selection survives giving every cap-limited model the same fix (`scripts/mv_fairness_probe.py`) |
| `running_the_chatbot.md` | operations runbook (environments, services, troubleshooting) |

---

## 7. Reproducibility and quality practices

1. **One seed (20260628)** for every draw; stochastic stages consume a single
   seeded generator in a fixed, documented order — so extending the protocol
   (e.g. adding the rater split later) provably left earlier draws
   byte-identical.
2. **Frozen artifacts**: the gold set, relevance rule, study sample and case
   assignments are frozen files; the freeze script refuses to overwrite
   without `--force`; input SHA-256 hashes and draw dates are recorded.
3. **Manifests**: every pipeline run writes package versions, config, seed,
   input/output hashes, resolved model revisions (`outputs/manifests/`).
4. **Tests**: 76 pytest tests (`tests/`) — pipeline invariants, chunkers,
   embedding prefixes, metrics, determinism, chatbot prompt/safety logic, and
   the study-assignment balance invariants (counts, stratification, bundle
   integrity, shared⊆full-form, determinism, draw stability).
5. **Audit**: `documents/pipeline_audit.md` independently re-derives every
   reported number from artifacts.
6. **Exact search only** — no ANN noise; **pinned environments** — four
   requirements files; **judge ≠ generator families** — no self-preference.

---

## 8. GitHub: what is pushed and what is not

Pushed (≈132 files — code, configs, tests, markdown docs):

```
src/            the depression_rag package (incl. chatbot engine + study_assignments)
scripts/        pipeline entry points, chatbot apps, study scripts
configs/        pipeline / chatbot / models / relevance / gold passages / prompts
tests/          the 76-test suite
static/         the web chat frontend
documents/      methodology notes + evaluation protocol/forms (markdown only)
data/README.md  how to rebuild data/ locally
README.md, pyproject.toml, requirements*.txt, .gitignore
```

Kept local (`.gitignore`), with reasons:

| Excluded | Why |
|---|---|
| `data/` (except its README) | guideline PDF and papers are **copyrighted**; `derived/` contains scraped Alodokter text the **ethics protocol** commits to storing securely, not redistributing |
| `outputs/` | fully regenerable; packets/answers also contain Alodokter text |
| `documents/thesis/`, `*.zip` | the assessed thesis submission (keep private until after marking) |
| `.venv*`, `.hf_cache/`, caches | environments and model weights (≈200 GB) |
| `archive/`, `*.log`, `*.bak`, `*.pptx/.docx/.pdf` | history, logs, binaries |

A fresh clone rebuilds everything from the guideline PDF via the README
quick-start commands. Never force-add anything under `data/` or `outputs/`.

### De-identification (2026-07-27)

The exclusions above are the primary protection, and they are unchanged. What
changed is the assumption *behind* them. The two frozen study files previously
held raw Alodokter text on the reasoning that they were researcher-only and
`.gitignore`d — de-identification happened at render time, in
`evaluation/deidentify.py`, only for participant-facing artifacts. That reasoning
is one `git add -f` away from failing, and it stops being true at all if any of
this is ever published as a dataset.

So the scrub was moved upstream into the files themselves
(`scripts/deidentify_frozen_files.py`):

| Change | Count | Note |
|---|---|---|
| self-introduced names → `[nama]` | 3 | Q05, Q26, **Q28** — the last found on 2026-07-27, from a 16-year-old, missed by all three pre-pass regexes |
| mojibake emoji stripped, sentence spaces restored | 7 | readability only, no clinical content |
| `url` → `source_sha256` | 50 | **the important one** |

The URL mattered more than the names. A scrubbed question sitting next to a link
to its unscrubbed original is not de-identification. The full mapping now lives in
`data/derived/SEALED_source_urls.csv`, researcher-only.

Cost: both files are hash-pinned in `eval_case_assignments.json`, so this forced
`freeze_eval_assignments.py --force`, a regeneration of all 50 briefings, and a
re-run of every automatic score. The draw was verified byte-identical across the
re-freeze (0 of 50 cases changed in any field), so the pre-specified
randomisation is intact.

**Residual risk to disclose in the write-up.** The questions remain *verbatim*
forum text. A distinctive sentence can still be pasted into a search engine and
traced to its source. Removing names and URLs defeats casual re-identification,
not determined re-identification. The only real fix is paraphrasing, which would
alter the very stimuli the study measures — so it was not done, and this is
stated as a limitation rather than solved.

The same reasoning retired `doctor_point_coverage` (§4.4): it was the only path
by which patient-facing doctor replies were sent to a judge model, and it scored
the briefing against a document of a different genre.

---

## 9. How the project was developed

**Method: specify → build → freeze → audit → extend, with the record kept.**

| Period (2026) | What happened |
|---|---|
| late June | Phase-1 design settled; extraction/cleaning/segmentation built; reference tokenizer decided; content-type labels defined and LLM-audited (κ ≈ 0.8) |
| ≈ Jun 29 – Jul 1 | embedding/indexing pipeline; gold passages + concept induction/audit; 124-question gold set generated, validated, frozen (reduced to 121 on 2026-07-26) |
| Jul 2 | retrieval bake-off run; **selection frozen**; independent `pipeline_audit.md` re-verified every number |
| Jul 5–7 | chatbot deployed (vLLM + web UI); evaluation instruments drafted (ESHRO + Park harmonisation, forms in Indonesian) |
| Jul 12 | BGE-M3 challenger episode (selection survives); k=5 decision recorded; earlier design explorations archived |
| Jul 13 | study design v2: direct briefing rating (P1) added with faithfulness items; counselor study sampled to 24 cases / P1 to 36 |
| Jul 16 | assignments frozen (seed 20260628); 36 packets + scoring keys built |
| Jul 19–20 | supervisor feedback folded in: two-psychologist split (16-item overlap), interleaved sequences, half-faithfulness economy — implemented, unit-tested, re-frozen (earlier draws verified byte-identical); terminology unified (with/without chatbot); repo restructured for release |
| Jul 21 | briefing prompt revised and **all 50 study briefings regenerated** (`study_answers_chatbot.oldprompt.bak` keeps the previous set); retrieved contexts and risk flags unchanged. The automatic evaluation was **not** re-run — see `pipeline_audit_20260726.md` §4.3 |
| Jul 25–26 | study kit rebuilt end to end: counselor workbooks, psychologist workbooks + sealed key, and the 36 P1 packets regenerated against the new briefings; assignments re-frozen with `--force` (the draw itself verified unchanged); risk-classifier audit run (0 negation false positives, 0 disagreements with the frozen flags); superseded forms and eval outputs moved to `archive/superseded_20260726/`; `ARCHITECTURE.md` written as the current-wiring index |
| Jul 26 | full-chain audit (`pipeline_audit_20260726.md`): chunking determinism, both frozen seeded draws and every published Phase-1 number re-verified from the artifacts; 30 findings raised, 3 of them critical |
| Jul 26 | audit findings worked through stage by stage: `content_type_overrides` made an explicit config key with a hard error (chunking was only conditionally deterministic); all 71 indexes rebuilt so manifests record their chunk inputs (provenance chain now verifies by string equality, 14/14 and 142/142); gold passage `MI1_0003` removed as a mis-flattened WHO Master Chart table and its 3 questions dropped (121 remain); the stale automatic evaluation re-run with Qwen3-14B as primary and 32B as a robustness check |
| Jul 27 | **de-identification moved into the frozen files** (§8): 3 names, 50 source URLs hashed; assignments re-frozen (draw byte-identical), all 50 briefings regenerated, judge + RAGAS re-run against them. `doctor_point_coverage` dropped — genre-mismatched, unused downstream, and the only route sending doctor replies to a judge. Superseded scores archived in `archive/eval_pre_deidentification_20260727/` |

Practices that shaped the code: a src-layout package with `ports/`
interfaces and pure, testable domain logic; scripts as thin entry points;
one config file per concern; superseded work moved to `archive/` rather than
deleted; every consequential decision written down in `documents/` *at the
time it was made*, with dates.

---

## 10. Explaining it to your supervisor

The one-breath story: *"A fixed knowledge source → a retriever we selected
rigorously (56 configs, frozen metric, corrected statistics, safety
tie-breaker) → a counselor-briefing chatbot with an escalation-first safety
layer → tested in three layers: cheap automatic checks on all 50 real
questions, direct expert rating of the tool, and a frozen, blinded,
counterbalanced study that isolates the causal question — does the tool make
the counselor better?"*

Questions to expect, with the honest answers:

- **"Why is the chatbot only a tool, not answering users?"** Safety and
  scope: a qualified human stays between the model and a vulnerable
  population — and it turns evaluation into a well-posed causal comparison.
- **"Why nDCG@5?"** The generator reads the top 5; rank-graded quality
  within that window is the deployed quantity. Frozen before scoring;
  recall/MRR reported alongside.
- **"Why only 2 counselors / 24 questions?"** Expert time. 24 pairs give
  80 % power for medium-to-large effects; questions are the unit of
  inference, counselor is a fixed factor. Adding counselors is the stated
  best next investment.
- **"Why two psychologists?"** Halves the per-rater load (~6 h each) at
  near-constant total cost and converts the single-rater weakness into a
  measured quantity (agreement on 16 shared items). Whole-case bundles keep
  rater strictness out of the with/without comparison.
- **"Why is BGE-M3 'post-hoc'?"** The selection was already frozen;
  re-opening a frozen decision after seeing results invalidates the
  statistics. As a labelled challenger it *strengthens* the claim: the
  procedure survived an attack it wasn't fitted to.
- **"Are the automatic scores trustworthy?"** Safety/structure: yes (stable
  across judges). Faithfulness: judge-relative (0.50–0.76 on identical
  outputs) — which is why the human study carries a calibration arm.
- **"What remains before data collection?"** Crisis line into the counselor
  forms, exit questionnaire, recruit the second psychologist. Ethics alignment: the submitted
  application describes the study at concept level; confirm the approved
  protocol matches the current design before running.

---

## 11. Glossary

| Term | Meaning here |
|---|---|
| **RAG** | retrieval-augmented generation: the model answers only from retrieved guideline passages |
| **chunk / chunking** | a retrievable piece of the corpus / the strategy that cuts the corpus into pieces |
| **nDCG@5** | rank-weighted retrieval quality over the top 5 (1.0 = all relevant, best-ranked) |
| **recall@5** | fraction of questions whose relevant passage appears in the top 5 |
| **gold set** | the 121 questions + 31 passages with frozen relevance judgements used to score retrieval |
| **briefing / packet** | the chatbot's 4-section output for the counselor / that output bundled with its question and passages for rating |
| **with/without chatbot** | the two study conditions a counselor answers under |
| **crossover (counterbalanced)** | each counselor does one block with the tool and the other without, mirrored — so every case gets both conditions |
| **blind rating** | psychologists rate answers under coded IDs, not knowing condition or counselor |
| **case bundle** | a case's two answers + its packet, always assigned to the same rater |
| **inter-rater agreement (ICC / κ)** | statistical agreement between the two psychologists on the 16 shared items |
| **pre-registered / frozen** | decided, seeded and written to file before outcomes could influence it |
| **RAGAS** | a standard library of automatic RAG metrics (faithfulness, relevancy, context precision) |
| **LLM-judge** | a different-family LLM (Qwen) scoring the briefings against defined rubrics |
| **P1 / P2 / K1 / K2** | direct briefing rating / blind counselor-answer rating / per-case counselor usefulness / exit questionnaire |
| **Holm correction** | multiple-comparison adjustment keeping the family-wise error rate at 5 % |
| **manifest** | a JSON record of exactly what produced an artifact (versions, config, seed, hashes) |
