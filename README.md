# Depression RAG — a counselor decision-support chatbot

A RAG chatbot that supports non-specialist counselors handling depression
cases, grounded exclusively in the Indonesian Ministry of Health primary-care
mental-health guideline. The pipeline has two phases:

- **Phase 1 — retrieval evaluation** (§1–§8): which combination of chunking
  strategy and embedding model best retrieves clinical guidance from the
  guideline?
- **Phase 2 — chatbot + evaluation** (§10): the deployed counselor-briefing
  chatbot built on the winning retriever, its safety layer, the automatic
  evaluation (LLM-judge + RAGAS), and the human evaluation study
  (with vs without chatbot).

## Phase 1 — retrieval evaluation

This phase answers the retrieval question. It indexes the Indonesian
Ministry of Health primary-care mental-health guideline (*Deteksi Dini dan
Penatalaksanaan Gangguan Jiwa bagi Dokter Umum di FKTP*, Kemenkes RI, 2017)
under **14 chunking configurations × 4 embedding models = 56 FAISS indexes**,
scores every index on a **121-question gold set** with span-level relevance
judgements, and selects one configuration with confidence intervals,
significance tests, and sensitivity checks.

**Result: `structure-512-0__e5-large`** — structure-aware chunking (512-token
target, no overlap) encoded by `intfloat/multilingual-e5-large`. nDCG@5 = 0.798
(95% CI [0.747, 0.846]), recall@5 = 0.950, and perfect recall@5 on the
safety-critical suicide-risk questions. Details in [Results](#7-results).

Two post-selection artifacts extend the scored index set to 71 without altering
the selection: a multi-vector variant of the winner (deployment insurance for
its four truncated atomic blocks) and a fifth embedding model, **BGE-M3**,
added 2026-07-12 as a robustness challenger — it joins the statistical tie but
loses the type-robustness tie-breaker
(`documents/bge_m3_robustness_addendum.md`).

The answer-generation chatbot built on this retriever is Phase 2 (§10).

---

## Repository scope

This repository is the code-and-method record for the study: the full pipeline
source, the scripts that produce every reported number, the configuration and
prompt files, the test suite, and the methodology documentation.

Four categories are deliberately **not** published here, so some file paths
named in the text below and in source comments will not resolve:

| Not included | Why |
| --- | --- |
| Source data (`data/`) | The Kemenkes guideline and the reference papers are copyrighted; the Alodokter question text is secondary data the ethics protocol commits to storing securely rather than disseminating. See [`data/README.md`](data/README.md). |
| Study instruments, forms and ethics amendments | Human data collection is still open, and publishing the instruments would prime the safety behaviour the study measures. |
| Study results and manuscript sources | Under review; they belong to the paper, not to its supplement. |
| Operational runbooks and internal process records | Specific to one HPC cluster and one launch, or a record of how the work was managed rather than how the method works. |

Generated artifacts (`outputs/`) are excluded too, but are reproducible from
the scripts documented below.

---

## 1. Pipeline overview

```
guideline PDF (5 in-scope units)
  │  extraction: PyMuPDF blocks, reading order
  │  cleaning:   headers, glyph normalization, reflow, de-hyphenation
  │  segmentation: clinical body only, labelled segments with char offsets
  ▼
cleaned_text.txt + segments.jsonl        ← the single char-offset coordinate system
  │  chunking: fixed / recursive / structure-aware × size × overlap  (14 configs)
  ▼
outputs/chunks/<config_id>.jsonl
  │  embedding + indexing: 4 models, exact cosine (FAISS IndexFlatIP)
  ▼
outputs/indexes/<config_id>__<model>/    (57: 56 grid + 1 deployment __mv)
  │  evaluation: 121 gold questions → per-query metrics → CIs → significance
  ▼
outputs/analysis/*.csv, *.md             → one selected configuration
```

Every segment, chunk, and gold passage carries `[char_start, char_end)` offsets
into the same `cleaned_text.txt`, with the invariant
`text == cleaned_text[char_start:char_end]` enforced by tests and at run time.
This shared coordinate system is what makes relevance judgements comparable
across chunking strategies that cut the text at different boundaries.

## 2. What gets compared

**Chunking strategies** (sizes measured by one reference tokenizer, the XLM-R
tokenizer used by e5, so "256 tokens" means the same text everywhere):

| strategy | behaviour | configs |
|---|---|---|
| `fixed` | naive token sliding window (baseline) | {128, 256, 512} × overlap {0, 64} |
| `recursive` | split on paragraph → line → sentence → word, then pack | {128, 256, 512} × {0, 64} |
| `structure` | one chunk per document section; **never splits a tightly-coupled clinical unit** (diagnostic criteria, dosage blocks, referral lists); oversize non-atomic sections fall back to the recursive splitter | 512-0, plus a context-enriched ablation (`-ctx`: heading path prepended at embed time only) |

**Embedding models** (`configs/embedding_models.yaml`):

| model | checkpoint | dim | max seq | notes |
|---|---|--:|--:|---|
| e5-large | `intfloat/multilingual-e5-large` | 1024 | 512 | asymmetric `query:`/`passage:` prefixes |
| nomic-indonesian | `asmud/nomic-embed-indonesian` | 768 | 8192 | `search_query:`/`search_document:` prefixes |
| minilm | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | truncates hard at 128 tokens |
| indobert | `indobenchmark/indobert-base-p1` | 768 | 512 | mean-pooled MLM — an intentional weak baseline (it is not a sentence-embedding model) |
| bge-m3 | `BAAI/bge-m3` | 1024 | 8192 | **post-selection robustness challenger** (added 2026-07-12, not part of the selection grid); dense/CLS, no prefixes — see `documents/bge_m3_robustness_addendum.md` |

Per-index truncation rates are logged in each index's `meta.json`; comparisons
across chunk sizes must be read together with them (a 512-token chunk is seen
in full by nomic but only its first 128 tokens by MiniLM).

## 3. The gold evaluation set

Questions are generated **from the source text, never from the chunks under
test**, so no chunking strategy is favoured:

1. **31 gold passages** (12 clinical concepts) are curated in
   `configs/gold_passages.yaml` — exact char spans over `cleaned_text.txt`,
   with concept labels cross-checked by two independent blind LLM judges.
   A 32nd, `MI1_0003`, was removed on 2026-07-26: it was the WHO mhGAP Master
   Chart, a table PyMuPDF flattened so that a psychosis row landed under a
   depression heading.
2. **121 Indonesian questions** (124 authored, 3 dropped with `MI1_0003`) are
   LLM-generated from those passages under a
   clinically constrained prompt (`configs/prompts/qa_generation.md`), tagged
   with one of 8 question types (diagnostic criteria, symptom recognition,
   pharmacotherapy/dosing, referral, suicide-risk/emergency, psychoeducation,
   special populations, differential/comorbidity) and a difficulty
   (97 factoid / 15 multi-hop / 9 applied-case), then schema-validated and
   author-reviewed. No near-duplicates (token-Jaccard checked).
3. **Relevance** is frozen once in `configs/relevance.yaml`: a chunk is
   relevant to a question iff its char span overlaps a cited gold-passage span
   by ≥ 50% of the *shorter* of the two spans. A sensitivity run under the
   answer-containment rule (≥ 50% of the gold span) confirms the selection does
   not depend on this choice **on the grid it was selected from**; across all
   scored indexes the post-selection BGE-M3 challenger takes nominal rank 1 by
   0.0017, which the report states explicitly
   (`outputs/analysis/sensitivity_gold/sensitivity_report.md`).

## 4. Metrics and statistics

Per query, at k ∈ {1, 3, 5, 10}: **recall@k** (hit indicator: is any relevant
chunk in the top k — note this is hit-rate, not |hits|/|relevant|),
**precision@k**, **MRR@k**, **nDCG@k** (binary gains), and **multi-hop
coverage@k** (does the top k cover *every* passage a multi-hop question needs).
Aggregates are reported overall, per question type, and per difficulty.

Uncertainty and significance, all seeded (seed 20260628):

- percentile **bootstrap 95% CIs** (10,000 resamples over questions);
- **paired bootstrap** and **Wilcoxon signed-rank** tests between the winner
  and its challengers, with **Holm correction** and **rank-biserial effect
  sizes** (`outputs/analysis/retrieval_significance.md`).

## 5. Quick start

Requires Python ≥ 3.10. The guideline PDF is **not** distributed with this
repository; place it at `data/Modul Keswa bagi Dokter Umum di  FKTP.pdf`
(note the double space) or point `configs/pipeline.yaml → source.pdf_path`
at your copy.

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt      # chunking stages
./.venv/bin/pip install -e ".[embed]"            # + torch / sentence-transformers / faiss
./.venv/bin/pip install scipy                    # Wilcoxon test in the significance script

# 1. extract + clean + segment + chunk (14 configs)
./.venv/bin/python -m depression_rag.cli

# 2. embed + build the single-vector indexes (GPU recommended; --device cpu works)
./.venv/bin/python -m depression_rag.cli_index
# 2b. the DEPLOYED index is the multi-vector variant of the winner and needs its
#     own command — configs/chatbot.yaml points at it, and step 2 does not build it
./.venv/bin/python -m depression_rag.cli_index \
    --models e5-large --configs structure-512-0 --multi-vector

# 3. build gold passages and validate the question set
./.venv/bin/python scripts/build_gold_passages.py
./.venv/bin/python scripts/validate_questions.py data/derived/questions_gold.jsonl

# 4. score all 57 indexes, then significance + sensitivity
#    (56 grid candidates + the __mv deployment build of the winner; the __mv
#     variant ranked 2nd on the primary metric and did not affect the pick —
#     report it separately, see documents/stage1_audit_20260802.md)
./.venv/bin/python scripts/run_retrieval_eval.py --device cuda
./.venv/bin/python scripts/retrieval_significance.py --top 6
./.venv/bin/python scripts/retrieval_sensitivity.py --device cuda

# tests
./.venv/bin/python -m pytest
```

Useful variants: `--models e5-large` / `--indexes <id>` restrict the eval and
write nowhere else. `--scope MI.4` (chunk one unit) and `--only structure-512-0`
(subset of chunk configs) are different: they rewrite `cleaned_text.txt`,
`segments.jsonl` and `outputs/chunks/` **in place** from a partial run, so the
char offsets the gold passages and all built indexes rely on no longer line up.
The CLI prints a loud banner when you use them; point `output.derived_dir` and
`output.chunks_dir` at a scratch directory first if you are just exploring.
Model downloads go to `./.hf_cache` inside the project.

## 6. Repository layout

```
configs/            pipeline.yaml (source/cleaning/segmentation/chunking sweep)
                    embedding_models.yaml · relevance.yaml (+ _gold variant)
                    gold_passages.yaml · content_type_overrides.yaml · prompts/
                    chatbot.yaml (deployed chatbot: retriever, generator, safety, prompts)
src/depression_rag/
  domain/           frozen models (Chunk, Segment, ChunkConfig, EmbeddingModelSpec)
  ports/            Protocol interfaces for every stage
  extraction/       pdf_loader (the only fitz import) · cleaning · segmentation
  chunking/         reference tokenizer · fixed · recursive · structure
  embedding/        sentence-transformers + raw-transformers backends
  indexing/         exact cosine FAISS wrapper
  evaluation/       gold passages · questions · relevance · metrics · bootstrap
  pipeline/         chunking + indexing orchestrators (composition roots)
  cli.py, cli_index.py
  chatbot.py        Phase-2 engine: config, prompt assembly, safety screen, generators
scripts/            run_* entry points, eval/significance/sensitivity scripts,
                    chatbot apps (chatbot_web.py, chatbot_app.py, serve_chatbot.sh),
                    study scripts (generate_study_answers, judge_study_answers,
                    ragas_eval, freeze_eval_assignments, build_eval_packets)
static/             chat frontend served by chatbot_web.py
tests/              70 tests: invariants, chunkers, prefixes, metrics, determinism, chatbot
documents/          methodology notes and the human-evaluation design/forms (markdown)
data/               NOT in git (copyrighted PDF + study data) — see data/README.md
outputs/            NOT in git — regenerated by the pipeline (indexes, analysis, packets)
```

Not in version control (see `.gitignore`): virtualenvs, `.hf_cache/`, all of
`outputs/` and `data/` (except `data/README.md`), `archive/`, logs, and binary
documents (pptx/docx/pdf). A fresh clone rebuilds everything from the guideline
PDF via the quick-start commands.

## 7. Results

All numbers are reproducible from the pipeline above; the evaluation is
deterministic given the built indexes.

**The embedding model dominates the chunking choice.** e5-large is the best
model on *every* metric at every k; within the selection grid its best nDCG@5
(0.798) is ~0.25 above the next model's best (MiniLM 0.548, nomic 0.490,
IndoBERT 0.394 — the weak baseline behaving as expected). The post-selection
challenger BGE-M3 later closed most of that gap (best 0.790, statistically
tied) but lost the type-robustness tie-breaker, leaving the selection unchanged
(`documents/bge_m3_robustness_addendum.md`).

**Within e5-large, 512-token chunks beat smaller ones** (mean nDCG@5: 512 →
0.761, 256 → 0.680, 128 → 0.633); this gap is the one chunking difference that
survives Holm correction. At 512 tokens like-for-like:

| config (e5-large) | nDCG@5 | recall@5 | MRR@5 |
|---|--:|--:|--:|
| **structure-512-0** | **0.798** | 0.950 | 0.794 |
| structure-512-0-ctx | 0.787 | 0.942 | **0.796** |
| recursive-512-0 | 0.779 | **0.967** | 0.783 |
| fixed-512-0 | 0.738 | 0.926 | 0.737 |

The top three are a statistical tie on the primary metric, so the selection is
decided by pre-stated tie-breakers — robustness across question types first:
`structure-512-0` keeps recall@5 = 1.000 on the 17 suicide-risk questions
(the `-ctx` variant drops one), has the highest worst-type floor, and needs no
enrichment step. Context enrichment did not help on this corpus, and small
chunks win only rank-1/MRR-style metrics, which is partly mechanical (overlap
and small sizes create many near-duplicate relevant chunks) and least relevant
when the top-k chunks are read by a generator rather than shown singly.

Only 2 of 121 questions have no relevant chunk in the winner's top 10.

## 8. Limitations

- **Single document, synthetic questions.** The corpus is one guideline; the
  questions are LLM-generated from the gold passages and validated by the
  author, not by an independent clinician panel. Absolute scores are therefore
  optimistic; the *relative* comparison between configurations is the claim.
- **`recall@k` is a hit indicator** (standard for single-gold-passage setups),
  not the |retrieved ∩ relevant| / |relevant| ratio — defined in
  `evaluation/metrics.py`.
- **Cross-config qrel sizes differ.** Small/overlapping configs turn one gold
  span into many near-duplicate relevant chunks, inflating hit-style metrics
  and precision@k; nDCG partially counteracts this. Metrics are compared with
  that in mind.
- **Oversize atomic chunks embed truncated.** Structure-aware chunking keeps
  four long dosage blocks whole (818–1,165 e5 tokens); e5 embeds only their
  first 512 tokens. The stored text is complete (only the vector is truncated)
  and the measured cost is a single missed question, but it is a real
  representation limit, logged per index in `meta.json`.
- **`content_type` labels are heuristic** (keyword rules plus a blind
  two-judge audit for the contested cases) and per-type buckets are small
  (n = 6–23), so per-type numbers carry wide intervals.
- **The between-model margin is inflated by a representation limit, not only
  by model quality.** The grid embeds every chunk as one vector, so a model is
  penalised for its context cap: MiniLM (cap 128) has 67 % of the selected
  config's chunks clipped, against 12 % for e5-large (cap 512) and 0 % for the
  two 8k-context models. Giving every cap-limited model the same multi-vector
  treatment the deployed index uses lifts MiniLM from 0.548 to **0.667** nDCG@5
  (paired Δ +0.119, *p* = 0.001) and IndoBERT by +0.020, while moving e5-large
  by −0.003 (n.s.). So the headline "~0.25 above the next model's best" in §7 is
  really **~0.13** under like-for-like representation. The selection is
  unchanged — e5-large still leads by 0.13 nDCG@5 (*p* = 5.0 × 10⁻⁵) and by 0.18
  on the worst-type floor the tie-breaker rests on (0.850 vs 0.667) — but the
  margin should be quoted with this caveat. Measured in
  `documents/multi_vector_fairness_probe.md`; reproduce with
  `scripts/mv_fairness_probe.py`.
- **De-identification is partial by construction.** The 50 study questions are
  real Alodokter posts. Direct identifiers are removed — three self-introduced
  names replaced with `[nama]`, and every source URL reduced to a hash, since a
  link to the unscrubbed original defeats any amount of text scrubbing. But the
  questions are kept **verbatim**, because paraphrasing would change the stimuli
  the study measures, and verbatim text is itself searchable: a distinctive
  sentence can be traced back to its source post. The protection is against
  casual re-identification, not determined re-identification. `data/` and
  `outputs/` stay out of version control for this reason, and
  `data/derived/SEALED_source_urls.csv` (the hash → URL map) is researcher-only.
  One question is from a self-reported 16-year-old; it is excluded from every
  participant-facing packet. See `PIPELINE_DOCUMENTATION.md` §8.
- **The automatic evaluation does not compare against the doctor's reply.**
  A `doctor_point_coverage` metric existed until 2026-07-27 and was dropped: the
  chatbot produces a counselor decision-support briefing and the Alodokter reply
  is a patient-facing answer, so the comparison mixes genres. Nothing downstream
  consumed it. This means the automatic layer certifies grounding, structure and
  safety only — whether the briefing is *clinically better or worse than a GP's
  answer* is not measured here, and is left to the human study.

## 9. Reproducibility

- Pinned dependencies (`requirements.txt`, `requirements.lock.txt`, exact pins
  in `pyproject.toml`); Python ≥ 3.10.
- One fixed seed (20260628) for every run and every bootstrap; the pipeline
  itself is sampling-free and deterministic (tested).
- Every run writes a JSON manifest (`outputs/manifests/`) with package
  versions, config, seed, input/output sha256s, and resolved model revisions;
  every index's `meta.json` pins the checkpoint revision, token stats,
  truncation rate, and build time.
- Exact search only (FAISS `IndexFlatIP` on L2-normalised vectors = cosine):
  no ANN variance in the comparison.

## 10. Phase 2 — chatbot and evaluation

The deployed chatbot has **one role**: a decision-support tool for the
counselor. Given a user question it retrieves k = 5 passages from the winning
index, screens for safety risk (keyword pre-screen + LLM classifier;
escalation-first response on risk), and generates a structured 4-section
briefing for the counselor — guideline information with `[n]` citations,
clinical considerations, danger signs, referral options. It never writes the
patient-facing answer. Everything is configured in `configs/chatbot.yaml`.

# run the chatbot (starts vLLM if needed, launches the app, prints a public link):
bash scripts/serve_chatbot.sh                 # one shared gradio.live link (port 7860)
bash scripts/serve_chatbot.sh counselor1      # per-counselor link: own port + own logs
bash scripts/serve_chatbot.sh counselor2      # a second counselor, running at the same time
bash scripts/serve_chatbot.sh link            # show the current link(s)
# full operations guide: documents/running_the_chatbot.md
.venv-chat/bin/python scripts/chatbot_web.py  # optional: FastAPI UI (reach via proxy/SSH tunnel)

# study pipeline (order):
.venv-chat/bin/python scripts/generate_study_answers.py     # briefings for the 50-question set
./.venv/bin/python scripts/freeze_eval_assignments.py       # frozen draws: case sample, rater split, form halves (seed 20260628)
./.venv/bin/python scripts/build_eval_packets.py            # rating packets + counselor/psychologist sheets
# after counselor answers exist: per-rater blind interleaved sequences
./.venv/bin/python scripts/build_psychologist_workbooks.py --counselor-answers data/derived/counselor_answers.jsonl
.venv-chat/bin/python scripts/judge_study_answers.py        # automatic LLM-judge metrics
.venv-ragas/bin/python scripts/ragas_eval.py --answers outputs/analysis/study_answers_chatbot.jsonl --tag chatbot --skip-correctness
# after human scores are collected: one-command pre-specified analysis
./.venv/bin/python scripts/analyze_study.py                 # → outputs/analysis/study_analysis_report.md
# counselor-side forms only (available before the psychologist scores come back):
./.venv/bin/python scripts/analyze_counselor_forms.py       # → outputs/analysis/counselor_forms_report.md + tidy CSV + stats JSON
./.venv/bin/python scripts/build_counselor_results_slides.py  # → documents/slides_counselor_results.pptx
```

A ready-to-run **study kit** for the human evaluation (forms, per-person
workbooks, the full distribution flow and timeline) is generated into
`outputs/evaluation/study_kit/` — see its `00_PANDUAN_STUDI_LENGKAP.md`.

Evaluation is three-layered: **automatic** (LLM-judge + RAGAS on all 50 study
questions), **direct expert rating** of the chatbot briefings
(quality/safety/faithfulness), and the **human study** — two counselors answer
a frozen, stratified sample of real Alodokter questions with and without the
chatbot (counterbalanced crossover), and two licensed psychologists rate all
answers blind (whole-case split with a 16-item shared overlap for inter-rater
agreement, one interleaved seeded sequence per rater). Full protocol:
`documents/evaluation/evaluation_design_v2.md` (design) and
`documents/evaluation/README_evaluation_design.md` (instruments).

Counselors are required to **paraphrase** each case rather than paste it, and the
two arms therefore use different inputs by design — verbatim questions for the
frozen briefings the psychologists rate, the counselor's own wording for the
with/without comparison. The measurement behind that rule, and the answer to the
obvious objection that paraphrasing adds uncontrolled variance, are in
`documents/query_phrasing_probe.md` §7. That probe is also where the euphemism gap
in the keyword safety screen is recorded (§2.2, §5).

Three questions about the safety layer come up repeatedly and are answered, with
measurements, in `documents/safety_layer.md`: why the fixed banner and the
generated briefing are separate mechanisms (§1 — the guideline corpus contains no
phone numbers, so only the banner can carry the crisis line); where the banner's
text comes from, including the two deliberate omissions (§2); and what the five
independent layers do when the screen misses, with the one limit that is not yet
established out-of-sample (§3–§4).
