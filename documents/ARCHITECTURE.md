# Architecture — what runs, in what order, on what

The one-page map. `PIPELINE_DOCUMENTATION.md` is the long-form explanation of
*why*; this file is the current wiring, kept short enough to stay true.

Verified against the code on 2026-07-27. If it disagrees with a script, the
script wins — and this file is the bug.

---

## The shape of it

```
                 configs/pipeline.yaml          configs/embedding_models.yaml
                        │                                │
  guideline PDF ──► extract ──► segment ──► chunk ──► embed ──► FAISS index
                                                                    │
   PHASE 1 (done, frozen)                            scored on 121 gold questions
   selection: structure-512-0 + multilingual-E5 (multi-vector)      │
                                                                    ▼
                                              configs/chatbot.yaml points here
                                                                    │
  50 real questions ──────────────────────────────► CHATBOT ────────┘
  (Alodokter, frozen)                        retrieve k=5 → Gemma-4-12B
                                             + safety screen → briefing
                                                       │
        ┌──────────────────────────────────────────────┼───────────────────────┐
        ▼                                              ▼                       ▼
   automatic eval                            expert rating of the       counselor study
   (all 50)                                  briefing (36 cases)        (24 cases × 2)
   LLM-judge + RAGAS                         2 psychologists            2 counselors
        └──────────────────────────────────────────────┴───────────────────────┘
                                     │
                            analyze_study.py → report
```

Two layers only: `src/depression_rag/` is the library (importable, tested);
`scripts/` are the entry points (argparse, read files, write files). Scripts never
import each other — anything shared lives in the library.

**Why the entry points are not symmetric.** Chunking and indexing are reachable
four ways (library class, `python -m depression_rag.cli[_index]`, the
`scripts/run_*.py` shim, the `depression-rag-*` console scripts); the retrieval
evaluation is reachable one way, `scripts/run_retrieval_eval.py`. That is
deliberate, not an oversight: the first two are *stateful pipelines* whose
orchestration is worth unit-testing and re-using, so the logic lives in
`pipeline/` behind a thin CLI. The evaluation is a *report generator* — it reads
finished artifacts and writes CSVs — with nothing worth importing, so it stays a
single script. If it ever grows a second caller, move the scoring loop into
`evaluation/` and leave the script as a shim, matching the other two.

`scripts/` holds 32 files with no subdirectories, so the filename does not tell
you the role — read it from these tables. **Phase 1** accounts for 15:

| role | scripts |
|---|---|
| pipeline | `run_chunking`, `run_indexing`, `run_retrieval_eval` |
| gold set | `build_gold_passages`, `validate_questions`, `build_question_review`, `convert_question_review` |
| analysis / robustness | `retrieval_significance`, `retrieval_sensitivity`, `reference_tokenizer_sensitivity`, `mv_fairness_probe`, `declare_selection` |
| one-off provenance | `inspect_pdf`, `export_segments_for_judge`, `compare_content_type` |

The last group is not part of any run; each backs a decision recorded in a config
or a note (page ranges, the blind content_type audit input, its comparison).

**Phases 2–3** account for the other 17:

| role | scripts |
|---|---|
| study corpus | `alodokter_scraper`, `prepare_alodokter_questions`, `sample_alodokter_questions`, `deidentify_frozen_files` |
| chatbot | `chatbot_app`, `chatbot_web`, `generate_study_answers` |
| automatic eval | `judge_study_answers`, `ragas_eval`, `audit_risk_verdicts` |
| study materials | `freeze_eval_assignments`, `build_eval_packets`, `build_counselor_workbooks`, `build_psychologist_workbooks` |
| results | `convert_psychologist_scores`, `analyze_study`, `export_study_questions_csv` |

`deidentify_frozen_files` is a **one-way** rewrite of hash-pinned artifacts — it
forces a re-freeze and a full regeneration downstream. It has been run once
(2026-07-27) and should not be needed again unless a new identifier is found.
`export_study_questions_csv` is the read-only counterpart: it joins the three
study files into one spreadsheet so the finalized question set can be reviewed
without tooling.

---

## Phase 1 — build and select the retriever  *(complete; do not re-run casually)*

| step | command | reads | writes |
|---|---|---|---|
| chunk | `scripts/run_chunking.py` | guideline PDF, `configs/pipeline.yaml`, `configs/content_type_overrides.yaml` | `data/derived/cleaned_text.txt` + `segments.jsonl`, `outputs/chunks/` |
| index | `scripts/run_indexing.py` | `outputs/chunks/*.jsonl`, `configs/embedding_models.yaml` | `outputs/indexes/<chunker>__<model>/` (`index.faiss`, `ids.json`, `chunks.jsonl`, `meta.json`) |
| score | `scripts/run_retrieval_eval.py` | `data/derived/questions_gold.jsonl`, `configs/relevance.yaml`, indexes | `outputs/analysis/` |

`cleaned_text.txt` is the coordinate system every later stage indexes into, and
`content_type_overrides.yaml` decides which segments are `atomic` and therefore
never split. Both are hashed into the run manifest. Re-running with `--scope` or
`--only` rewrites these shared paths from a partial run — the CLI prints a
blocking-loud banner when you do.

Sensitivity and significance checks: `retrieval_sensitivity.py`,
`retrieval_significance.py`, `reference_tokenizer_sensitivity.py`.

Gold-set construction: `build_gold_passages.py` → `validate_questions.py`
(→ `outputs/analysis/questions_gold_report.md`, the standing evidence that the
gold set validates: 121 questions, 31/31 passages covered, schema OK).
Step-3 human review of the questions: `build_question_review.py` writes a
worksheet grouped by gold passage with the passage text inline;
`convert_question_review.py` reads the completed worksheet back into
`questions_gold.jsonl` and records the worksheet's checksum beside it, so the
review is checkable from the artifacts rather than asserted. It refuses to run
while any question is still unreviewed.
The gold questions were LLM-authored under `configs/prompts/qa_generation.md`
(qa-gen-v2) and human-validated; there is no script that regenerates them. 124
were authored; **121 are in use** — three were dropped on 2026-07-26 with gold
passage `MI1_0003`, a WHO mhGAP Master Chart table that PyMuPDF flattened so that
a psychosis row landed under a depression heading.
The superseded 100-question v1 set survives as data in
`archive/questions_candidate_old.jsonl`; the script that embedded it was archived
2026-07-26 (it cited 4 gold passages that no longer exist, so it could not run).

**Selected configuration** — recorded in `outputs/analysis/selection.json`, written
by `scripts/declare_selection.py`: the winner, the frozen rule it was picked under,
its score and CI, the challengers it does **not** significantly beat, the
tie-breaker that resolves that tie, and the index `configs/chatbot.yaml` actually
deploys. `declare_selection.py --check` re-derives all of it from the scored CSVs
and exits non-zero if the recorded selection no longer follows — run it after any
change to the gold set, the indexes or the relevance rule.

Read the tie-break in the right order. On the current artifacts seven indexes are
statistically tied on nDCG@5; the safety tie-breaker
(`risk_suicide_emergency` recall@5) eliminates only **one** of them, and it is the
*worst-type floor* (0.850 vs 0.833 / 0.500 / 0.333) that narrows the field to
`structure-512-0__e5-large` and its own `__mv` variant. Leading with the safety
criterion alone overstates what it decides.

`configs/chatbot.yaml → retriever.index_dir` is the only line connecting Phase 1
to Phase 2, and `--check` verifies it still agrees with the winner. Note the `__mv` suffix: a default `cli_index` run
builds the 70 single-vector indexes the selection was scored on and **not** this
one, which needs `--models e5-large --configs structure-512-0 --multi-vector`.
Rebuilding the deployment from scratch is two commands (README §5, steps 2 and 2b).

## Phase 2 — the chatbot

`src/depression_rag/chatbot.py` is the whole runtime: config, safety screen,
retrieval, prompt assembly, two generator backends, session logging.

- serve it: `bash scripts/serve_chatbot.sh` (starts vLLM, then the Gradio app)
- one-off generation over the 50 study questions:
  `scripts/generate_study_answers.py` → `outputs/analysis/study_answers_chatbot.jsonl`

Safety screen: keyword list (`configs/chatbot.yaml → safety.keywords`) first, then
a single-prompt LLM classifier only if the keywords miss. Both the verdict and its
raw text are recorded per case.

## Phase 3 — evaluation

**Automatic (all 50 cases)**

| command | reads | writes |
|---|---|---|
| `scripts/judge_study_answers.py` | `study_answers_chatbot.jsonl` | `outputs/analysis/study_eval_chatbot.csv` |
| `scripts/ragas_eval.py` | `study_answers_chatbot.jsonl` | `outputs/analysis/ragas_chatbot.csv` |
| `scripts/audit_risk_verdicts.py` | the 50 questions | `outputs/analysis/risk_verdicts_audit.jsonl` |

**Human study.** The order below is the whole protocol; every arrow is a file.

```
freeze_eval_assignments.py          seed 20260628 + input hashes
        └─► data/derived/eval_case_assignments.json      ← FROZEN, the single
                │                                          source of truth for
                │                                          who rates/answers what
      ┌─────────┼──────────────────────────────┐
      ▼         ▼                              ▼
build_counselor_    build_eval_packets.py    build_psychologist_
 workbooks.py         └─► 02_PSIKOLOG/         workbooks.py
  └─► 01_KONSELOR/*.xlsx     p1_packets/       └─► 02_PSIKOLOG/Psikolog_{1,2}.xlsx
                             (reading material) └─► 03_ADMIN_PENELITI/SEALED_p2_key.csv
        │  counselors fill                             │
        ▼                                              │
convert_counselor_answers.py                           │
  └─► data/derived/counselor_answers.jsonl ────────────┘
        │                                    (rebuild the psychologist workbooks
        │                                     with --counselor-answers so the
        │                                     answer column is populated)
        │                                              │
        │                                     psychologists fill
        │                                              ▼
        │                              convert_psychologist_scores.py
        │                                └─► outputs/analysis/p{1,2}_scores.csv
        ▼                                              │
        └──────────────────► analyze_study.py ◄────────┘
                               └─► outputs/analysis/study_analysis_report.md
```

Run order, end to end:

```bash
.venv/bin/python scripts/freeze_eval_assignments.py          # once; refuses to overwrite
.venv/bin/python scripts/build_eval_packets.py               # 36 reading packets
.venv/bin/python scripts/build_counselor_workbooks.py        # 4 counselor workbooks
.venv/bin/python scripts/build_psychologist_workbooks.py     # 2 rater workbooks + sealed key
#   ... counselors answer ...
.venv/bin/python outputs/evaluation/study_kit/03_ADMIN_PENELITI/convert_counselor_answers.py
.venv/bin/python scripts/build_psychologist_workbooks.py \
    --counselor-answers data/derived/counselor_answers.jsonl
#   ... psychologists rate ...
.venv/bin/python scripts/convert_psychologist_scores.py
.venv/bin/python scripts/analyze_study.py
```

---

## Frozen artifacts — change these only on purpose

| file | why frozen | guard |
|---|---|---|
| `data/derived/questions_alodokter_sample50.jsonl` | the study corpus | hash recorded in the assignment file |
| `outputs/analysis/study_answers_chatbot.jsonl` | its `risky` flags define the sampling strata | hash recorded in the assignment file |
| `data/derived/eval_case_assignments.json` | the pre-specified draw | `freeze_eval_assignments.py` refuses to overwrite without `--force`; the previous freeze is kept as `*.superseded_20260720.json` |
| `03_ADMIN_PENELITI/SEALED_p2_key.csv` | unblinds the counselor answers | keep away from raters until analysis |
| `data/derived/SEALED_source_urls.csv` | maps each case to its original forum post | researcher-only; publishing it would undo the de-identification |

Everything else under `outputs/` regenerates from these plus the code.

**The two study files are de-identified as of 2026-07-27.** They previously held
raw Alodokter text on the argument that they were researcher-only; that stops
being true if the repository is published. `scripts/deidentify_frozen_files.py`
rewrote them (3 names → `[nama]`, plus `url` → `source_sha256`, the URL being a
sharper re-identification vector than any name), which forced
`freeze_eval_assignments.py --force` and a regeneration of all 50 briefings and
every automatic score. The draw was verified byte-identical across the re-freeze:
0 of 50 cases changed in any field.

Residual risk, stated for the write-up: the questions are still **verbatim** forum
text, so a distinctive sentence remains searchable back to its source. Scrubbing
defeats casual, not determined, re-identification. Paraphrasing would fix it and
would also change the stimuli the study measures, so it was not done.

## Where the seams are

Three places translate between a human-facing layout and an analysis schema. They
are the only places that know both, and they are the first thing to check when a
stage stops connecting:

- `convert_counselor_answers.py` — counselor workbooks → `counselor_answers.jsonl`,
  `counselor_k1.csv`, `counselor_k2.csv`
- `convert_psychologist_scores.py` — rater workbooks → `p1_scores.csv`, `p2_scores.csv`
- `src/depression_rag/evaluation/deidentify.py` — the one renderer for patient
  questions, used by every participant-facing artifact
- `src/depression_rag/evaluation/packets.py` — the one assembler of a P1 reading
  packet, rendered two ways: a markdown file (`build_eval_packets.py`) and a
  sheet inside the psychologist workbook (`build_psychologist_workbooks.py`). If
  those disagreed, two raters would be scoring different material and the P1
  comparison would be void, so the content is built once and only the
  presentation differs.

## Conventions worth knowing

- Workbook dropdowns store the anchor with the number (`"4 – lancar"`). Converters
  parse the leading integer and keep the raw string.
- `study_id` (`Q01`…) is the human-facing case code; `question_id` (`alo_…`) is
  internal and never shown to a participant.
- Response codes `R001`–`R048` are drawn from the frozen crossover, not from the
  answers file, so they exist before the answers do and never move.
- Seed `20260628` everywhere. Any new draw must consume the shared RNG in a fixed,
  documented order so appending a draw cannot disturb an earlier one.
