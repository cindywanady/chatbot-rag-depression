# Chunking Evaluation (Phase 1) — Metrics & Outputs

*Grounding: Indonesian MoH FKTP mental-health guideline (Kemenkes, 2017), depression submodule.*
*Every number below is re-derived from the actual run artifacts in `outputs/analysis/`, not from the spec or slides.*

---

## 1. What this evaluation answers

Phase 1 is **purely technical** — no participants. Its single question is:

> **Which way of cutting the guideline into retrievable units, paired with which embedding model, retrieves the right passages best?**

The winner is **frozen** before Phase 2 (the chatbot), so downstream results are never a moving target.

The evaluation is a grid: **chunking strategy × chunk size/overlap × embedding model**. Each cell is one FAISS index. Every index retrieves for the same 121 gold questions, and is scored deterministically by how well its retrieved chunks overlap the gold passage spans.

---

## 2. The moving parts

### 2.1 Chunking strategies (3 families → 14 configurations)

Defined in `configs/pipeline.yaml`, built via `src/depression_rag/chunking/`.

| Strategy | What it does | Sweep | # configs |
|---|---|---|--:|
| **fixed** | Cut into fixed token windows with overlap. Strategy-agnostic baseline. | sizes {128, 256, 512} × overlaps {0, 64} | 6 |
| **recursive** | Split at natural boundaries in order — paragraph → sentence → word. | sizes {128, 256, 512} × overlaps {0, 64} | 6 |
| **structure** | Follow the guideline's own structure (per *pokok bahasan* / diagnostic step). | size 512, overlap 0, `context_enriched ∈ {false, true}` | 2 |

That is **14 chunk configurations** total. `structure-512-0-ctx` is the context-enriched ablation of the structure chunker (prepends section context to each chunk).

Naming convention: `<strategy>-<size>-<overlap>` (e.g. `fixed-256-64`, `structure-512-0`).

### 2.2 Embedding models (4)

Defined in `configs/embedding_models.yaml`. All L2-normalised so inner product = cosine.

| Model | Checkpoint | Dim | Max seq | Note |
|---|---|--:|--:|---|
| **e5-large** | `intfloat/multilingual-e5-large` | 1024 | 512 | multilingual; uses `query:` / `passage:` prefixes |
| **nomic-indonesian** | `asmud/nomic-embed-indonesian` | 768 | 8192 | long-context, Indonesian-tuned |
| **indobert** | `indobenchmark/indobert-base-p1` | 768 | 512 | raw BERT + mean pool — deliberate weak baseline |
| **minilm** | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | compact multilingual baseline |
| **bge-m3** | `BAAI/bge-m3` | 1024 | 8192 | **post-selection robustness challenger** (added 2026-07-12, after the selection was frozen; dense/CLS, no prefixes — `bge_m3_robustness_addendum.md`) |

### 2.3 Indexes: 56 in the selection grid, 71 scored

14 chunk configs × 4 models = **56** indexes form the **selection grid**, plus **1** multi-vector variant of the winner (`structure-512-0__e5-large__mv`), plus **14** BGE-M3 indexes added **post-selection** (2026-07-12) as a robustness challenge — giving **71 indexes scored** in the current `outputs/analysis/` CSVs. The configuration selection was made on the original 56 (+1 mv); BGE-M3 results must always be presented with that provenance (`bge_m3_robustness_addendum.md`). Multi-vector mode re-embeds any chunk longer than the model cap as several sub-windows mapping to the same chunk id (search keeps the best window). It is written to a separate `__mv` directory so single-vector indexes are never overwritten.

Index type: FAISS `IndexFlatIP` (exact, not HNSW/IVF) — the corpus is small, so exact search removes approximation variance.

### 2.4 The evaluation set

- **31 gold passages** spanning **12 clinical concepts** — labelled slices of `cleaned_text.txt` with exact character spans. These are the *relevance units*.
- **121 gold questions**, synthesised **from the gold passages, never from the chunks under test** (anti-circularity — no chunking strategy is favoured), then human-validated. In Bahasa Indonesia, non-specialist counsellor persona.

Question mix (from `data/derived/questions_gold.jsonl`):

- **By difficulty:** factoid 100 · multi-hop 15 · applied-case 9
- **By type (8):** pharmacotherapy_dosing 23 · differential_comorbidity 21 · psychoeducation 21 · risk_suicide_emergency 17 · diagnostic_criteria 13 · symptom_recognition 12 · special_populations 11 · referral_criteria 6

---

## 3. How relevance is defined (the key design choice)

Because every strategy cuts the text at different boundaries, "the relevant chunk" cannot be defined per-index. Instead it is defined **at the source-span level** and projected onto each strategy's chunks identically. This is what makes Recall / MRR / nDCG comparable across the whole grid.

**Rule** (frozen once in `configs/relevance.yaml`, never tuned):

```
overlap(chunk, gold) = |chunk_span ∩ gold_span|   (characters)
fraction            = overlap / |shorter of the two spans|
chunk is RELEVANT   ⟺ fraction ≥ 0.5  for ANY gold passage the question cites
```

- **Deterministic, no LLM in the loop.** Both chunks and gold passages carry char offsets into the same `cleaned_text.txt`, so overlap is *exact*, not fuzzy.
- **Why normalise by the *shorter* span?** Gold passages span two size regimes (`main_symptoms` ≈ 330 chars … `referral_criteria` ≈ 24,803 chars). Normalising by the shorter span makes containment in *either* direction score 1.0, so a small gold passage inside a large chunk — and a small chunk inside a large gold passage — both count. Under a `gold`-denominator rule, large passages could never be satisfied by a single chunk. `scripts/retrieval_sensitivity.py` re-scores everything under the `gold` denominator and confirms the selection does not depend on this choice **on the selection grid** (the 4 models present when the pick was made). Across all 71 scored indexes the post-selection BGE-M3 challenger takes nominal rank 1 by 0.0017 — inside the statistical tie, and it loses the type-robustness tie-breaker. The report prints both index sets side by side rather than picking the flattering one.
- Relevance is **binary** (every gold passage graded 1). nDCG is written so graded gains can drop in later.

The qrels (set of relevant chunk ids) for each query is computed by scanning **every** chunk in the index — not just retrieved ones — so Recall/nDCG denominators are correct (`relevance.py:relevant_chunk_ids`).

---

## 4. The metrics (in detail)

Source: `src/depression_rag/evaluation/metrics.py`. Each metric takes a per-query ranked list of retrieved chunk ids (best first) and the query's relevant set. Cutoffs **k ∈ {1, 3, 5, 10}**. All metrics are macro-averaged over the 121 queries.

### 4.1 Recall@k  *(reported — but note the definition)*
```
recall@k = 1.0 if any relevant chunk is in the top k, else 0.0
```
⚠️ **This is a hit-rate@k, not textbook recall** (`|hits| / |relevant|`). Averaged over queries it reads as "on what fraction of questions did the top-k contain at least one right chunk." The code documents this deliberately; keep the distinction in the write-up.

### 4.2 Precision@k
```
precision@k = |top-k ∩ relevant| / k
```
Divisor is always `k` (standard IR convention). **Caveat:** with overlapping chunk configs the relevant set contains near-duplicate chunks, which inflates precision (and recall@1 / MRR) for those configs — do **not** compare precision across configs of different granularity.

### 4.3 MRR@k
```
mrr@k = 1 / (rank of the FIRST relevant chunk),  0 if none in top k
```
Rewards getting a relevant chunk high in the list.

### 4.4 nDCG@k  *(PRIMARY METRIC — `primary_metric: ndcg@5`)*
```
DCG@k  = Σ  gain_i / log2(i + 2)          over the top-k retrieved (i = 0-based)
IDCG@k = DCG of the ideal ranking (all relevant chunks first)
nDCG@k = DCG@k / IDCG@k                    (0 if no relevant chunk exists)
```
Position-aware and normalised to [0, 1]; the headline number is **nDCG@5**, frozen in `configs/relevance.yaml` before any index was scored — an implementation-time choice between the spec's two recommendations (nDCG@5 or Recall@5), not a formal pre-registration (`pipeline_audit.md` §4). (The `metrics.py` header now labels nDCG@k as primary; its earlier "Recall@k (primary)" wording was corrected 2026-07-12.)

### 4.5 Multi-hop coverage@k
```
coverage@k = 1.0 only if the top-k contains ≥1 relevant chunk for EVERY
             gold passage the question requires (not just one)
```
For single-passage questions this reduces to recall@k. It is the honest metric for the 15 multi-hop questions, which need two passages retrieved together.

### 4.6 Selection is not "highest number"
The primary metric ranks the grid, then differences are tested statistically:
- **Percentile bootstrap 95% CI** — 10,000 resamples over the 121 queries, seed `20260628` (`bootstrap.py`, α = 0.05).
- **Paired bootstrap** on the per-query difference (winner − challenger).
- **Wilcoxon signed-rank** on the same pairs, with **Holm** correction across the family of comparisons; significance = Holm-adjusted p < 0.05. Effect size = rank-biserial `r_rb`.

**Tie-breakers** (order fixed by spec §2.6 before any results were seen, and recorded as a comment in `relevance.yaml` when the primary metric was frozen): robustness across question types (esp. `risk_suicide_emergency`) → index size / latency → build cost.

---

## 5. How to run it

```bash
python scripts/run_retrieval_eval.py            # score all indexes
python scripts/run_retrieval_eval.py --models e5-large --device cpu
python scripts/run_retrieval_eval.py --indexes structure-512-0__e5-large
python scripts/retrieval_significance.py        # paired significance on the winner
python scripts/retrieval_sensitivity.py         # re-score under 'gold' denominator (robustness)
```

Scoring is deterministic given the built indexes; it reuses each index's own embedder and does **not** rebuild indexes.

---

## 6. The outputs

All written to `outputs/analysis/`.

| File | One row per… | Contents |
|---|---|---|
| `retrieval_metrics.csv` | index | Overall macro-mean of every metric@k + the primary-metric 95% CI bounds. |
| `retrieval_metrics_by_type.csv` | (index, question_type) | Same metrics broken down over the 8 question types — this is where the safety tie-breaker is read. |
| `retrieval_metrics_by_difficulty.csv` | (index, difficulty) | Same metrics split factoid / multi-hop / applied-case. |
| `retrieval_scores_per_query.csv` | (index, question) | The full long-form dump (~1.4 MB). Enables paired significance tests *without* re-running retrieval. |
| `retrieval_eval_report.md` | — | Human-readable ranked table on nDCG@5 + selection note. |
| `retrieval_significance.md` | challenger | Winner vs nearest challengers: Δ, CI, bootstrap p, Wilcoxon p, Holm p, effect size, sig? |
| `sensitivity_gold/` | — | The whole evaluation re-scored under the `gold` denominator (robustness check). |

### 6.1 Column schema of `retrieval_metrics.csv`
`index_id, chunk_config, model, n_queries,` then for each k ∈ {1,3,5,10}: `recall@k, precision@k, mrr@k, ndcg@k, multihop_coverage@k,` then `ndcg@5_ci_low, ndcg@5_ci_high`.

---

## 7. The result (verified from the artifacts)

### 7.1 Winner: `structure-512-0 × e5-large`

| Metric | Value |
|---|---|
| **nDCG@5** (primary) | **0.7980**  ·  95% CI **[0.7466, 0.8461]**, n = 121 |
| recall@5 (hit-rate) | 0.9516 |
| mrr@5 | 0.7858 |

### 7.2 Top of the leaderboard (nDCG@5)

| Rank | Index | nDCG@5 | 95% CI | recall@5 | mrr@5 |
|--:|---|--:|---|--:|--:|
| 1 | structure-512-0 · e5-large | 0.7980 | [0.7466, 0.8461] | 0.9504 | 0.7942 |
| 2 | structure-512-0 · e5-large (mv) | 0.7951 | [0.7435, 0.8438] | 0.9504 | 0.7956 |
| 3 | structure-512-0-ctx · bge-m3 *(post-sel.)* | 0.7900 | [0.7355, 0.8415] | 0.9256 | 0.8026 |
| 4 | structure-512-0-ctx · e5-large | 0.7867 | [0.7333, 0.8371] | 0.9421 | 0.7960 |
| 5 | structure-512-0 · bge-m3 *(post-sel.)* | 0.7826 | [0.7273, 0.8359] | 0.9174 | 0.8026 |
| 6 | recursive-512-0 · e5-large | 0.7788 | [0.7303, 0.8251] | 0.9669 | 0.7835 |
| 7 | fixed-512-0 · e5-large | 0.7379 | [0.6808, 0.7926] | 0.9256 | 0.7368 |

**Within the selection grid, e5-large swept the board:** all 15 top nDCG@5 rows were e5-large indexes; the best non-e5 index (`structure-512-0 · minilm`) ranked only 16th at 0.5377, and indobert (the intended weak baseline) filled the bottom of the table. The **post-selection** challenger bge-m3 now interleaves at ranks 3 and 5 (best 0.7744) — it joins the statistical tie but loses the type-robustness tie-breaker (§7.4; details in `bge_m3_robustness_addendum.md`).

### 7.3 The top configs are a statistical tie

From `retrieval_significance.md` — winner vs the nearest e5-large challengers:

| Challenger | Δ (winner−chal.) | Holm p | Significant? |
|---|--:|--:|:--:|
| structure-512-0 (mv) | +0.0029 | 1.000 | no |
| structure-512-0-ctx · bge-m3 *(post-sel.)* | +0.0185 | 1.000 | no |
| structure-512-0-ctx | +0.0218 | 1.000 | no |
| structure-512-0 · bge-m3 *(post-sel.)* | +0.0257 | 1.000 | no |
| recursive-512-0 | +0.0263 | 1.000 | no |
| fixed-512-0 | +0.0654 | 0.383 | no |

(The challenger family is the current top-6 of the 71-index table, so it now
includes the two post-selection bge-m3 configs; before the bge-m3 addition the
top-6 were all e5-large and the same conclusion held.)

**No challenger is beaten at Holm-adjusted p < 0.05.** So the choice rests on the tie-breakers whose order spec §2.6 fixes in advance of any results, not on the raw primary metric.

### 7.4 Tie broken on safety-critical retrieval

Read from `retrieval_metrics_by_type.csv`, question type `risk_suicide_emergency` (n = 17):

| Index | risk recall@5 | risk nDCG@5 | referral recall@5 | worst-type recall@5 |
|---|--:|--:|--:|--:|
| **structure-512-0 · e5-large** | **1.000** | 0.769 | **1.000** | **0.857** |
| recursive-512-0 · e5-large | 1.000 | 0.753 | 0.833 | 0.833 |
| fixed-512-0 · e5-large | 1.000 | 0.686 | 0.333 | 0.333 |
| structure-512-0-ctx · e5-large | 0.941 | 0.706 | 1.000 | 0.810 |
| structure-512-0 · bge-m3 *(post-sel.)* | 1.000 | **0.781** | 0.833 | 0.762 |
| structure-512-0-ctx · bge-m3 *(post-sel.)* | 0.941 | 0.739 | 0.833 | 0.762 |

The winner retrieves a relevant chunk in the top 5 for **all 17** suicide-risk / emergency questions **and** all 6 referral questions, with the highest worst-type floor (0.857) of any tied leader. That is the deciding criterion: within a statistical tie on the primary metric, pick the config that never misses a risk/referral passage and degrades least on its weakest question type. Disclosed for completeness: the post-selection `structure-512-0 · bge-m3` slightly beats the winner on risk-type nDCG@5 (0.781 vs 0.769), but the tie-breaker as pre-stated is recall robustness across types, and there it loses on every count (drops a referral question, floor 0.762 vs 0.857).

**`structure-512-0 · e5-large` is then frozen** as the retriever for the entire Phase 2 chatbot.

---

## 8. Honesty notes / caveats to carry into the write-up

1. **"Recall@k" here = hit-rate@k**, not `|hits|/|relevant|`. State the definition explicitly.
2. **Overlapping-chunk inflation** on precision, recall@1, and MRR — do not compare these across configs of different granularity.
3. **Primary metric is nDCG@5**, frozen in `configs/relevance.yaml` before any index was scored — an implementation-time choice between the spec's two recommendations, **not a formal pre-registration**; the selection argument therefore leans on the use case and full-metric-suite disclosure (`pipeline_audit.md` §4, `phase2_6_selection_notes.md` §3). (The formerly stale "Recall@k (primary)" comment in `metrics.py` was corrected 2026-07-12.)
4. **The top ~6 configs are statistically tied** — the win is a tie-break decision (safety-critical recall), not a significant margin on nDCG@5. Say so.
5. **Selection is robust to the relevance denominator, with one post-bge nuance** — within the original 4-model grid, `sensitivity_gold/` reproduced the pick exactly under the stricter `gold` rule (audit, 2026-07-02). After the post-selection BGE-M3 addition, the `gold`-rule nominal rank-1 is `structure-512-0-ctx__bge-m3` by +0.0017 over the winner — a reordering *inside* the statistically tied top-3 (set unchanged, Spearman 0.84). It does not change the selection: the tie rests on the tie-breakers either way, and that config drops a suicide-risk question (recall@5 = 0.941), so it loses the tie-break under both denominators. Stated here so the sensitivity report and this doc never disagree.
6. **BGE-M3 is a post-selection addition** (2026-07-12, ten days after the selection was frozen). Present it only with that provenance — as a robustness challenge the selection survived — never as part of the original 14 × 4 design (`bge_m3_robustness_addendum.md`).
