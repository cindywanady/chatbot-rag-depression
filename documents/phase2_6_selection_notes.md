# Phase 1 §2.6 — Retrieval bake-off & configuration selection: design notes

*Companion to `documents/build_spec.md` §2.5 Step 4 + §2.6. Written 2026-07-02.
Covers the cross-strategy relevance mapping, the rank-aware metrics, the
confidence intervals + paired significance, and the justified selection of the
single retrieval configuration. Style matches `phase2_5_gold_eval_notes.md` and
`qa_generation_summary.md`.*

---

## 1. What §2.6 has to produce

Score the **56 candidate indexes** (14 chunk configs × 4 embedding models) on the
depression-specific gold question set and pick **one** (chunking strategy + params
+ embedding model) with a written, defensible justification. The spec's Phase-1
acceptance criteria: a reproducible pipeline, all configs logged, a results table
**with CIs and significance**, a selected configuration + justification, and a
documented relevance-mapping method and threshold.

Inputs, all sharing the `cleaned_text.txt` char coordinate system:

```
data/derived/questions_gold.jsonl    121 validated questions, each tagged passage_ids   (§2.5 Step 3)
data/derived/gold_passages.jsonl     31 relevance units, exact char spans                (§2.5 Step 1)
outputs/indexes/<cfg>__<model>/      56 FAISS indexes + per-chunk char spans             (§2.4)
```

---

## 2. Cross-strategy relevance mapping (§2.5 Step 4)

Each chunking strategy cuts the source at different boundaries, so "the relevant
chunk" differs per index and there is no shared chunk id to score against.
Relevance is therefore defined at the **source-span level** and projected onto
each strategy's chunks by **char-span overlap**, identically for all 56 indexes.

**The frozen threshold** (`configs/relevance.yaml`, declared once and never
changed, spec line 243): a retrieved chunk is relevant to a question if, for
**any** gold passage the question cites,

> overlap(chunk, gold) / min(|chunk|, |gold|) ≥ **0.50**

The denominator is the **shorter** of the two spans (`denom: shorter`), not the
gold span. This is a deliberate deviation from the spec's literal "≥ 50 % of the
gold span," made because the gold passages span two size regimes
(`main_symptoms` ≈ 330 chars … `referral_criteria` ≈ 24,803 chars). Normalising
by gold makes a large gold passage impossible for any single small chunk to
satisfy → recall would be artificially 0 for those concepts. `shorter` is
size-robust and **subsumes the spec's "chunk contains the answer span"
recommendation** as the `min == gold` case (small gold fully inside a large chunk
→ fraction 1.0), while also scoring a small chunk fully inside a large gold
passage as relevant. Switch to `denom: gold` for the spec-literal rule.

**Verification.** On the real chunk/gold spans the mapping is well-behaved: every
one of the **121** questions has ≥ 1 relevant chunk in **every** chunk
configuration — 0 degenerate empty-qrels over all 14 × 121 = 1,694 pairs — and
relevant-chunks-per-query scales with granularity as expected (fixed-128 4.0,
fixed-512-64 1.6, structure-512-0 1.4). Re-derived 2026-07-27 after the
`MI1_0003` removal took the set from 124 to 121; the property survived unchanged.
(Relevance is a function of chunk spans, so it varies per chunk config, not per
index — the 71 indexes collapse to 14 distinct qrel sets.)
Code: `evaluation/relevance.py`
(pure, unit-tested).

---

## 3. Metrics, CIs, significance (§2.6)

**Metrics** (`evaluation/metrics.py`), per query then macro-averaged, at
k ∈ {1, 3, 5, 10}:

* **Recall@k** (primary family): the spec's "is a relevant chunk in the top k" —
  a per-query hit indicator averaged over queries.
* **MRR@k**: reciprocal rank of the first relevant chunk.
* **nDCG@k**: position-aware, binary gains, DCG / ideal-DCG (ideal places all of a
  query's relevant chunks at the top). Written to accept graded gains later.
* **multi-hop coverage@k**: does top-k contain a relevant chunk for **every**
  passage a multi-hop question requires (≥ 2 passages)? For single-passage
  questions this coincides with Recall@k.

**Primary selection metric: nDCG@5** — one of the two the spec *recommends*
("pick one primary metric (recommend nDCG@5 or Recall@5)"). To be precise about
provenance: the spec does **not** declare a primary and none was pre-registered
by the author; nDCG@5 was frozen in `configs/relevance.yaml` when the eval
harness was built (before any of the 56 indexes was scored), as an
implementation-time choice between the spec's two recommendations. Because it is
not a pre-registration, §4.4 discloses the full metric suite and §5.5 argues the
selection from the downstream use case rather than from this metric alone.
Tie-breakers, in order (spec §2.6): robustness across question types (esp. no
collapse on safety-critical suicide-risk) → index size / query latency → build cost.

**CIs.** Percentile bootstrap over the 121 queries (10 000 resamples, seed
20260628 → reproducible), reported per index on the primary metric.

**Significance.** Paired bootstrap on the per-query difference between two indexes
(same questions; pairing removes per-question difficulty as a nuisance). Reported
as Δ, its 95 % CI, and a two-sided bootstrap p. Code:
`evaluation/bootstrap.py` (unit-tested); driver `scripts/retrieval_significance.py`.

---

## 4. Results

56 indexes scored on 121 questions. Full table: `outputs/analysis/retrieval_metrics.csv`;
per-type: `..._by_type.csv`; per-query: `..._scores_per_query.csv`; ranked report:
`retrieval_eval_report.md`.

### 4.1 The model axis dominates the chunking axis

| model | best nDCG@5 | mean nDCG@5 (14 cfgs) |
|---|--:|--:|
| **e5-large** (multilingual-e5) | **0.798** | 0.701 |
| minilm | 0.548 | 0.453 |
| nomic-indonesian | 0.490 | 0.387 |
| indobert (weak baseline) | 0.394 | 0.302 |

**e5-large holds all top 14 rows — within the 4-model selection grid this table
describes.** State the scope: across all 71 indexes scored since, the
post-selection challenger BGE-M3 takes 5 of the top 14, so the unqualified
sentence is false outside the grid. (This is the same scoping trap as the
sensitivity claim in §5; both are true of the grid and not of the full set.)
IndoBERT lands last as designed (it is not a sentence-embedding model — the
intentional weak baseline; confirmed weak). Model choice moves nDCG@5 by ~0.4;
chunking moves it by ~0.1.

### 4.2 Within e5-large, larger chunks win; overlap / recursive-vs-fixed barely move it

Mean nDCG@5 by nominal chunk size (e5-large): **512 → 0.761**, 256 → 0.680,
128 → 0.633. Overlap (0 vs 64) and fixed-vs-recursive are within noise.

### 4.3 Top of the table (nDCG@5, 95 % CI)

| rank | index | nDCG@5 | 95 % CI | recall@5 |
|--:|---|--:|---|--:|
| 1 | **structure-512-0__e5-large** | **0.798** | [0.747, 0.846] | 0.950 |
| 2 | structure-512-0-**ctx**__e5-large | 0.787 | [0.733, 0.837] | 0.942 |
| 3 | recursive-512-0__e5-large | 0.779 | [0.730, 0.825] | 0.967 |
| 4 | fixed-512-0__e5-large | 0.738 | [0.681, 0.793] | 0.926 |

### 4.4 The selection is metric-contingent — other metrics favour other configs

nDCG@5 is the adopted primary (§3), but the full metric suite is not a clean sweep
for one config. Best e5-large config **per metric**:

| metric | winning config | value |
|---|---|--:|
| recall@1 = nDCG@1 | recursive-128-64 | 0.769 |
| recall@3 | recursive-128-64 | 0.926 |
| **recall@5** | recursive-512-0 | 0.967 |
| recall@10 | recursive-512-0 | 1.000 |
| **MRR@5 / MRR@10** | recursive-128-64 | 0.846 / 0.848 |
| nDCG@3 | structure-512-0-**ctx** | 0.761 |
| **nDCG@5** (primary) | **structure-512-0** | **0.798** |
| nDCG@10 | structure-512-0-**ctx** | 0.813 |

**This got sharper on the 121-question set, and in a way that cuts against the
selection.** On 124, `structure-512-0` won the entire nDCG family; it now wins
**only nDCG@5** — the declared primary. The `-ctx` variant takes nDCG@3
(0.7612 vs 0.7598) and nDCG@10 (0.8130 vs 0.8127), both by margins far inside any
plausible noise band. `recursive-512-0` now takes recall@10 outright at 1.000,
where it was previously an exact tie at 0.992.

The honest reading: the winner is the winner *on the frozen primary metric*, where
its margin is real (0.798 vs 0.787), and it is at worst indistinguishable on the
neighbouring cutoffs. It is not dominant across the suite, and this section should
not be quoted as if it were. The rank-1 / MRR metrics still go to the small-chunk
`recursive-128-64`, and recall@5 to `recursive-512-0` (0.967 vs structure's 0.950).
(At k=1, recall@1 and nDCG@1 coincide by definition — both reduce to "is rank-1
relevant".)

This is a real **small-vs-large-chunk tension**: small (128-token) chunks are more
*precise* — a hit is tightly on-target so it ranks first → best recall@1 / MRR;
large (512-token) structure chunks carry *more of the answer* per chunk → best nDCG
(relevant mass ranked high) and strong recall@5. Part of the small-chunk rank-1/MRR
advantage is also **mechanical**: overlap + small chunks manufacture near-duplicate
relevant targets (mean relevant chunks per query: 10.0 for `recursive-128-64` vs
1.4 for the 512-0 configs), and hit-style metrics credit a hit on *any* of them
(`pipeline_audit.md` §5.3). §5.5 explains why the selection survives this; full
suite in `outputs/analysis/retrieval_metrics.csv`.

> **Caveat on strategy-family means.** Aggregating by strategy (e5-large: structure
> recall@5 0.940 / nDCG@5 0.782 vs recursive 0.911 / 0.676 vs fixed 0.911 / 0.667)
> makes `structure` look dominant — but it has only **2 configs (both 512-token)**
> while the fixed/recursive means include the weaker 128/256 sizes, so the family
> comparison is confounded with chunk size. Compare like-for-like at 512 (§4.3).

### 4.5 The selection is invariant to the relevance denominator

`relevance.yaml` uses `denom: shorter` (§2), a deliberate size-robust deviation
from the spec-literal "≥ 50 % of the gold span" (`denom: gold`, ≈ DPR-style
answer-containment). To show this choice does not drive the result, every index was
re-scored under `configs/relevance_gold.yaml` (denom = gold) and the rankings
compared (`scripts/retrieval_sensitivity.py`):

* **winner unchanged on the selection grid** — `structure-512-0__e5-large`
  ranks 1 under both (nDCG@5 0.793 shorter → 0.797 gold);
* **top-3 set identical** (the tied contenders are the same);
* **Spearman rank correlation 0.823** across the 57 selection-grid indexes
  (0.839 across all 71).

  Scope matters here and the report now prints both. Including the
  post-selection BGE-M3 challenger, `structure-512-0-ctx__bge-m3` takes nominal
  rank 1 under `denom=gold` by **0.0017** — inside the statistical tie, and it
  loses the type-robustness tie-breaker (recall@5 0.941 on suicide-risk vs
  1.000). It is not part of the grid the selection was made on, so it does not
  bear on the pick; quote it alongside the claim rather than omitting it
  (`bge_m3_robustness_addendum.md`).

The denominator shifts absolute scores and mid-table ordering (mean |Δ nDCG@5|
≈ 0.11 per index; ranks 4–6 reshuffle) but **not** the pick or the tied top set.
Full comparison: `outputs/analysis/sensitivity_gold/sensitivity_report.md`.

---

## 5. Selection — `structure-512-0__e5-large`

### 5.1 The primary metric does **not** separate the top three (state this plainly)

Ranks 1–3 have overlapping 95 % CIs, and the paired bootstrap of rank-1 vs
rank-2 / rank-3 is **non-significant**: Δ = +0.022 (p = 0.25) vs `-ctx`,
Δ = +0.026 (p = 0.31) vs `recursive-512-0`. Against rank ≥ 4 the *uncorrected*
paired bootstrap rejects (fixed-512-0: Δ = +0.065, p = 0.039; recursive-512-64
p = 0.026; fixed-512-64 p = 0.020) — but those p-values are per-pair and not
adjusted for the multiple comparisons in `retrieval_significance.md`. Applying a
Holm correction to that table's six p-values, only the fixed-256-64 comparison
survives (0.0002 × 6 = 0.0012; the next-smallest becomes 0.020 × 5 = 0.102). The
statistically safe claims are therefore: the 512-token e5-large cluster clearly
beats the 128/256-token configs, and **the top ranks are a statistical tie** — so
the ordered tie-breakers decide, **not** the metric point estimate.

### 5.2 Tie-breaker 1 (robustness across question types) selects structure-512-0

This is the highest-priority tie-breaker and the one the spec weights toward
safety. Per-type nDCG@5 / recall@5 for the tied three:

| | structure-512-0 | structure-512-0-ctx | recursive-512-0 |
|---|--:|--:|--:|
| **suicide-risk recall@5** (n=17) | **1.00** | 0.94 ⚠ | 1.00 |
| suicide-risk nDCG@5 | **0.769** | 0.706 | 0.753 |
| worst-type nDCG@5 (floor) | **0.689** | 0.559 | 0.560 |
| mean-over-types nDCG@5 | **0.809** | 0.776 | 0.761 |
| referral_criteria recall@5 (n=6) | 1.00 | 1.00 | **0.83 ✗** |

Two decisive facts: `structure-512-0-ctx` **drops a safety-critical suicide-risk
question out of top-5** (recall@5 = 0.94 ≈ 1 miss of 17), which the spec's
"especially no collapse on safety-critical risk questions" treats as
disqualifying; and `recursive-512-0` **collapses on referral** (recall@5 = 0.83)
and has the lowest floor + mean. `structure-512-0` has the highest floor, the
highest mean, and perfect safety recall — it wins tie-breaker 1 outright, before
size or cost are even consulted.

### 5.3 Tie-breakers 2–3 reinforce (do not contradict)

* **Index size / latency.** `structure-512-0` and `-ctx` are identical (90
  chunks); `recursive-512-0` is smaller (52). Recursive would win this tie-breaker
  — but it already lost the higher-priority robustness one, so ordering resolves it.
* **Build cost / simplicity.** `structure-512-0` is **not** context-enriched;
  `-ctx` adds a heading-path enrichment step (more cost, more moving parts) for a
  *lower* score. So `structure-512-0` **strictly dominates its nearest
  competitor**: same index size, simpler + cheaper build, higher point estimate,
  and better safety robustness.

### 5.4 Selected configuration

**`structure-512-0__e5-large`** — structure-aware chunking, 512-token target, 0
overlap, no context enrichment, encoded by `intfloat/multilingual-e5-large`.

Full metrics: recall@{1,3,5,10} = 0.669 / 0.895 / 0.952 / 0.984; MRR@5 = 0.786;
nDCG@{1,3,5,10} = 0.669 / 0.756 / 0.793 / 0.807; nDCG@5 95 % CI [0.741, 0.840].
Balanced across difficulty (nDCG@5: factoid 0.788, applied_case 0.817, multi_hop
0.810 — persisted in `retrieval_metrics_by_difficulty.csv`) and multi-hop
coverage@5 = 0.933 (14 / 15 multi-hop questions retrieve all required passages
in top-5).

### 5.5 Why the pick survives the cross-metric view (metric choice + use case)

§4.4 shows `structure-512-0` does not win every metric — `recursive-128-64` wins
rank-1 / MRR. The selection nonetheless stands, for two reasons:

1. **nDCG@5 was frozen in `configs/relevance.yaml` before any index was scored,
   and it is one of the spec's two recommended primaries.** It was *not*
   pre-registered by the author (§3) — the spec only says "recommend nDCG@5 or
   Recall@5" — so this argument cannot carry the selection alone. But switching
   metrics *after* seeing that a different one favours another config would be a
   goalpost-shift; the honest posture is to keep the frozen choice, disclose the
   full suite (§4.4), and let reason 2 (the use case) do the real work.
2. **The downstream use is RAG-into-generator, not single-passage display.** Phase
   2 feeds the top-k retrieved chunks into Gemma-3-12B, which reads *all* of them —
   so the operative quantity is "is a relevant chunk in the top-k I pass to the
   generator?" (**recall@k**), not "is it at rank 1?" (**MRR / recall@1**). The
   rank-1 metrics that `recursive-128-64` wins matter *least* for this architecture.
   On the RAG-relevant recall@5, `structure-512-0` (0.952) is a near-tie with the
   leader `recursive-512-0` (0.960) — no meaningful recall is given up — and its
   larger chunks carry more answer context per retrieved unit, which suits a generator.

**Honest contingency (state it in the thesis).** The selection is contingent on
(a) nDCG@5 as the adopted primary (an implementation-time choice among the spec's
two recommendations, not a pre-registration) and (b) the top-k-into-generator use case. Were
the product instead to surface a *single* passage to the user, rank-1 quality would
dominate and `recursive-128-64` (best MRR@5 = 0.826, recall@1 = 0.750, small precise
chunks) would be the better pick. That is not the stated architecture. Claiming the
pick is *contingent and checked* is stronger than claiming it won every metric.

---

## 6. The `-ctx` ablation (context enrichment)

`structure-512-0` and `structure-512-0-ctx` produce **identical chunks** (same
boundaries, text, char offsets → both 90 chunks). The only difference is at
**embed time** (`indexing_pipeline._encoder_inputs`): for `-ctx`, the chunk's
`heading_path` is prepended to the passage before encoding —
`"MI.4 > Pokok Bahasan B > DEPRESI\n<passage>"` — document-side only (queries are
never enriched); stored text + offsets are untouched, so the relevance mapping is
byte-identical between the two. It is a clean ablation of "does section context in
the passage vector help?"

**On this corpus it did not** — it slightly hurt (nDCG@5 0.771 vs 0.793) and cost
a suicide-risk hit (recall@5 0.94 vs 1.00). Plausibly e5-large already encodes
each passage's topic, so a boilerplate heading path adds mostly redundant tokens
that dilute rather than sharpen the vector. This is a citable finding: context
enrichment provides no benefit here and a small safety cost — which is itself part
of the justification for the simpler non-enriched config.

---

## 7. Honest limitations

* **The choice is a tie-break, not a significant win** over ranks 2–3. Reported as
  such; do not claim `structure-512-0` "beat" them on the metric.
* **Small per-type buckets** (referral n=6, special_populations n=11) make per-type
  nDCG noisy; the selection anchors on the safety bucket (n=17, a concrete
  1-question miss) and the consistent floor/mean pattern, not any single small bucket.
* **Question set accepted whole at §2.5 Step 3** (124 candidates, mechanical
  validation only, no per-item clinical adjudication). The metrics are only as
  strong as that set; flagged in `qa_generation_summary.md` §8. Three were removed
  on 2026-07-26 with gold passage `MI1_0003` — a WHO mhGAP Master Chart table that
  PyMuPDF flattened across two conditions — leaving **121**. That defect was found
  by the content-type audit on 2026-06-30 and missed by the Stage-2 gold audit,
  which is exactly the gap this bullet warns about.
* **Difficulty skew** in the set (97 / 121 factoid) means factoid retrieval
  dominates the aggregate; the per-difficulty breakdown is reported to expose this.
* **No inter-annotator reliability** on gold labels (single annotator for the
  Claude Stage-2 gold assignment); noted in `phase2_5_gold_eval_notes.md`.
* **Not a universal winner across metrics, and less so on 121 than on 124.**
  `structure-512-0` now wins **only nDCG@5** — its declared primary — having lost
  nDCG@3 and nDCG@10 to the `-ctx` variant by 0.0014 and 0.0003. It does not win
  rank-1 / MRR (`recursive-128-64`), recall@5 or recall@10 (`recursive-512-0`,
  0.967 and 1.000). The selection rests on the adopted primary (nDCG@5, §3) plus
  the top-k-into-generator use case (§5.5) — contingent, not absolute. (It *is*
  robust to the relevance denominator, §4.5.)
* **Four atomic dosage blocks embed truncated in the selected config.** e5-large
  clips its input at 512 tokens, and structure-aware chunking deliberately keeps
  tightly-coupled units whole, so `MI7_0071` (1,165 e5 tokens), `MI7_0072` (1,074),
  `MI4_0049` (967) and `MI4_0050` (818) — all `dosage` content — are embedded from
  roughly their first half (7 further chunks exceed 512 by only 2–4 tokens from
  the prefix + special-token overhead). The stored chunk *text* is complete — only
  the vector is truncated — and the measured harm is small but not zero:
  11 of the 12 questions whose only relevant chunk is one of these blocks are
  still retrieved in the top-5 (pharmacotherapy_dosing remains the second-best
  type: recall@5 = 0.957, nDCG@5 = 0.878), but the one miss (`dep_0085`, the
  dosing type's single recall@5 miss) is retrieved in the top-5 by 12 of the
  other 13 e5-large configs, i.e. an isolated 1/121 cost of the oversized
  single-vector representation, priced into the winner's published numbers.
  Details: `phase2_embedding_indexing_notes.md` §5 (truncation sources),
  `pipeline_audit.md` §5.2 (inventory + per-query analysis).
* **Qrel sizes differ mechanically across configs.** Overlapping / small-chunk
  configs turn one gold span into many near-duplicate relevant chunks (mean
  relevant per query 10.0 for `recursive-128-64` vs 1.4 for the 512-0 configs), so
  hit-style metrics (recall@1, MRR, precision@k) are inflated for redundant
  configs; nDCG partially counteracts this via its ideal-DCG normalisation. This
  is one more reason the selection does not rest on the rank-1/MRR metrics
  (`pipeline_audit.md` §5.3).

---

## 8. Artifacts & how to reproduce

| file | role |
|---|---|
| `configs/relevance.yaml` | frozen threshold + primary metric + k values |
| `configs/relevance_gold.yaml` | denom=gold sensitivity variant (spec-literal / DPR-style) |
| `src/depression_rag/evaluation/relevance.py` | source-span → chunk relevance mapping |
| `src/depression_rag/evaluation/metrics.py` | Recall@k / Precision@k / MRR@k / nDCG@k / multi-hop coverage |
| `src/depression_rag/evaluation/bootstrap.py` | percentile CI + paired bootstrap significance |
| `src/depression_rag/evaluation/retrieval.py` | index + embedder retrieval harness |
| `scripts/run_retrieval_eval.py` | score all 56 indexes → metrics + CIs + per-query dump |
| `scripts/retrieval_significance.py` | paired significance, winner vs challengers |
| `scripts/retrieval_sensitivity.py` | re-score under denom=gold + compare rankings (robustness) |
| `tests/test_relevance_metrics.py`, `tests/test_bootstrap.py` | unit tests (pure core) |
| `outputs/analysis/retrieval_metrics{,_by_type,_by_difficulty}.csv` | results tables |
| `outputs/analysis/retrieval_scores_per_query.csv` | per-query scores (paired tests) |
| `outputs/analysis/retrieval_eval_report.md` | ranked table + CIs |
| `outputs/analysis/retrieval_significance.md` | paired significance report |
| `outputs/analysis/sensitivity_gold/` | denom=gold metrics + `sensitivity_report.md` |

```bash
# score all 56 indexes (GPU), the paired significance, and the denominator check
python scripts/run_retrieval_eval.py --device cuda
python scripts/retrieval_significance.py --top 6
python scripts/retrieval_sensitivity.py --device cuda   # re-score denom=gold + compare
```

Deterministic given the built indexes: query encoding reuses the exact Phase-2
embedder (prefixes, pooling, normalisation), FAISS is exact (`IndexFlatIP`), and
the bootstraps are seeded.

---

## 9. Status / next

* §2.6 complete: relevance mapping + frozen threshold, metrics with CIs +
  significance, selected config + justification, all logged. **Done.**
* **Selected for Phase 2: `structure-512-0__e5-large`.**
* Next (Phase 2): build the deployed chatbot on this configuration — Gemma-3-12B
  answer generation over the selected retriever; RAGAS reference-free evaluation
  (the gold passages can be reused as RAGAS reference contexts).
