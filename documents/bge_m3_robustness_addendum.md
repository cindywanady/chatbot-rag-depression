# BGE-M3 robustness addendum — a fifth embedding model, added after selection

*2026-07-12. Post-selection robustness check: `BAAI/bge-m3` was added to the model
grid ten days **after** the configuration selection was frozen
(`phase2_6_selection_notes.md`, audit `pipeline_audit.md`, both 2026-07-02). It is
therefore NOT part of the Phase-1 14 × 4 = 56-index design and must not be
presented as such. Its purpose is to answer one critique of that design: e5-large
was the only strong general-purpose multilingual retriever among the four models
(MiniLM truncates at 128 tokens, nomic-indonesian is a community finetune,
IndoBERT is the deliberate weak baseline) — so was the win just "best of a weak
field"? Every number below is re-derived from the regenerated
`outputs/analysis/*.csv`.*


> **Gold set: 124 questions (superseded).** Every number in this document was
> computed on the 124-question gold set as it stood before 2026-07-26, when gold
> passage `MI1_0003` and its three questions were removed (121 remain). The
> analysis has **not** been re-run, so read these figures as the record of what
> was measured at the time, not as current values. Where the current set matters,
> `outputs/analysis/retrieval_metrics.csv` is authoritative. The 2026-07-27 re-run
> of the closely-related multi-vector probe moved magnitudes by ~0.01 and changed
> no conclusion, which is the best available guide to the size of the drift here.
## 1. Verdict

**The selection survives.** BGE-M3 is the first genuine challenger — its best
configs join the statistical tie at the top of the leaderboard — but it does not
beat `structure-512-0__e5-large` on the primary metric, and it **loses the
pre-stated first tie-breaker (robustness across question types)**: both of its
top configs drop safety-relevant questions that the winner retrieves. The
deployed retriever (`structure-512-0__e5-large__mv`) is unchanged.

Suggested one-sentence thesis wording: *"BGE-M3, added after configuration
selection as a robustness challenger, joins the statistical tie on nDCG@5 but
drops safety-critical questions under the pre-stated type-robustness
tie-breaker; the selected configuration is unchanged."*

## 2. Setup and provenance

- Spec (`configs/embedding_models.yaml → bge-m3`): `BAAI/bge-m3`, revision
  `5617a9f61b028005a4858fdac845db406aefb181`, dense retrieval (CLS pooling from
  the model's own sentence-transformers config), **no** query/passage prefixes
  (per the model card), dim 1024, max_seq_len 8192, L2-normalised, exact
  `IndexFlatIP` — identical treatment to the other models.
- 14 indexes built 2026-07-12 (manifest `index-20260712-143311`), GPU.
  **Truncation rate 0.0 on all 14 indexes**: the 8192-token window sees every
  chunk in full, including the four atomic dosage blocks that e5 embeds
  truncated — BGE-M3 competes with a representational advantage e5 lacks.
- Full 71-index evaluation re-run (56 original + 1 `__mv` + 14 bge-m3).
  **Integrity: all 57 previously published index rows and all 7,068 per-query
  rows reproduce byte-identically** (checked against a pre-run backup, same
  protocol as audit R3). The addition is purely additive.

## 3. Results

### 3.1 Leaderboard (nDCG@5, top 5 of 71)

| rank | index | nDCG@5 | 95 % CI | recall@5 |
|--:|---|--:|---|--:|
| 1 | structure-512-0 · **e5-large** | **0.7929** | [0.741, 0.840] | 0.952 |
| 2 | structure-512-0 · e5-large (mv) | 0.7901 | [0.737, 0.838] | 0.952 |
| 3 | structure-512-0-ctx · **bge-m3** | 0.7744 | [0.717, 0.828] | 0.911 |
| 4 | structure-512-0-ctx · e5-large | 0.7712 | [0.715, 0.823] | 0.927 |
| 5 | structure-512-0 · **bge-m3** | 0.7672 | [0.708, 0.823] | 0.903 |

Per-model best nDCG@5: e5-large **0.7929** > bge-m3 **0.7744** > MiniLM 0.5377
> nomic-indonesian 0.4861 > IndoBERT 0.3847. The field now contains two strong
models, and e5-large still leads on every core metric at every k, including the
no-truncation common condition (`fixed-128-0`: e5 0.599 vs bge-m3 0.575) — so
the model ranking is not a truncation/window artifact in either direction.

### 3.2 Significance (regenerated `retrieval_significance.md`)

Winner vs the two bge-m3 challengers, per-query nDCG@5, n = 124:

| challenger | Δ (winner−chal.) | 95 % CI of Δ | Wilcoxon p | Holm p | sig? |
|---|--:|---|--:|--:|:--:|
| structure-512-0-ctx__bge-m3 | +0.0185 | [−0.0227, +0.0595] | 0.5181 | 1.0000 | no |
| structure-512-0__bge-m3 | +0.0257 | [−0.0141, +0.0653] | 0.2797 | 1.0000 | no |

Both join the statistical tie at the top (as before, nothing in the top group is
separated at Holm-adjusted p < 0.05).

### 3.3 The pre-stated tie-breaker decides — and it is not a sweep

First tie-breaker (spec §2.6, recorded in `relevance.yaml`): robustness across
question types, especially no collapse on safety-critical questions. From
`retrieval_metrics_by_type.csv`, for the tied leaders:

| index | risk recall@5 | risk nDCG@5 | referral recall@5 | worst-type recall@5 |
|---|--:|--:|--:|--:|
| **structure-512-0 · e5-large** | **1.000** | 0.769 | **1.000** | **0.857** (differential) |
| structure-512-0 · e5-large (mv) | 1.000 | 0.757 | 1.000 | 0.857 (differential) |
| structure-512-0-ctx · bge-m3 | 0.941 | 0.739 | 0.833 | 0.762 (differential) |
| structure-512-0 · bge-m3 | 1.000 | **0.781** | 0.833 | 0.762 (differential) |

Disclosed honestly: `structure-512-0__bge-m3` slightly **beats** the winner on
risk-type nDCG@5 (0.781 vs 0.769) — but the tie-breaker as stated is recall
robustness across types, and there bge-m3 loses on every count: its `-ctx`
variant drops a suicide-risk question (16/17), both variants drop a referral
question (5/6), and both have a markedly lower worst-type floor (0.762 vs
0.857, driven by differential/comorbidity questions). The second and third
tie-breakers (index size/latency, build cost) do not separate the leaders
(both are 1024-dim, 90-vector flat indexes). Selection unchanged.
(Per-type buckets are small — n = 6 for referral — so these per-type numbers
carry wide intervals; that caveat applies equally to every config compared.)

### 3.4 The chunking conclusions replicate in the new model

Within BGE-M3's own 14 configs, the Phase-1 chunking findings reproduce
independently: the two structure-512 configs rank first and second, and mean
nDCG@5 falls monotonically with chunk size (512: 0.718 > 256: 0.644 > 128:
0.617). A second strong retriever reproducing the chunking ordering
strengthens the claim that it is a property of the corpus and strategies, not
of e5-large.

One divergence worth a sentence: context enrichment **helps** BGE-M3
(ctx 0.7744 vs plain 0.7672) while it hurt e5-large (0.7712 vs 0.7929) — the
`-ctx` heading-path prefix is evidently model-dependent, which further supports
having tested it as an ablation rather than assuming it.

### 3.5 Relevance-denominator sensitivity, re-run over 71 indexes

`scripts/retrieval_sensitivity.py` was re-run (all 56 + 448 + 6,944 original
sensitivity rows reproduce exactly in shared columns). Under the frozen
`shorter` rule nothing changes. Under the `gold` (answer-containment) rule the
**nominal** rank-1 becomes `structure-512-0-ctx__bge-m3`, edging the winner by
**+0.0017** (0.7982 vs 0.7965) — a reordering inside the statistically tied
top-3, whose membership is identical under both rules (Spearman 0.839 across
71 indexes). This does not overturn the selection: under either denominator
the top group is a statistical tie decided by the pre-stated tie-breakers, and
`structure-512-0-ctx__bge-m3` drops a suicide-risk question (recall@5 = 0.941)
so it loses that tie-break under both rules. Disclose this alongside the
sensitivity claim rather than quoting the pre-bge "winner unchanged under
gold" line, which described the 4-model pool. (The report's conclusion line in
`retrieval_sensitivity.py` was hard-coded to the invariant case and has been
made conditional so the generated report states this correctly.)

## 4. Where the artifacts stand after this addendum

- `outputs/analysis/*.csv` and `retrieval_eval_report.md` now cover 71 indexes;
  all pre-existing rows are unchanged.
- `retrieval_significance.md` regenerated (winner vs top-6 challengers, which
  now include the two bge-m3 configs).
- `configs/embedding_models.yaml` keeps `bge-m3` in `index.models` so future
  full runs reproduce the shipped 71-row artifacts (config-matches-artifacts);
  the post-selection status is recorded in the stanza comment and here.
- The deployed chatbot retriever is untouched: `structure-512-0__e5-large__mv`
  (`configs/chatbot.yaml`).

## 5. Reproduction

```bash
./.venv/bin/python -m depression_rag.cli_index --models bge-m3 --device cuda
./.venv/bin/python scripts/run_retrieval_eval.py --device cuda
./.venv/bin/python scripts/retrieval_significance.py --top 6
```
