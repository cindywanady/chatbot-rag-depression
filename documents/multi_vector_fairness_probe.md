# Multi-vector fairness probe — is `__mv` an advantage handed only to the winner?

*Re-run 2026-07-27 on the current **121-question** gold set (the original run was
2026-07-26 on 124, before gold passage `MI1_0003` and its three questions were
removed). Every number below is produced by `scripts/mv_fairness_probe.py`
(`--out <scratch> --device cpu`) and re-derived from the artifacts, not quoted
from prose. The probe writes nothing into `outputs/indexes/` — see §6.*

**The conclusion did not change**: multi-vector helps exactly the models whose cap
was binding, e5-large is unaffected by it, and giving every cap-limited model the
same fix does not overturn the selection. Only the magnitudes moved (MiniLM's
gain +0.113 → +0.119; the winner's margin over the best mv challenger
+0.143 → +0.131).

---

## 1. The question

The deployed retriever is `structure-512-0__e5-large__mv`: the selected
configuration with oversize chunks embedded as several ≤cap windows instead of
being clipped at `max_seq_len`. No other model has an `__mv` index. That invites
a fair challenge in a viva:

> You applied a representation fix to your winner and not to the models it beat —
> and the model it beat by most, MiniLM, is the one truncation hurts most.

Before this probe the answer was an argument. It is now a measurement.

## 2. For two of the four other models the question dissolves

`__mv` only does anything when a chunk's encoding exceeds the model's cap. On the
selected chunk config (`structure-512-0`, 90 chunks, longest ≈ 1,200 reference
tokens):

| model | `max_seq_len` | truncated chunks | would `__mv` change anything? |
|---|--:|--:|---|
| bge-m3 | 8192 | **0 / 90** | **no** — byte-identical index |
| nomic-indonesian | 8192 | **0 / 90** | **no** — byte-identical index |
| e5-large | 512 | 11 / 90 (12.2 %) | yes — built, and deployed |
| indobert | 512 | 4 / 90 (4.4 %) | yes |
| minilm | 128 | **60 / 90 (66.7 %)** | yes |

The two long-context models cannot benefit: nothing they encode is clipped. The
probe derives this split from each index's own `meta.json` rather than assuming
it, and prints it before building anything:

```
[probe] __mv is a NO-OP for (nothing exceeds the cap): ['bge-m3', 'nomic-indonesian']
[probe] cap-limited, __mv is meaningful for:           ['e5-large', 'indobert', 'minilm']
```

So the real question is only about **IndoBERT and MiniLM**, and it is answerable
by building the two missing variants.

## 3. What `__mv` buys each cap-limited model

Paired over the same 121 gold questions, nDCG@5, mv − single-vector:

| model | Δ nDCG@5 | 95 % CI | Wilcoxon *p* | r_rb | truncated |
|---|--:|---|--:|--:|--:|
| **minilm** | **+0.1191** | [+0.0528, +0.1880] | **0.0011** | +0.483 | 60/90 |
| indobert | +0.0197 | [+0.0056, +0.0372] | 0.028 | +0.929 | 4/90 |
| e5-large | −0.0029 | [−0.0124, +0.0062] | 0.42 (n.s.) | −0.400 | 11/90 |

**The gain tracks truncation damage**, which is the mechanism behaving as
designed. MiniLM, clipped on two thirds of its chunks, gains most; IndoBERT
gains a little but almost every non-tied query moves the same way (r_rb = +0.93
on a small number of affected queries); e5-large gains nothing measurable on
nDCG@5.

For e5-large the deployment rationale was never nDCG@5 — it was tail coverage,
and that still holds: recall@10 goes 0.9835 → **0.9917**, recovering `dep_0085`
(the one question whose only relevant chunk is the 967-token `MI4_0049`). The
−0.003 nDCG@5 is noise, not a cost.

## 4. Does the selection change? No.

Full table, all six indexes on the selected chunk config:

| index | nDCG@5 | recall@5 | recall@10 | MRR@5 |
|---|--:|--:|--:|--:|
| **e5-large** | **0.7980** | 0.9504 | 0.9835 | 0.7942 |
| e5-large__mv *(deployed)* | 0.7951 | 0.9504 | **0.9917** | 0.7956 |
| minilm__mv | 0.6666 | 0.8678 | 0.9504 | 0.6804 |
| minilm | 0.5475 | 0.7190 | 0.8512 | 0.5548 |
| indobert__mv | 0.3115 | 0.4959 | 0.6281 | 0.3255 |
| indobert | 0.2918 | 0.4628 | 0.6281 | 0.3132 |

Best non-winner after the fix is `minilm__mv`. Paired against the selection:

| comparison | Δ nDCG@5 | 95 % CI | Wilcoxon *p* | r_rb |
|---|--:|---|--:|--:|
| e5-large − minilm__mv | +0.1314 | [+0.0699, +0.1975] | **5.0 × 10⁻⁵** | +0.553 |
| e5-large__mv − minilm__mv | +0.1285 | [+0.0659, +0.1946] | **1.0 × 10⁻⁴** | +0.529 |

Under the fairest available framing — give every cap-limited model the same fix,
then compare like with like — the selected configuration still wins, with a
confidence interval nowhere near zero.

## 5. The tie-breaker holds by a wider margin than the headline metric

The selection rests on type robustness before absolute score. Recall@5 by
question type:

| question_type | n | e5-large | e5-large__mv | minilm__mv |
|---|--:|--:|--:|--:|
| diagnostic_criteria | 13 | 1.000 | 1.000 | 0.846 |
| differential_comorbidity | 20 | 0.850 | 0.850 | 0.800 |
| pharmacotherapy_dosing | 23 | 0.957 | 0.957 | 0.913 |
| psychoeducation | 21 | 0.952 | 0.952 | 0.905 |
| referral_criteria | 6 | 1.000 | 1.000 | 0.667 |
| **risk_suicide_emergency** | 17 | **1.000** | **1.000** | **1.000** |
| special_populations | 11 | 1.000 | 1.000 | 0.909 |
| symptom_recognition | 10 | 0.900 | 0.900 | 0.700 |

**Worst-type floor** — the quantity the tie-breaker actually compares:

```
e5-large      0.850
e5-large__mv  0.850
minilm__mv    0.667
```

`minilm__mv` ties on the safety-critical slice (`risk_suicide_emergency`, n = 17)
but loses every other type, falling furthest on `referral_criteria` (0.667, n = 6)
and `symptom_recognition` (0.700, n = 10). The margin on the tie-breaker (0.183)
is still larger than the margin on the primary metric (0.131), though on the
121-question set the two are closer than they were on 124, where the tie-breaker
margin was roughly double. Both point the same way; neither is load-bearing
alone.

## 6. Why the probe writes to scratch

`run_retrieval_eval.py` discovers **every** directory under `--index-dir`. Two
extra `__mv` indexes left in `outputs/indexes/` would silently join the published
results table — the same mixing that already makes the relevance-denominator
sensitivity report read `winner unchanged: NO` once the post-selection BGE-M3
indexes are included (`pipeline_audit_20260726.md` §6.2). The probe therefore
refuses an `--out` inside `outputs/indexes/`, and copies the baselines it needs
into scratch so all six indexes are scored by one run.

Sanity check performed every run: the four copied indexes reproduce their live
numbers exactly, so the comparison is like-for-like and not an artefact of
re-scoring on CPU.

## 7. What this does and does not license

**Does.** The claim that `__mv` is a post-selection deployment variant rather
than a thumb on the scale is now measured, for every model where the variant is
not a no-op, on the configuration that was selected.

**Does not.** The probe covers **one chunk config** — `structure-512-0`, the
selected one. It says nothing about what `__mv` would do to each model's own best
config. That is the right scope for the fairness question (which is about the
deployed configuration) but it is not a general claim about multi-vector
indexing.

**Consequence for the write-up.** `README.md` §7 states that e5-large's 0.793 is
"~0.26 above the next model's best (MiniLM 0.538)". Under multi-vector, MiniLM's
best on this config becomes 0.651 and the true gap is **~0.14**. The headline
figure attributes part of the margin to a representation limitation rather than
to the model. This is now disclosed in `README.md` §8 (Limitations); the
selection is unaffected either way.

## 8. Reproducing

```bash
./.venv/bin/python scripts/mv_fairness_probe.py --out /tmp/mv_probe --device cpu
```

The script derives the no-op/cap-limited split from `meta.json`, builds only the
missing variants, scores everything in one run, and prints §3–§4 directly.
Runtime ≈ 5 min on CPU. Delete `--out` when done.
