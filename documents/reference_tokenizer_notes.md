# Reference Tokenizer — Notes & Justification

Working notes on *why* the chunking pipeline uses one fixed reference tokenizer
(`intfloat/multilingual-e5-large`, the XLM-RoBERTa SentencePiece tokenizer),
how it is used in each chunking strategy, and how to defend the choice in the
thesis. All numbers below were measured on this project's actual corpus
(MI.1/2/4/7/8, 126,979 cleaned chars) and are regenerable.

---

## 0. TL;DR

- The reference tokenizer is a **measuring stick** for `chunk_size` — nothing
  more. It never touches the embeddings; each embedding model still encodes
  with its own tokenizer in Phase 2.
- We fix **one** tokenizer so that "256 tokens" means **one constant amount of
  text** across every config. Otherwise the chunk-size factor and the
  embedding-model factor get confounded.
- We pick **e5/XLM-R** because it is spec-recommended, multilingual /
  Indonesian-aware, shared identically by 2 of the 4 candidate embedders, makes
  MiniLM's 128-token limit exact, and is a fast tokenizer with char offsets.
- It is **configurable** (`chunking.reference_tokenizer` in
  `configs/pipeline.yaml`) but should stay fixed within an experiment.

---

## 1. Why fix ONE reference tokenizer at all (the principle)

The bake-off varies `(chunking strategy × params × embedding model)`. If each
model defined `chunk_size` with its own tokenizer, "256" would be **four
different chunk corpora**, and a model's retrieval score would confound
*embedding quality* with *how much text its tokenizer put in each chunk*. Fixing
one reference removes that variance (same logic the spec uses for picking exact
FAISS search over approximate — "removes one source of variance").

**Evidence (same 126,979-char corpus, measured two ways):**

| measured by | corpus tokens | chars/token | ≈ chunks @256 | text per 256-tok chunk |
|---|---|---|---|---|
| e5 / MiniLM (XLM-R) | 26,152 | 4.86 | ~102 | ~1,243 chars |
| IndoBERT (WordPiece) | 22,894 | 5.55 | ~89 | ~1,420 chars |

→ A "256-token" IndoBERT chunk holds **~14% more actual text** than a "256-token"
e5 chunk (≈ an e5 ~290-token chunk). Same label, different slice.

---

## 2. What "256 tokens = the same amount of text in every config" means

A **token is a tokenizer-specific sub-word piece**, not a fixed unit. The same
word splits differently per tokenizer, e.g. **Sekurang**:
- e5 → `▁Se`, `ku`, `rang` (3 pieces)
- IndoBERT → `sekur`, `##ang` (2 pieces)

So "the first 8 tokens" of one phrase covers different text:
- e5 → `Sekurang-kurangnya 2 dari 3`
- IndoBERT → `sekurang - kurangnya 2 dari 3 gejala` (one word further)

**The fix:** cut the document into chunks **once**, with one ruler, then give
those identical chunks to all models. Every model embeds byte-for-byte the same
chunks, so the only thing varying between configs is the model.

**Analogy:** to compare 4 microscopes, put the **same slide** under all 4. The
reference tokenizer = the tool that prepares the slides (chunks); the embedding
models = the microscopes.

---

## 3. Why `multilingual-e5-large` specifically (5 arguments)

1. **Spec-recommended / conventional.** The build spec names "the XLM-RoBERTa
   tokenizer that e5 uses"; XLM-R is the de-facto multilingual subword standard.
2. **Most neutral ruler for this grid.** Shared *identically* by 2 of the 4
   models — e5 **and** multilingual-MiniLM (verified: same 250,002-token vocab,
   identical token counts). It is the common denominator, not one model's quirk.
3. **Indonesian-aware, not English-biased.** XLM-R was trained on 100 languages
   incl. Indonesian, so its token is a fair proxy for "amount of Indonesian
   text." A monolingual-English tokenizer would over-fragment Indonesian.
4. **Makes the fairness condition exact.** The spec's "128-token common
   no-truncation condition" is *exact* for MiniLM (it shares this tokenizer),
   not approximate — MiniLM truncates hard at 128.
5. **Practical + reproducible.** Fast tokenizer with char-offset mapping (the
   pipeline needs offsets), no PyTorch dependency, and its exact commit hash is
   pinned in every run manifest.

**Tokenizer comparison (verified):**

| model | tokenizer class | vocab | tokens on a sample sentence |
|---|---|---|---|
| e5-large (reference) | `XLMRobertaTokenizerFast` | 250,002 | 32 |
| minilm | `PreTrainedTokenizerFast` (XLM-R vocab) | 250,002 | **32** (same as e5) |
| indobert | `BertTokenizerFast` (WordPiece) | 30,521 | 28 |
| nomic-indonesian | different BPE | — | — |

---

## 4. What e5-large actually produces (the raw material)

XLM-R SentencePiece, 250,002-vocab, `▁` marks word starts, splits Indonesian
into sub-words, char offset per token. Example:

```
"Tidak bertenaga dan mudah lelah sehingga aktivitas menurun."
-> ['▁Tidak','▁berte','naga','▁dan','▁mudah','▁le','lah','▁sehingga','▁aktivitas','▁menurun','.']
   (bertenaga -> ▁berte+naga ; lelah -> ▁le+lah ; offsets: ▁berte=(6,11), naga=(11,15))
```

Every chunker only ever gets, from e5, either this full `(subword, start, end)`
list (`offsets`) or its length (`count`).

---

## 5. How the reference is used in EACH chunking strategy

Two operations: `offsets(text)` (per-token char spans) and `count(text)` (#tokens).

**Call tally on the real corpus:**

| strategy | `count()` calls | `offsets()` calls | tokenizer's role | who places boundaries |
|---|---|---|---|---|
| fixed | 0 | 1 (whole corpus) | defines **every** boundary (token grid) | the tokenizer |
| recursive | ~4,577 | 0 | size **budget** for split/pack | separators (¶/sentence/word) |
| structure | ~2,773 | 0 | size **gate**: split-or-keep per segment | document structure (segments) |

- **Fixed** — tokenizes the whole text once; the chunk grid *is* e5's sub-word
  grid. Boundaries land on e5's sub-word edges → **can cut mid-word** (22 of 102
  boundaries did; e.g. *penapisan* → `…pena` | `pisan…`). Most sensitive to the
  tokenizer choice.
- **Recursive** — cuts on separators (never mid-word: 0 cuts), uses `count()`
  only to decide what fits in the budget and to size overlap. Moderately
  dependent.
- **Structure** — boundaries from the document hierarchy; `count()` only asks
  per segment "is this section > size?". **Atomic units bypass it entirely**
  (kept whole even if oversize). Least dependent.

**Degree of dependence:** fixed (total) > recursive (partial) > structure (marginal).

### Atomic interaction (structure only)
An `atomic` segment (criteria / dosage / referral_criteria / somatic_symptoms,
when compact) is emitted as **one whole chunk**, skipping the token size-gate.
Demo — the 207-token criteria block at target size 128:
`fixed → 3 pieces`, `recursive → 2 pieces`, `structure → 1 whole chunk`.

---

## 6. What does NOT depend on the reference tokenizer

- The **embeddings**: every model encodes with its own tokenizer at embed time.
- **Per-model truncation rate**: computed in Phase 2 against each model's real
  `max_seq_len` (e5/IndoBERT 512, nomic ~2048, MiniLM 128) — logged separately.

So the reference choice can neither flatter nor penalize any model's
representation; it only sets what "256/512" physically means.

---

## 7. Justification for the thesis (defense structure)

1. **Principle:** a single fixed reference is required for internal validity
   (no confound between chunk-size and model).
2. **Specific choice:** e5/XLM-R is the neutral, Indonesian-aware,
   MiniLM-exact, spec-recommended ruler (Section 3).
3. **Concede the limit:** a reference is a *proxy* — "256" is exact for
   e5/MiniLM, approximate for IndoBERT/nomic; it is measurement-only.
4. **Robustness evidence (sensitivity analysis):** re-chunking with IndoBERT as
   the reference changes boundaries as follows —

   | strategy | boundary agreement vs e5 reference |
   |---|---|
   | fixed-256-0 | ~1% |
   | recursive-256-0 | ~3% |
   | structure-512-0 | **~74%** |

   The clinically-motivated, likely-winning **structure-aware** strategy is
   largely invariant; and because the reference is held fixed across models, it
   never confounds the model comparison regardless.
5. **Decisive check (do once Phase 2 exists):** re-run the bake-off with an
   alternative reference and show the **winning `(chunking × model)` config is
   unchanged**.

### Draftable methods paragraph
> "Token-based chunk sizes were measured with a single fixed reference
> tokenizer, the XLM-RoBERTa SentencePiece tokenizer used by
> `multilingual-e5-large`, so a nominal size denotes a constant quantity of text
> across all chunking configurations. This isolates the chunking and
> embedding-model factors: without a fixed reference, identically-labelled
> configurations would index different amounts of text per model, confounding
> the comparison. e5/XLM-R was chosen because it is multilingual and trained on
> Indonesian, it is shared identically by two of the four candidate embedders
> (e5 and multilingual-MiniLM), and it makes the 128-token no-truncation
> condition exact for the truncation-limited MiniLM. The reference tokenizer is
> used only to measure chunk size and to map token windows to character offsets;
> each embedding model encodes with its own tokenizer, and per-model truncation
> rates are reported separately. A sensitivity analysis re-deriving chunks with
> an alternative reference (IndoBERT) left structure-aware chunk boundaries ~74%
> unchanged, and as the reference is held constant across models it does not
> bias the model comparison."

### Citations to use (verify exact bibliographic details)
- XLM-R: Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale*, ACL 2020.
- multilingual-e5: Wang et al., *Multilingual E5 Text Embeddings: A Technical Report*, 2024.
- SentencePiece: Kudo & Richardson, EMNLP 2018 (system demonstrations).
- IndoBERT / IndoNLU: Wilie et al., AACL-IJCNLP 2020.

---

## 8. Reproduce these numbers

```bash
# tokenizer comparison + corpus token counts + sensitivity table + JSON
./.venv/bin/python scripts/reference_tokenizer_sensitivity.py \
    --out outputs/analysis/reference_tokenizer_sensitivity.json
```

Pinned tokenizer revisions (from the run manifest / sensitivity JSON):
- e5-large: `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`
- indobert-base-p1: `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2`

---

## 9. Caveats

- A reference tokenizer is a **proxy**: exact for e5/MiniLM, approximate for
  IndoBERT/nomic (4.86 vs 5.55 chars/token).
- **Fixed** chunk sizes are ±1–2 tokens of nominal (substring re-tokenization at
  boundaries); per-model truncation is what matters and is logged in Phase 2.
- Configurable via `chunking.reference_tokenizer`; keep fixed within an
  experiment for comparability.
