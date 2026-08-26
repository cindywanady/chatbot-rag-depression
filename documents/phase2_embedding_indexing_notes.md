# Phase 2 — Embedding + Indexing: Notes & Decisions

Working notes for build_spec §2.4 (embedding the chunks with four models and
building one exact-cosine FAISS index per configuration). Covers what was built,
the decisions and their justifications, the truncation analysis we worked
through, and how to reproduce everything. All numbers were measured on this
project on 2026-06-29 (GPU: NVIDIA A40, 46 GB).

---

## 0. TL;DR

- For each `(chunk config × embedding model)` we embed the chunks (normalized
  vectors, with the model's query/passage prefixes) and build a **FAISS
  `IndexFlatIP`** (exact cosine). Grid = 14 chunk configs × 4 models = **56
  indexes**, built in ~85 s on the A40.
- **sentence-transformers** encodes 3 of 4 models; **raw transformers + manual
  mean pooling** encodes IndoBERT (the weak baseline). **FAISS** is the index on
  top — they are complementary layers, not alternatives.
- Every index persists `index.faiss`, `ids.json`, `chunks.jsonl`, and a
  `meta.json` (checkpoint, **revision**, device, n_chunks, mean/max tokens,
  **truncation rate**, build time).
- Truncation is **reported, not engineered away** (the spec mandates logging it).
  Decision on the special-token reserve: **don't reserve for this thesis**
  (see §6).

---

## 1. What was built

| Component | File |
|---|---|
| `Embedder`, `VectorIndex` ports | `src/depression_rag/ports/interfaces.py` |
| `EmbeddingModelSpec` (+ prefix logic) | `src/depression_rag/domain/models.py` |
| Model + index config | `configs/embedding_models.yaml` |
| sentence-transformers backend (e5, nomic, MiniLM) | `embedding/sentence_transformers_embedder.py` |
| transformers-mean backend (IndoBERT) | `embedding/transformers_mean_embedder.py` |
| FAISS IndexFlatIP | `indexing/faiss_index.py` |
| Pipeline / CLI / runner | `pipeline/indexing_pipeline.py`, `cli_index.py`, `scripts/run_indexing.py` |
| Tests (incl. mandated prefix test) | `tests/test_embedding_prefix.py`, `tests/test_indexing.py` |

Installed (Phase-2 extra): torch 2.12.1+cu130, sentence-transformers 3.4.1,
faiss-cpu 1.14.3, einops 0.8.2 (transformers stays pinned at 4.46.2).

---

## 2. The four models (verified against live model cards)

| model | checkpoint | backend | dim | max_seq_len | query / doc prefix | trust_remote_code |
|---|---|---|---|---|---|---|
| e5-large | intfloat/multilingual-e5-large | sentence-transformers | 1024 | 512 | `query: ` / `passage: ` | no |
| indobert | indobenchmark/indobert-base-p1 | transformers-mean | 768 | 512 | none | no |
| nomic | asmud/nomic-embed-indonesian | sentence-transformers | 768 | **8192** | `search_query: ` / `search_document: ` | **yes** |
| minilm | …/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers | 384 | 128 | none | no |

**Build-time correction:** nomic's card says `max_seq_len = 8192`, NOT the
spec's placeholder 2048. (The spec explicitly told us to confirm this.)

Model revisions pinned in the manifest: e5 `3d7cfbda`, indobert `c2cd0b51`,
minilm `e8f8c211`, nomic `913b6d9a`.

---

## 3. Key decisions (Q&A)

### Which vector store? → FAISS `IndexFlatIP` on normalized vectors
- `Flat` = exact brute-force; `IP` = inner product; on L2-normalized vectors,
  inner product == cosine (the metric all four models are trained for).
- **Not** HNSW/IVF/PQ: the corpus is tiny (≤ ~106 chunks/index), so exact search
  is instant AND removes a source of variance (approximate recall noise would
  confound the model comparison). Same "remove variance" logic as fixing the
  reference tokenizer.
- One index per `(chunk config × model)`, dimension = the model's dim.

### Why not "just use sentence-transformers" (instead of FAISS)?
They do different jobs and we use both: sentence-transformers is the **encoder**
(text→vector); FAISS is the **index** (vector→neighbours). ST could do tiny-scale
search via `util.semantic_search`, but FAISS gives a **persistable, reloadable,
model-agnostic** index artifact (required by the spec), reused at chatbot
query-time in Phase 2; `IndexFlatIP` is mathematically identical to cosine
brute-force, so nothing is lost.

### Why does IndoBERT use `transformers-mean`, not sentence-transformers?
IndoBERT is a raw masked-LM, not a sentence-embedding model (no ST pooling
config). Loading it via ST would silently auto-wrap it with mean pooling. We pool
explicitly so it's transparent and controlled — and because IndoBERT is the
intentional weak baseline (Reimers & Gurevych, 2019).

### Prefixes (spec: "not optional")
`query_prefix` is applied to questions, `doc_prefix` to chunks, at encode time.
e5 and nomic use asymmetric prefixes; IndoBERT and MiniLM use none. Asserted by
`tests/test_embedding_prefix.py` (the spec-mandated test).

---

## 4. Validation (end-to-end on GPU)

- **56 indexes built in ~85 s** on the A40; all four backends work, including
  nomic's `trust_remote_code` custom architecture (+ einops) and IndoBERT's
  manual mean pooling.
- Sanity: a depression-criteria query into `structure-512-0__minilm` returns
  MI.4 diagnosis/referral chunks at ~0.76 cosine.
- 42 unit tests pass (chunking + embedding-prefix + FAISS).

---

## 5. Truncation analysis (the main discussion)

Chunk size is measured in **reference (XLM-R) content tokens**. Each model
truncates at its own `max_seq_len` (which counts special tokens). Per-index
`truncation_rate` is logged. Mean truncation by `(model × chunk size)`:

| model | size=128 | size=256 | size=512 | cap |
|---|---|---|---|---|
| e5-large | 0.00 | 0.00 | **0.45** | 512 |
| indobert | 0.00 | 0.00 | 0.01 | 512 |
| nomic | 0.00 | 0.00 | 0.00 | 8192 |
| minilm | **0.60** | 1.00 | 0.89 | 128 |

### Three distinct truncation sources
1. **Special-token overhead (+2).** Every encoder wraps text with special
   tokens. Confirmed `num_special_tokens_to_add = 2` for **all four** (XLM-R
   `<s>/</s>`; BERT `[CLS]/[SEP]`). So a 128-*content*-token chunk → 130 total.
2. **Size > cap.** e.g. MiniLM at size 256/512 truncates by size alone (≈1.0),
   regardless of specials.
3. **Atomic chunks > size.** Structure-aware keeps tightly-coupled units whole
   even when oversize (`tokens_max` up to ~1165), so e5/MiniLM clip those at 512
   (e5 `structure-512` 0.11, MiniLM 0.67). This is intended behaviour.

### Where the +2 "bites" — only at the cap-matching size, and only for
### models that share the reference tokenizer
- The +2 *causes* truncation only when `content + 2 > cap`, i.e. at the chunk
  size equal to a model's cap.
- It is sharpest for **e5 and MiniLM**, because they **use XLM-R — the same
  tokenizer as the reference**. So "N reference tokens" == N of their tokens, and
  +2 tips over: e5 `fixed-512` → **0.98** (`tokens_max=517`), MiniLM `fixed-128`
  → **0.99** (`tokens_max=131`, tokens cluster at 130).
- It does **not** bite **IndoBERT** even at 512: its WordPiece tokenizer is
  coarser (5.55 vs 4.86 chars/token), so a 512-reference-token chunk is only
  ~448 IndoBERT tokens (+2 < 512) → `fixed-512` = 0.00.
- **nomic** never bites (cap 8192; `tokens_max` ≈ 2009).

### `fixed` overflows; `recursive` mostly doesn't
`fixed` fills to *exactly* `size` content tokens → +2 reliably overflows at the
boundary (e5 `fixed-512` 0.98, MiniLM `fixed-128` 0.99). `recursive` stops at
separators, landing *under* `size`, so far fewer cross (e5 `recursive-512` 0.19,
MiniLM `recursive-128` 0.21). The boundary +2 effect is essentially a
fixed-strategy phenomenon.

---

## 5.5 Context-enrichment: the `-ctx` ablation (section-path prepending)

`structure-512-0` and `structure-512-0-ctx` are a **paired ablation** (spec §2.3
Strategy C): the *same chunks*, but the `-ctx` variant prepends each chunk's
**section path** (`heading_path`) to the text **before embedding**.

**"Section path" =** the breadcrumb built during segmentation,
`MI unit > Pokok Bahasan > subsection`, stored as `heading_path`. e.g. for the
criteria chunk:
`MI.4 > Pokok Bahasan B: PENGENALAN GEJALA DAN PENEGAKAN DIAGNOSIS GANGGUAN DEPRESI > …`

**What differs (and what doesn't):**

| layer | `structure-512-0` vs `structure-512-0-ctx` |
|---|---|
| chunk text, char offsets, ids, metadata | **same** (enrichment is embed-time, not chunk-time) |
| embedder input | **different** — `-ctx` prepends `heading_path` + newline |
| FAISS vectors / index | **different** |
| query side | unchanged (queries are never enriched — document-side only) |
| §2.5 relevance mapping | unchanged (judged on the clean text span) |

```
base →  embed( "Depresi memiliki gejala-gejala utama dan tambahan. 1. Sedih…" )
ctx  →  embed( "MI.4 > Pokok Bahasan B: …DIAGNOSIS… \n Depresi memiliki gejala… 1. Sedih…" )
                └────────── prepended section path ──────────┘└──── same body ────┘
```

**Why.** A chunk's body often lacks the words that name its topic (the criteria
body lists symptoms but never says "kriteria diagnosis"). Prepending the path
injects that context into the vector, so a diagnosis-themed query can match it.
It is the structural (free, deterministic) form of "contextual retrieval" —
using the document's own heading hierarchy instead of LLM-generated context.

**Trade-offs (why it's an ablation, not always-on):**
- *Helps:* short / keyword-poor chunks get disambiguated by their section.
- *Hurts:* the path text dilutes the body, and it eats token budget — for MiniLM,
  enrichment raised tokens_mean 289 → 317 and truncation 0.67 → 0.71 (heading
  tokens push more body past the 128 cap).
- Empirical → §2.6 measures whether it helps, per model.

**The bug that was fixed.** The indexing pipeline originally embedded the raw
chunk text and ignored the `context_enriched` flag, so the two indexes were
byte-identical (the ablation was a no-op). Fixed in
`pipeline/indexing_pipeline.py::_encoder_inputs` (prepend `heading_path` iff
`param_context_enriched`; `meta.json` now records `context_enriched`). After the
fix, mean cosine(base, ctx) ≈ **0.88–0.96** (e5 0.959, indobert 0.961,
nomic 0.949, minilm 0.884) — distinct, as intended.

**Known caveat.** For subsection-*marker* segments (e.g. "depresi memiliki
gejala"), the `heading_path` leaf is a truncated copy of the chunk's first
sentence, so the prepended context is partly redundant. Candidate segmentation
cleanup (give marker subsections a short label); does not affect correctness.

---

## 6. Decision: do we reserve the 2 special tokens?

**Recommendation: NO — do not reserve for this thesis; report it instead.**

Reasons:
1. The spec mandates logging truncation rate per config — it **expects**
   truncation to be reported, not engineered to zero. Reporting is faithful.
2. Reserving would muddy the clean, well-justified definition
   "`chunk_size` = N reference *content* tokens" (it would become N−2 content
   + 2 special).
3. Standard practice (MTEB/BEIR, the model papers) is to truncate at
   `max_seq_len` and report — reserving specials is a library nicety
   (LangChain's token splitter), not an evaluation convention.
4. The effect is ~2 trailing subwords (1.5% of a 128-token chunk), uniform
   across models, non-confounding.

**Honest framing to put in the write-up instead:**
> "Chunks are sized in reference (XLM-R) content tokens. The 128-token condition
> is encoded in full by e5, IndoBERT and nomic; MiniLM (max 128) clips ~2 tokens
> per chunk because XLM-R adds 2 special tokens. Per-config truncation rates are
> reported."

**Exception — reserve only if** a supervisor/examiner explicitly requires a
*literal* no-truncation condition for the truncation-limited model. Then set
`chunk_size ≤ 126` or add a `reserve_special_tokens: 2` knob (effective content =
size − 2). It is well-founded (all four models add exactly 2) and cheap
(re-chunk seconds + re-index ~90 s), but it does NOT remove the size-dominated
or atomic-chunk truncation, and it is not the default.

---

## 7. Outputs

```
outputs/indexes/<config_id>__<model>/
    index.faiss        # FAISS IndexFlatIP
    ids.json           # faiss-position -> chunk_id (+ dim, metric)
    chunks.jsonl       # chunk metadata in vector order (read by retrieval + chatbot)
    meta.json          # checkpoint, revision, device, n_chunks, tokens, truncation_rate, build_s
outputs/manifests/<run_id>.json   # full provenance for the run
```

---

## 8. Reproduce

```bash
# install Phase-2 deps (CUDA torch on Linux; CPU wheel if no GPU)
./.venv/bin/pip install -r requirements-embed.txt        # or: pip install -e ".[embed]"

# build the full grid (or subset with --models / --configs)
./.venv/bin/python -m depression_rag.cli_index
./.venv/bin/python -m depression_rag.cli_index --models minilm --configs structure-512-0

# tests (prefix test is the spec-mandated one)
./.venv/bin/python -m pytest -q
```

Per-model special-token count (the §5 confirmation):
```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(ckpt).num_special_tokens_to_add(pair=False)   # == 2 for all four
```

---

## 9. Still to come (spec 2.5)

`Retriever` + a gold depression-QA set + the relevance mapping (char-overlap of
retrieved chunks against `cleaned_text.txt`) + the metrics table that selects the
winning `(chunking × model)` configuration — consuming the 56 indexes built here.
