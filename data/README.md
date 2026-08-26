# data/ — not in version control

This directory holds inputs that cannot be published in a public repository.
Recreate it locally as follows:

| Path | What it is | How to obtain |
|---|---|---|
| `Modul Keswa bagi Dokter Umum di FKTP.pdf` | Kemenkes RI (2017) primary-care mental-health guideline — the knowledge base (copyrighted) | Kemenkes / institutional access |
| `*.pdf` (papers) | ESHRO, Park et al., and related evaluation papers | publisher / library access |
| `derived/` | All processed data: cleaned guideline text, segments, gold QA set, scraped Alodokter questions, frozen samples, scoring keys | regenerate with the pipeline scripts (see root README); the Alodokter questions are secondary data that the ethics protocol commits to storing securely, not redistributing |

## Page numbers in `derived/` are PRINTED pages, not PDF viewer pages

`page_start` / `page_end` in `segments.jsonl`, `gold_passages.jsonl` and every
chunk record are the number **printed on the paper**, which is offset from the
position your PDF viewer shows:

```
viewer page = printed page + 2          (configs/pipeline.yaml -> source.printed_offset)
```

So a segment recorded as `page_start: 38` is on **viewer page 40**; viewer page
38 prints "36". Verified anchor: "MATERI INTI 4" is physical page 38 and prints
36. Check any citation against the PDF with that offset applied — the field
names do not say which convention they use, and looking up the wrong page is the
easy mistake.

Pipeline order to rebuild `derived/` from the guideline PDF:
`run_chunking.py` → `run_indexing.py` → `build_gold_passages.py` →
`validate_questions.py` (the **121** gold questions are LLM-authored under
`configs/prompts/qa_generation.md` and human-validated; no script regenerates
them — 124 were authored, 3 dropped with gold passage `MI1_0003` on 2026-07-26) →
(`alodokter_scraper.py` → `prepare_alodokter_questions.py` →
`sample_alodokter_questions.py` → `deidentify_frozen_files.py`) →
`freeze_eval_assignments.py`.

## The study questions are de-identified (since 2026-07-27)

`questions_alodokter_sample50.jsonl` holds the **rendered** text: three
self-introduced names are `[nama]`, and `url` has been replaced by
`source_sha256`. A resolvable link to the original public post would undo the
text scrub, so the hash → URL mapping lives separately in
`SEALED_source_urls.csv`, which is researcher-only and must not be shared.

Two consequences if you rebuild:

- `deidentify_frozen_files.py` is **one-way** and rewrites hash-pinned files. It
  forces `freeze_eval_assignments.py --force` and a regeneration of every
  briefing and automatic score. Run it before freezing, not after.
- The questions are still verbatim forum text, so they remain searchable back to
  their source. Keep `data/` out of version control regardless.
