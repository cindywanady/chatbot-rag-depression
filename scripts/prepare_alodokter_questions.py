#!/usr/bin/env python
"""Prepare the real-question evaluation set from the Alodokter scrape.

Turns ``data/derived/alodokter_depresi_qa.csv`` (scripts/alodokter_scraper.py)
into ``data/derived/questions_alodokter.jsonl`` — the deployed-chatbot
evaluation set of real patient questions. Unlike ``questions_gold.jsonl`` these
carry NO gold passage mapping (they are scored reference-free at the answer
level); the doctor's forum reply is kept as ``doctor_answer`` for reference
and later human comparison, never as a gold label.

Steps: drop rows without question text -> deduplicate by thread URL (keep the
first occurrence) -> re-run the PII scrub as a guard -> heuristically redact
patient names to ``[nama]`` (self-introduction anchors like "perkenalkan saya
X" / "nama saya X", the "saya X, umur N" form, and doctors greeting the asker
by name; a PRE-pass only — manual review stays mandatory) -> count e5 query
tokens ("query: " prefix + special tokens, the deployed retriever's view) and
flag questions over the 512-token cap -> assign stable ids (``alo_NNNN`` over
URL-sorted threads, so re-scrapes keep existing pairs aligned) -> write the
JSONL + a summary report. Near-duplicate questions (token Jaccard) are
REPORTED but kept: similar real questions are ecologically valid.

    python scripts/prepare_alodokter_questions.py
    python scripts/prepare_alodokter_questions.py --no-tokens   # skip tokenizer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

E5_CHECKPOINT = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
QUERY_CAP = 512

_PHONE = re.compile(r"(?:\+62|62|0)8\d{7,11}")
_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

# ---- heuristic name redaction ------------------------------------------------
# Patterns grounded in the scraped data (see the preparation report). This is a
# PRE-pass: it catches the common self-introduction and greeting forms, but it
# cannot be complete — a manual PII review is still required before any text is
# shown to evaluators or quoted.

# words that legitimately follow "saya ..." or an anchor and are NOT names
_NOT_NAMES = {
    "remaja", "pria", "wanita", "perempuan", "laki", "laki-laki", "lelaki",
    "ibu", "bapak", "seorang", "cewek", "cowok", "gadis", "istri", "suami",
    "anak", "mahasiswa", "mahasiswi", "pelajar", "siswa", "siswi", "karyawan",
    "karyawati", "single", "janda", "duda", "dok", "dokter",
    "terus", "sudah", "masih", "juga", "baru", "sering", "sekarang", "kini",
    "hanya", "ingin", "mau", "akhir", "akhir-akhir", "yang", "dan", "asal",
    "umur", "usia", "berusia", "tahun", "saya", "aku", "adalah", "merasa",
}
# tokens that end a name run after an anchor phrase
_NAME_STOP = _NOT_NAMES
# introduction anchors: the following 1-2 word tokens are a name
_INTRO = re.compile(
    r"(?i)\b(perkenalkan\s+(?:nama\s+)?saya|nama\s+(?:saya|ku)|(?:saya|aku)\s+bernama)\s+")
# "saya <Name> (,) umur/usia/berusia" — needs capitalization since there is no
# introduction anchor, only the age phrase, and descriptors are common here
_NAME_AGE = re.compile(
    r"\b([Ss]aya|[Aa]ku)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s*,?\s*(umur|usia|berusia)\b")
# doctors greet the asker by name ("Alo Cahya, ...")
_GREETING = re.compile(r"(?i)\b(alo|halo|hai)[, ]+([A-Z][a-zA-Z]+)\b")
_GREET_STOP = {"dok", "dokter", "terimakasih", "terima", "kasih", "selamat",
               "salam", "kak", "bu", "pak", "mbak", "mas", "bunda", "sdr",
               "sdri", "saudara", "saudari", "sebelumnya", "sobat",
               # sentence-starters doctors use right after the greeting
               "apakah", "apa", "anda", "semoga", "mohon", "silakan",
               "menjawab", "terkait", "mengenai", "depresi"}
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")


def redact_names(text: str) -> tuple[str, int]:
    """Replace name spans matched by the introduction/greeting heuristics with
    ``[nama]``. Returns (redacted_text, number_of_redactions)."""
    n = 0
    out = text

    # introduction anchors: bound the following 1-2 token name run manually
    for m in list(_INTRO.finditer(out))[::-1]:  # right-to-left keeps offsets valid
        tail = out[m.end(): m.end() + 60]
        end = words = 0
        for w in _WORD.finditer(tail):
            if w.start() > end + 1 or words == 2:   # gap or run limit reached
                break
            if w.group(0).lower() in _NAME_STOP:
                break
            end, words = w.end(), words + 1
        if words:
            out = out[: m.end()] + "[nama]" + out[m.end() + end:]
            n += 1

    def _age_sub(m: re.Match) -> str:
        nonlocal n
        if m.group(2).split()[0].lower() in _NOT_NAMES:
            return m.group(0)
        n += 1
        return f"{m.group(1)} [nama] {m.group(3)}"

    out = _NAME_AGE.sub(_age_sub, out)

    def _greet_sub(m: re.Match) -> str:
        nonlocal n
        if m.group(2).lower() in _GREET_STOP or m.group(2).lower() in _NOT_NAMES:
            return m.group(0)
        n += 1
        return f"{m.group(1)} [nama]"

    out = _GREETING.sub(_greet_sub, out)
    return out, n


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _load_e5_token_counter():
    """Return a callable question -> token count, or None if unavailable."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            E5_CHECKPOINT, cache_dir=os.environ["HF_HOME"], local_files_only=True)
        return lambda q: len(
            tok(QUERY_PREFIX + q, add_special_tokens=True, truncation=False)["input_ids"])
    except Exception as e:  # tokenizer not cached / transformers missing
        print(f"[prep] e5 tokenizer unavailable ({e}); token fields omitted")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prepare-alodokter-questions", description=__doc__)
    ap.add_argument("--csv", default=str(ROOT / "data" / "derived" / "alodokter_depresi_qa.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "derived" / "questions_alodokter.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "outputs" / "analysis" / "questions_alodokter_report.md"))
    ap.add_argument("--near-dup", type=float, default=0.9,
                    help="token-Jaccard threshold for the (report-only) near-duplicate check")
    ap.add_argument("--no-tokens", action="store_true", help="skip the e5 token count")
    args = ap.parse_args(argv)

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_in = len(rows)

    rows = [r for r in rows if r["question"].strip() and r["url"].strip()]
    n_empty = n_in - len(rows)

    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:  # keep the first occurrence per thread URL
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        unique.append(r)
    n_dup = len(rows) - len(unique)

    pii_fixed = 0
    for r in unique:  # guard: the scraper already scrubs, but never trust one pass
        for col in ("question", "answer"):
            scrubbed = _PHONE.sub("[phone]", _EMAIL.sub("[email]", r[col]))
            if scrubbed != r[col]:
                pii_fixed += 1
                r[col] = scrubbed

    count_tokens = None if args.no_tokens else _load_e5_token_counter()

    unique.sort(key=lambda r: r["url"])  # stable id order across re-scrapes
    out_rows = []
    name_redacted: list[tuple[str, int]] = []
    for i, r in enumerate(unique, 1):
        rec = {
            "question_id": f"alo_{i:04d}",
            "source": "alodokter",
            "url": r["url"],
            "question_title": r["question_title"].strip() or r["title"].strip(),
            "question": r["question"].strip(),
            "question_date": r["question_date"].strip(),
            "doctor_answer": r["answer"].strip(),   # reference only, not a gold label
            "answer_doctors": r["answer_doctors"].strip(),
        }
        spans = 0
        for col in ("question_title", "question", "doctor_answer"):
            rec[col], k = redact_names(rec[col])
            spans += k
        if spans:
            name_redacted.append((rec["question_id"], spans))
        if count_tokens is not None:
            n_tok = count_tokens(rec["question"])
            rec["e5_query_tokens"] = n_tok
            rec["over_query_cap"] = n_tok > QUERY_CAP
        out_rows.append(rec)

    # near-duplicates: report, don't drop
    toks = [set(r["question"].lower().split()) for r in out_rows]
    near = [(out_rows[i]["question_id"], out_rows[j]["question_id"],
             round(_jaccard(toks[i], toks[j]), 2))
            for i in range(len(toks)) for j in range(i + 1, len(toks))
            if _jaccard(toks[i], toks[j]) >= args.near_dup]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for rec in out_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    qlens = sorted(len(r["question"]) for r in out_rows)
    lines = ["# Alodokter question set — preparation report\n",
             f"_Generated by `scripts/prepare_alodokter_questions.py` from `{Path(args.csv).name}`._\n",
             f"- scraped rows in: **{n_in}**",
             f"- dropped (no question text / no url): {n_empty}",
             f"- dropped (duplicate thread url): {n_dup}",
             f"- **final questions: {len(out_rows)}** -> `{out}`",
             f"- PII guard: {pii_fixed} field(s) re-scrubbed "
             f"({'clean on arrival' if pii_fixed == 0 else 'CHECK the scraper'})",
             f"- heuristic name redaction: **{sum(k for _, k in name_redacted)} span(s) "
             f"in {len(name_redacted)} thread(s)** replaced with `[nama]` "
             "(intro anchors, saya-Name-umur, doctor greetings). A pre-pass only — "
             "manual PII review is still mandatory before evaluator exposure. "
             "Affected: "
             + (", ".join(q for q, _ in name_redacted[:20])
                + (f" … (+{len(name_redacted) - 20} more)" if len(name_redacted) > 20 else "")
                if name_redacted else "none"),
             f"- question length (chars): min/median/max = {qlens[0]}/{qlens[len(qlens)//2]}/{qlens[-1]}",
             f"- with doctor answer: {sum(1 for r in out_rows if r['doctor_answer'])}"]
    if count_tokens is not None:
        tl = sorted(r["e5_query_tokens"] for r in out_rows)
        over = [r for r in out_rows if r["over_query_cap"]]
        lines += [f"- e5 query tokens (prefix incl.): min/median/mean/max = "
                  f"{tl[0]}/{tl[len(tl)//2]}/{mean(tl):.0f}/{tl[-1]}",
                  f"- **over the {QUERY_CAP}-token query cap: {len(over)}** "
                  f"({100 * len(over) / len(out_rows):.1f}%) — the retriever only sees the "
                  "first 512 tokens of these; patient posts usually put the actual "
                  "question LAST, so handle long queries deliberately (tail-keeping "
                  "truncation or query condensation):"]
        lines += [f"  - {r['question_id']} ({r['e5_query_tokens']} tokens) {r['url']}"
                  for r in over]
    lines.append(f"- near-duplicate pairs (Jaccard >= {args.near_dup}, kept — "
                 f"similar real questions are valid): {near if near else 'none'}")
    lines.append("\n> These questions have no gold passage mapping; they are the "
                 "reference-free evaluation set for the deployed chatbot. "
                 "`doctor_answer` is a forum reply kept for reference and human "
                 "comparison, not a gold answer.")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[wrote {out} and {report}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
