#!/usr/bin/env python
"""Do the guideline's referral passages actually reach the counselor at k=5?

The deployed system prompt requires a section "4) OPSI RUJUKAN sesuai pedoman" in
every briefing, and tells the generator to route risk cases "sesuai pedoman (MI.7
dan MI.8)".

The gold set does test referral retrieval — 6 questions of type referral_criteria,
recall@5 = 1.000 — but it tests it under two restrictions that the deployment does
not share: every one of those questions is clinician-voiced ("Kapan saya harus
merujuk pasien depresi ke spesialis?"), and every one is keyed to MI.4-D
(MI4_0042). MI.8, which carries the referral *procedure* in 11 of the index's 90
chunks, is never a gold passage. So recall@5 = 1.000 licenses a narrower claim
than it appears to: MI.4-D is retrievable for clinician-style queries. This probe
asks the two questions that leaves open — what happens on the patient narratives
the chatbot actually receives, and whether MI.8 is reachable at all.

It runs three stages, and the contrast between them is the whole point:

  1. CONTROL — clinician-voiced referral questions written for this probe
     ("Kapan pasien depresi harus dirujuk?"). Establishes that the referral
     chunks CAN be retrieved, so a miss in stage 2 is about the input
     distribution and not about a broken index.

  2. STUDY — the 50 frozen study questions (real patient narratives from a health
     forum). This is the distribution the chatbot actually faces.

  3. BRIEFINGS — for the answers already generated, parse the OPSI RUJUKAN
     section and resolve its [n] markers back to the chunks that were in context,
     asking what those citations actually point at.

Stage 3 needs no embedder and runs in seconds; stages 1-2 load e5-large.

    ./.venv/bin/python scripts/referral_retrieval_probe.py --device cpu
    ./.venv/bin/python scripts/referral_retrieval_probe.py --stage briefings
    ./.venv/bin/python scripts/referral_retrieval_probe.py --out outputs/analysis/referral_probe

Read-only with respect to the study: it never writes outside --out, and --out is
optional. Nothing here changes k, the prompt, or the index.

Findings as of 2026-07-31, deployed index structure-512-0__e5-large__mv, k=5:

  stage 1 (control, 14 questions)   >=1 referral chunk in top-5 : 79%
  stage 2 (50 frozen study cases)   >=1 referral chunk in top-5 :  8%
                                    MI.8 on the 24 risk cases   :  0/24 (also 0 at k=10)
  stage 3 (50 generated briefings)  48/50 write an OPSI RUJUKAN section;
                                    of the [n] it cites there, 5% are referral
                                    chunks (66% clinical_exposition, 22%
                                    risk_suicide). On risk cases, 2/24 cite an
                                    actual referral chunk.

So the section is written and cited, but the citations point at passages that are
not referral guidance — which is what the Tahap-2 items "Ketepatan rujukan [n]"
and "Dukungan konteks" are built to catch. Raising k does not fix it for risk
cases. Recorded here so the effect is a known property of the instrument rather
than a surprise in the rater data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.chatbot import (  # noqa: E402
    load_chatbot_config,
    load_chunk_records,
    risk_screen,
)

# The control set: referral-shaped questions in a clinician's voice. Deliberately
# NOT how the study's questions read — that contrast is what stage 1 exists to show.
CONTROL_QUESTIONS = [
    ("A · kapan merujuk", "Kapan pasien dengan gangguan depresi harus dirujuk ke spesialis?"),
    ("A · kapan merujuk", "Pasien sudah minum antidepresan dua bulan tapi belum membaik, apakah perlu dirujuk?"),
    ("A · kapan merujuk", "Apa saja kriteria rujukan untuk kasus gangguan depresi di layanan primer?"),
    ("A · kapan merujuk", "Penanya menunjukkan gejala psikotik disertai depresi, apakah harus dirujuk?"),
    ("B · tatacara rujukan", "Apa saja yang harus dicantumkan dalam surat rujukan pasien gangguan jiwa?"),
    ("B · tatacara rujukan", "Bagaimana tatacara sistem rujukan dan rujuk balik untuk pelayanan kesehatan jiwa?"),
    ("B · tatacara rujukan", "Apakah rujukan pasien memerlukan persetujuan dari pasien atau keluarganya?"),
    ("B · tatacara rujukan", "Ke fasilitas kesehatan mana pasien depresi dari puskesmas seharusnya dirujuk?"),
    ("C · tanda bahaya + rujukan", "Penanya mengatakan ingin mengakhiri hidupnya dan sudah punya rencana, ke mana harus dirujuk?"),
    ("C · tanda bahaya + rujukan", "Pasien baru saja melakukan percobaan bunuh diri, apa langkah rujukan segera yang harus dilakukan?"),
    ("C · tanda bahaya + rujukan", "Ada penanya yang bilang lebih baik mati saja, apa yang perlu dilakukan konselor dan ke mana merujuk?"),
    ("C · tanda bahaya + rujukan", "Kapan pasien dengan risiko bunuh diri perlu dirawat inap atau dihospitalisasi?"),
    ("D · persiapan rujukan", "Bagaimana menyiapkan pasien depresi yang akan dirujuk agar tidak merasa distigma?"),
    ("D · persiapan rujukan", "Bagaimana menjelaskan kepada keluarga bahwa pasien perlu dirujuk ke psikiater?"),
]

# "4) OPSI RUJUKAN sesuai pedoman" — the generator emits it with markdown emphasis,
# so match the words and stop at the next numbered section or end of answer.
REFERRAL_SECTION = re.compile(r"OPSI\s+RUJUKAN(.*?)(?=\n\s*\*{0,2}\d\)|\Z)", re.S | re.I)
CITATION = re.compile(r"\[(\d+)\]")


# ------------------------------------------------------------------ helpers ---
def is_referral(rec: dict) -> bool:
    return rec["meta_content_type"] == "referral_criteria"


def is_mi8(rec: dict) -> bool:
    return rec["meta_source_unit"] == "MI.8"


def is_mi4d(rec: dict) -> bool:
    return rec["meta_source_unit"] == "MI.4" and "Bahasan D" in rec["meta_heading_path"]


PREDICATES = [(">=1 chunk rujukan", is_referral),
              ("MI.8 (sistem rujukan)", is_mi8),
              ("MI.4-D (kapan merujuk)", is_mi4d)]


def build_runner(cfg, device: str):
    from depression_rag.evaluation.retrieval import (  # noqa: PLC0415
        RetrievalRunner,
        build_embedder_for_model,
    )
    embedder = build_embedder_for_model(cfg.model, ROOT / cfg.embedding_config,
                                        device=None if device == "auto" else device)
    return RetrievalRunner(ROOT / cfg.index_dir, embedder)


def retrieve(runner, records, items, k_max):
    """items: [(label, question_id, question)] -> rows with the resolved top-k."""
    qs = [{"question_id": qid, "question": q} for _, qid, q in items]
    results = runner.search(qs, k_max)
    rows = []
    for (label, qid, q), res in zip(items, results):
        rows.append({
            "label": label, "question_id": qid, "question": q,
            "top": [(records[cid], float(sc))
                    for cid, sc in zip(res.ranked_chunk_ids, res.scores)],
        })
    return rows


def hit(row, k, pred) -> bool:
    return any(pred(rec) for rec, _ in row["top"][:k])


def report(rows, k, k_alt, title, group_by=None):
    n = len(rows)
    print(f"\n{'=' * 78}\n{title}   (n={n})\n{'=' * 78}")
    print(f"{'':26} {f'k={k}':>14} {f'k={k_alt}':>14}")
    for name, pred in PREDICATES:
        a = sum(hit(r, k, pred) for r in rows)
        b = sum(hit(r, k_alt, pred) for r in rows)
        print(f"  {name:24} {a:>4}/{n} ({a/n:4.0%}) {b:>6}/{n} ({b/n:4.0%})")

    if group_by:
        print(f"\n  per {group_by}  (>=1 chunk rujukan di top-{k}):")
        by = defaultdict(list)
        for r in rows:
            by[r["label"]].append(hit(r, k, is_referral))
        for g, v in by.items():
            print(f"    {g:30} {sum(v)}/{len(v)}")

    units = Counter(rec["meta_source_unit"] for r in rows for rec, _ in r["top"][:k])
    types = Counter(rec["meta_content_type"] for r in rows for rec, _ in r["top"][:k])
    tot = sum(units.values())
    print(f"\n  yang mengisi top-{k} — unit: "
          + ", ".join(f"{u} {v} ({v/tot:.0%})" for u, v in units.most_common()))
    print("  " + " " * 21 + "content_type: "
          + ", ".join(f"{t} {v} ({v/tot:.0%})" for t, v in types.most_common(4)))


# ------------------------------------------------------------------- stages ---
def stage_control(runner, records, k, k_alt):
    items = [(theme, f"c{i:02d}", q) for i, (theme, q) in enumerate(CONTROL_QUESTIONS)]
    rows = retrieve(runner, records, items, k_alt)
    report(rows, k, k_alt, "STAGE 1 · KONTROL — pertanyaan gaya klinisi", group_by="tema")
    print("\n  Kontrol ini membuktikan chunk rujukan MEMANG bisa terambil.\n"
          "  Bandingkan dengan stage 2: bedanya ada di distribusi input, bukan di indeks.")
    return rows


def stage_study(runner, records, cfg, k, k_alt):
    cases_path = ROOT / "data" / "derived" / "eval_case_assignments.json"
    qs_path = ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl"
    if not cases_path.exists() or not qs_path.exists():
        print(f"\n[probe] lewati stage 2: {cases_path.name} / {qs_path.name} tidak ada")
        return []

    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    text = {}
    for line in qs_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        text[d["question_id"]] = d["question"]

    items = [("risiko" if c["risky"] else "non-risiko", c["study_id"],
              text[c["question_id"]]) for c in cases]
    rows = retrieve(runner, records, items, k_alt)
    for r, c in zip(rows, cases):
        r["risky"] = c["risky"]
        r["in_counselor_sample"] = c["in_counselor_sample"]
        r["kw"] = bool(risk_screen(r["question"], cfg.safety_keywords))

    report(rows, k, k_alt, "STAGE 2 · STUDI — 50 pertanyaan yang sudah dibekukan",
           group_by="kelompok")
    for sub, label in [([r for r in rows if r["risky"]], "hanya kasus BERISIKO"),
                       ([r for r in rows if r["in_counselor_sample"]],
                        "hanya 24 kasus yang dijawab konselor")]:
        if sub:
            report(sub, k, k_alt, f"STAGE 2b · {label}")

    risky = [r for r in rows if r["risky"]]
    if risky:
        nkw = sum(r["kw"] for r in risky)
        print(f"\n  safety keyword screen menyala pada {nkw}/{len(risky)} kasus berisiko "
              f"({nkw/len(risky):.0%}).")
        print(f"  Sisanya ({len(risky)-nkw}) bergantung pada llm_check — dilewati bila "
              "generator tidak tersedia\n  (chatbot.py: `if not matched and self.generator "
              "is not None and cfg.llm_check`).")
    return rows


def stage_briefings(records, k):
    path = ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl"
    if not path.exists():
        print(f"\n[probe] lewati stage 3: {path} tidak ada")
        return []

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_sec = n_cited = 0
    cited_types, cited_units = Counter(), Counter()
    risky_ok = risky_tot = 0
    audit = []

    for r in rows:
        m = REFERRAL_SECTION.search(r.get("generated_answer") or "")
        if not m:
            continue
        n_sec += 1
        nums = sorted({int(x) for x in CITATION.findall(m.group(1))})
        if nums:
            n_cited += 1
        ctx = r.get("retrieved_context") or []
        supported = False
        for n in nums:
            if 1 <= n <= len(ctx):
                rec = records[ctx[n - 1]["chunk_id"]]
                cited_types[rec["meta_content_type"]] += 1
                cited_units[rec["meta_source_unit"]] += 1
                supported |= is_referral(rec)
        if r.get("risky"):
            risky_tot += 1
            risky_ok += supported
        audit.append({"study_id": r.get("study_id"), "risky": r.get("risky"),
                      "n_citations": len(nums), "referral_supported": supported})

    print(f"\n{'=' * 78}\nSTAGE 3 · BRIEFING — apa yang sebenarnya ditunjuk sitasi "
          f"di bagian rujukan\n{'=' * 78}")
    print(f"  briefing dengan bagian OPSI RUJUKAN : {n_sec}/{len(rows)}")
    print(f"  di antaranya memuat sitasi [n]      : {n_cited}/{n_sec}")
    tot = sum(cited_types.values())
    if tot:
        print("\n  chunk yang ditunjuk sitasi — content_type:")
        for t, v in cited_types.most_common():
            flag = "  <-- panduan rujukan sungguhan" if t == "referral_criteria" else ""
            print(f"    {t:22} {v:3} ({v/tot:4.0%}){flag}")
        print(f"\n  ...unit: {dict(cited_units.most_common())}")
    if risky_tot:
        print(f"\n  kasus BERISIKO yang bagian rujukannya menunjuk chunk rujukan: "
              f"{risky_ok}/{risky_tot}")
    return audit


# --------------------------------------------------------------------- main ---
def _rel(p: Path) -> str:
    """Repo-relative when possible — --out may point anywhere."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_out(out: Path, control, study, audit, k, k_alt):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        if not rows:
            return
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "question_id", "risky", f"referral@{k}", f"referral@{k_alt}",
                        f"mi8@{k}", f"mi8@{k_alt}", f"mi4d@{k}", "top1_unit", "top1_type"])
            for r in rows:
                top = r["top"][0][0]
                w.writerow([r["label"], r["question_id"], r.get("risky", ""),
                            hit(r, k, is_referral), hit(r, k_alt, is_referral),
                            hit(r, k, is_mi8), hit(r, k_alt, is_mi8),
                            hit(r, k, is_mi4d),
                            top["meta_source_unit"], top["meta_content_type"]])
        print(f"[probe] wrote {_rel(out / name)}")

    dump("control_questions.csv", control)
    dump("study_questions.csv", study)
    if audit:
        with open(out / "briefing_citations.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(audit[0]))
            w.writeheader()
            w.writerows(audit)
        print(f"[probe] wrote {_rel(out / 'briefing_citations.csv')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "chatbot.yaml"))
    ap.add_argument("--stage", choices=["all", "control", "study", "briefings"],
                    default="all")
    ap.add_argument("--k", type=int, default=None, help="default: the deployed k")
    ap.add_argument("--k-alt", type=int, default=10, help="comparison k")
    ap.add_argument("--device", default="cpu", help="auto|cuda|cpu")
    ap.add_argument("--out", default=None, help="optional directory for CSV artifacts")
    args = ap.parse_args(argv)

    cfg = load_chatbot_config(args.config)
    k = args.k or cfg.k
    k_alt = max(args.k_alt, k)
    records = load_chunk_records(ROOT / cfg.index_dir)

    print(f"[probe] index    : {cfg.index_dir}")
    print(f"[probe] embedder : {cfg.model}   ·   k terdeploy = {cfg.k}   ·   probe k = {k}/{k_alt}")
    n_ref = sum(is_referral(r) for r in records.values())
    print(f"[probe] korpus   : {len(records)} chunk, {n_ref} bertipe referral_criteria "
          f"({n_ref/len(records):.0%})")

    control = study = []
    audit = []
    need_index = args.stage in ("all", "control", "study")
    runner = build_runner(cfg, args.device) if need_index else None

    if args.stage in ("all", "control"):
        control = stage_control(runner, records, k, k_alt)
    if args.stage in ("all", "study"):
        study = stage_study(runner, records, cfg, k, k_alt)
    if args.stage in ("all", "briefings"):
        audit = stage_briefings(records, k)

    if args.out:
        write_out(Path(args.out), control, study, audit, k, k_alt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
