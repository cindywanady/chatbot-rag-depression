#!/usr/bin/env python
"""Descriptive analysis of the four filled counselor workbooks.

This is the counselor-SIDE analysis: what the counselors did and what they
thought of the tool. It is NOT the study's primary outcome — that is the blind
psychologist rating of answer quality, which `scripts/analyze_study.py` runs
once the score workbooks come back. Until then those score files are empty
templates and the primary comparison cannot be made, so everything here is
process and perception evidence and is labelled as such.

Inputs (the returned workbooks, one per counselor per condition):

    data/evaluation_form_result/Konselor_{1,2}_{DENGAN,TANPA}_alat_bantu.xlsx

plus the frozen assignment (`data/derived/eval_case_assignments.json`) for the
block / risk-flag / condition key, so nothing about the design is re-derived
from the returned files.

Writes:

    outputs/analysis/counselor_forms_tidy.csv     one row per answer (48)
    outputs/analysis/counselor_forms_stats.json   every number the deck quotes
    outputs/analysis/counselor_forms_report.md    the readable write-up

    .venv/bin/python scripts/analyze_counselor_forms.py

Two things in the returned files need care, and both are reported rather than
silently repaired:

  * Counselor 2 typed clock times as decimals ("23.29" for 23:29), so the
    workbook's own `MOD(end-start,1)*1440` formula produced nothing. Durations
    are reconstructed here, with midnight wrap. Some reconstructions are
    implausibly long (a timer left running over a break), so duration is
    summarised with medians and the long cases are flagged, never dropped.
  * The danger-sign helpfulness item has a non-numeric escape option ("no
    danger signs in this case"). It is counted separately, never scored 0.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path

import openpyxl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FORMS = ROOT / "data" / "evaluation_form_result"
ASSIGN = ROOT / "data" / "derived" / "eval_case_assignments.json"
OUTDIR = ROOT / "outputs" / "analysis"

WORKBOOKS = [
    ("counselor_1", "chatbot", "Konselor_1_DENGAN_alat_bantu.xlsx"),
    ("counselor_1", "no_chatbot", "Konselor_1_TANPA_alat_bantu.xlsx"),
    ("counselor_2", "chatbot", "Konselor_2_DENGAN_alat_bantu.xlsx"),
    ("counselor_2", "no_chatbot", "Konselor_2_TANPA_alat_bantu.xlsx"),
]

# Columns are resolved from the header row, not hardcoded: the two conditions
# ship different layouts (the TANPA workbook has no per-case usefulness items,
# so «Seberapa yakin» sits five columns to the left of where it sits in DENGAN).
HEADER_KEYS = {
    "case": "Kode kasus",
    "answer": "Jawaban Anda",
    "words_cell": "Jumlah kata",
    "start": "Waktu mulai",
    "end": "Waktu selesai",
    "help_case": "membantu ringkasan untuk kasus ini",
    "help_risk": "mengenali tanda bahaya",
    "influence": "memengaruhi jawaban akhir",
    "confidence": "yakin Anda pada jawaban ini",
    "harm": "keliru atau berpotensi membahayakan",
    "harm_note": "jelaskan singkat",
}

NO_DANGER = "Tidak ada tanda bahaya pada kasus ini"
INFLUENCE = {"Tidak memengaruhi": 0, "Sedikit memengaruhi": 1,
             "Cukup memengaruhi": 2, "Sangat memengaruhi": 3}

# a single case is a sitting of minutes, not hours; past this the clock was
# almost certainly left running. Flagged, not deleted.
LONG_MIN = 90

# Known data-entry corrections, applied explicitly rather than by heuristic and
# reported in the data-quality section so no figure is silently altered.
#
# counselor_1 Q42 (no-tool): the «selesai» cell holds `21`, not a clock time. The
# start is a real Excel time (23:37) and the surrounding rows bracket this one
# (Q41 ends 22:32, Q44 starts 01:42), so an end of 21:00 is impossible. Confirmed
# with the researcher 2026-08-25: the counselor entered the ELAPSED MINUTES there,
# i.e. the answer took 21 minutes. Previously this row parsed as `ambiguous` and
# was dropped, leaving counselor_1's no-tool duration at n = 11.
DURATION_OVERRIDES = {
    ("counselor_1", "no_chatbot", "Q42"):
        (21, "«selesai» cell holds elapsed minutes (21), not a clock time; "
             "confirmed with the researcher 2026-08-25"),
}


# ------------------------------------------------------------- parsing -----
def scale(v):
    """"4 – yakin" -> 4. The workbook writes an en-dash; the escape option and
    blanks return None."""
    if v is None:
        return None
    s = str(v).strip()
    m = re.match(r"^([1-5])\s*[–\-]", s)
    return int(m.group(1)) if m else None


def clock(v):
    """A cell holding a time. Returns (minutes-since-midnight, how, raw).

    Three encodings turned up in the returned files: a real Excel time, a
    decimal typed as "HH.MM", and one bare integer that is ambiguous.
    """
    if v is None:
        return None, "blank", ""
    if isinstance(v, dt.time):
        return v.hour * 60 + v.minute, "time", v.strftime("%H:%M")
    if isinstance(v, dt.datetime):
        return v.hour * 60 + v.minute, "time", v.strftime("%H:%M")
    if isinstance(v, (int, float)):
        txt = f"{float(v):.10g}"
        if "." not in txt:                      # e.g. 21 — 21:00 or 00:21?
            return int(txt) % 24 * 60, "ambiguous", txt
        h, _, frac = txt.partition(".")
        mins = int((frac + "00")[:2])           # "23.29" -> 29 ; "21.0" -> 00
        if mins > 59:
            return None, "unparsed", txt
        return int(h) % 24 * 60 + mins, "decimal", f"{int(h):02d}:{mins:02d}"
    s = str(v).strip()
    m = re.match(r"^(\d{1,2})[:.](\d{2})", s)
    if m:
        return int(m.group(1)) % 24 * 60 + int(m.group(2)), "text", s
    return None, "unparsed", s


def duration(start, end):
    """Minutes from start to end, wrapping past midnight like the workbook's
    own MOD(end-start,1) formula."""
    if start is None or end is None:
        return None
    return (end - start) % (24 * 60)


def locate_columns(ws):
    """Map field -> column index by matching the header row."""
    header_row = next(r for r in range(1, 6)
                      if any("Kode kasus" == str(c.value).strip()
                             for c in ws[r] if c.value))
    cols = {}
    for c in ws[header_row]:
        text = " ".join(str(c.value or "").split())
        for field, needle in HEADER_KEYS.items():
            if needle in text:
                cols.setdefault(field, c.column)
    missing = {"case", "answer", "words_cell", "start", "end", "confidence"} - set(cols)
    if missing:
        raise SystemExit(f"{ws.title}: could not locate columns {sorted(missing)}")
    return cols, header_row


def read_workbook(counselor, condition, path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Jawaban"]
    cols, header_row = locate_columns(ws)

    def cell(r, field):
        idx = cols.get(field)
        return ws.cell(r, idx).value if idx else None

    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        case = cell(r, "case")
        if not case or not str(case).strip().startswith("Q"):
            continue
        s_min, s_how, s_raw = clock(cell(r, "start"))
        e_min, e_how, e_raw = clock(cell(r, "end"))
        dur = duration(s_min, e_min)
        ambiguous = "ambiguous" in (s_how, e_how) or "unparsed" in (s_how, e_how)
        override = DURATION_OVERRIDES.get((counselor, condition, str(case).strip()))
        if override:
            dur, _reason = override
            ambiguous = False
            # keep the raw cell visible; the reconstructed end is derived, not read
            e_raw = f"{e_raw} → +{dur}min"
            e_how = "override"
        risk_raw = cell(r, "help_risk")
        harm = cell(r, "harm")
        rows.append(dict(
            counselor=counselor,
            condition=condition,
            case_id=str(case).strip(),
            # Counted from the submitted text, not read from «Jumlah kata».
            # That cell holds a formula in every row but one: counselor_1 Q01 had
            # it overwritten with a literal 173 while the answer beside it is 235
            # words, and reading the cell propagated the stale figure into the arm
            # means. The cell is retained as `words_cell` and any disagreement is
            # reported, so a future overwrite cannot hide the same way.
            words=len(str(cell(r, "answer") or "").split()),
            words_cell=cell(r, "words_cell"),
            time_start=s_raw, time_end=e_raw,
            time_encoding=f"{s_how}/{e_how}",
            duration_min=None if ambiguous else dur,
            duration_flag=("ambiguous" if ambiguous else
                           "long" if dur is not None and dur > LONG_MIN else ""),
            confidence=scale(cell(r, "confidence")),
            help_case=scale(cell(r, "help_case")),
            help_risk=scale(risk_raw),
            help_risk_na=bool(risk_raw and NO_DANGER in str(risk_raw)),
            influence=INFLUENCE.get(str(cell(r, "influence")).strip()),
            harm_flag=(None if harm is None else str(harm).strip()),
            harm_note=cell(r, "harm_note"),
        ))
    return rows


def read_exit(path):
    """The «Kuesioner Akhir» sheet: 10 Likert items, one 0-10 item, free text."""
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Kuesioner Akhir" not in wb.sheetnames:
        return []
    ws = wb["Kuesioner Akhir"]
    header = next(r for r in range(1, ws.max_row + 1)
                  if str(ws.cell(r, 2).value).strip() == "#")
    items = []
    for r in range(header + 1, ws.max_row + 1):
        no = ws.cell(r, 2).value
        if not isinstance(no, (int, float)):
            continue                              # section headings, footer
        text = ws.cell(r, 3).value
        resp = ws.cell(r, 4).value
        # item 10 is the 0-10 overall rating, stored as a bare number; the rest
        # are the "4 – setuju" Likert strings.
        score = resp if isinstance(resp, (int, float)) else scale(resp)
        items.append(dict(item_no=int(no), statement=" ".join(str(text or "").split()),
                          response=None if resp is None else str(resp).strip(),
                          score=score,
                          kind="angka" if isinstance(resp, (int, float))
                               else ("skala" if scale(resp) else "teks")))
    return items


# --------------------------------------------------------------- stats -----
def describe(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return dict(n=0)
    return dict(n=len(vals), mean=round(statistics.mean(vals), 2),
                sd=round(statistics.stdev(vals), 2) if len(vals) > 1 else None,
                median=statistics.median(vals),
                min=min(vals), max=max(vals))


def paired_test(pairs):
    """Wilcoxon signed-rank on (with, without) pairs, plus the sign counts.

    n is small and the pairs carry ties, so the sign counts are the honest
    summary; the p-value is reported beside them, not instead of them.
    """
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    diffs = [a - b for a, b in pairs]
    nz = [d for d in diffs if d != 0]
    out = dict(n_pairs=len(pairs), n_ties=len(diffs) - len(nz),
               higher_with=sum(1 for d in diffs if d > 0),
               higher_without=sum(1 for d in diffs if d < 0),
               mean_diff=round(statistics.mean(diffs), 2) if diffs else None,
               median_diff=statistics.median(diffs) if diffs else None)
    if nz:
        w = stats.wilcoxon([a for a, b in pairs], [b for a, b in pairs],
                           zero_method="wilcox")
        out["wilcoxon_p"] = round(float(w.pvalue), 4)
    else:
        out["wilcoxon_p"] = None
    return out


def counts(values):
    out = {}
    for v in values:
        key = "blank" if v is None else str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def build_stats(rows, exits, key):
    by = {}
    for r in rows:
        by.setdefault((r["counselor"], r["condition"]), []).append(r)

    st = {"n_answers": len(rows),
          "n_cases": len({r["case_id"] for r in rows}),
          "design": "counterbalanced crossover: every case answered once with "
                    "and once without the tool, by different counselors; "
                    "counselor and condition are balanced across the two blocks"}

    # --- per arm ------------------------------------------------------------
    arms = {}
    for (c, cond), rs in sorted(by.items()):
        arms[f"{c}|{cond}"] = dict(
            n=len(rs),
            cases=[r["case_id"] for r in rs],
            block=sorted({key[r["case_id"]]["block"] for r in rs}),
            words=describe([r["words"] for r in rs]),
            duration=describe([r["duration_min"] for r in rs]),
            duration_excl_long=describe([r["duration_min"] for r in rs
                                         if r["duration_flag"] == ""]),
            confidence=describe([r["confidence"] for r in rs]),
            confidence_dist=counts([r["confidence"] for r in rs]),
        )
    st["arms"] = arms

    # --- paired by case, pooled over both counselors ------------------------
    with_ = {r["case_id"]: r for r in rows if r["condition"] == "chatbot"}
    without = {r["case_id"]: r for r in rows if r["condition"] == "no_chatbot"}
    shared = sorted(set(with_) & set(without))
    st["paired"] = {
        field: paired_test([(with_[c][field], without[c][field]) for c in shared])
        for field in ("confidence", "words", "duration_min")
    }
    # --- block split: DIAGNOSTIC ONLY, never a tool effect -------------------
    # Each case pairs ONE counselor with the tool against the OTHER without it,
    # so within a block the difference is
    #     toolEffect(counselor holding the tool) ± [baseline(C1) − baseline(C2)]
    # with the sign flipping between blocks. The baseline gap therefore dominates
    # each block figure and CANCELS when the blocks are pooled — which is exactly
    # what the counterbalancing is for. The pooled test is the estimate of the
    # tool effect; these per-block numbers are not, and were misread as such on
    # 2026-08-25. `baseline_gap` and `implied_tool_effect` are emitted alongside
    # them so the decomposition is visible wherever the block figures are quoted.
    st["paired_by_block"] = {}
    for blk in (1, 2):
        cs = [c for c in shared if key[c]["block"] == blk]
        entry = dict(n=len(cs),
                     with_counselor=sorted({with_[c]["counselor"] for c in cs}))
        for field in ("confidence", "words", "duration_min"):
            t = paired_test([(with_[c][field], without[c][field]) for c in cs])
            holder = sorted({with_[c]["counselor"] for c in cs})[0]
            other = "counselor_2" if holder == "counselor_1" else "counselor_1"
            base_h = describe([r[field] for r in by[(holder, "no_chatbot")]])["median"]
            base_o = describe([r[field] for r in by[(other, "no_chatbot")]])["median"]
            gap = None if base_h is None or base_o is None else base_h - base_o
            t["baseline_gap_holder_minus_other"] = gap
            t["implied_tool_effect"] = (None if gap is None or t["median_diff"] is None
                                        else round(t["median_diff"] - gap, 3))
            t["note"] = ("block figure = tool effect + baseline gap; subtract the gap "
                         "before interpreting, or use the pooled test instead")
            entry[field] = t
        st["paired_by_block"][f"block_{blk}"] = entry

    # --- perceived helpfulness, chatbot arm only ----------------------------
    perc = {}
    for c in ("counselor_1", "counselor_2"):
        rs = by[(c, "chatbot")]
        perc[c] = dict(
            help_case=describe([r["help_case"] for r in rs]),
            help_case_dist=counts([r["help_case"] for r in rs]),
            help_risk=describe([r["help_risk"] for r in rs]),
            help_risk_na=sum(1 for r in rs if r["help_risk_na"]),
            influence=describe([r["influence"] for r in rs]),
            influence_dist=counts([r["influence"] for r in rs]),
            # both framings, because "0 cases had no influence" is a double
            # negative that reads as its own opposite
            influence_none=sum(1 for r in rs if r["influence"] == 0),
            influence_any=sum(1 for r in rs if r["influence"]),
            harm_yes=[dict(case=r["case_id"], note=r["harm_note"]) for r in rs
                      if r["harm_flag"] and r["harm_flag"].lower().startswith("ya")],
        )
    st["perceived"] = perc

    # pooled over both counselors — the danger-sign item is the one place they
    # converge, so it gets a pooled figure the deck can quote directly
    tool_rows = [r for r in rows if r["condition"] == "chatbot"]
    hr = [r["help_risk"] for r in tool_rows if r["help_risk"] is not None]
    hc = [r["help_case"] for r in tool_rows if r["help_case"] is not None]
    st["pooled_perceived"] = dict(
        help_risk=describe(hr), help_risk_at_least_4=sum(1 for v in hr if v >= 4),
        help_case=describe(hc), help_case_at_least_4=sum(1 for v in hc if v >= 4),
    )

    # --- the danger-sign escape option against the frozen risk flag ---------
    # C2 used it; does "no danger signs here" line up with how the study
    # classified the case? A cheap external check on that judgement.
    na_check = {}
    for c in ("counselor_1", "counselor_2"):
        rs = by[(c, "chatbot")]
        risky = [r for r in rs if key[r["case_id"]]["risky"]]
        safe = [r for r in rs if not key[r["case_id"]]["risky"]]
        hit = [r["case_id"] for r in risky if not r["help_risk_na"]]
        miss = [r["case_id"] for r in risky if r["help_risk_na"]]
        false_alarm = [r["case_id"] for r in safe if not r["help_risk_na"]]
        correct_none = [r["case_id"] for r in safe if r["help_risk_na"]]
        na_check[c] = dict(
            n_cases=len(rs),
            used_na=sum(1 for r in rs if r["help_risk_na"]),
            risk_cases_in_arm=len(risky),
            risk_saw_danger=len(hit),
            risk_said_none=len(miss), risk_said_none_cases=miss,
            safe_saw_danger=len(false_alarm), safe_saw_danger_cases=false_alarm,
            safe_said_none=len(correct_none),
            agree=len(hit) + len(correct_none),
            # only meaningful for a counselor who actually used the escape option
            interpretable=bool(sum(1 for r in rs if r["help_risk_na"])),
        )
    st["danger_sign_check"] = na_check

    # --- exit questionnaire -------------------------------------------------
    st["exit"] = exits

    # --- data quality -------------------------------------------------------
    st["data_quality"] = dict(
        duration_missing_in_workbook=sum(
            1 for r in rows if "decimal" in r["time_encoding"]),
        duration_ambiguous=[dict(counselor=r["counselor"], case=r["case_id"],
                                 raw=f'{r["time_start"]}→{r["time_end"]}')
                            for r in rows if r["duration_flag"] == "ambiguous"],
        duration_overridden=[dict(counselor=c, condition=cond, case=case,
                                  minutes=mins, reason=reason)
                             for (c, cond, case), (mins, reason)
                             in DURATION_OVERRIDES.items()],
        duration_long=[dict(counselor=r["counselor"], case=r["case_id"],
                            minutes=r["duration_min"])
                       for r in rows if r["duration_flag"] == "long"],
        words_cell_mismatch=[dict(counselor=r["counselor"], case=r["case_id"],
                                  condition=r["condition"], cell=r["words_cell"],
                                  counted=r["words"],
                                  delta=round(float(r["words_cell"]) - r["words"]))
                             for r in rows
                             if isinstance(r["words_cell"], (int, float))
                             and float(r["words_cell"]) != r["words"]],
        words_out_of_range=[dict(counselor=r["counselor"], case=r["case_id"],
                                 condition=r["condition"], words=r["words"])
                            for r in rows
                            if isinstance(r["words"], (int, float))
                            and not 150 <= r["words"] <= 350],
    )
    return st


# ------------------------------------------------------------- reporting ---
def fmt(d, unit=""):
    if not d.get("n"):
        return "—"
    sd = f" ± {d['sd']}" if d.get("sd") is not None else ""
    return f"{d['mean']}{sd} (median {d['median']}, n={d['n']}){unit}"


def report(st) -> str:
    L = []
    A = st["arms"]
    add = L.append
    add("# Counselor evaluation forms — descriptive analysis\n")
    add(f"_{st['n_answers']} answers over {st['n_cases']} cases. "
        "Counselor-side process and perception data only._\n")

    add("> **This is not the study's primary result.** The primary outcome is the "
        "blind psychologist rating of the counselors' answers. Those workbooks "
        "have not come back — `outputs/analysis/p1_scores.csv` and `p2_scores.csv` "
        "are still empty templates — so whether the tool made the answers *better* "
        "is not answered here. What follows is what the counselors did and what "
        "they thought.\n")

    add("## 1. Design\n")
    add(f"{st['design']}.\n")
    add("| Counselor | Condition | Block | Cases |")
    add("|---|---|---|---|")
    for k, v in A.items():
        c, cond = k.split("|")
        add(f"| {c} | {cond} | {v['block']} | {v['n']} |")
    add("")

    add("## 2. What the counselors produced\n")
    add("| Counselor | Condition | Words | Duration (min) | Self-confidence (1–5) |")
    add("|---|---|---|---|---|")
    for k, v in A.items():
        c, cond = k.split("|")
        add(f"| {c} | {cond} | {fmt(v['words'])} | {fmt(v['duration'])} | "
            f"{fmt(v['confidence'])} |")
    add("")

    add("## 3. With vs without the tool, paired by case\n")
    add("Each case was answered once in each condition, so the 24 cases pair. "
        "Within a pair the two answers come from different counselors; across "
        "the two blocks that confound cancels, which is what the counterbalancing "
        "is for. Sign counts first — n is small and ties are common.\n")
    add("| Measure | Higher with tool | Higher without | Tied | Median diff | Wilcoxon p |")
    add("|---|---|---|---|---|---|")
    labels = dict(confidence="Self-confidence", words="Word count",
                  duration_min="Duration (min)")
    for f, p in st["paired"].items():
        add(f"| {labels[f]} | {p['higher_with']} | {p['higher_without']} | "
            f"{p['n_ties']} | {p['median_diff']} | {p['wilcoxon_p']} |")
    add("")

    add("## 4. Perceived helpfulness (tool arm only)\n")
    add("| Item | Counselor 1 | Counselor 2 |")
    add("|---|---|---|")
    p1, p2 = st["perceived"]["counselor_1"], st["perceived"]["counselor_2"]
    add(f"| Helpful for this case (1–5) | {fmt(p1['help_case'])} | {fmt(p2['help_case'])} |")
    add(f"| Helpful for spotting danger signs (1–5) | {fmt(p1['help_risk'])} | "
        f"{fmt(p2['help_risk'])} |")
    add(f"| Influenced my final answer (0–3) | {fmt(p1['influence'])} | "
        f"{fmt(p2['influence'])} |")
    add(f"| Briefings that influenced the final answer | {p1['influence_any']}/12 | "
        f"{p2['influence_any']}/12 |")
    add(f"| Flagged wrong / potentially harmful content | {len(p1['harm_yes'])}/12 | "
        f"{len(p2['harm_yes'])}/12 |")
    add("")
    for c, p in st["perceived"].items():
        for h in p["harm_yes"]:
            add(f"- {c}, {h['case']}: “{h['note']}”")
    add("")

    add("## 5. Exit questionnaire\n")
    add("| # | Statement | Counselor 1 | Counselor 2 |")
    add("|---|---|---|---|")
    e1 = {i["item_no"]: i for i in st["exit"]["counselor_1"]}
    e2 = {i["item_no"]: i for i in st["exit"]["counselor_2"]}
    for no in sorted(set(e1) | set(e2)):
        stmt = (e1.get(no) or e2.get(no))["statement"]
        add(f"| {no} | {stmt[:80]} | {(e1.get(no) or {}).get('response') or '—'} "
            f"| {(e2.get(no) or {}).get('response') or '—'} |")
    add("")

    add("## 6. Data quality\n")
    dq = st["data_quality"]
    add(f"- Counselor 2 typed clock times as decimals in "
        f"{dq['duration_missing_in_workbook']} rows, so the workbook's duration "
        "formula returned blank. Durations here are reconstructed from the raw "
        "entries, with midnight wrap.")
    if dq["duration_ambiguous"]:
        add(f"- {len(dq['duration_ambiguous'])} row(s) have an unreadable time entry "
            "and are excluded from duration: "
            + ", ".join(f"{d['counselor']} {d['case']} ({d['raw']})"
                        for d in dq["duration_ambiguous"]) + ".")
    else:
        add("- No rows are excluded for unreadable time entries.")
    for d in dq["duration_overridden"]:
        add(f"- **Corrected entry:** {d['counselor']} {d['case']} "
            f"({d['condition']}) duration set to {d['minutes']} min — {d['reason']}.")
    add(f"- {len(dq['duration_long'])} case(s) exceed {LONG_MIN} min, consistent "
        "with a timer left running: "
        + ", ".join(f"{d['counselor']} {d['case']} ({d['minutes']} min)"
                    for d in dq["duration_long"])
        + ". Duration is therefore read as a median, and not treated as a "
          "time-on-task result.")
    add(f"- {len(dq['words_out_of_range'])} answer(s) fall outside the 150–350 "
        "word instruction. Word counts are taken from the submitted text, not "
        "from the «Jumlah kata» cell.")
    if dq["words_cell_mismatch"]:
        add(f"- {len(dq['words_cell_mismatch'])} row(s) where «Jumlah kata» "
            "disagrees with the submitted text: "
            + ", ".join(f"{d['counselor']} {d['case']} ({d['condition']}: cell "
                        f"{d['cell']:g}, counted {d['counted']}, Δ{d['delta']:+d})"
                        for d in dq["words_cell_mismatch"])
            + ". A ±4 gap is the Excel formula's tokenisation; the large one is "
              "counselor_1 Q01, whose formula was overwritten with a literal.")
    add("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forms", type=Path, default=FORMS)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    a = ap.parse_args(argv)

    key = {c["study_id"]: c for c in json.load(open(ASSIGN))["cases"]}

    rows, exits = [], {}
    for counselor, condition, name in WORKBOOKS:
        path = a.forms / name
        if not path.exists():
            raise SystemExit(f"missing workbook: {path}")
        rows.extend(read_workbook(counselor, condition, path))
        if condition == "chatbot":
            exits[counselor] = read_exit(path)

    for r in rows:
        meta = key.get(r["case_id"], {})
        r["block"] = meta.get("block")
        r["risky"] = meta.get("risky")
        expected = meta.get("counselor_condition", {}).get(r["counselor"])
        r["condition_matches_frozen_key"] = (expected == r["condition"])

    bad = [r for r in rows if not r["condition_matches_frozen_key"]]
    if bad:
        print(f"WARNING: {len(bad)} row(s) not in the frozen condition: "
              + ", ".join(f'{r["counselor"]}/{r["case_id"]}' for r in bad))

    a.outdir.mkdir(parents=True, exist_ok=True)
    st = build_stats(rows, exits, key)

    import csv
    cols = list(rows[0].keys())
    with open(a.outdir / "counselor_forms_tidy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (a.outdir / "counselor_forms_stats.json").write_text(
        json.dumps(st, indent=2, ensure_ascii=False))
    (a.outdir / "counselor_forms_report.md").write_text(report(st))

    print(f"wrote {a.outdir/'counselor_forms_tidy.csv'} ({len(rows)} rows)")
    print(f"wrote {a.outdir/'counselor_forms_stats.json'}")
    print(f"wrote {a.outdir/'counselor_forms_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
