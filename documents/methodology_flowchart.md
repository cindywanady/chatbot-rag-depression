# Methodology flowchart — depression RAG chatbot, source → evaluation

Two diagrams: a **one-glance overview** to open the explanation, then a **detailed
stage-by-stage** flow. Speaker-notes at the bottom give one line to say per box.

Render anywhere Mermaid is supported: paste into <https://mermaid.live>, GitHub, or VS Code
(“Markdown Preview Mermaid Support”). Ask if you want this exported as an image/slide.

---

## A. One-glance overview (use this first)

```mermaid
flowchart LR
    S["Knowledge source<br/>MoH guideline (Kemenkes FKTP 2017)"] --> P1["PHASE 1<br/>Build &amp; select retriever"]
    P1 --> P2["PHASE 2<br/>Build RAG chatbot<br/>counsellor decision-support tool"]
    P2 --> EV{{"EVALUATION"}}
    QS["Real questions<br/>Alodokter — 50 sampled"] --> EV
    EV --> AE["Automatic<br/>LLM-judge + RAGAS"]
    EV --> HE["Human — 2-condition study<br/>with / without chatbot, 2 psychologists"]
    AE --> R["Analysis &amp;<br/>conclusions"]
    HE --> R
```

**The story in one breath:** a fixed knowledge source → a retriever we tuned and picked
(Phase 1) → a chatbot built on it as a counsellor decision-support tool (Phase 2) → tested on
real questions, first cheaply by machine, then definitively by humans → analysis.

---

## B. Detailed stage-by-stage flow

```mermaid
flowchart TD
    START(["Indonesian MoH mental-health guideline<br/>(Kemenkes FKTP, 2017) — depression in primary care"])

    subgraph P1 ["PHASE 1 — Retrieval pipeline &amp; bake-off"]
        direction TB
        A1["Extract &amp; clean guideline text"]
        A2["Chunking — 14 configurations"]
        A3["Embedding — 4 models"]
        A4["56 FAISS indexes (14 x 4)"]
        A5["Gold QA set<br/>121 validated questions over 31 passages"]
        A6["Retrieval evaluation<br/>nDCG@5, precision@k"]
        A7["SELECTED retriever<br/>structure-512-0 + e5-large"]
        A1 --> A2 --> A3 --> A4 --> A6
        A5 --> A6 --> A7
    end

    subgraph P2 ["PHASE 2 — RAG chatbot build"]
        direction TB
        B1["Retriever (selected index)"]
        B2["Generator — Gemma-4-12B-it (served via vLLM)"]
        B3["Safety layer<br/>keyword screen + LLM risk classifier + safe response"]
        B4["RAG chatbot"]
        B1 --> B4
        B2 --> B4
        B3 --> B4
        B4 --> DC["Decision-support tool<br/>4-section briefing for the counsellor"]
    end

    subgraph QS ["Evaluation question set"]
        direction TB
        Q1["Scrape Alodokter forum<br/>373 real depression questions"]
        Q2["Sample 50 (fixed seed)<br/>blocks Q1-25 / Q26-50, length-balanced<br/>24/50 flagged as risk cases"]
        Q1 --> Q2
    end

    subgraph AE ["Automatic evaluation — no human"]
        direction TB
        E1["Generate chatbot briefings"]
        E2["LLM-as-judge (Qwen3)<br/>faithfulness, relevancy, completeness, risk"]
        E3["RAGAS<br/>context precision, answer correctness"]
        E1 --> E2
        E1 --> E3
    end

    subgraph HE ["Human evaluation — 2-condition study"]
        direction TB
        H2["Without chatbot — counsellor alone"]
        H3["With chatbot — counsellor + tool"]
        H4["Counterbalanced block design<br/>24-case stratified sample, 2 counsellors, blocks swapped"]
        H5["Patient-facing answers (both conditions)"]
        H6["2 psychologists — BLIND, interleaved sequences<br/>ESHRO quality + Park safety + critical-safety gate<br/>case-level split; 16 shared items -> inter-rater agreement"]
        H7["Counsellor self-report (with-chatbot block only)<br/>per-case usefulness + exit questionnaire"]
        H2 --> H4
        H3 --> H4
        H4 --> H5
        H5 --> H6
        H3 --> H7
    end

    subgraph AN ["Analysis &amp; conclusions"]
        direction TB
        N1["Mixed model — score ~ condition + block + counsellor + random question effect"]
        N3["With vs without chatbot — does the tool help? (PRIMARY question)"]
        N5["Safety-gate failure rate per condition"]
        N6["Perceived usefulness vs objective quality"]
        N7["Expert vs automatic-judge calibration"]
        N1 --> N3
    end

    START --> A1
    A7 --> B1
    DC --> E1
    Q2 --> E1
    Q2 --> H4
    DC --> H3
    H6 --> N1
    H7 --> N6
    E2 --> N7
    H6 --> N7
    H6 --> N5
```

---

## C. Speaker-notes (one line per stage)

1. **Knowledge source.** Everything is grounded in one authoritative document — the Indonesian
   MoH primary-care mental-health guideline — so the chatbot can only speak from validated content.
2. **Phase 1 — retriever.** Before any chatbot, we found the best way to *retrieve* the right
   guideline passage: we built 56 index variants (chunking × embedding), scored them on a
   validated gold question set (nDCG@5), and selected the winner.
3. **Phase 2 — chatbot.** On that retriever we built a RAG chatbot with a safety layer, as a
   decision-support tool that briefs the *counsellor*.
4. **Question set.** We test on *real* questions — 373 scraped from Alodokter, a fixed 50-sample,
   half of them genuine risk cases (suicidal ideation) — not on toy inputs.
5. **Automatic evaluation.** First a cheap, fast machine pass (LLM-as-judge + RAGAS) on the
   chatbot's briefings — this screens quality but cannot judge whether the tool *helps a human*.
6. **Human evaluation.** The core study: two conditions — counsellor alone vs counsellor +
   tool — a frozen 24-case stratified sample, counterbalanced across two counsellors; all
   answers rated blind by two psychologists (case-level split, 16 shared items for
   inter-rater agreement) on a rubric (ESHRO quality + Park safety), plus the counsellor's
   own usefulness ratings in the with-chatbot block.
7. **Analysis.** The primary comparison — does the tool help (with vs without chatbot) —
   plus a safety failure rate per condition, a perception-vs-quality check, and calibration
   of the automatic judge against the expert.

> Numbers (56 indexes, 373→50, 24 risk cases, nDCG@5 value, model names) are from the project
> record; re-confirm against `outputs/analysis/` and the phase notes before quoting exact
> figures to your supervisor (per the no-hallucination rule).
