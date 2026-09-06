#!/usr/bin/env python3
"""Build the complete Multimedia Appendix document, end to end.

    python scripts/figures/build_full_appendix.py

Writes Multimedia_Appendices.docx (and .md) into documents/thesis_clean/appendix/.
Content that exists is rendered in full; content that does not yet exist is
marked [ADD LATER] with a note saying exactly what is needed.

Wide tables are placed in landscape sections. The CSVs in the same directory
remain the machine-readable companion to the rendered tables.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
APX = ROOT / "documents/thesis_clean/appendix"
APX.mkdir(parents=True, exist_ok=True)
DOCX = APX / "Multimedia_Appendices.docx"
MD = APX / "Multimedia_Appendices.md"

TODO = RGBColor(0x8A, 0x1C, 0x1C)
INK = RGBColor(0x1A, 0x1A, 0x1A)
md_lines: list[str] = []
WITH_LINKS = True   # set per build; links go in the combined document only


# ----------------------------------------------------------------- doc setup
def new_doc() -> Document:
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(7)
    st.paragraph_format.line_spacing = 1.12
    for i, sz in ((1, 15), (2, 12.5), (3, 11)):
        h = d.styles[f"Heading {i}"]
        h.font.name = "Times New Roman"
        h.font.size = Pt(sz)
        h.font.bold = True
        h.font.italic = i == 3
        h.font.color.rgb = INK
        h.paragraph_format.space_before = Pt(14 if i < 3 else 10)
        h.paragraph_format.space_after = Pt(5)
        h.paragraph_format.keep_with_next = True
    s = d.sections[0]
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(1.0)
    return d


def landscape(d: Document) -> None:
    s = d.add_section(WD_SECTION.NEW_PAGE)
    s.orientation = WD_ORIENT.LANDSCAPE
    s.page_width, s.page_height = Inches(11), Inches(8.5)
    s.left_margin = s.right_margin = Inches(0.7)
    s.top_margin = s.bottom_margin = Inches(0.8)


def portrait(d: Document) -> None:
    s = d.add_section(WD_SECTION.NEW_PAGE)
    s.orientation = WD_ORIENT.PORTRAIT
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(1.0)


def h(d, text, level=1, md=True):
    d.add_heading(text, level=level)
    if md:
        md_lines.append(f"\n{'#' * (level + 1)} {text}\n")


def p(d, text, size=None, italic=False, colour=None, md=True, space_after=None):
    par = d.add_paragraph()
    r = par.add_run(text)
    if size:
        r.font.size = Pt(size)
    r.italic = italic
    if colour:
        r.font.color.rgb = colour
    if space_after is not None:
        par.paragraph_format.space_after = Pt(space_after)
    if md:
        md_lines.append(f"\n{text}\n")
    return par


def todo(d, what: str, why: str):
    par = d.add_paragraph()
    r = par.add_run("[ADD LATER] ")
    r.bold = True
    r.font.color.rgb = TODO
    r2 = par.add_run(f"{what} — {why}")
    r2.font.color.rgb = TODO
    md_lines.append(f"\n**[ADD LATER]** {what} — {why}\n")


def verbatim(d, text: str, md=True):
    par = d.add_paragraph()
    par.paragraph_format.left_indent = Inches(0.25)
    par.paragraph_format.space_after = Pt(9)
    par.paragraph_format.space_before = Pt(3)
    r = par.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.2)
    if md:
        md_lines.append("\n```\n" + text + "\n```\n")


def shade(cell, hexcol="E8E8E8"):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcol)
    cell._tc.get_or_add_tcPr().append(el)


def table(d, header, rows, font=8.0, widths=None, md=True):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, cell in enumerate(t.rows[0].cells):
        cell.text = ""
        par = cell.paragraphs[0]
        par.paragraph_format.space_after = Pt(1)
        par.paragraph_format.space_before = Pt(1)
        r = par.add_run(str(header[i]))
        r.bold = True
        r.font.size = Pt(font)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.paragraph_format.space_after = Pt(1)
            par.paragraph_format.space_before = Pt(1)
            r = par.add_run("" if v is None else str(v))
            r.font.size = Pt(font)
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    if md:
        md_lines.append("\n| " + " | ".join(str(x) for x in header) + " |")
        md_lines.append("|" + "---|" * len(header))
        for row in rows:
            md_lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        md_lines.append("")
    return t


def read_csv(name):
    with (APX / name).open(encoding="utf-8") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


# ================================================================= APPENDIX 1
SPEC_FILES = [
    ("A1.1", "Question generation for the retrieval evaluation set", "configs/prompts/qa_generation.md"),
    ("A1.2", "Concept induction for the reference passages (stage 1)", "configs/prompts/gold_concept_induction.md"),
    ("A1.3", "Concept and split annotation for the reference passages (stage 2)", "configs/prompts/gold_passage_concept_audit.md"),
    ("A1.4", "Content-type audit of the source segments", "configs/prompts/content_type_audit.md"),
]
YAML_KEYS = [("A1.5", "Deployed system prompt: shared base", "base"),
             ("A1.6", "Deployed briefing prompt: counselor briefing", "counselor_briefing")]
CODE_CONSTS = [("A1.7", "Claim-support judge", "FAITHFULNESS_PROMPT"),
               ("A1.8", "Structural compliance judge", "STRUCTURE_PROMPT"),
               ("A1.9", "Question-level risk classification (evaluation)", "QUESTION_RISK_PROMPT"),
               ("A1.10", "Briefing safety-content assessment", "RISK_PROMPT")]
CODE_FILE = "scripts/judge_study_answers.py"


def prompt_body(path: Path) -> str:
    """Return the prompt exactly as it was sent to the model.

    Each source file wraps its prompt in a fenced block and surrounds it with the
    author's working notes: version tags, rationale tables and cross-references to
    internal design documents. Only the fenced block was ever sent to a model, so
    only the fenced block belongs in the appendix. Extracting it also keeps
    internal document references out of the published record.
    """
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^```text\n(.*?)^```", text, re.S | re.M)
    if not m:
        raise ValueError(f"no fenced prompt block in {path}")
    return m.group(1).strip("\n")


def appendix_1(d):
    h(d, "Multimedia Appendix 1. Prompts and evaluation criteria", 1)
    p(d, "Each prompt below is reproduced exactly as it was submitted to the model. Prompts A1.1 to "
         "A1.4 were run interactively and are given as issued; A1.5 and A1.6 are the deployed "
         "system prompts; A1.7 to A1.10 are the evaluation prompts. The deployed and evaluation "
         "prompts are in Indonesian, because the source corpus, the study questions, the "
         "counselors and the psychologists are Indonesian.")

    rows = [[t, n, s] for t, n, s in
            [(t, n, f"configs/prompts/{Path(r).name}") for t, n, r in SPEC_FILES]
            + [(t, n, f"configs/chatbot.yaml -> prompts.{k}") for t, n, k in YAML_KEYS]
            + [(t, n, f"{CODE_FILE} -> {c}") for t, n, c in CODE_CONSTS]]
    table(d, ["Section", "Prompt", "Source in the study repository"], rows,
          font=9, widths=[0.8, 3.3, 2.5])

    for tag, title, rel in SPEC_FILES:
        h(d, f"{tag} {title}", 2)
        p(d, f"Source: {rel}", size=9, italic=True, space_after=4)
        verbatim(d, prompt_body(ROOT / rel))

    cfg = yaml.safe_load((ROOT / "configs/chatbot.yaml").read_text(encoding="utf-8"))
    prompts = cfg.get("prompts", {})
    for tag, title, key in YAML_KEYS:
        h(d, f"{tag} {title}", 2)
        p(d, f"Source: configs/chatbot.yaml, key prompts.{key}", size=9, italic=True, space_after=4)
        verbatim(d, str(prompts.get(key, "")).strip())

    src = (ROOT / CODE_FILE).read_text(encoding="utf-8")
    for tag, title, name in CODE_CONSTS:
        h(d, f"{tag} {title}", 2)
        p(d, f"Source: {CODE_FILE}, constant {name}", size=9, italic=True, space_after=4)
        m = re.search(rf'^{name}\s*=\s*"""(.*?)"""', src, re.S | re.M)
        verbatim(d, m.group(1).strip("\n") if m else "[NOT FOUND]")

# ================================================================= APPENDIX 2
def appendix_2(d):
    h(d, "Multimedia Appendix 2. Ethics approval and consent materials", 1)
    p(d, "The study protocol was reviewed by the Research Ethics Committee, Institute for Research "
         "and Community Service, Universitas Negeri Malang, and determined to be ethically exempt "
         "(No. 05.08.24/UN32.14.2.8/LT/2026, 5 August 2026). Counselors and psychologists "
         "participated voluntarily, gave written informed consent, and were compensated.")
    p(d, "Signed consent forms carry participant names and signatures. With four participants in two "
         "named roles at a single organisation, a name and a role together are identifying. The "
         "signed forms are therefore retained in the study records and are not reproduced here; the "
         "blank template below documents the consent process.", size=10)

    table(d, ["Section", "Document", "Status"], [
        ["A2.1", "Ethical exemption determination", "Appended"],
        ["A2.2", "Participant information sheet", "[ADD LATER]"],
        ["A2.3", "Consent form, counselor role (blank template)", "[ADD LATER]"],
        ["A2.4", "Consent form, psychologist role (blank template)", "[ADD LATER]"],
        ["A2.5", "Recruitment flyer, counselor role", "[ADD LATER]"],
        ["A2.6", "Recruitment flyer, psychologist role", "[ADD LATER]"],
    ], font=9, widths=[0.8, 3.6, 2.2])

    h(d, "A2.1 Ethical exemption determination", 2)
    p(d, "Reference No. 05.08.24/UN32.14.2.8/LT/2026, issued 5 August 2026, one page, bilingual "
         "Indonesian and English.", size=10)
    todo(d, "Append the determination letter as a page image",
         "the source file is held in the study records as a single-page scanned PDF and is inserted "
         "here at submission")

    h(d, "A2.2 Participant information sheet", 2)
    todo(d, "Insert the information sheet issued to participants",
         "two versions exist in the study records, one dated 6 July 2026 and a revision dated "
         "9 August 2026; confirm which was actually distributed and include that one, with all "
         "bracketed fields completed")

    for tag, role in (("A2.3", "counselor"), ("A2.4", "psychologist")):
        h(d, f"{tag} Consent form, {role} role (blank template)", 2)
        todo(d, f"Insert the unsigned consent template for the {role} role",
             "the blank form with an empty signature block. Do not substitute a redacted scan of a "
             "signed form: redaction applied to a scanned image is frequently reversible")

    for tag, role in (("A2.5", "counselor"), ("A2.6", "psychologist")):
        h(d, f"{tag} Recruitment flyer, {role} role", 2)
        todo(d, f"Insert the {role} recruitment flyer as an image",
             "export the source slide to PNG or PDF and place it at roughly half to two-thirds page "
             "width. Remove any personal telephone number, messaging handle or email address before "
             "inclusion; the name of the recruiting organisation may remain")


# ================================================================= APPENDIX 3
S3_SHORT = {
    "clinical_correctness": "Correct", "completeness": "Complete", "case_relevance": "Relev",
    "clarity_practicality": "Clarity", "harm_avoidance": "Harm", "risk_recognition": "RiskRec",
    "immediate_escalation": "Escal", "resources_referral": "Refer",
    "decision_space_role_boundary": "Role", "scope_appropriateness": "Scope",
    "safety_gate": "Gate", "context_sufficient": "CtxSuf", "support_for_statements": "Suppt",
    "unsupported_statement": "Unsup", "global_0_10": "Global",
    "rater_danger_sign_judgment": "Danger", "condition_outside_depression": "OutDep",
}


def appendix_3(d):
    h(d, "Multimedia Appendix 3. Supplementary results", 1)
    p(d, "Tables S1 to S7 below support the Results. Each is also provided as a comma-separated "
         "value file in the submission package, which is the machine-readable version of the same "
         "content. Hit@k denotes the proportion of questions with at least one relevant passage in "
         "the top k, and corresponds to the column named recall@k in the analysis files.")

    h(d, "Code and data", 2)
    p(d, "The analysis code, the retrieval and indexing pipeline, the deployed chatbot, the "
         "evaluation harness and the prompts reproduced in Multimedia Appendix 1 are available in "
         "a public repository. The repository contains the code and configuration only; it holds "
         "no participant data. A supplementary archive additionally holds the complete original "
         "files behind the tables below, for a reader who wants to inspect a source document "
         "rather than the extract reproduced here. Nothing in this appendix depends on either "
         "link: every value reported is printed in full in the tables that follow.")
    if WITH_LINKS:
        for label, value in (
            ("Code repository", "[INSERT GITHUB URL]"),
            ("Supplementary archive", "[INSERT GOOGLE DRIVE LINK]"),
            ("Archive access", "view-only, open to anyone holding the link, no sign-in required"),
        ):
            par = d.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.25)
            par.paragraph_format.space_after = Pt(3)
            r = par.add_run(f"{label}: ")
            r.bold = True
            par.add_run(value)
            md_lines.append(f"\n**{label}:** {value}\n")
    table(d, ["Material", "Where it is available"], [
        ["Analysis code, pipeline, chatbot, evaluation harness",
         "Public repository, cited in Data Availability"],
        ["Prompts as submitted to the models", "Multimedia Appendix 1"],
        ["Retrieval results for every scored index", "Tables S1a to S1c, and as CSV"],
        ["Per-briefing automatic evaluation values", "Table S2, and as CSV"],
        ["Participant ratings, anonymised",
         "Tables S3 to S6, and as CSV. Not released in any fuller form"],
        ["Allocation record and input hashes", "Tables S7 and S7b, and as CSV"],
        ["Ministry of Health guideline module",
         "Third-party publication; cited, not redistributed"],
        ["Forum question corpus",
         "De-identified but not anonymous; source cited, not redistributed"],
    ], font=9, widths=[3.0, 3.4])
    if WITH_LINKS:
        todo(d, "Fill both address fields above, and confirm the archive opens without sign-in",
             "open the archive link in a private browsing window while signed out of every "
             "account; a request-access screen means the sharing setting is wrong and a reader "
             "will be blocked. If a university-issued account hosts the archive, move it to one "
             "that will outlast graduation. For the repository, consider archiving a tagged "
             "release to a service that mints a persistent identifier, so the version underlying "
             "this study stays retrievable even if the repository is renamed or removed")

    # ---- S1a (landscape)
    landscape(d)
    h(d, "Table S1a. Retrieval performance of every scored index", 2)
    p(d, "All 71 scored indexes: 56 from the selection grid (14 chunking configurations by 4 "
         "encoders), 14 post hoc BGE-M3 indexes, and the multi-vector variant that became the "
         "deployed index. Encoder window is the model's maximum sequence length in tokens; "
         "truncated chunks counts chunks longer than that window. Intervals are 95% percentile "
         "bootstrap intervals from 10,000 resamples over the 121 benchmark questions.", size=9)
    hdr, rows = read_csv("Table_S1a_retrieval_grid.csv")
    keep = ["index_id", "chunk_config", "embedding_model", "encoder_window_tokens", "n_chunks",
            "n_vectors", "truncated_chunks", "nDCG@5", "nDCG@5_CI_low", "nDCG@5_CI_high",
            "Hit@5", "MRR@5", "Hit@10"]
    idx = [hdr.index(c) for c in keep]
    out = []
    for r in rows:
        v = [r[i] for i in idx]
        out.append(v[:7] + [f"{v[7]} ({v[8]}–{v[9]})"] + v[10:])
    table(d, ["Index", "Chunk configuration", "Encoder", "Window", "Chunks", "Vectors",
              "Trunc.", "nDCG@5 (95% CI)", "Hit@5", "MRR@5", "Hit@10"], out,
          font=7.0,
          widths=[2.15, 1.25, 1.05, 0.55, 0.5, 0.5, 0.45, 1.35, 0.5, 0.5, 0.5])

    h(d, "Table S1b. Declared paired-comparison family", 2)
    p(d, "The leading index compared with the six next-highest-ranked indexes. Holm adjustment is "
         "across this family of six. The leader is structure-512-0 with multilingual E5, "
         "nDCG@5 = 0.798 (95% CI 0.747–0.846).", size=9)
    hdr, rows = read_csv("Table_S1b_holm_comparisons.csv")
    table(d, ["Challenger index", "Mean nDCG@5", "Difference", "95% CI of difference",
              "P bootstrap", "P Wilcoxon", "P Holm", "Rank-biserial", "Significant"],
          rows, font=7.5,
          widths=[2.3, 0.85, 0.8, 1.35, 0.8, 0.8, 0.7, 0.85, 0.75])
    portrait(d)

    h(d, "Table S1c. Deployed index by question type", 2)
    p(d, "The deployed multi-vector index across all 121 benchmark questions, ordered by "
         "nDCG@5. The manuscript reports only the two strata the recorded tie-breakers used, "
         "so the full profile is given here. Pharmacotherapy and dosing is the largest "
         "category, and the deployed chatbot does not present that content as a counselor "
         "action.", size=9)
    hdr, rows = read_csv("Table_S1c_deployed_by_question_type.csv")
    table(d, ["Question type", "Questions", "nDCG@5", "Hit@5", "Hit@10",
              "Questions with a relevant passage in top 5"],
          [[r[0].replace("_", " ").capitalize()] + r[1:] for r in rows], font=8.5,
          widths=[1.7, 0.8, 0.7, 0.7, 0.7, 1.8])

    # ---- S2
    h(d, "Table S2. Automatic evaluation of each chatbot output", 2)
    p(d, "All 50 outputs. Q03 and Q38 received the fixed out-of-scope reply, contain no clinical "
         "claims, and are excluded from the summary statistics reported in the Results, which cover "
         "the 48 in-scope briefings.", size=9)
    hdr, rows = read_csv("Table_S2_per_briefing_automatic.csv")
    table(d, ["Question", "In scope", "Sections", "Claim support", "RAGAS faithfulness",
              "RAGAS answer relevancy", "RAGAS context precision"], rows, font=8,
          widths=[0.7, 1.35, 0.65, 0.9, 1.05, 1.15, 1.15])

    # ---- S3 (landscape)
    landscape(d)
    h(d, "Table S3. Psychologist ratings of each briefing packet, by rater", 2)
    p(d, "All 44 ratings of the 36 briefing packets; 8 packets were rated by both psychologists. "
         "Items use 1–5 scales with 5 most favourable unless noted. Column abbreviations: "
         "Correct, clinical correctness; Complete, completeness for the case; Relev, case relevance; "
         "Clarity, clarity and practicality; Harm, harm avoidance in advice; RiskRec, risk "
         "recognition and handling; Escal, immediate escalation; Refer, resources and referral "
         "pathway; Role, decision space and role boundary; Scope, scope appropriateness "
         "(conditional, applied inconsistently between raters); Gate, binary safety gate; CtxSuf, "
         "retrieved passages sufficient; Suppt, support for the briefing's main statements; Unsup, "
         "unsupported clinical statement recorded; Global, global rating 0–10; Danger, the "
         "rater's own danger-sign judgment on the question; OutDep, a condition outside depression "
         "was recorded. Blank cells are items the rater did not apply.", size=8.5)
    hdr, rows = read_csv("Table_S3_psychologist_briefing_ratings.csv")
    short = ["Rater", "Case"] + [S3_SHORT.get(c, c) for c in hdr[2:]]
    rows2 = [[r[0].replace("psychologist_", "P")] + r[1:] for r in rows]
    table(d, short, rows2, font=7.0,
          widths=[0.42, 0.45] + [0.5] * (len(short) - 2))

    h(d, "Table S4. Scope appropriateness by rater", 2)
    p(d, "Reported by rater rather than pooled, because the item conditioning scope "
         "appropriateness was applied inconsistently between the two psychologists.", size=9)
    hdr, rows = read_csv("Table_S4_scope_appropriateness_by_rater.csv")
    table(d, ["Rater", "Ratings", "Condition outside depression, n",
              "Condition outside depression, %", "Scope item rated, n", "Scope item mean"],
          [[r[0].replace("psychologist_", "Psychologist ").replace("all_ratings", "All ratings")]
           + r[1:] for r in rows], font=8.5,
          widths=[1.5, 0.9, 1.9, 1.9, 1.5, 1.3])
    portrait(d)

    # ---- S5 (landscape)
    landscape(d)
    h(d, "Table S5. Counselor process measures for each case", 2)
    p(d, "All 48 counselor responses. Start and end times are clock times recorded by the counselor; "
         "elapsed time is their difference. Self-confidence and the two helpfulness items use "
         "1–5 scales. Influence and problem-flag items were collected only in the "
         "assigned-access condition. Free-text problem descriptions are held in the study records "
         "pending anonymisation review.", size=9)
    hdr, rows = read_csv("Table_S5_counselor_per_case_process.csv")
    table(d, ["Question", "Counselor", "Condition", "Start", "End", "Words", "Confidence",
              "Helpful, case", "Helpful, danger signs", "Influence on answer", "Problem flagged"],
          [[r[0], r[1].replace("counselor_", "C"), r[2].replace("no_chatbot", "no access")
            .replace("chatbot", "assigned access")] + r[3:] for r in rows], font=7.5,
          widths=[0.62, 0.62, 1.1, 0.55, 0.55, 0.5, 0.72, 0.72, 0.95, 1.35, 0.85])

    h(d, "Table S6. Pair-level counselor response ratings", 2)
    p(d, "The 24 matched question pairs. Each pair contributed one response written with assigned "
         "access and one written without, by different counselors. Ratings for the 8 responses "
         "rated by both psychologists were averaged before the pair was formed. A positive "
         "difference favours the response written with assigned access.", size=9)
    hdr, rows = read_csv("Table_S6_pair_level_response_ratings.csv")
    table(d, ["Question", "Response ID", "Counselor", "Rating", "Raters",
              "Response ID", "Counselor", "Rating", "Raters", "Difference", "Direction"],
          [[r[0], r[1], r[2].replace("counselor_", "C"), r[3], r[4],
            r[5], r[6].replace("counselor_", "C"), r[7], r[8], r[9], r[10]] for r in rows],
          font=7.5,
          widths=[0.65, 0.8, 0.7, 0.6, 0.55, 0.8, 0.7, 0.6, 0.55, 0.8, 0.8])
    p(d, "Columns 2 to 5 describe the response written with assigned access; columns 6 to 9 the "
         "response written without.", size=8, italic=True)
    portrait(d)

    # ---- S7
    h(d, "Table S7. Allocation record", 2)
    p(d, "The frozen allocation for all 50 study questions, generated from the project seed "
         "20260628 on 1 August 2026. Screen-positive marks questions flagged by the deployed "
         "two-stage risk screen. Instrument form records whether the psychologist packet used the "
         "short or the extended grounding form.", size=9)
    hdr, rows = read_csv("Table_S7_allocation_record.csv")
    table(d, ["Question", "Source ID", "Block", "Screen positive", "In counselor study",
              "Counselor 1", "Counselor 2", "In psychologist set", "Assigned rater", "Form"],
          [[r[0], r[1], r[2], r[3], r[4],
            r[5].replace("no_chatbot", "no access").replace("chatbot", "access"),
            r[6].replace("no_chatbot", "no access").replace("chatbot", "access"),
            r[7], r[8].replace("psychologist_", "P"), r[9]] for r in rows], font=7.5,
          widths=[0.6, 0.7, 0.4, 0.75, 0.85, 0.7, 0.7, 0.8, 0.55, 0.5])

    h(d, "Table S7b. Input hashes for the allocation record", 2)
    p(d, "SHA-256 of the frozen inputs from which the allocation was drawn, recorded so that the "
         "allocation can be verified against the exact files used.", size=9)
    hdr, rows = read_csv("Table_S7b_input_hashes.csv")
    table(d, ["Input file", "SHA-256"], rows, font=7.5, widths=[2.4, 4.0])

    h(d, "A3.1 Open-text participant feedback", 2)
    todo(d, "Insert the anonymised open-text feedback from both participant roles",
         "the psychologists' written feedback and the counselors' exit comments. These were "
         "excluded from the tables above because deciding what is safe to publish is a judgment "
         "that should not be automated. Before inclusion, remove anything identifying a "
         "participant, a colleague, a help-seeker or the service, and check for clinical detail "
         "quoted from a source forum post, since that text is de-identified but not anonymous")

    h(d, "A3.2 Mixed-effects model output", 2)
    todo(d, "Insert the fitted mixed-effects model summary for the primary outcome",
         "the model specified in the protocol, with its coefficient, standard error, interval and "
         "P value. Resolve the discrepancy between the analysis output and the manuscript before "
         "exporting, so that the two agree")


# ================================================================= APPENDIX 4
def appendix_4(d):
    h(d, "Multimedia Appendix 4. Evaluation instruments", 1)
    p(d, "The instruments issued to participants. Blank templates are reproduced here; the "
         "completed workbooks are the primary record and their contents are published in "
         "anonymised form as Tables S3, S5 and S6.")
    table(d, ["Section", "Instrument", "Status"], [
        ["A4.1", "Psychologist briefing and response rating workbook (blank)", "[ADD LATER]"],
        ["A4.2", "Counselor response workbook, assigned-access condition (blank)", "[ADD LATER]"],
        ["A4.3", "Counselor response workbook, no-access condition (blank)", "[ADD LATER]"],
        ["A4.4", "Counselor exit questionnaire (blank)", "[ADD LATER]"],
        ["A4.5", "Example briefing packet as presented to raters", "[ADD LATER]"],
        ["A4.6", "Study guide and participant briefing materials", "[ADD LATER]"],
    ], font=9, widths=[0.8, 4.2, 1.6])

    h(d, "A4.1 Psychologist rating workbook", 2)
    p(d, "The instrument used anchored 1–5 ratings of clinical correctness, completeness, case "
         "relevance, clarity and practicality, harm avoidance in advice, danger-sign recognition "
         "and handling, immediate escalation, resources and referral pathway, decision space and "
         "role boundary, and scope appropriateness, together with a binary safety gate and a 0–10 "
         "global rating. A first item recorded the rater's own judgment of whether the question "
         "carried danger signs. An extended form added three grounding items covering sufficiency "
         "of the retrieved passages, support for the briefing's main statements, and the presence "
         "of unsupported clinical statements.", size=10)
    todo(d, "Insert the blank workbook",
         "the template handed to raters, with all response cells empty")

    h(d, "A4.2 and A4.3 Counselor response workbooks", 2)
    p(d, "Counselors received one workbook per block. Each case recorded the response text, start "
         "and end time, and self-rated confidence. In the assigned-access condition the workbook "
         "additionally recorded perceived helpfulness for the case, helpfulness for recognising "
         "danger signs, whether and how far the briefing influenced the final response, and "
         "whether any briefing content was incorrect or potentially harmful.", size=10)
    todo(d, "Insert both blank workbooks",
         "the assigned-access and no-access templates, with all response cells empty")

    h(d, "A4.4 Counselor exit questionnaire", 2)
    p(d, "Administered once at the end of participation. Items adapted the perceived usefulness "
         "and perceived ease of use constructs of the technology acceptance model, and added items "
         "on trust in the briefing content and on independent checking of that content before use, "
         "together with a 0–10 overall usefulness rating and free-text fields.", size=10)
    todo(d, "Insert the blank questionnaire", "the template as issued")

    h(d, "A4.5 Example briefing packet", 2)
    p(d, "Each of the 36 packets presented the study question, the generated briefing, and the five "
         "retrieved passages supplied to the generator, in that order. The fixed safety banner was "
         "excluded from the packets, so raters assessed the model-generated briefing rather than "
         "the complete interface a counselor would see.", size=10)
    todo(d, "Insert two or three complete packets as examples",
         "reproducing all 36 is unnecessary; the per-packet ratings are in Table S3")

    h(d, "A4.6 Study guide and briefing materials", 2)
    todo(d, "Insert the study guide and the role briefing decks",
         "the written guide issued to participants and the briefing slides used at induction")


# ================================================================= assemble
BUILDERS = [
    (1, "Prompts_and_Evaluation_Criteria", appendix_1),
    (2, "Ethics_and_Consent", appendix_2),
    (3, "Supplementary_Results", appendix_3),
    (4, "Evaluation_Instruments", appendix_4),
]
TITLE = ("Counselor-Facing Retrieval-Augmented Generation Chatbot Grounded in an Indonesian "
         "Ministry of Health Depression Module: Development and Multistage Evaluation Study")


def cover(d, title: str, note: bool = True):
    par = d.add_paragraph()
    r = par.add_run(title)
    r.bold = True
    r.font.size = Pt(18)
    md_lines.append(f"# {title}\n")
    p(d, TITLE, size=11, italic=True)
    if note:
        p(d, "Items marked [ADD LATER] are documents to be inserted before submission; each says "
             "what is needed. Everything else is complete.", size=9.5, italic=True)


def build_combined() -> None:
    """One document containing every appendix, for the thesis PDF. Carries the links."""
    global WITH_LINKS
    WITH_LINKS = True
    md_lines.clear()
    d = new_doc()
    cover(d, "Multimedia Appendices")
    h(d, "Contents", 1)
    table(d, ["Appendix", "Contents", "Status"], [
        ["1", "Prompts and evaluation criteria", "Complete except A1.11"],
        ["2", "Ethics approval and consent materials", "Documents to be inserted"],
        ["3", "Supplementary results, Tables S1\u2013S7b", "Complete except A3.1 and A3.2"],
        ["4", "Evaluation instruments", "Documents to be inserted"],
    ], font=9.5, widths=[0.9, 3.6, 2.1])
    for _, _, fn in BUILDERS:
        fn(d)
    d.save(DOCX)
    MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  {DOCX.name}")
    print(f"  {MD.name}")


def build_separate() -> None:
    """One document per appendix, as the journal uploads them. No URLs: the journal's
    instructions require supplementary files to be uploaded and keep URLs out of the
    manuscript, so the address fields appear only in the combined thesis document."""
    global WITH_LINKS
    WITH_LINKS = False
    for n, slug, fn in BUILDERS:
        md_lines.clear()
        d = new_doc()
        cover(d, f"Multimedia Appendix {n}", note=True)
        fn(d)
        out = APX / f"Multimedia_Appendix_{n}_{slug}.docx"
        d.save(out)
        print(f"  {out.name}")


def main() -> int:
    print(f"writing to {APX.relative_to(ROOT)}/\n")
    print("combined, for the thesis PDF:")
    build_combined()
    print("\nseparate, for JMIR upload:")
    build_separate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
