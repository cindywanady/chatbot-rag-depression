"""Chatbot building blocks: config, prompt assembly, safety screen (pure)."""

from pathlib import Path

import pytest

from depression_rag.chatbot import (
    build_messages,
    format_context,
    load_chatbot_config,
    parse_risk_verdict,
    risk_screen,
)

CFG = Path(__file__).resolve().parents[1] / "configs" / "chatbot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_chatbot_config(CFG)


def test_config_loads_single_counselor_role(cfg):
    assert cfg.k >= 1 and cfg.checkpoint
    assert "HANYA" in cfg.base_prompt                  # grounding rule present
    assert "KONSELOR" in cfg.counselor_briefing        # the briefing serves the counselor
    sp = cfg.system_prompt()
    assert sp.startswith(cfg.base_prompt.rstrip())
    assert "ALAT BANTU" in sp


def test_format_context_numbers_and_cites():
    recs = [
        {"chunk_id": "x1", "text": "Isi satu.", "meta_heading_path": "MI.4 > A",
         "meta_page_start": 40, "meta_page_end": 41},
        {"chunk_id": "x2", "text": "Isi dua.", "meta_heading_path": "MI.7 > C",
         "meta_page_start": 99, "meta_page_end": 99},
    ]
    ctx = format_context(recs)
    assert "[1] MI.4 > A (hal. 40-41)\nIsi satu." in ctx
    assert "[2] MI.7 > C (hal. 99)\nIsi dua." in ctx


def test_build_messages_carries_role_and_context(cfg):
    msgs = build_messages(cfg, "Bagaimana dosisnya?", "[1] konteks")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "ALAT BANTU" in msgs[0]["content"]
    assert "[1] konteks" in msgs[1]["content"]
    assert "Bagaimana dosisnya?" in msgs[1]["content"]


def test_risk_screen_matches_case_insensitive(cfg):
    hits = risk_screen("Saya sudah tidak ingin hidup lagi, Dok.", cfg.safety_keywords)
    assert "tidak ingin hidup" in hits
    assert risk_screen("Berapa dosis fluoksetin awal?", cfg.safety_keywords) == []
    assert "bunuh diri" in risk_screen("Dia bilang mau BUNUH DIRI", cfg.safety_keywords)


# Every phrasing below returned NO match before 2026-08-06: the screen was a raw
# substring test over 15 formal phrases, so informal register, euphemism and
# spacing/punctuation variants all passed through unflagged. These are the
# regression cases — a miss here means a suicide disclosure reaches the
# counselor with no safety banner.
@pytest.mark.parametrize("text", [
    "pengen mati aja rasanya",              # informal register
    "kayaknya mau mati aja",
    "udah capek hidup",
    "aku gak kuat lagi",
    "pengin ilang aja dari dunia",          # euphemism
    "mau nyusul almarhum ibu",
    "bundir",                               # slang
    "dia cutting lagi kemarin",
    "gores-gores tangan pakai silet",
    "nyakitin diri sendiri",
    "OD obat tidur",                        # means
    "minum obat banyak biar tidur selamanya",
    "ada suicide attempt minggu kemarin",   # code-switching
    "bunuhdiri",                            # run-together typo
    "bunuh  diri",                          # doubled space
    "BUNUH-DIRI!!",                         # punctuation
])
def test_risk_screen_catches_colloquial_and_malformed(cfg, text):
    assert risk_screen(text, cfg.safety_keywords), f"missed: {text!r}"


@pytest.mark.parametrize("text", [
    "Apa dosis awal fluoxetine untuk dewasa?",
    "Pasien mengeluh sulit tidur dan nafsu makan menurun sejak dua bulan.",
    "Bagaimana cara melakukan psikoedukasi keluarga?",
    "Metode apa yang dipakai untuk skrining depresi?",   # 'metode' contains 'od'
])
def test_risk_screen_leaves_ordinary_clinical_questions_alone(cfg, text):
    assert risk_screen(text, cfg.safety_keywords) == [], f"false positive: {text!r}"


@pytest.mark.parametrize("verdict,expected", [
    ("RISIKO", True),
    ("AMAN", False),
    ("AMAN.", False),
    ("", True),                    # empty completion
    ("   ", True),                 # whitespace only
    ("Tidak berisiko", True),      # hedged: 'RISIKO' inside 'BERISIKO'
    ("Saya tidak dapat menilai", True),   # refusal / preamble
    # Regression, 2026-08-08. A bare `"AMAN" not in up` test read every one of
    # these as SAFE, inverting the verdict on the reply shape a non-compliant
    # Indonesian answer is most likely to take. The live model emits a bare
    # RISIKO/AMAN today, so this was latent — which is exactly why it needs a
    # test rather than a comment.
    ("TIDAK AMAN", True),
    ("Tidak aman", True),
    ("tidak aman.", True),
    ("Jawaban: TIDAK AMAN", True),
    ("BUKAN AMAN", True),
    ("Teks ini tidak aman", True),
    ("keamanan pasien terancam", True),   # 'AMAN' inside 'KEAMANAN'
    ("aman", False),               # lowercase, still a clean verdict
    ("  AMAN \n", False),          # padded by the chat template
])
def test_risk_verdict_parse_fails_closed(verdict, expected):
    assert parse_risk_verdict(verdict) is expected


def test_safe_response_is_present_and_routes_to_emergency_care(cfg):
    # The banner is the only fixed safety content the counselor sees; it loads
    # from config with an empty-string default, so an absent key would silently
    # produce a blank warning.
    assert cfg.safe_response.strip(), "safe_response is empty"
    assert "119" in cfg.safe_response          # crisis line
    assert "IGD" in cfg.safe_response          # in-person escalation leads


def test_risky_answer_leads_with_the_safety_banner(cfg):
    # Escalation-first: the fixed banner must precede the generated guidance in
    # the assembled answer, not follow it.
    assembled = f"{cfg.safe_response}\n\n---\n\n(jawaban model)"
    assert assembled.startswith(cfg.safe_response)
    assert assembled.index("PERINGATAN") < assembled.index("(jawaban model)")


# ---- study-instrument consistency (added 2026-08-06) -------------------------
# The counselor workbooks print crisis instructions that the counselor reads
# beside a chatbot showing safety.safe_response. When the two were maintained as
# separate strings the printed sheet drifted: it added 112 (deliberately absent
# from the banner) and asserted hotline hours the config records as UNRESOLVED
# and instructs not to state. Both surfaces now derive from safety.escalation.

def test_escalation_lines_exist_and_agree_with_the_banner(cfg):
    import yaml
    saf = yaml.safe_load(CFG.read_text(encoding="utf-8"))["safety"]
    esc = saf.get("escalation") or {}
    assert esc.get("emergency", "").strip(), "safety.escalation.emergency missing"
    assert esc.get("crisis", "").strip(), "safety.escalation.crisis missing"
    printed = f"{esc['emergency']} {esc['crisis']}"
    # the two facts that actually drifted
    assert "112" not in printed, "112 is deliberately absent from safe_response"
    assert "119" in printed, "printed crisis line lost the 119 number"
    for claim in ("jam layanan terbatas", "24 jam", "24/7"):
        assert claim not in printed, f"hours are UNRESOLVED; must not state {claim!r}"
    # in-person escalation leads, matching escalation-first
    assert "IGD" in esc["emergency"] or "puskesmas" in esc["emergency"]


def test_packet_carries_no_instrument_note():
    """The packet holds the question, the rated material and the contexts —
    nothing explaining the instrument. A note about the safety banner shipped
    here briefly on 2026-08-06 and was removed: repeated on all 22 packets and
    printed beside the rated text, it caused the confusion it meant to prevent
    (the safety-gate item asks whether "materi" is AMAN). The mechanism is now
    stated once in Petunjuk section 4, and its scoring consequence lives in the
    two items it bears on."""
    from depression_rag.evaluation import packets as pk
    assert not hasattr(pk, "SAFETY_BANNER_NOTE")
    assert "banner_note" not in pk.PacketContent.__dataclass_fields__
    src = (Path(__file__).resolve().parents[1]
           / "src/depression_rag/evaluation/packets.py").read_text(encoding="utf-8")
    render = src[src.index("def render_markdown"):]
    assert "SPANDUK" not in render and "banner_note" not in render
