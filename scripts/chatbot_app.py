#!/usr/bin/env python
"""Gradio UI for the depression case-management RAG chatbot (Phase 2).

Alternative frontend; the primary one is ``scripts/chatbot_web.py``, which
serves the project's own chat design from ``static/index.html``. This Gradio
variant exists because it can mint a free public ``gradio.live`` link
(``--share``), so it mirrors the same chat experience: message bubbles,
streamed typing, an in-bubble risk alert, and a citations panel. Both share the same
answer path (:class:`depression_rag.chatbot.ChatEngine`): retrieval over the
selected index, safety screen, counselor-briefing prompt assembly,
generation, and per-interaction logging to outputs/chat_logs/.

Run from the Phase-2 environment:

    .venv-chat/bin/python scripts/chatbot_app.py                 # port 7860
    .venv-chat/bin/python scripts/chatbot_app.py --port 7861 --dry-run

The generator is served separately by vLLM (scripts/serve_chatbot.sh).
Without it the app runs in DRY-RUN mode: retrieval, safety screen and prompt
assembly all work, and the assembled prompt is shown instead of an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CITES_PLACEHOLDER = ("*Kutipan pedoman yang dipakai untuk menjawab akan muncul di sini.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from depression_rag.chatbot import (  # noqa: E402
    ChatEngine,
    load_chatbot_config,
)


AUTH_ENV = "CHATBOT_AUTH"


def resolve_auth(require: bool) -> list[tuple[str, str]] | None:
    """Credential pairs for gradio's ``auth=``, read from the environment.

    Deliberately NOT a command-line flag: argv is world-readable through
    ``ps``/``/proc/<pid>/cmdline`` on a shared cluster, so a password passed
    that way would be visible to every other account on the node. An env var is
    readable only by the owning user.

    Format: ``CHATBOT_AUTH="user:pass"``, or several separated by commas.
    Fails loudly rather than silently launching an open link.
    """
    raw = os.environ.get(AUTH_ENV, "").strip()
    if not raw:
        if require:
            raise SystemExit(
                f"--require-auth was given but ${AUTH_ENV} is empty.\n"
                f'Set it first, e.g.:  export {AUTH_ENV}="konselor1:<password>"\n'
                "(export it rather than passing it as a flag: command lines are "
                "readable by other users on this node.)")
        return None
    pairs = []
    for chunk in raw.split(","):
        user, sep, password = chunk.partition(":")
        if not sep or not user.strip() or not password:
            raise SystemExit(
                f"${AUTH_ENV} entry {chunk!r} is malformed; expected user:pass")
        pairs.append((user.strip(), password))
    return pairs


def build_app(cfg_path: str, dry_run: bool, log_dir: str | None = None):
    import dataclasses

    import gradio as gr

    cfg = load_chatbot_config(cfg_path)
    if log_dir:  # per-instance log folder (e.g. one per counselor)
        cfg = dataclasses.replace(cfg, log_dir=log_dir)
    print(f"[app] loading engine: {cfg.index_dir} (model {cfg.model}, k={cfg.k}, "
          f"logs={cfg.log_dir})")
    engine = ChatEngine(cfg, root=ROOT, dry_run=dry_run)

    def respond(question: str, history: list):
        question = (question or "").strip()
        history = history or []
        if not question:
            yield history, CITES_PLACEHOLDER, ""
            return
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": ""}]
        cites = CITES_PLACEHOLDER
        # clear the input box immediately, then stream the answer in
        yield history, cites, ""
        for kind, payload in engine.answer_stream(question, cfg.k):
            if kind == "meta":
                if payload["risky"]:
                    # lead the bot bubble itself with the red risk alert
                    history[-1]["content"] = (
                        '<div class="risk-alert"><strong>Deteksi risiko aktif</strong> '
                        f'({payload["risk_reason"]}) — protokol keselamatan diterapkan.</div>\n\n')
                # Every line needs its own "> ": a blank line ENDS a markdown
                # blockquote, so quoting only the first line let the rest of the
                # passage render as ordinary body text — indistinguishable from
                # the chatbot's own answer. The snippet is already reflowed
                # (chatbot.py), so these are real content lines.
                cites = "\n\n".join(
                    f"**[{c['n']}] {c['heading']} (hal. {c['pages']})** "
                    f"`{c['chunk_id']}`\n\n"
                    # no trailing "…": the passage is complete now
                    + "\n".join(f"> {ln}" for ln in c["snippet"].split("\n"))
                    for c in payload["citations"])
                yield history, cites, ""
            elif kind == "delta":
                history[-1]["content"] += payload
                yield history, cites, ""
        yield history, cites, ""

    theme = getattr(gr.themes, "Soft", gr.themes.Base)(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
    )
    css = """
    .gradio-container {max-width: 1460px !important; margin: 0 auto !important;}
    #hdr {background: linear-gradient(120deg, #064e3b 0%, #047857 55%, #059669 100%);
          color: #fff; border-radius: 14px; padding: 20px 26px; margin: 4px 0 10px;}
    #hdr h1 {color: #fff; margin: 0 0 6px; font-size: 1.4rem; letter-spacing: .2px;}
    #hdr p {color: #a7f3d0; margin: 0; font-size: .9rem; line-height: 1.45;}
    #hdr .badge {display: inline-block; margin-top: 10px; margin-right: 8px;
          background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.35);
          color: #ecfdf5; padding: 3px 12px; border-radius: 999px; font-size: .78rem;
          letter-spacing: .3px;}
    #panel-kanan {background: #fbfcfd; border: 1px solid #eaeff5;
          border-radius: 14px; padding: 10px 12px 4px;}
    .dark #panel-kanan {background: #1e293b; border-color: #334155;}
    #chatbox {border: 1px solid #e2e8f0; border-radius: 14px;}
    .dark #chatbox {border-color: #334155;}
    #chatbox .message.user {background: #059669 !important; color: #fff !important;
          border-radius: 16px 16px 4px 16px !important;}
    #chatbox .message.user * {color: #fff !important;}
    #chatbox .message.bot {border-radius: 16px 16px 16px 4px !important;}
    #chatbox .message {max-width: 94% !important;}
    .risk-alert {background: #fef2f2; border: 1px solid #fca5a5;
          border-left: 5px solid #dc2626; color: #7f1d1d; border-radius: 10px;
          padding: 12px 16px; margin: 4px 0 8px; font-size: .95rem;}
    .dark .risk-alert {background: #450a0a; color: #fecaca; border-color: #7f1d1d;}
    /* Supplementary reference, deliberately quiet so the conversation holds the
       eye. Size is only half of that: the stronger lever is CONTRAST — small but
       near-black text still competes, whereas muted slate recedes at any size.
       Hence 0.70rem AND #64748b rather than shrinking further.

       0.70rem (~11px) is the floor. This is clinical text a counselor may need to
       READ to check a cited claim, not decoration, and #64748b on #f8fafc is
       ~4.9:1 — just past the WCAG AA 4.5:1 minimum. Going smaller or lighter
       would trade the panel's only purpose for tidiness.

       The [n] headings keep a touch of green and full weight so the panel can
       still be scanned for a specific citation without reading it. */
    #cites-box {font-size: .70rem; line-height: 1.5; max-height: 72vh;
          overflow-y: auto; padding-right: 6px; color: #64748b;}
    #cites-box strong {color: #0f766e; font-size: .71rem; font-weight: 600;}
    .dark #cites-box {color: #94a3b8;}
    .dark #cites-box strong {color: #5eead4;}
    #cites-box blockquote {margin: 2px 0 9px; padding: 0 0 0 8px;
          border-left: 2px solid #dbe3ec; color: #64748b;}
    .dark #cites-box blockquote {border-color: #3f4c5f; color: #8fa0b3;}
    #cites-box code {font-size: .63rem; color: #a3aebd;}
    #cites-box p {margin: 2px 0;}
    #cites-box::-webkit-scrollbar {width: 7px;}
    #cites-box::-webkit-scrollbar-thumb {background: #dbe3ec; border-radius: 4px;}
    /* the panel itself recedes too: paler fill, softer border, quieter label */
    #panel-kanan .label-wrap span {font-size: .78rem !important; color: #64748b !important;}
    #ftr {color: #64748b; font-size: .8rem; text-align: center; margin-top: 6px;}
    footer {display: none !important;}
    """
    badge = ('MODE DRY-RUN — generator belum dimuat' if engine.dry_run
             else 'Generator aktif: ' + cfg.checkpoint.split("/")[-1])

    # gradio 6 applies theme/css at launch(), not in the Blocks constructor
    style = {"theme": theme, "css": css}
    with gr.Blocks(title="Chatbot Pendamping Kasus Depresi") as demo:
        gr.HTML(
            '<div id="hdr"><h1>Chatbot Pendamping Kasus Depresi</h1>'
            '<p>Alat bantu konselor — berbasis <em>Pedoman Deteksi Dini dan '
            'Penatalaksanaan Gangguan Jiwa bagi Dokter Umum di FKTP</em> '
            # "dijawab independen dari pedoman" parsed as "answered independently
            # OF the guideline" — the opposite of the grounding claim the study
            # rests on. The intended meaning was that each question is processed
            # on its own; that is now stated plainly, together with the
            # consequence the counselor has to act on (no memory between turns).
            '(Kemenkes RI, 2017). Jawaban disusun dari pedoman tersebut. '
            'Chatbot ini <strong>tidak mengingat pesan sebelumnya</strong> — '
            'setiap pertanyaan diproses terpisah, jadi sertakan kembali konteks '
            'kasus pada tiap pertanyaan. Keluaran ditinjau konselor, bukan untuk '
            'pasien langsung.</p>'
            '<span class="badge">Alat Bantu Konselor</span>'
            f'<span class="badge">{badge}</span></div>')
        # Conversation left, sources right. The panel used to sit on the LEFT and
        # also held a study-question picker and a settings accordion; both were
        # removed 2026-07-27 for study integrity —
        #   * the picker pasted a study question verbatim, but the protocol needs
        #     the counselor to paraphrase, so the tool must see THEIR framing;
        #   * the k slider let a counselor change retrieval depth mid-study, which
        #     would make their cases non-comparable with each other and with the
        #     frozen briefings. k is fixed in configs/chatbot.yaml.
        # That left a quarter of the width holding one collapsed accordion, ahead
        # of the thing the user actually came for. The conversation is the primary
        # object so it now leads; the passages are reference material and read as
        # such on the right.
        with gr.Row(equal_height=False):
            with gr.Column(scale=9):
                # sanitize_html=False lets the risk alert render as a styled
                # block inside the bot bubble (content is our own + guideline text)
                chat = gr.Chatbot(elem_id="chatbox", height="74vh",
                                  show_label=False, avatar_images=None,
                                  sanitize_html=False)
                question = gr.Textbox(
                    show_label=False, lines=1, submit_btn=True, stop_btn=True,
                    placeholder="Tulis pertanyaan atau kasus pasien di sini… (Enter untuk kirim)")
                clear = gr.ClearButton([chat, question],
                                       value="Chat baru", size="sm")
            with gr.Column(scale=3, elem_id="panel-kanan"):
                # open by default: verifying a [n] is a normal step, not an
                # advanced one, and a collapsed panel makes it feel optional
                with gr.Accordion("Konteks pedoman yang digunakan", open=True):
                    cites = gr.Markdown(CITES_PLACEHOLDER, elem_id="cites-box")
        # added after the fact: `cites` is created later in the layout than the
        # button, so it cannot be in the constructor's list
        clear.add(cites)
        # The link is public and unauthenticated (see main()), so the boundary
        # that used to be a password is now this instruction plus the study
        # protocol's pre-anonymized question set. It has to be visible, not
        # buried in the briefing deck.
        gr.HTML('<div id="ftr"><strong>Gunakan hanya pertanyaan studi yang sudah '
                'dianonimkan. Jangan mengetikkan nama, kontak, atau data klien '
                'nyata</strong> — tautan ini bersifat publik dan teks yang '
                'dikirim melewati server pihak ketiga (gradio.live).<br>'
                'Semua interaksi dicatat untuk keperluan penelitian '
                '(outputs/chat_logs) · Konfigurasi: configs/chatbot.yaml</div>')

        question.submit(respond, [question, chat],
                        [chat, cites, question])
    return demo, style, engine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chatbot-app", description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "chatbot.yaml"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip loading the generator (retrieval + prompt only)")
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--log-dir", default=None,
                    help="override the interaction-log folder (e.g. per counselor)")
    ap.add_argument("--require-auth", action="store_true",
                    help=f"require login; credentials come from ${AUTH_ENV} "
                         '("user:pass", comma-separated for several)')
    args = ap.parse_args(argv)

    # Login is OPTIONAL as of 2026-08-08, by study decision: counselors work from
    # the frozen, pre-anonymized question set, and distributing per-person
    # credentials was costing more setup friction than the gate was buying. The
    # mechanism is kept, not deleted — set $CHATBOT_AUTH and pass --require-auth
    # to turn it back on for any run that handles non-anonymized text.
    auth = resolve_auth(args.require_auth)

    demo, style, engine = build_app(args.config, args.dry_run, args.log_dir)

    # A dry-run engine answers with the raw assembled prompt AND skips the LLM
    # risk stage for the whole session, leaving only the keyword screen. That
    # must never reach a counselor behind a public URL. --dry-run asks for it
    # explicitly and is allowed; falling into it because vLLM is down is not.
    if args.share and engine.dry_run and not args.dry_run:
        raise SystemExit(
            "--share refused: the generator could not be loaded, so the app is in "
            f"DRY-RUN mode and the LLM risk stage is disabled.\n  {engine.generator_error}\n"
            "Start vLLM first (scripts/serve_chatbot.sh), or pass --dry-run to "
            "publish a prompt-preview build deliberately.")

    if args.share and not auth:
        print("[app] WARNING: publishing a PUBLIC link with NO login. Anyone with "
              "the URL can use this tool and read its answers.\n"
              "[app]          Send the link only to the study counselors, and "
              "re-run serve_chatbot.sh to rotate it when the study ends.")

    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                auth=auth,
                auth_message=("Masukkan nama pengguna dan kata sandi yang "
                              "diberikan oleh peneliti."),
                **style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
