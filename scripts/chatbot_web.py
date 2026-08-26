#!/usr/bin/env python
"""Web app for the depression case-management RAG chatbot (Phase 2).

Serves the chat frontend in ``static/index.html`` (the project's own design:
green header, chat bubbles, typing indicator) over a small FastAPI backend.
The answer path — retrieval, safety screen, prompt assembly, generation,
logging — lives in :class:`depression_rag.chatbot.ChatEngine` and is shared
with the alternative Gradio UI (``scripts/chatbot_app.py``).

Run from the Phase-2 environment:

    .venv-chat/bin/python scripts/chatbot_web.py                # port 7860
    .venv-chat/bin/python scripts/chatbot_web.py --dry-run --port 7861

Endpoints: GET / (frontend) · GET /status ·
POST /chat {message, k?}.

The generator is ``google/gemma-4-12B-it``, served separately by vLLM (see
``scripts/serve_chatbot.sh``); the checkpoint is open-weights, not gated. If the
server is unreachable the app runs in dry-run mode — retrieval, safety screen and
prompt assembly all work and the assembled prompt is shown instead of an answer,
but the LLM risk stage is skipped, leaving only the keyword screen.

RESEARCHER-ONLY. Unlike the Gradio frontend this app has no authentication and
no public-link mode: bind it to 127.0.0.1 and reach it over an SSH tunnel. It
also accepts a client-supplied ``k``, which the Gradio UI deliberately does not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402  (ships with gradio)
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from depression_rag.chatbot import (  # noqa: E402
    ChatEngine,
    load_chatbot_config,
)

INDEX_HTML = ROOT / "static" / "index.html"


class ChatRequest(BaseModel):
    message: str
    k: int | None = None


def build_api(config_path: str, dry_run: bool) -> FastAPI:
    cfg = load_chatbot_config(config_path)
    print(f"[web] loading engine: {cfg.index_dir} (model {cfg.model}, k={cfg.k})")
    engine = ChatEngine(cfg, root=ROOT, dry_run=dry_run)
    print(f"[web] ready (dry_run={engine.dry_run})")

    api = FastAPI(title="Chatbot Pendamping Kasus Depresi")

    @api.get("/")
    def index() -> FileResponse:
        # no-cache so UI updates reach browsers without a hard refresh
        return FileResponse(INDEX_HTML,
                            headers={"Cache-Control": "no-cache, must-revalidate"})

    @api.get("/status")
    def status() -> dict:
        return {"dry_run": engine.dry_run,
                "generator": cfg.checkpoint.split("/")[-1],
                "index": cfg.index_dir.rsplit("/", 1)[-1], "k": cfg.k}

    # k arrives from the client and reaches FAISS and the prompt. Unbounded, one
    # request could pull every chunk in the index into the context window; None
    # falls back to cfg.k, which is what the study fixes.
    K_MAX = 20

    def _validate(req: ChatRequest) -> None:
        if not req.message.strip():
            raise HTTPException(400, "pertanyaan kosong")
        if req.k is not None and not (1 <= req.k <= K_MAX):
            raise HTTPException(400, f"k harus antara 1 dan {K_MAX}")

    @api.post("/chat")
    def chat(req: ChatRequest) -> dict:
        _validate(req)
        result = engine.answer(req.message, req.k)
        return {"reply": result["answer"], "risky": result["risky"],
                "risk_reason": result["risk_reason"],
                "dry_run": result["dry_run"], "citations": result["citations"]}

    # ---- submit + poll: proxy-proof transport (the frontend's default) ------
    # Cluster proxies (OnDemand/code-server) buffer responses and enforce a
    # total gateway deadline, killing both long requests AND streams. With
    # submit/poll every individual request returns in milliseconds; the browser
    # re-polls until the background generation finishes.
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()
    JOB_TTL_S = 900

    def _run_job(job_id: str, req: ChatRequest) -> None:
        try:
            for kind, payload in engine.answer_stream(req.message, req.k):
                with jobs_lock:
                    job = jobs.get(job_id)
                    if job is None:  # pruned
                        return
                    if kind == "meta":
                        job["meta"] = payload
                    elif kind == "delta":
                        job["text"] += payload
                    elif kind == "done":
                        job["status"] = "done"
        except Exception as e:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update(status="error", error=str(e))

    @api.post("/chat/submit")
    def chat_submit(req: ChatRequest) -> dict:
        _validate(req)
        now = time.time()
        job_id = uuid.uuid4().hex[:16]
        with jobs_lock:
            for jid in [j for j, v in jobs.items() if now - v["t0"] > JOB_TTL_S]:
                del jobs[jid]
            jobs[job_id] = {"status": "running", "text": "", "meta": None,
                            "error": "", "t0": now}
        threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
        return {"job_id": job_id}

    @api.get("/chat/poll/{job_id}")
    def chat_poll(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "job tidak ditemukan (kedaluwarsa?)")
            return {"status": job["status"], "text": job["text"],
                    "meta": job["meta"], "error": job["error"]}

    @api.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Server-sent events: a `meta` event (risk + citations), then `delta`
        text pieces as the model generates, then `done`. Streaming keeps bytes
        flowing so reverse proxies (OnDemand/code-server) do not 504 during
        ~30s generations."""
        _validate(req)

        def sse():
            for kind, payload in engine.answer_stream(req.message, req.k):
                if kind == "delta":
                    payload = {"text": payload}
                yield ("data: "
                       + json.dumps({"type": kind, **payload}, ensure_ascii=False)
                       + "\n\n")

        return StreamingResponse(
            sse(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"})  # disable proxy buffering

    return api


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chatbot-web", description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "chatbot.yaml"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip loading the generator (retrieval + prompt only)")
    args = ap.parse_args(argv)

    import uvicorn  # ships with gradio/fastapi

    uvicorn.run(build_api(args.config, args.dry_run),
                host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
