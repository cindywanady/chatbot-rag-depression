"""Deployed-chatbot building blocks (Phase 2).

Everything the Gradio app needs apart from the UI itself: typed config over
``configs/chatbot.yaml``, context formatting with citations, the
counselor-briefing message assembly, the keyword safety screen, the
interaction logger, and a lazy generator wrapper. The prompt/safety functions are pure so they are unit
tested without loading any model; torch/transformers are imported only inside
:class:`HFGenerator`.

Run the app itself from the Phase-2 environment (``.venv-chat``); this module
only needs PyYAML at import time.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ChatbotConfig:
    # retriever
    index_dir: str
    embedding_config: str
    model: str
    k: int
    # generator
    backend: str          # "openai" (vLLM / any OpenAI-compatible server) | "hf"
    checkpoint: str
    base_url: str         # openai backend only
    api_key_env: str      # env var holding the API key ("EMPTY" ok for local vLLM)
    dtype: str
    max_new_tokens: int
    temperature: float
    top_p: float
    # prompts
    base_prompt: str
    counselor_briefing: str   # the chatbot's single role: brief the counselor
    # safety
    safety_keywords: tuple[str, ...]
    llm_check: bool
    safe_response: str   # counselor-facing: keeps the MI.7/MI.8 pointers
    # run
    seed: int
    log_dir: str

    def system_prompt(self) -> str:
        return f"{self.base_prompt.rstrip()}\n\n{self.counselor_briefing.rstrip()}"


def load_chatbot_config(path: str | Path) -> ChatbotConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ret, gen, saf, pr = raw["retriever"], raw["generator"], raw["safety"], raw["prompts"]
    run = raw.get("run", {})
    return ChatbotConfig(
        index_dir=ret["index_dir"],
        embedding_config=ret["embedding_config"],
        model=ret.get("model", "e5-large"),
        k=int(ret.get("k", 5)),
        backend=gen.get("backend", "hf"),
        checkpoint=gen["checkpoint"],
        base_url=gen.get("base_url", "http://127.0.0.1:8000/v1"),
        api_key_env=gen.get("api_key_env", "VLLM_API_KEY"),
        dtype=gen.get("dtype", "bfloat16"),
        max_new_tokens=int(gen.get("max_new_tokens", 700)),
        temperature=float(gen.get("temperature", 0.2)),
        top_p=float(gen.get("top_p", 0.9)),
        base_prompt=pr["base"],
        counselor_briefing=pr["counselor_briefing"],
        safety_keywords=tuple(k.lower() for k in saf.get("keywords", [])),
        llm_check=bool(saf.get("llm_check", True)),
        safe_response=saf.get("safe_response", ""),
        seed=int(run.get("seed", 20260628)),
        log_dir=run.get("log_dir", "outputs/chat_logs"),
    )


# ---- retrieval-side helpers --------------------------------------------------

def load_chunk_records(index_dir: str | Path) -> dict[str, dict]:
    """chunk_id -> full chunk record (text + metadata) from an index directory."""
    records: dict[str, dict] = {}
    with open(Path(index_dir) / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                records[r["chunk_id"]] = r
    return records


def _citation(rec: dict) -> str:
    head = rec.get("meta_heading_path") or rec.get("meta_source_unit") or "Pedoman"
    p1, p2 = rec.get("meta_page_start"), rec.get("meta_page_end")
    pages = f"hal. {p1}" if p1 == p2 else f"hal. {p1}-{p2}"
    return f"{head} ({pages})"


def format_context(records: list[dict]) -> str:
    """Retrieved chunks as numbered, citable context blocks."""
    blocks = []
    for i, rec in enumerate(records, 1):
        blocks.append(f"[{i}] {_citation(rec)}\n{rec['text'].strip()}")
    return "\n\n".join(blocks)


RISK_NOTE = (
    "CATATAN SISTEM: Terdeteksi indikasi risiko keselamatan (bunuh diri / "
    "melukai diri) pada pertanyaan ini. Utamakan keselamatan: pada bagian "
    "TANDA BAHAYA, tegaskan risiko ini kepada konselor dan cantumkan langkah "
    "darurat serta alur rujukan segera sesuai pedoman yang perlu ditindaklanjuti "
    "konselor. Tetap dalam bentuk ringkasan untuk konselor (sudut pandang orang "
    "ketiga), BUKAN pesan langsung kepada penanya.")


def build_messages(cfg: ChatbotConfig, question: str, context: str,
                   risk_note: bool = False) -> list[dict]:
    """System + user messages for the generator's chat template."""
    user = (f"KONTEKS PEDOMAN:\n\n{context}\n\n---\n\n"
            f"PERTANYAAN:\n{question.strip()}")
    if risk_note:
        user += f"\n\n{RISK_NOTE}"
    return [{"role": "system", "content": cfg.system_prompt()},
            {"role": "user", "content": user}]


# ---- safety ------------------------------------------------------------------

def _fold(text: str) -> str:
    """Lowercase and collapse every run of non-alphanumerics to one space.

    A raw substring test misses the spellings people actually type. Measured on
    the deployed keyword list, "bunuh-diri", "bunuh  diri" and "BUNUH DIRI!!"
    all failed against the literal "bunuh diri" — folding makes them one form.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def risk_screen(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Normalised substring screen; returns the matched keywords.

    Two passes. The folded pass catches punctuation and spacing variants; the
    de-spaced pass additionally catches run-together typos ("bunuhdiri"), which
    the folded pass cannot, since folding leaves the keyword's own space in.

    The de-spaced pass can over-match across a word boundary. That direction is
    deliberate: this is a pre-screen whose only action is to raise a safety
    banner and add a risk instruction to the prompt, so a false positive costs a
    redundant warning and a false negative costs the warning entirely.
    """
    low = _fold(text)
    tight = low.replace(" ", "")
    out = []
    for k in keywords:
        folded = _fold(k)
        if folded in low or folded.replace(" ", "") in tight:
            out.append(k)
    return out


# DO NOT "harden" this wording without re-measuring on the 50 study questions.
#
# The pasted text is untrusted and is interpolated last, with no delimiter and no
# instruction after it, so instruction-shaped content sits in the most persuasive
# position. A fenced variant was written to fix that on 2026-08-08 (delimiters
# around the text, task restated afterwards) and REVERTED the same day: measured
# against the OLD wording over the 38 study questions that reach this stage, it
# flagged 10 where the old flagged 12, turning Q14 and Q42 — both frozen as
# risky — into AMAN. A prompt-injection defence that costs two real risk flags is
# a net safety loss, and the threat it addresses is a counselor typing study
# questions, not an adversary.
#
# If injection resistance is wanted later, add it as a SECOND check that can only
# escalate the verdict, never as an edit to this one.
LLM_RISK_PROMPT = (
    "Anda adalah pemeriksa keselamatan. Baca teks berikut dan tentukan apakah "
    "penulisnya menunjukkan risiko bunuh diri, keinginan mati, atau melukai "
    "diri. Jawab dengan SATU kata saja: RISIKO atau AMAN.\n\nTeks:\n{question}"
)

# One word plus stray punctuation or a stop token. The old cap of 8 could clip a
# hedged reply mid-word, which the parse below then has to read as risky.
RISK_VERDICT_MAX_TOKENS = 16


def parse_risk_verdict(raw: str) -> bool:
    """Verdict text -> risk detected. Ambiguity resolves to RISK.

    Two biases, both deliberately toward flagging:

    * "RISIKO" is a substring of "BERISIKO", so a hedged Indonesian reply
      ("tidak berisiko") reads as a positive.
    * Anything that does not clearly say AMAN is treated as risky, so an empty
      completion, a refusal, or a preamble the cap clipped fails closed rather
      than silently passing an at-risk disclosure through unflagged.

    Callers keep the raw text so both directions stay auditable after the fact
    (scripts/audit_risk_verdicts.py counts them).
    """
    up = raw.upper()
    if "RISIKO" in up:
        return True
    # AMAN must be the WHOLE verdict, not merely present. A bare `"AMAN" not in
    # up` test reads "TIDAK AMAN" (= not safe) as safe, inverting the verdict on
    # the single reply shape a non-compliant Indonesian answer is most likely to
    # take — and RISK_VERDICT_MAX_TOKENS leaves room for exactly that. Anything
    # that is not a clean AMAN (a negation, a preamble, "KEAMANAN", an empty
    # completion) therefore fails closed.
    return re.fullmatch(r"[^A-Z]*AMAN[^A-Z]*", up) is None


# ---- logging -----------------------------------------------------------------

class ChatLogger:
    """One JSONL file per app session; one line per interaction."""

    def __init__(self, log_dir: str | Path) -> None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(log_dir) / f"chat-{stamp}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A per-instance dir created here inherits outputs/'s default group ACL,
        # and directory write permission governs unlink regardless of the 0600 on
        # the files inside — so without this any group member could delete the
        # study's only record. Hardening the files alone was not enough.
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass  # not ours to chmod (shared parent); the file mode below still applies
        # These lines hold pasted patient narratives. The project sits on a
        # shared filesystem whose default ACL grants the group rwx, so a plain
        # open() leaves them readable by every member of the unix group; an
        # explicit chmod on the file resets its ACL mask. The parent directory
        # is not touched here — that is a deployment decision, not this class's.
        self.path.touch(exist_ok=True)
        os.chmod(self.path, 0o600)
        # The FastAPI UI runs generations in threads that share one logger, and
        # a record is 17-29 KB — far over the buffer, so unsynchronised writes
        # interleave and corrupt the JSONL.
        self._lock = threading.Lock()

    def log(self, **record) -> None:
        record.setdefault("ts", _dt.datetime.now().isoformat(timespec="seconds"))
        record.setdefault("interaction_id", uuid.uuid4().hex[:12])
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)


# ---- generator ----------------------------------------------------------------

class HFGenerator:
    """Chat-model wrapper (transformers). Heavy imports happen here only.

    Raises RuntimeError with actionable guidance when the model cannot be
    loaded (not downloaded / gated licence not accepted / wrong environment),
    so the app can fall back to dry-run mode.
    """

    def __init__(self, cfg: ChatbotConfig, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as e:
            raise RuntimeError(
                f"transformers/torch not available in this environment ({e}); "
                "run the app with .venv-chat/bin/python") from e
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, cfg.dtype, torch.bfloat16)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.checkpoint, dtype=dtype, device_map=self.device)
        except Exception as e:
            raise RuntimeError(
                f"could not load {cfg.checkpoint!r}: {e}\n"
                "If this is a gated model: accept its licence on huggingface.co, "
                "then `huggingface-cli login` (or export HF_TOKEN) and retry. "
                "Weights download into the directory HF_HOME points at.") from e
        self.model.eval()

    def _chat(self, messages: list[dict], max_new_tokens: int,
              temperature: float | None = None) -> str:
        import torch

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
        except Exception:
            # some chat templates reject a system role: fold it into the user turn
            merged = [{"role": "user", "content": "\n\n".join(m["content"] for m in messages)}]
            inputs = self.tokenizer.apply_chat_template(
                merged, add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        temp = self.cfg.temperature if temperature is None else temperature
        do_sample = temp > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp if do_sample else None,
                top_p=self.cfg.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, messages: list[dict]) -> str:
        return self._chat(messages, self.cfg.max_new_tokens)

    def llm_risk_check(self, question: str) -> tuple[bool, str]:
        """Second-stage safety classifier -> (risk detected, raw verdict text).

        See :func:`parse_risk_verdict` for the parse and its bias. Decoding is
        greedy regardless of ``cfg.temperature``: this is a binary safety
        verdict, and sampling it means the same input can flag on one send and
        not on a resend.
        """
        msg = [{"role": "user", "content": LLM_RISK_PROMPT.format(question=question)}]
        raw = self._chat(msg, max_new_tokens=RISK_VERDICT_MAX_TOKENS, temperature=0.0)
        return parse_risk_verdict(raw), raw


class OpenAIGenerator:
    """OpenAI-compatible chat client — a local vLLM server (`vllm serve`) or
    any hosted provider exposing /v1/chat/completions. The server applies the
    model's own chat template."""

    def __init__(self, cfg: ChatbotConfig) -> None:
        import os

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai client not installed in this environment "
                "(.venv-chat/bin/pip install openai)") from e
        self.cfg = cfg
        # Without an explicit timeout the client waits out openai-python's
        # 600 s default: a wedged vLLM would hold a counselor on a spinner for
        # ten minutes with no feedback. The longest healthy briefing is ~60 s
        # (1200 tokens at ~20 tok/s), so 180 s is generous and still bounded.
        self.client = OpenAI(base_url=cfg.base_url,
                             api_key=os.environ.get(cfg.api_key_env) or "EMPTY",
                             timeout=180.0, max_retries=1)
        try:
            served = [m.id for m in self.client.models.list().data]
        except Exception as e:
            raise RuntimeError(
                f"no OpenAI-compatible server reachable at {cfg.base_url}: {e}\n"
                f"Start one with: .venv-vllm/bin/vllm serve {cfg.checkpoint} "
                "--dtype bfloat16 --gpu-memory-utilization 0.85") from e
        if cfg.checkpoint not in served:
            raise RuntimeError(
                f"server at {cfg.base_url} serves {served}, not {cfg.checkpoint!r}")

    def _chat(self, messages: list[dict], max_tokens: int,
              temperature: float | None = None) -> str:
        import openai

        temp = self.cfg.temperature if temperature is None else temperature
        try:
            r = self.client.chat.completions.create(
                model=self.cfg.checkpoint, messages=messages,
                max_tokens=max_tokens, temperature=temp,
                top_p=self.cfg.top_p)
        except openai.BadRequestError:
            # ONLY a template rejecting the system role. A bare `except Exception`
            # also swallowed connection resets, 503s from a restarting vLLM and
            # the 180 s timeout — retrying those with the system prompt demoted to
            # free text silently drops the grounding, citation and counselor-scope
            # rules, while _log still records the two-message form. It also
            # doubled every timeout the counselor waits through.
            # some chat templates reject a system role: fold it into one user turn
            merged = [{"role": "user",
                       "content": "\n\n".join(m["content"] for m in messages)}]
            r = self.client.chat.completions.create(
                model=self.cfg.checkpoint, messages=merged,
                max_tokens=max_tokens, temperature=temp,
                top_p=self.cfg.top_p)
        # "length" = hit the max_new_tokens cap (truncated answer) - logged
        self.last_finish_reason = r.choices[0].finish_reason
        return (r.choices[0].message.content or "").strip()

    def generate(self, messages: list[dict]) -> str:
        return self._chat(messages, self.cfg.max_new_tokens)

    def generate_stream(self, messages: list[dict]):
        """Yield answer text incrementally (keeps proxies from timing out and
        lets the UI render tokens as they arrive)."""
        import openai

        kwargs = dict(model=self.cfg.checkpoint, max_tokens=self.cfg.max_new_tokens,
                      temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                      stream=True)
        try:
            stream = self.client.chat.completions.create(messages=messages, **kwargs)
        except openai.BadRequestError:  # see _chat: template-only, never transport
            merged = [{"role": "user",
                       "content": "\n\n".join(m["content"] for m in messages)}]
            stream = self.client.chat.completions.create(messages=merged, **kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                self.last_finish_reason = chunk.choices[0].finish_reason
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def llm_risk_check(self, question: str) -> tuple[bool, str]:
        """-> (risk detected, raw verdict text). See HFGenerator.llm_risk_check."""
        msg = [{"role": "user", "content": LLM_RISK_PROMPT.format(question=question)}]
        raw = self._chat(msg, RISK_VERDICT_MAX_TOKENS, temperature=0.0)
        return parse_risk_verdict(raw), raw


def build_generator(cfg: ChatbotConfig):
    """The configured generator backend ("openai" server client or in-process "hf")."""
    if cfg.backend == "openai":
        return OpenAIGenerator(cfg)
    if cfg.backend == "hf":
        return HFGenerator(cfg)
    raise RuntimeError(f"unknown generator backend {cfg.backend!r} (openai|hf)")


def timed(fn, *args, **kwargs):
    """(result, seconds) helper for the interaction log."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - t0, 2)


DRY_RUN_NOTE = ("(MODE DRY-RUN: generator belum tersedia — di bawah ini adalah "
                "prompt yang akan dikirim ke model, bukan jawaban.)")

# finish_reason == "length" means max_new_tokens cut the briefing off. It was
# recorded in the log but shown to the counselor unmarked, so a briefing missing
# its OPSI RUJUKAN section looked complete — the section most likely to be lost
# is the last one.
TRUNCATION_NOTE = ("\n\n---\n\n*[JAWABAN TERPOTONG — batas panjang tercapai. "
                   "Bagian akhir (mis. opsi rujukan) mungkin belum lengkap; "
                   "tanyakan ulang secara lebih spesifik bila perlu.]*")


def load_sample_questions(root: str | Path) -> dict[str, str]:
    """'Q01 — title' -> question text, from the 50-question study sample."""
    path = Path(root) / "data" / "derived" / "questions_alodokter_sample50.jsonl"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[f"{r['study_id']} — {r['question_title'][:70]}"] = r["question"]
    return out


from depression_rag.textdisplay import reflow_for_display  # noqa: E402


class ChatEngine:
    """One place for the full answer path, shared by every UI.

    retrieve -> safety screen -> prompt assembly -> generate (or dry-run
    preview) -> assemble the final answer per the study protocol (a risky
    input leads with the fixed safety banner, then the grounded answer
    generated under an explicit risk instruction) -> log.
    """

    def __init__(self, cfg: ChatbotConfig, root: str | Path = ".",
                 dry_run: bool = False) -> None:
        from depression_rag.evaluation.retrieval import (
            RetrievalRunner,
            build_embedder_for_model,
        )

        root = Path(root)
        self.cfg = cfg
        # Check the pairing before loading anything heavy. e5-large and bge-m3
        # are both 1024-d, so an index/embedder mismatch raises nothing: FAISS
        # accepts the query vector and returns confident, wrongly-encoded
        # neighbours, which reads as a retrieval-quality problem rather than a
        # misconfiguration.
        meta = json.loads(
            (root / cfg.index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model") != cfg.model:
            raise RuntimeError(
                f"index {cfg.index_dir} was built with model {meta.get('model')!r}, "
                f"but this config asks for {cfg.model!r} — refusing to search a "
                "mismatched index")
        embedder = build_embedder_for_model(cfg.model, root / cfg.embedding_config)
        self.runner = RetrievalRunner(root / cfg.index_dir, embedder)
        self.records = load_chunk_records(root / cfg.index_dir)
        self.generator = None
        self.generator_error = ""
        if not dry_run:
            try:
                self.generator = build_generator(cfg)
            except RuntimeError as e:  # server down / not downloaded / wrong venv
                self.generator_error = str(e)
                print(f"[chat] {e}\n[chat] continuing in DRY-RUN mode")
        self.logger = ChatLogger(root / cfg.log_dir)

    @property
    def dry_run(self) -> bool:
        return self.generator is None

    def _prepare(self, question: str, k: int | None,
                 risk_override: bool | None = None) -> dict:
        """Shared front half: safety assessment, retrieval, prompt, citations.

        `risk_override` pins the risk verdict instead of re-screening. The LLM
        stage decodes greedily (temperature 0.0, see llm_risk_check), so it is
        stable for a fixed input — but the verdict still moves if the model,
        the prompt or the keyword list changes, and the risk flag feeds
        freeze_eval_assignments.py. Pinning keeps a frozen study sample frozen
        across those, so a regeneration changes only what was intended.
        """
        cfg = self.cfg
        question = question.strip()
        k = int(k or cfg.k)
        matched = risk_screen(question, cfg.safety_keywords)
        llm_flag = False
        llm_verdict = None   # None = the LLM stage was not run for this input
        if risk_override is None:
            if not matched and self.generator is not None and cfg.llm_check:
                # Fail CLOSED, and here rather than at the call site: this runs
                # BEFORE answer()'s try/finally, so an exception escaping it also
                # means the interaction never reaches the log — the one case the
                # finally block exists to preserve. A restarting vLLM, an OOM or
                # the 180 s timeout must raise the banner, not silently clear it.
                try:
                    llm_flag, llm_verdict = self.generator.llm_risk_check(question)
                except Exception as e:
                    llm_flag = True
                    llm_verdict = f"ERROR: {type(e).__name__}: {e}"
            risky = bool(matched) or llm_flag
        else:
            risky = bool(risk_override)
        results, t_ret = timed(self.runner.search,
                               [{"question_id": "ui", "question": question}], k)
        recs = [self.records[cid] for cid in results[0].ranked_chunk_ids[:k]]
        return {
            "question": question, "k": k,
            "matched": matched, "llm_flag": llm_flag, "llm_verdict": llm_verdict,
            "risky": risky,
            "risk_reason": (f"kata kunci: {', '.join(matched)}" if matched
                            else ("klasifikasi LLM" if llm_flag else
                                  ("disematkan (pinned)" if risk_override else ""))),
            "results": results, "t_ret": t_ret, "recs": recs,
            "messages": build_messages(cfg, question, format_context(recs),
                                       risk_note=risky),
            "citations": [{
                "n": i,
                "heading": rec.get("meta_heading_path", "?"),
                "pages": f"{rec.get('meta_page_start')}–{rec.get('meta_page_end')}",
                "chunk_id": rec["chunk_id"],
                # Untruncated. It was capped at 400 chars, which cut off 57% of
                # passages — and half a passage is worse than useless for the one
                # job this panel has: letting a counselor check whether a cited
                # claim is actually in the source. Raising the cap does not help
                # (800 -> 2000 chars only moves 74% -> 82% complete), and the UI
                # panel now scrolls independently, so there is nothing to protect.
                "snippet": reflow_for_display(rec["text"]),
            } for i, rec in enumerate(recs, 1)],
        }

    def _dry_run_preview(self, messages: list[dict]) -> str:
        return (f"{DRY_RUN_NOTE}\n\n[system]\n{messages[0]['content']}"
                f"\n\n[user]\n{messages[1]['content'][:2500]} …")

    def _log(self, p: dict, answer: str, t_gen: float, error: str = "") -> None:
        self.logger.log(question=p["question"], k=p["k"], error=error,
                        retrieved=[{"chunk_id": c, "score": float(s)} for c, s in
                                   zip(p["results"][0].ranked_chunk_ids,
                                       p["results"][0].scores)],
                        risk_keywords=p["matched"], risk_llm=p["llm_flag"],
                        risk_llm_verdict=p["llm_verdict"],
                        system_prompt=p["messages"][0]["content"],
                        user_prompt=p["messages"][1]["content"], answer=answer,
                        dry_run=self.dry_run, t_retrieval_s=p["t_ret"],
                        t_generation_s=t_gen,
                        finish_reason=getattr(self.generator, "last_finish_reason", None),
                        generator=None if self.dry_run else self.cfg.checkpoint)

    def answer(self, question: str, k: int | None = None) -> dict:
        """Full pipeline for one question; returns a structured result dict."""
        cfg = self.cfg
        p = self._prepare(question, k)
        answer, t_gen, error = "", 0.0, ""
        try:
            if self.generator is not None:
                answer, t_gen = timed(self.generator.generate, p["messages"])
                if getattr(self.generator, "last_finish_reason", None) == "length":
                    answer += TRUNCATION_NOTE
            else:
                answer = self._dry_run_preview(p["messages"])
            if p["risky"]:  # escalation-first: template leads, guidance follows
                answer = f"{cfg.safe_response}\n\n---\n\n{answer}"
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            # in the finally block so a generator failure still leaves a study
            # record — otherwise the interactions most worth reviewing (the ones
            # that broke, including risk-flagged ones) are the ones that vanish
            self._log(p, answer, t_gen, error=error)
        return {"answer": answer, "risky": p["risky"],
                "risk_reason": p["risk_reason"], "dry_run": self.dry_run,
                "citations": p["citations"]}

    def answer_stream(self, question: str, k: int | None = None):
        """Streaming variant: yields ("meta", dict) once, then ("delta", str)
        pieces, then ("done", {}). Same protocol behaviour and logging as
        :meth:`answer`; streaming keeps bytes flowing so reverse proxies do not
        time out during long generations."""
        cfg = self.cfg
        p = self._prepare(question, k)
        yield "meta", {"risky": p["risky"], "risk_reason": p["risk_reason"],
                       "dry_run": self.dry_run, "citations": p["citations"]}
        parts: list[str] = []

        def emit(text: str):
            parts.append(text)
            return "delta", text

        t0 = time.perf_counter()
        error = ""
        try:
            if p["risky"]:  # escalation-first: template leads, guidance follows
                yield emit(f"{cfg.safe_response}\n\n---\n\n")
            if self.generator is None:
                yield emit(self._dry_run_preview(p["messages"]))
            elif hasattr(self.generator, "generate_stream"):
                for piece in self.generator.generate_stream(p["messages"]):
                    yield emit(piece)
                if getattr(self.generator, "last_finish_reason", None) == "length":
                    yield emit(TRUNCATION_NOTE)
            else:  # backend without streaming support: one final chunk
                yield emit(self.generator.generate(p["messages"]))
                if getattr(self.generator, "last_finish_reason", None) == "length":
                    yield emit(TRUNCATION_NOTE)
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            # BaseException and finally, not except Exception: the stop button
            # closes this generator, which arrives as GeneratorExit. Logging in
            # finally is what keeps a half-generated or abandoned answer — and
            # its risk verdict — in the study record. Nothing is yielded here;
            # yielding during GeneratorExit is an error.
            self._log(p, "".join(parts), round(time.perf_counter() - t0, 2),
                      error=error)
        yield "done", {}
