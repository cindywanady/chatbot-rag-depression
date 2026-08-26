#!/usr/bin/env python
"""Generate chatbot answers for the 50-question study set (for automatic eval).

For each question in questions_alodokter_sample50.jsonl this retrieves the top-k
guideline chunks and generates a counselor briefing with the deployed
generator (Gemma via vLLM), then records everything the automatic evaluation
needs:

  * the generated answer (the RAG output being evaluated)
  * the retrieved context it was grounded in (for the faithfulness metric)
  * the risk flag (whether the safety screen fired)
  * the doctor's forum reply (the reference for the agreement/similarity metrics)

The fixed safety template is deliberately NOT prepended here: we evaluate the
*generated* answer's quality separately from deterministic boilerplate, and
record `risky` so risk cases can be analysed on their own.

Writes outputs/analysis/study_answers_chatbot.jsonl incrementally (one line per
question, so a crash mid-run loses nothing). Requires the vLLM generator up
(scripts/serve_chatbot.sh or the vllm serve command); run in .venv-chat.

    .venv-chat/bin/python scripts/generate_study_answers.py
    .venv-chat/bin/python scripts/generate_study_answers.py --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from depression_rag.chatbot import ChatEngine, load_chatbot_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "chatbot.yaml"))
    ap.add_argument("--questions", default=str(ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl"))
    ap.add_argument("--k", type=int, default=None, help="top-k chunks (default: config)")
    ap.add_argument("--pin-risk-from", default=None, metavar="JSONL",
                    help="reuse the risk verdicts from a previous answers file "
                         "instead of re-screening. The risk flag feeds "
                         "freeze_eval_assignments.py, so a moved verdict silently "
                         "re-randomises the frozen sample. llm_risk_check decodes "
                         "greedily (temperature 0.0) and so is stable for a fixed "
                         "input, but the verdict still moves if the model, the prompt "
                         "or the keyword list changes. Pin it when regenerating for a "
                         "reason unrelated to risk detection, e.g. a prompt fix.")
    args = ap.parse_args(argv)

    pinned: dict[str, bool] = {}
    if args.pin_risk_from:
        for line in open(args.pin_risk_from, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                pinned[r["question_id"]] = bool(r["risky"])
        print(f"[gen] pinning risk verdicts for {len(pinned)} questions "
              f"from {args.pin_risk_from}")

    cfg = load_chatbot_config(args.config)
    engine = ChatEngine(cfg, root=ROOT, dry_run=False)
    if engine.dry_run:
        print("ERROR: generator not available (is vLLM up on :8000?). Aborting.")
        return 1
    k = args.k or cfg.k

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume support: skip question_ids already written
    done: set[str] = set()
    if out.exists():
        done = {json.loads(l)["question_id"] for l in open(out, encoding="utf-8") if l.strip()}
        if done:
            print(f"[resume] {len(done)} already done; skipping those")

    print(f"[gen] {len(questions)} questions, k={k}, "
          f"generator={cfg.checkpoint}")
    t_start = time.perf_counter()
    with open(out, "a", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            if q["question_id"] in done:
                continue
            p = engine._prepare(q["question"], k,
                                risk_override=pinned.get(q["question_id"]))
            t0 = time.perf_counter()
            answer = engine.generator.generate(p["messages"])
            dt = round(time.perf_counter() - t0, 1)
            rec = {
                "study_id": q.get("study_id"),
                "question_id": q["question_id"],
                "block": q.get("block"),
                "question": q["question"],
                "generated_answer": answer,
                "risky": p["risky"],
                "risk_reason": p["risk_reason"],
                # raw classifier output, null when the LLM stage was not run
                # (keyword hit, or llm_check off): lets the substring parse in
                # llm_risk_check be audited afterwards
                "risk_llm_verdict": p["llm_verdict"],
                "retrieved_context": [
                    {"chunk_id": c["chunk_id"], "heading": c["heading"],
                     "pages": c["pages"], "text": r["text"]}
                    for c, r in zip(p["citations"], p["recs"])
                ],
                "doctor_answer": q.get("doctor_answer", ""),
                "gen_seconds": dt,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[gen] {i:>2}/{len(questions)} {q['question_id']} "
                  f"({dt}s, {len(answer)} chars{', RISK' if p['risky'] else ''})")

    print(f"\n[gen] done in {round((time.perf_counter() - t_start) / 60, 1)} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
