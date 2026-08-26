#!/usr/bin/env python
"""Run the actual RAGAS library on the study answers (reference-free + correctness).

Complements the custom Qwen judge with the standard, citable RAGAS metrics
(the Phase-2 planning spec names RAGAS specifically). Uses:

  * LLM        = the Qwen judge already served on vLLM (:8000) — same
                 different-from-Gemma family as the custom judge
  * embeddings = multilingual-e5 on CPU (so Indonesian text embeds properly and
                 the judge server keeps the GPU)

Metrics (RAGAS names):
  reference-free : faithfulness, answer_relevancy, llm_context_precision_without_reference
  reference-based: answer_correctness  (uses the doctor reply as `reference` — a
                   GP forum answer used as reference, NOT a gold standard)

    # verify end to end on 2 items first
    .venv-ragas/bin/python scripts/ragas_eval.py --answers outputs/analysis/study_answers_chatbot.jsonl --tag chatbot --skip-correctness --limit 2
    # full run
    .venv-ragas/bin/python scripts/ragas_eval.py --answers outputs/analysis/study_answers_chatbot.jsonl --tag chatbot --skip-correctness

Writes outputs/analysis/ragas_<tag>.csv + ragas_<tag>_summary.md. Run with the
.venv-ragas environment; the Qwen server must be up on :8000.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--tag", required=True, help="label for output files, e.g. chatbot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N (for a quick check)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-call budget for RAGAS (s). The default 240 was too "
                         "tight: faithfulness makes two sequential calls over a long "
                         "briefing, and under concurrency each call slows enough to "
                         "blow the budget -> 5 retries all time out -> nan. The "
                         "failures were stochastic (a different set of cases each "
                         "run, intersection empty), which is the signature of a "
                         "timeout, not of unparseable content.")
    ap.add_argument("--workers", type=int, default=4,
                    help="RAGAS concurrency; low values avoid overloading a single "
                         "local vLLM server (the cause of parse-failure blanks)")
    ap.add_argument("--no-correctness", action="store_true",
                    help="skip answer_correctness (the counselor briefing is not "
                         "a like-for-like match for the doctor reply)")
    args = ap.parse_args(argv)

    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (faithfulness, answer_relevancy, answer_correctness,
                               LLMContextPrecisionWithoutReference)

    model = args.model
    if model is None:
        from openai import OpenAI
        served = [m.id for m in OpenAI(base_url=args.base_url, api_key="EMPTY").models.list().data]
        if len(served) != 1:
            raise SystemExit(f"specify --model; served: {served}")
        model = served[0]
    print(f"[ragas] LLM = {model} @ {args.base_url}; embeddings = multilingual-e5 (CPU)")

    # extra_body disables Qwen3's reasoning mode for every RAGAS call: with
    # thinking on, RAGAS's structured-output prompts time out and fail to parse
    # (faithfulness/answer_correctness -> nan). max_tokens keeps replies bounded.
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=model, base_url=args.base_url, api_key="EMPTY", temperature=0.0,
        timeout=args.timeout, max_tokens=4096, max_retries=4,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}))

    records = [json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"[ragas] {len(records)} answers from {Path(args.answers).name}")

    # The doctor reply is only ever read by answer_correctness. When correctness is
    # skipped (the normal case — the briefing is not a like-for-like match for a
    # patient-facing reply) do not put it in the sample at all, so no doctor text is
    # sent to the judge server. Matches judge_study_answers.py, which dropped its own
    # doctor-comparison metric on 2026-07-27 for the same reason.
    want_correctness = not args.no_correctness
    samples = [SingleTurnSample(
        user_input=r["question"],
        response=r["generated_answer"],
        retrieved_contexts=[c["text"] for c in r["retrieved_context"]] or [""],
        reference=(r.get("doctor_answer") or "") if want_correctness else "",
    ) for r in records]
    if want_correctness and not any(s.reference for s in samples):
        raise SystemExit("answer_correctness requested but no record has a "
                         "doctor_answer to use as `reference`; pass --no-correctness")
    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference()]
    if not args.no_correctness:
        metrics.append(answer_correctness)
    print(f"[ragas] metrics: {[getattr(m, 'name', type(m).__name__) for m in metrics]}")

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=args.workers, timeout=args.timeout, max_retries=5)
    print(f"[ragas] concurrency: max_workers={args.workers}, timeout={args.timeout}s")
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb,
                      run_config=run_config)
    df = result.to_pandas()

    # keep the id columns from the source, align by row order
    df.insert(0, "question_id", [r["question_id"] for r in records])
    df.insert(0, "study_id", [r.get("study_id") for r in records])
    metric_cols = [c for c in df.columns if c not in
                   ("study_id", "question_id", "user_input", "response",
                    "retrieved_contexts", "reference")]
    out = ROOT / "outputs" / "analysis" / f"ragas_{args.tag}.csv"
    df[["study_id", "question_id"] + metric_cols].to_csv(out, index=False)

    means = {c: round(float(df[c].mean(skipna=True)), 3) for c in metric_cols
             if df[c].dtype.kind in "fi"}
    # read both from the environment rather than hardcoding: a regenerated summary
    # that names the wrong RAGAS version or the wrong generator is worse than none
    try:
        from importlib.metadata import version as _v
        ragas_version = _v("ragas")
    except Exception:                                    # noqa: BLE001
        ragas_version = "unknown"
    try:
        import yaml
        generator = yaml.safe_load(
            (ROOT / "configs" / "chatbot.yaml").read_text(encoding="utf-8")
        )["generator"]["checkpoint"]
    except Exception:                                    # noqa: BLE001
        generator = "unknown"

    lines = [f"# RAGAS evaluation — {args.tag}\n",
             f"_RAGAS {ragas_version} · LLM {model} · embeddings multilingual-e5 · "
             f"{len(records)} answers · generator {generator}_\n",
             "## Mean scores (0-1, higher better)"]
    for c, v in means.items():
        lines.append(f"- **{c}**: {v}")
    # Describe only the metrics actually present. The unconditional version of this
    # note explained answer_correctness even on --no-correctness runs, i.e. every
    # real run — a summary that documents a metric it does not contain.
    note = ("\n> Reference-free metrics (faithfulness, answer_relevancy, context "
            "precision) measure the answer's own quality — no reference answer is "
            "involved.")
    if want_correctness:
        note += (" answer_correctness is reference-based against the doctor's forum "
                 "reply — a GP answer used as a reference, NOT a gold standard, so read "
                 "it as agreement-with-a-GP, not absolute correctness.")
    else:
        note += (" **answer_correctness was not run**: the briefing is a counselor "
                 "decision-support document and the doctor's reply is a patient-facing "
                 "answer, so comparing them mixes genres. The custom judge dropped its "
                 "equivalent metric (`doctor_point_coverage`) on 2026-07-27 for the same "
                 "reason.")
    lines += [note]
    summ = out.with_name(f"ragas_{args.tag}_summary.md")
    summ.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ragas] wrote {out} and {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
