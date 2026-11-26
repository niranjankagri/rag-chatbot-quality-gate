"""Ragas evaluation: answers compared with reference answers, and checked against their sources.

Two levels, like the DeepEval tests:

- **Offline** (default, gates every pull request): BLEU, ROUGE-L and Levenshtein string
  similarity between the answer and the reference answer a person wrote. Cheap and repeatable,
  but they reward shared words, not shared meaning, so they are read as trend lines.
- **Judged** (``--model``): Ragas' Faithfulness (is every claim supported by the retrieved
  sections), Context Precision and Context Recall (did retrieval bring the right sections, judged
  against the reference) and Answer Relevancy (does the answer address the question), with a
  local Ollama model as judge, through Ollama's OpenAI-compatible endpoint.

Every call goes through the HTTP API, like a real client, with ``include_contexts`` on.

    python -m qa.ragas_eval
    python -m qa.ragas_eval --model --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qa.cases import ROOT, SUITES, Case, load_cases
from qa.client import ChatApiClient, Reply

OFFLINE = ("bleu", "rouge_l", "string_similarity")


def offline_metrics() -> dict:
    from ragas.metrics.collections import BleuScore, NonLLMStringSimilarity, RougeScore

    return {
        "bleu": BleuScore(),
        "rouge_l": RougeScore(rouge_type="rougeL"),
        "string_similarity": NonLLMStringSimilarity(),
    }


def judged_metrics() -> dict:  # pragma: needs ollama
    """Ragas' LLM metrics with a local judge. Ollama speaks the OpenAI API under /v1."""
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/v1"
    client = OpenAI(base_url=base, api_key="ollama")  # Ollama ignores the key, the client needs one
    llm = llm_factory(os.environ.get("JUDGE_MODEL", "qwen2.5:7b"), provider="openai", client=client, temperature=0)
    # Answer Relevancy embeds asynchronously and refuses a synchronous client (first run, 26 Nov)
    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(base_url=base, api_key="ollama"),
        model=os.environ.get("BELAY_EMBED_MODEL", "nomic-embed-text"),
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }


def score_offline(metrics: dict, case: Case, reply: Reply) -> dict[str, float]:
    # The answer without its "[M1]" citation, so the markers don't count as words
    answer = reply.answer.rsplit(" [", 1)[0] if reply.answer.endswith("]") else reply.answer
    return {
        name: round(float(m.score(reference=case.reference, response=answer).value), 4) for name, m in metrics.items()
    }


def score_judged(metrics: dict, case: Case, reply: Reply) -> dict[str, float | None]:  # pragma: needs ollama
    contexts = reply.context_texts
    inputs = {
        "faithfulness": {"user_input": case.question, "response": reply.answer, "retrieved_contexts": contexts},
        "context_precision": {"user_input": case.question, "reference": case.reference, "retrieved_contexts": contexts},
        "context_recall": {"user_input": case.question, "retrieved_contexts": contexts, "reference": case.reference},
        "answer_relevancy": {"user_input": case.question, "response": reply.answer},
    }
    scores: dict[str, float | None] = {}
    for name, metric in metrics.items():
        if not contexts and name != "answer_relevancy":
            scores[name] = None  # nothing was retrieved: there is nothing to judge
            continue
        try:
            scores[name] = round(float(metric.score(**inputs[name]).value), 4)
        except Exception as error:  # a judge that returns broken JSON must not end the whole run
            print(f"  {case.id} {name}: judge error {type(error).__name__}: {error}")
            scores[name] = None
    return scores


def _mean(rows: list[dict], name: str) -> float | None:
    values = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
    return round(sum(values) / len(values), 4) if values else None


def run(api: ChatApiClient, judged: bool, limit: int | None) -> dict:
    metrics = judged_metrics() if judged else offline_metrics()
    report: dict = {"mode": "judged" if judged else "offline", "suites": {}}
    answerable = {suite: [c for c in load_cases(suite) if c.status == "answered"] for suite in SUITES}
    if limit:
        # The same number of cases from each suite, always the first ones, so runs compare
        answerable = {suite: cases[: max(1, limit // len(SUITES))] for suite, cases in answerable.items()}
    for suite, cases in answerable.items():
        rows = []
        for case in cases:
            reply = api.chat(case.question, client_id=f"ragas-{case.id}")
            scores = score_judged(metrics, case, reply) if judged else score_offline(metrics, case, reply)
            rows.append({"id": case.id, "status": reply.status, "scores": scores})
            print(f"  {suite:<10} {case.id} {reply.status:<9} " + " ".join(f"{k}={v}" for k, v in scores.items()))
        report["suites"][suite] = {"cases": len(rows), "metrics": {n: _mean(rows, n) for n in metrics}, "rows": rows}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="store_true", help="LLM-judged metrics with a local Ollama judge")
    parser.add_argument("--limit", type=int, default=None, help="at most this many answerable cases in total")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or ROOT / "reports" / ("ragas-judged.json" if args.model else "ragas.json")
    api = ChatApiClient.from_env(client_id="ragas")
    try:
        report = run(api, args.model, args.limit)
    finally:
        api.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for suite, result in report["suites"].items():
        print(f"{suite}: {result['cases']} cases, " + ", ".join(f"{k} {v}" for k, v in result["metrics"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
