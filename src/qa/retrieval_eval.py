"""Retrieval quality, measured on its own, for every retrieval mode side by side.

A wrong answer has two possible causes: the right section was never found, or it was found and
the generator got it wrong. Measuring retrieval separately tells them apart. Metrics:

- recall@k: the share of questions where a relevant section is in the top k
- MRR (mean reciprocal rank): 1 / position of the first relevant section, averaged
- no-answer accuracy: for questions the knowledge base can't answer, the share where the
  evidence rule correctly finds nothing (else the generator is handed a wrong section)

    python -m qa.retrieval_eval                    TF-IDF vectors (offline)
    python -m qa.retrieval_eval --embedder ollama  vectors from a local embedding model
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from belay.config import Settings
from belay.pipeline import build_pipeline
from belay.retrieval import MODES
from qa.cases import ROOT, SUITES, Case, load_cases


def reciprocal_rank(ranked: list[str], relevant: tuple[str, ...]) -> float:
    for position, section_id in enumerate(ranked, start=1):
        if section_id in relevant:
            return 1 / position
    return 0.0


def evaluate_mode(settings: Settings, cases: list[Case]) -> dict:
    pipeline = build_pipeline(settings)
    answerable, unanswerable, rows = [], [], []
    for case in cases:
        hits = pipeline.retriever.search(case.question, settings.top_k)
        ranked = [h.section.id for h in hits]
        found = [h.section.id for h in pipeline.evidence(hits)]
        row = {"id": case.id, "ranked": ranked, "evidence": found}
        if case.status == "answered":
            row["rr"] = reciprocal_rank(ranked, case.relevant)
            # Counted as found only when the evidence rule also lets it through
            row["hit"] = bool(set(found) & set(case.relevant))
            answerable.append(row)
        elif case.status == "not_found":
            row["abstained"] = not found
            unanswerable.append(row)
        rows.append(row)
    metrics = {
        "recall_at_k": _mean([r["hit"] for r in answerable]),
        "mrr": _mean([r["rr"] for r in answerable]),
        "no_answer_accuracy": _mean([r["abstained"] for r in unanswerable]) if unanswerable else None,
    }
    return {"cases": len(answerable) + len(unanswerable), "metrics": metrics, "rows": rows}


def _mean(values: list) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run(settings: Settings) -> dict:
    report: dict = {"embedder": settings.embedder, "top_k": settings.top_k, "selected": settings.retriever, "modes": {}}
    for mode in MODES:
        mode_settings = replace(settings, retriever=mode, generator="extractive")
        report["modes"][mode] = {suite: evaluate_mode(mode_settings, load_cases(suite)) for suite in SUITES}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", default=None, help="tfidf or ollama (default: BELAY_EMBEDDER or tfidf)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "retrieval.json")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.embedder:
        settings = replace(settings, embedder=args.embedder)
    report = run(settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Retrieval ({settings.embedder} vectors, top {settings.top_k}):")
    print(f"{'mode':<8} {'suite':<11} {'recall@k':>9} {'MRR':>6} {'no-answer':>10}")
    for mode, suites in report["modes"].items():
        for suite, result in suites.items():
            m = result["metrics"]
            no_answer = "-" if m["no_answer_accuracy"] is None else f"{m['no_answer_accuracy']:.3f}"
            print(f"{mode:<8} {suite:<11} {m['recall_at_k']:>9.3f} {m['mrr']:>6.3f} {no_answer:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
