"""Non-determinism: ask the same questions several times and measure how much the answers move.

A model at temperature 0 with a fixed seed usually repeats itself on one machine, but not
always, and not across machines or Ollama versions. A test that passed once proves little if the
next identical call gives a different answer, so this measures, per question over N repeats:

- same_status: every repeat had the same status (answered / not_found / refused)
- same_citations: every repeat cited the same sections
- identical_answer: every repeat was word for word the same
- rouge_l: mean ROUGE-L between each repeat and the first one (1.0 = same words)

    python -m qa.stability --repeats 5 --limit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from qa.cases import ROOT, load_cases
from qa.client import ChatApiClient


def measure(api: ChatApiClient, questions: list[tuple[str, str]], repeats: int) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    rows = []
    for case_id, question in questions:
        replies = [api.chat(question, client_id=f"stability-{case_id}-{i}") for i in range(repeats)]
        first = replies[0]
        rows.append(
            {
                "id": case_id,
                "statuses": [r.status for r in replies],
                "same_status": len({r.status for r in replies}) == 1,
                "same_citations": len({tuple(r.cited_ids) for r in replies}) == 1,
                "identical_answer": len({r.answer for r in replies}) == 1,
                "rouge_l": round(
                    sum(scorer.score(first.answer, r.answer)["rougeL"].fmeasure for r in replies[1:]) / (repeats - 1),
                    4,
                ),
            }
        )
        print(f"  {case_id}: {rows[-1]['statuses']} identical={rows[-1]['identical_answer']}")
    names = ("same_status", "same_citations", "identical_answer", "rouge_l")
    metrics = {n: round(sum(float(r[n]) for r in rows) / len(rows), 4) for n in names}
    return {"cases": len(rows), "repeats": repeats, "metrics": metrics, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=8, help="questions from the core suite")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "stability.json")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    questions = [(c.id, c.question) for c in load_cases("core")][: args.limit]
    api = ChatApiClient.from_env(client_id="stability")
    try:
        report = measure(api, questions, args.repeats)
    finally:
        api.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"stability over {args.repeats} repeats: {report['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
