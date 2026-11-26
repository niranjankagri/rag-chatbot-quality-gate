"""DeepEval adapter: our cases and replies as DeepEval test cases, our graders as DeepEval metrics.

Two groups of metrics (Adapter pattern: the graders don't know DeepEval exists):

- **Deterministic** (``grader_metrics``): wraps ``qa.graders`` in DeepEval's ``BaseMetric``. No
  model, no network, same result every run; these gate every pull request.
- **LLM-judged** (``judge_metrics``): DeepEval's own Faithfulness, Answer Relevancy and a G-Eval
  correctness rubric, judged by a local Ollama model. They catch what word matching can't (a
  correct paraphrase, an invented fact in new words) but vary between runs, so they run weekly.

``Recorder`` keeps every score so the quality gate can read them from ``reports/``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from qa import graders
from qa.cases import Attack, Case
from qa.client import Reply


class GraderMetric(BaseMetric):
    """A DeepEval metric backed by one of our deterministic graders."""

    def __init__(self, name: str, grade: Callable[[], float], threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False  # nothing to await: the grade is a plain function call
        self._name = name
        self._grade = grade

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = self._grade()
        self.success = self.score >= self.threshold
        self.reason = f"{self._name} = {self.score:.2f} (needs {self.threshold:.2f})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self) -> str:
        return self._name


def as_test_case(question: str, reply: Reply, reference: str | None = None, case_id: str = "") -> LLMTestCase:
    """The DeepEval view of one exchange: input, actual output, what it was built from."""
    return LLMTestCase(
        input=question,
        actual_output=reply.answer,
        expected_output=reference,
        retrieval_context=reply.context_texts or None,
        completion_time=reply.latency_ms / 1000,
        input_token_count=reply.usage.prompt_tokens,
        output_token_count=reply.usage.completion_tokens,
        name=case_id or None,
    )


def grader_metrics(case: Case, reply: Reply) -> list[BaseMetric]:
    metrics: list[BaseMetric] = [
        GraderMetric("status", lambda: graders.status_correct(case, reply)),
        GraderMetric("facts", lambda: graders.facts_present(case, reply)),
        GraderMetric("citation", lambda: graders.citation_correct(case, reply)),
        GraderMetric("grounded", lambda: graders.grounded(reply)),
    ]
    return metrics


def attack_metrics(attack: Attack, reply: Reply) -> list[BaseMetric]:
    return [
        GraderMetric("safe", lambda: graders.attack_safe(attack, reply)),
        GraderMetric("no_leak", lambda: graders.no_leak(attack, reply)),
    ]


def judge_model():  # pragma: needs ollama
    """The local judge: a different, larger model than the one answering (it grades, it doesn't answer)."""
    from deepeval.models import OllamaModel

    return OllamaModel(
        model=os.environ.get("JUDGE_MODEL", "qwen2.5:7b"),
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        temperature=0,
    )


def judge_metrics(threshold: float = 0.7) -> list[BaseMetric]:  # pragma: needs ollama
    """LLM-judged metrics for an answered case. Reasons are off: they double the model calls."""
    judge = judge_model()
    common = {"model": judge, "threshold": threshold, "async_mode": False, "include_reason": False}
    return [
        FaithfulnessMetric(**common),
        AnswerRelevancyMetric(**common),
        GEval(
            name="Correctness",
            criteria=(
                "Compare the actual output with the expected output. It is correct when it states the same "
                "facts (numbers, days, prices, conditions) and adds nothing that contradicts them. Missing "
                "extra detail is fine; a wrong or invented fact is not."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=judge,
            threshold=threshold,
            async_mode=False,
        ),
    ]


class Recorder:
    """Collects metric scores per suite and writes them for the gate (``reports/deepeval*.json``)."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = defaultdict(list)

    def add(self, suite: str, case_id: str, metrics: list[BaseMetric]) -> None:
        scores = {}
        for metric in metrics:
            score = getattr(metric, "score", None)
            scores[metric.__name__] = None if score is None else round(float(score), 4)
        self.rows[suite].append({"id": case_id, "scores": scores})

    def summary(self) -> dict:
        result = {}
        for suite, rows in self.rows.items():
            names = sorted({n for row in rows for n in row["scores"]})
            means = {}
            for name in names:
                values = [row["scores"][name] for row in rows if row["scores"].get(name) is not None]
                means[name] = round(sum(values) / len(values), 4) if values else None
            result[suite] = {"cases": len(rows), "metrics": means, "rows": rows}
        return result

    def write(self, path: Path) -> None:
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
