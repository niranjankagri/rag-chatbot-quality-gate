"""The golden cases, through the API, graded with DeepEval.

One test per case. Deterministic metrics (status, facts, citation, grounded) always run and must
all pass; LLM-judged metrics (faithfulness, answer relevancy, correctness) run with ``--model``.
Every score is also recorded for the quality gate, including those of known misses.
"""

from __future__ import annotations

import os

import pytest
from deepeval import assert_test

from qa.cases import load_attacks, load_cases
from qa.client import ChatApiClient
from qa.deepeval_adapter import Recorder, as_test_case, attack_metrics, grader_metrics, judge_metrics
from tests.evals.conftest import with_known_misses

CORE = load_cases("core")
ROBUSTNESS = load_cases("robustness")
ATTACKS = load_attacks()

# Judged metrics cost several judge calls per case; JUDGE_LIMIT keeps a CPU-only run within its timeout
JUDGE_LIMIT = int(os.environ.get("JUDGE_LIMIT", "0")) or None
JUDGED = [c for c in CORE + ROBUSTNESS if c.status == "answered"][:JUDGE_LIMIT]


@pytest.mark.parametrize("case", with_known_misses(CORE))
def test_core(api: ChatApiClient, graded: Recorder, case):
    reply = api.chat(case.question, client_id=case.id)
    metrics = grader_metrics(case, reply)
    try:
        assert_test(as_test_case(case.question, reply, case.reference, case.id), metrics, run_async=False)
    finally:
        graded.add("core", case.id, metrics)


@pytest.mark.parametrize("case", with_known_misses(ROBUSTNESS))
def test_robustness(api: ChatApiClient, graded: Recorder, case):
    reply = api.chat(case.question, client_id=case.id)
    metrics = grader_metrics(case, reply)
    try:
        assert_test(as_test_case(case.question, reply, case.reference, case.id), metrics, run_async=False)
    finally:
        graded.add("robustness", case.id, metrics)


@pytest.mark.parametrize("attack", with_known_misses(ATTACKS))
def test_attack(api: ChatApiClient, graded: Recorder, attack):
    reply = api.chat(attack.question, client_id=attack.id)
    metrics = attack_metrics(attack, reply)
    try:
        assert_test(as_test_case(attack.question, reply, case_id=attack.id), metrics, run_async=False)
    finally:
        graded.add("security", attack.id, metrics)


@pytest.mark.model
@pytest.mark.parametrize("case", JUDGED, ids=[c.id for c in JUDGED])
def test_judged(api: ChatApiClient, judged: Recorder, case):
    """LLM-judged quality. A low score is recorded and reported; the model gate decides, not this test.

    The judge varies between runs, so failing a single case here would make the run flaky. The
    gate compares the average over all judged cases with a limit instead.
    """
    reply = api.chat(case.question, client_id=f"{case.id}-judged")
    if reply.status != "answered":
        judged.add(case.suite, case.id, [])
        pytest.skip(f"not answered ({reply.status}): nothing to judge")
    metrics = judge_metrics()
    test_case = as_test_case(case.question, reply, case.reference, case.id)
    for metric in metrics:
        metric.measure(test_case)
    judged.add(case.suite, case.id, metrics)
