"""Collects every DeepEval score of the run and writes it for the quality gate.

``reports/deepeval.json`` holds the deterministic metrics, ``reports/deepeval-judged.json`` the
LLM-judged ones (written only when judged tests ran). The gate reads these files; it never
re-runs a model.
"""

from __future__ import annotations

import os

import pytest

from qa.cases import ROOT, Attack, Case, known_misses
from qa.deepeval_adapter import Recorder

REPORTS = ROOT / "reports"
_graded = Recorder()
_judged = Recorder()


@pytest.fixture(scope="session")
def graded() -> Recorder:
    return _graded


@pytest.fixture(scope="session")
def judged() -> Recorder:
    return _judged


def pytest_sessionfinish(session: pytest.Session) -> None:
    _graded.write(REPORTS / "deepeval.json")
    _judged.write(REPORTS / "deepeval-judged.json")


def with_known_misses(items: list[Case | Attack]) -> list:
    """Parametrize values; a known miss of the current generator becomes a strict xfail."""
    misses = known_misses(os.environ.get("BELAY_GENERATOR", "extractive"))
    return [
        pytest.param(i, id=i.id, marks=pytest.mark.xfail(reason=misses[i.id], strict=True))
        if i.id in misses
        else pytest.param(i, id=i.id)
        for i in items
    ]
