"""The quality gate and the case loaders: a gate that can't fail is decoration."""

import json

import pytest

from qa.cases import load_attacks, load_cases
from qa.client import _parse_sse
from qa.gate import _percentile, evaluate

CONFIG = {"suite": {"report": "r.json", "at": "core", "min_cases": 3, "min": {"facts": 0.9}, "max": {"cost": 1}}}


def write(tmp_path, cases=3, metrics=None):
    metrics = {"facts": 0.95, "cost": 0.5} if metrics is None else metrics
    (tmp_path / "r.json").write_text(json.dumps({"core": {"cases": cases, "metrics": metrics}}))


def test_passes_when_every_limit_holds(tmp_path):
    write(tmp_path)
    assert all(s.passed for s in evaluate(CONFIG, tmp_path))


@pytest.mark.parametrize(
    ("cases", "metrics", "why"),
    [
        (3, {"facts": 0.8, "cost": 0.5}, "below min"),
        (3, {"facts": 0.95, "cost": 2}, "above max"),
        (2, {"facts": 0.95, "cost": 0.5}, "too few cases"),
        (3, {"fact_share": 0.95, "cost": 0.5}, "renamed metric"),
        (3, {"facts": None, "cost": 0.5}, "metric without a value"),
    ],
)
def test_fails(tmp_path, cases, metrics, why):
    write(tmp_path, cases, metrics)
    assert not evaluate(CONFIG, tmp_path)[0].passed, why


def test_missing_report_fails(tmp_path):
    section = evaluate(CONFIG, tmp_path)[0]
    assert not section.passed and "missing" in section.problem


def test_promptfoo_and_traces_reports(tmp_path):
    (tmp_path / "pf.json").write_text(json.dumps({"results": {"stats": {"successes": 9, "failures": 1, "errors": 0}}}))
    (tmp_path / "t.jsonl").write_text(
        "\n".join(
            json.dumps({"latency_ms": ms, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            for ms in range(1, 101)
        )
    )
    config = {
        "pf": {"report": "pf.json", "kind": "promptfoo", "min_cases": 10, "min": {"pass_rate": 1.0}},
        "lat": {"report": "t.jsonl", "kind": "traces", "min_cases": 100, "max": {"p95_latency_ms": 95}},
    }
    pf, lat = evaluate(config, tmp_path)
    assert not pf.passed and pf.checks[0].value == 0.9
    assert lat.passed and lat.checks[0].value == 95


def test_percentile_is_nearest_rank():
    assert _percentile([5, 1, 3], 0.5) == 3
    assert _percentile(list(range(1, 21)), 0.95) == 19


def test_bundled_case_files_load():
    assert len(load_cases("core")) >= 28 and len(load_cases("robustness")) >= 14 and len(load_attacks()) >= 16


def test_answered_case_without_expected_facts_is_refused(tmp_path):
    (tmp_path / "core.jsonl").write_text(
        json.dumps(
            {"id": "X", "question": "q", "relevant": ["M1"], "status": "answered", "must_contain": [], "reference": ""}
        )
    )
    with pytest.raises(ValueError, match="must_contain"):
        load_cases("core", tmp_path)


def test_duplicate_case_ids_are_refused(tmp_path):
    row = json.dumps(
        {"id": "X", "question": "q", "relevant": [], "status": "not_found", "must_contain": [], "reference": ""}
    )
    (tmp_path / "core.jsonl").write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases("core", tmp_path)


def test_sse_parser_reads_events_and_multiline_data():
    lines = ["event: token", 'data: {"text": "Hi "}', "", "event: done", 'data: {"a":', "data: 1}", ""]
    assert list(_parse_sse(iter(lines))) == [("token", {"text": "Hi "}), ("done", {"a": 1})]


def test_manual_test_case_document_is_up_to_date():
    from qa.case_doc import DOC, render

    assert DOC.read_text(encoding="utf-8") == render(), "run: python -m qa.case_doc"
