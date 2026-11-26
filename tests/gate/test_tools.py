"""The evaluation tools themselves (retrieval eval, Ragas offline, stability, red-team confirm, gate CLI),
run in-process against the app so their logic is tested, not only their output in CI."""

import json

import pytest

from belay.api import create_app
from qa import gate, redteam, retrieval_eval, stability
from qa.client import ChatApiClient
from qa.deepeval_adapter import GraderMetric, Recorder
from qa.ragas_eval import run as ragas_run


@pytest.fixture(scope="module")
def local_api():
    client = ChatApiClient.for_app(create_app())
    yield client
    client.close()


def test_retrieval_eval_writes_every_mode(tmp_path, capsys):
    out = tmp_path / "retrieval.json"
    assert retrieval_eval.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["modes"]) == {"bm25", "vector", "hybrid"}
    assert report["modes"]["hybrid"]["core"]["metrics"]["recall_at_k"] == 1.0
    assert "hybrid" in capsys.readouterr().out


def test_reciprocal_rank():
    assert retrieval_eval.reciprocal_rank(["A", "B", "C"], ("B",)) == 0.5
    assert retrieval_eval.reciprocal_rank(["A"], ("Z",)) == 0.0


def test_ragas_offline_scores_identical_answers_as_one(local_api):
    report = ragas_run(local_api, judged=False, limit=4)
    assert report["mode"] == "offline"
    core = report["suites"]["core"]
    assert core["cases"] == 2 and set(core["metrics"]) == {"bleu", "rouge_l", "string_similarity"}
    assert all(0 <= v <= 1 for v in core["metrics"].values())


def test_stability_of_the_deterministic_generator_is_perfect(local_api):
    report = stability.measure(local_api, [("T-1", "How much is a day pass?"), ("T-2", "Is there a sauna?")], 3)
    assert report["metrics"] == {"same_status": 1.0, "same_citations": 1.0, "identical_answer": 1.0, "rouge_l": 1.0}


def promptfoo_file(path, rows):
    path.write_text(json.dumps({"results": {"results": rows}}))
    return path


def test_red_team_counts_a_failure_only_when_both_grades_fail(tmp_path, capsys):
    first = promptfoo_file(
        tmp_path / "first.json",
        [
            {"vars": {"prompt": "a"}, "success": True},
            {"vars": {"prompt": "b"}, "success": False, "response": {"output": "x"}},
            {"vars": {"prompt": "c"}, "success": False, "response": {"output": "y"}},
        ],
    )
    second = promptfoo_file(
        tmp_path / "second.json",
        [{"vars": {"prompt": "b"}, "success": False}, {"vars": {"prompt": "c"}, "success": True}],
    )
    assert redteam.main(["count", str(first)]) == 0
    assert capsys.readouterr().out.strip() == "2"
    out = tmp_path / "redteam.json"
    assert redteam.main(["confirm", str(first), str(second), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert (report["probes"], report["first_grade_failures"], report["confirmed_failures"]) == (3, 2, 1)
    assert report["pass_rate"] == round(2 / 3, 4)


def test_red_team_without_a_regrade_confirms_nothing(tmp_path):
    first = [{"vars": {"prompt": "b"}, "success": False}]
    assert redteam.confirm(first, [])["confirmed_failures"] == 0


def test_gate_cli_exit_codes_and_summary(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"s": {"cases": 5, "metrics": {"m": 0.9}}}))
    config = tmp_path / "g.toml"
    config.write_text('title = "T"\n[x]\nreport = "r.json"\nat = "s"\nmin_cases = 5\nmin = { m = 0.8 }\n')
    assert gate.main(["--config", str(config), "--reports", str(tmp_path)]) == 0
    assert "T: PASSED" in (tmp_path / "g.md").read_text()
    config.write_text('[x]\nreport = "r.json"\nat = "s"\nmin = { m = 0.95 }\n')
    assert gate.main(["--config", str(config), "--reports", str(tmp_path)]) == 1
    assert "FAIL" in (tmp_path / "g.md").read_text()


def test_gate_reports_an_unreadable_report(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"other": {}}))
    section = gate.evaluate({"x": {"report": "r.json", "at": "s"}}, tmp_path)[0]
    assert "unreadable" in section.problem


def test_gate_reads_the_red_team_report(tmp_path):
    (tmp_path / "rt.json").write_text(json.dumps({"probes": 12, "confirmed_failures": 1, "pass_rate": 0.9167}))
    config = {"rt": {"report": "rt.json", "kind": "redteam", "min_cases": 10, "max": {"confirmed_failures": 0}}}
    assert not gate.evaluate(config, tmp_path)[0].passed


def test_recorder_writes_means_per_suite(tmp_path):
    recorder = Recorder()
    for case_id, score in (("A", 1.0), ("B", 0.0)):
        metric = GraderMetric("facts", lambda s=score: s)
        metric.measure(None)
        recorder.add("core", case_id, [metric])
    recorder.write(tmp_path / "d.json")
    assert json.loads((tmp_path / "d.json").read_text())["core"]["metrics"] == {"facts": 0.5}
    Recorder().write(tmp_path / "empty.json")  # nothing recorded: nothing written
    assert not (tmp_path / "empty.json").exists()


def test_stability_and_ragas_command_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("BELAY_BASE_URL", raising=False)  # in-process app
    assert stability.main(["--repeats", "2", "--limit", "2", "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["cases"] == 2
    from qa import ragas_eval

    assert ragas_eval.main(["--limit", "2", "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["mode"] == "offline"
    with pytest.raises(SystemExit):
        stability.main(["--repeats", "1"])
