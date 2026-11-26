"""The quality gate: one verdict from every tool's report.

The tools write their results to ``reports/`` (retrieval eval, DeepEval, Ragas, Promptfoo, the
server's traces). The gate reads them, compares each metric with the limit in a TOML file, writes
a Markdown table and exits 1 if anything is below its limit, or if a report is missing or has
fewer cases than expected: a suite that quietly ran half its cases must not pass.

    python -m qa.gate                                   configs/quality_gate.toml (every pull request)
    python -m qa.gate --config configs/model_gate.toml  the weekly real-model run
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from qa.cases import ROOT


@dataclass
class Check:
    name: str
    metric: str
    value: float | None
    limit: float
    kind: str  # "min" or "max"

    @property
    def passed(self) -> bool:
        if self.value is None:
            return False
        return self.value >= self.limit if self.kind == "min" else self.value <= self.limit


@dataclass
class Section:
    name: str
    report: str
    cases: int | None
    min_cases: int
    checks: list[Check] = field(default_factory=list)
    problem: str | None = None  # a missing or short report

    @property
    def passed(self) -> bool:
        return self.problem is None and all(c.passed for c in self.checks)


def _dig(data: dict, path: str) -> dict:
    for key in path.split(".") if path else []:
        data = data[key]
    return data


def _percentile(values: list[float], share: float) -> float:
    """Nearest-rank percentile: the smallest value with at least ``share`` of the values at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def read_metrics(kind: str, path: Path, at: str) -> tuple[int, dict[str, float | None]]:
    """(number of cases, metric values) from one report, whatever tool wrote it."""
    if kind == "promptfoo":
        stats = json.loads(path.read_text(encoding="utf-8"))["results"]["stats"]
        total = stats["successes"] + stats["failures"] + stats["errors"]
        return total, {"pass_rate": stats["successes"] / total if total else None, "errors": stats["errors"]}
    if kind == "traces":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        latencies = [r["latency_ms"] for r in rows]
        tokens = [r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rows]
        return len(rows), {
            "p50_latency_ms": _percentile(latencies, 0.5) if rows else None,
            "p95_latency_ms": _percentile(latencies, 0.95) if rows else None,
            "mean_tokens": sum(tokens) / len(tokens) if rows else None,
        }
    if kind == "redteam":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["probes"], {"confirmed_failures": data["confirmed_failures"], "pass_rate": data["pass_rate"]}
    section = _dig(json.loads(path.read_text(encoding="utf-8")), at)
    return section["cases"], section["metrics"]


def evaluate(config: dict, reports: Path) -> list[Section]:
    sections = []
    for name, spec in config.items():
        path = reports / spec["report"]
        section = Section(name, spec["report"], None, spec.get("min_cases", 1))
        sections.append(section)
        if not path.exists():
            section.problem = f"report {spec['report']} is missing"
            continue
        try:
            cases, metrics = read_metrics(spec.get("kind", "json"), path, spec.get("at", ""))
        except (KeyError, ValueError) as error:
            section.problem = f"report {spec['report']} is unreadable: {error!r}"
            continue
        section.cases = cases
        if cases < section.min_cases:
            section.problem = f"{cases} cases, expected at least {section.min_cases}"
        for kind in ("min", "max"):
            for metric, limit in spec.get(kind, {}).items():
                # A renamed metric shows up as missing (None) and fails, instead of being skipped
                section.checks.append(Check(name, metric, metrics.get(metric), limit, kind))
    return sections


def markdown(sections: list[Section], title: str) -> str:
    verdict = "PASSED" if all(s.passed for s in sections) else "FAILED"
    lines = [
        f"### {title}: {verdict}",
        "",
        "| Check | Cases | Metric | Value | Limit | |",
        "|---|---:|---|---:|---:|---|",
    ]
    for s in sections:
        cases = "-" if s.cases is None else str(s.cases)
        if s.problem:
            lines.append(f"| {s.name} | {cases} | {s.problem} | | | FAIL |")
        for c in s.checks:
            value = "missing" if c.value is None else f"{c.value:.3f}".rstrip("0").rstrip(".")
            sign = ">=" if c.kind == "min" else "<="
            lines.append(
                f"| {s.name} | {cases} | {c.metric} | {value} | {sign} {c.limit:g} | {'ok' if c.passed else 'FAIL'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "quality_gate.toml")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--summary", type=Path, default=None, help="where to write the Markdown table")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    title = config.pop("title", "Quality gate")
    sections = evaluate(config, args.reports)
    table = markdown(sections, title)
    summary = args.summary or args.reports / f"{args.config.stem}.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(table, encoding="utf-8")
    print(table)
    return 0 if all(s.passed for s in sections) else 1


if __name__ == "__main__":
    raise SystemExit(main())
