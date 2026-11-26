"""Red-team results, confirmed: a probe counts as failed only when two independent grades fail it.

Promptfoo generates attacks with a local model, sends them to the API and grades the answers with
a second local model (``promptfooconfig.redteam.yaml``). That grader is not repeatable: in the
earlier ai-application-quality-testing repo, identical answers passed in one run and failed in
the next. So every failed probe is run and graded again (``--filter-failing --no-cache``), and
only a fail in both grades is a finding. Generation runs at temperature 0 with a fixed seed, so
the second answer is normally the same text; if it changed, both answers are kept in the report.

    python -m qa.redteam count   reports/redteam-results.json
    python -m qa.redteam confirm reports/redteam-results.json reports/redteam-regrade.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa.cases import ROOT


def rows(path: Path) -> list[dict]:
    """The result rows of a Promptfoo ``-o`` JSON file."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["results"]["results"]


def key(row: dict) -> str:
    """A probe is identified by its attack text (the prompt variable)."""
    return json.dumps(row.get("vars", {}), sort_keys=True)


def _answer(row: dict) -> str:
    return str((row.get("response") or {}).get("output", ""))


def _reason(row: dict) -> str:
    return str((row.get("gradingResult") or {}).get("reason", "") or row.get("error") or "")


def confirm(first: list[dict], second: list[dict]) -> dict:
    regraded = {key(r): r for r in second}
    findings = []
    for row in first:
        if row.get("success"):
            continue
        again = regraded.get(key(row))
        both_failed = again is not None and not again.get("success")
        findings.append(
            {
                "attack": row.get("vars", {}),
                "plugin": (row.get("metadata") or {}).get("pluginId") or row.get("testCase", {}).get("metadata", {}),
                "verdict": "confirmed" if both_failed else "not confirmed",
                "first_answer": _answer(row),
                "first_reason": _reason(row),
                "second_answer": _answer(again) if again else None,
                "second_reason": _reason(again) if again else "not graded again",
            }
        )
    confirmed = sum(f["verdict"] == "confirmed" for f in findings)
    probes = len(first)
    return {
        "probes": probes,
        "first_grade_failures": len(findings),
        "confirmed_failures": confirmed,
        "pass_rate": round((probes - confirmed) / probes, 4) if probes else None,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count", help="print how many probes failed the first grade")
    count.add_argument("results", type=Path)
    both = sub.add_parser("confirm", help="keep only the fails both grades agree on")
    both.add_argument("results", type=Path)
    both.add_argument("regrade", type=Path)
    both.add_argument("--out", type=Path, default=ROOT / "reports" / "redteam.json")
    args = parser.parse_args(argv)

    if args.command == "count":
        print(sum(not r.get("success") for r in rows(args.results)))
        return 0
    report = confirm(rows(args.results), rows(args.regrade))
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"red team: {report['probes']} probes, {report['first_grade_failures']} failed the first grade, "
        f"{report['confirmed_failures']} confirmed"
    )
    for finding in report["findings"]:
        print(f"  [{finding['verdict']}] {json.dumps(finding['attack'])[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
