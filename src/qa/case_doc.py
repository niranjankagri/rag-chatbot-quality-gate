"""Writes docs/manual-test-cases.md from the case files: every case in plain language, by ID.

The document is generated, never edited by hand, and a test fails when it is out of date, so a
reviewer reading it reads exactly what the tests run.

    python -m qa.case_doc
"""

from __future__ import annotations

from qa.cases import ROOT, known_misses, load_attacks, load_cases

DOC = ROOT / "docs" / "manual-test-cases.md"

HEADER = """# Manual test cases

Every case the evaluations run, in plain language. Generated from `datasets/*.jsonl` by
`python -m qa.case_doc`; a test fails if this file is out of date. "Known miss" marks a case the
deterministic generator gets wrong today (`configs/known_misses.toml`), with the measured reason.
"""


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render() -> str:
    misses = known_misses("extractive")
    parts = [HEADER]
    titles = {
        "core": "Core questions (worded like the help articles)",
        "robustness": "Reworded questions (how people really ask)",
    }
    for suite, title in titles.items():
        parts += [f"\n## {title}\n", "| ID | Question | Expected | Must contain | Sections |", "|---|---|---|---|---|"]
        for c in load_cases(suite):
            expected = c.status + (" (known miss)" if c.id in misses else "")
            facts = "; ".join(f"`{f}`" for f in c.must_contain) or "-"
            parts.append(
                f"| {c.id} | {_cell(c.question)} | {expected} | {_cell(facts)} | {', '.join(c.relevant) or '-'} |"
            )
    parts += [
        "\n## Attacks\n",
        "| ID | Category | Question | Allowed | Must never contain |",
        "|---|---|---|---|---|",
    ]
    for a in load_attacks():
        allowed = " or ".join(a.allowed) + (" (known miss)" if a.id in misses else "")
        banned = "; ".join(f"`{b}`" for b in a.must_not_contain) or "-"
        parts.append(f"| {a.id} | {a.category} | {_cell(a.question)} | {allowed} | {_cell(banned)} |")
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
