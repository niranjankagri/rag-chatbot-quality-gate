"""Typed test data: the versioned case files in ``datasets/`` as Python objects.

A case file is reviewed like code: one line per case, an ID that never changes, and the expected
facts written by a person from the knowledge base. Loading fails on a broken file instead of
quietly running fewer cases.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SUITES = ("core", "robustness")
STATUSES = ("answered", "not_found", "refused")


@dataclass(frozen=True)
class Case:
    """A question with a known right answer (or a known right "I don't know")."""

    id: str
    suite: str
    question: str
    relevant: tuple[str, ...]  # sections that hold the answer; empty when nothing does
    status: str  # the expected status
    must_contain: tuple[str, ...]  # facts the answer must state, word for word
    reference: str  # a correct answer written by a person (Ragas compares against it)


@dataclass(frozen=True)
class Attack:
    """An adversarial question: what the assistant may answer with, and what must never appear."""

    id: str
    category: str
    question: str
    allowed: tuple[str, ...]  # acceptable statuses
    must_not_contain: tuple[str, ...]


def _lines(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} line {number}: {error}") from error
    return rows


def _unique(ids: list[str], name: str) -> None:
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{name}: duplicate ids {duplicates}")


def load_cases(suite: str, folder: Path = DATASETS) -> list[Case]:
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")
    cases = []
    for row in _lines(folder / f"{suite}.jsonl"):
        if row["status"] not in STATUSES:
            raise ValueError(f"{row['id']}: unknown status {row['status']!r}")
        if row["status"] == "answered" and not (row["relevant"] and row["must_contain"]):
            # An "answered" case without expected facts would pass any answer
            raise ValueError(f"{row['id']}: an answered case needs relevant sections and must_contain")
        cases.append(
            Case(
                row["id"],
                suite,
                row["question"],
                tuple(row["relevant"]),
                row["status"],
                tuple(row["must_contain"]),
                row["reference"],
            )
        )
    _unique([c.id for c in cases], suite)
    return cases


def load_attacks(folder: Path = DATASETS) -> list[Attack]:
    attacks = [
        Attack(r["id"], r["category"], r["question"], tuple(r["allowed"]), tuple(r["must_not_contain"]))
        for r in _lines(folder / "attacks.jsonl")
    ]
    _unique([a.id for a in attacks], "attacks")
    for attack in attacks:
        unknown = set(attack.allowed) - set(STATUSES)
        if unknown:
            raise ValueError(f"{attack.id}: unknown statuses {sorted(unknown)}")
    return attacks


def known_misses(generator: str, path: Path = ROOT / "configs" / "known_misses.toml") -> dict[str, str]:
    """Case ID -> reason, for cases this generator is known to get wrong today.

    The tests turn these into strict expected failures: still visible in every run, and the run
    fails when one starts to pass, so the list can't go stale.
    """
    table = tomllib.loads(path.read_text(encoding="utf-8"))
    key = generator.split(":")[0]  # "ollama:llama3.2:3b" -> "ollama"
    return dict(table.get(key, {}))
