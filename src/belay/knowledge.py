"""Loads the knowledge base: Markdown files split into numbered sections.

Each file starts with a small front matter block (title, access). Each ``## <ID> <Title>`` heading
starts a section. A section is the unit the retriever ranks and the answer cites, so a citation
like ``[M2]`` always points at one paragraph a person can check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from belay.guards import looks_like_instruction
from belay.text import sentences

_HEADING = re.compile(r"^##\s+([A-Z]\d+)\s+(.+)$")


@dataclass(frozen=True)
class Section:
    """One citable paragraph of the knowledge base."""

    id: str
    title: str
    text: str
    access: str  # "public" or "staff"; the chat API only ever searches public sections
    source: str  # file name, for traces and the docs
    # Sentences removed at indexing time because they read like instructions to the assistant
    dropped: tuple[str, ...] = field(default=())


def _front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Splits ``---`` front matter from the body. Only flat ``key: value`` lines are supported."""
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def parse(raw: str, source: str) -> list[Section]:
    """Turns one Markdown file into sections.

    Sentences that look like instructions (``ignore all previous instructions ...``) are dropped
    here, before indexing: text from a document is data, and a forum post must not be able to
    steer the assistant. The dropped sentences are kept on the section so tests and traces can
    show what was removed.
    """
    meta, body = _front_matter(raw)
    access = meta.get("access", "public")
    found: list[tuple[str, str, list[str]]] = []
    for line in body.splitlines():
        heading = _HEADING.match(line.strip())
        if heading:
            found.append((heading.group(1), heading.group(2).strip(), []))
        elif found and line.strip():
            found[-1][2].append(line.strip())

    result = []
    for section_id, title, lines in found:
        kept, dropped = [], []
        for sentence in sentences(" ".join(lines)):
            (dropped if looks_like_instruction(sentence) else kept).append(sentence)
        result.append(Section(section_id, title, " ".join(kept), access, source, tuple(dropped)))
    return result


class KnowledgeBase:
    """All sections, with lookups by ID and by access level."""

    def __init__(self, sections: list[Section]):
        ids = [s.id for s in sections]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            # Two sections with one ID would make citations ambiguous
            raise ValueError(f"duplicate section ids: {sorted(duplicates)}")
        self.sections = sections
        self._by_id = {s.id: s for s in sections}

    @classmethod
    def load(cls, folder: Path) -> KnowledgeBase:
        """Reads every ``*.md`` file in the folder, in name order so the index is stable."""
        sections: list[Section] = []
        for path in sorted(folder.glob("*.md")):
            sections.extend(parse(path.read_text(encoding="utf-8"), path.name))
        if not sections:
            raise ValueError(f"no sections found in {folder}")
        return cls(sections)

    def get(self, section_id: str) -> Section:
        return self._by_id[section_id]

    def visible_to(self, access: str) -> list[Section]:
        """Sections a caller with this access level may see; staff also see public sections."""
        allowed = {"public"} if access == "public" else {"public", access}
        return [s for s in self.sections if s.access in allowed]
