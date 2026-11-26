"""One JSON line per answered request: what was asked, what was found, what came back, how long it took.

Traces are how a failed eval gets explained ("the right section was ranked 4th"). They pass
through the same redaction as answers, so a trace file can be shared as a CI artifact.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from belay.guards import OutputGuard


class Tracer:
    """Appends redacted trace records to a JSON Lines file; thread safe (the API serves in threads)."""

    def __init__(self, path: Path | None):
        self._path = path
        self._lock = threading.Lock()
        self._guard = OutputGuard()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **fields: Any) -> None:
        if not self._path:
            return
        fields["question"], _ = self._guard.redact(fields.get("question", ""))
        fields["answer"], _ = self._guard.redact(fields.get("answer", ""))
        line = json.dumps({"time": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields})
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
