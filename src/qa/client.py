"""The API client every test and evaluation uses (the API's equivalent of a page object).

Tests never build URLs or parse JSON themselves: they call ``chat()`` and get a typed ``Reply``.
The same client talks to the app in-process (fast, for unit-level API tests) or to a real server
over HTTP (``BELAY_BASE_URL``), so CI runs the very same tests against a started uvicorn.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

# --- The contract, as a client sees it. Kept separate from the server's models on purpose: if the
# --- server renames a field, these fail, which is what a real client would experience.


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CitedSection(_Strict):
    id: str
    title: str


class ContextSection(CitedSection):
    text: str


class Usage(_Strict):
    prompt_tokens: int
    completion_tokens: int


class Reply(_Strict):
    answer: str
    status: Literal["answered", "not_found", "refused"]
    citations: list[CitedSection]
    contexts: list[ContextSection] | None = None
    trace_id: str
    latency_ms: float
    generator: str
    usage: Usage

    @property
    def cited_ids(self) -> list[str]:
        return [c.id for c in self.citations]

    @property
    def context_texts(self) -> list[str]:
        return [c.text for c in self.contexts or []]


@dataclass
class StreamResult:
    """What a streaming call produced: the raw events, the joined tokens and the final reply."""

    content_type: str
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        return [data["text"] for event, data in self.events if event == "token"]

    @property
    def done(self) -> Reply:
        finals = [data for event, data in self.events if event == "done"]
        if len(finals) != 1:
            raise AssertionError(f"expected exactly one 'done' event, got {len(finals)}")
        return Reply.model_validate(finals[0])


def _parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Reads Server-Sent Events: ``event:`` and ``data:`` lines, a blank line ends one event."""
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, json.loads("\n".join(data))


class ChatApiClient:
    def __init__(self, http: httpx.Client, client_id: str = "qa-tests"):
        self._http = http
        self.client_id = client_id

    @classmethod
    def for_url(cls, base_url: str, client_id: str = "qa-tests", timeout: float = 300.0) -> ChatApiClient:
        """A client for a running server. The long timeout covers a CPU-only model answering."""
        return cls(httpx.Client(base_url=base_url, timeout=timeout), client_id)

    @classmethod
    def for_app(cls, app: Any, client_id: str = "qa-tests") -> ChatApiClient:
        """A client for an app object, in the same process (no port, no server)."""
        from fastapi.testclient import TestClient

        return cls(TestClient(app), client_id)

    @classmethod
    def from_env(cls, client_id: str = "qa-tests") -> ChatApiClient:
        """``BELAY_BASE_URL`` set: talk to that server. Not set: build the app from the environment."""
        url = os.environ.get("BELAY_BASE_URL")
        if url:
            return cls.for_url(url, client_id)
        from belay.api import create_app

        return cls.for_app(create_app(), client_id)

    def _headers(self, client_id: str | None) -> dict[str, str]:
        return {"X-Client-Id": client_id or self.client_id}

    def health(self) -> dict[str, Any]:
        response = self._http.get("/health")
        response.raise_for_status()
        return response.json()

    def post_chat(self, body: Any, client_id: str | None = None, path: str = "/chat") -> httpx.Response:
        """The raw call, for tests about status codes and error bodies."""
        return self._http.post(path, json=body, headers=self._headers(client_id))

    def chat(self, question: str, include_contexts: bool = True, client_id: str | None = None) -> Reply:
        response = self.post_chat({"question": question, "include_contexts": include_contexts}, client_id)
        response.raise_for_status()
        return Reply.model_validate(response.json())

    def stream(self, question: str, include_contexts: bool = False, client_id: str | None = None) -> StreamResult:
        body = {"question": question, "include_contexts": include_contexts}
        with self._http.stream("POST", "/chat/stream", json=body, headers=self._headers(client_id)) as response:
            response.raise_for_status()
            result = StreamResult(response.headers.get("content-type", ""))
            result.events.extend(_parse_sse(response.iter_lines()))
        return result

    def close(self) -> None:
        self._http.close()
