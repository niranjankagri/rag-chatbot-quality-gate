"""The HTTP API of the assistant (FastAPI).

    GET  /health        is the service up, which generator and retriever does it use
    POST /chat          one question, one JSON answer with citations
    POST /chat/stream   the same answer as Server-Sent Events: "token" events, then one "done"

Every chat call counts against a per-client rate limit (429 with Retry-After). Run it with
``uvicorn belay.api:app``; settings come from the environment (see ``config.py``).
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from belay import __version__
from belay.config import Settings
from belay.pipeline import ChatPipeline, ChatResult, build_pipeline


class ChatRequest(BaseModel):
    # Unknown fields are an error (422), so a typo like "questoin" fails loudly instead of silently
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    include_contexts: bool = False  # evaluators set it to see what the answer was built from

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class Citation(BaseModel):
    id: str
    title: str


class Context(Citation):
    text: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "not_found", "refused"]
    citations: list[Citation]
    contexts: list[Context] | None = None
    trace_id: str
    latency_ms: float
    generator: str
    usage: TokenUsage


class RateLimiter:
    """Sliding window: at most ``limit`` requests per client in the last ``window`` seconds.

    In memory and per process, which is fine for one instance; several instances would need a
    shared store such as Redis. ``clock`` is injectable so tests don't have to sleep.
    """

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, client: str) -> int | None:
        """Records a call. Returns None if allowed, else the seconds until the next call is allowed."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            calls = self._calls[client]
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if len(calls) >= self.limit:
                return max(1, math.ceil(self.window - (now - calls[0])))
            calls.append(now)
            return None


def to_response(result: ChatResult, include_contexts: bool) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        status=result.status,
        citations=[Citation(id=s.id, title=s.title) for s in result.citations],
        contexts=[Context(id=s.id, title=s.title, text=s.text) for s in result.contexts] if include_contexts else None,
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
        generator=result.generator,
        usage=TokenUsage(prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens),
    )


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event. JSON never contains a raw newline, so one data line is enough."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunks(text: str) -> Iterator[str]:
    """Splits the answer into word-sized pieces, keeping the spaces, so the client can join them back."""
    yield from re.findall(r"\S+\s*", text)


def create_app(
    pipeline: ChatPipeline | None = None, settings: Settings | None = None, limiter: RateLimiter | None = None
) -> FastAPI:
    """App factory: tests pass their own pipeline or limiter; production builds both from the environment."""
    settings = settings or Settings.from_env()
    pipeline = pipeline or build_pipeline(settings)
    limiter = limiter or RateLimiter(settings.rate_limit, settings.rate_window_s)
    app = FastAPI(title="Belay - Northpeak help assistant", version=__version__)

    def rate_limited(request: Request) -> None:
        # A client may name itself (X-Client-Id); otherwise its address is the key
        client = request.headers.get("x-client-id") or (request.client.host if request.client else "unknown")
        wait = limiter.retry_after(client)
        if wait is not None:
            raise HTTPException(429, "Too many requests, please slow down.", headers={"Retry-After": str(wait)})

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "generator": pipeline.generator.name,
            "retriever": pipeline.retriever.mode,
            "sections": len(pipeline.retriever.sections),
            # Published so clients (and the live API tests) can pace themselves
            "rate_limit": {"limit": limiter.limit, "window_s": limiter.window},
        }

    @app.post("/chat", dependencies=[Depends(rate_limited)])
    def chat(body: ChatRequest) -> ChatResponse:
        return to_response(pipeline.answer(body.question), body.include_contexts)

    @app.post("/chat/stream", dependencies=[Depends(rate_limited)])
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        # The answer is complete and guarded BEFORE the first token goes out: streaming raw model
        # tokens would let a phone number reach the client before the redaction could see it whole.
        response = to_response(pipeline.answer(body.question), body.include_contexts)

        def events() -> Iterator[str]:
            for piece in _chunks(response.answer):
                yield _sse("token", {"text": piece})
            yield _sse("done", response.model_dump())

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def __getattr__(name: str):
    """``uvicorn belay.api:app`` builds the app on first access, so importing this module stays cheap."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
