"""API tests: the HTTP contract of the assistant, independent of how good its answers are.

Run in-process by default; with BELAY_BASE_URL set, the same tests hit a real server (CI starts
one with uvicorn), which also covers serialisation, headers and the SSE framing over a socket.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from qa.client import ChatApiClient, Reply

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "chat_response.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def schema_errors(body: dict) -> list[str]:
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in VALIDATOR.iter_errors(body)]


def unique_client() -> str:
    """A fresh rate-limit identity, so one test's calls never count against another's."""
    return f"api-test-{uuid.uuid4().hex[:12]}"


class TestHealth:
    def test_health_reports_ok_and_configuration(self, api: ChatApiClient):
        health = api.health()
        assert health["status"] == "ok"
        assert health["retriever"] in {"bm25", "vector", "hybrid"}
        assert health["sections"] > 0
        assert health["rate_limit"]["limit"] >= 0


class TestChatContract:
    @pytest.mark.parametrize(
        "question",
        ["How much is a day pass?", "Do you sell protein shakes?", "What is the staff door code?"],
        ids=["answered", "not_found", "refused"],
    )
    def test_every_status_matches_the_json_schema(self, api: ChatApiClient, question: str):
        response = api.post_chat({"question": question, "include_contexts": True}, client_id=unique_client())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert schema_errors(response.json()) == []

    def test_contexts_are_left_out_unless_asked_for(self, api: ChatApiClient):
        body = api.post_chat({"question": "How much is a day pass?"}, client_id=unique_client()).json()
        assert body["contexts"] is None

    def test_answered_reply_cites_a_section_it_was_given(self, api: ChatApiClient):
        reply = api.chat("How much is a day pass?", client_id=unique_client())
        assert reply.status == "answered"
        assert set(reply.cited_ids) <= {c.id for c in reply.contexts or []}

    def test_each_request_gets_its_own_trace_id(self, api: ChatApiClient):
        ids = {api.chat("What time is last entry?", client_id=unique_client()).trace_id for _ in range(3)}
        assert len(ids) == 3

    def test_question_is_trimmed(self, api: ChatApiClient):
        padded = api.chat("   How much is a day pass?   ", client_id=unique_client())
        plain = api.chat("How much is a day pass?", client_id=unique_client())
        assert padded.answer == plain.answer


class TestInvalidRequests:
    @pytest.mark.parametrize(
        ("body", "field"),
        [
            ({}, "question"),
            ({"question": ""}, "question"),
            ({"question": "   "}, "question"),
            ({"question": "x" * 501}, "question"),
            ({"question": 42}, "question"),
            ({"question": "Hi", "session_id": "has spaces"}, "session_id"),
            ({"question": "Hi", "questoin": "typo"}, "questoin"),
        ],
        ids=["missing", "empty", "blank", "too-long", "not-a-string", "bad-session-id", "unknown-field"],
    )
    def test_bad_body_is_rejected_with_422_naming_the_field(self, api: ChatApiClient, body: dict, field: str):
        response = api.post_chat(body, client_id=unique_client())
        assert response.status_code == 422
        locations = [error["loc"][-1] for error in response.json()["detail"]]
        assert field in locations

    def test_longest_allowed_question_is_accepted(self, api: ChatApiClient):
        response = api.post_chat({"question": "x" * 500}, client_id=unique_client())
        assert response.status_code == 200

    def test_get_on_chat_is_not_allowed(self, api: ChatApiClient):
        assert api._http.get("/chat").status_code == 405

    def test_unknown_path_is_404(self, api: ChatApiClient):
        assert api._http.get("/admin").status_code == 404


class TestStreaming:
    def test_stream_is_server_sent_events(self, api: ChatApiClient):
        result = api.stream("How much is shoe rental?", client_id=unique_client())
        assert result.content_type.startswith("text/event-stream")

    def test_tokens_join_to_the_final_answer(self, api: ChatApiClient):
        result = api.stream("How much is shoe rental?", client_id=unique_client())
        assert len(result.tokens) > 1, "a streamed answer should arrive in several pieces"
        assert "".join(result.tokens) == result.done.answer

    def test_stream_ends_with_one_done_event_that_matches_the_schema(self, api: ChatApiClient):
        result = api.stream("How much is shoe rental?", include_contexts=True, client_id=unique_client())
        assert [event for event, _ in result.events][-1] == "done"
        done = [data for event, data in result.events if event == "done"][0]
        assert schema_errors(done) == []

    def test_stream_and_plain_call_give_the_same_answer(self, api: ChatApiClient):
        streamed = api.stream("Can I use chalk powder?", client_id=unique_client()).done
        plain = api.chat("Can I use chalk powder?", include_contexts=False, client_id=unique_client())
        assert (streamed.status, streamed.answer, streamed.cited_ids) == (plain.status, plain.answer, plain.cited_ids)

    def test_refusal_is_streamed_too(self, api: ChatApiClient):
        done = api.stream("What is the staff door code?", client_id=unique_client()).done
        assert done.status == "refused"
        assert "4417" not in done.answer


class TestRateLimit:
    def test_client_over_the_limit_gets_429_with_retry_after(self, api: ChatApiClient):
        limit = api.health()["rate_limit"]["limit"]
        if limit == 0 or limit > 200:
            pytest.skip(f"rate limit is {limit or 'off'}; too high to reach in a test")
        client = unique_client()
        statuses = [api.post_chat({"question": "Hi"}, client_id=client).status_code for _ in range(limit)]
        assert set(statuses) == {200}
        blocked = api.post_chat({"question": "Hi"}, client_id=client)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1
        assert "too many requests" in blocked.json()["detail"].lower()

    def test_limit_is_per_client(self, api: ChatApiClient):
        limit = api.health()["rate_limit"]["limit"]
        if limit == 0 or limit > 200:
            pytest.skip(f"rate limit is {limit or 'off'}")
        busy = unique_client()
        for _ in range(limit + 1):
            api.post_chat({"question": "Hi"}, client_id=busy)
        assert api.post_chat({"question": "Hi"}, client_id=unique_client()).status_code == 200

    def test_streaming_counts_against_the_same_limit(self, api: ChatApiClient):
        limit = api.health()["rate_limit"]["limit"]
        if limit == 0 or limit > 200:
            pytest.skip(f"rate limit is {limit or 'off'}")
        client = unique_client()
        for _ in range(limit):
            api.post_chat({"question": "Hi"}, client_id=client)
        response = api.post_chat({"question": "Hi"}, client_id=client, path="/chat/stream")
        assert response.status_code == 429


def test_reply_model_rejects_an_unexpected_field():
    """The client contract is strict: a new server field must be added here on purpose."""
    body = {
        "answer": "a",
        "status": "answered",
        "citations": [],
        "trace_id": "0" * 32,
        "latency_ms": 1,
        "generator": "g",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "surprise": 1,
    }
    with pytest.raises(ValueError):
        Reply.model_validate(body)
