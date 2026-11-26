"""Shared fixtures: the API client, and the switch for tests that need a real model.

By default every test runs offline against the app in-process. Two environment variables change
where the tests point, without changing a test:

- ``BELAY_BASE_URL``: run against a started server (CI starts uvicorn and sets it)
- ``BELAY_GENERATOR=ollama``: the app answers with a local model (weekly model run)

Tests marked ``model`` need Ollama for judging or embeddings and run only with ``--model``.
"""

from __future__ import annotations

import os

import pytest

from qa.client import ChatApiClient


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model", action="store_true", help="also run tests that need a running Ollama")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--model"):
        return
    skip = pytest.mark.skip(reason="needs Ollama: run with --model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def api() -> ChatApiClient:
    """One client for the whole run: in-process app, or the server at BELAY_BASE_URL."""
    client = ChatApiClient.from_env()
    yield client
    client.close()


@pytest.fixture(scope="session")
def generator_name() -> str:
    return os.environ.get("BELAY_GENERATOR", "extractive")
