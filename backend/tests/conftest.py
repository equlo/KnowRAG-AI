"""
conftest.py - Shared pytest fixtures.

The tests never call the real Claude API or download a model:
  * `HashEmbedder` gives deterministic offline vectors;
  * `FakeLLM` returns canned text and records what it was sent.
"""

# Async generator type for the fake LLM.
from collections.abc import AsyncIterator

# Loose typing for message dictionaries.
from typing import Any

# The pytest framework.
import pytest

# FastAPI's in-process test client (no real network server needed).
from fastapi.testclient import TestClient

# The application factory and its settings.
from app.config import Settings
from app.main import create_app

# The offline embedder.
from app.rag.embeddings import HashEmbedder


class FakeLLM:
    """Stands in for ClaudeLLM; records calls and streams a fixed answer."""

    def __init__(self) -> None:
        # Every (system, messages) pair passed to stream_answer.
        self.answer_calls: list[tuple[str, list[dict[str, Any]]]] = []
        # Every (system, messages) pair passed to complete.
        self.complete_calls: list[tuple[str, list[dict[str, Any]]]] = []
        # What complete() returns (the "rewritten query").
        self.rewrite_result = "rewritten standalone query"

    async def stream_answer(self, system: str, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        # Record the call for later assertions.
        self.answer_calls.append((system, messages))
        # Stream an answer in two tokens, then a "done" event.
        yield {"type": "token", "text": "The answer "}
        yield {"type": "token", "text": "is 42 [1]."}
        yield {"type": "done", "model": "fake-model", "input_tokens": 10, "output_tokens": 5}

    async def complete(self, system: str, messages: list[dict[str, Any]]) -> str:
        # Record the call and return the canned rewrite.
        self.complete_calls.append((system, messages))
        return self.rewrite_result


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Settings pointing at a fresh temporary folder, with small chunks and a 1 MB upload limit.
    return Settings(
        data_dir=tmp_path, embedding_provider="hash", chunk_size=200, chunk_overlap=50, top_k=3, max_upload_mb=1
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    # A new fake per test, so recorded calls never leak between tests.
    return FakeLLM()


@pytest.fixture
def client(settings, fake_llm):
    # Build the app with the offline embedder and the fake LLM.
    app = create_app(settings=settings, embedder=HashEmbedder(), llm=fake_llm)
    # Using TestClient as a context manager runs the lifespan (startup/shutdown).
    with TestClient(app) as test_client:
        yield test_client
