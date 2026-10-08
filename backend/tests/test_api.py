"""End-to-end tests of the HTTP API (with a fake LLM and offline embeddings)."""

# Used to parse the SSE "data: {...}" lines.
import json

# Used to build a small in-memory .docx file.
import io

# python-docx, to create a Word document for the upload test.
from docx import Document


def _upload(client, name: str, content: bytes):
    """Helper: upload one file and return the HTTP response."""
    # Send a multipart/form-data request with a field called "file".
    return client.post("/api/documents", files={"file": (name, content)})


def _read_sse(response) -> list[dict]:
    """Helper: turn an SSE response body into a list of event dictionaries."""
    # Every event line starts with "data: "; strip that prefix and parse the JSON.
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_health(client):
    # The health endpoint reports status and an empty index.
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 0


def test_upload_list_and_delete(client):
    # Upload a text file.
    response = _upload(client, "notes.txt", b"The launch window opens on Tuesday at 9am.")
    # It should be created and report at least one chunk.
    assert response.status_code == 201
    document = response.json()
    assert document["filename"] == "notes.txt"
    assert document["num_chunks"] >= 1
    # It appears in the list.
    assert [d["id"] for d in client.get("/api/documents").json()] == [document["id"]]
    # Delete it.
    assert client.delete(f"/api/documents/{document['id']}").status_code == 204
    # The list is empty again.
    assert client.get("/api/documents").json() == []
    # Deleting an unknown ID gives 404.
    assert client.delete(f"/api/documents/{document['id']}").status_code == 404


def test_upload_docx(client):
    # Build a Word document in memory.
    doc = Document()
    doc.add_paragraph("Quarterly revenue grew by 12 percent.")
    buffer = io.BytesIO()
    doc.save(buffer)
    # Upload it and check that it was indexed.
    response = _upload(client, "report.docx", buffer.getvalue())
    assert response.status_code == 201
    assert response.json()["num_chunks"] == 1


def test_unsupported_and_empty_files_are_rejected(client):
    # Unknown extension -> 400.
    assert _upload(client, "image.png", b"\x89PNG").status_code == 400
    # A text file with only whitespace -> 400.
    assert _upload(client, "blank.txt", b"   \n\n  ").status_code == 400


def test_oversized_upload_is_rejected(client, settings):
    # One byte more than the limit.
    too_big = b"a" * (settings.max_upload_mb * 1024 * 1024 + 1)
    # The API must answer 413.
    assert _upload(client, "big.txt", too_big).status_code == 413


def test_search_returns_ranked_chunks(client):
    # Index two unrelated documents.
    _upload(client, "cats.txt", b"Cats are small felines that purr and sleep a lot.")
    _upload(client, "rockets.txt", b"Rockets burn liquid fuel to reach orbit around Earth.")
    # Search for rocket-related text.
    hits = client.post("/api/search", json={"query": "rocket fuel orbit", "top_k": 2}).json()
    # The rocket document ranks first, results are numbered from 1.
    assert hits[0]["filename"] == "rockets.txt"
    assert hits[0]["number"] == 1


def test_chat_streams_sources_then_tokens(client, fake_llm):
    # Index a document.
    _upload(client, "facts.txt", b"The answer to life, the universe and everything is 42.")
    # Ask a question.
    response = client.post("/api/chat", json={"question": "What is the answer to everything?"})
    # The response is an SSE stream.
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Parse the events.
    events = _read_sse(response)
    # The order is: sources, tokens..., done.
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    # The sources include our document.
    assert events[0]["sources"][0]["filename"] == "facts.txt"
    # The answer text is the concatenation of the tokens.
    assert "".join(e["text"] for e in events if e["type"] == "token") == "The answer is 42 [1]."
    # The LLM received the retrieved text inside the final user message.
    _system, messages = fake_llm.answer_calls[0]
    assert "<sources>" in messages[-1]["content"]
    assert "42" in messages[-1]["content"]
    # No history -> no query rewriting call.
    assert fake_llm.complete_calls == []


def test_follow_up_questions_are_rewritten(client, fake_llm):
    # Index a document.
    _upload(client, "facts.txt", b"Mars has two moons named Phobos and Deimos.")
    # Ask a follow-up question with history.
    history = [
        {"role": "user", "content": "How many moons does Mars have?"},
        {"role": "assistant", "content": "Mars has two moons [1]."},
    ]
    events = _read_sse(client.post("/api/chat", json={"question": "What are their names?", "history": history}))
    # The rewriter was called once and its output was used as the search query.
    assert len(fake_llm.complete_calls) == 1
    assert events[0]["query"] == "rewritten standalone query"
    # The history was forwarded to the answering LLM before the new question.
    _system, messages = fake_llm.answer_calls[0]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


def test_chat_validates_input(client):
    # An empty question is rejected by Pydantic with 422 Unprocessable Entity.
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
