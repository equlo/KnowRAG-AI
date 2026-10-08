"""
schemas.py - The *shapes* of data that travel over HTTP.

FastAPI uses these Pydantic models to:
  * validate incoming JSON (bad input -> automatic 422 error),
  * serialise outgoing Python objects to JSON,
  * generate interactive API docs at http://localhost:8000/docs.
"""

# `Literal` restricts a value to specific strings (here: "user" or "assistant").
from typing import Literal

# `BaseModel` is the Pydantic base class; `Field` adds validation constraints.
from pydantic import BaseModel, Field


# Metadata about one uploaded document, returned by the /documents endpoints.
class DocumentOut(BaseModel):
    # Unique ID generated when the document was uploaded.
    id: str
    # Original file name, e.g. "handbook.pdf".
    filename: str
    # How many chunks the document was split into.
    num_chunks: int
    # Total number of characters of extracted text.
    num_characters: int
    # Upload time as an ISO-8601 string, e.g. "2026-10-08T12:00:00+00:00".
    created_at: str


# One retrieved chunk plus its similarity score (used by /search and /chat).
class ChunkOut(BaseModel):
    # Citation number shown to the user and to Claude ([1], [2], ...).
    number: int
    # Which document the chunk came from.
    document_id: str
    # The document's file name (so the UI can display it).
    filename: str
    # Page number for PDFs; None for formats without pages.
    page: int | None
    # The chunk's text.
    text: str
    # Cosine similarity between the question and the chunk (-1..1, higher = closer).
    score: float


# Request body for POST /api/search.
class SearchRequest(BaseModel):
    # The search text; must contain at least one character.
    query: str = Field(min_length=1, max_length=4000)
    # Optional override of how many chunks to return.
    top_k: int | None = Field(default=None, ge=1, le=20)


# One previous message in the conversation, sent back by the browser.
class ChatTurn(BaseModel):
    # Who said it.
    role: Literal["user", "assistant"]
    # What they said (plain text / markdown).
    content: str = Field(max_length=50_000)


# Request body for POST /api/chat.
class ChatRequest(BaseModel):
    # The new question the user just typed.
    question: str = Field(min_length=1, max_length=4000)
    # Earlier turns of the conversation (empty list for the first question).
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    # Optional override of how many chunks to retrieve.
    top_k: int | None = Field(default=None, ge=1, le=20)


# Response body for GET /api/health.
class HealthOut(BaseModel):
    # Always "ok" if the server is running.
    status: str
    # The Claude model in use.
    model: str
    # The embedding provider/model in use.
    embedding: str
    # How many documents are currently indexed.
    documents: int
