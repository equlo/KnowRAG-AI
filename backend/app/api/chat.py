"""
chat.py - The streaming question-answering endpoint.

    POST /api/chat   {"question": "...", "history": [...]}

The response is a Server-Sent Events (SSE) stream: a long-lived HTTP response
made of lines like

    data: {"type": "token", "text": "Hello"}

followed by a blank line. The browser reads these events as they arrive and
appends each token to the answer, giving the familiar "typing" effect.

Event types sent, in order:
    sources  - the retrieved chunks (once, before the answer)
    token    - the next piece of answer text (many times)
    notice   - optional informational message (fallback model, truncation)
    done     - final model name + token usage
    error    - something went wrong (replaces `done`)
"""

# `json.dumps` serialises each event dictionary to a JSON string.
import json

# The standard logging module records unexpected errors on the server console.
import logging

# AsyncIterator: type of our async generator.
from collections.abc import AsyncIterator

# Router, dependency injection, and the streaming response class.
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

# Dependency that provides the RAG service.
from app.api.deps import get_rag

# Service type for hints.
from app.rag.pipeline import RAGService

# Request body shape.
from app.schemas import ChatRequest

# A logger named after this module ("app.api.chat").
logger = logging.getLogger(__name__)

# Group these routes under "chat" in the API docs.
router = APIRouter(tags=["chat"])


def _sse(event: dict) -> str:
    """Format one event in SSE wire format: 'data: <json>' + blank line."""
    # ensure_ascii=False keeps non-English characters readable (and smaller).
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(body: ChatRequest, rag: RAGService = Depends(get_rag)) -> StreamingResponse:
    """Answer a question about the uploaded documents, streaming the reply."""
    # Convert the validated history models into simple (role, text) pairs.
    history = [(turn.role, turn.content) for turn in body.history]

    # An async generator: each `yield` sends one chunk of the HTTP response.
    async def event_stream() -> AsyncIterator[str]:
        # Catch unexpected errors so the browser gets a clean "error" event
        # instead of a connection that silently dies.
        try:
            # Run the whole RAG pipeline and forward each event.
            async for event in rag.answer(body.question, history, body.top_k):
                yield _sse(event)
        # Exception (not BaseException) - so client disconnects still cancel normally.
        except Exception:
            # Log the full stack trace for the developer.
            logger.exception("Chat request failed")
            # Send a generic message to the user (no internal details leaked).
            yield _sse({"type": "error", "message": "Internal server error while answering."})

    # Wrap the generator in a streaming HTTP response.
    return StreamingResponse(
        event_stream(),
        # The MIME type browsers and proxies recognise as SSE.
        media_type="text/event-stream",
        # Disable caching, and tell nginx not to buffer (buffering would delay tokens).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
