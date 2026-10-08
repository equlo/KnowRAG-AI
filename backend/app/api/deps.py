"""
deps.py - FastAPI "dependencies": small functions that hand shared objects to routes.

A route declares `rag: RAGService = Depends(get_rag)` and FastAPI calls `get_rag`
for it on every request. This keeps routes free of global variables and makes
them easy to test.
"""

# `Request` gives access to the running application object.
from fastapi import Request

# The service class we hand out.
from app.rag.pipeline import RAGService


def get_rag(request: Request) -> RAGService:
    """Return the RAGService created at startup (see app/main.py lifespan)."""
    # `app.state` is a place to keep objects for the lifetime of the server.
    return request.app.state.rag
