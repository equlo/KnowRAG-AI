"""
search.py - Retrieval-only endpoint, handy for learning and debugging.

    POST /api/search   {"query": "...", "top_k": 5}  -> ranked chunks with scores

It runs the "R" of RAG without the "G": you can see exactly which chunks
Claude WOULD receive for a question, and how similar each one is.
"""

# `asdict` converts dataclasses to dicts.
from dataclasses import asdict

# Router and dependency injection.
from fastapi import APIRouter, Depends

# Dependency that provides the RAG service.
from app.api.deps import get_rag

# Service type for hints.
from app.rag.pipeline import RAGService

# Request and response shapes.
from app.schemas import ChunkOut, SearchRequest

# Group these routes under "search" in the API docs.
router = APIRouter(tags=["search"])


# Sync `def` so the CPU-bound embedding runs in FastAPI's thread pool.
@router.post("/search", response_model=list[ChunkOut])
def search(body: SearchRequest, rag: RAGService = Depends(get_rag)) -> list[ChunkOut]:
    """Return the chunks most similar to the query."""
    # Run the semantic search.
    hits = rag.search(body.query, body.top_k)
    # Number the results from 1 and convert them to the response model.
    return [ChunkOut(number=n, **asdict(hit)) for n, hit in enumerate(hits, start=1)]
