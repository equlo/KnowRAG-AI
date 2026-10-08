"""
pipeline.py - The conductor that wires every RAG step together.

    INGEST  (upload time):  file -> load -> chunk -> embed -> store
    ANSWER  (question time): question -> (rewrite) -> embed -> search
                             -> build prompt -> Claude streams the answer

The web layer (app/api/*) only calls methods on `RAGService`; it never needs to
know how loading, chunking, embedding or prompting work.
"""

# AsyncIterator: the return type of an async generator (a function that `yield`s).
from collections.abc import AsyncIterator

# `asdict` converts a dataclass instance into a plain dictionary.
from dataclasses import asdict

# Any is used for loosely-typed event dictionaries.
from typing import Any

# Runs blocking (CPU-heavy) code in a worker thread so the async server stays responsive.
from fastapi.concurrency import run_in_threadpool

# Settings (chunk size, top_k, rewrite flag...).
from app.config import Settings

# The chunking function.
from app.rag.chunker import split_text

# The embedder interface.
from app.rag.embeddings import Embedder

# The LLM interface.
from app.rag.llm import LLM

# File loading and its error type.
from app.rag.loaders import UnsupportedFileError, load_document

# Prompt text and prompt builders.
from app.rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    build_answer_message,
    build_rewrite_message,
)

# Storage types and the vector store itself.
from app.rag.vector_store import ChunkInput, SearchHit, StoredDocument, VectorStore


class RAGService:
    """High-level operations: ingest, list, delete, search, answer."""

    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, llm: LLM):
        # Configuration values.
        self.settings = settings
        # Turns text into vectors.
        self.embedder = embedder
        # Persists chunks and searches them.
        self.store = store
        # Generates answers (Claude in production, a fake in tests).
        self.llm = llm

    # ------------------------------------------------------------- ingestion

    def ingest(self, filename: str, data: bytes) -> StoredDocument:
        """Load, chunk, embed and store one uploaded file."""
        # 1. LOAD: bytes -> list of pages of text.
        pages = load_document(filename, data)
        # 2. CHUNK: split every page; each chunk remembers its page number.
        chunks = [
            ChunkInput(text=chunk, page=page.page)
            for page in pages
            for chunk in split_text(page.text, self.settings.chunk_size, self.settings.chunk_overlap)
        ]
        # A file of pure whitespace/punctuation could produce zero chunks.
        if not chunks:
            raise UnsupportedFileError(f"'{filename}' contains no indexable text.")
        # 3. EMBED: one vector per chunk, computed in a single batch for speed.
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        # 4. STORE: save the document, chunks and vectors together.
        return self.store.add_document(filename, chunks, embeddings)

    def list_documents(self) -> list[StoredDocument]:
        """All indexed documents."""
        # Delegate to the store.
        return self.store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        """Remove a document; True if it existed."""
        # Delegate to the store.
        return self.store.delete_document(document_id)

    # ------------------------------------------------------------- retrieval

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Semantic search: embed the query and find the closest chunks."""
        # Embed the query with the SAME model used for the chunks.
        query_vector = self.embedder.embed_query(query)
        # Return the best matches (request override or the configured default).
        return self.store.search(query_vector, top_k or self.settings.top_k)

    async def _standalone_query(self, history: list[tuple[str, str]], question: str) -> str:
        """Rewrite a follow-up like 'and the second one?' into a self-contained query."""
        # First question of a chat, or feature switched off -> use the question as-is.
        if not history or not self.settings.query_rewrite:
            return question
        # Ask Claude to rewrite, showing only the last 6 turns to keep it small and fast.
        rewritten = await self.llm.complete(
            REWRITE_SYSTEM_PROMPT,
            [{"role": "user", "content": build_rewrite_message(history[-6:], question)}],
        )
        # If the rewrite failed (empty string), fall back to the original question.
        return rewritten or question

    # ------------------------------------------------------------ generation

    async def answer(
        self, question: str, history: list[tuple[str, str]], top_k: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the full RAG loop and yield events for the browser."""
        # A. Make the search query standalone (only matters for follow-ups).
        search_query = await self._standalone_query(history, question)
        # B. RETRIEVE: embedding + search are CPU work, so run them in a thread.
        hits = await run_in_threadpool(self.search, search_query, top_k)
        # C. Send the sources to the UI first, so they appear before the answer.
        yield {
            "type": "sources",
            "query": search_query,
            "sources": [{"number": n, **asdict(hit)} for n, hit in enumerate(hits, start=1)],
        }
        # D. AUGMENT: build the message list for Claude.
        messages = _history_to_messages(history)
        # The new user turn = numbered sources + the original question.
        messages.append({"role": "user", "content": build_answer_message(question, hits)})
        # E. GENERATE: stream Claude's answer events straight through to the caller.
        async for event in self.llm.stream_answer(ANSWER_SYSTEM_PROMPT, messages):
            yield event


def _history_to_messages(history: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Convert (role, text) pairs into Messages API format, cleaning edge cases."""
    # Keep only turns that actually contain text (the API rejects empty content).
    messages = [{"role": role, "content": text} for role, text in history if text.strip()]
    # The API requires the conversation to start with a user turn.
    while messages and messages[0]["role"] != "user":
        # Drop any leading assistant turns.
        messages.pop(0)
    # Return the cleaned list.
    return messages
