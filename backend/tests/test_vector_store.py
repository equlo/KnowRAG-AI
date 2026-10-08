"""Unit tests for the SQLite + NumPy vector store and the offline embedder."""

# pytest for `raises`.
import pytest

# The embedder and the store under test.
from app.rag.embeddings import HashEmbedder
from app.rag.vector_store import ChunkInput, VectorStore


def _store_with_docs(tmp_path):
    """Helper: a store holding two small documents."""
    # The offline embedder.
    embedder = HashEmbedder()
    # A store in the temporary folder.
    store = VectorStore(tmp_path / "test.db", embedder.name)
    # Document 1: about cats.
    cat_chunks = [ChunkInput("Cats are small domesticated felines that purr.", page=1)]
    store.add_document("cats.txt", cat_chunks, embedder.embed_documents([c.text for c in cat_chunks]))
    # Document 2: about rockets.
    rocket_chunks = [ChunkInput("Rockets use liquid fuel engines to reach orbit.", page=None)]
    store.add_document("rockets.txt", rocket_chunks, embedder.embed_documents([c.text for c in rocket_chunks]))
    # Return both for the tests.
    return embedder, store


def test_search_ranks_the_relevant_chunk_first(tmp_path):
    # Build the store.
    embedder, store = _store_with_docs(tmp_path)
    # Ask about rockets.
    hits = store.search(embedder.embed_query("How do rockets reach orbit?"), top_k=2)
    # The rocket chunk must rank first and score higher than the cat chunk.
    assert hits[0].filename == "rockets.txt"
    assert hits[0].score > hits[1].score


def test_search_on_empty_store_returns_nothing(tmp_path):
    # A fresh store with no documents.
    embedder = HashEmbedder()
    store = VectorStore(tmp_path / "empty.db", embedder.name)
    # Searching should simply return an empty list.
    assert store.search(embedder.embed_query("anything"), top_k=5) == []


def test_delete_removes_document_and_chunks(tmp_path):
    # Build the store.
    embedder, store = _store_with_docs(tmp_path)
    # Find the cats document's ID.
    cats = next(d for d in store.list_documents() if d.filename == "cats.txt")
    # Delete it.
    assert store.delete_document(cats.id) is True
    # Only one document is left...
    assert store.count_documents() == 1
    # ...and its chunks no longer appear in search results.
    hits = store.search(embedder.embed_query("cats purr"), top_k=5)
    assert all(hit.filename != "cats.txt" for hit in hits)
    # Deleting again reports "not found".
    assert store.delete_document(cats.id) is False


def test_switching_embedder_is_rejected(tmp_path):
    # Build a store with vectors from the default HashEmbedder (1024 dimensions).
    _store_with_docs(tmp_path)
    # Re-opening the same database with a different embedder must fail loudly.
    with pytest.raises(RuntimeError):
        VectorStore(tmp_path / "test.db", HashEmbedder(dimensions=256).name)
