"""
vector_store.py - Step 4 of RAG: STORE chunks + vectors and SEARCH them.

A "vector database" does two jobs:
  1. persist each chunk's text, metadata and embedding vector;
  2. given a query vector, return the chunks whose vectors are most similar.

We build a minimal one from two standard tools so every step is visible:
  * SQLite (built into Python) stores documents, chunks and vectors on disk;
  * NumPy computes cosine similarity against ALL vectors in one matrix
    multiplication ("brute-force" or "exact" search).

Brute force is perfectly fast up to a few hundred thousand chunks. Beyond
that you would switch to a dedicated vector DB (pgvector, Qdrant, Chroma...)
that uses approximate-nearest-neighbour indexes - the interface stays the same.
"""

# `sqlite3` is Python's built-in driver for the SQLite file-based database.
import sqlite3

# A lock stops two web requests from modifying the database at the same moment.
import threading

# `uuid4` generates random unique IDs for documents.
import uuid

# `dataclass` creates simple record classes.
from dataclasses import dataclass

# Used to timestamp uploads in UTC.
from datetime import datetime, timezone

# `Path` represents the database file location.
from pathlib import Path

# NumPy holds the vectors and does the similarity math.
import numpy as np


# SQL that creates our three tables the first time the database is opened.
_SCHEMA = """
-- Key/value settings for the store itself (e.g. which embedder built the vectors).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- One row per uploaded file.
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    num_chunks     INTEGER NOT NULL,
    num_characters INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
-- One row per chunk; deleting a document deletes its chunks (ON DELETE CASCADE).
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page        INTEGER,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
-- Speeds up "find all chunks of document X".
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


# A chunk ready to be stored (text + optional page number).
@dataclass(frozen=True)
class ChunkInput:
    # Chunk text.
    text: str
    # Page number (PDF only).
    page: int | None


# A document row read back from the database.
@dataclass(frozen=True)
class StoredDocument:
    # Unique document ID.
    id: str
    # Original file name.
    filename: str
    # Number of chunks.
    num_chunks: int
    # Number of characters of extracted text.
    num_characters: int
    # ISO-8601 upload timestamp.
    created_at: str


# One search result.
@dataclass(frozen=True)
class SearchHit:
    # ID of the document the chunk belongs to.
    document_id: str
    # That document's file name.
    filename: str
    # Page number (PDF only).
    page: int | None
    # The chunk's text.
    text: str
    # Cosine similarity with the query (higher = more relevant).
    score: float


class VectorStore:
    """SQLite-backed storage with exact (brute-force) cosine-similarity search."""

    def __init__(self, db_path: Path, embedder_name: str):
        # Make sure the folder for the database file exists.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Open (or create) the database file. check_same_thread=False allows
        # FastAPI's worker threads to share it; our lock keeps access safe.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # Return rows that can be accessed by column name: row["filename"].
        self._conn.row_factory = sqlite3.Row
        # SQLite ignores foreign keys unless asked; we need them for cascading deletes.
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Create the tables if they do not exist yet.
        self._conn.executescript(_SCHEMA)
        # A re-entrant lock serialises all database access.
        self._lock = threading.RLock()
        # In-memory search index: chunk IDs and the matching vector matrix.
        # None means "not loaded yet / out of date".
        self._index_ids: np.ndarray | None = None
        self._index_matrix: np.ndarray | None = None
        # Refuse to mix vectors from different embedding models (they are incompatible).
        self._check_embedder(embedder_name)

    # ------------------------------------------------------------------ setup

    def _check_embedder(self, embedder_name: str) -> None:
        """Remember which embedder built the vectors and detect switches."""
        # Hold the lock for the whole check.
        with self._lock:
            # Look up the embedder recorded in the meta table (if any).
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
            # Count stored chunks.
            chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            # If vectors exist and were made by a DIFFERENT model, searching would
            # compare apples with oranges - stop with a helpful message.
            if row is not None and row["value"] != embedder_name and chunk_count > 0:
                raise RuntimeError(
                    f"The index was built with '{row['value']}' but the current embedder is "
                    f"'{embedder_name}'. Delete the data folder (or switch EMBEDDING_PROVIDER "
                    "back) and re-upload your documents."
                )
            # Record the current embedder (insert or overwrite).
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder', ?)", (embedder_name,)
            )
            # Save the change to disk.
            self._conn.commit()

    # ------------------------------------------------------------- documents

    def add_document(self, filename: str, chunks: list[ChunkInput], embeddings: np.ndarray) -> StoredDocument:
        """Insert one document and all of its chunks + vectors in a single transaction."""
        # Every chunk needs exactly one vector.
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        # Build the document record with a fresh random ID and the current UTC time.
        document = StoredDocument(
            id=uuid.uuid4().hex,
            filename=filename,
            num_chunks=len(chunks),
            num_characters=sum(len(c.text) for c in chunks),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        # Lock, then use the connection as a context manager: it COMMITs on success
        # and ROLLs BACK on error, so we never store half a document.
        with self._lock, self._conn:
            # Insert the document row.
            self._conn.execute(
                "INSERT INTO documents (id, filename, num_chunks, num_characters, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (document.id, document.filename, document.num_chunks, document.num_characters, document.created_at),
            )
            # Insert all chunk rows at once. Vectors are stored as raw float32 bytes.
            self._conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (document.id, i, chunk.page, chunk.text, vector.astype(np.float32).tobytes())
                    for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ],
            )
            # The in-memory index is now stale; it will be rebuilt on the next search.
            self._invalidate_index()
        # Return the stored document's metadata.
        return document

    def list_documents(self) -> list[StoredDocument]:
        """Return all documents, newest first."""
        # Lock while reading.
        with self._lock:
            # Fetch every document row ordered by upload time.
            rows = self._conn.execute(
                "SELECT id, filename, num_chunks, num_characters, created_at FROM documents ORDER BY created_at DESC"
            ).fetchall()
        # Convert each sqlite Row into a StoredDocument (column names match field names).
        return [StoredDocument(**dict(row)) for row in rows]

    def count_documents(self) -> int:
        """Return how many documents are indexed."""
        # Lock while reading.
        with self._lock:
            # COUNT(*) returns a single number.
            return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and (via CASCADE) its chunks. Returns False if not found."""
        # Lock and run inside a transaction.
        with self._lock, self._conn:
            # Delete the document row; chunks are removed automatically.
            cursor = self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            # The search index must be rebuilt without the deleted chunks.
            self._invalidate_index()
        # rowcount tells us how many rows were deleted (0 = unknown ID).
        return cursor.rowcount > 0

    # ---------------------------------------------------------------- search

    def _invalidate_index(self) -> None:
        """Forget the cached in-memory index (called after every write)."""
        # Drop the cached IDs.
        self._index_ids = None
        # Drop the cached vector matrix.
        self._index_matrix = None

    def _load_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all chunk vectors into one NumPy matrix (cached until the next write)."""
        # Reuse the cache if it is still valid.
        if self._index_ids is not None and self._index_matrix is not None:
            return self._index_ids, self._index_matrix
        # Read every chunk ID and its vector bytes.
        rows = self._conn.execute("SELECT id, embedding FROM chunks").fetchall()
        # An empty store gives empty arrays.
        if not rows:
            self._index_ids = np.empty(0, dtype=np.int64)
            self._index_matrix = np.empty((0, 0), dtype=np.float32)
        else:
            # Collect the chunk IDs in an integer array.
            self._index_ids = np.array([row["id"] for row in rows], dtype=np.int64)
            # Decode each BLOB back into a float32 vector and stack them into rows.
            self._index_matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        # Return the freshly built index.
        return self._index_ids, self._index_matrix

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Return the `top_k` chunks most similar to `query_vector`."""
        # Lock so a concurrent upload cannot change the index mid-search.
        with self._lock:
            # Get (or build) the in-memory index.
            ids, matrix = self._load_index()
            # Nothing indexed yet -> no results.
            if len(ids) == 0:
                return []
            # Cosine similarity of the query with EVERY chunk in one operation:
            # (num_chunks x dims) @ (dims,) -> (num_chunks,). Vectors are unit
            # length, so the dot product IS the cosine similarity.
            scores = matrix @ query_vector.astype(np.float32)
            # We cannot return more results than there are chunks.
            k = min(top_k, len(scores))
            # argpartition finds the k best positions in linear time (unordered)...
            best = np.argpartition(-scores, k - 1)[:k]
            # ...then we sort just those k by descending score.
            best = best[np.argsort(-scores[best])]
            # Map matrix positions back to database chunk IDs (as plain Python ints).
            best_ids = [int(ids[i]) for i in best]
            # Remember each chunk's score by ID.
            score_by_id = {int(ids[i]): float(scores[i]) for i in best}
            # Build "?, ?, ?" placeholders for a parameterised SQL IN (...) query.
            placeholders = ", ".join("?" for _ in best_ids)
            # Fetch the text and metadata of the winning chunks, joined with their document.
            rows = self._conn.execute(
                f"SELECT c.id, c.document_id, c.page, c.text, d.filename "
                f"FROM chunks c JOIN documents d ON d.id = c.document_id "
                f"WHERE c.id IN ({placeholders})",
                best_ids,
            ).fetchall()
        # SQL does not preserve our ranking, so index rows by chunk ID...
        row_by_id = {row["id"]: row for row in rows}
        # ...and rebuild the results in best-first order.
        return [
            SearchHit(
                document_id=row_by_id[cid]["document_id"],
                filename=row_by_id[cid]["filename"],
                page=row_by_id[cid]["page"],
                text=row_by_id[cid]["text"],
                score=score_by_id[cid],
            )
            for cid in best_ids
        ]

    def close(self) -> None:
        """Close the database connection (called when the server shuts down)."""
        # Lock so no request is mid-query while we close.
        with self._lock:
            # Release the file handle.
            self._conn.close()
