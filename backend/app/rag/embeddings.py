"""
embeddings.py - Step 3 of RAG: turn TEXT into VECTORS (lists of numbers).

An embedding model maps a piece of text to a point in a high-dimensional space
so that texts with similar MEANING land close together. "How do I reset my
password?" and "Steps to change a forgotten login" end up near each other even
though they share almost no words. That closeness is what makes semantic
search possible.

Two interchangeable providers are implemented:
  * FastEmbedEmbedder - a real neural embedding model (BAAI/bge-small-en-v1.5)
    that runs locally on the CPU. Best quality; downloads ~70 MB on first use.
  * HashEmbedder      - a dependency-free "bag of words" vector built with
    feature hashing. No download and no network, but it only matches words,
    not meaning. Perfect for offline demos and unit tests.

Both return vectors that are L2-NORMALISED (length 1). For unit vectors the dot
product equals the cosine similarity, which keeps the search code trivial.
"""

# `hashlib` provides stable hash functions (used by the HashEmbedder).
import hashlib

# `re` finds word tokens in text.
import re

# `Path` is used for the model download folder.
from pathlib import Path

# `Protocol` describes an interface: "anything with these methods is an Embedder".
from typing import Protocol

# NumPy stores vectors as efficient float arrays.
import numpy as np

# Our settings object (provider choice, model name, data folder).
from app.config import Settings


class Embedder(Protocol):
    """The interface every embedding provider must implement."""

    # A human-readable identifier stored in the database, e.g. "fastembed:BAAI/bge-small-en-v1.5".
    name: str

    # Embed many document chunks at once -> matrix of shape (len(texts), dimensions).
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    # Embed one search query -> vector of shape (dimensions,).
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row of `matrix` to length 1 (so dot product == cosine similarity)."""
    # Compute each row's Euclidean length; keepdims=True keeps shape (rows, 1) for broadcasting.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid dividing by zero for an all-zero vector (e.g. text with no words).
    norms[norms == 0] = 1.0
    # Divide each row by its length.
    return matrix / norms


class FastEmbedEmbedder:
    """Semantic embeddings from a small open-source model, run locally via ONNX."""

    def __init__(self, model_name: str, cache_dir: Path):
        # Import here (not at the top) so the "hash" provider works even if
        # fastembed is not installed.
        from fastembed import TextEmbedding

        # Make sure the folder for downloaded model files exists.
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Load (and on first run, download) the model into the cache folder.
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        # Remember which model produced our vectors.
        self.name = f"fastembed:{model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # `passage_embed` applies the model's recommended formatting for documents
        # and yields one vector per text; list(...) collects them.
        vectors = list(self._model.passage_embed(texts))
        # Stack into a float32 matrix (float32 halves memory vs float64) and normalise.
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        # `query_embed` adds the query instruction that BGE models were trained with
        # ("Represent this sentence for searching relevant passages: ...").
        vector = next(iter(self._model.query_embed(text)))
        # Turn the vector into a 1-row matrix, normalise it, and return the single row.
        return _normalize(np.asarray(vector, dtype=np.float32)[None, :])[0]


# Regex that matches "words": runs of letters/digits (Unicode-aware).
_TOKEN = re.compile(r"\w+")
# Very common words carry little meaning, so the HashEmbedder ignores them.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or "
    "that the this to was were what when where which who why will with you your".split()
)


def _normalize_word(word: str) -> str:
    """A crude 'stemmer': fold simple plurals so 'vectors' matches 'vector'."""
    # Strip one trailing "s" from longer words, but keep words like "class" or "glass".
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    # Everything else is used unchanged.
    return word


class HashEmbedder:
    """Offline keyword vectors using the 'hashing trick' (no model download)."""

    def __init__(self, dimensions: int = 1024):
        # Length of every vector produced.
        self.dimensions = dimensions
        # Identifier stored in the database.
        self.name = f"hash:{dimensions}"

    def _vector(self, text: str) -> np.ndarray:
        """Build one un-normalised vector for `text`."""
        # Lower-case the text, split it into words, drop stop-words, fold plurals.
        words = [_normalize_word(w) for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        # Features = single words + adjacent word pairs ("machine learning").
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        # Start from an all-zero vector.
        vector = np.zeros(self.dimensions, dtype=np.float32)
        # Add every feature into one "bucket" of the vector.
        for feature in features:
            # Hash the feature to 8 stable pseudo-random bytes.
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            # Use the first 4 bytes to choose a bucket index.
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            # Use one bit of the 5th byte as a +1/-1 sign, which reduces collision bias.
            sign = 1.0 if digest[4] & 1 else -1.0
            # Accumulate the feature into its bucket.
            vector[index] += sign
        # Return the raw vector (normalised by the caller).
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # Build one vector per text, stack them into a matrix, then normalise.
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        # Same procedure as for documents, for a single string.
        return _normalize(self._vector(text)[None, :])[0]


def create_embedder(settings: Settings) -> Embedder:
    """Factory: build the embedder chosen in the settings."""
    # Offline keyword embedder.
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    # Default: semantic FastEmbed model, cached under <data_dir>/models.
    return FastEmbedEmbedder(settings.embedding_model, settings.data_dir / "models")
