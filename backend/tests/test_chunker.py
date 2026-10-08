"""Unit tests for the text splitter."""

# pytest provides `raises` for asserting exceptions.
import pytest

# The function under test.
from app.rag.chunker import split_text


def test_short_text_is_a_single_chunk():
    # Text shorter than the chunk size should come back unchanged as one chunk.
    assert split_text("Hello world.", chunk_size=100, chunk_overlap=10) == ["Hello world."]


def test_chunks_respect_the_size_limit():
    # 50 sentences of ~30 characters each = ~1,500 characters of text.
    text = " ".join(f"This is sentence number {i}." for i in range(50))
    # Split into chunks of at most 200 characters.
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    # We expect several chunks...
    assert len(chunks) > 5
    # ...and none may exceed the limit.
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_consecutive_chunks_overlap():
    # Same long text as above.
    text = " ".join(f"This is sentence number {i}." for i in range(50))
    # Split with overlap.
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    # The last sentence of each chunk should reappear at the start of the next one.
    for previous, following in zip(chunks, chunks[1:]):
        last_sentence = previous.split("\n")[-1]
        assert following.startswith(last_sentence)


def test_paragraphs_are_kept_together_when_they_fit():
    # Two short paragraphs separated by a blank line.
    text = "First paragraph here.\n\nSecond paragraph here."
    # They fit in one chunk and are joined by a newline.
    assert split_text(text, chunk_size=100, chunk_overlap=0) == ["First paragraph here.\nSecond paragraph here."]


def test_very_long_words_are_hard_split():
    # A 250-character "word" with no spaces or punctuation.
    text = "x" * 250
    # It must still be cut into pieces of at most 100 characters.
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    # Check sizes and that nothing was lost.
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_overlap_must_be_smaller_than_chunk():
    # An overlap as large as the chunk is a configuration error.
    with pytest.raises(ValueError):
        split_text("anything", chunk_size=100, chunk_overlap=100)
