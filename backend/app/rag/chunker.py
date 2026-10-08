"""
chunker.py - Step 2 of RAG ingestion: split long TEXT into small CHUNKS.

Why chunk at all?
  * An embedding captures the meaning of a short passage far better than of a
    whole book, so small chunks make search more precise.
  * We only send the few most relevant chunks to Claude, which keeps prompts
    short, fast and cheap.

Strategy (a simplified "recursive" splitter):
  1. Break the text into paragraphs.
  2. Break any paragraph that is too long into sentences.
  3. Break any sentence that is STILL too long into fixed-size pieces.
  4. Greedily pack these pieces into chunks of at most `chunk_size` characters,
     repeating the last ~`chunk_overlap` characters at the start of the next
     chunk so that ideas spanning a boundary are not cut in half.
"""

# `re` is Python's regular-expression module, used to find paragraph/sentence breaks.
import re

# A paragraph break is a newline, optional whitespace, then another newline.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# A sentence break is whitespace that FOLLOWS ".", "!" or "?".
# `(?<=...)` is a "look-behind": it checks the punctuation without consuming it,
# so the punctuation stays attached to its sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` characters."""
    # Overlap must be smaller than the chunk, otherwise we would never make progress.
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Steps 1-3: get a flat list of small text "units" (paragraphs/sentences/pieces).
    units = _split_into_units(text, chunk_size)

    # The finished chunks.
    chunks: list[str] = []
    # The units that make up the chunk currently being built.
    current: list[str] = []
    # Running length of `current`, counting one joining newline per unit.
    current_len = 0

    # Step 4: pack units into chunks.
    for unit in units:
        # Length this unit adds to the chunk (+1 for the "\n" that joins it).
        unit_len = len(unit) + 1
        # If adding the unit would overflow the chunk, finish the current chunk first.
        if current and current_len + unit_len > chunk_size:
            # Save the full chunk.
            chunks.append("\n".join(current))
            # Keep only a tail of the old chunk as overlap: drop units from the front
            # until what remains is no longer than `chunk_overlap` AND leaves room
            # for the new unit.
            while current and (current_len > chunk_overlap or current_len + unit_len > chunk_size):
                # Remove the oldest unit...
                removed = current.pop(0)
                # ...and subtract its length from the running total.
                current_len -= len(removed) + 1
        # Add the new unit to the chunk being built.
        current.append(unit)
        # Update the running length.
        current_len += unit_len

    # After the loop, whatever is left forms the final chunk.
    if current:
        chunks.append("\n".join(current))

    # Return the list of chunk strings.
    return chunks


def _split_into_units(text: str, max_len: int) -> list[str]:
    """Break text into paragraphs, then sentences, then fixed-size pieces."""
    # Collected units, in reading order.
    units: list[str] = []
    # Walk over each paragraph.
    for paragraph in _PARAGRAPH_BREAK.split(text):
        # Collapse runs of spaces/newlines/tabs into single spaces
        # (PDFs often break lines mid-sentence).
        paragraph = " ".join(paragraph.split())
        # Skip empty paragraphs.
        if not paragraph:
            continue
        # A paragraph that already fits is used as-is.
        if len(paragraph) <= max_len:
            units.append(paragraph)
            # Move on to the next paragraph.
            continue
        # Otherwise split the long paragraph into sentences.
        for sentence in _SENTENCE_BREAK.split(paragraph):
            # A sentence that fits is used as-is.
            if len(sentence) <= max_len:
                units.append(sentence)
            # A gigantic "sentence" (e.g. a table without punctuation) is cut into
            # fixed-size slices: [0:max_len], [max_len:2*max_len], ...
            else:
                units.extend(sentence[i : i + max_len] for i in range(0, len(sentence), max_len))
    # Return the flat list of units.
    return units
