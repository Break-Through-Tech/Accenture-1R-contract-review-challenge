from src.chunking.chunk import Chunk


def chunk_contract(contract_id: str, text: str, window_chars: int, overlap_chars: int) -> list[Chunk]:
    """
    Split a contract into fixed-size, overlapping character windows (naive stub).
    Offsets are absolute and end-exclusive into the original text, which is sliced
    unaltered, so chunk.text == text[chunk.start:chunk.end]. Ignores sentence &
    clause boundaries. Returns [] for empty text and raises ValueError if
    window_chars <= 0, overlap_chars < 0, or overlap_chars >= window_chars.
    """
    # Handle empty input and reject window/overlap values that would never advance
    if len(text) == 0:
        return []
    if window_chars <= 0 or overlap_chars < 0:
        raise ValueError(
            f"window_chars must be > 0 and overlap_chars must be >= 0 "
            f"(got window_chars={window_chars}, overlap_chars={overlap_chars})"
        )
    if overlap_chars >= window_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be less than window_chars ({window_chars})"
        )

    # Each chunk starts `step` characters after the previous one
    step = window_chars - overlap_chars
    start, chunk_index = 0, 0
    chunks = []

    # Slice the original text window by window until a chunk reaches the end
    while True:
        end = min(start + window_chars, len(text))
        piece = text[start:end]
        chunk = Chunk(contract_id=contract_id, start=start, end=end,
                      text=piece, chunk_index=chunk_index)
        chunks.append(chunk)
        chunk_index += 1

        # Stop at the final chunk so no redundant tail chunk is created
        if end == len(text):
            break
        start += step

    return chunks
