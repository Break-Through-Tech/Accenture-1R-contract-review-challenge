from src.chunking.chunk import Chunk

def token_chunk_contract(
    contract,
    tokenizer,
    window_tokens=480,
    overlap_tokens=40
):
    """Split a contract by tokens while preserving character offsets."""

    if window_tokens <= 0 or not 0 <= overlap_tokens < window_tokens:
        raise ValueError("Invalid window or overlap size")

    # Tokenize without truncating the original contract
    encoded = tokenizer(
        contract.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    offsets = encoded["offset_mapping"]
    if not offsets:
        return []

    chunks = []
    step = window_tokens - overlap_tokens
    start_token = 0

    while start_token < len(offsets):
        end_token = min(start_token + window_tokens, len(offsets))

        # Convert token positions back to original character positions
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = (
            len(contract.text)
            if end_token == len(offsets)
            else offsets[end_token - 1][1]
        )

        piece = contract.text[start_char:end_char]

        chunks.append(
            Chunk(
                contract_id=contract.contract_id,
                start=start_char,
                end=end_char,
                text=piece,
                chunk_index=len(chunks)
            )
        )

        if end_token == len(offsets):
            break

        start_token += step

    return chunks