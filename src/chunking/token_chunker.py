"""Experimental token windows over exact original-text slices (not the shared chunker)."""

from bisect import bisect_right
from pydoc import text

from nupunkt import sent_spans
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast

from src.chunking.chunk import Chunk

TOKENIZER_NAME = "distilbert/distilroberta-base"
# The experiment runner and real-tokenizer tests load the same immutable revision.
TOKENIZER_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"


class SentenceFallback(BaseModel):
    """One oversized source sentence and the indices of its token-window chunks."""
    start: int
    end: int
    chunk_indices: list[int]


def token_length(text: str, tokenizer: PreTrainedTokenizerFast, special_tokens: bool = True) -> int:
    """Count untruncated, unpadded tokens, optionally including model special tokens."""
    return len(tokenizer(text, add_special_tokens=special_tokens,
                         truncation=False, padding=False, verbose=False)["input_ids"])


def _validate_budget(tokenizer: PreTrainedTokenizerFast, max_tokens: int, overlap_tokens: int) -> int:
    """Reject budgets that leave no content room or permit nonadvancing overlap."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with original-character offsets is required")
    content_budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if max_tokens > 512 or content_budget <= 0 or not 0 <= overlap_tokens < content_budget:
        raise ValueError("Require max_tokens <= 512, positive content budget, and smaller nonnegative overlap")
    return content_budget


def _fit_end(
    text: str, start: int, covered_end: int, ends: list[int], token_ends: list[int],
    tokenizer: PreTrainedTokenizerFast, max_tokens: int, content_budget: int,
) -> int:
    """Estimate a token endpoint, retreat until its exact source slice fits, and advance."""
    # Repeated token endpoints consume tokens but only distinct endpoints can split characters.
    first = bisect_right(token_ends, start)
    estimate = token_ends[min(first + content_budget, len(token_ends)) - 1] if first < len(token_ends) else len(text)
    if first + content_budget >= len(token_ends):
        estimate = len(text)
    index = bisect_right(ends, estimate) - 1
    while index >= 0 and ends[index] > covered_end:
        end = ends[index]
        if token_length(text[start:end], tokenizer) <= max_tokens:
            return end
        index -= 1

    # Offset trimming can omit whitespace; a character endpoint still preserves the source.
    end = covered_end + 1
    if end <= len(text) and token_length(text[start:end], tokenizer) <= max_tokens:
        return end
    raise ValueError("Cannot fit an advancing source slice in the token budget")


def _overlap_start(text: str, start: int, end: int, tokenizer: PreTrainedTokenizerFast,
                   overlap_tokens: int) -> int:
    """Map the trailing token target to an advancing character boundary, reducing if needed."""
    if not overlap_tokens:
        return end
    encoding = tokenizer(text[start:end], add_special_tokens=False, truncation=False,
                         padding=False, return_offsets_mapping=True, verbose=False)
    offsets = encoding["offset_mapping"]
    target = max(0, len(offsets) - overlap_tokens)
    candidates = sorted({start + a for a, b in offsets[target:] if start < start + a < end})
    # Retokenizing a suffix may change its count; never exceed the overlap target.
    for candidate in candidates:
        if token_length(text[candidate:end], tokenizer, special_tokens=False) <= overlap_tokens:
            return candidate
    return end


def chunk_token_windows(
    contract_id: str, text: str, tokenizer: PreTrainedTokenizerFast,
    max_tokens: int, overlap_tokens: int,
) -> list[Chunk]:
    """Return covering token windows with explicit input budget and content-token overlap.

    Offsets refer to the unaltered source. Each final slice is retokenized with
    special tokens and must fit max_tokens (at most 512). Overlap targets the last
    overlap_tokens content tokens; offset grouping and suffix retokenization can
    reduce it. If overlap prevents new coverage, drop it. No annotations are used.
    """
    content_budget = _validate_budget(tokenizer, max_tokens, overlap_tokens)
    if not text:
        return []
    encoding = tokenizer(text, add_special_tokens=False, truncation=False, padding=False,
                         return_offsets_mapping=True, verbose=False)
    # Trimmed whitespace often has (position, position) offsets but still consumes tokens.
    # Keep those tokens in the estimate and positive coordinates as candidate cuts;
    # coverage comes from original-text slices, never from token offset spans.
    token_ends = sorted(b for a, b in encoding["offset_mapping"])
    ends = sorted({b for b in token_ends if b > 0} | {len(text)})
    chunks = []
    start = covered_end = 0
    while covered_end < len(text):
        # Overlap is secondary to the budget and strict progress of the covered endpoint.
        try:
            end = _fit_end(text, start, covered_end, ends, token_ends,
                           tokenizer, max_tokens, content_budget)
        except ValueError:
            if start == covered_end:
                raise
            start = covered_end
            end = _fit_end(text, start, covered_end, ends, token_ends,
                           tokenizer, max_tokens, content_budget)
        chunks.append(Chunk(contract_id=contract_id, start=start, end=end,
                            text=text[start:end], chunk_index=len(chunks)))
        covered_end = end
        start = _overlap_start(text, start, end, tokenizer, overlap_tokens)
    return chunks


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return NUPunkt's nonadaptive, layout-free, contiguous original-character spans."""
    spans = sent_spans(text) if text else []
    cursor = 0
    for start, end in spans:
        if start != cursor or not start < end <= len(text):
            raise ValueError("Sentence detector did not return contiguous source spans")
        cursor = end
    if cursor != len(text):
        raise ValueError("Sentence detector omitted source text")
    return spans


def chunk_sentence_windows(
    contract_id: str, text: str, tokenizer: PreTrainedTokenizerFast,
    max_tokens: int, overlap_tokens: int,
    fallback_cases: list[SentenceFallback] | None = None,
) -> list[Chunk]:
    """Pack complete sentences, using token windows for each oversized sentence.

    Carry the longest complete-sentence suffix with at most overlap_tokens content
    tokens, measured on its combined original slice. Reduce overlap to fit at least
    one new sentence; never repeat a whole chunk. Long sentences use token overlap
    internally, with zero overlap at fallback transitions. Optional fallback_cases
    records their source offsets and chunk indices without changing Chunk outputs.
    """
    _validate_budget(tokenizer, max_tokens, overlap_tokens)
    spans = sentence_spans(text)

    sentence_texts = [text[start:end] for start, end in spans]

    if sentence_texts:
        sentence_lengths = [
            len(ids)
            for ids in tokenizer(
                sentence_texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                verbose=False,
            )["input_ids"]
        ]
    else:
        sentence_lengths = []

    chunks = []
    next_sentence = 0
    while next_sentence < len(spans):
        # A sentence that cannot fit alone is kept out of normal sentence packing.
        start, end = spans[next_sentence]
        if sentence_lengths[next_sentence] > max_tokens:
            pieces = chunk_token_windows(contract_id, text[start:end], tokenizer,
                                        max_tokens, overlap_tokens)
            indices = []
            for piece in pieces:
                index = len(chunks)
                indices.append(index)
                chunks.append(Chunk(contract_id=contract_id, start=start + piece.start,
                                    end=start + piece.end, text=piece.text, chunk_index=index))
            if fallback_cases is not None:
                fallback_cases.append(SentenceFallback(start=start, end=end, chunk_indices=indices))
            next_sentence += 1
            continue

        # Greedily add new complete sentences while each combined source slice fits.
        first_sentence = next_sentence
        last_sentence = next_sentence
        while last_sentence + 1 < len(spans):
            candidate_start, candidate_end = spans[last_sentence + 1]
            if sentence_lengths[last_sentence + 1] > max_tokens:
                break
            if token_length(text[start:candidate_end], tokenizer) > max_tokens:
                break
            last_sentence += 1
        end = spans[last_sentence][1]
        chunks.append(Chunk(contract_id=contract_id, start=start, end=end,
                            text=text[start:end], chunk_index=len(chunks)))
        next_sentence = last_sentence + 1
        if next_sentence == len(spans):
            break

        # Do not overlap into a fallback; it has its own separate token-window policy.
        new_start, new_end = spans[next_sentence]
        if sentence_lengths[next_sentence] > max_tokens:
            continue
        if token_length(text[spans[last_sentence][0]:end], tokenizer, False) > overlap_tokens:
            continue

        # Choose the earliest fitting suffix start (longest suffix), including new content.
        for candidate in range(first_sentence + 1, last_sentence + 1):
            overlap_start = spans[candidate][0]
            if token_length(text[overlap_start:end], tokenizer, False) <= overlap_tokens:
                if token_length(text[overlap_start:new_end], tokenizer) <= max_tokens:
                    next_sentence = candidate
                    break
    return chunks
