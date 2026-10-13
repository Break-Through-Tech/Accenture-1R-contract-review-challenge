"""Real DistilRoBERTa checks for exact source coverage and bounded token windows."""

import pytest
from transformers import AutoTokenizer

from src.chunking.token_chunker import (
    TOKENIZER_NAME, TOKENIZER_REVISION, chunk_token_windows, chunk_sentence_windows,
    sentence_spans, token_length,
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the pinned real tokenizer from the experiment cache without network access."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION,
                                        use_fast=True, cache_dir=".venv/hf-cache", local_files_only=True)


def assert_chunk_invariants(text: str, chunks: list, tokenizer, budget: int) -> None:
    """Verify exact slices, indices, bounded input lengths, gap-free coverage, and progress."""
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for index, chunk in enumerate(chunks):
        assert chunk.text == text[chunk.start:chunk.end]
        assert chunk.chunk_index == index
        assert token_length(chunk.text, tokenizer) <= budget
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.start < current.start <= previous.end
        assert previous.end < current.end


# Real input includes Unicode offset groups and whitespace omitted/trimmed in mappings.
@pytest.mark.parametrize("text", [
    " word" * 1400,
    "  Acme Inc. operates in the U.S.  Section 1.2 applies.\n\n" * 120,
    "\n\t   " * 500,
    "  café 中文🙂 e\u0301 § 1.2.3\t\n" * 140,
])
def test_token_windows_real_text_preserves_coverage_and_budget(tokenizer, text: str) -> None:
    """Adversarial real-tokenizer chunks cover unchanged text and fit 512 including specials."""
    chunks = chunk_token_windows("T", text, tokenizer, 512, 64)
    assert_chunk_invariants(text, chunks, tokenizer, 512)
    assert chunks == chunk_token_windows("T", text, tokenizer, 512, 64)
    for previous, current in zip(chunks, chunks[1:]):
        assert token_length(text[current.start:previous.end], tokenizer, False) <= 64


def test_token_windows_golden_offsets_and_special_token_budget(tokenizer) -> None:
    """Six content slots plus two special tokens split repeated words at exact offsets."""
    text = "word " * 12
    chunks = chunk_token_windows("T", text, tokenizer, 8, 2)
    assert [(c.start, c.end) for c in chunks] == [(0, 29), (20, 49), (40, 60)]
    assert_chunk_invariants(text, chunks, tokenizer, 8)
    assert tokenizer.num_special_tokens_to_add(pair=False) == 2


def test_token_windows_empty_and_short_text(tokenizer) -> None:
    """Empty text has no chunks and short text preserves its leading/trailing whitespace."""
    assert chunk_token_windows("T", "", tokenizer, 512, 64) == []
    text = "  Agreement.\n "
    chunks = chunk_token_windows("T", text, tokenizer, 512, 64)
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.parametrize("budget,overlap", [(2, 0), (513, 64), (8, -1), (8, 6)])
def test_token_windows_invalid_budget_raises(tokenizer, budget: int, overlap: int) -> None:
    """Invalid settings fail even with no text."""
    with pytest.raises(ValueError):
        chunk_token_windows("T", "", tokenizer, budget, overlap)


def test_token_windows_small_budget_and_high_overlap_advances(tokenizer) -> None:
    """A near-full overlap still increases starts and covered endpoints."""
    text = "🙂中文 word  \n" * 40
    chunks = chunk_token_windows("T", text, tokenizer, 10, 7)
    assert_chunk_invariants(text, chunks, tokenizer, 10)


def test_token_windows_trimmed_whitespace_counts_toward_budget(tokenizer) -> None:
    """Zero-length whitespace offsets still consume tokens and produce useful bounded windows."""
    text = " " * 2000
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert any(a == b for a, b in offsets)
    chunks = chunk_token_windows("T", text, tokenizer, 512, 64)
    assert_chunk_invariants(text, chunks, tokenizer, 512)
    assert len(chunks) < 10
    assert all(token_length(text[b.start:a.end], tokenizer, False) <= 64
               for a, b in zip(chunks, chunks[1:]))


def test_token_windows_repeated_unicode_offsets_do_not_drop_characters(tokenizer) -> None:
    """Byte-level tokens sharing a character offset do not justify dropping or decoding it."""
    text = "🙂中文" * 300
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert len(set(offsets)) < len(offsets)
    assert_chunk_invariants(text, chunk_token_windows("T", text, tokenizer, 512, 64), tokenizer, 512)


# B uses sentence boundaries but keeps layout, offsets, and model-input length intact.
def test_sentence_detector_abbreviations_and_numbering_preserves_source() -> None:
    """Inc., U.S., and decimal section numbers do not create spurious sentence breaks."""
    text = "  Acme Inc. operates in the U.S. market. Next sentence.  "
    spans = sentence_spans(text)
    assert spans == [(0, 41), (41, len(text))]
    numbered = "Section 1.2 applies.\n2.3 Other terms follow.  "
    assert sentence_spans(numbered) == [(0, 21), (21, len(numbered))]
    assert "".join(text[a:b] for a, b in spans) == text


@pytest.mark.parametrize("text", [
    "  Acme Inc. operates in the U.S. market. Section 1.2 applies.\n\n" * 100,
    "\n\t   " * 500,
    "  café 中文🙂 e\u0301 § 1.2.3 applies.\t\n" * 140,
])
def test_sentence_windows_real_text_preserves_budget_and_determinism(tokenizer, text: str) -> None:
    """Sentence packing and fallback preserve source coverage and a real 512-token limit."""
    chunks = chunk_sentence_windows("T", text, tokenizer, 512, 64)
    assert_chunk_invariants(text, chunks, tokenizer, 512)
    assert chunks == chunk_sentence_windows("T", text, tokenizer, 512, 64)
    for previous, current in zip(chunks, chunks[1:]):
        assert token_length(text[current.start:previous.end], tokenizer, False) <= 64


def test_sentence_windows_oversized_sentence_records_fallback(tokenizer) -> None:
    """A long sentence uses A internally and has zero overlap at its transitions."""
    text = "Intro. " + "word " * 1400 + ". Next sentence. "
    cases = []
    chunks = chunk_sentence_windows("T", text, tokenizer, 512, 64, cases)
    assert_chunk_invariants(text, chunks, tokenizer, 512)
    assert len(cases) == 1
    case = cases[0]
    assert len(case.chunk_indices) > 1
    pieces = chunk_token_windows("T", text[case.start:case.end], tokenizer, 512, 64)
    assert [(chunks[i].start, chunks[i].end) for i in case.chunk_indices] == [
        (case.start + c.start, case.start + c.end) for c in pieces
    ]
    assert chunks[case.chunk_indices[0] - 1].end == case.start
    assert chunks[case.chunk_indices[-1] + 1].start == case.end


def test_sentence_windows_longest_complete_suffix_and_reduced_overlap(tokenizer) -> None:
    """The longest fitting trailing suffix is repeated, never a partial normal sentence."""
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. "
    spans = sentence_spans(text)
    chunks = chunk_sentence_windows("T", text, tokenizer, 14, 5)
    assert_chunk_invariants(text, chunks, tokenizer, 14)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start in {a for a, _ in spans}
        suffix = text[current.start:previous.end]
        assert token_length(suffix, tokenizer, False) <= 5
        # Any earlier advancing sentence start either exceeds overlap or cannot fit new content.
        new_end = next(b for a, b in spans if a == previous.end)
        for a, _ in spans:
            if previous.start < a < current.start:
                assert (token_length(text[a:previous.end], tokenizer, False) > 5 or
                        token_length(text[a:new_end], tokenizer) > 14)


def test_sentence_windows_last_sentence_above_overlap_target_has_zero_overlap(tokenizer) -> None:
    """A final sentence above the target is not cut merely to obtain overlap."""
    text = "A short sentence here. " * 12
    chunks = chunk_sentence_windows("T", text, tokenizer, 14, 2)
    assert_chunk_invariants(text, chunks, tokenizer, 14)
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))


def test_sentence_windows_empty_text_has_no_chunks(tokenizer) -> None:
    """Sentence-aware empty input emits no chunks or fallback cases."""
    assert chunk_sentence_windows("T", "", tokenizer, 512, 64) == []
