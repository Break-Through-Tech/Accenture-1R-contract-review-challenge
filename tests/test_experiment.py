"""Hand-calculated report checks for source overhead, realized overlap, and examples."""

import pytest

from src.chunking.chunk import Chunk
from src.chunking.chunker import chunk_contract
from src.chunking.experiment import compare_examples, describe_lengths, measure_chunks
from src.chunking.token_chunker import SentenceFallback
from src.labels.loader import AnswerSpan, Contract


class CharacterTokenizer:
    """Count one token per source character for transparent reporting-only golden values."""
    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs) -> dict:
        """Return a hand-computable token count; real tokenizer tests cover actual budgets."""
        return {"input_ids": [1] * (len(text) + 2 * add_special_tokens)}


# Three source windows duplicate two characters in a ten-character contract.
def test_report_golden_overhead_and_overlap_counts() -> None:
    """[0,4), [3,7), [6,10) produce 20% overhead and two one-token overlaps."""
    contract = Contract(contract_id="T", text="abcdefghij", spans=[])
    chunks = chunk_contract("T", contract.text, 4, 1)
    report, rows = measure_chunks([contract], {"T": chunks}, CharacterTokenizer(), 512)
    assert report.chunk_count == 3
    assert report.duplicated_char_overhead_pct == 20
    assert report.adjacency_count == 2
    assert report.overlap_content_tokens.mean == 1
    assert report.zero_overlap_pct == 0
    assert report.input_token_lengths.maximum == 6
    assert rows[0].overlap_chars is None


def test_report_no_overlap_excludes_first_chunks_from_denominator() -> None:
    """Two first chunks are excluded and all four within-contract adjacencies are zero overlap."""
    contracts = [Contract(contract_id=name, text="abcdefghij", spans=[]) for name in ("T", "U")]
    chunks = {c.contract_id: chunk_contract(c.contract_id, c.text, 4, 0) for c in contracts}
    report, _ = measure_chunks(contracts, chunks, CharacterTokenizer(), 512)
    assert report.adjacency_count == 4
    assert report.zero_overlap_pct == 100
    assert report.duplicated_char_overhead_pct == 0


def test_report_rejects_incomplete_coverage_before_overhead() -> None:
    """Overhead is never calculated for chunk sets with omitted source characters."""
    contract = Contract(contract_id="T", text="abcdefghij", spans=[])
    chunks = [Chunk(contract_id="T", start=1, end=10, text="bcdefghij", chunk_index=0)]
    with pytest.raises(ValueError, match="cover source endpoints"):
        measure_chunks([contract], {"T": chunks}, CharacterTokenizer(), 512)


def test_report_distribution_uses_exact_overflow_boundary() -> None:
    """512 is in budget and 513 is counted in the overflow histogram bin."""
    distribution = describe_lengths([1, 512, 513])
    assert distribution.p50 == 512
    assert distribution.histogram[">512"] == 1
    assert distribution.histogram["385-512"] == 1


def test_report_comparison_examples_match_containment_changes() -> None:
    """A grounded [3,5) annotation is rescued by overlap and lost when comparison is reversed."""
    text = "abcdefghij"
    contract = Contract(contract_id="T", text=text, spans=[
        AnswerSpan(category="Parties", start=3, end=5, text="de")])
    baseline = {"T": chunk_contract("T", text, 4, 0)}
    candidate = {"T": chunk_contract("T", text, 4, 1)}
    gained = compare_examples([contract], "old", "new", baseline, candidate)[0]
    lost = compare_examples([contract], "new", "old", candidate, baseline)[0]
    assert gained.outcome == "improved" and gained.candidate_contained and not gained.baseline_contained
    assert lost.outcome == "worsened" and lost.baseline_contained and not lost.candidate_contained


def test_report_fallback_overlap_is_separate_from_transitions() -> None:
    """Fallback internal overlap is separated from zero-overlap entry and exit."""
    contract = Contract(contract_id="T", text="abcdefghijklmn", spans=[])
    offsets = [(0, 2), (2, 7), (6, 12), (12, 14)]
    chunks = [Chunk(contract_id="T", start=a, end=b, text=contract.text[a:b], chunk_index=i)
              for i, (a, b) in enumerate(offsets)]
    cases = {"T": [SentenceFallback(start=2, end=12, chunk_indices=[1, 2])]}
    stats, rows = measure_chunks([contract], {"T": chunks}, CharacterTokenizer(), 512, cases)
    assert stats.fallback_sentences == 1
    assert stats.fallback_chunks == 2
    assert [row.adjacency_kind for row in rows[1:]] == [
        "fallback_transition", "fallback_internal", "fallback_transition",
    ]
    assert stats.zero_overlap_pct == pytest.approx(200 / 3)
    assert stats.zero_overlap_pct_by_kind == {"fallback_transition": 100, "fallback_internal": 0}
