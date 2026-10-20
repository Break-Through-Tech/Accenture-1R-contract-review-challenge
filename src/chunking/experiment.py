"""Small reporting helpers for the development-only token chunking comparison."""

import numpy as np
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast

from src.chunking.chunk import Chunk
from src.chunking.evaluation import validate_chunks
from src.chunking.token_chunker import SentenceFallback, token_length
from src.labels.loader import Contract


class LengthDistribution(BaseModel):
    """Token counts summarized with linear-interpolated percentiles and fixed bins."""
    count: int
    minimum: int | None
    mean: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: int | None
    histogram: dict[str, int]


class ChunkMeasurement(BaseModel):
    """One chunk's untruncated model length and realized overlap with its predecessor."""
    contract_id: str
    chunk_index: int
    start: int
    end: int
    input_tokens: int
    overlap_chars: int | None
    overlap_content_tokens: int | None
    adjacency_kind: str | None


class ChunkStatistics(BaseModel):
    """Coverage-verified chunk cost, model-length, overlap, and sentence-fallback statistics."""
    chunk_count: int
    source_chars: int
    emitted_chars: int
    duplicated_char_overhead_pct: float | None
    overflow_chunks: int
    overflow_pct: float | None
    input_token_lengths: LengthDistribution
    adjacency_count: int
    zero_overlap_adjacencies: int
    zero_overlap_pct: float | None
    overlap_content_tokens: LengthDistribution
    overlap_characters: LengthDistribution
    overlap_by_kind: dict[str, LengthDistribution]
    zero_overlap_pct_by_kind: dict[str, float]
    fallback_sentences: int
    fallback_chunks: int


class ComparisonExample(BaseModel):
    """A unique annotation gaining or losing containment relative to a character baseline."""
    outcome: str
    baseline: str
    candidate: str
    contract_id: str
    category: str
    start: int
    end: int
    annotation_preview: str
    preview_truncated: bool
    baseline_contained: bool
    candidate_contained: bool
    baseline_cut: bool
    candidate_cut: bool
    baseline_chunks: list[Chunk]
    candidate_chunks: list[Chunk]


def describe_lengths(lengths: list[int]) -> LengthDistribution:
    """Describe natural token counts, including a separate bin for overflow above 512."""
    limits = [(0, 64), (65, 128), (129, 256), (257, 384), (385, 512)]
    histogram = {f"{low}-{high}": sum(low <= n <= high for n in lengths) for low, high in limits}
    histogram[">512"] = sum(n > 512 for n in lengths)
    values = np.percentile(lengths, [50, 90, 95, 99]).tolist() if lengths else [None] * 4
    return LengthDistribution(
        count=len(lengths), minimum=min(lengths) if lengths else None,
        mean=float(np.mean(lengths)) if lengths else None,
        p50=values[0], p90=values[1], p95=values[2], p99=values[3],
        maximum=max(lengths) if lengths else None, histogram=histogram,
    )


def measure_chunks(
    contracts: list[Contract], chunks_by_contract: dict[str, list[Chunk]],
    tokenizer: PreTrainedTokenizerFast, max_tokens: int,
    fallback_by_contract: dict[str, list[SentenceFallback]] | None = None,
) -> tuple[ChunkStatistics, list[ChunkMeasurement]]:
    """Validate full coverage before computing overhead and measure all final source slices.

    Overlap is measured on the original shared substring without special tokens.
    The first chunk of each contract is excluded from adjacency denominators.
    Oversized-sentence internal adjacencies and transitions are reported separately.
    """
    # Check every contract first: duplicated-text overhead is meaningful only with full coverage.
    for contract in contracts:
        validate_chunks(contract, chunks_by_contract[contract.contract_id])

    measurements = []
    source_chars = emitted_chars = fallback_sentences = fallback_chunks = 0
    overlap_by_kind: dict[str, list[int]] = {}
    for contract in contracts:
        source_chars += len(contract.text)
        chunks = chunks_by_contract[contract.contract_id]
        cases = (fallback_by_contract or {}).get(contract.contract_id, [])
        membership = {index: group for group, case in enumerate(cases) for index in case.chunk_indices}
        fallback_sentences += len(cases)
        fallback_chunks += len(membership)
        for index, chunk in enumerate(chunks):
            emitted_chars += len(chunk.text)
            overlap_chars = overlap_tokens = kind = None
            if index:
                previous = chunks[index - 1]
                overlap_chars = previous.end - chunk.start
                overlap_tokens = token_length(contract.text[chunk.start:previous.end], tokenizer, False)
                if index in membership and membership.get(index - 1) == membership[index]:
                    kind = "fallback_internal"
                elif index in membership or index - 1 in membership:
                    kind = "fallback_transition"
                else:
                    kind = "regular"
                overlap_by_kind.setdefault(kind, []).append(overlap_tokens)
            measurements.append(ChunkMeasurement(
                contract_id=contract.contract_id, chunk_index=index, start=chunk.start, end=chunk.end,
                input_tokens=token_length(chunk.text, tokenizer), overlap_chars=overlap_chars,
                overlap_content_tokens=overlap_tokens, adjacency_kind=kind,
            ))

    lengths = [m.input_tokens for m in measurements]
    overlaps = [m.overlap_content_tokens for m in measurements if m.overlap_content_tokens is not None]
    overlap_chars = [m.overlap_chars for m in measurements if m.overlap_chars is not None]
    zero = sum(n == 0 for n in overlap_chars)
    overflows = sum(n > max_tokens for n in lengths)
    stats = ChunkStatistics(
        chunk_count=len(lengths), source_chars=source_chars, emitted_chars=emitted_chars,
        duplicated_char_overhead_pct=100 * (emitted_chars - source_chars) / source_chars if source_chars else None,
        overflow_chunks=overflows, overflow_pct=100 * overflows / len(lengths) if lengths else None,
        input_token_lengths=describe_lengths(lengths), adjacency_count=len(overlaps),
        zero_overlap_adjacencies=zero, zero_overlap_pct=100 * zero / len(overlaps) if overlaps else None,
        overlap_content_tokens=describe_lengths(overlaps), overlap_characters=describe_lengths(overlap_chars),
        overlap_by_kind={kind: describe_lengths(values) for kind, values in overlap_by_kind.items()},
        zero_overlap_pct_by_kind={kind: 100 * sum(
            m.overlap_chars == 0 for m in measurements if m.adjacency_kind == kind
        ) / len(values) for kind, values in overlap_by_kind.items()},
        fallback_sentences=fallback_sentences, fallback_chunks=fallback_chunks,
    )
    return stats, measurements


def annotation_lengths(contracts: list[Contract], tokenizer: PreTrainedTokenizerFast) -> dict[str, LengthDistribution]:
    """Measure deduplicated annotations standalone, with special tokens, as a diagnostic."""
    seen = set()
    by_category: dict[str, list[int]] = {}
    overall = []
    for contract in contracts:
        for span in contract.spans:
            key = (contract.contract_id, span.category, span.start, span.end)
            if key not in seen:
                seen.add(key)
                length = token_length(span.text, tokenizer)
                overall.append(length)
                by_category.setdefault(span.category, []).append(length)
    return {"overall": describe_lengths(overall),
            **{category: describe_lengths(values) for category, values in by_category.items()}}


def compare_examples(
    contracts: list[Contract], baseline_name: str, candidate_name: str,
    baseline: dict[str, list[Chunk]], candidate: dict[str, list[Chunk]],
) -> list[ComparisonExample]:
    """Select at most two gains and two losses of full annotation containment in input order."""
    selected = []
    counts = {"improved": 0, "worsened": 0}
    seen = set()
    for contract in contracts:
        for span in contract.spans:
            key = (contract.contract_id, span.category, span.start, span.end)
            if key in seen:
                continue
            seen.add(key)
            old = [c for c in baseline[contract.contract_id] if c.start < span.end and span.start < c.end]
            new = [c for c in candidate[contract.contract_id] if c.start < span.end and span.start < c.end]
            old_contained = any(c.start <= span.start and span.end <= c.end for c in old)
            new_contained = any(c.start <= span.start and span.end <= c.end for c in new)
            if old_contained == new_contained:
                continue
            outcome = "improved" if new_contained else "worsened"
            if counts[outcome] == 2:
                continue
            counts[outcome] += 1
            selected.append(ComparisonExample(
                outcome=outcome, baseline=baseline_name, candidate=candidate_name,
                contract_id=contract.contract_id, category=span.category, start=span.start, end=span.end,
                annotation_preview=span.text[:240], preview_truncated=len(span.text) > 240,
                baseline_contained=old_contained, candidate_contained=new_contained,
                baseline_cut=any(span.start < b < span.end for c in old for b in (c.start, c.end)),
                candidate_cut=any(span.start < b < span.end for c in new for b in (c.start, c.end)),
                baseline_chunks=old, candidate_chunks=new,
            ))
    return selected
